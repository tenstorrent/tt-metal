# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host side of TimesFM-3: everything around the network, in fp64, with no ttnn import.

Mirrors ``timesfm3.torch`` at google-research/timesfm@e51928e (package 3.0.2):
- ``TimesFM3Forecaster.predict_batch``: NaN trimming, interpolation, symmetric duplication,
  left padding to the batch context (``format_batch``);
- ``TimesFM3Torch.decode`` / ``forward`` / ``_preprocess``: linear detrending, horizon (CPM) patches,
  causal running RevIN, token features (``model_inputs``);
- the post-processing: iterative CPM RevIN refine, de-normalization, stitching, trend, sort,
  symmetric averaging and ``make_positive`` (``postprocess``).

Departures from the reference, all exact in real arithmetic:
- running statistics are masked cumulative sums of (count, sum, sum of squares) of shifted values in
  fp64, instead of the sequential Chan merge in fp32;
- the detrend variances are computed centred, in fp64;
- only target variates are supported (no covariates), so every token's future window is fully masked.

This file must stay importable without the ``models`` package: the reference venv loads it by path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

PATCH = 32
OUTPUT_PATCH = 64
ROLLS = OUTPUT_PATCH // PATCH
NUM_QUANTILES = 9
MEDIAN_QUANTILE = NUM_QUANTILES // 2
TOKEN_FEATURES = 2 * (PATCH + OUTPUT_PATCH)
MAX_CONTEXT = 15360
MAX_VARIATES = 32
VALUE_CLIP = 1e20
SIGMA_TOLERANCE = 1e-6
DETREND_THRESHOLD = 0.5
STITCH_EXTRACT = min(2 * PATCH, OUTPUT_PATCH)
STITCH_OVERLAP = STITCH_EXTRACT - PATCH


def num_forecast_patches(horizon: int) -> int:
    """Patches read out for a decode horizon (``model.decode`` with stitching)."""
    return max(math.ceil((horizon - STITCH_OVERLAP) / PATCH), 1)


def num_horizon_patches(horizon: int) -> int:
    return num_forecast_patches(horizon) + ROLLS - 1


def decode_horizon(horizon: int) -> int:
    """``predict_batch`` rounds the requested horizon up to whole output patches."""
    return math.ceil(horizon / OUTPUT_PATCH) * OUTPUT_PATCH


@dataclass
class FormattedBatch:
    """What ``predict_batch`` hands to ``model.decode`` (one batch), plus what its post-processing needs."""

    target: np.ndarray  # (b, u, C) float32
    mask: np.ndarray  # (b, C) bool, True = padded
    horizon: int  # decode horizon (multiple of 64)
    requested_horizon: int
    symmetric: bool
    make_positive: bool
    was_1d: bool
    nonnegative: np.ndarray | None  # (items, u) bool, for make_positive


def _linear_interpolation(arr: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    out = arr.copy()
    for r in range(out.shape[0]):
        if nan_mask[r].any():
            valid = ~nan_mask[r]
            if valid.any():
                out[r, nan_mask[r]] = np.interp(nan_mask[r].nonzero()[0], valid.nonzero()[0], out[r, valid])
            else:
                out[r, nan_mask[r]] = 0.0
    return out


def _nonnegative(arr: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(arr)
    res = np.zeros(arr.shape[0], dtype=bool)
    for r in range(arr.shape[0]):
        valid = arr[r][~np.isnan(arr[r])]
        res[r] = valid.size > 0 and bool(np.all(valid >= 0))
    return res


def format_batch(
    contexts,
    horizon: int,
    *,
    use_symmetric_averaging: bool = False,
    make_positive: bool = False,
    max_context: int = MAX_CONTEXT,
) -> FormattedBatch:
    """``predict_batch`` up to the ``decode`` call, with every query in one batch."""
    targets = []
    for ctx in contexts:
        t = np.atleast_2d(np.array(ctx, dtype=np.float32))
        isnan = np.isnan(t).all(axis=0)
        first = t.shape[-1] if isnan.all() else int(np.argmax(~isnan))
        if 0 < first < t.shape[-1]:
            t = t[:, first:]
        elif first == t.shape[-1] and t.shape[-1] > 0:
            t = np.zeros_like(t)
        targets.append(_linear_interpolation(t))
    u = targets[0].shape[0]
    if any(t.shape[0] != u for t in targets):
        raise ValueError("all contexts must have the same number of target variates")
    if u > MAX_VARIATES:
        raise ValueError(f"at most {MAX_VARIATES} variates per item (the evaluator subsamples above that)")
    if use_symmetric_averaging:
        targets = [s for t in targets for s in (t, -t)]
    longest = max(t.shape[-1] for t in targets)
    ceiling = math.ceil(max_context / PATCH) * PATCH
    context = max(min(math.ceil(longest / PATCH) * PATCH, ceiling), PATCH)
    tgt = np.zeros((len(targets), u, context), dtype=np.float32)
    mask = np.zeros((len(targets), context), dtype=bool)
    for i, t in enumerate(targets):
        if t.shape[-1] >= context:
            tgt[i] = t[:, -context:]
        else:
            pad = context - t.shape[-1]
            tgt[i, :, pad:] = t
            mask[i, :pad] = True
    nonneg = np.stack([_nonnegative(np.asarray(c, dtype=np.float32)) for c in contexts]) if make_positive else None
    return FormattedBatch(
        target=tgt,
        mask=mask,
        horizon=decode_horizon(horizon),
        requested_horizon=horizon,
        symmetric=use_symmetric_averaging,
        make_positive=make_positive,
        was_1d=np.ndim(contexts[0]) == 1,
        nonnegative=nonneg,
    )


@dataclass
class HostInputs:
    """Model inputs and the statistics the post-processing needs. Rows are ``item * variates + variate``."""

    tokens: torch.Tensor  # (rows, n, 192) float64, the tokenizer input (``resblock_input``)
    num_front_masked: torch.Tensor  # (rows,) int64: leading fully masked patches (attention key padding)
    variates: int
    num_context_patches: int
    horizon: int  # decode horizon
    context_len: int
    running_n: torch.Tensor  # (rows, n) float64
    running_mu: torch.Tensor
    running_sigma: torch.Tensor
    trend_m: torch.Tensor  # (rows,) float64
    trend_c: torch.Tensor
    trend_apply: torch.Tensor  # (rows,) bool
    batch: FormattedBatch | None = None

    @property
    def rows(self) -> int:
        return self.tokens.shape[0]

    @property
    def items(self) -> int:
        return self.rows // self.variates

    @property
    def num_patches(self) -> int:
        return self.tokens.shape[1]

    @property
    def num_forecast_patches(self) -> int:
        return num_forecast_patches(self.horizon)

    @property
    def readout_start(self) -> int:
        """The first read-out token: the last context patch."""
        return self.num_context_patches - 1

    @property
    def num_kept_patches(self) -> int:
        """Tokens the device runs: everything up to the last read-out token (causal attention, per-patch rest)."""
        return self.readout_start + self.num_forecast_patches

    @property
    def readout_tokens(self) -> range:
        return range(self.readout_start, self.readout_start + self.num_forecast_patches)

    @property
    def token_valid(self) -> torch.Tensor:
        """(rows, n) bool: False for the leading fully masked patches."""
        return torch.arange(self.num_patches)[None, :] >= self.num_front_masked[:, None]


def _detrend(y: torch.Tensor, valid: torch.Tensor, context: int):
    """Masked least-squares line over t in [-(C-1)/C, 0]; fp64 ``model.decode`` steps 3a."""
    t = torch.arange(-(context - 1), 1, dtype=torch.float64) / context
    t = t.expand_as(y)
    vf = valid.to(torch.float64)
    n = vf.sum(-1)
    st = (t * vf).sum(-1)
    st2 = (t * t * vf).sum(-1)
    yv = torch.where(valid, y, 0.0)
    sy = yv.sum(-1)
    sty = (t * yv).sum(-1)
    det = n * st2 - st * st
    zero = det == 0.0
    m = torch.where(zero, 0.0, (n * sty - st * sy) / torch.where(zero, 1.0, det))
    nn_ = n.clamp_min(1.0)
    c = torch.where(zero, torch.where(n > 0, sy / nn_, 0.0), (sy - m * st) / nn_)
    yd = y - (m[..., None] * t + c[..., None])

    def masked_std(x):
        mean = torch.where(valid, x, 0.0).sum(-1) / nn_
        return torch.sqrt((torch.where(valid, x - mean[..., None], 0.0) ** 2).sum(-1) / nn_)

    apply = masked_std(yd) < DETREND_THRESHOLD * masked_std(y)
    return torch.where(apply[..., None], yd, y), m, c, apply


def running_stats(values: torch.Tensor, masks: torch.Tensor):
    """Causal per-patch (count, mean, population std) over patches 0..i as masked cumsums in fp64.

    ``values``/``masks``: (rows, n, p). Equal in exact arithmetic to ``util.get_running_stats``; the sums
    are of values shifted by the row's mean of valid points, which keeps ``S2/N - mean^2`` well conditioned.
    """
    valid = ~masks
    vf = valid.to(torch.float64)
    count = vf.sum(-1)
    total = count.sum(-1).clamp_min(1.0)
    shift = (torch.where(valid, values, 0.0).sum((-1, -2)) / total)[:, None, None]
    xs = torch.where(valid, values - shift, 0.0)
    n = count.cumsum(-1)
    s1 = xs.sum(-1).cumsum(-1)
    s2 = (xs * xs).sum(-1).cumsum(-1)
    safe_n = n.clamp_min(1.0)
    mean_s = s1 / safe_n
    var = (s2 / safe_n - mean_s * mean_s).clamp_min(0.0)
    empty = n == 0
    mu = torch.where(empty, 0.0, mean_s + shift[..., 0])
    sigma = torch.where(empty, 0.0, torch.sqrt(var))
    return n, mu, sigma


def _safe_sigma(sigma: torch.Tensor) -> torch.Tensor:
    return torch.where(sigma < SIGMA_TOLERANCE, 1.0, sigma)


def model_inputs(target, mask, horizon: int, *, batch: FormattedBatch | None = None) -> HostInputs:
    """``model.decode`` + ``forward._preprocess`` on the host. ``target`` (b, u, C), ``mask`` (b, C)."""
    target = torch.as_tensor(target).to(torch.float64)
    mask = torch.as_tensor(mask).to(torch.bool)
    b, u, context = target.shape
    if context % PATCH:
        raise ValueError("context must be a multiple of the patch length (format_batch pads it)")
    rows = b * u
    y = target.reshape(rows, context)
    ctx_masks = mask[:, None, :].expand(b, u, context).reshape(rows, context)
    y, m, c, apply = _detrend(y, ~ctx_masks, context)
    y = torch.where(ctx_masks, 0.0, y)
    hp = num_horizon_patches(horizon)
    nc = context // PATCH
    n = nc + hp
    values = torch.cat([y, torch.zeros(rows, hp * PATCH, dtype=torch.float64)], -1)
    masks = torch.cat([ctx_masks, torch.ones(rows, hp * PATCH, dtype=torch.bool)], -1)
    values = values.nan_to_num(0.0).clamp(-VALUE_CLIP, VALUE_CLIP).reshape(rows, n, PATCH)
    masks = masks.reshape(rows, n, PATCH)
    rn, rmu, rsig = running_stats(values, masks)
    norm = (values - rmu[..., None]) / _safe_sigma(rsig)[..., None]
    norm = torch.where(masks, 0.0, norm)
    # Targets never see a future window: values 0, mask 1 (model.py:245).
    tokens = torch.cat(
        [
            norm,
            torch.zeros(rows, n, OUTPUT_PATCH, dtype=torch.float64),
            masks.to(torch.float64),
            torch.ones(rows, n, OUTPUT_PATCH, dtype=torch.float64),
        ],
        -1,
    )
    fully_masked = masks.all(-1)
    num_front_masked = torch.cumprod(fully_masked.to(torch.int64), -1).sum(-1)
    return HostInputs(
        tokens=tokens,
        num_front_masked=num_front_masked,
        variates=u,
        num_context_patches=nc,
        horizon=horizon,
        context_len=context,
        running_n=rn,
        running_mu=rmu,
        running_sigma=rsig,
        trend_m=m,
        trend_c=c,
        trend_apply=apply,
        batch=batch,
    )


def prepare(contexts, horizon: int, *, use_symmetric_averaging=False, make_positive=False, max_context=MAX_CONTEXT):
    """``predict_batch`` (one batch) up to the network input."""
    batch = format_batch(
        contexts,
        horizon,
        use_symmetric_averaging=use_symmetric_averaging,
        make_positive=make_positive,
        max_context=max_context,
    )
    return model_inputs(batch.target, batch.mask, batch.horizon, batch=batch)


def _merge(n, mu, sigma, x):
    """Chan merge of an unmasked patch ``x`` (rows, p) into (n, mu, sigma): ``util.update_running_stats``."""
    k = float(x.shape[-1])
    m = x.mean(-1)
    s2 = ((x - m[..., None]) ** 2).mean(-1)
    new_n = n + k
    new_mu = (n * mu + k * m) / new_n
    new_var = (n * sigma * sigma + k * s2 + n * (mu - new_mu) ** 2 + k * (m - new_mu) ** 2) / new_n
    return new_n, new_mu, torch.sqrt(new_var)


def readout_stats(raw: torch.Tensor, host: HostInputs):
    """De-normalization (mu, sigma) at the read-out patches, with the iterative CPM RevIN refine.

    ``raw``: (rows, F, 576) head output at the read-out tokens, normalized space. The first read-out is the
    last context patch (actual stats); the others are CPM patches, whose stats merge the model's own
    de-normalized medians of the preceding horizon patches (``cpm_revin_refine.py``). Every anchor token
    the refine reads is itself a read-out token.
    """
    raw = raw.to(torch.float64)
    p0 = host.readout_start
    f = raw.shape[1]
    med = raw.reshape(raw.shape[0], f, ROLLS, PATCH, NUM_QUANTILES)[..., MEDIAN_QUANTILE]
    n, mu, sigma = host.running_n[:, p0], host.running_mu[:, p0], host.running_sigma[:, p0]
    mus, sigmas = [mu], [sigma]
    anchor = (med[:, 0] * sigma[:, None, None] + mu[:, None, None]).clamp(-VALUE_CLIP, VALUE_CLIP)
    offset = 0
    for j in range(1, f):
        n, mu, sigma = _merge(n, mu, sigma, anchor[:, offset])
        mus.append(mu)
        sigmas.append(sigma)
        offset = (offset + 1) % ROLLS
        if offset == 0:
            anchor = (med[:, j] * sigma[:, None, None] + mu[:, None, None]).clamp(-VALUE_CLIP, VALUE_CLIP)
    return torch.stack(mus, 1), torch.stack(sigmas, 1)


def stitch(preds: torch.Tensor) -> torch.Tensor:
    """``util.stitch_patches``: (rows, F, 64, q) -> (rows, (F + 1) * 32, q), 32-step linear cross-fades."""
    f = preds.shape[1]
    if f == 1:
        return preds[:, 0]
    w = torch.linspace(1.0, 0.0, STITCH_OVERLAP, dtype=preds.dtype)[None, None, :, None]
    first = preds[:, 0, :PATCH]
    over = w * preds[:, :-1, PATCH:] + (1.0 - w) * preds[:, 1:, :STITCH_OVERLAP]
    mid = torch.cat([over, preds[:, 1:, STITCH_OVERLAP:PATCH]], 2).reshape(preds.shape[0], -1, preds.shape[-1])
    return torch.cat([first, mid, preds[:, -1, PATCH:]], 1)


def horizon_quantiles(raw: torch.Tensor, host: HostInputs) -> torch.Tensor:
    """``decode``'s return value: (items, u, H, 9) data-space quantiles, unsorted, with the trend added."""
    mu, sigma = readout_stats(raw, host)
    y = raw.to(torch.float64).reshape(raw.shape[0], raw.shape[1], OUTPUT_PATCH, NUM_QUANTILES)
    y = (y * sigma[..., None, None] + mu[..., None, None]).clamp(-VALUE_CLIP, VALUE_CLIP)
    y = stitch(y[:, :, :STITCH_EXTRACT])[:, : host.horizon]
    t = torch.arange(1, host.horizon + 1, dtype=torch.float64) / host.context_len
    trend = host.trend_m[:, None] * t[None, :] + host.trend_c[:, None]
    y = y + torch.where(host.trend_apply[:, None], trend, 0.0)[..., None]
    return y.reshape(host.items, host.variates, host.horizon, NUM_QUANTILES)


def postprocess(raw: torch.Tensor, host: HostInputs, *, sort_quantiles: bool = True) -> np.ndarray:
    """The forecaster's quantiles: (items, H, 9) for 1-D inputs, else (items, u, H, 9); float64."""
    batch = host.batch
    y = horizon_quantiles(raw, host).numpy()
    if sort_quantiles:
        y = np.sort(y, axis=-1)
    if batch is not None and batch.symmetric:
        y = (y[0::2] - y[1::2][..., ::-1]) / 2
    if batch is not None and batch.make_positive:
        y = np.where(batch.nonnegative[:, :, None, None], np.maximum(y, 0.0), y)
    h = host.horizon if batch is None else batch.requested_horizon
    y = y[:, :, :h]
    if batch is not None and batch.was_1d:
        y = y[:, 0]
    return y
