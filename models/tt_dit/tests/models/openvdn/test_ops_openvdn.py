# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN operations on device vs the CPU reference (`reference_openvdn.py`, copied from OpenVDN/vdn-minimax-h3).

Every check compares PCC and relative RMS error (RMSE / std of the reference). fp32 operations (frame statistics,
the inverse, the delta-rule factors, the scan, the gather) must reach PCC >= 0.99999 and RMSE/std <= 0.5 %; the bf16
window softmax PCC >= 0.9999 and RMSE/std <= 1 %. Keys are SiLU'd and L2-normalised as in the model; "correlated"
keys share a per-frame direction, which gives `I + A` the large condition numbers real activations do (up to
~1 + tokens per frame).

Set OPENVDN_CHECKS=<file> to append one JSON line per check.

Run (one chip): pytest models/tt_dit/tests/models/openvdn/test_ops_openvdn.py -s --tb=short
"""

from __future__ import annotations

import json
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.openvdn import linear_ops_openvdn as ops
from models.tt_dit.models.transformers.openvdn import reference_openvdn as ref
from models.tt_dit.models.transformers.openvdn.window_softmax_openvdn import WindowLayout, WindowSoftmax
from models.tt_dit.utils import tensor

FP32_PCC, FP32_RMSE = 0.99999, 0.005
BF16_PCC, BF16_RMSE = 0.9999, 0.01

HEAD_DIM = 128
TOKENS_PER_FRAME = 1008  # 24 x 42 latent grid
CHUNK, RADIUS = 5, 1  # released checkpoints: c1 window

ONE_CHIP = pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)


def _check(name: str, out: torch.Tensor, expected: torch.Tensor, *, pcc: float, rmse: float, **details) -> None:
    a = expected.detach().flatten().double()
    b = out.detach().flatten().double()
    cov = torch.cov(torch.stack([a, b]))
    pcc_found = (cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())).item()
    rmse_found = (torch.nn.functional.mse_loss(a, b).sqrt() / cov[0, 0].sqrt()).item()
    passed = not math.isnan(pcc_found) and pcc_found >= pcc and rmse_found <= rmse
    logger.info(f"{name}: PCC {pcc_found:.7f} (>= {pcc}), RMSE/std {rmse_found:.2e} (<= {rmse}) {details}")
    path = os.environ.get("OPENVDN_CHECKS")
    if path:
        with open(path, "a") as f:
            for metric, value, threshold in (("pcc", pcc_found, pcc), ("rel_rmse", rmse_found, rmse)):
                record = dict(name=name, metric=metric, value=value, threshold=threshold, passed=passed, **details)
                f.write(json.dumps(record) + "\n")
    assert passed, f"{name}: PCC {pcc_found:.7f} < {pcc} or RMSE/std {rmse_found:.2e} > {rmse}"


def _keys(shape: tuple[int, ...], kind: str, gen: torch.Generator) -> torch.Tensor:
    """[F, H, S, d] SiLU + L2-normalised features, bf16."""
    f, h, s, d = shape
    noise = torch.randn(shape, generator=gen)
    if kind == "random":
        raw = noise
    elif kind == "correlated":
        raw = torch.randn(f, h, 1, d, generator=gen) + 0.5 * noise
    elif kind == "strong":  # nearly one direction per frame: the worst conditioning
        raw = 2.0 * torch.randn(f, h, 1, d, generator=gen) + 0.1 * noise
    else:
        raise ValueError(kind)
    x = torch.nn.functional.silu(raw)
    return (x / x.norm(dim=-1, keepdim=True)).bfloat16()


def _beta(shape: tuple[int, ...], kind: str, gen: torch.Generator) -> torch.Tensor:
    if kind == "strong":
        return torch.full(shape, 0.99).bfloat16()
    return torch.sigmoid(torch.randn(shape, generator=gen)).bfloat16()


def _alpha(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    """fp32 per-frame decay in (0.8, 1), like exp(-exp(A_log) * softplus(.)) after training."""
    return 0.8 + 0.2 * torch.rand(shape, generator=gen)


def _stats(f: int, h: int, s: int, kind: str, gen: torch.Generator):
    key = _keys((f, h, s, HEAD_DIM), kind, gen)
    value = torch.randn(f, h, s, HEAD_DIM, generator=gen).bfloat16()
    beta = _beta((f, h, s), kind, gen)
    return key, value, beta


def _to_device(x: torch.Tensor, device, dtype=ttnn.float32) -> ttnn.Tensor:
    return tensor.from_torch(x, device=device, dtype=dtype)


def _alpha_rows(alpha: torch.Tensor, device) -> ttnn.Tensor:
    f, h, d = alpha.shape
    return _to_device(alpha.reshape(1, 1, f, h * d), device)


def _timed(device, fn):
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    out = fn()
    ttnn.synchronize_device(device)
    return out, time.perf_counter() - start


@ONE_CHIP
@pytest.mark.parametrize("tokens", [TOKENS_PER_FRAME, 64])
def test_frame_statistics(mesh_device, tokens):
    gen = torch.Generator().manual_seed(0)
    key, value, beta = _stats(4, 4, tokens, "correlated", gen)
    a_ref, b_ref = ref.frame_statistics(key, value, beta)

    config = ops.compute_config(mesh_device)
    (a, b), seconds = _timed(
        mesh_device,
        lambda: ops.frame_statistics(
            _to_device(key, mesh_device, ttnn.bfloat16),
            _to_device(value, mesh_device, ttnn.bfloat16),
            _to_device(beta.unsqueeze(-1), mesh_device, ttnn.bfloat16),
            config=config,
        ),
    )
    shape = f"F4 H4 S{tokens}"
    _check(f"frame_statistics_A[S{tokens}]", ttnn.to_torch(a), a_ref, pcc=FP32_PCC, rmse=FP32_RMSE, shape=shape)
    _check(f"frame_statistics_B[S{tokens}]", ttnn.to_torch(b), b_ref, pcc=FP32_PCC, rmse=FP32_RMSE, shape=shape)
    logger.info(f"frame_statistics {shape}: {seconds * 1e3:.1f} ms")


@ONE_CHIP
@pytest.mark.parametrize(("kind", "batch"), [("random", 224), ("correlated", 224), ("strong", 224), ("correlated", 5712)])
def test_spd_inverse(mesh_device, kind, batch):
    """(I + A)^-1 vs the reference's Cholesky. batch 5712 = 102 frames x 56 heads, one layer and step."""
    gen = torch.Generator().manual_seed(1)
    key, value, beta = _stats(4, 56, TOKENS_PER_FRAME, kind, gen)
    a_ref, _ = ref.frame_statistics(key, value, beta)
    a_ref = a_ref.reshape(1, 224, HEAD_DIM, HEAD_DIM)
    if batch > 224:
        a_ref = a_ref.repeat(1, math.ceil(batch / 224), 1, 1)[:, :batch]
    inv_ref = ref.vdn_inverse(a_ref)
    eig = torch.linalg.eigvalsh(a_ref.double() + torch.eye(HEAD_DIM, dtype=torch.float64))
    cond = (eig[..., -1] / eig[..., 0]).max().item()

    config = ops.compute_config(mesh_device)
    a_tt = _to_device(a_ref, mesh_device)
    max_eig = 1.0 + TOKENS_PER_FRAME
    eye = ops.identity(mesh_device, HEAD_DIM)
    inv, seconds = _timed(
        mesh_device, lambda: ops.spd_inverse(ttnn.add(a_tt, eye), max_eig=max_eig, config=config)
    )
    inv = ttnn.to_torch(inv)
    m = a_ref.double() + torch.eye(HEAD_DIM, dtype=torch.float64)
    residual = (m @ inv.double() - torch.eye(HEAD_DIM, dtype=torch.float64)).abs().max().item()
    iterations = ops.newton_schulz_iterations(max_eig)
    logger.info(f"spd_inverse {kind} B{batch}: cond {cond:.1f}, {iterations} iterations, {seconds:.2f} s")
    _check(
        f"spd_inverse[{kind}-B{batch}]",
        inv,
        inv_ref,
        pcc=FP32_PCC,
        rmse=FP32_RMSE,
        max_cond=round(cond, 1),
        max_abs_residual=residual,
        iterations=iterations,
        seconds=round(seconds, 3),
    )


@ONE_CHIP
def test_factor_apply(mesh_device):
    gen = torch.Generator().manual_seed(2)
    f, h = 6, 4
    key, value, beta = _stats(f, h, TOKENS_PER_FRAME, "correlated", gen)
    a_ref, b_ref = ref.frame_statistics(key, value, beta)
    alpha = _alpha((f, h, HEAD_DIM), gen)
    t_ref, j_ref = ref.vdn_factor_apply(alpha, a_ref, b_ref)

    config = ops.compute_config(mesh_device)
    alpha_heads = ops.rows_to_heads(_alpha_rows(alpha, mesh_device), f, h, HEAD_DIM)
    t, j = ops.vdn_factor_apply(
        alpha_heads,
        _to_device(a_ref, mesh_device),
        _to_device(b_ref, mesh_device),
        max_eig=1.0 + TOKENS_PER_FRAME,
        config=config,
    )
    _check("vdn_factor_apply_transition", ttnn.to_torch(t), t_ref, pcc=FP32_PCC, rmse=FP32_RMSE)
    _check("vdn_factor_apply_injection", ttnn.to_torch(j), j_ref, pcc=FP32_PCC, rmse=FP32_RMSE)


@ONE_CHIP
@pytest.mark.parametrize("text_state", [False, True], ids=["zero_start", "text_start"])
def test_bidirectional_scan(mesh_device, text_state):
    """102 frames, the production count: error compounds through the serial recurrence."""
    gen = torch.Generator().manual_seed(3)
    f, h = 102, 2
    key, value, beta = _stats(f, h, TOKENS_PER_FRAME, "correlated", gen)
    a_ref, b_ref = ref.frame_statistics(key, value, beta)
    t_ref, j_ref = ref.vdn_factor_apply(_alpha((f, h, HEAD_DIM), gen), a_ref, b_ref)
    start = 0.5 * j_ref[0] if text_state else None
    prefix_ref, suffix_ref = ref.run_scans(t_ref, j_ref, start)

    config = ops.compute_config(mesh_device)
    start_tt = None if start is None else _to_device(start.unsqueeze(0), mesh_device)
    (prefix, suffix), seconds = _timed(
        mesh_device,
        lambda: ops.bidirectional_scan(
            _to_device(t_ref, mesh_device), _to_device(j_ref, mesh_device), text_state=start_tt, config=config
        ),
    )
    tag = "text_start" if text_state else "zero_start"
    logger.info(f"bidirectional_scan F{f} H{h}: {seconds:.2f} s")
    _check(f"scan_prefix[{tag}]", ttnn.to_torch(prefix), prefix_ref, pcc=FP32_PCC, rmse=FP32_RMSE, frames=f)
    _check(f"scan_suffix[{tag}]", ttnn.to_torch(suffix), suffix_ref, pcc=FP32_PCC, rmse=FP32_RMSE, frames=f)


@ONE_CHIP
@pytest.mark.parametrize("text_state", [False, True], ids=["zero_start", "text_start"])
def test_gather_linear_state(mesh_device, text_state):
    """Anchor frames dense (`skip_ends`): 19 frames -> 17 inner frames with rebased c1 bounds."""
    gen = torch.Generator().manual_seed(4)
    total_frames, h = 19, 2
    bounds = [(lo - 1, hi - 1) for lo, hi in ref.window_bounds(total_frames, RADIUS, CHUNK)[1:-1]]
    f = total_frames - 2
    prefix = torch.randn(f, h, HEAD_DIM, HEAD_DIM, generator=gen)
    suffix = torch.randn(f, h, HEAD_DIM, HEAD_DIM, generator=gen)
    alpha = _alpha((f, h, HEAD_DIM), gen)
    start = torch.randn(h, HEAD_DIM, HEAD_DIM, generator=gen) if text_state else None
    expected = ref.gather_linear_state(prefix, suffix, alpha, bounds, start)

    config = ops.compute_config(mesh_device)
    out = ops.gather_linear_state(
        _to_device(prefix, mesh_device),
        _to_device(suffix, mesh_device),
        _alpha_rows(alpha, mesh_device),
        bounds,
        text_state=None if start is None else _to_device(start.unsqueeze(0), mesh_device),
        config=config,
    )
    tag = "text_start" if text_state else "zero_start"
    _check(f"gather_linear_state[{tag}]", ttnn.to_torch(out), expected, pcc=FP32_PCC, rmse=FP32_RMSE)


@ONE_CHIP
def test_linear_state_chain(mesh_device):
    """Statistics -> text state -> factors -> scans -> gather, as the branch chains them (`_readout` up to the
    query readout), with anchor frames dropped and the text state on, as the released checkpoints run."""
    gen = torch.Generator().manual_seed(5)
    total_frames, h, text_rows = 14, 2, 128
    f = total_frames - 2
    bounds = [(lo - 1, hi - 1) for lo, hi in ref.window_bounds(total_frames, RADIUS, CHUNK)[1:-1]]
    key, value, beta = _stats(f, h, TOKENS_PER_FRAME, "correlated", gen)
    t_key, t_value, t_beta = _stats(1, h, text_rows, "correlated", gen)
    alpha = _alpha((f, h, HEAD_DIM), gen)

    a_ref, b_ref = ref.frame_statistics(key, value, beta)
    ta_ref, tb_ref = ref.frame_statistics(t_key, t_value, t_beta)
    text_ref = 0.5 * ref.vdn_factor_apply(torch.ones(1, h, HEAD_DIM), ta_ref, tb_ref)[1][0]
    t_ref, j_ref = ref.vdn_factor_apply(alpha, a_ref, b_ref)
    prefix_ref, suffix_ref = ref.run_scans(t_ref, j_ref, text_ref)
    expected = ref.gather_linear_state(prefix_ref, suffix_ref, alpha, bounds, text_ref)

    config = ops.compute_config(mesh_device)
    bf16 = ttnn.bfloat16

    def run():
        a, b = ops.frame_statistics(
            _to_device(key, mesh_device, bf16),
            _to_device(value, mesh_device, bf16),
            _to_device(beta.unsqueeze(-1), mesh_device, bf16),
            config=config,
        )
        ta, tb = ops.frame_statistics(
            _to_device(t_key, mesh_device, bf16),
            _to_device(t_value, mesh_device, bf16),
            _to_device(t_beta.unsqueeze(-1), mesh_device, bf16),
            config=config,
        )
        _, text_inj = ops.vdn_factor_apply(None, ta, tb, max_eig=1.0 + text_rows, config=config)
        text = ttnn.multiply(text_inj, 0.5)
        alpha_rows = _alpha_rows(alpha, mesh_device)
        t, j = ops.vdn_factor_apply(
            ops.rows_to_heads(alpha_rows, f, h, HEAD_DIM), a, b, max_eig=1.0 + TOKENS_PER_FRAME, config=config
        )
        prefix, suffix = ops.bidirectional_scan(t, j, text_state=text, config=config)
        return ops.gather_linear_state(prefix, suffix, alpha_rows, bounds, text_state=text, config=config)

    out, seconds = _timed(mesh_device, run)
    logger.info(f"linear state chain F{f} H{h}: {seconds:.2f} s")
    _check("linear_state_chain", ttnn.to_torch(out), expected, pcc=FP32_PCC, rmse=FP32_RMSE, frames=f, heads=h)


@ONE_CHIP
@pytest.mark.parametrize(
    ("tokens", "globals_rows"), [(64, 64), (48, 40)], ids=["tile_aligned", "unaligned"]
)
def test_window_softmax(mesh_device, tokens, globals_rows):
    """c1 window with anchor frames dense both ways, over 12 frames. "unaligned" has frames and globals that are not
    whole tiles, like the production 1008-row frames."""
    gen = torch.Generator().manual_seed(6)
    f, h = 12, 2
    total = globals_rows + f * tokens
    bounds = ref.window_bounds(f, RADIUS, CHUNK)
    scale = HEAD_DIM**-0.5
    q, k, v = (torch.randn(total, h, HEAD_DIM, generator=gen).bfloat16() for _ in range(3))
    expected = ref.window_softmax(
        q.float(),
        k.float(),
        v.float(),
        video_start=globals_rows,
        num_frames=f,
        tokens_per_frame=tokens,
        bounds=bounds,
        scale=scale,
    )

    layout = WindowLayout(total=total, video_start=globals_rows, num_frames=f, tokens_per_frame=tokens)
    attention = WindowSoftmax(mesh_device, layout, bounds, scale=scale)

    def device_rows(x):
        return _to_device(x.permute(1, 0, 2).unsqueeze(0), mesh_device, ttnn.bfloat16)

    out = attention(device_rows(q), device_rows(k), device_rows(v))
    out = ttnn.to_torch(out)[0].permute(1, 0, 2)
    tag = "tile_aligned" if tokens % 32 == 0 and globals_rows % 32 == 0 else "unaligned"
    _check(
        f"window_softmax[{tag}]",
        out,
        expected,
        pcc=BF16_PCC,
        rmse=BF16_RMSE,
        tokens_per_frame=tokens,
        globals_rows=globals_rows,
        groups=len(attention.groups),
    )
