# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device embedding (tt/embedding.py) and final norm + LM head (tt/lm_head.py) vs the pure-torch
ground truth (reference/lm_head.py), tenstorrent/tt-metal#65892.

Stage by stage, so a failure names where it is:
    embedding:  the row each chip looks up (host and device arithmetic) | lookup | spread to the
                hidden-split layout -- all three exact, the table is BF16 on both sides
    LM head:    gathered state | final norm | logits | end to end
    next token: top-k | greedy, on device, and the greedy token fed back to the embedding;
                ttnn.sampling over the gathered candidates (8x4 only, 32 rows: see TtHy4LMHead.sample)
    both ends:  embedding -> ihc_expand -> TtIHCHead -> final norm -> LM head -> greedy token ->
                embedding, a model with no layers
On one chip first (the arithmetic), then on the 8x4 Galaxy (heads on mesh axis 0, hidden on mesh
axis 1: tt/common.py) with both tables sharded by vocab over the 32 chips, at a small width and at
Hy4's own, and with the checkpoint's tables.

Each LM head stage is fed the reference's input for that stage, so the per-stage numbers do not
compound; the end-to-end check is the one that does. PCC does not see a scale, and a norm that
drops its 1/hidden is a pure scale error, so every float stage is also held to a relative error
(``_check_close``); one case runs the norm on readouts small enough that eps decides the result.

``test_lm_head_precision`` is the device half of the open question of the issue (does the LM head
need fp32), with the checkpoint's head. The real final hidden states need the 78 layers, which are
not on the device yet, so it runs on STAND-INS: embedding rows, peaked mixtures of LM-head rows
and random vectors (reference/lm_head.py ``stand_in_readouts``). The host half, the same formats
in exact arithmetic, is tests/test_lm_head_reference_host.py ``test_lm_head_precision_formats``.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead, ihc_expand
from models.experimental.hy4_preview.reference.lm_head import (
    Hy4Embedding,
    Hy4LMHead,
    LMHeadConfig,
    format_agreement,
    logit_agreement,
    stand_in_readouts,
)
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
)
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.embedding import TtHy4Embedding
from models.experimental.hy4_preview.tt.ihc import TtIHCHead
from models.experimental.hy4_preview.tt.ihc import ihc_expand as tt_ihc_expand
from models.experimental.hy4_preview.tt.lm_head import MAX_TOP_K, TtHy4LMHead

HY4 = LMHeadConfig()
# Hy4's structure at 1/8 of the hidden width: 128 token ids to a chip on the 8x4 mesh.
SMALL = LMHeadConfig(hidden_size=768, vocab_size=4096)
CONFIGS = {"small": SMALL, "hy4": HY4}

# name -> (head weight dtype, dtype of the normed state, logits dtype, per-stage PCC, end-to-end PCC)
# The floors follow what the Galaxy measured (2026-11-19, one chip and 8x4, random and checkpoint
# weights), as 1 - PCC: fp32 logits 1.2e-7 or less (relative error 5e-4, the HiFi4 matmul), BF16
# logits 1.4e-6 to 2.8e-6 (their own rounding), a BFP8 head 2.9e-5 to 3.0e-5 (the format's 7-bit
# mantissa). Each floor leaves about a factor of 3 over the worst of them.
VARIANTS = {
    "fp32": (ttnn.float32, ttnn.float32, ttnn.float32, 0.99999, 0.99999),
    "hf": (ttnn.float32, ttnn.bfloat16, ttnn.float32, 0.99999, 0.99999),  # HF's recipe: BF16 state, fp32 head
    "bf16-acc32": (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, 0.99999, 0.99999),
    "bf16": (ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, 0.99999, 0.99999),
    "bfp8-acc32": (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32, 0.9999, 0.9999),
    "bfp8": (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, 0.9999, 0.9999),
}
# The final norm alone: exact for a BF16 state, 8e-8 relative error in fp32 (measured).
NORM_PCC = 0.999999
# A BF16-state device against HF's recipe when the readout is not BF16-valued: HF rounds the
# readout to BF16 before the norm and the device, whose streams are fp32, does not. Measured
# 1 - PCC 5e-6, the size of HF's own rounding (2.8e-6 in test_lm_head_precision's yardstick).
ENDS_VS_HF_PCC = 0.9999


def _bf16(t):
    """Round to BF16 values, kept in fp32: what a BF16 checkpoint tensor is once loaded."""
    return t.bfloat16().float()


_CACHE = {}


def _cached(key, build):
    """Hy4's two tables are 3 GB each; build or read them once per session."""
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def _reference(cfg, seed, logit_scale=3.0):
    return _cached(("random", cfg.hidden_size, cfg.vocab_size, seed), lambda: _random_reference(cfg, seed, logit_scale))


def _random_reference(cfg, seed, logit_scale):
    """Reference embedding and head with random BF16-valued parameters at a trained scale: logits
    of a few units, a norm weight away from its trivial init."""
    g = torch.Generator().manual_seed(seed)
    emb, head = Hy4Embedding(cfg, init=False).eval(), Hy4LMHead(cfg, init=False).eval()
    with torch.no_grad():
        emb.weight.copy_(_bf16(torch.randn(emb.weight.shape, generator=g)))
        head.norm_weight.copy_(_bf16(1.0 + 0.25 * torch.randn(cfg.hidden_size, generator=g)))
        head.weight.copy_(_bf16(logit_scale * cfg.hidden_size**-0.5 * torch.randn(head.weight.shape, generator=g)))
    return emb, head


def _real_embedding():
    emb = Hy4Embedding(HY4, init=False).eval()
    (emb.weight.data,) = hy4_tensors(["model.embed_tokens.weight"])
    return emb


def _real_head():
    head = Hy4LMHead(HY4, init=False).eval()
    names = [Hy4LMHead.CHECKPOINT_NAMES[name] for name in Hy4LMHead.PARAMETER_NAMES]
    head.norm_weight.data, head.weight.data = hy4_tensors(names)
    return head


def _real_reference(names=("embedding", "head")):
    """The checkpoint's embedding and / or head (norm weight and LM head)."""
    emb = _cached("real embedding", _real_embedding) if "embedding" in names else None
    head = _cached("real head", _real_head) if "head" in names else None
    return emb, head


def _token_ids(cfg, shards, count, seed):
    """Random token ids plus both ends of the vocab and both sides of a chip boundary. A single
    token is the last id of the vocab: the last row of the last chip, and on every other chip the
    far end of the clamp."""
    ids = torch.randint(0, cfg.vocab_size, (count,), generator=torch.Generator().manual_seed(seed))
    special = [cfg.vocab_size - 1, 0]
    if shards.num_chips > 1:
        r = shards.rows_per_chip
        special += [r - 1, r, cfg.vocab_size - r - 1, cfg.vocab_size - r]
    special = special[:count]
    ids[: len(special)] = torch.tensor(special)
    return ids


def _states(cfg, tokens, seed, scale=3.0):
    """BF16-valued readouts [T, hidden], so the BF16 and fp32 recipes start from the same numbers."""
    return _bf16(scale * torch.randn(tokens, cfg.hidden_size, generator=torch.Generator().manual_seed(seed)))


def _up(plan, t, mesh_mapper=None, dtype=ttnn.float32):
    """[T, width] -> device [1, 1, T, width]."""
    return ttnn.from_torch(
        t.float().reshape(1, 1, *t.shape).contiguous(),
        layout=ttnn.TILE_LAYOUT,
        device=plan.mesh_device,
        dtype=dtype,
        mesh_mapper=mesh_mapper,
    )


def _chips(t):
    """Every chip's copy of a device tensor, as float torch tensors."""
    return [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]


def _replicated(t, label):
    """Host copy of a tensor that must be the same on every chip."""
    chips = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]
    for i, chip in enumerate(chips[1:], 1):
        assert torch.equal(chips[0], chip), f"{label}: chip {i} differs from chip 0"
    return chips[0]


def _check_close(name, ref, dev, threshold):
    """check_pcc plus the scale PCC is blind to: the relative error ||dev - ref|| / ||ref||.

    Noise that costs exactly ``threshold`` in PCC is a relative error of sqrt(2 (1 - threshold));
    the bound is 1.5 times that (0.0067 at 0.99999, 0.021 at 0.9999), so the two say the same thing
    about noise and only this one fails a result that is off by a factor."""
    check_pcc(name, ref, dev, threshold)
    a, b = ref.flatten().double(), dev.flatten().double()
    rel = ((a - b).norm() / a.norm()).item()
    bound = 1.5 * (2.0 * (1.0 - threshold)) ** 0.5
    logger.info(f"{name}: relative error {rel:.2e} (bound {bound:.2e})")
    assert rel <= bound, f"{name}: relative error {rel:.3e} above {bound:.3e}: right direction, wrong size"


def _ref_unrounded_readout(head, x):
    """The logits of the device's own recipe for a BF16 state, in exact arithmetic: the readout
    reaches the norm in fp32 (the device's streams are fp32; HF's readout is BF16 by then), the
    normalised state is cast to BF16 and takes the BF16 weight, as in HF."""
    hf = head.as_hf_bf16()
    with torch.no_grad():
        xf = x.float()
        h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + head.cfg.norm_eps)
        return hf.logits(hf.norm_weight * h.bfloat16())


def _ref_stages(head, x, hidden_dtype):
    """The reference's normed state and logits for a device variant: HF's BF16 recipe or fp32 throughout."""
    with torch.no_grad():
        if hidden_dtype == ttnn.bfloat16:
            hf = head.as_hf_bf16()
            h = hf.norm(x.bfloat16())
            return h.float(), hf.logits(h)
        h = head.norm(x)
        return h, head.logits(h)


# ---------------------------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------------------------
def _check_embedding(plan, cfg, ref, tokens, seed, label):
    tt = TtHy4Embedding(plan, cfg, ref.weight.detach())
    shards = tt.shards
    ids = _token_ids(cfg, shards, tokens, seed)
    with torch.no_grad():
        r_out = ref(ids)  # [T, hidden]
    owner = shards.chip(ids)

    # the row each chip looks up: host arithmetic, and the same on device from replicated ids
    r_rows = shards.table_rows_for(ids)  # [chips, T]
    for name, d_rows in (("host", tt.host_table_rows(ids)), ("device", tt.table_rows(tt.host_token_ids(ids)))):
        got = torch.stack([ttnn.to_torch(s).reshape(-1).long() for s in ttnn.get_device_tensors(d_rows)])
        assert torch.equal(got, r_rows), f"{label}: {name} table rows differ from the reference"
    logger.info(f"{label}: table rows exact on {shards.num_chips} chip(s), host and device arithmetic")

    # lookup: the token's row on its owner, zeros on every other chip
    d_rows = tt.table_rows(tt.host_token_ids(ids))
    looked_up = torch.stack(_chips(tt.lookup(d_rows)))[:, 0, 0]  # [chips, T, hidden]
    expected = torch.zeros_like(looked_up)
    expected[owner, torch.arange(tokens)] = r_out
    assert torch.equal(looked_up, expected), f"{label}: lookup differs (owner row or a non-zero row elsewhere)"

    # spread to the hidden-split layout, cast to the streams' dtype
    d_out = tt(d_rows)
    assert tuple(d_out.shape) == (1, 1, tokens, cfg.hidden_size // plan.hidden_tp)
    assert d_out.dtype == ttnn.float32
    out = plan.to_torch_hidden_concat(d_out, dim=-1)[0, 0].float()
    check_pcc(f"{label} embedding", r_out, out, 0.99999)
    assert torch.equal(out, r_out), f"{label}: embedding is not exact, max|Δ|={(out - r_out).abs().max().item():.3e}"
    if plan.hidden_axis is not None:  # every chip of a hidden group holds the same slice
        chips = torch.stack(_chips(d_out)).reshape(*plan.mesh_shape, tokens, -1).movedim(plan.hidden_axis, 0)
        assert torch.equal(chips, chips[:, :1].expand_as(chips)), f"{label}: chips of a hidden group differ"
    logger.info(f"{label}: embedding exact for {tokens} tokens on {shards.num_chips} chip(s)")


# ---------------------------------------------------------------------------------------------
# Final norm and LM head
# ---------------------------------------------------------------------------------------------
def _check_lm_head(plan, cfg, head, variant, x, label):
    weight_dtype, hidden_dtype, logits_dtype, pcc_stage, pcc_e2e = VARIANTS[variant]
    T = x.shape[0]
    r_h, r_logits = _ref_stages(head, x, hidden_dtype)

    tt = TtHy4LMHead(
        plan, cfg, head.weights(), weight_dtype=weight_dtype, hidden_dtype=hidden_dtype, logits_dtype=logits_dtype
    )
    d_x = _up(plan, x, plan.shard_hidden(3))

    # the readout whole on every chip
    d_whole = tt.whole(d_x)
    whole = _replicated(d_whole, f"{label} gathered state").float()[0, 0]
    assert torch.equal(whole, x), f"{label}: the gathered state is not the readout"

    # final norm, on the reference's readout
    d_h = tt.norm(_up(plan, x))
    assert d_h.dtype == hidden_dtype
    h = _replicated(d_h, f"{label} norm").float()[0, 0]
    _check_close(f"{label} norm", r_h, h, NORM_PCC)
    logger.info(f"{label} norm: max relative error {((h - r_h).abs().max() / r_h.abs().max()).item():.2e}")

    # logits, on the reference's normed state
    d_logits = tt.logits(_up(plan, r_h, dtype=hidden_dtype))
    assert tuple(d_logits.shape) == (1, 1, T, cfg.vocab_size // tt.num_chips) and d_logits.dtype == logits_dtype
    _check_close(f"{label} logits", r_logits, tt.to_torch(d_logits), pcc_stage)

    # end to end
    d_logits = tt(d_x)
    logits = tt.to_torch(d_logits)
    assert tuple(logits.shape) == (T, cfg.vocab_size)
    _check_close(f"{label} end to end", r_logits, logits, pcc_e2e)
    return tt, d_logits, logits, r_logits


def _bf16_step(v):
    """The spacing of BF16 values at ``v``: ttnn.sampling sees the logits rounded to it."""
    return torch.exp2(torch.floor(torch.log2(v.abs().clamp_min(1e-30))) - 7)


def _check_next_token(plan, cfg, tt, d_logits, logits, label, emb=None, ref_emb=None):
    """Top-k and greedy tokens against the logits read back from the device, so the check is of
    the selection alone: values exact, every id naming a logit of that value."""
    T = logits.shape[0]
    rows = torch.arange(T)[:, None]
    for k in (1, 5, MAX_TOP_K):
        d_values, d_ids = tt.top_k(d_logits, k)
        assert tuple(d_values.shape) == tuple(d_ids.shape) == (1, 1, T, k)
        values = _replicated(d_values, f"{label} top-{k} values").float()[0, 0]
        ids = _replicated(d_ids, f"{label} top-{k} ids").long()[0, 0]
        want_values, want_ids = torch.topk(logits, k, dim=-1)
        assert torch.equal(values, want_values), f"{label}: top-{k} values differ from torch.topk"
        assert torch.equal(logits[rows, ids], values), f"{label}: top-{k} ids do not name their values"
        assert all(row.unique().numel() == k for row in ids), f"{label}: a token id repeats in a top-{k}"
        same = (ids == want_ids).all(-1).float().mean().item()
        logger.info(f"{label} top-{k}: values exact; ids identical to torch.topk on {same:.4f} of {T} rows")

    d_greedy = tt.greedy(d_logits)
    assert tuple(d_greedy.shape) == (1, 1, 1, T) and d_greedy.dtype == ttnn.uint32
    greedy = _replicated(d_greedy, f"{label} greedy").long().flatten()
    assert torch.equal(logits[torch.arange(T), greedy], logits.max(-1).values), f"{label}: greedy is not an argmax"
    logger.info(f"{label} greedy: an argmax on all {T} rows, the same on every chip")

    if tt.num_chips == 1 or T != 32:
        # the shapes ttnn.sampling is not trusted with are refused before anything reaches the device
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
            tt.sample(d_logits, k=1, p=1.0, temperature=1.0, seed=1)

    if emb is not None:
        # the greedy token goes back into the embedding without leaving the device
        d_next = emb(emb.table_rows(d_greedy))
        nxt = plan.to_torch_hidden_concat(d_next, dim=-1)[0, 0].float()
        with torch.no_grad():
            assert torch.equal(nxt, ref_emb(greedy)), f"{label}: the fed-back embedding is not the greedy token's row"
        logger.info(f"{label}: greedy token embedded on device, exact for {T} rows")


def _check_sampling(tt, d_logits, logits, label):
    """ttnn.sampling over the gathered candidates. It sees BF16-rounded logits, so a draw is checked
    to within one BF16 step of the logits read back."""
    T = logits.shape[0]
    at = torch.arange(T)
    # parameters the op would read as they stand are refused before anything reaches the device
    for bad in (
        dict(k=0, p=1.0, temperature=1.0),
        dict(k=MAX_TOP_K + 1, p=1.0, temperature=1.0),
        dict(k=1, p=1.5, temperature=1.0),
        dict(k=1, p=1.0, temperature=0.0),
    ):
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
            tt.sample(d_logits, seed=1, **bad)
    # k = 1: an argmax of the BF16-rounded logits
    logger.info(f"{label}: sampling k=1, {T} row(s)")
    d_tok = tt.sample(d_logits, k=1, p=1.0, temperature=1.0, seed=1)
    assert tuple(d_tok.shape) == (1, 1, 1, T) and d_tok.dtype == ttnn.uint32
    tok = _replicated(d_tok, f"{label} sample k=1").long().flatten()
    best = logits.max(-1).values
    assert (
        logits[at, tok] >= best - _bf16_step(best)
    ).all(), f"{label}: ttnn.sampling with k=1 is not an argmax of the BF16 logits"
    logger.info(f"{label} sample k=1: argmax of the BF16 logits on all {T} rows, the same on every chip")
    # p = 0 keeps the leading candidate only, whatever k is
    d_tok = tt.sample(d_logits, k=MAX_TOP_K, p=0.0, temperature=1.0, seed=3)
    tok = _replicated(d_tok, f"{label} sample p=0").long().flatten()
    assert (logits[at, tok] >= best - _bf16_step(best)).all(), f"{label}: p=0 should keep the argmax only"
    # k = 5 and k = 32 at p = 1: every draw is one of the top k, and every chip draws the same token
    top = torch.topk(logits, MAX_TOP_K, dim=-1).values
    for k in (5, MAX_TOP_K):
        last = top[:, k - 1]
        draws = []
        for seed in (7, 8, 9):
            d_tok = tt.sample(d_logits, k=k, p=1.0, temperature=1.0, seed=seed)
            tok = _replicated(d_tok, f"{label} sample k={k} seed {seed}").long().flatten()
            in_top = logits[at, tok] >= last - _bf16_step(last)
            assert in_top.all(), f"{label}: a sampled token is outside the top {k}"
            draws.append(tok)
        # it has to be a draw: the softmax over the top 32 of these logits leaves the argmax a
        # quarter of the mass on average (0.07 to 0.71 by row, host arithmetic), so some token
        # must fall clearly below it, and two seeds must not agree everywhere
        not_greedy = torch.stack([logits[at, d] < best - _bf16_step(best) for d in draws]).float().mean().item()
        differ = torch.stack([draws[0] != d for d in draws[1:]]).float().mean().item()
        logger.info(
            f"{label} sample k={k}: all draws in the top {k}, the same on every chip; "
            f"{not_greedy:.2f} of draws are below the argmax, {differ:.2f} differ from seed 7's"
        )
        assert not_greedy > 0, f"{label}: k={k} at temperature 1 only ever returned the argmax"
        assert differ > 0, f"{label}: k={k} draws do not depend on the seed"
        d_tok = tt.sample(d_logits, k=k, p=1.0, temperature=1.0, seed=7)
        again = _replicated(d_tok, f"{label} sample k={k} seed 7 again").long().flatten()
        assert torch.equal(again, draws[0]), f"{label}: the same seed drew different tokens"
    # a cold temperature collapses the draw onto the argmax
    d_tok = tt.sample(d_logits, k=MAX_TOP_K, p=1.0, temperature=0.01, seed=7)
    tok = _replicated(d_tok, f"{label} sample cold").long().flatten()
    cold = (logits[at, tok] >= best - _bf16_step(best)).float().mean().item()
    logger.info(f"{label} sample temperature 0.01: {cold:.4f} of {T} rows draw the argmax")
    assert cold == 1.0, f"{label}: temperature 0.01 should always draw the argmax"


def _draw_intervals(top, k, p, temperature, values):
    """Where the random number of one ttnn.sampling call must lie, per row, for the op to have drawn
    a token whose BF16 logit is ``values`` [T]: (lower [T], upper [T], ambiguous [T]).

    The op as its kernels read (sampling/device/kernels: compute/sampling.cpp masks all but the top
    k and takes the softmax, dataflow/writer_interleaved.cpp draws): the softmax of the ``k``
    leading candidates at the given temperature; walk them and stop after the one that takes the
    cumulative mass above ``p``; renormalise what is kept; draw the first token whose cumulative
    share exceeds a uniform random number. ``top`` [T, 32] is the leading BF16 logits, sorted.
    Tokens of equal logit share one interval. A row is ambiguous when its cumulative mass passes
    within 0.01 of ``p`` at some token: the device's BF16 probabilities may then cut one token
    earlier or later."""
    in_k = torch.arange(top.shape[-1]) < k
    probs = torch.softmax((top.double() / temperature).masked_fill(~in_k, float("-inf")), dim=-1)
    before = probs.cumsum(-1) - probs
    keep = in_k & (before <= p)
    keep[:, 0] = True
    kept = probs * keep
    cdf = (kept / kept.sum(-1, keepdim=True)).cumsum(-1)
    above = (top > values[:, None]).sum(-1)
    at_least = (top >= values[:, None]).sum(-1)
    assert (at_least > above).all(), "a sampled token is not one of the 32 leading candidates"
    lower = torch.where(above > 0, cdf.gather(-1, (above - 1).clamp_min(0)[:, None])[:, 0], 0.0)
    upper = cdf.gather(-1, (at_least - 1)[:, None])[:, 0]
    if p < 1:
        ambiguous = (((before - p).abs() < 0.01) & in_k)[:, 1:].any(-1)
    else:
        ambiguous = torch.zeros(len(top), dtype=torch.bool)
    return lower, upper, ambiguous


def _check_model_ends(plan, cfg, ref_emb, head, ihc_params, variant, tokens, seed, label):
    """A model with no layers, end to end on device: token ids -> embedding -> 4 streams ->
    stream readout -> final norm -> LM head -> logits -> greedy token -> embedding."""
    weight_dtype, hidden_dtype, logits_dtype, _, pcc_e2e = VARIANTS[variant]
    fn, base, scale = ihc_params
    ihc_cfg = IHCConfig(dim=cfg.hidden_size, norm_eps=cfg.norm_eps)
    readout = IHCHead(ihc_cfg).eval()
    readout.fn.data, readout.base.data, readout.scale.data = fn, base, scale

    tt_emb = TtHy4Embedding(plan, cfg, ref_emb.weight.detach())
    tt_readout = TtIHCHead(plan, ihc_cfg, fn, base, scale)
    tt_head = TtHy4LMHead(
        plan, cfg, head.weights(), weight_dtype=weight_dtype, hidden_dtype=hidden_dtype, logits_dtype=logits_dtype
    )
    ids = _token_ids(cfg, tt_emb.shards, tokens, seed)
    with torch.no_grad():
        r_readout = readout(ihc_expand(ref_emb(ids)[None], ihc_cfg.n))[0]  # [T, hidden]
    _, r_logits = _ref_stages(head, r_readout, hidden_dtype)

    d_streams = tt_ihc_expand(tt_emb(tt_emb.host_table_rows(ids)), ihc_cfg.n)
    assert tuple(d_streams.shape) == (1, 1, tokens, ihc_cfg.n * cfg.hidden_size // plan.hidden_tp)
    d_readout = tt_readout(d_streams)
    _check_close(f"{label} readout", r_readout, plan.to_torch_hidden_concat(d_readout, dim=-1)[0, 0], 0.99999)
    d_logits = tt_head(d_readout)
    logits = tt_head.to_torch(d_logits)
    if hidden_dtype == ttnn.bfloat16:
        # the readout is fp32 here, not BF16-valued: against HF's recipe the device carries HF's
        # rounding of the readout as an error; against its own recipe only its arithmetic
        _check_close(f"{label} logits", r_logits, logits, ENDS_VS_HF_PCC)
        _check_close(f"{label} logits, unrounded readout", _ref_unrounded_readout(head, r_readout), logits, pcc_e2e)
    else:
        _check_close(f"{label} logits", r_logits, logits, pcc_e2e)
    top1 = (logits.argmax(-1) == r_logits.argmax(-1)).float().mean().item()
    logger.info(f"{label}: top-1 agreement with the reference {top1:.4f} over {tokens} tokens")
    # the next token chosen on device from these logits and embedded again: the decode loop's two ends
    _check_next_token(plan, cfg, tt_head, d_logits, logits, label, tt_emb, ref_emb)
    return top1


def _ihc_head_params(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(4, 4 * cfg.hidden_size, generator=g) * 0.02
    return fn, torch.randn(4, generator=g), torch.full((1,), 0.01)


def _small_readouts(cfg, tokens, seed):
    """Readouts whose mean square (1e-6) is below the norm's eps (1e-5), so eps decides the normed
    state: without it the result is 3.3 times too large. The last row is all zeros, a padded user."""
    x = _states(cfg, tokens, seed, scale=1e-3)
    x[-1] = 0.0
    assert (x.pow(2).mean(-1) < 0.2 * cfg.norm_eps).all()
    return x


def _padded_states(cfg, tokens, live, seed):
    """A batch of ``live`` users padded to ``tokens`` rows with zeros. A zero readout norms to zero
    and every one of its logits is exactly 0: the whole vocab ties, on every chip."""
    x = _states(cfg, tokens, seed)
    x[live:] = 0.0
    return x


# Every variant on a decode-sized batch, a single row for the two that matter most, and a row
# count that is not a whole tile.
LM_HEAD_CASES = [(variant, 32) for variant in VARIANTS] + [("fp32", 1), ("bf16-acc32", 1), ("bf16-acc32", 7)]
# (config, logits variant, rows): fp32 logits and BF16 logits (which tie), one row, a part tile and a batch.
NEXT_TOKEN_CASES = [
    ("small", "fp32", 1),
    ("small", "fp32", 7),
    ("small", "fp32", 32),
    ("small", "bf16", 1),
    ("small", "bf16", 32),
    ("hy4", "fp32", 1),
    ("hy4", "fp32", 32),
    ("hy4", "bf16", 32),
]
# ttnn.sampling only in the shape TtHy4LMHead.sample accepts: 32 rows, the candidates of all 32 chips.
SAMPLING_CASES = [("small", "fp32", 32), ("small", "bf16", 32), ("hy4", "fp32", 32)]


def _case_id(case):
    return "-".join(f"T{part}" if isinstance(part, int) else part for part in case)


# ---------------------------------------------------------------------------------------------
# One chip: the arithmetic, no collectives
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tokens", [1, 7, 32], ids=["T1", "T7", "T32"])
def test_embedding_one_chip(device, tokens):
    plan = MeshPlan(device, hidden_axis=None)
    emb, _ = _reference(SMALL, seed=1)
    _check_embedding(plan, SMALL, emb, tokens, 2, "one chip")


@pytest.mark.parametrize("variant, tokens", LM_HEAD_CASES, ids=[_case_id(case) for case in LM_HEAD_CASES])
def test_lm_head_one_chip(device, variant, tokens):
    cfg = SMALL
    plan = MeshPlan(device, hidden_axis=None)
    ref_emb, head = _reference(cfg, seed=1)
    tt, d_logits, logits, _ = _check_lm_head(plan, cfg, head, variant, _states(cfg, tokens, 2), "one chip")
    emb = TtHy4Embedding(plan, cfg, ref_emb.weight.detach())
    _check_next_token(plan, cfg, tt, d_logits, logits, "one chip", emb, ref_emb)


@pytest.mark.parametrize("variant", ["fp32", "bf16-acc32"])
def test_lm_head_one_chip_eps(device, variant):
    """The norm where eps decides the result (see ``_small_readouts``)."""
    cfg = SMALL
    plan = MeshPlan(device, hidden_axis=None)
    _, head = _reference(cfg, seed=1)
    _check_lm_head(plan, cfg, head, variant, _small_readouts(cfg, 32, 4), "one chip eps")


def test_next_token_one_chip_padded_rows(device):
    """Top-k and greedy on a batch whose padding rows tie over the whole vocab."""
    cfg = SMALL
    plan = MeshPlan(device, hidden_axis=None)
    _, head = _reference(cfg, seed=1)
    tt, d_logits, logits, _ = _check_lm_head(plan, cfg, head, "fp32", _padded_states(cfg, 32, 24, 2), "one chip padded")
    assert (logits[24:] == 0).all(), "a zero readout must give logits of exactly 0"
    _check_next_token(plan, cfg, tt, d_logits, logits, "one chip padded")


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy: both tables sharded by vocab over the 32 chips, the state split over the hidden axis
# (mesh axis 1, 4 chips)
# ---------------------------------------------------------------------------------------------
def _galaxy(fn):
    fn = pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)(fn)
    return pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)(fn)


def _sc1_plan(mesh_device):
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    return plan


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("tokens", [1, 7, 96], ids=["T1", "T7", "T96"])
def test_embedding_sc1(mesh_device, config, tokens):
    cfg = CONFIGS[config]
    emb, _ = _reference(cfg, seed=1)
    _check_embedding(_sc1_plan(mesh_device), cfg, emb, tokens, 2, f"sc1 {config}")


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("variant, tokens", LM_HEAD_CASES, ids=[_case_id(case) for case in LM_HEAD_CASES])
def test_lm_head_sc1(mesh_device, config, variant, tokens):
    cfg = CONFIGS[config]
    plan = _sc1_plan(mesh_device)
    _, head = _reference(cfg, seed=1)
    _check_lm_head(plan, cfg, head, variant, _states(cfg, tokens, 2), f"sc1 {config}")


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("variant", ["fp32", "bf16-acc32"])
def test_lm_head_sc1_eps(mesh_device, config, variant):
    """The norm where eps decides the result (see ``_small_readouts``)."""
    cfg = CONFIGS[config]
    _, head = _reference(cfg, seed=1)
    _check_lm_head(_sc1_plan(mesh_device), cfg, head, variant, _small_readouts(cfg, 32, 4), f"sc1 {config} eps")


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
def test_next_token_sc1_padded_rows(mesh_device, config):
    """Top-k and greedy on a batch whose padding rows tie over the whole vocab, on every chip.
    Sampling is not tried on such a batch: see TtHy4LMHead.sample."""
    cfg = CONFIGS[config]
    plan = _sc1_plan(mesh_device)
    _, head = _reference(cfg, seed=1)
    label = f"sc1 {config} padded"
    tt, d_logits, logits, _ = _check_lm_head(plan, cfg, head, "fp32", _padded_states(cfg, 32, 24, 2), label)
    assert (logits[24:] == 0).all(), "a zero readout must give logits of exactly 0"
    _check_next_token(plan, cfg, tt, d_logits, logits, label)


@_galaxy
@pytest.mark.parametrize("config, variant, tokens", NEXT_TOKEN_CASES, ids=[_case_id(c) for c in NEXT_TOKEN_CASES])
def test_next_token_sc1(mesh_device, config, variant, tokens):
    """Top-k and greedy over the sharded logits, with fp32 logits and with BF16 logits (which
    tie), and the greedy token fed back to the embedding."""
    cfg = CONFIGS[config]
    plan = _sc1_plan(mesh_device)
    ref_emb, head = _reference(cfg, seed=1)
    label = f"sc1 {config}"
    tt, d_logits, logits, _ = _check_lm_head(plan, cfg, head, variant, _states(cfg, tokens, 2), label)
    emb = TtHy4Embedding(plan, cfg, ref_emb.weight.detach())
    _check_next_token(plan, cfg, tt, d_logits, logits, label, emb, ref_emb)


@_galaxy
@pytest.mark.parametrize("config, variant, tokens", SAMPLING_CASES, ids=[_case_id(c) for c in SAMPLING_CASES])
def test_sampling_sc1(mesh_device, config, variant, tokens):
    cfg = CONFIGS[config]
    plan = _sc1_plan(mesh_device)
    _, head = _reference(cfg, seed=1)
    label = f"sc1 {config}"
    tt, d_logits, logits, _ = _check_lm_head(plan, cfg, head, variant, _states(cfg, tokens, 2), label)
    _check_sampling(tt, d_logits, logits, label)


# (k, p, temperature). Temperatures are powers of two, so the BF16 scaling of the logits is exact.
DISTRIBUTION_SETTINGS = [
    (5, 1.0, 1.0),
    (5, 1.0, 2.0),
    (5, 1.0, 0.5),
    (32, 1.0, 1.0),
    (32, 1.0, 2.0),
    (32, 1.0, 0.5),
    (32, 0.3, 1.0),
    (32, 0.7, 1.0),
    (8, 0.2, 2.0),
]
DISTRIBUTION_SEEDS = range(1, 13)


def _interval_gap(intervals):
    """How far the intervals [(lower [n], upper [n]), ...] are from having a point in common:
    (the gap, negative when they overlap; the middle of what they leave)."""
    low = max(lower.max().item() for lower, _ in intervals)
    high = min(upper.min().item() for _, upper in intervals)
    return low - high, (low + high) / 2


@_galaxy
def test_sampling_distribution_sc1(mesh_device):
    """Is a draw of ttnn.sampling the draw of the distribution asked for: top-k, top-p and
    temperature other than 1?

    The op seeds every row's core with the call's seed and draws before k, p or the temperature are
    looked at, so all rows of a call, and all calls with one seed, are drawn with one random
    number. Each token drawn then confines that number to an interval of its row's expected
    distribution (``_draw_intervals``), and for one seed the intervals of all 32 rows under all
    settings must have a point in common. A temperature applied upside down, a softmax over more
    than the top k, a top-p cut in the wrong place or a missing renormalisation moves them apart
    by 0.2 or more. What the shared number costs is logged too: within a call the users' draws
    are not independent, and neither are two calls with the same seed."""
    cfg, T = SMALL, 32
    plan = _sc1_plan(mesh_device)
    _, head = _reference(cfg, seed=1)
    label = "sc1 small"
    tt, d_logits, logits, _ = _check_lm_head(plan, cfg, head, "fp32", _states(cfg, T, 2), label)
    at = torch.arange(T)
    seen = _bf16(logits)  # ttnn.sampling takes BF16 values
    top = torch.topk(seen, MAX_TOP_K, dim=-1).values

    def unseeded(when):
        """A measurement: do the rows of a call share the random number when no seed is given?"""
        gaps = []
        for _ in range(6):
            d_tok = tt.sample(d_logits, k=MAX_TOP_K, p=1.0, temperature=1.0)
            tok = ttnn.to_torch(ttnn.get_device_tensors(d_tok)[0]).long().flatten()
            lower, upper, _ = _draw_intervals(top, MAX_TOP_K, 1.0, 1.0, seen[at, tok])
            gaps.append(_interval_gap([(lower, upper)]))
        logger.info(
            f"{label} sample without a seed, {when}: gap between the rows' intervals per call "
            f"{' '.join(f'{gap:+.3f}' for gap, _ in gaps)} (negative: one random number fits all 32 rows; "
            f"it would be {' '.join(f'{number:.3f}' for _, number in gaps)})"
        )

    unseeded("before any seeded call")
    per_setting = {setting: [] for setting in DISTRIBUTION_SETTINGS}
    ranks = {setting: [] for setting in DISTRIBUTION_SETTINGS}
    unsure = {setting: 0 for setting in DISTRIBUTION_SETTINGS}
    uncut, every, numbers = [], [], []
    for seed in DISTRIBUTION_SEEDS:
        intervals = {}
        for setting in DISTRIBUTION_SETTINGS:
            k, p, temperature = setting
            d_tok = tt.sample(d_logits, k=k, p=p, temperature=temperature, seed=seed)
            tok = ttnn.to_torch(ttnn.get_device_tensors(d_tok)[0]).long().flatten()
            lower, upper, ambiguous = _draw_intervals(top, k, p, temperature, seen[at, tok])
            unsure[setting] += int(ambiguous.sum())
            assert (~ambiguous).sum() >= T // 4, f"{label}: {setting}: too few rows with a clear top-p cut"
            intervals[setting] = (lower[~ambiguous], upper[~ambiguous])
            per_setting[setting].append(_interval_gap([intervals[setting]])[0])
            ranks[setting].append((top > seen[at, tok][:, None]).sum(-1))
        gap, number = _interval_gap([v for (_, p, _), v in intervals.items() if p == 1])
        uncut.append(gap)
        numbers.append(number)
        every.append(_interval_gap(list(intervals.values()))[0])
    for setting in DISTRIBUTION_SETTINGS:
        k, p, temperature = setting
        drawn = torch.stack(ranks[setting])
        logger.info(
            f"{label} sample k={k} p={p} temperature={temperature}: widest gap between two rows' intervals "
            f"{max(per_setting[setting]):+.4f} over {len(DISTRIBUTION_SEEDS)} seeds (negative: they overlap), "
            f"{unsure[setting]} ambiguous rows left out; mean rank drawn {drawn.float().mean().item():.2f}, "
            f"deepest {int(drawn.max())}"
        )
        assert int(drawn.max()) < k, f"{label}: k={k} drew rank {int(drawn.max())}"
    logger.info(
        f"{label} sampling distribution: widest gap per seed over the settings without a top-p cut "
        f"{max(uncut):+.4f}, over all settings {max(every):+.4f}; random numbers implied, seeds "
        f"{DISTRIBUTION_SEEDS[0]}..{DISTRIBUTION_SEEDS[-1]}: {' '.join(f'{u:.3f}' for u in numbers)}"
    )
    unseeded("after the seeded calls")
    # The op's softmax is BF16. Measured on the Galaxy: the intervals of one seed miss each other
    # by at most 0.0098 over all nine settings. A torch model of the kernel with any one of the
    # four faults named above gives 0.28 to 0.92.
    assert max(uncut) <= 0.03, f"{label}: top-k / temperature draws are {max(uncut):.4f} from one random number"
    assert max(every) <= 0.03, f"{label}: top-p draws are {max(every):.4f} from the random number of the others"


@_galaxy
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("variant", ["fp32", "bf16-acc32"])
def test_model_ends_sc1(mesh_device, config, variant):
    cfg = CONFIGS[config]
    ref_emb, head = _reference(cfg, seed=1)
    _check_model_ends(
        _sc1_plan(mesh_device), cfg, ref_emb, head, _ihc_head_params(cfg, 3), variant, 32, 2, f"sc1 {config} ends"
    )


def _timed(mesh_device, fn, runs=10):
    """Wall time of ``fn`` in ms per call, op by op from Python with no trace (an upper bound): one
    untimed call (it compiles), then ``runs`` calls against the clock. Returns (up to
    synchronize_device, up to a read of the last result from every chip). The read is queued behind
    the work, so it cannot return early; what the same read costs on an idle mesh is taken off."""

    def barrier(out):
        ttnn.synchronize_device(mesh_device)
        for shard in ttnn.get_device_tensors(out[0] if isinstance(out, (tuple, list)) else out):
            ttnn.to_torch(shard)

    barrier(fn())
    start = time.perf_counter()
    for _ in range(runs):
        out = fn()
    ttnn.synchronize_device(mesh_device)
    synced = time.perf_counter() - start
    barrier(out)
    total = time.perf_counter() - start
    start = time.perf_counter()
    barrier(out)
    idle = time.perf_counter() - start
    return synced * 1e3 / runs, (total - idle) * 1e3 / runs


@_galaxy
def test_ends_timing_sc1(mesh_device):
    """What the two ends cost per decode step at Hy4's sizes, 32 users, the default dtypes (BF16
    tables, BF16 normed state, fp32 logits): every stage on its own, then the whole turn-around
    readout -> logits -> greedy token -> embedding. A measurement: it asserts shapes only."""
    cfg, variant, T = HY4, "bf16-acc32", 32
    plan = _sc1_plan(mesh_device)
    ref_emb, ref_head = _reference(cfg, seed=1)
    weight_dtype, hidden_dtype, logits_dtype, _, _ = VARIANTS[variant]
    emb = TtHy4Embedding(plan, cfg, ref_emb.weight.detach())
    head = TtHy4LMHead(
        plan, cfg, ref_head.weights(), weight_dtype=weight_dtype, hidden_dtype=hidden_dtype, logits_dtype=logits_dtype
    )
    ids = _token_ids(cfg, emb.shards, T, 2)
    d_ids = emb.host_token_ids(ids)
    d_rows = emb.table_rows(d_ids)
    d_x = _up(plan, _states(cfg, T, 2), plan.shard_hidden(3))
    d_whole = head.whole(d_x)
    d_h = head.norm(d_whole)
    d_logits = head.logits(d_h)
    assert tuple(d_logits.shape) == (1, 1, T, cfg.vocab_size // head.num_chips)
    d_values, d_cand_ids = head.candidates(d_logits)
    assert tuple(d_values.shape) == (1, 1, T, MAX_TOP_K * head.num_chips)

    stages = {
        "embedding: row arithmetic": lambda: emb.table_rows(d_ids),
        "embedding: lookup": lambda: emb.lookup(d_rows),
        "embedding: lookup + 2 all-reduces + partition + cast": lambda: emb(d_rows),
        "head: all-gather of the readout": lambda: head.whole(d_x),
        "head: final norm": lambda: head.norm(d_whole),
        "head: matmul": lambda: head.logits(d_h),
        "next token: per-chip top 32 + 4 all-gathers": lambda: head.candidates(d_logits),
        "next token: top 1 of the gathered candidates": lambda: ttnn.topk(
            d_values, k=1, dim=-1, largest=True, sorted=True, indices_tensor=d_cand_ids
        ),
        "next token: greedy (both of the above)": lambda: head.greedy(d_logits),
        "whole turn-around: readout -> logits -> greedy -> embedding": lambda: emb(
            emb.table_rows(head.greedy(head(d_x)))
        ),
    }
    for name, fn in stages.items():
        synced, read = _timed(mesh_device, fn)
        logger.info(f"timing | {T} rows | {name}: {synced:.2f} ms per call ({read:.2f} ms up to a read of the result)")


# ---------------------------------------------------------------------------------------------
# Checkpoint weights
# ---------------------------------------------------------------------------------------------
@_galaxy
def test_embedding_sc1_real_weights(mesh_device):
    emb, _ = _real_reference(["embedding"])
    _check_embedding(_sc1_plan(mesh_device), HY4, emb, 96, 2, "sc1 real")


@_galaxy
@pytest.mark.parametrize("variant", list(VARIANTS), ids=list(VARIANTS))
def test_lm_head_sc1_real_weights(mesh_device, variant):
    """The checkpoint's final norm and head on random readouts (the real ones need the 78 layers)."""
    _, head = _real_reference(["head"])
    _check_lm_head(_sc1_plan(mesh_device), HY4, head, variant, _states(HY4, 32, 2), "sc1 real")


@_galaxy
@pytest.mark.parametrize("variant", ["fp32", "bf16-acc32"])
def test_model_ends_sc1_real_weights(mesh_device, variant):
    """Both ends with the checkpoint's embedding, stream readout, final norm and head, and the
    next token chosen and embedded on device."""
    cfg = HY4
    plan = _sc1_plan(mesh_device)
    ref_emb, head = _real_reference()
    fn, base, scale = hy4_tensors([f"model.hc_head.hc_head_{name}" for name in ("fn", "base", "scale")])
    label = "sc1 real ends"
    _check_model_ends(plan, cfg, ref_emb, head, (fn, base, scale), variant, 32, 2, label)


# ---------------------------------------------------------------------------------------------
# Does the LM head need fp32? (the open question of the issue)
# ---------------------------------------------------------------------------------------------
PRECISION_TOKENS = 256


@_galaxy
def test_lm_head_precision(mesh_device):
    """BF16 vs fp32 (and BFP8 weights) for the LM head, with the checkpoint's final norm and head.

    The reference is HF's recipe on the same readout: BF16 normed state, fp32 head. Reported per
    variant and per kind of stand-in: logit PCC, top-1 agreement, top-5 agreement as a set and in
    order, and the widest reference margin (top-1 minus top-2, 5th minus 6th) at which a
    disagreement happened. The exact fp32 arithmetic is reported against the same reference as a
    yardstick: that is how far HF's own BF16 rounding of the state moves the logits.
    """
    cfg = HY4
    plan = _sc1_plan(mesh_device)
    ref_emb, head = _real_reference()
    with torch.no_grad():
        stand_ins = stand_in_readouts(ref_emb.weight, head.weight, PRECISION_TOKENS, seed=5)
    del ref_emb

    references = {}
    for kind, x in stand_ins.items():
        _, r_logits = _ref_stages(head, x, ttnn.bfloat16)
        _, exact = _ref_stages(head, x, ttnn.float32)
        references[kind] = r_logits
        top = torch.topk(r_logits, 2, dim=-1).values
        margin = top[:, 0] - top[:, 1]
        logger.info(
            f"precision | reference | {kind}: logit std {r_logits.std().item():.3f}, top-1 logit mean "
            f"{top[:, 0].mean().item():.2f}, top-1 margin median {margin.median().item():.4f} "
            f"min {margin.min().item():.5f}"
        )
        yardstick = format_agreement(logit_agreement(r_logits, exact))
        logger.info(f"precision | host fp32 exact vs HF recipe | {kind}: {yardstick}")

    results = {}
    for variant, (weight_dtype, hidden_dtype, logits_dtype, _, _) in VARIANTS.items():
        tt = TtHy4LMHead(
            plan, cfg, head.weights(), weight_dtype=weight_dtype, hidden_dtype=hidden_dtype, logits_dtype=logits_dtype
        )
        for kind, x in stand_ins.items():
            d_x = _up(plan, x, plan.shard_hidden(3))
            logits = tt.to_torch(tt(d_x))
            a = results[variant, kind] = logit_agreement(references[kind], logits)
            logger.info(f"precision | {variant} | {kind}: {format_agreement(a)}")
        # decode-sized batch, the head alone
        d_h = _up(plan, _bf16(torch.randn(32, cfg.hidden_size)), dtype=hidden_dtype)
        synced, read = _timed(mesh_device, lambda: tt.logits(d_h))
        logger.info(
            f"precision | {variant}: head matmul, 32 rows: {synced:.2f} ms per call "
            f"({read:.2f} ms up to a read of the result)"
        )
        ttnn.deallocate(tt.weight)

    for (variant, kind), a in results.items():
        floor = VARIANTS[variant][4]
        assert a["pcc"] >= floor, f"{variant} | {kind}: logits pcc {a['pcc']:.6f} below {floor}"
