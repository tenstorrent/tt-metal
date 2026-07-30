# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 7: production sequence length, no host fallback.

Every other test in this module runs `T = 64`, which fits in a couple of tile rows
and hides everything shape-related. Production prefill is `T = 5120`, where
`block_residual` alone is 560 MiB and one read touches ~7.3 GB of DRAM traffic.

The sharpest gate here needs no host reference at all. Every reduction in the op is
over `d` or over the candidate axis, never over `T`, so `T` is a pure batch axis and
a token slice of the production run must reproduce the `T = 64` run **bit for bit**.
Any padding leak, wrong reduction axis, or `T`-dependent work split breaks that
equality, and it costs one extra device run instead of a 10-minute fp32 walk.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import BLOCK_SIZE, EPS, NUM_LAYERS, attn_res
from models.experimental.kimi_k3_attn_res.tests.test_tt_attn_res_depth import _make_stack, _walk_device
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes

PCC_GATE = 0.9999
STAT_REL_TOL = 2e-2
PROJ_STD = 0.02

HIDDEN_SIZE = 7168
PRODUCTION_TOKENS = 5120
SMALL_TOKENS = 64
READ_SITES = 24  # read sites per 12-layer block, the real per-block count


def _pcc(a, b):
    stacked = torch.stack((a.double().reshape(-1), b.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _rel_err(got, want):
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


def _make_case(num_tokens, num_sealed, seed=0):
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(num_tokens, HIDDEN_SIZE),
        randn(num_tokens, num_sealed, HIDDEN_SIZE),
        (1.0 + 0.1 * randn(HIDDEN_SIZE)) * (PROJ_STD * randn(HIDDEN_SIZE)),
    )


def _to_device(mesh_device, prefix_sum, block_residual, query, op):
    to_tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)) if block_residual.shape[1] else None,
        op.to_query(query),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_sealed", [0, 8])
def test_production_forward_matches_torch(mesh_device, num_sealed):
    """`T = 5120`, `S = 8`: 560 MiB of snapshots, 660 MiB of concatenated candidates."""
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, num_sealed)
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE)
    tt_prefix, tt_block, tt_query = _to_device(mesh_device, prefix_sum, block_residual, query, op)

    out = op.forward(tt_prefix, tt_block, tt_query)
    got = ttnn.to_torch(out).reshape(PRODUCTION_TOKENS, HIDDEN_SIZE)
    ttnn.deallocate(out)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    pcc, rel_err = _pcc(got, want), _rel_err(got, want)
    logger.info(f"S={num_sealed}: production forward PCC {pcc:.7f}, rel err {rel_err:.2e}")
    assert pcc >= PCC_GATE, f"S={num_sealed}: production PCC {pcc:.7f} < {PCC_GATE}"
    assert rel_err <= STAT_REL_TOL, f"S={num_sealed}: production rel err {rel_err:.2e} > {STAT_REL_TOL}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_production_split_matches_forward(mesh_device):
    """All 24 read sites of a block through the split form, at production `T`.

    `inter_block` returns before any `merge` runs, so 24 partials of `[1, 1, T, d]`
    coexist — 1.7 GiB on top of the snapshots. That co-residency is the split form's
    real cost and this is the first shape where it is worth stating."""
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, 8)
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE)
    tt_prefix, tt_block, tt_query = _to_device(mesh_device, prefix_sum, block_residual, query, op)

    direct = op.forward(tt_prefix, tt_block, tt_query)
    want = ttnn.to_torch(direct).reshape(PRODUCTION_TOKENS, HIDDEN_SIZE)
    ttnn.deallocate(direct)

    partials, shifts, masses = op.inter_block(tt_block, [tt_query] * READ_SITES)
    for read_site, (partial, shift, mass) in enumerate(zip(partials, shifts, masses)):
        merged = op.merge(partial, shift, mass, tt_prefix, tt_query)
        got = ttnn.to_torch(merged).reshape(PRODUCTION_TOKENS, HIDDEN_SIZE)
        ttnn.deallocate(merged)
        pcc = _pcc(got, want)
        assert pcc >= PCC_GATE, f"read site {read_site}: split PCC {pcc:.7f} < {PCC_GATE}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_tokens", [1000, 5119])
def test_ragged_token_count_matches_torch(mesh_device, num_tokens):
    """Token counts that are not multiples of 32, which real prompts are not.

    `T` tile-pads to the next multiple of 32 — 1000 -> 1024, 5119 -> 5120 — so the
    padded rows are real tiles carrying whatever the allocator left there. Nothing
    reduces over `T`, so they must stay isolated from the live rows; if any of the
    dim-1 reductions leaked across `T` this is where it would show."""
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, 8)
    prefix_sum, block_residual = prefix_sum[:num_tokens], block_residual[:num_tokens]
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE)
    tt_prefix, tt_block, tt_query = _to_device(mesh_device, prefix_sum, block_residual, query, op)

    out = op.forward(tt_prefix, tt_block, tt_query)
    got = ttnn.to_torch(out).reshape(num_tokens, HIDDEN_SIZE)
    ttnn.deallocate(out)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    pcc, rel_err = _pcc(got, want), _rel_err(got, want)
    logger.info(f"T={num_tokens}: ragged forward PCC {pcc:.7f}, rel err {rel_err:.2e}")
    assert pcc >= PCC_GATE, f"T={num_tokens}: ragged PCC {pcc:.7f} < {PCC_GATE}"
    assert rel_err <= STAT_REL_TOL, f"T={num_tokens}: ragged rel err {rel_err:.2e} > {STAT_REL_TOL}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("large_tokens", [1000, PRODUCTION_TOKENS])
def test_token_axis_is_pure_batch(mesh_device, large_tokens):
    """A token slice of the larger read equals the `T = 64` read, bit for bit.

    Gated at `max|delta| == 0`, not at a tolerance: nothing in the op reduces over
    `T`, so any difference at all is a real `T`-dependent defect rather than
    rounding. This is the one gate here that needs no host reference. `T = 1000`
    carries the same check across a tile-padded boundary."""
    prefix_sum, block_residual, query = _make_case(PRODUCTION_TOKENS, 8)
    op = TtAttnRes(mesh_device, hidden_size=HIDDEN_SIZE)

    outputs = {}
    for num_tokens in (SMALL_TOKENS, large_tokens):
        tt_prefix, tt_block, tt_query = _to_device(
            mesh_device, prefix_sum[:num_tokens], block_residual[:num_tokens], query, op
        )
        out = op.forward(tt_prefix, tt_block, tt_query)
        outputs[num_tokens] = ttnn.to_torch(out).reshape(num_tokens, HIDDEN_SIZE)
        ttnn.deallocate(out)

    delta = (outputs[large_tokens][:SMALL_TOKENS].float() - outputs[SMALL_TOKENS].float()).abs()
    logger.info(f"T={large_tokens}: token-slice max|delta| {delta.max().item():.3e} over {delta.numel()} elements")
    assert delta.max().item() == 0.0, (
        f"the shared token slice differs between T={SMALL_TOKENS} and T={large_tokens} "
        f"by up to {delta.max().item():.3e} — T is not a pure batch axis"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_production_depth_walk(mesh_device):
    """The whole 93-layer walk at production `T`, gated by the same slice equality.

    The depth harness already established fidelity at `T = 64`; what is unproven at
    `T = 5120` is that 186 reads, 8 seals and the seal-time concat all fit and stay
    shape-correct. Both walks share weights and their first 64 token rows, so the
    slice must match bit for bit after 186 chained reads — a single `T`-dependent
    slip anywhere in the walk breaks it."""
    hidden_states, q_pre, q_post, q_out, weights = _make_stack(PRODUCTION_TOKENS, HIDDEN_SIZE, NUM_LAYERS)

    outputs, sealed_after = {}, []
    for num_tokens in (SMALL_TOKENS, PRODUCTION_TOKENS):
        recorded = sealed_after if num_tokens == PRODUCTION_TOKENS else []
        outputs[num_tokens], _ = _walk_device(
            mesh_device,
            hidden_states[:num_tokens],
            weights,
            q_pre,
            q_post,
            q_out,
            HIDDEN_SIZE,
            record=lambda _, stream: recorded.append(stream.num_sealed),
        )

    production = outputs[PRODUCTION_TOKENS]
    assert torch.isfinite(production).all(), "production-shape walk diverged"
    logger.info(f"production walk output norm {production.double().norm().item():.1f}")
    assert sealed_after[0] == 1 and sealed_after[-1] == 8
    assert len(sealed_after) == NUM_LAYERS and sealed_after == sorted(sealed_after)
    assert sealed_after.count(1) == BLOCK_SIZE, "first block must hold exactly one snapshot"

    delta = (production[:SMALL_TOKENS] - outputs[SMALL_TOKENS]).abs()
    logger.info(f"depth token-slice max|delta| {delta.max().item():.3e} after {2 * NUM_LAYERS} reads")
    assert delta.max().item() == 0.0, (
        f"the shared token slice diverged over 93 layers by up to {delta.max().item():.3e} — "
        "a T-dependent defect somewhere in the walk"
    )
