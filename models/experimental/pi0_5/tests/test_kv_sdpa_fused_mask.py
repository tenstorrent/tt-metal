# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Differential tests for kv_sdpa's FUSED attention-mask path.

These exist because the mask path was silently lost in a merge and nothing caught it: the 16-chip perf
gate feeds every camera present, so no prefix column is ever invalid and the maskless fast path is the
only one it exercises. The single-layer PCC gate is barely better -- with all-real inputs the pi0.5
expert mask only blocks the phantom suffix tail, so masked and unmasked agree to 6 decimal places and a
dropped mask looks fine. A mask test has to supply a mask that CHANGES THE ANSWER.

Assertions are kv_sdpa-vs-kv_sdpa (or invariance under perturbing masked inputs), which keeps them
independent of absolute scaling. An absolute torch oracle IS also valid for this op -- see the correction
below -- and is pinned here for each prefix/suffix/mask phase.

CORRECTION (2026-07-30). An earlier version of this file asserted that a standalone torch reference
"does not match the op's contract", citing scratchpad/kvs_bench.py's PCC ~0.38 for stock kv_sdpa. That was
WRONG, twice over: first blamed on a bad reference, then on harness sensitivity. 0.38 was a REAL BUG --
flash_fused.cpp ran the QK matmul with transpose=false, so each K tile's contents were never transposed
(only the tile grid was). kvs_bench.py had been correctly reporting a genuine defect all along. Tiny-Q
then exposed two more defects: whole-tile SFPU traversal used the four-face RC mode on a two-face 16x32
score tile, and the row-sum identity was incorrectly 16x32, summing only columns 0..15 of 32-column prefix
scores. Mixed 32x32-prefix/16x32-suffix matmul also hits tt-metal's documented tile-descriptor
reconfiguration gap (#46769); flash_fused now explicitly reprograms unpack dimensions/stride whenever
it switches K/V sources rather than promoting the model's tiny Q/suffix tensors.

  test_tile_aligned_mask_matches_tile_skipping
      A mask that blocks WHOLE prefix tiles must agree with dropping those tiles via
      prefix_valid_tiles. Two independent mechanisms, same math -- and both are kv_sdpa, so any
      disagreement is in the mask handling itself.

  test_ragged_mask_matches_general_sdpa
      A mask that blocks a PARTIAL tile (which tile skipping fundamentally cannot express -- this is
      exactly LIBERO's prompt ending mid-tile) must agree with the general
      ttnn.transformer.scaled_dot_product_attention given the same mask.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0_5.tt.tile_config import TILE_HEIGHT
from models.experimental.pi0_5.tt._ttnn_compat import kv_sdpa as pi05_kv_sdpa

# pi0.5 decode expert-attention shape: MQA, 8 Q heads / 1 KV head, head_dim 256, a 1024-row resident
# prefix at a 32-row tile and a single suffix K tile at the model tile height.
NQH, NKH, HD = 8, 1, 256
PREFIX = 1024
_MASK_VAL = -1e4
_L1 = ttnn.L1_MEMORY_CONFIG


def _mask_to_tt(mask_2d, device, tile_h):
    """[Sq, KV] additive mask -> [1, 1, Sq, KV] bf16 TILE tensor at the q tile height."""
    return ttnn.from_torch(
        mask_2d.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        tile=ttnn.Tile((tile_h, 32)),
        memory_config=_L1,
    )


def _inputs(device, *, prefix_mutator=None):
    """(q, k, v, past_k, past_v) at the production dtypes/tiles, plus the torch suffix rows."""
    sq = TILE_HEIGHT
    torch.manual_seed(0)
    q_t = torch.randn(1, NQH, sq, HD)
    k_t = torch.randn(1, NKH, sq, HD)
    v_t = torch.randn(1, NKH, sq, HD)
    pk_t = torch.randn(1, NKH, PREFIX, HD)
    pv_t = torch.randn(1, NKH, PREFIX, HD)
    if prefix_mutator is not None:
        prefix_mutator(pk_t, pv_t)

    def to_tt(x, tile_h):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            tile=ttnn.Tile((tile_h, 32)),
            memory_config=_L1,
        )

    # Suffix rides the model tile height; the prefix is always a 32-row tile (the fused mask path
    # requires it, so prefix tile g aligns with mask column-tile g).
    return (
        to_tt(q_t, sq),
        to_tt(k_t, sq),
        to_tt(v_t, sq),
        to_tt(pk_t, 32),
        to_tt(pv_t, 32),
        sq,
    )


def _pcc(a, b):
    a, b = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _unmasked_baseline(q, k, v, pk, pv):
    """The unmasked kv_sdpa output, used as the 'mask ignored' comparison point."""
    base = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv)).float()
    assert torch.isfinite(base).all() and base.abs().max() < 1e6, (
        f"unmasked mixed-tile kv_sdpa is invalid at TILE_HEIGHT={TILE_HEIGHT}: "
        f"absmax={base.abs().max():.3g}"
    )
    return base


def test_tile_aligned_mask_matches_tile_skipping(device):
    """A whole-tile mask must equal dropping those tiles. Blocks prefix tiles 16..23 (an absent
    camera: 256 columns = 8 whole 32-wide tiles) plus the phantom suffix tail."""
    q, k, v, pk, pv, sq = _inputs(device)
    kv_total = PREFIX + 32  # one suffix column-tile

    blocked = list(range(16, 24))
    mask = torch.zeros(sq, kv_total, dtype=torch.bfloat16)
    for t in blocked:
        mask[:, t * 32 : (t + 1) * 32] = _MASK_VAL

    valid = [t for t in range(PREFIX // 32) if t not in blocked]

    unmasked = _unmasked_baseline(q, k, v, pk, pv)
    masked = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=_mask_to_tt(mask, device, sq)))
    skipped = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, prefix_valid_tiles=valid))

    pcc = _pcc(masked, skipped)
    print(f"\n[fused mask] tile-aligned mask vs tile skipping: PCC={pcc:.6f}  (TILE_HEIGHT={sq})")
    # Not bit-identical: masking adds -1e4 then exponentiates (a true zero only in the limit), while
    # skipping never reads the tile at all. They must agree to well within bf8 noise.
    assert pcc > 0.999, f"fused mask disagrees with tile skipping: PCC={pcc}"

    # And it must actually DIFFER from ignoring the mask -- otherwise this test proves nothing.
    pcc_vs_unmasked = _pcc(masked, unmasked)
    print(f"[fused mask] masked vs UNMASKED: PCC={pcc_vs_unmasked:.6f}  (must be clearly < 1)")
    assert pcc_vs_unmasked < 0.99, (
        f"masking 8 of 32 prefix tiles changed nothing (PCC={pcc_vs_unmasked}) -- the mask is being "
        "ignored, which is exactly the regression this test exists to catch"
    )


def test_partial_tile_mask_is_strictly_between_open_and_blocked(device):
    """Validate PARTIAL-tile masking -- what tile skipping fundamentally cannot express, and what LIBERO
    actually needs (its prompt ends mid-tile, at prefix tile 24).

    Phrased as a bracket so it stays valid in a harness where absolute PCC is not (see
    _unmasked_baseline): masking HALF of tile 24 must land strictly between leaving that tile fully open
    and blocking it entirely, and must differ from both. A mask that silently ignored the partial
    columns would coincide with 'open'; one that over-masked the whole tile would coincide with
    'blocked'. Only correct per-column masking sits in between.
    """
    q, k, v, pk, pv, sq = _inputs(device)
    kv_total = PREFIX + 32
    _unmasked_baseline(q, k, v, pk, pv)

    def run(partial_cols):
        # Tiles 16..23 blocked (absent camera) in every variant; tile 24 gets `partial_cols` blocked.
        m = torch.zeros(sq, kv_total, dtype=torch.bfloat16)
        m[:, 16 * 32 : 24 * 32] = _MASK_VAL
        if partial_cols:
            m[:, 24 * 32 + (32 - partial_cols) : 25 * 32] = _MASK_VAL
        return ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=_mask_to_tt(m, device, sq))).float()

    open_t, half, blocked = run(0), run(16), run(32)

    d_open = (half - open_t).abs().mean().item()
    d_blocked = (half - blocked).abs().mean().item()
    span = (blocked - open_t).abs().mean().item()
    print(
        f"\n[fused mask] partial tile: |half-open|={d_open:.6g}  |half-blocked|={d_blocked:.6g}  "
        f"|blocked-open|={span:.6g}  (TILE_HEIGHT={sq})"
    )
    assert span > 0, "blocking a whole prefix tile changed nothing -- the mask is being ignored"
    # Strictly between: closer to each endpoint than the endpoints are to each other.
    assert d_open > 0.05 * span, (
        f"masking 16 of tile 24's 32 columns is indistinguishable from leaving it OPEN "
        f"(|half-open|={d_open:.6g} vs span={span:.6g}) -- partial columns are being dropped"
    )
    assert d_blocked > 0.05 * span, (
        f"masking 16 of tile 24's 32 columns is indistinguishable from blocking it ENTIRELY "
        f"(|half-blocked|={d_blocked:.6g} vs span={span:.6g}) -- the whole tile is being over-masked"
    )


def test_partial_tile_mask_blocks_the_requested_columns(device):
    """Changing only masked K/V rows must not change the output.

    The open/half/blocked bracket proves that a partial mask has an effect, but cannot detect a
    within-tile column permutation: masking the wrong 22 columns still lands between the endpoints.
    This mirrors LIBERO's tile 24, where ten prompt tokens are real and the remaining 22 are padding.
    """
    prompt_end = 24 * 32 + 10

    q, k, v, pk, pv, sq = _inputs(device)
    mask = torch.zeros(sq, PREFIX + 32, dtype=torch.bfloat16)
    mask[:, prompt_end:PREFIX] = _MASK_VAL
    m_tt = _mask_to_tt(mask, device, sq)
    reference = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=m_tt)).float()
    assert torch.isfinite(reference).all(), f"masked mixed-tile kv_sdpa is non-finite at TILE_HEIGHT={TILE_HEIGHT}"

    def mutate_masked(pk_t, pv_t):
        pk_t[:, :, prompt_end:PREFIX, :] = torch.randn_like(pk_t[:, :, prompt_end:PREFIX, :]) * 100
        pv_t[:, :, prompt_end:PREFIX, :] = torch.randn_like(pv_t[:, :, prompt_end:PREFIX, :]) * 100

    q2, k2, v2, pk2, pv2, _ = _inputs(device, prefix_mutator=mutate_masked)
    changed_masked = ttnn.to_torch(ttnn.kv_sdpa(q2, k2, v2, past_k=pk2, past_v=pv2, attn_mask=m_tt)).float()
    pcc = _pcc(reference, changed_masked)
    mae = (reference - changed_masked).abs().mean().item()
    print(f"\n[fused mask] mutate masked columns: PCC={pcc:.6f} MAE={mae:.6g}")
    assert pcc > 0.9999 and mae < 1e-3, (
        f"changing masked K/V rows changed the result (PCC={pcc}, MAE={mae}); "
        "the partial mask is landing on different within-tile columns"
    )


@pytest.mark.parametrize("mask_mode", ["unmasked", "prefix-only", "suffix-only", "partial"])
def test_absolute_torch_phase_isolation(device, mask_mode, monkeypatch):
    """Pin absolute correctness and identify which folded-KV phase first diverges."""
    monkeypatch.setenv("PI05_FUSED_KV_MASK", "1")
    q, k, v, pk, pv, sq = _inputs(device)
    kv_total = PREFIX + 32
    mask = torch.zeros(sq, kv_total, dtype=torch.bfloat16)
    if mask_mode == "prefix-only":
        mask[:, PREFIX:] = _MASK_VAL
    elif mask_mode == "suffix-only":
        mask[:, :PREFIX] = _MASK_VAL
        mask[:, PREFIX + sq :] = _MASK_VAL
    elif mask_mode == "partial":
        mask[:, 16 * 32 : 24 * 32] = _MASK_VAL
        mask[:, 24 * 32 + 10 : PREFIX] = _MASK_VAL
        mask[:, PREFIX + sq :] = _MASK_VAL

    kwargs = {}
    if mask_mode != "unmasked":
        kwargs["attn_mask"] = _mask_to_tt(mask, device, sq)
    actual = ttnn.to_torch(pi05_kv_sdpa(q, k, v, past_k=pk, past_v=pv, **kwargs)).float()

    q_t = ttnn.to_torch(q).float()
    k_t = torch.cat([ttnn.to_torch(pk).float(), ttnn.to_torch(k).float()], dim=2)
    v_t = torch.cat([ttnn.to_torch(pv).float(), ttnn.to_torch(v).float()], dim=2)
    torch_mask = None if mask_mode == "unmasked" else mask[:, : PREFIX + sq].unsqueeze(0).unsqueeze(0).float()
    expected = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=torch_mask)

    finite = bool(torch.isfinite(actual).all())
    pcc = _pcc(actual, expected) if finite else float("nan")
    mae = (actual - expected).abs().mean().item() if finite else float("inf")
    print(
        f"\n[absolute {mask_mode}] tile={sq} finite={finite} PCC={pcc:.6f} MAE={mae:.6g} "
        f"actual_absmax={actual.abs().max().item():.6g}"
    )
    assert finite and pcc > 0.999 and mae < 1e-2, (
        f"{mask_mode} kv_sdpa diverges from torch: finite={finite}, PCC={pcc}, MAE={mae}"
    )


def test_absolute_attention_probabilities(device):
    """Use V=I so the output directly exposes the computed attention probabilities."""
    sq = TILE_HEIGHT
    torch.manual_seed(11)
    q_t = torch.randn(1, 1, sq, 32, dtype=torch.bfloat16)
    k_t = torch.randn(1, 1, sq, 32, dtype=torch.bfloat16)
    v_t = torch.eye(sq, 32, dtype=torch.bfloat16).reshape(1, 1, sq, 32)

    def upload(x):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((sq, 32)),
            device=device,
            memory_config=_L1,
        )

    q, k, v = upload(q_t), upload(k_t), upload(v_t)
    actual = ttnn.to_torch(ttnn.kv_sdpa(q, k, v)).float()
    expected = F.scaled_dot_product_attention(
        ttnn.to_torch(q).float(), ttnn.to_torch(k).float(), ttnn.to_torch(v).float()
    )
    pcc = _pcc(actual, expected)
    mae = (actual - expected).abs().mean().item()
    row_sum = actual[..., :sq].sum(dim=-1)
    print(
        f"\n[absolute probabilities] tile={sq} PCC={pcc:.6f} MAE={mae:.6g} "
        f"row_sum=[{row_sum.min().item():.6g},{row_sum.max().item():.6g}] "
        f"tail_absmax={actual[..., sq:].abs().max().item() if sq < 32 else 0:.6g}"
    )
    print(f"expected row0={expected[0, 0, 0, :sq].tolist()}")
    print(f"actual row0={actual[0, 0, 0, :sq].tolist()}")
    assert pcc > 0.999 and mae < 1e-3


def test_absolute_attention_probabilities_two_chunks(device):
    """V=I with two one-tile chunks isolates the online-softmax combine."""
    sq = TILE_HEIGHT
    kv_len = 2 * sq
    torch.manual_seed(13)
    q_t = torch.randn(1, 1, sq, 32, dtype=torch.bfloat16)
    k_t = torch.randn(1, 1, kv_len, 32, dtype=torch.bfloat16)
    v_t = torch.eye(kv_len, 32, dtype=torch.bfloat16).reshape(1, 1, kv_len, 32)

    def upload(x):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((sq, 32)),
            device=device,
            memory_config=_L1,
        )

    q, k, v = upload(q_t), upload(k_t), upload(v_t)
    actual = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, max_kv_chunk_tiles=1)).float()
    expected = F.scaled_dot_product_attention(q_t.float(), k_t.float(), v_t.float())
    finite = bool(torch.isfinite(actual).all())
    pcc = _pcc(actual, expected) if finite else float("nan")
    mae = (actual - expected).abs().mean().item() if finite else float("inf")
    print(f"\n[absolute probabilities 2 chunks] tile={sq} finite={finite} PCC={pcc:.6f} MAE={mae:.6g}")
    assert finite and pcc > 0.999 and mae < 1e-3


def test_absolute_attention_probabilities_mixed_tiles(device):
    """V=I for the production tiny-Q/full-height-prefix geometry."""
    sq = TILE_HEIGHT
    torch.manual_seed(17)
    q_t = torch.randn(1, 1, sq, 32, dtype=torch.bfloat16)
    k_t = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    v_t = torch.eye(32, dtype=torch.bfloat16).reshape(1, 1, 32, 32)

    def upload(x, tile_h):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((tile_h, 32)),
            device=device,
            memory_config=_L1,
        )

    q, k, v = upload(q_t, sq), upload(k_t, 32), upload(v_t, 32)
    actual = ttnn.to_torch(ttnn.kv_sdpa(q, k, v)).float()
    expected = F.scaled_dot_product_attention(q_t.float(), k_t.float(), v_t.float())
    finite = bool(torch.isfinite(actual).all())
    pcc = _pcc(actual, expected) if finite else float("nan")
    mae = (actual - expected).abs().mean().item() if finite else float("inf")
    print(f"\n[absolute probabilities mixed] q_tile={sq} finite={finite} PCC={pcc:.6f} MAE={mae:.6g}")
    print(f"expected row0={expected[0, 0, 0].tolist()}")
    print(f"actual row0={actual[0, 0, 0].tolist()}")
    assert finite and pcc > 0.999 and mae < 1e-3


def test_absolute_wide_prefix_chunks(device):
    """Match the production prefix's two 16-tile chunks without the suffix phase."""
    sq = TILE_HEIGHT
    kv_len = 1024
    torch.manual_seed(19)
    q_t = torch.randn(1, 1, sq, 32, dtype=torch.bfloat16)
    k_t = torch.randn(1, 1, kv_len, 32, dtype=torch.bfloat16)
    v_t = torch.randn(1, 1, kv_len, 32, dtype=torch.bfloat16)

    def upload(x, tile_h):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((tile_h, 32)),
            device=device,
            memory_config=_L1,
        )

    q, k, v = upload(q_t, sq), upload(k_t, 32), upload(v_t, 32)
    actual = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, max_kv_chunk_tiles=16)).float()
    expected = F.scaled_dot_product_attention(q_t.float(), k_t.float(), v_t.float())
    finite = bool(torch.isfinite(actual).all())
    pcc = _pcc(actual, expected) if finite else float("nan")
    mae = (actual - expected).abs().mean().item() if finite else float("inf")
    print(f"\n[absolute wide prefix] tile={sq} finite={finite} PCC={pcc:.6f} MAE={mae:.6g}")
    assert finite and pcc > 0.99 and mae < 1e-2


def test_absolute_wide_prefix_production_dtype(device):
    """Production bf8/HD256 prefix geometry without the folded suffix."""
    sq = TILE_HEIGHT
    kv_len = 1024
    torch.manual_seed(23)
    q_t = torch.randn(1, NQH, sq, HD)
    k_t = torch.randn(1, NKH, kv_len, HD)
    v_t = torch.randn(1, NKH, kv_len, HD)

    def upload(x, tile_h):
        return ttnn.from_torch(
            x,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((tile_h, 32)),
            device=device,
            memory_config=_L1,
        )

    q, k, v = upload(q_t, sq), upload(k_t, 32), upload(v_t, 32)
    actual = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, max_kv_chunk_tiles=64)).float()
    expected = F.scaled_dot_product_attention(
        ttnn.to_torch(q).float(), ttnn.to_torch(k).float(), ttnn.to_torch(v).float()
    )
    finite = bool(torch.isfinite(actual).all())
    pcc = _pcc(actual, expected) if finite else float("nan")
    mae = (actual - expected).abs().mean().item() if finite else float("inf")
    print(f"\n[absolute wide bf8] tile={sq} finite={finite} PCC={pcc:.6f} MAE={mae:.6g}")
    assert finite


def test_mask_composes_with_prefix_tile_skipping(device):
    """Mask AND prefix_valid_tiles together -- the combination the pi0.5 denoise block actually passes.

    LIBERO's mask blocks prefix tiles 16..23 (absent camera), part of 24 (prompt ends mid-tile), and
    25..31 (language padding), while the pipeline independently hands kv_sdpa a prefix_valid_tiles list
    that DROPS the wholly-invalid tiles. Skipping shortens the prefix, so the mask's column-tile g no
    longer sits at loop position g -- the reader has to indirect the mask index through the same MAPPER
    as the K/V page index. If it does not, the mask lands on the wrong columns.

    Reference is the same op with skipping disabled: identical mask, full prefix. The wholly-invalid
    tiles are already masked to -1e4 there, so dropping them must not change the result.
    """
    q, k, v, pk, pv, sq = _inputs(device)
    kv_total = PREFIX + 32
    _unmasked_baseline(q, k, v, pk, pv)

    mask = torch.zeros(sq, kv_total, dtype=torch.bfloat16)
    mask[:, 16 * 32 : 24 * 32] = _MASK_VAL  # absent camera: whole tiles
    mask[:, 24 * 32 + 16 : 25 * 32] = _MASK_VAL  # prompt ends mid-tile: PARTIAL
    mask[:, 25 * 32 : 32 * 32] = _MASK_VAL  # language padding: whole tiles
    m_tt = _mask_to_tt(mask, device, sq)

    # Drop exactly the wholly-invalid tiles; tile 24 must stay (it is only partly invalid).
    valid = [t for t in range(PREFIX // 32) if not (16 <= t < 24) and not (25 <= t < 32)]

    mask_only = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=m_tt))
    mask_and_skip = ttnn.to_torch(ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=m_tt, prefix_valid_tiles=valid))

    pcc = _pcc(mask_only, mask_and_skip)
    print(f"\n[fused mask] mask+skip vs mask-only: PCC={pcc:.6f}  (TILE_HEIGHT={sq}, kept {len(valid)}/32 tiles)")
    assert pcc > 0.999, (
        f"mask does not compose with prefix tile skipping: PCC={pcc}. The mask column index is not "
        "following the same tile mapping as K/V, so it is being applied to the wrong columns."
    )


@pytest.mark.parametrize("max_chunk", [128, 64], ids=["default-128", "model-64"])
def test_mask_across_chunk_geometries(device, max_chunk):
    """The mask must hold for the chunk geometry the MODEL actually uses, not just the default.

    The unit tests above ran at the default max_kv_chunk_tiles=128; the denoise block passes 64. With
    DHt=8 that caps a chunk at 8 tiles, and a prefix of 17 effective tiles (LIBERO's count after
    dropping wholly-invalid ones -- and prime) forces prefix_Sk_chunk_t == 1, i.e. 17 single-tile chunks.
    That regime is worth pinning independently: a 1-tile chunk is the case where add_block_inplace's
    pop/reserve/push cycle on cb_qk_im does NOT wrap back onto the tile the matmul just wrote, since the
    chunk no longer fills the whole single-buffered CB.

    Reference is the same masked call at the same chunk cap WITHOUT the mask, plus the tile-skip
    equivalent -- all kv_sdpa-vs-kv_sdpa, so valid despite the harness distortion.
    """
    q, k, v, pk, pv, sq = _inputs(device)
    kv_total = PREFIX + 32
    _unmasked_baseline(q, k, v, pk, pv)

    blocked = list(range(16, 24))
    mask = torch.zeros(sq, kv_total, dtype=torch.bfloat16)
    for t in blocked:
        mask[:, t * 32 : (t + 1) * 32] = _MASK_VAL
    valid = [t for t in range(PREFIX // 32) if t not in blocked]

    masked = ttnn.to_torch(
        ttnn.kv_sdpa(
            q, k, v, past_k=pk, past_v=pv, attn_mask=_mask_to_tt(mask, device, sq), max_kv_chunk_tiles=max_chunk
        )
    )
    skipped = ttnn.to_torch(
        ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, prefix_valid_tiles=valid, max_kv_chunk_tiles=max_chunk)
    )
    pcc = _pcc(masked, skipped)
    print(f"\n[fused mask] max_kv_chunk_tiles={max_chunk}: mask vs skip PCC={pcc:.6f}  (TILE_HEIGHT={sq})")
    assert pcc > 0.999, f"fused mask breaks at max_kv_chunk_tiles={max_chunk}: PCC={pcc}"


def test_mask_with_single_tile_chunks(device):
    """Force the 1-tile-per-chunk regime explicitly: a prime effective prefix under a tight cap."""
    q, k, v, pk, pv, sq = _inputs(device)
    kv_total = PREFIX + 32
    _unmasked_baseline(q, k, v, pk, pv)

    # 17 valid prefix tiles (prime) -> prefix_Sk_chunk_t == 1 whatever the cap. Mask tiles 16..23 and
    # 25..31 (matching what is dropped) plus HALF of tile 24, so a partial tile rides along.
    valid = [t for t in range(PREFIX // 32) if not (16 <= t < 24) and not (25 <= t < 32)]
    assert len(valid) == 17
    mask = torch.zeros(sq, kv_total, dtype=torch.bfloat16)
    mask[:, 16 * 32 : 24 * 32] = _MASK_VAL
    mask[:, 24 * 32 + 16 : 25 * 32] = _MASK_VAL
    mask[:, 25 * 32 : 32 * 32] = _MASK_VAL
    m_tt = _mask_to_tt(mask, device, sq)

    a = ttnn.to_torch(
        ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=m_tt, prefix_valid_tiles=valid, max_kv_chunk_tiles=64)
    )
    b = ttnn.to_torch(
        ttnn.kv_sdpa(q, k, v, past_k=pk, past_v=pv, attn_mask=m_tt, prefix_valid_tiles=valid, max_kv_chunk_tiles=128)
    )
    pcc = _pcc(a, b)
    print(f"\n[fused mask] 17 single-tile chunks (cap 64) vs cap 128: PCC={pcc:.6f}  (TILE_HEIGHT={sq})")
    assert pcc > 0.999, f"fused mask is chunk-geometry dependent: PCC={pcc}"
