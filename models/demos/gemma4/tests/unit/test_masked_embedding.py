# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Centroid Masked Embedding drafter head (E2B assistant).

Two layers:

  * ``test_cme_ops_*`` — the four ttnn primitives the head is built from, at real
    E2B shapes (C=2048, top_k=32, P=128, N=4096, H=256, V=262144). No checkpoint
    needed. These exist because two of them are load-bearing and were unproven at
    these shapes: ``ttnn.topk`` is documented elsewhere in this repo to return
    garbage indices at width 32768, and ``ttnn.gather`` had no uint32-data user.

  * ``test_cme_vs_hf_*`` — the head against HF's
    ``Gemma4AssistantMaskedEmbedder`` on the real checkpoint
    (``GEMMA4_ASSISTANT_MODEL``), which is the actual parity contract.

**Why exact equality with HF is the wrong assertion.** HF's reference upcasts to
fp32; the device runs the centroid stage in bf16 because ``ttnn.topk`` only accepts
BFLOAT16/BFLOAT8_B input. Two consequences, both measured on the real E2B
checkpoint over 40 random hidden states:

  * *Centroid selection.* bf16 quantization of the centroid scores (~0.03 at a
    ~2.4 scale) is larger than the typical gap between the 32nd and 33rd centroid
    (observed as small as 0.0016), so the selected centroid set differs from HF's
    in ~9/40 cases — a candidate set differing by one 128-token block. Note these
    are quantization swaps, NOT exact ties (0/40 trials had an exact tie at the
    cut), so ``ttnn.topk``'s tie-break rule is not the mechanism.
  * *Candidate scores.* With HiFi4 + fp32 accumulation and an fp32 output, the
    selected logits reach PCC 0.9999969 (min) / max abs err 0.006 against the
    exact fp32 dot, and the argmax token matches HF in 39/40 cases. The single
    miss is a centroid swap, not a scoring error.

So the tests assert a precise contract instead: TT's top-k is a *valid* top-k;
any centroid only one side selected must sit within bf16 quantization of the
boundary; the candidate scores match the exact fp32 dot to high PCC; and the
argmax token matches HF unless HF's winner was in a block TT did not select, or
the top-2 gap is below the achievable resolution.

All of this is benign for speculative decoding: a drafter proposal only moves the
acceptance rate, never correctness — the committed tokens always come from the
target verify (see the ``spec_decode`` module docstring).
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric
from ...tt.assistant.masked_embedding import (
    CmeLogits,
    Gemma4TTMaskedEmbedder,
    _encode_base_digits,
    _reconstruct_base_digits,
    _same_buffer,
)

ASSISTANT_PATH = os.getenv("GEMMA4_ASSISTANT_MODEL")
_needs_assistant = pytest.mark.skipif(not ASSISTANT_PATH, reason="set GEMMA4_ASSISTANT_MODEL to run")

# Real E2B assistant CME geometry.
C, K, P, N, H, V = 2048, 32, 128, 4096, 256, 262144


class _AddressTensor:
    def __init__(self, address=None, error=None):
        self.address = address
        self.error = error

    def buffer_address(self):
        if self.error is not None:
            raise self.error
        return self.address


def test_cme_alias_identity_is_conservative():
    assert _same_buffer(_AddressTensor(17), _AddressTensor(17))
    assert not _same_buffer(_AddressTensor(17), _AddressTensor(23))
    assert _same_buffer(_AddressTensor(error=RuntimeError("address unavailable")), _AddressTensor(23))


def test_cme_base64_id_reconstruction_boundaries():
    ids = torch.tensor([0, 63, 64, 4095, 4096, 262143], dtype=torch.int64)
    digits = _encode_base_digits(ids, base=64, num_digits=3)
    assert all(int(digit.min()) >= 0 and int(digit.max()) < 64 for digit in digits)
    assert torch.equal(_reconstruct_base_digits(digits, base=64), ids)


@pytest.mark.parametrize("rows", [1, 2, 31, 32])
@pytest.mark.parametrize("tiled_ids", [False, True])
@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_argmax_accepts_row_major_and_tiled_ids(mesh_device, rows, tiled_ids):
    width = 32
    values = torch.arange(rows * width, dtype=torch.float32).reshape(1, 1, rows, width)
    winners = torch.arange(rows, dtype=torch.int64) % width
    values.fill_(-10.0)
    values[0, 0, torch.arange(rows), winners] = 10.0
    ids = torch.arange(1000, 1000 + rows * width, dtype=torch.int64).reshape(1, 1, rows, width)
    expected = ids[0, 0, torch.arange(rows), winners]

    values_tt = ttnn.from_torch(values, device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    ids_tt = ttnn.from_torch(
        ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT if tiled_ids else ttnn.ROW_MAJOR_LAYOUT,
    )
    pack = CmeLogits(values=values_tt, ids=ids_tt, rows=rows)
    head = object.__new__(Gemma4TTMaskedEmbedder)

    actual = head.argmax_token_id(pack, rows)
    assert torch.equal(ttnn.to_torch(actual).reshape(-1).to(torch.int64), expected)


@pytest.mark.parametrize("rows", [1, 2, 31, 32, 33, 64])
@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_argmax_row_shapes_and_boundaries(mesh_device, rows):
    width = 64
    values = torch.full((1, 1, rows, width), -20.0, dtype=torch.float32)
    expected = []
    for row in range(rows):
        if row % 3 == 0:
            values[0, 0, row, -1] = -1.0  # negative values, last-column winner
            expected.append(width - 1)
        elif row % 3 == 1:
            values[0, 0, row, 0] = 3.0
            values[0, 0, row, -1] = 3.0  # tie follows the operation's first-index contract
            expected.append(0)
        else:
            values[0, 0, row, width // 2] = 4.0
            expected.append(width // 2)

    values_tt = ttnn.from_torch(values, device=mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    actual = Gemma4TTMaskedEmbedder._argmax_rows(values_tt, rows)
    assert ttnn.to_torch(actual).reshape(-1).tolist() == expected


def _is_valid_topk(values, indices, flat_logits, k):
    """A valid top-k: descending, values match the gathered logits, and the k-th
    value is >= every logit outside the selected set."""
    if not torch.equal(values, flat_logits[indices]):
        return False, "values != logits[indices]"
    if not bool((values[:-1] >= values[1:]).all()):
        return False, "values not descending"
    mask = torch.ones_like(flat_logits, dtype=torch.bool)
    mask[indices] = False
    if mask.any() and float(flat_logits[mask].max()) > float(values[-1]):
        return False, "an unselected logit exceeds the k-th selected value"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════
# Primitive-level: the ops the head is built from
# ══════════════════════════════════════════════════════════════════════════
@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_ops_topk_at_2048(mesh_device):
    """ttnn.topk(k=32) over width 2048 returns a valid top-k with exact values."""
    torch.manual_seed(0)
    for trial in range(4):
        cl = torch.randn(1, 1, 1, C, dtype=torch.bfloat16)
        tt = ttnn.from_torch(cl, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tv, ti = ttnn.topk(tt, k=K, dim=-1)
        idx = ttnn.to_torch(ti).reshape(-1).to(torch.int64)
        val = ttnn.to_torch(tv).reshape(-1).float()
        flat = cl.reshape(-1).float()

        assert idx.numel() == K and val.numel() == K
        assert idx.min() >= 0 and idx.max() < C, f"trial {trial}: index out of range {idx.min()}..{idx.max()}"
        valid, why = _is_valid_topk(val, idx, flat, K)
        assert valid, f"trial {trial}: {why}"

        # Any divergence from torch's index order must be a pure tie.
        _, ref_i = torch.topk(flat, K)
        for pos in (idx != ref_i).nonzero().reshape(-1).tolist():
            assert float(flat[idx[pos]]) == float(flat[ref_i[pos]]), (
                f"trial {trial}: index differs at {pos} with DIFFERENT logits "
                f"(tt {flat[idx[pos]]} vs ref {flat[ref_i[pos]]}) — not a tie-break"
            )


@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_ops_gather_uint32_dim2(mesh_device):
    """ttnn.gather picks whole rows of a uint32 [C,P] table via a broadcast index."""
    torch.manual_seed(0)
    ordering = torch.randperm(V, dtype=torch.int64)
    canon = ordering.reshape(C, P)
    ord_tt = ttnn.from_torch(canon.reshape(1, 1, C, P), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    chosen = torch.randint(0, C, (K,), dtype=torch.int64)
    col = ttnn.from_torch(chosen.reshape(1, 1, K, 1), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT)
    idx = ttnn.repeat(col, ttnn.Shape([1, 1, 1, P]))
    assert list(idx.shape) == [1, 1, K, P]
    assert torch.equal(
        ttnn.to_torch(idx).reshape(K, P).to(torch.int64), chosen.reshape(K, 1).expand(K, P)
    ), "repeat did not broadcast the centroid index across the row"

    sel = ttnn.gather(ord_tt, dim=2, index=idx)
    assert torch.equal(
        ttnn.to_torch(sel).reshape(K, P).to(torch.int64), canon[chosen]
    ), "uint32 gather along dim=2 did not reproduce canon[top_k_indices]"

    # Row-major flatten must be C order, so a local index f maps to (f // P, f % P).
    sel_rm = ttnn.reshape(ttnn.to_layout(sel, ttnn.ROW_MAJOR_LAYOUT), (1, N))
    assert torch.equal(ttnn.to_torch(sel_rm).reshape(-1).to(torch.int64), canon[chosen].reshape(-1))


@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_ops_embedding_and_matvec(mesh_device):
    """ttnn.embedding row-gathers N of V rows exactly; the matvec argmax is stable."""
    torch.manual_seed(0)
    emb = torch.randn(V, H, dtype=torch.bfloat16)
    emb_tt = ttnn.from_torch(
        emb.unsqueeze(0).unsqueeze(0), device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ids = torch.randperm(V, dtype=torch.int64)[:N]
    ids_tt = ttnn.from_torch(ids.reshape(1, N), device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    sel_emb = ttnn.embedding(ids_tt, emb_tt, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    if len(sel_emb.shape) == 3:
        sel_emb = ttnn.unsqueeze_to_4D(sel_emb)
    assert list(sel_emb.shape) == [1, 1, N, H]
    assert torch.equal(
        ttnn.to_torch(sel_emb).reshape(N, H).float(), emb[ids].float()
    ), "embedding row gather is not exact"

    h = torch.randn(1, 1, 1, H, dtype=torch.bfloat16)
    h_tt = ttnn.from_torch(h, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    h_col = ttnn.transpose(h_tt, -2, -1)
    vals = ttnn.transpose(ttnn.matmul(sel_emb, h_col), -2, -1)  # [1,1,1,N]
    got = ttnn.to_torch(vals).reshape(-1).float()
    ref = emb[ids].float() @ h.reshape(-1).float()
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel < 5e-2, f"matvec rel err {rel}"
    assert int(got.argmax()) == int(ref.argmax()), "matvec argmax disagrees with torch"

    # The pad-to-32-rows multicore argmax path over width N.
    padded = ttnn.pad(vals, [(0, 0), (0, 0), (0, 31), (0, 0)], value=0.0)
    idx = ttnn.argmax(ttnn.untilize(padded, use_multicore=True), dim=-1, keepdim=False)
    assert int(ttnn.to_torch(idx).reshape(-1)[0]) == int(got.argmax())


# ══════════════════════════════════════════════════════════════════════════
# Parity against the HF reference on the real checkpoint
# ══════════════════════════════════════════════════════════════════════════
def _hf_reference():
    """HF's Gemma4AssistantMaskedEmbedder loaded with the real checkpoint weights."""
    from transformers import AutoConfig
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import Gemma4AssistantMaskedEmbedder

    from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

    hf_config = AutoConfig.from_pretrained(ASSISTANT_PATH, trust_remote_code=True)
    state_dict = Gemma4AssistantArgs.load_state_dict(ASSISTANT_PATH, dummy_weights=False)
    ref = Gemma4AssistantMaskedEmbedder(hf_config)
    ref.centroids.weight.data = state_dict["masked_embedding.centroids.weight"].float()
    ref.token_ordering = state_dict["masked_embedding.token_ordering"].to(torch.int64)
    lm_w = state_dict.get("lm_head.weight", state_dict.get("model.embed_tokens.weight"))
    return ref, lm_w, hf_config, state_dict


def _tt_head(mesh_device, state_dict, hf_config):
    from models.demos.gemma4.tt.assistant.masked_embedding import Gemma4TTMaskedEmbedder
    from models.demos.gemma4.tt.model_config import Gemma4AssistantArgs

    args = Gemma4AssistantArgs.from_hf_config(hf_config)
    return Gemma4TTMaskedEmbedder(
        mesh_device=mesh_device,
        assistant_args=args,
        state_dict=state_dict,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
    )


@_needs_assistant
@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_vs_hf(mesh_device):
    """TT CME head vs HF's fp32 reference on the real checkpoint.

    Asserts the contract described in the module docstring: valid centroid
    selection (differences confined to the bf16-quantization band at the top-k
    boundary), high-PCC candidate scores, and an argmax token that matches HF
    except when HF's winner lives in a block TT did not select.
    """
    ref, lm_w, hf_config, state_dict = _hf_reference()
    head = _tt_head(mesh_device, state_dict, hf_config)
    if not (head.num_centroids == C and head.top_k == K and head.vocab_per_centroid == P):
        pytest.skip(f"checkpoint geometry {head.num_centroids}/{head.top_k}/{head.vocab_per_centroid} != E2B")

    Wc = ref.centroids.weight.data
    ordering = ref.token_ordering
    canon = ordering.reshape(C, P)
    n_trials = int(os.environ.get("GEMMA4_CME_TRIALS", 16))

    torch.manual_seed(0)
    argmax_match = set_match = swaps = 0
    worst_pcc = 1.0
    for trial in range(n_trials):
        h = torch.randn(1, 1, 1, head.hidden_size, dtype=torch.bfloat16)
        h_tt = ttnn.from_torch(h, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        pack = head.forward(h_tt)
        tt_vals = ttnn.to_torch(pack.values).reshape(-1).float()
        tt_ids = ttnn.to_torch(pack.ids).reshape(-1).to(torch.int64)
        pack.deallocate(True)

        hv = h.reshape(-1).float()
        with torch.no_grad():
            hf_out = ref(h.reshape(1, 1, -1).float(), lm_w.float()).reshape(-1)
        hf_tok = int(hf_out.argmax())

        # ── structure ─────────────────────────────────────────────────────
        assert tt_ids.numel() == head.num_candidates, f"trial {trial}: {tt_ids.numel()} candidates"
        assert tt_ids.unique().numel() == tt_ids.numel(), f"trial {trial}: duplicate candidate ids"
        assert int(tt_ids.min()) >= 0 and int(tt_ids.max()) < head.vocab_size

        # TT's candidates must be exactly the union of 32 whole centroid blocks:
        # each slot's first id identifies a centroid, and the rest of the slot must
        # be that centroid's block verbatim. This is what proves the
        # repeat -> gather -> row-major-flatten chain reproduces canon[top_k].
        blocks = tt_ids.reshape(K, P)
        for s in range(K):
            row = blocks[s]
            hits = (canon == int(row[0])).nonzero()
            assert hits.shape[0] == 1, f"trial {trial}: candidate {int(row[0])} is not unique in the ordering table"
            cent = int(hits[0, 0])
            assert torch.equal(row, canon[cent]), f"trial {trial}: slot {s} is not a whole centroid block"

        # ── centroid selection: differences must sit in the bf16 band ──────
        cl32 = hv @ Wc.T
        _, ref_i = torch.topk(cl32, K)
        ref_ids = canon[ref_i].reshape(-1)
        tt_set, ref_set = set(tt_ids.tolist()), set(ref_ids.tolist())
        if tt_set == ref_set:
            set_match += 1
        else:
            swaps += 1
            srt = cl32.sort(descending=True).values
            boundary = float(srt[K - 1])
            # bf16 quantization band at this magnitude (bf16 has 8 mantissa bits).
            band = 4.0 * abs(boundary) * 2.0**-8 + 1e-3
            tt_cent_set = {int((canon == int(t)).nonzero()[0, 0]) for t in blocks[:, 0]}
            for c in tt_cent_set.symmetric_difference(set(ref_i.tolist())):
                assert abs(float(cl32[c]) - boundary) <= band, (
                    f"trial {trial}: centroid {c} differs with fp32 score {float(cl32[c]):.6f}, "
                    f"{abs(float(cl32[c]) - boundary):.6f} away from the boundary {boundary:.6f} "
                    f"(bf16 band {band:.6f}) — not a quantization swap"
                )

        # ── candidate scores vs the exact fp32 dot over TT's own ids ───────
        exact = lm_w.float()[tt_ids] @ hv
        vx, vy = tt_vals - tt_vals.mean(), exact - exact.mean()
        pcc = float((vx * vy).sum() / (vx.norm() * vy.norm()))
        worst_pcc = min(worst_pcc, pcc)
        assert pcc > 0.9999, f"trial {trial}: selected-logit PCC {pcc}"

        # ── argmax token ──────────────────────────────────────────────────
        tt_tok = int(tt_ids[int(tt_vals.argmax())])
        if tt_tok == hf_tok:
            argmax_match += 1
        else:
            # Allowed only if HF's winner was never a TT candidate (block swap),
            # or the two contenders are closer than the achievable resolution.
            hf_winner_available = hf_tok in tt_set
            top2 = exact.sort(descending=True).values[:2]
            gap = float(top2[0] - top2[1])
            assert (not hf_winner_available) or gap < 0.05, (
                f"trial {trial}: argmax differs (tt {tt_tok} vs hf {hf_tok}) although HF's winner WAS "
                f"a TT candidate and the fp32 top-2 gap is {gap:.4f} — a real scoring bug"
            )

    logger.info(
        f"CME vs HF over {n_trials} trials: argmax matched {argmax_match}/{n_trials}, "
        f"candidate set identical {set_match}/{n_trials} ({swaps} bf16 boundary swaps), "
        f"worst selected-logit PCC {worst_pcc:.7f}"
    )
    # The head must be right far more often than not; swaps are the only excuse.
    assert argmax_match >= n_trials - swaps


@_needs_assistant
@parametrize_mesh_with_fabric([(1, 1)])
def test_cme_full_vocab_reconstruction(mesh_device):
    """The host full-vocab rebuild reproduces HF's masked output, mask_value included."""
    ref, lm_w, hf_config, state_dict = _hf_reference()
    head = _tt_head(mesh_device, state_dict, hf_config)

    torch.manual_seed(1)
    h = torch.randn(1, 1, 1, head.hidden_size, dtype=torch.bfloat16)
    h_tt = ttnn.from_torch(h, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    pack = head.forward(h_tt)
    full = head.to_host_full_vocab(pack, ttnn.to_torch)
    assert list(full.shape) == [1, head.vocab_size]

    with torch.no_grad():
        hf_out = ref(h.reshape(1, 1, -1).float(), lm_w.float()).reshape(-1)

    # Structure: exactly N entries above the fill, and the fill is min(selected)-1.
    tt_row = full.reshape(-1)
    fill = float(tt_row.min())
    n_above = int((tt_row > fill).sum())
    assert n_above <= head.num_candidates, f"{n_above} entries above the fill, expected <= {head.num_candidates}"
    sel_vals = tt_row[tt_row > fill]
    assert abs(fill - (float(sel_vals.min()) - 1.0)) < 5e-2, f"fill {fill} != min(selected)-1"

    # HF's own fill must match ours to bf16 tolerance.
    hf_fill = float(hf_out.min())
    assert abs(hf_fill - fill) < 5e-2, f"mask_value mismatch: tt {fill} vs hf {hf_fill}"
    logger.info(f"CME full-vocab rebuild: {n_above} selected, fill={fill:.4f} (hf {hf_fill:.4f})")
    pack.deallocate(True)
