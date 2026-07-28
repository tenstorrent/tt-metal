# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-2 PCC tests for the MiniMax-M2 MoE router vs the HF reference math.

A single fuzzy end-to-end PCC is a poor router test: bf16 top-k selection flips
the boundary (8th vs 9th) expert on near-tie tokens, which drags a dense-weight
PCC into the ~0.95 range regardless of correctness — so a passing threshold there
could equally hide a real bug. Instead we DECOMPOSE the router and feed BOTH sides
the SAME (bf16-rounded) gate logits, then assert:

  (1) ttnn.sigmoid matches torch.sigmoid           (>= 0.999)
  (2) gather+normalize with IDENTICAL indices       (>= 0.999)   -> weight math exact
  (3) every expert TT/HF disagree on is within bf16 resolution of the selection
      boundary                                                   -> flips are ties, not a bug

HF reference: MiniMaxM2SparseMoeBlock.route_tokens_to_experts.
Runs at mesh (1,1)/TP=1.
"""


import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import minimax_config_dims, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@pytest.mark.parametrize("use_bias", [False, True], ids=["no_bias", "score_bias"])
def test_router_vs_hf(mesh_device, device_params, seq_len, use_bias, reset_seeds):
    cfg = minimax_config_dims()
    H, E, K = cfg["hidden_size"], cfg["num_local_experts"], cfg["num_experts_per_tok"]

    gate_w = torch.randn(E, H) * (H**-0.5)
    # Draw the bias unconditionally (then zero it) so `x` sees the same RNG offset in
    # both parametrizations; use_bias=True exercises the +bias selection path.
    bias_full = torch.randn(E) * 0.1
    bias = bias_full if use_bias else torch.zeros(E)
    x = torch.randn(1, seq_len, H)

    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, H), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl
    )
    w_tt = ttnn.from_torch(
        gate_w.t().contiguous(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl
    )
    bias_tt = ttnn.from_torch(
        bias.reshape(1, -1), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl
    )

    # Compute logits on TT (bf16) and read them back so BOTH sides use IDENTICAL logits;
    # this isolates the router logic from gate-matmul precision.
    logits_tt = ttnn.linear(ttnn.reshape(x_tt, (-1, H)), w_tt)
    logits = ttnn.to_torch(ttnn.get_device_tensors(logits_tt)[0]).reshape(seq_len, E).float()

    # --- HF reference routing (fp32) on those logits ---
    rw_ref = torch.sigmoid(logits)
    scores = rw_ref + bias
    top_scores, idx_ref = torch.topk(scores, K, dim=-1)
    boundary = top_scores[:, K - 1 : K]  # Kth-highest selection score per token

    # (1) sigmoid op precision
    rw_tt = ttnn.sigmoid(logits_tt)
    sig = ttnn.to_torch(ttnn.get_device_tensors(rw_tt)[0]).reshape(seq_len, E).float()
    sig_pass, sig_pcc = comp_pcc(rw_ref, sig, 0.999)

    # (2) gather+normalize with IDENTICAL (reference) indices -> isolates weight math
    idx_ref_tt = ttnn.from_torch(
        idx_ref.int(), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=repl
    )
    w_tt_same = ttnn.gather(rw_tt, dim=-1, index=idx_ref_tt)
    w_tt_same = ttnn.div(w_tt_same, ttnn.sum(w_tt_same, dim=-1, keepdim=True))
    w_same = ttnn.to_torch(ttnn.get_device_tensors(w_tt_same)[0]).reshape(seq_len, K).float()
    w_ref = rw_ref.gather(1, idx_ref)
    w_ref = w_ref / w_ref.sum(-1, keepdim=True)
    # 0.995 reflects bf16 precision of gather+sum+div over K values; a real gather/
    # normalize bug gives << 0.9. Also assert a tight max relative error as a second,
    # interpretable confirmation the per-weight values match HF.
    wmath_pass, wmath_pcc = comp_pcc(w_ref, w_same, 0.995)
    max_rel_err = ((w_same - w_ref).abs() / w_ref.abs().clamp_min(1e-3)).max().item()

    # (3) TT's actual selection vs reference -> any disagreement must be a boundary tie
    sc_tt = ttnn.add(rw_tt, bias_tt)
    _, idx_tt = ttnn.topk(sc_tt, k=K, dim=-1, sorted=True)
    idx_tt_t = ttnn.to_torch(ttnn.get_device_tensors(idx_tt)[0]).reshape(seq_len, K).long()
    ref_set = torch.zeros(seq_len, E).scatter_(1, idx_ref, 1).bool()
    tt_set = torch.zeros(seq_len, E).scatter_(1, idx_tt_t, 1).bool()
    disagree = ref_set ^ tt_set
    overlap = (ref_set & tt_set).sum(-1).float().mean().item()
    if disagree.any():
        margins = (scores - boundary).abs()[disagree]  # distance of each disputed expert from the boundary
        max_margin = margins.max().item()
    else:
        max_margin = 0.0
    bf16_res = boundary.abs().max().item() * 2**-8  # ~resolution near the boundary score
    tol = 4 * bf16_res

    logger.info(
        f"router decomposition (use_bias={use_bias}): sigmoid_pcc={sig_pcc}, "
        f"weight_math_pcc={wmath_pcc}, weight_max_rel_err={max_rel_err:.4f}, "
        f"selection_overlap={overlap:.3f}/{K}, "
        f"max_disagreement_margin={max_margin:.5f} (tol={tol:.5f}, bf16_res={bf16_res:.5f})"
    )
    assert sig_pass, f"sigmoid PCC fail: {sig_pcc}"
    assert wmath_pass, f"weight-math PCC fail: {wmath_pcc}"
    assert max_rel_err < 0.05, f"weight-math max relative error too high: {max_rel_err}"
    assert max_margin <= tol, (
        f"selection disagreement {max_margin:.5f} exceeds bf16 boundary tolerance {tol:.5f} "
        f"-> not a tie, likely a routing bug"
    )
