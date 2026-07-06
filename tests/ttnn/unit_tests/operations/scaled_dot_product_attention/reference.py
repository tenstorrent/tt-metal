# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch reference for scaled_dot_product_attention (Flash Attention).

Decomposed into the same Compute Phases as op_design.md — one function per
phase. Each phase function computes and prints the checkpoint slice for the
pinned tiny shape, so the implementer's TDD loop can compare DPRINT output
against these values.

Pinned shape: (B=1, H=1, S=128, D=64) → S_t=4, D_t=2.
B_q=4, B_kv=4 (one Q block, one KV block — exercises all phases with
multi-tile paths).

The reference implements the standard (non-flash) math but decomposed to
mirror the flash-attention phase structure: QK^T → scale → mask → rowmax →
exp → rowsum → rescale → PV → normalize. With a single KV block, the online
softmax recurrence is mathematically identical to the two-pass formulation.
"""

import math
import struct

import torch

torch.manual_seed(0)

# ── Pinned inputs ──────────────────────────────────────────────────────
B, H, S, D = 1, 1, 128, 64
SCALE = 1.0 / math.sqrt(D)

Q = torch.randn(B, H, S, D, dtype=torch.float32)
K = torch.randn(B, H, S, D, dtype=torch.float32)
V = torch.randn(B, H, S, D, dtype=torch.float32)

# Tile counts
TILE_DIM = 32
S_t = S // TILE_DIM  # 4
D_t = D // TILE_DIM  # 2

# Block sizes (must match op_design.md defaults)
B_q = 4  # Q block size in tiles (128 rows)
B_kv = 4  # KV block size in tiles (128 rows)


def _tile_slice(tile_row, tile_col, rows=4, cols=4):
    """Extract the first rows×cols elements from a tile at (tile_row, tile_col).

    Tiles are 32×32. We return the top-left rows×cols sub-block.
    The tensor is (B, H, S, D) → tile (tr, tc) covers
    rows [tr*32 : (tr+1)*32] × cols [tc*32 : (tc+1)*32].
    """
    r0 = tile_row * TILE_DIM
    c0 = tile_col * TILE_DIM
    return (r0, r0 + rows, c0, c0 + cols)


def _print_slice(name, tensor, tile_row=0, tile_col=0, rows=4, cols=4):
    """Print a rows×cols slice from tile (tile_row, tile_col) of a (S, D) tensor.

    tensor is expected to be (S, D) for 2D phases or (S, 1) for 1D phases.
    """
    r0, r1, c0, c1 = _tile_slice(tile_row, tile_col, rows, cols)
    if tensor.dim() == 2:
        sub = tensor[r0:r1, c0:c1]
    elif tensor.dim() == 1:
        sub = tensor[r0:r1]
    else:
        sub = tensor[0, 0, r0:r1, c0:c1]
    print(f"{name} (tile ({tile_row},{tile_col}), {rows}x{cols} slice):")
    for r in range(sub.shape[0]):
        row_vals = []
        for c in range(sub.shape[1]):
            row_vals.append(f"{sub[r, c].item():.6f}")
        print(f"  [{', '.join(row_vals)}]")


# ── Compute Phases (mirrors op_design.md) ──────────────────────────────


def reference_phase_init(state):
    """Phase 0: Initialize running state.

    m_i = -inf (per row), l_i = 0 (per row), O_i = 0 (per row, per D).
    Also compute the scale factor.
    """
    S_q_dim = B_q * TILE_DIM
    m_i = torch.full((B_q, 1, TILE_DIM), float("-inf"), dtype=torch.float32)
    l_i = torch.zeros((B_q, 1, TILE_DIM), dtype=torch.float32)
    O_i = torch.zeros((B_q * TILE_DIM, D), dtype=torch.float32)
    print(f"scale = {SCALE:.6f}")
    print(f"m_i init: {m_i[0, 0, :4].tolist()}")
    print(f"l_i init: {l_i[0, 0, :4].tolist()}")
    return {"scale": SCALE, "m_i": m_i, "l_i": l_i, "O_i": O_i}


def reference_phase_qkt(state):
    """Phase 1: QK^T — S = Q_block @ K_block^T.

    For the pinned shape with B_q=4, B_kv=4, S_t=4, this is the full
    (128, 128) score matrix. Tile (0,0) is the first 32×32 block.
    """
    Q_block = Q[0, 0, : B_q * TILE_DIM, :]  # (128, 64)
    K_block = K[0, 0, : B_kv * TILE_DIM, :]  # (128, 64)
    scores = Q_block @ K_block.T  # (128, 128)
    state["scores"] = scores
    _print_slice("cb_scores tile (0,0)", scores, 0, 0, 4, 4)
    return state


def reference_phase_scale(state):
    """Phase 2: Scale — S *= scale."""
    state["scores"] = state["scores"] * state["scale"]
    _print_slice("cb_scores (scaled) tile (0,0)", state["scores"], 0, 0, 4, 4)
    return state


def reference_phase_mask(state):
    """Phase 2b: Mask (optional) — S += attn_mask.

    Phase 0 with no mask: this is a no-op. Included for completeness.
    """
    # No mask in Phase 0 base path
    print("mask_mode=none: no mask applied")
    return state


def reference_phase_rowmax(state):
    """Phase 3: RowMax — m_block = rowmax(S).

    Reduce MAX along the W dimension (B_kv columns), output B_q rows.
    """
    scores = state["scores"]  # (128, 128) = (B_q*32, B_kv*32)
    # Reduce along dim=1 (the KV dimension), keepdim for broadcast
    m_block = scores.max(dim=1, keepdim=True)[0]  # (128, 1)
    state["m_block"] = m_block
    # Print as tile 0 of the reduce output (first 4 elements of row 0)
    sub = m_block[:4, 0]
    print(f"cb_m_new tile 0, first 4 elements: [{', '.join(f'{v:.6f}' for v in sub.tolist())}]")
    return state


def reference_phase_onlinemax(state):
    """Phase 4: OnlineMax — m_new = max(m_i, m_block).

    m_i is (B_q, 1, 32) per-tile; m_block is (128, 1). We compute per-row.
    """
    m_i = state["m_i"]  # (B_q, 1, 32) → flatten to (128, 1)
    m_i_flat = m_i.reshape(-1, 1)  # (128, 1)
    m_new = torch.maximum(m_i_flat, state["m_block"])  # (128, 1)
    state["m_new"] = m_new
    sub = m_new[:4, 0]
    print(f"cb_m_new (m_new) tile 0, first 4 elements: [{', '.join(f'{v:.6f}' for v in sub.tolist())}]")
    return state


def reference_phase_expscores(state):
    """Phase 5: ExpScores — P = exp(S - m_new).

    m_new is (128, 1), broadcast across 128 columns of S.
    """
    scores = state["scores"]  # (128, 128)
    m_new = state["m_new"]  # (128, 1)
    P = torch.exp(scores - m_new)  # (128, 128)
    state["P"] = P
    _print_slice("cb_scores (P) tile (0,0)", P, 0, 0, 4, 4)
    return state


def reference_phase_copyp(state):
    """Phase 6: CopyP — copy P to cb_pv.

    This is a data movement step. The values are identical.
    """
    state["P_copy"] = state["P"].clone()
    _print_slice("cb_pv (P copy) tile (0,0)", state["P_copy"], 0, 0, 4, 4)
    return state


def reference_phase_rescale_l(state):
    """Phase 7: Rescale l_i — l_i *= exp(m_i - m_new).

    factor_old = exp(m_i - m_new). l_i = l_i * factor_old.
    """
    m_i = state["m_i"].reshape(-1, 1)  # (128, 1)
    m_new = state["m_new"]  # (128, 1)
    l_i = state["l_i"].reshape(-1, 1)  # (128, 1)
    factor_old = torch.exp(m_i - m_new)  # (128, 1)
    l_i = l_i * factor_old  # (128, 1)
    state["l_i"] = l_i.reshape(B_q, 1, TILE_DIM)
    state["factor_old"] = factor_old
    sub = l_i[:4, 0]
    print(f"cb_l (scaled l_i) tile 0, first 4 elements: [{', '.join(f'{v:.6f}' for v in sub.tolist())}]")
    return state


def reference_phase_rescale_o(state):
    """Phase 8: Rescale O_i — O_i *= exp(m_i - m_new).

    Same factor_old as l_i, broadcast across D columns.
    """
    factor_old = state["factor_old"]  # (128, 1)
    O_i = state["O_i"]  # (128, 64)
    O_i = O_i * factor_old  # (128, 64)
    state["O_i"] = O_i
    _print_slice("cb_o (scaled O_i) tile (0,0)", O_i, 0, 0, 4, 4)
    return state


def reference_phase_update_m(state):
    """Phase 9: Update m_i — m_i = m_new.

    Pop old m_i, copy m_new → m_i.
    """
    m_new = state["m_new"]  # (128, 1)
    state["m_i"] = m_new.reshape(B_q, 1, TILE_DIM)
    sub = state["m_i"].reshape(-1, 1)[:4, 0]
    print(f"cb_m (new m_i) tile 0, first 4 elements: [{', '.join(f'{v:.6f}' for v in sub.tolist())}]")
    return state


def reference_phase_rowsum(state):
    """Phase 10: RowSum — psum = rowsum(P).

    Reduce SUM along W dimension of P.
    """
    P = state["P_copy"]  # (128, 128)
    psum = P.sum(dim=1, keepdim=True)  # (128, 1)
    state["psum"] = psum
    sub = psum[:4, 0]
    print(f"cb_psum (rowsum) tile 0, first 4 elements: [{', '.join(f'{v:.6f}' for v in sub.tolist())}]")
    return state


def reference_phase_l_accumulate(state):
    """Phase 11: l_i += psum."""
    l_i = state["l_i"].reshape(-1, 1)  # (128, 1)
    psum = state["psum"]  # (128, 1)
    l_i = l_i + psum
    state["l_i"] = l_i.reshape(B_q, 1, TILE_DIM)
    sub = l_i[:4, 0]
    print(f"cb_l (updated l_i) tile 0, first 4 elements: [{', '.join(f'{v:.6f}' for v in sub.tolist())}]")
    return state


def reference_phase_pv(state):
    """Phase 12: PV — PV = P @ V.

    P is (128, 128), V is (128, 64). PV = P @ V = (128, 64).
    """
    P = state["P_copy"]  # (128, 128)
    V_block = V[0, 0, : B_kv * TILE_DIM, :]  # (128, 64)
    PV = P @ V_block  # (128, 64)
    state["PV"] = PV
    _print_slice("cb_scores (PV) tile (0,0)", PV, 0, 0, 4, 4)
    return state


def reference_phase_o_accumulate(state):
    """Phase 13: O_i += PV."""
    O_i = state["O_i"]  # (128, 64)
    PV = state["PV"]  # (128, 64)
    O_i = O_i + PV
    state["O_i"] = O_i
    _print_slice("cb_o (updated O_i) tile (0,0)", O_i, 0, 0, 4, 4)
    return state


def reference_phase_normalize(state):
    """Phase 14: Normalize — O = O_i / l_i.

    Broadcast l_i across D columns.
    """
    O_i = state["O_i"]  # (128, 64)
    l_i = state["l_i"].reshape(-1, 1)  # (128, 1)
    output = O_i / l_i  # (128, 64)
    state["output"] = output
    _print_slice("cb_output (final) tile (0,0)", output, 0, 0, 4, 4)
    return state


def reference_phase_reset(state):
    """Phase 15: Reset — clean state for next Q block.

    For the pinned shape there is only one Q block, so this is a no-op
    beyond clearing state.
    """
    print("reset: clearing state for next Q block (none needed — single Q block)")
    return state


# ── Stage index (machine-readable, parsed by the staged drafter driver) ─

STAGES = [
    {
        "index": 0,
        "name": "init",
        "implement": "Op file + program descriptor + kernel scaffolding. Reader pushes scaler/scale tiles; compute initializes m_i=-inf, l_i=0, O_i=0. Kernel is runnable — completes without hang.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_init",
            "checkpoint": "cb_m tile 0 init (-inf), cb_l tile 0 init (0), scale value",
            "pass_criteria": "m_i tiles contain -inf, l_i tiles contain 0.0, scale printed matches 1/sqrt(D)",
        },
    },
    {
        "index": 1,
        "name": "qkt",
        "implement": "QK^T matmul: matmul_block<transpose=true, TileRowMajor> reading cb_q/cb_k → cb_scores. Reader streams Q/K tile blocks from DRAM.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_qkt",
            "checkpoint": "cb_scores tile (0,0), first 4x4 elements",
            "pass_criteria": "score values match reference Q@K^T within PCC 0.99",
        },
    },
    {
        "index": 2,
        "name": "scale",
        "implement": "Scale scores: transform_in_place<MulUnary> on cb_scores. Reader pushes scale tile (filled with 1/sqrt(D) or user scale).",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_scale",
            "checkpoint": "cb_scores tile (0,0) after scaling, first 4x4 elements",
            "pass_criteria": "scaled values = raw scores * scale within PCC 0.99",
        },
    },
    {
        "index": 3,
        "name": "rowmax",
        "implement": "Row max: reduce<MAX, REDUCE_ROW, WaitUpfrontNoPop> on cb_scores → cb_m_new. Reader pushes MAX scaler tile.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rowmax",
            "checkpoint": "cb_m_new tile 0, first 4 elements (rowmax of first 4 rows)",
            "pass_criteria": "max values match torch max of score rows within PCC 0.99",
        },
    },
    {
        "index": 4,
        "name": "onlinemax",
        "implement": "Online max: eltwise_chain with BinaryMax on cb_m (HeldBulk) and cb_m_new (streaming) → cb_m_new (in-place).",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_onlinemax",
            "checkpoint": "cb_m_new tile 0 after BinaryMax, first 4 elements",
            "pass_criteria": "m_new = max(m_i, m_block) matches reference within PCC 0.99",
        },
    },
    {
        "index": 5,
        "name": "expscores",
        "implement": "Exp scores: eltwise_chain with Sub (broadcast m_new across cols) + Exp on cb_scores (in-place). cb_m_new as Col+HeldBulk.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_expscores",
            "checkpoint": "cb_scores tile (0,0) after exp, first 4x4 elements",
            "pass_criteria": "P = exp(S - m_new) matches reference within PCC 0.99",
        },
    },
    {
        "index": 6,
        "name": "copyp",
        "implement": "Copy P: copy<cb_scores, cb_pv> — stream P from cb_scores to cb_pv.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_copyp",
            "checkpoint": "cb_pv tile (0,0), first 4x4 elements (should match P from phase 5)",
            "pass_criteria": "P copy matches phase 5 output exactly",
        },
    },
    {
        "index": 7,
        "name": "rescale_l",
        "implement": "Rescale l_i: eltwise_chain with SubBinary(m_i, m_new) + Exp + MulBinary(l_i, factor_old) → cb_l (in-place). cb_m and cb_m_new as HeldBulk.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rescale_l",
            "checkpoint": "cb_l tile 0 after rescaling, first 4 elements",
            "pass_criteria": "l_i = l_i * exp(m_i - m_new) matches reference within PCC 0.99",
        },
    },
    {
        "index": 8,
        "name": "rescale_o",
        "implement": "Rescale O_i: eltwise_chain with SubBinary(m_i, m_new) + Exp + MulBinary(O_i, factor_old) → cb_o (in-place). cb_m and cb_m_new as Col+HeldBulk. After: manual pop cb_m and cb_m_new.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rescale_o",
            "checkpoint": "cb_o tile (0,0) after rescaling, first 4x4 elements",
            "pass_criteria": "O_i = O_i * exp(m_i - m_new) matches reference within PCC 0.99",
        },
    },
    {
        "index": 9,
        "name": "update_m",
        "implement": "Update m_i: manual pop old m_i from cb_m, then copy<cb_m_new, cb_m> to set m_i = m_new. Pop cb_m_new.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_update_m",
            "checkpoint": "cb_m tile 0 after update, first 4 elements (should match m_new from phase 4)",
            "pass_criteria": "m_i = m_new matches reference within PCC 0.99",
        },
    },
    {
        "index": 10,
        "name": "rowsum",
        "implement": "Row sum: reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop> on cb_pv (P) → cb_psum. Reader pushes SUM scaler tile. P tiles not popped.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_rowsum",
            "checkpoint": "cb_psum tile 0, first 4 elements (rowsum of P)",
            "pass_criteria": "psum = sum of P rows matches reference within PCC 0.99",
        },
    },
    {
        "index": 11,
        "name": "l_accumulate",
        "implement": "l_i += psum: add<cb_l, cb_psum, cb_l> (in-place). Pops cb_psum.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_l_accumulate",
            "checkpoint": "cb_l tile 0 after accumulation, first 4 elements",
            "pass_criteria": "l_i = l_i + psum matches reference within PCC 0.99",
        },
    },
    {
        "index": 12,
        "name": "pv",
        "implement": "PV matmul: matmul_block<false, TileRowMajor> reading cb_pv (P) and cb_v → cb_scores (reused as output). Reader streams V tiles.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_pv",
            "checkpoint": "cb_scores tile (0,0) PV output, first 4x4 elements",
            "pass_criteria": "PV = P @ V matches reference within PCC 0.99",
        },
    },
    {
        "index": 13,
        "name": "o_accumulate",
        "implement": "O_i += PV: add<cb_o, cb_scores, cb_o> (in-place). Pops cb_scores (PV tiles).",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_o_accumulate",
            "checkpoint": "cb_o tile (0,0) after accumulation, first 4x4 elements",
            "pass_criteria": "O_i = O_i + PV matches reference within PCC 0.99",
        },
    },
    {
        "index": 14,
        "name": "normalize",
        "implement": "Normalize: eltwise_chain with CopyTile(cb_o) + CopyTile(cb_l, Col+HeldBulk) + DivBinary → cb_output. Writer reads cb_output and writes to DRAM.",
        "verify": {
            "method": "dprint",
            "reference_fn": "reference_phase_normalize",
            "checkpoint": "cb_output tile (0,0) final output, first 4x4 elements",
            "pass_criteria": "output = O_i / l_i matches reference within PCC 0.99",
        },
    },
    {
        "index": 15,
        "name": "reset",
        "implement": "Reset: pop remaining cb_m, cb_l remnants. Reinit m_i=-inf, l_i=0, O_i=0 for next Q block. For single-Q-block case, just clean exit.",
        "verify": {
            "method": "none",
            "reference_fn": None,
            "checkpoint": "no reference function",
            "pass_criteria": "kernel completes, no hang, all CBs drained",
        },
    },
]


# ── Main: run all phases in sequence and print checkpoints ─────────────

if __name__ == "__main__":
    print("=" * 72)
    print("scaled_dot_product_attention — reference.py")
    print(f"Pinned shape: B={B}, H={H}, S={S}, D={D}")
    print(f"Tiles: S_t={S_t}, D_t={D_t}")
    print(f"Block sizes: B_q={B_q}, B_kv={B_kv}")
    print(f"Scale: {SCALE:.6f} (1/sqrt({D}))")
    print("=" * 72)
    print()

    state = {}
    phase_fns = [
        reference_phase_init,
        reference_phase_qkt,
        reference_phase_scale,
        reference_phase_mask,
        reference_phase_rowmax,
        reference_phase_onlinemax,
        reference_phase_expscores,
        reference_phase_copyp,
        reference_phase_rescale_l,
        reference_phase_rescale_o,
        reference_phase_update_m,
        reference_phase_rowsum,
        reference_phase_l_accumulate,
        reference_phase_pv,
        reference_phase_o_accumulate,
        reference_phase_normalize,
        reference_phase_reset,
    ]

    for fn in phase_fns:
        print(f"--- {fn.__name__} ---")
        state = fn(state)
        print()

    # Final verification: compare against torch.nn.functional.scaled_dot_product_attention
    print("--- final torch reference comparison ---")
    torch_ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=SCALE)
    our_output = state["output"].unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 64)
    max_diff = (torch_ref - our_output).abs().max().item()
    print(f"Max abs diff vs torch.nn.functional.scaled_dot_product_attention: {max_diff:.8f}")
    print(f"Output matches torch reference: {max_diff < 1e-5}")
