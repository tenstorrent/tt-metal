# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic debugging tests for moe_fused_swiglu — DO NOT DELETE.

Every case uses inputs whose exact result is hand-calculable, and each one isolates ONE
structural mapping in the blocking scheme:

  * `test_all_ones`            — is the whole chain the right magnitude, and is the output
                                 UNIFORM? Any non-uniform column indicts the Ne split
                                 (per-core emb assignment / W_down read / output write).
  * `test_hidden_identity`     — W_down is a hidden->emb identity, so the output IS `h`.
                                 W_gate is per-hidden-tile constant, so the expected output
                                 encodes the hidden tile index. Any permutation indicts the Hn
                                 split, the h all-gather round order, or the phase-2 K indexing.
  * `test_emb_contraction`     — x is per-emb-tile constant, so the gate/up sum encodes the emb
                                 axis. Isolates the Kg row split + the cross-column reduce tree.
"""

import pytest
import torch

import ttnn

from ttnn.operations.moe_fused_swiglu import moe_fused_swiglu

TILE = 32
HIDDEN = 2048
NUM_GLOBAL_EXPERTS = 256
NUM_LOCAL_EXPERTS = 8
LOCAL_EXPERT_ID = 3
GLOBAL_EXPERT_ID = 137


def _count_tensors(count, device):
    counts = torch.zeros(NUM_GLOBAL_EXPERTS, dtype=torch.int32)
    counts[GLOBAL_EXPERT_ID] = count
    idx = torch.tensor([(11 + 37 * i) % NUM_GLOBAL_EXPERTS for i in range(NUM_LOCAL_EXPERTS)], dtype=torch.int32)
    idx[LOCAL_EXPERT_ID] = GLOBAL_EXPERT_ID
    to_dev = lambda t: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return to_dev(counts), to_dev(idx)


def _run(device, x, wg, wu, wd, count):
    tt_x = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = [
        ttnn.from_torch(
            w.to(torch.bfloat16),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for w in (wg, wu, wd)
    ]
    tt_counts, tt_idx = _count_tensors(count, device)
    out = moe_fused_swiglu(tt_x, tt_w[0], tt_w[1], tt_w[2], tt_counts, tt_idx, LOCAL_EXPERT_ID)
    return ttnn.to_torch(out)[0, 0, :, :].to(torch.float32)


def _bfp4(t, device):
    """Round-trip a weight through bfp4_b so the reference sees the bytes the device saw.

    bfp4_b keeps a sign + 3 magnitude bits per datum against a shared exponent, so a value like
    9.0 lands on 8.0 — a ~11% shift that has nothing to do with the kernel.
    """
    tt = ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.to_torch(tt).to(torch.float32)


def _reference(x_rows, wg, wu, wd):
    xf = x_rows.to(torch.float32)
    h = torch.nn.functional.silu(torch.matmul(xf, wg.to(torch.float32)))
    h = h * torch.matmul(xf, wu.to(torch.float32))
    return torch.matmul(h, wd.to(torch.float32))


def test_all_ones(device):
    """x, W_gate, W_up, W_down all 1.0 -> every output element is the SAME value.

    gate = up = emb; h = SiLU(emb) * emb = emb^2 (SiLU(z) ~ z for large z);
    out  = hidden * emb^2.
    """
    emb, capacity, count = 7168, 1024, 32
    x = torch.ones((1, 1, capacity, emb))
    wg = torch.ones((emb, HIDDEN))
    wu = torch.ones((emb, HIDDEN))
    wd = torch.ones((HIDDEN, emb))
    actual = _run(device, x, wg, wu, wd, count)[:count]
    expect = float(HIDDEN) * float(emb) * float(emb)
    rel = (actual - expect).abs() / expect
    # Per-emb-column max relative error: a non-uniform profile localises the bad columns.
    col = rel.max(dim=0).values
    bad = (col > 0.05).nonzero().flatten().tolist()
    print(f"[all_ones] expect {expect:.4e} actual[0,:8]={actual[0,:8].tolist()}")
    print(f"[all_ones] max rel {rel.max().item():.4f}  bad emb cols {len(bad)}/{emb} first={bad[:16]}")
    assert rel.max().item() < 0.05, f"max rel {rel.max().item()}"


def test_hidden_identity(device):
    """W_down = hidden->emb identity, so out[:, :HIDDEN] IS h. W_gate is per-hidden-tile
    constant so the expected value encodes the hidden TILE index."""
    emb, capacity, count = 7168, 1024, 32
    x = torch.ones((1, 1, capacity, emb))
    wg = torch.zeros((emb, HIDDEN))
    for nt in range(HIDDEN // TILE):
        wg[:, nt * TILE : (nt + 1) * TILE] = float(nt + 1)
    wu = torch.ones((emb, HIDDEN))
    wd = torch.zeros((HIDDEN, emb))
    for i in range(HIDDEN):
        wd[i, i] = 1.0
    actual = _run(device, x, wg, wu, wd, count)[:count, :HIDDEN]
    ref = _reference(x[0, 0, :count, :].to(torch.bfloat16), _bfp4(wg, device), _bfp4(wu, device), _bfp4(wd, device))[
        :, :HIDDEN
    ]
    # Per-hidden-tile ratio: 1.0 everywhere means the hidden mapping is right; a permutation
    # shows up as tile t carrying tile t''s value.
    got = actual[0].reshape(HIDDEN // TILE, TILE)[:, 0]
    want = ref[0].reshape(HIDDEN // TILE, TILE)[:, 0]
    print(f"[hidden_id] want[:16]={want[:16].tolist()}")
    print(f"[hidden_id] got [:16]={got[:16].tolist()}")
    ratio = got / want
    print(f"[hidden_id] ratio[:24]={[round(v, 3) for v in ratio[:24].tolist()]}")
    bad = (ratio - 1.0).abs().gt(0.05).nonzero().flatten().tolist()
    print(f"[hidden_id] bad hidden tiles {len(bad)}/{HIDDEN // TILE}: {bad}")
    assert not bad, f"hidden tiles wrong: {bad}"


def test_emb_contraction(device):
    """x is per-emb-tile constant, so the gate/up contraction sum encodes the emb axis.
    Isolates the Kg row split + the cross-column reduce tree."""
    emb, capacity, count = 7168, 1024, 32
    x = torch.zeros((1, 1, capacity, emb))
    for kt in range(emb // TILE):
        x[:, :, :, kt * TILE : (kt + 1) * TILE] = float(kt % 4 + 1)
    wg = torch.ones((emb, HIDDEN))
    wu = torch.ones((emb, HIDDEN))
    wd = torch.zeros((HIDDEN, emb))
    for i in range(HIDDEN):
        wd[i, i] = 1.0
    actual = _run(device, x, wg, wu, wd, count)[:count, :HIDDEN]
    ref = _reference(x[0, 0, :count, :].to(torch.bfloat16), _bfp4(wg, device), _bfp4(wu, device), _bfp4(wd, device))[
        :, :HIDDEN
    ]
    print(f"[emb_contract] want[0,:4]={ref[0, :4].tolist()} got[0,:4]={actual[0, :4].tolist()}")
    rel = ((actual - ref).abs() / ref.abs().clamp(min=1e-6)).max().item()
    print(f"[emb_contract] max rel {rel:.4f}")
    assert rel < 0.05, f"max rel {rel}"
