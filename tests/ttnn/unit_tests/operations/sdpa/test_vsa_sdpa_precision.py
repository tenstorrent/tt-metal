# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-row numerics of the streaming vsa_sdpa kernel at long lists (up to ~860 listed blocks per row):
every row must sit at the bf16 noise floor against an fp32 reference (no running-sum saturation or
truncation drift), which the exact writer-side row sums and the lazy anchor rescale guarantee."""

import os

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.test_vsa_sdpa import BLOCK, build_counts, build_indices, fine_attention_ref


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("kv_len", [65536], ids=["kv1024blocks"])
def test_vsa_sdpa_row_precision(device, kv_len):
    torch.manual_seed(3)
    heads, seq_len, dim = 2, 512, 128
    n_q_tiles, n_blocks = seq_len // BLOCK, kv_len // BLOCK
    w = ((n_blocks + 15) // 16) * 16
    q = torch.randn(1, heads, seq_len, dim, dtype=torch.bfloat16)
    k = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16)
    v = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16)
    indices = build_indices(heads, n_q_tiles, n_blocks, w, "nonuniform", 4)
    counts = build_counts(n_blocks, w, False, 5)
    dev = lambda t, lay, dt: ttnn.from_torch(t, device=device, layout=lay, dtype=dt)
    tt_q, tt_k, tt_v = (dev(x, ttnn.TILE_LAYOUT, ttnn.bfloat16) for x in (q, k, v))
    tt_idx = dev(indices.view(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    tt_counts = dev(counts.view(torch.int32).reshape(1, 1, 1, w), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    ref = fine_attention_ref(q.float(), k.float(), v.float(), indices, counts)
    out = ttnn.to_torch(ttnn.transformer.vsa_sdpa(tt_q, tt_k, tt_v, tt_idx, tt_counts, streaming=True)).float()
    ok, pcc = comp_pcc(ref, out, 0.999)
    print(f"\nPREC overall pcc={pcc:.6f}")
    worst = (1.0, 0.0, 1.0)
    for h in range(heads):
        for r in range(n_q_tiles):
            listed = int((indices[0, h, r] != 0xFFFFFFFF).sum())
            a, b = ref[0, h, r * BLOCK : (r + 1) * BLOCK], out[0, h, r * BLOCK : (r + 1) * BLOCK]
            row_pcc = comp_pcc(a, b, 0.0)[1]
            rel_err = ((a - b).norm() / a.norm()).item()
            scale = ((a * b).sum() / (a * a).sum()).item()  # least-squares gain of out vs ref
            print(f"PREC  h{h} r{r} listed={listed:4d} pcc={row_pcc:.6f} rel_err={rel_err:.5f} scale={scale:.4f}")
            worst = (min(worst[0], row_pcc), max(worst[1], rel_err), min(worst[2], 1.0 - abs(1.0 - scale)))
            assert row_pcc > 0.999, f"row h{h} r{r} pcc {row_pcc}"
            assert rel_err < 0.05, f"row h{h} r{r} rel_err {rel_err}"  # bf16 noise floor is ~0.025-0.03
            assert abs(scale - 1.0) < 0.01, f"row h{h} r{r} gain {scale}"  # no row-sum drift
    assert ok, pcc
    print(f"PREC worst row pcc={worst[0]:.6f} rel_err={worst[1]:.5f} gain_err={1.0 - worst[2]:.4f}")
