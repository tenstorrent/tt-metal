# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The streaming vsa_sdpa kernel must be bit-exact across repeated launches on one input: it holds
per-row state in resident L1 tiles across chunks and hands row sums to the writer core, so any
uninitialized read or missing cross-RISC ordering shows up here as run-to-run differences."""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.test_vsa_sdpa import BLOCK, SENTINEL, build_counts


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("heads,seq_len,kv_len", [(2, 4096, 16384)])
def test_vsa_sdpa_repeat_bit_exact(device, heads, seq_len, kv_len):
    torch.manual_seed(0)
    dim = 128
    n_q_tiles, n_blocks = seq_len // BLOCK, kv_len // BLOCK
    k_sel = 32
    w_full = ((max(n_blocks, k_sel) + 15) // 16) * 16
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(1, heads, seq_len, dim, dtype=torch.bfloat16)
    k = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16)
    v = torch.randn(1, heads, kv_len, dim, dtype=torch.bfloat16)
    counts = build_counts(n_blocks, w_full, True, 2)
    assembled = torch.full((1, heads, n_q_tiles, w_full), SENTINEL, dtype=torch.int64)
    for h in range(heads):
        for qt in range(n_q_tiles):
            row = torch.randperm(n_blocks, generator=gen)[:k_sel].sort().values
            assembled[0, h, qt, : row.numel()] = row
    dev = lambda x, lay=ttnn.TILE_LAYOUT, dt=ttnn.bfloat16: ttnn.from_torch(x, device=device, layout=lay, dtype=dt)
    tt_q, tt_k, tt_v = dev(q), dev(k), dev(v)
    tt_counts = dev(counts.view(torch.int32).reshape(1, 1, 1, w_full), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    tt_idx = dev(assembled.to(torch.uint32).view(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    outs = [ttnn.to_torch(ttnn.transformer.vsa_sdpa(tt_q, tt_k, tt_v, tt_idx, tt_counts)) for _ in range(4)]
    for i, o in enumerate(outs[1:], start=1):
        diff_rows = (outs[0] != o).any(dim=-1).sum().item()
        assert torch.equal(outs[0], o), f"launch {i} differs from launch 0 on {diff_rows} rows"
