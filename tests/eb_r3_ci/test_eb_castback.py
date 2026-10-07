# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723): test_token_count_aware_cast_back as the module runs it on Blackhole CI (scales from the
metadata), but only the rows the op writes, [0, total_valid_rows), are read back. The op leaves the tail of its output
unwritten, so a whole-output hash also compares stale DRAM contents."""
import importlib.util

import pytest
import torch
import ttnn

_P = "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_deepseek_prefill_per_token_cast.py"
_spec = importlib.util.spec_from_file_location("eb_castback_src", _P)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("label, counts", m.TOKEN_COUNT_AWARE_CASES, ids=[c[0] for c in m.TOKEN_COUNT_AWARE_CASES])
@pytest.mark.parametrize("readback", ["prefix", "whole"])
def test_castback_rows(device, label, counts, output_dtype, readback):
    torch.manual_seed(0)
    H = m.KimiK27Config.EMB_SIZE
    experts_per_chip = len(counts)
    num_routed_experts = 2 * experts_per_chip
    global_expert_idx_table = [2 * s + 1 for s in range(experts_per_chip)]
    expert_region_offsets = [0] * num_routed_experts
    expert_token_counts = [0] * num_routed_experts
    running_offset = 0
    for local_slot, token_count in enumerate(counts):
        global_id = global_expert_idx_table[local_slot]
        expert_region_offsets[global_id] = running_offset
        expert_token_counts[global_id] = token_count
        running_offset += m._ceil_tile(token_count)
    capacity = m.MAX_DISPATCH_BUFFER_TOKENS
    total_valid_rows = min(running_offset, capacity)

    input_e4m3 = (torch.randn(capacity, H) * 3.0).clamp(-m.E4M3_MAX, m.E4M3_MAX).to(torch.float8_e4m3fn)
    input_scale = torch.rand(capacity, H // m.BLOCK_W) * 4.0 - 2.0
    e4m3_tt = m._make_e4m3_from_torch(input_e4m3, device=device)
    metadata_tt = ttnn.from_torch(
        m._pack_scale_metadata(input_scale),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = ttnn.experimental.deepseek_prefill.per_token_cast_back(
        e4m3_tt,
        None,
        token_count_aware=True,
        expert_region_offsets=m.create_u32_tensor(device, expert_region_offsets),
        expert_token_counts=m.create_u32_tensor(device, expert_token_counts),
        global_expert_idx_table=m.create_u32_tensor(device, global_expert_idx_table),
        experts_per_chip=experts_per_chip,
        output_dtype=output_dtype,
        metadata=metadata_tt,
    )
    if readback == "prefix":
        out_tt = ttnn.slice(out_tt, [0, 0], [total_valid_rows, H])
    out = ttnn.to_torch(out_tt).float()[:total_valid_rows]
    golden = input_e4m3.float()[:total_valid_rows] * input_scale[:total_valid_rows].repeat_interleave(m.BLOCK_W, dim=-1)
    if output_dtype == ttnn.bfloat16:
        golden = golden.to(torch.bfloat16).float()
    normal = input_e4m3.float()[:total_valid_rows].abs() > 2.0**-6
    assert torch.allclose(out[normal], golden[normal], rtol=1e-2, atol=1e-3)
