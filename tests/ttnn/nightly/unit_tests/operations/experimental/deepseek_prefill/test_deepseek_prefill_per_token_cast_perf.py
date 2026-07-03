# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time profiling harness for masked per_token_cast_back.

Runs the masked decompress over a hand-built dispatch buffer with different per-expert token
distributions that all carry the SAME total token count, so the device time isolates grid
utilization (the masked path splits work per expert -> one core per expert) from total work.
Each scenario is wrapped in a tracy signpost; run under the profiler and compare the balanced
case (work spread over experts_per_chip cores) against the hot cases (one core carries it all).

Run:
    pytest <this file> -k test_masked_decompress_perf --profile     # or under tt tracy tooling
"""

import pytest
import torch
import ttnn
from loguru import logger
from tracy import signpost

from models.common.utility_functions import is_blackhole

BLOCK_W = 128
E4M3_MAX = 448.0

# All distributions sum to TOTAL_TOKENS, spread over EXPERTS_PER_CHIP local experts. Counts are
# tile-multiples so region padding does not skew the comparison. experts_per_chip = len(counts).
EXPERTS_PER_CHIP = 8
# ISL = 5000: full_ISL caps a single expert at 5000 tokens, so total valid = 5000 lets one_hot put the
# whole load on one expert (peak == cap). Buffer capacity T is derived per pattern (tile-padded) and
# does not affect decompress time — only the valid token count / distribution does.
TOTAL_TOKENS = 5000
H = 7168  # real DeepSeek-V3 emb_dim -> blocks_per_row = 56, tokens span multiple compute blocks
ITERS = 10  # measured op invocations per scenario (after one warmup)


FULL_ISL = 8 * 5120  # dispatch_group_size(8) * ISL_per_chip(5120) = 40960 = per-expert cap ("max used")


def _distributions():
    """Dispatch-buffer fill scenarios at ISL=5120/chip. Two axes: fill level (max/half/sparse valid
    tokens) x distribution (balanced vs one_hot). one_hot piles the whole fill on one expert, up to the
    full_ISL cap. For masked, device time tracks the valid-token count (fill level) and — for the old
    per-expert split — the peak expert; the buffer capacity itself does not affect it."""
    e = EXPERTS_PER_CHIP
    # Realistic balanced steady state at ISL=5120/chip: per-chip valid = full_ISL(40960) * top_k(8) *
    # experts_per_chip/num_routed_experts(8/256) = 10240, evenly spread -> 1280/expert.
    return [
        # Balanced steady state: 1280 tokens/group * top_k(8) / 64 experts_in_group = 160 tokens/expert;
        # experts_per_chip=8 -> 1280 valid/device.
        ("bal_160_per_expert", [160] * e),  # 160/expert, 1280 valid/device
    ]


@pytest.fixture(autouse=True)
def _require_blackhole():
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")


def _build_inputs(counts, device):
    """Build (e4m3, dummy scale, counts, offsets, metadata) for a masked decompress of `counts`.

    Mirrors test_cast_back_masked: offsets are tile-padded cumulative (the real dispatch rule) and
    each valid row's per-128-block scales ride in its metadata tail (fields 5.., bit-cast to int32).
    """
    n_blocks = H // BLOCK_W
    metadata_len = 5 + n_blocks
    experts_per_chip = len(counts)

    offsets, acc = [], 0
    for c in counts:
        offsets.append(acc)
        acc += ((c + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    T = acc

    valid = [offsets[e] + i for e in range(experts_per_chip) for i in range(counts[e])]

    raw_e4m3 = (torch.randn(T, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    scales = (torch.rand(T, n_blocks) * 2.0 + 0.5).to(torch.float32)
    metadata = torch.zeros((1, 1, T, metadata_len), dtype=torch.int32)
    for r in valid:
        metadata[0, 0, r, 5:] = scales[r].view(torch.int32)

    e4m3_tt = ttnn.from_torch(
        raw_e4m3.reshape(1, 1, T, H),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale_dummy_tt = ttnn.from_torch(
        torch.zeros(1, 1, 1, n_blocks, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    counts_tt = ttnn.from_torch(
        torch.tensor(counts, dtype=torch.int32).reshape(1, 1, 1, experts_per_chip),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    offsets_tt = ttnn.from_torch(
        torch.tensor(offsets, dtype=torch.int32).reshape(1, 1, 1, experts_per_chip),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    metadata_tt = ttnn.from_torch(
        metadata, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return e4m3_tt, scale_dummy_tt, counts_tt, offsets_tt, metadata_tt, experts_per_chip


@pytest.mark.parametrize("M", [204800], ids=lambda v: f"M{v}")
def test_plain_decompress_perf_full_buffer(device, M):
    """Plain (non-masked) decompress over the full worst-case dispatch buffer (M x H).

    M=204800 = dispatch_group_size(8) * seq_len_per_chip(3200) * capacity_factor(8) — the whole flat
    per-chip buffer. Plain path splits M rows over the full grid, so this is the full-grid baseline.
    """
    torch.manual_seed(0)
    n_blocks = H // BLOCK_W
    raw_e4m3 = (torch.randn(M, H) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    scale = (torch.rand(M, n_blocks) * 2.0 + 0.5).to(torch.float32)

    e4m3_tt = ttnn.from_torch(
        raw_e4m3.reshape(1, 1, M, H),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale_tt = ttnn.from_torch(
        scale.reshape(1, 1, M, n_blocks),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def _run():
        return ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.bfloat16)

    _run()
    ttnn.synchronize_device(device)
    logger.info(f"plain full-buffer decompress M={M} H={H} iters={ITERS}")
    signpost(f"plain_decompress M={M} H={H}")
    for _ in range(ITERS):
        _run()
        ttnn.synchronize_device(device)


@pytest.mark.parametrize("per_expert", [5 * 1024, 25 * 1024], ids=lambda v: f"per{v}")
def test_masked_packed_perf(device, per_expert):
    """Fully-packed buffer: every one of EXPERTS_PER_CHIP experts holds `per_expert` valid tokens, so
    total valid = per_expert * 8 (40960 for 5*1024, 204800 for 25*1024). Measures the masked decompress
    over the whole packed buffer; run under both the per-expert build (no full grid) and block-parallel."""
    torch.manual_seed(0)
    counts = [per_expert] * EXPERTS_PER_CHIP
    e4m3_tt, scale_tt, counts_tt, offsets_tt, metadata_tt, experts_per_chip = _build_inputs(counts, device)

    def _run():
        return ttnn.experimental.deepseek_prefill.per_token_cast_back(
            e4m3_tt,
            scale_tt,
            output_dtype=ttnn.bfloat16,
            expert_token_counts=counts_tt,
            expert_region_offsets=offsets_tt,
            metadata=metadata_tt,
            experts_per_chip=experts_per_chip,
            dispatch_group_size=1,
        )

    _run()
    ttnn.synchronize_device(device)
    logger.info(f"packed per_expert={per_expert} total={per_expert * EXPERTS_PER_CHIP} H={H} iters={ITERS}")
    signpost(f"masked_packed per_expert={per_expert} total={per_expert * EXPERTS_PER_CHIP}")
    for _ in range(ITERS):
        _run()
        ttnn.synchronize_device(device)


@pytest.mark.parametrize("output_dtype", ["bfloat16"])
@pytest.mark.parametrize("name, counts", _distributions(), ids=lambda v: v if isinstance(v, str) else "")
def test_masked_decompress_perf(device, output_dtype, name, counts):
    """One scenario per invocation: warm up, then ITERS signpost-bracketed op calls to profile."""
    torch.manual_seed(0)
    e4m3_tt, scale_tt, counts_tt, offsets_tt, metadata_tt, experts_per_chip = _build_inputs(counts, device)
    ttnn_out_dtype = getattr(ttnn, output_dtype)

    def _run():
        return ttnn.experimental.deepseek_prefill.per_token_cast_back(
            e4m3_tt,
            scale_tt,
            output_dtype=ttnn_out_dtype,
            expert_token_counts=counts_tt,
            expert_region_offsets=offsets_tt,
            metadata=metadata_tt,
            experts_per_chip=experts_per_chip,
            dispatch_group_size=1,
        )

    # Warmup: compiles + caches the program so the measured region excludes JIT.
    _run()
    ttnn.synchronize_device(device)

    logger.info(f"scenario={name} counts={counts} total={sum(counts)} H={H} iters={ITERS}")
    # Synchronize per iteration so each op's DEVICE FW DURATION is isolated: back-to-back dispatch
    # on shared buffers lets the profiler's per-op window absorb the next op's double-buffer stall.
    signpost(f"masked_decompress {name} total={sum(counts)} H={H} experts={experts_per_chip}")
    for _ in range(ITERS):
        _run()
        ttnn.synchronize_device(device)
