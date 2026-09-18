# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mixed Llama-3B TP1 decode traffic benchmark for Tensor Prefetcher MPFE policies.

Each replay queues one receiver-contiguous per-device FF1 weight, runs decode
SDPA against DRAM-resident K/V while the DRISCs fill a whole-layer GCB, then
consumes the weight with the FF1 matmul. Full buffering lets the request finish
and restore dynamic idle weights while ordinary SDPA traffic is still active.
"""

import math
import os
import time

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.prefetcher_common import (
    make_recv_contig_weight,
    require_tensor_prefetcher,
    round_up,
)
from tests.ttnn.unit_tests.operations.transformers.mpfe_benchmark_utils import (
    append_benchmark_jsonl,
    resolve_mpfe_benchmark_weights,
    weight_result_fields,
)


pytestmark = run_for_blackhole("Tensor prefetcher requires Blackhole")

_BATCH = 32
_NUM_HEADS = 24
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_FF1_K = 3072
_FF1_N = 8192
_RECEIVERS_PER_BANK = 8
_BF8_BYTES_PER_ELEMENT = 1088 / 1024.0


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    require_tensor_prefetcher(device)


def _production_or_compact_ring(num_dram_banks: int, ring_size: int) -> list[tuple[int, int]]:
    if num_dram_banks != 8:
        return [(position % 8, position // 8) for position in range(ring_size)]

    from models.tt_transformers.tt.prefetcher import ARCH_CONFIG, generate_sender_receiver_mapping

    config = ARCH_CONFIG["blackhole"]
    mapping = generate_sender_receiver_mapping(num_receivers_per_sender=_RECEIVERS_PER_BANK)
    ordered_senders = [
        (config["sender_cols"]["left"], y) for y in config["bank_ordered_y_coords"]["left"]
    ] + [(config["sender_cols"]["right"], y) for y in config["bank_ordered_y_coords"]["right"]]
    receivers_by_y: dict[int, list[tuple[int, int]]] = {}
    for sender in ordered_senders:
        receivers_by_y.setdefault(sender[1], []).extend(mapping[sender])
    return [core for y in sorted(receivers_by_y) for core in sorted(receivers_by_y[y])]


def _singleton_core_set(cores: list[tuple[int, int]]) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
            for x, y in sorted(set(cores), key=lambda core: (core[1], core[0]))
        ]
    )


def _ff1_program_config(ring_cols: int, ring_rows: int, n_padded: int, ring_size: int):
    out_block_w = n_padded // ring_size // ttnn.TILE_SIZE
    out_subblock_w = min(out_block_w, 8)
    while out_subblock_w > 1 and out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(ring_cols, ring_rows),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=1,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=_RECEIVERS_PER_BANK,
        untilize_out=False,
        stream_in1=False,
    )


def _sdpa_k_chunk_size(context: int) -> int:
    return 128 if context <= 1024 else 512


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 23887872}],
    indirect=True,
)
def test_mpfe_mixed_llama3b_ff1_sdpa(device):
    """Overlap DRISC-prefetched FF1 with ordinary Llama-3B TP1 decode SDPA traffic."""
    context = int(os.environ.get("BENCH_SDPA_CONTEXT", "1024"))
    trace_repeats = int(os.environ.get("BENCH_TRACE_REPEATS", "20"))
    assert context >= 128 and context % 128 == 0
    assert trace_repeats > 0

    num_dram_banks = device.dram_grid_size().x
    ring_size = num_dram_banks * _RECEIVERS_PER_BANK
    ring_cols = 8
    assert ring_size % ring_cols == 0
    ring_rows = ring_size // ring_cols
    ring_cores = _production_or_compact_ring(num_dram_banks, ring_size)
    assert len(ring_cores) == ring_size
    receiver_core_set = _singleton_core_set(ring_cores)

    compute_grid = device.compute_with_storage_grid_size()
    if compute_grid.x < 8 or compute_grid.y < 8:
        pytest.skip(f"Llama-3B SDPA benchmark requires an 8x8 worker grid, got {compute_grid}")
    sdpa_cores = [(x, y) for y in range(8) for x in range(8)]
    sdpa_core_set = _singleton_core_set(sdpa_cores)
    worker_core_set = _singleton_core_set(ring_cores + sdpa_cores)
    worker_sub_device = ttnn.SubDevice([worker_core_set])
    sub_device_manager = device.create_sub_device_manager([worker_sub_device], 0)
    device.load_sub_device_manager(sub_device_manager)
    worker_sub_device_id = ttnn.SubDeviceId(0)
    device.set_sub_device_stall_group([worker_sub_device_id])

    k_padded = round_up(_FF1_K, ring_size * ttnn.TILE_SIZE)
    n_padded = round_up(_FF1_N, ring_size * ttnn.TILE_SIZE)
    torch.manual_seed(0x4D495845)
    pt_weight = torch.zeros((1, 1, k_padded, n_padded))
    pt_weight[:, :, :_FF1_K, :_FF1_N] = torch.randn((1, 1, _FF1_K, _FF1_N))
    pt_activation = torch.zeros((1, 1, _BATCH, k_padded))
    pt_activation[:, :, :, :_FF1_K] = torch.randn((1, 1, _BATCH, _FF1_K))

    tt_weight = make_recv_contig_weight(
        device,
        pt_weight,
        num_dram_banks,
        ring_size,
        ttnn.bfloat8_b,
        distribution_strategy=ttnn.ShardDistributionStrategy.CONTIGUOUS_1D,
    )
    activation_mem_config = ttnn.create_sharded_memory_config(
        shape=(_BATCH, k_padded // ring_size),
        core_grid=receiver_core_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_activation = ttnn.from_torch(
        pt_activation,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=activation_mem_config,
    )

    bank_to_receivers = [
        (
            bank,
            _singleton_core_set(
                ring_cores[bank * _RECEIVERS_PER_BANK : (bank + 1) * _RECEIVERS_PER_BANK]
            ),
        )
        for bank in range(num_dram_banks)
    ]
    ff1_program_config = _ff1_program_config(ring_cols, ring_rows, n_padded, ring_size)
    # The production TP1 weight fits one complete receiver shard under the
    # 65,535-page GCB limit even on a seven-bank harvested device. Unlike the
    # previous shallow streaming benchmark, this allows prefetch to finish
    # before SDPA and creates a real dynamic-idle interval.
    gcb_size = int(k_padded * (n_padded // ring_size) * _BF8_BYTES_PER_ELEMENT)
    assert gcb_size // 16 < 65535
    gcb = ttnn.experimental.create_global_circular_buffer_for_matmul_1d(
        device,
        [ff1_program_config],
        [tt_weight],
        bank_to_receivers,
        gcb_size,
        support_multi_receiver_shards=False,
    )
    output_mem_config = ttnn.create_sharded_memory_config(
        shape=(_BATCH, n_padded // ring_size),
        core_grid=receiver_core_set,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
        dst_full_sync_en=True,
    )
    ff1_output = ttnn.from_torch(
        torch.zeros((1, 1, _BATCH, n_padded)),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_mem_config,
    )

    def run_ff1():
        return ttnn.linear(
            tt_activation,
            tt_weight,
            program_config=ff1_program_config,
            memory_config=output_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            global_cb=gcb,
            optional_output_tensor=ff1_output,
            sub_device_id=worker_sub_device_id,
        )

    pt_k = torch.randn((_BATCH, _NUM_KV_HEADS, context, _HEAD_DIM))
    pt_v = torch.randn((_BATCH, _NUM_KV_HEADS, context, _HEAD_DIM))
    pt_q = torch.randn((1, _BATCH, _NUM_HEADS, _HEAD_DIM))
    tt_k = ttnn.as_tensor(
        pt_k, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_v = ttnn.as_tensor(
        pt_v, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    q_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            _singleton_core_set([(x, y) for y in range(4) for x in range(8)]),
            (_NUM_HEADS, _HEAD_DIM),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    # Llama-3B has 24 logical Q heads, while tiled shards require a physical
    # multiple of 32. Upload the padded storage first, preserve 24 as the
    # logical shape, then shard the resulting logical/padded tensor into L1.
    pt_q_padded = torch.zeros((1, _BATCH, 32, _HEAD_DIM))
    pt_q_padded[:, :, :_NUM_HEADS, :] = pt_q
    tt_q_padded = ttnn.as_tensor(
        pt_q_padded,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_q = ttnn.reshape(
        tt_q_padded,
        (1, _BATCH, _NUM_HEADS, _HEAD_DIM),
        padded_shape=(1, _BATCH, 32, _HEAD_DIM),
    )
    tt_q = ttnn.to_memory_config(tt_q, q_mem_config)
    current_positions = ttnn.Tensor(
        torch.full((_BATCH,), context - 1, dtype=torch.int32), ttnn.int32
    ).to(device)
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        sub_core_grids=sdpa_core_set,
        q_chunk_size=32,
        k_chunk_size=_sdpa_k_chunk_size(context),
        exp_approx_mode=False,
    )
    sdpa_compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    def run_sdpa():
        return ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=current_positions,
            scale=_HEAD_DIM**-0.5,
            program_config=sdpa_program_config,
            compute_kernel_config=sdpa_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    block_count = ttnn.experimental.tensor_prefetcher_block_count_for_matmul_1d(
        ff1_program_config, tt_weight, gcb
    )
    policy = resolve_mpfe_benchmark_weights()
    trace_id = None
    prefetcher_started = False
    try:
        ttnn.experimental.start_tensor_prefetcher(device, **policy.start_kwargs())
        prefetcher_started = True

        # One model-shaped warmup also validates that both cached programs consume
        # exactly one balanced prefetch request before trace capture.
        ttnn.experimental.queue_tensor_prefetcher_request(
            device, [(tt_weight, block_count)], global_cb=gcb
        )
        sdpa_warmup = run_sdpa()
        ff1_warmup = run_ff1()
        sdpa_torch = ttnn.to_torch(sdpa_warmup)
        ff1_torch = ttnn.to_torch(ff1_warmup)

        # Validate four query heads sharing KV head zero without materializing a
        # full repeated-head PyTorch reference.
        reference_scores = torch.matmul(
            pt_q[0, 0, :4].float(), pt_k[0, 0].float().transpose(-2, -1)
        ) * (_HEAD_DIM**-0.5)
        reference_sdpa = torch.matmul(torch.softmax(reference_scores, dim=-1), pt_v[0, 0].float())
        sdpa_pass, sdpa_message = comp_pcc(reference_sdpa, sdpa_torch[0, 0, :4].float(), 0.99)
        assert sdpa_pass, f"SDPA PCC failed: {sdpa_message}"
        reference_ff1 = pt_activation[:, :, :, :_FF1_K].float() @ pt_weight[
            :, :, :_FF1_K, :_FF1_N
        ].float()
        ff1_pass, ff1_message = comp_pcc(reference_ff1, ff1_torch[:, :, :, :_FF1_N], 0.99)
        assert ff1_pass, f"FF1 PCC failed: {ff1_message}"

        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        capture_open = True
        try:
            ttnn.experimental.queue_tensor_prefetcher_request(
                device,
                [(tt_weight, block_count)],
                global_cb=gcb,
                capture_into_trace=True,
            )
            traced_sdpa_output = run_sdpa()
            traced_ff1_output = run_ff1()
        finally:
            if capture_open:
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
                capture_open = False

        start = time.perf_counter()
        for _ in range(trace_repeats):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start

        # Keep trace-owned outputs live through the final replay.
        assert traced_sdpa_output is not None and traced_ff1_output is not None
    finally:
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        if prefetcher_started:
            ttnn.experimental.stop_tensor_prefetcher(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sub_device_manager)

    prefetch_bytes_per_step = k_padded * n_padded * _BF8_BYTES_PER_ELEMENT
    ordinary_bytes_per_step = 2 * _BATCH * _NUM_KV_HEADS * context * _HEAD_DIM * _BF8_BYTES_PER_ELEMENT
    ratio = prefetch_bytes_per_step / ordinary_bytes_per_step
    per_step_us = elapsed / trace_repeats * 1e6
    ff1_tflops = 2 * _BATCH * _FF1_K * _FF1_N * trace_repeats / elapsed / 1e12
    logger.info(
        f"[mpfe_mixed] policy={policy.name} context={context} repeats={trace_repeats} "
        f"prefetch/ordinary={ratio:.3f} step={per_step_us:.2f}us FF1={ff1_tflops:.4f}TFLOP/s"
    )
    append_benchmark_jsonl(
        {
            "benchmark": "mpfe_mixed_llama3b_ff1_sdpa",
            **weight_result_fields(policy),
            "num_dram_banks": num_dram_banks,
            "ring_size": ring_size,
            "dual_senders": True,
            "gcb_buffered_blocks": block_count,
            "sdpa_context": context,
            "trace_repeats": trace_repeats,
            "elapsed_ms": elapsed * 1e3,
            "per_step_us": per_step_us,
            "ff1_tflops": ff1_tflops,
            "prefetch_bytes_per_step": prefetch_bytes_per_step,
            "ordinary_bytes_per_step": ordinary_bytes_per_step,
            "prefetch_to_ordinary_ratio": ratio,
        }
    )
