# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The real flat gate_up_compute.cpp, compiled with the GLM e16 plan's named args, alone on one core: x and weight
blocks handed over as fast as the rings free (kf_feed.cpp), h drained (kf_drain.cpp). Its GU-SB zone (run under
TT_METAL_DEVICE_PROFILER=1) against the bench's 20.3 us: the kernel by itself vs its surroundings in the expert."""

import pytest
from loguru import logger

import ttnn
from models.demos.glm_ks_expert.bench_util import cb, crs, kpath

FLAT_K = "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/device/kernels/gate_up_compute.cpp"
NAMED = dict(
    combine_handoff_arg=0, combine_report_arg=0, combine_writer_index_arg=0, down_block_tiles=0, down_columns=0,
    down_core_count=32, down_done_sem=7, down_k_block_tiles=0, down_weight_banks=8, down_weight_ring_blocks=0,
    down_weights_per_load=1, gate_up_block_bytes=9216, gate_up_block_tiles=16, gate_up_blocks_per_expert=24,
    gate_up_core_count=64, gate_up_data_sem=4, gate_up_grant_sem=1, gate_up_mailbox_sem=0, gate_up_read_pieces=1,
    gate_up_reads_in_flight=2, gate_up_ring_regions=2, gate_up_tile_pairs=1, gate_up_units=0, gate_up_units_h_tiles=0,
    gate_up_weight_ring_blocks=48, go_sem=6, h_arrived_sem=5, h_buffers=3, h_gather_sem_0=8, h_gather_sem_1=14,
    h_gather_sem_2=15, h_slot_free_sem=12, h_tile_bytes=1088, half_width_gate_up=0, hidden_tiles=192,
    intermediate_tiles=64, local_experts=16, max_experts=16, pin_chunk_sub_blocks=1, pin_min_ratio=8, row_pass_rows=2,
    rows_per_sub_block=128, schedule_page_bytes=512, schedule_scratch_bytes=2048, super_block_tiles=32,
    super_blocks_per_sub_block=6, x_arrived_sem=9, x_block_k_tiles=8, x_chain_pieces=4, x_consumed_sem=2,
    x_credit_sem=0, x_end_sem=3, x_indexed=0, x_ring_blocks=24, x_slot_free_sem=10, x_tile_bytes=1088, y_row_major=0,
    y_row_major_rows=0, expert_activation=0, expert_activation_beta_bits=1082130432,
    expert_activation_linear_beta_bits=1103626240, expert_activation_limit_bits=1092616192,
    expert_activation_alpha_bits=1071242019,
)  # fmt: skip


def test_flat_compute_alone(device):
    core = (0, 0)
    g = crs([core])
    ne, sbe, nb, mt = 16, 3, 24, 4
    named = list(NAMED.items())
    rt_f, rt_d = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rt_f[0][0] = [ne, sbe, nb, mt * 8, 24, 16, mt]
    rt_d[0][0] = [ne * sbe, mt]
    dm = lambda p, n: ttnn.DataMovementConfigDescriptor(processor=p, noc=n)
    kernels = [
        ttnn.KernelDescriptor(kernel_source=kpath("kf_feed.cpp"), core_ranges=g, named_compile_time_args=named,
                              runtime_args=rt_f, config=dm(ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1)),
        ttnn.KernelDescriptor(kernel_source=kpath("kf_drain.cpp"), core_ranges=g, runtime_args=rt_d,
                              config=dm(ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0)),
        ttnn.KernelDescriptor(kernel_source=FLAT_K, core_ranges=g, named_compile_time_args=named,
                              config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi,
                                                                  fp32_dest_acc_en=True, dst_full_sync_en=False)),
    ]  # fmt: skip
    cbs = [
        cb(0, ttnn.bfloat8_b, 24 * mt * 8, g),
        cb(1, ttnn.bfloat4_b, 48 * 16, g),
        cb(3, ttnn.bfloat8_b, 3 * mt, g),
        cb(6, ttnn.uint32, 1, g, page=512),
        cb(16, ttnn.bfloat16, 1, g),
    ]
    ttnn.generic_op([ttnn.from_torch(__import__("torch").zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)] * 2,
                    ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    logger.info("FLATALONE done")
