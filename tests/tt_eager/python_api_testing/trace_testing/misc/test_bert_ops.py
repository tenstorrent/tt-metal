# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import ttnn


from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from loguru import logger
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


@pytest.mark.skipif(is_wormhole_b0(), reason="Unsupported parallelizations for WH B0")
@pytest.mark.parametrize("fidelity", [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2], ids=["LoFi", "HiFi2"])
@pytest.mark.parametrize(
    "in1_in_dram, out_sharded, in0_sharded, M, K, N, activation",
    [
        # (False, True, True, 12*128, 1024, 1024, None),
        # (False, True, True, 12*128, 4096, 1024, None),
        # (False, True, True, 12*128, 8192, 1024, None),
        # one core
        # (False, False, False, 128, 256, 128, None),
        # # in1-L1-fusedQKV
        (False, True, True, 4608, 1024, 3072, None),  # both sharded
        (False, True, False, 4608, 1024, 3072, None),  # out sharded, in0 interleaved
        (False, False, True, 4608, 1024, 3072, None),  # out interleaved, in0 sharded
        (False, False, False, 4608, 1024, 3072, None),  # out interleaved, in0 interleaved
    ],
)
@pytest.mark.parametrize("enable_async", [True, False])
class TestBertOpsTrace:
    # TODO: Not all ops here take in cq id, only works with 0 for now
    def run_bert_linear(
        self,
        device,
        fidelity,
        in0_sharded,
        out_sharded,
        in1_in_dram,
        M,
        K,
        N,
        activation,
        enable_async,
        cq_id,
    ):
        device.enable_async(enable_async)
        has_bias = False
        in0_shape = [1, 1, M, K]
        in1_shape = [1, 1, K, N]
        bias_shape = [1, 1, N]
        out_shape = [1, 1, M, N]
        grid_size = (12, 8)
        # grid_size = (2, 2)
        shard_shape = [M // grid_size[0], K // grid_size[1]]  # shard height, width

        in0_block_w = K // grid_size[1] // 32  # 16
        in0_block_h = M // grid_size[0] // 32
        out_block_h = M // grid_size[0] // 32
        out_block_w = N // grid_size[1] // 32

        if out_block_w <= 8:
            out_subblock_w = out_block_w
            out_subblock_h = 8 // out_subblock_w
        else:
            out_subblock_h = 1
            out_subblock_w = 8 // out_subblock_h
            while out_block_w % out_subblock_w != 0:
                out_subblock_w = out_block_w // 2

        # in0_block_w = K // grid_size[1] // 32
        # out_subblock_w = 4
        # out_subblock_h = 4

        logger.debug("in0 block w h " + str(in0_block_w * 32) + " " + str(in0_block_h * 32))
        logger.debug("in1 block w h " + str(out_block_w * 32) + " " + str(in0_block_w * 32))
        logger.debug("out block w h " + str(out_block_w * 32) + " " + str(out_block_h * 32))
        logger.debug("out subblock w h " + str(out_subblock_w * 32) + " " + str(out_subblock_h * 32))

        interleaved_mem_config_L1 = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.L1,
        )
        interleaved_mem_config_DRAM = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
            buffer_type=ttnn.BufferType.DRAM,
        )
        sharded_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )

        in0 = torch.randn(in0_shape).bfloat16().float()
        in1 = torch.randn(in1_shape).bfloat16().float()
        bias = torch.randn(bias_shape).bfloat16().float()
        in0_t_res = torch2tt_tensor(in0, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat8_b)

        if in1_in_dram:
            in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat8_b)
        else:
            in1_t = torch2tt_tensor(in1, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttnn.bfloat8_b)

        output_mem_config = sharded_mem_config if out_sharded else interleaved_mem_config_L1

        bias_t = pad_by_zero(bias, device, tt_memory_config=interleaved_mem_config_L1, tt_dtype=ttnn.bfloat8_b)[0]

        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            transpose_mcast=True,
            # transpose_mcast=False,
            fused_activation=activation,
        )

        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=True)

        trace_loops = 4

        def run_ops(in0_t_res):
            if in0_sharded:
                in0_t = ttnn.interleaved_to_sharded(
                    in0_t_res,
                    grid_size,
                    [M // grid_size[0], K // grid_size[1]],
                    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    ttnn.ShardOrientation.COL_MAJOR,
                )
            else:
                in0_t = ttnn.clone(in0_t_res, memory_config=interleaved_mem_config_L1)

            if has_bias:
                output_t = ttnn.linear(
                    in0_t,
                    in1_t,
                    bias=bias_t,
                    program_config=program_config,
                    memory_config=output_mem_config,
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                output_t = ttnn.matmul(
                    in0_t,
                    in1_t,
                    program_config=program_config,
                    memory_config=output_mem_config,
                    compute_kernel_config=compute_kernel_config,
                )
            if out_sharded:
                output_t = ttnn.sharded_to_interleaved(output_t, interleaved_mem_config_L1)
            return output_t

        # Compile
        run_ops(in0_t_res)
        # Capture
        logger.info("Start Trace capture")
        tid = ttnn.begin_trace_capture(device, cq_id=cq_id)
        output_t_res = run_ops(in0_t_res)
        ttnn.end_trace_capture(device, tid, cq_id=cq_id)
        logger.info("Trace captured")

        for iter in range(trace_loops):
            in0 = torch.randn(in0_shape).bfloat16().float()
            in0_t_updated = torch2tt_tensor(
                in0, None, tt_memory_config=interleaved_mem_config_DRAM, tt_dtype=ttnn.bfloat8_b
            )
            ttnn.copy_host_to_device_tensor(in0_t_updated, in0_t_res)
            logger.info(f"Running iteration {iter}")
            ttnn.execute_trace(device, tid, cq_id=cq_id, blocking=True)

            pt_out = in0 @ in1

            if has_bias:
                pt_out = pt_out + bias

            if activation != None:
                pt_out = torch.nn.functional.gelu(pt_out)
            tt_out = tt2torch_tensor(output_t_res)

            passing, output = comp_pcc(pt_out, tt_out)
            logger.info(output)
            assert passing

        # Done with the trace, can deallocate the buffers now.
        ttnn.release_trace(device, tid)
        device.enable_async(False)

    @pytest.mark.parametrize("device_params", [{"trace_region_size": 34816}], indirect=True)
    def test_bert_linear_1cq_initialized(
        self,
        device,
        fidelity,
        in0_sharded,
        out_sharded,
        in1_in_dram,
        M,
        K,
        N,
        activation,
        use_program_cache,
        function_level_defaults,
        enable_async,
    ):
        self.run_bert_linear(
            device,
            fidelity,
            in0_sharded,
            out_sharded,
            in1_in_dram,
            M,
            K,
            N,
            activation,
            enable_async,
            0,
        )

    @pytest.mark.parametrize("cq_id", [0])
    @pytest.mark.parametrize("device_params", [{"trace_region_size": 34816, "num_hw_cqs": 2}], indirect=True)
    def test_bert_linear_2cqs_initialized(
        self,
        device,
        fidelity,
        in0_sharded,
        out_sharded,
        in1_in_dram,
        M,
        K,
        N,
        activation,
        use_program_cache,
        function_level_defaults,
        enable_async,
        cq_id,
    ):
        self.run_bert_linear(
            device,
            fidelity,
            in0_sharded,
            out_sharded,
            in1_in_dram,
            M,
            K,
            N,
            activation,
            enable_async,
            cq_id,
        )
