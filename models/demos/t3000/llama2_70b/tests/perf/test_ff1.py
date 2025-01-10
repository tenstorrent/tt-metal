# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import math
import torch
import pytest
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
    # get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor

FF_DIM = int(32 * 1024 / 8)
USE_ACC = True


class TtFF1:
    def __init__(self, device, state_dict, base_url, layer_num, model_config, configuration):
        self.model_config = model_config
        self.weight = torch2tt_tensor(
            torch.randn(8 * 1024, FF_DIM),
            device,
            tt_memory_config=self.model_config["FF1_MM_WEIGHTS_MEMCFG"],
            tt_dtype=self.model_config["FF1_MM_WEIGHTS_DTYPE"],
        )

    def __call__(self, x, prog_config, output_config):
        # Assume interleaved input
        ff_out = ttnn.matmul(
            x,
            self.weight,
            program_config=prog_config,
            memory_config=output_config,
            dtype=self.model_config["FF1_MM_OUTPUT_DTYPE"],
        )
        x.deallocate()

        return ff_out


def run_test_ff1(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config,
    num_devices,
    # tt_cache_path,
    # model_location_generator,
):
    pt_in = torch.randn(seq_len, batch, 8 * 1024)

    # Loop over valid grid ranges for width sharding outputs and activations
    start_n_cores = 9 if USE_ACC else 5
    for n_cores in range(start_n_cores, 64):
        start_idx = (0, 0)
        num_rows = (n_cores - 1) // 8
        extra_cores = (n_cores - 1) % 8
        end_idx = (num_rows, extra_cores)

        cols1_tiles = int(8 * 1024 / 32)
        cols2_tiles = int(FF_DIM / 32)
        # common factors of the above variables
        if not (cols1_tiles % n_cores == 0 and cols2_tiles % n_cores == 0):
            print(f"num_cores: {n_cores}. core_range {start_idx}, {end_idx} not valid")
            continue

        print(f"num_cores: {n_cores}. core_range {start_idx}, {end_idx}")

        compute_with_storage_grid_size = ((end_idx[1] - start_idx[1]) + 1, (end_idx[0] - start_idx[0]) + 1)
        in0_block_w = int(cols1_tiles / n_cores)
        per_core_N = int(cols2_tiles / n_cores)
        max_dst_size = 4 if USE_ACC else 8
        out_subblock_w = max([i for i in range(1, max_dst_size + 1) if (per_core_N % i) == 0])

        prog_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=compute_with_storage_grid_size,
            in0_block_w=in0_block_w,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=out_subblock_w,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=per_core_N,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
        )

        output_config = model_config["FF1_MM_OUTPUT_MEMCFG"]

        inp_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(*start_idx),
                            ttnn.CoreCoord(*end_idx),
                        ),
                    }
                ),
                [
                    32,
                    int(8 * 1024 / n_cores),
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

        tt_in = torch2tt_tensor(pt_in, device, tt_memory_config=inp_mem_config)

        # TT hardware execution -------------------------------------------------------------
        tt_model = TtFF1(device, None, None, None, model_config, None)

        tt_out = tt_model(tt_in, prog_config, output_config)
        tt_out.deallocate()


@pytest.mark.parametrize("n_devices", (8, 4))
@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (("llama-2-70B", 32, 1, 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_ff1(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
    n_devices,
    use_program_cache,
):
    model_config = get_model_config(model_config_str, num_devices=n_devices)

    run_test_ff1(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
        # tt_cache_path,
        # model_location_generator,
    )
