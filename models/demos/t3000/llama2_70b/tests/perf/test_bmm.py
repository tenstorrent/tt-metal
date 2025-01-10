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


TILE_SIZE = 32

N_HEADS = 8
N_HEADS_PADDED = 32
BATCH = 32
HEAD_DIM = 128
SEQ_LEN = 128


class TT_bmm:
    def __init__(self, model_config):
        self.model_config = model_config

    def __call__(self, q, k, q_mem_config, prog_config, output_config):
        # Q: (1, n_heads, batch, head_dim) sharded by heads on 8 cores
        # K: (1, batch, head_dim, seq_len) sharded by batch on 8 cores

        #### OPTION 1: transpose then tilize with padding
        # # Transpose Q in DRAM. This converts to RM
        # q = ttnn.transpose(
        #     q,
        #     -2,
        #     -3,
        # )

        # # Tilize with zero padding plus shard to L1 by batch
        # # output_mem_config leads to `bad optional access` error
        # q = ttnn.tilize_with_zero_padding(
        #     q,
        #     # output_mem_config=q_mem_config,
        #     # output_dtype=ttnn.bfloat16,
        # )

        # q = ttnn.interleaved_to_sharded(
        #     q,
        #     q_mem_config,
        # )

        #### OPTION 2: pad, then tranpsose, then shard?

        q = ttnn.pad(q, [1, N_HEADS_PADDED, BATCH, HEAD_DIM], [0, 0, 0, 0], 0.0)

        q = ttnn.transpose(q, -2, -3)

        q = ttnn.interleaved_to_sharded(
            q,
            q_mem_config,
        )

        q = ttnn.interleaved_to_sharded(
            q,
            q_mem_config,
        )

        out = ttnn.matmul(
            q,
            k,
            program_config=prog_config,
            memory_config=output_config,
            dtype=ttnn.bfloat16,
        )

        return out


def run_test(
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
    inp_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 3),
                    ),
                }
            ),
            [N_HEADS_PADDED, HEAD_DIM],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    k_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 3),
                    ),
                }
            ),
            [HEAD_DIM, SEQ_LEN],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    output_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 3),
                    ),
                }
            ),
            [N_HEADS_PADDED, SEQ_LEN],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    q_in = torch.randn(1, N_HEADS, BATCH, HEAD_DIM)
    k_in = torch.randn(1, BATCH, HEAD_DIM, SEQ_LEN)

    q_tt = torch2tt_tensor(q_in, device)  # , tt_memory_config=inp_mem_config)
    k_tt = torch2tt_tensor(k_in, device, tt_memory_config=k_mem_config)

    prog_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        in0_block_w=HEAD_DIM // TILE_SIZE,
        out_subblock_h=1,  # TODO: Maximize
        out_subblock_w=1,  # TODO: Maximize
        per_core_M=N_HEADS_PADDED // TILE_SIZE,
        per_core_N=SEQ_LEN // TILE_SIZE,
    )

    DRAM_MEMCFG = ttnn.BufferType.DRAM

    # TT hardware execution -------------------------------------------------------------
    tt_model = TT_bmm(model_config)

    tt_out = tt_model(q_tt, k_tt, inp_mem_config, prog_config, DRAM_MEMCFG)


@pytest.mark.parametrize("n_devices", (8,))
@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (("llama-2-70B", 32, 1, 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test(
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

    run_test(
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
