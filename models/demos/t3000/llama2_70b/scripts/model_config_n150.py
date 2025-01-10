# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from pathlib import Path

OP_KEYS = (
    # Inputs
    "INPUT",
    "ATTN_MASK",
    "KV_CACHE",
    # Embeddings
    "WORD_EMBEDDING_WEIGHTS",
    "WORD_EMBEDDING_OUTPUT",
    # Decoder
    "LN_ATTN_WEIGHTS",
    "LN_ATTN_BIAS",
    "LN_ATTN_OUTPUT",
    "LN_MLP_WEIGHTS",
    "LN_MLP_BIAS",
    "LN_MLP_OUTPUT",
    # Rotary
    "SIN_CACHED_WEIGHTS",
    "COS_CACHED_WEIGHTS",
    # Attention
    "FUSED_QKV_MM_WEIGHTS",
    "FUSED_QKV_MM_OUTPUT",
    "CREATE_QKV_HEADS_OUTPUT",
    "ROTARY_EMBEDDING_OUTPUT",
    "K_CACHE_SLICE_OUTPUT",
    "V_CACHE_SLICE_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "PRE_SOFTMAX_MM_OUTPUT",
    "PRE_SOFTMAX_SCALE_OUTPUT",
    "PRE_SOFTMAX_MASK_OUTPUT",
    "SOFTMAX_OUTPUT",
    "POST_SOFTMAX_MM_OUTPUT",
    "CONCAT_HEADS_OUTPUT",
    "SELFOUT_MM_WEIGHTS",
    "SELFOUT_MM_OUTPUT",
    # MLP
    "FF1_MM_WEIGHTS",
    "FF1_MM_OUTPUT",
    "FF3_MM_WEIGHTS",
    "FF3_MM_OUTPUT",
    "FF2_MM_WEIGHTS",
    "FF2_MM_OUTPUT",
    # Decoder Cont
    "PARALLEL_ATTN_ADD_OUTPUT",
    "DROPOUT_ADD_OUTPUT",
    # Model
    "LN_F_WEIGHTS",
    "LN_F_BIAS",
    "LN_F_OUTPUT",
    # LM Head
    "LM_HEAD_MM_WEIGHTS",
    "LM_HEAD_MM_OUTPUT",
)

NO_MEMCFG = ("SOFTMAX_OUTPUT",)

NO_DTYPE = (
    # Attention
    "ROTARY_EMBEDDING_OUTPUT",
    "CREATE_QKV_HEADS_OUTPUT",
    "K_TRANSPOSED_OUTPUT",
    "PRE_SOFTMAX_SCALE_OUTPUT",
    "PRE_SOFTMAX_MASK_OUTPUT",
    "SOFTMAX_OUTPUT",
    "CONCAT_HEADS_OUTPUT",
    # MLP
    "MLP_ALL_GATHER_OUTPUT",
    # Decoder Cont
    "PARALLEL_ATTN_ADD_OUTPUT",
    "DROPOUT_ADD_OUTPUT",
)

ACCEPTABLE_MODEL_CONFIG_STRS = ("BFLOAT16-DRAM", "BFLOAT8_B-DRAM")


def pretty_print_model_config(model_config):
    print_str = []
    for key, val in model_config.items():
        if key.endswith("MEMCFG"):
            print_str.append(f"{key}: {val.buffer_type}")

        elif key.endswith("DTYPE") or key.endswith("BOOL"):
            print_str.append(f"{key}: {val}")

        else:
            raise NotImplementedError("Unknown key: {key}!")

    return "\n".join(print_str)


def get_model_config(model_config_str, num_devices=1, all_gather=True):
    assert model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS
    DRAM_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    L1_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    WIDTH_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)
    HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    BFLOAT16_DTYPE = ttnn.bfloat16
    BFP8_DTYPE = ttnn.bfloat8_b
    # BFP2_DTYPE = ttl.tensor.DataType.BFLOAT2_B

    # Set default dtype and mem_config based on model_config_str
    if model_config_str in ACCEPTABLE_MODEL_CONFIG_STRS:
        dtype_str, mem_config_str = model_config_str.split("-")
        # TODO: Set default memcfg for BFLOAT16-L1 to L1
        mem_config = DRAM_MEMCFG if mem_config_str == "DRAM" else L1_MEMCFG
        dtype = getattr(ttnn.DataType, dtype_str)
    else:
        raise NotImplementedError(f"Model config {model_config_str} is not supported!")

    # Set defaults for dtype and mem_config for all ops
    model_config = {
        "DEFAULT_DTYPE": dtype,
        "DEFAULT_MEMCFG": mem_config,
        "MOVE_DECODER_OUTPUT_BOOL": False,
        "NUM_DEVICES": 4,
        "MAX_GRID_SIZE": (8, 8),
        "DEFAULT_CACHE_PATH": Path(f"models/demos/llama70b/datasets/"),
    }
    model_config.update({f"{key}_MEMCFG": mem_config for key in OP_KEYS if key not in NO_MEMCFG})
    model_config.update({f"{key}_DTYPE": dtype for key in OP_KEYS if key not in NO_DTYPE})

    # Matmul Weights must always be BFP8_B
    # Override defaults for certain configs
    for key in model_config.keys():
        if "MM_WEIGHTS_DTYPE" in key:
            model_config[key] = BFP8_DTYPE
        elif "WEIGHTS_MEMCFG" in key or "BIAS_MEMCFG" in key:
            model_config[key] = DRAM_MEMCFG
        elif "LN" in key and ("WEIGHTS_DTYPE" in key or "BIAS_DTYPE" in key):
            model_config[key] = BFLOAT16_DTYPE

    model_config["KV_CACHE_MEMCFG"] = DRAM_MEMCFG
    model_config["KV_CACHE_DTYPE"] = BFP8_DTYPE

    if model_config_str in ("BFLOAT16-L1",):
        model_config["ROTARY_EMBEDDING_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_SCALE_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["PRE_SOFTMAX_MASK_OUTPUT_MEMCFG"] = L1_MEMCFG
        model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = L1_MEMCFG

    # Embeddings
    model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                64,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if num_devices == 4:
        model_config["ATTN_MASK_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            # Volume must match # of attn heads
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 1),
                        ),
                    }
                ),
                [
                    32,  # Each core has 32 users
                    1,  # Dynamic - must set before using this config
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["ATTN_MASK_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            # Volume must match # of attn heads
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 0),
                        ),
                    }
                ),
                [
                    32,  # Each core has 32 users
                    1,  # Dynamic - must set before using this config
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

    model_config["ATTN_MASK_DTYPE"] = BFP8_DTYPE

    model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                64,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["DROPOUT_ADD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                64,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    # Decoder
    model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,  # per_core_M * TILE_HEIGHT
                256,  # per_core_N * TILE_WIDTH
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_ATTN_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                256,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_ATTN_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=1,
        block_w=8,
        inplace=False,
    )
    model_config["LN_MLP_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=1,
        block_w=8,
        inplace=True,
    )
    # LLama2 Attention Module
    # Fused QKV Matmul Config
    if num_devices == 1:
        model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
                [
                    32,
                    256,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 4:
        model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 1),
                        ),
                    }
                ),
                [
                    32,
                    512,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        # Unpadded version
        model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 0),
                        ),
                    }
                ),
                [
                    32,
                    1024,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        # Padded version
        # model_config["FUSED_QKV_MM_INPUT_MEMCFG"] = ttnn.MemoryConfig(
        #     ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #     ttnn.BufferType.L1,
        #     ttnn.ShardSpec(
        #         ttnn.CoreRangeSet(
        #             {
        #                 ttnn.CoreRange(
        #                     ttnn.CoreCoord(0, 0),
        #                     ttnn.CoreCoord(7, 7),
        #                 ),
        #             }
        #         ),
        #         [
        #             32,
        #             128,
        #         ],
        #         ttnn.ShardOrientation.ROW_MAJOR,
        #         False,
        #     ),
        # )
    model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    if num_devices == 1:
        model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=1,
            per_core_N=10,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 4:
        model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=1,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
        # Unpadded version
        model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=32,
            out_subblock_h=1,
            out_subblock_w=5,
            per_core_M=1,
            per_core_N=5,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        # Padded version
        # model_config["FUSED_QKV_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        #     compute_with_storage_grid_size=(8, 8),
        #     in0_block_w=4,
        #     out_subblock_h=1,
        #     out_subblock_w=1,
        #     per_core_M=1,
        #     per_core_N=1,
        #     fuse_batch=True,
        #     fused_activation=None,
        #     mcast_in0=True,
        # )

    # Split QKV Matmul Config
    model_config["WQ_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=1,
        per_core_N=8,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    model_config["WK_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    model_config["WK_MM_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                32,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    model_config["WQ_MM_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                256,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if num_devices == 4:
        model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(1, 0),
                        ),
                    }
                ),
                [
                    32,
                    1280,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(0, 0),
                        ),
                    }
                ),
                [
                    32,
                    1280,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    # model_config["ROT_MAT_K_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
    #     ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    #     ttnn.BufferType.L1,
    #     ttnn.ShardSpec(
    #         ttnn.CoreRangeSet(
    #             {
    #                 ttnn.CoreRange(
    #                     ttnn.CoreCoord(0, 0),
    #                     ttnn.CoreCoord(0, 0),
    #                 ),
    #             }
    #         ),
    #         [
    #             128,
    #             128,  # Dynamic
    #         ],
    #         ttnn.ShardOrientation.ROW_MAJOR,
    #         False,
    #     ),
    # )

    if num_devices == 8:
        model_config["ROT_MAT_K_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(1, 1),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        model_config["ROT_MAT_K_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG

        model_config["ROT_MAT_Q_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        model_config["ROT_MAT_Q_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG

    model_config["ROT_MAT_MEMCFG"] = DRAM_MEMCFG  # L1_MEMCFG
    model_config["L1_K_HEADS_INTERLEAVED_MEMCFG"] = L1_MEMCFG

    model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
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
            [
                1,
                128,  # Dynamic
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if num_devices == 4:
        model_config["Q_ROTARY_EMB_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 1),
                        ),
                    }
                ),
                [
                    32,
                    128,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["Q_ROTARY_EMB_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 0),
                        ),
                    }
                ),
                [
                    32,
                    128,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    model_config["K_TRANSPOSED_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    if num_devices == 4:
        model_config["SOFTMAX_PROGCFG"] = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 2),
            subblock_w=1,
            block_h=1,
            block_w=1,  # Dynamic
        )
    elif num_devices == 8:
        model_config["SOFTMAX_PROGCFG"] = ttnn.SoftmaxShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            subblock_w=1,
            block_h=1,
            block_w=1,  # Dynamic
        )
    model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"] = HEIGHT_SHARDED_MEMCFG
    model_config["CONCAT_HEADS_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    if num_devices == 4:
        model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 7),
                        ),
                    }
                ),
                [
                    32,
                    128,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
                [
                    32,
                    256,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    model_config["SELFOUT_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    if num_devices == 4:
        # (32 x 8k) x (8k x 2k) = (32 x 2k)
        model_config["SELFOUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 8:
        # (32 x 8k) x (8k x 1k) = (32 x 1k)
        model_config["SELFOUT_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    # Llama2 LN_MLP config
    if num_devices == 1:
        model_config["LN_MLP_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 7),  # Full core grid
                        ),
                    }
                ),
                [
                    32,
                    128,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 4:
        model_config["LN_MLP_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
                [
                    32,
                    256,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["LN_MLP_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 1),
                        ),
                    }
                ),
                [
                    32,
                    512,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    # Llama MLP config
    model_config["FF1_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["FF3_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["FF2_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["FF13_MUL_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["ALL_REDUCE_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    if num_devices == 4:
        model_config["ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
                [
                    32,
                    896,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    elif num_devices == 8:
        model_config["ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
                [
                    32,
                    896,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
    # FF1 & FF3 Matmul Config Variables
    compute_with_storage_grid_size = {
        1: (8, 8),  # - For 1 device, use full 8x8 grid
        4: (8, 4),  # - For 4 devices weight_dim sharded, GCF for 8192 and 7168 is 32, so use 8x4 grid.
        8: (8, 2),  # - For 8 devices, GCF for 8192 and 3584 is 16, so use 8x2 grid.
    }[num_devices]

    in0_block_w = {1: 4, 4: 8, 8: 16}[
        num_devices
    ]  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size

    per_core_N = 14 if num_devices == 1 else 7  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size

    model_config["FF1_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_with_storage_grid_size,
        in0_block_w=in0_block_w,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=7,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=per_core_N,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fuse_batch=True,
        fused_activation=ttnn.UnaryOpType.SILU,
        mcast_in0=True,
    )

    model_config["FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=compute_with_storage_grid_size,
        in0_block_w=in0_block_w,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=7,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
        per_core_M=1,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    # FF2 Matmul Config Variables, using dim=-2 shard, All-gather/All-reduce(1D-Weight Stationary) schemes,
    if num_devices == 1:
        model_config["FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=14,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
    elif num_devices == 4:
        if all_gather:
            model_config["FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=28,  # K = 28672 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=1,
                per_core_N=2,  # 2048 / 32 / 32
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        else:  # All Reduce Case
            model_config["FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=7,  # K = 7168 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=1,
                per_core_N=8,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
    elif num_devices == 8:
        if all_gather:
            model_config["FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=28,  # K = 28672 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        else:
            model_config["FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 2),
                in0_block_w=7,  # K = 3584 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
                out_subblock_h=1,
                out_subblock_w=8,
                per_core_M=1,
                per_core_N=16,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
    # ***IMPORTANT***
    # Padded MLP 32K config:
    model_config["PADDED_LN_MLP_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    ),
                }
            ),
            [
                32,
                128,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    if num_devices == 4:
        model_config["PADDED_FF1_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 8192 for num_device=8
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
        )
        model_config["PADDED_FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=4,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["PADDED_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 7),
                        ),
                    }
                ),
                [
                    32,
                    512,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["PADDED_FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=16,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    elif num_devices == 8:
        model_config["PADDED_FF1_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=2,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size, N = 4096 for num_device=8
            fuse_batch=True,
            fused_activation=ttnn.UnaryOpType.SILU,
            mcast_in0=True,
        )
        model_config["PADDED_FF3_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=4,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 8
            per_core_M=1,  # M / TILE_HEIGHT = 32 / 32
            per_core_N=2,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        model_config["PADDED_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
                [
                    32,
                    1024,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )
        model_config["PADDED_FF2_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=32,  # K = 32768 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                256,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_F_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
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
            [
                32,
                256,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )
    model_config["LN_F_PROGCFG"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[8, 4],
        subblock_w=8,
        block_h=1,
        block_w=8,
        inplace=True,
    )

    # LM Head
    model_config["LM_HEAD_MM_OUTPUT_MEMCFG"] = WIDTH_SHARDED_MEMCFG
    model_config["LM_HEAD_MM_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 4),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=1,
        per_core_N=16,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    # uncomment if need to see all the configs
    # logger.debug(f"Falcon model config: \n{pretty_print_model_config(model_config)}")

    return model_config


model_config_entries = {
    "_name_or_path": "tiiuae/falcon-40b-instruct",
    "alibi": False,
    "apply_residual_connection_post_layernorm": False,
    "architectures": ["FalconForCausalLM"],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_falcon.FalconConfig",
        "AutoModel": "modeling_falcon.FalconModel",
        "AutoModelForCausalLM": "modeling_falcon.FalconForCausalLM",
        "AutoModelForQuestionAnswering": "modeling_falcon.FalconForQuestionAnswering",
        "AutoModelForSequenceClassification": "modeling_falcon.FalconForSequenceClassification",
        "AutoModelForTokenClassification": "modeling_falcon.FalconForTokenClassification",
    },
    "bias": False,
    "bos_token_id": 11,
    "eos_token_id": 11,
    "hidden_dropout": 0.0,
    "hidden_size": 8192,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "llama2",
    "multi_query": True,
    "new_decoder_architecture": True,
    "num_attention_heads": 64,
    "num_hidden_layers": 80,
    "num_kv_heads": 8,
    "parallel_attn": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.28.1",
    "use_cache": True,
    "vocab_size": 65024,
}


# def main():
#     model_config = get_model_config("BFLOAT16-DRAM")
#     print(model_config["FF1_MM_WEIGHTS_DTYPE"])
#     print(model_config["FF1_MM_OUTPUT_MEMCFG"])
#     # print(pretty_print_model_config(model_config))


# main()
