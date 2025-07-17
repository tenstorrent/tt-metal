# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
import json
import ttnn
from pathlib import Path
from loguru import logger
import torch
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer
from models.tt_transformers.tt.common import (
    precompute_freqs,
    freqs_to_rotation_matrix,
    num_to_core_range_set,
    calculate_hidden_dim,
    get_base_model_name,
    get_out_subblock_w,
    encode_prompt_instruct,
    encode_prompt_hf,
    nearest_multiple,
)
from typing import Tuple
from models.utility_functions import nearest_32
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from models.demos.llama3_subdevices.tt.load_checkpoints import (
    load_meta_state_dict,
    load_hf_state_dict,
    convert_hf_to_meta,
    standardize_hf_keys,
)


PREFETCHER_NOC1_GRID = [
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
]

LM_HEAD_16_GRID = [
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
]

LM_HEAD_32_GRID = [
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 7),
    (2, 8),
    (2, 9),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 9),
    (5, 0),
    (5, 1),
]

LM_HEAD_INPUT_GRID = [
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 0),
    (6, 1),
    (6, 2),
    (6, 4),
    (6, 5),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 0),
    (5, 1),
    (5, 2),
    (5, 4),
    (1, 4),
    (1, 5),
    (1, 9),
    (1, 0),
    (2, 0),
    (2, 4),
    (2, 5),
    (2, 9),
]

LM_HEAD_OUTPUT_GRID = [
    (1, 9),
    (2, 9),
    (1, 0),
    (2, 0),
    (1, 4),
    (2, 4),
    (1, 5),
    (2, 5),
    (5, 0),
    (6, 0),
    (5, 9),
    (6, 9),
    (5, 1),
    (6, 1),
    (5, 7),
    (6, 7),
    (5, 6),
    (6, 6),
    (5, 2),
    (6, 2),
    (5, 4),
    (6, 4),
    (5, 5),
    (6, 5),
]


def get_core_ranges(num_reader_cores, num_global_cb_receivers, is_functional_test):
    """
    Helper function to get all the relevant core ranges for dram prefetcher + matmul configuration
    """

    all_dram_cores = [ttnn.CoreCoord(idx, 0) for idx in range(12)]  # 12 DRAM banks

    all_sender_cores = [
        ttnn.CoreCoord(0, 9),
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(0, 4),
        ttnn.CoreCoord(0, 5),
        ttnn.CoreCoord(4, 0),
        ttnn.CoreCoord(4, 9),
        ttnn.CoreCoord(4, 1),
        ttnn.CoreCoord(4, 7),
        ttnn.CoreCoord(4, 6),
        ttnn.CoreCoord(4, 2),
        ttnn.CoreCoord(4, 4),
        ttnn.CoreCoord(4, 5),
    ]
    dummy_sender_cores = [
        ttnn.CoreCoord(0, 1),
        ttnn.CoreCoord(0, 2),
        ttnn.CoreCoord(0, 3),
        ttnn.CoreCoord(0, 6),
        ttnn.CoreCoord(0, 7),
        ttnn.CoreCoord(0, 8),
        ttnn.CoreCoord(4, 3),
        ttnn.CoreCoord(4, 8),
    ]

    all_receiver_cores_list = [
        (1, 9),
        (2, 9),
        (1, 0),
        (2, 0),
        (1, 4),
        (2, 4),
        (1, 5),
        (2, 5),
        (5, 0),
        (6, 0),
        (5, 9),
        (6, 9),
        (5, 1),
        (6, 1),
        (5, 7),
        (6, 7),
        (5, 6),
        (6, 6),
        (5, 2),
        (6, 2),
        (5, 4),
        (6, 4),
        (5, 5),
        (6, 5),
    ]

    all_receiver_cores = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(*all_receiver_cores_list[idx]),
                    ttnn.CoreCoord(*all_receiver_cores_list[idx + 1 if num_global_cb_receivers == 2 else idx]),
                ),
            ]
        )
        for idx in range(0, len(all_receiver_cores_list), 2)
    ]

    dummy_receiver_cores = [
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 0),
                    ttnn.CoreCoord(3, 0),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 1),
                    ttnn.CoreCoord(3, 1),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 2),
                    ttnn.CoreCoord(3, 2),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 3),
                    ttnn.CoreCoord(3, 3),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 4),
                    ttnn.CoreCoord(3, 4),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 5),
                    ttnn.CoreCoord(3, 5),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 6),
                    ttnn.CoreCoord(3, 6),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 7),
                    ttnn.CoreCoord(3, 7),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(1, 8),
                    ttnn.CoreCoord(3, 8),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(3, 9),
                    ttnn.CoreCoord(3, 9),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 3),
                    ttnn.CoreCoord(6, 3),
                ),
            ]
        ),
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(5, 8),
                    ttnn.CoreCoord(6, 8),
                ),
            ]
        ),
    ]

    dram_cores = all_dram_cores[:num_reader_cores]
    hop_grid = []
    mm_optimised_ring_cores = []
    if not is_functional_test:
        sender_cores = all_sender_cores
        active_sender_cores = all_sender_cores[:num_reader_cores]
        sender_cores.extend(dummy_sender_cores)
        active_receiver_cores_list = all_receiver_cores_list[: num_reader_cores * num_global_cb_receivers]
        receiver_cores = all_receiver_cores
        receiver_cores.extend(dummy_receiver_cores)

        worker_cores_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )

        mm_optimised_ring_cores = PREFETCHER_NOC1_GRID
        hop_grid = [
            (3, 6),
        ]
    else:
        sender_cores = [ttnn.CoreCoord(0, y) for y in range(num_reader_cores)]
        active_sender_cores = sender_cores

        active_receiver_cores_list = [
            (x, y) for y in range(num_reader_cores) for x in range(1, num_global_cb_receivers + 1)
        ]

        receiver_cores = [
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(*active_receiver_cores_list[idx * num_global_cb_receivers]),
                        ttnn.CoreCoord(*active_receiver_cores_list[(idx + 1) * num_global_cb_receivers - 1]),
                    )
                ]
            )
            for idx in range(num_reader_cores)
        ]

        worker_cores_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1)),
            ]
        )

    return (
        active_sender_cores,
        dram_cores,
        sender_cores,
        active_receiver_cores_list,
        receiver_cores,
        worker_cores_range_set,
        mm_optimised_ring_cores,
        hop_grid,
    )


@dataclass
class LlamaOptimizations:
    bfp4_mlp: bool
    # Future fields will go here:
    # bfp8_activations: bool
    # bfp8_layernorm: bool
    # bfp8_ccl: bool

    @classmethod
    def accuracy(cls, model_name):
        """Configuration optimized for accuracy
        Only 3.1-70B uses bfp4 MLPs in this configuration
        """
        return cls(bfp4_mlp="70B" in model_name)

    @classmethod
    def performance(cls, model_name):
        """Configuration optimized for performance
        All models use bfp4 MLPs in this configuration
        """
        return cls(bfp4_mlp=True)


class CheckpointType(Enum):
    Meta = auto()
    HuggingFace = auto()


class TtModelArgs:
    OP_KEYS = (
        # Embedding
        "EMB_WEIGHTS",
        # Feed forward
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        # Attention
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        # Decoder
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )

    LOCAL_LLAMA_PARAMS = {
        "LLAMA3_1_70B_PARAMS": "models/demos/llama3_subdevices/model_params/Llama-3.1-70B-Instruct",
        "LLAMA3_3_70B_PARAMS": "models/demos/llama3_subdevices/model_params/Llama-3.3-70B-Instruct",
    }

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=LlamaOptimizations.accuracy,
    ):
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.device_name = {0: "CPU", 1: "N150", 2: "N300", 8: "T3K", 32: "TG"}[self.num_devices]
        self.model_name = "Unknown"  # Llama model name will be dependent on the checkpoint directory
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.tile_size = 32
        self.is_70b = False
        self.from_hf_url = False  # updated below if true
        self.max_prefill_chunk_size = max_seq_len
        self.use_prefetcher = False
        self.max_top_k = 32

        if self.num_devices == 32:
            self.use_prefetcher = True

        # Set up prefetcher stuff
        _, _, _, self.pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)

        self.sub_core_grids = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
            ]
        )
        self.sub_core_grid_topk = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ]
        )
        self.start_core = ttnn.CoreCoord(1, 0)

        LLAMA_DIR = os.getenv("LLAMA_DIR")
        HF_MODEL = os.getenv("HF_MODEL")
        assert not (LLAMA_DIR and HF_MODEL), "Only one of LLAMA_DIR or HF_MODEL should be set"
        if LLAMA_DIR:
            if any([os.getenv("LLAMA_CKPT_DIR"), os.getenv("LLAMA_TOKENIZER_PATH")]):
                logger.warning("LLAMA_DIR will override LLAMA_CKPT_DIR and LLAMA_TOKENIZER_PATH")
            self.CKPT_DIR = LLAMA_DIR
            self.TOKENIZER_PATH = LLAMA_DIR
            self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join(LLAMA_DIR, self.device_name)
            self.model_name = os.path.basename(LLAMA_DIR)  # May be overridden by config
        elif HF_MODEL:
            self.CKPT_DIR = HF_MODEL
            self.TOKENIZER_PATH = HF_MODEL
            self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join("model_cache", HF_MODEL, self.device_name)
            else:  # For HF models, always append the device name (e.g. N150/N300/T3K/TG) to the cache path
                self.CACHE_PATH = os.path.join(self.CACHE_PATH, self.device_name)
            self.model_name = HF_MODEL  # May be overridden by config
            self.from_hf_url = True
        else:
            assert (
                False
            ), "Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct or LLAMA_DIR to a Meta-style checkpoint directory"

        if not dummy_weights and not HF_MODEL:
            # Assert if all folders and files exist
            assert os.path.exists(
                self.CKPT_DIR
            ), f"Checkpoint directory {self.CKPT_DIR} does not exist, please set LLAMA_DIR=... or LLAMA_CKPT_DIR=..."
            os.makedirs(self.CACHE_PATH, exist_ok=True)

        logger.info(f"Checkpoint directory: {self.CKPT_DIR}")
        logger.info(f"Tokenizer file: {self.TOKENIZER_PATH + '/tokenizer.model'}")
        logger.info(f"Cache directory: {self.CACHE_PATH}")

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = self.CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.TOKENIZER_PATH + "/tokenizer.model"

        self.instruct = instruct
        # If the weights file contain the keyword `instruct` also set self.instruct to true
        if "instruct" in self.CKPT_DIR.lower():
            self.instruct = True

        # Load model params
        if HF_MODEL:
            self.checkpoint_type = CheckpointType.HuggingFace
            self._set_hf_params(self.CKPT_DIR)
        elif not dummy_weights:
            self.checkpoint_type = self.detect_checkpoint_type()
            self._set_model_params(self.CKPT_DIR)
        else:  # With Dummy weights, set the params from the local copy inside the model folder. This is required for CI pipeline that doesn't mount the external folders.
            self.checkpoint_type = CheckpointType.Meta
            if "3.2-1B" in self.CKPT_DIR:
                local_params = "LLAMA3_2_1B_PARAMS"
            elif "3.2-3B" in self.CKPT_DIR:
                local_params = "LLAMA3_2_3B_PARAMS"
            elif "3.1-8B" in self.CKPT_DIR:
                local_params = "LLAMA3_1_8B_PARAMS"
            elif "3.2-11B" in self.CKPT_DIR:
                local_params = "LLAMA3_2_11B_PARAMS"
            elif "3.1-70B" in self.CKPT_DIR:
                local_params = "LLAMA3_1_70B_PARAMS"
            elif "3.3-70B" in self.CKPT_DIR:
                local_params = "LLAMA3_3_70B_PARAMS"
            else:
                raise ValueError(
                    f"No local params found for {self.CKPT_DIR}, dummy weights are not supported for this model"
                )
            self._set_model_params(self.LOCAL_LLAMA_PARAMS[local_params])

        if callable(optimizations):
            self.optimizations = optimizations(self.model_name)
        else:
            self.optimizations = optimizations

        self.dummy_weights = dummy_weights
        self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

        # Enable workarounds by default until di/dt issues are fixed
        self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"
        if not self.di_dt_workaround:
            logger.info("Disabling di/dt workaround, re-enable if you see hangs")

        self.TG = self.num_devices == 32
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
        L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
        self.model_config = {}
        # Update memory configs (weights->DRAM, activations->L1)
        self.model_config.update(
            {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
        )
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        self.cos, self.sin = precompute_freqs(
            self.head_dim, self.max_seq_len * 2, self.rope_theta, self.rope_scaling_factor, self.orig_context_len
        )  # for prefill
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode

        self.tokenizer = None if dummy_weights else self.create_tokenizer()

        device = mesh_device if mesh_device is not None else None
        self.cluster_shape = list(mesh_device.shape)
        self.is_galaxy = self.num_devices == 32
        self.galaxy_type = None

        if self.is_galaxy:
            self.galaxy_type = "6U" if ttnn.GetNumPCIeDevices() == 32 else "4U"
        else:
            raise ValueError(
                f"Unsupported number of devices: {self.num_devices}. Only 32 devices (Galaxy) are supported."
            )
        self.model_config["GALAXY_NUM_LINKS"] = {"6U": 4, "4U": 3}.get(self.galaxy_type)
        self.model_config["CCL_TOPOLOGY"] = {"6U": ttnn.Topology.Ring, "4U": ttnn.Topology.Linear}.get(self.galaxy_type)
        if device is not None:  # Avoid issue with test_llama_torch.py not having a device
            self.n_local_heads = self.n_heads // self.cluster_shape[1]

            grid = device.compute_with_storage_grid_size()
            self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

            # DRAM weight grid specs for dram sharding matmuls
            self.dram_weight_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                    )
                }
            )

            # Compute kernels. FP32 acc does not appear to be needed for accuracy in model tests or demo runs.
            self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            )
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
                dst_full_sync_en=True,
            )
            self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            self.model_config["COMPUTE_KERNEL_CONFIG_HIFI2"] = self.compute_kernel_config_hifi2
            core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(1, 0)
            core_range = ttnn.CoreRange(
                grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
            )
            num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
            residual_grid = self.dram_shard_core_grid_for_k(self.dim // self.num_devices)
            self.model_config["DECODE_RESIDUAL_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(1, 1, 32, 2048 // num_cores_ln),
                    core_grid=ttnn.CoreRangeSet(
                        {
                            core_range,
                        }
                    ),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.is_galaxy
                else ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        self.dim // residual_grid.num_cores // self.num_devices,
                    ),
                    residual_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            start_core = ttnn.CoreCoord(1, 0)
            core_grid = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            )
            num_cores = self.cluster_shape[0]
            shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                start_core, num_cores, core_grid, row_wise=False
            )

            self.model_config["DECODE_SAMPLING_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(1, 1, max(self.max_batch_size, self.tile_size), self.max_top_k),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            num_cores = 32
            shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                start_core, num_cores, core_grid, row_wise=False
            )
            self.model_config["DECODE_LOGITS_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(1, 1, max(self.max_batch_size, self.tile_size), self.padded_vocab_size // num_cores),
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(7, 10),
                exp_approx_mode=False,
                q_chunk_size=256 if seqlen >= 2048 else 64,
                k_chunk_size=256 if seqlen >= 2048 else 64,
            )

            def find_largest_divisor(n, max_divisor=8):
                for i in range(max_divisor, 0, -1):
                    if n % i == 0:
                        return i
                return 1  # Fallback to 1 if no divisor found

            # nlp_concat_heads_decode will shard the data across this number of cores
            assert (
                self.n_heads % self.cluster_shape[1] == 0
            ), f"n_heads must be divisible by num_devices: {self.n_heads} % {self.cluster_shape[1]}"

            self.model_config["ATTN_OUTPUT_PROGCFG"] = (
                None
                if self.is_galaxy
                else self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=self.dim // self.num_devices,
                    n=self.dim,
                    num_cores=self.n_heads // self.num_devices,
                )
            )

            # All Gather Matmul for Dense Out (DO)
            # TODO: Is there a better way to decide if fused all gather matmul should be used? And is there a better way to use the flag, instead of passing it into model_config?
            # NOTE: Fused all gather matmul only suppports a core grid of size num_devices x 1
            self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
                self.ccl_topology() == ttnn.Topology.Ring
                and (self.dim // self.tile_size // self.num_devices) % self.num_devices == 0
                and self.num_devices > 1
            )

            if self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]:
                do_core_grid_size = (8, 1)
                do_per_core_N = (
                    self.dim // self.num_devices // self.tile_size // (do_core_grid_size[0] * do_core_grid_size[1])
                )
                self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=do_core_grid_size,
                    in0_block_w=self.dim
                    // self.tile_size
                    // (do_core_grid_size[0] * do_core_grid_size[1]),  # [32 x 8k] x [8k x 1k] = [32 x 1k]
                    out_subblock_h=1,
                    out_subblock_w=get_out_subblock_w(
                        do_per_core_N, out_subblock_h=1
                    ),  # Max out_subblock_w = 4, needs to be divisible by per_core_N
                    per_core_M=self.tile_padded_batch_rows // self.tile_size,
                    per_core_N=do_per_core_N,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"] = None

            def w1_w3_prg_config(seq_len, use_interleaved):
                if seq_len == 128:
                    return self.matmul_1d_config(128, 2048, 3584, grid=ttnn.CoreGrid(x=7, y=4), overwrite_per_core_k=16)
                if not use_interleaved:
                    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 10),
                        in0_block_w=8,
                        out_subblock_h=1,  # Must be divisible by per_core_M
                        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                        per_core_M=max(
                            1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8  # 8 rows
                        ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                        per_core_N=math.ceil(28672 / 8 / 32 / 7),  # N / TILE_WIDTH / grid width
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=seq_len <= 2048,
                    )

                if seq_len % 4096 == 0:
                    per_core_M = 20 * seq_len // 4096
                    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 7),
                        in0_block_w=4,
                        out_subblock_h=1,
                        out_subblock_w=8,
                        out_block_h=10,
                        out_block_w=16,
                        per_core_M=per_core_M,
                        per_core_N=16,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=False,
                    )
                elif seq_len % 2048 == 0:
                    per_core_M = 10 * seq_len // 2048

                    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 7),
                        in0_block_w=4,
                        out_subblock_h=1,
                        out_subblock_w=8,
                        out_block_h=10,
                        out_block_w=16,
                        per_core_M=per_core_M,
                        per_core_N=16,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=False,
                    )
                else:
                    raise NotImplementedError(
                        f"W1 Program config generation for sequence length {seq_len} not implemented"
                    )

            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = w1_w3_prg_config

            def w2_prg_config(seq_len):
                if seq_len == 128:
                    return self.matmul_1d_config(
                        128, 3584, 2048, grid=ttnn.CoreGrid(x=7, y=10), overwrite_per_core_k=14
                    )
                # For sequence lengths < 4096, we use this config as it performs better that what would be generated below
                if seq_len < 4096:
                    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 10),
                        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                        out_subblock_h=1,  # Must be divisible by per_core_M
                        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                        per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),  # 8~10 rows
                        per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=seq_len <= 2048,
                    )

                # For very large activation heights (arbitrarily chosen to be > 320) we want the per_core_M to have many divisors
                # so that there are many options for out_block_h and out_block_w. Padding to the next multiple of 8 ensures that
                # per_core_M can at least be divisible by 2, 4, and 8 in addition to 1 and itself.
                #
                # If the number is less than or equal to 320 we still wouldn't want it to be prime so we'll add one if thats the case.
                next_multiple_of_8 = lambda x: int(x + (8 - x % 8) % 8)
                add_one_if_prime = (
                    lambda n: n + 1 if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)) else n
                )
                total_per_core_out_M = add_one_if_prime(math.ceil(seq_len / (7 * self.tile_size)))
                per_core_M = (
                    next_multiple_of_8(total_per_core_out_M) if total_per_core_out_M > 320 else total_per_core_out_M
                )
                per_core_N = 10

                # Want out_block_h and out_block_w such that:
                # out_block_h * out block_w <= 320
                # out_block_h % per_core_M == 0
                # out_block_w % per_core_N == 0
                # Since we're fixing per_core_N = 10, out_block_w can only be 5 or 10

                def find_out_block_h(out_block_w):
                    max_out_block_h = -1
                    for i in range(1, per_core_M + 1):
                        if i * out_block_w > 320:
                            break
                        if per_core_M % i == 0:
                            if i > max_out_block_h:
                                max_out_block_h = i
                    if max_out_block_h == -1:
                        return None
                    return max_out_block_h

                out_block_h_if_w_5 = find_out_block_h(5)
                out_block_h_if_w_10 = find_out_block_h(10)

                if out_block_h_if_w_5 is None and out_block_h_if_w_10 is None:
                    assert False, "This should never happen"

                # Pick the configuration that exists if one of them does not
                if out_block_h_if_w_5 is None:
                    out_block_w = 10
                    out_block_h = out_block_h_if_w_10
                elif out_block_h_if_w_10 is None:
                    out_block_w = 5
                    out_block_h = out_block_h_if_w_5
                # If both exist, pick the one that is larger in volume
                elif out_block_h_if_w_5 * 5 > out_block_h_if_w_10 * 10:
                    out_block_h = out_block_h_if_w_5
                    out_block_w = 5
                elif out_block_h_if_w_10 * 10 > out_block_h_if_w_5 * 5:
                    out_block_h = out_block_h_if_w_10
                    out_block_w = 10
                # If both have the same volume, pick the configuration that is more "square"
                else:
                    # Want to use the out_block_h/w combination which is the most "square"
                    # This calculates the height/width ratio of the blocks and then gets their
                    # distance from 1 (1 is the ideal ratio) to determine which is more square
                    squareness_5 = abs(1 - (max(out_block_h_if_w_5, 5) / min(out_block_h_if_w_5, 5)))
                    squareness_10 = abs(1 - (max(out_block_h_if_w_10, 10) / min(out_block_h_if_w_10, 10)))

                    if squareness_5 < squareness_10:
                        out_block_w = 5
                        out_block_h = out_block_h_if_w_5
                    else:
                        out_block_w = 10
                        out_block_h = out_block_h_if_w_10

                return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(7, 7),
                    in0_block_w=2,  # seeing this to 2 because 4 gives oom for long seqlen continuous batching
                    out_subblock_h=1,
                    out_subblock_w=5,
                    out_block_h=out_block_h,
                    out_block_w=out_block_w,
                    per_core_M=per_core_M,
                    per_core_N=per_core_N,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )

            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = w2_prg_config

            self.model_config["WO_PREFILL_PROGCFG"] = (
                lambda seq_len: self.matmul_1d_config(
                    seq_len, 1024, 2048, grid=ttnn.CoreGrid(x=7, y=10), overwrite_per_core_k=16
                )
                if seq_len == 128
                else (
                    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 10),
                        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                        out_subblock_h=1,  # Must be divisible by per_core_M
                        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                        per_core_M=max(1, 4 if seq_len >= 1024 else seq_len // self.tile_size // 8),  # 8~10 rows
                        per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=seq_len <= 1024,
                    )
                )
            )

            # Calculate largest number of lm_head_num_rows such that self.dim % (lm_head_num_rows * 8) == 0
            if self.num_devices == 32:
                lm_head_num_rows = 4
                while self.dim % (32 * 32 * lm_head_num_rows) != 0:
                    lm_head_num_rows -= 1
            else:
                lm_head_num_rows = 8
                while self.dim % (32 * lm_head_num_rows * 8) != 0:
                    lm_head_num_rows -= 1
            assert (
                lm_head_num_rows > 0
            ), f"Could not find a lm_head_num_rows such that self.dim(={self.dim}) % (lm_head_num_rows * 4) == 0"
            self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=8)

            self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    nearest_32((self.dim // (4 if self.is_galaxy else 1)) // self.lm_head_core_grid.num_cores),
                ),  # Shard shape: [32, 128] -> 1 shard per core
                self.lm_head_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
            self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / (self.n_kv_heads // self.cluster_shape[1])
            self.model_config["XQKV_PREFILL_PROGCFG"] = (
                lambda seq_len: self.matmul_1d_config(
                    seq_len, 2048, 1280, grid=ttnn.CoreGrid(x=4, y=10), overwrite_per_core_k=16
                )
                if seq_len == 128
                else (
                    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 10),
                        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                        out_subblock_h=1,  # Must be divisible by per_core_M
                        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                        per_core_M=max(
                            1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8  # 8 rows
                        ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                        per_core_N=math.ceil(1280 / 32 / 7),  # N / TILE_WIDTH / grid width
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=seq_len <= 2048,
                    )
                )
            )

            assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"
            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                (((self.n_kv_heads // self.cluster_shape[1]) * seq_len // (8 * 8)), self.head_dim),
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["PAGED_SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    self.start_core, 32, self.sub_core_grids, row_wise=True
                ),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )

            # TODO: Need to uplift UpdateCache to support dynamic chunk sizes if non-paged
            self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    self.start_core, 32, self.sub_core_grids, row_wise=True
                ),
                exp_approx_mode=False,
                q_chunk_size=256,
                k_chunk_size=256,
            )

            self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            # Useful core grid based on batch size
            if self.max_batch_size == 32:
                grid_by_batch = (8, 4)
            elif self.max_batch_size == 16:
                grid_by_batch = (8, 2)
            elif self.max_batch_size == 8:
                grid_by_batch = (8, 1)
            elif self.max_batch_size == 4:
                grid_by_batch = (4, 1)
            elif self.max_batch_size == 2:
                grid_by_batch = (2, 1)
            elif self.max_batch_size == 1:
                grid_by_batch = (1, 1)
            else:
                raise ValueError(f"Batch size {self.max_batch_size} not supported")
            core_grid_by_batch = ttnn.CoreGrid(y=grid_by_batch[1], x=grid_by_batch[0])
            core_range_set_by_batch = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(grid_by_batch[0] - 1, grid_by_batch[1] - 1),
                    ),
                }
            )

            self.model_config[
                "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
            ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
                shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
                core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    self.start_core, batch_size_per_device_group, self.sub_core_grids, row_wise=True
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["ROT_MAT_MEMCONFIG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    core_range_set_by_batch,
                    [
                        128,
                        128,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            # MLP configs
            mlp_core_grid = (
                self.dram_shard_core_grid_for_k(self.dim)
                if self.is_galaxy
                else self.dram_shard_core_grid_for_k_and_n(self.dim, self.hidden_dim // self.num_devices)
            )

            self.model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.dim // mlp_core_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                mlp_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.dim,
                n=self.hidden_dim // self.cluster_shape[1],
                num_cores=mlp_core_grid.num_cores,
            )

            mlp2_core_grid = (
                ttnn.CoreGrid(y=1, x=8)
                if self.is_galaxy
                else self.dram_shard_core_grid_for_k_and_n(self.hidden_dim // self.num_devices, self.dim)
            )

            self.model_config["SHARDED_MLP2_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    32 if self.is_galaxy else self.tile_padded_batch_rows,
                    self.hidden_dim // self.cluster_shape[1] // mlp2_core_grid.num_cores,
                ),
                mlp2_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = self.dram_matmul_config(
                m=self.tile_padded_batch_rows,
                k=self.hidden_dim // self.cluster_shape[1],
                n=self.dim,
                num_cores=mlp2_core_grid.num_cores,
            )

            ##### Prefetcher stuff #####
            self.model_config["USE_PREFETCHER"] = self.use_prefetcher
            RING_SIZE = 24
            ring_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in PREFETCHER_NOC1_GRID
                ]
            )
            pf_mm_out_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in self.pf_receiver_cores_list
                ]
            )

            # QKV
            self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, 2304 // RING_SIZE),  # Use padded K
                    core_grid=ring_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.is_galaxy
                else ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        self.dim // attn_input_grid.num_cores,
                    ),  # Shard shape: [32, 128] -> 1 shard per core
                    attn_input_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )
            qkv_shape_ring = (8192 // 4, 12288 // 8)  # Use padded K and N
            self.model_config["SHARDED_QKV_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                k=qkv_shape_ring[0],
                n=qkv_shape_ring[1],
            )

            qkv_out_shard_shape_ring = (32, 12288 // 8 // RING_SIZE)  # Use padded N
            self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=qkv_out_shard_shape_ring,
                core_grid=pf_mm_out_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["XQKV_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
                1,
                32,
                8192 // 4,
                12288 // 8,  # Use padded N
                RING_SIZE,
                untilize_out=True,
            )
            RS_CREATE_HEADS_PACKET_WORKER_CRS = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
                ]
            )
            self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 512),
                core_grid=RS_CREATE_HEADS_PACKET_WORKER_CRS,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # WO
            self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 12288 // 8 // RING_SIZE),  # Use padded K
                core_grid=ring_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            wo_shape_ring = (8192 // 8, 9216 // 4)  # Use padded K and N
            self.model_config["SHARDED_WO_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                k=wo_shape_ring[0],
                n=wo_shape_ring[1],
            )

            wo_out_shard_shape_ring = (32, 9216 // 4 // RING_SIZE)  # Use padded N
            self.model_config["SHARDED_WO_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=wo_out_shard_shape_ring,
                core_grid=pf_mm_out_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["WO_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
                1,
                32,
                10240 // 8,
                9216 // 4,  # Use padded N
                RING_SIZE,
            )

            # Use padded K and N
            self.model_config["W1W3_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                k=8192 // 4,
                n=3840,
            )

            # Use padded K and N
            self.model_config["W2_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                k=3584,
                n=9216 // 4,
            )

            self.model_config["FF1_3_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
                1,
                32,
                8192 // 4,
                3840,  # Use padded N
                RING_SIZE,
            )

            self.model_config["FF2_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
                1,
                32,
                3584,
                9216 // 4,  # Use padded N
                RING_SIZE,
            )

            self.model_config["SHARDED_FF12_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 9216 // 4 // RING_SIZE),  # Use padded N
                core_grid=ring_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 3840 // RING_SIZE),  # Use padded N
                core_grid=pf_mm_out_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["SHARDED_FF12_PRE_MUL_RING_REDUCE_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 3840 // 30),  # Use padded N
                core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    self.start_core, 30, self.sub_core_grids, row_wise=True
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            mul_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 28, self.sub_core_grids, row_wise=True
            )
            self.model_config["MUL_IN_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 3584 // 28),  # Use padded K
                core_grid=mul_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["FF2_IN_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 3840 // RING_SIZE),  # Use padded K
                core_grid=ring_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["FF2_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 9216 // 4 // RING_SIZE),  # Use padded N
                core_grid=pf_mm_out_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(1, 0)
            core_range = ttnn.CoreRange(
                grid_offset, ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1)
            )
            LM_HEAD_RING_SIZE = 24
            self.lm_head_shape = (8192 // 4, 128 * 1024 // 8)

            lm_head_ring_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in LM_HEAD_32_GRID
                ]
            )

            lm_head_ring_core_input_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in LM_HEAD_INPUT_GRID
                ]
            )

            lm_head_ring_core_output_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in LM_HEAD_OUTPUT_GRID
                ]
            )

            lm_head_ring_16_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in LM_HEAD_16_GRID
                ]
            )
            self.model_config["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 2304 // LM_HEAD_RING_SIZE),  # padded shape
                core_grid=lm_head_ring_core_input_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["SHARDED_LM_HEAD_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, self.lm_head_shape[0] // 16),
                core_grid=lm_head_ring_16_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["LM_HEAD_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 16896 // LM_HEAD_RING_SIZE),  # padded shape
                core_grid=lm_head_ring_core_output_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["LM_HEAD_OUT_RING_RESHARD_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, self.lm_head_shape[1] // 32),
                core_grid=lm_head_ring_core_range_set,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["LM_HEAD_TG_RING_PROGCFG"] = self.matmul_1d_ring_lm_head_config(
                1,
                32,
                self.dim // 4,
                16896,  # use padded shape
                LM_HEAD_RING_SIZE,
                prefetch=False,
            )

            self.model_config["LM_HEAD_PREFILL_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                in0_shape=(1, 1, 32, 2048),
                in1_shape=(1, 1, 2048, 16384),
                grid=ttnn.CoreGrid(x=7, y=7),  # (7,10) leads to hangs
                act=None,
                is_fp32_accumulate=False,
                # overwrite_subblock_w=1,
                # overwrite_subblock_h=1,
            )

            attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
            attn_input_sub_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, 32, self.sub_core_grids, row_wise=True
            )
            self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, nearest_32(self.dim // (8 * lm_head_num_rows) // 4)),
                    core_grid=attn_input_sub_core_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.is_galaxy
                else ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        self.dim // attn_input_grid.num_cores,
                    ),  # Shard shape: [32, 128] -> 1 shard per core
                    attn_input_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            # glx doesn't support DRAM sharded matmuls yet
            self.model_config["XQKV_DECODE_PROGCFG"] = (
                ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(8, 5 if self.is_70b else lm_head_num_rows),
                    in0_block_w=2 if self.is_70b else 1,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                if self.is_galaxy
                else self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=self.dim,
                    n=self.qkv_size // self.num_devices,
                    num_cores=attn_input_grid.num_cores,
                )
            )

            full_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 7),
                    )
                }
            )
            self.model_config["FULL_GRID_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    full_grid,
                    [
                        32,
                        nearest_32(56),
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            self.model_config["MLP_ACT_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, self.dim // 4 // 16),  # dim / num devices / 16 cores
                    core_grid=ttnn.CoreGrid(x=8, y=2),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.dim >= 4096
                else self.model_config["FULL_GRID_MEMCFG"]
            )

            self.model_config["FF1_3_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                (
                    1,
                    1,
                    32,
                    self.dim // 4,
                ),
                (
                    1,
                    1,
                    self.dim // 4,
                    self.hidden_dim // 8,
                ),
                grid=ttnn.CoreGrid(x=8, y=2),
                overwrite_subblock_h=1,
                overwrite_subblock_w=1,
            )

            self.model_config["FF2_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                (
                    1,
                    1,
                    32,
                    self.hidden_dim // 8,
                ),
                (
                    1,
                    1,
                    self.hidden_dim // 8,
                    self.dim // 4,
                ),
                grid=ttnn.CoreGrid(x=8, y=2),
                overwrite_subblock_h=1,
                overwrite_subblock_w=1,
            )
            self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, self.hidden_dim // 28 // 8),  # shard_grid_cores = 28, num_devices=8
                core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 3))}),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )  # if self.dim==8192 else ttnn.DRAM_MEMORY_CONFIG

            self.model_config["FF1_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32 * 4, self.hidden_dim // 8 // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, self.dim // 8 // 4),  # shard_grid_cores = 8, num_devices=4
                    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.dim == 8192
                else ttnn.create_sharded_memory_config(
                    shape=(32 * 8, self.dim // 4 // 8),
                    core_grid=ttnn.CoreGrid(y=1, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            # Note PACKET_WORKER_CRS is 8 cores and it can NOT use any core in the following ranges:
            # {2,8}-{3,8},{5,3}-{6,3}  (CCL cores in 6u),
            # {3,3},{5,3}-{6,3}  (CCL cores in TG)
            # {1,0}-{2,0}, {1,4}-{2,5}, {1,9}-{2,9}, {5,0}-{6,2}, {5,4}-{6,7}, {5,9}-{6,9} (Matmul)
            # {0,0}-{0,9}, {4,0}-{4,9} (Prefetcher)
            # {3,6} (Matmul hop core)
            PACKET_WORKER_CRS = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
                    ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
                ]
            )
            self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32, 512),
                core_grid=PACKET_WORKER_CRS,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            FF1_CRS_RS_OUT = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                ttnn.CoreCoord(1, 0), 30, self.sub_core_grids, row_wise=True
            )
            self.model_config["REDUCE_SCATTER_OUT_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    FF1_CRS_RS_OUT,
                    [32, 32],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            self.model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, 2048 // 8 // 8),  # mesh_rows = 8, num_cores=8
                    core_grid=ttnn.CoreGrid(y=1, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if self.dim == 8192
                else ttnn.create_sharded_memory_config(
                    shape=(32 * 8, nearest_32(self.dim // 4 // 32)),  # mesh_rows = 8
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
            )

            self.model_config["FF2_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
                shape=(32 * 8, self.dim // 4 // 8),
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Vision model configs
            self.model_config["IMAGE_MLP_FC_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=self.vision_hidden_dim // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_MLP_PROJ_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_hidden_dim // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_ATTN_QKV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.vision_dim,
                n=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3)
                // self.num_devices,  # Head dim was padded to nearest 32
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["IMAGE_ATTN_OUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3) // self.num_devices,
                n=self.vision_dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["VISION_XATTN_Q_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim,
                n=(self.head_dim * self.n_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= 1024,
            )
            self.model_config["VISION_XATTN_KV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.dim,
                n=(self.head_dim * self.n_kv_heads) // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )
            self.model_config["VISION_XATTN_SCORE_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                m=seq_len,
                k=self.head_dim,
                n=cache_seq_len,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )
            self.model_config["VISION_XATTN_OUTPUT_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                m=seq_len,
                k=cache_seq_len,
                n=self.head_dim,
                grid_size=(8, 8),
                # in0_block_w=1, # TODO: Remove this when we get non-causal FlashDecode
                fuse_batch=False,
            )
            self.model_config["VISION_XATTN_DENSE_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim // self.num_devices,
                n=self.dim,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )

            self.model_config["VISION_PROJ_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=seq_len,
                k=self.vision_dim * 6,
                n=self.dim // self.num_devices,
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=False,
            )

            self.model_config["CROSS_TRANSFORMER_TEXT_OUTPUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                m=min(seq_len, max_seq),
                k=self.dim,
                n=self.vocab_size // 8,  # Magic number. LM Head always contains 8 splits
                grid_size=(8, 8),
                in0_block_w=1,
                fuse_batch=seq_len <= max_seq,
            )

            def _get_xattn_kv_prefill_mem_cfg(seq_len):
                M = (self.n_kv_heads // self.num_devices) * seq_len
                cores_x, cores_y = self.find_grid(M // self.tile_size)
                return ttnn.create_sharded_memory_config(
                    (
                        nearest_32(M // (cores_x * cores_y)),
                        self.head_dim,
                    ),
                    ttnn.CoreGrid(y=cores_y, x=cores_x),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

            self.model_config["XATTN_KV_PREFILL_MEM_CFG"] = _get_xattn_kv_prefill_mem_cfg

            self.VISION_MAX_MM_SEQ = nearest_32(self.vision_chunk_ntok)
            # RMS NORM
            self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(attn_input_grid)
            self.model_config["SHARDED_NORM_MLP_PRGM_CFG"] = self.create_sharded_norm_config(mlp_core_grid)
            self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"] = self.create_sharded_norm_config(self.lm_head_core_grid)

            # All gather matmuls currently only supported on T3K
            # We need it sharded on num_cores = num_devices
            self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    num_to_core_range_set(self.num_devices),
                    [
                        self.tile_padded_batch_rows,
                        self.dim // self.num_devices,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

            self.model_config = set_tg_attention_config(self.model_config, self.dim)

            self.is_multichip = self.num_devices > 1
            self.num_reduce_scatter_links = 1
            self.num_all_gather_links = (
                2 if self.is_galaxy else 1
            )  # TODO: try out 3 for short axis and 4 for long axis (TG only) <- should work but untested in model
            self.ccl_dtype = ttnn.bfloat8_b

    def is_distributed_norm(self, mode):
        if not self.is_multichip:
            return False
        if all([dim > 1 for dim in list(self.mesh_device.shape)]):  # 2D grid
            return True
        elif self.dim >= 8192 and mode == "prefill":  # Somewhere between 4k and 8k WH runs out of L1 if not distributed
            return True
        return False

    def ccl_topology(self):
        if self.num_devices == 8 and os.getenv("ACTUAL_DEVICE", "") != "TG":  # T3K
            return ttnn.Topology.Ring
        elif self.num_devices > 1:  # All other multi chip devices
            return ttnn.Topology.Linear
        return None

    def prepare_residual_tensor_decode(self, x, input_mem_cfg, force_replicated=False, on_host=False):
        """
        Prepare inputs for decode mode.
        x: (batch, seq, dim)
        """
        dims = (None, None) if force_replicated else (None, -1)
        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)

        if len(x.shape) == 3:
            batch = x.shape[0]
            seq_len = x.shape[1]
            # assert x.shape[2] == self.dim #TODO : pad self.dim at model level
        elif len(x.shape) == 4:
            seq_len = x.shape[0]
            assert x.shape[1] == 1
            batch = x.shape[2]
            assert x.shape[3] == self.dim

        assert seq_len == 1, "Only supporting decode mode"

        # Support input on device
        if torch.is_tensor(x):  # Input on host -> Use torch
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, dim]
            # Pad small batches to 32
            if batch < 32:
                zeros = torch.zeros(1, seq_len, 32, self.dim)
                zeros[:, :, :batch, :] = x
                x = zeros
        elif len(x.shape) == 3:  # Input on device -> Use ttnn
            x = ttnn.reshape(x, (batch, seq_len, 1, self.dim))  # [batch, seqlen, dim] -> [batch, seqlen, 1, dim]
            x = ttnn.permute(x, (1, 2, 0, 3))  # [seq_len, 1, batch, dim]
        elif len(x.shape) == 4:
            pass  # already in [seq_len, 1, batch, dim]

        if torch.is_tensor(x):
            x = ttnn.from_torch(
                x,
                device=self.mesh_device if not on_host else None,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
                memory_config=input_mem_cfg if not on_host else None,
            )
        else:  # Convert the row major layout from embedding back to tile layout
            x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
        return x

    def prepare_residual_tensor_prefill(self, x_bsh, force_replicated=False):
        """
        Prepare inputs for prefill mode.
        x: (batch, seq, hidden_dim)
        B: batch (1)
        S: sequence len
        H: dim
        """

        x_1BSH = x_bsh.unsqueeze(0)
        dims = (None, None) if force_replicated else (None, -1)

        mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)

        # input goes to DRAM
        xs_1BSH = ttnn.from_torch(
            x_1BSH,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    def _set_params_from_dict(self, params, is_hf=False):
        # Common params with different names between Meta and HF
        self.dim = params.get("dim", params.get("hidden_size"))
        self.n_heads = params.get("n_heads", params.get("num_attention_heads"))
        self.n_kv_heads = params.get("n_kv_heads", params.get("num_key_value_heads"))
        self.n_layers = params.get("n_layers", params.get("num_hidden_layers"))
        self.full_model_n_layers = self.n_layers
        self.norm_eps = params.get("norm_eps", params.get("rms_norm_eps"))
        self.vocab_size = params["vocab_size"]
        self.padded_vocab_size = 128 * 1024
        self.head_dim = params.get("head_dim", self.dim // self.n_heads)
        if is_hf:
            self.max_context_len = params.get("max_position_embeddings")
        else:
            self.max_context_len = (
                128 * 1024
            )  # For Llama3 Meta weights TODO: Remove this when we move to HF weights only

        # Handle different MLP dimension specifications
        if "intermediate_size" in params:
            self.hidden_dim = params["intermediate_size"]
            self.ffn_dim_multiplier = None
            self.multiple_of = None
        else:
            self.ffn_dim_multiplier = params["ffn_dim_multiplier"]
            self.multiple_of = params["multiple_of"]
            self.hidden_dim = calculate_hidden_dim(self.dim, self.ffn_dim_multiplier, self.multiple_of)

        if "_name_or_path" in params:
            if is_hf:
                normalized_path = os.path.normpath(params["_name_or_path"])
                # For HF paths, they might end with `<model_name>/snapshots/<snapshot_id>/`
                if "snapshots" in normalized_path:
                    full_model_name = normalized_path.split(os.path.sep)[-3]
                    self.model_name = full_model_name.split("--")[-1]
                else:
                    self.model_name = os.path.basename(normalized_path)
            else:
                self.model_name = os.path.basename(params["_name_or_path"])
            logger.info(f"Model name from params: {self.model_name}")

        if self.base_model_name == "Qwen2.5-7B" and self.num_devices not in [0, 2, 4]:
            raise AssertionError(
                "Qwen2.5-7B is only supported on 2 or 4 devices, run on an N300 or use MESH_DEVICE=N150x4"
            )

        self.unpadded_hidden_dim = self.hidden_dim
        # Don't need to pad for CPU runs
        if self.num_devices:
            # Default padding cores for each model, 0 if not set here
            default_padded_cores = {
                "Qwen2.5-72B": 32,
                "Qwen2.5-7B": 16,
                "QwQ-32B": 16,
            }.get(self.base_model_name, 0)

            # Override MLP padding cores from env var
            mlp_padded_cores = int(os.environ.get("PAD_MLP_CORES", default_padded_cores))

            # Only pad if MLP_PADDED_CORES is non-zero
            if mlp_padded_cores > 0:
                padded_hidden_dim = nearest_multiple(
                    self.hidden_dim, mlp_padded_cores * self.tile_size * self.num_devices
                )
                if padded_hidden_dim != self.hidden_dim:
                    logger.info(
                        f"PAD_MLP_CORES={mlp_padded_cores}, padding hidden dim from {self.hidden_dim} to {padded_hidden_dim}"
                    )
                    self.hidden_dim = padded_hidden_dim

        # RoPE params
        self.rope_theta = params.get("rope_theta")
        # If use_scaled_rope is not present, assume setting rope_scaling means use scaled rope
        # If it is present and is set to false, do not use scaled rope
        # Setting self.rope_scaling_factor to None is our way of saying do not use scaled rope
        rope_scaling_params = params.get("rope_scaling", None)
        if rope_scaling_params:
            self.rope_scaling_factor = rope_scaling_params.get("factor", None)
            self.orig_context_len = rope_scaling_params.get("original_max_position_embeddings", None)
        else:
            self.rope_scaling_factor = None
            self.orig_context_len = None

        # Vision params (Meta-specific)
        self.vision_chunk_size = params.get("vision_chunk_size", -1)
        self.vision_max_num_chunks = params.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = params.get("vision_num_cross_attention_layers", -1)

        # Vision constants
        self.vision_dim = 1280
        self.vision_mlp_ratio = 4
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_act_layer = ttnn.UnaryOpType.GELU
        self.vision_dropout = 0.0
        self.vision_attn_n_heads = 16
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = 32
        self.vision_n_global_layers = 8
        self.vision_max_num_tiles = 4
        self.vision_patch_size = 14
        self.vision_in_channels = 3

    @property
    def use_scaled_rope(self):
        return self.rope_scaling_factor is not None

    @property
    def base_model_name(self):
        return get_base_model_name(self.model_name)

    @property
    def vision_chunk_ntok(self):
        """
        Returns the number of tokens per chunk, accounting for the extra class token
        """
        return (self.vision_chunk_size // self.vision_patch_size) ** 2 + 1

    def _set_model_params(self, checkpoint_dir):
        if self.checkpoint_type == CheckpointType.Meta:
            self._set_params(checkpoint_dir)
        elif self.checkpoint_type == CheckpointType.HuggingFace:
            self._set_hf_params(checkpoint_dir)
        else:
            raise ValueError(f"Unsupported checkpoint type: {self.checkpoint_type}")

    def _set_params(self, checkpoint_dir):
        params_file = os.path.join(checkpoint_dir, "params.json")
        assert os.path.exists(params_file), f"params.json file not found at {params_file}"
        with open(params_file, "r") as f:
            params = json.load(f)
        self._set_params_from_dict(params)

        # Meta-style config dicts don't specity model name or rope_scaling_factor so hard-code these
        # Set the model name based on the checkpoint directory being loaded
        if "3.2-1B" in checkpoint_dir:
            self.model_name = "Llama-3.2-1B" + ("-Instruct" if self.instruct else "")
            self.rope_scaling_factor = 32
        elif "3.2-3B" in checkpoint_dir:
            self.model_name = "Llama-3.2-3B" + ("-Instruct" if self.instruct else "")
            self.rope_scaling_factor = 32
        elif "3.1-8B" in checkpoint_dir:
            self.model_name = "Llama-3.1-8B" + ("-Instruct" if self.instruct else "")
            self.rope_scaling_factor = 8
        elif "3.2-11B" in checkpoint_dir:
            self.model_name = "Llama-3.2-11B" + ("-Instruct" if self.instruct else "")
            self.rope_scaling_factor = 8  # shared with 3.1-8B
        elif "3.1-70B" in checkpoint_dir:
            self.model_name = "Llama-3.1-70B" + ("-Instruct" if self.instruct else "")
            self.rope_scaling_factor = 8
            self.is_70b = True  # self.dim == 8192 and self.n_layers == 80
            self.max_prefill_chunk_size = 128 * 1024
        elif "3.3-70B" in checkpoint_dir:
            self.model_name = "Llama-3.3-70B" + ("-Instruct" if self.instruct else "")
            self.rope_scaling_factor = 8
            self.is_70b = True  # self.dim == 8192 and self.n_layers == 80
            self.max_prefill_chunk_size = 128 * 1024
        else:
            logger.warning(f"Unknown Meta-style model: {checkpoint_dir}")
        self.orig_context_len = 8192

    def _set_hf_params(self, checkpoint_dir):
        if self.from_hf_url:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name).to_dict()
        else:
            config_file = os.path.join(checkpoint_dir, "config.json")
            assert os.path.exists(config_file), f"config.json file not found at {config_file}"
            with open(config_file, "r") as f:
                config = json.load(f)
        self._set_params_from_dict(config, is_hf=True)
        self.is_70b = self.dim == 8192 and self.n_layers == 80
        if self.is_70b:
            self.max_prefill_chunk_size = 128 * 1024
        # TODO Hack for deepseek distill 70b. generalize if needed
        if "Llama-70B" in checkpoint_dir:  # if we're using a distill version of 70B, use same settings as Llama-70b
            self.model_name = "Deepseek-R1-Distill-70B"
            self.rope_scaling_factor = 8
            self.is_70b = True  # self.dim == 8192 and self.n_layers == 80
            self.max_prefill_chunk_size = 128 * 1024
            self.orig_context_len = 8192

    def __repr__(self):
        return f"""ModelArgs(
    model_name={self.model_name}
    dim={self.dim},
    n_layers={self.n_layers},
    n_heads={self.n_heads},
    n_kv_heads={self.n_kv_heads},
    vocab_size={self.vocab_size},
    multiple_of={self.multiple_of},
    ffn_dim_multiplier={self.ffn_dim_multiplier},
    norm_eps={self.norm_eps},
    rope_theta={self.rope_theta},
    use_scaled_rope={self.use_scaled_rope},
    max_batch_size={self.max_batch_size},
    max_seq_len={self.max_seq_len},
    vision_chunk_size={self.vision_chunk_size},
    vision_max_num_chunks={self.vision_max_num_chunks},
    vision_num_cross_attention_layers={self.vision_num_cross_attention_layers}
)"""

    def is_vision(self):
        return self.vision_chunk_size > 0

    def get_state_dict_prefix(self, module_name, layer_num):
        text_prefix = "text_model." if self.is_vision() else ""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "TtLlamaMLP": "feed_forward",
            "TtLlamaAttention": "attention",
            "TtTransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }
        return text_prefix + layer_prefix + module_map[module_name]

    def weight_cache_path(self, dtype):
        # Keep the weight cache separate for generative and instruct weights
        if self.instruct:
            return (
                self.model_cache_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        else:
            return (
                self.model_cache_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]
            )

    def get_model_config(self):
        return self.model_config

    # TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
    def load_state_dict(self):
        """Generate or load state_dict for n_layers of the model"""
        if self.dummy_weights:
            reference_model = Transformer(self)
            state_dict = reference_model.state_dict()
            state_dict_prefix = self.get_state_dict_prefix("", None)
            state_dict = {f"{state_dict_prefix}{k}": torch.randn_like(v) for k, v in state_dict.items()}
        elif self.checkpoint_type == CheckpointType.Meta:
            state_dict = load_meta_state_dict(self.CKPT_DIR, self.n_layers)
        else:
            assert self.checkpoint_type == CheckpointType.HuggingFace
            if self.from_hf_url:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR)
                state_dict = model.state_dict()
            else:
                state_dict = load_hf_state_dict(self.CKPT_DIR)
            state_dict = standardize_hf_keys(state_dict)
            state_dict = convert_hf_to_meta(state_dict, self.head_dim)
        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

    def create_dram_sharded_mem_config(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors"""
        dram_cores = 12
        padded_size = math.ceil(n / (self.tile_size * dram_cores)) * (self.tile_size * dram_cores)
        shard_spec = ttnn.ShardSpec(
            self.dram_weight_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR
        )
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def create_dram_sharded_mem_config_lm_head(self, k, n):
        """Create DRAM-sharded memory config for width-sharded tensors for LM_HEAD"""

        def round_up(a, b):
            """
            Round up a to the nearest multiple of b
            """
            return b * math.ceil(a / b)

        num_cores = 24
        N_per_shard = round_up(math.ceil(n // num_cores), ttnn.TILE_SIZE)
        N_per_shard_in_dram = N_per_shard * 2
        in1_shard_shape = [k, N_per_shard_in_dram]
        in1_shard_spec = ttnn.ShardSpec(self.dram_weight_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
        return memory_config

    def matmul_config(
        self,
        m: int,
        k: int,
        n: int,
        grid_size: Tuple[int, int],
        in0_block_w: int = None,
        fuse_batch: bool = False,
        fused_activation=None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        per_core_M = math.ceil(m / (self.tile_size * grid_size[1]))
        per_core_N = math.ceil(n / (self.tile_size * grid_size[0]))

        out_subblock_h = 1
        out_subblock_w = (
            get_out_subblock_w(per_core_N, out_subblock_h) if not self.is_galaxy else 1
        )  # TODO: Needed for TG hang workaround

        if in0_block_w is None:
            in0_block_w = min(4, max(1, k // (self.tile_size * grid_size[0])))

        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid_size,
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=fused_activation,
            fuse_batch=fuse_batch,
        )

    def dram_shard_core_grid_for_k(self, k: int) -> Tuple[int, int]:
        rows, cols = self.find_grid(k // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid(self, N):
        """
        Find the number of rows and columns for a grid of cores such that
        the total number of tiles N can be evenly divided among the cores.
        Each core will have the same integer number of tiles.
        The grid size is limited to a maximum of 2 rows and 8 columns.

        Parameters:
            N (int): Total number of tiles to be distributed.

        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
        """
        max_rows = 8
        max_cols = 8
        max_cores = max_rows * max_cols

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        target = 32
        possible_cores = [k for k in range(1, max_cores + 1) if N % k == 0]
        possible_cores.sort(key=lambda x: abs(x - target))  # Sort by closest to target

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration for {N} tiles that evenly divides into {max_cores} cores of max size {max_rows}x{max_cols}."
        )

    def dram_shard_core_grid_for_k_and_n(self, k: int, n: int) -> Tuple[int, int]:
        rows, cols = self.find_grid_k_n(k // self.tile_size, n // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid_k_n(self, K, N):
        """
        Find the number of rows and columns for a grid of cores such that
        the total number of tiles N can be evenly divided among the cores.
        Each core will have the same integer number of tiles.
        The grid size is limited to a maximum of 2 rows and 8 columns.

        Parameters:
            N (int): Total number of tiles to be distributed.

        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
        """
        max_rows = 4
        max_cols = 8  # Maximum number of rows or columns
        max_cores = max_rows * max_cols  # Maximum number of cores (8x2 grid)

        # Find all possible numbers of cores that divide N and are less than or equal to max_cores
        possible_cores = [c for c in range(1, max_cores + 1) if K % c == 0 and N % c == 0]
        possible_cores.sort(reverse=True)  # Start checking from the largest number of cores

        for cores in possible_cores:
            # Try to find a grid configuration with the current number of cores
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols

        # If no configuration is found, assert an error
        raise AssertionError(
            f"Cannot find a grid configuration such that both {K} and {N} tiles evenly divide into cores of max size {max_rows}x{max_cols}."
        )

    def dram_matmul_config(
        self, m: int, k: int, n: int, num_cores=None
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
        if num_cores is None:
            # num_cores = self.dram_shard_core_grid_for_k_and_n(k).num_cores
            num_cores = self.dram_shard_core_grid_for_k_and_n(k, n).num_cores
            assert (
                k % (self.tile_size * num_cores) == 0
            ), f"k must be divisible by tile_size * num_cores: {k} % {self.tile_size * num_cores} != 0"
            # assert n % (self.tile_size * num_cores) == 0, f"n must be divisible by tile_size * num_cores: {n} % {self.tile_size * num_cores} != 0"
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=math.ceil(k / (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=None,
        )

    def matmul_1d_ring_config(
        self,
        B,
        M,
        K,
        N,
        num_cores,
        prefetch=True,
        untilize_out=False,
    ):
        M *= B  # Fuse batch always enabled

        in0_block_h = M // ttnn.TILE_SIZE
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N // num_cores // ttnn.TILE_SIZE

        num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
        num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
        num_blocks_total = num_blocks_y * num_blocks_x

        if num_blocks_total != num_cores:
            assert False, f"num_blocks_total {num_blocks_total} != num_cores {num_cores}"

        out_subblock_h = 1
        out_subblock_w = 8
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1

        hop_grid = [(3, 6)] if prefetch else []  # FIXME: Make not hard coded
        hop_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in hop_grid
            }
        )
        grid = num_to_coregrid(num_cores)

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=hop_core_range_set,
            num_global_cb_receivers=2 if prefetch else 1,
            untilize_out=untilize_out,
        )

        return program_config

    def matmul_1d_ring_lm_head_config(
        self,
        B,
        M,
        K,
        N,
        num_cores,
        prefetch=True,
    ):
        M *= B  # Fuse batch always enabled

        in0_block_h = M // ttnn.TILE_SIZE
        in0_block_w = K // num_cores // ttnn.TILE_SIZE
        out_block_h = M // ttnn.TILE_SIZE
        out_block_w = N // num_cores // ttnn.TILE_SIZE

        num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
        num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
        num_blocks_total = num_blocks_y * num_blocks_x

        if num_blocks_total != num_cores:
            assert False, f"num_blocks_total {num_blocks_total} != num_cores {num_cores}"

        out_subblock_h = 1
        out_subblock_w = 8
        while out_block_w % out_subblock_w != 0:
            out_subblock_w -= 1

        hop_grid = [(3, 6)]
        hop_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in hop_grid
            }
        )
        grid = num_to_coregrid(num_cores)

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=out_block_h,
            per_core_N=out_block_w,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=True,
            hop_cores=hop_core_range_set,
        )

        return program_config

    def matmul_1d_config(
        self,
        m,
        k,
        n,
        grid=ttnn.CoreGrid(x=8, y=8),
        act=None,
        is_fp32_accumulate=False,
        overwrite_per_core_k=None,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
    ):
        tile_width = 32
        tile_height = 32

        if (
            n // tile_width // grid.num_cores < 1
        ):  # use less number of cores in case we have more N num tiles than cores
            # assert (n // tile_width) % grid.x == 0
            grid_y = n // tile_width // grid.x
            grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

        per_core_m = m // tile_height
        per_core_k = math.ceil(k / tile_width / grid.num_cores)
        per_core_n = math.ceil(n / tile_width / grid.num_cores)

        if is_fp32_accumulate:
            max_subblock_w_h = 4
        else:
            max_subblock_w_h = 8

        # find the largest value between 1 and 8 that is a factor of per_core_n
        # e.g. if per_core_n is 14, then out_subblock_w = 7
        out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

        # find the largest value that is a factor of per_core_m such that
        # out_subblock_w * out_subblock_h <= 8
        out_subblock_h = max(
            [
                i
                for i in range(1, max_subblock_w_h + 1)
                if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h
            ]
        )

        if overwrite_per_core_k is not None:
            per_core_k = overwrite_per_core_k

        if overwrite_subblock_w is not None:
            out_subblock_w = overwrite_subblock_w

        if overwrite_subblock_h is not None:
            out_subblock_h = overwrite_subblock_h

        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=per_core_k,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=act,
            mcast_in0=True,
        )

    def matmul_1d_config_from_tensor_shapes(
        self,
        in0_shape,
        in1_shape,
        grid=ttnn.CoreGrid(x=8, y=8),
        act=None,
        is_fp32_accumulate=False,
        overwrite_subblock_w=None,
        overwrite_subblock_h=None,
    ):
        m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
        return self.matmul_1d_config(
            m,
            k,
            n,
            grid,
            act,
            is_fp32_accumulate,
            overwrite_subblock_w=overwrite_subblock_w,
            overwrite_subblock_h=overwrite_subblock_h,
        )

    def create_sharded_norm_config(self, grid):
        """Helper function to create LayerNormShardedMultiCoreProgramConfig for RMS NORM.

        Args:
            grid (ttnn.CoreGrid): Grid specification for the norm operation
        """
        block_w = self.dim // grid.num_cores // self.tile_size
        # Find largest value <= 4 that evenly divides block_w
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=self.tile_padded_batch_rows // self.tile_size,
            block_w=block_w,
            inplace=False,
        )

    def detect_checkpoint_type(self) -> CheckpointType:
        """Detect if checkpoint directory contains Meta or HuggingFace format weights.

        Returns:
            CheckpointType: Meta or HuggingFace enum value

        Raises:
            ValueError: If neither Meta nor HuggingFace checkpoint format is detected
        """
        config_path = os.path.join(self.CKPT_DIR, "config.json")
        params_path = os.path.join(self.CKPT_DIR, "params.json")

        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                if "transformers_version" in config:
                    return CheckpointType.HuggingFace

        if os.path.exists(params_path):
            return CheckpointType.Meta

        raise ValueError(
            f"Could not detect Meta or HuggingFace checkpoint format in {self.CKPT_DIR}. "
            "Directory should contain either config.json (HuggingFace) or params.json (Meta)."
        )

    def create_tokenizer(self):
        """Create and return a Tokenizer instance based on the checkpoint type."""
        if self.checkpoint_type == CheckpointType.Meta:
            # Use the Meta Tokenizer
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer

            return Tokenizer(self.tokenizer_path)
        else:
            # Create a HuggingFace AutoTokenizer
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH)

            # Add meta-compatible stop token list to the HF tokenizer
            if not "stop_tokens" in tokenizer.__dict__:
                tokenizer.stop_tokens = [tokenizer.eos_token_id]
            return tokenizer

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=True):
        if self.checkpoint_type == CheckpointType.Meta:
            if instruct:
                return encode_prompt_instruct(self.tokenizer, prompt_text, system_prompt_text)
            else:
                return self.tokenizer.encode(prompt_text, bos=True, eos=False)
        else:
            if instruct:
                try:
                    return encode_prompt_hf(self.tokenizer, prompt_text, system_prompt_text)
                except ValueError as e:
                    logger.warning(f"Failed to encode chat prompt, are you sure this is an instruct model? Error: {e}")
                    logger.warning(f"Falling back to base model encoding with no chat template")

            return self.tokenizer.encode(prompt_text, add_special_tokens=False)


def num_to_corerange(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(num_x - 1, num_y - 1),
    )


def num_to_coregrid(x):
    if x % 8 == 0:
        return ttnn.CoreGrid(y=x // 8, x=8)
    if x == 12:
        return ttnn.CoreGrid(y=2, x=6)
    if x == 20:
        return ttnn.CoreGrid(y=4, x=5)


def set_tg_attention_config(model_config, dim):
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    start_core = ttnn.CoreCoord(1, 0)
    shard_spec_n_cores_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
        start_core, 10, sub_core_grids, row_wise=False
    )

    model_config["CREATE_HEAD_INPUT_MEMCFG"] = (
        None
        if dim < 4096
        else ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec_n_cores_grid,
                [
                    32,
                    128,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
    )
    model_config["CREATE_HEAD_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            sub_core_grids,
            [
                32,
                128,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    num_cores = 40 if dim == 8192 else (24 if dim == 4096 else (20 if dim == 3072 else 12))

    model_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
        shape=(32 * mesh_cols, 32),  # mesh_cols = 4
        core_grid=num_to_coregrid(num_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    model_config["SELF_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
        shape=(32 * mesh_rows, dim // 4 // min(32, dim // 4 // 32)),
        core_grid=num_to_coregrid(min(32, dim // 4 // 32)),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
        shape=(32, 128),  # mesh_cols = 4
        core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(start_core, 32, sub_core_grids, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    return model_config
