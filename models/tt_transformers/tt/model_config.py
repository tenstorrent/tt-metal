# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
    get_out_subblock_w,
    encode_prompt_instruct,
    encode_prompt_hf,
    nearest_multiple,
)
from typing import Tuple
from models.utility_functions import nearest_32
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, auto
from models.tt_transformers.tt.load_checkpoints import (
    load_meta_state_dict,
    load_hf_state_dict,
    convert_hf_to_meta,
    convert_meta_to_hf,
    reverse_permute,
    standardize_hf_keys,
)


@dataclass
class ModelOptimizations:
    bfp4_mlp: bool
    # Future fields will go here:
    # bfp8_activations: bool
    # bfp8_layernorm: bool
    # bfp8_ccl: bool

    @classmethod
    def accuracy(cls, model_name):
        """Configuration optimized for accuracy
        Only 70B models uses bfp4 MLPs in this configuration
        """
        bfp4 = model_name in ["Llama3.1-70B", "DeepSeek-R1-Distill-Llama-70B", "Qwen2.5-72B"]
        return cls(bfp4_mlp=bfp4)

    @classmethod
    def performance(cls, model_name):
        """Configuration optimized for performance
        All models use bfp4 MLPs in this configuration
        """
        return cls(bfp4_mlp=True)


class CheckpointType(Enum):
    Meta = auto()
    HuggingFace = auto()


class ModelArgs:
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
        "LLAMA3_2_1B_PARAMS": "models/tt_transformers/model_params/Llama3.2-1B-Instruct",
        "LLAMA3_2_3B_PARAMS": "models/tt_transformers/model_params/Llama3.2-3B-Instruct",
        "LLAMA3_1_8B_PARAMS": "models/tt_transformers/model_params/Llama3.1-8B-Instruct",
        "LLAMA3_2_11B_PARAMS": "models/tt_transformers/model_params/Llama3.2-11B-Vision-Instruct",
        "LLAMA3_1_70B_PARAMS": "models/tt_transformers/model_params/Llama3.1-70B-Instruct",
    }

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=ModelOptimizations.accuracy,
    ):
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.device_name = {0: "CPU", 1: "N150", 2: "N300", 4: "N150x4", 8: "T3K", 32: "TG"}[self.num_devices]
        self.model_name = "Unknown"  # Llama model name will be dependent on the checkpoint directory
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.tile_size = 32
        self.is_70b = False
        self.from_hf_url = False  # updated below if true

        assert not os.getenv(
            "FAKE_DEVICE"
        ), "FAKE_DEVICE has been renamed to MESH_DEVICE for consistency with vLLM, please update your environment variables and run again."

        LLAMA_DIR = os.getenv("LLAMA_DIR")
        HF_MODEL = os.getenv("HF_MODEL")
        assert not (LLAMA_DIR and HF_MODEL), "Only one of LLAMA_DIR or HF_MODEL should be set"
        if LLAMA_DIR:
            if any([os.getenv("LLAMA_CKPT_DIR"), os.getenv("LLAMA_TOKENIZER_PATH"), os.getenv("TT_CACHE_PATH")]):
                logger.warning(
                    "LLAMA_DIR is set and will override LLAMA_CKPT_DIR, LLAMA_TOKENIZER_PATH, and TT_CACHE_PATH"
                )
            self.CKPT_DIR = LLAMA_DIR
            self.TOKENIZER_PATH = LLAMA_DIR
            self.CACHE_PATH = os.path.join(LLAMA_DIR, self.device_name)
            self.model_name = os.path.basename(LLAMA_DIR)  # May be overridden by config
        elif HF_MODEL:
            self.CKPT_DIR = HF_MODEL
            self.TOKENIZER_PATH = HF_MODEL
            self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join("model_cache", HF_MODEL, self.device_name)
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
            # Check if weights exist in the specified folder. If not warn the user to run the download and untar script.
        #            assert os.path.isfile(
        #                self.CKPT_DIR + "/consolidated.00.pth"
        #            ), f"weights consolidated.00.pth file does not exist. Please use the script `models/tt_transformers/scripts/get_weights.py` to download and untar the weights."

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
        if "instruct" in self.CACHE_PATH.lower():
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
            else:
                raise ValueError(
                    f"No local params found for {self.CKPT_DIR}, dummy weights are not supported for this model"
                )
            self._set_model_params(self.LOCAL_LLAMA_PARAMS[local_params])

        # Set the max number of tokens for each prefill chunk based on the model and device
        max_prefill_chunk_size_div1024 = os.getenv("MAX_PREFILL_CHUNK_SIZE")
        if max_prefill_chunk_size_div1024 is None:
            MAX_PREFILL_CHUNK_SIZES_DIV1024 = {
                "Llama3.2-1B": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128},
                "Llama3.2-3B": {"N150": 8, "N300": 128, "T3K": 128, "TG": 128},
                "Llama3.1-8B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128},
                "Llama3.2-11B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128},
                "Llama3.1-70B": {"N150": None, "N300": None, "T3K": 32, "TG": 128},
                "DeepSeek-R1-Distill-Llama-70B": {"N150": None, "N300": None, "T3K": 32, "TG": 128},
                "Qwen2.5-7B": {"N150": 4, "N300": 64, "T3K": 128, "TG": 128},
                "Qwen2.5-72B": {"N150": None, "N300": None, "T3K": 32, "TG": 128},
                "Phi-3.5-mini-instruct": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128},
                "QwQ-32B": {"N150": None, "N300": None, "T3K": 64, "TG": 128},
            }
            try:
                max_prefill_chunk_size_div1024 = MAX_PREFILL_CHUNK_SIZES_DIV1024[self.base_model_name][self.device_name]
            except KeyError:
                logger.warning(
                    f"Unknown model {self.model_name} on device {self.device_name}, setting MAX_PREFILL_CHUNK_SIZE to 4 for compatibility"
                )
                logger.warning(
                    f"Try setting MAX_PREFILL_CHUNK_SIZE to larger powers of 2 up to e.g. 128 for faster performance (if you run out of L1 memory it was too high)"
                )
                max_prefill_chunk_size_div1024 = 4
            assert (
                max_prefill_chunk_size_div1024 is not None
            ), f"Unsupported model {self.model_name} on device {self.device_name}"
        else:
            max_prefill_chunk_size_div1024 = int(max_prefill_chunk_size_div1024)
        self.max_prefill_chunk_size = max_prefill_chunk_size_div1024 * 1024

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

        device = mesh_device.get_devices()[0] if mesh_device is not None else None
        self.cluster_shape = list(mesh_device.shape)
        self.is_galaxy = self.num_devices == 32
        if device is not None:  # Avoid issue with test_torch.py not having a device
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
            )
            self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
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

            residual_grid = self.dram_shard_core_grid_for_k(self.dim // self.num_devices)
            self.model_config["DECODE_RESIDUAL_MEMCFG"] = (
                ttnn.L1_MEMORY_CONFIG  # FIXME: when residual add support typecasting for sharded tensors
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

            # Chunk values based on what works best empirically
            self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=256 if seqlen >= 2048 else 64,
                k_chunk_size=256 if seqlen >= 2048 else 64,
            )

            # nlp_concat_heads_decode will shard the data across this number of cores
            assert (
                self.n_heads % self.cluster_shape[1] == 0
            ), f"n_heads must be divisible by num_devices: {self.n_heads} % {self.cluster_shape[1]}"

            # Note: for some models (e.g. Mistral-Small) n_heads * head_dim != dim
            self.model_config["ATTN_OUTPUT_PROGCFG"] = (
                None
                if self.is_galaxy
                else self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=(self.n_heads * self.head_dim) // self.num_devices,
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

            # For maximum performance, set the prefill grid row to 8, even if it can fit in a smaller grid
            # prefill_rows = lambda seq_len: min(seq_len, 1024) // self.tile_size
            prefill_rows = 8
            mlp1_3_grid = lambda seq_len: (
                (8, min(min(seq_len, 1024) // 32, 4))
                if self.is_galaxy
                else self.find_prefill_grid(prefill_rows, self.dim // self.tile_size)
            )
            mlp2_grid = lambda seq_len: (
                (8, min(min(seq_len, 1024) // 32, 4))
                if self.is_galaxy
                else self.find_prefill_grid(prefill_rows, self.hidden_dim // self.tile_size)
            )

            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.dim // self.cluster_shape[0],
                n=self.hidden_dim // self.cluster_shape[1],
                grid_size=mlp1_3_grid(seq_len),
            )
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, 1024),
                k=self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1),
                n=self.dim,
                grid_size=mlp2_grid(seq_len),
            )

            k_dim = self.dim // self.cluster_shape[0] if self.is_galaxy else self.dim
            # n_dim = self.dim // self.cluster_shape[1] if self.is_galaxy else self.dim
            n_dim = (
                self.dim // self.cluster_shape[1]
                if self.is_galaxy
                else (
                    1024
                    if self.ccl_topology() == ttnn.Topology.Ring and 1024 % (self.dim / self.num_devices) == 0
                    else self.dim
                )
            )
            num_rows = lambda seq_len: min(seq_len, 1024 if self.is_galaxy else 2048)
            self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=num_rows(seq_len),
                k=k_dim,
                n=n_dim,
                grid_size=self.find_prefill_grid(num_rows(seq_len), n_dim // self.tile_size),
                in0_block_w=1 if self.is_galaxy else self.dim // 1024,
                fuse_batch=seq_len <= 1024,  # if self.is_galaxy else 2048),
            )

            # Calculate largest number of lm_head_num_rows such that self.dim % (lm_head_num_rows * lm_head_cores_per_row) == 0
            if self.num_devices == 32:
                lm_head_num_rows = 4
                while self.dim % (32 * 32 * lm_head_num_rows) != 0:
                    lm_head_num_rows -= 1
            else:
                lm_head_num_rows = 8
            lm_head_cores_per_row = 8
            while self.dim % (32 * lm_head_num_rows * lm_head_cores_per_row) != 0:
                lm_head_num_rows -= 1
                if lm_head_num_rows == 0:
                    lm_head_cores_per_row -= 1
                    if lm_head_cores_per_row == 0:
                        raise ValueError(
                            f"Could not find a lm_head_num_rows such that self.dim(={self.dim}) % (lm_head_num_rows * 8) == 0"
                        )
                    lm_head_num_rows = 8
            self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=lm_head_cores_per_row)

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
            self.MAX_QKV_MM_SEQ_LEN = 2048
            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, 8 if seq_len >= self.MAX_QKV_MM_SEQ_LEN else seq_len // self.tile_size // 8  # 8 rows
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=math.ceil(self.qkv_size / self.cluster_shape[1] / 32 / 8),  # N / TILE_WIDTH / grid width
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
            )

            assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"
            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                (((self.n_kv_heads // self.cluster_shape[1]) * seq_len // (8 * 8)), self.head_dim),
                ttnn.CoreGrid(y=8, x=8),
                ttnn.ShardStrategy.HEIGHT,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
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
                core_grid=ttnn.CoreRangeSet({num_to_corerange(batch_size_per_device_group)}),
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
            attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
            self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = (
                ttnn.create_sharded_memory_config(
                    shape=(32, nearest_32(self.dim // (8 * lm_head_num_rows) // 4)),
                    core_grid=ttnn.CoreGrid(y=lm_head_num_rows, x=8),
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

            if self.is_galaxy:
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

            self.set_tg_attention_config()

            self.is_multichip = self.num_devices > 1
            self.num_reduce_scatter_links = 1
            self.num_all_gather_links = (
                2 if self.is_galaxy else 1
            )  # TODO: try out 3 for short axis and 4 for long axis (TG only) <- should work but untested in model
            self.ccl_dtype = ttnn.bfloat8_b

            logger.info(f"Attention grid: {attn_input_grid}")
            logger.info(f"MLP grid: {mlp_core_grid}")
            logger.info(f"MLP prefill grids @ 32: w1/w3: {mlp1_3_grid(32)}, w2: {mlp2_grid(32)}")
            logger.info(
                f"MLP prefill grids @ max_seq_len({self.max_seq_len}): w1/w3: {mlp1_3_grid(self.max_seq_len)}, w2: {mlp2_grid(self.max_seq_len)}"
            )
            logger.info(f"LM head grid: {self.lm_head_core_grid}")

    def is_distributed_norm(self, mode):
        if not self.is_multichip:
            return False
        if all([dim > 1 for dim in list(self.mesh_device.shape)]):  # 2D grid
            return True
        elif self.dim > 4096 and mode == "prefill":  # Somewhere between 4k and 8k WH runs out of L1 if not distributed
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
            assert x.shape[2] == self.dim
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
                dtype=ttnn.bfloat16,
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
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        return xs_1BSH

    def _set_params_from_dict(self, params):
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
            self.model_name = os.path.basename(params["_name_or_path"])

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
        if "rope_scaling" in params and params.get("use_scaled_rope", True):
            self.rope_scaling_factor = params.get("factor", None)
            self.orig_context_len = params.get("original_max_position_embeddings", None)
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
        return self.model_name.split("B-")[0] + "B" if "B-" in self.model_name else self.model_name

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
            self.model_name = "Llama3.2-1B" + "-Instruct" if self.instruct else ""
            self.rope_scaling_factor = 32
        elif "3.2-3B" in checkpoint_dir:
            self.model_name = "Llama3.2-3B" + "-Instruct" if self.instruct else ""
            self.rope_scaling_factor = 32
        elif "3.1-8B" in checkpoint_dir:
            self.model_name = "Llama3.1-8B" + "-Instruct" if self.instruct else ""
            self.rope_scaling_factor = 8
        elif "3.2-11B" in checkpoint_dir:
            self.model_name = "Llama3.2-11B" + "-Instruct" if self.instruct else ""
            self.rope_scaling_factor = 8  # shared with 3.1-8B
        elif "3.1-70B" in checkpoint_dir:
            self.model_name = "Llama3.1-70B" + "-Instruct" if self.instruct else ""
            self.rope_scaling_factor = 8
            self.is_70b = True  # self.dim == 8192 and self.n_layers == 80
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
        self._set_params_from_dict(config)

    def __repr__(self):
        return f"""ModelArgs(
    dim={self.dim},
    n_layers={self.n_layers},
    n_heads={self.n_heads},
    n_kv_heads={self.n_kv_heads},
    vocab_size={self.vocab_size},
    multiple_of={self.multiple_of},
    ffn_dim_multiplier={self.ffn_dim_multiplier},
    norm_eps={self.norm_eps},
    rope_theta={self.rope_theta},
    rope_scaling_factor={self.rope_scaling_factor},
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
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
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

    def matmul_config(
        self,
        m: int,
        k: int,
        n: int,
        grid_size: Tuple[int, int],
        in0_block_w: int = None,
        fuse_batch: bool = False,
        fused_activation=None,
    ):
        per_core_M = math.ceil(m / (self.tile_size * grid_size[1]))
        per_core_N = math.ceil(n / (self.tile_size * grid_size[0]))

        out_subblock_h = 1
        out_subblock_w = (
            get_out_subblock_w(per_core_N, out_subblock_h) if not self.is_galaxy else 1
        )  # TODO: Needed for TG hang workaround

        if in0_block_w is None:
            in0_block_w = self.find_largest_divisor(k // (self.tile_size * grid_size[1]))

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

    def find_prefill_grid(self, row_tiles, col_tiles):
        """Find a grid such that the number of row tiles evenly divides into the number
        of rows and the number of column tiles evenly divides into the number of columns
        """
        max_rows = 8
        max_cols = 8

        # Find number of cols that evenly divides into the number of columns
        cols = None
        rows = None

        for i in range(max_cols, 0, -1):
            if col_tiles % i == 0:
                cols = i
                break

        for i in range(max_rows, 0, -1):
            if row_tiles % i == 0:
                rows = i
                break

        assert cols is not None, f"Cannot find a number of columns that evenly divides into {col_tiles}, not even 1(!)."
        assert rows is not None, f"Cannot find a number of rows that evenly divides into {row_tiles}, not even 1(!)."
        return rows, cols

    def dram_shard_core_grid_for_k_and_n(self, k: int, n: int) -> Tuple[int, int]:
        rows, cols = self.find_grid_k_n(k // self.tile_size, n // self.tile_size)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_grid_k_n(self, K, N):
        """
        Find the number of rows and columns for a grid of cores such that
        the total number of tiles N can be evenly divided among the cores.
        Each core will have the same integer number of tiles.

        Parameters:
            N (int): Total number of tiles to be distributed.

        Returns:
            tuple: A tuple (rows, cols) representing the grid dimensions.

        Raises:
            AssertionError: If it's not possible to find such a grid configuration.
        """
        max_rows = 8
        max_cols = 8  # Maximum number of rows or columns
        max_cores = max_rows * max_cols  # Maximum number of cores

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

    def find_largest_divisor(self, n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1  # Fallback to 1 if no divisor found

    def dram_matmul_config(self, m: int, k: int, n: int, num_cores=None):
        # in0_block_w must evenly divide k and be no larger than tile_size * num_cores
        if num_cores is None:
            # num_cores = self.dram_shard_core_grid_for_k(k).num_cores
            num_cores = self.dram_shard_core_grid_for_k_and_n(k, n).num_cores
            assert (
                k % (self.tile_size * num_cores) == 0
            ), f"k must be divisible by tile_size * num_cores: {k} % {self.tile_size * num_cores} != 0"
            # assert n % (self.tile_size * num_cores) == 0, f"n must be divisible by tile_size * num_cores: {n} % {self.tile_size * num_cores} != 0"
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.find_largest_divisor(k // (self.tile_size * num_cores)),
            per_core_M=math.ceil(m / self.tile_size),
            per_core_N=math.ceil(n / (self.tile_size * num_cores)),
            fused_activation=None,
        )

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
        per_core_k = self.find_largest_divisor(k // (self.tile_size * grid.num_cores))
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

    def reference_lm_head(self):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import ColumnParallelLinear

            return ColumnParallelLinear(self.dim, self.vocab_size, bias=False, init_method=lambda x: x)
        else:
            model = self.reference_transformer(wrap=False)
            layer = model.lm_head
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
            return layer

    def reference_transformer(self, wrap=True, load_checkpoint=False):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Transformer

            model = Transformer(self)
            if load_checkpoint:
                model.load_state_dict(self.load_state_dict())
            return model
        else:
            from transformers import AutoConfig, AutoModelForCausalLM

            # HF is much faster at loading from a checkpoint than generating from config
            # so use that by preference unless we don't have a checkpoint
            if self.dummy_weights and not load_checkpoint:
                config = AutoConfig.from_pretrained(self.CKPT_DIR)
                config.num_layers = self.n_layers
                model = AutoModelForCausalLM.from_config(config)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR)
            if wrap:
                wrapper = HfModelWrapper(model, self.head_dim)
                return wrapper
            else:
                return model

    def reference_rms_norm(self):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm

            return RMSNorm(self.dim, self.norm_eps)
        else:
            model = self.reference_transformer(wrap=False)
            layer = model.model.norm
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
            return layer

    def reference_mlp(self):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward

            return FeedForward(self.dim, 4 * self.dim, self.multiple_of, self.ffn_dim_multiplier)
        else:
            model = self.reference_transformer(wrap=False)
            layer = model.model.layers[0].mlp
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
            return layer

    def reference_embedding(self, reference_model=None):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.tt_transformers.tt.common import HostEmbedding

            return HostEmbedding(self)
        else:
            if reference_model is None:
                model = self.reference_transformer(wrap=False)
            else:
                model = reference_model
            layer = model.model.embed_tokens
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
            return layer

    def reference_decoder(self):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import TransformerBlock

            return TransformerBlock(layer_id=0, args=self)
        else:
            model = self.reference_transformer(wrap=False)
            layer = model.model.layers[0]
            wrapper = HfDecoderWrapper(layer, self.head_dim)
            return wrapper

    def reference_attention(self):
        if self.checkpoint_type == CheckpointType.Meta:
            from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import Attention

            return Attention(self)
        else:
            model = self.reference_transformer(wrap=False)
            layer = model.model.layers[0].self_attn
            wrapper = HfAttentionWrapper(layer, self.head_dim)
            return wrapper

    def set_tg_attention_config(self):
        shard_spec_n_cores_grid = ttnn.CoreRangeSet({num_to_corerange(40)})

        self.model_config["CREATE_HEAD_INPUT_MEMCFG"] = (
            None
            if self.dim < 4096
            else ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_n_cores_grid,
                    [
                        32,
                        32,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        )

        if self.is_galaxy:
            num_cores = 40 if self.dim == 8192 else (24 if self.dim == 4096 else (20 if self.dim == 3072 else 12))

            self.model_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
                shape=(32 * mesh_cols, 32),  # mesh_cols = 4
                core_grid=num_to_coregrid(num_cores),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.model_config["SELF_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
                shape=(32 * mesh_rows, self.dim // 4 // min(32, self.dim // 4 // 32)),
                core_grid=num_to_coregrid(min(32, self.dim // 4 // 32)),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
                shape=(32 * mesh_cols, 32),  # mesh_cols = 4
                core_grid=num_to_coregrid(min(32, self.dim // 8 // 32)),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            qkv_core_grid = self.dram_shard_core_grid_for_k(self.dim)
            self.model_config["QKV_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
                (
                    self.tile_size * mesh_rows,
                    self.dim // qkv_core_grid.num_cores,
                ),  # Shard shape: [32, 128] -> 1 shard per core
                core_grid=qkv_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            gather_core_grid = self.dram_shard_core_grid_for_k(self.dim // 4)
            self.model_config["SELF_OUT_GATHERED_MEMCFG"] = lambda mesh_rows: ttnn.create_sharded_memory_config(
                (
                    self.tile_size * mesh_rows,
                    self.dim // 4 // gather_core_grid.num_cores,
                ),
                core_grid=gather_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            users_core_grid = self.dram_shard_core_grid_for_k(self.dim // 8)
            self.model_config["GATHER_USERS_MEMCFG"] = lambda mesh_cols: ttnn.create_sharded_memory_config(
                (
                    self.tile_size * mesh_cols,
                    self.dim // 8 // users_core_grid.num_cores,
                ),
                core_grid=users_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )


class HfAttentionWrapper:
    def __init__(self, attention, head_dim):
        from transformers import DynamicCache

        super().__init__()
        self.attention = attention
        self.past_key_value = DynamicCache()
        self.head_dim = head_dim

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        output, _, self.past_key_value = self.attention(
            x,
            past_key_value=self.past_key_value,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
        )
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        return self.attention.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim))

    @property
    def cache_k(self):
        [(k, v)] = self.past_key_value.to_legacy_cache()
        hf_k = k.permute(0, 2, 1, 3)  # match meta-style reference which uses (batch_size, seq, n_kv_heads, head_dim)
        batch_size, seq_len, n_heads, head_dim = hf_k.shape

        meta_k = torch.zeros_like(hf_k)
        for b in range(batch_size):
            for s in range(seq_len):
                # Flatten just heads and head_dim
                flat = hf_k[b, s].flatten()
                # Apply reverse_permute
                transformed = reverse_permute(flat.unsqueeze(-1), n_heads, flat.shape[0], 1).squeeze(-1)
                # Restore heads and head_dim shape
                meta_k[b, s] = transformed.reshape(n_heads, head_dim)

        return meta_k

    @property
    def cache_v(self):
        [(k, v)] = self.past_key_value.to_legacy_cache()
        return v.permute(0, 2, 1, 3)  # match meta-style reference which uses (batch_size, seq, n_kv_heads, head_dim)


class HfDecoderWrapper:
    def __init__(self, decoder, head_dim):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        output, self.past_key_values = self.decoder.forward(
            x,
            past_key_value=self.past_key_values,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
        )
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        return self.decoder.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim))


class HfModelWrapper:
    def __init__(self, model, head_dim):
        from transformers import DynamicCache

        self.model = model
        self.head_dim = head_dim
        self.past_key_values = DynamicCache()

    def forward(self, inputs_embeds, start_pos, mode="decode"):
        position_ids = torch.tensor(
            [list(range(start_pos, start_pos + inputs_embeds.shape[1]))] * inputs_embeds.shape[0]
        )
        logits, new_cache, hidden_states = self.model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
            past_key_values=self.past_key_values,
            return_dict=False,
            output_hidden_states=True,
        )
        self.past_key_values = new_cache
        return logits if mode == "decode" else hidden_states[-2]  # last hidden state is final norm

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim))

    def eval(self):
        self.model.eval()


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
