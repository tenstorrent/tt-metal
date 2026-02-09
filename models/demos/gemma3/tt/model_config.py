# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0, nearest_32
from models.demos.gemma3.tt.load_checkpoints import convert_vision_hf_to_meta, convert_vision_meta_to_hf
from models.tt_transformers.tt.common import (
    calculate_prefill_warmup_seq_lens,
    cap_seq_lens_to_max_prefill_chunk_size,
    get_out_subblock_w,
    num_to_core_range_set,
)
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, convert_meta_to_hf, standardize_hf_keys
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    HfAttentionWrapper,
    HfDecoderWrapper,
    HfModelWrapper,
)
from models.tt_transformers.tt.model_config import ModelArgs as TTModelArgs
from models.tt_transformers.tt.model_config import determine_device_name, num_to_corerange

# file names for performance and accuracy mode override files
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME = "accuracy_decoder_config.json"


class ModelArgs(TTModelArgs):
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

    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,  # Set to False to reduce memory usage by not caching HF model
    ):
        self.num_devices = mesh_device.get_num_devices() if mesh_device else 0
        self.mesh_device = mesh_device
        self.arch_name = ttnn.get_arch_name()
        self.dram_grid_size = mesh_device.dram_grid_size() if mesh_device else None  # CoreCoord with (x, y)

        self.device_name = determine_device_name(self.mesh_device)

        logger.info(f"Inferring device name: {self.device_name}")
        device = mesh_device if mesh_device is not None else None
        self.cluster_shape = list(mesh_device.shape) if mesh_device is not None else None
        self.is_galaxy = self.num_devices == 32

        self.model_name = "Unknown"  # Llama model name will be dependent on the checkpoint directory
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.tile_size = 32
        self.is_70b = False
        self.is_90b = False
        self.use_qk_fused = False  # For Gemma 3, we do not use qk fused ops (rotary embedding + paged cache update)
        self.prefill_len_cutoff = 512 if is_blackhole() else 1024
        self.dummy_weights = dummy_weights
        self.cache_hf_flag = cache_hf  # Whether to cache HF model to avoid multiple loads (uses extra memory)
        self.cached_hf_model = None  # Save any HF model object to avoid loading it multiple times for reference methods

        self.rms_norm_add_unit_offset = False
        self.embed_scale = None

        assert not os.getenv(
            "FAKE_DEVICE"
        ), "FAKE_DEVICE has been renamed to MESH_DEVICE for consistency with vLLM, please update your environment variables and run again."

        # Remove trailing slashes so basename gets the right model name
        HF_MODEL = os.getenv("HF_MODEL")
        self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
        if HF_MODEL:
            self.CKPT_DIR = HF_MODEL
            self.TOKENIZER_PATH = HF_MODEL
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join("model_cache", HF_MODEL, self.device_name)
            else:  # For HF models, always append the device name (e.g. N150/N300/T3K/TG) to the cache path
                self.CACHE_PATH = os.path.join(self.CACHE_PATH, self.device_name)
            self.model_name = HF_MODEL.strip("/").split("/")[
                -1
            ]  # HF model names use / even on windows. May be overridden by config.
        else:
            assert False, "Please set HF_MODEL to a HuggingFace name e.g. google/gemma-3-27b-it"

        if not dummy_weights and not HF_MODEL:
            # Assert if all folders and files exist
            assert os.path.exists(self.CKPT_DIR), f"Checkpoint directory {self.CKPT_DIR} does not exist"
            os.makedirs(self.CACHE_PATH, exist_ok=True)

        logger.info(f"Checkpoint directory: {self.CKPT_DIR}")
        logger.info(f"Tokenizer file: {os.path.join(self.TOKENIZER_PATH, 'tokenizer.model')}")
        logger.info(f"Cache directory: {self.CACHE_PATH}")
        logger.info(f"Model name: {self.model_name}")

        # Some consumers like SentencePiece only accept str not Path for files
        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)

        # Load weights and tokenizer
        self.consolidated_weights_path = os.path.join(self.CKPT_DIR, "consolidated.00.pth")
        self.tokenizer_path = os.path.join(self.TOKENIZER_PATH, "tokenizer.model")

        self.instruct = instruct
        # If the weights file contain the keyword `instruct` also set self.instruct to true
        if any(keyword in self.CKPT_DIR.lower() for keyword in ("instruct", "it")):
            self.instruct = True

        # Check for supported batches since previous logic that contained the check was removed because it was unused
        supported_batches = {1, 2, 4, 8, 16, 32}
        if self.max_batch_size not in supported_batches:
            raise ValueError(f"Batch size {self.max_batch_size} not supported")

        # Load model params
        self._set_hf_params(self.CKPT_DIR)

        # Set the max number of tokens for each prefill chunk based on the model and device
        max_prefill_chunk_size_div1024 = os.getenv("MAX_PREFILL_CHUNK_SIZE")
        if max_prefill_chunk_size_div1024 is None:
            # TODO Improve this to be more general to more devices and models
            MAX_PREFILL_CHUNK_SIZES_DIV1024 = {
                "gemma-3-1b": {"N150": 32, "N300": 32, "T3K": 32, "TG": 32, "P150x4": 32},
                "gemma-3-4b": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
                "gemma-3-27b": {"N150": 128, "N300": 128, "T3K": 128, "TG": 128, "P150x4": 128},
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

        if self.base_model_name in ["gemma-3-27b", "gemma-3-4b"] and self.device_name == "N150":
            logger.info(f"Reducing prefill_len_cutoff to 512 for {self.model_name} on {self.device_name}")
            self.prefill_len_cutoff = 512

        if callable(optimizations):
            self.optimizations = optimizations(self)
        else:
            self.optimizations = optimizations

        # Configure data precision and math fidelity for tensors and kernels
        if self.optimizations is None:
            self.optimizations = DecodersPrecision.accuracy(num_decoders=self.n_layers, model_name=self.model_name)

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
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations
        # Update memory layouts (Tile, except MLP)
        self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

        self.tokenizer = None if dummy_weights else self.create_tokenizer()

        if device is not None:  # Avoid issue with test_torch.py not having a device
            self.n_local_heads = self.n_heads // self.cluster_shape[1]

            grid = device.compute_with_storage_grid_size()
            self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

            # DRAM weight grid specs for dram sharding matmuls
            self.dram_weight_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(self.dram_grid_size.x - 1, self.dram_grid_size.y - 1),
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
            self.compute_kernel_config_hifi4_fp32 = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
                dst_full_sync_en=False,
            )
            self.compute_kernel_config_hifi2_na = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            # Create memory config for sharded tensors
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
            # TODO: #26657 (self.num_devices == 8 and os.getenv("ACTUAL_DEVICE", "") != "TG") should be refactored, and investigate if ACTUAL_DEVICE environment variable is still used
            self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
                self.num_devices == 8
                and os.getenv("ACTUAL_DEVICE", "") != "TG"
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
            prefill_rows = 8  # TODO if BH = 10, if wh = 8
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

            mlp_w_dram_sharded = not self.is_galaxy
            n_w1_w3 = self.hidden_dim // self.cluster_shape[1]
            # Using dram_shard_grid_width to ensure per_core_N matches DRAM shard width for P100, otherwise matmuls silently give bad PCC
            dram_shard_grid_width = 8 if is_wormhole_b0() else self.dram_grid_size.x  # 7 for P100, 8 for P150
            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
                k=self.dim // self.cluster_shape[0],
                n=n_w1_w3,
                grid_size=mlp1_3_grid(seq_len),
                per_core_N=math.ceil(n_w1_w3 / (self.tile_size * dram_shard_grid_width))
                if mlp_w_dram_sharded
                else None,
            )
            n_w2 = self.dim
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),  # 512 if BH, 1024 if WH
                k=self.hidden_dim // (self.cluster_shape[1] if self.is_galaxy else 1),
                n=n_w2,
                grid_size=mlp2_grid(seq_len),
                per_core_N=math.ceil(n_w2 / (self.tile_size * dram_shard_grid_width)) if mlp_w_dram_sharded else None,
            )

            # Attention output is not necessarily the same dimension as the self.dim, e.g. in Mistral
            k_dim = (
                (self.n_heads * self.head_dim) // self.cluster_shape[0]
                if self.is_galaxy
                else (self.n_heads * self.head_dim) // self.num_devices
            )
            # TODO: #26657 (if self.num_devices == 8 and os.getenv("ACTUAL_DEVICE", "") != "TG") should be refactored, and investigate if ACTUAL_DEVICE environment variable is still used
            n_dim = (
                self.dim // self.cluster_shape[1]
                if self.is_galaxy
                else (
                    1024
                    if self.num_devices == 8
                    and os.getenv("ACTUAL_DEVICE", "") != "TG"
                    and 1024 % (self.dim / self.num_devices) == 0
                    else self.dim
                )
            )
            num_rows = lambda seq_len: min(seq_len, 1024)
            dram_sharded_wo = not (self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] or self.is_galaxy)
            self.model_config["WO_PREFILL_PROGCFG"] = lambda seq_len: self.matmul_config(
                m=num_rows(seq_len),
                k=k_dim,
                n=n_dim,
                grid_size=self.find_prefill_grid(prefill_rows, k_dim // self.tile_size),
                in0_block_w=1 if self.is_galaxy else None,
                fuse_batch=seq_len <= 1024,
                per_core_N=math.ceil(n_dim / (self.tile_size * dram_shard_grid_width)) if dram_sharded_wo else None,
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
            # 128256 comes from original llama 3 vocab size. 128256 / 4 was experimentally the maximum columns that worked per device.
            # The LM head for that was on 48 cores, so we know 128256 / 4 / 48 = 668 columns per core is close to the L1 limit.
            # FIXME: Update blackhole figure to be per-core as well.
            self.max_columns_per_device_lm_head = (
                128256 // 8 if is_blackhole() else 668 * self.lm_head_core_grid.num_cores
            )

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
            self.model_config["XQKV_PREFILL_PROGCFG"] = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                out_subblock_h=1,  # Must be divisible by per_core_M
                out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                per_core_M=max(
                    1, 8 if seq_len >= self.MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / self.tile_size / 8)  # 8 rows
                ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                per_core_N=math.ceil(
                    self.qkv_size / self.cluster_shape[1] / 32 / dram_shard_grid_width
                ),  # N / TILE_WIDTH / grid width
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
            )

            assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"
            self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: self.get_xqkv_prefill_mem_cfg(seq_len)

            self.model_config["CREATE_QKV_DECODE_SHARD"] = (
                ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, self.head_dim),
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                if is_blackhole()
                else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
            )

            self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                exp_approx_mode=False,
                q_chunk_size=128 if is_blackhole() else 256,
                k_chunk_size=128 if is_blackhole() else 256,
            )

            self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
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
                    compute_with_storage_grid_size=(8, 5 if self.is_70b or self.is_90b else lm_head_num_rows),
                    in0_block_w=2 if self.is_70b or self.is_90b else 1,
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
            )

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
            if self.is_multimodal:
                self.VISION_MAX_MM_SEQ = self.vision_chunk_ntok

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

            self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
            self.lm_head_dtype = ttnn.bfloat16

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

        self.capped_warmup_seq_len = min(self.max_prefill_chunk_size, self.max_seq_len)
        self.trace_prefill_supported_seq_lens = self.get_trace_prefill_supported_seq_lens()

    def get_warmup_prefill_supported_seq_lens(self):
        DEFAULT_VALUE = self.capped_warmup_seq_len
        # This dictionary is used to override the default ceil warmup prefill value
        model_specific_ceil_warmup_lengths = {
            # e.g. "gemma-3-4b": 4096
        }

        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(self.base_model_name, DEFAULT_VALUE)
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )

        to_warmup_seq_lens = self.filter_warmup_seq_lens(to_warmup_seq_lens)

        return to_warmup_seq_lens

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        # TODO: Add more model-specific filtering here
        # This filtering is based on the current PR's (https://github.com/tenstorrent/tt-metal/pull/33143) sequence lengths that are used for warmup
        return to_warmup_seq_lens

    def get_trace_prefill_supported_seq_lens(self):
        default_supported_seq_lens = {
            # for gemma we have different default supported seq lens than in tt_transformers
            # TODO: should be empty until https://github.com/tenstorrent/tt-metal/issues/33041 is fixed
            "N150": [],
            "N300": [],
            "T3K": [],
            "TG": [],
        }

        # TODO: If no specific sequence lengths are listed for a model and device, the default one will be used (from the default_supported_seq_lens dictionary)
        # TODO: should be empty until https://github.com/tenstorrent/tt-metal/issues/33041 is fixed
        model_specific_supported_seq_lens = {
            # EXAMPLE: "gemma-3-4b": {
            #     "N150": [128, 1024, 2048],
            # }
        }

        model_name = self.base_model_name
        device_name = self.device_name

        # Try model-specific sequence lengths first
        result = model_specific_supported_seq_lens.get(model_name, {}).get(device_name)
        if result:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)

        # Fall back to default sequence lengths
        result = default_supported_seq_lens.get(device_name)
        if result:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)

        # No supported sequence lengths found, return empty list
        return []

    def _set_model_specific_params(self):
        self.rms_norm_add_unit_offset = True
        self.embed_scale = self.dim**0.5

    # def _set_vision_params(self, vision_config):
    #     self.vision_dim = vision_config.get("hidden_size", 1280)
    #     self.vision_mlp_ratio = vision_config.get("intermediate_size", self.vision_dim * 4) // self.vision_dim
    #     self.vision_hidden_dim = vision_config.get("intermediate_size", self.vision_dim * self.vision_mlp_ratio)
    #     self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
    #     self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
    #     self.vision_n_layers = vision_config.get("num_hidden_layers", 32)
    #     self.vision_patch_size = vision_config.get("patch_size", 14)
    #     self.vision_in_channels = vision_config.get("num_channels", 3)
    #     self.vision_act_layer = ttnn.UnaryOpType.GELU  # or read from config if variable
    #     self.vision_dropout = vision_config.get("attention_dropout", 0.0)
    #     self.vision_max_num_tiles = 4
    #     self.vision_n_global_layers = 8

    def _set_vision_params(self, vision_config):
        self.vision_chunk_size = vision_config.get("vision_chunk_size", 896)
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get("vision_num_cross_attention_layers", 8)
        self.vision_dim = vision_config.get("hidden_size", 1152)

        intermediate_size = vision_config.get("intermediate_size", self.vision_dim * 4)
        self.vision_mlp_ratio = intermediate_size // self.vision_dim
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads

        self.vision_n_layers = vision_config.get("num_hidden_layers", 27)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_in_channels = vision_config.get("num_channels", 3)

        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.mm_tokens_per_image = vision_config.get("mm_tokens_per_image", 256)

        # Optional vision activation layer, defaults to GELU
        act_layer = vision_config.get("act_layer", "gelu").lower()
        self.vision_act_layer = {
            "gelu": ttnn.UnaryOpType.GELU,
            "relu": ttnn.UnaryOpType.RELU,
            "silu": ttnn.UnaryOpType.SILU,
        }.get(act_layer, ttnn.UnaryOpType.GELU)

        self.vision_n_global_layers = vision_config.get("n_global_layers", 8)

    def _set_hf_params(self, checkpoint_dir):
        def merge_text_config(base_config):
            text_config = base_config.get("text_config", {})
            # Merge non-nested keys into text_config
            text_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return text_config

        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            # Merge non-nested keys into vision_config
            vision_config.update({k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]})
            return vision_config

        from transformers import AutoConfig

        if self.dummy_weights:
            raise NotImplementedError("Dummy weights not supported for gemma models for now.")
        else:
            self.hf_config = AutoConfig.from_pretrained(self.CKPT_DIR).to_dict()

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(self.hf_config)
            if "vision_config" in self.hf_config:
                merged_vision_config = merge_vision_config(self.hf_config)
                self._set_vision_params(merged_vision_config)
        else:
            self._set_params_from_dict(self.hf_config)

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        if is_vision:
            text_prefix = "model.vision_tower.vision_model.encoder."
        else:
            text_prefix = ""

        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",  # If no module is given, just get layer prefix
        }

        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "self_attn.",
            "TransformerBlock": "",
            "": "",
        }

        module_map = vision_module_map if is_vision else module_map

        return text_prefix + layer_prefix + module_map[module_name]

    # TODO Update function for large models: For 1 layer tests we only want to load 1 checkpoint file, instead of all.
    def load_state_dict(self):
        if self.dummy_weights:
            from transformers import AutoModelForCausalLM

            raise NotImplementedError("Dummy weights not supported for gemma models for now.")
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.CKPT_DIR,
                torch_dtype="auto"
                # Note that the default setting is torch.dtype.float32, but model weights are
                # may come in any dtype. If the model's weights are in torch.dtype.bfloat16, this would result in 2x memory usage from an
                # unnecessary cast.
            )
            if self.cache_hf_flag:
                self.cached_hf_model = model
            state_dict = model.state_dict()

        if self.is_multimodal:
            state_dict = convert_vision_hf_to_meta(state_dict, self.head_dim)
        else:
            state_dict = standardize_hf_keys(state_dict)
            state_dict = convert_hf_to_meta(state_dict, self.head_dim)

        keys_dict = list(state_dict.keys())[:]
        remv = [f"layers.{i}." for i in list(range(self.n_layers, self.full_model_n_layers))]
        for k in keys_dict:
            if any([r in k for r in remv]):
                state_dict.pop(k)

        return state_dict

    def create_tokenizer(self):
        from transformers import AutoTokenizer

        # Mapping of base model names to their known tokenizer paths
        # These are the original models that have proper tokenizers
        base_model_tokenizer_mapping = {
            "gemma-3-4b-it": "google/gemma-3-4b-it",
        }

        logger.info(f"Tokenizer path: {self.TOKENIZER_PATH}")
        logger.info(f"Model name: {self.model_name}")
        logger.info(f"Base model name: {self.base_model_name}")

        try:
            # Try to load tokenizer from the original model path
            # If there is no Processor, it will return Tokenizer (useful for multimodal models)
            tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_PATH, local_files_only=os.getenv("CI") == "true")
            logger.info(f"Successfully loaded tokenizer from {self.TOKENIZER_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.TOKENIZER_PATH}: {e}")

            # Try to use base model tokenizer as fallback
            fallback_tokenizer_path = base_model_tokenizer_mapping.get(self.base_model_name)

            if fallback_tokenizer_path:
                logger.info(f"Attempting to use fallback tokenizer: {fallback_tokenizer_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        fallback_tokenizer_path, local_files_only=os.getenv("CI") == "true"
                    )
                    logger.info(f"Successfully loaded fallback tokenizer from {fallback_tokenizer_path}")
                except Exception as fallback_e:
                    logger.error(f"Failed to load fallback tokenizer from {fallback_tokenizer_path}: {fallback_e}")
                    raise fallback_e
            else:
                logger.error(f"No fallback tokenizer found for base model: {self.base_model_name}")
                raise e

        # Add meta-compatible stop token list to the HF tokenizer
        if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
            tokenizer.stop_tokens = [tokenizer.eos_token_id]
        return tokenizer

    def reference_vision_multi_modal(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.multi_modal_projector
        return layer

    def reference_vision_rms_norm(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.multi_modal_projector.mm_soft_emb_norm
        return layer

    def reference_rms_norm(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[i].self_attn.q_norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_rms_norm_text(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.norm
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def get_hf_model_cls(self):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

        if not self.is_multimodal:
            return AutoModelForCausalLM

        for model_cls in (AutoModelForVision2Seq, AutoModelForImageTextToText):
            if type(self.hf_config) == dict:
                return model_cls

        raise ValueError(f"Unknown model for config {type(self.hf_config)}")

    def reference_mlp(self):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].mlp
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        pass

        if self.dummy_weights and not load_checkpoint:
            raise NotImplementedError("Dummy weights not supported for gemma models for now.")
        else:
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(self.CKPT_DIR)
        if wrap:
            wrapper = HfModelWrapper(model, self.head_dim)
            return wrapper
        else:
            return model

    def reference_gemma_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model
        layer._load_state_dict = layer.load_state_dict
        layer.load_state_dict = lambda x: layer._load_state_dict(convert_vision_meta_to_hf(x, self.head_dim))
        return layer

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model
        return layer

    def reference_vision_mlp(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0].mlp
        return layer

    def reference_siglip_patch_embed(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings.patch_embedding
        return layer

    def reference_vision_pos_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings.position_embedding
        return layer

    def reference_vision_embedding(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.embeddings
        return layer

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        model = self.reference_vision_transformer(wrap=False)
        if layer_name == "layer_norm1":
            layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm1
        elif layer_name == "layer_norm2":
            layer = model.vision_tower.vision_model.encoder.layers[0].layer_norm2
        else:
            layer = model.vision_tower.vision_model.post_layernorm
        return layer

    def reference_vision_attention(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0].self_attn  # Common naming
        return layer

    def reference_vision_encoder_block(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder.layers[0]
        return layer

    def reference_vision_encoder(self):
        model = self.reference_vision_transformer(wrap=False)
        layer = model.vision_tower.vision_model.encoder
        return layer

    def reference_decoder(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[i]
        rotary_emb = model.model.rotary_emb

        rotary_emb_local = model.model.rotary_emb_local
        wrapper = HfGemmaDecoderWrapper(layer, self.head_dim, rotary_emb, rotary_emb_local)

        return wrapper

    def reference_decoder_text(self, i=0):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0]
        use_position_embeddings = layer.__class__.__name__ != "Phi3DecoderLayer" or self.base_model_name in ("phi-4",)
        if hasattr(model.model, "rotary_emb_local"):
            rotary_emb_local = model.model.rotary_emb_local
        else:
            rotary_emb_local = None
        wrapper = HfDecoderWrapper(
            layer, self.head_dim, model.model.rotary_emb if use_position_embeddings else None, rotary_emb_local
        )
        return wrapper

    def reference_attention(self, rope_embeddings="global"):
        model = self.reference_transformer(wrap=False)
        layer = model.model.layers[0].self_attn
        use_position_embeddings = layer.__class__.__name__ in ("Gemma3Attention",)
        if "gemma-3" in self.model_name:
            if rope_embeddings == "local":
                rotary_emb = model.model.rotary_emb_local
            else:
                rotary_emb = model.model.rotary_emb
        else:
            rotary_emb = model.model.rotary_emb
        wrapper = HfAttentionWrapper(layer, self.head_dim, rotary_emb if use_position_embeddings else None)
        return wrapper


class HfGemmaDecoderWrapper:
    def __init__(self, decoder, head_dim, rotary_emb, rotary_emb_local):
        from transformers import DynamicCache

        self.decoder = decoder
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rotary_emb_local = rotary_emb_local
        self.past_key_values = DynamicCache()

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        position_ids = torch.tensor([list(range(start_pos, start_pos + x.shape[1]))] * x.shape[0])
        # TODO: Generalize for other HF models

        position_embeddings_global = self.rotary_emb(x, position_ids)
        position_embeddings_local = self.rotary_emb_local(x, position_ids)
        if mask is not None:
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
        result = self.decoder.forward(
            x,
            position_embeddings_global=position_embeddings_global,
            position_embeddings_local=position_embeddings_local,
            past_key_value=self.past_key_values,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=mask,
        )
        output = result[0]
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        return self.decoder.load_state_dict(convert_meta_to_hf(state_dict, self.head_dim))
