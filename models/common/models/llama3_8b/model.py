# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.1-8B Transformer model and executors.

Model:
    Llama3Transformer1D — pure forward methods, no input/output processing

Executors (thin wrappers around engines in models/common/models/executor.py):
    EagerLlamaExecutor  — direct execution
    TracedLlamaExecutor — traced execution with capture/replay

Architecture:
    Llama3Transformer1D (1D only — non-TG)
    ├── Embedding1D
    ├── RotarySetup1D
    ├── TransformerBlock1D × n_layers
    │   ├── RMSNorm1D  (attention_norm)
    │   ├── Attention1D
    │   ├── RMSNorm1D  (ff_norm)
    │   └── MLP1D
    ├── RMSNorm1D  (final norm)
    ├── LMHead1D
    └── Sampling1D (optional)

Loop policy functions (run_teacher_forcing, run_perf_benchmark) are in
models/common/models/executor.py.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.executor import EagerLLMExecutor, TracedLLMExecutor
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _compute_kernel_config_hifi2
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _create_dram_sharded_mem_config
from models.common.modules.rmsnorm.rmsnorm_1d import SHARD_HEIGHT, RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import TT_CCL, default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, pad_dim_to_size

# =============================================================================
# Runtime Config
# =============================================================================


class Llama31DecoderPrecision:
    """Per-decoder tensor dtype and math-fidelity selection."""

    _DTYPES = {
        "bfp4": ttnn.bfloat4_b,
        "bfp8": ttnn.bfloat8_b,
        "bf16": ttnn.bfloat16,
        None: None,
    }

    @classmethod
    def from_string(cls, optimizations: str):
        if optimizations == "performance":
            return cls.performance
        if optimizations == "accuracy":
            return cls.accuracy
        raise ValueError(
            f"Invalid optimization configuration: {optimizations}. Allowed values are 'performance' or 'accuracy'"
        )

    @classmethod
    def performance(cls, num_decoders: int, model_name: str):
        inst = cls(num_decoders, model_name, cls._performance_settings(model_name))
        if model_name == "Llama-3.1-8B-Instruct" and num_decoders > 31:
            inst._tensor_precision[31]["ff1_ff3"] = "bfp8"
            inst._op_fidelity[31]["li_ff1_ff3"] = "hifi2fp16"
            inst._update_full_name()
        inst.__name__ = "performance"
        return inst

    @classmethod
    def accuracy(cls, num_decoders: int, model_name: str):
        inst = cls(num_decoders, model_name, cls._accuracy_settings(model_name))
        inst.__name__ = "accuracy"
        return inst

    def __init__(self, num_decoders: int, model_name: str, settings: dict | None = None):
        self.model_name = model_name
        default_tensor_precision, default_op_fidelity = self._default_settings()
        settings = settings or {}
        default_tensor_precision.update(settings.get("tensor_precision", {}))
        default_op_fidelity.update(settings.get("op_fidelity", {}))
        self._tensor_precision = {decoder_id: dict(default_tensor_precision) for decoder_id in range(num_decoders)}
        self._op_fidelity = {decoder_id: dict(default_op_fidelity) for decoder_id in range(num_decoders)}
        self._update_full_name()

    @staticmethod
    def _base_model_name(model_name: str):
        for suffix in ("-Instruct", "-instruct"):
            if model_name.endswith(suffix):
                return model_name[: -len(suffix)]
        return model_name

    @classmethod
    def _accuracy_settings(cls, model_name: str):
        base_model_name = cls._base_model_name(model_name)
        if base_model_name.startswith("Llama-3") or base_model_name.startswith("Meta-Llama-3"):
            return {
                "tensor_precision": {
                    "wqkv": "bfp8",
                    "kv_cache": "bfp8",
                    "wo": "bfp8",
                },
                "op_fidelity": {
                    "li_ff1_ff3": "hifi2fp16",
                    "li_ff2": "hifi2fp16",
                },
            }
        return {
            "tensor_precision": {
                "wqkv": "bf16",
                "kv_cache": "bf16",
                "wo": "bf16",
            },
            "op_fidelity": {
                "li_qkv_decode": "hifi4",
                "li_qkv_prefill": "hifi4",
                "sdpa_decode": "hifi4",
                "sdpa_prefill": "hifi4",
                "li_o_decode": "hifi4",
                "li_o_prefill": "hifi4",
            },
        }

    @classmethod
    def _performance_settings(cls, model_name: str):
        return {
            "tensor_precision": {"ff1_ff3": "bfp4"},
            "op_fidelity": {"li_ff1_ff3": "lofi"},
        }

    @staticmethod
    def _default_settings():
        return (
            {
                "ff1_ff3": "bfp8",
                "ff2": "bfp8",
                "wqkv": "bfp8",
                "wo": "bfp8",
                "kv_cache": "bfp8",
                "activation": None,
            },
            {
                "li_ff1_ff3": "hifi2fp16",
                "li_ff2": "hifi2fp16",
                "li_qkv_decode": "hifi2",
                "sdpa_decode": "hifi2",
                "li_o_decode": "hifi2",
                "li_qkv_prefill": "hifi2",
                "sdpa_prefill": "hifi4",
                "li_o_prefill": "hifi2",
                "accuracy": "hifi4fp32",
            },
        )

    def get_tensor_dtype(self, decoder_id: int, tensor: str, prefetcher: bool = False):
        effective_decoder_id = 0 if prefetcher else decoder_id
        value = self._tensor_precision.get(effective_decoder_id, {}).get(tensor)
        if prefetcher and value is None and tensor != "activation":
            return ttnn.bfloat8_b
        return self._DTYPES.get(value)

    def get_math_fidelity(self, decoder_id: int, op: str, configuration):
        kernel_lookup = {
            "lofi": configuration.compute_kernel_config_lofi,
            "hifi2": configuration.compute_kernel_config_hifi2,
            "hifi2na": configuration.compute_kernel_config_hifi2_na,
            "hifi2fp16": configuration.compute_kernel_config_hifi2_fp16,
            "hifi2nol1acc": configuration.compute_kernel_config_hifi2_nol1acc,
            "hifi4": configuration.compute_kernel_config_hifi4,
            "hifi4fp32": configuration.compute_kernel_config_hifi4_fp32,
        }
        return kernel_lookup[self._op_fidelity[decoder_id][op]]

    def _update_full_name(self):
        self._full_name = " | ".join(
            f"Decoder {decoder_id}: precision_cfg = {self._tensor_precision[decoder_id]}, fidelity_cfg = {self._op_fidelity[decoder_id]}"
            for decoder_id in self._tensor_precision
        )


def _nearest_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _nearest_32(value: int) -> int:
    return _nearest_multiple(value, 32)


def _num_to_core_range_set(num_cores: int):
    assert num_cores < 8 or num_cores % 8 == 0
    num_x = min(num_cores, 8)
    num_y = num_cores // num_x
    assert num_x * num_y == num_cores
    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            )
        }
    )


def _get_out_subblock_w(per_core_n: int, out_subblock_h: int):
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_n % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


def _device_name(mesh_device) -> str:
    num_devices = mesh_device.get_num_devices()
    dram_grid_size = mesh_device.dram_grid_size()
    if ttnn.device.is_blackhole(mesh_device):
        return {
            1: "P100" if dram_grid_size and dram_grid_size.x == 7 else "P150",
            2: "P300",
            4: "P150x4",
            8: "P150x8",
            32: "BHGLX",
        }[num_devices]
    if ttnn.device.is_wormhole_b0(mesh_device):
        return {1: "N150", 2: "N300", 4: "N150x4", 8: "T3K", 32: "TG"}[num_devices]
    raise ValueError(f"Unsupported architecture: {ttnn.get_arch_name()}")


def _base_model_name(model_name: str) -> str:
    for suffix in ("-Instruct", "-instruct"):
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


class Llama31RuntimeArgs:
    """TTTv2 runtime/config object for the Llama-3.1-8B 1D path."""

    def __init__(
        self,
        mesh_device,
        *,
        instruct: bool,
        max_batch_size: int,
        max_seq_len: int,
        model_name: str,
        model_info: dict,
        model_cache_path: str | Path,
        optimizations="performance",
        n_layers: int | None = None,
        tokenizer=None,
        prompt_encoder: Callable | None = None,
        state_dict_loader: Callable | None = None,
    ):
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.dram_grid_size = mesh_device.dram_grid_size()
        self.device_name = _device_name(mesh_device)
        self.cluster_shape = list(mesh_device.shape)
        self.cluster_type = ttnn.cluster.get_cluster_type()
        self.is_galaxy_cluster = self.cluster_type in (
            ttnn.cluster.ClusterType.GALAXY,
            ttnn.cluster.ClusterType.TG,
            ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        )
        self.is_galaxy = self.num_devices == 32
        if self.is_galaxy:
            raise ValueError("Llama31RuntimeArgs only supports 1D non-Galaxy meshes.")

        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.batch_size_per_device_group = max_batch_size
        self.tile_size = ttnn.TILE_SIZE
        self.dummy_weights = False
        self.rms_norm_add_unit_offset = False
        self.embed_scale = None
        self.prefetcher = None
        self.prefill_len_cutoff = 512 if ttnn.device.is_blackhole(mesh_device) else 1024
        self.instruct = instruct

        self.model_name = model_name
        self.model_cache_path = Path(model_cache_path)
        self.tokenizer = tokenizer
        self._prompt_encoder = prompt_encoder
        self._state_dict_loader = state_dict_loader
        self._set_model_params(model_info)
        if n_layers is not None:
            self.n_layers = n_layers
        self.full_model_n_layers = getattr(self, "full_model_n_layers", self.n_layers)
        self.max_prefill_chunk_size = self.get_max_prefill_chunk_size()
        self.disable_batched_prefill = self.base_model_name == "Llama-3.1-8B" and self.device_name in (
            "P150",
            "P300",
            "P150x4",
            "P150x8",
        )
        if self.base_model_name == "Llama-3.1-8B" and self.device_name in ("N150",):
            self.prefill_len_cutoff = 512

        if optimizations is None:
            self.optimizations = Llama31DecoderPrecision.performance(self.n_layers, self.model_name)
        elif isinstance(optimizations, str):
            self.optimizations = Llama31DecoderPrecision.from_string(optimizations)(self.n_layers, self.model_name)
        else:
            self.optimizations = optimizations

        self.tile_padded_batch_rows = ttnn.TILE_SIZE * int(math.ceil(self.max_batch_size / ttnn.TILE_SIZE))
        self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"
        self.model_config = {}
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations
        self.processor = None
        self.use_qk_fused = True

        assert self.n_heads % self.cluster_shape[1] == 0
        assert self.n_kv_heads % self.cluster_shape[1] == 0
        self.n_local_heads = self.n_heads // self.cluster_shape[1]
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
        self.min_kv_prefill_shard_seqlen = (ttnn.TILE_SIZE * 8 * 8) / (self.n_kv_heads // self.cluster_shape[1])

        self._use_t3k_fused_agmm_config = not self.is_galaxy_cluster
        self._use_fused_all_gather_matmul = (
            self.num_devices == 8
            and self._use_t3k_fused_agmm_config
            and (self.dim // ttnn.TILE_SIZE // self.num_devices) % self.num_devices == 0
            and self.num_devices > 1
            and self.ccl_topology() == ttnn.Topology.Ring
        )
        self.dram_shard_grid_width = 8 if ttnn.device.is_wormhole_b0(mesh_device) else self.dram_grid_size.x
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)
        self.dram_weight_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.dram_grid_size.x - 1, self.dram_grid_size.y - 1))}
        )
        lm_head_num_rows = 8
        lm_head_cores_per_row = 8
        while self.dim % (ttnn.TILE_SIZE * lm_head_num_rows * lm_head_cores_per_row) != 0:
            lm_head_num_rows -= 1
            if lm_head_num_rows == 0:
                lm_head_cores_per_row -= 1
                if lm_head_cores_per_row == 0:
                    raise ValueError("Could not find a valid LM head core grid")
                lm_head_num_rows = 8
        self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=lm_head_cores_per_row)
        self.max_columns_per_device_lm_head = 668 * self.lm_head_core_grid.num_cores
        self.prefill_rows = 8
        self.attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
        self.mlp_core_grid = self.dram_shard_core_grid_for_k_and_n(self.dim, self.hidden_dim // self.num_devices)
        self.mlp2_core_grid = self.dram_shard_core_grid_for_k_and_n(self.hidden_dim // self.num_devices, self.dim)
        self._init_compute_kernel_configs()
        self._init_model_config()
        self.capped_warmup_seq_len = min(self.max_prefill_chunk_size, self.max_seq_len)
        self.trace_prefill_supported_seq_lens = self.get_trace_prefill_supported_seq_lens()

    @property
    def base_model_name(self):
        return _base_model_name(self.model_name)

    @property
    def use_fused_all_gather_matmul(self):
        return self._use_fused_all_gather_matmul

    def _set_model_params(self, model_info):
        for key, value in model_info.items():
            if value is not None or key != "model_name":
                setattr(self, key, value)

    def _init_compute_kernel_configs(self):
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
        self.compute_kernel_config_hifi2_nol1acc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _init_model_config(self):
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations
        self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        self.model_config["CREATE_QKV_DECODE_SHARD"] = (
            ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.head_dim),
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            if ttnn.device.is_blackhole(self.mesh_device)
            else ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        self.model_config["ATTN_OUTPUT_PROGCFG"] = self.dram_matmul_config(
            m=self.tile_padded_batch_rows,
            k=(self.n_heads * self.head_dim) // self.num_devices,
            n=self.dim,
            num_cores=self.n_heads // self.num_devices,
        )
        self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"] = self.get_decode_all_gather_matmul_program_config()
        self.model_config[
            "ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"
        ] = self.get_decode_all_gather_matmul_output_mem_config()
        self.model_config["ATTN_AGMM_CONFIG"] = {"num_links": 1, "chunks_per_sync": 10, "num_workers_per_link": 2}
        self.model_config["MLP_RS_CONFIG"] = {
            "num_links": 1,
            "chunks_per_sync": 10,
            "num_workers_per_link": 2,
            "rs_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
        self.model_config["SAMPLING_AG_CONFIG"] = {
            "allow_force_argmax": False,
            "num_links": 1,
            "chunks_per_sync": 10,
            "num_workers_per_link": 2,
            "topology": ttnn.Topology.Linear,
        }

    def get_decode_all_gather_matmul_program_config(self):
        if not self.use_fused_all_gather_matmul:
            return None
        do_core_grid_size = (8, 1)
        do_per_core_n = self.dim // self.num_devices // ttnn.TILE_SIZE // (do_core_grid_size[0] * do_core_grid_size[1])
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=do_core_grid_size,
            in0_block_w=self.dim // ttnn.TILE_SIZE // (do_core_grid_size[0] * do_core_grid_size[1]),
            out_subblock_h=1,
            out_subblock_w=_get_out_subblock_w(do_per_core_n, out_subblock_h=1),
            per_core_M=self.tile_padded_batch_rows // ttnn.TILE_SIZE,
            per_core_N=do_per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def get_decode_all_gather_matmul_output_mem_config(self):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                _num_to_core_range_set(self.num_devices),
                [
                    self.tile_padded_batch_rows,
                    self.dim // self.num_devices,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        return (
            prefill_seq_len in self.trace_prefill_supported_seq_lens
            and prefill_seq_len <= self.max_prefill_chunk_size
            and prefill_seq_len <= self.max_seq_len
        )

    def get_trace_prefill_supported_seq_lens(self):
        return [seq_len for seq_len in (128, 1024) if seq_len <= self.capped_warmup_seq_len]

    def get_max_prefill_chunk_size(self):
        override = os.getenv("MAX_PREFILL_CHUNK_SIZE")
        if override is not None:
            return int(override) * 1024
        return {"N150": 4, "N300": 64, "T3K": 128}.get(self.device_name, 128) * 1024

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {"MLP": "feed_forward", "Attention": "attention", "TransformerBlock": "", "": ""}
        return layer_prefix + module_map[module_name]

    def weight_cache_path(self, dtype):
        if self.instruct:
            return (
                self.model_cache_path
                / {ttnn.bfloat16: "tensor_cache_instruct_bf16", ttnn.bfloat8_b: "tensor_cache_instruct_bfp8"}[dtype]
            )
        return self.model_cache_path / {ttnn.bfloat16: "tensor_cache_bf16", ttnn.bfloat8_b: "tensor_cache_bfp8"}[dtype]

    def get_model_config(self):
        return self.model_config

    def load_state_dict(self):
        if self._state_dict_loader is None:
            raise ValueError("No state_dict_loader was provided for this runtime config.")
        state_dict, self.fuse_qkv, self.fuse_mlp = self._state_dict_loader(self)
        return state_dict

    def create_tokenizer(self):
        if self.tokenizer is None:
            raise ValueError("No tokenizer was provided for this runtime config.")
        return self.tokenizer

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=True):
        if self._prompt_encoder is None:
            raise ValueError("No prompt_encoder was provided for this runtime config.")
        return self._prompt_encoder(prompt_text, system_prompt_text, instruct=instruct)

    def create_dram_sharded_mem_config(self, k, n, dram_grid=None):
        dram_cores = self.dram_grid_size.x
        padded_size = math.ceil(n / (ttnn.TILE_SIZE * dram_cores)) * (ttnn.TILE_SIZE * dram_cores)
        if dram_grid is None:
            dram_grid = self.dram_weight_grid
        shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    def find_grid(self, n):
        max_rows = 8 if ttnn.device.is_wormhole_b0(self.mesh_device) else 10
        max_cols = 8 if ttnn.device.is_wormhole_b0(self.mesh_device) else 12
        possible_cores = [k for k in range(1, max_rows * max_cols + 1) if n % k == 0]
        possible_cores.sort(key=lambda x: abs(x - 32))
        for cores in possible_cores:
            for rows in range(1, max_rows + 1):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= max_cols:
                        return rows, cols
        raise AssertionError(f"Cannot find grid for {n} tiles")

    def find_grid_k_n(self, k, n):
        possible_cores = [c for c in range(1, 65) if k % c == 0 and n % c == 0]
        possible_cores.sort(reverse=True)
        for cores in possible_cores:
            for rows in range(1, 9):
                if cores % rows == 0:
                    cols = cores // rows
                    if cols <= 8:
                        return rows, cols
        raise AssertionError(f"Cannot find grid for K={k}, N={n}")

    def dram_shard_core_grid_for_k(self, k):
        rows, cols = self.find_grid(k // ttnn.TILE_SIZE)
        return ttnn.CoreGrid(x=cols, y=rows)

    def dram_shard_core_grid_for_k_and_n(self, k, n):
        rows, cols = self.find_grid_k_n(k // ttnn.TILE_SIZE, n // ttnn.TILE_SIZE)
        return ttnn.CoreGrid(x=cols, y=rows)

    def find_largest_divisor(self, n, max_divisor=8):
        for i in range(max_divisor, 0, -1):
            if n % i == 0:
                return i
        return 1

    def dram_matmul_config(self, m, k, n, num_cores=None, fused_activation=None):
        if num_cores is None:
            num_cores = self.dram_shard_core_grid_for_k_and_n(k, n).num_cores
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.find_largest_divisor(k // (ttnn.TILE_SIZE * num_cores)),
            per_core_M=math.ceil(m / ttnn.TILE_SIZE),
            per_core_N=math.ceil(n / (ttnn.TILE_SIZE * num_cores)),
            fused_activation=fused_activation,
        )

    def create_sharded_norm_config(self, grid):
        block_w = self.dim // grid.num_cores // ttnn.TILE_SIZE
        subblock_w = 4
        while subblock_w > 0:
            if block_w % subblock_w == 0:
                break
            subblock_w -= 1
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid.x, grid.y],
            subblock_w=subblock_w,
            block_h=self.tile_padded_batch_rows // ttnn.TILE_SIZE,
            block_w=block_w,
            inplace=False,
        )

    def get_decode_residual_mem_config(self):
        residual_grid = self.dram_shard_core_grid_for_k(self.dim // self.num_devices)
        return ttnn.create_sharded_memory_config(
            (self.tile_padded_batch_rows, self.dim // residual_grid.num_cores // self.num_devices),
            residual_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def get_decode_norm_config(self, norm_type):
        if norm_type == "attn":
            grid = self.attn_input_grid
            mem = ttnn.create_sharded_memory_config(
                (self.tile_padded_batch_rows, self.dim // grid.num_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        elif norm_type == "ff":
            grid = self.mlp_core_grid
            mem = ttnn.create_sharded_memory_config(
                (self.tile_padded_batch_rows, self.dim // grid.num_cores),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        elif norm_type == "lm_head":
            grid = self.lm_head_core_grid
            mem = ttnn.create_sharded_memory_config(
                (self.tile_padded_batch_rows, _nearest_32(self.dim // grid.num_cores)),
                grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            raise ValueError(f"Invalid norm_type: {norm_type}")
        return {
            "sharded_program_config": self.create_sharded_norm_config(grid),
            "sharded_output_config": mem,
            "output_mem_config": None,
        }

    def get_decode_mlp_ff1_3_prg_config(self):
        return self.dram_matmul_config(
            self.tile_padded_batch_rows,
            self.dim,
            self.hidden_dim // self.cluster_shape[1],
            self.mlp_core_grid.num_cores,
        )

    def get_decode_mlp_ff2_prg_config(self):
        return self.dram_matmul_config(
            self.tile_padded_batch_rows,
            self.hidden_dim // self.cluster_shape[1],
            self.dim,
            self.mlp2_core_grid.num_cores,
        )

    def get_decode_mlp_binary_mult_mem_config(self):
        return ttnn.create_sharded_memory_config(
            (self.tile_padded_batch_rows, self.hidden_dim // self.cluster_shape[1] // self.mlp2_core_grid.num_cores),
            self.mlp2_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def get_decode_mlp_output_mem_config(self):
        return self.get_decode_residual_mem_config()

    def get_tensor_dtype(self, layer_num, tensor):
        return self.optimizations.get_tensor_dtype(layer_num, tensor)

    def get_math_fidelity(self, layer_num, op):
        return self.optimizations.get_math_fidelity(layer_num, op, self)

    def get_kv_cache_dtype(self, layer_num):
        return self.get_tensor_dtype(layer_num, "kv_cache")

    def ccl_topology(self):
        cluster_type = ttnn.cluster.get_cluster_type()
        if cluster_type in (
            ttnn.cluster.ClusterType.P300_X2,
            ttnn.cluster.ClusterType.P150_X4,
            ttnn.cluster.ClusterType.P150_X8,
        ):
            return ttnn.Topology.Ring
        if cluster_type in (
            ttnn.cluster.ClusterType.T3K,
            ttnn.cluster.ClusterType.GALAXY,
            ttnn.cluster.ClusterType.TG,
            ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        ):
            return ttnn.Topology.Ring if self.num_devices >= 8 else ttnn.Topology.Linear
        return ttnn.Topology.Linear if self.num_devices > 1 else None


def create_llama31_runtime_args(**kwargs):
    return Llama31RuntimeArgs(**kwargs)


# =============================================================================
# TransformerBlock1D
# =============================================================================


@dataclass
class TransformerBlock1DConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Attention1DConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtype: ttnn.DataType | None = None


class TransformerBlock1D(LightweightModule):
    """Single transformer block for 1D topologies (N150, N300, T3K).

    Happy path (takes pre-built sub-modules):
        block = TransformerBlock1D(attn_norm, attention, ff_norm, mlp)

    Power-user path (builds from config):
        block = TransformerBlock1D.from_config(config)
    """

    def __init__(
        self,
        attention_norm: RMSNorm1D,
        attention: Attention1D,
        ff_norm: RMSNorm1D,
        feed_forward: MLP1D,
        decode_residual_memcfg: ttnn.MemoryConfig | None = None,
        prefill_residual_memcfg: ttnn.MemoryConfig | None = None,
        activation_dtype: ttnn.DataType | None = None,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.ff_norm = ff_norm
        self.feed_forward = feed_forward
        self.decode_residual_memcfg = decode_residual_memcfg
        self.prefill_residual_memcfg = prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtype = activation_dtype

    @classmethod
    def from_config(cls, config: TransformerBlock1DConfig):
        return cls(
            attention_norm=RMSNorm1D.from_config(config.attention_norm_config),
            attention=Attention1D.from_config(config.attention_config),
            ff_norm=RMSNorm1D.from_config(config.ff_norm_config),
            feed_forward=MLP1D.from_config(config.mlp_config),
            decode_residual_memcfg=config.decode_residual_memcfg,
            prefill_residual_memcfg=config.prefill_residual_memcfg,
            activation_dtype=config.activation_dtype,
        )

    def decode_forward(self, x: ttnn.Tensor, current_pos, rot_mats, page_table) -> ttnn.Tensor:
        residual = x

        x = _all_gather_rmsnorm_tensor(
            self.attention_norm, x, memory_config=self.attention_norm.config.decode_memory_config
        )
        attn_in = self.attention_norm.decode_forward(x)
        attn_out = self.attention.decode_forward(attn_in, current_pos, rot_mats, page_table=page_table)
        attn_out = ttnn.to_memory_config(attn_out, self.decode_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.decode_residual_memcfg)
        residual = hidden_states

        hidden_states = _all_gather_rmsnorm_tensor(
            self.ff_norm, hidden_states, memory_config=self.ff_norm.config.decode_memory_config
        )
        hidden_states = self.ff_norm.decode_forward(hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.decode_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.decode_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def prefill_forward(
        self, x: ttnn.Tensor, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx
    ) -> ttnn.Tensor:
        residual = x

        attn_in = self.attention_norm.prefill_forward(x)
        attn_in = _all_gather_rmsnorm_tensor(self.attention_norm, attn_in)
        attn_out = self.attention.prefill_forward(
            attn_in,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        attn_out = ttnn.to_memory_config(attn_out, self.prefill_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.prefill_residual_memcfg)
        residual = hidden_states
        x.deallocate(True)

        hidden_states = self.ff_norm.prefill_forward(hidden_states)
        hidden_states = _all_gather_rmsnorm_tensor(self.ff_norm, hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.prefill_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.prefill_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)
        return self.decode_forward(x, current_pos, rot_mats, page_table)


# =============================================================================
# Llama3Transformer1D
# =============================================================================


@dataclass
class Llama31_8BPagedAttentionConfig:
    block_size: int
    max_num_blocks: int


@dataclass
class Llama3Transformer1DConfig:
    """Full TTTv2 model config."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    num_devices: int
    mesh_device: ttnn.MeshDevice

    # Sub-module configs
    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    block_configs: list[TransformerBlock1DConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None

    # Model-level memory configs
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None

    # Per-layer activation dtypes (from decoders_optimizations)
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)

    # CCL
    tt_ccl: TT_CCL | None = None

    # Weight cache path
    cache_path: "str | None" = None


def build_llama3_transformer_1d_config(
    *,
    mesh_device,
    args,
    state_dict,
    weight_cache_path,
    dtype=None,
    paged_attention_config=None,
    use_paged_kv_cache=None,
) -> Llama3Transformer1DConfig:
    """Translate Llama runtime args into explicit TTTv2 module configs."""
    if args.is_galaxy:
        raise ValueError("Llama3Transformer1D only supports 1D mesh topologies.")

    if use_paged_kv_cache is None:
        use_paged_kv_cache = paged_attention_config is not None

    num_devices = mesh_device.get_num_devices()
    tt_ccl_inst = get_tt_ccl(mesh_device) if num_devices > 1 else None
    model_config = args.get_model_config()
    model_config["DECODE_RESIDUAL_MEMCFG"] = args.get_decode_residual_mem_config()
    weight_cache_path = Path(weight_cache_path) if weight_cache_path else None
    embedding_cache_path = args.weight_cache_path(dtype or ttnn.bfloat8_b)

    def mesh_shard(dim: int) -> ttnn.MeshMapperConfig:
        return ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(dim)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        )

    def cache_path_for(base, *parts):
        if base is None or getattr(args, "dummy_weights", False):
            return None
        return Path(base).joinpath(*parts)

    def make_embedding_config() -> Embedding1DConfig:
        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_dir = cache_path_for(embedding_cache_path, "embedding")
        return Embedding1DConfig(
            weights=LazyWeight(
                source=torch_weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_dir, "tok_embeddings") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            weights_dtype=ttnn.bfloat16,
            weights_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )

    def make_rope_config() -> Rope1DConfig:
        return Rope1DConfig(
            cos_matrix=LazyWeight(source=args.rope_cos, device=mesh_device),
            sin_matrix=LazyWeight(source=args.rope_sin, device=mesh_device),
            max_batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            device=mesh_device,
            use_qk_fused=getattr(args, "use_qk_fused", False),
            datatype=ttnn.bfloat16,
        )

    def norm_weight_name(layer_num: int | None, weight_key: str, state_dict_prefix: str | None = None) -> str:
        if state_dict_prefix:
            return f"{state_dict_prefix}{weight_key}.weight"
        if layer_num is None:
            return f"{weight_key}.weight"
        return f"layers.{layer_num}.{weight_key}.weight"

    def make_norm_config(
        *,
        layer_num: int | None,
        weight_key: str,
        state_dict_prefix: str | None = None,
        sharded_program_config=None,
        sharded_output_config=None,
    ) -> RMSNorm1DConfig:
        weight_name = norm_weight_name(layer_num, weight_key, state_dict_prefix)
        torch_weight = (
            state_dict[weight_name]
            .unsqueeze(0)
            .view(1, 1, args.dim)
            .reshape([1, 1, args.dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )
        if args.rms_norm_add_unit_offset:
            torch_weight = torch_weight + 1.0
        return RMSNorm1DConfig(
            weight=LazyWeight(
                source=torch_weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(weight_cache_path, weight_name) if weight_cache_path else None,
                mesh_mapper_config=(
                    ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementReplicate()],
                        mesh_shape_override=ttnn.MeshShape([num_devices]),
                    )
                    if num_devices > 1
                    else None
                ),
            ),
            eps=args.norm_eps,
            add_unit_offset=False,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            max_batch_size=args.max_batch_size,
            prefill_distributed=num_devices > 1 and args.dim >= 4096,
            decode_program_config=sharded_program_config,
            decode_memory_config=sharded_output_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )

    def make_attention_config(layer_num: int, transformation_mats: dict[str, ttnn.Tensor]) -> Attention1DConfig:
        layer_name = args.get_state_dict_prefix("Attention", layer_num)
        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        wqkv_dtype = args.get_tensor_dtype(layer_num, "wqkv")
        wo_dtype = args.get_tensor_dtype(layer_num, "wo")
        kv_cache_dtype = args.get_tensor_dtype(layer_num, "kv_cache")
        activation_dtype = args.get_tensor_dtype(layer_num, "activation")

        qkv_list = []
        for device_idx in range(num_devices):
            wq = torch.transpose(torch.chunk(state_dict[f"{wq_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            wk = torch.transpose(torch.chunk(state_dict[f"{wk_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            wv = torch.transpose(torch.chunk(state_dict[f"{wv_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        use_fused_all_gather_matmul = getattr(args, "use_fused_all_gather_matmul", False)
        wqkv = LazyWeight(
            source=qkv_cat,
            dtype=wqkv_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=args.create_dram_sharded_mem_config(args.dim, args.qkv_size // num_devices),
            mesh_mapper_config=mesh_shard(-1),
            cache_dir_weight_name=(weight_cache_path / layer_name, "wqkv_sharded") if weight_cache_path else None,
        )
        wo = LazyWeight(
            source=state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0),
            dtype=wo_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_fused_all_gather_matmul
                else args.create_dram_sharded_mem_config((args.n_heads * args.head_dim) // num_devices, args.dim)
            ),
            mesh_mapper_config=mesh_shard(-1 if use_fused_all_gather_matmul else -2),
            cache_dir_weight_name=(
                (weight_cache_path / layer_name, "wo_width_sharded" if use_fused_all_gather_matmul else "wo")
                if weight_cache_path
                else None
            ),
        )

        qk_norm_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def make_qk_norm_config(name: str) -> RMSNorm1DConfig | None:
            weight_name = f"{name}.weight"
            if weight_name not in state_dict:
                return None
            return RMSNorm1DConfig(
                weight=LazyWeight(
                    source=state_dict[weight_name].reshape(1, 1, -1, TILE_SIZE),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_dir_weight_name=(
                        (weight_cache_path / layer_name, name.rsplit(".", 1)[-1]) if weight_cache_path else None
                    ),
                ),
                mesh_device=mesh_device,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
                decode_in_sharded=False,
                decode_out_sharded=False,
                prefill_distributed=False,
                compute_kernel_config=qk_norm_compute_kernel,
            )

        wqkv_bias = None
        if f"{wq_str}.bias" in state_dict:
            wqkv_bias = LazyWeight(
                source=torch.concat(
                    [
                        torch.concat(
                            [
                                torch.chunk(state_dict[f"{wq_str}.bias"], num_devices)[device_idx],
                                torch.chunk(state_dict[f"{wk_str}.bias"], num_devices)[device_idx],
                                torch.chunk(state_dict[f"{wv_str}.bias"], num_devices)[device_idx],
                            ],
                            dim=-1,
                        )
                        for device_idx in range(num_devices)
                    ],
                    dim=-1,
                )
            )

        scale = args.query_pre_attn_scalar**-0.5 if args.query_pre_attn_scalar is not None else args.head_dim**-0.5
        attn_agmm_cfg = model_config.get("ATTN_AGMM_CONFIG", {})
        return Attention1DConfig(
            wqkv=wqkv,
            wo=wo,
            q_norm_config=make_qk_norm_config(q_norm_str),
            k_norm_config=make_qk_norm_config(k_norm_str),
            wqkv_bias=wqkv_bias,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            topology=args.ccl_topology(),
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            qkv_size=args.qkv_size,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            scale=scale,
            sliding_window=args.sliding_window if hasattr(args, "sliding_window") else None,
            use_qk_fused=getattr(args, "use_qk_fused", False),
            use_vllm_paged_kv_cache=use_paged_kv_cache,
            paged_attention_config=paged_attention_config,
            kv_cache_dtype=kv_cache_dtype,
            min_kv_prefill_shard_seqlen=args.min_kv_prefill_shard_seqlen,
            wqkv_dtype=wqkv_dtype,
            wo_dtype=wo_dtype,
            activation_dtype=activation_dtype,
            decode_xqkv_prg_config=model_config.get("XQKV_DECODE_PROGCFG"),
            decode_sdpa_prg_config=model_config.get("SDPA_DECODE_PROGCFG"),
            decode_attn_output_prg_config=model_config.get("ATTN_OUTPUT_PROGCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
            decode_create_qkv_head_memcfg=model_config.get("CREATE_QKV_DECODE_SHARD"),
            decode_scores_memcfg=model_config.get("SCORES_BATCHED_MM_OUTPUT_MEMCFG"),
            prefill_xqkv_prg_config=model_config.get("XQKV_PREFILL_PROGCFG"),
            prefill_sdpa_prg_config=model_config.get("SDPA_PROGCFG"),
            prefill_wo_prg_config=model_config.get("WO_PREFILL_PROGCFG"),
            prefill_kv_memcfg=model_config.get("KV_PREFILL_MEM_CFG"),
            use_fused_all_gather_matmul=use_fused_all_gather_matmul,
            decode_all_gather_matmul_prg_config=model_config.get("ATTN_ALL_GATHER_MATMUL_PROGCFG"),
            decode_all_gather_matmul_memcfg=model_config.get("ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"),
            decode_agmm_num_links=attn_agmm_cfg.get("num_links", 1),
            decode_agmm_chunks_per_sync=attn_agmm_cfg.get("chunks_per_sync", 10),
            decode_agmm_num_workers_per_link=attn_agmm_cfg.get("num_workers_per_link", 2),
            li_qkv_decode_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_qkv_decode"),
            sdpa_decode_compute_kernel_cfg=args.get_math_fidelity(layer_num, "sdpa_decode"),
            li_o_decode_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_o_decode"),
            li_qkv_prefill_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_qkv_prefill"),
            sdpa_prefill_compute_kernel_cfg=args.get_math_fidelity(layer_num, "sdpa_prefill"),
            li_o_prefill_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_o_prefill"),
            transformation_mat_decode=transformation_mats.get("decode"),
            transformation_mat_prefill=transformation_mats.get("prefill"),
        )

    def make_mlp_config(layer_num: int) -> MLP1DConfig:
        state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)
        ff1_3_dtype = args.get_tensor_dtype(layer_num, "ff1_ff3")
        ff2_dtype = args.get_tensor_dtype(layer_num, "ff2")
        activation_dtype = args.get_tensor_dtype(layer_num, "activation")
        mlp_rs_cfg = model_config.get("MLP_RS_CONFIG", {})

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        w1_w3_mem_config = _create_dram_sharded_mem_config(
            k=args.dim,
            n=args.hidden_dim // num_devices,
            dram_grid=dram_grid,
            tile_size=TILE_SIZE,
            dram_cores=dram_size.x,
        )
        w2_mem_config = _create_dram_sharded_mem_config(
            k=args.hidden_dim // num_devices,
            n=args.dim,
            dram_grid=dram_grid,
            tile_size=TILE_SIZE,
            dram_cores=dram_size.x,
        )
        cache_dir = cache_path_for(weight_cache_path, state_dict_prefix)

        def make_weight_source(name: str, shard_dim: int):
            tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
            return pad_dim_to_size(tensor, dim=shard_dim, size=args.hidden_dim)

        return MLP1DConfig(
            w1=LazyWeight(
                source=make_weight_source("w1", -1),
                dtype=ff1_3_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w1_w3_mem_config,
                cache_dir_weight_name=(cache_dir, "w1_sharded") if cache_dir else None,
            ),
            w2=LazyWeight(
                source=make_weight_source("w2", -2),
                dtype=ff2_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-2),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w2_mem_config,
                cache_dir_weight_name=(cache_dir, "w2_sharded") if cache_dir else None,
            ),
            w3=LazyWeight(
                source=make_weight_source("w3", -1),
                dtype=ff1_3_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w1_w3_mem_config,
                cache_dir_weight_name=(cache_dir, "w3_sharded") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            max_batch_size=args.max_batch_size,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            topology=args.ccl_topology(),
            decode_rs_memory_config=mlp_rs_cfg.get("rs_memory_config", ttnn.L1_MEMORY_CONFIG),
            decode_rs_chunks_per_sync=mlp_rs_cfg.get("chunks_per_sync", 1),
            decode_rs_num_workers_per_link=mlp_rs_cfg.get("num_workers_per_link", 1),
            decode_w1_w3_prg_config=args.get_decode_mlp_ff1_3_prg_config(),
            decode_w2_prg_config=args.get_decode_mlp_ff2_prg_config(),
            decode_mlp2_input_memcfg=args.get_decode_mlp_binary_mult_mem_config(),
            decode_residual_memcfg=args.get_decode_mlp_output_mem_config(),
            w1_w3_dtype=ff1_3_dtype,
            w2_dtype=ff2_dtype,
            activation_dtype=activation_dtype,
            ff1_3_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff1_ff3"),
            ff2_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff2"),
            decode_ff1_3_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff1_ff3"),
            decode_ff2_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff2"),
            decode_spill_w1_to_dram_before_w3=False,
            prefill_len_cutoff=args.prefill_len_cutoff,
        )

    def make_lm_head_config() -> LMHead1DConfig:
        vocab_size = args.vocab_size
        padded_vocab_size = math.ceil(vocab_size / TILE_SIZE) * TILE_SIZE
        size_per_device = padded_vocab_size // num_devices
        num_splits = math.ceil(size_per_device / args.max_columns_per_device_lm_head)
        split_sizes = [min(size_per_device, args.max_columns_per_device_lm_head)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        state_dict_prefix = args.get_state_dict_prefix("", None)
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)
        if vocab_size < padded_vocab_size:
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(
                        torch_output_weights.shape[0],
                        padded_vocab_size - vocab_size,
                        dtype=torch_output_weights.dtype,
                    ),
                ],
                dim=-1,
            )

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        cache_dir = cache_path_for(weight_cache_path, "lm_head")
        output_weights = []
        weights_memcfgs = []
        for split_idx, split_size in enumerate(split_sizes):
            device_splits = []
            for device_idx in range(num_devices):
                start = device_idx * size_per_device + sum(split_sizes[:split_idx])
                end = start + split_size
                device_splits.append(torch_output_weights[:, start:end])
            combined_split = torch.cat(device_splits, dim=-1)
            mem_cfg = _create_dram_sharded_mem_config(
                k=args.dim,
                n=math.ceil(combined_split.shape[-1] / num_devices),
                dram_grid=dram_grid,
                tile_size=TILE_SIZE,
                dram_cores=dram_size.x,
            )
            weights_memcfgs.append(mem_cfg)
            output_weights.append(
                LazyWeight(
                    source=combined_split,
                    dtype=dtype if dtype is not None else ttnn.bfloat8_b,
                    device=mesh_device,
                    mesh_mapper_config=mesh_shard(-1),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem_cfg,
                    cache_dir_weight_name=(cache_dir, f"output_split_{split_idx}_{combined_split.shape[-1]}")
                    if cache_dir
                    else None,
                )
            )

        tile_padded_batch_rows = TILE_SIZE * math.ceil(args.max_batch_size / TILE_SIZE)
        lm_head_core_grid = args.lm_head_core_grid
        input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, math.ceil((args.dim // lm_head_core_grid.num_cores) / TILE_SIZE) * TILE_SIZE),
            lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return LMHead1DConfig(
            output_weights=output_weights,
            mesh_device=mesh_device,
            dim=args.dim,
            max_batch_size=args.max_batch_size,
            program_configs=[
                args.dram_matmul_config(tile_padded_batch_rows, args.dim, split_size, lm_head_core_grid.num_cores)
                for split_size in split_sizes
            ],
            compute_kernel_config=_compute_kernel_config_hifi2(),
            lm_head_dtype=getattr(args, "lm_head_dtype", ttnn.bfloat8_b),
            output_memcfg=ttnn.L1_MEMORY_CONFIG,
            input_memcfg=input_memcfg,
            weights_memcfgs=weights_memcfgs,
        )

    def make_sampling_config() -> Sampling1DConfig | None:
        sampling_splits = num_devices if list(mesh_device.shape) != [1, 1] else 2
        if args.vocab_size // sampling_splits > 64 * 1024:
            return None

        num_gather_links = 1
        if "GALAXY_NUM_LINKS" in model_config:
            max_links = model_config["GALAXY_NUM_LINKS"]
            max_top_k = getattr(args, "max_top_k", 32)
            num_gather_links = min(max_top_k // 32, max_links) if max_top_k // 32 <= max_links else max_links

        ag_cfg = model_config.get("SAMPLING_AG_CONFIG", {})
        return Sampling1DConfig(
            vocab_size=getattr(args, "padded_vocab_size", args.vocab_size),
            valid_vocab_size=args.vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            max_batch_size=getattr(args, "max_batch_size", 32),
            max_top_k=getattr(args, "max_top_k", 32),
            sub_core_grids=getattr(args, "sub_core_grids", None),
            sub_core_grid_topk=getattr(args, "sub_core_grid_topk", None),
            start_core=getattr(args, "start_core", ttnn.CoreCoord(0, 0)),
            num_gather_links=num_gather_links,
            sampling_memory_config=model_config.get("DECODE_SAMPLING_INPUT_MEMCFG", ttnn.DRAM_MEMORY_CONFIG),
            allow_force_argmax=ag_cfg.get("allow_force_argmax", False),
            num_argmax_gather_links=ag_cfg.get("num_links", num_gather_links),
            ag_topology=ag_cfg.get("topology", ttnn.Topology.Linear),
            argmax_chunks_per_sync=ag_cfg.get("chunks_per_sync", 10),
            argmax_num_workers_per_link=1,
            pad_to_power_of_2=getattr(args, "pad_logits_to_power_of_2", False),
        )

    rope_config = make_rope_config()
    trans_mats_dict = RotarySetup1D.from_config(rope_config).get_both_trans_mats()
    attn_norm_cfg = args.get_decode_norm_config("attn")
    ff_norm_cfg = args.get_decode_norm_config("ff")
    lm_head_norm_cfg = args.get_decode_norm_config("lm_head")
    activation_dtypes = [args.get_tensor_dtype(i, "activation") for i in range(args.n_layers)]

    return Llama3Transformer1DConfig(
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        num_devices=num_devices,
        mesh_device=mesh_device,
        embedding_config=make_embedding_config(),
        rope_config=rope_config,
        block_configs=[
            TransformerBlock1DConfig(
                attention_norm_config=make_norm_config(
                    layer_num=i,
                    weight_key="attention_norm",
                    sharded_program_config=attn_norm_cfg.get("sharded_program_config"),
                    sharded_output_config=attn_norm_cfg.get("sharded_output_config"),
                ),
                attention_config=make_attention_config(i, trans_mats_dict),
                ff_norm_config=make_norm_config(
                    layer_num=i,
                    weight_key="ffn_norm",
                    sharded_program_config=ff_norm_cfg.get("sharded_program_config"),
                    sharded_output_config=ff_norm_cfg.get("sharded_output_config"),
                ),
                mlp_config=make_mlp_config(i),
                decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
                prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                activation_dtype=activation_dtypes[i],
            )
            for i in range(args.n_layers)
        ],
        norm_config=make_norm_config(
            layer_num=None,
            weight_key="norm",
            state_dict_prefix=args.get_state_dict_prefix("", None),
            sharded_program_config=lm_head_norm_cfg.get("sharded_program_config"),
            sharded_output_config=lm_head_norm_cfg.get("sharded_output_config"),
        ),
        lm_head_config=make_lm_head_config(),
        sampling_config=make_sampling_config(),
        decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtypes=activation_dtypes,
        tt_ccl=tt_ccl_inst,
        cache_path=str(weight_cache_path) if weight_cache_path else None,
    )


class Llama3Transformer1D(LightweightModule):
    """TTTv2 Llama 3.1-8B Transformer.

    Constructor takes a config and builds everything internally:
        model = Llama3Transformer1D(config)

    Public sub-modules (accessible by executor for trace support):
        - embedding: Embedding1D
        - rope_setup: RotarySetup1D
        - layers: list[TransformerBlock1D]
        - norm: RMSNorm1D (final)
        - lm_head: LMHead1D
        - sampling: Sampling1D | None

    Forward methods take pre-embedded tensors. The executor handles
    embedding, input preparation, and output processing.
    """

    def __init__(self, config: Llama3Transformer1DConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config

        tt_ccl_inst = config.tt_ccl
        if tt_ccl_inst is None and config.num_devices > 1:
            tt_ccl_inst = get_tt_ccl(config.mesh_device)

        self.embedding = Embedding1D.from_config(config.embedding_config)
        self.rope_setup = RotarySetup1D.from_config(config.rope_config)

        self.layers = [
            TransformerBlock1D.from_config(config.block_configs[i])
            for i in tqdm(range(config.n_layers), desc="Building layers")
        ]

        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)

        self.sampling = None
        if config.sampling_config is not None:
            self.sampling = Sampling1D.from_config(config.sampling_config)
        self.supports_on_device_sampling = self.sampling is not None

        self.mesh_device = config.mesh_device
        self.tt_ccl = tt_ccl_inst
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers

    # =========================================================================
    # KV Cache binding
    # =========================================================================

    def set_kv_cache(self, kv_cache: list):
        """Bind static kv_cache pool via each attention layer's config.

        Must be called before the first forward (before load_device_weights runs).
        The kv_cache is resolved from config during load_device_weights(), just
        like all other weights.
        """
        assert len(kv_cache) == len(
            self.layers
        ), f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers"
        for i, layer in enumerate(self.layers):
            layer.attention.config.kv_cache = tuple(kv_cache[i])

    # =========================================================================
    # Forward methods — take pre-embedded tensors
    # =========================================================================

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Decode forward. x_embed is already embedded, unsqueezed, and in decode_residual_memcfg."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            x = ttnn.to_memory_config(x, self.decode_residual_memcfg, self.activation_dtypes[i])

            x = layer.decode_forward(x, current_pos, rot_mats, page_table)

        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        x = self.lm_head.forward(x)
        return x

    def prefill_forward(
        self,
        x_embed: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        """Prefill forward. x_embed is already embedded and unsqueezed to 4D."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None and x.dtype != activation_dtype:
                old = x
                x = ttnn.typecast(x, activation_dtype)
                ttnn.deallocate(old)

            x = layer.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)

        if get_last_token == -1:
            return x

        get_last_token_floor = (get_last_token // 32) * 32
        old = x
        x = ttnn.slice(x, (0, 0, get_last_token_floor, 0), (1, 1, get_last_token_floor + 32, x.shape[-1]))
        ttnn.deallocate(old)

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def post_process_prefill_output(self, hidden_states: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
        """Convert traced prefill hidden states into logits for the last token block."""
        get_last_token_floor = (last_token_idx // 32) * 32
        x = ttnn.slice(
            hidden_states,
            (0, 0, get_last_token_floor, 0),
            (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
        )

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        """Dispatcher for backward compatibility. Llama 3.1-8B has no local rope."""
        rot_mats = rot_mats_global
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
            )
        return self.decode_forward(
            x,
            current_pos,
            rot_mats,
            page_table=page_table,
        )

    # =========================================================================
    # Embedding + output processing helpers (called by executor)
    # =========================================================================

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens and prepare for decode. Returns tensor in decode_residual_memcfg."""
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        x = ttnn.to_memory_config(x, self.decode_residual_memcfg)
        return x

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens for prefill. Returns tensor in DRAM interleaved."""
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return x

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather logits across devices and untilize for host argmax."""
        if self.num_devices > 1:
            logits = ttnn.experimental.all_gather_async(
                logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                memory_config=logits.memory_config(),
                topology=default_topology(self.mesh_device),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor):
        """Increment decode position counters on device."""
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)


# =============================================================================
# RMSNorm gather helpers
# =============================================================================


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=tt_ccl.get_num_links(),
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


# =============================================================================
# EagerLlamaExecutor — thin wrapper
# =============================================================================


class EagerLlamaExecutor:
    """Thin wrapper: passes Llama model to EagerLLMExecutor.

    All actual logic lives in the engine. This class exists to:
    1. Provide a model-specific type for type hints
    2. Preserve the existing API for demos and tests
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        # Attach model_args to model so engine can access it via model.model_args
        if model_args is not None:
            model.model_args = model_args
        self._engine = EagerLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    # =========================================================================
    # KV Cache — delegate to engine
    # =========================================================================

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def _assert_kv_cache_identity(self, kv_cache):
        return self._engine._assert_kv_cache_identity(kv_cache)

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_host(tokens, current_pos, page_table)

    def prepare_decode_inputs_device(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_device(tokens, current_pos, page_table)

    # =========================================================================
    # Compile — delegate to engine
    # =========================================================================

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        sampling_params=None,
    ):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    # =========================================================================
    # Forward — delegate to engine
    # =========================================================================

    def prefill_forward(
        self,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def _prefill_single_user(self, tokens, page_table, user_id, last_token_idx, num_cached_tokens=0):
        return self._engine._prefill_single_user(tokens, page_table, user_id, last_token_idx, num_cached_tokens)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
    ):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    # =========================================================================
    # Cleanup — delegate to engine
    # =========================================================================

    def cleanup(self):
        return self._engine.cleanup()


# =============================================================================
# TracedLlamaExecutor — thin wrapper
# =============================================================================


class TracedLlamaExecutor:
    """Thin wrapper: passes Llama model to TracedLLMExecutor.

    All actual logic lives in the engine. This class exists to:
    1. Provide a model-specific type for type hints
    2. Preserve the existing API for demos and tests
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        # Attach model_args to model so engine can access it via model.model_args
        if model_args is not None:
            model.model_args = model_args
        self._engine = TracedLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    # Expose internal state for tests/debugging
    @property
    def trace_id_prefill(self):
        return self._engine.trace_id_prefill

    @property
    def trace_ids_decode(self):
        return self._engine.trace_ids_decode

    @property
    def already_warmed_up_prefill(self):
        return self._engine.already_warmed_up_prefill

    # =========================================================================
    # KV Cache — delegate to engine
    # =========================================================================

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    # =========================================================================
    # Warmup — delegate to engine
    # =========================================================================

    def warmup_model_prefill(self, seq_lens, make_tokens, make_page_table):
        return self._engine.warmup_model_prefill(seq_lens, make_tokens, make_page_table)

    # =========================================================================
    # Compile — delegate to engine
    # =========================================================================

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        sampling_params=None,
    ):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    # =========================================================================
    # Forward — delegate to engine
    # =========================================================================

    def prefill_forward(
        self,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
    ):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    # =========================================================================
    # Cleanup — delegate to engine
    # =========================================================================

    def cleanup(self):
        return self._engine.cleanup()


# =============================================================================
# Executor Validation Traversal
# =============================================================================


def _iter_llama_executor_named_modules(model):
    """Yield named submodules that declare executor input contracts."""
    if not hasattr(model, "layers"):
        return

    for i, layer in enumerate(model.layers):
        for suffix, submodule in [
            ("attn_norm", getattr(layer, "attention_norm", None)),
            ("attention", getattr(layer, "attention", None)),
            ("ff_norm", getattr(layer, "ff_norm", None)),
            ("mlp", getattr(layer, "mlp", None)),
        ]:
            if submodule is not None:
                yield f"layer[{i}].{suffix}", submodule

    if hasattr(model, "norm"):
        yield "final_norm", model.norm
    if hasattr(model, "lm_head"):
        yield "lm_head", model.lm_head
