# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTTv2-owned runtime/config object for Llama-3.1-8B 1D demos."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.models.llama3_8b.precision import Llama31DecoderPrecision
from models.common.models.llama3_8b.rope import rope_scaling_model_factory


def _nearest_multiple(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def _should_pad_sampling_logits_to_power_of_2(padded_vocab_size: int, sampling_splits: int) -> bool:
    if sampling_splits < 1:
        return False
    per_device_vocab = padded_vocab_size // sampling_splits
    return per_device_vocab > 0 and (per_device_vocab & (per_device_vocab - 1)) != 0


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


def _replace_keys(state_dict, replacements):
    output = {}
    for key, value in state_dict.items():
        new_key = key
        for pattern, repl in replacements:
            new_key = re.sub(pattern, repl, new_key)
        output[new_key] = value
    return output


def _standardize_hf_keys(state_dict):
    key_meta = "lm_head.weight"
    key_hf = "model.embed_tokens.weight"
    if key_meta not in state_dict and key_hf in state_dict:
        state_dict[key_meta] = state_dict[key_hf]
        del state_dict[key_hf]
    return state_dict


def _split_hf_keys(loaded_weights, n_heads=None, n_kv_heads=None):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "qkv_proj" in key:
            q_key = key.replace("qkv_proj", "q_proj")
            k_key = key.replace("qkv_proj", "k_proj")
            v_key = key.replace("qkv_proj", "v_proj")
            if n_heads is not None and n_kv_heads is not None and n_heads != n_kv_heads:
                head_dim = tensor.shape[0] // (n_heads + 2 * n_kv_heads)
                q_size = n_heads * head_dim
                kv_size = n_kv_heads * head_dim
                q_tensor = tensor[:q_size]
                k_tensor = tensor[q_size : q_size + kv_size]
                v_tensor = tensor[q_size + kv_size : q_size + 2 * kv_size]
            else:
                q_tensor, k_tensor, v_tensor = torch.split(tensor, tensor.shape[0] // 3, dim=0)
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        elif "gate_up_proj" in key:
            gate_key = key.replace("gate_up_proj", "gate_proj")
            up_key = key.replace("gate_up_proj", "up_proj")
            gate_tensor, up_tensor = torch.split(tensor, tensor.shape[0] // 2, dim=0)
            converted_weights[gate_key] = gate_tensor
            converted_weights[up_key] = up_tensor
        else:
            converted_weights[key] = tensor
    return converted_weights


def _reverse_permute(tensor, n_heads, dim1, dim2):
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _reverse_permute_1d(tensor):
    dim = tensor.shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    return torch.stack((reals, imags), dim=-1).flatten(start_dim=len(tensor.shape) - 1)


def _convert_hf_qkv_to_meta_format(loaded_weights, head_dim):
    converted_weights = {}
    for key, tensor in loaded_weights.items():
        if "q_proj.weight" in key or "k_proj.weight" in key:
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = _reverse_permute(tensor, n_heads, tensor.shape[0], tensor.shape[1])
        elif "q_proj.bias" in key or "k_proj.bias" in key:
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = _reverse_permute(tensor, n_heads, tensor.shape[0], 1).squeeze(-1)
        elif "q_norm.weight" in key or "k_norm.weight" in key:
            converted_weights[key] = _reverse_permute_1d(tensor)
        else:
            converted_weights[key] = tensor
    return converted_weights


def _map_hf_to_meta_keys(loaded_weights):
    replacements = [
        ("^emb.weight", "weight"),
        ("model.", ""),
        ("embed_tokens", "tok_embeddings"),
        ("lm_head", "output"),
        ("input_layernorm", "attention_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("self_attn", "attention"),
        ("mlp", "feed_forward"),
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
        ("q_norm", "q_norm"),
        ("k_norm", "k_norm"),
    ]
    return _replace_keys(loaded_weights, replacements)


def _convert_hf_to_meta(state_dict, head_dim, n_heads, n_kv_heads):
    state_dict = _split_hf_keys(state_dict, n_heads, n_kv_heads)
    state_dict = _convert_hf_qkv_to_meta_format(state_dict, head_dim)
    return _map_hf_to_meta_keys(state_dict)


def _chat_template_ids(encoded):
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)) and len(encoded) == 1 and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return list(encoded)


def _encode_prompt_hf(tokenizer, prompt_text, system_prompt_text=None):
    chat = []
    if isinstance(prompt_text, str):
        if system_prompt_text:
            chat.append({"role": "system", "content": system_prompt_text})
        if prompt_text:
            chat.append({"role": "user", "content": prompt_text})
        encoded = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    else:
        encoded = tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)
    return _chat_template_ids(encoded)


class Llama31RuntimeArgs:
    """Narrow replacement for TTTv1 ModelArgs for the Llama-3.1-8B 1D path."""

    def __init__(
        self,
        mesh_device,
        *,
        instruct: bool,
        max_batch_size: int,
        max_seq_len: int,
        optimizations="performance",
        n_layers: int | None = None,
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
        self.cache_hf_flag = False
        self.cached_hf_model = None
        self.rms_norm_add_unit_offset = False
        self.embed_scale = None
        self.use_hf_rope = False
        self.trust_remote_code_hf = False
        self.prefetcher = None
        self.prefill_len_cutoff = 512 if ttnn.device.is_blackhole(mesh_device) else 1024
        self.instruct = instruct

        hf_model = os.getenv("HF_MODEL")
        if not hf_model:
            raise ValueError("Please set HF_MODEL to a HuggingFace name e.g. meta-llama/Llama-3.1-8B-Instruct")
        self.CKPT_DIR = hf_model
        self.TOKENIZER_PATH = hf_model
        self.CACHE_PATH = os.getenv("TT_CACHE_PATH")
        if self.CACHE_PATH:
            self.CACHE_PATH = os.path.join(self.CACHE_PATH, self.device_name)
        else:
            self.CACHE_PATH = os.path.join("model_cache", hf_model, self.device_name)
        self.model_name = hf_model.strip("/").split("/")[-1]
        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)
        self.consolidated_weights_path = self.CKPT_DIR + "/consolidated.00.pth"
        self.tokenizer_path = self.TOKENIZER_PATH + "/tokenizer.model"

        self._set_hf_params()
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
        self.tokenizer = self.create_tokenizer()
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

    def _set_hf_params(self):
        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(
            self.CKPT_DIR,
            trust_remote_code=self.trust_remote_code_hf,
            local_files_only=os.getenv("CI") == "true",
        )
        config = self.hf_config.to_dict()
        text_config = config.get("text_config", config)
        self.dim = text_config.get("dim", text_config.get("hidden_size"))
        self.n_heads = text_config.get("n_heads", text_config.get("num_attention_heads"))
        self.n_kv_heads = text_config.get("n_kv_heads", text_config.get("num_key_value_heads"))
        self.n_layers = text_config.get("n_layers", text_config.get("num_hidden_layers"))
        self.full_model_n_layers = self.n_layers
        self.norm_eps = text_config.get("norm_eps", text_config.get("rms_norm_eps"))
        self.vocab_size = text_config["vocab_size"]
        self.padded_vocab_size = _nearest_multiple(self.vocab_size, ttnn.TILE_SIZE * self.num_devices)
        self.head_dim = text_config.get("head_dim", self.dim // self.n_heads) or self.dim // self.n_heads
        self.max_context_len = text_config.get("max_position_embeddings")
        self.hidden_dim = text_config["intermediate_size"]
        if "_name_or_path" in config and config["_name_or_path"]:
            self.model_name = os.path.basename(os.path.normpath(config["_name_or_path"]))
        sampling_splits = self.num_devices if self.cluster_shape != [1, 1] else 2
        self.pad_logits_to_power_of_2 = self.cluster_shape != [1, 1] and _should_pad_sampling_logits_to_power_of_2(
            self.padded_vocab_size, sampling_splits
        )
        self.unpadded_hidden_dim = self.hidden_dim
        self.layer_types = text_config.get("layer_types", None)
        self.sliding_window = text_config.get("sliding_window", None)
        rope_parameters = text_config.get("rope_parameters") or {}
        self.rope_theta = text_config.get("rope_theta") or rope_parameters.get("rope_theta")
        self.rope_theta_local = text_config.get("rope_local_base_freq")
        self.use_sliding_window = text_config.get("use_sliding_window", None)
        rope_scaling_params = text_config.get("rope_scaling")
        if not rope_scaling_params and rope_parameters.get("rope_type") not in (None, "default"):
            rope_scaling_params = rope_parameters
        self.rope_scaling_params = rope_scaling_params
        self.original_max_context_len = text_config.get("original_max_position_embeddings", None)
        self.rope_scaling = (
            rope_scaling_model_factory(rope_scaling_params, self.original_max_context_len)
            if rope_scaling_params
            else None
        )
        self.query_pre_attn_scalar = text_config.get("query_pre_attn_scalar", None)
        self.mlp_activation_type = ttnn.UnaryOpType.SILU
        self.is_multimodal = False
        self.state_dict_text_prefix = ""
        self.state_dict_vision_prefix = "visual."

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
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self.CKPT_DIR,
            torch_dtype="auto",
            trust_remote_code=self.trust_remote_code_hf,
            local_files_only=os.getenv("CI") == "true",
        )
        state_dict = model.state_dict()
        self.fuse_qkv = any("qkv" in layer_name for layer_name in state_dict)
        self.fuse_mlp = any("gate_up" in layer_name for layer_name in state_dict)
        state_dict = _standardize_hf_keys(state_dict)
        state_dict = _convert_hf_to_meta(state_dict, self.head_dim, self.n_heads, self.n_kv_heads)
        for key in list(state_dict.keys()):
            if "layers." in key:
                layer_num = int(key.split("layers.")[1].split(".")[0])
                if layer_num >= self.n_layers:
                    state_dict.pop(key)
        return state_dict

    def create_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.TOKENIZER_PATH,
            local_files_only=os.getenv("CI") == "true",
            trust_remote_code=self.trust_remote_code_hf,
        )
        if not hasattr(tokenizer, "stop_tokens") or tokenizer.stop_tokens is None:
            tokenizer.stop_tokens = [tokenizer.eos_token_id]
        return tokenizer

    def encode_prompt(self, prompt_text, system_prompt_text=None, instruct=True):
        if instruct:
            try:
                return _encode_prompt_hf(self.tokenizer, prompt_text, system_prompt_text)
            except ValueError as exc:
                logger.warning(f"Failed to encode chat prompt, falling back to base encoding: {exc}")
        return self.tokenizer.encode(prompt_text, add_special_tokens=False)

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
