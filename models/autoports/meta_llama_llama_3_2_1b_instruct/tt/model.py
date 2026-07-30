# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full TTNN autoregressive model for meta-llama/Llama-3.2-1B-Instruct.

The decoder stack is the optimized 1x8 multichip decoder from this autoport.
The stack boundary remains the selected replicated full-hidden residual stream:
terminal work is localized to final RMSNorm, a vocab-sharded tied LM head, and
the generator-owned split sampler.
"""

from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
import ttnn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.functional_decoder import (
    MODEL_ID,
    _state_tensor,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    MultichipDecoder,
    set_multichip_ccl_dtype_policy,
)
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoderPrecisionPolicy,
    _compute_kernel_config_hifi2_fp16,
    _dram_matmul_config,
    precision_policy_from_config,
)
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.common.modules.tt_ccl import get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_rot_transformation_mat
from models.common.utility_functions import nearest_32


DEFAULT_PAGE_BLOCK_SIZE = 64
# Attention1D caps the allocated KV-token budget at 128K tokens.
# With the standard readiness max_batch_size=32, max_seq_len must be <=4096.
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_MAX_TOP_K = 32
LM_HEAD_MAX_COLUMNS_PER_DEVICE = 8192
PRECISION_CONFIG_ENV = "MD_LLAMA32_PRECISION_CONFIG"
DEFAULT_SELECTED_PRECISION_CONFIG_RELPATH = Path("doc/datatype_sweep/selected_precision_config.json")


def _resolve_precision_config_path(
    *,
    cache_path: str | Path | None,
    precision_config_path: str | Path | None,
) -> Path | None:
    raw_path = precision_config_path if precision_config_path is not None else os.getenv(PRECISION_CONFIG_ENV)
    if raw_path is not None:
        raw_text = str(raw_path).strip()
        if raw_text.lower() in {"", "none", "default", "builtin"}:
            return None
        path = Path(raw_text)
        return path if path.is_absolute() else path.resolve()
    if cache_path is None:
        return None
    cache_dir = Path(cache_path)
    model_dir = cache_dir.parent if cache_dir.name == ".ttnn_cache" else cache_dir
    selected_path = model_dir / DEFAULT_SELECTED_PRECISION_CONFIG_RELPATH
    return selected_path if selected_path.exists() else None


def _load_precision_policy_from_config(
    *,
    cache_path: str | Path | None,
    precision_config_path: str | Path | None,
) -> OptimizedDecoderPrecisionPolicy:
    resolved_path = _resolve_precision_config_path(
        cache_path=cache_path,
        precision_config_path=precision_config_path,
    )
    if resolved_path is None:
        set_multichip_ccl_dtype_policy(all_gather_dtype=ttnn.bfloat8_b, reduce_scatter_dtype=ttnn.bfloat8_b)
        return OptimizedDecoderPrecisionPolicy()
    config = json.loads(resolved_path.read_text())
    ccl_config = config.get("ccl", {})
    for key, value in ccl_config.get("runtime_env_overrides", {}).items():
        os.environ.setdefault(str(key), str(value))
    set_multichip_ccl_dtype_policy(
        all_gather_dtype=ccl_config.get("all_gather_dtype", ttnn.bfloat8_b),
        reduce_scatter_dtype=ccl_config.get("reduce_scatter_dtype", ttnn.bfloat8_b),
    )
    return precision_policy_from_config(config)


def _mesh_shape_tuple(mesh_device: ttnn.MeshDevice) -> tuple[int, int]:
    shape = tuple(mesh_device.shape)
    if len(shape) != 2:
        raise ValueError(f"expected 2D mesh shape, got {shape}")
    return int(shape[0]), int(shape[1])


def _validate_target_mesh(mesh_device: ttnn.MeshDevice) -> None:
    shape = _mesh_shape_tuple(mesh_device)
    if shape != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != 8:
        raise ValueError(
            f"{MODEL_ID} full model is specialized for the optimized 1x8 T3K decoder, "
            f"got shape={shape} num_devices={mesh_device.get_num_devices()}"
        )


def _replicated_mapper_config(mesh_device: ttnn.MeshDevice) -> ttnn.MeshMapperConfig | None:
    if mesh_device.get_num_devices() == 1:
        return None
    return ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate()],
        mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
    )


def _vocab_shard_mapper_config(mesh_device: ttnn.MeshDevice) -> ttnn.MeshMapperConfig:
    return ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementShard(-1)],
        mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
    )


def _load_hf_state_dict(hf_model_id: str = MODEL_ID, revision: str | None = None) -> dict[str, torch.Tensor]:
    snapshot = Path(snapshot_download(hf_model_id, revision=revision, local_files_only=True))
    model_file = snapshot / "model.safetensors"
    if not model_file.exists():
        raise FileNotFoundError(f"expected safetensors checkpoint at {model_file}")

    state: dict[str, torch.Tensor] = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            state[key] = f.get_tensor(key).to(torch.bfloat16)
    return state


def _permute_to_meta_format(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = torch.stack((cos[:, : cos.shape[1] // 2], cos[:, : cos.shape[1] // 2]), dim=-1).flatten(-2)
    sin = torch.stack((sin[:, : sin.shape[1] // 2], sin[:, : sin.shape[1] // 2]), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


class RotaryState(LightweightModule):
    """Meta-format RoPE table lookup used by the decoder stack."""

    def __init__(self, *, hf_config: Any, mesh_device: ttnn.MeshDevice, max_seq_len: int) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.max_seq_len = int(max_seq_len)
        self.head_dim = int(hf_config.head_dim)
        self._device_weights_loaded = False

        rotary_emb = LlamaRotaryEmbedding(hf_config)
        dummy = torch.zeros(1, self.max_seq_len, self.head_dim, dtype=torch.bfloat16)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            cos_hf, sin_hf = rotary_emb(dummy, position_ids)
        cos_meta, sin_meta = _permute_to_meta_format(cos_hf.float(), sin_hf.float())
        self.cos_source = cos_meta.to(torch.bfloat16)
        self.sin_source = sin_meta.to(torch.bfloat16)

        self.decode_trans_mat_source = get_rot_transformation_mat().repeat(1, 1, TILE_SIZE, 1)
        self.decode_trans_mat_memcfg = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, TILE_SIZE),
            core_grid=ttnn.num_cores_to_corerangeset(
                TILE_SIZE,
                mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.decode_rot_memcfg = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, self.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(
                TILE_SIZE,
                mesh_device.compute_with_storage_grid_size(),
                row_wise=True,
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.cos_matrix = ttnn.from_torch(
            self.cos_source,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.sin_matrix = ttnn.from_torch(
            self.sin_source,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.transformation_mat = ttnn.from_torch(
            self.decode_trans_mat_source,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.decode_trans_mat_memcfg,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._device_weights_loaded = True

    def prefill_forward(self, start_pos: int, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.load_device_weights()
        end_pos = int(start_pos) + int(seq_len)
        if end_pos > self.max_seq_len:
            raise ValueError(f"RoPE range [{start_pos}, {end_pos}) exceeds max_seq_len={self.max_seq_len}")
        return (
            self.cos_matrix[:, :, int(start_pos) : end_pos, :],
            self.sin_matrix[:, :, int(start_pos) : end_pos, :],
        )

    def decode_forward(self, rot_idxs: ttnn.Tensor, *, batch_size: int = DEFAULT_MAX_BATCH_SIZE) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        self.load_device_weights()
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.transpose(ttnn.unsqueeze_to_4D(cos), 1, 2)
        sin = ttnn.transpose(ttnn.unsqueeze_to_4D(sin), 1, 2)
        if batch_size % TILE_SIZE != 0:
            cos = cos[:, :batch_size, :, :]
            sin = sin[:, :batch_size, :, :]
        return (
            ttnn.interleaved_to_sharded(cos, self.decode_rot_memcfg),
            ttnn.interleaved_to_sharded(sin, self.decode_rot_memcfg),
        )


@dataclass(frozen=True)
class FullModelConfig:
    hf_model_id: str
    revision: str | None
    max_seq_len: int
    max_batch_size: int
    page_block_size: int
    num_layers: int | None = None
    cache_path: str | Path | None = None
    use_vllm_paged_kv_cache: bool = False


class Llama32FullModel(LightweightModule):
    """Full autoregressive TTNN model around the optimized multichip decoder."""

    def __init__(
        self,
        *,
        hf_config: Any,
        state_dict: dict[str, torch.Tensor],
        mesh_device: ttnn.MeshDevice,
        config: FullModelConfig,
        precision_policy: OptimizedDecoderPrecisionPolicy | None = None,
    ) -> None:
        super().__init__()
        _validate_target_mesh(mesh_device)
        if not bool(getattr(hf_config, "tie_word_embeddings", False)):
            raise ValueError(f"{MODEL_ID} full model expects tied token embeddings and LM head")

        self.hf_config = hf_config
        self.mesh_device = mesh_device
        self.config = config
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.vocab_size = int(hf_config.vocab_size)
        self.hidden_size = int(hf_config.hidden_size)
        requested_layers = int(config.num_layers or hf_config.num_hidden_layers)
        if requested_layers < 1 or requested_layers > int(hf_config.num_hidden_layers):
            raise ValueError(
                f"num_layers must be in [1, {int(hf_config.num_hidden_layers)}], got {requested_layers}"
            )
        self.n_layers = requested_layers
        self.num_devices = mesh_device.get_num_devices()
        self.max_seq_len = int(config.max_seq_len)
        self.max_batch_size = int(config.max_batch_size)
        self.page_block_size = int(config.page_block_size)
        self.padded_vocab_size = self.vocab_size
        if self.padded_vocab_size % (TILE_SIZE * self.num_devices) != 0:
            self.padded_vocab_size = math.ceil(self.padded_vocab_size / (TILE_SIZE * self.num_devices)) * (
                TILE_SIZE * self.num_devices
            )
        self.per_device_vocab_size = self.padded_vocab_size // self.num_devices
        self.precision_policy = precision_policy or OptimizedDecoderPrecisionPolicy()
        cache_path = Path(config.cache_path) if config.cache_path is not None else None

        embed_weight = _state_tensor(state_dict, "", "model.embed_tokens.weight")
        self.embedding_weight = LazyWeight(
            source=embed_weight.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper_config=None,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=(cache_path / "full_model", "tok_embeddings_replicated") if cache_path else None,
        )

        self.layers = [
            MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                page_block_size=self.page_block_size,
                max_seq_len=self.max_seq_len,
                max_batch_size=self.max_batch_size,
                precision_policy=self.precision_policy,
                cache_path=cache_path,
                materialize=False,
                use_vllm_paged_kv_cache=bool(config.use_vllm_paged_kv_cache),
            )
            for layer_idx in range(self.n_layers)
        ]
        self.decode_residual_memcfg = self.layers[0].decode_residual_memcfg

        norm_weight = _state_tensor(state_dict, "", "model.norm.weight")
        norm_weight = norm_weight.unsqueeze(0).view(1, 1, self.hidden_size).reshape(
            1,
            1,
            self.hidden_size // TILE_SIZE,
            TILE_SIZE,
        )
        self.norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=LazyWeight(
                    source=norm_weight,
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    mesh_mapper_config=_replicated_mapper_config(mesh_device),
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_dir_weight_name=(cache_path / "full_model", "final_norm") if cache_path else None,
                ),
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                eps=float(hf_config.rms_norm_eps),
                max_batch_size=self.max_batch_size,
                prefill_distributed=False,
                decode_memory_config=self.decode_residual_memcfg,
                compute_kernel_config=_compute_kernel_config_hifi2_fp16(),
            )
        )
        self.lm_head = self._make_lm_head(embed_weight, cache_path=cache_path)
        self.rotary = RotaryState(hf_config=hf_config, mesh_device=mesh_device, max_seq_len=self.max_seq_len)
        self.sampling_args = SimpleNamespace(
            vocab_size=self.vocab_size,
            padded_vocab_size=self.padded_vocab_size,
            max_batch_size=self.max_batch_size,
            max_top_k=DEFAULT_MAX_TOP_K,
            cluster_shape=TARGET_MESH_SHAPE,
            sampling_all_gather_axis=0,
            pad_logits_to_power_of_2=True,
            sub_core_grids=None,
            sub_core_grid_topk=None,
            start_core=ttnn.CoreCoord(0, 0),
        )
        self._device_weights_loaded = False

    @classmethod
    def from_pretrained(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = MODEL_ID,
        revision: str | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        num_layers: int | None = None,
        cache_path: str | Path | None = None,
        precision_policy: OptimizedDecoderPrecisionPolicy | None = None,
        precision_config_path: str | Path | None = None,
        use_vllm_paged_kv_cache: bool = False,
    ) -> "Llama32FullModel":
        hf_config = AutoConfig.from_pretrained(hf_model_id, revision=revision, local_files_only=True)
        state_dict = _load_hf_state_dict(hf_model_id, revision=revision)
        if precision_policy is None:
            precision_policy = _load_precision_policy_from_config(
                cache_path=cache_path,
                precision_config_path=precision_config_path,
            )
        config = FullModelConfig(
            hf_model_id=hf_model_id,
            revision=revision,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_block_size=page_block_size,
            num_layers=num_layers,
            cache_path=cache_path,
            use_vllm_paged_kv_cache=use_vllm_paged_kv_cache,
        )
        return cls(
            hf_config=hf_config,
            state_dict=state_dict,
            mesh_device=mesh_device,
            config=config,
            precision_policy=precision_policy,
        )

    @property
    def kv_cache(self) -> list[tuple[ttnn.Tensor, ttnn.Tensor] | None]:
        return [layer.kv_cache for layer in self.layers]

    def bind_external_kv_cache(self, kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]]) -> None:
        if len(kv_cache) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} KV cache layers, got {len(kv_cache)}")
        for layer, cache_pair in zip(self.layers, kv_cache):
            key_cache, value_cache = cache_pair
            layer.attention.kv_cache = (key_cache, value_cache)
            layer.attention.config.kv_cache = (key_cache, value_cache)
            layer.attention.config.use_vllm_paged_kv_cache = False

    def _make_lm_head(self, embed_weight: torch.Tensor, *, cache_path: Path | None) -> LMHead1D:
        output_weight = embed_weight.transpose(0, 1).contiguous()
        if self.padded_vocab_size > self.vocab_size:
            output_weight = F.pad(output_weight, (0, self.padded_vocab_size - self.vocab_size), value=0.0)

        split_sizes = []
        remaining = self.per_device_vocab_size
        while remaining > 0:
            split = min(remaining, LM_HEAD_MAX_COLUMNS_PER_DEVICE)
            split_sizes.append(split)
            remaining -= split

        dram_size = self.mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        tile_padded_batch_rows = TILE_SIZE * math.ceil(self.max_batch_size / TILE_SIZE)
        lm_head_core_grid = ttnn.CoreGrid(x=8, y=8)
        output_weights = []
        weight_memcfgs = []
        program_configs = []
        for split_idx, split_size in enumerate(split_sizes):
            device_splits = []
            for device_idx in range(self.num_devices):
                start = device_idx * self.per_device_vocab_size + sum(split_sizes[:split_idx])
                end = start + split_size
                device_splits.append(output_weight[:, start:end])
            combined_split = torch.cat(device_splits, dim=-1)
            memcfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (
                        self.hidden_size,
                        math.ceil(split_size / (TILE_SIZE * dram_size.x)) * TILE_SIZE,
                    ),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            weight_memcfgs.append(memcfg)
            output_weights.append(
                LazyWeight(
                    source=combined_split,
                    dtype=self.precision_policy.lm_head_weight_dtype,
                    device=self.mesh_device,
                    mesh_mapper_config=_vocab_shard_mapper_config(self.mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memcfg,
                    cache_dir_weight_name=(cache_path / "full_model_lm_head", f"output_split_{split_idx}_{split_size}")
                    if cache_path
                    else None,
                )
            )
            program_configs.append(
                _dram_matmul_config(
                    m=tile_padded_batch_rows,
                    k=self.hidden_size,
                    n=split_size,
                    num_cores=lm_head_core_grid.num_cores,
                )
            )

        input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, nearest_32(self.hidden_size // lm_head_core_grid.num_cores)),
            lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=output_weights,
                mesh_device=self.mesh_device,
                dim=self.hidden_size,
                max_batch_size=self.max_batch_size,
                program_configs=program_configs,
                compute_kernel_config=_compute_kernel_config_hifi2_fp16(),
                lm_head_dtype=ttnn.bfloat16,
                output_memcfg=ttnn.L1_MEMORY_CONFIG,
                input_memcfg=input_memcfg,
                weights_memcfgs=weight_memcfgs,
            )
        )

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.embedding = self.embedding_weight.get_device_weight()
        for layer in self.layers:
            layer.load_device_weights()
        self.norm.load_device_weights()
        self.lm_head.load_device_weights()
        self.rotary.load_device_weights()
        self._device_weights_loaded = True

    def reset_kv_cache(self) -> None:
        self.load_device_weights()
        for key_cache, value_cache in self.kv_cache:
            ttnn.fill(key_cache, 0.0, memory_config=key_cache.memory_config(), output_tensor=key_cache)
            ttnn.fill(value_cache, 0.0, memory_config=value_cache.memory_config(), output_tensor=value_cache)

    def make_page_table(self, batch_size: int | None = None, max_seq_len: int | None = None) -> torch.Tensor:
        batch_size = int(batch_size or self.max_batch_size)
        max_seq_len = int(max_seq_len or self.max_seq_len)
        blocks_per_user = math.ceil(max_seq_len / self.page_block_size)
        page_table = torch.zeros(batch_size, blocks_per_user, dtype=torch.int32)
        for user_id in range(batch_size):
            start = user_id * blocks_per_user
            page_table[user_id] = torch.arange(start, start + blocks_per_user, dtype=torch.int32)
        return page_table

    def page_table_to_device(self, page_table: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            page_table.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prepare_prefill_tokens_device(self, tokens: torch.Tensor) -> ttnn.Tensor:
        if tokens.dim() != 2 or tokens.shape[0] != 1:
            raise ValueError(f"prefill tokens must be [1, seq_len], got {tuple(tokens.shape)}")
        token_view = tokens.to(torch.uint32).reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            token_view,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prepare_decode_inputs_host(
        self,
        tokens: torch.Tensor,
        current_pos: torch.Tensor,
        page_table: torch.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        tokens = tokens.reshape(-1).to(torch.int64)
        current_pos = current_pos.reshape(-1).to(torch.int64)
        if tokens.numel() > self.max_batch_size:
            raise ValueError(f"decode batch {tokens.numel()} exceeds max_batch_size={self.max_batch_size}")

        token_pad = torch.zeros(self.max_batch_size, dtype=torch.uint32)
        token_pad[: tokens.numel()] = tokens.to(torch.uint32)
        current_pad = torch.full((self.max_batch_size,), -1, dtype=torch.int32)
        current_pad[: current_pos.numel()] = current_pos.to(torch.int32)
        rot_idxs = torch.maximum(current_pad, torch.zeros_like(current_pad)).to(torch.uint32).reshape(1, self.max_batch_size)

        tokens_tt = ttnn.from_torch(
            token_pad.reshape(1, 1, 1, self.max_batch_size),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        current_tt = ttnn.from_torch(
            current_pad,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        rot_idxs_tt = ttnn.from_torch(
            rot_idxs,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        page_table_tt = ttnn.from_torch(
            page_table.to(torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return tokens_tt, current_tt, rot_idxs_tt, page_table_tt

    def copy_decode_inputs_to_device(
        self,
        host_inputs: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor],
        device_inputs: tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor] | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        if device_inputs is None:
            return tuple(
                ttnn.to_device(tensor, self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tensor in host_inputs
            )  # type: ignore[return-value]
        for host_tensor, device_tensor in zip(host_inputs, device_inputs):
            if host_tensor is not None:
                ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        return device_inputs

    def copy_page_table_to_device(
        self,
        page_table_host: ttnn.Tensor,
        page_table_device: ttnn.Tensor,
    ) -> None:
        ttnn.copy_host_to_device_tensor(page_table_host, page_table_device)

    def embed_tokens(self, tokens_tt: ttnn.Tensor) -> ttnn.Tensor:
        self.load_device_weights()
        return ttnn.embedding(
            tokens_tt,
            self.embedding,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def embed_decode(self, tokens_tt: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embed_tokens(tokens_tt)
        return ttnn.to_memory_config(x, self.decode_residual_memcfg)

    def embed_prefill(self, tokens_tt: ttnn.Tensor) -> ttnn.Tensor:
        return self.embed_tokens(tokens_tt)

    def prefill_forward_device(
        self,
        tokens_tt: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        start_pos: int = 0,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        x = self.embed_prefill(tokens_tt)
        seq_len = int(x.shape[-2])
        rot_mats = self.rotary.prefill_forward(start_pos, seq_len)
        for layer in self.layers:
            x = layer.prefill_forward(x, rot_mats=rot_mats, page_table=page_table, user_id=user_id)
        x = self.norm.prefill_forward(x)
        return self._lm_head_prefill_forward(x)

    def _lm_head_decode_rows_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.memory_config().is_sharded():
            x = ttnn.to_memory_config(x, self.lm_head.config.input_memcfg)
        else:
            x = ttnn.interleaved_to_sharded(x, self.lm_head.config.input_memcfg)
        logits = self.lm_head.forward(x)
        return ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)

    def _lm_head_prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        seq_len = int(x.shape[-2])
        if seq_len % TILE_SIZE != 0:
            raise ValueError(f"prefill LM head expects tile-multiple rows, got {seq_len}")
        outputs = []
        for row_start in range(0, seq_len, TILE_SIZE):
            block = ttnn.slice(x, (0, 0, row_start, 0), (1, 1, row_start + TILE_SIZE, x.shape[-1]))
            outputs.append(self._lm_head_decode_rows_forward(block))
        if len(outputs) == 1:
            return outputs[0]
        return ttnn.concat(outputs, dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward_device(
        self,
        tokens_tt: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_idxs: ttnn.Tensor,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        x = self.embed_decode(tokens_tt)
        rot_mats = self.rotary.decode_forward(rot_idxs, batch_size=self.max_batch_size)
        for layer in self.layers:
            x = layer.decode_forward(x, current_pos=current_pos, rot_mats=rot_mats, page_table=page_table)
        x = self.norm.decode_forward(x)
        logits = self._lm_head_decode_rows_forward(x)
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_idxs)
        return logits

    def logits_to_torch(self, logits: ttnn.Tensor, *, rows: int | None = None) -> torch.Tensor:
        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        device_tensors = ttnn.get_device_tensors(logits)
        parts = [ttnn.to_torch(device_tensor).float() for device_tensor in device_tensors]
        full = torch.cat(parts, dim=-1)
        if rows is not None:
            full = full[:, :, :rows, :]
        return full[..., : self.vocab_size]


__all__ = [
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_PAGE_BLOCK_SIZE",
    "Llama32FullModel",
    "FullModelConfig",
    "MODEL_ID",
]
