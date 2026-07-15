# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full TP4 autoregressive Gemma 4 31B model.

This module deliberately wraps :class:`MultichipDecoder` instead of the demo
Gemma decoder.  The decoder's replicated-BF16 layer boundary, TP-local weights,
BFP8 paged cache, persistent reductions, and precision policy are therefore the
same in a one-layer test and in the complete 60-layer stack.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_31b.tt.functional_decoder import HF_ADVERTISED_CONTEXT, HF_MODEL_ID
from models.autoports.google_gemma_4_31b.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    MultichipDecoder,
    release_multichip_decoder_resources,
)
from models.demos.gemma4.tt.ccl import ccl_allgather
from models.demos.gemma4.tt.rms_norm import RMSNorm


def _text_config(config):
    return getattr(config, "text_config", config)


def _resolve_checkpoint(model_id_or_path: str | Path = HF_MODEL_ID) -> Path:
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    if str(model_id_or_path) != HF_MODEL_ID:
        raise FileNotFoundError(f"checkpoint is not local: {model_id_or_path}")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-31B" / "snapshots"
    matches = sorted(path for path in snapshots.glob("*") if (path / "config.json").exists())
    if not matches:
        raise FileNotFoundError(f"cached {HF_MODEL_ID} checkpoint not found under {snapshots}")
    return matches[-1].resolve()


def _load_checkpoint_state(checkpoint: Path, *, layer_indices: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shard_names = sorted(set(index["weight_map"].values()))
    else:
        weight_map = {}
        shard_names = [path.name for path in sorted(checkpoint.glob("*.safetensors"))]
    if not shard_names:
        raise FileNotFoundError(f"no safetensors weights found under {checkpoint}")
    required: set[str] | None = None
    if weight_map:
        required = {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    if layer_indices is not None and required is not None:
        prefixes = tuple(f"model.language_model.layers.{index}." for index in layer_indices)
        required.update(key for key in weight_map if key.startswith(prefixes))
    elif layer_indices is None:
        required = None
    state: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if required is None or key in required:
                    state[key] = handle.get_tensor(key)
    return state


def _replicated_tensor(
    source: torch.Tensor,
    mesh_device,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        source.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _copy_host_to_mesh(host_tensor: ttnn.Tensor, mesh_tensor: ttnn.Tensor) -> None:
    """Copy one host TT tensor into every physical tensor of a replicated mesh tensor."""
    for device_tensor in ttnn.get_device_tensors(mesh_tensor):
        ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)


ROPE_POSITION_INACTIVE_SENTINEL = 1 << 31
ROPE_POSITION_LOOKUP_MASK = ROPE_POSITION_INACTIVE_SENTINEL - 1


def _pad_rope_positions(positions: torch.Tensor, *, width: int = 32) -> torch.Tensor:
    positions = positions.reshape(-1).to(torch.int32)
    if positions.numel() > width:
        raise ValueError(f"decode batch {positions.numel()} exceeds RoPE position width {width}")
    result = torch.full((1, width), ROPE_POSITION_INACTIVE_SENTINEL, dtype=torch.int64)
    active = positions.ge(0)
    result[0, : positions.numel()] = torch.where(active, positions.to(torch.int64), ROPE_POSITION_INACTIVE_SENTINEL)
    return result


def _kv_cache_identity(kv_cache: Sequence[Sequence[ttnn.Tensor]]) -> tuple[tuple[int, ...], ...]:
    """Identify the cache allocation rather than a caller-created list wrapper."""
    return tuple(tuple(id(tensor) for tensor in cache_pair) for cache_pair in kv_cache)


def _sequence_tile_ranges(logical_rows: int) -> tuple[tuple[int, int], ...]:
    """Partition a positive logical sequence into contiguous TT tile-height ranges."""
    if logical_rows < 1:
        raise ValueError("logical_rows must be positive")
    return tuple((start, min(start + ttnn.TILE_SIZE, logical_rows)) for start in range(0, logical_rows, ttnn.TILE_SIZE))


@dataclass(frozen=True)
class Gemma4FullModelConfig:
    max_seq_len: int = HF_ADVERTISED_CONTEXT
    layer_indices: tuple[int, ...] | None = None
    max_batch_size: int = 1
    lm_head_weight_dtype: ttnn.DataType = ttnn.bfloat16
    logits_dtype: ttnn.DataType = ttnn.bfloat16
    lm_head_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    lm_head_dram_sharded: bool = True
    lm_head_num_cores: int = 4
    lm_head_in0_block_w: int = 2
    lm_head_split_size: int = 8192
    trace_region_size_bytes: int = 256 << 20


@dataclass
class DecodeTraceState:
    trace_id: int | None = None
    logits: ttnn.Tensor | None = None
    token_input: ttnn.Tensor | None = None
    rope_position: ttnn.Tensor | None = None
    rope_position_lookup_mask: ttnn.Tensor | None = None
    cache_position: ttnn.Tensor | None = None
    page_tables: list[ttnn.Tensor] = field(default_factory=list)
    page_table_identities: list[int] = field(default_factory=list)
    page_table_generations: list[int] = field(default_factory=list)
    batch_size: int = 0
    active_batch_size: int = 0
    prompt_lengths: tuple[int, ...] = ()
    kv_cache_identity: tuple[tuple[int, ...], ...] | None = None
    initial_positions: torch.Tensor | None = None
    initial_replay_count: int = 0
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "model_trace_replays": 0,
            "token_host_refreshes": 0,
            "position_host_refreshes": 0,
            "rope_host_refreshes": 0,
            "page_table_refreshes": 0,
            "synchronizations": 0,
            "sampled_token_readbacks": 0,
            "full_logits_readbacks": 0,
        }
    )


class Gemma4FullModel:
    """Full autoregressive path over the optimized TP4 decoder stack."""

    def __init__(
        self,
        *,
        hf_config,
        state_dict: dict[str, torch.Tensor],
        mesh_device,
        config: Gemma4FullModelConfig | None = None,
        tensor_cache_path: str | Path | None = None,
    ) -> None:
        self.config = config or Gemma4FullModelConfig()
        self.hf_config = _text_config(hf_config)
        self.mesh_device = mesh_device
        if tuple(mesh_device.shape) != TARGET_MESH_SHAPE or mesh_device.get_num_devices() != 4:
            raise ValueError(f"Gemma4FullModel requires MeshShape{TARGET_MESH_SHAPE}, got {tuple(mesh_device.shape)}")
        if not 1 <= self.config.max_seq_len <= int(self.hf_config.max_position_embeddings):
            raise ValueError("max_seq_len must preserve a positive prefix of the HF context")
        if not 1 <= self.config.max_batch_size <= 32:
            raise ValueError("max_batch_size must be in [1, 32]")

        all_indices = tuple(range(int(self.hf_config.num_hidden_layers)))
        self.layer_indices = self.config.layer_indices or all_indices
        if not self.layer_indices or any(index not in all_indices for index in self.layer_indices):
            raise ValueError(f"invalid layer_indices={self.layer_indices}")
        self.layer_kinds = [self.hf_config.layer_types[index] for index in self.layer_indices]
        self.hidden_size = int(self.hf_config.hidden_size)
        self.vocab_size = int(self.hf_config.vocab_size)
        if self.vocab_size % mesh_device.get_num_devices():
            raise ValueError("Gemma vocabulary must divide exactly over TP4")
        self.vocab_per_device = self.vocab_size // mesh_device.get_num_devices()
        if self.vocab_per_device != 65_536:
            raise ValueError(f"unexpected TP4 vocabulary shard: {self.vocab_per_device}")
        self.final_logit_softcapping = float(self.hf_config.final_logit_softcapping)
        self.embed_scale = math.sqrt(self.hidden_size)
        self.trace_state = DecodeTraceState()

        lm_head_k_tiles = self.hidden_size // ttnn.TILE_SIZE
        if self.hidden_size % (ttnn.TILE_SIZE * self.config.lm_head_num_cores):
            raise ValueError("LM-head hidden width must tile-divide over lm_head_num_cores")
        if (lm_head_k_tiles // self.config.lm_head_num_cores) % self.config.lm_head_in0_block_w:
            raise ValueError("LM-head in0_block_w must divide the per-core hidden K tiles")
        if self.vocab_per_device % self.config.lm_head_split_size:
            raise ValueError("TP-local vocabulary must divide exactly over lm_head_split_size")

        embed_key = "model.language_model.embed_tokens.weight"
        norm_key = "model.language_model.norm.weight"
        embedding = state_dict[embed_key].to(torch.bfloat16)
        self.embedding_weight = ttnn.from_torch(
            embedding,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        lm_head_source = embedding.transpose(0, 1).contiguous()
        lm_head_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        self.lm_head_weights: list[ttnn.Tensor] = []
        if self.config.lm_head_dram_sharded:
            # Direct BF16 tilization into this very wide sharded allocation asks
            # the device tilizer for an 8+ MiB circular buffer.  Tile on the host
            # first, then perform only the final DRAM-sharded placement on device.
            for split_start in range(0, self.vocab_per_device, self.config.lm_head_split_size):
                device_splits = [
                    lm_head_source[
                        :,
                        device_idx * self.vocab_per_device
                        + split_start : device_idx * self.vocab_per_device
                        + split_start
                        + self.config.lm_head_split_size,
                    ]
                    for device_idx in range(mesh_device.get_num_devices())
                ]
                split_source = torch.cat(device_splits, dim=-1).contiguous()
                lm_head_host = ttnn.from_torch(
                    split_source,
                    dtype=self.config.lm_head_weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=lm_head_mapper,
                )
                self.lm_head_weights.append(
                    ttnn.to_device(
                        lm_head_host,
                        mesh_device,
                        memory_config=self._lm_head_weight_memory_config(self.config.lm_head_split_size),
                    )
                )
            self.lm_head_weight = None
        else:
            self.lm_head_weight = ttnn.from_torch(
                lm_head_source,
                device=mesh_device,
                dtype=self.config.lm_head_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=lm_head_mapper,
            )
        self.final_norm = RMSNorm(
            mesh_device,
            self.hf_config,
            {"weight": state_dict[norm_key]},
            tensor_cache_path=str(Path(tensor_cache_path) / "final_norm") if tensor_cache_path else None,
        )
        self.lm_head_compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.config.lm_head_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.lm_head_input_memory_config = self._lm_head_input_memory_config()
        self.lm_head_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=self.config.lm_head_in0_block_w,
            per_core_M=1,
            per_core_N=math.ceil(self.config.lm_head_split_size / (ttnn.TILE_SIZE * self.config.lm_head_num_cores)),
        )

        self.decode_rope = self._build_decode_rope_caches()
        self.layers: list[MultichipDecoder] = []
        for layer_idx in self.layer_indices:
            layer_cache = str(Path(tensor_cache_path) / f"layer_{layer_idx}") if tensor_cache_path else None
            self.layers.append(
                MultichipDecoder.from_state_dict(
                    state_dict,
                    hf_config=self.hf_config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    tensor_cache_path=layer_cache,
                )
            )

    def _lm_head_weight_memory_config(self, local_vocab_width: int) -> ttnn.MemoryConfig:
        """Width-shard each TP-local vocab projection over Blackhole DRAM banks."""
        grid = self.mesh_device.dram_grid_size()
        num_banks = grid.x * grid.y
        padded_n = math.ceil(local_vocab_width / (ttnn.TILE_SIZE * num_banks)) * ttnn.TILE_SIZE * num_banks
        dram_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(
                dram_cores,
                (self.hidden_size, padded_n // num_banks),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _lm_head_input_memory_config(self) -> ttnn.MemoryConfig:
        core_grid = ttnn.num_cores_to_corerangeset(
            self.config.lm_head_num_cores,
            self.mesh_device.compute_with_storage_grid_size(),
            row_wise=True,
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.hidden_size // self.config.lm_head_num_cores),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    @classmethod
    def from_pretrained(
        cls,
        *,
        mesh_device,
        model_id_or_path: str | Path = HF_MODEL_ID,
        config: Gemma4FullModelConfig | None = None,
        hf_config=None,
        state_dict: dict[str, torch.Tensor] | None = None,
        tensor_cache_path: str | Path | None = None,
    ) -> "Gemma4FullModel":
        checkpoint = _resolve_checkpoint(model_id_or_path)
        hf_config = hf_config or AutoConfig.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
        requested_layers = None if config is None else config.layer_indices
        state_dict = state_dict or _load_checkpoint_state(checkpoint, layer_indices=requested_layers)
        return cls(
            hf_config=hf_config,
            state_dict=state_dict,
            mesh_device=mesh_device,
            config=config,
            tensor_cache_path=tensor_cache_path,
        )

    def _build_decode_rope_caches(self) -> dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]:
        rotary = Gemma4TextRotaryEmbedding(self.hf_config)
        positions = torch.arange(self.config.max_seq_len, dtype=torch.long).unsqueeze(0)
        dummy = torch.zeros((1, 1, self.hidden_size), dtype=torch.bfloat16)
        result = {}
        for layer_kind in sorted(set(self.layer_kinds)):
            cos, sin = rotary(dummy, positions, layer_type=layer_kind)
            result[layer_kind] = tuple(
                _replicated_tensor(table.squeeze(0), self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
                for table in (cos, sin)
            )
        return result

    def _prefill_rope(self, logical_len: int) -> dict[str, tuple[ttnn.Tensor, ttnn.Tensor]]:
        result = {}
        for layer_kind, tables in self.decode_rope.items():
            tiled = []
            for table in tables:
                sliced = ttnn.slice(table, [0, 0], [logical_len, table.shape[-1]])
                view = ttnn.reshape(sliced, [1, 1, logical_len, table.shape[-1]])
                tiled.append(ttnn.to_layout(view, ttnn.TILE_LAYOUT))
            result[layer_kind] = tuple(tiled)
        return result

    def embed_tokens(self, tokens: torch.Tensor | ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        owns_tokens = isinstance(tokens, torch.Tensor)
        if isinstance(tokens, torch.Tensor):
            if mode == "prefill":
                host_shape = tuple(tokens.shape)
                token_source = tokens.to(torch.int32).contiguous()
            else:
                host_shape = tuple(tokens.shape)
                token_source = tokens.to(torch.int32).reshape(1, 1, 1, tokens.shape[0]).contiguous()
            tokens = _replicated_tensor(token_source, self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            batch_size = host_shape[0]
        else:
            batch_size = int(tokens.shape[-1]) if mode == "decode" else int(tokens.shape[0])
        embedded = ttnn.embedding(tokens, self.embedding_weight, dtype=ttnn.bfloat16)
        if owns_tokens:
            tokens.deallocate(True)
        local = ttnn.mul(embedded, self.embed_scale)
        embedded.deallocate(True)
        local_4d = ttnn.unsqueeze_to_4D(local)
        # ccl_allgather owns/deallocates its input. It returns ROW_MAJOR, while
        # the optimized decoder's replicated BF16 residual contract is TILE.
        # Establish the logical M first so non-aligned prefills get physical
        # tile padding only during the layout conversion.
        gathered = ccl_allgather(local_4d, self.layers[0].mesh_config, self.layers[0].ccl_manager)
        if mode == "decode":
            shape = [1, 1, batch_size, self.hidden_size]
        else:
            shape = [1, 1, gathered.shape[-2], self.hidden_size]
        reshaped = ttnn.reshape(gathered, shape)
        tiled = ttnn.to_layout(reshaped, ttnn.TILE_LAYOUT)
        gathered.deallocate(True)
        return tiled

    def allocate_paged_kv_cache(
        self, *, max_context: int | None = None, batch_size: int | None = None
    ) -> tuple[list[tuple[ttnn.Tensor, ttnn.Tensor]], list[ttnn.Tensor]]:
        max_context = int(max_context or self.config.max_seq_len)
        batch_size = int(batch_size or self.config.max_batch_size)
        if max_context > self.config.max_seq_len:
            raise ValueError("requested KV context exceeds model context")
        caches: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
        page_tables: list[ttnn.Tensor] = []
        shared_tables: dict[str, ttnn.Tensor] = {}
        for layer, layer_kind in zip(self.layers, self.layer_kinds):
            cache, table = layer.init_paged_kv_cache(max_context=max_context, batch_size=batch_size)
            caches.append(cache)
            if layer_kind in shared_tables:
                table.deallocate(True)
                table = shared_tables[layer_kind]
            else:
                shared_tables[layer_kind] = table
            page_tables.append(table)
        return caches, page_tables

    def _normalize_page_tables(self, page_table: Any) -> list[ttnn.Tensor]:
        if isinstance(page_table, dict):
            result = [page_table[layer_kind] for layer_kind in self.layer_kinds]
        elif isinstance(page_table, Sequence) and not isinstance(page_table, ttnn.Tensor):
            result = list(page_table)
        else:
            result = [page_table] * len(self.layers)
        if len(result) != len(self.layers) or any(table is None for table in result):
            raise ValueError("page_table must resolve to one device table per decoder layer")
        return result

    def _project_sharded_lm_head_tile(self, normed_tile: ttnn.Tensor) -> ttnn.Tensor:
        """Project one logical sequence tile with the fixed-M DRAM-sharded LM head."""
        logical_rows = int(normed_tile.shape[-2])
        if not 1 <= logical_rows <= ttnn.TILE_SIZE:
            raise ValueError(f"sharded LM-head tile must have 1..{ttnn.TILE_SIZE} logical rows, got {logical_rows}")
        lm_head_input = ttnn.to_memory_config(normed_tile, self.lm_head_input_memory_config)
        normed_tile.deallocate(True)
        split_logits = []
        for weight in self.lm_head_weights:
            local_logits = ttnn.linear(
                lm_head_input,
                weight,
                dtype=self.config.logits_dtype,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=self.lm_head_program_config,
                compute_kernel_config=self.lm_head_compute,
            )
            split_logits.append(ttnn.sharded_to_interleaved(local_logits, ttnn.DRAM_MEMORY_CONFIG))
            local_logits.deallocate(True)
        lm_head_input.deallocate(True)
        logits = ttnn.concat(split_logits, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for split in split_logits:
            split.deallocate(True)
        return logits

    def _terminal(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        normed = self.final_norm.forward(hidden)
        hidden.deallocate(True)
        if self.config.lm_head_dram_sharded:
            logical_rows = int(normed.shape[-2])
            if logical_rows <= ttnn.TILE_SIZE:
                logits = self._project_sharded_lm_head_tile(normed)
            else:
                tile_logits = []
                for start, end in _sequence_tile_ranges(logical_rows):
                    normed_tile = ttnn.slice(
                        normed,
                        [0, 0, start, 0],
                        [1, 1, end, self.hidden_size],
                    )
                    tile_logits.append(self._project_sharded_lm_head_tile(normed_tile))
                normed.deallocate(True)
                logits = ttnn.concat(tile_logits, dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tile in tile_logits:
                    tile.deallocate(True)
        else:
            logits = ttnn.linear(
                normed,
                self.lm_head_weight,
                dtype=self.config.logits_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.lm_head_compute,
            )
            normed.deallocate(True)
        if self.final_logit_softcapping > 0:
            scaled = ttnn.mul(logits, 1.0 / self.final_logit_softcapping)
            logits.deallocate(True)
            capped = ttnn.tanh(scaled)
            scaled.deallocate(True)
            logits = ttnn.mul(capped, self.final_logit_softcapping)
            capped.deallocate(True)
        return logits

    def logits_to_torch(self, logits: ttnn.Tensor) -> torch.Tensor:
        self.trace_state.counters["full_logits_readbacks"] += 1
        return ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1)).float()[
            ..., : self.vocab_size
        ]

    def prefill_hidden(
        self,
        tokens: torch.Tensor,
        *,
        page_tables: Sequence[ttnn.Tensor],
        kv_cache: Sequence[Sequence[ttnn.Tensor]],
        user_id: int,
        prompt_len: int,
    ) -> ttnn.Tensor:
        hidden = self.embed_tokens(tokens[:, :prompt_len], mode="prefill")
        rope = self._prefill_rope(prompt_len)
        try:
            for layer, layer_kind, cache, table in zip(self.layers, self.layer_kinds, kv_cache, page_tables):
                previous = hidden
                hidden = layer.prefill_forward(
                    hidden,
                    rope_mats=rope[layer_kind],
                    page_table=table,
                    kv_cache=cache,
                    user_id=user_id,
                    valid_seq_len=prompt_len,
                )
                previous.deallocate(True)
            return hidden
        finally:
            for tables in rope.values():
                for table in tables:
                    table.deallocate(True)

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: list[int],
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        if tokens.ndim != 2 or len(prompt_lens) != tokens.shape[0]:
            raise ValueError("prefill requires tokens [batch, seq] and one prompt length per row")
        page_tables = self._normalize_page_tables(page_table)
        outputs = []
        for user_id, prompt_len in enumerate(prompt_lens):
            prompt_len = int(prompt_len)
            if not 1 <= prompt_len <= self.config.max_seq_len or prompt_len > tokens.shape[1]:
                raise ValueError(f"invalid logical prompt length {prompt_len}")
            hidden = self.prefill_hidden(
                tokens[user_id : user_id + 1],
                page_tables=page_tables,
                kv_cache=kv_cache,
                user_id=user_id,
                prompt_len=prompt_len,
            )
            if not return_all_logits:
                tile_start = ((prompt_len - 1) // 32) * 32
                tile_end = min(tile_start + 32, hidden.shape[-2])
                if tile_start == 0 and tile_end == hidden.shape[-2]:
                    # A full-range slice aliases its input.  Hand ownership of
                    # the original short-prompt tensor to _terminal instead of
                    # deallocating the backing buffer before RMSNorm consumes it.
                    last_hidden = hidden
                else:
                    last_hidden = ttnn.slice(
                        hidden,
                        [0, 0, tile_start, 0],
                        [1, 1, tile_end, self.hidden_size],
                    )
                    hidden.deallocate(True)
                logits = self._terminal(last_hidden)
                host = self.logits_to_torch(logits).reshape(1, -1, self.vocab_size)
                outputs.append(host[:, (prompt_len - 1) - tile_start : (prompt_len - 1) - tile_start + 1])
                logits.deallocate(True)
            else:
                logits = self._terminal(hidden)
                outputs.append(self.logits_to_torch(logits).reshape(1, prompt_len, self.vocab_size))
                logits.deallocate(True)
        if return_all_logits and len({output.shape[1] for output in outputs}) != 1:
            max_len = max(output.shape[1] for output in outputs)
            outputs = [torch.nn.functional.pad(output, (0, 0, 0, max_len - output.shape[1])) for output in outputs]
        return torch.cat(outputs, dim=0)

    def prefill_forward_device_logits(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: list[int],
    ) -> ttnn.Tensor:
        """Prefill mixed prompt rows and return one sampler-ready TP-sharded row per user."""
        if tokens.ndim != 2 or tokens.shape[0] != len(prompt_lens):
            raise ValueError("device-logit prefill requires tokens [batch, seq] and one prompt length per row")
        if not 1 <= tokens.shape[0] <= self.config.max_batch_size:
            raise ValueError(f"device-logit prefill batch must be in [1, {self.config.max_batch_size}]")

        page_tables = self._normalize_page_tables(page_table)
        row_logits = []
        for user_id, requested_len in enumerate(prompt_lens):
            prompt_len = int(requested_len)
            if not 1 <= prompt_len <= tokens.shape[1] or prompt_len > self.config.max_seq_len:
                raise ValueError(f"invalid logical prompt length {prompt_len}")
            hidden = self.prefill_hidden(
                tokens[user_id : user_id + 1],
                page_tables=page_tables,
                kv_cache=kv_cache,
                user_id=user_id,
                prompt_len=prompt_len,
            )
            tile_start = ((prompt_len - 1) // 32) * 32
            tile_end = min(tile_start + 32, hidden.shape[-2])
            local_index = (prompt_len - 1) - tile_start
            if tile_start == 0 and tile_end == hidden.shape[-2]:
                # Short prompts occupy the complete logical tensor. TTNN returns
                # an alias for that full-range slice, so _terminal must receive
                # the original allocation and take ownership of its deallocation.
                last_tile = hidden
            else:
                last_tile = ttnn.slice(
                    hidden,
                    [0, 0, tile_start, 0],
                    [1, 1, tile_end, self.hidden_size],
                )
                hidden.deallocate(True)
            tile_logits = self._terminal(last_tile)
            row_view = ttnn.slice(
                tile_logits,
                [0, 0, local_index, 0],
                [1, 1, local_index + 1, tile_logits.shape[-1]],
                memory_config=tile_logits.memory_config(),
            )
            # A full-range M=1 slice aliases its input. Materialize a distinct
            # sampler-owned row before releasing the terminal tile backing.
            row_logits.append(ttnn.clone(row_view, memory_config=tile_logits.memory_config()))
            tile_logits.deallocate(True)

        if len(row_logits) == 1:
            return row_logits[0]
        logits = ttnn.concat(row_logits, dim=-2, memory_config=row_logits[0].memory_config())
        for row in row_logits:
            row.deallocate(True)
        return logits

    def decode_hidden_device(
        self,
        token_input: ttnn.Tensor,
        *,
        rope_position: ttnn.Tensor,
        rope_position_lookup_mask: ttnn.Tensor,
        cache_position: ttnn.Tensor,
        page_tables: Sequence[ttnn.Tensor],
        kv_cache: Sequence[Sequence[ttnn.Tensor]],
        batch_size: int,
    ) -> ttnn.Tensor:
        # RoPE state uses the UINT32 sign bit as the inactive fixed-slot marker.
        # Masking is an ordinary captured device op: active positions are unchanged,
        # while inactive/padded slots look up the safe row-zero RoPE values.
        rope_lookup_position = ttnn.bitwise_and(rope_position, rope_position_lookup_mask)
        hidden = self.embed_tokens(token_input, mode="decode")
        for layer, layer_kind, cache, table in zip(self.layers, self.layer_kinds, kv_cache, page_tables):
            previous = hidden
            hidden = layer.decode_forward(
                hidden,
                rope_mats=self.decode_rope[layer_kind],
                page_table=table,
                kv_cache=cache,
                current_position=rope_lookup_position,
                current_position_cache=cache_position,
                batch_size=batch_size,
            )
            previous.deallocate(True)
        rope_lookup_position.deallocate(True)
        return hidden

    def decode_forward_device_state(
        self,
        token_input: ttnn.Tensor,
        *,
        rope_position: ttnn.Tensor,
        rope_position_lookup_mask: ttnn.Tensor,
        cache_position: ttnn.Tensor,
        page_tables: Sequence[ttnn.Tensor],
        kv_cache: Sequence[Sequence[ttnn.Tensor]],
        batch_size: int,
        advance_position: bool,
    ) -> ttnn.Tensor:
        hidden = self.decode_hidden_device(
            token_input,
            rope_position=rope_position,
            rope_position_lookup_mask=rope_position_lookup_mask,
            cache_position=cache_position,
            page_tables=page_tables,
            kv_cache=kv_cache,
            batch_size=batch_size,
        )
        logits = self._terminal(hidden)
        if advance_position:
            ttnn.plus_one(cache_position, skip_negative_entries=True)
            # Active UINT32 positions advance in place; values with the sign bit
            # set are treated as negative and remain unchanged.
            ttnn.plus_one(rope_position, skip_negative_entries=True)
        return logits

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table,
        kv_cache,
        return_device_logits: bool = False,
    ) -> torch.Tensor | ttnn.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 1 or start_pos.numel() != tokens.shape[0]:
            raise ValueError("decode requires tokens [batch,1] and start_pos [batch]")
        batch_size = int(tokens.shape[0])
        token_input = _replicated_tensor(
            tokens.to(torch.int32).reshape(1, 1, 1, batch_size),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        rope_position = _replicated_tensor(
            _pad_rope_positions(start_pos), self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        rope_position_lookup_mask = _replicated_tensor(
            torch.full((1, 32), ROPE_POSITION_LOOKUP_MASK, dtype=torch.int64),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cache_position = _replicated_tensor(
            start_pos.to(torch.int32).reshape(-1),
            self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        sliced_tables: dict[int, ttnn.Tensor] = {}
        try:
            page_tables = self._normalize_page_tables(page_table)
            decode_tables: list[ttnn.Tensor] = []
            for table in page_tables:
                table_batch = int(table.shape[0])
                if table_batch < batch_size:
                    raise ValueError("decode page tables cannot have fewer rows than the token batch")
                if table_batch == batch_size:
                    decode_tables.append(table)
                    continue
                identity = id(table)
                if identity not in sliced_tables:
                    sliced_tables[identity] = ttnn.slice(table, [0, 0], [batch_size, int(table.shape[1])])
                decode_tables.append(sliced_tables[identity])
            logits = self.decode_forward_device_state(
                token_input,
                rope_position=rope_position,
                rope_position_lookup_mask=rope_position_lookup_mask,
                cache_position=cache_position,
                page_tables=decode_tables,
                kv_cache=kv_cache,
                batch_size=batch_size,
                advance_position=False,
            )
        finally:
            token_input.deallocate(True)
            rope_position.deallocate(True)
            rope_position_lookup_mask.deallocate(True)
            cache_position.deallocate(True)
            for table in sliced_tables.values():
                table.deallocate(True)
        if return_device_logits:
            return logits
        host = self.logits_to_torch(logits).reshape(batch_size, self.vocab_size)
        logits.deallocate(True)
        return host

    def initialize_trace_state(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_tables: Sequence[ttnn.Tensor],
        page_table_generations: Sequence[int],
        prompt_lengths: Sequence[int],
        active_batch_size: int,
    ) -> DecodeTraceState:
        batch_size = int(tokens.shape[0])
        positions = tuple(int(value) for value in start_pos.reshape(-1).tolist())
        lengths = tuple(int(value) for value in prompt_lengths)
        if len(positions) != batch_size or len(lengths) != batch_size:
            raise ValueError("tokens, positions, and prompt lengths must describe the same fixed slots")
        if len(page_table_generations) != len(page_tables):
            raise ValueError("page-table generations must match the layer count")
        if int(active_batch_size) < 1 or int(active_batch_size) != sum(position >= 0 for position in positions):
            raise ValueError("active_batch_size must match non-negative decode positions")
        for position, prompt_length in zip(positions, lengths):
            if position < 0 and (position != -1 or prompt_length != 0):
                raise ValueError("inactive slots require position -1 and prompt length 0")
            if position >= 0 and (prompt_length < 0 or prompt_length > position):
                raise ValueError("active prompt lengths must be non-negative and not exceed position")
        if self.trace_state.trace_id is not None and self.trace_state.batch_size != batch_size:
            self.release_decode_trace()
        if self.trace_state.token_input is None:
            self.trace_state.token_input = _replicated_tensor(
                tokens.to(torch.int32).reshape(1, 1, 1, batch_size),
                self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.trace_state.rope_position = _replicated_tensor(
                _pad_rope_positions(start_pos),
                self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.trace_state.rope_position_lookup_mask = _replicated_tensor(
                torch.full((1, 32), ROPE_POSITION_LOOKUP_MASK, dtype=torch.int64),
                self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            self.trace_state.cache_position = _replicated_tensor(
                start_pos.to(torch.int32).reshape(-1),
                self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            # Trace inputs are model-owned stable buffers.  Never bind the
            # captured graph directly to a scheduler/caller allocation because
            # refreshing it would overwrite the caller's page table.
            stable_tables: dict[int, ttnn.Tensor] = {}
            self.trace_state.page_tables = []
            for table in page_tables:
                identity = id(table)
                if identity not in stable_tables:
                    stable_tables[identity] = ttnn.clone(table)
                self.trace_state.page_tables.append(stable_tables[identity])
            self.trace_state.page_table_identities = [id(table) for table in page_tables]
            self.trace_state.page_table_generations = [int(generation) for generation in page_table_generations]
            self.trace_state.batch_size = batch_size
        else:
            self.write_trace_tokens_from_host(tokens)
            self.write_trace_positions_from_host(start_pos)
        self.trace_state.active_batch_size = int(active_batch_size)
        self.trace_state.prompt_lengths = lengths
        self.trace_state.initial_positions = start_pos.to(torch.int32).reshape(-1).clone()
        self.trace_state.initial_replay_count = int(self.trace_state.counters["model_trace_replays"])
        return self.trace_state

    def write_trace_tokens_from_host(self, tokens: torch.Tensor | Sequence[int]) -> None:
        state = self.trace_state
        if state.token_input is None:
            raise RuntimeError("trace token input is not initialized")
        values = torch.as_tensor(tokens, dtype=torch.int32).reshape(1, 1, 1, state.batch_size)
        _copy_host_to_mesh(ttnn.from_torch(values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), state.token_input)
        state.counters["token_host_refreshes"] += 1

    def write_trace_positions_from_host(self, positions: torch.Tensor | Sequence[int]) -> None:
        state = self.trace_state
        if state.rope_position is None or state.cache_position is None:
            raise RuntimeError("trace position inputs are not initialized")
        values = torch.as_tensor(positions, dtype=torch.int32).reshape(-1)
        rope_host = ttnn.from_torch(_pad_rope_positions(values), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        cache_host = ttnn.from_torch(values, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        _copy_host_to_mesh(rope_host, state.rope_position)
        _copy_host_to_mesh(cache_host, state.cache_position)
        state.counters["position_host_refreshes"] += 1
        state.counters["rope_host_refreshes"] += 1

    def refresh_trace_page_tables(
        self, page_tables: Sequence[ttnn.Tensor], *, generations: Sequence[int] | None = None
    ) -> None:
        state = self.trace_state
        if len(page_tables) != len(state.page_tables):
            raise ValueError("page-table layer count changed after trace capture")
        if generations is None or len(generations) != len(page_tables):
            raise ValueError("page-table generations must be explicit and match the layer count")
        generations = list(generations)
        copied_targets: set[tuple[int, int, int]] = set()
        for index, (source, target, generation) in enumerate(zip(page_tables, state.page_tables, generations)):
            source_identity = id(source)
            if (
                source_identity == state.page_table_identities[index]
                and int(generation) == state.page_table_generations[index]
            ):
                continue
            if source is target:
                # The stable trace buffer was updated in place. Record the new
                # generation without copying a tensor onto itself.
                state.page_table_identities[index] = source_identity
                state.page_table_generations[index] = int(generation)
                continue
            copy_key = (id(source), id(target), int(generation))
            if copy_key in copied_targets:
                state.page_table_identities[index] = source_identity
                state.page_table_generations[index] = int(generation)
                continue
            # Submit one distributed operation so every device observes the
            # same mesh command ordering before the decode trace is replayed.
            # Per-physical-device copies can leave the mesh on divergent op
            # IDs and deadlock the following trace.
            ttnn.copy(source, target)
            copied_targets.add(copy_key)
            state.page_table_identities[index] = source_identity
            state.page_table_generations[index] = int(generation)
            state.counters["page_table_refreshes"] += 1

    def capture_decode_trace(self, *, kv_cache) -> DecodeTraceState:
        state = self.trace_state
        cache_identity = _kv_cache_identity(kv_cache)
        if state.trace_id is not None:
            if state.kv_cache_identity != cache_identity:
                raise ValueError("decode trace is bound to a different KV cache allocation")
            return state
        if any(
            value is None
            for value in (state.token_input, state.rope_position, state.rope_position_lookup_mask, state.cache_position)
        ):
            raise RuntimeError("initialize_trace_state must be called before capture")
        state.kv_cache_identity = cache_identity
        warmup_logits = self.decode_forward_device_state(
            state.token_input,
            rope_position=state.rope_position,
            rope_position_lookup_mask=state.rope_position_lookup_mask,
            cache_position=state.cache_position,
            page_tables=state.page_tables,
            kv_cache=kv_cache,
            batch_size=state.batch_size,
            advance_position=True,
        )
        ttnn.synchronize_device(self.mesh_device)
        state.counters["synchronizations"] += 1
        warmup_logits.deallocate(True)
        if state.initial_positions is None:
            raise RuntimeError("initial trace positions were lost")
        self.write_trace_positions_from_host(state.initial_positions)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        state.logits = self.decode_forward_device_state(
            state.token_input,
            rope_position=state.rope_position,
            rope_position_lookup_mask=state.rope_position_lookup_mask,
            cache_position=state.cache_position,
            page_tables=state.page_tables,
            kv_cache=kv_cache,
            batch_size=state.batch_size,
            advance_position=True,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        state.trace_id = trace_id
        return state

    def execute_decode_trace(self) -> ttnn.Tensor:
        state = self.trace_state
        if state.trace_id is None or state.logits is None:
            raise RuntimeError("decode trace is not captured")
        ttnn.execute_trace(self.mesh_device, state.trace_id, cq_id=0, blocking=False)
        state.counters["model_trace_replays"] += 1
        return state.logits

    def release_decode_trace(self) -> None:
        state = self.trace_state
        if state.trace_id is not None:
            ttnn.release_trace(self.mesh_device, state.trace_id)
        for tensor in (
            state.logits,
            state.token_input,
            state.rope_position,
            state.rope_position_lookup_mask,
            state.cache_position,
            *state.page_tables,
        ):
            if tensor is not None and tensor.is_allocated():
                tensor.deallocate(True)
        self.trace_state = DecodeTraceState()

    def zero_kv_cache(self, kv_cache: Iterable[Sequence[ttnn.Tensor]]) -> None:
        for cache_pair in kv_cache:
            for cache in cache_pair:
                ttnn.fill(cache, 0.0, memory_config=cache.memory_config(), output_tensor=cache)

    def teardown(self) -> None:
        self.release_decode_trace()
        release_multichip_decoder_resources(self.mesh_device)


__all__ = [
    "DecodeTraceState",
    "Gemma4FullModel",
    "Gemma4FullModelConfig",
    "HF_MODEL_ID",
    "ROPE_POSITION_INACTIVE_SENTINEL",
    "_resolve_checkpoint",
]
