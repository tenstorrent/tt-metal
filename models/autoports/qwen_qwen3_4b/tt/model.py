# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import HF_MODEL_ID, RMS_NORM_EPS, Qwen3DecoderConfig
from models.autoports.qwen_qwen3_4b.tt.multichip_decoder import DEFAULT_MULTICHIP_KV_CONFIG, MultichipDecoder
from models.autoports.qwen_qwen3_4b.tt.optimized_decoder import PagedKVConfig
from models.common.lightweightmodule import LightweightModule


def _pad_to(value: int, multiple: int) -> int:
    remainder = value % multiple
    return value if remainder == 0 else value + multiple - remainder


def _mesh_replicated_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.detach().contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _load_hf_state_dict(hf_model_id: str) -> tuple[Any, dict[str, torch.Tensor]]:
    hf_config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    del model
    return hf_config, state_dict


@dataclass(frozen=True)
class Qwen3FullModelConfig:
    hf_model_id: str = HF_MODEL_ID
    max_seq_len: int = DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len
    paged_kv_config: PagedKVConfig = DEFAULT_MULTICHIP_KV_CONFIG
    num_layers: int | None = None
    lm_head_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    lm_head_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    attention_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    mlp_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi
    topology: ttnn.Topology = ttnn.Topology.Ring
    num_links: int = 1
    trace_region_size_bytes: int = 128 << 20


@dataclass
class Qwen3FullModelTimings:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    traced_decode_ms: float | None = None
    token_out_decode_ms: float | None = None


@dataclass
class DecodeTraceState:
    trace_id: int | None = None
    logits: ttnn.Tensor | None = None
    token_input: ttnn.Tensor | None = None
    current_pos: ttnn.Tensor | None = None
    rope_pos: ttnn.Tensor | None = None
    position_cos: ttnn.Tensor | None = None
    position_sin: ttnn.Tensor | None = None
    page_table: ttnn.Tensor | None = None
    prompt_lens: list[int] = field(default_factory=list)
    start_pos_host: torch.Tensor | None = None
    batch_size: int = 0
    page_table_generation: int = 0
    page_table_identity: int | None = None
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "trace_replays": 0,
            "token_host_refreshes": 0,
            "position_host_refreshes": 0,
            "page_table_host_refreshes": 0,
            "syncs": 0,
            "readbacks": 0,
        }
    )


class Qwen3FullModel(LightweightModule):
    """Qwen3-4B full autoregressive wrapper around the optimized TP4 decoder stack."""

    def __init__(
        self,
        *,
        hf_config,
        state_dict: dict[str, torch.Tensor],
        mesh_device,
        config: Qwen3FullModelConfig | None = None,
    ) -> None:
        self.config = config or Qwen3FullModelConfig()
        self.hf_config = hf_config
        self.decoder_cfg = Qwen3DecoderConfig.from_hf_config(hf_config)
        self.mesh_device = mesh_device
        self.tp = mesh_device.get_num_devices()
        if tuple(mesh_device.shape) != (1, 4) or self.tp != 4:
            raise ValueError(
                f"Qwen3FullModel requires the optimized 1x4 TP4 mesh, got shape={tuple(mesh_device.shape)}"
            )
        if self.config.max_seq_len > self.config.paged_kv_config.max_seq_len:
            raise ValueError(
                f"max_seq_len={self.config.max_seq_len} exceeds paged KV capacity "
                f"{self.config.paged_kv_config.max_seq_len}"
            )

        self.num_layers = self.config.num_layers or int(hf_config.num_hidden_layers)
        self.vocab_size = int(hf_config.vocab_size)
        self.padded_vocab_size = _pad_to(self.vocab_size, 32 * self.tp)
        self.vocab_per_device = self.padded_vocab_size // self.tp
        self.timings = Qwen3FullModelTimings()
        self.trace_state = DecodeTraceState()

        embedding = state_dict["model.embed_tokens.weight"].to(torch.bfloat16)
        if self.padded_vocab_size != self.vocab_size:
            padding = torch.zeros(
                self.padded_vocab_size - self.vocab_size,
                embedding.shape[1],
                dtype=embedding.dtype,
            )
            embedding = torch.cat([embedding, padding], dim=0)

        self.embed_tokens_weight = _mesh_replicated_tensor(
            embedding,
            mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.final_norm_weight = _mesh_replicated_tensor(state_dict["model.norm.weight"], mesh_device)
        self.decode_position_cos, self.decode_position_sin = self._build_decode_rope_embedding_tables()
        self.lm_head_weight = self._load_lm_head_weight(state_dict, embedding)
        self.compute_kernel_config_lofi = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=self.config.lm_head_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.layers = [
            MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_seq_len=self.config.max_seq_len,
                paged_kv_config=self.config.paged_kv_config,
                attention_weight_dtype=self.config.attention_weight_dtype,
                mlp_weight_dtype=self.config.mlp_weight_dtype,
                attention_math_fidelity=self.config.attention_math_fidelity,
                mlp_math_fidelity=self.config.mlp_math_fidelity,
                topology=self.config.topology,
                num_links=self.config.num_links,
                static_position_table_seq_len=1,
            )
            for layer_idx in range(self.num_layers)
        ]
        shared_all_reduce_cache = {}
        shared_all_reduce_index = {}
        for layer in self.layers:
            layer._persistent_all_reduce_cache = shared_all_reduce_cache
            layer._persistent_all_reduce_index = shared_all_reduce_index

    @classmethod
    def from_pretrained(
        cls,
        *,
        mesh_device,
        hf_model_id: str = HF_MODEL_ID,
        hf_config=None,
        state_dict: dict[str, torch.Tensor] | None = None,
        config: Qwen3FullModelConfig | None = None,
        **kwargs,
    ) -> "Qwen3FullModel":
        if kwargs:
            raise TypeError(f"unsupported Qwen3FullModel kwargs: {sorted(kwargs)}")
        if state_dict is None:
            hf_config, state_dict = _load_hf_state_dict(hf_model_id)
        elif hf_config is None:
            hf_config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        cfg = config or Qwen3FullModelConfig(hf_model_id=hf_model_id)
        return cls(hf_config=hf_config, state_dict=state_dict, mesh_device=mesh_device, config=cfg)

    def _load_lm_head_weight(self, state_dict: dict[str, torch.Tensor], embedding: torch.Tensor) -> ttnn.Tensor:
        output_weight = state_dict.get("lm_head.weight")
        if output_weight is None:
            output_weight = embedding
        else:
            output_weight = output_weight.to(torch.bfloat16)
            if output_weight.shape[0] != self.padded_vocab_size:
                padding = torch.zeros(
                    self.padded_vocab_size - output_weight.shape[0],
                    output_weight.shape[1],
                    dtype=output_weight.dtype,
                )
                output_weight = torch.cat([output_weight, padding], dim=0)
        transposed = output_weight.transpose(0, 1).contiguous()
        return ttnn.from_torch(
            transposed,
            dtype=self.config.lm_head_weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
        )

    def _build_decode_rope_embedding_tables(self) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        positions = torch.arange(self.config.max_seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.decoder_cfg.rope_theta
            ** (torch.arange(0, self.decoder_cfg.head_dim, 2, dtype=torch.float32) / self.decoder_cfg.head_dim)
        )
        angles = torch.einsum("s,d->sd", positions, inv_freq)
        emb = torch.cat((angles, angles), dim=-1)
        return (
            _mesh_replicated_tensor(torch.cos(emb), self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
            _mesh_replicated_tensor(torch.sin(emb), self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT),
        )

    def _lm_head_weight_memory_config(self, k: int, n_per_device: int) -> ttnn.MemoryConfig:
        dram_banks = self.mesh_device.dram_grid_size().x * self.mesh_device.dram_grid_size().y
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(self.mesh_device.dram_grid_size().x - 1, self.mesh_device.dram_grid_size().y - 1),
                )
            }
        )
        shard_shape = [k, _pad_to(n_per_device, 32 * dram_banks) // dram_banks]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _lm_head_program_config(self, seq_or_batch: int) -> ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig:
        grid = self.mesh_device.compute_with_storage_grid_size()
        k_tiles = self.decoder_cfg.hidden_size // 32
        n_tiles = self.vocab_per_device // 32
        num_cores = grid.x * grid.y
        in0_block_w = min(32, k_tiles)
        while k_tiles % in0_block_w != 0:
            in0_block_w //= 2
        per_core_n = math.ceil(n_tiles / num_cores)
        out_subblock_w = min(4, per_core_n)
        while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
            out_subblock_w -= 1
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=max(1, math.ceil(seq_or_batch / 32)),
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def _lm_head_output_memory_config(self, seq_or_batch: int) -> ttnn.MemoryConfig:
        grid = self.mesh_device.compute_with_storage_grid_size()
        n_tiles = self.vocab_per_device // 32
        num_cores = grid.x * grid.y
        per_core_n = math.ceil(n_tiles / num_cores)
        active_cores = math.ceil(n_tiles / per_core_n)
        shard_grid = _core_range_set_for_row_major_cores(active_cores, grid)
        shard_shape = [
            max(32, _pad_to(seq_or_batch, 32)),
            per_core_n * 32,
        ]
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    def _tokens_to_device(self, tokens: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tokens.to(torch.uint32).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _decode_tokens_to_device(self, tokens: torch.Tensor) -> ttnn.Tensor:
        tokens = tokens.to(torch.uint32)
        if int(tokens.shape[0]) == 1:
            tokens = tokens.reshape(1, 1, 1, 1).contiguous()
        else:
            tokens = tokens.reshape(tokens.shape[0], -1).contiguous()
        return ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def embed_tokens(self, tokens: torch.Tensor | ttnn.Tensor, *, mode: str) -> ttnn.Tensor:
        if isinstance(tokens, torch.Tensor):
            if mode == "decode" and tokens.shape[0] > 1:
                rows = []
                for row in tokens:
                    tt_row = self._decode_tokens_to_device(row.reshape(1, -1))
                    row_hidden = ttnn.embedding(
                        tt_row,
                        self.embed_tokens_weight,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    rows.append(ttnn.reshape(row_hidden, [1, 1, 1, self.decoder_cfg.hidden_size]))
                return ttnn.concat(rows, dim=2)
            tt_tokens = self._tokens_to_device(tokens) if mode == "prefill" else self._decode_tokens_to_device(tokens)
            batch_size = int(tokens.shape[0])
        else:
            tt_tokens = tokens
            batch_size = int(tokens.shape[-2]) if mode == "decode" else int(tokens.shape[0])
        hidden = ttnn.embedding(
            tt_tokens,
            self.embed_tokens_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if mode == "prefill":
            return ttnn.reshape(hidden, [1, 1, hidden.shape[-2], self.decoder_cfg.hidden_size])
        if mode == "decode":
            return ttnn.reshape(hidden, [1, 1, batch_size, self.decoder_cfg.hidden_size])
        raise ValueError(f"unsupported embed mode: {mode!r}")

    def init_paged_kv_cache(self) -> list[list[ttnn.Tensor]]:
        return [layer.init_paged_kv_cache() for layer in self.layers]

    def make_identity_page_table(self, batch_size: int = 1) -> ttnn.Tensor:
        return self.layers[0].make_identity_page_table(batch_size=batch_size)

    def make_current_pos(self, positions: list[int] | torch.Tensor) -> ttnn.Tensor:
        return self.layers[0].make_current_pos(positions)

    def position_tables_for_decode(self, position: int, *, batch_size: int = 1) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position < 0 or position >= self.config.max_seq_len:
            raise ValueError(f"decode position must be in [0, {self.config.max_seq_len}), got {position}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        positions = torch.full((batch_size,), int(position), dtype=torch.uint32)
        rope_pos = ttnn.from_torch(
            positions,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return self.position_tables_from_device_pos(rope_pos, batch_size=batch_size)

    def position_tables_from_device_pos(
        self, rope_pos: ttnn.Tensor, *, batch_size: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cos = ttnn.embedding(
            rope_pos,
            self.decode_position_cos,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sin = ttnn.embedding(
            rope_pos,
            self.decode_position_sin,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return (
            ttnn.reshape(cos, [1, 1, batch_size, self.decoder_cfg.head_dim]),
            ttnn.reshape(sin, [1, 1, batch_size, self.decoder_cfg.head_dim]),
        )

    def position_tables_for_prefill(self, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if seq_len <= 0 or seq_len > self.config.max_seq_len:
            raise ValueError(f"prefill seq_len must be in [1, {self.config.max_seq_len}], got {seq_len}")
        positions = torch.arange(seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.decoder_cfg.rope_theta
            ** (torch.arange(0, self.decoder_cfg.head_dim, 2, dtype=torch.float32) / self.decoder_cfg.head_dim)
        )
        angles = torch.einsum("d,s->sd", inv_freq, positions)
        emb = torch.cat((angles, angles), dim=-1).reshape(1, 1, seq_len, self.decoder_cfg.head_dim)
        return (
            _mesh_replicated_tensor(torch.cos(emb), self.mesh_device),
            _mesh_replicated_tensor(torch.sin(emb), self.mesh_device),
        )

    def prefill_hidden(
        self,
        tokens: torch.Tensor | ttnn.Tensor,
        *,
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        page_table: ttnn.Tensor | None = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        hidden = self.embed_tokens(tokens, mode="prefill")
        position_cos, position_sin = self.position_tables_for_prefill(int(hidden.shape[-2]))
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.prefill_forward(
                hidden,
                position_cos=position_cos,
                position_sin=position_sin,
                kv_cache=None if kv_cache is None else kv_cache[layer_idx],
                page_table=page_table,
                user_id=user_id,
            )
        return hidden

    def decode_hidden(
        self,
        tokens: torch.Tensor | ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[list[ttnn.Tensor]],
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        hidden = self.embed_tokens(tokens, mode="decode")
        for layer_idx, layer in enumerate(self.layers):
            hidden = layer.decode_forward(
                hidden,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=kv_cache[layer_idx],
                position_cos=position_cos,
                position_sin=position_sin,
            )
        return hidden

    def apply_final_norm_and_lm_head(self, hidden: ttnn.Tensor, *, output_tile=None) -> ttnn.Tensor:
        normed = ttnn.rms_norm(
            hidden,
            epsilon=RMS_NORM_EPS,
            weight=self.final_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_lofi,
        )
        logits = ttnn.linear(
            normed,
            self.lm_head_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_lofi,
            output_tile=output_tile,
        )
        return logits

    def logits_to_torch(self, logits: ttnn.Tensor, *, trim_vocab: bool = True) -> torch.Tensor:
        torch_logits = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1))
        torch_logits = torch_logits.to(torch.float32)
        if trim_vocab:
            torch_logits = torch_logits[..., : self.vocab_size]
        return torch_logits

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table: ttnn.Tensor | None = None,
        kv_cache: list[list[ttnn.Tensor]] | None = None,
        prompt_lens: list[int] | None = None,
        return_all_logits: bool = False,
        user_id: int = 0,
    ) -> torch.Tensor:
        if tokens.dim() != 2:
            raise ValueError(f"prefill tokens must be [batch, seq], got {tuple(tokens.shape)}")
        if tokens.shape[0] != 1:
            prompt_lens = prompt_lens or [int(tokens.shape[1])] * int(tokens.shape[0])
            if len(prompt_lens) != int(tokens.shape[0]):
                raise ValueError("prompt_lens must have one entry per prefill row")
            row_outputs = [
                self.prefill_forward(
                    tokens[row_idx : row_idx + 1, : int(prompt_lens[row_idx])],
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=[int(prompt_lens[row_idx])],
                    return_all_logits=return_all_logits,
                    user_id=user_id + row_idx,
                )
                for row_idx in range(int(tokens.shape[0]))
            ]
            if return_all_logits:
                max_len = max(output.shape[1] for output in row_outputs)
                padded_outputs = []
                for output in row_outputs:
                    if output.shape[1] == max_len:
                        padded_outputs.append(output)
                    else:
                        pad_shape = (output.shape[0], max_len - output.shape[1], output.shape[2])
                        padded_outputs.append(torch.cat([output, torch.zeros(pad_shape, dtype=output.dtype)], dim=1))
                return torch.cat(padded_outputs, dim=0)
            return torch.cat(row_outputs, dim=0)
        prompt_len = int(prompt_lens[0]) if prompt_lens else int(tokens.shape[1])
        logical_tokens = tokens[:, :prompt_len]
        padded_len = _pad_to(prompt_len, max(32, self.config.paged_kv_config.block_size))
        if padded_len != prompt_len:
            pad_width = padded_len - prompt_len
            pad_value = 0
            logical_tokens = torch.nn.functional.pad(logical_tokens, (0, pad_width), value=pad_value)
        hidden = self.prefill_hidden(logical_tokens, kv_cache=kv_cache, page_table=page_table, user_id=user_id)
        if not return_all_logits:
            last_tile_start = ((prompt_len - 1) // 32) * 32
            hidden = ttnn.slice(
                hidden,
                [0, 0, last_tile_start, 0],
                [1, 1, last_tile_start + 32, self.decoder_cfg.hidden_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        logits = self.apply_final_norm_and_lm_head(hidden)
        torch_logits = self.logits_to_torch(logits)
        if return_all_logits:
            return torch_logits.reshape(1, padded_len, self.vocab_size)[:, :prompt_len, :]
        return torch_logits.reshape(1, -1, self.vocab_size)[:, (prompt_len - 1) % 32 : (prompt_len - 1) % 32 + 1, :]

    def decode_forward(
        self,
        tokens: torch.Tensor | ttnn.Tensor,
        start_pos: torch.Tensor | ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        kv_cache: list[list[ttnn.Tensor]],
        return_device_logits: bool = False,
    ) -> torch.Tensor | ttnn.Tensor:
        batch_size = int(tokens.shape[0]) if isinstance(tokens, torch.Tensor) else int(tokens.shape[-2])
        if isinstance(start_pos, torch.Tensor):
            if start_pos.numel() != batch_size:
                raise ValueError("start_pos must have one entry per decode row")
            start_pos = start_pos.reshape(batch_size)
            current_pos = self.make_current_pos(start_pos.to(torch.int32))
            rope_pos = ttnn.from_torch(
                start_pos.to(torch.uint32).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            position_cos, position_sin = self.position_tables_from_device_pos(rope_pos, batch_size=batch_size)
        else:
            current_pos = start_pos
            position_host = ttnn.to_torch(ttnn.get_device_tensors(start_pos)[0]).reshape(-1).to(torch.uint32)
            if position_host.numel() != batch_size:
                raise ValueError("start_pos must have one entry per decode row")
            rope_pos = ttnn.from_torch(
                position_host.contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            position_cos, position_sin = self.position_tables_from_device_pos(rope_pos, batch_size=batch_size)
        hidden = self.decode_hidden(
            tokens,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        logits = self.apply_final_norm_and_lm_head(hidden)
        if return_device_logits:
            return logits
        return self.logits_to_torch(logits).reshape(batch_size, self.vocab_size)

    def decode_forward_device_state(
        self,
        token_input: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        rope_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[list[ttnn.Tensor]],
        advance_position: bool = False,
    ) -> ttnn.Tensor:
        batch_size = int(token_input.shape[-2])
        position_cos, position_sin = self.position_tables_from_device_pos(rope_pos, batch_size=batch_size)
        hidden = self.decode_hidden(
            token_input,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            position_cos=position_cos,
            position_sin=position_sin,
        )
        logits = self.apply_final_norm_and_lm_head(hidden)
        if advance_position:
            ttnn.plus_one(current_pos)
            ttnn.plus_one(rope_pos)
        return logits

    def reset_decode_trace_state(
        self,
        *,
        token_input: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: ttnn.Tensor,
        prompt_lens: list[int],
    ) -> DecodeTraceState:
        self.trace_state = DecodeTraceState(
            trace_id=None,
            logits=None,
            token_input=self._decode_tokens_to_device(token_input),
            current_pos=self.make_current_pos(start_pos.to(torch.int32)),
            rope_pos=ttnn.from_torch(
                start_pos.to(torch.uint32).contiguous(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            ),
            page_table=page_table,
            prompt_lens=list(prompt_lens),
            start_pos_host=start_pos.to(torch.int32).contiguous().clone(),
            batch_size=int(token_input.shape[0]),
            page_table_generation=self.trace_state.page_table_generation + 1,
            page_table_identity=id(page_table),
        )
        return self.trace_state

    def _reset_trace_positions_from_host(self) -> None:
        state = self.trace_state
        if state.start_pos_host is None or state.current_pos is None or state.rope_pos is None:
            raise RuntimeError("decode trace positions are not initialized")
        current_host = ttnn.from_torch(
            state.start_pos_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        rope_host = ttnn.from_torch(
            state.start_pos_host.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(current_host, state.current_pos)
        ttnn.copy_host_to_device_tensor(rope_host, state.rope_pos)
        state.counters["position_host_refreshes"] += 1

    def capture_decode_trace(self, *, kv_cache: list[list[ttnn.Tensor]]) -> DecodeTraceState:
        state = self.trace_state
        if state.token_input is None or state.current_pos is None or state.rope_pos is None or state.page_table is None:
            raise RuntimeError("decode trace state has not been reset")
        warmup = self.decode_forward_device_state(
            state.token_input,
            current_pos=state.current_pos,
            rope_pos=state.rope_pos,
            page_table=state.page_table,
            kv_cache=kv_cache,
            advance_position=True,
        )
        ttnn.synchronize_device(self.mesh_device)
        state.counters["syncs"] += 1
        self._reset_trace_positions_from_host()
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        logits = self.decode_forward_device_state(
            state.token_input,
            current_pos=state.current_pos,
            rope_pos=state.rope_pos,
            page_table=state.page_table,
            kv_cache=kv_cache,
            advance_position=True,
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(self.mesh_device)
        state.trace_id = trace_id
        state.logits = logits
        state.counters["syncs"] += 1
        return state

    def execute_decode_trace(self) -> ttnn.Tensor:
        state = self.trace_state
        if state.trace_id is None or state.logits is None:
            raise RuntimeError("decode trace has not been captured")
        ttnn.execute_trace(self.mesh_device, state.trace_id, cq_id=0, blocking=False)
        state.counters["trace_replays"] += 1
        return state.logits

    def refresh_trace_page_table(self, page_table: ttnn.Tensor, *, generation: int | None = None) -> None:
        state = self.trace_state
        if state.page_table is None:
            raise RuntimeError("decode trace page table is not initialized")
        if page_table is state.page_table or id(page_table) == state.page_table_identity:
            if generation is not None:
                state.page_table_generation = generation
            return
        ttnn.copy(page_table, state.page_table)
        state.page_table_identity = id(page_table)
        if generation is not None:
            state.page_table_generation = generation
        else:
            state.page_table_generation += 1
        state.counters["page_table_host_refreshes"] += 1

    def write_trace_token_from_host(self, token: int) -> None:
        state = self.trace_state
        if state.token_input is None:
            raise RuntimeError("decode trace token input is not initialized")
        host_token = torch.tensor([[[[int(token)]]]], dtype=torch.uint32)
        host_tensor = ttnn.from_torch(
            host_token,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host_tensor, state.token_input)
        state.counters["token_host_refreshes"] += 1

    def write_context_contract(self, model_dir: str | Path) -> Path:
        """Update doc/context_contract.json with full-model persistent memory accounting."""
        model_dir = Path(model_dir)
        path = model_dir / "doc" / "context_contract.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        embedding_bytes = self.padded_vocab_size * self.decoder_cfg.hidden_size * 2
        lm_head_bytes_per_device = self.decoder_cfg.hidden_size * self.vocab_per_device // 2
        final_norm_bytes = self.decoder_cfg.hidden_size * 2
        data["decode_status"] = "full_model_optimized_multichip_autoregressive_path_in_progress"
        data["capacity_evidence"]["full_model_weight_estimate"] = {
            "status": "full-model wrapper weights added",
            "embedding_weight_bytes_replicated_per_device": embedding_bytes,
            "lm_head_weight_bytes_per_device_bfloat4_b": lm_head_bytes_per_device,
            "final_norm_weight_bytes_replicated_per_device": final_norm_bytes,
            "decoder_layers": self.num_layers,
            "note": "Embedding is BF16 replicated; tied LM head is BFP4 vocab-sharded over TP4.",
        }
        data["current_supported_context"] = self.config.max_seq_len
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return path
