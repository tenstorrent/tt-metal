# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import RMS_NORM_EPS, Qwen3DecoderConfig, _get_layer_tensor
from models.autoports.qwen_qwen3_4b.tt.optimized_decoder import OptimizedDecoder, PagedKVConfig
from models.common.lightweightmodule import LightweightModule

try:
    from tracy import signpost
except ImportError:  # pragma: no cover - tracy is optional outside profiling runs.

    def signpost(header):
        return None


DEFAULT_MULTICHIP_KV_CONFIG = PagedKVConfig(max_num_blocks=2560, block_size=16)


@dataclass(frozen=True)
class MultichipDecoderTimings:
    prefill_ms: float | None = None
    decode_ms: float | None = None
    traced_decode_ms: float | None = None


class MultichipDecoder(LightweightModule):
    """Qwen3-4B dense decoder layer with 1x4 tensor parallel execution.

    This stage intentionally starts from the optimized single-chip decoder's
    math and precision policy, but shards projection weights and KV heads across
    a four-device ring. Layer inputs and outputs are replicated hidden-state
    tensors so stacked decoder layers can be composed without boundary gathers.
    """

    baseline_cls = OptimizedDecoder
    mesh_profile = {
        "name": "qwen3_4b_multichip_decoder_1x4_tp4_v1",
        "single_chip_baseline": OptimizedDecoder.optimization_profile["name"],
        "target_mesh": "1x4 Blackhole ring",
        "tp": 4,
        "activation_contract": "replicated hidden state at layer input/output; local TP shards inside attention/MLP",
        "attention": "QKV column parallel with per-device Q/K/V packing; local paged SDPA; row-parallel WO + ring all_reduce",
        "mlp": "gate/up column parallel; down row parallel; ring all_reduce",
        "kv_cache": "paged BF16 cache with local KV heads per device",
        "weight_dtype": "bfloat4_b",
        "math_fidelity": "LoFi",
    }

    def __init__(
        self,
        *,
        cfg: Qwen3DecoderConfig,
        layer_idx: int,
        mesh_device,
        qkv_prefill_weight: ttnn.Tensor,
        qkv_decode_weight: ttnn.Tensor,
        o_proj_weight: ttnn.Tensor,
        gate_proj_weight: ttnn.Tensor,
        up_proj_weight: ttnn.Tensor,
        down_proj_weight: ttnn.Tensor,
        q_norm_weight: ttnn.Tensor,
        k_norm_weight: ttnn.Tensor,
        input_layernorm_weight: ttnn.Tensor,
        post_attention_layernorm_weight: ttnn.Tensor,
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        max_seq_len: int,
        paged_kv_config: PagedKVConfig,
        attention_math_fidelity: ttnn.MathFidelity,
        mlp_math_fidelity: ttnn.MathFidelity,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        num_links: int = 1,
    ) -> None:
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.tp = mesh_device.get_num_devices()
        if tuple(mesh_device.shape) != (1, 4) or self.tp != 4:
            raise ValueError(f"MultichipDecoder targets a 1x4 mesh, got shape={tuple(mesh_device.shape)}")
        if cfg.num_attention_heads % self.tp != 0 or cfg.num_key_value_heads % self.tp != 0:
            raise ValueError("attention and KV heads must divide evenly across TP=4")
        if cfg.intermediate_size % self.tp != 0:
            raise ValueError("MLP intermediate size must divide evenly across TP=4")

        self.local_num_attention_heads = cfg.num_attention_heads // self.tp
        self.local_num_key_value_heads = cfg.num_key_value_heads // self.tp
        self.local_q_width = self.local_num_attention_heads * cfg.head_dim
        self.local_kv_width = self.local_num_key_value_heads * cfg.head_dim
        self.local_qkv_width = self.local_q_width + 2 * self.local_kv_width
        self.local_intermediate_size = cfg.intermediate_size // self.tp

        self.qkv_prefill_weight = qkv_prefill_weight
        self.qkv_decode_weight = qkv_decode_weight
        self.o_proj_weight = o_proj_weight
        self.gate_proj_weight = gate_proj_weight
        self.up_proj_weight = up_proj_weight
        self.down_proj_weight = down_proj_weight
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.input_layernorm_weight = input_layernorm_weight
        self.post_attention_layernorm_weight = post_attention_layernorm_weight
        self.position_cos = position_cos
        self.position_sin = position_sin
        self.attention_mask = attention_mask
        self.max_seq_len = max_seq_len
        self.paged_kv_config = paged_kv_config
        self.topology = topology
        self.num_links = num_links
        self.timings = MultichipDecoderTimings()

        self.compute_kernel_config_hifi2 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_lofi = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.mlp_compute_kernel_config = (
            self.compute_kernel_config_lofi
            if mlp_math_fidelity == ttnn.MathFidelity.LoFi
            else self.compute_kernel_config_hifi2
        )
        self.attention_compute_kernel_config = (
            self.compute_kernel_config_lofi
            if attention_math_fidelity == ttnn.MathFidelity.LoFi
            else self.compute_kernel_config_hifi2
        )
        self.auxiliary_compute_kernel_config = self.compute_kernel_config_lofi
        self.sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    @staticmethod
    def _mesh_replicated_tensor(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            tensor.detach().contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def _mesh_sharded_tensor(
        tensor: torch.Tensor,
        mesh_device,
        *,
        dim: int,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    ):
        return ttnn.from_torch(
            tensor.detach().contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=dim),
        )

    @classmethod
    def _build_rope_tables(cls, cfg: Qwen3DecoderConfig, seq_len: int, mesh_device) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        angles = torch.einsum("d,s->sd", inv_freq, positions)
        emb = torch.cat((angles, angles), dim=-1).reshape(1, 1, seq_len, cfg.head_dim)
        return (
            cls._mesh_replicated_tensor(torch.cos(emb), mesh_device),
            cls._mesh_replicated_tensor(torch.sin(emb), mesh_device),
        )

    @classmethod
    def _build_causal_mask(cls, seq_len: int, mesh_device) -> ttnn.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
        return cls._mesh_replicated_tensor(torch.triu(mask, diagonal=1), mesh_device)

    @staticmethod
    def _pack_qkv_by_device(
        q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor, *, tp: int, prefill: bool
    ):
        q_shards = torch.chunk(q_proj.transpose(0, 1).detach().contiguous(), tp, dim=1)
        k_shards = torch.chunk(k_proj.transpose(0, 1).detach().contiguous(), tp, dim=1)
        v_shards = torch.chunk(v_proj.transpose(0, 1).detach().contiguous(), tp, dim=1)
        packed = []
        for q, k, v in zip(q_shards, k_shards, v_shards):
            packed.append(torch.cat((v, q, k), dim=1) if prefill else torch.cat((q, k, v), dim=1))
        return torch.cat(packed, dim=1)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        max_seq_len: int = DEFAULT_MULTICHIP_KV_CONFIG.max_seq_len,
        paged_kv_config: PagedKVConfig | None = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        attention_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        mlp_weight_dtype: ttnn.DataType | None = ttnn.bfloat4_b,
        attention_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        mlp_math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.LoFi,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Ring,
        **kwargs,
    ) -> "MultichipDecoder":
        if kwargs:
            raise TypeError(f"unsupported MultichipDecoder kwargs: {sorted(kwargs)}")
        cfg = Qwen3DecoderConfig.from_hf_config(hf_config)
        if max_seq_len <= 0 or max_seq_len > cfg.max_position_embeddings:
            raise ValueError(f"max_seq_len must be in [1, {cfg.max_position_embeddings}], got {max_seq_len}")
        paged_kv_config = paged_kv_config or DEFAULT_MULTICHIP_KV_CONFIG
        if max_seq_len > paged_kv_config.max_seq_len:
            raise ValueError(
                f"max_seq_len={max_seq_len} exceeds paged KV capacity {paged_kv_config.max_seq_len}; "
                "increase PagedKVConfig.max_num_blocks or lower max_seq_len"
            )
        attention_weight_dtype = attention_weight_dtype or weight_dtype
        mlp_weight_dtype = mlp_weight_dtype or weight_dtype
        tp = mesh_device.get_num_devices()
        if tp != 4:
            raise ValueError(f"MultichipDecoder expects TP=4, got {tp}")

        q_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_proj = _get_layer_tensor(state_dict, layer_idx, "self_attn.v_proj.weight")
        qkv_prefill = cls._pack_qkv_by_device(q_proj, k_proj, v_proj, tp=tp, prefill=True)
        qkv_decode = cls._pack_qkv_by_device(q_proj, k_proj, v_proj, tp=tp, prefill=False)

        position_cos, position_sin = cls._build_rope_tables(cfg, max_seq_len, mesh_device)
        attention_mask = cls._build_causal_mask(max_seq_len, mesh_device)

        return cls(
            cfg=cfg,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            qkv_prefill_weight=cls._mesh_sharded_tensor(qkv_prefill, mesh_device, dim=1, dtype=attention_weight_dtype),
            qkv_decode_weight=cls._mesh_sharded_tensor(qkv_decode, mesh_device, dim=1, dtype=attention_weight_dtype),
            o_proj_weight=cls._mesh_sharded_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.o_proj.weight"),
                mesh_device,
                dim=1,
                dtype=attention_weight_dtype,
            ),
            gate_proj_weight=cls._mesh_sharded_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.gate_proj.weight"),
                mesh_device,
                dim=0,
                dtype=mlp_weight_dtype,
            ),
            up_proj_weight=cls._mesh_sharded_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.up_proj.weight"),
                mesh_device,
                dim=0,
                dtype=mlp_weight_dtype,
            ),
            down_proj_weight=cls._mesh_sharded_tensor(
                _get_layer_tensor(state_dict, layer_idx, "mlp.down_proj.weight"),
                mesh_device,
                dim=1,
                dtype=mlp_weight_dtype,
            ),
            q_norm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.q_norm.weight"), mesh_device
            ),
            k_norm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "self_attn.k_norm.weight"), mesh_device
            ),
            input_layernorm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "input_layernorm.weight"), mesh_device
            ),
            post_attention_layernorm_weight=cls._mesh_replicated_tensor(
                _get_layer_tensor(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device
            ),
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=attention_mask,
            max_seq_len=max_seq_len,
            paged_kv_config=paged_kv_config,
            attention_math_fidelity=attention_math_fidelity,
            mlp_math_fidelity=mlp_math_fidelity,
            topology=topology,
            num_links=num_links,
        )

    def init_paged_kv_cache(self) -> list[ttnn.Tensor]:
        cache_shape = (
            self.paged_kv_config.max_num_blocks,
            self.local_num_key_value_heads,
            self.paged_kv_config.block_size,
            self.cfg.head_dim,
        )
        zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)
        return [
            self._mesh_replicated_tensor(
                zeros,
                self.mesh_device,
                dtype=self.paged_kv_config.cache_dtype,
                layout=ttnn.TILE_LAYOUT,
            )
            for _ in range(2)
        ]

    def make_identity_page_table(self, batch_size: int = 1) -> ttnn.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if self.paged_kv_config.max_num_blocks % batch_size != 0:
            raise ValueError(
                f"max_num_blocks={self.paged_kv_config.max_num_blocks} must divide evenly across batch_size={batch_size}"
            )
        blocks_per_user = self.paged_kv_config.max_num_blocks // batch_size
        pages = torch.arange(self.paged_kv_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_size, blocks_per_user
        )
        return ttnn.from_torch(
            pages,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def make_current_pos(self, positions: list[int] | torch.Tensor) -> ttnn.Tensor:
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.int32)
        return ttnn.from_torch(
            positions.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def position_tables_for_decode(self, position: int, *, batch_size: int = 1) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if position < 0 or position >= self.max_seq_len:
            raise ValueError(f"decode position must be in [0, {self.max_seq_len}), got {position}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > 1:
            positions = torch.full((batch_size,), position, dtype=torch.float32)
            inv_freq = 1.0 / (
                self.cfg.rope_theta ** (torch.arange(0, self.cfg.head_dim, 2, dtype=torch.float32) / self.cfg.head_dim)
            )
            angles = torch.einsum("d,s->sd", inv_freq, positions)
            emb = torch.cat((angles, angles), dim=-1).reshape(1, 1, batch_size, self.cfg.head_dim)
            return (
                self._mesh_replicated_tensor(torch.cos(emb), self.mesh_device),
                self._mesh_replicated_tensor(torch.sin(emb), self.mesh_device),
            )
        start = [0, 0, position, 0]
        end = [1, 1, position + 1, self.cfg.head_dim]
        return (
            ttnn.slice(self.position_cos, start, end, [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.slice(self.position_sin, start, end, [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG),
        )

    def _all_reduce_hidden(self, partial: ttnn.Tensor, *, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        return ttnn.all_reduce(
            partial,
            num_links=self.num_links,
            topology=self.topology,
            cluster_axis=1,
            memory_config=memory_config,
        )

    def _fill_paged_kv_cache(self, k, v, kv_cache, page_table, *, user_id: int = 0) -> None:
        k_cache, v_cache = kv_cache
        if k.dtype != k_cache.dtype:
            k = ttnn.typecast(k, k_cache.dtype)
        if v.dtype != v_cache.dtype:
            v = ttnn.typecast(v, v_cache.dtype)
        ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_cache, v, page_table, batch_idx=user_id)

    def _prefill_attention(self, hidden_states, seq_len, cos, sin, mask, kv_cache=None, page_table=None, user_id=0):
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=RMS_NORM_EPS,
            weight=self.input_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        qkv = ttnn.matmul(
            normed,
            self.qkv_prefill_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        v = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, seq_len, self.local_kv_width], [1, 1, 1, 1])
        q = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_kv_width],
            [1, 1, seq_len, self.local_kv_width + self.local_q_width],
            [1, 1, 1, 1],
        )
        k = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_kv_width + self.local_q_width],
            [1, 1, seq_len, self.local_qkv_width],
            [1, 1, 1, 1],
        )

        v = ttnn.reshape(v, [1, seq_len, self.local_num_key_value_heads, self.cfg.head_dim])
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, [1, seq_len, self.local_num_key_value_heads, self.cfg.head_dim])
        k = ttnn.rms_norm(k, epsilon=RMS_NORM_EPS, weight=self.k_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(k, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(k, [0, 0, 0, 0], [1, self.local_num_key_value_heads, seq_len, self.cfg.head_dim], [1, 1, 1, 1])

        if kv_cache is not None:
            if page_table is None:
                raise ValueError("page_table is required when prefill_forward fills paged kv_cache")
            self._fill_paged_kv_cache(k, v, kv_cache, page_table, user_id=user_id)

        q = ttnn.reshape(q, [1, seq_len, self.local_num_attention_heads, self.cfg.head_dim])
        q = ttnn.rms_norm(q, epsilon=RMS_NORM_EPS, weight=self.q_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(q, cos, sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(q, [0, 0, 0, 0], [1, self.local_num_attention_heads, seq_len, self.cfg.head_dim], [1, 1, 1, 1])

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        attn = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(attn, [1, 1, seq_len, self.local_q_width], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _mlp(self, post_norm):
        gate = ttnn.matmul(
            post_norm,
            self.gate_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        gate = ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(
            post_norm,
            self.up_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        gated = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_partial = ttnn.matmul(
            gated,
            self.down_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        return self._all_reduce_hidden(down_partial)

    def prefill_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        position_cos: ttnn.Tensor | None = None,
        position_sin: ttnn.Tensor | None = None,
        attention_mask: ttnn.Tensor | None = None,
        kv_cache: list[ttnn.Tensor] | None = None,
        page_table: ttnn.Tensor | None = None,
        user_id: int = 0,
    ) -> ttnn.Tensor:
        signpost("PERF_MULTICHIP_PREFILL")
        start = time.perf_counter()
        seq_len = hidden_states.shape[-2]
        if seq_len > self.max_seq_len and (position_cos is None or position_sin is None):
            raise ValueError(
                f"prefill seq_len {seq_len} exceeds setup max_seq_len {self.max_seq_len}; "
                "provide matching RoPE tables or rebuild the decoder"
            )
        cos = position_cos if position_cos is not None else self.position_cos
        sin = position_sin if position_sin is not None else self.position_sin
        attn = self._prefill_attention(hidden_states, seq_len, cos, sin, attention_mask, kv_cache, page_table, user_id)
        attn_partial = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        attn_out = self._all_reduce_hidden(attn_partial)
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=RMS_NORM_EPS,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        output = ttnn.add(
            self._mlp(post_norm), attn_residual, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_MULTICHIP_PREFILL_END")
        self.timings = MultichipDecoderTimings(
            prefill_ms=elapsed_ms,
            decode_ms=self.timings.decode_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output

    def _decode_head_memory_config(self, batch_size: int) -> ttnn.MemoryConfig:
        if batch_size <= 0:
            raise ValueError(f"decode batch_size must be positive, got {batch_size}")
        return ttnn.create_sharded_memory_config(
            shape=(32, self.cfg.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(
                batch_size, self.mesh_device.compute_with_storage_grid_size(), row_wise=True
            ),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_qkv(self, hidden_states, position_cos, position_sin, batch_size):
        decode_head_memcfg = self._decode_head_memory_config(batch_size)
        qkv = ttnn.matmul(
            hidden_states,
            self.qkv_decode_weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        q = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, batch_size, self.local_q_width], [1, 1, 1, 1])
        k = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_q_width],
            [1, 1, batch_size, self.local_q_width + self.local_kv_width],
            [1, 1, 1, 1],
        )
        v = ttnn.slice(
            qkv,
            [0, 0, 0, self.local_q_width + self.local_kv_width],
            [1, 1, batch_size, self.local_qkv_width],
            [1, 1, 1, 1],
        )

        q = ttnn.reshape(q, [1, batch_size, self.local_num_attention_heads, self.cfg.head_dim])
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.rms_norm(q, epsilon=RMS_NORM_EPS, weight=self.q_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.experimental.rotary_embedding(
            q, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q = ttnn.slice(
            q, [0, 0, 0, 0], [1, batch_size, self.local_num_attention_heads, self.cfg.head_dim], [1, 1, 1, 1]
        )
        q = ttnn.to_memory_config(q, decode_head_memcfg)

        k = ttnn.reshape(k, [1, batch_size, self.local_num_key_value_heads, self.cfg.head_dim])
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.rms_norm(k, epsilon=RMS_NORM_EPS, weight=self.k_norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.experimental.rotary_embedding(
            k, position_cos, position_sin, None, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.slice(
            k, [0, 0, 0, 0], [1, batch_size, self.local_num_key_value_heads, self.cfg.head_dim], [1, 1, 1, 1]
        )
        k = ttnn.to_memory_config(k, decode_head_memcfg)

        v = ttnn.reshape(v, [1, batch_size, self.local_num_key_value_heads, self.cfg.head_dim])
        v = ttnn.to_memory_config(v, decode_head_memcfg)
        return q, k, v

    def decode_forward(
        self,
        hidden_states: ttnn.Tensor,
        *,
        current_pos: ttnn.Tensor,
        page_table: ttnn.Tensor,
        kv_cache: list[ttnn.Tensor],
        position_cos: ttnn.Tensor,
        position_sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        signpost("PERF_MULTICHIP_DECODE")
        start = time.perf_counter()
        batch_size = hidden_states.shape[-2]
        if hidden_states.shape[-3] != 1:
            raise ValueError(f"decode expects one logical token per user, got shape {hidden_states.shape}")
        q, k, v = self._decode_qkv(hidden_states, position_cos, position_sin, batch_size)
        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_pos, page_table=page_table)

        sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=1.0 / math.sqrt(self.cfg.head_dim),
            program_config=self.sdpa_decode_program_config,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sdpa = ttnn.to_memory_config(sdpa, self._decode_head_memory_config(batch_size))
        attn = ttnn.experimental.nlp_concat_heads_decode(sdpa, num_heads=self.local_num_attention_heads)
        attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, batch_size, self.local_q_width], [1, 1, 1, 1])
        attn_partial = ttnn.matmul(
            attn,
            self.o_proj_weight,
            transpose_b=True,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        attn_out = self._all_reduce_hidden(attn_partial)
        attn_residual = ttnn.add(attn_out, hidden_states, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        post_norm = ttnn.rms_norm(
            attn_residual,
            epsilon=RMS_NORM_EPS,
            weight=self.post_attention_layernorm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.auxiliary_compute_kernel_config,
        )
        output = ttnn.add(
            self._mlp(post_norm), attn_residual, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_MULTICHIP_DECODE_END")
        self.timings = MultichipDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=elapsed_ms,
            traced_decode_ms=self.timings.traced_decode_ms,
        )
        return output

    def trace_decode_once(self, *args, **kwargs) -> tuple[int, ttnn.Tensor]:
        warmup = self.decode_forward(*args, **kwargs)
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        output = self.decode_forward(*args, **kwargs)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        signpost("PERF_MULTICHIP_TRACE_DECODE")
        start = time.perf_counter()
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        signpost("PERF_MULTICHIP_TRACE_DECODE_END")
        self.timings = MultichipDecoderTimings(
            prefill_ms=self.timings.prefill_ms,
            decode_ms=self.timings.decode_ms,
            traced_decode_ms=elapsed_ms,
        )
        return trace_id, output

    def forward(self, hidden_states: ttnn.Tensor, *, mode: str = "prefill", **kwargs) -> ttnn.Tensor:
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"unsupported MultichipDecoder mode: {mode!r}")
