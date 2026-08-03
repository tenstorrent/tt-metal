# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Optimized single-device Phi-3.5 Mini decoder layer.

The public tensor, paging, position, and LongRoPE contracts intentionally match
``FunctionalDecoder``.  This implementation owns its complete runtime path:
decode uses a width-sharded L1 residual stream, sharded RMSNorm, DRAM-sharded
projection weights, explicit matmul/compute configs, default paged SDPA, and no
host fallback. Prefill stays DRAM interleaved and uses explicit large-M 2D
matmul configs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import ttnn
from models.common.lightweightmodule import LightweightModule

HF_ADVERTISED_CONTEXT = 131_072
DEFAULT_PAGE_SIZE = 32
PCC_ACCEPTANCE = 0.995
PREFILL_SDPA_MAX_SEQ = 32_768
PREFILL_LINEAR_MAX_ROWS = 1_024


def _layer_key(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.layers.{layer_idx}.{suffix}",
        f"model.model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )


def _require(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    for key in _layer_key(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"Missing Phi-3.5 tensor {suffix!r}; tried {_layer_key(layer_idx, suffix)}")


def _to_device(
    tensor,
    mesh_device,
    *,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )


def _largest_divisor(value: int, candidates: tuple[int, ...]) -> int:
    for candidate in candidates:
        if value % candidate == 0:
            return candidate
    return 1


@dataclass(frozen=True)
class OptimizationPolicy:
    """Named cumulative contract for precision and decode block geometry."""

    attention_weight_dtype: object = ttnn.bfloat4_b
    gate_up_weight_dtype: object = ttnn.bfloat4_b
    down_weight_dtype: object = ttnn.bfloat4_b
    kv_cache_dtype: object = ttnn.bfloat8_b
    attention_math_fidelity: object = ttnn.MathFidelity.LoFi
    mlp_math_fidelity: object = ttnn.MathFidelity.LoFi
    decode_core_grid: tuple[int, int] = (8, 1)
    qkv_in0_block_w: int = 12
    o_proj_in0_block_w: int = 12
    gate_up_in0_block_w: int = 6
    down_in0_block_w: int = 16
    gate_up_split_interleaved: bool = True
    separate_gate_up_projections: bool = False
    fused_paged_cache_update: bool = True
    explicit_decode_sdpa: bool = False
    fused_rope: bool = False
    advisor_rope_l1_chain: bool = True
    fused_prefill_rope: bool = False
    prefill_core_grid: tuple[int, int] = (8, 8)
    prefill_qkv_in0_block_w: int | None = 2
    prefill_o_proj_in0_block_w: int | None = 2
    prefill_gate_up_in0_block_w: int | None = 2
    prefill_down_in0_block_w: int | None = 2
    prefill_qkv_out_block_h: int | None = None
    prefill_o_proj_out_block_h: int | None = None
    prefill_gate_up_out_block_h: int | None = None
    prefill_down_out_block_h: int | None = None
    prefill_qkv_out_block_w: int | None = None
    prefill_o_proj_out_block_w: int | None = None
    prefill_gate_up_out_block_w: int | None = None
    prefill_down_out_block_w: int | None = None


class OptimizedDecoder(LightweightModule):
    """One optimized dense Phi-3.5 decoder layer with paged KV cache."""

    def __init__(
        self,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        batch: int,
        max_context: int,
        page_size: int,
        weights: dict[str, ttnn.Tensor],
        short_cos: ttnn.Tensor,
        short_sin: ttnn.Tensor,
        long_cos: ttnn.Tensor,
        long_sin: ttnn.Tensor,
        short_cos_decode: ttnn.Tensor,
        short_sin_decode: ttnn.Tensor,
        long_cos_decode: ttnn.Tensor,
        long_sin_decode: ttnn.Tensor,
        rope_transformation_prefill: ttnn.Tensor | None,
        rope_transformation_decode: ttnn.Tensor | None,
        policy: OptimizationPolicy,
    ):
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_context = max_context
        self.page_size = page_size
        self.weights = weights
        self.short_cos = short_cos
        self.short_sin = short_sin
        self.long_cos = long_cos
        self.long_sin = long_sin
        self.short_cos_decode = short_cos_decode
        self.short_sin_decode = short_sin_decode
        self.long_cos_decode = long_cos_decode
        self.long_sin_decode = long_sin_decode
        self.rope_transformation_prefill = rope_transformation_prefill
        self.rope_transformation_decode = rope_transformation_decode
        self.policy = policy
        self.hidden_size = int(hf_config.hidden_size)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.original_context = int(hf_config.original_max_position_embeddings)
        self.scale = self.head_dim**-0.5
        self.eps = float(hf_config.rms_norm_eps)

        grid_x, grid_y = policy.decode_core_grid
        device_grid = mesh_device.compute_with_storage_grid_size()
        if grid_x > device_grid.x or grid_y > device_grid.y:
            raise ValueError(
                f"decode grid {policy.decode_core_grid} exceeds device grid {(device_grid.x, device_grid.y)}"
            )
        prefill_grid_x, prefill_grid_y = policy.prefill_core_grid
        if prefill_grid_x > device_grid.x or prefill_grid_y > device_grid.y:
            raise ValueError(
                f"prefill grid {policy.prefill_core_grid} exceeds device grid {(device_grid.x, device_grid.y)}"
            )
        self.decode_grid = ttnn.CoreGrid(x=grid_x, y=grid_y)
        self.decode_cores = grid_x * grid_y
        if self.hidden_size % (self.decode_cores * ttnn.TILE_SIZE):
            raise ValueError("decode grid must divide the hidden dimension in whole tiles")
        if self.intermediate_size % (self.decode_cores * ttnn.TILE_SIZE):
            raise ValueError("decode grid must divide the intermediate dimension in whole tiles")

        self.residual_memory_config = self._width_sharded_memory_config(self.hidden_size)
        self.qkv_memory_config = self._width_sharded_memory_config(3 * self.hidden_size)
        self.gate_up_memory_config = self._width_sharded_memory_config(2 * self.intermediate_size)
        self.intermediate_memory_config = self._width_sharded_memory_config(self.intermediate_size)
        block_w = self.hidden_size // self.decode_cores // ttnn.TILE_SIZE
        self.norm_program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[grid_x, grid_y],
            subblock_w=_largest_divisor(block_w, (4, 3, 2, 1)),
            block_h=1,
            block_w=block_w,
            inplace=False,
        )
        self.attention_compute_kernel_config = self._compute_kernel_config(policy.attention_math_fidelity)
        self.mlp_compute_kernel_config = self._compute_kernel_config(policy.mlp_math_fidelity)
        self.decode_program_configs = {
            "qkv": self._decode_matmul_config(self.hidden_size, 3 * self.hidden_size, policy.qkv_in0_block_w),
            "o_proj": self._decode_matmul_config(self.hidden_size, self.hidden_size, policy.o_proj_in0_block_w),
            "gate_up": self._decode_matmul_config(
                self.hidden_size, 2 * self.intermediate_size, policy.gate_up_in0_block_w
            ),
            "gate": self._decode_matmul_config(self.hidden_size, self.intermediate_size, policy.gate_up_in0_block_w),
            "up": self._decode_matmul_config(self.hidden_size, self.intermediate_size, policy.gate_up_in0_block_w),
            "down": self._decode_matmul_config(self.intermediate_size, self.hidden_size, policy.down_in0_block_w),
        }
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=1,
        max_context=HF_ADVERTISED_CONTEXT,
        page_size=DEFAULT_PAGE_SIZE,
        policy=None,
        **_kwargs,
    ):
        """Load canonical HF tensors directly into the optimized weight layouts."""
        import torch

        policy = policy or OptimizationPolicy()
        if policy.fused_prefill_rope and policy.fused_rope:
            raise ValueError("phase-specific fused prefill requires canonical manual decode RoPE")
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"OptimizedDecoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx {layer_idx} is outside the configured layer range")
        if int(hf_config.hidden_size) != 3072 or int(hf_config.intermediate_size) != 8192:
            raise ValueError("This translation targets the real Phi-3.5-mini shape (hidden=3072, intermediate=8192)")
        if int(hf_config.num_attention_heads) != 32 or int(hf_config.num_key_value_heads) != 32:
            raise ValueError("This translation targets Phi-3.5-mini's 32 Q heads and 32 KV heads")
        if not 1 <= max_context <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_context must be in [1, {hf_config.max_position_embeddings}], got {max_context}")
        if page_size <= 0 or page_size % ttnn.TILE_SIZE:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        hidden = int(hf_config.hidden_size)
        heads = int(hf_config.num_attention_heads)
        head_dim = hidden // heads
        inter = int(hf_config.intermediate_size)
        qkv = _require(state_dict, layer_idx, "self_attn.qkv_proj.weight")
        o_proj = _require(state_dict, layer_idx, "self_attn.o_proj.weight")
        gate_up = _require(state_dict, layer_idx, "mlp.gate_up_proj.weight")
        down = _require(state_dict, layer_idx, "mlp.down_proj.weight")
        input_norm = _require(state_dict, layer_idx, "input_layernorm.weight")
        post_norm = _require(state_dict, layer_idx, "post_attention_layernorm.weight")
        expected = {
            "qkv": (3 * hidden, hidden),
            "o_proj": (hidden, hidden),
            "gate_up": (2 * inter, hidden),
            "down": (hidden, inter),
        }
        for name, tensor in (("qkv", qkv), ("o_proj", o_proj), ("gate_up", gate_up), ("down", down)):
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected[name]}")

        dram_size = mesh_device.dram_grid_size()
        if dram_size.y != 1:
            raise ValueError(f"optimized DRAM weight sharding requires a 1D DRAM grid, got {dram_size}")
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, 0))})

        def weight_memory_config(k: int, n: int):
            padded_n = math.ceil(n / (ttnn.TILE_SIZE * dram_size.x)) * ttnn.TILE_SIZE * dram_size.x
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(dram_grid, (k, padded_n // dram_size.x), ttnn.ShardOrientation.ROW_MAJOR),
            )

        rope = hf_config.rope_scaling
        positions = torch.arange(max_context, dtype=torch.float32).unsqueeze(1)
        exponent = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        base = float(hf_config.rope_theta)
        amplitude = math.sqrt(
            1
            + math.log(int(hf_config.max_position_embeddings) / int(hf_config.original_max_position_embeddings))
            / math.log(int(hf_config.original_max_position_embeddings))
        )

        def rope_table(factors):
            inv_freq = 1.0 / (torch.tensor(factors, dtype=torch.float32) * base**exponent)
            freqs = positions * inv_freq.unsqueeze(0)
            emb = torch.cat((freqs, freqs), dim=-1)
            return (emb.cos() * amplitude).to(torch.bfloat16), (emb.sin() * amplitude).to(torch.bfloat16)

        short_cos_canonical, short_sin_canonical = rope_table(rope["short_factor"])
        long_cos_canonical, long_sin_canonical = rope_table(rope["long_factor"])
        short_cos, short_sin = short_cos_canonical, short_sin_canonical
        long_cos, long_sin = long_cos_canonical, long_sin_canonical
        qkv_canonical = qkv
        qkv_prefill = None
        rope_transformation_prefill = None
        rope_transformation_decode = None
        if policy.fused_prefill_rope or policy.fused_rope:
            pair_index = torch.stack((torch.arange(head_dim // 2), torch.arange(head_dim // 2, head_dim)), dim=-1)
            pair_index = pair_index.flatten()
            qkv_pair = qkv_canonical.reshape(3, heads, head_dim, hidden)
            qkv_pair = torch.cat((qkv_pair[:2, :, pair_index, :], qkv_pair[2:]), dim=0).reshape(3 * hidden, hidden)
            short_cos, short_sin = short_cos[:, pair_index], short_sin[:, pair_index]
            long_cos, long_sin = long_cos[:, pair_index], long_sin[:, pair_index]

            transformation = torch.zeros(1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE, dtype=torch.bfloat16)
            even = torch.arange(0, ttnn.TILE_SIZE, 2)
            odd = torch.arange(1, ttnn.TILE_SIZE, 2)
            transformation[..., even, odd] = 1
            transformation[..., odd, even] = -1
            rope_transformation_prefill = _to_device(transformation, mesh_device)
            if policy.fused_rope:
                qkv = qkv_pair
            else:
                qkv = qkv_canonical
                qkv_prefill = qkv_pair
        else:
            qkv = qkv_canonical
        if policy.fused_rope:
            batch_grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True)
            decode_transformation_memory = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            rope_transformation_decode = _to_device(
                transformation.repeat(1, 1, batch, 1),
                mesh_device,
                memory_config=decode_transformation_memory,
            )
        norm_shape = (1, 1, hidden // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
        weights = {
            "input_norm": _to_device(
                input_norm.reshape(norm_shape).to(torch.bfloat16), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "post_norm": _to_device(
                post_norm.reshape(norm_shape).to(torch.bfloat16), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "qkv": _to_device(
                qkv.transpose(-2, -1),
                mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=weight_memory_config(hidden, 3 * hidden),
            ),
            "o_proj": _to_device(
                o_proj.transpose(-2, -1),
                mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=weight_memory_config(hidden, hidden),
            ),
            "gate_up": _to_device(
                gate_up.transpose(-2, -1),
                mesh_device,
                dtype=policy.gate_up_weight_dtype,
                memory_config=weight_memory_config(hidden, 2 * inter),
            ),
            "down": _to_device(
                down.transpose(-2, -1),
                mesh_device,
                dtype=policy.down_weight_dtype,
                memory_config=weight_memory_config(inter, hidden),
            ),
        }
        if qkv_prefill is not None:
            weights["qkv_prefill"] = _to_device(
                qkv_prefill.transpose(-2, -1),
                mesh_device,
                dtype=policy.attention_weight_dtype,
                memory_config=weight_memory_config(hidden, 3 * hidden),
            )
        if policy.separate_gate_up_projections:
            gate, up = gate_up.chunk(2, dim=0)
            weights["gate"] = _to_device(
                gate.transpose(-2, -1),
                mesh_device,
                dtype=policy.gate_up_weight_dtype,
                memory_config=weight_memory_config(hidden, inter),
            )
            weights["up"] = _to_device(
                up.transpose(-2, -1),
                mesh_device,
                dtype=policy.gate_up_weight_dtype,
                memory_config=weight_memory_config(hidden, inter),
            )
        short_cos_device = _to_device(
            short_cos,
            mesh_device,
            layout=ttnn.TILE_LAYOUT if policy.fused_prefill_rope or policy.fused_rope else ttnn.ROW_MAJOR_LAYOUT,
        )
        short_sin_device = _to_device(
            short_sin,
            mesh_device,
            layout=ttnn.TILE_LAYOUT if policy.fused_prefill_rope or policy.fused_rope else ttnn.ROW_MAJOR_LAYOUT,
        )
        long_cos_device = _to_device(
            long_cos,
            mesh_device,
            layout=ttnn.TILE_LAYOUT if policy.fused_prefill_rope or policy.fused_rope else ttnn.ROW_MAJOR_LAYOUT,
        )
        long_sin_device = _to_device(
            long_sin,
            mesh_device,
            layout=ttnn.TILE_LAYOUT if policy.fused_prefill_rope or policy.fused_rope else ttnn.ROW_MAJOR_LAYOUT,
        )
        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            weights=weights,
            short_cos=short_cos_device,
            short_sin=short_sin_device,
            long_cos=long_cos_device,
            long_sin=long_sin_device,
            # Embedding consumes row-major weights.  Phase-specific decode
            # tables avoid an implicit untilize on every traced token.
            short_cos_decode=(
                _to_device(
                    short_cos if policy.fused_rope else short_cos_canonical,
                    mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                if policy.fused_prefill_rope or policy.fused_rope
                else short_cos_device
            ),
            short_sin_decode=(
                _to_device(
                    short_sin if policy.fused_rope else short_sin_canonical,
                    mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                if policy.fused_prefill_rope or policy.fused_rope
                else short_sin_device
            ),
            long_cos_decode=(
                _to_device(
                    long_cos if policy.fused_rope else long_cos_canonical,
                    mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                if policy.fused_prefill_rope or policy.fused_rope
                else long_cos_device
            ),
            long_sin_decode=(
                _to_device(
                    long_sin if policy.fused_rope else long_sin_canonical,
                    mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                if policy.fused_prefill_rope or policy.fused_rope
                else long_sin_device
            ),
            rope_transformation_prefill=rope_transformation_prefill,
            rope_transformation_decode=rope_transformation_decode,
            policy=policy,
        )

    def _compute_kernel_config(self, fidelity):
        return ttnn.types.BlackholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _width_sharded_memory_config(self, width: int):
        return ttnn.create_sharded_memory_config(
            (ttnn.TILE_SIZE, width // self.decode_cores),
            self.decode_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _decode_matmul_config(self, k: int, n: int, in0_block_w: int):
        k_tiles = k // ttnn.TILE_SIZE
        n_tiles = n // ttnn.TILE_SIZE
        if k_tiles % self.decode_cores or n_tiles % self.decode_cores:
            raise ValueError(f"decode grid of {self.decode_cores} cores does not divide Kt={k_tiles}, Nt={n_tiles}")
        if (k_tiles // self.decode_cores) % in0_block_w:
            raise ValueError(
                f"in0_block_w={in0_block_w} does not divide input shard width " f"{k_tiles // self.decode_cores} tiles"
            )
        return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=in0_block_w,
            per_core_M=1,
            per_core_N=n_tiles // self.decode_cores,
            fused_activation=None,
        )

    def _prefill_matmul_config(self, *, rows: int, row_tiles: int, k: int, n: int, weight_name: str):
        configured_grid_x, configured_grid_y = self.policy.prefill_core_grid
        grid_y = min(configured_grid_y, row_tiles)
        grid_x = min(configured_grid_x, math.ceil(n / ttnn.TILE_SIZE))
        per_core_m = math.ceil(row_tiles / grid_y)
        per_core_n = math.ceil(n / ttnn.TILE_SIZE / grid_x)
        role = "gate_up" if weight_name in ("gate", "up") else weight_name
        if role == "qkv_prefill":
            role = "qkv"
        in0_override = getattr(self.policy, f"prefill_{role}_in0_block_w")
        out_block_h_override = getattr(self.policy, f"prefill_{role}_out_block_h")
        out_block_w_override = getattr(self.policy, f"prefill_{role}_out_block_w")
        if rows >= 2048 and in0_override is not None:
            in0_block_w = in0_override
            out_block_h = out_block_h_override or per_core_m
            out_block_w = out_block_w_override or per_core_n
        else:
            in0_candidates = (1,) if rows >= 2048 else (4, 3, 2, 1)
            in0_block_w = _largest_divisor(k // ttnn.TILE_SIZE, in0_candidates)
            out_block_h = per_core_m
            out_block_w = per_core_n
        if k // ttnn.TILE_SIZE % in0_block_w:
            raise ValueError(f"prefill {role} in0_block_w={in0_block_w} does not divide Kt={k // ttnn.TILE_SIZE}")
        if per_core_m % out_block_h or per_core_n % out_block_w:
            raise ValueError(
                f"prefill {role} inner block {(out_block_h, out_block_w)} does not divide "
                f"per-core output {(per_core_m, per_core_n)}"
            )
        out_subblock_h = 1
        out_subblock_w = _largest_divisor(out_block_w, (4, 3, 2, 1))
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            out_block_h=out_block_h,
            out_block_w=out_block_w,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )

    def create_paged_kv_cache(self, *, num_physical_blocks=None):
        blocks_per_user = math.ceil(self.max_context / self.page_size)
        num_physical_blocks = num_physical_blocks or self.batch * blocks_per_user
        shape = (num_physical_blocks, self.num_kv_heads, self.page_size, self.head_dim)
        cache_kwargs = dict(
            dtype=self.policy.kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.zeros(shape, **cache_kwargs), ttnn.zeros(shape, **cache_kwargs)

    def _norm_prefill(self, hidden_states, weight):
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _norm_decode(self, hidden_states, weight):
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=weight,
            program_config=self.norm_program_config,
            memory_config=self.residual_memory_config,
            compute_kernel_config=self.attention_compute_kernel_config,
        )

    def _linear_prefill(self, hidden_states, weight_name: str, n: int, compute_kernel_config):
        shape = tuple(hidden_states.shape)
        rows = math.prod(shape[:-1])
        # A tiled tensor pads the sequence axis independently for every batch
        # row.  Flattening logical rows first undercounts M tiles for shapes
        # such as [1, 2, 33, H] (4 physical tiles, not ceil(66 / 32) == 3).
        row_tiles = math.prod(shape[:-2]) * math.ceil(shape[-2] / ttnn.TILE_SIZE)
        # The explicit reuse/multicast factory's CB allocation scales with M.
        # Bound only advertised-context internal linears; measured serving
        # shapes keep their independently tuned single-program configs.
        if rows > PREFILL_SDPA_MAX_SEQ:
            leading_rows = math.prod(shape[:-2])
            chunk_length = max(1, PREFILL_LINEAR_MAX_ROWS // leading_rows)
            outputs = []
            for start in range(0, shape[-2], chunk_length):
                end = min(start + chunk_length, shape[-2])
                starts = [0] * len(shape)
                ends = list(shape)
                starts[-2] = start
                ends[-2] = end
                outputs.append(
                    self._linear_prefill(
                        ttnn.slice(hidden_states, starts, ends),
                        weight_name,
                        n,
                        compute_kernel_config,
                    )
                )
            return ttnn.concat(outputs, dim=-2)
        return ttnn.linear(
            hidden_states,
            self.weights[weight_name],
            dtype=ttnn.bfloat16,
            program_config=self._prefill_matmul_config(
                rows=rows, row_tiles=row_tiles, k=shape[-1], n=n, weight_name=weight_name
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )

    def _linear_decode(self, hidden_states, weight_name: str, output_memory_config, compute_kernel_config):
        return ttnn.linear(
            hidden_states,
            self.weights[weight_name],
            dtype=ttnn.bfloat16,
            program_config=self.decode_program_configs[weight_name],
            memory_config=output_memory_config,
            compute_kernel_config=compute_kernel_config,
        )

    def _mlp_prefill_chunk(self, hidden_states, normalized):
        if self.policy.separate_gate_up_projections:
            gate = self._linear_prefill(normalized, "gate", self.intermediate_size, self.mlp_compute_kernel_config)
            up = self._linear_prefill(normalized, "up", self.intermediate_size, self.mlp_compute_kernel_config)
        else:
            gate_up = self._linear_prefill(
                normalized, "gate_up", 2 * self.intermediate_size, self.mlp_compute_kernel_config
            )
            gate_up_shape = tuple(gate_up.shape)
            gate = ttnn.slice(gate_up, [0, 0, 0, 0], [*gate_up_shape[:-1], self.intermediate_size])
            up = ttnn.slice(
                gate_up,
                [0, 0, 0, self.intermediate_size],
                [*gate_up_shape[:-1], 2 * self.intermediate_size],
            )
        activated = ttnn.multiply(ttnn.silu(gate), up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down = self._linear_prefill(activated, "down", self.hidden_size, self.mlp_compute_kernel_config)
        return ttnn.add(hidden_states, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _mlp_prefill(self, hidden_states):
        normalized = self._norm_prefill(hidden_states, self.weights["post_norm"])
        # The packed 16384-wide serving-batch output exceeds Blackhole L1 when
        # all 4096 rows share one program. Keep the public batch contract and
        # packed projection, but execute device-only batch chunks of eight.
        if self.batch > 8:
            outputs = []
            for batch_start in range(0, self.batch, 8):
                batch_end = min(batch_start + 8, self.batch)
                starts = [0, batch_start, 0, 0]
                ends = [1, batch_end, hidden_states.shape[-2], self.hidden_size]
                outputs.append(
                    self._mlp_prefill_chunk(
                        ttnn.slice(hidden_states, starts, ends),
                        ttnn.slice(normalized, starts, ends),
                    )
                )
            return ttnn.concat(outputs, dim=1)
        return self._mlp_prefill_chunk(hidden_states, normalized)

    def _mlp_decode(self, hidden_states):
        normalized = self._norm_decode(hidden_states, self.weights["post_norm"])
        if self.policy.separate_gate_up_projections:
            gate = self._linear_decode(
                normalized, "gate", self.intermediate_memory_config, self.mlp_compute_kernel_config
            )
            up = self._linear_decode(normalized, "up", self.intermediate_memory_config, self.mlp_compute_kernel_config)
        else:
            gate_up = self._linear_decode(
                normalized, "gate_up", self.gate_up_memory_config, self.mlp_compute_kernel_config
            )
            if self.policy.gate_up_split_interleaved:
                # Some few-core shard geometries cannot represent both
                # 8192-wide halves with ttnn.split. Cross the helper boundary
                # once and restore the working shard in the multiply.
                gate_up = ttnn.to_memory_config(gate_up, ttnn.DRAM_MEMORY_CONFIG)
            gate, up = ttnn.split(gate_up, self.intermediate_size, dim=-1)
        activated = ttnn.multiply(
            ttnn.silu(gate),
            up,
            memory_config=self.intermediate_memory_config,
        )
        down = self._linear_decode(activated, "down", self.residual_memory_config, self.mlp_compute_kernel_config)
        return ttnn.add(hidden_states, down, memory_config=self.residual_memory_config)

    def _apply_rope(self, value, cos, sin):
        leading = list(tuple(value.shape)[:-1])
        first = ttnn.slice(value, [0] * len(leading) + [0], leading + [self.head_dim // 2])
        second = ttnn.slice(
            value,
            [0] * len(leading) + [self.head_dim // 2],
            leading + [self.head_dim],
        )
        rotated = ttnn.concat((ttnn.neg(second), first), dim=-1)
        return ttnn.add(ttnn.multiply(value, cos), ttnn.multiply(rotated, sin))

    def _pair_basis_to_canonical(self, value):
        """Restore split-half HF coordinates after the fused adjacent-pair RoPE."""
        shape = list(value.shape)
        row_major = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT)
        starts = [0] * len(shape)
        ends = shape
        steps = [1] * len(shape)
        steps[-1] = 2
        even = ttnn.slice(row_major, starts, ends, steps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        starts = starts.copy()
        starts[-1] = 1
        odd = ttnn.slice(row_major, starts, ends, steps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        canonical = ttnn.concat((even, odd), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.to_layout(canonical, ttnn.TILE_LAYOUT)

    def _prefill_rope(self, query, key, seq_len):
        cos_table = self.long_cos if seq_len > self.original_context else self.short_cos
        sin_table = self.long_sin if seq_len > self.original_context else self.short_sin
        cos = ttnn.reshape(ttnn.slice(cos_table, [0, 0], [seq_len, self.head_dim]), [1, 1, seq_len, self.head_dim])
        sin = ttnn.reshape(ttnn.slice(sin_table, [0, 0], [seq_len, self.head_dim]), [1, 1, seq_len, self.head_dim])
        if self.policy.fused_prefill_rope or self.policy.fused_rope:
            query = ttnn.experimental.rotary_embedding_llama(
                query, cos, sin, self.rope_transformation_prefill, is_decode_mode=False
            )
            key = ttnn.experimental.rotary_embedding_llama(
                key, cos, sin, self.rope_transformation_prefill, is_decode_mode=False
            )
            if self.policy.fused_prefill_rope and not self.policy.fused_rope:
                query = self._pair_basis_to_canonical(query)
                key = self._pair_basis_to_canonical(key)
            return query, key
        return self._apply_rope(query, cos, sin), self._apply_rope(key, cos, sin)

    def _offset_causal_mask(self, *, chunk_start, query_len, key_len):
        query_positions = ttnn.arange(
            chunk_start,
            chunk_start + query_len,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        key_positions = ttnn.arange(
            0,
            key_len,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        query_positions = ttnn.reshape(query_positions, [1, 1, query_len, 1])
        key_positions = ttnn.reshape(key_positions, [1, 1, 1, key_len])
        allowed = ttnn.typecast(ttnn.ge(query_positions, key_positions), ttnn.bfloat16)
        return ttnn.to_layout(ttnn.add(ttnn.multiply(allowed, 1.0e4), -1.0e4), ttnn.TILE_LAYOUT)

    def prefill_forward(self, hidden_states, *, key_cache, value_cache, page_table, user_id=0):
        shape = tuple(hidden_states.shape)
        if len(shape) != 4 or shape[:2] != (1, self.batch) or shape[3] != self.hidden_size:
            raise ValueError(f"prefill hidden_states must be [1,{self.batch},S,{self.hidden_size}], got {shape}")
        seq_len = shape[2]
        if not 1 < seq_len <= self.max_context:
            raise ValueError(f"prefill sequence must be in [2,{self.max_context}], got {seq_len}")
        residual = hidden_states
        normalized = self._norm_prefill(hidden_states, self.weights["input_norm"])
        qkv_weight = "qkv_prefill" if self.policy.fused_prefill_rope else "qkv"
        fused = self._linear_prefill(normalized, qkv_weight, 3 * self.hidden_size, self.attention_compute_kernel_config)
        fused = ttnn.reshape(fused, [self.batch, seq_len, 3 * self.hidden_size])
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            fused,
            None,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_key=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        query, key = self._prefill_rope(query, key, seq_len)
        query = ttnn.slice(query, [0, 0, 0, 0], [self.batch, self.num_heads, seq_len, self.head_dim])
        key = ttnn.slice(key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        value = ttnn.slice(value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, seq_len, self.head_dim])
        for batch_idx in range(self.batch):
            user_key = ttnn.slice(key, [batch_idx, 0, 0, 0], [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim])
            user_value = ttnn.slice(
                value, [batch_idx, 0, 0, 0], [batch_idx + 1, self.num_kv_heads, seq_len, self.head_dim]
            )
            if user_key.dtype != key_cache.dtype:
                user_key = ttnn.typecast(user_key, key_cache.dtype)
                user_value = ttnn.typecast(user_value, value_cache.dtype)
            ttnn.experimental.paged_fill_cache(
                key_cache, user_key, page_table, batch_idx=user_id + batch_idx, block_size=self.page_size
            )
            ttnn.experimental.paged_fill_cache(
                value_cache, user_value, page_table, batch_idx=user_id + batch_idx, block_size=self.page_size
            )
        prefill_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=64,
            k_chunk_size=64,
        )
        if seq_len <= PREFILL_SDPA_MAX_SEQ:
            attended = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=self.scale,
                program_config=prefill_sdpa_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attended_chunks = []
            chunk_start = 0
            while chunk_start < seq_len:
                chunk_capacity = PREFILL_SDPA_MAX_SEQ if chunk_start == 0 else 4 * ttnn.TILE_SIZE
                chunk_len = min(chunk_capacity, seq_len - chunk_start)
                padded_len = math.ceil(chunk_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
                query_chunk = ttnn.slice(
                    query,
                    [0, 0, chunk_start, 0],
                    [self.batch, self.num_heads, chunk_start + chunk_len, self.head_dim],
                )
                if padded_len != chunk_len:
                    query_chunk = ttnn.pad(
                        query_chunk, [(0, 0), (0, 0), (0, padded_len - chunk_len), (0, 0)], value=0.0
                    )
                if chunk_start == 0 and chunk_len == PREFILL_SDPA_MAX_SEQ:
                    prefix_key = ttnn.slice(
                        key, [0, 0, 0, 0], [self.batch, self.num_kv_heads, chunk_len, self.head_dim]
                    )
                    prefix_value = ttnn.slice(
                        value, [0, 0, 0, 0], [self.batch, self.num_kv_heads, chunk_len, self.head_dim]
                    )
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        prefix_key,
                        prefix_value,
                        is_causal=True,
                        scale=self.scale,
                        program_config=prefill_sdpa_config,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                else:
                    mask = self._offset_causal_mask(chunk_start=chunk_start, query_len=padded_len, key_len=seq_len)
                    output_chunk = ttnn.transformer.scaled_dot_product_attention(
                        query_chunk,
                        key,
                        value,
                        attn_mask=mask,
                        is_causal=False,
                        scale=self.scale,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=ttnn.types.BlackholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.HiFi4,
                            math_approx_mode=False,
                            fp32_dest_acc_en=True,
                            packer_l1_acc=False,
                        ),
                    )
                if padded_len != chunk_len:
                    output_chunk = ttnn.slice(
                        output_chunk, [0, 0, 0, 0], [self.batch, self.num_heads, chunk_len, self.head_dim]
                    )
                attended_chunks.append(output_chunk)
                chunk_start += chunk_len
            attended = attended_chunks[0] if len(attended_chunks) == 1 else ttnn.concat(attended_chunks, dim=2)
        attended = ttnn.transformer.concatenate_heads(attended, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        projected = self._linear_prefill(attended, "o_proj", self.hidden_size, self.attention_compute_kernel_config)
        projected = ttnn.reshape(projected, [1, self.batch, seq_len, self.hidden_size])
        return self._mlp_prefill(ttnn.add(residual, projected, memory_config=ttnn.DRAM_MEMORY_CONFIG))

    def _decode_rope(self, query, key, current_positions, *, use_long_rope):
        cos_table = self.long_cos_decode if use_long_rope else self.short_cos_decode
        sin_table = self.long_sin_decode if use_long_rope else self.short_sin_decode
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        batch_grid = ttnn.num_cores_to_corerangeset(self.batch, ttnn.CoreCoord(8, 8), row_wise=True)
        rope_memory_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        cos = ttnn.reshape(
            ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT),
            [1, 1, self.batch, self.head_dim],
        )
        sin = ttnn.reshape(
            ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT),
            [1, 1, self.batch, self.head_dim],
        )
        cos = ttnn.transpose(cos, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin = ttnn.transpose(sin, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.policy.fused_rope:
            # A direct sharded transpose was measured, but corrupts recorded
            # target activations in later b32 lanes.  Keep this explicit,
            # correctness-preserving boundary after removing table untilizes.
            cos = ttnn.to_memory_config(cos, rope_memory_config)
            sin = ttnn.to_memory_config(sin, rope_memory_config)
            return (
                ttnn.experimental.rotary_embedding_llama(
                    query, cos, sin, self.rope_transformation_decode, is_decode_mode=True
                ),
                ttnn.experimental.rotary_embedding_llama(
                    key, cos, sin, self.rope_transformation_decode, is_decode_mode=True
                ),
            )
        query_memory_config = query.memory_config()
        key_memory_config = key.memory_config()
        if self.policy.advisor_rope_l1_chain:
            cos = ttnn.to_memory_config(cos, query_memory_config)
            sin = ttnn.to_memory_config(sin, query_memory_config)

            def apply_l1_rope(value):
                leading = list(tuple(value.shape)[:-1])
                first = ttnn.slice(
                    value,
                    [0] * len(leading) + [0],
                    leading + [self.head_dim // 2],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                second = ttnn.slice(
                    value,
                    [0] * len(leading) + [self.head_dim // 2],
                    leading + [self.head_dim],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                rotated = ttnn.concat((ttnn.neg(second), first), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                rotated = ttnn.to_memory_config(rotated, query_memory_config)
                return ttnn.add(
                    ttnn.multiply(value, cos, memory_config=query_memory_config),
                    ttnn.multiply(rotated, sin, memory_config=query_memory_config),
                    memory_config=query_memory_config,
                )

            query = apply_l1_rope(query)
            key = apply_l1_rope(key)
            return query, ttnn.to_memory_config(key, key_memory_config)
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        return (
            ttnn.to_memory_config(query, query_memory_config),
            ttnn.to_memory_config(key, key_memory_config),
        )

    def _decode_concat_memory_config(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = min(self.batch, grid.x)
        while self.batch % grid_x != 0 or self.batch // grid_x > grid.y:
            grid_x -= 1
        cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, self.batch // grid_x - 1))}
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _fused_cache_value_memory_config(self):
        """Place V on a disjoint core set, as required by fused cache update."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        grid_x = min(self.batch, grid.x)
        while self.batch % grid_x != 0:
            grid_x -= 1
        grid_y = self.batch // grid_x
        if 2 * grid_y > grid.y:
            raise ValueError(
                f"fused cache update needs two disjoint {grid_x}x{grid_y} input grids on {grid.x}x{grid.y}"
            )
        cores = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, grid_y),
                    ttnn.CoreCoord(grid_x - 1, 2 * grid_y - 1),
                )
            }
        )
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def decode_forward(
        self,
        hidden_states,
        *,
        key_cache,
        value_cache,
        page_table,
        current_positions,
        use_long_rope,
    ):
        shape = tuple(hidden_states.shape)
        if shape != (1, 1, self.batch, self.hidden_size):
            raise ValueError(f"decode hidden_states must be [1,1,{self.batch},{self.hidden_size}], got {shape}")
        if tuple(current_positions.shape) != (self.batch,):
            raise ValueError(f"current_positions must have shape [{self.batch}], got {tuple(current_positions.shape)}")
        residual = ttnn.to_memory_config(hidden_states, self.residual_memory_config)
        normalized = self._norm_decode(residual, self.weights["input_norm"])
        fused = self._linear_decode(normalized, "qkv", self.qkv_memory_config, self.attention_compute_kernel_config)
        query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        query, key = self._decode_rope(query, key, current_positions, use_long_rope=use_long_rope)
        if self.policy.fused_paged_cache_update:
            value = ttnn.to_memory_config(value, self._fused_cache_value_memory_config())
            ttnn.experimental.paged_fused_update_cache(
                key_cache,
                key,
                value_cache,
                value,
                update_idxs_tensor=current_positions,
                page_table=page_table,
            )
        else:
            ttnn.experimental.paged_update_cache(
                key_cache, key, update_idxs_tensor=current_positions, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                value_cache, value, update_idxs_tensor=current_positions, page_table=page_table
            )
        attended = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            query,
            key_cache,
            value_cache,
            cur_pos_tensor=current_positions,
            page_table_tensor=page_table,
            scale=self.scale,
            program_config=self.decode_sdpa_program_config if self.policy.explicit_decode_sdpa else None,
            compute_kernel_config=self.attention_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attended = ttnn.to_memory_config(attended, self._decode_concat_memory_config())
        attended = ttnn.experimental.nlp_concat_heads_decode(attended, num_heads=self.num_heads)
        if self.batch < ttnn.TILE_SIZE:
            attended = ttnn.slice(attended, [0, 0, 0, 0], [1, 1, self.batch, self.hidden_size])
        attended = ttnn.to_memory_config(attended, self.residual_memory_config)
        projected = self._linear_decode(
            attended, "o_proj", self.residual_memory_config, self.attention_compute_kernel_config
        )
        projected = ttnn.reshape(projected, [1, 1, self.batch, self.hidden_size])
        return self._mlp_decode(ttnn.add(residual, projected, memory_config=self.residual_memory_config))

    def forward(self, hidden_states, *, mode, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(hidden_states, **kwargs)
        if mode == "decode":
            return self.decode_forward(hidden_states, **kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
