# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Performance-optimized single-device Qwen3.6-27B decoder layer.

Public tensor contracts
-----------------------
``prefill_forward`` accepts a device tensor shaped ``[1, batch, sequence,
5120]`` and returns the same logical shape. ``decode_forward`` accepts
``[1, 1, batch, 5120]`` and returns the same shape. Both accept device-resident
``page_table`` and ``current_positions`` tensors. Full-attention layers use a
paged KV cache. Linear-attention layers use persistent convolution and
gated-delta recurrent states.

All host work (canonical-key lookup, shape validation, transposition, RoPE
table construction, cache allocation, dtype conversion, and transfer) belongs
to :meth:`from_state_dict`. Runtime methods contain TTNN operations only.

Qwen3.6 resolves to the Transformers Qwen3.5-text implementation. Its RMSNorm
parameters are offsets and therefore become ``1 + weight`` during setup.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import ttnn
from models.common.lightweightmodule import LightweightModule

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
ADVERTISED_CONTEXT = 262_144
REPRESENTATIVE_LAYERS = {"linear_attention": 0, "full_attention": 3}
LINEAR_PREFILL_CHUNK_SIZE = 64


@dataclass(frozen=True)
class OptimizationPolicy:
    attention_weight_dtype: object
    mlp_gate_up_dtype: object
    mlp_down_dtype: object
    cache_dtype: object
    attention_fidelity: object
    mlp_fidelity: object
    packed_qkv: bool = True
    packed_gate_up: bool = True
    advisor_seed: bool = False
    qkv_in0_block_w: int = 4
    qkv_per_core_n: int = 5
    o_in0_block_w: int = 12
    down_in0_block_w: int = 4
    gate_up_ds_in0_block_w: int | None = None
    mlp_l1_chain: bool = False


POLICIES = {
    "default": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
        down_in0_block_w=17,
    ),
    "bfp8_hifi2": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.HiFi2,
        ttnn.MathFidelity.HiFi2,
        advisor_seed=True,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
        down_in0_block_w=17,
    ),
    "bfp4_attention": OptimizationPolicy(
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
        down_in0_block_w=17,
    ),
    "bfp4_down": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
    ),
    "bf16_cache": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.bfloat16,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
    ),
    "bfp8_down_final": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
        down_in0_block_w=17,
    ),
    "split_bfp8_hifi2": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.HiFi2,
        ttnn.MathFidelity.HiFi2,
        packed_qkv=False,
        packed_gate_up=False,
    ),
    "advisor_seed": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
    ),
    "geometry_qkv_iw10_n7": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        qkv_in0_block_w=10,
        qkv_per_core_n=7,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
    ),
    "geometry_o_iw24_n2": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        o_in0_block_w=24,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
    ),
    "geometry_down_iw17_n2": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        down_in0_block_w=17,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
    ),
    "gate_up_ds_iw2": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=2,
    ),
    "gate_up_ds_iw5": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=5,
    ),
    "gate_up_ds_iw10": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=10,
    ),
    "packed_interleaved": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
    ),
    "split_default": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        packed_qkv=False,
        packed_gate_up=False,
    ),
    "mlp_l1_chain": OptimizationPolicy(
        ttnn.bfloat8_b,
        ttnn.bfloat4_b,
        ttnn.bfloat4_b,
        ttnn.bfloat8_b,
        ttnn.MathFidelity.LoFi,
        ttnn.MathFidelity.LoFi,
        advisor_seed=True,
        gate_up_ds_in0_block_w=2,
        mlp_l1_chain=True,
    ),
}


def _candidate_keys(layer_idx: int, suffix: str) -> tuple[str, ...]:
    return (
        f"model.language_model.layers.{layer_idx}.{suffix}",
        f"language_model.layers.{layer_idx}.{suffix}",
        f"model.layers.{layer_idx}.{suffix}",
        f"layers.{layer_idx}.{suffix}",
        suffix,
    )


def _require_tensor(state_dict: Mapping[str, object], layer_idx: int, suffix: str):
    for key in _candidate_keys(layer_idx, suffix):
        if key in state_dict:
            return state_dict[key]
    raise KeyError(f"Missing Qwen3.6 tensor {suffix!r}; tried {', '.join(_candidate_keys(layer_idx, suffix))}")


def _to_device(tensor, *, mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class _BalancedSequenceConcat:
    """Incrementally concatenate sequence chunks with logarithmic retention.

    A simple ``outputs.append(...)`` followed by one final concat retains one
    device tensor per token. At Qwen3.6's 262K context that host-side graph is
    itself a capability limit. This binary-counter reducer keeps at most one
    completed chunk at each level; the final model output is necessarily still
    proportional to sequence length, but transient Python/device references are
    bounded by ``chunk_size + log2(num_chunks)``.
    """

    def __init__(self, *, dim: int, memory_config):
        self.dim = dim
        self.memory_config = memory_config
        self.levels = []

    def append(self, tensor):
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            tensor = ttnn.concat(
                [self.levels[level], tensor],
                dim=self.dim,
                memory_config=self.memory_config,
            )
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(tensor)
        else:
            self.levels[level] = tensor

    def finish(self):
        # High levels contain earlier chunks than low levels.
        chunks = [tensor for tensor in reversed(self.levels) if tensor is not None]
        if not chunks:
            raise ValueError("cannot concatenate an empty linear-attention prefill")
        while len(chunks) > 1:
            chunks = [
                ttnn.concat(
                    chunks[index : index + 2],
                    dim=self.dim,
                    memory_config=self.memory_config,
                )
                for index in range(0, len(chunks), 2)
            ]
        return chunks[0]


class OptimizedDecoder(LightweightModule):
    """One Qwen3.6 text decoder layer on a single 1x1 TTNN mesh.

    The class supports both configured layer kinds rather than silently
    treating the hybrid model as ordinary dense attention.
    """

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
        caches: dict[str, ttnn.Tensor],
        rope: dict[str, ttnn.Tensor],
        policy: OptimizationPolicy,
        candidate: str,
        decode_attention_memory_config=None,
    ):
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.layer_kind = hf_config.layer_types[layer_idx]
        self.mesh_device = mesh_device
        self.batch = batch
        self.max_context = max_context
        self.page_size = page_size
        self.weights = weights
        self.caches = caches
        self.rope = rope
        self.policy = policy
        self.candidate = candidate
        self.decode_attention_memory_config = decode_attention_memory_config

        self.hidden_size = int(hf_config.hidden_size)
        self.intermediate_size = int(hf_config.intermediate_size)
        self.num_heads = int(hf_config.num_attention_heads)
        self.num_kv_heads = int(hf_config.num_key_value_heads)
        self.head_dim = int(hf_config.head_dim)
        self.eps = float(hf_config.rms_norm_eps)
        self.attention_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.attention_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.mlp_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=policy.mlp_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        grid = mesh_device.compute_with_storage_grid_size()
        self.decode_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )
        if policy.advisor_seed:
            input_grid = ttnn.CoreGrid(y=1, x=8)
            self.advisor_input_memcfg = {
                width: ttnn.create_sharded_memory_config(
                    shape=(32, width // 8),
                    core_grid=input_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                for width in (self.hidden_size, self.num_heads * self.head_dim, self.intermediate_size)
            }
            qkv_width = 2 * self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim

            def output_memcfg(width, cores):
                return ttnn.create_sharded_memory_config(
                    shape=(32, math.ceil(width / (32 * cores)) * 32),
                    core_grid=ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(11, 10), row_wise=True),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

            qkv_cores = math.ceil(math.ceil(qkv_width / 32) / policy.qkv_per_core_n)
            self.advisor_output_memcfg = {
                qkv_width: output_memcfg(qkv_width, qkv_cores),
                self.hidden_size: output_memcfg(self.hidden_size, 80),
                2 * self.intermediate_size: output_memcfg(2 * self.intermediate_size, 109),
            }
            self.qkv_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=policy.qkv_in0_block_w, per_core_M=1, per_core_N=policy.qkv_per_core_n
            )
            self.o_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=policy.o_in0_block_w, per_core_M=1, per_core_N=2
            )
            if policy.gate_up_ds_in0_block_w is None:
                self.gate_up_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
                    in0_block_w=2,
                    out_subblock_h=1,
                    out_subblock_w=5,
                    per_core_M=1,
                    per_core_N=10,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
            else:
                self.gate_up_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=policy.gate_up_ds_in0_block_w,
                    per_core_M=1,
                    per_core_N=10,
                )
            self.down_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=policy.down_in0_block_w, per_core_M=1, per_core_N=2
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
        max_context=ADVERTISED_CONTEXT,
        page_size=64,
        candidate="default",
        **_kwargs,
    ):
        """Validate and transfer one canonical HF layer.

        ``hf_config`` must be the text config (``AutoConfig(...).text_config``),
        not the outer multimodal config.
        """
        import math

        import torch
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

        if candidate not in POLICIES:
            raise ValueError(f"unknown optimized-decoder candidate {candidate!r}; choices={tuple(POLICIES)}")
        policy = POLICIES[candidate]

        if not isinstance(mesh_device, ttnn.MeshDevice):
            raise TypeError("OptimizedDecoder requires a ttnn.MeshDevice")
        if tuple(mesh_device.shape) != (1, 1):
            raise ValueError(f"OptimizedDecoder requires a 1x1 mesh, got {tuple(mesh_device.shape)}")
        if not 0 <= layer_idx < int(hf_config.num_hidden_layers):
            raise ValueError(f"layer_idx={layer_idx} is outside the configured layer range")
        if batch < 1 or batch > 32:
            raise ValueError(f"batch must be in [1, 32], got {batch}")
        if max_context < 1 or max_context > int(hf_config.max_position_embeddings):
            raise ValueError(f"max_context must be in [1, {hf_config.max_position_embeddings}], got {max_context}")
        if page_size < 32 or page_size % 32:
            raise ValueError(f"page_size must be a positive tile multiple, got {page_size}")

        hidden = int(hf_config.hidden_size)
        intermediate = int(hf_config.intermediate_size)
        head_dim = int(hf_config.head_dim)
        q_heads = int(hf_config.num_attention_heads)
        kv_heads = int(hf_config.num_key_value_heads)
        kind = hf_config.layer_types[layer_idx]
        expected = {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "head_dim": 256,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
        }
        actual = {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "head_dim": head_dim,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
        }
        if actual != expected:
            raise ValueError(f"Qwen3.6-27B shape contract mismatch: expected {expected}, got {actual}")
        if kind not in REPRESENTATIVE_LAYERS:
            raise ValueError(f"Unsupported Qwen3.6 layer kind {kind!r}")

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )

        def dram_weight_memory_config(k: int, n: int):
            padded_n = math.ceil(n / (32 * dram_size.x)) * (32 * dram_size.x)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (k, padded_n // dram_size.x),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )

        def weight(
            suffix: str,
            expected_shape: tuple[int, ...],
            *,
            transpose=False,
            add_one=False,
            dtype=ttnn.bfloat16,
        ):
            value = _require_tensor(state_dict, layer_idx, suffix)
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"{suffix} has shape {tuple(value.shape)}, expected {expected_shape}")
            value = value.to(torch.bfloat16)
            if transpose:
                value = value.transpose(-2, -1)
            if add_one:
                value = value + 1
            return _to_device(value, mesh_device=mesh_device, dtype=dtype)

        weights = {
            "input_norm": weight("input_layernorm.weight", (hidden,), add_one=True),
            "post_attention_norm": weight("post_attention_layernorm.weight", (hidden,), add_one=True),
            "mlp_gate": weight(
                "mlp.gate_proj.weight", (intermediate, hidden), transpose=True, dtype=policy.mlp_gate_up_dtype
            ),
            "mlp_up": weight(
                "mlp.up_proj.weight", (intermediate, hidden), transpose=True, dtype=policy.mlp_gate_up_dtype
            ),
            "mlp_down": weight(
                "mlp.down_proj.weight", (hidden, intermediate), transpose=True, dtype=policy.mlp_down_dtype
            ),
        }
        caches: dict[str, ttnn.Tensor] = {}
        rope: dict[str, ttnn.Tensor] = {}

        if kind == "full_attention":
            q_width = q_heads * head_dim
            kv_width = kv_heads * head_dim
            q_proj_value = _require_tensor(state_dict, layer_idx, "self_attn.q_proj.weight")
            if tuple(q_proj_value.shape) != (2 * q_width, hidden):
                raise ValueError(
                    f"self_attn.q_proj.weight has shape {tuple(q_proj_value.shape)}, "
                    f"expected {(2 * q_width, hidden)}"
                )
            # HF stores [q, gate] within each head.  Runtime slicing consumes
            # [all q heads, all gate heads], so canonicalize before packing.
            q_proj_value = q_proj_value.to(torch.bfloat16).reshape(q_heads, 2, head_dim, hidden)
            q_proj_value = torch.cat((q_proj_value[:, 0], q_proj_value[:, 1]), dim=0).reshape(2 * q_width, hidden)
            weights.update(
                {
                    "q_proj": _to_device(
                        q_proj_value.transpose(-2, -1),
                        mesh_device=mesh_device,
                        dtype=policy.attention_weight_dtype,
                    ),
                    "k_proj": weight(
                        "self_attn.k_proj.weight",
                        (kv_width, hidden),
                        transpose=True,
                        dtype=policy.attention_weight_dtype,
                    ),
                    "v_proj": weight(
                        "self_attn.v_proj.weight",
                        (kv_width, hidden),
                        transpose=True,
                        dtype=policy.attention_weight_dtype,
                    ),
                    "o_proj": weight(
                        "self_attn.o_proj.weight",
                        (hidden, q_width),
                        transpose=True,
                        dtype=policy.attention_weight_dtype,
                    ),
                    "q_norm": weight("self_attn.q_norm.weight", (head_dim,), add_one=True),
                    "k_norm": weight("self_attn.k_norm.weight", (head_dim,), add_one=True),
                }
            )
            # Each batch row owns an integral set of pages.  Rounding only
            # after multiplying aliases/underallocates the tail page whenever
            # max_context is not page-aligned (for example B32,C65 needs 64
            # blocks, not ceil(32*65/64)=33).
            num_blocks = batch * math.ceil(max_context / page_size)
            cache_shape = (num_blocks, kv_heads, page_size, head_dim)
            zeros = torch.zeros(cache_shape, dtype=torch.bfloat16)
            caches["key"] = _to_device(zeros, mesh_device=mesh_device, dtype=policy.cache_dtype)
            caches["value"] = _to_device(zeros, mesh_device=mesh_device, dtype=policy.cache_dtype)
            caches["batch_indices"] = _to_device(
                torch.arange(batch, dtype=torch.int32),
                mesh_device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            )

            rotary = Qwen3_5TextRotaryEmbedding(hf_config)
            positions = torch.arange(max_context, dtype=torch.long).reshape(1, -1)
            # Rotary only reads dtype/device from x; its output length comes
            # from position_ids. Avoid a needless max_context x hidden host
            # allocation (2.5+ GiB at the advertised context).
            dummy = torch.empty(1, 1, hidden, dtype=torch.bfloat16)
            cos, sin = rotary(dummy, positions)
            # Decode performs a trace-safe embedding lookup with device
            # current_positions, so keep these as 2D embedding tables.
            rope["cos"] = _to_device(cos.squeeze(0), mesh_device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
            rope["sin"] = _to_device(sin.squeeze(0), mesh_device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            key_width = int(hf_config.linear_num_key_heads) * int(hf_config.linear_key_head_dim)
            value_width = int(hf_config.linear_num_value_heads) * int(hf_config.linear_value_head_dim)
            conv_width = 2 * key_width + value_width
            value_heads = int(hf_config.linear_num_value_heads)
            value_dim = int(hf_config.linear_value_head_dim)
            kernel = int(hf_config.linear_conv_kernel_dim)
            weights.update(
                {
                    "in_qkv": weight("linear_attn.in_proj_qkv.weight", (conv_width, hidden), transpose=True),
                    "in_z": weight("linear_attn.in_proj_z.weight", (value_width, hidden), transpose=True),
                    "in_b": weight("linear_attn.in_proj_b.weight", (value_heads, hidden), transpose=True),
                    "in_a": weight("linear_attn.in_proj_a.weight", (value_heads, hidden), transpose=True),
                    "conv": _to_device(
                        _require_tensor(state_dict, layer_idx, "linear_attn.conv1d.weight")
                        .to(torch.bfloat16)
                        .reshape(1, 1, conv_width, kernel),
                        mesh_device=mesh_device,
                    ),
                    "dt_bias": _to_device(
                        _require_tensor(state_dict, layer_idx, "linear_attn.dt_bias")
                        .to(torch.float32)
                        .reshape(1, 1, 1, value_heads),
                        mesh_device=mesh_device,
                        dtype=ttnn.float32,
                    ),
                    "a": _to_device(
                        -_require_tensor(state_dict, layer_idx, "linear_attn.A_log")
                        .float()
                        .exp()
                        .reshape(1, 1, 1, value_heads),
                        mesh_device=mesh_device,
                        dtype=ttnn.float32,
                    ),
                    "gated_norm": weight("linear_attn.norm.weight", (value_dim,)),
                    "out_proj": weight("linear_attn.out_proj.weight", (hidden, value_width), transpose=True),
                    # The chunked affine scan composes 128x128 recurrent
                    # transforms.  Materialize its neutral element during
                    # setup so runtime remains device-only.
                    "linear_identity": _to_device(
                        torch.eye(value_dim, dtype=torch.bfloat16).reshape(1, 1, value_dim, value_dim),
                        mesh_device=mesh_device,
                    ),
                }
            )
            caches["conv"] = _to_device(
                torch.zeros((1, batch, conv_width, kernel), dtype=torch.bfloat16),
                mesh_device=mesh_device,
            )
            caches["recurrent"] = _to_device(
                torch.zeros((batch, value_heads, value_dim, value_dim), dtype=torch.float32),
                mesh_device=mesh_device,
                dtype=ttnn.float32,
            )

        decode_attention_memory_config = None
        if kind == "full_attention":
            # Cache update, decode SDPA, and decode head-concat require an
            # L1-height-sharded tensor. This is the minimal workload-derived
            # layout: one shard per batch row, not a tuned program grid.
            device_grid = mesh_device.compute_with_storage_grid_size()
            grid_x = min(batch, device_grid.x)
            while batch % grid_x or batch // grid_x > device_grid.y:
                grid_x -= 1
            batch_grid = ttnn.CoreGrid(y=batch // grid_x, x=grid_x)
            decode_attention_memory_config = ttnn.create_sharded_memory_config(
                shape=(32, head_dim),
                core_grid=batch_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        if kind == "full_attention" and policy.packed_qkv:
            weights["packed_qkv"] = ttnn.concat(
                [weights["q_proj"], weights["k_proj"], weights["v_proj"]],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        elif kind == "linear_attention":
            weights["packed_linear_inputs"] = ttnn.concat(
                [weights["in_qkv"], weights["in_z"], weights["in_b"], weights["in_a"]],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if policy.packed_gate_up:
            weights["mlp_gate_up"] = ttnn.concat(
                [weights["mlp_gate"], weights["mlp_up"]],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if policy.advisor_seed:
            if kind == "full_attention":
                weights["packed_qkv_prefill"] = weights["packed_qkv"]
                weights["o_proj_prefill"] = weights["o_proj"]
                weights["packed_qkv"] = ttnn.to_memory_config(
                    weights["packed_qkv"],
                    dram_weight_memory_config(hidden, 2 * q_width + 2 * kv_width),
                )
                weights["o_proj"] = ttnn.to_memory_config(
                    weights["o_proj"],
                    dram_weight_memory_config(q_width, hidden),
                )
            weights["mlp_down_prefill"] = weights["mlp_down"]
            weights["mlp_down"] = ttnn.to_memory_config(
                weights["mlp_down"],
                dram_weight_memory_config(intermediate, hidden),
            )
            if policy.gate_up_ds_in0_block_w is not None:
                weights["mlp_gate_up_prefill"] = weights["mlp_gate_up"]
                weights["mlp_gate_up"] = ttnn.to_memory_config(
                    weights["mlp_gate_up"],
                    dram_weight_memory_config(hidden, 2 * intermediate),
                )

        return cls(
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            batch=batch,
            max_context=max_context,
            page_size=page_size,
            weights=weights,
            caches=caches,
            rope=rope,
            policy=policy,
            candidate=candidate,
            decode_attention_memory_config=decode_attention_memory_config,
        )

    def _rms_norm(self, hidden_states, name):
        return ttnn.rms_norm(
            hidden_states,
            epsilon=self.eps,
            weight=self.weights[name],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _mlp(self, hidden_states):
        advisor_decode = (
            self.policy.advisor_seed and hidden_states.shape[1] == 1 and hidden_states.shape[2] == self.batch
        )
        if self.policy.packed_gate_up:
            matmul_input = (
                ttnn.to_memory_config(hidden_states, self.advisor_input_memcfg[self.hidden_size])
                if advisor_decode
                else hidden_states
            )
            gate_up = ttnn.linear(
                matmul_input,
                (
                    self.weights["mlp_gate_up"]
                    if advisor_decode
                    else self.weights.get("mlp_gate_up_prefill", self.weights["mlp_gate_up"])
                ),
                memory_config=(
                    self.advisor_output_memcfg[2 * self.intermediate_size]
                    if advisor_decode
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
                program_config=self.gate_up_decode_program_config if advisor_decode else None,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            if advisor_decode and not self.policy.mlp_l1_chain:
                gate_up = ttnn.to_memory_config(gate_up, ttnn.DRAM_MEMORY_CONFIG)
            gate = gate_up[..., : self.intermediate_size]
            up = gate_up[..., self.intermediate_size :]
        else:
            gate = ttnn.linear(
                hidden_states,
                self.weights["mlp_gate"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
            up = ttnn.linear(
                hidden_states,
                self.weights["mlp_up"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.mlp_compute_kernel_config,
            )
        hidden_states = ttnn.multiply(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=(
                self.advisor_input_memcfg[self.intermediate_size]
                if advisor_decode and self.policy.mlp_l1_chain
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
        down_input = (
            (
                hidden_states
                if self.policy.mlp_l1_chain
                else ttnn.to_memory_config(hidden_states, self.advisor_input_memcfg[self.intermediate_size])
            )
            if advisor_decode
            else hidden_states
        )
        down_weight = (
            self.weights["mlp_down"]
            if advisor_decode
            else self.weights.get("mlp_down_prefill", self.weights["mlp_down"])
        )
        output = ttnn.linear(
            down_input,
            down_weight,
            memory_config=(self.advisor_output_memcfg[self.hidden_size] if advisor_decode else ttnn.DRAM_MEMORY_CONFIG),
            program_config=self.down_decode_program_config if advisor_decode else None,
            compute_kernel_config=self.mlp_compute_kernel_config,
        )
        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG) if advisor_decode else output

    def _token_mixer_prefill(self, hidden_states, page_table, current_positions):
        if self.layer_kind == "linear_attention":
            return self._linear_attention_prefill(hidden_states)
        return self._full_attention_prefill(hidden_states, page_table, current_positions)

    def _token_mixer_decode(self, hidden_states, page_table, current_positions):
        if self.layer_kind == "linear_attention":
            return self._linear_attention_decode(hidden_states)
        return self._full_attention_decode(hidden_states, page_table, current_positions)

    def _linear_attention_prefill(self, hidden_states):
        output = _BalancedSequenceConcat(dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sequence = hidden_states.shape[2]
        for start in range(0, sequence, LINEAR_PREFILL_CHUNK_SIZE):
            stop = min(start + LINEAR_PREFILL_CHUNK_SIZE, sequence)
            chunk = hidden_states[:, :, start:stop, :]
            chunk = self._linear_attention_prefill_chunk(chunk)
            output.append(chunk)
        return output.finish()

    def _linear_attention_prefill_chunk(self, hidden_states):
        """Run one 64-token gated-delta chunk with a logarithmic affine scan.

        For each token the recurrent update is ``R' = A R + B``, where
        ``A = d (I - beta k.T k)`` and ``B = beta k.T v``.  Affine transforms
        compose associatively, so a Hillis-Steele scan produces every token
        state in log2(chunk) batched matmuls instead of submitting one decode
        graph per token.
        """
        key_heads = int(self.hf_config.linear_num_key_heads)
        value_heads = int(self.hf_config.linear_num_value_heads)
        key_dim = int(self.hf_config.linear_key_head_dim)
        value_dim = int(self.hf_config.linear_value_head_dim)
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim
        sequence = hidden_states.shape[2]
        groups = self.batch * value_heads

        projected = ttnn.linear(
            hidden_states, self.weights["packed_linear_inputs"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        conv_width = 2 * key_width + value_width
        mixed = projected[..., :conv_width]
        z = projected[..., conv_width : conv_width + value_width]
        beta = projected[..., conv_width + value_width : conv_width + value_width + value_heads]
        decay = projected[..., conv_width + value_width + value_heads :]

        # Stateful depthwise causal convolution, vectorized across the chunk.
        mixed = ttnn.permute(mixed, (0, 1, 3, 2))
        conv_input = ttnn.concat([self.caches["conv"], mixed], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        next_conv_state = conv_input[..., -self.caches["conv"].shape[-1] :]
        # ``conv_state`` stores the last ``kernel`` inputs, while the HF
        # update concatenates it with this chunk and retains the last L
        # *valid* convolution windows.  Their starts are 1..L, not 0..L-1.
        convolved = ttnn.multiply(conv_input[..., 1 : sequence + 1], self.weights["conv"][..., 0:1])
        for kernel_index in range(1, self.caches["conv"].shape[-1]):
            convolved = ttnn.add(
                convolved,
                ttnn.multiply(
                    conv_input[..., kernel_index + 1 : kernel_index + sequence + 1],
                    self.weights["conv"][..., kernel_index : kernel_index + 1],
                ),
            )
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.silu(ttnn.permute(convolved, (0, 1, 3, 2)))

        query = mixed[..., :key_width]
        key = mixed[..., key_width : 2 * key_width]
        value = mixed[..., 2 * key_width :]
        query = ttnn.reshape(query, (self.batch, sequence, key_heads, key_dim))
        key = ttnn.reshape(key, (self.batch, sequence, key_heads, key_dim))
        value = ttnn.reshape(value, (self.batch, sequence, value_heads, value_dim))
        query = ttnn.repeat_interleave(ttnn.permute(query, (0, 2, 1, 3)), value_heads // key_heads, dim=1)
        key = ttnn.repeat_interleave(ttnn.permute(key, (0, 2, 1, 3)), value_heads // key_heads, dim=1)
        value = ttnn.permute(value, (0, 2, 1, 3))
        query = self._l2_norm(query)
        key = self._l2_norm(key)
        query = ttnn.multiply(query, key_dim**-0.5)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(
            self.weights["a"],
            ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])),
        )
        beta = ttnn.permute(
            ttnn.reshape(beta, (self.batch, sequence, value_heads, 1)),
            (0, 2, 1, 3),
        )
        decay = ttnn.exp(
            ttnn.permute(
                ttnn.reshape(decay, (self.batch, sequence, value_heads, 1)),
                (0, 2, 1, 3),
            )
        )
        query = ttnn.reshape(query, (groups, sequence, 1, key_dim))
        key = ttnn.reshape(key, (groups, sequence, 1, key_dim))
        value = ttnn.reshape(value, (groups, sequence, 1, value_dim))
        beta = ttnn.reshape(beta, (groups, sequence, 1, 1))
        decay = ttnn.reshape(decay, (groups, sequence, 1, 1))
        # Projection/bias math above intentionally follows decode's FP32
        # decay policy.  The verified affine scan is BF16, so cast its scalar
        # coefficients explicitly instead of relying on mixed-dtype promotion.
        beta = ttnn.typecast(beta, ttnn.bfloat16)
        decay = ttnn.typecast(decay, ttnn.bfloat16)

        identity = ttnn.repeat(self.weights["linear_identity"], ttnn.Shape([groups, sequence, 1, 1]))
        zero = ttnn.multiply(identity, 0.0)
        key_t = ttnn.transpose(key, -2, -1)
        transform = ttnn.multiply(
            decay,
            ttnn.subtract(
                identity,
                ttnn.multiply(beta, ttnn.matmul(key_t, key)),
            ),
        )
        bias = ttnn.multiply(beta, ttnn.matmul(key_t, value))

        distance = 1
        while distance < sequence:
            previous_transform = ttnn.concat([identity[:, :distance], transform[:, :-distance]], dim=1)
            previous_bias = ttnn.concat([zero[:, :distance], bias[:, :-distance]], dim=1)
            old_transform = transform
            transform = ttnn.matmul(old_transform, previous_transform)
            bias = ttnn.add(ttnn.matmul(old_transform, previous_bias), bias)
            distance *= 2

        initial = ttnn.typecast(self.caches["recurrent"], ttnn.bfloat16)
        initial = ttnn.reshape(initial, (groups, 1, value_dim, value_dim))
        initial = ttnn.repeat(initial, ttnn.Shape([1, sequence, 1, 1]))
        states = ttnn.add(ttnn.matmul(transform, initial), bias)
        final_state = ttnn.reshape(
            states[:, -1:],
            (self.batch, value_heads, value_dim, value_dim),
        )
        ttnn.copy(ttnn.typecast(final_state, ttnn.float32), self.caches["recurrent"])

        output = ttnn.matmul(query, states)
        output = ttnn.reshape(output, (self.batch, value_heads, sequence, value_dim))
        output = ttnn.rms_norm(
            output,
            epsilon=self.eps,
            weight=self.weights["gated_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.permute(
            ttnn.reshape(z, (self.batch, sequence, value_heads, value_dim)),
            (0, 2, 1, 3),
        )
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.reshape(output, (1, self.batch, sequence, value_width))
        return ttnn.linear(output, self.weights["out_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _linear_attention_decode(self, hidden_states):
        key_heads = int(self.hf_config.linear_num_key_heads)
        value_heads = int(self.hf_config.linear_num_value_heads)
        key_dim = int(self.hf_config.linear_key_head_dim)
        value_dim = int(self.hf_config.linear_value_head_dim)
        key_width = key_heads * key_dim
        value_width = value_heads * value_dim

        projected = ttnn.linear(
            hidden_states, self.weights["packed_linear_inputs"], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        conv_width = 2 * key_width + value_width
        mixed = projected[..., :conv_width]
        z = projected[..., conv_width : conv_width + value_width]
        beta = projected[..., conv_width + value_width : conv_width + value_width + value_heads]
        decay = projected[..., conv_width + value_width + value_heads :]

        mixed = ttnn.permute(mixed, (0, 2, 3, 1))
        next_conv_state = ttnn.concat(
            [self.caches["conv"][..., 1:], mixed],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mixed = ttnn.sum(
            ttnn.multiply(next_conv_state, self.weights["conv"]),
            dim=-1,
            keepdim=True,
        )
        mixed = ttnn.silu(mixed)
        ttnn.copy(next_conv_state, self.caches["conv"])
        mixed = ttnn.permute(mixed, (0, 3, 1, 2))

        query = mixed[..., :key_width]
        key = mixed[..., key_width : 2 * key_width]
        value = mixed[..., 2 * key_width :]
        query = ttnn.reshape(query, (self.batch, 1, key_heads, key_dim))
        key = ttnn.reshape(key, (self.batch, 1, key_heads, key_dim))
        query = ttnn.permute(query, (0, 2, 1, 3))
        key = ttnn.permute(key, (0, 2, 1, 3))
        value = ttnn.reshape(value, (self.batch, 1, value_heads, value_dim))
        value = ttnn.permute(value, (0, 2, 1, 3))
        repeats = value_heads // key_heads
        query = ttnn.repeat_interleave(query, repeats, dim=1)
        key = ttnn.repeat_interleave(key, repeats, dim=1)
        query = self._l2_norm(query)
        key = self._l2_norm(key)
        query = ttnn.multiply(query, key_dim**-0.5)

        beta = ttnn.sigmoid(beta)
        decay = ttnn.multiply(
            self.weights["a"],
            ttnn.softplus(ttnn.add(decay, self.weights["dt_bias"])),
        )
        beta = ttnn.reshape(beta, (self.batch, value_heads, 1, 1))
        decay = ttnn.reshape(decay, (self.batch, value_heads, 1, 1))
        decay = ttnn.exp(decay)

        recurrent = ttnn.multiply(self.caches["recurrent"], decay)
        memory_value = ttnn.matmul(key, recurrent)
        delta = ttnn.multiply(ttnn.subtract(value, memory_value), beta)
        update = ttnn.matmul(ttnn.transpose(key, -2, -1), delta)
        recurrent = ttnn.add(recurrent, update)
        output = ttnn.matmul(query, recurrent)
        ttnn.copy(recurrent, self.caches["recurrent"])

        output = ttnn.rms_norm(
            output,
            epsilon=self.eps,
            weight=self.weights["gated_norm"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        z = ttnn.reshape(z, (self.batch, value_heads, 1, value_dim))
        output = ttnn.multiply(output, ttnn.silu(z))
        output = ttnn.permute(output, (2, 0, 1, 3))
        output = ttnn.reshape(output, (1, 1, self.batch, value_width))
        return ttnn.linear(output, self.weights["out_proj"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    @staticmethod
    def _l2_norm(tensor):
        norm = ttnn.sum(ttnn.multiply(tensor, tensor), dim=-1, keepdim=True)
        return ttnn.multiply(tensor, ttnn.rsqrt(ttnn.add(norm, 1e-6)))

    def _full_attention_prefill(self, hidden_states, page_table, current_positions):
        if self.policy.packed_qkv:
            projected = ttnn.linear(
                hidden_states,
                self.weights.get("packed_qkv_prefill", self.weights["packed_qkv"]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.attention_compute_kernel_config,
            )
        else:
            projected = ttnn.concat(
                [
                    ttnn.linear(
                        hidden_states,
                        self.weights[name],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=self.attention_compute_kernel_config,
                    )
                    for name in ("q_proj", "k_proj", "v_proj")
                ],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        q_and_gate = projected[..., : 2 * q_width]
        q = q_and_gate[..., :q_width]
        gate = q_and_gate[..., q_width:]
        k = projected[..., 2 * q_width : 2 * q_width + kv_width]
        v = projected[..., 2 * q_width + kv_width :]

        sequence = hidden_states.shape[2]
        q = ttnn.reshape(q, (self.batch, sequence, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (self.batch, sequence, self.num_kv_heads, self.head_dim))
        v = ttnn.reshape(v, (self.batch, sequence, self.num_kv_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        q = self._per_head_norm_prefill(q, "q_norm")
        k = self._per_head_norm_prefill(k, "k_norm")
        q = self._partial_rope_prefill(q, current_positions)
        k = self._partial_rope_prefill(k, current_positions)

        cache_k = ttnn.typecast(k, self.policy.cache_dtype)
        cache_v = ttnn.typecast(v, self.policy.cache_dtype)
        ttnn.experimental.paged_fill_cache(
            self.caches["key"], cache_k, page_table, batch_idx_tensor=self.caches["batch_indices"]
        )
        ttnn.experimental.paged_fill_cache(
            self.caches["value"], cache_v, page_table, batch_idx_tensor=self.caches["batch_indices"]
        )
        if sequence <= 32768:
            attention = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.head_dim**-0.5,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            chunks = []
            start = 0
            while start < sequence:
                logical_chunk = min(32768, sequence - start)
                q_chunk = ttnn.slice(
                    q,
                    (0, 0, start, 0),
                    (self.batch, self.num_heads, start + logical_chunk, self.head_dim),
                )
                padding = (-logical_chunk) % 32
                if padding:
                    q_chunk = ttnn.pad(
                        q_chunk,
                        ((0, 0), (0, 0), (0, padding), (0, 0)),
                        value=0.0,
                    )
                chunk = ttnn.transformer.chunked_scaled_dot_product_attention(
                    q_chunk,
                    self.caches["key"],
                    self.caches["value"],
                    page_table,
                    chunk_start_idx=start,
                    scale=self.head_dim**-0.5,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if padding:
                    chunk = ttnn.slice(
                        chunk,
                        (0, 0, 0, 0),
                        (self.batch, self.num_heads, logical_chunk, self.head_dim),
                    )
                chunks.append(chunk)
                start += logical_chunk
            attention = chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2)
        attention = ttnn.experimental.nlp_concat_heads(attention, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.permute(attention, (1, 0, 2, 3))
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        return ttnn.linear(
            attention,
            self.weights.get("o_proj_prefill", self.weights["o_proj"]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_compute_kernel_config,
        )

    def _per_head_norm_prefill(self, tensor, weight_name):
        shape = tensor.shape
        flat = ttnn.reshape(
            tensor,
            (1, 1, shape[0] * shape[1] * shape[2], shape[3]),
        )
        flat = ttnn.rms_norm(
            flat,
            epsilon=self.eps,
            weight=self.weights[weight_name],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(flat, shape)

    def _partial_rope_prefill(self, tensor, current_positions):
        rotary_dim = int(self.head_dim * float(self.hf_config.partial_rotary_factor))
        rotary = tensor[..., :rotary_dim]
        passthrough = tensor[..., rotary_dim:]
        cos = ttnn.embedding(current_positions, self.rope["cos"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(current_positions, self.rope["sin"], layout=ttnn.TILE_LAYOUT)
        cos = ttnn.reshape(cos, (self.batch, 1, tensor.shape[2], rotary_dim))
        sin = ttnn.reshape(sin, (self.batch, 1, tensor.shape[2], rotary_dim))
        heads = tensor.shape[1]
        cos = ttnn.repeat(cos, ttnn.Shape([1, heads, 1, 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([1, heads, 1, 1]))
        rotary = ttnn.add(
            ttnn.multiply(rotary, cos),
            ttnn.multiply(self._rotate_half(rotary), sin),
        )
        return ttnn.concat([rotary, passthrough], dim=-1)

    def _full_attention_decode(self, hidden_states, page_table, current_positions):
        cache_positions = ttnn.typecast(current_positions, ttnn.int32)
        advisor_decode = self.policy.advisor_seed
        if self.policy.packed_qkv:
            qkv_input = (
                ttnn.to_memory_config(hidden_states, self.advisor_input_memcfg[self.hidden_size])
                if advisor_decode
                else hidden_states
            )
            projected = ttnn.linear(
                qkv_input,
                self.weights["packed_qkv"],
                memory_config=(
                    self.advisor_output_memcfg[
                        2 * self.num_heads * self.head_dim + 2 * self.num_kv_heads * self.head_dim
                    ]
                    if advisor_decode
                    else ttnn.DRAM_MEMORY_CONFIG
                ),
                program_config=self.qkv_decode_program_config if advisor_decode else None,
                compute_kernel_config=self.attention_compute_kernel_config,
            )
            if advisor_decode:
                projected = ttnn.to_memory_config(projected, ttnn.DRAM_MEMORY_CONFIG)
        else:
            projected = ttnn.concat(
                [
                    ttnn.linear(
                        hidden_states,
                        self.weights[name],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=self.attention_compute_kernel_config,
                    )
                    for name in ("q_proj", "k_proj", "v_proj")
                ],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        q_and_gate = projected[..., : 2 * q_width]
        q = q_and_gate[..., :q_width]
        gate = q_and_gate[..., q_width:]
        k = projected[..., 2 * q_width : 2 * q_width + kv_width]
        v = projected[..., 2 * q_width + kv_width :]

        fused_qkv = ttnn.concat([q, k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=self.decode_attention_memory_config,
        )
        q = self._per_head_norm(q, "q_norm")
        k = self._per_head_norm(k, "k_norm")
        q = self._partial_rope_decode(q, current_positions)
        k = self._partial_rope_decode(k, current_positions)

        ttnn.experimental.paged_update_cache(
            self.caches["key"],
            k,
            update_idxs_tensor=cache_positions,
            page_table=page_table,
        )
        ttnn.experimental.paged_update_cache(
            self.caches["value"],
            v,
            update_idxs_tensor=cache_positions,
            page_table=page_table,
        )
        attention = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.caches["key"],
            self.caches["value"],
            cur_pos_tensor=cache_positions,
            page_table_tensor=page_table,
            scale=self.head_dim**-0.5,
            program_config=self.decode_sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = ttnn.to_memory_config(attention, self.decode_attention_memory_config)
        attention = ttnn.experimental.nlp_concat_heads_decode(attention, num_heads=self.num_heads)
        attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = ttnn.multiply(attention, ttnn.sigmoid(gate))
        o_input = (
            ttnn.to_memory_config(
                attention,
                self.advisor_input_memcfg[self.num_heads * self.head_dim],
            )
            if advisor_decode
            else attention
        )
        attention = ttnn.linear(
            o_input,
            self.weights["o_proj"],
            memory_config=(self.advisor_output_memcfg[self.hidden_size] if advisor_decode else ttnn.DRAM_MEMORY_CONFIG),
            program_config=self.o_decode_program_config if advisor_decode else None,
            compute_kernel_config=self.attention_compute_kernel_config,
        )
        if advisor_decode:
            attention = ttnn.to_memory_config(attention, ttnn.DRAM_MEMORY_CONFIG)
        attention = attention[:, :, : self.batch, :]
        return ttnn.reshape(attention, (1, 1, self.batch, self.hidden_size))

    def _per_head_norm(self, tensor, weight_name):
        tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
        shape = tensor.shape
        flat = ttnn.reshape(tensor, (1, 1, shape[1] * shape[2], shape[3]))
        flat = ttnn.rms_norm(
            flat,
            epsilon=self.eps,
            weight=self.weights[weight_name],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(flat, shape)

    @staticmethod
    def _rotate_half(tensor):
        half = tensor.shape[-1] // 2
        return ttnn.concat([ttnn.neg(tensor[..., half:]), tensor[..., :half]], dim=-1)

    def _partial_rope_decode(self, tensor, current_positions):
        rotary_dim = int(self.head_dim * float(self.hf_config.partial_rotary_factor))
        rotary = tensor[..., :rotary_dim]
        passthrough = tensor[..., rotary_dim:]
        cos = ttnn.embedding(current_positions, self.rope["cos"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(current_positions, self.rope["sin"], layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)
        cos = cos[:, : self.batch, :, :]
        sin = sin[:, : self.batch, :, :]
        heads = tensor.shape[2]
        cos = ttnn.repeat(cos, ttnn.Shape([1, 1, heads, 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([1, 1, heads, 1]))
        rotary = ttnn.add(
            ttnn.multiply(rotary, cos),
            ttnn.multiply(self._rotate_half(rotary), sin),
        )
        return ttnn.to_memory_config(
            ttnn.concat([rotary, passthrough], dim=-1),
            self.decode_attention_memory_config,
        )

    def prefill_forward(self, *, hidden_states, page_table, current_positions):
        """Run paged/stateful prefill for either configured decoder kind."""
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "input_norm")
        hidden_states = self._token_mixer_prefill(hidden_states, page_table, current_positions)
        hidden_states = ttnn.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "post_attention_norm")
        hidden_states = self._mlp(hidden_states)
        return ttnn.add(residual, hidden_states)

    def decode_forward(self, *, hidden_states, page_table, current_positions):
        """Run one trace-safe decode step using device-resident mutable cache state."""
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "input_norm")
        hidden_states = self._token_mixer_decode(hidden_states, page_table, current_positions)
        hidden_states = ttnn.add(residual, hidden_states)
        residual = hidden_states
        hidden_states = self._rms_norm(hidden_states, "post_attention_norm")
        hidden_states = self._mlp(hidden_states)
        return ttnn.add(residual, hidden_states)

    def forward(self, *, hidden_states, page_table, current_positions, mode):
        if mode == "prefill":
            return self.prefill_forward(
                hidden_states=hidden_states,
                page_table=page_table,
                current_positions=current_positions,
            )
        if mode == "decode":
            return self.decode_forward(
                hidden_states=hidden_states,
                page_table=page_table,
                current_positions=current_positions,
            )
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")
