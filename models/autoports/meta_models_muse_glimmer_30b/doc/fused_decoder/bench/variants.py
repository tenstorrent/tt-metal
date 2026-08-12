# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Candidate graph rewrites measured *against* the shipped ``FusedDecoder``.

Each class here is one graph-fusing candidate that was considered and had to be
measured before it could be accepted or rejected.  They live in ``doc/`` rather
than in ``tt/`` because none of them won: keeping them here documents what was
tried (and lets the measurement be reproduced) without carrying dead paths in
the shipped runtime module.

Run with ``bench/ab_latency.py --impl fused,<variant>``; results are in
``doc/fused_decoder/logs/variant_sweep.log`` and the work log.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import _get_layer_tensor, _to_device
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import FusedDecoder, _dense
from models.common.lightweightmodule import LightweightModule


def _packed_gate_up(state_dict, layer_idx, mesh_device, dtype, order):
    """``[a | b]`` column-packed MLP projection, ``order`` picks the halves."""
    tensors = {}
    for name in ("gate", "up"):
        w = _get_layer_tensor(state_dict, layer_idx, f"mlp.{name}_proj.weight")
        tensors[name] = w.to(torch.float32).transpose(-2, -1).contiguous()
    packed = torch.cat([tensors[order[0]], tensors[order[1]]], dim=-1).unsqueeze(0).unsqueeze(0)
    return _to_device(packed, mesh_device=mesh_device, dtype=dtype)


class _SlicedGateUpMLP(LightweightModule):
    """One packed matmul + two slices + ``mul`` with a SILU input activation."""

    def __init__(self, packed, down, intermediate_size, activation_dtype, compute_kernel_config):
        super().__init__()
        self.packed = packed
        self.down = down
        self.intermediate_size = intermediate_size
        self.activation_dtype = activation_dtype
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x):
        gu = _dense(
            x,
            self.packed,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        n = self.intermediate_size
        rows = gu.shape[-2]
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, rows, n])
        up = ttnn.slice(gu, [0, 0, 0, n], [1, 1, rows, 2 * n])
        ttnn.deallocate(gu)
        hidden = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = _dense(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(hidden)
        return out


class _SwigluMLP(LightweightModule):
    """One packed ``[up | gate]`` matmul + ``ttnn.swiglu`` (a composite op)."""

    def __init__(self, packed, down, activation_dtype, compute_kernel_config):
        super().__init__()
        self.packed = packed
        self.down = down
        self.activation_dtype = activation_dtype
        self.compute_kernel_config = compute_kernel_config

    def forward(self, x):
        gu = _dense(
            x,
            self.packed,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden = ttnn.swiglu(gu, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gu)
        out = _dense(
            hidden,
            self.down,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(hidden)
        return out


class PackedGateUpDecoder(FusedDecoder):
    """Shared-LHS packing of the MLP gate/up projections (+ SILU on the mul)."""

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        weight_dtype = kwargs.get("weight_dtype", ttnn.bfloat16)
        decoder = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs
        )
        packed = _packed_gate_up(state_dict, layer_idx, mesh_device, weight_dtype, ("gate", "up"))
        decoder.mlp = _SlicedGateUpMLP(
            packed,
            decoder.mlp.down,
            decoder.config.intermediate_size,
            decoder.activation_dtype,
            decoder.dense_compute_kernel_config,
        )
        return decoder


class SwigluDecoder(FusedDecoder):
    """Shared-LHS packing consumed by the composite ``ttnn.swiglu``."""

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        weight_dtype = kwargs.get("weight_dtype", ttnn.bfloat16)
        decoder = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs
        )
        # ttnn.swiglu is split[0] * swish(split[1]), so pack [up | gate].
        packed = _packed_gate_up(state_dict, layer_idx, mesh_device, weight_dtype, ("up", "gate"))
        decoder.mlp = _SwigluMLP(
            packed, decoder.mlp.down, decoder.activation_dtype, decoder.dense_compute_kernel_config
        )
        return decoder


class PackedQkvGateDecoder(FusedDecoder):
    """Shared-LHS packing of ``wqkv`` and the attention output-gate projection.

    Both matmuls read the same ``input_layernorm`` output, so they can be one
    matmul over ``concat([wqkv, w_attn_gate], -1)`` plus two slices.  The
    sigmoid then has to move from the matmul's pack-time activation onto the
    gating ``ttnn.mul``'s input activation.
    """

    @classmethod
    def from_state_dict(cls, state_dict, *, hf_config, layer_idx, mesh_device, **kwargs):
        weight_dtype = kwargs.get("weight_dtype", ttnn.bfloat16)
        decoder = super().from_state_dict(
            state_dict, hf_config=hf_config, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs
        )

        def col(suffix):
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        packed = (
            torch.cat(
                [
                    col("self_attn.q_proj.weight"),
                    col("self_attn.k_proj.weight"),
                    col("self_attn.v_proj.weight"),
                    col("self_attn.gate_proj.weight"),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        decoder.w_qkv_gate = _to_device(packed, mesh_device=mesh_device, dtype=weight_dtype)
        return decoder

    def _project_qkv(self, normed, *, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        cfg = self.config
        qkv_width = (cfg.num_attention_heads + 2 * cfg.num_key_value_heads) * cfg.head_dim
        gate_width = cfg.num_attention_heads * cfg.head_dim
        rows = normed.shape[-2]
        fused = _dense(
            normed,
            self.w_qkv_gate,
            dtype=self.activation_dtype,
            memory_config=memory_config,
            compute_kernel_config=self.dense_compute_kernel_config,
        )
        xqkv = ttnn.slice(fused, [0, 0, 0, 0], [1, 1, rows, qkv_width])
        # Held for the matching _attn_gate() call later in the same forward.
        self._pending_gate = ttnn.slice(fused, [0, 0, 0, qkv_width], [1, 1, rows, qkv_width + gate_width])
        ttnn.deallocate(fused)
        return xqkv

    def _attn_gate(self, normed):
        gate = self._pending_gate
        self._pending_gate = None
        return gate


class FusedKvUpdateDecoder(FusedDecoder):
    """``paged_fused_update_cache`` + the V reshard it needs from this layout.

    The op asserts its two update tensors live on disjoint cores.
    ``nlp_create_qkv_heads_decode`` can produce that — but only via
    ``overlap_qk_coregrid=False``, which the frontend *drops* for an interleaved
    input (``nlp_create_qkv_heads_decode.cpp:23``) and which the device op then
    constrains to a width-sharded QKV with ``head_dim % shard_width == 0``.
    This layer's decode QKV projection is L1 interleaved, so the flag is a no-op
    here (measured: identical Q/K/V grids at batch 1/4/32) and the only
    reachable form is a manual V reshard — which is what this variant measures.
    See ``doc/fused_decoder/logs/kv_coregrid_probe.log``.
    """

    def _decode_kv_update(self, k, v, current_pos, page_table):
        shard = v.memory_config().shard_spec
        grid = self.mesh_device.compute_with_storage_grid_size()
        batch = shard.grid.num_cores()
        cores = [ttnn.CoreCoord((batch + i) % grid.x, (batch + i) // grid.x) for i in range(batch)]
        disjoint = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in cores}),
                shard.shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        v_moved = ttnn.to_memory_config(v, disjoint)
        ttnn.experimental.paged_fused_update_cache(
            self.k_cache, k, self.v_cache, v_moved, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(v_moved)
