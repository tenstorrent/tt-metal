# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Candidate optimized decoders that were built, measured and rejected.

Kept in the evidence directory rather than in ``tt/`` so the shipped module stays
one path, but kept *runnable* so every rejection in the README is reproducible
rather than asserted.  ``bench/layer_ab.py --candidates packed_qkv_gate,...``
measures them against the shipped layer under the same harness.

Each class overrides the smallest possible seam of ``OptimizedDecoder``.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import TILE_SIZE, _get_layer_tensor
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    OptimizedDecoder,
    decode_matmul_program_config,
    dram_sharded_weight_memcfg,
)


class PackedQkvGateDecoder(OptimizedDecoder):
    """One matmul for ``wqkv`` and the attention output gate ($optimize OPT-001).

    Both consume the ``input_layernorm`` output, so they are the classic shared-LHS
    pair.  The packed weight is ``[6656, 4608 + 4096]``; the packed output is split
    on device after the ``sharded_to_interleaved`` that the QKV head-creation op
    needs anyway, so the split costs two slices and one reshard rather than a
    round trip.

    Expected to lose, and the reason is visible in the isolated sweep
    (``logs/decode_matmul_geometry_packed.log``): on the 8-core boundary grid the
    doubled output width pushes the largest legal ``in0_block_w`` from 13 down to
    2 (13 overflows L1), so the packed matmul is 0.1345 ms against 0.1326 for the
    two separate dispatches -- *before* the split.  Measured here anyway, because
    OPT-001 asks for the whole-layer number and not an inference from op counts.
    """

    def __init__(self, *, packed_qkv_gate: ttnn.Tensor | None = None, **kwargs) -> None:
        # ``from_state_dict`` builds the base layer first and attaches the packed
        # weight afterwards, so the constructor cannot require it.
        super().__init__(**kwargs)
        self.packed_qkv_gate = packed_qkv_gate
        self.qkv_width = int(self.wqkv.shape[-1])

    @classmethod
    def from_state_dict(cls, state_dict, *, layer_idx: int, mesh_device, **kwargs):
        decoder = super().from_state_dict(state_dict, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs)

        def weight(suffix: str) -> torch.Tensor:
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        packed = torch.cat(
            [
                weight("self_attn.q_proj.weight"),
                weight("self_attn.k_proj.weight"),
                weight("self_attn.v_proj.weight"),
                weight("self_attn.gate_proj.weight"),
            ],
            dim=-1,
        )
        k, n = packed.shape[-2], packed.shape[-1]
        decoder.packed_qkv_gate = ttnn.from_torch(
            packed.reshape(1, 1, k, n),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=decoder.precision.attn_weight_dtype,
            memory_config=dram_sharded_weight_memcfg(k, n, mesh_device),
        )
        decoder.qkv_width = int(decoder.wqkv.shape[-1])
        return decoder

    def decode_forward(self, hidden_states, *, current_pos, page_table, rope_pos_ids=None):
        cfg = self.config
        batch = int(hidden_states.shape[-2])
        rows = ((batch + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE
        norm_prg, norm_memcfg = self._decode_norm_configs(rows)
        cores, in0_block_w = self.decode_matmul["wqkv"]
        packed_n = int(self.packed_qkv_gate.shape[-1])

        residual = ttnn.interleaved_to_sharded(hidden_states, norm_memcfg)
        normed = self.input_layernorm.sharded_forward(residual, norm_prg, norm_memcfg)
        packed = ttnn.linear(
            normed,
            self.packed_qkv_gate,
            dtype=self.activation_dtype,
            memory_config=self._sharded_memcfg(rows, packed_n, cores),
            program_config=decode_matmul_program_config(rows, packed_n, cores, in0_block_w),
            compute_kernel_config=self.decode_compute_kernel_config,
        )
        ttnn.deallocate(normed)
        packed_l1 = ttnn.sharded_to_interleaved(packed, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(packed)
        xqkv_l1 = ttnn.slice(packed_l1, [0, 0, 0, 0], [1, 1, batch, self.qkv_width])
        gate_l1 = ttnn.slice(packed_l1, [0, 0, 0, self.qkv_width], [1, 1, batch, packed_n])
        ttnn.deallocate(packed_l1)

        q, k, v = self._create_qkv_heads_decode(xqkv_l1)
        ttnn.deallocate(xqkv_l1)
        q = self._sharded_per_head_rmsnorm(q, q.memory_config())
        k = self._sharded_per_head_rmsnorm(k, k.memory_config())
        if cfg.uses_rope:
            cos_q, sin_q = self._decode_rope_tables(rope_pos_ids, q)
            q_rot = ttnn.experimental.rotary_embedding_hf(
                q, cos_q, sin_q, is_decode_mode=True, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(q)
            k_rot = ttnn.experimental.rotary_embedding_hf(
                k, cos_q, sin_q, is_decode_mode=True, compute_kernel_config=self.rope_compute_kernel_config
            )
            ttnn.deallocate(k)
            ttnn.deallocate(cos_q)
            ttnn.deallocate(sin_q)
            q, k = q_rot, k_rot
        self._decode_kv_update(k, v, current_pos, page_table)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            self.k_cache,
            self.v_cache,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            scale=cfg.sdpa_scale,
            sliding_window_size=cfg.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_sdpa_program_config,
        )
        ttnn.deallocate(q)
        out = self._concat_heads_decode(attn, batch)
        ttnn.deallocate(attn)

        gate_memcfg = self._sharded_memcfg(rows, int(gate_l1.shape[-1]), self.decode_matmul["attn_gate"][0])
        gate = ttnn.interleaved_to_sharded(gate_l1, gate_memcfg)
        ttnn.deallocate(gate_l1)
        out_sharded = ttnn.interleaved_to_sharded(out, gate_memcfg)
        ttnn.deallocate(out)
        gated = ttnn.mul(
            out_sharded,
            gate,
            input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
            dtype=self.activation_dtype,
            memory_config=gate_memcfg,
        )
        ttnn.deallocate(out_sharded)
        ttnn.deallocate(gate)
        attn_out = self._decode_projection(gated, self.wo, role="o_proj", rows=rows)
        ttnn.deallocate(gated)

        attn_normed = self.post_attention_layernorm.sharded_forward(attn_out, norm_prg, norm_memcfg)
        ttnn.deallocate(attn_out)
        hidden = ttnn.add(residual, attn_normed, memory_config=norm_memcfg)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_normed)
        mlp_in = self.pre_feedforward_layernorm.sharded_forward(hidden, norm_prg, norm_memcfg)
        mlp_in = self._reshard_to(mlp_in, self.decode_matmul["mlp_gate"][0], rows)
        mlp_out = self.mlp.decode_forward(mlp_in, rows)
        ttnn.deallocate(mlp_in)
        mlp_out = self._reshard_to(mlp_out, self.boundary_cores, rows)
        mlp_normed = self.post_feedforward_layernorm.sharded_forward(mlp_out, norm_prg, norm_memcfg)
        ttnn.deallocate(mlp_out)
        out_sharded = ttnn.add(hidden, mlp_normed, memory_config=norm_memcfg)
        ttnn.deallocate(hidden)
        ttnn.deallocate(mlp_normed)
        out = ttnn.sharded_to_interleaved(out_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_sharded)
        return out


class PackedGateUpDecoder(OptimizedDecoder):
    """One matmul for the MLP gate and up projections ($optimize OPT-010).

    The packed weight is ``[6656, 2 * 19968]``.  The isolated sweep already puts
    this behind: the 39936-wide output caps ``in0_block_w`` at 2 at every legal
    core count, so the packed matmul is 0.4851 ms against 0.4600 for the two
    separate dispatches, before the slice that splits the halves for the SwiGLU
    multiply.  Measured in the layer for the same reason as ``PackedQkvGateDecoder``.
    """

    @classmethod
    def from_state_dict(cls, state_dict, *, layer_idx: int, mesh_device, **kwargs):
        decoder = super().from_state_dict(state_dict, layer_idx=layer_idx, mesh_device=mesh_device, **kwargs)

        def weight(suffix: str) -> torch.Tensor:
            return _get_layer_tensor(state_dict, layer_idx, suffix).to(torch.float32).transpose(-2, -1).contiguous()

        packed = torch.cat([weight("mlp.gate_proj.weight"), weight("mlp.up_proj.weight")], dim=-1)
        k, n = packed.shape[-2], packed.shape[-1]
        decoder.packed_gate_up = ttnn.from_torch(
            packed.reshape(1, 1, k, n),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=decoder.precision.mlp_gate_up_weight_dtype,
            memory_config=dram_sharded_weight_memcfg(k, n, mesh_device),
        )

        def packed_decode(x_sharded, rows, _decoder=decoder):
            cores, in0_block_w = _decoder.decode_matmul["mlp_gate"]
            packed_n = int(_decoder.packed_gate_up.shape[-1])
            half = packed_n // 2
            both = ttnn.linear(
                x_sharded,
                _decoder.packed_gate_up,
                dtype=_decoder.activation_dtype,
                memory_config=_decoder._sharded_memcfg(rows, packed_n, cores),
                program_config=decode_matmul_program_config(rows, packed_n, cores, in0_block_w),
                compute_kernel_config=_decoder.decode_compute_kernel_config,
            )
            both_l1 = ttnn.sharded_to_interleaved(both, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(both)
            batch = int(x_sharded.shape[-2])
            gate = ttnn.slice(both_l1, [0, 0, 0, 0], [1, 1, batch, half])
            up = ttnn.slice(both_l1, [0, 0, 0, half], [1, 1, batch, packed_n])
            ttnn.deallocate(both_l1)
            half_memcfg = _decoder._sharded_memcfg(rows, half, cores)
            gate_s = ttnn.interleaved_to_sharded(gate, half_memcfg)
            up_s = ttnn.interleaved_to_sharded(up, half_memcfg)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            hidden = ttnn.mul(
                gate_s,
                up_s,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                dtype=_decoder.activation_dtype,
                memory_config=half_memcfg,
            )
            ttnn.deallocate(gate_s)
            ttnn.deallocate(up_s)
            out = _decoder._decode_projection(hidden, _decoder.mlp.down, role="mlp_down", rows=rows)
            ttnn.deallocate(hidden)
            return out

        decoder.mlp.decode_forward = packed_decode
        return decoder


class FusedSdpaDecoder(OptimizedDecoder):
    """The decode SDPA config the fused stage inherited: 11x10, q=32, k=64.

    Kept so the shipped ``DEFAULT_DECODE_SDPA`` is measured against the previous
    stage's choice under this stage's layer, not only against itself
    ($optimize OPT-002).
    """

    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        return super().from_state_dict(state_dict, decode_sdpa=(11, 10, 32, 64), **kwargs)
