# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Encoder`] (prefill / inference)."""

from __future__ import annotations

import math
from typing import Optional

import ttnn

from models.common.utility_functions import nearest_32


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


class TTSeamlessM4Tv2Encoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2Encoder``.

    ``forward`` takes tensors already placed on the device. Use
    ``create_text_encoder_parameters`` to build ``parameters`` from the PyTorch encoder.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        *,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_size: int,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._layernorm_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Sharded-LN program config + shard spec are shape-dependent; we build
        # them once per (M_tiles, N_tiles) shape and cache the result.  Default
        # ``LayerNormDefaultProgramConfig`` runs on a single core (~44 us per
        # call x 49 calls = 22 % of device time); the sharded variant spreads
        # the reduction across grid_x cores -- typically 8 cores at seq=32 --
        # for a ~4-5x per-op speedup.
        self._ln_sharded_cache: dict = {}

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        q_chunk = max(64, min(256, nearest_32(seq_q)))
        k_chunk = max(64, min(256, nearest_32(seq_k)))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: Optional[str] = None,
    ) -> ttnn.Tensor:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _layer_norm(self, x: ttnn.Tensor, *, weight: ttnn.Tensor, bias: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._layernorm_compute_cfg,
        )

    def _build_ln_sharded_config(self, m_tiles: int, n_tiles: int):
        """Build width-sharded LN program config + memory config for [M_tiles, N_tiles].

        At our typical shape (B=1, S=32, hidden=1024) -> M_tiles=1, N_tiles=32:
        we pick an ``(8, 1)`` grid (8 cores in x, 1 in y) with ``block_h=1``,
        ``block_w=4`` so the 32 N-tiles split evenly across 8 cores.  For
        larger M we add core rows up to ``grid.y``.
        """
        key = (m_tiles, n_tiles)
        cached = self._ln_sharded_cache.get(key)
        if cached is not None:
            return cached

        device_grid = self.device.compute_with_storage_grid_size()

        # Pick grid_x that divides N_tiles evenly, prefer the largest available.
        grid_x = device_grid.x
        while grid_x > 1 and n_tiles % grid_x != 0:
            grid_x -= 1
        block_w = n_tiles // grid_x

        # Pick grid_y that divides M_tiles evenly (for M=1 we end up at grid_y=1).
        grid_y = min(device_grid.y, m_tiles)
        while grid_y > 1 and m_tiles % grid_y != 0:
            grid_y -= 1
        block_h = m_tiles // grid_y

        # ``out_subblock_w * out_subblock_h`` must be <= 4 with
        # ``fp32_dest_acc_en=True`` -- block_h is typically 1, so we cap
        # subblock_w at min(block_w, 4) and let the kernel handle the rest.
        subblock_w = min(block_w, 4)
        while block_w % subblock_w != 0:
            subblock_w -= 1

        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=subblock_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            [block_h * 32, block_w * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED if grid_y == 1 else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        cached = (memory_config, program_config)
        self._ln_sharded_cache[key] = cached
        return cached

    def _layer_norm_sharded(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        m_tiles: int,
        n_tiles: int,
    ) -> ttnn.Tensor:
        """Width-sharded LN: reshard -> sharded LN -> reshard back to DRAM."""
        sharded_mem_config, sharded_pc = self._build_ln_sharded_config(m_tiles, n_tiles)

        x_sharded = ttnn.to_memory_config(x, sharded_mem_config)
        normed_sharded = ttnn.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=sharded_mem_config,
            program_config=sharded_pc,
            compute_kernel_config=self._layernorm_compute_cfg,
        )
        ttnn.deallocate(x_sharded)
        normed = ttnn.to_memory_config(normed_sharded, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(normed_sharded)
        return normed

    def _attention(
        self,
        hidden_states: ttnn.Tensor,
        attn_module,
        attn_mask: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_q: int,
        seq_k: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
    ) -> ttnn.Tensor:
        # Fused QKV projection (Stage 3a): one matmul producing
        # ``[B, S, 3 * hidden]`` instead of three separate Q/K/V matmuls.
        qkv = self._linear(hidden_states, attn_module.qkv.weight, attn_module.qkv.bias)

        # ``nlp_create_qkv_heads`` consumes a 4-D ``[B, 1, S, 3*H]`` input and
        # returns Q/K/V already shaped as ``[B, num_heads, S, head_dim]`` --
        # this fuses the per-tensor reshape + permute (HC transpose) into a
        # single device kernel.
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, 3 * hidden_size))
        ttnn.deallocate(qkv)

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_4d)

        # Scale is folded into SDPA so we drop the explicit ``ttnn.multiply``.
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0 / math.sqrt(head_dim),
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # ``nlp_concat_heads`` undoes ``nlp_create_qkv_heads``: it merges heads
        # back into ``[B, 1, S, hidden]``. We then drop the singleton batch-1
        # axis so the residual ``ttnn.add`` consumes the same 3-D layout as
        # the rest of the encoder.
        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
        ttnn.deallocate(merged_4d)

        proj = self._linear(merged, attn_module.out_proj.weight, attn_module.out_proj.bias)
        ttnn.deallocate(merged)
        return proj

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor],
        position_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        *,
        inputs_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``uint32`` ``[batch, seq]`` on device (mutually exclusive with ``inputs_embeds``).
            position_ids: ``uint32`` ``[batch, seq]`` on device (sinusoidal table indices).
            attention_mask: optional additive mask ``[batch, 1, seq, seq]`` (bfloat16).
            inputs_embeds: optional ``bfloat16`` ``[batch, seq, hidden_size]`` on device; matches HF
                ``SeamlessM4Tv2Encoder`` when ``inputs_embeds`` is passed instead of ``input_ids``.

        Returns:
            Last hidden states ``bfloat16`` ``[batch, seq, hidden_size]`` on device.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("One of input_ids or inputs_embeds is required.")

        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers

        if inputs_embeds is not None:
            batch = int(inputs_embeds.shape[0])
            seq = int(inputs_embeds.shape[1])
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
            )
            hidden = ttnn.add(inputs_embeds, pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pos)
        else:
            batch = int(input_ids.shape[0])  # type: ignore[union-attr]
            seq = int(input_ids.shape[1])

            tok = ttnn.embedding(
                input_ids,
                weight=parameters.embed_tokens.weight,
                layout=ttnn.TILE_LAYOUT,
            )
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
            )

            hidden = ttnn.add(tok, pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tok)
            ttnn.deallocate(pos)

        sdpa_self = self._sdpa_program_config(seq, seq)

        # M tiles count for the activation -- folded ``[B*S, hidden]`` view.
        # padded up to a multiple of 32 to match TILE layout.  N tiles is the
        # tile count of ``hidden_size`` (always a multiple of 32 in this model).
        m_tiles = (batch * seq + 31) // 32
        n_tiles = hidden_size // 32

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            attn_out = self._attention(
                normed,
                layer.self_attn,
                attention_mask,
                batch=batch,
                seq_q=seq,
                seq_k=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_self,
            )
            ttnn.deallocate(normed)
            hidden = ttnn.add(hidden, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            ff = self._linear(
                normed,
                layer.ffn.fc1.weight,
                layer.ffn.fc1.bias,
                activation="relu",
            )
            ttnn.deallocate(normed)
            ff = self._linear(ff, layer.ffn.fc2.weight, layer.ffn.fc2.bias)
            hidden = ttnn.add(hidden, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(ff)

        hidden = self._layer_norm_sharded(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
        )
        return hidden
