# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Encoder`] (prefill / inference)."""

from __future__ import annotations

import math
from typing import Optional

import ttnn

from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_ln_sharded_config,
    dram_linear_input_mem_config,
    dram_matmul_program_config,
    ensure_interleaved_bsh,
    ensure_l1_width_sharded_activation,
    sdpa_program_config,
    width_sharded_to_l1_interleaved,
)


class TTSeamlessM4Tv2Encoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2Encoder``.

    ``forward`` takes tensors already placed on the device. Use
    ``create_text_encoder_parameters`` to build ``parameters`` from the PyTorch encoder.

    Prefill matmuls use L1 width-sharded activations + DRAM width-sharded BFP8 weights
    (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``).
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
        self._sdpa_pc_cache: dict = {}
        self._dram_matmul_pc_cache: dict = {}
        self._width_shard_mem_cache: dict = {}

    def _width_shard_mem_config(self, token_rows: int, channels: int, out_channels: int) -> ttnn.MemoryConfig:
        key = (token_rows, channels, out_channels)
        cached = self._width_shard_mem_cache.get(key)
        if cached is not None:
            return cached
        cached = dram_linear_input_mem_config(self.device, token_rows, channels, out_channels)
        self._width_shard_mem_cache[key] = cached
        return cached

    def _to_matmul_width_sharded(
        self, x: ttnn.Tensor, token_rows: int, channels: int, out_channels: int
    ) -> ttnn.Tensor:
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (token_rows, channels))
        return ensure_l1_width_sharded_activation(self.device, x, token_rows, channels, out_channels)

    @staticmethod
    def _width_sharded_to_3d(x: ttnn.Tensor, batch: int, seq: int, channels: int) -> ttnn.Tensor:
        return ensure_interleaved_bsh(x, batch=batch, seq=seq, channels=channels)

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        return sdpa_program_config(self.device, seq_q, seq_k, self._sdpa_pc_cache)

    @staticmethod
    def _linear_token_rows(x: ttnn.Tensor) -> int:
        if len(x.shape) == 3:
            return int(x.shape[0]) * int(x.shape[1])
        if len(x.shape) == 2:
            return int(x.shape[0])
        return int(x.shape[-2])

    def _dram_matmul_pc(
        self,
        m: int,
        k: int,
        n: int,
        *,
        fused_activation=None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        key = (m, k, n, fused_activation)
        cached = self._dram_matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        cached = dram_matmul_program_config(
            self.device,
            m,
            k,
            n,
            fused_activation=fused_activation,
        )
        self._dram_matmul_pc_cache[key] = cached
        return cached

    @staticmethod
    def _bias_token_rows(bias: ttnn.Tensor) -> int:
        if len(bias.shape) == 4:
            return int(bias.shape[2])
        return 32

    @staticmethod
    def _pad_token_rows(x: ttnn.Tensor, m_actual: int, m_padded: int) -> ttnn.Tensor:
        if m_actual >= m_padded:
            return x
        k = int(x.shape[-1])
        pad_rows = m_padded - m_actual
        pad = ttnn.full(
            [pad_rows, k],
            0.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        padded = ttnn.concat([x, pad], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pad)
        return padded

    def _finalize_dram_sharded_linear(
        self,
        x: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        m_actual: int,
        out_dim: int,
    ) -> ttnn.Tensor:
        x = width_sharded_to_l1_interleaved(x)
        if len(x.shape) == 4 and int(x.shape[1]) == 1:
            x = ttnn.reshape(x, (batch, seq, int(x.shape[-1])))
        if len(x.shape) == 2 and int(x.shape[0]) > m_actual:
            x = ttnn.slice(x, [0, 0], [m_actual, int(x.shape[-1])], [1, 1])
        padded_n = int(x.shape[-1])
        if padded_n > out_dim:
            if len(x.shape) == 2:
                x = ttnn.slice(x, [0, 0], [m_actual, out_dim], [1, 1])
            elif len(x.shape) == 3:
                x = ttnn.slice(x, [0, 0, 0], [batch, seq, out_dim], [1, 1, 1])
        if len(x.shape) == 2:
            return ttnn.reshape(x, (batch, seq, out_dim))
        if len(x.shape) == 3 and (int(x.shape[0]) != batch or int(x.shape[1]) != seq):
            x = ttnn.slice(x, [0, 0, 0], [batch, seq, out_dim], [1, 1, 1])
            return ttnn.reshape(x, (batch, seq, out_dim))
        return x

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: Optional[str] = None,
        logical_out_dim: Optional[int] = None,
        keep_sharded_output: bool = False,
        accept_sharded_input: bool = False,
        batch: Optional[int] = None,
        seq: Optional[int] = None,
    ) -> ttnn.Tensor:
        k = int(weight.shape[-2])
        n = int(weight.shape[-1])
        out_dim = logical_out_dim if logical_out_dim is not None else n

        if (accept_sharded_input or (len(x.shape) == 2 and ttnn.is_sharded(x))) and ttnn.is_sharded(x):
            if batch is None or seq is None:
                raise ValueError("batch and seq are required for sharded linear input")
            x_flat = x
            m_actual = batch * seq
            m = self._bias_token_rows(bias)
            x_sharded = ensure_l1_width_sharded_activation(self.device, x_flat, m, k, n)
        elif len(x.shape) == 3:
            batch = int(x.shape[0])
            seq = int(x.shape[1])
            x_flat = ttnn.reshape(x, (batch * seq, k))
            m_actual = batch * seq
            x_sharded = None
        elif len(x.shape) == 2:
            batch = batch if batch is not None else int(x.shape[0])
            seq = seq if seq is not None else 1
            x_flat = x
            m_actual = int(x.shape[0])
            x_sharded = None
        else:
            batch = batch if batch is not None else 1
            seq = seq if seq is not None else 1
            x_flat = x
            m_actual = self._linear_token_rows(x)
            x_sharded = None

        m = self._bias_token_rows(bias)
        if m_actual > m:
            raise ValueError(
                f"Text encoder DRAM-sharded linear expects at most {m} token rows, got {m_actual}. "
                "Recreate parameters with a larger prefill_token_rows."
            )

        if x_sharded is None:
            x_flat = self._pad_token_rows(x_flat, m_actual, m)
            x_sharded = ensure_l1_width_sharded_activation(self.device, x_flat, m, k, n)
        fused_activation = ttnn.UnaryOpType.RELU if activation == "relu" else None
        out = ttnn.linear(
            x_sharded,
            weight,
            bias=bias,
            program_config=self._dram_matmul_pc(m, k, n, fused_activation=fused_activation),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        if x_sharded is not x_flat and x_sharded is not x:
            ttnn.deallocate(x_sharded)
        if keep_sharded_output:
            return out
        return self._finalize_dram_sharded_linear(
            out,
            batch=batch,
            seq=seq,
            m_actual=m_actual,
            out_dim=out_dim,
        )

    def _build_ln_sharded_config(self, m_tiles: int, n_tiles: int):
        return build_ln_sharded_config(self.device, m_tiles, n_tiles, self._ln_sharded_cache)

    def _layer_norm_sharded(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        m_tiles: int,
        n_tiles: int,
        input_sharded: bool = False,
        output_sharded: bool = False,
    ) -> ttnn.Tensor:
        """Width-sharded multicore LN. Set ``output_sharded=True`` to feed matmul without S2I."""
        sharded_mem_config, sharded_pc = self._build_ln_sharded_config(m_tiles, n_tiles)

        if input_sharded and ttnn.is_sharded(x):
            x_sharded = x
        else:
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
        if x_sharded is not x:
            ttnn.deallocate(x_sharded)
        if output_sharded:
            return normed_sharded
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
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
        # Fused QKV projection: one matmul producing
        # ``[B, S, 3 * hidden]`` instead of three separate Q/K/V matmuls.
        qkv = self._linear(
            hidden_states,
            attn_module.qkv.weight,
            attn_module.qkv.bias,
            logical_out_dim=3 * hidden_size,
            accept_sharded_input=ttnn.is_sharded(hidden_states),
            batch=batch,
            seq=seq_q,
        )

        # ``nlp_create_qkv_heads`` consumes a 4-D ``[B, 1, S, 3*H]`` input and
        # returns Q/K/V already shaped as ``[B, num_heads, S, head_dim]`` --
        # this fuses the per-tensor reshape + permute (HC transpose) into a
        # single device kernel.
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, 3 * hidden_size))

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_4d)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0 / math.sqrt(head_dim),
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
        ttnn.deallocate(merged_4d)

        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            logical_out_dim=hidden_size,
            keep_sharded_output=True,
            batch=batch,
            seq=seq_q,
        )
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
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            hidden = ttnn.add(inputs_embeds, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(pos)
        else:
            batch = int(input_ids.shape[0])  # type: ignore[union-attr]
            seq = int(input_ids.shape[1])

            tok = ttnn.embedding(
                input_ids,
                weight=parameters.embed_tokens.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            hidden = ttnn.add(tok, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(tok)
            ttnn.deallocate(pos)

        sdpa_self = self._sdpa_program_config(seq, seq)

        m_tiles = (batch * seq + 31) // 32
        n_tiles = hidden_size // 32
        ffn_dim = 8 * hidden_size
        token_rows = batch * seq
        qkv_n = int(parameters.layers[0].self_attn.qkv.weight.shape[-1])
        hidden = self._to_matmul_width_sharded(hidden, token_rows, hidden_size, qkv_n)
        sharded_hidden_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                input_sharded=ttnn.is_sharded(hidden),
                output_sharded=True,
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
            hidden = ttnn.add(hidden, attn_out, memory_config=sharded_hidden_mem)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                input_sharded=ttnn.is_sharded(hidden),
                output_sharded=True,
            )
            ff = self._linear(
                normed,
                layer.ffn.fc1.weight,
                layer.ffn.fc1.bias,
                activation="relu",
                logical_out_dim=ffn_dim,
                keep_sharded_output=True,
                accept_sharded_input=True,
                batch=batch,
                seq=seq,
            )
            ttnn.deallocate(normed)
            ff = self._linear(
                ff,
                layer.ffn.fc2.weight,
                layer.ffn.fc2.bias,
                logical_out_dim=hidden_size,
                accept_sharded_input=True,
                keep_sharded_output=True,
                batch=batch,
                seq=seq,
            )
            hidden = ttnn.add(hidden, ff, memory_config=sharded_hidden_mem)
            ttnn.deallocate(ff)

        hidden = self._layer_norm_sharded(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
            input_sharded=True,
            output_sharded=False,
        )
        return self._width_sharded_to_3d(hidden, batch, seq, hidden_size)
