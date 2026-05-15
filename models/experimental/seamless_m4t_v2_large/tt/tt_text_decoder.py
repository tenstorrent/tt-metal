# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Decoder`] (prefill, ``use_cache=False``). Host I/O belongs in callers."""

from __future__ import annotations

import math
from typing import Optional

import ttnn

from models.common.utility_functions import nearest_32


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


class TTSeamlessM4Tv2Decoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2Decoder`` for one prefill step (no KV cache).

    ``forward`` takes only tensors already placed on the device. Use
    ``create_text_decoder_parameters`` to build ``parameters`` from the PyTorch decoder.

    Args:
        device: TTNN device.
        parameters: Nested parameter dict from preprocessing.
        layer_norm_eps: ``config.layer_norm_eps``.
        num_hidden_layers: ``config.decoder_layers``.
        num_attention_heads: ``config.decoder_attention_heads``.
        hidden_size: ``config.hidden_size``.
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
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Pretrained checkpoints amplify bf16 matmul drift over 24 decoder layers; match Whisper-style
        # HiFi4 + FP32 dest accumulation on linear + layer norm so deep stacks stay aligned with PyTorch.
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._ffn_fc1_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ffn_fc2_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # Sharded LN (Stage 11): same recipe as the text encoder — default LN uses one core;
        # width-sharded ``LayerNormShardedMultiCoreProgramConfig`` spreads the reduction across
        # ``grid_x`` cores for typical ``[B, S, 1024]`` (many LN ops per decoder forward).
        self._ln_sharded_cache: dict = {}
        # Stage 19: reuse identical matmul ``ProgramConfig`` objects across layers (same shapes
        # every block) for stable JIT/program-cache keys and less host-side churn.
        self._projection_pc_cache: dict = {}

    def _sdpa_program_config(self, seq_q: int, seq_k: int, *, large_chunks: bool = True) -> ttnn.SDPAProgramConfig:
        """Chunk sizes for ``ttnn.transformer.scaled_dot_product_attention``.

        Long sequences use Whisper-style 64-wide chunks to limit bf16 drift. Short *encoder* keys
        (speech after adaptor subsampling, often ``< 32``) must not force ``k_chunk=64`` against a
        narrow key sequence; that schedule misaligns SDPA with PyTorch and tanks full-model speech PCC.
        """
        if large_chunks:
            q_chunk = max(64, min(256, nearest_32(seq_q)))
            k_chunk = max(64, min(256, nearest_32(seq_k)))
        else:
            q_chunk = max(32, min(256, nearest_32(seq_q)))
            k_chunk = max(32, min(256, nearest_32(seq_k)))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    _PROJECTION_1D_SEQ_THRESHOLD = 384

    def _prefill_projection_program_config(self, token_rows: int, in_dim: int, out_dim: int):
        """Matmul program config for batched prefill linears ``[B,S,*] @ [K,N]`` (1D multicast for short M)."""
        key = (token_rows, in_dim, out_dim)
        cached = self._projection_pc_cache.get(key)
        if cached is not None:
            return cached

        cg = self.device.compute_with_storage_grid_size()
        k_tiles = max(1, in_dim // 32)
        in0_block_w = min(4, k_tiles)
        if token_rows <= self._PROJECTION_1D_SEQ_THRESHOLD:
            per_core_n = (out_dim + cg.x * 32 - 1) // (cg.x * 32)
            per_core_m = (token_rows + 31) // 32
            out_subblock_w = min(4, max(1, per_core_n))
            while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
                out_subblock_w -= 1
            result = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cg.x, 1),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=max(1, per_core_m),
                per_core_N=max(1, per_core_n),
                fuse_batch=True,
                mcast_in0=True,
            )
        else:
            grid_y = min(cg.y, (token_rows + 31) // 32)
            per_core_m = (token_rows + grid_y * 32 - 1) // (grid_y * 32)
            per_core_n = (out_dim + cg.x * 32 - 1) // (cg.x * 32)
            out_subblock_w = min(4, max(1, per_core_n))
            while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
                out_subblock_w -= 1
            result = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(cg.x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=max(1, per_core_m),
                per_core_N=max(1, per_core_n),
                transpose_mcast=False,
                fused_activation=None,
            )
        self._projection_pc_cache[key] = result
        return result

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        compute_cfg: Optional[ttnn.DeviceComputeKernelConfig] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        program_config=None,
        activation: Optional[str] = None,
    ) -> ttnn.Tensor:
        ck = compute_cfg if compute_cfg is not None else self._linear_ln_compute_cfg
        if program_config is not None:
            return ttnn.linear(
                x,
                weight,
                bias=bias,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=ck,
                activation=activation,
            )
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            core_grid=_core_grid(self.device),
            memory_config=memory_config,
            compute_kernel_config=ck,
            activation=activation,
        )

    def _build_ln_sharded_config(self, m_tiles: int, n_tiles: int):
        """Program + L1 memory config for width-sharded LN (see ``tt_text_encoder``)."""
        key = (m_tiles, n_tiles)
        cached = self._ln_sharded_cache.get(key)
        if cached is not None:
            return cached

        device_grid = self.device.compute_with_storage_grid_size()

        grid_x = device_grid.x
        while grid_x > 1 and n_tiles % grid_x != 0:
            grid_x -= 1
        block_w = n_tiles // grid_x

        grid_y = min(device_grid.y, m_tiles)
        while grid_y > 1 and m_tiles % grid_y != 0:
            grid_y -= 1
        block_h = m_tiles // grid_y

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
        sharded_mem_config, sharded_pc = self._build_ln_sharded_config(m_tiles, n_tiles)
        x_sharded = ttnn.to_memory_config(x, sharded_mem_config)
        normed_sharded = ttnn.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=sharded_mem_config,
            program_config=sharded_pc,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        ttnn.deallocate(x_sharded)
        # L1 interleaved activations for the following linears (vs DRAM interleaved ``in0`` on Matmul).
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        ttnn.deallocate(normed_sharded)
        return normed

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _attention(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: Optional[ttnn.Tensor],
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
        q_src = hidden_states
        kv_src = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        # Stage 12: self-attention fuses Q|K|V when KV comes from the same tensor as Q.
        # Stage 15: cross-attention fuses K|V over ``encoder_hidden_states``; Q stays on ``hidden_states``.
        if encoder_hidden_states is None and hasattr(attn_module, "qkv"):
            pc_qkv = self._prefill_projection_program_config(batch * seq_q, hidden_size, 3 * hidden_size)
            qkv = self._linear(
                q_src,
                attn_module.qkv.weight,
                attn_module.qkv.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_qkv,
            )
            qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, 3 * hidden_size))
            ttnn.deallocate(qkv)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_4d,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_4d)
            qh = ttnn.multiply(q, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q)
            kh, vh = k, v
        else:
            pc_q = self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size)
            pc_kv_single = (
                pc_q
                if seq_k == seq_q and kv_src is q_src
                else self._prefill_projection_program_config(batch * seq_k, hidden_size, hidden_size)
            )

            # Stage 8: explicit prefill matmul program config on attention projections; L1 outputs (Stage 6–7).
            q = self._linear(
                q_src,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
            )
            kv_packed = None
            if hasattr(attn_module, "kv"):
                pc_kv2 = self._prefill_projection_program_config(batch * seq_k, hidden_size, 2 * hidden_size)
                kv_packed = self._linear(
                    kv_src,
                    attn_module.kv.weight,
                    attn_module.kv.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv2,
                )
                k = ttnn.slice(
                    kv_packed,
                    [0, 0, 0],
                    [batch, seq_k, hidden_size],
                    [1, 1, 1],
                )
                v = ttnn.slice(
                    kv_packed,
                    [0, 0, hidden_size],
                    [batch, seq_k, 2 * hidden_size],
                    [1, 1, 1],
                )
            else:
                k = self._linear(
                    kv_src,
                    attn_module.k_proj.weight,
                    attn_module.k_proj.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv_single,
                )
                v = self._linear(
                    kv_src,
                    attn_module.v_proj.weight,
                    attn_module.v_proj.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv_single,
                )

            qh = self._heads(q, batch, seq_q, num_heads, head_dim)
            kh = self._heads(k, batch, seq_k, num_heads, head_dim)
            vh = self._heads(v, batch, seq_k, num_heads, head_dim)

            ttnn.deallocate(q)
            if kv_packed is not None:
                ttnn.deallocate(kv_packed)
            else:
                ttnn.deallocate(k)
                ttnn.deallocate(v)

            qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Match Hugging Face ``SeamlessM4Tv2Attention``: scale Q by head_dim**-0.5 before QKᵀ.
        # Use SDPA ``scale=1.0`` so ttnn does not premultiply ``attn_mask`` by 1/scale (see
        # ``sdpa.cpp``); that premultiply overflows HF-style masks (``finfo(dtype).min``).
        is_causal = encoder_hidden_states is None and attn_mask is None

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=1.0,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        # Stage 18: omit post-SDPA slice when output is already unpadded on the head axis (tile
        # layout can pad ``head_dim``; when ``padded_shape[-1] == head_dim`` the slice is a no-op
        # that still showed up as non-trivial work in Tracy ``ReshapeView``/slice buckets).
        ls = attn_out.shape
        ps = attn_out.padded_shape
        if (
            len(ls) == 4
            and int(ls[0]) == batch
            and int(ls[1]) == num_heads
            and int(ls[2]) == seq_q
            and int(ls[3]) == head_dim
            and len(ps) >= 4
            and int(ps[3]) == head_dim
        ):
            attn_for_concat = attn_out
        else:
            attn_for_concat = ttnn.slice(
                attn_out,
                [0, 0, 0, 0],
                [batch, num_heads, seq_q, head_dim],
                [1, 1, 1, 1],
            )
            ttnn.deallocate(attn_out)

        # Stage 13: ``nlp_concat_heads`` fuses the HC transpose + merge (vs separate permute/reshape).
        merged_4d = ttnn.experimental.nlp_concat_heads(attn_for_concat, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_for_concat)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
        ttnn.deallocate(merged_4d)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size),
        )
        ttnn.deallocate(merged)
        return proj

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor],
        position_ids: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        cross_attention_mask: Optional[ttnn.Tensor] = None,
        *,
        inputs_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``uint32`` ``[batch, seq]`` on device (mutually exclusive with ``inputs_embeds``).
            position_ids: ``uint32`` ``[batch, seq]`` on device (sinusoidal table indices).
            encoder_hidden_states: ``bfloat16`` ``[batch, enc_seq, hidden_size]`` on device.
            causal_attention_mask: ``bfloat16`` additive mask ``[batch, 1, seq, seq]`` on device.
            cross_attention_mask: optional ``bfloat16`` ``[batch, 1, seq, enc_seq]`` on device; when
                ``None``, cross-attention runs with no additive mask (all encoder keys visible).
            inputs_embeds: optional decoder ``bfloat16`` ``[batch, seq, hidden_size]`` (HF ``decoder_inputs_embeds``).

        Returns:
            Last hidden states ``bfloat16`` ``[batch, seq, hidden_size]`` on device (after ``decoder.layer_norm``).
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

        enc_seq = int(encoder_hidden_states.shape[1])

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

        sdpa_self = self._sdpa_program_config(seq, seq, large_chunks=True)
        # Cross-attn over subsampled speech (``enc_seq`` ~7) keeps ``nearest_32(enc_seq)`` small; do not
        # promote to 64-wide K chunks or parity vs HF logits collapses.
        sdpa_cross = self._sdpa_program_config(seq, enc_seq, large_chunks=(enc_seq >= 32))

        # Stage 9: FFN linears use the same prefill matmul program config as attention projections.
        # ``fc1.weight`` is ``[hidden, intermediate]`` after ``preprocess_linear_weight`` (torch ``W.T``).
        ffn_intermediate = int(parameters.layers[0].ffn.fc1.weight.shape[1])
        token_m = batch * seq
        pc_ffn_fc1 = self._prefill_projection_program_config(token_m, hidden_size, ffn_intermediate)
        pc_ffn_fc2 = self._prefill_projection_program_config(token_m, ffn_intermediate, hidden_size)

        # Stage 11: width-sharded layer norm (folded ``[B*S, hidden]`` tile grid; matches text encoder).
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
                None,
                layer.self_attn,
                causal_attention_mask,
                batch=batch,
                seq_q=seq,
                seq_k=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_self,
            )
            ttnn.deallocate(normed)
            residual = hidden
            hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.cross_attention_layer_norm.weight,
                bias=layer.cross_attention_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            attn_out = self._attention(
                normed,
                encoder_hidden_states,
                layer.cross_attention,
                cross_attention_mask,
                batch=batch,
                seq_q=seq,
                seq_k=enc_seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_cross,
            )
            ttnn.deallocate(normed)
            residual = hidden
            hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            # Stage 14: fuse fc1 + ReLU in matmul (matches text encoder FFN; drops standalone ReLU).
            ff = self._linear(
                normed,
                layer.ffn.fc1.weight,
                layer.ffn.fc1.bias,
                compute_cfg=self._ffn_fc1_compute_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_ffn_fc1,
                activation="relu",
            )
            ttnn.deallocate(normed)
            ff_in = ff
            ff = self._linear(
                ff_in,
                layer.ffn.fc2.weight,
                layer.ffn.fc2.bias,
                compute_cfg=self._ffn_fc2_compute_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_ffn_fc2,
            )
            ttnn.deallocate(ff_in)
            residual = hidden
            hidden = ttnn.add(residual, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(ff)

        out = self._layer_norm_sharded(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
        )
        ttnn.deallocate(hidden)
        return out
