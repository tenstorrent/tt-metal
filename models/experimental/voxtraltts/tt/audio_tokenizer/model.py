# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT audio tokenizer decode path; encoder modules optional (gated on checkpoint keys)."""

from __future__ import annotations

import torch
import ttnn
from types import SimpleNamespace

from models.experimental.voxtraltts.reference.audio_tokenizer_ops import (
    audio_tokenizer_mm_embedding_offsets,
    audio_tokenizer_sliding_window_attention_bias,
)
from models.experimental.voxtraltts.reference.voxtral_config import (
    DEFAULT_VOXTRAL_MODEL,
    VoxtralAudioTokenizerConfig,
    audio_tokenizer_latent_dim,
    load_voxtral_config,
    parse_csv_ints,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.conv import (
    VoxtralTTAudioTokenizerDecoderCausalConv1d,
    VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d,
    VoxtralTTAudioTokenizerEncoderDownsampleConv,
    VoxtralTTAudioTokenizerInputProj,
    resolve_encoder_block_strided_conv_weight,
    resolve_output_proj_causal_conv_fused_weight,
)
from models.experimental.voxtraltts.tt.audio_tokenizer.embedding import VoxtralTTAudioCodebookEmbedding
from models.experimental.voxtraltts.tt.audio_tokenizer.quantizer import VoxtralTTSemanticCodebookQuantizer
from models.experimental.voxtraltts.tt.audio_tokenizer.transformer import (
    VoxtralTTAudioTokenizerDecoderTransformerBlock,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict
from models.experimental.voxtraltts.utils.config_helpers import COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER

AUDIO_TOKENIZER_ENCODER_OPTIONAL_PREFIXES = ("input_proj.", "encoder_blocks.")


def extract_audio_tokenizer_state_dict(full_state_dict: dict) -> dict:
    prefix = "audio_tokenizer."
    return {k[len(prefix) :]: v for k, v in full_state_dict.items() if k.startswith(prefix)}


class VoxtralTTAudioTokenizer:
    """TT decode stack (+ optional encoder / quantizer when weights exist)."""

    cfg: VoxtralAudioTokenizerConfig

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        *,
        state_dict: dict,
        tokenizer_cfg: VoxtralAudioTokenizerConfig,
        dtype: ttnn.DataType = ttnn.bfloat16,
        full_checkpoint: dict | None = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.cfg = tokenizer_cfg
        self._dtype = dtype
        self._audio_tokenizer_sd = state_dict
        self._compute_kernel_config = COMPUTE_KERNEL_CONFIG_VOXTRAL_AUDIO_TOKENIZER

        self.input_proj: VoxtralTTAudioTokenizerInputProj | None = None
        try:
            self.input_proj = VoxtralTTAudioTokenizerInputProj(
                mesh_device,
                state_dict=state_dict,
                in_channels=tokenizer_cfg.pretransform_patch_size,
                out_channels=tokenizer_cfg.dim,
                kernel_size=tokenizer_cfg.patch_proj_kernel_size,
                stride=1,
                causal=tokenizer_cfg.causal,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.decoder_blocks_1_layer0: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_1_layer0 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=1,
                layer_index=0,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_1_layer1: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_1_layer1 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=1,
                layer_index=1,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_3_layer0: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_3_layer0 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=3,
                layer_index=0,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_3_layer1: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_3_layer1 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=3,
                layer_index=1,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_2_conv_transpose: VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d | None = None
        try:
            kerns = parse_csv_ints(tokenizer_cfg.decoder_convs_kernels_str)
            strides = parse_csv_ints(tokenizer_cfg.decoder_convs_strides_str)
            self.decoder_blocks_2_conv_transpose = VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d(
                mesh_device,
                state_dict=state_dict,
                block_index=2,
                kernel_size=kerns[1],
                stride=strides[1],
                in_channels=tokenizer_cfg.dim,
                out_channels=tokenizer_cfg.dim,
                output_channel_splits=16,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.decoder_blocks_4_conv_transpose: VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d | None = None
        try:
            kerns = parse_csv_ints(tokenizer_cfg.decoder_convs_kernels_str)
            strides = parse_csv_ints(tokenizer_cfg.decoder_convs_strides_str)
            self.decoder_blocks_4_conv_transpose = VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d(
                mesh_device,
                state_dict=state_dict,
                block_index=4,
                kernel_size=kerns[2],
                stride=strides[2],
                in_channels=tokenizer_cfg.dim,
                out_channels=tokenizer_cfg.dim,
                output_channel_splits=16,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.decoder_blocks_5_layer0: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_5_layer0 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=5,
                layer_index=0,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_5_layer1: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_5_layer1 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=5,
                layer_index=1,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_6_conv_transpose: VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d | None = None
        try:
            kerns = parse_csv_ints(tokenizer_cfg.decoder_convs_kernels_str)
            strides = parse_csv_ints(tokenizer_cfg.decoder_convs_strides_str)
            self.decoder_blocks_6_conv_transpose = VoxtralTTAudioTokenizerDecoderCausalConvTranspose1d(
                mesh_device,
                state_dict=state_dict,
                block_index=6,
                kernel_size=kerns[3],
                stride=strides[3],
                in_channels=tokenizer_cfg.dim,
                out_channels=tokenizer_cfg.dim,
                output_channel_splits=16,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.decoder_blocks_7_layer0: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_7_layer0 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=7,
                layer_index=0,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.decoder_blocks_7_layer1: VoxtralTTAudioTokenizerDecoderTransformerBlock | None = None
        try:
            self.decoder_blocks_7_layer1 = VoxtralTTAudioTokenizerDecoderTransformerBlock(
                mesh_device,
                state_dict=state_dict,
                tokenizer_cfg=tokenizer_cfg,
                block_index=7,
                layer_index=1,
                weight_dtype=dtype,
                output_dtype=dtype,
                compute_kernel_config=self._compute_kernel_config,
            )
        except KeyError:
            pass

        self.encoder_downsample_after_transformer_0: VoxtralTTAudioTokenizerEncoderDownsampleConv | None = None
        try:
            w_ds = resolve_encoder_block_strided_conv_weight(state_dict, 1)
            oc, ic, ks = (int(w_ds.shape[0]), int(w_ds.shape[1]), int(w_ds.shape[2]))
            self.encoder_downsample_after_transformer_0 = VoxtralTTAudioTokenizerEncoderDownsampleConv(
                mesh_device,
                state_dict=state_dict,
                block_index=1,
                in_channels=ic,
                out_channels=oc,
                kernel_size=ks,
                stride=2,
                causal=tokenizer_cfg.causal,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.decoder_blocks_0_conv: VoxtralTTAudioTokenizerDecoderCausalConv1d | None = None
        try:
            kerns = parse_csv_ints(tokenizer_cfg.decoder_convs_kernels_str)
            strides = parse_csv_ints(tokenizer_cfg.decoder_convs_strides_str)
            self.decoder_blocks_0_conv = VoxtralTTAudioTokenizerDecoderCausalConv1d(
                mesh_device,
                state_dict=state_dict,
                block_index=0,
                kernel_size=kerns[0],
                stride=strides[0],
                pad_mode="replicate",
                in_channels=audio_tokenizer_latent_dim(tokenizer_cfg),
                out_channels=tokenizer_cfg.dim,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.output_proj_conv: VoxtralTTAudioTokenizerDecoderCausalConv1d | None = None
        try:
            w_op = resolve_output_proj_causal_conv_fused_weight(state_dict)
            oc, ic, ks = (int(w_op.shape[0]), int(w_op.shape[1]), int(w_op.shape[2]))
            self.output_proj_conv = VoxtralTTAudioTokenizerDecoderCausalConv1d(
                mesh_device,
                state_dict=state_dict,
                conv_weight_base="output_proj.conv",
                kernel_size=ks,
                stride=1,
                pad_mode="replicate",
                in_channels=ic,
                out_channels=oc,
                output_channel_splits=8,
                weight_dtype=dtype,
                activations_dtype=dtype,
                output_dtype=dtype,
            )
        except KeyError:
            pass

        self.mm_audio_codebook_embedding: VoxtralTTAudioCodebookEmbedding | None = None
        if full_checkpoint is not None:
            emb_key = "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
            if emb_key in full_checkpoint:
                self.mm_audio_codebook_embedding = VoxtralTTAudioCodebookEmbedding(
                    mesh_device,
                    weight_bf16=full_checkpoint[emb_key].to(torch.bfloat16),
                )

        self.semantic_codebook_quantizer: VoxtralTTSemanticCodebookQuantizer | None = None
        if (
            "quantizer.semantic_codebook.embedding_sum" in state_dict
            and "quantizer.semantic_codebook.cluster_usage" in state_dict
        ):
            try:
                self.semantic_codebook_quantizer = VoxtralTTSemanticCodebookQuantizer(
                    mesh_device, state_dict=state_dict, dtype=dtype
                )
            except (KeyError, ValueError, RuntimeError):
                pass

        self._mm_offsets_tt: ttnn.Tensor | None = None
        if self.mm_audio_codebook_embedding is not None:
            off_cpu = (
                audio_tokenizer_mm_embedding_offsets(
                    SimpleNamespace(
                        semantic_codebook_size=tokenizer_cfg.semantic_codebook_size,
                        acoustic_codebook_size=tokenizer_cfg.acoustic_codebook_size,
                        n_acoustic_codebook=tokenizer_cfg.acoustic_dim,
                    )
                )
                .to(torch.int32)
                .view(1, -1, 1)
                .contiguous()
            )
            self._mm_offsets_tt = ttnn.from_torch(
                off_cpu,
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        nl = max(int(tokenizer_cfg.acoustic_codebook_size), 2)
        sc = 2.0 / (nl - 1)
        self._acoustic_fsq_scale_tt = ttnn.from_torch(
            torch.full((1, 1, 1), sc, dtype=torch.float32),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._acoustic_fsq_one_tt = ttnn.from_torch(
            torch.full((1, 1, 1), 1.0, dtype=torch.float32),
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def encoder_checkpoint_present(self) -> bool:
        """True if ``input_proj`` (and typically encoder) tensors exist in the loaded slice."""
        return self.input_proj is not None

    @classmethod
    def create_from_model_name(
        cls,
        mesh_device: ttnn.MeshDevice,
        *,
        model_name_or_path: str = DEFAULT_VOXTRAL_MODEL,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> "VoxtralTTAudioTokenizer":
        cfg = load_voxtral_config(model_name_or_path)
        full = _load_safetensors_state_dict(model_name_or_path)
        sd = extract_audio_tokenizer_state_dict(full)
        return cls(
            mesh_device,
            state_dict=sd,
            tokenizer_cfg=cfg.audio_tokenizer_args,
            dtype=dtype,
            full_checkpoint=full,
        )

    def input_projection(self, mel_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """Mel ``[B, 1, T, pretransform_patch_size]`` tile → ``[B, 1, T, dim]`` tile."""
        if self.input_proj is None:
            raise RuntimeError(
                "input_proj weights are not in this checkpoint (encoder stem absent). "
                f"Optional prefixes: {AUDIO_TOKENIZER_ENCODER_OPTIONAL_PREFIXES}."
            )
        return self.input_proj(mel_b1tc)

    def decoder_blocks_0_forward(self, latent_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """First decoder ``CausalConv1d``: ``[B,1,T,latent_dim]`` → ``[B,1,T,dim]`` (``decoder_blocks.0``)."""
        if self.decoder_blocks_0_conv is None:
            raise RuntimeError("decoder_blocks.0 conv is not loaded for this checkpoint.")
        return self.decoder_blocks_0_conv(latent_b1tc)

    def _decoder_sliding_window_attn_mask_tt(self, seq_len: int) -> ttnn.Tensor:
        """ALiBi + causal + sliding-window bias ``[1,1,H,T,T]`` on device (bf16 tile), current ``seq_len``."""
        mask = audio_tokenizer_sliding_window_attention_bias(
            self.cfg.n_heads,
            seq_len,
            self.cfg.attn_sliding_window_size,
        ).to(torch.bfloat16)
        return ttnn.from_torch(
            mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_full_forward(self, latent_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``[B,1,T,latent_dim]`` → ``[B,1,T_out,dim]``: all 12 decoder blocks, no ``output_proj``."""
        required = (
            ("decoder_blocks.0", self.decoder_blocks_0_conv),
            ("decoder_blocks.1.layers.0", self.decoder_blocks_1_layer0),
            ("decoder_blocks.1.layers.1", self.decoder_blocks_1_layer1),
            ("decoder_blocks.2", self.decoder_blocks_2_conv_transpose),
            ("decoder_blocks.3.layers.0", self.decoder_blocks_3_layer0),
            ("decoder_blocks.3.layers.1", self.decoder_blocks_3_layer1),
            ("decoder_blocks.4", self.decoder_blocks_4_conv_transpose),
            ("decoder_blocks.5.layers.0", self.decoder_blocks_5_layer0),
            ("decoder_blocks.5.layers.1", self.decoder_blocks_5_layer1),
            ("decoder_blocks.6", self.decoder_blocks_6_conv_transpose),
            ("decoder_blocks.7.layers.0", self.decoder_blocks_7_layer0),
            ("decoder_blocks.7.layers.1", self.decoder_blocks_7_layer1),
        )
        missing = [name for name, mod in required if mod is None]
        if missing:
            raise RuntimeError(
                "decode_full_forward requires the full decoder stack; missing weights for: " + ", ".join(missing)
            )

        b, _, input_t, input_c = (int(latent_b1tc.shape[i]) for i in range(4))
        # Pad T to a tile multiple for SDPA; trim after upsample (causal stack).
        min_decode_t = 32
        decode_t = max(min_decode_t, ((input_t + 31) // 32) * 32)
        stack_input = latent_b1tc
        padded_stack_input = None
        if input_t < decode_t:
            pad = ttnn.from_torch(
                torch.zeros((b, 1, decode_t - input_t, input_c), dtype=torch.bfloat16),
                device=self.mesh_device,
                dtype=self._dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            stack_input = ttnn.concat([latent_b1tc, pad], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pad)
            padded_stack_input = stack_input

        x = self.decoder_blocks_0_forward(stack_input)
        if padded_stack_input is not None:
            ttnn.deallocate(padded_stack_input)
        m = self._decoder_sliding_window_attn_mask_tt(int(x.shape[2]))
        x = self.decoder_blocks_1_forward(x, attn_mask=m)
        ttnn.deallocate(m)
        x = self.decoder_blocks_2_forward(x)
        m = self._decoder_sliding_window_attn_mask_tt(int(x.shape[2]))
        x = self.decoder_blocks_3_forward(x, attn_mask=m)
        ttnn.deallocate(m)
        x = self.decoder_blocks_4_forward(x)
        m = self._decoder_sliding_window_attn_mask_tt(int(x.shape[2]))
        x = self.decoder_blocks_5_forward(x, attn_mask=m)
        ttnn.deallocate(m)
        # Free L1 from prior SDPA before decoder_blocks_6 conv static-CB compile (P150).
        ttnn.synchronize_device(self.mesh_device)
        x = self.decoder_blocks_6_forward(x)
        m = self._decoder_sliding_window_attn_mask_tt(int(x.shape[2]))
        x = self.decoder_blocks_7_forward(x, attn_mask=m)
        ttnn.deallocate(m)
        if input_t < decode_t:
            upsample = 1
            for stride in parse_csv_ints(self.cfg.decoder_convs_strides_str)[1:]:
                upsample *= int(stride)
            target_t = input_t * upsample
            if int(x.shape[2]) > target_t:
                trimmed = ttnn.slice(x, [0, 0, 0, 0], [int(x.shape[0]), 1, target_t, int(x.shape[3])])
                ttnn.deallocate(x)
                x = trimmed
        return x

    def output_proj_forward(self, hidden_b1td: ttnn.Tensor) -> ttnn.Tensor:
        """``output_proj`` causal conv: ``[B,1,T,dim]`` → ``[B,1,T_out,C_mel]`` (``pretransform_patch_size`` channels)."""
        if self.output_proj_conv is None:
            raise RuntimeError("output_proj.conv is not loaded for this checkpoint.")
        return self.output_proj_conv(hidden_b1td)

    def decode_latent_to_mel_b1tc(self, latent_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``decode_full_forward`` then ``output_proj_forward`` (full decoder to mel features)."""
        h = self.decode_full_forward(latent_b1tc)
        return self.output_proj_forward(h)

    def mm_audio_codebook_embed_forward(self, indices_bt: ttnn.Tensor) -> ttnn.Tensor:
        """``mm_audio_embeddings.audio_codebook_embeddings`` lookup (requires ``full_checkpoint`` at construction)."""
        if self.mm_audio_codebook_embedding is None:
            raise RuntimeError(
                "MM audio codebook embedding was not loaded (missing consolidated "
                "'mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight')."
            )
        return self.mm_audio_codebook_embedding(indices_bt)

    def semantic_codebook_quantize_forward(self, x_b1ts: ttnn.Tensor) -> ttnn.Tensor:
        """Semantic VQ argmin from ``quantizer.semantic_codebook`` EMA buffers; returns ``[B,T]`` indices on device."""
        if self.semantic_codebook_quantizer is None:
            raise RuntimeError(
                "quantizer.semantic_codebook (embedding_sum / cluster_usage) not in audio_tokenizer state_dict."
            )
        return self.semantic_codebook_quantizer(x_b1ts)

    def mm_audio_encode_tokens_summed_forward(self, codes_b37t: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, 37, T]`` codes → ``[B, T, mm_dim]`` (or ``[T, mm_dim]`` for B=1): offset + embed + sum over codebooks."""
        if self.mm_audio_codebook_embedding is None or self._mm_offsets_tt is None:
            raise RuntimeError(
                "MM audio codebook embedding and offsets require full checkpoint "
                "(mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight)."
            )
        if len(codes_b37t.shape) != 3:
            raise ValueError(f"Expected [B, 37, T] codes, got {tuple(codes_b37t.shape)}")
        b, k, t = (int(codes_b37t.shape[i]) for i in range(3))
        if k != 37:
            raise ValueError(f"Expected 37 codebooks, got k={k}")

        codes_rm = ttnn.to_layout(codes_b37t, ttnn.ROW_MAJOR_LAYOUT)
        idx_rm = ttnn.add(codes_rm, self._mm_offsets_tt)
        flat_rm = ttnn.reshape(idx_rm, (b * k, t))
        emb_flat = self.mm_audio_codebook_embedding(flat_rm)
        d = int(emb_flat.shape[2])
        emb4 = ttnn.reshape(emb_flat, (b, k, t, d))
        out = ttnn.sum(emb4, dim=1)
        if b == 1:
            return ttnn.reshape(out, (t, d))
        return out

    def latent_from_codes_tt(self, codes_b37t: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, 37, T]`` codes → ``[B, 1, T, latent_dim]`` bf16 tile: semantic centroid embed + acoustic FSQ rescale, all on device."""
        if self.semantic_codebook_quantizer is None:
            raise RuntimeError("latent_from_codes_tt requires quantizer.semantic_codebook (centroid table on device).")
        if len(codes_b37t.shape) != 3:
            raise ValueError(f"Expected [B, 37, T] codes, got {tuple(codes_b37t.shape)}")
        b, k37, t = (int(codes_b37t.shape[i]) for i in range(3))
        if k37 != 37:
            raise ValueError(f"Expected 37 codebooks, got {k37}")

        rm = ttnn.to_layout(codes_b37t, ttnn.ROW_MAJOR_LAYOUT)
        sem_cb = ttnn.slice(rm, [0, 0, 0], [b, 1, t])
        ac_cb = ttnn.slice(rm, [0, 1, 0], [b, 37, t])
        ttnn.deallocate(rm)

        sem_bt = ttnn.reshape(sem_cb, (b, t))
        ttnn.deallocate(sem_cb)
        sem_bt_tile = ttnn.to_layout(sem_bt, ttnn.TILE_LAYOUT)
        ttnn.deallocate(sem_bt)
        sem_bts = self.semantic_codebook_quantizer.decode_semantic_embeddings(sem_bt_tile)
        ttnn.deallocate(sem_bt_tile)

        ac_f32 = ttnn.typecast(ac_cb, ttnn.float32)
        ttnn.deallocate(ac_cb)
        ac_scaled = ttnn.multiply(ac_f32, self._acoustic_fsq_scale_tt)
        ttnn.deallocate(ac_f32)
        ac_fsq = ttnn.subtract(ac_scaled, self._acoustic_fsq_one_tt)
        ttnn.deallocate(ac_scaled)
        ac_fsq_bf16 = ttnn.typecast(ac_fsq, ttnn.bfloat16)
        ttnn.deallocate(ac_fsq)
        ac_bt36 = ttnn.permute(ac_fsq_bf16, (0, 2, 1))
        ttnn.deallocate(ac_fsq_bf16)

        sem_bt_rm = ttnn.to_layout(sem_bts, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(sem_bts)
        ac_bt36_rm = ttnn.to_layout(ac_bt36, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(ac_bt36)
        latent_bt = ttnn.concat([sem_bt_rm, ac_bt36_rm], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sem_bt_rm)
        ttnn.deallocate(ac_bt36_rm)
        latent_b1tc = ttnn.reshape(latent_bt, (b, 1, t, audio_tokenizer_latent_dim(self.cfg)))
        ttnn.deallocate(latent_bt)
        return ttnn.to_layout(latent_b1tc, ttnn.TILE_LAYOUT)

    def decoder_blocks_1_layer0_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """First transformer layer inside ``decoder_blocks.1``."""
        if self.decoder_blocks_1_layer0 is None:
            raise RuntimeError("decoder_blocks.1.layers.0 is not loaded for this checkpoint.")
        return self.decoder_blocks_1_layer0(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_1_layer1_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """Second transformer layer inside ``decoder_blocks.1``."""
        if self.decoder_blocks_1_layer1 is None:
            raise RuntimeError("decoder_blocks.1.layers.1 is not loaded for this checkpoint.")
        return self.decoder_blocks_1_layer1(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_1_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """First decoder transformer stack (2 layers for the released config)."""
        x_b1td = self.decoder_blocks_1_layer0_forward(x_b1td, attn_mask=attn_mask)
        return self.decoder_blocks_1_layer1_forward(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_3_layer0_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """First transformer layer inside ``decoder_blocks.3``."""
        if self.decoder_blocks_3_layer0 is None:
            raise RuntimeError("decoder_blocks.3.layers.0 is not loaded for this checkpoint.")
        return self.decoder_blocks_3_layer0(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_3_layer1_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """Second transformer layer inside ``decoder_blocks.3``."""
        if self.decoder_blocks_3_layer1 is None:
            raise RuntimeError("decoder_blocks.3.layers.1 is not loaded for this checkpoint.")
        return self.decoder_blocks_3_layer1(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_3_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """Decoder transformer stack after ``decoder_blocks.2`` transpose (2 layers)."""
        x_b1td = self.decoder_blocks_3_layer0_forward(x_b1td, attn_mask=attn_mask)
        return self.decoder_blocks_3_layer1_forward(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_2_forward(self, x_b1td: ttnn.Tensor) -> ttnn.Tensor:
        """First decoder upsample ``CausalConvTranspose1d`` (``decoder_blocks.2``)."""
        if self.decoder_blocks_2_conv_transpose is None:
            raise RuntimeError("decoder_blocks.2 conv transpose is not loaded for this checkpoint.")
        return self.decoder_blocks_2_conv_transpose(x_b1td)

    def decoder_blocks_4_forward(self, x_b1td: ttnn.Tensor) -> ttnn.Tensor:
        """Second decoder upsample ``CausalConvTranspose1d`` (``decoder_blocks.4``)."""
        if self.decoder_blocks_4_conv_transpose is None:
            raise RuntimeError("decoder_blocks.4 conv transpose is not loaded for this checkpoint.")
        return self.decoder_blocks_4_conv_transpose(x_b1td)

    def decoder_blocks_5_layer0_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        if self.decoder_blocks_5_layer0 is None:
            raise RuntimeError("decoder_blocks.5.layers.0 is not loaded for this checkpoint.")
        return self.decoder_blocks_5_layer0(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_5_layer1_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        if self.decoder_blocks_5_layer1 is None:
            raise RuntimeError("decoder_blocks.5.layers.1 is not loaded for this checkpoint.")
        return self.decoder_blocks_5_layer1(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_5_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """Decoder transformer stack after ``decoder_blocks.4`` transpose (2 layers)."""
        x_b1td = self.decoder_blocks_5_layer0_forward(x_b1td, attn_mask=attn_mask)
        return self.decoder_blocks_5_layer1_forward(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_6_forward(self, x_b1td: ttnn.Tensor) -> ttnn.Tensor:
        """Third decoder upsample ``CausalConvTranspose1d`` (``decoder_blocks.6``)."""
        if self.decoder_blocks_6_conv_transpose is None:
            raise RuntimeError("decoder_blocks.6 conv transpose is not loaded for this checkpoint.")
        return self.decoder_blocks_6_conv_transpose(x_b1td)

    def decoder_blocks_7_layer0_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        if self.decoder_blocks_7_layer0 is None:
            raise RuntimeError("decoder_blocks.7.layers.0 is not loaded for this checkpoint.")
        return self.decoder_blocks_7_layer0(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_7_layer1_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        if self.decoder_blocks_7_layer1 is None:
            raise RuntimeError("decoder_blocks.7.layers.1 is not loaded for this checkpoint.")
        return self.decoder_blocks_7_layer1(x_b1td, attn_mask=attn_mask)

    def decoder_blocks_7_forward(self, x_b1td: ttnn.Tensor, *, attn_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """Final decoder transformer stack (2 layers)."""
        x_b1td = self.decoder_blocks_7_layer0_forward(x_b1td, attn_mask=attn_mask)
        return self.decoder_blocks_7_layer1_forward(x_b1td, attn_mask=attn_mask)

    def encoder_downsample_after_transformer_0_forward(self, x_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """First encoder strided conv after ``encoder_blocks.0``. Raises if encoder weights are absent."""
        if self.encoder_downsample_after_transformer_0 is None:
            raise RuntimeError("encoder_blocks.1 downsample conv is not loaded for this checkpoint.")
        return self.encoder_downsample_after_transformer_0(x_b1tc)

    def latent_from_codes(self, codes_b37t: torch.Tensor) -> ttnn.Tensor:
        """``[B, 37, T]`` host integer codes → ``[B, 1, T, latent_dim]`` bf16 tile on device."""
        codes_tt = ttnn.from_torch(
            codes_b37t.to(torch.int64).clamp(min=0).to(torch.uint32).contiguous(),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = self.latent_from_codes_tt(codes_tt)
        ttnn.deallocate(codes_tt)
        return out

    def pretransform_decode_tt(self, mel_b1tc: ttnn.Tensor) -> ttnn.Tensor:
        """``[B,1,T,C_mel]`` → ``[B,1,T*C_mel]`` waveform on device (reshape only, no weights)."""
        if len(mel_b1tc.shape) != 4 or int(mel_b1tc.shape[1]) != 1:
            raise ValueError(f"Expected [B,1,T,C_mel] mel, got {tuple(mel_b1tc.shape)}")
        b, _, t, c_mel = (int(mel_b1tc.shape[i]) for i in range(4))
        mel_rm = ttnn.to_layout(mel_b1tc, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(mel_rm, (b, 1, t * c_mel))

    def pretransform_decode_torch(self, mel_b1tc: ttnn.Tensor) -> torch.Tensor:
        """``[B,1,T,C_mel]`` TT mel → ``[B,1,T*C_mel]`` float32 host tensor."""
        if len(mel_b1tc.shape) != 4 or int(mel_b1tc.shape[1]) != 1:
            raise ValueError(f"Expected [B,1,T,C_mel] mel, got {tuple(mel_b1tc.shape)}")
        b, _, t, c_mel = (int(mel_b1tc.shape[i]) for i in range(4))
        # Flatten on host — large TT reshape blows L1 on P150.
        mel_host = ttnn.to_torch(mel_b1tc).float()
        return mel_host.reshape(b, 1, t * c_mel)
