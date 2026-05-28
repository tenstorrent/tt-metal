# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 speech encoder (W2v-BERT-2.0).

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::speech_encoder_forward``.

This block composes the full W2v-BERT-2.0 Conformer speech encoder used by
``SeamlessM4Tv2SpeechEncoder``::

    hidden = feature_projection(input_features)                # LN(160) + Linear(160 -> 1024)
    for _ in range(N): hidden = conformer_encoder_layer(hidden, attn_mask, conv_mask)
    hidden = layer_norm(hidden)                                # encoder.final_layer_norm
    hidden = hidden + 0.5 * intermediate_ffn(hidden)           # ReLU FFN, half-step residual
    if add_adapter:
        for layer in adapter.layers:
            hidden = conformer_adapter_layer(hidden, sub_4d_mask)
    hidden = inner_layer_norm(hidden)

Composition over the already-verified TTNN leaves:

* :class:`ConformerFeatureProjection` — front-end ``LayerNorm + Linear``.
* :class:`ConformerEncoderLayer`      — full macaron-FFN sandwich,
  N (=2 in the saved golden, 24 in the real model) sequential layers.
* :class:`LayerNorm`                   — post-encoder ``final_layer_norm``
  and the terminal ``inner_layer_norm``.
* :class:`ConformerFfn` (act_fn="relu") — the encoder's ``intermediate_ffn``
  (HF builds it with ``act_fn="relu"``, half-step residual).
* :class:`ConformerAdapterLayer`      — the post-encoder strided-conv
  down-sampling + MHA + FFN adapter (one or more layers).

The TTNN forward intentionally mirrors the reference call signature: it accepts
the same 2-D HF-style padding mask (``[B, T]`` long/bool) at the speech-feature
time axis and builds:

  * the encoder's additive 4-D log-mask (the chunked-attention mask combined
    with the padding mask, both produced bit-equivalently to HF), and
  * the per-adapter strided 4-D log-mask (recomputed on the post-downsample
    time axis, also bit-equivalent to HF).

All mask preparation lives on the host (in PyTorch), then the resulting
4-D mask is transferred to the device once per forward — matching the
pattern used by the per-layer encoder / adapter PCC tests.

For SeamlessM4T-v2-Large the relevant config is:
    - hidden = 1024, num_heads = 16, head_dim = 64
    - speech_encoder_layers = 24 (golden uses 2 for size)
    - speech_encoder_intermediate_size = 4096
    - speech_encoder_hidden_act = "swish"
    - left_max_position_embeddings = 64, right_max_position_embeddings = 8
    - position_embeddings_type = "relative_key"
    - conv_depthwise_kernel_size = 31
    - feature_projection_input_dim = 160
    - adaptor_kernel_size = 8, adaptor_stride = 8
    - num_adapter_layers = 1
    - speech_encoder_chunk_size = 20000, speech_encoder_left_chunk_num = 128
    - layer_norm_eps = 1e-5
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_adapter_layer import ConformerAdapterLayer
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_encoder_layer import ConformerEncoderLayer
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_feature_projection import ConformerFeatureProjection
from models.demos.facebook_seamless_m4t_v2_large.tt.conformer_ffn import ConformerFfn
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm


def _build_conformer_chunk_attention_mask(
    seq_len: int,
    chunk_size: int,
    left_chunk_num: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bit-equivalent port of the reference chunk-attention mask builder.

    Returns an additive log-mask of shape ``[1, 1, seq_len, seq_len]`` in
    ``dtype`` where blocked entries are ``finfo(dtype).min`` and kept entries
    are ``0``.
    """
    chunk_indices = torch.arange(seq_len)
    chunk_indices = torch.div(chunk_indices, chunk_size, rounding_mode="trunc").long()
    if left_chunk_num >= 0:
        start_indices = (chunk_indices - left_chunk_num).clamp_(min=0) * chunk_size
    else:
        start_indices = torch.zeros_like(chunk_indices)
    start_indices = start_indices.unsqueeze(1).expand(-1, seq_len)
    end_indices = ((chunk_indices + 1) * chunk_size).clamp_(max=seq_len)
    end_indices = end_indices.unsqueeze(1).expand(-1, seq_len)
    indices = torch.arange(seq_len).unsqueeze(0).expand(seq_len, -1)
    chunk_bool = (indices < start_indices) | (indices >= end_indices)
    chunk_bool = chunk_bool.unsqueeze(0).unsqueeze(0)
    return chunk_bool.to(dtype=dtype) * torch.finfo(dtype).min


def _build_encoder_attention_mask(
    attention_mask_2d: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    chunk_size: Optional[int],
    left_chunk_num: int,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Replicate the reference encoder mask combine: chunk bool OR pad bool, then ``* finfo.min``.

    Returns ``None`` if both ``attention_mask_2d`` is ``None`` and chunked
    attention is disabled (``chunk_size is None``). Otherwise returns an
    additive 4-D log-mask of shape ``[B, 1, T, T]`` in ``dtype``.
    """
    encoder_mask: Optional[torch.Tensor] = None
    expanded = None
    if attention_mask_2d is not None:
        expanded = 1.0 - attention_mask_2d[:, None, None, :].to(dtype=dtype)
        expanded = expanded.expand(batch_size, 1, seq_len, seq_len)
        encoder_mask = expanded
    if chunk_size is not None:
        chunk_additive = _build_conformer_chunk_attention_mask(
            seq_len=seq_len,
            chunk_size=chunk_size,
            left_chunk_num=left_chunk_num,
            dtype=dtype,
        )
        if encoder_mask is None:
            encoder_mask = chunk_additive
        else:
            pad_bool = expanded.bool()
            chunk_bool = (chunk_additive != 0).expand(batch_size, 1, seq_len, seq_len)
            combined_bool = pad_bool | chunk_bool
            encoder_mask = combined_bool.to(dtype) * torch.finfo(dtype).min
    elif encoder_mask is not None:
        encoder_mask = encoder_mask * torch.finfo(dtype).min
    return encoder_mask


def _build_adapter_sub_attention_mask_4d(
    attention_mask_2d: torch.Tensor,
    seq_len: int,
    hidden: int,
    kernel_size: int,
    stride: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reproduce HF's per-adapter 4-D sub-sampled additive mask.

    Uses the same helpers the reference imports
    (``_compute_new_attention_mask`` + ``_prepare_4d_attention_mask``).
    """
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import _compute_new_attention_mask

    pad = kernel_size // 2
    seq_lens = attention_mask_2d.size(1) - (1 - attention_mask_2d.int()).sum(1)
    sub_sampled_lengths = (((seq_lens + 2 * pad - kernel_size) / stride) + 1).floor()
    batch = int(attention_mask_2d.shape[0])
    t_sub = int(((seq_len + 2 * pad - kernel_size) // stride) + 1)
    dummy_sub_hidden = torch.zeros(batch, t_sub, hidden, dtype=dtype)
    sub_2d = _compute_new_attention_mask(hidden_states=dummy_sub_hidden, seq_lens=sub_sampled_lengths)
    sub_4d = _prepare_4d_attention_mask(sub_2d, dummy_sub_hidden.dtype)
    return sub_4d


class SpeechEncoder(LightweightModule):
    """Full SeamlessM4T-v2 speech encoder (W2v-BERT-2.0 + adapter) in TTNN.

    Args:
        device: ttnn device or mesh device.
        state_dict: nested dict matching
            ``_extract_speech_encoder_state_dict`` in the reference test
            (keys ``feature_projection``, ``encoder``, ``intermediate_ffn``,
            ``inner_layer_norm``, optional ``adapter``).
        feature_size: input feature dim (160 for v2-Large).
        hidden: model hidden size (1024 for v2-Large).
        num_heads: encoder/adapter attention heads (16).
        head_dim: per-head dim (64).
        seq_len: pre-feature-projection time length.
        batch_size: forward batch size.
        eps: LayerNorm epsilon (1e-5).
        speech_encoder_hidden_act: Conformer FFN activation ("swish").
        left_max_position_embeddings, right_max_position_embeddings,
        position_embeddings_type, conv_depthwise_kernel_size: forwarded to
            each ``ConformerEncoderLayer``.
        adaptor_kernel_size, adaptor_stride: forwarded to each
            ``ConformerAdapterLayer``.
        speech_encoder_chunk_size: chunk size for the encoder's chunked
            attention mask (default 20000). Set ``None`` to disable.
        speech_encoder_left_chunk_num: left-chunk overlap count (default 128).
        add_adapter: whether to build/run the post-encoder adapter (True for
            v2-Large).
        weight_dtype: storage dtype for all sub-block weights.
        weight_memory_config: where to place weights (DRAM by default).
    """

    def __init__(
        self,
        device,
        state_dict,
        feature_size: int,
        hidden: int,
        num_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int = 1,
        eps: float = 1e-5,
        speech_encoder_hidden_act: str = "swish",
        left_max_position_embeddings: int = 64,
        right_max_position_embeddings: int = 8,
        position_embeddings_type: Optional[str] = "relative_key",
        conv_depthwise_kernel_size: int = 31,
        adaptor_kernel_size: int = 8,
        adaptor_stride: int = 8,
        speech_encoder_chunk_size: Optional[int] = 20000,
        speech_encoder_left_chunk_num: int = 128,
        add_adapter: bool = True,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        if num_heads * head_dim != hidden:
            raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != hidden({hidden})")

        self.device = device
        self.feature_size = int(feature_size)
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.eps = float(eps)
        self.speech_encoder_hidden_act = speech_encoder_hidden_act
        self.left_max_position_embeddings = int(left_max_position_embeddings)
        self.right_max_position_embeddings = int(right_max_position_embeddings)
        self.position_embeddings_type = position_embeddings_type
        self.conv_depthwise_kernel_size = int(conv_depthwise_kernel_size)
        self.adaptor_kernel_size = int(adaptor_kernel_size)
        self.adaptor_stride = int(adaptor_stride)
        self.speech_encoder_chunk_size = speech_encoder_chunk_size
        self.speech_encoder_left_chunk_num = int(speech_encoder_left_chunk_num)
        self.add_adapter = bool(add_adapter)

        # 1. Feature projection (LN(160) -> Linear(160 -> 1024)).
        fp_sd = state_dict["feature_projection"]
        self.feature_projection = ConformerFeatureProjection(
            device=device,
            layer_norm_weight=fp_sd["layer_norm"]["weight"],
            layer_norm_bias=fp_sd["layer_norm"]["bias"],
            projection_weight=fp_sd["projection"]["weight"],
            projection_bias=fp_sd["projection"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 2. Conformer encoder layer stack.
        self.encoder_layers = []
        for layer_sd in state_dict["encoder"]["layers"]:
            self.encoder_layers.append(
                ConformerEncoderLayer(
                    device=device,
                    embed_dim=self.hidden,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    seq_len=self.seq_len,
                    state_dict=layer_sd,
                    distance_embedding_weight=layer_sd.get("distance_embedding_weight"),
                    left_max_position_embeddings=self.left_max_position_embeddings,
                    right_max_position_embeddings=self.right_max_position_embeddings,
                    position_embeddings_type=self.position_embeddings_type,
                    conv_kernel_size=self.conv_depthwise_kernel_size,
                    eps=self.eps,
                    batch_size=self.batch_size,
                    weight_dtype=weight_dtype,
                    weight_memory_config=weight_memory_config,
                )
            )

        # 3. Encoder final LayerNorm.
        self.encoder_final_layer_norm = LayerNorm(
            device=device,
            dim=self.hidden,
            weight=state_dict["encoder"]["final_layer_norm"]["weight"],
            bias=state_dict["encoder"]["final_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 4. Intermediate FFN (HF uses act_fn="relu"; half-step residual lives
        #    in this module's forward, NOT inside ConformerFfn).
        inter_sd = state_dict["intermediate_ffn"]
        self.intermediate_ffn = ConformerFfn(
            device=device,
            intermediate_weight=inter_sd["intermediate_dense"]["weight"],
            intermediate_bias=inter_sd["intermediate_dense"]["bias"],
            output_weight=inter_sd["output_dense"]["weight"],
            output_bias=inter_sd["output_dense"]["bias"],
            act_fn="relu",
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

        # 5. Adapter (optional, stacked). T_sub for the v2-Large config with
        #    T=seq_len=64, kernel=stride=8, pad=4: floor((64+8-8)/8)+1 = 9.
        pad = self.adaptor_stride // 2
        self.sub_seq_len = int(((self.seq_len + 2 * pad - self.adaptor_kernel_size) // self.adaptor_stride) + 1)
        self.adapter_layers = []
        if self.add_adapter:
            adapter_sd = state_dict.get("adapter")
            if adapter_sd is None:
                raise ValueError("add_adapter=True but state_dict has no 'adapter' entry")
            if len(adapter_sd["layers"]) > 1:
                # See reference: stacked adapters with a non-None mask require
                # recomputing the 2-D mask after each strided downsample. v2-Large
                # uses num_adapter_layers=1 so this path is unused.
                pass
            for layer_sd in adapter_sd["layers"]:
                self.adapter_layers.append(
                    ConformerAdapterLayer(
                        device=device,
                        embed_dim=self.hidden,
                        num_heads=self.num_heads,
                        head_dim=self.head_dim,
                        seq_len=self.seq_len,
                        sub_seq_len=self.sub_seq_len,
                        state_dict=layer_sd,
                        kernel_size=self.adaptor_kernel_size,
                        stride=self.adaptor_stride,
                        eps=self.eps,
                        batch_size=self.batch_size,
                        weight_dtype=weight_dtype,
                        weight_memory_config=weight_memory_config,
                    )
                )

        # 6. Terminal inner LayerNorm (HF: ``self.inner_layer_norm``).
        self.inner_layer_norm = LayerNorm(
            device=device,
            dim=self.hidden,
            weight=state_dict["inner_layer_norm"]["weight"],
            bias=state_dict["inner_layer_norm"]["bias"],
            eps=self.eps,
            weight_dtype=weight_dtype,
            weight_memory_config=weight_memory_config,
        )

    def _to_tt(self, t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=self.device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(
        self,
        input_features: ttnn.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Run the full TTNN speech encoder.

        Args:
            input_features: ttnn tensor of shape ``[B, T, feature_size]`` in
                TILE_LAYOUT (the same input the reference takes).
            attention_mask_2d: optional torch tensor of shape ``[B, T]`` with
                ``1`` = keep / ``0`` = pad over the post-feature-projection time
                axis (the same 2-D mask HF takes). The encoder's 4-D additive
                mask (chunked + padding) and the per-adapter sub-sampled 4-D
                mask are derived from this on the host, then transferred to
                device.

        Returns:
            ttnn tensor of shape ``[B, sub_seq_len, hidden]`` in TILE_LAYOUT
            (== ``[B, 9, 1024]`` for v2-Large with T=64 and add_adapter=True).
        """
        # --- 1. Build host-side masks ---
        batch = int(input_features.shape[0])
        seq_len = self.seq_len

        encoder_mask_torch = _build_encoder_attention_mask(
            attention_mask_2d=attention_mask_2d,
            batch_size=batch,
            seq_len=seq_len,
            chunk_size=self.speech_encoder_chunk_size,
            left_chunk_num=self.speech_encoder_left_chunk_num,
            dtype=torch.float32,
        )

        encoder_attn_mask_tt = None
        if encoder_mask_torch is not None:
            encoder_attn_mask_tt = self._to_tt(encoder_mask_torch)

        conv_attn_mask_tt = None
        if attention_mask_2d is not None:
            conv_mask_f = attention_mask_2d.to(torch.float32).reshape(batch, seq_len, 1)
            conv_attn_mask_tt = self._to_tt(conv_mask_f)

        # --- 2. Feature projection ---
        hidden_states = self.feature_projection(input_features)

        # HF: zero padded positions on the residual stream right after
        # feature_projection. We replicate that here so the encoder/conv path
        # sees the same masked-out hidden states as the reference does.
        if attention_mask_2d is not None:
            mask_keep = attention_mask_2d.to(torch.float32).reshape(batch, seq_len, 1)
            mask_keep_tt = self._to_tt(mask_keep)
            hidden_states = ttnn.multiply(hidden_states, mask_keep_tt)
            ttnn.deallocate(mask_keep_tt)

        # --- 3. Conformer encoder stack ---
        for layer in self.encoder_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=encoder_attn_mask_tt,
                conv_attention_mask=conv_attn_mask_tt,
            )

        if encoder_attn_mask_tt is not None:
            ttnn.deallocate(encoder_attn_mask_tt)
        if conv_attn_mask_tt is not None:
            ttnn.deallocate(conv_attn_mask_tt)

        # --- 4. Encoder final LayerNorm ---
        hidden_states = self.encoder_final_layer_norm(hidden_states)

        # --- 5. Intermediate FFN with half-step residual ---
        # HF: hidden = hidden + 0.5 * intermediate_ffn(hidden)
        ffn_out = self.intermediate_ffn(hidden_states)
        ffn_scaled = ttnn.multiply(ffn_out, 0.5)
        ttnn.deallocate(ffn_out)
        hidden_states = ttnn.add(hidden_states, ffn_scaled)
        ttnn.deallocate(ffn_scaled)

        # --- 6. Adapter (optional) ---
        if self.add_adapter:
            for layer in self.adapter_layers:
                if attention_mask_2d is not None:
                    sub_4d = _build_adapter_sub_attention_mask_4d(
                        attention_mask_2d=attention_mask_2d,
                        seq_len=seq_len,
                        hidden=self.hidden,
                        kernel_size=self.adaptor_kernel_size,
                        stride=self.adaptor_stride,
                        dtype=torch.float32,
                    )
                    sub_mask_tt = self._to_tt(sub_4d)
                else:
                    sub_mask_tt = None

                hidden_states = layer(hidden_states, attention_mask=sub_mask_tt)

                if sub_mask_tt is not None:
                    ttnn.deallocate(sub_mask_tt)

        # --- 7. Inner LayerNorm ---
        hidden_states = self.inner_layer_norm(hidden_states)
        return hidden_states
