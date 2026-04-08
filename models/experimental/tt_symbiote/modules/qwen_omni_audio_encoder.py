# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HF ``Qwen3OmniMoeAudioEncoder`` → TTNN (strided convs + ``conv_out`` linear + encoder stack + ``ln_post`` + MLP head)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import ttnn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import _get_feat_extract_output_lengths

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.activation import _ttnn_mesh_to_torch_one_replica
from models.experimental.tt_symbiote.modules.conv import TTNNQwenOmniConv2dNHWC
from models.experimental.tt_symbiote.modules.encoder_layer import TTNNQwen3OmniMoeAudioEncoderLayer
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNQwenLayerNorm
from models.experimental.tt_symbiote.modules.qwen_omni_vision_patch import _ensure_ttnn, _replicate_mapper


def _to_torch_tensor(x, mesh_device=None):
    """Materialize TTNN / wrappers to torch for masking, padding, and ``torch.cat``."""
    if isinstance(x, torch.Tensor) and not isinstance(x, TorchTTNNTensor):
        return x
    if isinstance(x, TorchTTNNTensor):
        return x.to_torch
    if isinstance(x, ttnn.Tensor):
        return _ttnn_mesh_to_torch_one_replica(x, mesh_device)
    return x


def _gelu_after_conv(x):
    """Child convs use bypass: outputs stay ``ttnn.Tensor``; PyTorch ``F.gelu`` cannot accept them."""
    if isinstance(x, ttnn.Tensor):
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return F.gelu(x)


def _activation_kind_from_config(name: str) -> str:
    n = (name or "gelu").lower()
    if "silu" in n or "swish" in n:
        return "silu"
    return "gelu"


@trace_enabled
class TTNNQwen3OmniMoeAudioEncoder(TTNNModule):
    """Same control flow as HF ``Qwen3OmniMoeAudioEncoder.forward``; TTNN modules for compute-heavy ops."""

    @classmethod
    def from_torch(cls, enc):
        m = cls()
        m._fallback_torch_layer = enc
        # Symbiote must not convert ``feature_lens`` / ``input_features`` to ttnn: HF uses torch
        # ops and ``_get_feat_extract_output_lengths`` on lengths (see ``device_management``).
        m._symbiote_force_bypass_inputs = True
        m.config = enc.config
        m.dropout = enc.dropout
        m.embed_scale = enc.embed_scale
        m.num_mel_bins = enc.num_mel_bins
        m.max_source_positions = enc.max_source_positions
        m.n_window = enc.n_window
        m.n_window_infer = enc.n_window_infer
        m.conv_chunksize = enc.conv_chunksize
        m._positional_embedding = enc.positional_embedding.positional_embedding.data.clone()
        m.conv2d1 = TTNNQwenOmniConv2dNHWC.from_torch(enc.conv2d1)
        m.conv2d2 = TTNNQwenOmniConv2dNHWC.from_torch(enc.conv2d2)
        m.conv2d3 = TTNNQwenOmniConv2dNHWC.from_torch(enc.conv2d3)
        m.conv_out = TTNNLinear.from_torch(enc.conv_out)
        m.layers = tuple(TTNNQwen3OmniMoeAudioEncoderLayer.from_torch(layer) for layer in enc.layers)
        m.ln_post = TTNNQwenLayerNorm.from_torch(enc.ln_post)
        if isinstance(m.ln_post, TTNNQwenLayerNorm):
            # Same as encoder-layer norms: hidden states are replicated full-width before ``ln_post``.
            m.ln_post._force_replicated_input_layernorm = True
        m.proj1 = TTNNLinear.from_torch(enc.proj1)
        m.proj2 = TTNNLinear.from_torch(enc.proj2)
        m._activation_kind = _activation_kind_from_config(enc.config.activation_function)
        return m

    def preprocess_weights_impl(self):
        for mod in (self.conv2d1, self.conv2d2, self.conv2d3, self.conv_out, self.ln_post, self.proj1, self.proj2):
            if isinstance(mod, TTNNModule):
                mod.preprocess_weights()
        for layer in self.layers:
            layer.preprocess_weights()

    def move_weights_to_device_impl(self):
        for mod in (self.conv2d1, self.conv2d2, self.conv2d3, self.conv_out, self.ln_post, self.proj1, self.proj2):
            if isinstance(mod, TTNNModule):
                mod.move_weights_to_device()
        for layer in self.layers:
            layer.move_weights_to_device()

    def deallocate_weights_impl(self):
        for mod in (self.conv2d1, self.conv2d2, self.conv2d3, self.conv_out, self.ln_post, self.proj1, self.proj2):
            if isinstance(mod, TTNNModule):
                mod.deallocate_weights()
        for layer in self.layers:
            layer.deallocate_weights()
        super().deallocate_weights_impl()

    def _act_ttnn(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._activation_kind == "silu":
            return ttnn.silu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.gelu(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _ln_post(self, hidden_states):
        if isinstance(self.ln_post, TTNNQwenLayerNorm):
            return self.ln_post(hidden_states)
        th = _to_torch_tensor(hidden_states, self.device)
        th = self.ln_post(th)
        return _ensure_ttnn(th, self.device, mesh_mapper=_replicate_mapper(self.device))

    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
        **kwargs,
    ):
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.full((chunk_num.sum(),), self.n_window * 2, dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Parent encoder uses ``_symbiote_force_bypass_inputs`` so ``chunk`` stays torch; child convs also
        # use bypass and receive torch unchanged — upload here before ``TTNNQwenOmniConv2dNHWC`` (expects ttnn).
        mapper = _replicate_mapper(self.device)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            chunk_ttnn = _ensure_ttnn(chunk, self.device, mesh_mapper=mapper)
            padded_embed = self.conv2d1(chunk_ttnn)
            padded_embed = _gelu_after_conv(padded_embed)
            padded_embed = self.conv2d2(padded_embed)
            padded_embed = _gelu_after_conv(padded_embed)
            padded_embed = self.conv2d3(padded_embed)
            padded_embed = _gelu_after_conv(padded_embed)
            padded_embed = _to_torch_tensor(padded_embed, self.device)
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        linear_in = padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        linear_in_ttnn = _ensure_ttnn(linear_in, self.device, mesh_mapper=mapper)
        padded_embed = self.conv_out(linear_in_ttnn)

        padded_embed = _to_torch_tensor(padded_embed, self.device)
        pos = (
            self._positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(device=padded_embed.device, dtype=padded_embed.dtype)
        )
        padded_embed = padded_embed + pos
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self._ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self._act_ttnn(hidden_states)
        hidden_states = self.proj2(hidden_states)
        # HF ``get_audio_features`` does ``last_hidden_state.to(inputs_embeds.device, dtype)`` (torch API).
        hidden_states = _to_torch_tensor(hidden_states, self.device)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)
