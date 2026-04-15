# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Embedding layer implementations for TTNN."""

from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.run_config import (
    DistributedTensorConfig,
    trace_enabled,
)
from models.experimental.tt_symbiote.modules.decoder_layer import _next_power_of_2


@trace_enabled
class TTNNEmbedding(TTNNModule):
    """TTNN-accelerated embedding lookup for Ling/Bailing models.

    Replaces nn.Embedding (word_embeddings). Weight is replicated across all
    devices on a mesh — no CCL needed since downstream column-parallel linears
    expect replicated input.
    """

    @classmethod
    def from_torch(cls, embedding: nn.Embedding):
        new_layer = cls()
        new_layer._fallback_torch_layer = embedding
        return new_layer

    def preprocess_weights_impl(self):
        self.tt_weight_host = ttnn.from_torch(
            self.torch_layer.weight.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        mesh_mapper = ttnn.ShardTensor2dMesh(self.device, dims=(None, 1), mesh_shape=list(self.device.shape))
        self.tt_weight = ttnn.to_device(
            ttnn.from_torch(
                self.torch_layer.weight.data,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mesh_mapper,
            ),
            self.device,
        )

    def deallocate_weights_impl(self):
        ttnn.deallocate(self.tt_weight)
        super().deallocate_weights_impl()

    def forward(self, tt_indices):
        out = ttnn.embedding(
            tt_indices,
            self.tt_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out


class TTNNBailingPaddedEmbedding(TTNNModule):
    """Padded embedding wrapper that pads sequence to power-of-2 before lookup.

    Weight is sharded along hidden dim across mesh devices. Indices are
    replicated. Each device produces a slice of the hidden dimension.
    """

    @staticmethod
    def _pad_dim(tensor, dim, pad_amount, value=0.0):
        """Pad a single dimension of a tensor by ``pad_amount``."""
        rank = len(tensor.shape)
        padding = tuple((0, pad_amount if i == dim else 0) for i in range(rank))
        return ttnn.pad(tensor, padding=padding, value=value)

    @staticmethod
    def _slice_dim(tensor, dim, length):
        """Slice a tensor along ``dim`` to ``length``."""
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        ends[dim] = length
        return ttnn.slice(tensor, starts, ends)

    @classmethod
    def from_torch(cls, embedding: nn.Embedding):
        new_layer = cls()
        new_layer.embedder = TTNNEmbedding.from_torch(embedding)
        return new_layer

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_ids):
        rank = len(input_ids.shape)
        seq_dim = rank - 1  # sequence length is always second-to-last
        seq_len = input_ids.shape[seq_dim]
        padded_seq_len = _next_power_of_2(seq_len)
        pad_amount = padded_seq_len - seq_len
        if pad_amount > 0:
            input_ids = self._pad_dim(input_ids, seq_dim, pad_amount, value=0)

        input_ids = self.embedder(input_ids)
        if pad_amount > 0:
            input_ids = self._slice_dim(input_ids, seq_dim, seq_len)

        return input_ids


@trace_enabled
class TTNNRotaryEmbeddingCompute(TTNNModule):
    """Computes cos/sin from inv_freq and position_ids."""

    @staticmethod
    def from_torch(rotary_emb):
        new_layer = TTNNRotaryEmbeddingCompute()
        new_layer._fallback_torch_layer = rotary_emb
        return new_layer

    def forward(self, tt_inv_freq, position_ids):
        # Typecast int32 -> bfloat16 (requires last dim multiple of 32)
        position_ids = ttnn.typecast(position_ids, ttnn.bfloat16)
        position_ids = ttnn.to_layout(position_ids, ttnn.TILE_LAYOUT)

        # Reshape [batch, padded_seq] -> [batch, 1, padded_seq]
        position_ids = ttnn.reshape(position_ids, (position_ids.shape[0], 1, -1))

        # freqs = inv_freq @ position_ids -> [batch, rotary_dim/2, seq] -> transpose -> [batch, seq, rotary_dim/2]
        freqs = ttnn.matmul(tt_inv_freq, position_ids)
        freqs = ttnn.transpose(freqs, -2, -1)

        # emb = cat(freqs, freqs) -> [batch, seq, rotary_dim]
        emb = ttnn.concat([freqs, freqs], dim=-1)

        # cos/sin
        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)

        if self.torch_layer.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.torch_layer.attention_scaling)
            sin = ttnn.multiply(sin, self.torch_layer.attention_scaling)

        return cos, sin


class TTNNBailingRotaryEmbedding(TTNNModule):
    """TTNN-accelerated rotary position embedding for Ling/Bailing models.

    Replaces BailingMoeV2RotaryEmbedding. Pre-computes cos/sin caches on device
    via BailingRotarySetup. Returns torch cos/sin in HF doubled-half format —
    the downstream TTNNBailingMoEAttention handles format conversion internally.
    """

    @classmethod
    def from_torch(cls, rotary_emb):
        new_layer = cls()
        new_layer._fallback_torch_layer = rotary_emb
        config = rotary_emb.config
        new_layer._head_dim = config.head_dim
        new_layer._rope_theta = config.rope_theta
        new_layer._partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        new_layer._max_seq_len = config.max_position_embeddings
        new_layer._attention_scaling = rotary_emb.attention_scaling
        new_layer._inv_freq_torch = rotary_emb.inv_freq.float()  # to be populated in preprocess_weights
        new_layer.rotary_compute = TTNNRotaryEmbeddingCompute.from_torch(rotary_emb)
        return new_layer

    @staticmethod
    def _pad_dim(tensor, dim, pad_amount, value=0.0):
        """Pad a single dimension of a tensor by ``pad_amount``."""
        rank = len(tensor.shape)
        padding = tuple((0, pad_amount if i == dim else 0) for i in range(rank))
        return ttnn.pad(tensor, padding=padding, value=value)

    @staticmethod
    def _slice_dim(tensor, dim, length):
        """Slice a tensor along ``dim`` to ``length``."""
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        ends[dim] = length
        return ttnn.slice(tensor, starts, ends)

    def set_output_tensors_config_impl(self, output_tensors):
        """Override to use replicated config — cos/sin are replicated across all devices."""
        from models.experimental.tt_symbiote.core.module import set_distributed_tensor_config
        from models.experimental.tt_symbiote.core.utils import tree_map

        replicated_config = DistributedTensorConfig(
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
        )
        return tree_map(set_distributed_tensor_config(replicated_config), output_tensors)

    def move_weights_to_device_impl(self):
        inv_freq_expanded = self._inv_freq_torch[None, :, None]
        self.tt_inv_freq = ttnn.from_torch(
            inv_freq_expanded,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states: ttnn.Tensor, position_ids: ttnn.Tensor):
        # Replicate HF BailingMoeV2RotaryEmbedding.forward in TTNN:
        #   freqs = inv_freq_expanded @ position_ids_expanded  -> [batch, seq, rotary_dim/2]
        #   emb = cat(freqs, freqs)  -> [batch, seq, rotary_dim]
        #   cos = emb.cos() * attention_scaling
        #   sin = emb.sin() * attention_scaling
        # Now position_ids is torch [batch, seq]
        batch = position_ids.shape[0]
        seq_len = position_ids.shape[-1]
        padded_seq_len = max(_next_power_of_2(seq_len), 32)
        pad_amount = padded_seq_len - seq_len
        if pad_amount > 0:
            position_ids = self._pad_dim(position_ids, 1, pad_amount, value=0)

        cos, sin = self.rotary_compute.forward(self.tt_inv_freq, position_ids)

        # Slice back to original seq_len: [batch, 1, seq_len]
        if pad_amount > 0:
            cos = self._slice_dim(cos, 1, seq_len)
            sin = self._slice_dim(sin, 1, seq_len)
        return cos, sin
