# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Embedding layer implementations for TTNN."""

import torch
from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import (
    DistributedTensorConfig,
    logical_shape_for_batch_channel_sharding,
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
    def __call__(self, input_ids):
        rank = len(input_ids.shape)
        seq_dim = rank - 1  # sequence length is always second-to-last
        seq_len = input_ids.shape[seq_dim]
        padded_seq_len = _next_power_of_2(seq_len)
        pad_amount = padded_seq_len - seq_len
        tt_indices = ttnn.from_torch(
            input_ids.cpu().to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        if pad_amount > 0:
            tt_indices = self._pad_dim(tt_indices, seq_dim, pad_amount, value=0)

        tt_out = self.embedder(tt_indices)
        if pad_amount > 0:
            tt_out = self._slice_dim(tt_out, seq_dim, seq_len)

        result = TorchTTNNTensor(tt_out)
        # Output is sharded along hidden dim (last dim), matching weight sharding
        result.set_distributed_tensor_config(
            DistributedTensorConfig(
                mesh_mapper=ttnn.ShardTensor2dMesh(self.device, self.device.shape, (0, -1)),
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.device, self.device.shape, (0, -1)),
                logical_shape_fn=logical_shape_for_batch_channel_sharding(self.device.shape),
            )
        )
        return result
