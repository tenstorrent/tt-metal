# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Token embedding module for TTNN-accelerated LLMs using ttnn.embedding."""

from torch import nn

import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


class TTNNEmbedding(TTNNModule):
    """TTNN-accelerated token embedding using ttnn.embedding.

    Replaces nn.Embedding for LLM token lookup. Uses ttnn.embedding which
    supports pad_token for models with padding_idx (e.g. GLM-4).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    @classmethod
    def from_torch(cls, torch_embedding: nn.Embedding) -> "TTNNEmbedding":
        """Create TTNNEmbedding from PyTorch nn.Embedding layer."""
        new_embedding = cls(
            num_embeddings=torch_embedding.num_embeddings,
            embedding_dim=torch_embedding.embedding_dim,
            padding_idx=torch_embedding.padding_idx if torch_embedding.padding_idx is not None else None,
        )
        new_embedding._fallback_torch_layer = torch_embedding
        new_embedding.weight = torch_embedding.weight
        return new_embedding

    def preprocess_weights_impl(self):
        """Preprocess embedding weights for TTNN."""
        # nn.Embedding weight: [num_embeddings, embedding_dim]
        # ttnn.embedding expects weight in ROW_MAJOR (cpp converts TILE to ROW_MAJOR)
        self._weight_host = ttnn.from_torch(
            self.weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        """Move embedding weights to TTNN device.

        Note: ttnn.to_device does not accept mesh_mapper. For mesh devices,
        the host tensor is replicated to all devices automatically.
        """
        self._weight_tt = ttnn.to_device(
            self._weight_host,
            self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self._weight_tt)
        super().deallocate_weights_impl()

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_ids):
        """Forward pass: token indices -> embeddings.

        Args:
            input_ids: Token indices [batch_size, seq_len]. Can be TorchTTNNTensor
                      or torch.Tensor (will be wrapped and converted to ttnn).

        Returns:
            Embeddings [batch_size, seq_len, embedding_dim].
        """
        if isinstance(input_ids, TorchTTNNTensor):
            input_ids = input_ids.to_ttnn
        elif not isinstance(input_ids, ttnn.Tensor):
            input_ids = ttnn.from_torch(
                input_ids,
                device=self.device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        batch_size, seq_len = input_ids.shape[0], input_ids.shape[-1]
        # ttnn.embedding expects 4D input: (batch, 1, 1, seq_len)
        if len(input_ids.shape) == 2:
            input_ids = ttnn.reshape(input_ids, (batch_size, 1, 1, seq_len))

        output = ttnn.embedding(
            input_ids,
            self._weight_tt,
            pad_token=self.padding_idx,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Output may be (batch, seq, embed) or need reshape
        if len(output.shape) == 3:
            return output
        return ttnn.reshape(output, (batch_size, seq_len, self.embedding_dim))
