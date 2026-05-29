# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 language-model token embedding.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`embedding_forward`

    output = weight[input_ids]

A plain embedding-table gather: ``nn.Embedding(vocab=151936, hidden=1536)``.
In TTNN this is :func:`ttnn.embedding`. The lookup table is ~233M params
(~466MB in bf16), so it is stored in DRAM (never L1). The op gathers exact
rows, so PCC is essentially 1.0 (bf16 row casting is the only error source).

Reference TTNN impl this follows: models/tt_transformers/tt/embedding.py
"""
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtEmbedding(LightweightModule):
    """dots.ocr LM token embedding (gather over a DRAM-resident table).

    Args:
        device: ttnn Device or MeshDevice.
        weight: torch.Tensor of shape [vocab_size, hidden_size] (the
            ``embed_tokens.weight`` parameter).
        weight_dtype: dtype the embedding table is stored in (bf16 default).
        weight_memory_config: where the table lives (DRAM — it is large).
    """

    def __init__(
        self,
        device,
        weight,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        # ttnn.embedding wants the table in row-major layout.
        self.weight = ttnn.as_tensor(
            weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Gather embeddings.

        Args:
            input_ids: row-major uint32 ttnn tensor of shape [batch, seq_len].
        Returns:
            tile-layout ttnn tensor of shape [batch, seq_len, hidden_size].
        """
        return ttnn.embedding(
            input_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
        )
