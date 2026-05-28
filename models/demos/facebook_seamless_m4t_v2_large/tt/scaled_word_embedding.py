# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the SeamlessM4T-v2 scaled word embedding.

PyTorch reference is
`models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::scaled_word_embedding_forward`
which performs the standard embedding lookup followed by a scalar multiply
by ``sqrt(hidden_size)`` (32.0 for the 1024-dim large model).

We pre-scale the embedding table at load time so the forward path is a single
``ttnn.embedding`` lookup — numerically identical to a post-multiply by ``scale``
but cheaper at inference time. This matches the lookup-then-multiply semantics
used by the HuggingFace ``SeamlessM4Tv2ScaledWordEmbedding`` module while
avoiding an extra elementwise op.

Loading pattern mirrors the embedding lookup in
``models/demos/audio/whisper/tt/ttnn_optimized_functional_whisper.py``
(``ttnn.embedding`` with TILE_LAYOUT output in DRAM).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class ScaledWordEmbedding(LightweightModule):
    """SeamlessM4T-v2 scaled word embedding.

    Args:
        device: ttnn device or mesh device.
        weight: torch.Tensor of shape ``(vocab_size, hidden_size)`` — the
            embedding table.
        scale: scalar multiplier (``sqrt(hidden_size)`` for SeamlessM4T-v2,
            i.e. 32.0 for the large 1024-dim model).
        padding_idx: optional padding index. The HF module follows
            ``nn.Embedding`` semantics and the padding row is already zeroed
            by ``reset_parameters`` at init time, so we don't need to mask
            anything here — keeping the arg for API parity with the
            reference signature.
        weight_dtype: storage dtype for the (pre-scaled) embedding table.
        weight_memory_config: where to place the embedding table.
    """

    def __init__(
        self,
        device,
        weight,
        scale: float,
        padding_idx=None,
        weight_dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device
        self.scale = float(scale)
        self.padding_idx = padding_idx
        self.vocab_size, self.hidden_size = int(weight.shape[0]), int(weight.shape[1])

        # Fold the scale into the embedding table once at load time. This is
        # numerically equivalent to a per-call elementwise multiply but saves
        # one op on the forward path.
        scaled_weight = weight.to(weight.dtype).mul(self.scale)

        # Embedding tables must be ROW_MAJOR for ttnn.embedding's gather op.
        self.weight = ttnn.from_torch(
            scaled_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
        )

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Gather pre-scaled embedding rows for ``input_ids``.

        Args:
            input_ids: ttnn uint32 ROW_MAJOR_LAYOUT tensor of shape
                ``(batch, seq_len)``.

        Returns:
            ttnn TILE_LAYOUT tensor of shape ``(batch, seq_len, hidden_size)``
            placed in DRAM.
        """
        return ttnn.embedding(
            input_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
