# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Serving-facing wrapper for llama-nemotron-rerank-1b-v2, same shape as bge_m3's ``BgeM3ForEmbedding``.

A runner (tt-media-server's EmbeddingRunner pattern, or ``demo/serve.py``) builds it once
per chip and calls ``forward(input_ids, attention_mask)`` with tokenized
``question:{q} \\n \\n passage:{p}`` pairs; it returns one relevance logit per row.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID, RerankConfig
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.runner import BATCH_BUCKETS, TracedReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.weights import default_cache_dir, load_state_dict


class NemotronRerankForScoring:
    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: int = 512,
        model_name: str = MODEL_ID,
        vllm_config=None,
        warmup: bool = True,
        config: RerankConfig = None,
        **kwargs,
    ):
        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device
        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")
        self.device = device
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        buckets = tuple(b for b in BATCH_BUCKETS if b <= max_batch_size) or (1,)
        model = NemotronReranker(
            device, config or RerankConfig(), load_state_dict(model_name), default_cache_dir(model_name)
        )
        self.runner = TracedReranker(model, max_seq_len=max_seq_len, batch_buckets=buckets)
        if warmup:
            self.runner.warmup()

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config=None,
        mesh_device: ttnn.Device = None,
        max_batch_size: int = 32,
        max_seq_len: Optional[int] = 512,
        vllm_config=None,
        **kwargs,
    ) -> "NemotronRerankForScoring":
        return cls(
            device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            vllm_config=vllm_config,
            **kwargs,
        )

    def score_ids(self, ids_list: list[list[int]]) -> list[float]:
        return self.runner.score_ids(ids_list)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """input_ids [B, L] (right- or left-padded), attention_mask [B, L] -> logits [B] float32."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        ids_list = [row[m.bool()].tolist() for row, m in zip(input_ids, attention_mask)]
        return torch.tensor(self.score_ids(ids_list), dtype=torch.float32)

    def release(self):
        self.runner.close()
