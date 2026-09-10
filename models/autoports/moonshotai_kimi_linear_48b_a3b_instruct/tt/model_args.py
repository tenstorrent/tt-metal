# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The attribute bag models/tt_transformers/tt/generator.py expects as ``model_args`` (no tt_transformers ModelArgs here)."""

from __future__ import annotations

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig


class KimiModelArgs:
    def __init__(
        self, mesh_device, cfg: KimiLinearConfig, *, max_batch_size: int, max_seq_len: int, prefill_chunk: int = 2048
    ):
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = cfg.vocab_size
        self.padded_vocab_size = -(-cfg.vocab_size // 32) * 32
        self.dim = cfg.hidden_size
        self.n_layers = cfg.num_hidden_layers
        self.cluster_shape = list(mesh_device.shape)
        self.num_devices = mesh_device.get_num_devices()
        self.max_prefill_chunk_size = prefill_chunk
        self.trace_prefill_supported_seq_lens: list[int] = []  # prefill is eager (model-owned)
        self.model_config: dict = {}
        self.max_top_k = 32
        self.model_name = "Kimi-Linear-48B-A3B-Instruct"

    def is_llama_vision(self) -> bool:
        return False

    def can_enable_trace(self, seq_len: int, batch: int) -> bool:
        return False  # no traced prefill; decode tracing is handled by the Generator base class

    def get_warmup_prefill_supported_seq_lens(self):
        return []
