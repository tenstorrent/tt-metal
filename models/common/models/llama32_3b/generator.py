# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-shaped adapter and greedy helpers for Llama-3.2-3B-Instruct TTTv2.

Uses ``AutoConfig`` / ``Llama32_3BTransformer1D.from_pretrained`` and
:class:`Llama32_3BGeneratorConfig` only — no v1 ``ModelArgs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.common.models.executor import make_contiguous_page_table
from models.common.models.llama32_3b.model import (
    LLAMA32_3B_ACCURACY,
    EagerLlama32_3BExecutor,
    Llama32_3BPrecisionConfig,
    Llama32_3BTransformer1D,
    TracedLlama32_3BExecutor,
)


@dataclass
class Llama32_3BGeneratorConfig:
    """Port-local serving / executor knobs (HF-free beyond ``hf_model_id``)."""

    hf_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_revision: str | None = None
    max_batch_size: int = 32
    max_seq_len: int = 4096
    num_layers: int | None = None
    cache_dir: Path | str | None = None
    precision: Llama32_3BPrecisionConfig = LLAMA32_3B_ACCURACY
    traced: bool = False


class Llama32_3BGenerator:
    """Thin vLLM adapter: ``allocate_kv_cache``, ``prefill_forward``, ``decode_forward``."""

    model_capabilities = {"supports_prefix_caching": True}

    def __init__(
        self,
        executor: EagerLlama32_3BExecutor | TracedLlama32_3BExecutor,
        *,
        gen_cfg: Llama32_3BGeneratorConfig,
    ):
        self.executor = executor
        self.model = executor.model
        self.gen_cfg = gen_cfg
        self.mesh_device = executor.mesh_device

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        gen_cfg: Llama32_3BGeneratorConfig | None = None,
    ) -> Llama32_3BGenerator:
        gen_cfg = gen_cfg or Llama32_3BGeneratorConfig()
        model = Llama32_3BTransformer1D.from_pretrained(
            mesh_device,
            gen_cfg.hf_model_id,
            max_batch_size=gen_cfg.max_batch_size,
            max_seq_len=gen_cfg.max_seq_len,
            num_layers=gen_cfg.num_layers,
            cache_dir=gen_cfg.cache_dir,
            precision=gen_cfg.precision,
            executor_mode=True,
        )
        ex: EagerLlama32_3BExecutor | TracedLlama32_3BExecutor
        if gen_cfg.traced:
            ex = TracedLlama32_3BExecutor(model, mesh_device)
        else:
            ex = EagerLlama32_3BExecutor(model, mesh_device)
        return cls(ex, gen_cfg=gen_cfg)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self.executor.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def prefill_forward(self, *args, **kwargs):
        return self.executor.prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self.executor.decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, *args, **kwargs):
        if hasattr(self.executor, "warmup_model_prefill"):
            return self.executor.warmup_model_prefill(*args, **kwargs)

    @staticmethod
    def default_page_table(batch_size: int, max_seq_len: int, block_size: int = 32) -> "torch.Tensor":  # noqa: F821
        return make_contiguous_page_table(batch_size, max_seq_len, block_size)
