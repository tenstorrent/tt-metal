# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-shaped adapter and greedy helpers for Llama-3.3-70B-Instruct TTTv2.

Uses ``AutoConfig`` / ``Llama33_70BTransformer1D.from_pretrained`` and
:class:`Llama33_70BGeneratorConfig` only — no v1 ``ModelArgs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ttnn
from models.common.models.executor import make_contiguous_page_table
from models.common.models.llama33_70b.model import (
    LLAMA33_70B_ACCURACY,
    EagerLlama33_70BExecutor,
    Llama33_70BPrecisionConfig,
    Llama33_70BTransformer1D,
    TracedLlama33_70BExecutor,
)


@dataclass
class Llama33_70BGeneratorConfig:
    """Port-local serving / executor knobs (HF-free beyond ``hf_model_id``)."""

    hf_model_id: str = "meta-llama/Llama-3.3-70B-Instruct"
    hf_revision: str | None = None
    max_batch_size: int = 32
    max_seq_len: int = 4096
    num_layers: int | None = None
    cache_dir: Path | str | None = None
    precision: Llama33_70BPrecisionConfig = LLAMA33_70B_ACCURACY
    traced: bool = False


class Llama33_70BGenerator:
    """Thin vLLM adapter: ``allocate_kv_cache``, ``prefill_forward``, ``decode_forward``."""

    model_capabilities = {"supports_prefix_caching": True}

    def __init__(
        self,
        executor: EagerLlama33_70BExecutor | TracedLlama33_70BExecutor,
        *,
        gen_cfg: Llama33_70BGeneratorConfig,
    ):
        self.executor = executor
        self.model = executor.model
        self.gen_cfg = gen_cfg
        self.mesh_device = executor.mesh_device

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        gen_cfg: Llama33_70BGeneratorConfig | None = None,
    ) -> Llama33_70BGenerator:
        gen_cfg = gen_cfg or Llama33_70BGeneratorConfig()
        model = Llama33_70BTransformer1D.from_pretrained(
            mesh_device,
            gen_cfg.hf_model_id,
            max_batch_size=gen_cfg.max_batch_size,
            max_seq_len=gen_cfg.max_seq_len,
            num_layers=gen_cfg.num_layers,
            cache_dir=gen_cfg.cache_dir,
            precision=gen_cfg.precision,
            executor_mode=True,
        )
        ex: EagerLlama33_70BExecutor | TracedLlama33_70BExecutor
        if gen_cfg.traced:
            ex = TracedLlama33_70BExecutor(model, mesh_device)
        else:
            ex = EagerLlama33_70BExecutor(model, mesh_device)
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
