# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM-shaped adapter and greedy helpers for Qwen2.5-72B-Instruct TTTv2 (T3K).

Uses ``AutoConfig`` / ``Qwen25_72B.from_pretrained`` and
:class:`Qwen25_72BGeneratorConfig` only — no v1 ``ModelArgs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.models.executor import make_contiguous_page_table
from models.common.models.qwen25_72b.executor import EagerQwen25_72BExecutor, TracedQwen25_72BExecutor
from models.common.models.qwen25_72b.model import DEFAULT_HF_REVISION, Qwen25_72B


@dataclass
class Qwen25_72BGeneratorConfig:
    """Port-local serving / executor knobs (HF-free beyond ``hf_model_id``)."""

    hf_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    hf_revision: str | None = DEFAULT_HF_REVISION
    max_batch_size: int = 32
    max_seq_len: int = 4096
    num_layers: int | None = None
    cache_dir: Path | str | None = None
    traced: bool = False


class Qwen25_72BGenerator:
    """Thin vLLM adapter: ``allocate_kv_cache``, ``prefill_forward``, ``decode_forward``."""

    model_capabilities = {"supports_prefix_caching": True}

    def __init__(
        self,
        executor: EagerQwen25_72BExecutor | TracedQwen25_72BExecutor,
        *,
        gen_cfg: Qwen25_72BGeneratorConfig,
    ):
        self.executor = executor
        self.model = executor.model
        self.gen_cfg = gen_cfg
        self.mesh_device = executor.mesh_device

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        gen_cfg: Qwen25_72BGeneratorConfig | None = None,
    ) -> Qwen25_72BGenerator:
        gen_cfg = gen_cfg or Qwen25_72BGeneratorConfig()
        model = Qwen25_72B.from_pretrained(
            mesh_device,
            gen_cfg.hf_model_id,
            revision=gen_cfg.hf_revision,
            max_batch_size=gen_cfg.max_batch_size,
            max_seq_len=gen_cfg.max_seq_len,
            num_layers=gen_cfg.num_layers,
            cache_dir=gen_cfg.cache_dir,
            executor_mode=True,
        )
        ex: EagerQwen25_72BExecutor | TracedQwen25_72BExecutor
        if gen_cfg.traced:
            ex = TracedQwen25_72BExecutor(model, mesh_device)
        else:
            ex = EagerQwen25_72BExecutor(model, mesh_device)
        return cls(ex, gen_cfg=gen_cfg)

    def allocate_kv_cache(self, kv_cache_shape: tuple[int, ...], dtype: torch.dtype, num_layers: int):
        return self.executor.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def prefill_forward(self, *args, **kwargs):
        return self.executor.prefill_forward(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return self.executor.decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, *args, **kwargs):
        if hasattr(self.executor, "warmup_model_prefill"):
            return self.executor.warmup_model_prefill(*args, **kwargs)

    @staticmethod
    def default_page_table(batch_size: int, max_seq_len: int, block_size: int = 32) -> torch.Tensor:
        return make_contiguous_page_table(batch_size, max_seq_len, block_size)

    @property
    def cache_path(self) -> Path | None:
        if self.model.model_args:
            return self.model.model_args.model_cache_path
        return None


def greedy_argmax_from_logits(logits: ttnn.Tensor, *, mesh_device: ttnn.MeshDevice) -> int:
    """Return global argmax token id from (possibly sharded) logits ``[1,1,B,V_local]``."""
    lt = to_torch_auto_compose(logits, device=mesh_device).float()
    if lt.dim() == 4:
        lt = lt[0, 0, 0]
    elif lt.dim() == 3:
        lt = lt[0, 0]
    return int(torch.argmax(lt).item())


def greedy_decode_one_step(model: Qwen25_72B, token_id: int, *, current_pos: int) -> int:
    """Decode one token at ``current_pos``; returns next token id (greedy)."""
    tid = torch.tensor([[[[token_id]]]], dtype=torch.int32)
    x = ttnn.from_torch(
        tid,
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(model.mesh_device),
    )
    h = model.decode_from_token_ids(x, current_pos=current_pos)
    logits = model.lm_logits(h)
    return greedy_argmax_from_logits(logits, mesh_device=model.mesh_device)
