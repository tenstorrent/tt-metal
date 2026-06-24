# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optimized, minimal vLLM embedding wrapper for BGE-M3 (batch 1, ISL 512).

This is a stripped-down alternative to ``BgeM3ForEmbedding`` in
``generator_vllm.py``, specialized for the common serving case:

  * batch size 1, sequence length <= 512
  * dense (CLS) embeddings only

It leans on three model-level abstractions so the wrapper stays tiny:

  1. **On-device pooling** -- ``create_tt_model(..., pooling="cls")`` runs CLS
     pooling inside ``model.forward``, so the pooled vector is produced on
     device and captured in the trace (no host-side pooling math).
  2. **Trace capture** -- ``model.forward(**inputs, mode="trace")`` warms up and
     captures the trace on the first call, then replays it on every later call.
     This matches the vLLM plugin's lazy-warmup design (the worker's
     ``compile_or_warm_up_model`` is a no-op; capture happens on first execute).
  3. **Optimized D2H** -- ``copy_device_to_torch`` DMAs the pooled result
     straight into a pre-allocated torch tensor (no intermediate host tensor).

vLLM serving contract (see ``tt_model_runner_pooling.py``):
  * vLLM tokenizes text upstream; ``forward`` receives ``input_ids`` (token IDs),
    not text. ``token_type_ids`` / ``position_ids`` are derived by the model.
  * ``forward`` returns a ``[batch_size, embedding_dim]`` torch tensor.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import transformers

import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

MODEL_NAME = "BAAI/bge-m3"
MAX_BATCH_SIZE = 1
MAX_SEQ_LEN = 512
HIDDEN = 1024  # BGE-M3 embedding dim


class BgeM3ForEmbeddingOptimized:
    """Minimal BGE-M3 dense (CLS) embedding model for vLLM, fixed to B1/S512."""

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = MAX_BATCH_SIZE,
        max_seq_len: int = MAX_SEQ_LEN,
        dtype=ttnn.bfloat8_b,
        model_name: str = MODEL_NAME,
        vllm_config=None,
        **kwargs,
    ):
        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device
        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.model_name = model_name
        self.config = transformers.AutoConfig.from_pretrained(model_name)

        # Build the model once with CLS pooling baked in (on-device, in-trace).
        self.model_args, self.model, _ = create_tt_model(
            mesh_device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            hf_model_name=model_name,
            pooling="cls",
        )

        # Optimized D2H staging, allocated lazily on the first forward (once the
        # pooled output shape is known) and reused on every later call.
        self._dram_staging = None
        self._dest_torch = None

        # Optimized H2D state (perf.py pattern): a persistent device input slot
        # that the captured trace reads from, plus a reused host-side torch
        # buffer. The first forward captures the trace and adopts the slot; later
        # forwards do a single copy_host_to_device_tensor into that fixed address
        # (no per-call device allocation, no device->device copy).
        self._input_ids_dev = None
        self._trace_out = None
        self._padded_ids = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: transformers.PretrainedConfig,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = MAX_SEQ_LEN,
        vllm_config=None,
        dtype=ttnn.bfloat8_b,
        **kwargs,
    ) -> "BgeM3ForEmbeddingOptimized":
        """vLLM entry point (called by ``TTModelLoader``).

        This optimized model is fixed to batch 1 / ISL 512; ``max_batch_size``
        and ``max_seq_len`` from vLLM are clamped to those limits.
        """
        return cls(
            device=mesh_device,
            max_batch_size=MAX_BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            dtype=dtype,
            vllm_config=vllm_config,
        )

    def _allocate_d2h(self, pooled_dev: ttnn.Tensor) -> None:
        """One-time setup of the optimized device->host readback buffers."""
        b, _, s, _ = pooled_dev.shape
        sample_rm = ttnn.untilize_with_unpadding(
            pooled_dev, output_tensor_end=(b - 1, 0, s - 1, HIDDEN - 1), use_multicore=True
        )
        self._dram_staging = ttnn.clone(sample_rm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(sample_rm)
        self._dest_torch = torch.empty((b, 1, s, HIDDEN), dtype=torch.bfloat16)

    def _read_output(self, pooled_dev: ttnn.Tensor) -> torch.Tensor:
        """Optimized D2H: untilize -> DRAM staging -> copy_device_to_torch."""
        b, _, s, _ = pooled_dev.shape
        out_rm = ttnn.untilize_with_unpadding(
            pooled_dev, output_tensor_end=(b - 1, 0, s - 1, HIDDEN - 1), use_multicore=True
        )
        ttnn.copy(out_rm, self._dram_staging)
        ttnn.deallocate(out_rm)
        ttnn.copy_device_to_torch(self._dram_staging, self._dest_torch)
        return self._dest_torch

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return dense (CLS) embeddings of shape ``[batch_size, HIDDEN]``.

        ``input_ids`` are token IDs ``[B, S]`` (vLLM tokenizes upstream). Only
        ``input_ids`` and ``attention_mask`` are used; ``token_type_ids`` and
        ``position_ids`` are derived inside the model.
        """
        batch_size, seq_len = input_ids.shape
        if batch_size != MAX_BATCH_SIZE:
            raise ValueError(f"optimized BGE-M3 supports batch size {MAX_BATCH_SIZE} only, got {batch_size}")
        if seq_len > MAX_SEQ_LEN:
            raise ValueError(f"optimized BGE-M3 supports seq_len <= {MAX_SEQ_LEN}, got {seq_len}")

        # vLLM already tokenized upstream -- use the token IDs directly. Pad to
        # the fixed trace length (S512) with pad_token_id; the model derives the
        # attention mask, token_type_ids and position_ids from input_ids in-graph
        # (trace-safe). CLS pooling reads only token 0, so padding is harmless.
        import os
        import time

        _prof = os.environ.get("BGE_PROFILE") == "1"
        _t0 = time.perf_counter() if _prof else 0.0

        # Reuse a single pinned host buffer; refill in place each call.
        if self._padded_ids is None:
            pad_id = int(self.model_args.pad_token_id)
            self._padded_ids = torch.full((MAX_BATCH_SIZE, MAX_SEQ_LEN), pad_id, dtype=torch.int32)
        self._padded_ids.fill_(int(self.model_args.pad_token_id))
        self._padded_ids[:, :seq_len] = input_ids.to(torch.int32)

        # Wrap the fresh tokens as a host ttnn tensor (host-only, cheap). This is
        # the H2D source streamed into the persistent device slot each call.
        host_ids = ttnn.from_torch(self._padded_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        if self._input_ids_dev is None:
            # First call: build the persistent device input slot, JIT-compile,
            # and capture the trace that reads from this fixed device address.
            self._input_ids_dev = host_ids.to(self.device, ttnn.DRAM_MEMORY_CONFIG)

            # Warmup (eager) to compile kernels, then capture the trace bound to
            # the persistent input slot. capture_trace returns the trace output.
            warm = self.model.forward(input_ids=self._input_ids_dev, mode="eager")
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(warm)
            self._trace_out = self.model.capture_trace(input_ids=self._input_ids_dev, mesh_device=self.device)
            self._allocate_d2h(self._trace_out)
            _t1 = time.perf_counter() if _prof else 0.0
        else:
            # Optimized H2D: refresh the persistent device slot in place with a
            # single host->device copy (no new device allocation, no D2D copy).
            ttnn.copy_host_to_device_tensor(host_ids, self._input_ids_dev)
            ttnn.synchronize_device(self.device)
            _t1 = time.perf_counter() if _prof else 0.0

        # Replay the captured trace from the fixed input address.
        self.model.execute_trace(blocking=True)
        _t2 = time.perf_counter() if _prof else 0.0

        pooled = self._read_output(self._trace_out)  # [B, 1, 1, HIDDEN] bf16
        pooled = pooled.reshape(batch_size, HIDDEN).to(torch.float32)
        out = F.normalize(pooled, p=2, dim=-1)
        if _prof:
            _t3 = time.perf_counter()
            print(
                f"[BGE_PROFILE] h2d={1e3*(_t1-_t0):.3f}ms fwd={1e3*(_t2-_t1):.3f}ms "
                f"d2h={1e3*(_t3-_t2):.3f}ms total={1e3*(_t3-_t0):.3f}ms",
                flush=True,
            )
        return out

    def get_embedding_dim(self) -> int:
        return HIDDEN

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def _init_pooler(self, vllm_config, prefix: str = "") -> None:
        del vllm_config, prefix
        self.pooler = None


def register_model() -> None:
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        ModelRegistry.register_model(
            "BAAI/bge-m3",
            BgeM3ForEmbeddingOptimized,
            architecture="XLMRobertaModel",
        )
    except ImportError:
        return
