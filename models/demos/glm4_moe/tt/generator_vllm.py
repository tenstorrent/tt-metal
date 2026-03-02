# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM entrypoint for GLM-4.7-REAP-218B on TT hardware.

Implements the vLLM TT model interface expected by TTModelRunner:
- initialize_vllm_model() classmethod for model loading
- allocate_kv_cache() for standard GQA KV cache (NOT MLA KVPE)
- prefill_forward() / decode_forward() for inference
- read_decode_output() / process_decode_output_host() for output handling

KV cache format: standard GQA with separate K and V tensors
  (num_blocks, n_local_kv_heads=1, block_size=64, head_dim=128), dtype BF8

No MTP support (not applicable to REAP).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from loguru import logger

import ttnn

from models.demos.glm4_moe.tt.model_tt import Glm4MoeTT, _torch_dtype_to_ttnn
from models.demos.glm4_moe.tt.weights import resolve_best_effort_snapshot_dir, find_missing_shards


class Glm4MoeForCausalLM(nn.Module):
    """TT model entrypoint for vLLM (V1) — GLM-4.7-REAP-218B.

    Standard GQA attention (96Q/8KV heads, head_dim=128).
    KV cache: separate K and V, dtype BF8, paged.
    """

    model_capabilities = {"supports_prefix_caching": False}

    def __init__(
        self,
        *,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: int,
        model_id: str,
        snapshot_dir: Path,
        cache_dir: Path,
    ) -> None:
        super().__init__()
        self.hf_config = hf_config

        from models.demos.glm4_moe.tt.config import Glm4MoeHParams
        self.hparams = Glm4MoeHParams.from_hf_config(hf_config)
        self.hparams.validate()

        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.model_id = model_id
        self.snapshot_dir = snapshot_dir
        self.cache_dir = cache_dir

        self._kv_cache: Optional[list] = None
        self._kv_cache_shape: Optional[tuple] = None
        self._tt_runner: Optional[Glm4MoeTT] = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        """Factory method called by vLLM TTModelLoader."""
        model_id = os.environ.get("HF_MODEL") or os.environ.get("GLM4_MOE_HF_MODEL")
        if not model_id:
            raise ValueError(
                "Missing model id. Set HF_MODEL or GLM4_MOE_HF_MODEL to a "
                "HuggingFace repo id (e.g. 'cerebras/GLM-4.7-REAP-218B-A32B')."
            )

        snapshot_dir = resolve_best_effort_snapshot_dir(model_id)

        default_cache_root = Path(os.path.expanduser("~/.cache/ttnn/models"))
        cache_dir_env = os.environ.get("GLM4_MOE_CACHE_DIR", "").strip()
        cache_dir = Path(cache_dir_env) if cache_dir_env else (default_cache_root / "glm4_moe" / "vllm")
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing GLM-4.7-REAP TT model: model_id={} snapshot_dir={} cache_dir={} "
            "max_batch={} max_seq_len={} dp={} opt={}",
            model_id, str(snapshot_dir), str(cache_dir),
            max_batch_size, max_seq_len, tt_data_parallel, optimizations,
        )

        try:
            missing = find_missing_shards(snapshot_dir)
            if missing:
                logger.warning(
                    "GLM snapshot is missing {} safetensors shards (example: {}). "
                    "Inference will fail until weights are fully downloaded.",
                    len(missing), missing[0],
                )
        except Exception as e:
            logger.warning("GLM snapshot completeness check failed: {}", e)

        return cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            model_id=model_id,
            snapshot_dir=snapshot_dir,
            cache_dir=cache_dir,
        )

    @property
    def cache_path(self) -> Path:
        return self.cache_dir

    # -------------------------------------------------------------------
    # KV Cache
    # -------------------------------------------------------------------

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate standard GQA KV cache (separate K and V per layer).

        KV cache shape per tensor: (num_blocks, n_local_kv_heads=1, block_size, head_dim=128)
        Dtype: BF8 by default (sufficient for 100K context at 218B).
        """
        if hasattr(self.hf_config, "num_hidden_layers"):
            assert num_layers == self.hf_config.num_hidden_layers, (
                f"allocate_kv_cache: num_layers={num_layers} does not match "
                f"hf_config.num_hidden_layers={self.hf_config.num_hidden_layers}"
            )
        self._kv_cache_shape = tuple(int(x) for x in kv_cache_shape)
        self._kv_cache_dtype = dtype

        num_layers = int(num_layers)
        num_layers_env = os.environ.get("GLM4_MOE_NUM_LAYERS", "").strip()
        if num_layers_env and os.environ.get("GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS", "").strip() != "1":
            raise ValueError(
                "GLM4_MOE_NUM_LAYERS is debug-only. "
                "Set GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS=1 to run a partial model."
            )
        num_layers_to_alloc = int(num_layers_env) if num_layers_env else num_layers
        num_layers_to_alloc = max(1, min(num_layers_to_alloc, num_layers))
        if num_layers_to_alloc != num_layers:
            logger.warning(
                "GLM4_MOE_NUM_LAYERS: allocating KV cache for {} layers (vLLM requested {})",
                num_layers_to_alloc, num_layers,
            )

        # Override KV cache dtype from env (default: use vLLM's dtype)
        kv_dtype_env = os.environ.get("GLM4_MOE_KV_CACHE_TT_DTYPE", "").strip().lower()
        if kv_dtype_env == "bf8":
            tt_dtype = ttnn.bfloat8_b
            logger.info("GLM KV cache dtype overridden to BF8 from env")
        elif kv_dtype_env == "bf4":
            tt_dtype = ttnn.bfloat4_b
            logger.info("GLM KV cache dtype overridden to BF4 from env")
        else:
            tt_dtype = _torch_dtype_to_ttnn(dtype)

        num_blocks = int(self._kv_cache_shape[0])
        block_size = int(self._kv_cache_shape[2])
        head_dim = int(self.hparams.head_dim)  # 128
        n_local_kv_heads = int(self.hparams.num_key_value_heads) // max(
            1, self._get_tp_size()
        )  # 8 / 8 = 1

        # Standard GQA: separate K and V caches.
        cache_k = torch.zeros((num_blocks, n_local_kv_heads, block_size, head_dim), dtype=torch.bfloat16, device="cpu")
        cache_v = torch.zeros((num_blocks, n_local_kv_heads, block_size, head_dim), dtype=torch.bfloat16, device="cpu")

        is_mesh = self.mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None

        kv_cache = []
        for layer_idx in range(num_layers_to_alloc):
            if layer_idx == 0 or (layer_idx + 1) % 10 == 0 or (layer_idx + 1) == num_layers_to_alloc:
                logger.info("Allocating GLM KV cache layer {}/{}", layer_idx + 1, num_layers_to_alloc)

            tt_k = ttnn.as_tensor(
                cache_k,
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=tt_dtype,
                cache_file_name=None,
            )
            tt_v = ttnn.as_tensor(
                cache_v,
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=tt_dtype,
                cache_file_name=None,
            )
            kv_cache.append([tt_k, tt_v])

        self._kv_cache = kv_cache
        return kv_cache

    def _get_tp_size(self) -> int:
        """Return TP size from mesh device.

        Galaxy TG Mesh(8,4): TP=axis 0 (rows=8), DP=axis 1 (cols=4).
        T3K Mesh(1,8): TP=axis 1 (cols=8).
        """
        if self.mesh_device.__class__.__name__ != "MeshDevice":
            return 1
        mesh_rows = int(self.mesh_device.shape[0])
        mesh_cols = int(self.mesh_device.shape[1])
        if mesh_rows > 1 and mesh_cols > 1:
            # 2D mesh (TG): TP is axis 0 (rows)
            return mesh_rows
        return mesh_cols if mesh_cols > 1 else mesh_rows

    # -------------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------------

    def warmup_model_prefill(self, *, kv_cache, enable_trace, sampling_params):
        """vLLM TT backend warmup hook."""
        self._ensure_tt_runner()

        num_blocks = int(self._kv_cache_shape[0]) if self._kv_cache_shape else 1
        page_table = torch.zeros((1, max(1, num_blocks)), dtype=torch.int32)
        page_table[0, 0] = 0

        _ = self.decode_forward(
            tokens=torch.zeros((1, 1), dtype=torch.int32),
            page_table=page_table,
            kv_cache=kv_cache,
            start_pos=torch.zeros((1,), dtype=torch.int32),
            enable_trace=False,
            read_from_device=True,
        )

    # -------------------------------------------------------------------
    # Prefill Forward
    # -------------------------------------------------------------------

    def prefill_forward(self, *args, **kwargs):
        """Prefill: process prompt tokens, fill KV cache, return logits for last token."""
        tokens: torch.Tensor = kwargs["tokens"]
        prompt_lens = kwargs["prompt_lens"]
        page_table: torch.Tensor = kwargs["page_table"]
        kv_cache = kwargs["kv_cache"]
        start_pos = kwargs.get("start_pos", None)

        batch = int(tokens.shape[0])
        vocab = int(self.hparams.vocab_size)

        if all(int(x) == 0 for x in prompt_lens):
            return torch.zeros((batch, 1, vocab), dtype=torch.float32)

        if start_pos is not None:
            start_pos_t = torch.as_tensor(start_pos, dtype=torch.int32) if not isinstance(start_pos, torch.Tensor) else start_pos.to(torch.int32)
            if (start_pos_t != 0).any():
                raise ValueError(f"Prefix caching not supported; got non-zero start_pos: {start_pos_t.tolist()}")

        self._ensure_tt_runner()

        return self._tt_runner.prefill(
            tokens=tokens,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
        )

    # -------------------------------------------------------------------
    # Decode Forward
    # -------------------------------------------------------------------

    def decode_forward(self, *args, **kwargs):
        """Decode: single-token step, update KV cache, return logits or token ids."""
        tokens: torch.Tensor = kwargs["tokens"]
        page_table: torch.Tensor = kwargs["page_table"]
        kv_cache = kwargs["kv_cache"]
        start_pos: torch.Tensor = kwargs["start_pos"]
        sampling_params = kwargs.get("sampling_params", None)
        enable_trace: bool = bool(kwargs.get("enable_trace", False))
        read_from_device: bool = bool(kwargs.get("read_from_device", False))

        # Patch all-zero page table for warmup.
        if page_table.numel() > 0 and int(page_table.sum().item()) == 0:
            page_table = page_table.clone()
            page_table[:, 0] = torch.arange(page_table.shape[0], dtype=torch.int32)

        self._ensure_tt_runner()
        tt_out = self._tt_runner.decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            enable_trace=enable_trace,
        )

        if read_from_device:
            tt_host = self.read_decode_output(tt_out, async_read=False)
            return self.process_decode_output_host(tt_host, is_tokens=(sampling_params is not None))
        return tt_out

    # -------------------------------------------------------------------
    # Output Handling
    # -------------------------------------------------------------------

    def read_decode_output(self, tt_out, async_read: bool):
        """Read TT decode output to host."""
        if isinstance(tt_out, torch.Tensor):
            return (tt_out, []) if async_read else tt_out

        if not async_read:
            if isinstance(tt_out, tuple):
                max_tt, argmax_tt = tt_out
                return (max_tt.cpu(), argmax_tt.cpu())
            return tt_out.cpu()

        if isinstance(tt_out, tuple):
            max_tt, argmax_tt = tt_out
            max_cpu = max_tt.cpu(blocking=False, cq_id=0)
            argmax_cpu = argmax_tt.cpu(blocking=False, cq_id=0)
            return (max_cpu, argmax_cpu), [ttnn.record_event(self.mesh_device, 0)]

        tt_cpu = tt_out.cpu(blocking=False, cq_id=0)
        return tt_cpu, [ttnn.record_event(self.mesh_device, 0)]

    def process_decode_output_host(self, tt_out, is_tokens: bool):
        """Post-process decode output on host."""
        if not is_tokens:
            return tt_out

        def _tt_to_torch_device0(tensor: ttnn.Tensor) -> torch.Tensor:
            if self.mesh_device.__class__.__name__ == "MeshDevice":
                try:
                    device_tensors = ttnn.get_device_tensors(tensor)
                except Exception:
                    device_tensors = []
                if device_tensors:
                    return ttnn.to_torch(device_tensors[0])
            return ttnn.to_torch(tensor)

        if isinstance(tt_out, torch.Tensor):
            return tt_out.to(dtype=torch.int32).cpu().reshape(-1)

        # Vocab-sharded LM head: (local_max, local_argmax) per shard.
        if isinstance(tt_out, tuple):
            if self._tt_runner is None:
                raise RuntimeError("TT runner is not initialized")
            local_max_tt, local_argmax_tt = tt_out

            max_dts = ttnn.get_device_tensors(local_max_tt)
            idx_dts = ttnn.get_device_tensors(local_argmax_tt)
            if not max_dts or not idx_dts:
                raise RuntimeError("expected mesh tensors for vocab-sharded decode output")

            num_devices = len(max_dts)
            vocab = int(self.hparams.vocab_size)
            vocab_per_shard = int(self._tt_runner.lm_head_vocab_per_shard)
            batch = int(ttnn.to_torch(max_dts[0]).reshape(-1).numel())
            next_ids = torch.empty((batch,), dtype=torch.int32)

            for b in range(batch):
                best_val = None
                best_global = None
                for shard_idx in range(num_devices):
                    max_val = float(ttnn.to_torch(max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = shard_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global
            return next_ids

        next_ids_torch = _tt_to_torch_device0(tt_out).reshape(-1).to(dtype=torch.int32).cpu()
        return next_ids_torch

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _ensure_tt_runner(self) -> None:
        """Lazy-initialize the TT model runner."""
        if self._tt_runner is not None:
            return
        self._tt_runner = Glm4MoeTT.create(
            device=self.mesh_device,
            snapshot_dir=self.snapshot_dir,
            cache_dir=self.cache_dir,
            max_seq_len=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            hparams=self.hparams,
        )
