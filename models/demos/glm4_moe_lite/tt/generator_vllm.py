# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from loguru import logger

import ttnn

from models.demos.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT, _torch_dtype_to_ttnn
from models.demos.glm4_moe_lite.tt.weights import resolve_best_effort_snapshot_dir


class Glm4MoeLiteForCausalLM(nn.Module):
    """TT model entrypoint for vLLM (V1) - skeleton.

    This file is intentionally a minimal, integration-first scaffold:
    - It makes vLLM TT backend able to import and instantiate the model.
    - It provides method signatures expected by TTModelRunner.
    - It returns placeholder logits so the HTTP stack can exercise the full
      request path during early bring-up.

    Numerical correctness and TT execution are implemented in later phases.
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
        # Parsed, validated hyperparameters (stable interface for tt-metal code).
        from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams

        self.hparams = Glm4MoeLiteHParams.from_hf_config(hf_config)
        self.hparams.validate()

        self.mesh_device = mesh_device
        self.max_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.model_id = model_id
        self.snapshot_dir = snapshot_dir
        self.cache_dir = cache_dir

        self._kv_cache: Optional[Any] = None
        self._tt_runner: Optional[Glm4MoeLiteDenseOnlyTT] = None

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
        # TTModelLoader does not pass the model ID/path; docker_tt sets HF_MODEL.
        model_id = os.environ.get("HF_MODEL") or os.environ.get(
            "GLM4_MOE_LITE_HF_MODEL")
        if not model_id:
            raise ValueError(
                "Missing model id. Set HF_MODEL or GLM4_MOE_LITE_HF_MODEL to a "
                "HuggingFace repo id (e.g. 'zai-org/GLM-4.7-Flash').")

        # Single source of truth for snapshot resolution: offline scan of the local HF cache.
        snapshot_dir = resolve_best_effort_snapshot_dir(model_id)

        # Converted TT weights / compiled artifacts cache.
        default_cache_root = Path(os.path.expanduser("~/.cache/ttnn/models"))
        cache_dir_env = os.environ.get("GLM4_MOE_LITE_CACHE_DIR", "").strip()
        cache_dir = Path(cache_dir_env) if cache_dir_env else (default_cache_root / "glm4_moe_lite" / "vllm")
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing GLM TT model (skeleton): model_id={} snapshot_dir={} cache_dir={} max_batch={} max_seq_len={} dp={} opt={}",
            model_id,
            str(snapshot_dir),
            str(cache_dir),
            max_batch_size,
            max_seq_len,
            tt_data_parallel,
            optimizations,
        )

        # Early diagnostic: GLM snapshots can exist locally with only a subset
        # of the sharded safetensors downloaded. We don't need weights for the
        # Phase-3 placeholder forward, but we *will* need them for real compute.
        try:
            from models.demos.glm4_moe_lite.tt.weights import find_missing_shards

            missing = find_missing_shards(snapshot_dir)
            if missing:
                logger.warning(
                    "GLM snapshot is missing {} safetensors shards (example: {}). Real inference will fail until weights are fully downloaded.",
                    len(missing),
                    missing[0],
                )
        except Exception as e:
            logger.warning("GLM snapshot completeness check failed (ignored in skeleton): {}", e)

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

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        if hasattr(self.hf_config, "num_hidden_layers"):
            assert num_layers == self.hf_config.num_hidden_layers, (
                f"allocate_kv_cache: num_layers={num_layers} does not match "
                f"hf_config.num_hidden_layers={self.hf_config.num_hidden_layers}"
            )
        self._kv_cache_shape = tuple(int(x) for x in kv_cache_shape)
        self._kv_cache_dtype = dtype

        # Allocate per-layer KVPE cache tensors.
        #
        # NOTE: Avoid `ttnn.zeros(..., device=MeshDevice)` here. We've seen it
        # hang during server startup (KV cache allocation). Instead, stage a
        # CPU zero tensor and upload it via `ttnn.as_tensor`, which is the
        # pattern used by other TT vLLM model entrypoints.
        num_layers = int(num_layers)
        num_layers_env = os.environ.get("GLM4_MOE_LITE_NUM_LAYERS", "").strip()
        if num_layers_env and os.environ.get("GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS", "").strip() != "1":
            raise ValueError(
                "GLM4_MOE_LITE_NUM_LAYERS is debug-only. "
                "Set GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1 to run a partial model."
            )
        num_layers_to_alloc = int(num_layers_env) if num_layers_env else num_layers
        num_layers_to_alloc = max(1, min(num_layers_to_alloc, num_layers))
        if num_layers_to_alloc != num_layers:
            logger.warning(
                "GLM4_MOE_LITE_NUM_LAYERS is set; allocating KV cache for {} layers (vLLM requested {}). "
                "This is bring-up only; full model runs require allocating all layers.",
                num_layers_to_alloc,
                num_layers,
            )

        tt_dtype = _torch_dtype_to_ttnn(dtype)
        cache_kv = torch.zeros(self._kv_cache_shape, dtype=dtype, device="cpu")
        mesh_mapper = None
        if self.mesh_device.__class__.__name__ == "MeshDevice":
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        cache_file = self.cache_dir / f"empty_kvpe_cache_paged_attention{self._kv_cache_shape}_dtype_{tt_dtype}"

        kv_cache = []
        for layer_idx in range(num_layers_to_alloc):
            if layer_idx == 0 or (layer_idx + 1) % 8 == 0 or (layer_idx + 1) == num_layers_to_alloc:
                logger.info("Allocating GLM KV cache layer {}/{}", layer_idx + 1, num_layers_to_alloc)
            kv_cache.append(
                ttnn.as_tensor(
                    cache_kv,
                    device=self.mesh_device,
                    mesh_mapper=mesh_mapper,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=tt_dtype,
                    cache_file_name=cache_file,
                )
            )
        self._kv_cache = kv_cache
        return kv_cache

    def warmup_model_prefill(self, *, kv_cache, enable_trace, sampling_params):
        """vLLM TT backend warmup hook.

        The TT runner uses this to compile/trace common prefill paths early.
        For this integration scaffold, we simply exercise `prefill_forward` at
        least once (and for each sampling config, if provided) to satisfy the
        runner contract without doing any heavy work.
        """
        # Prefill in the dense-only bring-up uses decode iteratively; decode warmup
        # already captures the heavy path. Keep prefill warmup minimal and safe.
        _ = enable_trace
        _ = sampling_params

        # Ensure the runtime is initialized.
        self._ensure_tt_runner()

        num_blocks = int(self._kv_cache_shape[0]) if hasattr(self, "_kv_cache_shape") else 1
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

    def prefill_forward(self, *args, **kwargs):
        # Required kwargs used by TTModelRunner:
        # tokens: (B, S) int32
        # page_table: (B, max_num_blocks_per_req) int32
        # kv_cache: opaque object from allocate_kv_cache
        # start_pos: np.ndarray or torch.Tensor (B,) positions
        # prompt_lens: list[int] length B
        tokens: torch.Tensor = kwargs["tokens"]
        prompt_lens = kwargs["prompt_lens"]
        page_table: torch.Tensor = kwargs["page_table"]
        kv_cache = kwargs["kv_cache"]
        start_pos = kwargs.get("start_pos", None)

        batch = int(tokens.shape[0])
        vocab = int(self.hparams.vocab_size)

        if all(int(x) == 0 for x in prompt_lens):
            return torch.zeros((batch, 1, vocab), dtype=torch.float32)

        # Prefix caching is disabled for GLM bring-up.
        if start_pos is not None:
            if not isinstance(start_pos, torch.Tensor):
                start_pos_t = torch.as_tensor(start_pos, dtype=torch.int32)
            else:
                start_pos_t = start_pos.to(torch.int32)
            if (start_pos_t != 0).any():
                raise ValueError(f"Prefix caching not supported for GLM bring-up; got non-zero start_pos: {start_pos_t.tolist()}")

        self._ensure_tt_runner()

        impl = os.environ.get("GLM4_MOE_LITE_PREFILL_IMPL", "").strip().lower() or "flash_mla_prefill"
        if impl in {"mla", "flash_mla", "flash_mla_prefill", "prefill"}:
            return self._tt_runner.prefill(tokens=tokens, prompt_lens=prompt_lens, page_table=page_table, kv_cache=kv_cache)

        if impl not in {"decode_loop", "decode"}:
            raise ValueError(
                f"Invalid GLM4_MOE_LITE_PREFILL_IMPL={impl!r}; expected one of "
                "['flash_mla_prefill', 'decode_loop']"
            )

        # Fallback: iterative decode-per-token prefill (slow, correctness-only).
        last_logits = []
        for i in range(batch):
            prompt_len = int(prompt_lens[i])
            if prompt_len <= 0:
                last_logits.append(torch.zeros((1, vocab), dtype=torch.float32))
                continue

            user_page_table = page_table[i : i + 1, :]
            logits_i = None
            for t in range(prompt_len):
                tok = tokens[i, t].view(1, 1).to(torch.int32)
                pos = torch.tensor([t], dtype=torch.int32)
                logits_i = self._tt_runner.decode(
                    tokens=tok, start_pos=pos, page_table=user_page_table, kv_cache=kv_cache
                )  # [1,1,V]
            assert logits_i is not None
            last_logits.append(logits_i[0])  # [1,V]

        return torch.stack(last_logits, dim=0)  # [B,1,V]

    def decode_forward(self, *args, **kwargs):
        # Required kwargs used by TTModelRunner:
        # tokens: (B, 1) int32 (padded to max_num_seqs)
        # page_table: (B, max_num_blocks_per_req) int32
        # kv_cache: opaque object from allocate_kv_cache
        # start_pos: torch.Tensor (B,) (padded with -1 for inactive slots)
        tokens: torch.Tensor = kwargs["tokens"]
        page_table: torch.Tensor = kwargs["page_table"]
        kv_cache = kwargs["kv_cache"]
        start_pos: torch.Tensor = kwargs["start_pos"]
        sampling_params = kwargs.get("sampling_params", None)
        enable_trace: bool = bool(kwargs.get("enable_trace", False))

        # Warmup can pass a dummy all-zero page table; patch it to avoid
        # overlapping updates to the same physical block.
        if page_table.numel() > 0 and int(page_table.sum().item()) == 0:
            page_table = page_table.clone()
            page_table[:, 0] = torch.arange(page_table.shape[0], dtype=torch.int32)

        self._ensure_tt_runner()
        return self._tt_runner.decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            enable_trace=enable_trace,
        )

    def _ensure_tt_runner(self) -> None:
        if self._tt_runner is not None:
            return
        self._tt_runner = Glm4MoeLiteDenseOnlyTT.create(
            device=self.mesh_device,
            snapshot_dir=self.snapshot_dir,
            cache_dir=self.cache_dir,
            max_seq_len=self.max_seq_len,
            hparams=self.hparams,
        )

    # ---- vLLM TT v0 output hooks ----
    # vLLM's legacy TT runner expects TT models to expose a two-step read/process
    # pipeline for decode outputs. In production this is where we'd copy device
    # tensors back to host and post-process (e.g. layout, dtype).
    #
    # For this integration scaffold we return torch tensors already on host,
    # so these are essentially identity operations.

    def read_decode_output(self, tt_out, async_read: bool):
        if async_read:
            # We currently return torch tensors directly from `decode_forward`,
            # so there is no device->host read to synchronize. Return an empty
            # event list to satisfy TTModelRunner's async_read_decode contract.
            return tt_out, []
        return tt_out

    def process_decode_output_host(self, tt_out, is_tokens: bool):
        _ = is_tokens  # skeleton always returns logits
        return tt_out
