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

import models.demos.glm4_moe_lite.tt.debug_runtime  # noqa: F401

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
        self._last_draft_token_ids: Optional[torch.Tensor] = None

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
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
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
        
        # Ensure runner is created
        self._ensure_tt_runner()

        # Allocate per-layer KVPE cache tensors.
        #
        # Important: vLLM passes a "standard" KV cache shape in the form
        # (num_blocks, num_kv_heads, block_size, head_size). GLM-4.7-Flash uses
        # *Multi-Latent Attention* and the TT kernels expect a packed KVPE cache:
        # (num_blocks, 1, block_size, kvpe_dim) where kvpe_dim = kv_lora_rank + rope_dim.
        # Using the standard shape here will silently corrupt cache updates and
        # produce garbled outputs.
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

        num_blocks = int(self._kv_cache_shape[0])
        block_size = int(self._kv_cache_shape[2])
        kvpe_dim = int(self.hparams.kv_lora_rank + self.hparams.qk_rope_head_dim)
        cache_kv = torch.zeros((num_blocks, 1, block_size, kvpe_dim), dtype=dtype, device="cpu")
        mesh_mapper = None
        if self.mesh_device.__class__.__name__ == "MeshDevice":
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

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
                    # Do not use `cache_file_name` here. KV caches must be unique
                    # per layer; disk caching can lead to accidentally reusing
                    # the same backing buffer across layers.
                    cache_file_name=None,
                )
            )
        # Allocate 1 extra KV cache layer for MTP (layer 47)
        mtp_enabled = os.environ.get("GLM4_MOE_LITE_MTP", "").strip() == "1"
        if mtp_enabled:
            logger.info("Allocating MTP KV cache layer (layer {})", num_layers_to_alloc)
            kv_cache.append(
                ttnn.as_tensor(
                    cache_kv,
                    device=self.mesh_device,
                    mesh_mapper=mesh_mapper,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=tt_dtype,
                    cache_file_name=None,
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

        # Debug-only correctness knob: clear the *first* paged KV block referenced by
        # this request before running prefill via the decode-loop fallback.
        #
        # Motivation: if the paged MLA decode kernel incorrectly reads tokens beyond
        # `cur_pos` within a KV cache block, stale values (from previously served
        # requests that reused the same physical KV block) can leak into attention
        # and make greedy decode nondeterministic.
        #
        # This is not a production solution, but it is a fast A/B test to confirm
        # whether stale KV data is the root cause of gibberish/nondeterminism.
        clear_block0 = os.environ.get("GLM4_MOE_LITE_CLEAR_KV_CACHE_BLOCK0", "").strip() == "1"
        if clear_block0:
            try:
                block_size = int(getattr(self, "_kv_cache_shape", (0, 0, 64, 0))[2]) or 64
                kvpe_dim = int(self.hparams.kv_lora_rank + self.hparams.qk_rope_head_dim)
                is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"
                mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh_device else None

                # Reuse a single zero-fill tensor across all layers for this prefill call.
                kv_dtype = kv_cache[0].dtype if isinstance(kv_cache, list) and kv_cache else ttnn.bfloat8_b
                zero_fill_torch = torch.zeros((1, 1, block_size, kvpe_dim), dtype=torch.bfloat16, device="cpu")
                zero_fill_tt = ttnn.from_torch(
                    zero_fill_torch,
                    device=self.mesh_device,
                    dtype=kv_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )

                for i in range(batch):
                    if int(prompt_lens[i]) <= 0:
                        continue
                    block0 = int(page_table[i, 0].item())
                    page_table_fill = torch.full_like(page_table[i : i + 1, :], fill_value=block0, dtype=torch.int32)
                    page_table_tt = ttnn.from_torch(
                        page_table_fill,
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mapper,
                    )
                    for layer_cache in kv_cache:
                        ttnn.experimental.paged_fill_cache(layer_cache, zero_fill_tt, page_table=page_table_tt, batch_idx=0)
                    ttnn.deallocate(page_table_tt, force=False)

                ttnn.deallocate(zero_fill_tt, force=False)
            except Exception as e:
                logger.warning("GLM4_MOE_LITE_CLEAR_KV_CACHE_BLOCK0 failed (ignored): {}", e)

        impl = os.environ.get("GLM4_MOE_LITE_PREFILL_IMPL", "").strip().lower() or "flash_mla_prefill"
        if impl in {"mla", "flash_mla", "flash_mla_prefill", "prefill"}:
            return self._tt_runner.prefill(tokens=tokens, prompt_lens=prompt_lens, page_table=page_table, kv_cache=kv_cache)

        if impl not in {"decode_loop", "decode", "decode_loop_trace"}:
            raise ValueError(
                f"Invalid GLM4_MOE_LITE_PREFILL_IMPL={impl!r}; expected one of "
                "['flash_mla_prefill', 'decode_loop', 'decode_loop_trace']"
            )

        # Fallback: iterative decode-per-token prefill. This is a pragmatic
        # workaround for large/slow shape-specialized prefill graphs.
        #
        # decode_loop:
        # - correctness-only
        # - runs non-traced decode and reads logits each step (slow)
        #
        # decode_loop_trace:
        # - uses decode trace replay for each prompt token
        # - uses on-device greedy sampling for intermediate tokens to avoid
        #   reading full logits back to host for every token
        #
        # Note: this still computes the LM head for intermediate tokens (the
        # trace graph produces logits). A future optimization is to add an
        # "update-cache-only" trace that stops before the LM head.
        last_logits = []
        for i in range(batch):
            prompt_len = int(prompt_lens[i])
            if prompt_len <= 0:
                last_logits.append(torch.zeros((1, vocab), dtype=torch.float32))
                continue

            user_page_table = page_table[i : i + 1, :]
            logits_i = None
            # Debug knob: avoid on-device sampling during decode-loop prefill.
            #
            # Motivation: If on-device argmax has correctness issues (e.g. memory
            # lifetime, kernel bug), it can corrupt subsequent computation even
            # though we discard intermediate prompt-token outputs. Disabling this
            # forces host logits readback for intermediate prompt tokens.
            use_intermediate_sampling = os.environ.get(
                "GLM4_MOE_LITE_PREFILL_INTERMEDIATE_SAMPLING", "1"
            ).strip() not in {"0", "false", "no", "off"}
            for t in range(prompt_len):
                tok = tokens[i, t].view(1, 1).to(torch.int32)
                pos = torch.tensor([t], dtype=torch.int32)
                if t < (prompt_len - 1):
                    # Intermediate prompt tokens: update KV cache, but avoid reading
                    # back full logits. Use on-device greedy sampling (ids only) and
                    # discard the result.
                    kwargs_decode: dict[str, object] = {
                        "tokens": tok,
                        "start_pos": pos,
                        "page_table": user_page_table,
                        "kv_cache": kv_cache,
                        "enable_trace": (impl == "decode_loop_trace"),
                    }
                    if use_intermediate_sampling:
                        kwargs_decode["sampling_params"] = {"temperature": 0.0}
                    _ = self._tt_runner.decode(**kwargs_decode)
                    continue

                logits_i = self._tt_runner.decode(
                    tokens=tok,
                    start_pos=pos,
                    page_table=user_page_table,
                    kv_cache=kv_cache,
                    enable_trace=(impl == "decode_loop_trace"),
                )  # [1,1,V]
            assert logits_i is not None
            last_logits.append(logits_i[0])  # [1,V]

        if os.environ.get("GLM4_MOE_LITE_SYNC_DEVICE", "").strip() == "1":
            # Debug-only: force full device sync at the end of prefill to rule out
            # cross-request overlap or async lifetime issues.
            ttnn.synchronize_device(self.mesh_device)

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
        read_from_device: bool = bool(kwargs.get("read_from_device", False))

        # Warmup can pass a dummy all-zero page table; patch it to avoid
        # overlapping updates to the same physical block.
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
            num_main_lanes=int(kwargs.get("num_main_lanes", 0)),
        )
        if os.environ.get("GLM4_MOE_LITE_SYNC_DEVICE", "").strip() == "1":
            # Debug-only: force full device sync at the end of decode to rule out
            # cross-request overlap or async lifetime issues.
            ttnn.synchronize_device(self.mesh_device)

        # Handle MTP output: eager path returns (main, draft) tuple
        self._last_draft_token_ids = None
        if isinstance(tt_out, tuple) and len(tt_out) == 2:
            first = tt_out[0]
            # Distinguish MTP tuple (torch, torch) from vocab-sharded tuple (ttnn, ttnn)
            if isinstance(first, torch.Tensor):
                main_out, draft_ids = tt_out
                self._last_draft_token_ids = draft_ids  # [active] int32 on host
                tt_out = main_out
        # Check trace path stored draft tokens on the runner
        if self._last_draft_token_ids is None and hasattr(self._tt_runner, '_last_draft_token_ids'):
            self._last_draft_token_ids = self._tt_runner._last_draft_token_ids
            self._tt_runner._last_draft_token_ids = None

        if read_from_device:
            # Used by vLLM warmup. Force a synchronous readback so compilation/tracing
            # work happens during warmup rather than at first user request.
            tt_host = self.read_decode_output(tt_out, async_read=False)
            return self.process_decode_output_host(tt_host, is_tokens=(sampling_params is not None))
        return tt_out

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
        # tt_out can be:
        # - torch.Tensor: already on host (compat sampling/logits path)
        # - ttnn.Tensor: device tensor (sample-on-device tokens)
        # - tuple[ttnn.Tensor, ttnn.Tensor]: per-shard (max, argmax) for vocab-sharded LM head
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

        # Vocab-sharded LM head returns (local_max, local_argmax) for each shard.
        if isinstance(tt_out, tuple):
            if self._tt_runner is None:
                raise RuntimeError("TT runner is not initialized")
            local_max_tt, local_argmax_tt = tt_out

            max_dts = ttnn.get_device_tensors(local_max_tt)
            idx_dts = ttnn.get_device_tensors(local_argmax_tt)
            if not max_dts or not idx_dts:
                raise RuntimeError("expected mesh tensors for vocab-sharded decode output")

            mesh_rows, mesh_cols = int(self.mesh_device.shape[0]), int(self.mesh_device.shape[1])
            tp_axis = self._tt_runner.lm_head_tp_axis
            if tp_axis is None:
                tp_size = int(mesh_rows * mesh_cols)
                selected_device_ids = list(range(tp_size))
                shard_indices = list(range(tp_size))
            elif int(tp_axis) == 1:
                tp_size = mesh_cols
                selected_device_ids = [c for c in range(tp_size)]
                shard_indices = [c for c in range(tp_size)]
            else:
                tp_size = mesh_rows
                selected_device_ids = [r * mesh_cols for r in range(tp_size)]
                shard_indices = [r for r in range(tp_size)]

            local_max_torch = [_tt_to_torch_device0(max_dts[device_id]).reshape(-1) for device_id in selected_device_ids]
            local_idx_torch = [
                _tt_to_torch_device0(idx_dts[device_id]).reshape(-1) for device_id in selected_device_ids
            ]

            vocab = int(self._tt_runner.hparams.vocab_size)
            vocab_per_shard = int(self._tt_runner.lm_head_vocab_per_shard)
            batch = int(local_idx_torch[0].numel())
            next_ids = torch.empty((batch,), dtype=torch.int32)
            for b in range(batch):
                best_val = None
                best_global = None
                for shard_idx, (val_tensor, idx_tensor) in enumerate(zip(local_max_torch, local_idx_torch)):
                    max_val = float(val_tensor[b].item())
                    local_idx = int(idx_tensor[b].item())
                    global_idx = int(shard_indices[shard_idx] * vocab_per_shard + local_idx)
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = int(best_global)
            return next_ids

        next_ids_torch = _tt_to_torch_device0(tt_out).reshape(-1).to(dtype=torch.int32).cpu()
        return next_ids_torch

    def get_spec_token_ids(self, num_reqs: int) -> list[list[int]] | None:
        """Return MTP draft tokens from the last decode step, or None."""
        draft = self._last_draft_token_ids
        if draft is None:
            return None
        result = [[int(draft[b].item())] for b in range(min(int(draft.shape[0]), num_reqs))]
        while len(result) < num_reqs:
            result.append([])
        self._last_draft_token_ids = None
        return result
