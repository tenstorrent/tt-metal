# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

import ttnn
from loguru import logger

from models.common.rmsnorm import RMSNorm
from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import (
    prepare_decode_rope_and_positions_tt,
    prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt,
    run_decoder_layer_decode_one_step_update_cache_tt,
    run_decoder_layer_prefill_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.layer0_tt import _rot_transformation_mat_torch, make_rope_tensors
from models.demos.glm4_moe_lite.tt.layer_weights import (
    _env_dense_dtype,
    _env_tp_enabled,
    _linear_weight_tt,
    _tp_axis_and_size,
    _tp_mesh_mapper,
    convert_decoder_layer_weights,
)
from models.demos.glm4_moe_lite.tt.tt_embedding import (
    convert_embedding_weight_to_tt,
    run_tt_embedding,
)
from models.demos.glm4_moe_lite.tt.weights import LazyStateDict, load_glm_lazy_state_dict


@dataclass
class _DecodeTraceSamplingState:
    """Per-bucket state for batch-bucketed decode traces."""
    trace_id: Any | None = None
    batch: int = 0
    page_table_width: int = 0
    # Persistent inputs
    tokens_tt: ttnn.Tensor | None = None
    positions_tt: ttnn.Tensor | None = None
    rot_idxs_tt: ttnn.Tensor | None = None
    cos_batch_tt: ttnn.Tensor | None = None
    sin_batch_tt: ttnn.Tensor | None = None
    trans_matrix_tt: ttnn.Tensor | None = None
    page_table_tt: ttnn.Tensor | None = None
    rope_sharded_mem_config: Any | None = None
    rot_idxs_padded_batch: int = 0
    # Persistent outputs
    logits_tt: ttnn.Tensor | None = None
    top1_values_tt: ttnn.Tensor | None = None
    top1_indices_tt: ttnn.Tensor | None = None


def _torch_dtype_to_ttnn(dtype: torch.dtype) -> ttnn.DataType:
    # NOTE: vLLM passes a torch dtype for sizing/accounting, but TT kernels
    # often prefer BF8 KV caches for both memory and kernel constraints.
    override = os.environ.get("GLM4_MOE_LITE_KV_CACHE_TT_DTYPE", "").strip().lower()
    if override:
        if override in {"bf8", "bfloat8_b"}:
            return ttnn.bfloat8_b
        if override in {"bf16", "bfloat16"}:
            return ttnn.bfloat16
        if override in {"f16", "fp16", "float16"}:
            return ttnn.float16
        if override in {"f32", "fp32", "float32"}:
            return ttnn.float32
        raise ValueError(f"Invalid GLM4_MOE_LITE_KV_CACHE_TT_DTYPE={override!r}")

    # Default: BF8 KV cache (production-intended).
    return ttnn.bfloat8_b


def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tt_to_torch_for_vllm_output(*, tensor: ttnn.Tensor, device: Any) -> torch.Tensor:
    """
    Convert a TT tensor to torch for vLLM outputs.

    Design choice (current GLM bring-up):
    - When running on a mesh, we read back **only device 0**. This matches the
      "replicated model" bring-up mode where every device executes identical work.

    If/when GLM is truly sharded across the mesh (DP/TP/EP), this must be
    replaced with a topology-aware composer matching vLLM’s batch layout
    contract.
    """
    if not _is_mesh_device(device):
        # `ttnn.to_torch` can be async depending on runtime settings; stage
        # through a blocking device->host transfer for correctness.
        return ttnn.to_torch(tensor.cpu())
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    return ttnn.to_torch(device_tensors[0].cpu())


def _mesh_to_torch_selected(*, tensor: ttnn.Tensor, device_ids: list[int]) -> list[torch.Tensor]:
    """Convert a subset of device tensors from a mesh tensor into torch tensors."""
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    out: list[torch.Tensor] = []
    for device_id in device_ids:
        if device_id < 0 or device_id >= len(device_tensors):
            raise IndexError(f"device_id {device_id} out of range for mesh tensor with {len(device_tensors)} devices")
        out.append(ttnn.to_torch(device_tensors[device_id].cpu()))
    return out


def _load_hparams_from_snapshot(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _profile_enabled() -> bool:
    return os.environ.get("GLM4_MOE_LITE_PROFILE", "").strip() == "1"


def _profile_layer_filter() -> int:
    raw = os.environ.get("GLM4_MOE_LITE_PROFILE_LAYER", "").strip()
    if not raw:
        return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _profile_print_every() -> int:
    raw = os.environ.get("GLM4_MOE_LITE_PROFILE_PRINT_EVERY", "").strip()
    if not raw:
        return 32
    try:
        return max(1, int(raw))
    except ValueError:
        return 32


@dataclass
class Glm4MoeLiteDenseOnlyTT:
    """Correctness-first TT runner for GLM-4.7-Flash (dense-only bring-up).

    Current scope:
    - MLA attention decode with vLLM paged KV cache update semantics.
    - Dense MLP for layer0; shared-expert-as-dense for layers >= first_k_dense_replace.

    Not yet supported:
    - Routed experts (true MoE), expert parallelism, performance tuning.
    - Prefix caching (start_pos offsets) in prefill.
    """

    device: Any
    snapshot_dir: Path
    cache_dir: Path
    max_seq_len: int

    hparams: Glm4MoeLiteHParams
    state: LazyStateDict
    embed_w: ttnn.Tensor
    rope: dict[str, Any]
    final_norm: RMSNorm
    lm_head_w: ttnn.Tensor
    lm_head_sharded_vocab: bool
    lm_head_tp_axis: int | None
    lm_head_tp_size: int
    lm_head_vocab_per_shard: int
    layer_weights: dict[int, Any]
    num_layers_to_run: int
    enable_moe: bool
    moe_runtime: Any | None
    # ---- MTP (Multi-Token Prediction) state ----
    mtp_enabled: bool = False
    mtp_enorm: Any | None = None
    mtp_hnorm: Any | None = None
    mtp_eh_proj_w: ttnn.Tensor | None = None
    mtp_shared_head_norm: Any | None = None
    mtp_shared_head_w: ttnn.Tensor | None = None
    mtp_decoder_w: Any | None = None
    # ---- Decode trace state (vLLM trace_mode=all) ----
    # Batch-bucketed traces: one _DecodeTraceSamplingState per batch bucket.
    # At runtime, decode pads to the nearest bucket instead of MAX_NUM_SEQS,
    # giving small batches more cores/seq and lower ITL.
    _decode_trace_states: dict[int, _DecodeTraceSamplingState] = field(init=False, default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        device: Any,
        snapshot_dir: Path,
        cache_dir: Path,
        max_seq_len: int,
        hparams: Optional[Glm4MoeLiteHParams] = None,
    ) -> "Glm4MoeLiteDenseOnlyTT":
        snapshot_dir = Path(snapshot_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hparams = _load_hparams_from_snapshot(snapshot_dir) if hparams is None else hparams

        # Lazy view over safetensors (do not materialize all weights).
        state = load_glm_lazy_state_dict(snapshot_dir, num_layers=int(hparams.num_hidden_layers))

        embed_cache = cache_dir / "embed_w"
        embed_w = convert_embedding_weight_to_tt(
            device=device,
            embed_weight=state["model.embed_tokens.weight"],
            cache_file_name=embed_cache,
            dtype=ttnn.bfloat16,
        )

        # Shared RoPE cache across all layers.
        rope = make_rope_tensors(
            device=device,
            seq_len=int(max_seq_len),
            rope_dim=int(hparams.qk_rope_head_dim),
            rope_theta=float(hparams.rope_theta),
        )

        # Output norm + LM head.
        dense_dtype = _env_dense_dtype()
        final_norm = RMSNorm(
            device=device,
            dim=int(hparams.hidden_size),
            eps=float(hparams.rms_norm_eps),
            state_dict=state,
            state_dict_prefix="model.",
            weight_key="norm",
            weight_cache_path=cache_dir,
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
        )
        tp_enabled = _env_tp_enabled()
        lm_head_mapper = None
        lm_head_variant = ""
        lm_head_sharded_vocab = False
        lm_head_tp_axis = None
        lm_head_tp_size = 1
        lm_head_vocab_per_shard = int(hparams.vocab_size)
        num_devices = 1
        if _is_mesh_device(device):
            num_devices = int(device.shape[0]) * int(device.shape[1])
        if tp_enabled and num_devices > 1:
            vocab = int(hparams.vocab_size)
            if vocab % int(num_devices) != 0:
                raise ValueError(
                    f"LM head TP requires vocab divisible by num_devices. Got vocab={vocab} num_devices={num_devices}. "
                    "Disable GLM4_MOE_LITE_TP or add vocab padding support."
                )
            # Shard vocab across all mesh devices.
            #
            # NOTE: we intentionally use a 1D mesh sharding mapper here to avoid
            # sub-grid write patterns that can hang during warmup on some meshes.
            lm_head_mapper = ttnn.ShardTensorToMesh(device, dim=3)
            lm_head_variant = f"shard{num_devices}_v1"
            lm_head_sharded_vocab = True
            lm_head_tp_axis = None
            lm_head_tp_size = int(num_devices)
            lm_head_vocab_per_shard = vocab // int(num_devices)
        lm_head_w = _linear_weight_tt(
            device=device,
            torch_weight_out_in=state["lm_head.weight"],
            cache_file=cache_dir / f"lm_head_w_{lm_head_variant}" if lm_head_variant else cache_dir / "lm_head_w",
            dtype=dense_dtype,
            mesh_mapper=lm_head_mapper,
        )

        num_layers_env = os.environ.get("GLM4_MOE_LITE_NUM_LAYERS", "").strip()
        if num_layers_env and os.environ.get("GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS", "").strip() != "1":
            raise ValueError(
                "GLM4_MOE_LITE_NUM_LAYERS is debug-only. "
                "Set GLM4_MOE_LITE_DEBUG_ALLOW_PARTIAL_LAYERS=1 to run a partial model."
            )
        num_layers_to_run = int(num_layers_env) if num_layers_env else int(hparams.num_hidden_layers)
        num_layers_to_run = max(1, min(num_layers_to_run, int(hparams.num_hidden_layers)))

        enable_moe = os.environ.get("GLM4_MOE_LITE_ENABLE_MOE", "").strip() == "1"
        moe_runtime = None
        if enable_moe:
            from models.demos.glm4_moe_lite.tt.moe_tt import create_moe_runtime

            moe_runtime = create_moe_runtime(device=device, hparams=hparams)

        # ---- MTP Layer 47 (optional) ----
        mtp_enabled = os.environ.get("GLM4_MOE_LITE_MTP", "").strip() == "1"
        mtp_enorm = None
        mtp_hnorm = None
        mtp_eh_proj_w = None
        mtp_shared_head_norm = None
        mtp_shared_head_w = None
        mtp_decoder_w = None

        if mtp_enabled:
            logger.info("MTP enabled: loading layer 47 weights")
            mtp_cache = cache_dir / "mtp"
            mtp_cache.mkdir(parents=True, exist_ok=True)

            # Unfiltered state dict that includes layer 47
            mtp_state = load_glm_lazy_state_dict(snapshot_dir)

            hidden = int(hparams.hidden_size)

            # enorm: RMSNorm(hidden_size)
            mtp_enorm = RMSNorm(
                device=device,
                dim=hidden,
                eps=float(hparams.rms_norm_eps),
                state_dict=mtp_state,
                state_dict_prefix="model.layers.47.",
                weight_key="enorm",
                weight_cache_path=mtp_cache,
                weight_dtype=ttnn.bfloat16,
                is_distributed=False,
            )

            # hnorm: RMSNorm(hidden_size)
            mtp_hnorm = RMSNorm(
                device=device,
                dim=hidden,
                eps=float(hparams.rms_norm_eps),
                state_dict=mtp_state,
                state_dict_prefix="model.layers.47.",
                weight_key="hnorm",
                weight_cache_path=mtp_cache,
                weight_dtype=ttnn.bfloat16,
                is_distributed=False,
            )

            # eh_proj: Linear(2*hidden -> hidden, no bias)
            mtp_eh_proj_w = _linear_weight_tt(
                device=device,
                torch_weight_out_in=mtp_state["model.layers.47.eh_proj.weight"],
                cache_file=mtp_cache / "eh_proj_w",
                dtype=dense_dtype,
            )

            # shared_head.norm: RMSNorm(hidden_size)
            mtp_shared_head_norm = RMSNorm(
                device=device,
                dim=hidden,
                eps=float(hparams.rms_norm_eps),
                state_dict=mtp_state,
                state_dict_prefix="model.layers.47.shared_head.",
                weight_key="norm",
                weight_cache_path=mtp_cache,
                weight_dtype=ttnn.bfloat16,
                is_distributed=False,
            )

            # shared_head.head: Linear(hidden -> vocab_size), SEPARATE from main lm_head
            mtp_shared_head_w = _linear_weight_tt(
                device=device,
                torch_weight_out_in=mtp_state["model.layers.47.shared_head.head.weight"],
                cache_file=mtp_cache / f"shared_head_w_{lm_head_variant}" if lm_head_variant else mtp_cache / "shared_head_w",
                dtype=dense_dtype,
                mesh_mapper=lm_head_mapper,
            )

            # Full decoder layer 47 (attention + MoE)
            mtp_decoder_w = convert_decoder_layer_weights(
                device=device,
                state=mtp_state,
                layer_idx=47,
                hparams=hparams,
                cache_dir=cache_dir / "layers",
                force_shared_expert_dense=False,
                enable_moe=enable_moe,
            )

            logger.info("MTP layer 47 weights loaded successfully")

        return cls(
            device=device,
            snapshot_dir=snapshot_dir,
            cache_dir=cache_dir,
            max_seq_len=int(max_seq_len),
            hparams=hparams,
            state=state,
            embed_w=embed_w,
            rope=rope,
            final_norm=final_norm,
            lm_head_w=lm_head_w,
            lm_head_sharded_vocab=lm_head_sharded_vocab,
            lm_head_tp_axis=lm_head_tp_axis,
            lm_head_tp_size=lm_head_tp_size,
            lm_head_vocab_per_shard=lm_head_vocab_per_shard,
            layer_weights={},
            num_layers_to_run=num_layers_to_run,
            enable_moe=enable_moe,
            moe_runtime=moe_runtime,
            mtp_enabled=mtp_enabled,
            mtp_enorm=mtp_enorm,
            mtp_hnorm=mtp_hnorm,
            mtp_eh_proj_w=mtp_eh_proj_w,
            mtp_shared_head_norm=mtp_shared_head_norm,
            mtp_shared_head_w=mtp_shared_head_w,
            mtp_decoder_w=mtp_decoder_w,
        )

    def _ensure_layer_weights(self, layer_idx: int) -> Any:
        layer_idx = int(layer_idx)
        w = self.layer_weights.get(layer_idx)
        if w is not None:
            return w

        w = convert_decoder_layer_weights(
            device=self.device,
            state=self.state,
            layer_idx=layer_idx,
            hparams=self.hparams,
            cache_dir=self.cache_dir / "layers",
            force_shared_expert_dense=False,
            enable_moe=self.enable_moe,
        )
        self.layer_weights[layer_idx] = w
        return w

    def _profile_record(
        self,
        *,
        phase: str,
        stage_totals: dict[str, float],
        token_count: int,
        layer_filter: int,
    ) -> None:
        if token_count <= 0:
            return
        state_key = f"_profile_state_{phase}"
        state = getattr(self, state_key, None)
        if state is None:
            state = {"calls": 0, "tokens": 0, "stages": {}}
        state["calls"] = int(state["calls"]) + 1
        state["tokens"] = int(state["tokens"]) + int(token_count)
        stages = state["stages"]
        for key, value in stage_totals.items():
            stages[key] = float(stages.get(key, 0.0)) + float(value)
        setattr(self, state_key, state)

        every = _profile_print_every()
        if int(state["calls"]) % every != 0:
            return

        tokens = max(1, int(state["tokens"]))
        total_s = float(stages.get("total_s", 0.0))
        agg_tps = (tokens / total_s) if total_s > 0 else 0.0

        top = [
            (k, float(v))
            for k, v in stages.items()
            if k != "total_s" and float(v) > 0.0
        ]
        top.sort(key=lambda item: item[1], reverse=True)
        top = top[:8]
        top_str = ", ".join(f"{k}={1000.0 * v / tokens:.3f}ms/tok" for k, v in top)
        layer_txt = f" layer_filter={layer_filter}" if layer_filter >= 0 else ""
        print(
            f"[glm4_moe_lite][profile][{phase}]{layer_txt} calls={state['calls']} "
            f"tokens={tokens} agg_tps={agg_tps:.3f} {top_str}",
            flush=True,
        )

    @torch.no_grad()
    def prefill(
        self,
        *,
        tokens: torch.Tensor,  # [B,S] int32
        prompt_lens: list[int],  # length B
        page_table: torch.Tensor,  # [B,W] int32
        kv_cache: list[ttnn.Tensor],  # per-layer KVPE cache tensors
        seq_pad_multiple: int = 32,
    ) -> torch.Tensor:
        """Compute logits for the last prompt token for each request and fill KV caches.

        This is a correctness-first (non-optimized) prefill implementation:
        - Runs each request independently (B loop).
        - Runs full-sequence FlashMLA prefill for each layer.
        - Writes KVPE to the paged KV cache using `paged_fill_cache`.
        """
        # Release any active decode trace before prefill. The trace reserves
        # device memory that conflicts with prefill's dynamic allocations.
        # The trace will be lazily re-captured on the next decode call
        # (via _decode_trace_sampling → _capture_decode_trace_sampling).
        # Trace INPUT tensors are preserved for fast re-capture.
        #
        # When GLM4_MOE_LITE_PRESERVE_TRACE=1, skip the release to avoid the
        # ~6s re-capture overhead after each prefill. If the prefill OOMs
        # without the trace being released, we catch it, release the trace,
        # and retry.
        preserve_trace = os.environ.get("GLM4_MOE_LITE_PRESERVE_TRACE", "").strip() == "1"
        if self._decode_trace_states and not preserve_trace:
            self._release_all_decode_traces()

        if tokens.ndim != 2:
            raise ValueError(f"expected tokens [B,S], got {tuple(tokens.shape)}")
        if page_table.ndim != 2:
            raise ValueError(f"expected page_table [B,W], got {tuple(page_table.shape)}")
        batch, seq_total = tokens.shape
        if len(prompt_lens) != int(batch):
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {int(batch)}")

        return self._prefill_compute(
            tokens=tokens,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            seq_pad_multiple=seq_pad_multiple,
            preserve_trace=preserve_trace,
        )

    def _release_all_decode_traces(self) -> None:
        """Release ALL bucket decode traces and deallocate trace output tensors."""
        if not self._decode_trace_states:
            return
        ttnn.synchronize_device(self.device)
        for bucket, state in list(self._decode_trace_states.items()):
            if state.trace_id is not None:
                try:
                    ttnn.release_trace(self.device, state.trace_id)
                except Exception:
                    pass
                state.trace_id = None
            for t in (state.logits_tt, state.top1_values_tt, state.top1_indices_tt):
                if t is not None:
                    try:
                        ttnn.deallocate(t, force=True)
                    except Exception:
                        pass
            state.logits_tt = None
            state.top1_values_tt = None
            state.top1_indices_tt = None
        ttnn.synchronize_device(self.device)

    def _release_decode_trace(self) -> None:
        """Release ALL bucket decode traces (compat alias)."""
        self._release_all_decode_traces()

    def _prefill_compute(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
        seq_pad_multiple: int,
        preserve_trace: bool,
    ) -> torch.Tensor:
        """Inner prefill compute. Separated to allow OOM retry with trace release."""
        try:
            return self._prefill_compute_inner(
                tokens=tokens,
                prompt_lens=prompt_lens,
                page_table=page_table,
                kv_cache=kv_cache,
                seq_pad_multiple=seq_pad_multiple,
            )
        except Exception as e:
            if preserve_trace and any(s.trace_id is not None for s in self._decode_trace_states.values()):
                logger.warning(
                    "Prefill failed with preserved trace; releasing trace and retrying: {}", e
                )
                self._release_decode_trace()
                return self._prefill_compute_inner(
                    tokens=tokens,
                    prompt_lens=prompt_lens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    seq_pad_multiple=seq_pad_multiple,
                )
            raise

    def _prefill_compute_inner(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
        seq_pad_multiple: int,
    ) -> torch.Tensor:
        batch, seq_total = tokens.shape
        vocab = int(self.hparams.vocab_size)
        hidden = int(self.hparams.hidden_size)
        rope_dim = int(self.hparams.qk_rope_head_dim)
        pad_multiple = max(1, int(seq_pad_multiple))

        # Check if batched prefill is enabled and applicable.
        batched_prefill = (
            os.environ.get("GLM4_MOE_LITE_BATCHED_PREFILL", "").strip() == "1"
            and int(batch) > 1
            and all(int(pl) > 0 for pl in prompt_lens)
        )
        if batched_prefill:
            return self._prefill_compute_inner_batched(
                tokens=tokens,
                prompt_lens=prompt_lens,
                page_table=page_table,
                kv_cache=kv_cache,
                pad_multiple=pad_multiple,
            )

        is_mesh_device = self.device.__class__.__name__ == "MeshDevice"
        profile_on = _profile_enabled()
        profile_layer = _profile_layer_filter()
        profile_token_count = 0
        prefill_profile: dict[str, float] = {}
        t_prefill0 = time.perf_counter() if profile_on else 0.0

        out_logits: list[torch.Tensor] = []

        for i in range(int(batch)):
            prompt_len = int(prompt_lens[i])
            if prompt_len <= 0:
                out_logits.append(torch.zeros((1, vocab), dtype=torch.float32))
                continue
            profile_token_count += prompt_len
            if prompt_len > int(seq_total):
                raise ValueError(f"prompt_len[{i}]={prompt_len} > tokens.shape[1]={int(seq_total)}")

            padded_len = ((prompt_len + pad_multiple - 1) // pad_multiple) * pad_multiple
            if padded_len > int(self.max_seq_len):
                raise ValueError(
                    f"padded_len={padded_len} exceeds max_seq_len={int(self.max_seq_len)} "
                    f"(prompt_len={prompt_len}, seq_pad_multiple={pad_multiple})"
                )

            prompt_ids = tokens[i, :prompt_len].to(torch.int32).cpu()
            input_padded = torch.zeros((1, padded_len), dtype=torch.int32)
            input_padded[0, :prompt_len] = prompt_ids

            # Convert page table row to TT.
            page_row = page_table[i : i + 1, :].to(torch.int32)
            page_table_tt = ttnn.from_torch(
                page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh_device else None,
            )

            # Slice RoPE tables to the padded sequence length (shared across layers).
            #
            # IMPORTANT: `ttnn.slice` may return the input tensor itself when the slice spans the
            # entire tensor. Since TTNN does not refcount view aliases, deallocating the slice can
            # accidentally deallocate the cached base RoPE tensor. Avoid slicing when we want the
            # full table.
            rope_slices_owned = True
            if (
                int(padded_len) == int(self.rope["cos_matrix"].shape[2])
                and int(rope_dim) == int(self.rope["cos_matrix"].shape[3])
            ):
                cos_matrix = self.rope["cos_matrix"]
                sin_matrix = self.rope["sin_matrix"]
                rope_slices_owned = False
            else:
                cos_matrix = ttnn.slice(self.rope["cos_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim])
                sin_matrix = ttnn.slice(self.rope["sin_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim])

            # Embedding.
            t0 = time.perf_counter() if profile_on else 0.0
            x = run_tt_embedding(device=self.device, token_ids=input_padded, tt_weight=self.embed_w)
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, (1, 1, padded_len, hidden))
            if profile_on:
                prefill_profile["embed_s"] = prefill_profile.get("embed_s", 0.0) + (time.perf_counter() - t0)

            # Decoder stack (prefill).
            for layer_idx in range(self.num_layers_to_run):
                w = self._ensure_layer_weights(layer_idx)
                layer_profile: dict[str, float] | None = None
                if profile_on and (profile_layer < 0 or profile_layer == layer_idx):
                    layer_profile = {}
                x_next = run_decoder_layer_prefill_update_cache_tt(
                    device=self.device,
                    x_embed=x,
                    page_table_tt=page_table_tt,
                    kvpe_cache=kv_cache[layer_idx],
                    cos_matrix=cos_matrix,
                    sin_matrix=sin_matrix,
                    trans_matrix=self.rope["trans_matrix"],
                    w=w,
                    hparams=self.hparams,
                    prompt_len=prompt_len,
                    moe_runtime=self.moe_runtime,
                    profile=layer_profile,
                )
                if layer_profile is not None:
                    for key, value in layer_profile.items():
                        stage_key = f"layer_{key}"
                        prefill_profile[stage_key] = prefill_profile.get(stage_key, 0.0) + float(value)
                ttnn.deallocate(x, force=False)
                x = x_next

            # Logits for the last *real* prompt token only.
            x_last = ttnn.slice(x, [0, 0, prompt_len - 1, 0], [1, 1, prompt_len, hidden])
            ttnn.deallocate(x, force=False)

            t0 = time.perf_counter() if profile_on else 0.0
            x_last = self.final_norm(x_last, mode="decode")
            logits_tt = ttnn.linear(x_last, self.lm_head_w)  # [1,1,1,vocab]
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                cluster_axis = None if self.lm_head_tp_axis is None else int(self.lm_head_tp_axis)
                logits_tt_full = ttnn.all_gather(
                    logits_tt,
                    dim=3,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    cluster_axis=cluster_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(logits_tt, force=False)
                logits_tt = logits_tt_full
            if profile_on:
                prefill_profile["head_s"] = prefill_profile.get("head_s", 0.0) + (time.perf_counter() - t0)

            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
            logits_torch = logits_torch[..., :vocab]
            logits_flat = logits_torch.reshape(-1, vocab)
            if logits_flat.shape[0] != 1:
                raise RuntimeError(
                    f"prefill logits shape mismatch: expected 1 row, got {int(logits_flat.shape[0])} "
                    f"(logits_torch.shape={tuple(logits_torch.shape)})"
                )
            logits_i = logits_flat.to(dtype=torch.float32).cpu()
            out_logits.append(logits_i)

            # Cleanup.
            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(x_last, force=False)
            if rope_slices_owned:
                ttnn.deallocate(cos_matrix, force=False)
                ttnn.deallocate(sin_matrix, force=False)
            ttnn.deallocate(page_table_tt, force=False)

        if profile_on and profile_token_count > 0:
            prefill_profile["total_s"] = prefill_profile.get("total_s", 0.0) + (time.perf_counter() - t_prefill0)
            self._profile_record(
                phase="prefill",
                stage_totals=prefill_profile,
                token_count=profile_token_count,
                layer_filter=profile_layer,
            )
        # vLLM expects prefill logits as [B, 1, vocab] so it can slice the last
        # prompt position with `logits[:, -1, :]`. Each entry in out_logits is
        # [1, vocab], so stacking already yields [B, 1, vocab].
        return torch.stack(out_logits, dim=0)  # [B, 1, vocab]

    def _prefill_compute_inner_batched(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
        pad_multiple: int,
    ) -> torch.Tensor:
        """Batched prefill: process all B requests through the decoder stack simultaneously.

        All B requests are padded to a common S_max and concatenated along dim-2
        as [1,1,B*S_max,hidden].  The decoder layer's existing batch>1 support
        handles reshaping to [B,...] for RoPE, FlashMLA, and per-request KV cache fill.
        """
        batch = int(tokens.shape[0])
        seq_total = int(tokens.shape[1])
        vocab = int(self.hparams.vocab_size)
        hidden = int(self.hparams.hidden_size)
        rope_dim = int(self.hparams.qk_rope_head_dim)

        is_mesh_device = self.device.__class__.__name__ == "MeshDevice"
        profile_on = _profile_enabled()
        profile_layer = _profile_layer_filter()
        prefill_profile: dict[str, float] = {}
        t_prefill0 = time.perf_counter() if profile_on else 0.0

        # Compute per-request padded lengths and find the common S_max.
        int_prompt_lens: list[int] = [int(pl) for pl in prompt_lens]
        padded_lens: list[int] = []
        for pl in int_prompt_lens:
            if pl > seq_total:
                raise ValueError(f"prompt_len={pl} > tokens.shape[1]={seq_total}")
            padded = ((pl + pad_multiple - 1) // pad_multiple) * pad_multiple
            if padded > int(self.max_seq_len):
                raise ValueError(
                    f"padded_len={padded} exceeds max_seq_len={int(self.max_seq_len)} "
                    f"(prompt_len={pl}, seq_pad_multiple={pad_multiple})"
                )
            padded_lens.append(padded)

        s_max = max(padded_lens)
        profile_token_count = sum(int_prompt_lens)
        logger.info(
            "Batched prefill: B={}, S_max={}, prompt_lens={}",
            batch, s_max, int_prompt_lens,
        )

        # Build concatenated token tensor [1, B*S_max] with per-request padding.
        input_concat = torch.zeros((1, batch * s_max), dtype=torch.int32)
        for i in range(batch):
            pl = int_prompt_lens[i]
            offset = i * s_max
            input_concat[0, offset : offset + pl] = tokens[i, :pl].to(torch.int32).cpu()

        # Build page table [B, W] on device.
        page_table_all = page_table[:batch, :].to(torch.int32)
        page_table_tt = ttnn.from_torch(
            page_table_all,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh_device else None,
        )

        # Slice RoPE tables to S_max.
        rope_slices_owned = True
        if (
            int(s_max) == int(self.rope["cos_matrix"].shape[2])
            and int(rope_dim) == int(self.rope["cos_matrix"].shape[3])
        ):
            cos_matrix = self.rope["cos_matrix"]
            sin_matrix = self.rope["sin_matrix"]
            rope_slices_owned = False
        else:
            cos_matrix = ttnn.slice(self.rope["cos_matrix"], [0, 0, 0, 0], [1, 1, s_max, rope_dim])
            sin_matrix = ttnn.slice(self.rope["sin_matrix"], [0, 0, 0, 0], [1, 1, s_max, rope_dim])

        # Embedding: [1, B*S_max] -> [1, 1, B*S_max, hidden].
        t0 = time.perf_counter() if profile_on else 0.0
        x = run_tt_embedding(device=self.device, token_ids=input_concat, tt_weight=self.embed_w)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, 1, batch * s_max, hidden))
        if profile_on:
            prefill_profile["embed_s"] = time.perf_counter() - t0

        # Decoder stack (batched prefill).
        for layer_idx in range(self.num_layers_to_run):
            w = self._ensure_layer_weights(layer_idx)
            layer_profile: dict[str, float] | None = None
            if profile_on and (profile_layer < 0 or profile_layer == layer_idx):
                layer_profile = {}
            x_next = run_decoder_layer_prefill_update_cache_tt(
                device=self.device,
                x_embed=x,
                page_table_tt=page_table_tt,
                kvpe_cache=kv_cache[layer_idx],
                cos_matrix=cos_matrix,
                sin_matrix=sin_matrix,
                trans_matrix=self.rope["trans_matrix"],
                w=w,
                hparams=self.hparams,
                prompt_len=s_max,  # padded length (max across batch)
                batch=batch,
                prompt_lens=int_prompt_lens,
                moe_runtime=self.moe_runtime,
                profile=layer_profile,
            )
            if layer_profile is not None:
                for key, value in layer_profile.items():
                    stage_key = f"layer_{key}"
                    prefill_profile[stage_key] = prefill_profile.get(stage_key, 0.0) + float(value)
            ttnn.deallocate(x, force=False)
            x = x_next

        # Extract per-request logits from the batched output [1,1,B*S_max,hidden].
        # For each request i, the last real token is at offset i*S_max + prompt_lens[i]-1.
        out_logits: list[torch.Tensor] = []
        for i in range(batch):
            offset = i * s_max
            pos = offset + int_prompt_lens[i] - 1
            x_last = ttnn.slice(x, [0, 0, pos, 0], [1, 1, pos + 1, hidden])

            t0 = time.perf_counter() if profile_on else 0.0
            x_last = self.final_norm(x_last, mode="decode")
            logits_tt = ttnn.linear(x_last, self.lm_head_w)  # [1,1,1,vocab]
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                cluster_axis = None if self.lm_head_tp_axis is None else int(self.lm_head_tp_axis)
                logits_tt_full = ttnn.all_gather(
                    logits_tt,
                    dim=3,
                    num_links=1,
                    topology=ttnn.Topology.Linear,
                    cluster_axis=cluster_axis,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(logits_tt, force=False)
                logits_tt = logits_tt_full
            if profile_on:
                prefill_profile["head_s"] = prefill_profile.get("head_s", 0.0) + (time.perf_counter() - t0)

            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
            logits_torch = logits_torch[..., :vocab]
            logits_flat = logits_torch.reshape(-1, vocab)
            logits_i = logits_flat.to(dtype=torch.float32).cpu()
            out_logits.append(logits_i)

            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(x_last, force=False)

        ttnn.deallocate(x, force=False)
        if rope_slices_owned:
            ttnn.deallocate(cos_matrix, force=False)
            ttnn.deallocate(sin_matrix, force=False)
        ttnn.deallocate(page_table_tt, force=False)

        if profile_on and profile_token_count > 0:
            prefill_profile["total_s"] = time.perf_counter() - t_prefill0
            self._profile_record(
                phase="prefill_batched",
                stage_totals=prefill_profile,
                token_count=profile_token_count,
                layer_filter=profile_layer,
            )

        return torch.stack(out_logits, dim=0)  # [B, 1, vocab]

    @torch.no_grad()
    def decode(
        self,
        *,
        tokens: torch.Tensor,  # [B,1] int32
        start_pos: torch.Tensor,  # [B] int32 (padded with -1 for inactive)
        page_table: torch.Tensor,  # [B,W] int32
        kv_cache: list[ttnn.Tensor],  # per-layer KVPE cache tensors
        sampling_params: Any | None = None,
        enable_trace: bool = False,
    ) -> Any:
        """Run a decode step, updating KV cache.

        Returns:
        - logits on host as torch float32 [active, 1, vocab] when sampling_params is None
        - when sampling_params is not None (sample-on-device):
          - enable_trace=True: returns TT device tensors for async readback by the vLLM wrapper
          - enable_trace=False: returns next token ids on host as torch int32 [active]
        """
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError(f"expected tokens [B,1], got {tuple(tokens.shape)}")
        if start_pos.ndim != 1:
            raise ValueError(f"expected start_pos [B], got {tuple(start_pos.shape)}")
        if page_table.ndim != 2:
            raise ValueError(f"expected page_table [B,W], got {tuple(page_table.shape)}")

        start_pos = start_pos.to(torch.int32)
        active = int((start_pos >= 0).sum().item())
        if active <= 0:
            return torch.zeros((0, 1, int(self.hparams.vocab_size)), dtype=torch.float32)
        if enable_trace:
            if sampling_params is not None:
                # Greedy on-device sampling path (trace captures local top-1).
                return self._decode_trace_sampling(
                    tokens=tokens.to(torch.int32),
                    start_pos=start_pos,
                    page_table=page_table.to(torch.int32),
                    kv_cache=kv_cache,
                )
            # Host-sampling path: return full logits on host without device all-gather.
            return self._decode_trace_logits(
                tokens=tokens.to(torch.int32),
                start_pos=start_pos,
                page_table=page_table.to(torch.int32),
                kv_cache=kv_cache,
            )
        profile_on = _profile_enabled()
        profile_layer = _profile_layer_filter()
        decode_profile: dict[str, float] = {}
        t_decode0 = time.perf_counter() if profile_on else 0.0

        # Make the host inputs own their storage. vLLM can reuse/pad input
        # buffers across engine steps, and TTNN host->device copies can be
        # asynchronous under some runtime configurations.
        tokens = tokens[:active].to(torch.int32).contiguous().clone()
        positions = start_pos[:active].to(torch.int32).contiguous().clone()
        page_table = page_table[:active].to(torch.int32).contiguous()

        if os.environ.get("GLM4_MOE_LITE_DECODE_EMBED_ONLY", "").strip() == "1" and sampling_params is None:
            # Debug-only: skip KV cache update + all decoder layers and return
            # logits from (embed -> final_norm -> lm_head) only. This is useful
            # for isolating nondeterminism in low-level matmul/readback paths.
            x = run_tt_embedding(device=self.device, token_ids=tokens, tt_weight=self.embed_w)
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, (1, active, 1, int(self.hparams.hidden_size)))
            x = ttnn.permute(x, (0, 2, 1, 3))  # [1,1,B,D]
            x_view = ttnn.slice(x, [0, 0, 0, 0], [1, 1, active, int(self.hparams.hidden_size)])
            x_tight = ttnn.clone(x_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x, force=False)
            x = x_tight

            x = self.final_norm(x, mode="decode")
            logits_tt = ttnn.linear(x, self.lm_head_w)  # [1,1,B,vocab]

            vocab = int(self.hparams.vocab_size)
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                shards = ttnn.get_device_tensors(logits_tt)
                if not shards:
                    raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
                logits_shards = [ttnn.to_torch(t)[..., : int(t.shape[-1])] for t in shards]
                logits_full = torch.cat(logits_shards, dim=-1)[..., :vocab]
                logits_flat = logits_full.reshape(-1, vocab)
            else:
                logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
                logits_torch = logits_torch[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)

            if logits_flat.shape[0] != active:
                raise RuntimeError(
                    f"decode logits shape mismatch: expected {active} rows, got {int(logits_flat.shape[0])} "
                    f"(logits_flat.shape={tuple(logits_flat.shape)})"
                )
            if os.environ.get("GLM4_MOE_LITE_DEBUG_LOGITS_SANITY", "").strip() == "1":
                try:
                    finite = torch.isfinite(logits_flat)
                    finite_count = int(finite.sum().item())
                    total = int(logits_flat.numel())
                    # Quick proxy for distribution sharpness on the first row.
                    row0 = logits_flat[0]
                    top2 = torch.topk(row0, k=2).values
                    gap = float((top2[0] - top2[1]).item())
                    spread = float((row0.max() - row0.min()).item())
                    if finite_count != total or gap < 0.1:
                        logger.warning(
                            "GLM decode embed-only logits look suspicious: finite={}/{} top2_gap={:.6f} spread={:.6f} top1={:.6f}",
                            finite_count,
                            total,
                            gap,
                            spread,
                            float(top2[0].item()),
                        )
                except Exception as e:
                    logger.warning("GLM4_MOE_LITE_DEBUG_LOGITS_SANITY failed: {}", e)
            logits = logits_flat.reshape(active, 1, vocab).to(dtype=torch.float32).cpu()

            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(x, force=False)
            return logits

        # Correctness: vLLM uses a fixed-width page table (max blocks per req),
        # but decode only needs the prefix of blocks that cover the current
        # position. Some kernels have historically been sensitive to garbage in
        # unused page-table slots, which can manifest as nondeterministic greedy
        # outputs when KV blocks are reused across requests.
        #
        # Slice the page table down to the minimal required width for this
        # decode step to ensure we never pass unneeded block IDs to kernels.
        try:
            block_size = int(kv_cache[0].shape[2])
        except Exception:
            block_size = 64
        max_pos = int(positions.max().item()) if positions.numel() else 0
        blocks_needed = 1
        if block_size > 0:
            blocks_needed = max(1, max_pos // block_size + 1)

        if blocks_needed > int(page_table.shape[1]):
            msg = (
                f"decode page_table too narrow: blocks_needed={blocks_needed} "
                f"page_table.shape={tuple(page_table.shape)} max_pos={max_pos} block_size={block_size}. "
                "This indicates a vLLM block table allocation/hand-off bug; continuing will corrupt output."
            )
            if os.environ.get("GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY", "").strip() == "1":
                raise ValueError(msg)
            logger.warning(msg)
            blocks_needed = int(page_table.shape[1])

        if blocks_needed < int(page_table.shape[1]):
            page_table = page_table[:, :blocks_needed].contiguous()
        page_table = page_table.clone()

        if os.environ.get("GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY", "").strip() == "1":
            try:
                if 58 <= max_pos <= 70:
                    head_w = min(4, int(page_table.shape[1]))
                    head = page_table[0, :head_w].tolist() if active > 0 else []
                    logger.info(
                        "GLM boundary debug (model_tt.decode): active={} max_pos={} blocks_needed={} "
                        "page_table_shape={} page_table_head={}",
                        active,
                        max_pos,
                        blocks_needed,
                        tuple(page_table.shape),
                        head,
                    )
            except Exception as e:  # pragma: no cover
                logger.warning("GLM boundary debug (model_tt.decode) failed: {}", e)

        # vLLM uses a constant page_table width; accept any W here.
        t0 = time.perf_counter() if profile_on else 0.0
        is_mesh_device = _is_mesh_device(self.device)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh_device else None,
        )

        tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(
            device=self.device, rope=self.rope, positions=positions
        )

        # Decode-mode RoPE is faster but has been observed to be brittle during
        # bring-up. Default to the non-decode rotary kernel for correctness,
        # and only enable decode-mode when tracing (or explicitly requested).
        use_decode_rope = enable_trace or os.environ.get("GLM4_MOE_LITE_USE_DECODE_ROPE", "").strip() == "1"
        cos_decode = sin_decode = trans_decode = rope_sharded_cfg = None
        if use_decode_rope:
            cos_decode, sin_decode, trans_decode, rope_sharded_cfg = (
                prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt(
                    device=self.device,
                    cos_batch=cos_batch,
                    sin_batch=sin_batch,
                    trans_matrix=self.rope["trans_matrix"],
                    batch=active,
                    rope_dim=int(self.hparams.qk_rope_head_dim),
                )
            )
        if profile_on:
            decode_profile["prep_inputs_s"] = decode_profile.get("prep_inputs_s", 0.0) + (time.perf_counter() - t0)

        t0 = time.perf_counter() if profile_on else 0.0
        x = run_tt_embedding(device=self.device, token_ids=tokens, tt_weight=self.embed_w)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, active, 1, int(self.hparams.hidden_size)))
        x = ttnn.permute(x, (0, 2, 1, 3))  # [1,1,B,D]
        # Some TT tile-layout ops materialize/preserve tile padding in the logical
        # shape. Keep the decode batch dimension tight so downstream kernels (MoE,
        # FlashMLA, RoPE) do not execute inflated work on padded lanes.
        #
        # `slice` can return a view that aliases the source buffer (no refcounting).
        # Materialize the sliced tensor and then free the padded source to avoid
        # intermittent use-after-free corruption during decode.
        x_view = ttnn.slice(x, [0, 0, 0, 0], [1, 1, active, int(self.hparams.hidden_size)])
        x_tight = ttnn.clone(x_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # NOTE: `x_view` may alias `x`; do not deallocate it separately.
        ttnn.deallocate(x, force=False)
        x = x_tight
        if profile_on:
            decode_profile["embed_s"] = decode_profile.get("embed_s", 0.0) + (time.perf_counter() - t0)

        # Run decoder stack.
        for layer_idx in range(self.num_layers_to_run):
            w = self._ensure_layer_weights(layer_idx)
            layer_profile: dict[str, float] | None = None
            if profile_on and (profile_layer < 0 or profile_layer == layer_idx):
                layer_profile = {}
            x_next = run_decoder_layer_decode_one_step_update_cache_tt(
                device=self.device,
                x_embed_tok=x,
                tt_positions=tt_positions,
                page_table_tt=page_table_tt,
                kvpe_cache=kv_cache[layer_idx],
                cos_batch=cos_batch,
                sin_batch=sin_batch,
                trans_matrix=self.rope["trans_matrix"],
                cos_decode=cos_decode,
                sin_decode=sin_decode,
                trans_decode=trans_decode,
                rope_sharded_cfg=rope_sharded_cfg,
                w=w,
                hparams=self.hparams,
                moe_runtime=self.moe_runtime,
                profile=layer_profile,
                use_decode_rope=use_decode_rope,
            )
            if layer_profile is not None:
                for key, value in layer_profile.items():
                    stage_key = f"layer_{key}"
                    decode_profile[stage_key] = decode_profile.get(stage_key, 0.0) + float(value)
            ttnn.deallocate(x, force=False)
            x = x_next

        if use_decode_rope:
            assert cos_decode is not None and sin_decode is not None and trans_decode is not None
            ttnn.deallocate(cos_decode, force=False)
            ttnn.deallocate(sin_decode, force=False)
            ttnn.deallocate(trans_decode, force=False)

        t0 = time.perf_counter() if profile_on else 0.0
        x = self.final_norm(x, mode="decode")
        logits_tt = ttnn.linear(x, self.lm_head_w)  # [1,1,B,vocab]
        if profile_on:
            decode_profile["head_s"] = decode_profile.get("head_s", 0.0) + (time.perf_counter() - t0)

        if sampling_params is not None:
            # Basic on-device sampling: greedy (argmax).
            # vLLM will only request on-device sampling when the platform
            # indicates support; for GLM bring-up we intentionally implement
            # greedy-only first.
            # Multicore argmax expects ROW_MAJOR input.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            # Correctness: TT matmul outputs are tile-padded. If we run argmax over
            # the padded vocab region, we can select an out-of-range token id,
            # which can appear as gibberish/nondeterminism depending on whatever
            # values happen to be present in the padded lanes.
            vocab = int(self.hparams.vocab_size)

            if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
                logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
                # `slice` can return a view with non-trivial strides. Some reduction
                # kernels are sensitive to strided inputs; materialize a tight
                # buffer before taking top-1 to avoid rare garbage tokens.
                logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
                if isinstance(max_out, tuple):
                    local_max_tt, next_ids_tt = max_out
                    ttnn.deallocate(local_max_tt, force=False)
                else:
                    next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(logits_rm_tight, force=False)

                next_ids_torch = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
                next_ids_flat = next_ids_torch.reshape(-1).to(dtype=torch.int32).cpu()
                if int(next_ids_flat.numel()) != active:
                    raise RuntimeError(
                        f"decode token ids shape mismatch: expected {active} values, got {int(next_ids_flat.numel())} "
                        f"(next_ids_torch.shape={tuple(next_ids_torch.shape)})"
                    )
                if int(next_ids_flat.max().item()) >= vocab or int(next_ids_flat.min().item()) < 0:
                    # This should be impossible once the padded vocab region is excluded.
                    raise RuntimeError(
                        f"decode produced out-of-range token ids: min={int(next_ids_flat.min().item())} "
                        f"max={int(next_ids_flat.max().item())} vocab={vocab}"
                    )
                ttnn.deallocate(logits_rm, force=False)
                ttnn.deallocate(logits_tt, force=False)
                ttnn.deallocate(next_ids_tt, force=False)
            else:
                # Vocab-sharded LM head: avoid all-gathering full logits. Compute per-device
                # max+argmax and reduce on host (only a few scalars per token).
                mesh_rows, mesh_cols = int(self.device.shape[0]), int(self.device.shape[1])
                tp_axis = self.lm_head_tp_axis
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

                vocab_per_shard = int(self.lm_head_vocab_per_shard)
                logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
                max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
                if isinstance(max_out, tuple):
                    local_max_tt, local_argmax_tt = max_out
                else:
                    local_max_tt = max_out
                    local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

                local_argmax_torch = _mesh_to_torch_selected(tensor=local_argmax_tt, device_ids=selected_device_ids)
                local_max_torch = _mesh_to_torch_selected(tensor=local_max_tt, device_ids=selected_device_ids)
                ttnn.deallocate(local_argmax_tt, force=False)
                ttnn.deallocate(local_max_tt, force=False)
                ttnn.deallocate(logits_rm, force=False)
                ttnn.deallocate(logits_tt, force=False)

                next_ids = torch.empty((active,), dtype=torch.int32)
                for b in range(active):
                    best_val = None
                    best_global = None
                    for shard_idx, max_tensor, argmax_tensor in zip(
                        shard_indices, local_max_torch, local_argmax_torch
                    ):
                        max_val = float(max_tensor.reshape(-1)[b].item())
                        local_idx = int(argmax_tensor.reshape(-1)[b].item())
                        global_idx = int(shard_idx * vocab_per_shard + local_idx)
                        if global_idx >= vocab:
                            continue
                        if best_val is None or max_val > best_val:
                            best_val = max_val
                            best_global = global_idx
                    if best_global is None:
                        best_global = max(0, vocab - 1)
                    next_ids[b] = int(best_global)
                next_ids_flat = next_ids

            ttnn.deallocate(x, force=False)
            ttnn.deallocate(tt_positions, force=False)
            ttnn.deallocate(cos_batch, force=False)
            ttnn.deallocate(sin_batch, force=False)
            ttnn.deallocate(page_table_tt, force=False)
            if profile_on:
                decode_profile["total_s"] = decode_profile.get("total_s", 0.0) + (time.perf_counter() - t_decode0)
                self._profile_record(
                    phase="decode",
                    stage_totals=decode_profile,
                    token_count=active,
                    layer_filter=profile_layer,
                )
            return next_ids_flat

        vocab = int(self.hparams.vocab_size)
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            # Avoid device all_gather to compose logits.
            #
            # We've observed device-side all_gather to hang during bring-up for
            # some mesh configurations. For correctness-first operation (and for
            # prefill decode-loop fallbacks), it is acceptable to read each vocab
            # shard to the host and concatenate.
            shards = ttnn.get_device_tensors(logits_tt)
            if not shards:
                raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
            logits_shards = [ttnn.to_torch(t)[..., : int(t.shape[-1])] for t in shards]
            logits_full = torch.cat(logits_shards, dim=-1)[..., :vocab]
            logits_flat = logits_full.reshape(-1, vocab)
        else:
            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
            logits_torch = logits_torch[..., :vocab]
            logits_flat = logits_torch.reshape(-1, vocab)

        if logits_flat.shape[0] != active:
            raise RuntimeError(
                f"decode logits shape mismatch: expected {active} rows, got {int(logits_flat.shape[0])} "
                f"(logits_flat.shape={tuple(logits_flat.shape)})"
            )
        logits = logits_flat.reshape(active, 1, vocab).to(dtype=torch.float32).cpu()

        # Cleanup.
        ttnn.deallocate(logits_tt, force=False)
        ttnn.deallocate(x, force=False)
        ttnn.deallocate(tt_positions, force=False)
        ttnn.deallocate(cos_batch, force=False)
        ttnn.deallocate(sin_batch, force=False)
        ttnn.deallocate(page_table_tt, force=False)
        if profile_on:
            decode_profile["total_s"] = decode_profile.get("total_s", 0.0) + (time.perf_counter() - t_decode0)
            self._profile_record(
                phase="decode",
                stage_totals=decode_profile,
                token_count=active,
                layer_filter=profile_layer,
            )

        return logits

    def _get_or_create_trace_state(self, *, batch: int, page_table_width: int) -> _DecodeTraceSamplingState:
        """Get or create a per-bucket decode trace state with persistent inputs."""
        batch = int(batch)
        page_table_width = int(page_table_width)
        if batch <= 0:
            raise ValueError("trace batch must be > 0")
        if page_table_width <= 0:
            raise ValueError("trace page_table_width must be > 0")

        state = self._decode_trace_states.get(batch)
        if (
            state is not None
            and state.tokens_tt is not None
            and state.positions_tt is not None
            and state.rot_idxs_tt is not None
            and state.cos_batch_tt is not None
            and state.sin_batch_tt is not None
            and state.trans_matrix_tt is not None
            and state.rope_sharded_mem_config is not None
            and state.page_table_tt is not None
            and int(state.page_table_width) == page_table_width
        ):
            return state

        # New bucket or page_table_width changed. Create fresh state.
        # Release old trace for this bucket if it exists.
        if state is not None and state.trace_id is not None:
            try:
                ttnn.synchronize_device(self.device)
            except Exception:
                pass
            try:
                ttnn.release_trace(self.device, state.trace_id)
            except Exception:
                pass
            state.trace_id = None
            for t in (state.logits_tt, state.top1_values_tt, state.top1_indices_tt):
                if t is not None:
                    try:
                        ttnn.deallocate(t, force=False)
                    except Exception:
                        pass
            state.logits_tt = None
            state.top1_values_tt = None
            state.top1_indices_tt = None

        # Free old persistent inputs if they exist.
        if state is not None:
            for t in (
                state.tokens_tt, state.positions_tt, state.rot_idxs_tt,
                state.cos_batch_tt, state.sin_batch_tt, state.trans_matrix_tt,
                state.page_table_tt,
            ):
                if t is not None:
                    try:
                        ttnn.deallocate(t, force=False)
                    except Exception:
                        pass

        state = _DecodeTraceSamplingState(batch=batch, page_table_width=page_table_width)

        is_mesh_device = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh_device else None

        state.tokens_tt = ttnn.from_torch(
            torch.zeros((batch, 1), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        state.positions_tt = ttnn.from_torch(
            torch.zeros((batch,), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        rope_dim = int(self.hparams.qk_rope_head_dim)
        padded_batch = ((batch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        state.rot_idxs_tt = ttnn.from_torch(
            torch.zeros((1, padded_batch), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        state.rot_idxs_padded_batch = int(padded_batch)

        grid_size = self.device.compute_with_storage_grid_size()
        user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)
        state.rope_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, int(rope_dim)),
            core_grid=user_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        state.cos_batch_tt = ttnn.from_torch(
            torch.zeros((1, batch, 1, rope_dim), dtype=torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=state.rope_sharded_mem_config,
            mesh_mapper=mapper,
        )
        state.sin_batch_tt = ttnn.from_torch(
            torch.zeros((1, batch, 1, rope_dim), dtype=torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=state.rope_sharded_mem_config,
            mesh_mapper=mapper,
        )
        trans_mat_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=user_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        trans_mat_torch = _rot_transformation_mat_torch().to(dtype=torch.bfloat16)
        trans_mat_torch = trans_mat_torch.repeat(1, 1, int(batch), 1).contiguous()
        state.trans_matrix_tt = ttnn.from_torch(
            trans_mat_torch,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=trans_mat_mem_config,
            mesh_mapper=mapper,
        )
        state.page_table_tt = ttnn.from_torch(
            torch.zeros((batch, page_table_width), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._decode_trace_states[batch] = state
        return state

    def _copy_decode_trace_inputs(
        self,
        *,
        state: _DecodeTraceSamplingState,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        """Copy host decode inputs into persistent device tensors for a bucket state."""
        if (
            state.tokens_tt is None
            or state.positions_tt is None
            or state.rot_idxs_tt is None
            or state.cos_batch_tt is None
            or state.sin_batch_tt is None
            or state.trans_matrix_tt is None
            or state.page_table_tt is None
        ):
            raise RuntimeError("trace inputs not allocated")
        batch = int(tokens.shape[0])
        if int(state.batch) != batch:
            raise RuntimeError(f"trace batch mismatch: allocated={int(state.batch)} got={batch}")
        if int(state.page_table_width) != int(page_table.shape[1]):
            raise RuntimeError(
                f"trace page_table_width mismatch: allocated={int(state.page_table_width)} got={int(page_table.shape[1])}"
            )

        # Debug-only: print the page table and positions at KV block boundaries.
        #
        # The most common correctness failure mode we've seen is a sudden
        # degradation to gibberish output exactly when the total sequence length
        # crosses the vLLM KV-cache `--block-size` boundary (currently 64).
        #
        # This log helps validate whether:
        # - vLLM is passing absolute positions (expected) vs modulo positions.
        # - page_table contains a valid second block id at pos==64.
        if os.environ.get("GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY", "").strip() == "1":
            try:
                if batch == 1 and start_pos.numel() >= 1 and tokens.numel() >= 1:
                    pos0 = int(start_pos[0].item())
                    # Log a small window around the first block boundary.
                    # This avoids relying on an exact convention (0-based vs 1-based),
                    # and makes it easy to see if positions wrap modulo block_size.
                    if 60 <= pos0 <= 70:
                        pt0 = page_table[0].to(torch.int32)
                        block = pos0 // 64
                        lo = max(0, block - 1)
                        hi = min(int(pt0.numel()), block + 2)
                        logger.info(
                            "GLM4 boundary: pos={} block={} token={} page_table_first8={} page_table_near={}",
                            pos0,
                            block,
                            int(tokens[0, 0].item()),
                            pt0[:8].tolist(),
                            pt0[lo:hi].tolist(),
                        )
            except Exception as e:
                logger.warning("GLM4_MOE_LITE_DEBUG_PAGE_TABLE_BOUNDARY failed: {}", e)

        is_mesh_device = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh_device else None

        host_tokens = ttnn.from_torch(
            tokens.to(torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_tokens, state.tokens_tt)

        host_pos = ttnn.from_torch(
            start_pos.to(torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pos, state.positions_tt)

        # Trace-mode RoPE: copy rot_idxs (positions) to device and generate cos/sin
        # *inside* the trace graph. Copying host-side RoPE slices directly into
        # sharded tensors is not layout-safe and can corrupt decode.
        pos_clamped = start_pos.to(torch.int32).clamp_min(0)
        pos_clamped = torch.clamp(pos_clamped, 0, max(0, int(self.max_seq_len) - 1)).to(torch.int32)
        padded_batch = int(state.rot_idxs_padded_batch)
        if padded_batch < batch:
            raise RuntimeError(f"trace rot_idxs padded batch too small: padded={padded_batch} batch={batch}")
        if padded_batch != batch:
            rot_idxs_padded = torch.nn.functional.pad(
                pos_clamped.view(1, batch),
                (0, padded_batch - batch),
                "constant",
                0,
            )
        else:
            rot_idxs_padded = pos_clamped.view(1, batch)
        host_rot_idxs = ttnn.from_torch(
            rot_idxs_padded,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_rot_idxs, state.rot_idxs_tt)

        host_pt = ttnn.from_torch(
            page_table.to(torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pt, state.page_table_tt)

    def _decode_step_tt_logits(self, *, state: _DecodeTraceSamplingState, kv_cache: list[ttnn.Tensor]) -> ttnn.Tensor:
        """Decode step using persistent TT inputs from a bucket state, producing device logits."""
        assert state.tokens_tt is not None
        assert state.positions_tt is not None
        assert state.rot_idxs_tt is not None
        assert state.cos_batch_tt is not None
        assert state.sin_batch_tt is not None
        assert state.trans_matrix_tt is not None
        assert state.rope_sharded_mem_config is not None
        assert state.page_table_tt is not None

        batch = int(state.batch)

        # Generate RoPE cos/sin inside the trace from rot_idxs, then shard into
        # the decode layout expected by rotary_embedding_llama(is_decode_mode=True).
        rope_dim = int(self.hparams.qk_rope_head_dim)
        padded_batch = int(state.rot_idxs_padded_batch)
        cos_rows = ttnn.embedding(state.rot_idxs_tt, self.rope["cos_matrix"], layout=ttnn.TILE_LAYOUT)
        sin_rows = ttnn.embedding(state.rot_idxs_tt, self.rope["sin_matrix"], layout=ttnn.TILE_LAYOUT)
        cos_batch_view = ttnn.unsqueeze_to_4D(cos_rows)  # [1,1,B_pad,D]
        sin_batch_view = ttnn.unsqueeze_to_4D(sin_rows)  # [1,1,B_pad,D]
        if padded_batch != batch:
            cos_batch_view = ttnn.slice(cos_batch_view, [0, 0, 0, 0], [1, 1, batch, rope_dim])
            sin_batch_view = ttnn.slice(sin_batch_view, [0, 0, 0, 0], [1, 1, batch, rope_dim])
        cos_batch_rm = ttnn.clone(cos_batch_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_batch_rm = ttnn.clone(sin_batch_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cos_decode = ttnn.transpose(cos_batch_rm, 1, 2)  # [1,B,1,D]
        sin_decode = ttnn.transpose(sin_batch_rm, 1, 2)  # [1,B,1,D]
        cos_decode_sharded = ttnn.interleaved_to_sharded(cos_decode, state.rope_sharded_mem_config)
        sin_decode_sharded = ttnn.interleaved_to_sharded(sin_decode, state.rope_sharded_mem_config)
        ttnn.copy(cos_decode_sharded, state.cos_batch_tt)
        ttnn.copy(sin_decode_sharded, state.sin_batch_tt)
        cos_batch = state.cos_batch_tt
        sin_batch = state.sin_batch_tt
        trans_matrix = state.trans_matrix_tt
        hidden = int(self.hparams.hidden_size)
        # Match DeepSeek trace pattern: embed tokens *inside* the trace graph.
        # This avoids device allocations outside trace replay and keeps input
        # staging limited to host->device copies.
        x = ttnn.embedding(
            state.tokens_tt,
            self.embed_w,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, batch, 1, hidden))
        x = ttnn.permute(x, (0, 2, 1, 3))  # [1,1,B,D]
        x = ttnn.slice(x, [0, 0, 0, 0], [1, 1, batch, hidden])

        for layer_idx in range(self.num_layers_to_run):
            w = self._ensure_layer_weights(layer_idx)
            x_next = run_decoder_layer_decode_one_step_update_cache_tt(
                device=self.device,
                x_embed_tok=x,
                tt_positions=state.positions_tt,
                page_table_tt=state.page_table_tt,
                kvpe_cache=kv_cache[layer_idx],
                cos_batch=cos_batch,
                sin_batch=sin_batch,
                trans_matrix=trans_matrix,
                cos_decode=cos_batch,
                sin_decode=sin_batch,
                trans_decode=trans_matrix,
                rope_sharded_cfg=state.rope_sharded_mem_config,
                w=w,
                hparams=self.hparams,
                moe_runtime=self.moe_runtime,
                profile=None,
                use_decode_rope=True,
            )
            ttnn.deallocate(x, force=False)
            x = x_next

        x = self.final_norm(x, mode="decode")
        logits_tt = ttnn.linear(x, self.lm_head_w)  # [1,1,B,vocab_shard?]
        ttnn.deallocate(x, force=False)

        return logits_tt

    def _capture_decode_trace_sampling(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> _DecodeTraceSamplingState:
        batch = int(tokens.shape[0])
        page_table_width = int(page_table.shape[1])
        state = self._get_or_create_trace_state(batch=batch, page_table_width=page_table_width)

        # Warm-up compile run (no trace) to keep compilation out of capture.
        _ = self.decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=None,
            enable_trace=False,
        )
        ttnn.synchronize_device(self.device)

        self._copy_decode_trace_inputs(state=state, tokens=tokens, start_pos=start_pos, page_table=page_table)
        ttnn.synchronize_device(self.device)

        # Warm-up the trace path itself (not captured) so ops like `ttnn.embedding`
        # do any one-time compilation/program uploads outside trace capture.
        logits_warm = self._decode_step_tt_logits(state=state, kv_cache=kv_cache)
        # Warm-up any sampling ops we plan to run inside the trace. After trace
        # capture, allocating device buffers becomes unsafe, so sampling must be
        # fully trace-contained.
        logits_rm_warm = ttnn.to_layout(logits_warm, ttnn.ROW_MAJOR_LAYOUT)
        vocab = int(self.hparams.vocab_size)
        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            logits_rm_warm_view = ttnn.slice(logits_rm_warm, [0, 0, 0, 0], [1, 1, batch, vocab])
            next_ids_warm = ttnn.argmax(logits_rm_warm_view, dim=3, keepdim=False, use_multicore=True)
            ttnn.deallocate(next_ids_warm, force=False)
        else:
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_warm_view = ttnn.slice(logits_rm_warm, [0, 0, 0, 0], [1, 1, batch, vocab_per_shard])
            max_out_warm = ttnn.max(logits_rm_warm_view, dim=3, keepdim=True)
            if isinstance(max_out_warm, tuple):
                local_max_warm, local_argmax_warm = max_out_warm
                ttnn.deallocate(local_max_warm, force=False)
                ttnn.deallocate(local_argmax_warm, force=False)
            else:
                local_max_warm = max_out_warm
                local_argmax_warm = ttnn.argmax(logits_rm_warm_view, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(local_max_warm, force=False)
                ttnn.deallocate(local_argmax_warm, force=False)
        ttnn.deallocate(logits_rm_warm, force=False)
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(logits_warm, force=False)

        # Re-copy inputs since the warm-up decode step updated KV cache and may
        # have consumed the previous values.
        self._copy_decode_trace_inputs(state=state, tokens=tokens, start_pos=start_pos, page_table=page_table)
        ttnn.synchronize_device(self.device)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        logits_tt = self._decode_step_tt_logits(state=state, kv_cache=kv_cache)
        # Capture greedy sampling inside the trace to avoid allocating any
        # device buffers while an active trace exists.
        logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
        vocab = int(self.hparams.vocab_size)
        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            top1_values_tt = None
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, batch, vocab])
            top1_indices_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)
        else:
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, batch, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                top1_values_tt, top1_indices_tt = max_out
            else:
                top1_values_tt = max_out
                top1_indices_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        state.trace_id = trace_id
        state.logits_tt = logits_tt
        state.top1_values_tt = top1_values_tt
        state.top1_indices_tt = top1_indices_tt
        return state

    def _decode_trace_sampling(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        batch = int(tokens.shape[0])
        page_table_width = int(page_table.shape[1])
        state = self._get_or_create_trace_state(batch=batch, page_table_width=page_table_width)
        if state.trace_id is None:
            state = self._capture_decode_trace_sampling(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)
        assert state.trace_id is not None
        assert state.logits_tt is not None
        assert state.top1_indices_tt is not None

        self._copy_decode_trace_inputs(state=state, tokens=tokens, start_pos=start_pos, page_table=page_table)
        if os.environ.get("GLM4_MOE_LITE_SYNC_BEFORE_TRACE", "").strip() == "1":
            ttnn.synchronize_device(self.device)
        ttnn.execute_trace(self.device, state.trace_id, cq_id=0, blocking=True)

        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            return state.top1_indices_tt

        assert state.top1_values_tt is not None
        return (state.top1_values_tt, state.top1_indices_tt)

    def _decode_trace_logits(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> torch.Tensor:
        """Execute the decode trace and return logits on host [B,1,vocab] float32."""
        batch = int(tokens.shape[0])
        page_table_width = int(page_table.shape[1])
        state = self._get_or_create_trace_state(batch=batch, page_table_width=page_table_width)
        if state.trace_id is None:
            state = self._capture_decode_trace_sampling(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)
        assert state.trace_id is not None
        assert state.logits_tt is not None

        self._copy_decode_trace_inputs(state=state, tokens=tokens, start_pos=start_pos, page_table=page_table)
        if os.environ.get("GLM4_MOE_LITE_SYNC_BEFORE_TRACE", "").strip() == "1":
            ttnn.synchronize_device(self.device)
        ttnn.execute_trace(self.device, state.trace_id, cq_id=0, blocking=True)

        logits_tt = state.logits_tt
        vocab = int(self.hparams.vocab_size)

        if not _is_mesh_device(self.device) or not self.lm_head_sharded_vocab:
            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)[..., :vocab]
            logits_flat = logits_torch.reshape(-1, vocab)
            return logits_flat.reshape(batch, 1, vocab).to(dtype=torch.float32).cpu()

        shards = ttnn.get_device_tensors(logits_tt)
        if not shards:
            raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
        logits_shards = [ttnn.to_torch(t)[..., : int(t.shape[-1])] for t in shards]
        logits_full = torch.cat(logits_shards, dim=-1)[..., :vocab]
        logits_flat = logits_full.reshape(-1, vocab)
        return logits_flat.reshape(batch, 1, vocab).to(dtype=torch.float32).cpu()


__all__ = ["Glm4MoeLiteDenseOnlyTT", "_torch_dtype_to_ttnn"]
