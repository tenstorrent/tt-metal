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
        return ttnn.to_torch(tensor)
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    return ttnn.to_torch(device_tensors[0])


def _mesh_to_torch_selected(*, tensor: ttnn.Tensor, device_ids: list[int]) -> list[torch.Tensor]:
    """Convert a subset of device tensors from a mesh tensor into torch tensors."""
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    out: list[torch.Tensor] = []
    for device_id in device_ids:
        if device_id < 0 or device_id >= len(device_tensors):
            raise IndexError(f"device_id {device_id} out of range for mesh tensor with {len(device_tensors)} devices")
        out.append(ttnn.to_torch(device_tensors[device_id]))
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
    # ---- Decode trace state (vLLM trace_mode=all) ----
    # We start with the on-device greedy sampling path, which is what vLLM uses
    # in `sample_on_device_mode=decode_only`. Trace capture/replay removes the
    # per-op host overhead for 47-layer decode.
    _decode_trace_id_sampling: Any | None = field(init=False, default=None)
    _trace_tokens_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_x_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_positions_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_rot_idxs_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_cos_batch_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_sin_batch_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_trans_matrix_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_rope_sharded_mem_config: Any | None = field(init=False, default=None)
    _trace_rot_idxs_padded_batch: int = field(init=False, default=0)
    _trace_page_table_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_logits_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_top1_values_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_top1_indices_tt: ttnn.Tensor | None = field(init=False, default=None)
    _trace_batch: int = field(init=False, default=0)
    _trace_page_table_width: int = field(init=False, default=0)

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
        if tokens.ndim != 2:
            raise ValueError(f"expected tokens [B,S], got {tuple(tokens.shape)}")
        if page_table.ndim != 2:
            raise ValueError(f"expected page_table [B,W], got {tuple(page_table.shape)}")
        batch, seq_total = tokens.shape
        if len(prompt_lens) != int(batch):
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {int(batch)}")

        vocab = int(self.hparams.vocab_size)
        hidden = int(self.hparams.hidden_size)
        rope_dim = int(self.hparams.qk_rope_head_dim)
        pad_multiple = max(1, int(seq_pad_multiple))

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
                ttnn.deallocate(x)
                x = x_next

            # Logits for the last *real* prompt token only.
            x_last = ttnn.slice(x, [0, 0, prompt_len - 1, 0], [1, 1, prompt_len, hidden])
            ttnn.deallocate(x)

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
                ttnn.deallocate(logits_tt)
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
            ttnn.deallocate(logits_tt)
            ttnn.deallocate(x_last)
            if rope_slices_owned:
                ttnn.deallocate(cos_matrix)
                ttnn.deallocate(sin_matrix)
            ttnn.deallocate(page_table_tt)

        if profile_on and profile_token_count > 0:
            prefill_profile["total_s"] = prefill_profile.get("total_s", 0.0) + (time.perf_counter() - t_prefill0)
            self._profile_record(
                phase="prefill",
                stage_totals=prefill_profile,
                token_count=profile_token_count,
                layer_filter=profile_layer,
            )
        return torch.stack(out_logits, dim=0).unsqueeze(1)  # [B,1,vocab]

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
    ) -> torch.Tensor:
        """Run a decode step, updating KV cache.

        Returns:
        - logits on host as torch float32 [active, 1, vocab] when sampling_params is None
        - next token ids on host as torch int32 [active] when sampling_params is not None
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

        tokens = tokens[:active].to(torch.int32)
        positions = start_pos[:active].to(torch.int32)
        page_table = page_table[:active].to(torch.int32)

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
        if profile_on:
            decode_profile["prep_inputs_s"] = decode_profile.get("prep_inputs_s", 0.0) + (time.perf_counter() - t0)

        t0 = time.perf_counter() if profile_on else 0.0
        x = run_tt_embedding(device=self.device, token_ids=tokens, tt_weight=self.embed_w)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (1, active, 1, int(self.hparams.hidden_size)))
        x = ttnn.permute(x, (0, 2, 1, 3))  # [1,1,B,D]
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
                w=w,
                hparams=self.hparams,
                moe_runtime=self.moe_runtime,
                profile=layer_profile,
            )
            if layer_profile is not None:
                for key, value in layer_profile.items():
                    stage_key = f"layer_{key}"
                    decode_profile[stage_key] = decode_profile.get(stage_key, 0.0) + float(value)
            ttnn.deallocate(x)
            x = x_next

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
            ttnn.deallocate(logits_tt)

            if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
                next_ids_tt = ttnn.argmax(logits_rm, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(logits_rm)

                next_ids_torch = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
                next_ids_flat = next_ids_torch.reshape(-1).to(dtype=torch.int32).cpu()
                if int(next_ids_flat.numel()) != active:
                    raise RuntimeError(
                        f"decode token ids shape mismatch: expected {active} values, got {int(next_ids_flat.numel())} "
                        f"(next_ids_torch.shape={tuple(next_ids_torch.shape)})"
                    )
                ttnn.deallocate(next_ids_tt)
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

                max_out = ttnn.max(logits_rm, dim=3, keepdim=True)
                if isinstance(max_out, tuple):
                    local_max_tt, local_argmax_tt = max_out
                else:
                    local_max_tt = max_out
                    local_argmax_tt = ttnn.argmax(logits_rm, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(logits_rm)

                local_argmax_torch = _mesh_to_torch_selected(tensor=local_argmax_tt, device_ids=selected_device_ids)
                local_max_torch = _mesh_to_torch_selected(tensor=local_max_tt, device_ids=selected_device_ids)
                ttnn.deallocate(local_argmax_tt)
                ttnn.deallocate(local_max_tt)

                vocab_per_shard = int(self.lm_head_vocab_per_shard)
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
                        if best_val is None or max_val > best_val:
                            best_val = max_val
                            best_global = global_idx
                    assert best_global is not None
                    next_ids[b] = int(best_global)
                next_ids_flat = next_ids

            ttnn.deallocate(x)
            ttnn.deallocate(tt_positions)
            ttnn.deallocate(cos_batch)
            ttnn.deallocate(sin_batch)
            ttnn.deallocate(page_table_tt)
            if profile_on:
                decode_profile["total_s"] = decode_profile.get("total_s", 0.0) + (time.perf_counter() - t_decode0)
                self._profile_record(
                    phase="decode",
                    stage_totals=decode_profile,
                    token_count=active,
                    layer_filter=profile_layer,
                )
            return next_ids_flat

        # On multi-device meshes, logits_tt is often distributed. Converting to torch
        # requires composing shards/replicas; otherwise ttnn will error when trying
        # to convert a multi-buffer host tensor into a single row-major buffer.
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
            ttnn.deallocate(logits_tt)
            logits_tt = logits_tt_full
        logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)

        # Some distributed topologies may compose replicas by concatenation; slice
        # down to vocab_size to match vLLM expectations.
        logits_torch = logits_torch[..., : int(self.hparams.vocab_size)]
        vocab = int(self.hparams.vocab_size)
        logits_flat = logits_torch.reshape(-1, vocab)
        if logits_flat.shape[0] != active:
            raise RuntimeError(
                f"decode logits shape mismatch: expected {active} rows, got {int(logits_flat.shape[0])} "
                f"(logits_torch.shape={tuple(logits_torch.shape)})"
            )
        logits = logits_flat.reshape(active, 1, vocab).to(dtype=torch.float32).cpu()

        # Cleanup.
        ttnn.deallocate(logits_tt)
        ttnn.deallocate(x)
        ttnn.deallocate(tt_positions)
        ttnn.deallocate(cos_batch)
        ttnn.deallocate(sin_batch)
        ttnn.deallocate(page_table_tt)
        if profile_on:
            decode_profile["total_s"] = decode_profile.get("total_s", 0.0) + (time.perf_counter() - t_decode0)
            self._profile_record(
                phase="decode",
                stage_totals=decode_profile,
                token_count=active,
                layer_filter=profile_layer,
            )

        return logits

    def _ensure_decode_trace_inputs(self, *, batch: int, page_table_width: int) -> None:
        """Allocate persistent decode inputs for trace capture/replay."""
        batch = int(batch)
        page_table_width = int(page_table_width)
        if batch <= 0:
            raise ValueError("trace batch must be > 0")
        if page_table_width <= 0:
            raise ValueError("trace page_table_width must be > 0")

        if (
            self._trace_tokens_tt is not None
            and self._trace_x_tt is not None
            and self._trace_positions_tt is not None
            and self._trace_rot_idxs_tt is not None
            and self._trace_cos_batch_tt is not None
            and self._trace_sin_batch_tt is not None
            and self._trace_trans_matrix_tt is not None
            and self._trace_rope_sharded_mem_config is not None
            and self._trace_page_table_tt is not None
            and int(self._trace_batch) == batch
            and int(self._trace_page_table_width) == page_table_width
        ):
            return

        # Shapes changed. Drop any previous trace.
        self._decode_trace_id_sampling = None
        self._trace_logits_tt = None
        self._trace_top1_values_tt = None
        self._trace_top1_indices_tt = None

        is_mesh_device = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh_device else None

        self._trace_tokens_tt = ttnn.from_torch(
            torch.zeros((batch, 1), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        hidden = int(self.hparams.hidden_size)
        self._trace_x_tt = ttnn.from_torch(
            torch.zeros((1, 1, batch, hidden), dtype=torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._trace_positions_tt = ttnn.from_torch(
            torch.zeros((batch,), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        rope_dim = int(self.hparams.qk_rope_head_dim)
        padded_batch = ((batch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self._trace_rot_idxs_tt = ttnn.from_torch(
            torch.zeros((1, padded_batch), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._trace_rot_idxs_padded_batch = int(padded_batch)

        grid_size = self.device.compute_with_storage_grid_size()
        user_grid = ttnn.num_cores_to_corerangeset(int(batch), grid_size, row_wise=True)
        self._trace_rope_sharded_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, int(rope_dim)),
            core_grid=user_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self._trace_cos_batch_tt = ttnn.from_torch(
            torch.zeros((1, batch, 1, rope_dim), dtype=torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._trace_rope_sharded_mem_config,
            mesh_mapper=mapper,
        )
        self._trace_sin_batch_tt = ttnn.from_torch(
            torch.zeros((1, batch, 1, rope_dim), dtype=torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._trace_rope_sharded_mem_config,
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
        self._trace_trans_matrix_tt = ttnn.from_torch(
            trans_mat_torch,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=trans_mat_mem_config,
            mesh_mapper=mapper,
        )
        self._trace_page_table_tt = ttnn.from_torch(
            torch.zeros((batch, page_table_width), dtype=torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        self._trace_batch = batch
        self._trace_page_table_width = page_table_width

    def _copy_decode_trace_inputs(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        """Copy host decode inputs into persistent device tensors."""
        if (
            self._trace_tokens_tt is None
            or self._trace_x_tt is None
            or self._trace_positions_tt is None
            or self._trace_rot_idxs_tt is None
            or self._trace_cos_batch_tt is None
            or self._trace_sin_batch_tt is None
            or self._trace_trans_matrix_tt is None
            or self._trace_page_table_tt is None
        ):
            raise RuntimeError("trace inputs not allocated")
        batch = int(tokens.shape[0])
        if int(self._trace_batch) != batch:
            raise RuntimeError(f"trace batch mismatch: allocated={int(self._trace_batch)} got={batch}")
        if int(self._trace_page_table_width) != int(page_table.shape[1]):
            raise RuntimeError(
                f"trace page_table_width mismatch: allocated={int(self._trace_page_table_width)} got={int(page_table.shape[1])}"
            )

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
        ttnn.copy_host_to_device_tensor(host_tokens, self._trace_tokens_tt)

        # Build and copy embedded token input for the trace.
        x = ttnn.embedding(
            self._trace_tokens_tt,
            self.embed_w,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        batch = int(tokens.shape[0])
        hidden = int(self.hparams.hidden_size)
        x = ttnn.reshape(x, (1, batch, 1, hidden))
        x = ttnn.permute(x, (0, 2, 1, 3))  # [1,1,B,D]
        ttnn.copy(x, self._trace_x_tt)
        ttnn.deallocate(x)

        host_pos = ttnn.from_torch(
            start_pos.to(torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pos, self._trace_positions_tt)

        # Trace-mode RoPE: copy small per-step cos/sin slices from host into
        # persistent sharded tensors. This avoids any host writes during trace
        # capture (e.g. `ttnn.embedding`) on mesh.
        pos_clamped = start_pos.to(torch.int64).clamp_min(0)
        pos_clamped = torch.clamp(pos_clamped, 0, max(0, int(self.max_seq_len) - 1)).to(torch.int64)
        rope_dim = int(self.hparams.qk_rope_head_dim)
        cos_rows = self.rope["cos_matrix_host"][0, 0, pos_clamped, :rope_dim].to(dtype=torch.bfloat16)
        sin_rows = self.rope["sin_matrix_host"][0, 0, pos_clamped, :rope_dim].to(dtype=torch.bfloat16)
        cos_batch = cos_rows.unsqueeze(0).unsqueeze(2).contiguous()  # [1,B,1,D]
        sin_batch = sin_rows.unsqueeze(0).unsqueeze(2).contiguous()  # [1,B,1,D]

        host_cos = ttnn.from_torch(
            cos_batch,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        host_sin = ttnn.from_torch(
            sin_batch,
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_cos, self._trace_cos_batch_tt)
        ttnn.copy_host_to_device_tensor(host_sin, self._trace_sin_batch_tt)

        host_pt = ttnn.from_torch(
            page_table.to(torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pt, self._trace_page_table_tt)

    def _decode_step_tt_logits(self, *, kv_cache: list[ttnn.Tensor]) -> ttnn.Tensor:
        """Decode step using persistent TT inputs, producing device logits."""
        assert self._trace_tokens_tt is not None
        assert self._trace_x_tt is not None
        assert self._trace_positions_tt is not None
        assert self._trace_cos_batch_tt is not None
        assert self._trace_sin_batch_tt is not None
        assert self._trace_trans_matrix_tt is not None
        assert self._trace_page_table_tt is not None

        batch = int(self._trace_batch)

        cos_batch = self._trace_cos_batch_tt
        sin_batch = self._trace_sin_batch_tt
        trans_matrix = self._trace_trans_matrix_tt
        x = self._trace_x_tt

        for layer_idx in range(self.num_layers_to_run):
            w = self._ensure_layer_weights(layer_idx)
            x_next = run_decoder_layer_decode_one_step_update_cache_tt(
                device=self.device,
                x_embed_tok=x,
                tt_positions=self._trace_positions_tt,
                page_table_tt=self._trace_page_table_tt,
                kvpe_cache=kv_cache[layer_idx],
                cos_batch=cos_batch,
                sin_batch=sin_batch,
                trans_matrix=trans_matrix,
                w=w,
                hparams=self.hparams,
                moe_runtime=self.moe_runtime,
                profile=None,
                use_decode_rope=True,
            )
            if x is not self._trace_x_tt:
                ttnn.deallocate(x)
            x = x_next

        x = self.final_norm(x, mode="decode")
        logits_tt = ttnn.linear(x, self.lm_head_w)  # [1,1,B,vocab_shard?]
        ttnn.deallocate(x)

        return logits_tt

    def _capture_decode_trace_sampling(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> None:
        batch = int(tokens.shape[0])
        page_table_width = int(page_table.shape[1])
        self._ensure_decode_trace_inputs(batch=batch, page_table_width=page_table_width)

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

        self._copy_decode_trace_inputs(tokens=tokens, start_pos=start_pos, page_table=page_table)
        ttnn.synchronize_device(self.device)

        # Warm-up the trace path itself (not captured) so ops like `ttnn.embedding`
        # do any one-time compilation/program uploads outside trace capture.
        logits_warm = self._decode_step_tt_logits(kv_cache=kv_cache)
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(logits_warm)

        # Re-copy inputs since the warm-up decode step updated KV cache and may
        # have consumed the previous values.
        self._copy_decode_trace_inputs(tokens=tokens, start_pos=start_pos, page_table=page_table)
        ttnn.synchronize_device(self.device)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        logits_tt = self._decode_step_tt_logits(kv_cache=kv_cache)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

        self._decode_trace_id_sampling = trace_id
        self._trace_logits_tt = logits_tt
        self._trace_top1_values_tt = None
        self._trace_top1_indices_tt = None

    def _decode_trace_sampling(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> torch.Tensor:
        if self._decode_trace_id_sampling is None:
            self._capture_decode_trace_sampling(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)
        assert self._decode_trace_id_sampling is not None
        assert self._trace_logits_tt is not None

        self._copy_decode_trace_inputs(tokens=tokens, start_pos=start_pos, page_table=page_table)
        ttnn.execute_trace(self.device, self._decode_trace_id_sampling, cq_id=0, blocking=True)

        batch = int(tokens.shape[0])
        vocab = int(self.hparams.vocab_size)

        # Greedy sampling outside the trace. This avoids any uint32 layout
        # conversions or top-k index writes during trace capture.
        logits_rm = ttnn.to_layout(self._trace_logits_tt, ttnn.ROW_MAJOR_LAYOUT)

        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            next_ids_tt = ttnn.argmax(logits_rm, dim=3, keepdim=False, use_multicore=True)
            ttnn.deallocate(logits_rm)

            next_ids_torch = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids_flat = next_ids_torch.reshape(-1).to(dtype=torch.int64).cpu()
            ttnn.deallocate(next_ids_tt)

            next_ids_flat = torch.clamp(next_ids_flat, 0, max(0, vocab - 1))
            return next_ids_flat.to(dtype=torch.int32)[:batch]

        # Vocab-sharded LM head: compute per-shard max+argmax and reduce on host.
        tp_size = int(self.device.shape[0]) * int(self.device.shape[1])
        selected_device_ids = list(range(tp_size))
        shard_indices = list(range(tp_size))

        max_out = ttnn.max(logits_rm, dim=3, keepdim=True)
        if isinstance(max_out, tuple):
            local_max_tt, local_argmax_tt = max_out
        else:
            local_max_tt = max_out
            local_argmax_tt = ttnn.argmax(logits_rm, dim=3, keepdim=False, use_multicore=True)
        ttnn.deallocate(logits_rm)

        local_argmax_torch = _mesh_to_torch_selected(tensor=local_argmax_tt, device_ids=selected_device_ids)
        local_max_torch = _mesh_to_torch_selected(tensor=local_max_tt, device_ids=selected_device_ids)
        ttnn.deallocate(local_argmax_tt)
        ttnn.deallocate(local_max_tt)

        vocab_per_shard = int(self.lm_head_vocab_per_shard)
        next_ids = torch.empty((batch,), dtype=torch.int32)
        for b in range(batch):
            best_val = None
            best_global = None
            for shard_idx, (val_tensor, idx_tensor) in enumerate(zip(local_max_torch, local_argmax_torch)):
                max_val = float(val_tensor.reshape(-1)[b].item())
                local_idx = int(idx_tensor.reshape(-1)[b].item())
                global_idx = int(shard_idx * vocab_per_shard + local_idx)
                if global_idx >= vocab:
                    continue
                if best_val is None or max_val > best_val:
                    best_val = max_val
                    best_global = global_idx
            if best_global is None:
                best_global = max(0, vocab - 1)
            next_ids[b] = int(best_global)
        return next_ids

    def _decode_trace_logits(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[ttnn.Tensor],
    ) -> torch.Tensor:
        """Execute the decode trace and return logits on host [B,1,vocab] float32.

        This path avoids any post-trace device allocations by composing sharded
        logits on the host (no device all-gather).
        """
        if self._decode_trace_id_sampling is None:
            self._capture_decode_trace_sampling(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)
        assert self._decode_trace_id_sampling is not None
        assert self._trace_logits_tt is not None

        self._copy_decode_trace_inputs(tokens=tokens, start_pos=start_pos, page_table=page_table)
        ttnn.execute_trace(self.device, self._decode_trace_id_sampling, cq_id=0, blocking=True)

        logits_tt = self._trace_logits_tt
        vocab = int(self.hparams.vocab_size)
        batch = int(tokens.shape[0])

        if not _is_mesh_device(self.device) or not self.lm_head_sharded_vocab:
            # Bring-up contract: in mesh-replicated mode, read back device 0.
            # (ttnn.to_torch(mesh_tensor) requires an explicit mesh composer.)
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
