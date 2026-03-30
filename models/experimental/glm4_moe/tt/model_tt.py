# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Top-level model runner for GLM-4.7-REAP-218B on TT hardware.

Provides Glm4MoeTT: a frozen dataclass with create() factory that loads all 92 layers
of weights, creates decoder layers, embedding, final norm, and LM head, and exposes
decode() and prefill() entry points.

Key architecture:
- 92 layers (num_hidden_layers=92)
- Layers 0-2: dense MLP (first_k_dense_replace=3)
- Layers 3-91: MoE (96 routed experts EP=32 + 1 shared expert TP=8)
- Standard GQA attention (96Q/8KV heads, head_dim=128, NOT MLA)
- Galaxy Wormhole: Mesh(8,4), TP=8 (axis 0), EP=32, DP=4 (axis 1)
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

import ttnn
from loguru import logger

from models.common.rmsnorm import RMSNorm
from models.experimental.glm4_moe.tt.config import Glm4MoeHParams
from models.experimental.glm4_moe.tt.decoder_layer_tt import Glm4MoeDecoderLayer, _sharded_rms_norm
from models.experimental.glm4_moe.tt.layer_weights import (
    DecoderLayerTTWeights,
    _env_dense_dtype,
    _tp_axis_and_size,
    _tp_mesh_mapper,
    _linear_weight_tt,
    convert_decoder_layer_weights,
)
from models.experimental.glm4_moe.tt.attention_tt import _simple_all_gather
from models.experimental.glm4_moe.tt.ccl import CCL
from models.experimental.glm4_moe.tt.moe_tt import create_moe_runtime, Glm4MoeMoERuntime
from models.experimental.glm4_moe.tt.tt_embedding import (
    convert_embedding_weight_to_tt,
)
from models.experimental.glm4_moe.tt.weights import (
    load_glm_lazy_state_dict,
)


# ---------------------------------------------------------------------------
# RoPE utilities (adapted from glm4_moe_lite/layer0_tt.py for GQA head_dim)
# ---------------------------------------------------------------------------


def _rot_transformation_mat_torch() -> torch.Tensor:
    """Transformation matrix for ttnn.experimental.rotary_embedding_llama."""
    dhead = 32
    rot = torch.zeros(1, 1, dhead, dhead, dtype=torch.float32)
    rot[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1.0
    rot[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1.0
    return rot


def _rope_cos_sin_torch(*, seq_len: int, dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin matrices in NeoX half-rotation form: [1,1,seq_len,dim].

    Last dim layout: [t0, t1, ..., t_{d/2-1}, t0, t1, ..., t_{d/2-1}]
    to match NeoX-style rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2]).
    GLM-4.7 uses NeoX-style RoPE (confirmed from HuggingFace transformers glm4_moe).
    """
    if dim % 2 != 0:
        raise ValueError(f"rope dim must be even, got dim={dim}")
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * (2.0 / dim)))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _make_rope_tensors(
    *,
    device,
    seq_len: int,
    rope_dim: int,
    rope_theta: float,
) -> dict[str, object]:
    """Create RoPE cos/sin/trans tensors for GQA with partial RoPE.

    For GLM-4.7-REAP: partial_rotary_factor=0.5, so rope_dim=64 (half of head_dim=128).
    RoPE tables are created at rope_dim (the portion that gets rotated).
    """
    cos_t, sin_t = _rope_cos_sin_torch(seq_len=seq_len, dim=rope_dim, base=rope_theta)
    trans_t = _rot_transformation_mat_torch().to(dtype=torch.bfloat16)

    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    cos_host = cos_t.to(dtype=torch.bfloat16).cpu()
    sin_host = sin_t.to(dtype=torch.bfloat16).cpu()

    cos = ttnn.from_torch(
        cos_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    sin = ttnn.from_torch(
        sin_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    trans = ttnn.from_torch(
        trans_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    return {
        "cos_matrix": cos,
        "sin_matrix": sin,
        "trans_matrix": trans,
        "cos_matrix_host": cos_host,
        "sin_matrix_host": sin_host,
    }


# ---------------------------------------------------------------------------
# Decode RoPE input preparation (per-step cos/sin for batch of positions)
# ---------------------------------------------------------------------------


def _prepare_decode_rope_and_positions_tt(
    *,
    device,
    rope: dict,
    positions: torch.Tensor,
    dp_shard_axis: int | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prepare per-step RoPE cos/sin/sin_neg and position tensors for decode.

    positions: [B] int32 tensor of current positions.
    dp_shard_axis: If set, shard cos/sin along dim=1 (batch) across this mesh axis
        instead of replicating.  Used on TG (Galaxy) where DP groups each process
        a subset of the batch.  The mesh axis must evenly divide the batch size.
    Returns: (tt_positions, cos_batch, sin_batch, sin_neg_batch) on device.
    sin_neg_batch: [-sin[:half], sin[half:]] for fused addcmul RoPE (OPT-3).
    """
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    batch = int(positions.shape[0])

    # Upload positions to device.
    # When dp_shard_axis is set, shard positions across DP groups (each group gets
    # only its subset of positions).
    if is_mesh_device and dp_shard_axis is not None:
        mesh_shape = list(device.shape)
        pos_dims = [None, None]
        pos_dims[dp_shard_axis] = 0  # shard tensor dim=0 across dp_shard_axis
        pos_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(pos_dims), mesh_shape=mesh_shape)
    elif is_mesh_device:
        pos_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        pos_mapper = None
    tt_positions = ttnn.from_torch(
        positions.view(-1).contiguous().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=pos_mapper,
    )

    # Gather per-position cos/sin from host-side RoPE tables.
    cos_host = rope["cos_matrix_host"]  # [1, 1, max_seq_len, rope_dim]
    sin_host = rope["sin_matrix_host"]
    rope_dim = int(cos_host.shape[3])

    positions_cpu = positions.to(torch.long).clamp(min=0, max=int(cos_host.shape[2]) - 1)
    cos_batch_t = cos_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)
    sin_batch_t = sin_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)

    # Determine mesh mapper for cos/sin.
    if is_mesh_device and dp_shard_axis is not None:
        # Shard batch dim across DP groups so each group gets only its positions' cos/sin.
        mesh_shape = list(device.shape)
        dims = [None, None]
        dims[dp_shard_axis] = 1  # shard tensor dim=1 (batch) across dp_shard_axis
        rope_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(dims), mesh_shape=mesh_shape)
    elif is_mesh_device:
        rope_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        rope_mapper = None

    cos_batch = ttnn.from_torch(
        cos_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )
    sin_batch = ttnn.from_torch(
        sin_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )

    # Pre-compute sin_neg = [-sin[:half], sin[half:]] for addcmul RoPE fusion (OPT-3).
    half = rope_dim // 2
    sin_neg_batch_t = torch.cat([-sin_batch_t[..., :half], sin_batch_t[..., half:]], dim=-1)
    sin_neg_batch = ttnn.from_torch(
        sin_neg_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )

    return tt_positions, cos_batch, sin_batch, sin_neg_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tt_to_torch_for_vllm_output(*, tensor: ttnn.Tensor, device: Any) -> torch.Tensor:
    """Convert a TT tensor to torch for vLLM outputs.

    On a mesh, reads back device 0 (replicated model bring-up mode).
    """
    if not _is_mesh_device(device):
        return ttnn.to_torch(tensor.cpu())
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    return ttnn.to_torch(device_tensors[0].cpu())


def _load_hparams_from_snapshot(snapshot_dir: Path) -> Glm4MoeHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _torch_dtype_to_ttnn(dtype: torch.dtype) -> ttnn.DataType:
    """Map vLLM torch dtype to TT KV cache dtype, with env override."""
    override = os.environ.get("GLM4_MOE_KV_CACHE_TT_DTYPE", "").strip().lower()
    if override:
        if override in {"bf8", "bfloat8_b"}:
            return ttnn.bfloat8_b
        if override in {"bf16", "bfloat16"}:
            return ttnn.bfloat16
        if override in {"f32", "fp32", "float32"}:
            return ttnn.float32
        raise ValueError(f"Invalid GLM4_MOE_KV_CACHE_TT_DTYPE={override!r}")
    return ttnn.bfloat8_b


@dataclass
class _DecodeTraceState:
    """Per-bucket state for batch-bucketed decode traces."""

    trace_id: Any | None = None
    batch: int = 0
    page_table_width: int = 0
    tokens_tt: ttnn.Tensor | None = None
    positions_tt: ttnn.Tensor | None = None
    cos_batch_tt: ttnn.Tensor | None = None
    sin_batch_tt: ttnn.Tensor | None = None
    sin_neg_batch_tt: ttnn.Tensor | None = None
    trans_matrix_tt: ttnn.Tensor | None = None
    page_table_tt: ttnn.Tensor | None = None
    logits_tt: ttnn.Tensor | None = None
    top1_values_tt: ttnn.Tensor | None = None
    top1_indices_tt: ttnn.Tensor | None = None
    embed_tt: ttnn.Tensor | None = None

    # MTP trace state
    mtp_hidden_tt: ttnn.Tensor | None = None  # [1,1,B,hidden] clone from main trace
    mtp_positions_tt: ttnn.Tensor | None = None  # [B] int32 — MTP positions (start_pos+1)
    mtp_cos_batch_tt: ttnn.Tensor | None = None  # [1,B,1,rope_dim] bf16
    mtp_sin_batch_tt: ttnn.Tensor | None = None  # [1,B,1,rope_dim] bf16
    mtp_sin_neg_batch_tt: ttnn.Tensor | None = None  # [1,B,1,rope_dim] bf16
    mtp_page_table_tt: ttnn.Tensor | None = None  # [B,W] int32 (or None to reuse main)
    mtp_embed_tt: ttnn.Tensor | None = None  # [1,1,B,hidden] — MTP embedding buffer
    mtp_trace_id: int | None = None
    mtp_logits_tt: ttnn.Tensor | None = None  # MTP logits (for host-side argmax)


# ---------------------------------------------------------------------------
# Model Runner
# ---------------------------------------------------------------------------


@dataclass
class Glm4MoeTT:
    """TT runner for GLM-4.7-REAP-218B (92 layers, GQA attention, MoE).

    Loaded via `Glm4MoeTT.create(...)` factory.
    """

    device: Any
    snapshot_dir: Path
    cache_dir: Path
    max_seq_len: int

    hparams: Glm4MoeHParams
    state: Any  # LazyStateDict
    embed_w: Optional[ttnn.Tensor]
    embed_w_cpu: torch.Tensor
    rope: dict[str, Any]
    final_norm: RMSNorm
    lm_head_w: ttnn.Tensor
    lm_head_sharded_vocab: bool
    lm_head_tp_axis: int | None
    lm_head_tp_size: int
    lm_head_vocab_per_shard: int
    layer_weights: dict[int, DecoderLayerTTWeights]
    decoder_layers: dict[int, Glm4MoeDecoderLayer]
    num_layers_to_run: int
    enable_moe: bool
    moe_runtime: Glm4MoeMoERuntime | None
    configuration: dict[str, Any]
    tt_ccl: Any | None

    # DRAM weight prefetcher — initialized when GLM4_MOE_PREFETCH=1
    prefetcher: Any | None = None

    # MTP (Multi-Token Prediction) fields — loaded when GLM4_MOE_MTP=1
    mtp_enabled: bool = False
    mtp_enorm: Any | None = None  # RMSNorm(hidden_size)
    mtp_hnorm: Any | None = None  # RMSNorm(hidden_size)
    mtp_eh_proj_w: ttnn.Tensor | None = None  # Linear [hidden, 2*hidden] -> [hidden] (UNUSED, kept for reference)
    mtp_eh_proj_e_w: ttnn.Tensor | None = None  # Linear [hidden, hidden] -> [hidden] (embed half)
    mtp_eh_proj_h_w: ttnn.Tensor | None = None  # Linear [hidden, hidden] -> [hidden] (hidden half)
    mtp_shared_head_norm: Any | None = None  # RMSNorm(hidden_size)
    mtp_shared_head_w: ttnn.Tensor | None = None  # Linear [hidden] -> [vocab/tp]
    mtp_decoder_layer: Any | None = None  # Glm4MoeDecoderLayer for MTP layer
    mtp_max_batch: int = 16  # Skip MTP when active batch > this threshold

    _decode_trace_states: dict[int, _DecodeTraceState] = field(init=False, default_factory=dict)
    _last_draft_token_ids: torch.Tensor | None = field(init=False, default=None)
    _prev_draft_ids: torch.Tensor | None = field(init=False, default=None)
    _prev_main_ids: torch.Tensor | None = field(init=False, default=None)
    _mtp_batch_skip_logged: bool = field(init=False, default=False)

    @classmethod
    def create(
        cls,
        *,
        device: Any,
        snapshot_dir: Path,
        cache_dir: Path,
        max_seq_len: int,
        max_batch_size: int = 32,
        hparams: Optional[Glm4MoeHParams] = None,
    ) -> "Glm4MoeTT":
        snapshot_dir = Path(snapshot_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hparams = _load_hparams_from_snapshot(snapshot_dir) if hparams is None else hparams

        state = load_glm_lazy_state_dict(snapshot_dir, num_layers=int(hparams.num_hidden_layers))

        # Embedding.
        embed_w_cpu = state["model.embed_tokens.weight"].clone().to(torch.bfloat16)
        # MTP in-trace embedding: when MTP is enabled AND GLM4_MOE_MTP_DEVICE_EMBED=1,
        # load embedding weight to device for in-trace argmax→embedding→MTP chain.
        # Costs ~1.49 GB DRAM per device but enables current-step MTP predictions.
        mtp_enabled_env = os.environ.get("GLM4_MOE_MTP", "0").strip() == "1"
        mtp_device_embed = os.environ.get("GLM4_MOE_MTP_DEVICE_EMBED", "0").strip() == "1"
        if mtp_enabled_env and mtp_device_embed:
            logger.info("Loading embedding weight to device for MTP in-trace embedding (~1.49 GB/device)")
            embed_w = convert_embedding_weight_to_tt(
                device=device,
                embed_weight=state["model.embed_tokens.weight"],
                cache_file_name=cache_dir / "embed_tokens_w",
                dtype=ttnn.bfloat16,
            )
        else:
            # Skip device embedding — TG mesh uses host-side lookup (embed_w_cpu).
            # Saves ~1.49 GB DRAM per device.
            embed_w = None

        # RoPE for partial rotary (rope_dim = head_dim * partial_rotary_factor = 64).
        rope_dim = int(hparams.head_dim * hparams.partial_rotary_factor)
        rope = _make_rope_tensors(
            device=device,
            seq_len=int(max_seq_len),
            rope_dim=rope_dim,
            rope_theta=float(hparams.rope_theta),
        )

        # Final norm + LM head.
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

        # LM head sharding: shard vocab across TP devices (axis 0), replicate across DP.
        lm_head_mapper = None
        lm_head_variant = ""
        lm_head_sharded_vocab = False
        lm_head_tp_axis = None
        lm_head_tp_size = 1
        lm_head_vocab_per_shard = int(hparams.vocab_size)
        num_devices = 1
        if _is_mesh_device(device):
            num_devices = int(device.get_num_devices())

        tp_axis, tp_size_detected = _tp_axis_and_size(device)
        if tp_size_detected > 1 and num_devices > 1:
            vocab = int(hparams.vocab_size)
            if vocab % tp_size_detected == 0:
                # Shard vocab across TP axis only, replicate across DP.
                lm_head_mapper = _tp_mesh_mapper(device, shard_dim=3)
                lm_head_variant = f"tp{tp_size_detected}_shard_v1"
                lm_head_sharded_vocab = True
                lm_head_tp_axis = tp_axis
                lm_head_tp_size = tp_size_detected
                lm_head_vocab_per_shard = vocab // tp_size_detected
            else:
                logger.warning(
                    "LM head vocab {} not divisible by TP={} devices, replicating instead of sharding",
                    vocab,
                    tp_size_detected,
                )

        lm_head_w = _linear_weight_tt(
            device=device,
            torch_weight_out_in=state["lm_head.weight"],
            cache_file=cache_dir / f"lm_head_w_{lm_head_variant}" if lm_head_variant else cache_dir / "lm_head_w",
            dtype=dense_dtype,
            mesh_mapper=lm_head_mapper,
        )

        # Layers.
        num_layers_env = os.environ.get("GLM4_MOE_NUM_LAYERS", "").strip()
        if num_layers_env and os.environ.get("GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS", "").strip() != "1":
            raise ValueError(
                "GLM4_MOE_NUM_LAYERS is debug-only. "
                "Set GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS=1 to run a partial model."
            )
        num_layers_to_run = int(num_layers_env) if num_layers_env else int(hparams.num_hidden_layers)
        num_layers_to_run = max(1, min(num_layers_to_run, int(hparams.num_hidden_layers)))

        enable_moe = os.environ.get("GLM4_MOE_ENABLE_MOE", "1").strip() != "0"
        moe_runtime = None
        if enable_moe:
            moe_runtime = create_moe_runtime(device=device, hparams=hparams)

        # Configuration dict for attention and decoder layers.
        mesh_rows = int(device.shape[0]) if _is_mesh_device(device) else 1
        mesh_cols = int(device.shape[1]) if _is_mesh_device(device) else 1
        configuration = {
            "num_devices": mesh_rows * mesh_cols,
            "max_batch_size": max_batch_size,
            "MAX_QKV_MM_SEQ_LEN": 4096,
            "ccl_dtype": ttnn.bfloat16,
        }

        # CCL for attention all-reduce/all-gather (async semaphore management).
        if _is_mesh_device(device):
            logger.info("Glm4MoeTT.create: initializing CCL semaphores for mesh {}", list(device.shape))
            tt_ccl = CCL(device)
        else:
            tt_ccl = None

        # Load decoder layer weights lazily (one at a time to save host memory).
        layer_weights_dict: dict[int, DecoderLayerTTWeights] = {}
        decoder_layers_dict: dict[int, Glm4MoeDecoderLayer] = {}

        logger.info(
            "Glm4MoeTT.create: loading {} layers (total={}), moe={}, devices={}, max_seq_len={}",
            num_layers_to_run,
            hparams.num_hidden_layers,
            enable_moe,
            num_devices,
            max_seq_len,
        )

        # DEBUG: test device sync before loading layers
        if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            import sys

            print("  [DEBUG MODEL] synchronizing before layer loading ...", flush=True, file=sys.stderr)
            ttnn.synchronize_device(device)
            print("  [DEBUG MODEL] synchronize before layers OK", flush=True, file=sys.stderr)

        for layer_idx in range(num_layers_to_run):
            t0 = time.perf_counter()
            logger.info("  [DEBUG] Starting layer {} weight conversion", layer_idx)
            lw = convert_decoder_layer_weights(
                device=device,
                state=state,
                layer_idx=layer_idx,
                hparams=hparams,
                cache_dir=cache_dir / "layers",
                enable_moe=enable_moe and (layer_idx >= int(hparams.first_k_dense_replace)),
            )
            layer_weights_dict[layer_idx] = lw
            logger.info("  [DEBUG] Layer {} weights converted, creating decoder layer", layer_idx)

            dl = Glm4MoeDecoderLayer(
                mesh_device=device,
                tt_ccl=tt_ccl,
                hparams=hparams,
                layer_weights=lw,
                configuration=configuration,
                paged_attention_config=None,
                moe_runtime=moe_runtime,
            )
            decoder_layers_dict[layer_idx] = dl

            elapsed = time.perf_counter() - t0
            if layer_idx == 0 or (layer_idx + 1) % 10 == 0 or (layer_idx + 1) == num_layers_to_run:
                logger.info("  Layer {}/{} loaded ({:.1f}s)", layer_idx + 1, num_layers_to_run, elapsed)

            # DEBUG: sync after each layer to find which one hangs the device
            if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
                print(f"  [DEBUG MODEL] sync after layer {layer_idx} ...", flush=True, file=sys.stderr)
                ttnn.synchronize_device(device)
                print(f"  [DEBUG MODEL] sync after layer {layer_idx} OK", flush=True, file=sys.stderr)

        # DEBUG: test device sync after model init
        if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            import sys

            print("  [DEBUG MODEL] synchronizing after all layers loaded ...", flush=True, file=sys.stderr)
            ttnn.synchronize_device(device)
            print("  [DEBUG MODEL] synchronize after layers OK", flush=True, file=sys.stderr)

        # ---- DRAM Weight Prefetcher (optional, for decode latency optimization) ----
        prefetcher = None
        if os.environ.get("GLM4_MOE_PREFETCH", "0").strip() == "1":
            from models.experimental.glm4_moe.tt.prefetcher_setup import Glm4MoePrefetcherSetup

            n_tensors_per_layer = 2  # QKV + O-proj (attention weights only for Phase 2)
            prefetcher = Glm4MoePrefetcherSetup(
                mesh_device=device,
                n_tensors_per_layer=n_tensors_per_layer,
                n_layers=num_layers_to_run,
            )
            # Register attention weights from all layers
            for li in range(num_layers_to_run):
                lw = layer_weights_dict[li]
                prefetcher.insert_tensor(lw.w_qkv)
                prefetcher.insert_tensor(lw.w_o)
            logger.info(
                "Prefetcher: registered {} attention weights ({} layers × {} tensors)",
                len(prefetcher.tensor_addrs),
                num_layers_to_run,
                n_tensors_per_layer,
            )

        # ---- MTP Layer (optional, for GLM-4.7 Full with num_nextn_predict_layers=1) ----
        mtp_enabled = os.environ.get("GLM4_MOE_MTP", "").strip() == "1"
        mtp_max_batch = int(os.environ.get("GLM4_MOE_MTP_MAX_BATCH", "16") or "16")
        mtp_enorm = None
        mtp_hnorm = None
        mtp_eh_proj_w = None
        mtp_eh_proj_e_w = None
        mtp_eh_proj_h_w = None
        mtp_shared_head_norm = None
        mtp_shared_head_w = None
        mtp_decoder_layer = None

        if mtp_enabled:
            mtp_layer_idx = int(hparams.num_hidden_layers)  # 92
            logger.info("MTP enabled: loading layer {} weights (mtp_max_batch={})", mtp_layer_idx, mtp_max_batch)
            mtp_cache = cache_dir / "mtp"
            mtp_cache.mkdir(parents=True, exist_ok=True)

            # Need unfiltered state dict to access MTP layer weights
            mtp_state = load_glm_lazy_state_dict(snapshot_dir)

            hidden = int(hparams.hidden_size)  # 5120

            # enorm: RMSNorm(hidden_size)
            mtp_enorm = RMSNorm(
                device=device,
                dim=hidden,
                eps=float(hparams.rms_norm_eps),
                state_dict=mtp_state,
                state_dict_prefix=f"model.layers.{mtp_layer_idx}.",
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
                state_dict_prefix=f"model.layers.{mtp_layer_idx}.",
                weight_key="hnorm",
                weight_cache_path=mtp_cache,
                weight_dtype=ttnn.bfloat16,
                is_distributed=False,
            )

            # eh_proj: Linear(2*hidden -> hidden, no bias)
            # Split into two halves to avoid ttnn.concat (concat fails on TG mesh)
            # Original weight: [hidden, 2*hidden] in HF layout (out_dim, in_dim)
            # First hidden cols correspond to embed input, second hidden cols to hidden input
            eh_proj_full = mtp_state[f"model.layers.{mtp_layer_idx}.eh_proj.weight"]  # [hidden, 2*hidden]
            mtp_eh_proj_w = None  # Not used anymore
            mtp_eh_proj_e_w = _linear_weight_tt(
                device=device,
                torch_weight_out_in=eh_proj_full[:, :hidden].contiguous(),  # [hidden, hidden]
                cache_file=mtp_cache / "eh_proj_e_w",
                dtype=dense_dtype,
            )
            mtp_eh_proj_h_w = _linear_weight_tt(
                device=device,
                torch_weight_out_in=eh_proj_full[:, hidden:].contiguous(),  # [hidden, hidden]
                cache_file=mtp_cache / "eh_proj_h_w",
                dtype=dense_dtype,
            )

            # shared_head.norm: RMSNorm(hidden_size)
            mtp_shared_head_norm = RMSNorm(
                device=device,
                dim=hidden,
                eps=float(hparams.rms_norm_eps),
                state_dict=mtp_state,
                state_dict_prefix=f"model.layers.{mtp_layer_idx}.shared_head.",
                weight_key="norm",
                weight_cache_path=mtp_cache,
                weight_dtype=ttnn.bfloat16,
                is_distributed=False,
            )

            # shared_head.head: Linear(hidden -> vocab), SAME vocab sharding as main lm_head
            mtp_shared_head_w = _linear_weight_tt(
                device=device,
                torch_weight_out_in=mtp_state[f"model.layers.{mtp_layer_idx}.shared_head.head.weight"],
                cache_file=mtp_cache / f"shared_head_w_{lm_head_variant}"
                if lm_head_variant
                else mtp_cache / "shared_head_w",
                dtype=dense_dtype,
                mesh_mapper=lm_head_mapper,
            )

            # Full decoder layer for MTP (attention + MoE)
            mtp_lw = convert_decoder_layer_weights(
                device=device,
                state=mtp_state,
                layer_idx=mtp_layer_idx,
                hparams=hparams,
                cache_dir=cache_dir / "layers",
                enable_moe=enable_moe,  # MTP layer has MoE (layer 92 >= first_k_dense_replace=3)
            )
            layer_weights_dict[mtp_layer_idx] = mtp_lw

            mtp_decoder_layer = Glm4MoeDecoderLayer(
                mesh_device=device,
                tt_ccl=tt_ccl,
                hparams=hparams,
                layer_weights=mtp_lw,
                configuration=configuration,
                paged_attention_config=None,
                moe_runtime=moe_runtime,
            )

            logger.info("MTP layer {} weights loaded successfully", mtp_layer_idx)

        logger.info("=== All weights loaded, creating model object (MTP={}) ===", mtp_enabled)
        return cls(
            device=device,
            snapshot_dir=snapshot_dir,
            cache_dir=cache_dir,
            max_seq_len=int(max_seq_len),
            hparams=hparams,
            state=state,
            embed_w=embed_w,
            embed_w_cpu=embed_w_cpu,
            rope=rope,
            final_norm=final_norm,
            lm_head_w=lm_head_w,
            lm_head_sharded_vocab=lm_head_sharded_vocab,
            lm_head_tp_axis=lm_head_tp_axis,
            lm_head_tp_size=lm_head_tp_size,
            lm_head_vocab_per_shard=lm_head_vocab_per_shard,
            layer_weights=layer_weights_dict,
            decoder_layers=decoder_layers_dict,
            num_layers_to_run=num_layers_to_run,
            enable_moe=enable_moe,
            moe_runtime=moe_runtime,
            configuration=configuration,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            mtp_enabled=mtp_enabled,
            mtp_max_batch=mtp_max_batch,
            mtp_enorm=mtp_enorm,
            mtp_hnorm=mtp_hnorm,
            mtp_eh_proj_w=mtp_eh_proj_w,
            mtp_eh_proj_e_w=mtp_eh_proj_e_w,
            mtp_eh_proj_h_w=mtp_eh_proj_h_w,
            mtp_shared_head_norm=mtp_shared_head_norm,
            mtp_shared_head_w=mtp_shared_head_w,
            mtp_decoder_layer=mtp_decoder_layer,
        )

    # -------------------------------------------------------------------
    # Decode
    # -------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
        enable_trace: bool = False,
    ) -> Any:
        """Run a single-token decode step through all layers.

        Args:
            tokens: [B, 1] int32
            start_pos: [B] int32 (padded with -1 for inactive slots)
            page_table: [B, W] int32
            kv_cache: list of [cache_k, cache_v] per layer
            sampling_params: if not None, do on-device greedy sampling
            enable_trace: use trace capture/replay for decode

        Returns:
            logits [active, 1, vocab] as torch float32 when sampling_params is None,
            or next token ids [active] as torch int32 when sampling_params is not None.
        """
        _PROF = os.environ.get("GLM4_MOE_PROFILE", "0") != "0"
        _t_decode_start = time.perf_counter_ns() if _PROF else 0
        if _PROF:
            logger.info("TTPROF decode_start active={} trace={}", int((start_pos >= 0).sum().item()), enable_trace)
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError(f"expected tokens [B,1], got {tuple(tokens.shape)}")
        if start_pos.ndim != 1:
            raise ValueError(f"expected start_pos [B], got {tuple(start_pos.shape)}")

        start_pos = start_pos.to(torch.int32)
        active = int((start_pos >= 0).sum().item())
        if active <= 0:
            return torch.zeros((0, 1, int(self.hparams.vocab_size)), dtype=torch.float32)

        if enable_trace:
            return self._decode_trace(
                tokens=tokens[:active].to(torch.int32),
                positions=start_pos[:active].to(torch.int32),
                page_table=page_table[:active].to(torch.int32),
                kv_cache=kv_cache,
                sampling_params=sampling_params,
            )

        return self._decode_eager(
            tokens=tokens[:active].to(torch.int32),
            positions=start_pos[:active].to(torch.int32),
            page_table=page_table[:active].to(torch.int32),
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def _decode_eager(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
    ) -> Any:
        """Eager (non-traced) decode step."""
        active = int(tokens.shape[0])
        hidden = int(self.hparams.hidden_size)
        is_mesh = _is_mesh_device(self.device)

        # Prepare inputs.
        tokens = tokens.contiguous().clone()
        positions = positions.contiguous().clone()
        page_table = page_table.contiguous().clone()

        # On TG (2D mesh) with multi-user batch, shard batch-dimension inputs
        # across DP groups (page_table, positions, cos/sin).
        dp_shard_axis = None
        dp_batch_mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                dp_shard_axis = 1
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0),
                    mesh_shape=mesh_shape,
                )

        page_table_tt = ttnn.from_torch(
            page_table,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        tt_positions, cos_batch, sin_batch, sin_neg_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device,
            rope=self.rope,
            positions=positions,
            dp_shard_axis=dp_shard_axis,
        )

        # Embedding: host-side lookup to avoid device-side tile conversion hang on TG mesh.
        embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
        x = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
        )

        # RoPE mats for attention (cos, sin, trans_matrix).
        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"], sin_neg_batch)

        # Decoder stack.
        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(
                x,
                tt_positions,
                rot_mats,
                page_table_tt,
                kv_cache[layer_idx],
                mode="decode",
                active_batch=active,
            )
            ttnn.deallocate(x, force=False)
            x = x_next

        # Preserve hidden state for MTP before final_norm (skip clone if batch exceeds threshold)
        mtp_hidden = None
        if self.mtp_enabled:
            if active <= self.mtp_max_batch:
                mtp_hidden = ttnn.typecast(x, dtype=x.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            elif not self._mtp_batch_skip_logged:
                logger.info(
                    "MTP skipped: active batch {} > mtp_max_batch {} (GLM4_MOE_MTP_MAX_BATCH)",
                    active,
                    self.mtp_max_batch,
                )
                object.__setattr__(self, "_mtp_batch_skip_logged", True)

        # Final norm + LM head (sharded norm to avoid L1 overflow with hidden=5120).
        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)  # [1, 1, B, vocab]

        if sampling_params is not None:
            next_ids_flat = self._sample_greedy(logits_tt, active, x, tt_positions, cos_batch, sin_batch, page_table_tt)

            # MTP draft token generation (skip when batch exceeds threshold)
            if self.mtp_enabled and mtp_hidden is not None and active <= self.mtp_max_batch:
                try:
                    draft_token_ids = self._mtp_forward_eager(
                        main_token_ids=next_ids_flat,
                        hidden_state=mtp_hidden,
                        mtp_positions=positions + 1,
                        page_table=page_table,
                        kv_cache=kv_cache,
                    )
                    object.__setattr__(self, "_last_draft_token_ids", draft_token_ids)
                except Exception as e:
                    logger.warning("MTP eager forward failed (non-fatal): {}", e)
                    object.__setattr__(self, "_last_draft_token_ids", None)
                ttnn.deallocate(mtp_hidden, force=False)
            elif mtp_hidden is not None:
                ttnn.deallocate(mtp_hidden, force=False)

            return next_ids_flat

        # Return full logits on host.
        vocab = int(self.hparams.vocab_size)
        result = self._logits_to_host(logits_tt, active, vocab)

        # MTP: compute main_ids from logits if sampling_params is None (host-sampling mode)
        if self.mtp_enabled and mtp_hidden is not None and active <= self.mtp_max_batch:
            try:
                logits_host = result  # already on host from _logits_to_host
                main_ids_eager = torch.argmax(logits_host, dim=-1).to(torch.int32).cpu().flatten()
                draft_token_ids = self._mtp_forward_eager(
                    main_token_ids=main_ids_eager,
                    hidden_state=mtp_hidden,
                    mtp_positions=positions + 1,
                    page_table=page_table,
                    kv_cache=kv_cache,
                )
                object.__setattr__(self, "_last_draft_token_ids", draft_token_ids)
            except Exception as e:
                logger.warning("MTP eager forward failed (non-fatal): {}", e)
                object.__setattr__(self, "_last_draft_token_ids", None)
            ttnn.deallocate(mtp_hidden, force=False)
            mtp_hidden = None  # prevent double-free
        elif mtp_hidden is not None:
            ttnn.deallocate(mtp_hidden, force=False)

        ttnn.deallocate(logits_tt, force=False)
        ttnn.deallocate(x, force=False)
        ttnn.deallocate(tt_positions, force=False)
        ttnn.deallocate(cos_batch, force=False)
        ttnn.deallocate(sin_batch, force=False)
        ttnn.deallocate(page_table_tt, force=False)

        # Reset CCL semaphore counters after eager decode.
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        return result

    def _sample_greedy(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        x: ttnn.Tensor,
        tt_positions: ttnn.Tensor,
        cos_batch: ttnn.Tensor,
        sin_batch: ttnn.Tensor,
        page_table_tt: ttnn.Tensor,
    ) -> torch.Tensor:
        """On-device greedy sampling (argmax) from logits."""
        vocab = int(self.hparams.vocab_size)

        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.typecast(
                logits_rm_view, dtype=logits_rm_view.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, next_ids_tt = max_out
                ttnn.deallocate(local_max_tt, force=False)
            else:
                next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            ttnn.deallocate(logits_rm_tight, force=False)

            next_ids_torch = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids_flat = next_ids_torch.reshape(-1).to(dtype=torch.int32).cpu()
            ttnn.deallocate(logits_rm, force=False)
            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(next_ids_tt, force=False)
        else:
            # Vocab-sharded: per-device max+argmax, reduce on host.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, local_argmax_tt = max_out
            else:
                local_max_tt = max_out
                local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

            num_devices = int(self.device.get_num_devices())
            local_max_dts = ttnn.get_device_tensors(local_max_tt)
            local_idx_dts = ttnn.get_device_tensors(local_argmax_tt)

            next_ids = torch.empty((active,), dtype=torch.int32)
            for b in range(active):
                best_val = None
                best_global = None
                for shard_idx in range(num_devices):
                    max_val = float(ttnn.to_torch(local_max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(local_idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = shard_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global
            next_ids_flat = next_ids

            ttnn.deallocate(local_max_tt, force=False)
            ttnn.deallocate(local_argmax_tt, force=False)
            ttnn.deallocate(logits_rm, force=False)
            ttnn.deallocate(logits_tt, force=False)

        ttnn.deallocate(x, force=False)
        ttnn.deallocate(tt_positions, force=False)
        ttnn.deallocate(cos_batch, force=False)
        ttnn.deallocate(sin_batch, force=False)
        ttnn.deallocate(page_table_tt, force=False)

        # Reset CCL semaphore counters after eager decode (sampling path).
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        return next_ids_flat

    def _sample_from_trace_logits(self, logits_tt: ttnn.Tensor, active: int) -> torch.Tensor:
        """Device-side greedy sampling from trace-owned logits (OUTSIDE trace).

        Called after execute_trace + synchronize_device. The logits_tt tensor
        is trace-owned and must NOT be deallocated.

        For TP-sharded vocab: per-shard max+argmax on device, then pick global
        best on host. Transfers only ~64 bytes instead of 9.6 MB.
        """
        vocab = int(self.hparams.vocab_size)

        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            # Vocab-sharded: per-device max+argmax, reduce on host.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, local_argmax_tt = max_out
            else:
                local_max_tt = max_out
                local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

            num_devices = int(self.device.get_num_devices())
            tp_size = self.lm_head_tp_size
            local_max_dts = ttnn.get_device_tensors(local_max_tt)
            local_idx_dts = ttnn.get_device_tensors(local_argmax_tt)

            # Skip DP duplicates: pick one shard per TP position
            dp_stride = num_devices // tp_size if num_devices > tp_size else 1

            next_ids = torch.empty((active,), dtype=torch.int32)
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    shard_idx = tp_idx * dp_stride
                    max_val = float(ttnn.to_torch(local_max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(local_idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global

            # Clean up temporaries (NOT logits_tt — trace-owned)
            ttnn.deallocate(local_max_tt, force=True)
            ttnn.deallocate(local_argmax_tt, force=True)
            ttnn.deallocate(logits_rm_view, force=False)
            ttnn.deallocate(logits_rm, force=True)
        else:
            # Non-sharded: single argmax
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.typecast(
                logits_rm_view, dtype=logits_rm_view.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                _, next_ids_tt = max_out
            else:
                next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)

            next_ids = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids = next_ids.reshape(-1).to(dtype=torch.int32).cpu()

            ttnn.deallocate(next_ids_tt, force=True)
            ttnn.deallocate(logits_rm_tight, force=True)
            ttnn.deallocate(logits_rm_view, force=False)
            ttnn.deallocate(logits_rm, force=True)

        return next_ids

    def _host_argmax_from_trace_logits(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        vocab: int,
    ) -> torch.Tensor:
        """Per-shard device-side top-1 from trace-owned logits.

        Uses ttnn.topk(k=1) per TP shard on device, then reads only the top-1
        value+index (~256 bytes total across 8 shards) instead of transferring
        the full logit shards to host (~304 KB).  Falls back to full-transfer
        host argmax if topk fails.

        Does NOT deallocate logits_tt (trace-owned).
        """
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            shards = ttnn.get_device_tensors(logits_tt)
            num_shards = len(shards)
            tp_size = self.lm_head_tp_size
            vocab_per_shard = int(self.lm_head_vocab_per_shard)

            # Select one shard per TP position (skip DP duplicates).
            dp_stride = num_shards // tp_size if num_shards > tp_size else 1

            next_ids = torch.empty((active,), dtype=torch.int32)

            # Device-side per-shard top-1: topk(k=1) on each TP shard.
            # Transfers ~32 bytes/shard instead of ~38 KB/shard.
            all_vals = []  # [tp_size] lists of [active] floats
            all_idxs = []  # [tp_size] lists of [active] ints
            try:
                for tp_idx in range(tp_size):
                    shard_idx = tp_idx * dp_stride
                    shard = shards[shard_idx]

                    topk_values, topk_indices = ttnn.topk(
                        shard,
                        k=1,
                        dim=-1,
                        largest=True,
                        sorted=False,
                    )

                    val_host = ttnn.to_torch(topk_values.cpu())  # [1,1,B_pad,1]
                    idx_host = ttnn.to_torch(topk_indices.cpu())  # [1,1,B_pad,1]

                    ttnn.deallocate(topk_values)
                    ttnn.deallocate(topk_indices)

                    all_vals.append([float(val_host[0, 0, b, 0]) for b in range(active)])
                    all_idxs.append([int(idx_host[0, 0, b, 0]) for b in range(active)])
            except Exception:
                # topk failed on per-shard tensor — fall back to host transfer
                logger.warning("Device topk failed, falling back to host argmax")
                return self._host_argmax_fallback(logits_tt, active, vocab)

            # Pick global best across TP shards
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    v = all_vals[tp_idx][b]
                    local_idx = all_idxs[tp_idx][b]
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or v > best_val:
                        best_val = v
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global

            return next_ids
        else:
            # Non-sharded: simple host transfer + argmax
            logits_host = self._logits_to_host(logits_tt, active, vocab)
            return logits_host.reshape(active, -1)[:, :vocab].argmax(dim=-1).to(torch.int32)

    def _host_argmax_fallback(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        vocab: int,
    ) -> torch.Tensor:
        """Full-transfer host argmax fallback (transfers ~38 KB per TP shard)."""
        shards = ttnn.get_device_tensors(logits_tt)
        num_shards = len(shards)
        tp_size = self.lm_head_tp_size
        vocab_per_shard = int(self.lm_head_vocab_per_shard)
        dp_stride = num_shards // tp_size if num_shards > tp_size else 1

        next_ids = torch.empty((active,), dtype=torch.int32)
        shard_maxvals = []
        shard_argmaxes = []
        for tp_idx in range(tp_size):
            shard_idx = tp_idx * dp_stride
            shard_torch = ttnn.to_torch(shards[shard_idx].cpu())
            shard_torch = shard_torch[..., :active, :vocab_per_shard]
            shard_flat = shard_torch.reshape(active, -1).to(torch.float32)
            shard_maxvals.append(shard_flat.max(dim=-1))
            shard_argmaxes.append(shard_flat.argmax(dim=-1))

        for b in range(active):
            best_val = None
            best_global = None
            for tp_idx in range(tp_size):
                max_val = float(shard_maxvals[tp_idx].values[b].item())
                local_idx = int(shard_argmaxes[tp_idx][b].item())
                global_idx = tp_idx * vocab_per_shard + local_idx
                if global_idx >= vocab:
                    continue
                if best_val is None or max_val > best_val:
                    best_val = max_val
                    best_global = global_idx
            if best_global is None:
                best_global = max(0, vocab - 1)
            next_ids[b] = best_global

        return next_ids

    def _logits_to_host(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        vocab: int,
    ) -> torch.Tensor:
        """Convert logits TT tensor to host torch [active, 1, vocab]."""
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            shards = ttnn.get_device_tensors(logits_tt)
            num_shards = len(shards)
            tp_size = self.lm_head_tp_size
            # Select one shard per TP position (skip DP duplicates).
            if num_shards == tp_size:
                # Already one per TP device (ShardTensor2dMesh collapsed DP).
                tp_shards = list(shards)
            elif num_shards > tp_size:
                # All mesh devices returned; pick every (num_shards/tp_size)-th shard.
                dp_stride = num_shards // tp_size
                tp_shards = [shards[i * dp_stride] for i in range(tp_size)]
            else:
                tp_shards = list(shards)
            logits_shards = [ttnn.to_torch(t.cpu())[..., : int(t.shape[-1])] for t in tp_shards]
            logits_full = torch.cat(logits_shards, dim=-1)[..., :vocab]
            # Slice off TILE_SIZE padding — decode pads batch to 32 but we only need `active`.
            logits_full = logits_full[..., :active, :]
            logits_flat = logits_full.reshape(-1, vocab)
        else:
            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
            logits_torch = logits_torch[..., :vocab]
            # Slice off TILE_SIZE padding — decode pads batch to 32 but we only need `active`.
            logits_torch = logits_torch[..., :active, :]
            logits_flat = logits_torch.reshape(-1, vocab)

        if logits_flat.shape[0] != active:
            raise RuntimeError(f"decode logits shape mismatch: expected {active} rows, got {int(logits_flat.shape[0])}")

        return logits_flat.reshape(active, 1, vocab).to(dtype=torch.float32).cpu()

    # -------------------------------------------------------------------
    # MTP (Multi-Token Prediction)
    # -------------------------------------------------------------------

    @torch.no_grad()
    def _mtp_forward_eager(
        self,
        *,
        main_token_ids: torch.Tensor,  # [B] int32 — main model's predicted token IDs
        hidden_state: ttnn.Tensor,  # [1,1,B,hidden] TILE on device — pre-final_norm hidden
        mtp_positions: torch.Tensor,  # [B] int32 (= main start_pos + 1)
        page_table: torch.Tensor,  # [B,W] int32
        kv_cache: list,  # full list including kv_cache[mtp_layer_idx]
    ) -> torch.Tensor:
        """Run MTP layer eagerly. Returns draft_token_ids [B] int32 on CPU."""
        batch = int(main_token_ids.shape[0])
        hidden = int(self.hparams.hidden_size)  # 5120
        is_mesh = _is_mesh_device(self.device)
        mtp_layer_idx = int(self.hparams.num_hidden_layers)  # 92

        # 1. Embed main model's predicted tokens (HOST-SIDE to avoid TG mesh hangs)
        embed_torch = self.embed_w_cpu[main_token_ids.long()]  # [batch, hidden]
        # Pad embed to match hidden state batch (tile-padded from trace)
        hidden_batch = int(hidden_state.shape[-2])
        if batch < hidden_batch:
            pad = torch.zeros(hidden_batch - batch, hidden, dtype=embed_torch.dtype)
            embed_torch = torch.cat([embed_torch, pad], dim=0)
        x_embed = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1,1,hidden_batch,hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
        )

        # 2. enorm(embedded), hnorm(hidden_state)
        enorm_out = _sharded_rms_norm(x_embed, self.mtp_enorm, hidden)
        ttnn.deallocate(x_embed, force=False)
        hnorm_out = _sharded_rms_norm(hidden_state, self.mtp_hnorm, hidden)

        # 3. Split-matmul projection (avoids ttnn.concat which fails on TG mesh)
        proj_e = ttnn.linear(enorm_out, self.mtp_eh_proj_e_w)
        ttnn.deallocate(enorm_out, force=False)
        proj_h = ttnn.linear(hnorm_out, self.mtp_eh_proj_h_w)
        ttnn.deallocate(hnorm_out, force=False)
        proj = ttnn.add(proj_e, proj_h)
        ttnn.deallocate(proj_e, force=False)
        ttnn.deallocate(proj_h, force=False)

        # 4. Prepare RoPE for MTP positions (= main_position + 1)
        #    Use `batch` (not hidden_batch) — positions/cos/sin must match attention's active_batch
        dp_shard_axis = None
        dp_batch_mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if batch > 1 and batch % dp_size == 0:
                dp_shard_axis = 1
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0),
                    mesh_shape=mesh_shape,
                )

        mtp_pos_clamped = mtp_positions[:batch].to(torch.int32).clamp(min=0, max=max(0, int(self.max_seq_len) - 1))
        tt_positions, cos_batch, sin_batch, sin_neg_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device,
            rope=self.rope,
            positions=mtp_pos_clamped,
            dp_shard_axis=dp_shard_axis,
        )

        # Page table for MTP (use batch, not hidden_batch)
        pt = page_table[:batch].to(torch.int32).contiguous()
        page_table_tt = ttnn.from_torch(
            pt,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"], sin_neg_batch)

        # 5. Run MTP decoder layer (OOP forward)
        x = self.mtp_decoder_layer.forward(
            proj,
            tt_positions,
            rot_mats,
            page_table_tt,
            kv_cache[mtp_layer_idx],  # [tt_k, tt_v]
            mode="decode",
            active_batch=batch,  # real batch for attention masking
        )
        ttnn.deallocate(proj, force=False)

        # 6. shared_head: sharded norm + LM head
        x = _sharded_rms_norm(x, self.mtp_shared_head_norm, hidden)
        logits_tt = ttnn.linear(x, self.mtp_shared_head_w)  # [1,1,hidden_batch,vocab/tp]
        ttnn.deallocate(x, force=False)

        # 7. Host-side argmax (TG mesh constraint — device argmax hangs)
        draft_token_ids = self._host_argmax_from_trace_logits(logits_tt, hidden_batch, int(self.hparams.vocab_size))

        # Cleanup
        ttnn.deallocate(logits_tt, force=False)
        ttnn.deallocate(tt_positions, force=False)
        ttnn.deallocate(cos_batch, force=False)
        ttnn.deallocate(sin_batch, force=False)
        ttnn.deallocate(sin_neg_batch, force=False)
        ttnn.deallocate(page_table_tt, force=False)

        return draft_token_ids[:batch]  # Slice to real batch size

    def _mtp_decode_step_tt(
        self,
        *,
        state: _DecodeTraceState,
        kv_cache: list,
        mtp_current_embed_tt: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """MTP decode step using persistent device tensors. Returns MTP logits on device.

        If mtp_current_embed_tt is provided, uses it directly as the token embedding
        (from in-trace argmax→embedding chain). Otherwise falls back to state.mtp_embed_tt
        (persistent buffer uploaded between replays).
        """
        batch = int(state.batch)
        hidden = int(self.hparams.hidden_size)  # 5120
        mtp_layer_idx = int(self.hparams.num_hidden_layers)  # 92

        # 1. Embed: use in-trace embedding if available, else persistent buffer
        if mtp_current_embed_tt is not None:
            x_embed = mtp_current_embed_tt  # [1,1,B,hidden] from argmax→embedding inside trace
        else:
            x_embed = state.mtp_embed_tt  # [1,1,B,hidden] uploaded by _copy_mtp_trace_inputs

        # 2. enorm(embedded), hnorm(hidden_state from main trace)
        enorm_out = _sharded_rms_norm(x_embed, self.mtp_enorm, hidden)
        # DO NOT deallocate x_embed — it may be a persistent buffer
        hnorm_out = _sharded_rms_norm(state.mtp_hidden_tt, self.mtp_hnorm, hidden)

        # 3. Split-matmul projection (avoids ttnn.concat which fails on TG mesh)
        # proj = W_e @ enorm + W_h @ hnorm (equivalent to W @ concat(enorm, hnorm))
        proj_e = ttnn.linear(enorm_out, self.mtp_eh_proj_e_w)
        ttnn.deallocate(enorm_out, force=False)
        proj_h = ttnn.linear(hnorm_out, self.mtp_eh_proj_h_w)
        ttnn.deallocate(hnorm_out, force=False)
        proj = ttnn.add(proj_e, proj_h)
        ttnn.deallocate(proj_e, force=False)
        ttnn.deallocate(proj_h, force=False)

        # 4. RoPE from persistent cos/sin tensors
        rot_mats = (
            state.mtp_cos_batch_tt,
            state.mtp_sin_batch_tt,
            self.rope["trans_matrix"],
            state.mtp_sin_neg_batch_tt,
        )

        # 5. Run MTP decoder layer
        x = self.mtp_decoder_layer.forward(
            proj,
            state.mtp_positions_tt,
            rot_mats,
            state.mtp_page_table_tt if state.mtp_page_table_tt is not None else state.page_table_tt,
            kv_cache[mtp_layer_idx],
            mode="decode",
            active_batch=batch,
        )
        ttnn.deallocate(proj, force=False)

        # 6. shared_head: sharded norm + LM head
        x = _sharded_rms_norm(x, self.mtp_shared_head_norm, hidden)
        logits_tt = ttnn.linear(x, self.mtp_shared_head_w)
        ttnn.deallocate(x, force=False)

        return logits_tt

    def _copy_mtp_trace_inputs(
        self,
        *,
        state: _DecodeTraceState,
        main_token_ids: torch.Tensor,  # [B] int32
        mtp_positions: torch.Tensor,  # [B] int32 (= main start_pos + 1)
    ) -> None:
        """Copy MTP inputs into persistent device tensors."""
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        batch = int(state.batch)  # = active (the trace batch, e.g. 1 or 32)
        # Embed uses tile-padded batch (for _sharded_rms_norm / linear compatibility)
        mtp_batch = int(state.mtp_embed_tt.shape[-2])
        # Positions/cos/sin use active batch (must match attention's q/k batch)
        active = batch

        # DP sharding for TG mesh
        dp_batch_mapper = mapper
        rope_mapper = mapper
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0),
                    mesh_shape=mesh_shape,
                )
                rope_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 1),
                    mesh_shape=mesh_shape,
                )

        # 1. Copy MTP embedding (host-side lookup + upload)
        actual_batch = int(main_token_ids.shape[0])
        embed_torch = self.embed_w_cpu[main_token_ids.long()]  # [actual_batch, hidden]
        if actual_batch < mtp_batch:
            pad = torch.zeros(mtp_batch - actual_batch, embed_torch.shape[1], dtype=embed_torch.dtype)
            embed_torch = torch.cat([embed_torch, pad], dim=0)
        host_embed = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1,1,mtp_batch,hidden]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,  # replicate (not DP-sharded — embedding is replicated)
        )
        ttnn.copy_host_to_device_tensor(host_embed, state.mtp_embed_tt)

        # 2. Copy MTP positions (use `active` batch, not mtp_batch)
        mtp_pos_padded = mtp_positions.view(-1).contiguous().to(torch.int32)
        if len(mtp_pos_padded) < active:
            pad = torch.zeros(active - len(mtp_pos_padded), dtype=torch.int32)
            mtp_pos_padded = torch.cat([mtp_pos_padded, pad])
        elif len(mtp_pos_padded) > active:
            mtp_pos_padded = mtp_pos_padded[:active]
        host_pos = ttnn.from_torch(
            mtp_pos_padded,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pos, state.mtp_positions_tt)

        # 3. Copy MTP cos/sin/sin_neg (use `active` batch, not mtp_batch)
        mtp_pos_clamped = mtp_positions.to(torch.long).clamp(min=0, max=int(self.rope["cos_matrix_host"].shape[2]) - 1)
        if len(mtp_pos_clamped) < active:
            pad = torch.zeros(active - len(mtp_pos_clamped), dtype=torch.long)
            mtp_pos_clamped = torch.cat([mtp_pos_clamped, pad])
        elif len(mtp_pos_clamped) > active:
            mtp_pos_clamped = mtp_pos_clamped[:active]
        cos_host = self.rope["cos_matrix_host"]
        sin_host = self.rope["sin_matrix_host"]
        rope_dim = int(cos_host.shape[3])

        cos_batch_t = cos_host[0, 0, mtp_pos_clamped, :].reshape(1, active, 1, rope_dim).to(torch.bfloat16)
        sin_batch_t = sin_host[0, 0, mtp_pos_clamped, :].reshape(1, active, 1, rope_dim).to(torch.bfloat16)

        host_cos = ttnn.from_torch(cos_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rope_mapper)
        host_sin = ttnn.from_torch(sin_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rope_mapper)
        ttnn.copy_host_to_device_tensor(host_cos, state.mtp_cos_batch_tt)
        ttnn.copy_host_to_device_tensor(host_sin, state.mtp_sin_batch_tt)

        # sin_neg for addcmul RoPE fusion
        half = rope_dim // 2
        sin_neg_batch_t = torch.cat([-sin_batch_t[..., :half], sin_batch_t[..., half:]], dim=-1)
        host_sin_neg = ttnn.from_torch(
            sin_neg_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rope_mapper
        )
        ttnn.copy_host_to_device_tensor(host_sin_neg, state.mtp_sin_neg_batch_tt)

    def _copy_mtp_trace_inputs_positions_only(
        self,
        *,
        state: _DecodeTraceState,
        mtp_positions: torch.Tensor,  # [B] int32 (= main start_pos + 1)
    ) -> None:
        """Copy only MTP positions + RoPE into persistent device tensors.

        Used in in-trace embedding mode where the token embedding is computed
        inside the trace (from argmax → embedding), so only positions and RoPE
        need to be updated between replays.
        """
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        batch = int(state.batch)
        active = batch

        # DP sharding for TG mesh
        dp_batch_mapper = mapper
        rope_mapper = mapper
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0),
                    mesh_shape=mesh_shape,
                )
                rope_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 1),
                    mesh_shape=mesh_shape,
                )

        # 1. Copy MTP positions
        mtp_pos_padded = mtp_positions.view(-1).contiguous().to(torch.int32)
        if len(mtp_pos_padded) < active:
            pad = torch.zeros(active - len(mtp_pos_padded), dtype=torch.int32)
            mtp_pos_padded = torch.cat([mtp_pos_padded, pad])
        elif len(mtp_pos_padded) > active:
            mtp_pos_padded = mtp_pos_padded[:active]
        host_pos = ttnn.from_torch(
            mtp_pos_padded,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pos, state.mtp_positions_tt)

        # 2. Copy MTP cos/sin/sin_neg
        mtp_pos_clamped = mtp_positions.to(torch.long).clamp(min=0, max=int(self.rope["cos_matrix_host"].shape[2]) - 1)
        if len(mtp_pos_clamped) < active:
            pad = torch.zeros(active - len(mtp_pos_clamped), dtype=torch.long)
            mtp_pos_clamped = torch.cat([mtp_pos_clamped, pad])
        elif len(mtp_pos_clamped) > active:
            mtp_pos_clamped = mtp_pos_clamped[:active]
        cos_host = self.rope["cos_matrix_host"]
        sin_host = self.rope["sin_matrix_host"]
        rope_dim = int(cos_host.shape[3])

        cos_batch_t = cos_host[0, 0, mtp_pos_clamped, :].reshape(1, active, 1, rope_dim).to(torch.bfloat16)
        sin_batch_t = sin_host[0, 0, mtp_pos_clamped, :].reshape(1, active, 1, rope_dim).to(torch.bfloat16)

        host_cos = ttnn.from_torch(cos_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rope_mapper)
        host_sin = ttnn.from_torch(sin_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rope_mapper)
        ttnn.copy_host_to_device_tensor(host_cos, state.mtp_cos_batch_tt)
        ttnn.copy_host_to_device_tensor(host_sin, state.mtp_sin_batch_tt)

        # sin_neg for addcmul RoPE fusion
        half = rope_dim // 2
        sin_neg_batch_t = torch.cat([-sin_batch_t[..., :half], sin_batch_t[..., half:]], dim=-1)
        host_sin_neg = ttnn.from_torch(
            sin_neg_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rope_mapper
        )
        ttnn.copy_host_to_device_tensor(host_sin_neg, state.mtp_sin_neg_batch_tt)

    # -------------------------------------------------------------------
    # Decode Trace (capture/replay)
    # -------------------------------------------------------------------

    def _decode_trace(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
    ) -> Any:
        """Decode using trace capture/replay for performance.

        On first call with a given batch size, captures a trace.
        On subsequent calls, replays the trace with updated inputs.
        """
        active = int(tokens.shape[0])
        page_table_w = int(page_table.shape[1])

        # Round batch size UP to nearest pre-defined bucket to avoid
        # unnecessary trace recaptures when requests complete mid-batch.
        # E.g., 31 active users → use bucket 32's trace (pad extra slot).
        _DECODE_BUCKETS = [1, 4, 8, 16, 32]
        # Verify traces: bucket=4 minimum for DP=4 compatibility (batch=2 triggers
        # subtile broadcast error on TG mesh). With 2 real entries padded to 4,
        # DP=4 slices 1 per group: DP0=main, DP1=draft, DP2-3=pad.
        _VERIFY_BUCKETS = [4, 8, 16, 32]

        is_verify_batch = getattr(self, "_is_verify_batch", False)
        buckets = _VERIFY_BUCKETS if is_verify_batch else _DECODE_BUCKETS
        bucket = next((b for b in buckets if b >= active), active)

        # Pad inputs to bucket size if needed.
        if active < bucket:
            pad = bucket - active
            if is_verify_batch and active >= 2:
                # Pad by replicating the last PAIR to keep [main, draft] structure.
                last_pair_t = tokens[-2:]
                last_pair_p = positions[-2:]
                last_pair_pt = page_table[-2:]
                n_pairs = (pad + 1) // 2
                tokens = torch.cat([tokens, last_pair_t.repeat(n_pairs, 1)[:pad]], dim=0)
                positions = torch.cat([positions, last_pair_p.repeat(n_pairs)[:pad]], dim=0)
                page_table = torch.cat([page_table, last_pair_pt.repeat(n_pairs, 1)[:pad]], dim=0)
            else:
                tokens = torch.cat([tokens, tokens[-1:].expand(pad, -1)], dim=0)
                positions = torch.cat([positions, positions[-1:].expand(pad)], dim=0)
                page_table = torch.cat([page_table, page_table[-1:].expand(pad, -1)], dim=0)

        trace_cache_key = f"verify_{bucket}" if is_verify_batch else f"decode_{bucket}"
        state = self._decode_trace_states.get(trace_cache_key)

        if state is None or state.trace_id is None:
            # Release any existing traces before capturing a new one.
            # TTNN forbids device allocations while a trace is active.
            self._release_all_decode_traces()

            # Capture a new trace for this batch bucket.
            try:
                state = self._capture_decode_trace(
                    tokens=tokens,
                    positions=positions,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    sampling_params=sampling_params,
                    active=bucket,
                )
                self._decode_trace_states[trace_cache_key] = state
            except RuntimeError as e:
                if "trace_region_size" in str(e) or "trace buffers" in str(e).lower():
                    logger.warning(
                        "Trace capture failed for batch={} (trace region too small). "
                        "Falling back to eager decode for this bucket.",
                        bucket,
                    )
                    return self._decode_eager(
                        tokens=tokens[:active],
                        positions=positions[:active],
                        page_table=page_table[:active],
                        kv_cache=kv_cache,
                        sampling_params=sampling_params,
                    )
                raise
        else:
            _PROF = os.environ.get("GLM4_MOE_PROFILE", "0") != "0"

            # Update persistent inputs and replay.
            _t0 = time.perf_counter_ns() if _PROF else 0
            self._update_trace_inputs(state, tokens, positions, page_table)

            # MTP input update before trace replay.
            if self.mtp_enabled and state.mtp_logits_tt is not None and active <= self.mtp_max_batch:
                mtp_positions = positions[:active] + 1
                if self.embed_w is not None:
                    self._copy_mtp_trace_inputs_positions_only(
                        state=state,
                        mtp_positions=mtp_positions,
                    )
                else:
                    prev_main_ids = getattr(self, "_prev_main_ids", None)
                    if prev_main_ids is not None:
                        self._copy_mtp_trace_inputs(
                            state=state,
                            main_token_ids=prev_main_ids[:active],
                            mtp_positions=mtp_positions,
                        )
            _t1 = time.perf_counter_ns() if _PROF else 0

            if self.tt_ccl is not None:
                self.tt_ccl.reset_sem_counters()
            ttnn.synchronize_device(self.device)
            _t2 = time.perf_counter_ns() if _PROF else 0
            ttnn.execute_trace(self.device, state.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)
            _t3 = time.perf_counter_ns() if _PROF else 0

            if _PROF:
                logger.info(
                    "TTPROF trace_replay h2d={:.1f}ms sync={:.1f}ms exec={:.1f}ms",
                    (_t1 - _t0) / 1e6,
                    (_t2 - _t1) / 1e6,
                    (_t3 - _t2) / 1e6,
                )

        # Read outputs (use real active count, not padded bucket).
        # First, get main model token IDs (needed for both return AND MTP).
        main_ids = None
        cached_logits_host = None
        if sampling_params is not None and state.top1_indices_tt is not None:
            next_ids_torch = (
                _tt_to_torch_for_vllm_output(tensor=state.top1_indices_tt, device=self.device)
                .reshape(-1)
                .to(dtype=torch.int32)
                .cpu()
            )
            main_ids = next_ids_torch[:active]
        elif state.logits_tt is not None:
            if sampling_params is not None:
                vocab = int(self.hparams.vocab_size)
                if _is_mesh_device(self.device):
                    main_ids = self._host_argmax_from_trace_logits(state.logits_tt, active, vocab)
                else:
                    main_ids = self._sample_from_trace_logits(state.logits_tt, active)
            elif self.mtp_enabled and active <= self.mtp_max_batch:
                vocab = int(self.hparams.vocab_size)
                logits_host = self._logits_to_host(state.logits_tt, active, vocab)
                cached_logits_host = logits_host
                main_ids = logits_host.reshape(active, -1)[:, :vocab].argmax(dim=-1).to(torch.int32)

        # Store main_ids for next step's MTP input (Approach A: previous-step embedding)
        if main_ids is not None:
            object.__setattr__(self, "_prev_main_ids", main_ids.cpu().clone())

        # Diagnostic: verify in-trace argmax gives same token as host-side argmax.
        # The in-trace argmax operates on all_gathered full-vocab logits. The host-side
        # sampling reads per-shard logits and reduces. If they differ, the all_gather
        # or argmax is wrong on TG mesh.
        if self.embed_w is not None and main_ids is not None and state.logits_tt is not None:
            diag_count = getattr(self, "_intrace_diag_count", 0)
            if diag_count < 10:
                try:
                    # Host-side full-vocab argmax for comparison
                    vocab = int(self.hparams.vocab_size)
                    full_logits = self._logits_to_host(state.logits_tt, active, vocab)
                    host_full_argmax = full_logits.reshape(active, -1)[:, :vocab].argmax(dim=-1).to(torch.int32)
                    logger.info(
                        "INTRACE-DIAG: host_argmax={} main_ids={} match={}",
                        host_full_argmax[:4].tolist(),
                        main_ids[:4].tolist(),
                        (host_full_argmax[:active] == main_ids[:active]).all().item(),
                    )
                except Exception as e:
                    logger.warning("INTRACE-DIAG failed: {}", e)
                object.__setattr__(self, "_intrace_diag_count", diag_count + 1)

        # MTP: read draft tokens from combined trace output (MTP ran inside the trace)
        object.__setattr__(self, "_last_draft_token_ids", None)
        if (
            self.mtp_enabled
            and state.mtp_logits_tt is not None
            and main_ids is not None
            and active <= self.mtp_max_batch
        ):
            try:
                # MTP acceptance tracking
                prev = self._prev_draft_ids
                if prev is not None:
                    n = min(len(prev), len(main_ids))
                    if n > 0:
                        match = int((prev[:n] == main_ids[:n]).sum().item())
                        if not hasattr(self, "_mtp_total"):
                            object.__setattr__(self, "_mtp_total", 0)
                            object.__setattr__(self, "_mtp_accepted", 0)
                        object.__setattr__(self, "_mtp_total", self._mtp_total + n)
                        object.__setattr__(self, "_mtp_accepted", self._mtp_accepted + match)
                        if self._mtp_total < 20:
                            logger.info(
                                "MTP diag: prev_draft={} main_ids={} match={}/{}",
                                prev[: min(n, 4)].tolist(),
                                main_ids[: min(n, 4)].tolist(),
                                match,
                                n,
                            )
                        if self._mtp_total <= 5 or self._mtp_total % 500 == 0:
                            mode = "intrace-embed" if self.embed_w is not None else "prev-step"
                            logger.info(
                                "MTP acceptance [{}]: {}/{} = {:.1%}",
                                mode,
                                self._mtp_accepted,
                                self._mtp_total,
                                self._mtp_accepted / self._mtp_total,
                            )
                    object.__setattr__(self, "_prev_draft_ids", None)

                # Read MTP logits from combined trace output (no separate trace execution needed)
                vocab = int(self.hparams.vocab_size)
                mtp_logits_host = self._logits_to_host(state.mtp_logits_tt, active, vocab)
                draft_token_ids = mtp_logits_host.reshape(active, -1)[:, :vocab].argmax(dim=-1).to(torch.int32)
                if not hasattr(self, "_mtp_diag_count"):
                    object.__setattr__(self, "_mtp_diag_count", 0)
                if self._mtp_diag_count < 10:
                    logger.info("MTP draft [combined-trace]: ids={}", draft_token_ids[:4].tolist())
                    object.__setattr__(self, "_mtp_diag_count", self._mtp_diag_count + 1)
                object.__setattr__(self, "_last_draft_token_ids", draft_token_ids)

                if self._last_draft_token_ids is not None:
                    object.__setattr__(self, "_prev_draft_ids", self._last_draft_token_ids.cpu().clone())
            except Exception as e:
                logger.warning("MTP combined-trace read failed (non-fatal): {}\n{}", e, traceback.format_exc())
                object.__setattr__(self, "_last_draft_token_ids", None)
        elif self.mtp_enabled and active > self.mtp_max_batch and not self._mtp_batch_skip_logged:
            logger.info(
                "MTP skipped: active batch {} > mtp_max_batch {} (GLM4_MOE_MTP_MAX_BATCH)", active, self.mtp_max_batch
            )
            object.__setattr__(self, "_mtp_batch_skip_logged", True)

        # Return main model output
        if main_ids is not None and sampling_params is not None:
            return main_ids

        if state.logits_tt is not None:
            vocab = int(self.hparams.vocab_size)
            if cached_logits_host is not None:
                logits_host = cached_logits_host
            else:
                logits_host = self._logits_to_host(state.logits_tt, active, vocab)
            return logits_host

        return torch.zeros((active, 1, int(self.hparams.vocab_size)), dtype=torch.float32)

    def _capture_decode_trace(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None,
        active: int,
    ) -> _DecodeTraceState:
        """Capture a decode trace for the given batch size."""
        logger.info("=== _capture_decode_trace: starting, batch={} ===", active)
        hidden = int(self.hparams.hidden_size)
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None

        # On TG (2D mesh) with multi-user batch, shard batch-dimension inputs (cos/sin,
        # page_table, positions) across DP groups so each group gets only its subset.
        # Required because attention slices QKV per DP group: Q/K have
        # logical batch = batch_per_group (not full batch).
        dp_shard_axis = None
        dp_batch_mapper = mapper  # replicate by default
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                dp_shard_axis = 1
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0),
                    mesh_shape=mesh_shape,
                )

        # Create persistent input tensors (BEFORE trace capture AND warm-up).
        page_table_tt = ttnn.from_torch(
            page_table.contiguous().clone(),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        tt_positions, cos_batch, sin_batch, sin_neg_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device,
            rope=self.rope,
            positions=positions,
            dp_shard_axis=dp_shard_axis,
        )

        # Pre-allocate persistent token buffer on device (outside trace).
        tokens_tt = ttnn.from_torch(
            tokens.contiguous().clone().to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        # Compute rope_mapper for MTP (DP-sharded cos/sin on batch dim)
        rope_mapper = mapper  # replicate by default
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                mesh_shape = list(self.device.shape)
                rope_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 1),
                    mesh_shape=mesh_shape,
                )

        # Embedding strategy: use in-trace ttnn.embedding when embed_w is on device,
        # otherwise fall back to host-side lookup + pre-allocated device buffer.
        use_intrace_main_embed = self.embed_w is not None
        embed_tt = None
        if use_intrace_main_embed:
            logger.info("Using in-trace ttnn.embedding for main model (batch={})", active)
        else:
            # Pre-allocate embedding tensor BEFORE compile warm-up.
            embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
            embed_tt = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

        # Allocate MTP persistent tensors (before compile warm-up)
        mtp_embed_tt = None
        mtp_positions_tt = None
        mtp_cos_batch_tt = None
        mtp_sin_batch_tt = None
        mtp_sin_neg_batch_tt = None
        mtp_hidden_tt = None
        mtp_trace_id = None
        mtp_logits_tt = None

        mtp_traced = os.environ.get("GLM4_MOE_MTP_TRACED", "").strip() != "0"  # default ON

        if self.mtp_enabled:
            rope_dim = int(self.hparams.head_dim * self.hparams.partial_rotary_factor)
            # MTP batch = active batch. TILE_LAYOUT handles physical padding internally.
            # mtp_embed_tt must match mtp_hidden_tt's logical batch for ttnn.add compatibility.
            mtp_batch = active

            mtp_embed_tt = ttnn.from_torch(
                torch.zeros(1, 1, mtp_batch, hidden, dtype=torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            # Positions, cos, sin use `active` (NOT mtp_batch) because the decoder layer's
            # attention uses active_batch for batch slicing, and cos/sin must match q/k batch.
            mtp_positions_tt = ttnn.from_torch(
                torch.zeros(active, dtype=torch.int32),
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=dp_batch_mapper,
            )
            mtp_cos_batch_tt = ttnn.from_torch(
                torch.zeros(1, active, 1, rope_dim, dtype=torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rope_mapper,
            )
            mtp_sin_batch_tt = ttnn.from_torch(
                torch.zeros(1, active, 1, rope_dim, dtype=torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rope_mapper,
            )
            mtp_sin_neg_batch_tt = ttnn.from_torch(
                torch.zeros(1, active, 1, rope_dim, dtype=torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rope_mapper,
            )

        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"], sin_neg_batch)

        # Prefetcher shorthand variables — set before compile warmup so warmup
        # matches trace exactly (same SubDevice configs, same buffer shapes).
        _gcb = None  # global circular buffer
        _sdid = None  # worker sub_device_id
        _pf_qkv_pc = None
        _pf_oproj_pc = None
        _pf_qkv_in_mc = None
        _pf_qkv_out_mc = None
        _pf_oproj_in_mc = None
        _pf_oproj_out_mc = None
        _worker_scg = None

        # Load SubDevice manager BEFORE compile warmup so that warmup ops
        # create buffers with the same configuration as trace capture.
        # (Matches Llama Galaxy pattern: SubDevice loaded at model init.)
        if self.prefetcher:
            logger.info("=== PREFETCH: calling ensure_ready() ===")
            self.prefetcher.ensure_ready()
            logger.info("=== PREFETCH: ensure_ready() done, compiling prefetcher program ===")
            self.prefetcher.compile_prefetch()
            logger.info("=== PREFETCH: compile_prefetch() done, reading configs ===")
            _gcb = self.prefetcher.global_circular_buffer
            _sdid = self.prefetcher.worker_sub_device_id
            _pf_qkv_pc = self.prefetcher.qkv_program_config
            _pf_oproj_pc = self.prefetcher.oproj_program_config
            _pf_qkv_in_mc = self.prefetcher.qkv_input_mem_cfg
            _pf_qkv_out_mc = self.prefetcher.qkv_output_mem_cfg
            _pf_oproj_in_mc = self.prefetcher.oproj_input_mem_cfg
            _pf_oproj_out_mc = self.prefetcher.oproj_output_mem_cfg
            # Worker sub_core_grids: cols 0-5 (avoid sender cols 6-7)
            _worker_scg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 8))])
            logger.info("=== PREFETCH: configs read, gcb={}, sdid={} ===", _gcb, _sdid)

        # Run forward once (compile warm-up).
        # With prefetcher: runs WITH SubDevice active so buffer configs match trace.
        # Without prefetcher: runs without SubDevice (original path).
        logger.info("=== _capture_decode_trace: starting compile warmup ===")
        if use_intrace_main_embed:
            # In-trace embedding: ttnn.embedding(tokens_tt, embed_w) — same as trace capture.
            logger.info("  warmup: embedding (host-side for SubDevice compat)...")
            # Host-side embedding: ttnn.embedding on large vocab (151K) dispatches to
            # full device grid, crashing with SubDevice active. Do CPU lookup + upload.
            tokens_host = ttnn.to_torch(tokens_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            tokens_1d = tokens_host[0].to(torch.int32).flatten()[:active]
            embed_host = self.embed_w_cpu[tokens_1d.long()]  # [B, hidden]
            x = ttnn.from_torch(
                embed_host.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1,1,B,hidden]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )
            logger.info("  warmup: embed block done (host-side)")
        else:
            # Use embed_tt directly (no copy). Skip dealloc of layer 0 input in loop below.
            x = embed_tt

        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(
                x,
                tt_positions,
                rot_mats,
                page_table_tt,
                kv_cache[layer_idx],
                mode="decode",
                active_batch=active,
                global_cb=_gcb,
                sub_device_id=_sdid,
                prefetch_qkv_pc=_pf_qkv_pc,
                prefetch_oproj_pc=_pf_oproj_pc,
                prefetch_qkv_in_mc=_pf_qkv_in_mc,
                prefetch_qkv_out_mc=_pf_qkv_out_mc,
                prefetch_oproj_in_mc=_pf_oproj_in_mc,
                prefetch_oproj_out_mc=_pf_oproj_out_mc,
            )
            if x is not embed_tt:
                ttnn.deallocate(x, force=False)
            x = x_next

        # MTP compile warm-up: clone hidden state and run MTP in SAME sequence as trace
        mtp_hidden_warmup = None
        if self.mtp_enabled:
            mtp_hidden_warmup = ttnn.typecast(x, dtype=x.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Final norm: use worker core range when prefetcher is active (matches trace path)
        _warmup_norm_cr = None
        if _gcb is not None:
            _warmup_norm_cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))])
        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size), worker_core_range=_warmup_norm_cr)
        logits_tt = ttnn.linear(x, self.lm_head_w)

        # In-trace argmax compile warm-up: include sampling ops for ALL devices (including TG mesh).
        # The "TG mesh sampling broken" belief was stale — argmax works correctly inside trace on TG
        # (proven by MTP in-trace argmax producing correct INTRACE-DIAG match=True results).
        top1_values_tt = None
        top1_indices_tt = None
        if sampling_params is not None:
            tp_axis, tp_size = _tp_axis_and_size(self.device)
            vocab = int(self.hparams.vocab_size)
            if self.lm_head_sharded_vocab and is_mesh and tp_size > 1:
                # All-gather vocab shards → full vocab → argmax (same as MTP path)
                logits_gathered_warm = _simple_all_gather(
                    logits_tt,
                    self.device,
                    cluster_axis=tp_axis,
                    dim=3,
                    subdevice_id=_sdid,
                )
                logits_rm_warm = ttnn.to_layout(logits_gathered_warm, ttnn.ROW_MAJOR_LAYOUT)
                logits_rm_view_warm = ttnn.slice(logits_rm_warm, [0, 0, 0, 0], [1, 1, active, vocab])
                top1_indices_tt = ttnn.argmax(logits_rm_view_warm, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(logits_rm_warm, force=False)
                ttnn.deallocate(logits_gathered_warm, force=False)
                ttnn.deallocate(top1_indices_tt, force=False)
                top1_indices_tt = None  # freed; will be captured fresh in trace
            else:
                logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
                logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
                logits_rm_tight = ttnn.typecast(
                    logits_rm_view, dtype=logits_rm_view.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
                if isinstance(max_out, tuple):
                    top1_values_tt, top1_indices_tt = max_out
                else:
                    top1_values_tt = max_out
                    top1_indices_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(logits_rm_tight, force=False)
                ttnn.deallocate(logits_rm, force=False)

        # MTP compile warm-up: run MTP forward in the SAME sequence as trace capture
        # (main layers → clone(x) → norm → LM head → [argmax → embedding] → MTP forward).
        # This matches the combined trace's op sequence so programs are pre-compiled.
        mtp_use_intrace_embed = self.mtp_enabled and mtp_traced and self.embed_w is not None
        if self.mtp_enabled and mtp_traced and mtp_hidden_warmup is not None:
            try:
                mtp_current_embed_warmup = None
                if mtp_use_intrace_embed:
                    # In-trace embedding chain: all_gather → argmax → embedding
                    # Must match trace capture sequence exactly.
                    tp_axis, tp_size = _tp_axis_and_size(self.device)
                    vocab = int(self.hparams.vocab_size)
                    if self.lm_head_sharded_vocab and is_mesh and tp_size > 1:
                        logits_gathered = _simple_all_gather(
                            logits_tt,
                            self.device,
                            cluster_axis=tp_axis,
                            dim=3,
                            subdevice_id=_sdid,
                        )
                    else:
                        logits_gathered = logits_tt
                    logits_rm_mtp = ttnn.to_layout(logits_gathered, ttnn.ROW_MAJOR_LAYOUT)
                    logits_rm_mtp_view = ttnn.slice(logits_rm_mtp, [0, 0, 0, 0], [1, 1, active, vocab])
                    mtp_token_ids_tt = ttnn.argmax(logits_rm_mtp_view, dim=3, keepdim=False, use_multicore=True)
                    ttnn.deallocate(logits_rm_mtp, force=False)
                    if logits_gathered is not logits_tt:
                        ttnn.deallocate(logits_gathered, force=False)
                    # Reshape argmax output for embedding: [1,1,B,1] → [B,1]
                    mtp_token_ids_2d = ttnn.reshape(mtp_token_ids_tt, (active, 1))
                    mtp_current_embed_warmup = ttnn.embedding(
                        mtp_token_ids_2d,
                        self.embed_w,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    # Reshape to [1,1,B,hidden] for MTP forward
                    mtp_current_embed_warmup = ttnn.reshape(mtp_current_embed_warmup, (1, 1, active, hidden))
                    ttnn.deallocate(mtp_token_ids_tt, force=False)
                    ttnn.deallocate(mtp_token_ids_2d, force=False)
                    logger.info("MTP in-trace embedding warm-up completed for batch={}", active)

                mtp_state_warmup = _DecodeTraceState(
                    batch=active,
                    page_table_tt=page_table_tt,
                    mtp_hidden_tt=mtp_hidden_warmup,
                    mtp_embed_tt=mtp_embed_tt,
                    mtp_positions_tt=mtp_positions_tt,
                    mtp_cos_batch_tt=mtp_cos_batch_tt,
                    mtp_sin_batch_tt=mtp_sin_batch_tt,
                    mtp_sin_neg_batch_tt=mtp_sin_neg_batch_tt,
                )
                mtp_logits_warm = self._mtp_decode_step_tt(
                    state=mtp_state_warmup,
                    kv_cache=kv_cache,
                    mtp_current_embed_tt=mtp_current_embed_warmup,
                )
                ttnn.synchronize_device(self.device)
                ttnn.deallocate(mtp_logits_warm, force=True)
                if mtp_current_embed_warmup is not None:
                    ttnn.deallocate(mtp_current_embed_warmup, force=True)
                logger.info("MTP compile warm-up completed for batch={}", active)
            except Exception as e:
                logger.warning("MTP compile warm-up failed, will use eager fallback: {}", e)
                import traceback as _tb

                logger.warning("Traceback: {}", _tb.format_exc())
                mtp_traced = False
                mtp_use_intrace_embed = False

        # Free compile warm-up outputs
        if mtp_hidden_warmup is not None:
            ttnn.deallocate(mtp_hidden_warmup, force=True)
            mtp_hidden_warmup = None

        # Synchronize device to drain all async ops from compile-forward
        # before starting trace capture.
        logger.info("=== _capture_decode_trace: compile warmup DONE, syncing device ===")
        ttnn.synchronize_device(self.device)

        # Explicitly free compile-forward outputs (skip if already freed for MTP).
        if x is not None:
            ttnn.deallocate(x, force=True)
        if logits_tt is not None:
            ttnn.deallocate(logits_tt, force=True)
        if top1_values_tt is not None:
            ttnn.deallocate(top1_values_tt, force=True)
        if top1_indices_tt is not None:
            ttnn.deallocate(top1_indices_tt, force=True)
        x = logits_tt = top1_values_tt = top1_indices_tt = None
        ttnn.synchronize_device(self.device)

        # Re-create persistent inputs before trace capture.
        # embed_tt may have been deallocated during compile warmup (typecast same-dtype
        # can alias), so we always re-create from scratch.
        if not use_intrace_main_embed:
            embed_tt = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )
        # For in-trace embedding, tokens_tt is the persistent input — re-copy it.
        host_tokens_recopy = ttnn.from_torch(
            tokens.contiguous().clone().to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_tokens_recopy, tokens_tt)

        # Now capture trace.
        logger.info("=== _capture_decode_trace: starting trace capture ===")
        logger.info("Capturing decode trace for batch={}", active)

        # SubDevice manager already loaded before compile warmup (above).
        if self.tt_ccl is not None:
            logger.info("=== PREFETCH: resetting CCL sem counters ===")
            self.tt_ccl.reset_sem_counters()
            logger.info("=== PREFETCH: CCL sem counters reset done ===")
        logger.info("=== PREFETCH: beginning trace capture ===")
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        logger.info("=== PREFETCH: trace capture begun, trace_id={} ===", trace_id)

        # Launch async DRAM prefetch INSIDE trace so it replays on every trace execution.
        _pf_garbage = None
        if self.prefetcher:
            logger.info("=== PREFETCH: calling start_prefetch() ===")
            _pf_garbage = self.prefetcher.start_prefetch()
            logger.info("=== PREFETCH: start_prefetch() done ===")

        # Embedding inside trace: either in-trace ttnn.embedding or clone of pre-allocated buffer.
        if use_intrace_main_embed:
            x = ttnn.embedding(
                tokens_tt,
                self.embed_w,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Skip to_layout — embedding already requests TILE_LAYOUT.
            # ttnn.to_layout crashes with SubDevice (dispatches to full grid).
            # Reshape directly to [1,1,B,D] — avoid ttnn.permute which crashes
            # with SubDevice active (dispatches to full grid).
            x = ttnn.reshape(x, (1, 1, active, hidden))
            x = ttnn.slice(
                x, [0, 0, 0, 0], [1, 1, active, hidden], sub_core_grids=_worker_scg if self.prefetcher else None
            )
        else:
            # Use embed_tt directly — no copy needed. The embed buffer is dedicated
            # to this trace and overwritten (via write_tensor) before each execution.
            x = embed_tt

        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(
                x,
                tt_positions,
                rot_mats,
                page_table_tt,
                kv_cache[layer_idx],
                mode="decode",
                active_batch=active,
                global_cb=_gcb,
                sub_device_id=_sdid,
                prefetch_qkv_pc=_pf_qkv_pc,
                prefetch_oproj_pc=_pf_oproj_pc,
                prefetch_qkv_in_mc=_pf_qkv_in_mc,
                prefetch_qkv_out_mc=_pf_qkv_out_mc,
                prefetch_oproj_in_mc=_pf_oproj_in_mc,
                prefetch_oproj_out_mc=_pf_oproj_out_mc,
            )
            # Don't deallocate embed_tt — it's the persistent trace input buffer.
            if x is not embed_tt:
                ttnn.deallocate(x, force=False)
            x = x_next

        # MTP: multiply by 1.0 to create trace-owned copy (clone/typecast don't support sub_core_grids in trace)
        if self.mtp_enabled:
            mtp_hidden_tt = ttnn.multiply(
                x, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=_worker_scg if self.prefetcher else None
            )

        # Final norm: use worker core range when prefetcher is active
        # Must be within worker SubDevice (cols 0-5), avoiding sender cols 6,7.
        _final_norm_cr = None
        if _gcb is not None:
            _final_norm_cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))])
        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size), worker_core_range=_final_norm_cr)
        logits_tt = ttnn.linear(x, self.lm_head_w)

        # In-trace argmax for main model sampling (ALL devices including TG mesh).
        # The "TG mesh sampling broken" belief was stale — argmax works correctly
        # inside trace on TG (proven by MTP in-trace argmax match=True results).
        # Widened gate: also produce top1_indices_tt when MTP needs it for in-trace embedding
        # (Bug 3 fix: reuse main argmax for MTP instead of running a second all_gather chain).
        _need_intrace_argmax = sampling_params is not None or (
            self.mtp_enabled and mtp_traced and mtp_use_intrace_embed and mtp_hidden_tt is not None
        )
        if _need_intrace_argmax:
            tp_axis, tp_size = _tp_axis_and_size(self.device)
            vocab = int(self.hparams.vocab_size)
            if self.lm_head_sharded_vocab and is_mesh and tp_size > 1:
                # All-gather vocab shards → full vocab → argmax
                logits_gathered_main = _simple_all_gather(
                    logits_tt,
                    self.device,
                    cluster_axis=tp_axis,
                    dim=3,
                    subdevice_id=_sdid,
                )
                logits_rm_main = ttnn.to_layout(logits_gathered_main, ttnn.ROW_MAJOR_LAYOUT)
                logits_rm_main_view = ttnn.slice(logits_rm_main, [0, 0, 0, 0], [1, 1, active, vocab])
                top1_indices_tt = ttnn.argmax(logits_rm_main_view, dim=3, keepdim=False, use_multicore=True)
                top1_values_tt = None
                ttnn.deallocate(logits_rm_main, force=False)
                ttnn.deallocate(logits_gathered_main, force=False)
            else:
                logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
                logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
                logits_rm_tight = ttnn.multiply(
                    logits_rm_view,
                    1.0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    sub_core_grids=_worker_scg if self.prefetcher else None,
                )
                max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
                if isinstance(max_out, tuple):
                    top1_values_tt, top1_indices_tt = max_out
                else:
                    top1_values_tt = max_out
                    top1_indices_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
                ttnn.deallocate(logits_rm_tight, force=False)
                ttnn.deallocate(logits_rm, force=False)

        # MTP forward INSIDE the main trace (combined trace).
        # Two modes:
        #   (a) In-trace embedding (GLM4_MOE_MTP_DEVICE_EMBED=1): all_gather logits →
        #       argmax → embedding → MTP. Gets CURRENT-step token embedding. ~30% acceptance.
        #   (b) Fallback: uses previous-step draft embedding (uploaded into mtp_embed_tt
        #       between replays). First replay uses zeros → garbage draft. 0% acceptance.
        if self.mtp_enabled and mtp_traced and mtp_hidden_tt is not None:
            mtp_current_embed_trace = None
            if mtp_use_intrace_embed:
                # Bug 3 fix: reuse top1_indices_tt from main sampling chain instead of
                # running a second all_gather+argmax. Eliminates redundant CCL op and
                # prevents potential TG mesh divergence between the two chains.
                if top1_indices_tt is not None:
                    mtp_token_ids_2d = ttnn.reshape(top1_indices_tt, (active, 1))
                else:
                    # Fallback: run argmax if main chain didn't produce top1_indices_tt
                    # (shouldn't happen — gate was widened above)
                    tp_axis, tp_size = _tp_axis_and_size(self.device)
                    vocab = int(self.hparams.vocab_size)
                    if self.lm_head_sharded_vocab and is_mesh and tp_size > 1:
                        logits_gathered = _simple_all_gather(
                            logits_tt,
                            self.device,
                            cluster_axis=tp_axis,
                            dim=3,
                            subdevice_id=_sdid,
                        )
                    else:
                        logits_gathered = logits_tt
                    logits_rm_mtp = ttnn.to_layout(logits_gathered, ttnn.ROW_MAJOR_LAYOUT)
                    logits_rm_mtp_view = ttnn.slice(logits_rm_mtp, [0, 0, 0, 0], [1, 1, active, vocab])
                    mtp_token_ids_2d = ttnn.argmax(logits_rm_mtp_view, dim=3, keepdim=False, use_multicore=True)
                    mtp_token_ids_2d = ttnn.reshape(mtp_token_ids_2d, (active, 1))
                    ttnn.deallocate(logits_rm_mtp, force=False)
                    if logits_gathered is not logits_tt:
                        ttnn.deallocate(logits_gathered, force=False)
                mtp_current_embed_trace = ttnn.embedding(
                    mtp_token_ids_2d,
                    self.embed_w,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                # Reshape to [1,1,B,hidden] for MTP forward
                mtp_current_embed_trace = ttnn.reshape(mtp_current_embed_trace, (1, 1, active, hidden))
                # Don't deallocate mtp_token_ids_2d if it's a reshape of top1_indices_tt
                # (shared with main sampling chain — will be deallocated there)

            mtp_state_for_trace = _DecodeTraceState(
                batch=active,
                page_table_tt=page_table_tt,
                mtp_hidden_tt=mtp_hidden_tt,
                mtp_embed_tt=mtp_embed_tt,
                mtp_positions_tt=mtp_positions_tt,
                mtp_cos_batch_tt=mtp_cos_batch_tt,
                mtp_sin_batch_tt=mtp_sin_batch_tt,
                mtp_sin_neg_batch_tt=mtp_sin_neg_batch_tt,
            )
            mtp_logits_tt = self._mtp_decode_step_tt(
                state=mtp_state_for_trace,
                kv_cache=kv_cache,
                mtp_current_embed_tt=mtp_current_embed_trace,
            )
            logger.info(
                "MTP forward included in main trace for batch={} (intrace_embed={})", active, mtp_use_intrace_embed
            )

        # Stop prefetch inside trace (deallocate garbage, reset stall group)
        if _pf_garbage is not None:
            self.prefetcher.stop_prefetch(_pf_garbage)

        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        logger.info("Decode trace captured for batch={} (combined main+MTP)", active)

        # All devices now use in-trace argmax. Keep logits_tt for MTP diagnostics.
        return _DecodeTraceState(
            trace_id=trace_id,
            batch=active,
            page_table_width=int(page_table.shape[1]),
            tokens_tt=tokens_tt,
            positions_tt=tt_positions,
            cos_batch_tt=cos_batch,
            sin_batch_tt=sin_batch,
            sin_neg_batch_tt=sin_neg_batch,
            trans_matrix_tt=self.rope["trans_matrix"],
            page_table_tt=page_table_tt,
            logits_tt=logits_tt,
            top1_values_tt=top1_values_tt,
            top1_indices_tt=top1_indices_tt,
            embed_tt=embed_tt,
            # MTP fields
            mtp_hidden_tt=mtp_hidden_tt,
            mtp_embed_tt=mtp_embed_tt,
            mtp_positions_tt=mtp_positions_tt,
            mtp_cos_batch_tt=mtp_cos_batch_tt,
            mtp_sin_batch_tt=mtp_sin_batch_tt,
            mtp_sin_neg_batch_tt=mtp_sin_neg_batch_tt,
            mtp_page_table_tt=None,  # reuse main page_table_tt
            mtp_trace_id=mtp_trace_id,
            mtp_logits_tt=mtp_logits_tt,
        )

    def _update_trace_inputs(
        self,
        state: _DecodeTraceState,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        """Update persistent trace input tensors with new values."""
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        batch = int(tokens.shape[0])

        # On TG with multi-user batch, batch-dimension inputs (positions, cos/sin,
        # page_table) were sharded across DP groups during trace capture.
        # Must use the same sharding here for copy_host_to_device_tensor.
        dp_batch_mapper = mapper  # for page_table (dim=0), positions (dim=0)
        rope_mapper = mapper  # for cos/sin (dim=1)
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if batch > 1 and batch % dp_size == 0:
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 0),
                    mesh_shape=mesh_shape,
                )
                rope_mapper = ttnn.ShardTensor2dMesh(
                    self.device,
                    dims=(None, 1),
                    mesh_shape=mesh_shape,
                )

        # Update embedding: in-trace mode just updates tokens_tt (device does lookup);
        # fallback mode does host lookup + copy to pre-allocated embed buffer.
        if self.embed_w is not None:
            # In-trace embedding: only need token IDs (ttnn.embedding runs inside trace)
            host_tokens = ttnn.from_torch(
                tokens.contiguous().clone().to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host_tokens, state.tokens_tt)
        elif state.embed_tt is not None:
            embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
            host_embed = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mapper,
            )
            ttnn.copy_host_to_device_tensor(host_embed, state.embed_tt)

        # Update positions (host tensor, then copy).
        host_positions = ttnn.from_torch(
            positions.view(-1).contiguous().to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_positions, state.positions_tt)

        # Update cos/sin for new positions.
        cos_host = self.rope["cos_matrix_host"]
        sin_host = self.rope["sin_matrix_host"]
        rope_dim = int(cos_host.shape[3])
        positions_cpu = positions.to(torch.long).clamp(min=0, max=int(cos_host.shape[2]) - 1)
        cos_batch_t = cos_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)
        sin_batch_t = sin_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)

        host_cos = ttnn.from_torch(
            cos_batch_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        host_sin = ttnn.from_torch(
            sin_batch_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_cos, state.cos_batch_tt)
        ttnn.copy_host_to_device_tensor(host_sin, state.sin_batch_tt)

        # Update sin_neg for addcmul RoPE fusion (OPT-3).
        half = rope_dim // 2
        sin_neg_batch_t = torch.cat([-sin_batch_t[..., :half], sin_batch_t[..., half:]], dim=-1)
        host_sin_neg = ttnn.from_torch(
            sin_neg_batch_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_sin_neg, state.sin_neg_batch_tt)

        # Update page table (host tensor, then copy).
        host_pt = ttnn.from_torch(
            page_table.contiguous().clone(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pt, state.page_table_tt)

    def _release_all_decode_traces(self) -> None:
        """Release ALL bucket decode traces and deallocate trace output tensors."""
        if not self._decode_trace_states:
            return
        ttnn.synchronize_device(self.device)
        for bucket, state in list(self._decode_trace_states.items()):
            # Release MTP trace first
            if state.mtp_trace_id is not None:
                try:
                    ttnn.release_trace(self.device, state.mtp_trace_id)
                except Exception:
                    pass
                state.mtp_trace_id = None

            # Release main trace
            if state.trace_id is not None:
                try:
                    ttnn.release_trace(self.device, state.trace_id)
                except Exception:
                    pass
                state.trace_id = None

            # Deallocate all tensors (existing + MTP)
            for t in (
                state.logits_tt,
                state.top1_values_tt,
                state.top1_indices_tt,
                state.tokens_tt,
                state.positions_tt,
                state.cos_batch_tt,
                state.sin_batch_tt,
                state.sin_neg_batch_tt,
                state.page_table_tt,
                state.embed_tt,
                # MTP tensors
                state.mtp_embed_tt,
                state.mtp_positions_tt,
                state.mtp_cos_batch_tt,
                state.mtp_sin_batch_tt,
                state.mtp_sin_neg_batch_tt,
                state.mtp_page_table_tt,
                state.mtp_logits_tt,
                # mtp_hidden_tt is trace-owned — released with main trace
            ):
                if t is not None:
                    try:
                        ttnn.deallocate(t, force=True)
                    except Exception:
                        pass
        self._decode_trace_states.clear()
        # Reset CCL semaphore counters — decode trace replay advances counters,
        # and the next prefill's all_reduce needs them starting at 0.
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()
        ttnn.synchronize_device(self.device)

    # -------------------------------------------------------------------
    # Prefill
    # -------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list,
        seq_pad_multiple: int = 128,
    ) -> torch.Tensor:
        """Compute logits for the last prompt token for each request and fill KV caches.

        Processes each request independently (B loop). For long prompts (>16K tokens),
        uses chunked prefill to avoid activation memory OOM.

        Args:
            tokens: [B, S] int32
            prompt_lens: length B, actual prompt lengths
            page_table: [B, W] int32
            kv_cache: list of [cache_k, cache_v] per layer
            seq_pad_multiple: pad prompt lengths to this multiple

        Returns:
            Logits [B, 1, vocab] as torch float32
        """
        # Release decode traces before prefill to avoid buffer corruption.
        # Prefill allocates device buffers that can overlap with trace-owned buffers,
        # producing garbled output on subsequent trace replays.
        # With synchronize_device before trace capture (line ~931), re-capture is safe.
        self._release_all_decode_traces()  # Required: prefill buffers overlap with trace-owned buffers

        if tokens.ndim != 2:
            raise ValueError(f"expected tokens [B,S], got {tuple(tokens.shape)}")
        if page_table.ndim != 2:
            raise ValueError(f"expected page_table [B,W], got {tuple(page_table.shape)}")
        batch, seq_total = tokens.shape
        if len(prompt_lens) != int(batch):
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {batch}")

        hidden = int(self.hparams.hidden_size)
        vocab = int(self.hparams.vocab_size)
        rope_dim = int(self.hparams.head_dim * self.hparams.partial_rotary_factor)
        pad_multiple = max(128, int(seq_pad_multiple))
        is_mesh = _is_mesh_device(self.device)

        # Prefill chunk size (default 128K; override with GLM4_MOE_PREFILL_CHUNK_SIZE; lower to reduce peak activation memory).
        PREFILL_CHUNK_SIZE = int(os.environ.get("GLM4_MOE_PREFILL_CHUNK_SIZE", "131072") or "131072")

        out_logits: list[torch.Tensor] = []

        for i in range(int(batch)):
            prompt_len = int(prompt_lens[i])
            if prompt_len <= 0:
                out_logits.append(torch.zeros((1, vocab), dtype=torch.float32))
                continue

            padded_len = ((prompt_len + pad_multiple - 1) // pad_multiple) * pad_multiple
            padded_len = min(padded_len, int(self.max_seq_len))

            prompt_ids = tokens[i, :prompt_len].to(torch.int32).cpu()
            input_padded = torch.zeros((1, padded_len), dtype=torch.int32)
            input_padded[0, :prompt_len] = prompt_ids

            # Page table for this request.
            page_row = page_table[i : i + 1, :].to(torch.int32)
            page_table_tt = ttnn.from_torch(
                page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            # Slice RoPE tables.
            rope_slices_owned = True
            if int(padded_len) == int(self.rope["cos_matrix"].shape[2]) and int(rope_dim) == int(
                self.rope["cos_matrix"].shape[3]
            ):
                cos_matrix = self.rope["cos_matrix"]
                sin_matrix = self.rope["sin_matrix"]
                rope_slices_owned = False
            else:
                cos_matrix = ttnn.slice(self.rope["cos_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim])
                sin_matrix = ttnn.slice(self.rope["sin_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim])

            # DEBUG: sync checkpoints for prefill pipeline
            _dbg = os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"

            def _psync(label):
                if _dbg:
                    import sys

                    print(f"  [DEBUG PREFILL-MODEL] {label} ...", flush=True, file=sys.stderr)
                    ttnn.synchronize_device(self.device)
                    print(f"  [DEBUG PREFILL-MODEL] {label} OK", flush=True, file=sys.stderr)

            _psync("after page_table + rope_slice")

            # Embedding: do host-side lookup to avoid device-side tile conversion hang on TG.
            # ttnn.embedding with TILE_LAYOUT and ttnn.to_layout both hang on 32-device TG mesh.
            embed_torch = self.embed_w_cpu[input_padded[0].long()]  # [padded_len, hidden]
            x = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, padded_len, hidden]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )
            _psync("after embedding (host)")

            rot_mats = (cos_matrix, sin_matrix, self.rope["trans_matrix"])

            _psync("before layer loop")

            # Chunked prefill: process PREFILL_CHUNK_SIZE tokens at a time.
            if padded_len > PREFILL_CHUNK_SIZE:
                x = self._prefill_chunked(
                    x=x,
                    padded_len=padded_len,
                    prompt_len=prompt_len,
                    page_table_tt=page_table_tt,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    rot_mats=rot_mats,
                    user_id=i,
                    chunk_size=PREFILL_CHUNK_SIZE,
                )
            else:
                # Single-pass prefill.
                for layer_idx in range(self.num_layers_to_run):
                    dl = self.decoder_layers[layer_idx]
                    x_next = dl.forward(
                        x,
                        None,
                        rot_mats,
                        page_table_tt,
                        kv_cache[layer_idx],
                        mode="prefill",
                    )
                    ttnn.deallocate(x, force=False)
                    x = x_next

            x_last = ttnn.slice(x, [0, 0, prompt_len - 1, 0], [1, 1, prompt_len, hidden])
            ttnn.deallocate(x, force=False)

            x_last = _sharded_rms_norm(x_last, self.final_norm, int(self.hparams.hidden_size))
            logits_tt = ttnn.linear(x_last, self.lm_head_w)

            # Assemble vocab-sharded logits on host (NOT all_gather — all_gather
            # corrupts shard ordering on 2D mesh, inflating Chinese token scores).
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                shards = ttnn.get_device_tensors(logits_tt)
                num_shards = len(shards)
                tp_size = self.lm_head_tp_size
                if num_shards == tp_size:
                    tp_shards = list(shards)
                elif num_shards > tp_size:
                    dp_stride = num_shards // tp_size
                    tp_shards = [shards[i * dp_stride] for i in range(tp_size)]
                else:
                    tp_shards = list(shards)
                logits_shards = [ttnn.to_torch(t.cpu())[..., : int(t.shape[-1])] for t in tp_shards]
                logits_torch = torch.cat(logits_shards, dim=-1)[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            else:
                logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
                logits_torch = logits_torch[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            logits_i = logits_flat.to(dtype=torch.float32).cpu()
            out_logits.append(logits_i)

            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(x_last, force=False)
            if rope_slices_owned:
                ttnn.deallocate(cos_matrix, force=False)
                ttnn.deallocate(sin_matrix, force=False)
            ttnn.deallocate(page_table_tt, force=False)

            # Reset CCL semaphore counters after each prefill to keep state consistent.
            if self.tt_ccl is not None:
                self.tt_ccl.reset_sem_counters()

        return torch.stack(out_logits, dim=0)  # [B, 1, vocab]

    def _prefill_chunked(
        self,
        *,
        x: ttnn.Tensor,
        padded_len: int,
        prompt_len: int,
        page_table_tt: ttnn.Tensor,
        page_table,  # raw torch tensor for host-side page table slicing
        kv_cache: list,
        rot_mats: tuple,
        user_id: int,
        chunk_size: int,
    ) -> ttnn.Tensor:
        """Run prefill in chunks for long sequences to avoid activation OOM.

        Processes chunk_size tokens at a time through all layers, writing KV cache
        incrementally. Returns the full hidden state [1,1,padded_len,hidden].
        """
        import sys as _sys

        _dbg_prefill = os.environ.get("GLM4_MOE_DEBUG_PREFILL", "0") != "0"
        hidden = int(self.hparams.hidden_size)
        num_chunks = (padded_len + chunk_size - 1) // chunk_size

        if _dbg_prefill:
            print(f"\n{'='*80}", flush=True, file=_sys.stderr)
            print(
                f"[PREFILL_CHUNKED] START: padded_len={padded_len}, prompt_len={prompt_len}, "
                f"chunk_size={chunk_size}, num_chunks={num_chunks}, user_id={user_id}, "
                f"num_layers={self.num_layers_to_run}",
                flush=True,
                file=_sys.stderr,
            )
            print(f"[PREFILL_CHUNKED] x.shape={list(x.shape)}, x.dtype={x.dtype}", flush=True, file=_sys.stderr)
            print(f"[PREFILL_CHUNKED] page_table_tt.shape={list(page_table_tt.shape)}", flush=True, file=_sys.stderr)
            print(f"[PREFILL_CHUNKED] rot_mats[0].shape={list(rot_mats[0].shape)} (cos)", flush=True, file=_sys.stderr)
            print(f"[PREFILL_CHUNKED] rot_mats[1].shape={list(rot_mats[1].shape)} (sin)", flush=True, file=_sys.stderr)
            print(
                f"[PREFILL_CHUNKED] kv_cache has {len(kv_cache)} layers, "
                f"kv_cache[0][0].shape={list(kv_cache[0][0].shape)} (keys)",
                flush=True,
                file=_sys.stderr,
            )
            _total_t0 = time.time()

        # Pre-compute per-chunk RoPE slices, page tables, and chunk_start_idx.
        # These are the same for every layer, so compute once outside the layer loop.
        rope_dim = rot_mats[0].shape[-1]
        is_mesh = _is_mesh_device(self.device)

        # Determine block_size from KV cache shape: kv_cache[0] = [keys, values],
        # keys.shape[2] = block_size (number of positions per page/block).
        block_size = kv_cache[0][0].shape[2]

        chunk_infos = []
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, padded_len)

            # Bug 5 fix: slice cos/sin for correct absolute positions
            cos_chunk = ttnn.slice(rot_mats[0], [0, 0, start, 0], [1, 1, end, rope_dim])
            sin_chunk = ttnn.slice(rot_mats[1], [0, 0, start, 0], [1, 1, end, rope_dim])
            chunk_rot_mats = (cos_chunk, sin_chunk, rot_mats[2])

            # Bug 2 fix: create per-chunk page table for KV cache fill.
            # The fill kernel maps input tile 0 -> page_table[0], so for chunk N
            # starting at position `start`, we need page_table entries starting at
            # start // block_size so the fill writes to the correct physical blocks.
            start_page = start // block_size
            chunk_len = end - start
            num_chunk_pages = chunk_len // block_size
            # Host-side slice of page table (page_table_tt is on device, need torch slice then re-upload)
            chunk_page_row = page_table[user_id : user_id + 1, start_page : start_page + num_chunk_pages].to(
                torch.int32
            )
            chunk_page_table_tt = ttnn.from_torch(
                chunk_page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            # Bug 1 fix: for SDPA, pass full page table covering positions 0..end-1
            # so attention can read all previously cached KV.
            end_page = end // block_size
            sdpa_page_row = page_table[user_id : user_id + 1, :end_page].to(torch.int32)
            sdpa_page_table_tt = ttnn.from_torch(
                sdpa_page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            chunk_infos.append(
                {
                    "start": start,
                    "end": end,
                    "rot_mats": chunk_rot_mats,
                    "chunk_page_table": chunk_page_table_tt,
                    "sdpa_page_table": sdpa_page_table_tt,
                    "chunk_start_idx": start,
                }
            )

        if _dbg_prefill:
            print(
                f"[PREFILL_CHUNKED] Pre-computed {len(chunk_infos)} chunk infos, " f"block_size={block_size}",
                flush=True,
                file=_sys.stderr,
            )

        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next_chunks = []

            if _dbg_prefill:
                _layer_t0 = time.time()

            for chunk_idx in range(num_chunks):
                ci = chunk_infos[chunk_idx]
                start = ci["start"]
                end = ci["end"]

                if _dbg_prefill:
                    _chunk_t0 = time.time()
                    print(
                        f"  [PREFILL_CHUNKED] layer={layer_idx}, chunk={chunk_idx}/{num_chunks}, "
                        f"tokens[{start}:{end}] (len={end-start})",
                        flush=True,
                        file=_sys.stderr,
                    )

                x_chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, hidden])

                if _dbg_prefill:
                    print(f"    x_chunk.shape={list(x_chunk.shape)}", flush=True, file=_sys.stderr)

                # Pass chunk-specific RoPE, page tables, and start index
                x_chunk_out = dl.forward(
                    x_chunk,
                    None,
                    ci["rot_mats"],
                    ci["sdpa_page_table"],
                    kv_cache[layer_idx],
                    mode="prefill",
                    chunk_page_table=ci["chunk_page_table"],
                    chunk_start_idx=ci["chunk_start_idx"],
                )

                if _dbg_prefill:
                    _chunk_elapsed = time.time() - _chunk_t0
                    print(
                        f"    x_chunk_out.shape={list(x_chunk_out.shape)}, " f"chunk_time={_chunk_elapsed:.3f}s",
                        flush=True,
                        file=_sys.stderr,
                    )

                x_next_chunks.append(x_chunk_out)
                ttnn.deallocate(x_chunk, force=False)

            ttnn.deallocate(x, force=False)
            if len(x_next_chunks) == 1:
                x = x_next_chunks[0]
            else:
                if _dbg_prefill:
                    _concat_t0 = time.time()
                x = ttnn.concat(x_next_chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if _dbg_prefill:
                    print(
                        f"  [PREFILL_CHUNKED] layer={layer_idx} concat {len(x_next_chunks)} chunks -> "
                        f"shape={list(x.shape)}, concat_time={time.time()-_concat_t0:.3f}s",
                        flush=True,
                        file=_sys.stderr,
                    )
                for chunk_t in x_next_chunks:
                    ttnn.deallocate(chunk_t, force=False)

            if _dbg_prefill:
                _layer_elapsed = time.time() - _layer_t0
                print(
                    f"  [PREFILL_CHUNKED] layer={layer_idx} DONE, layer_time={_layer_elapsed:.3f}s",
                    flush=True,
                    file=_sys.stderr,
                )
                # Print every 10 layers to avoid spam but show progress
                if layer_idx % 10 == 9 or layer_idx == self.num_layers_to_run - 1:
                    _total_so_far = time.time() - _total_t0
                    print(
                        f"  [PREFILL_CHUNKED] Progress: {layer_idx+1}/{self.num_layers_to_run} layers, "
                        f"elapsed={_total_so_far:.1f}s",
                        flush=True,
                        file=_sys.stderr,
                    )

        if _dbg_prefill:
            _total_elapsed = time.time() - _total_t0
            print(
                f"[PREFILL_CHUNKED] DONE: total_time={_total_elapsed:.1f}s, " f"output.shape={list(x.shape)}",
                flush=True,
                file=_sys.stderr,
            )
            print(f"{'='*80}\n", flush=True, file=_sys.stderr)

        return x
