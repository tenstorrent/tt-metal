# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in capture/replay helpers for HF decode ``rotary_embedding_hf`` (#36378).

Set ``TT_HF_ROPE_DECODE_CAPTURE_DIR`` to a writable directory; each decode step writes
``step_NNNN/`` with ``ttnn.dump_tensor`` payloads and ``manifest.json``.
See ``HF_ROPE_PCC_INVESTIGATION.md`` for usage.

Set ``TT_HF_ROPE_DECODE_INSTRUMENT_SHAPES=1`` to log Q/K ``shape``, ``logical_shape()``,
``padded_shape``, and a small memory-config digest before and after decode
``rotary_embedding_hf`` (see ``maybe_log_hf_rope_decode_tensor_shapes``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import precompute_freqs

CAPTURE_DIR_ENV = "TT_HF_ROPE_DECODE_CAPTURE_DIR"
CAPTURE_MAX_STEPS_ENV = "TT_HF_ROPE_DECODE_CAPTURE_MAX_STEPS"
CAPTURE_TORCH_SIDECAR_ENV = "TT_HF_ROPE_DECODE_CAPTURE_TORCH"
REPLAY_DIR_ENV = "TT_HF_ROPE_DECODE_CAPTURE_REPLAY_DIR"
MANIFEST_VERSION = 1

INSTRUMENT_SHAPES_ENV = "TT_HF_ROPE_DECODE_INSTRUMENT_SHAPES"
INSTRUMENT_MAX_DECODE_STEPS_ENV = "TT_HF_ROPE_DECODE_INSTRUMENT_MAX_DECODE_STEPS"


def tensor_memory_config_shard_digest(tensor: ttnn.Tensor) -> Dict[str, Any]:
    """Comparable fields for sharded tensors (decode heads / cos-sin)."""
    mem_cfg = tensor.memory_config()
    out: Dict[str, Any] = {
        "memory_layout": str(mem_cfg.memory_layout),
        "buffer_type": str(mem_cfg.buffer_type),
    }
    spec = getattr(mem_cfg, "shard_spec", None)
    if spec is not None:
        out["shard_shape"] = list(spec.shape)
        out["shard_orientation"] = str(spec.orientation)
        out["grid_repr"] = repr(spec.grid)
    else:
        out["shard_spec"] = None
    return out


def _tensor_shape_digest(tt: ttnn.Tensor) -> Dict[str, Any]:
    """Host-side view of logical vs padded and sharding (for one ttnn tensor).

    Many ``ttnn.Tensor`` Python builds do not expose ``logical_shape()``; in that case
    ``shape`` is still useful: for decode Q/K it often matches the logical head counts on
    axes that are tile-padded (compare to ``padded_shape``).
    """
    out: Dict[str, Any] = {"shape": list(tt.shape)}
    padded = getattr(tt, "padded_shape", None)
    out["padded_shape"] = list(padded) if padded is not None else None

    logical_shape_source: str
    logical_val: Any
    fn = getattr(tt, "logical_shape", None)
    if callable(fn):
        try:
            logical_val = list(fn())
            logical_shape_source = "logical_shape()"
        except Exception as exc:
            logical_val = list(tt.shape)
            logical_shape_source = f"logical_shape() failed ({exc}); using ttnn.Tensor.shape"
    else:
        logical_val = list(tt.shape)
        logical_shape_source = (
            "ttnn.Tensor.shape (no callable logical_shape(); compare to padded_shape for tile padding)"
        )

    out["logical_shape"] = logical_val
    out["logical_shape_source"] = logical_shape_source
    out["memory"] = tensor_memory_config_shard_digest(tt)
    dtype = getattr(tt, "dtype", None)
    out["dtype"] = str(dtype) if dtype is not None else None
    layout = getattr(tt, "layout", None)
    out["layout"] = str(layout) if layout is not None else None
    return out


def tensor_device_layout_digest(tt: ttnn.Tensor) -> Dict[str, Any]:
    """Public alias for :func:`_tensor_shape_digest` (replay / sweep parity logging)."""
    return _tensor_shape_digest(tt)


def maybe_log_hf_rope_decode_tensor_shapes(
    attention: Any,
    phase: str,
    q_tt: ttnn.Tensor,
    k_tt: ttnn.Tensor,
) -> None:
    """If ``TT_HF_ROPE_DECODE_INSTRUMENT_SHAPES`` is set, log Q/K shapes around decode RoPE.

    Emits one log per call. For a full before/after picture, ``Attention._hf_rope_new_decode`` invokes this
    with ``phase="pre_rot"`` then ``phase="post_rot"``. Decode iteration is advanced only on ``post_rot`` so
    pre/post share the same step index.

    ``TT_HF_ROPE_DECODE_INSTRUMENT_MAX_DECODE_STEPS`` (default ``64``) caps how many decode iterations log.
    """
    if os.environ.get(INSTRUMENT_SHAPES_ENV, "0") not in ("1", "true", "True"):
        return

    max_steps = int(os.environ.get(INSTRUMENT_MAX_DECODE_STEPS_ENV, "64"))
    step = int(getattr(attention, "_hf_rope_shape_instrument_decode_step", 0))
    if step >= max_steps:
        return

    layer = getattr(attention, "layer_num", None)
    payload = {
        "layer": layer,
        "decode_step": step,
        "phase": phase,
        "Q": _tensor_shape_digest(q_tt),
        "K": _tensor_shape_digest(k_tt),
    }
    logger.info("HF RoPE decode shape instrument: {}", json.dumps(payload, default=str))

    if phase == "post_rot":
        setattr(attention, "_hf_rope_shape_instrument_decode_step", step + 1)


def _rope_scaling_manifest(rs: Any) -> Optional[Dict[str, Any]]:
    if rs is None:
        return None
    return {
        "factor": getattr(rs, "factor", None),
        "original_max_position_embeddings": getattr(rs, "original_max_position_embeddings", None),
        "rope_type": rs.rope_type.value if getattr(rs, "rope_type", None) is not None else "llama3",
    }


def _to_torch_mesh_composer(attn: Any) -> Tuple[Any, List[int]]:
    """Match ``test_attention_inference`` / multi-device attention output composition."""
    mesh_device = attn.mesh_device
    cluster_shape = list(attn.args.cluster_shape)
    if attn.args.is_galaxy:
        return ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=cluster_shape), cluster_shape
    return ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=cluster_shape), cluster_shape


def _maybe_torch_sidecar(path: Path, tensor: ttnn.Tensor, composer) -> None:
    try:
        t_torch = ttnn.to_torch(tensor, mesh_composer=composer).contiguous()
        torch.save(t_torch, path)
    except Exception:
        # Optional sidecar; manifest still lists tensorbin as source of truth.
        pass


def maybe_capture_hf_rope_decode_step(
    attention: Any,
    q_bf16: ttnn.Tensor,
    k_bf16: ttnn.Tensor,
    cos_tt: ttnn.Tensor,
    sin_tt: ttnn.Tensor,
) -> None:
    """If ``TT_HF_ROPE_DECODE_CAPTURE_DIR`` is set, dump decode RoPE inputs for this step."""
    base = os.environ.get(CAPTURE_DIR_ENV)
    if not base:
        return
    max_steps = int(os.environ.get(CAPTURE_MAX_STEPS_ENV, "1000000"))
    step_id = getattr(attention, "_hf_rope_capture_step", 0)
    if step_id >= max_steps:
        return
    setattr(attention, "_hf_rope_capture_step", step_id + 1)

    root = Path(base)
    root.mkdir(parents=True, exist_ok=True)
    step_dir = root / f"step_{step_id:04d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    ttnn.synchronize_device(attention.mesh_device)

    files = {
        "q_pre_rot": "q_pre_rot.tensorbin",
        "k_pre_rot": "k_pre_rot.tensorbin",
        "cos": "cos.tensorbin",
        "sin": "sin.tensorbin",
    }
    ttnn.dump_tensor(str(step_dir / files["q_pre_rot"]), q_bf16)
    ttnn.dump_tensor(str(step_dir / files["k_pre_rot"]), k_bf16)
    ttnn.dump_tensor(str(step_dir / files["cos"]), cos_tt)
    ttnn.dump_tensor(str(step_dir / files["sin"]), sin_tt)

    host_pos = getattr(attention, "_hf_rope_capture_host_positions", None)
    logical_b = int(host_pos.shape[0]) if isinstance(host_pos, torch.Tensor) else None
    padded_b = int(q_bf16.padded_shape[1]) if q_bf16.padded_shape is not None else int(q_bf16.shape[1])

    if isinstance(host_pos, torch.Tensor):
        torch.save(host_pos.detach().cpu().to(torch.int32), step_dir / "position_indices.pt")

    composer, cluster_shape = _to_torch_mesh_composer(attention)
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_VERSION,
        "step": step_id,
        "tensor_files": files,
        "q_digest": tensor_memory_config_shard_digest(q_bf16),
        "k_digest": tensor_memory_config_shard_digest(k_bf16),
        "cos_digest": tensor_memory_config_shard_digest(cos_tt),
        "sin_digest": tensor_memory_config_shard_digest(sin_tt),
        "logical_batch_size": logical_b,
        "padded_batch_size": padded_b,
        "head_dim": attention.head_dim,
        "n_local_heads": attention.n_local_heads,
        "n_local_kv_heads": attention.n_local_kv_heads,
        "max_seq_len": attention.max_seq_len,
        "rope_theta": attention.args.rope_theta,
        "rope_scaling": _rope_scaling_manifest(attention.args.rope_scaling),
        "cluster_shape": cluster_shape,
        "is_galaxy": bool(attention.args.is_galaxy),
        "mesh_device_env": os.environ.get("MESH_DEVICE"),
        "model_name": getattr(attention.args, "model_name", None),
        "prefetcher": attention.prefetcher is not None,
        "to_torch_composer": {"type": "ConcatMesh2dToTensor", "dims": [1, 3], "mesh_shape": cluster_shape},
    }
    with open(step_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if os.environ.get(CAPTURE_TORCH_SIDECAR_ENV, "0") == "1":
        _maybe_torch_sidecar(step_dir / "q_pre_rot_torch.pt", q_bf16, composer)
        _maybe_torch_sidecar(step_dir / "k_pre_rot_torch.pt", k_bf16, composer)
        _maybe_torch_sidecar(step_dir / "cos_torch.pt", cos_tt, composer)
        _maybe_torch_sidecar(step_dir / "sin_torch.pt", sin_tt, composer)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _pt_hf_rope_heads(q_or_k: torch.Tensor, cos_b: torch.Tensor, sin_b: torch.Tensor) -> torch.Tensor:
    """HF RoPE: ``(x * cos) + (rotate_half(x) * sin)`` with broadcast cos/sin."""
    return (q_or_k * cos_b) + (_rotate_half(q_or_k) * sin_b)


def hf_cos_sin_cache_torch(
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF-layout cos/sin ``[1,1,S,D]`` float32 (same concat as ``HfRotarySetupNew``)."""
    factor = rope_scaling.get("factor") if rope_scaling else None
    orig_max = rope_scaling.get("original_max_position_embeddings") if rope_scaling else None
    rope_type = (rope_scaling or {}).get("rope_type", "llama3")
    cos_freqs, sin_freqs = precompute_freqs(
        head_dim,
        max_seq_len * 2,
        rope_theta,
        factor,
        orig_max,
        rope_type,
    )
    cos_hf = torch.cat([cos_freqs[:max_seq_len], cos_freqs[:max_seq_len]], dim=-1)
    sin_hf = torch.cat([sin_freqs[:max_seq_len], sin_freqs[:max_seq_len]], dim=-1)
    cos_hf = cos_hf.unsqueeze(0).unsqueeze(0).float()
    sin_hf = sin_hf.unsqueeze(0).unsqueeze(0).float()
    return cos_hf, sin_hf


def torch_golden_hf_rope_decode_from_cos_sin(
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    cos_torch: torch.Tensor,
    sin_torch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HF RoPE in torch using explicit ``cos``/``sin`` (same broadcast math as the device op inputs).

    Shapes:
        ``q_bf16`` / ``k_bf16``: ``[1, B_pad, H, head_dim]``.
        ``cos_torch`` / ``sin_torch``: ``[1, B_pad, S_pad, head_dim]`` with decode row at ``S_pad`` index ``0``
        (matches ``HfRotarySetupNew.get_rot_mats`` / ``rotary_embedding_hf`` capture).
    """
    padded_b = int(q_bf16.shape[1])
    n_q_heads = int(q_bf16.shape[2])
    n_k_heads = int(k_bf16.shape[2])
    head_dim = int(q_bf16.shape[3])
    if int(k_bf16.shape[1]) != padded_b or int(k_bf16.shape[3]) != head_dim:
        raise ValueError("q_bf16 and k_bf16 must match on batch and head_dim")
    q_ref = torch.zeros((1, padded_b, n_q_heads, head_dim), dtype=torch.float32)
    k_ref = torch.zeros((1, padded_b, n_k_heads, head_dim), dtype=torch.float32)
    for b in range(padded_b):
        cos_row = cos_torch[0:1, b : b + 1, 0:1, :].float()
        sin_row = sin_torch[0:1, b : b + 1, 0:1, :].float()
        cos_q = cos_row.expand(1, 1, n_q_heads, -1)
        sin_q = sin_row.expand(1, 1, n_q_heads, -1)
        qb = q_bf16[0, b, :, :].float().unsqueeze(0).unsqueeze(0)
        q_ref[:, b : b + 1, :, :] = _pt_hf_rope_heads(qb, cos_q, sin_q)
        cos_k = cos_row.expand(1, 1, n_k_heads, -1)
        sin_k = sin_row.expand(1, 1, n_k_heads, -1)
        kb = k_bf16[0, b, :, :].float().unsqueeze(0).unsqueeze(0)
        k_ref[:, b : b + 1, :, :] = _pt_hf_rope_heads(kb, cos_k, sin_k)
    return q_ref, k_ref


def torch_golden_hf_rope_decode_1b32d(
    q_bf16: torch.Tensor,
    k_bf16: torch.Tensor,
    position_logical: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    rope_theta: float,
    rope_scaling: Optional[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for decode layout ``[1, B_pad, H, D]`` (parity with ``test_hf_rope_decode_reassembly_parity``)."""
    padded_b = q_bf16.shape[1]
    n_q_heads = q_bf16.shape[2]
    n_k_heads = k_bf16.shape[2]
    cos_cache, sin_cache = hf_cos_sin_cache_torch(head_dim, max_seq_len, rope_theta, rope_scaling)
    pos = position_logical.to(torch.int64)
    q_ref = torch.zeros((1, padded_b, n_q_heads, head_dim), dtype=torch.float32)
    k_ref = torch.zeros((1, padded_b, n_k_heads, head_dim), dtype=torch.float32)
    logical_len = int(pos.shape[0])
    for b in range(padded_b):
        if logical_len == 0:
            p = 0
        elif b < logical_len:
            p = int(pos[b].item())
        else:
            p = 0
        cos_row = cos_cache[0, 0, p : p + 1, :]
        sin_row = sin_cache[0, 0, p : p + 1, :]

        qb = q_bf16[0, b, :, :].float().unsqueeze(0).unsqueeze(0)
        cos_q = cos_row.expand(1, 1, n_q_heads, -1)
        sin_q = sin_row.expand(1, 1, n_q_heads, -1)
        q_ref[:, b : b + 1, :, :] = _pt_hf_rope_heads(qb, cos_q, sin_q)

        kb = k_bf16[0, b, :, :].float().unsqueeze(0).unsqueeze(0)
        cos_k = cos_row.expand(1, 1, n_k_heads, -1)
        sin_k = sin_row.expand(1, 1, n_k_heads, -1)
        k_ref[:, b : b + 1, :, :] = _pt_hf_rope_heads(kb, cos_k, sin_k)
    return q_ref, k_ref


def list_capture_step_dirs(replay_root: Path) -> List[Path]:
    """Sorted ``step_*`` directories under ``replay_root``."""
    if not replay_root.is_dir():
        return []
    dirs = [p for p in replay_root.iterdir() if p.is_dir() and p.name.startswith("step_")]
    return sorted(dirs, key=lambda p: p.name)


def load_manifest(step_dir: Path) -> Dict[str, Any]:
    with open(step_dir / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)
