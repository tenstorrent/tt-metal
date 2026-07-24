# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE 5 Hz LM decode: sharded Q/K head RMSNorm without L1 interleaved ping-pong.

Stock ``tt_transformers`` ``Attention`` wraps Qwen3 ``q_norm`` / ``k_norm`` with
``norm_reshard`` (HEIGHT → L1 interleaved → norm → HEIGHT) because RMSNorm did not
accept height-sharded I/O. That adds two ``to_memory_config`` round-trips per norm
(~2,128 ``LayerNormDeviceOperation`` launches per long decode trace).

When ``in_sharded=True`` and ``out_sharded=True``, :class:`models.common.rmsnorm.RMSNorm`
runs directly on the HEIGHT-sharded head tensors from ``nlp_create_qkv_heads_decode``.
"""

from __future__ import annotations

from typing import Any, Callable

import ttnn
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode


def _unwrap_rmsnorm_from_norm_lambda(norm_fn: Callable) -> RMSNorm | None:
    """Extract the ``RMSNorm`` module closed over by ``norm_reshard`` lambdas."""
    closure = getattr(norm_fn, "__closure__", None)
    if not closure or len(closure) < 2:
        return None
    inner = closure[1].cell_contents
    return inner if isinstance(inner, RMSNorm) else None


def ace_step_head_qk_norm_sharded_config(model_args: Any) -> dict[str, Any]:
    """Norm config for HEIGHT-sharded Q/K heads (matches create_qkv_heads decode output)."""
    sharded_output = model_args.get_attn_create_head_output_mem_config(Mode.DECODE, None)
    tile = int(getattr(ttnn, "TILE_SIZE", 32))
    head_dim = int(model_args.head_dim)
    block_w = max(1, head_dim // tile)
    lnpc_cls = getattr(ttnn, "LayerNormShardedMultiCoreProgramConfig", None)
    if lnpc_cls is None or sharded_output is None:
        return {"sharded_output_config": sharded_output}

    shard_spec = getattr(sharded_output, "shard_spec", None)
    grid = getattr(shard_spec, "grid", None) if shard_spec is not None else None
    if grid is None:
        return {"sharded_output_config": sharded_output}

    try:
        num_cores = int(grid.num_cores())
    except Exception:
        return {"sharded_output_config": sharded_output}

    subblock_w = 1
    for sw in range(min(block_w, 4), 0, -1):
        if block_w % sw == 0:
            subblock_w = sw
            break

    gx = max(1, int(getattr(grid, "grid_size", lambda: (1, num_cores))().x)) if hasattr(grid, "grid_size") else 1
    gy = max(1, (num_cores + gx - 1) // gx)

    program_config = lnpc_cls(
        compute_with_storage_grid_size=(gx, gy),
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )
    return {
        "sharded_program_config": program_config,
        "sharded_output_config": sharded_output,
    }


def _make_sharded_qk_norm_fn(rms: RMSNorm, model_args: Any) -> Callable:
    head_cfg = ace_step_head_qk_norm_sharded_config(model_args)

    def _norm(x, mode, norm_config):
        if mode == Mode.DECODE:
            return rms(
                x,
                mode,
                in_sharded=True,
                out_sharded=True,
                norm_config=head_cfg,
            )
        return rms(x, mode, norm_config=norm_config)

    return _norm


def ace_step_patch_attention_qk_norm_decode(attn: Any, model_args: Any) -> None:
    """Replace decode ``q_norm`` / ``k_norm`` lambdas that use ``norm_reshard``."""

    def _patched_norm(norm_fn: Any) -> Callable | None:
        if not callable(norm_fn):
            return None
        rms = _unwrap_rmsnorm_from_norm_lambda(norm_fn)
        if rms is None:
            return None
        return _make_sharded_qk_norm_fn(rms, model_args)

    # Assign known attributes directly (avoid dynamic setattr; SAST false-positive).
    patched_q = _patched_norm(getattr(attn, "q_norm", None))
    if patched_q is not None:
        attn.q_norm = patched_q
    patched_k = _patched_norm(getattr(attn, "k_norm", None))
    if patched_k is not None:
        attn.k_norm = patched_k


def ace_step_apply_qwen_decode_qk_norm(tt_model: Any, model_args: Any) -> None:
    """Patch all decoder ``Attention`` layers for sharded Q/K head norms."""
    for layer in getattr(tt_model, "layers", []):
        attn = getattr(layer, "attention", None)
        if attn is not None:
            ace_step_patch_attention_qk_norm_decode(attn, model_args)


__all__ = [
    "ace_step_apply_qwen_decode_qk_norm",
    "ace_step_head_qk_norm_sharded_config",
    "ace_step_patch_attention_qk_norm_decode",
]
