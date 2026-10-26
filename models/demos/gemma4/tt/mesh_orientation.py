# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh orientation defaults of the 4-chip Blackhole line (QB2 emulation / P150x4), pure host logic.

The vLLM plugin opens the 4 visible chips as ``MeshShape(1, 4)`` (TP over the columns). The DeepSeek token-dispatch
/ combine ops of the routed-expert prefill run along mesh axis 0 only, so the 26B-A4B's dispatch prefill needs the
same chips viewed as the column mesh ``MeshShape(4, 1)`` with the dense TP group over the rows. Measured on line
23..20 (2026-10-23, 3 warm reps): single-user TTFT 128 / 2048 / 8192 tokens = 160 / 389 / 1169 ms on (4,1) +
dispatch vs 130-150 / 889-904 / 3560-3620 ms on (1,4) with the masked-dense sparse_matmul experts.

Defaults (every one overridable):

* ``GEMMA4_QB2_COLUMN_MESH`` (default on): a Blackhole 4-chip MoE build reshapes the plugin's (1,4) mesh to (4,1)
  (``should_reshape_to_column``); ``0`` keeps the row mesh;
* ``GEMMA4_TP_AXIS``: unset -> 0 on a column mesh (rows > 1, one column), else 1 (``tp_axis_for``);
* ``GEMMA4_MOE_DISPATCH_PREFILL``: unset -> on for a Blackhole (4,1) mesh (``dispatch_prefill_default``), with
  ``GEMMA4_MOE_DISPATCH_MIN_TOKENS`` defaulting to 128 there (the measured configuration) and 256 elsewhere;
* ``GEMMA4_MOE_DISPATCH_FIT_CHUNK``: unset -> on wherever the dispatch path is the default (``fit_chunk_default``):
  a short call runs on a dispatch chunk of its own length instead of the 4096-row padded chunk.
"""

import os

from loguru import logger

COLUMN_MESH_ENV = "GEMMA4_QB2_COLUMN_MESH"
TP_AXIS_ENV = "GEMMA4_TP_AXIS"
DISPATCH_ENV = "GEMMA4_MOE_DISPATCH_PREFILL"
DISPATCH_MIN_TOKENS_ENV = "GEMMA4_MOE_DISPATCH_MIN_TOKENS"
COLUMN_MESH_SHAPE = (4, 1)
ROW_MESH_SHAPE = (1, 4)


def _shape(shape):
    return tuple(int(d) for d in shape) if shape is not None else None


def is_column_mesh(shape) -> bool:
    """A mesh of several rows and one column: the only TP group it can hold is over the rows."""
    s = _shape(shape)
    return s is not None and len(s) == 2 and s[0] > 1 and s[1] == 1


def tp_axis_for(shape=None) -> int:
    """Mesh axis of the plain (non-fractured) TP group: ``GEMMA4_TP_AXIS`` when set, else 0 on a column mesh and 1
    (the columns) everywhere else."""
    env = os.environ.get(TP_AXIS_ENV)
    if env is not None and env.strip() != "":
        return 0 if env.strip() == "0" else 1
    return 0 if is_column_mesh(shape) else 1


def column_mesh_enabled() -> bool:
    return os.environ.get(COLUMN_MESH_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def should_reshape_to_column(shape, *, blackhole: bool, moe: bool, data_parallel: int = 1, lanes: bool = False):
    """The ``MeshShape`` a vLLM model build should reshape its mesh to before building, or None to keep it: the
    (1,4) row mesh of a Blackhole MoE build (one model per mesh, no lane fold) becomes (4,1)."""
    if not column_mesh_enabled() or not blackhole or not moe:
        return None
    if int(data_parallel) != 1 or lanes:
        return None
    if _shape(shape) != ROW_MESH_SHAPE:
        return None
    return COLUMN_MESH_SHAPE


def dispatch_prefill_default(shape, *, blackhole: bool) -> bool:
    """Whether the routed-expert prefill takes the token-dispatch path: ``GEMMA4_MOE_DISPATCH_PREFILL`` when set,
    else on for a Blackhole (4,1) column mesh (the measured configuration) and off elsewhere."""
    env = os.environ.get(DISPATCH_ENV)
    if env is not None and env.strip() != "":
        return env.strip() == "1"
    return bool(blackhole) and _shape(shape) == COLUMN_MESH_SHAPE


FIT_CHUNK_ENV = "GEMMA4_MOE_DISPATCH_FIT_CHUNK"


def fit_chunk_default(shape, *, blackhole: bool) -> bool:
    """Whether a dispatch-prefill call shorter than the full chunk runs on a chunk of its own length:
    ``GEMMA4_MOE_DISPATCH_FIT_CHUNK`` when set, else on wherever the dispatch path is the default (the Blackhole (4,1)
    column mesh: 128-token TTFT 213 -> 151 ms, 2048-token 382 -> 303 ms, logits PCC 1.000, 2026-10-23) and off elsewhere.
    """
    env = os.environ.get(FIT_CHUNK_ENV)
    if env is not None and env.strip() != "":
        return env.strip() == "1"
    return dispatch_prefill_default(shape, blackhole=blackhole)


def dispatch_min_tokens_default(shape, *, blackhole: bool) -> int:
    """Rows from which a prefill call takes the dispatch path: ``GEMMA4_MOE_DISPATCH_MIN_TOKENS`` when set, else 128
    where the dispatch path is the default and 256 elsewhere."""
    env = os.environ.get(DISPATCH_MIN_TOKENS_ENV)
    if env is not None and env.strip() != "":
        return int(env)
    return 128 if dispatch_prefill_default(shape, blackhole=blackhole) else 256


# --- decode fusion defaults of the 4-chip Blackhole MoE build -------------------------------------------------
#
# The fused decode step (explicit 1D decode matmul configs, the custom_ops fusions and the call-site switches) is
# the default for the 26B-A4B on 4 Blackhole chips, on the (1,4) row mesh and the (4,1) column mesh alike:
# 64-position HF gate b1 52/64 top-1 (decode PCC 0.98481), b32 50/64 (0.98558); traced step at 32 distinct
# users 35.7 -> 30.2 ms (28.0 -> 33.1 tok/s/user), b1 23.8 -> 23.0 ms (2026-10-23, line 31..28, device
# sampling). Every flag stays an explicit override: a set env var wins ("0" restores the old path, "2" for
# GEMMA4_NORM_FUSED selects the fused_norm_decode form). Other meshes (the Galaxy lane engine), Wormhole and the
# dense models keep today's defaults (off). The model build records its profile once
# (``set_decode_fusion_profile``) and the read sites go through ``decode_flag``.

DECODE_FUSION_ON = {
    "GEMMA4_MLP_DECODE_1D": "1",
    "GEMMA4_ATTN_DECODE_1D": "1",
    "GEMMA4_ROUTER_DECODE_1D": "1",
    "GEMMA4_HEAD_NORM_4D": "1",
    "GEMMA4_MOE_FUSED": "1",
    "GEMMA4_MLP_GEGLU_FUSED": "1",
    "GEMMA4_SOFTCAP_FUSED": "1",
    "GEMMA4_CONCAT_HEADS_FUSED": "1",
    "GEMMA4_QKV_L1_OUT": "1",
    "GEMMA4_NORM_FUSED": "2",
    "GEMMA4_LAYER_NORM_FUSED": "1",
    "GEMMA4_QKV_NORM_ROPE_FUSED": "1",
    # 32..128-token prefill calls: the small-row expert path on its 128-row plan instead of the fabric dispatch
    # (replay kernel sum 76.7 -> 45.7 ms on the 128 bucket; served quiet-box A/B/A/B 2026-10-24, devq 0401: c1 128/128
    # TTFT 141 / 142 -> 112 / 112.5 ms, c32 TTFT 1294 / 1300 -> 1388 / 1358 ms with TPOT 34.5 / 31.0 -> 32.7 / 32.6);
    # inert on the (1,4) row mesh (masked-dense experts there); GEMMA4_SMALL_ROW_MIN/MAX_TOKENS bound it (32..128)
    "GEMMA4_SMALL_ROW_EXPERTS": "1",
}
# The dense models (31B / 12B) on the same 4-chip Blackhole meshes default to the combined dense configuration
# measured in the layer lab and the model-level A/B (row view, QKV heads + norms + RoPE as one program, concat heads
# into the o_proj layout, the native norms writing the next matmul's activation layout, the sharded GeGLU, the
# fused softcap); the fused norm family (GEMMA4_NORM_FUSED=2) stays off on dense widths (slower than the native
# width-sharded LayerNorm there). Env overrides win here as well.
DENSE_DECODE_FUSION_ON = {
    "GEMMA4_DRAM_SHARD_ROW_VIEW": "1",
    "GEMMA4_QKV_L1_OUT": "1",
    "GEMMA4_QKV_HEADS_FUSED": "1",
    "GEMMA4_CONCAT_HEADS_FUSED": "1",
    "GEMMA4_CONCAT_HEADS_SHARDED": "1",
    "GEMMA4_MLP_GEGLU_SHARDED": "1",
    "GEMMA4_NORM_SHARDED_OUT": "1",
    "GEMMA4_SOFTCAP_FUSED": "1",
    # prefill: the per-head q / k / v norms, the q / k RoPE and their slices as one program (QkvNormRopePrefill);
    # judged on the 2048-position natural-prose fp32 gates (12B 512 / 2048: 0.795 / 0.979 and 0.729 / 0.957 vs the
    # composite 0.746 / 0.951 and 0.714 / 0.945; on the QB2 runner 31B 0.7993 / 0.9624, 12B 0.7285 / 0.9570 vs the
    # composite 0.7202 / 0.9448); Blackhole only (qkv_norm_rope_prefill_enabled refuses the flag on Wormhole)
    "GEMMA4_QKV_NORM_ROPE_PREFILL": "1",
}
DECODE_FUSION_OFF = {name: "0" for name in {**DECODE_FUSION_ON, **DENSE_DECODE_FUSION_ON}}
FOUR_CHIP_SHAPES = (ROW_MESH_SHAPE, COLUMN_MESH_SHAPE)
# The Galaxy lane engine (the 26B-A4B on the (8,4) Blackhole mesh): the fused QKV norm + RoPE program on the lane
# decode frame, the rest as before (the lane entry's catalog carries its own set). Window 13 (2026-10-25, lane layer
# (2,4) gate fused vs composite pcc 0.999755; served): c1 32.4 -> 37.1 tok/s/user, c32 33.4 -> 36.3, c128 29.3 ->
# 31.4, capitals 8/8 + 32/32, the distinct-document recipe 4/4 + 20/20 x2. Env overrides win here as well.
LANE_MESH_SHAPE = (8, 4)
LANE_DECODE_FUSION_ON = {**DECODE_FUSION_OFF, "GEMMA4_QKV_NORM_ROPE_FUSED": "1"}

_decode_fusion_profile = {}


def fused_decode_build(shape, *, blackhole: bool, moe: bool) -> bool:
    """Whether a build runs the fused decode step by default: a Blackhole MoE model on the 4-chip (1,4) or (4,1)
    mesh. Wormhole, dense models and every other mesh shape (one chip, the Galaxy lane engine) do not."""
    return bool(blackhole) and bool(moe) and _shape(shape) in FOUR_CHIP_SHAPES


def dense_fused_decode_build(shape, *, blackhole: bool, moe: bool) -> bool:
    """Whether a build runs the combined dense decode configuration by default: a Blackhole dense model on the
    4-chip (1,4) or (4,1) mesh. Wormhole, the MoE build and the other mesh shapes (one chip, the (1,8) line, the
    Galaxy lane engine) do not."""
    return bool(blackhole) and not moe and _shape(shape) in FOUR_CHIP_SHAPES


def lane_fused_decode_build(shape, *, blackhole: bool, moe: bool) -> bool:
    """Whether a build is the Galaxy lane engine (a Blackhole MoE model on the (8,4) mesh), which fuses the QKV
    norms and RoPE of the lane decode frame by default."""
    return bool(blackhole) and bool(moe) and _shape(shape) == LANE_MESH_SHAPE


def decode_fusion_defaults(shape, *, blackhole: bool, moe: bool) -> dict:
    """The per-flag default value (the string an env var would carry) for this build."""
    if fused_decode_build(shape, blackhole=blackhole, moe=moe):
        return dict(DECODE_FUSION_ON)  # the dense-only names are absent: decode_flag reads them as "0"
    if dense_fused_decode_build(shape, blackhole=blackhole, moe=moe):
        return dict(DENSE_DECODE_FUSION_ON)  # the MoE-only names and the fused norm family absent: "0"
    if lane_fused_decode_build(shape, blackhole=blackhole, moe=moe):
        return dict(LANE_DECODE_FUSION_ON)
    return dict(DECODE_FUSION_OFF)


def set_decode_fusion_profile(shape, *, blackhole: bool, moe: bool) -> dict:
    """Record the build's defaults process-wide (the model build calls it before its layers are built, since the
    weight loaders and the norm modules read the flags at construction). Returns the recorded defaults."""
    global _decode_fusion_profile
    _decode_fusion_profile = decode_fusion_defaults(shape, blackhole=blackhole, moe=moe)
    return dict(_decode_fusion_profile)


def clear_decode_fusion_profile():
    global _decode_fusion_profile
    _decode_fusion_profile = {}


def decode_flag(name: str) -> str:
    """The effective value of a decode fusion flag: the env var when set (non-empty), else the recorded build
    profile's default, else "0" (no profile: a module used outside a model build keeps the old path unless the
    env var asks for the new one)."""
    env = os.environ.get(name)
    if env is not None and env.strip() != "":
        return env.strip()
    return _decode_fusion_profile.get(name, "0")


_QKV_PREFILL_WH_WARNED = False


def qkv_norm_rope_prefill_enabled(arch_is_blackhole: bool) -> bool:
    """Whether the fused QkvNormRopePrefill program runs: GEMMA4_QKV_NORM_ROPE_PREFILL resolves to "1" (the env when
    set, else the build profile: on for the 4-chip Blackhole dense builds) and the device is Blackhole. The op's
    kernels are Blackhole-validated only (its CI arms live on the bh_quietbox_2 legs); on Wormhole the flag is
    refused with one warning and the composite path runs, so a stray env can never hang a Wormhole box (CI at
    66238f4353b: the fused-op token-accuracy row on wh_llmbox_perf hung the device). A Wormhole-safe kernel, not
    this guard, is the long-term answer (custom kernels generic over WH / BH)."""
    global _QKV_PREFILL_WH_WARNED
    if decode_flag("GEMMA4_QKV_NORM_ROPE_PREFILL") != "1":  # the env when set, else the build profile's default
        return False
    if arch_is_blackhole:
        return True
    if not _QKV_PREFILL_WH_WARNED:
        _QKV_PREFILL_WH_WARNED = True
        logger.warning(
            "GEMMA4_QKV_NORM_ROPE_PREFILL=1 ignored: the fused qkv-norm-rope prefill op is Blackhole-only; "
            "the composite per-head norm + RoPE path runs on this Wormhole device"
        )
    return False
