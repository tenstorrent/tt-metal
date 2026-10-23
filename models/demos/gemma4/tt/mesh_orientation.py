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
  ``GEMMA4_MOE_DISPATCH_MIN_TOKENS`` defaulting to 128 there (the measured configuration) and 256 elsewhere.
"""

import os

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


def dispatch_min_tokens_default(shape, *, blackhole: bool) -> int:
    """Rows from which a prefill call takes the dispatch path: ``GEMMA4_MOE_DISPATCH_MIN_TOKENS`` when set, else 128
    where the dispatch path is the default and 256 elsewhere."""
    env = os.environ.get(DISPATCH_MIN_TOKENS_ENV)
    if env is not None and env.strip() != "":
        return int(env)
    return 128 if dispatch_prefill_default(shape, blackhole=blackhole) else 256
