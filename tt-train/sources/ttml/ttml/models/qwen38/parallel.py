# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Megatron tensor-parallel helpers for Qwen3.8.

Qwen3.8 is trained here on a ``[2, 4]`` mesh -- 2-way data parallel x 4-way
tensor parallel over 8 chips.  TP is capped at 4 rather than 8 by the attention
layers: there are only ``num_key_value_heads = 4`` KV heads, and ttml's
distributed GQA requires ``num_groups % tp_size == 0``.  The 48 DeltaNet layers
would shard happily at 8 (16 key / 48 value heads), but a single TP width has to
serve the whole stack.

Head counts at TP=4, all exact:

======================  ======  =========
tensor                  global  per chip
======================  ======  =========
DeltaNet key heads      16      4
DeltaNet value heads    48      12
attention query heads   24      6
attention KV heads      4       1
======================  ======  =========

The GVA ratio is preserved locally (12/4 == 48/16 == 3), so the DeltaNet needs
no cross-chip communication inside the delta rule itself.

Sharding a fused QKV projection
-------------------------------
``ColumnParallelLinear`` shards output features *contiguously*, which is correct
for any projection whose output is a whole number of heads per shard -- but not
for the DeltaNet's fused ``in_proj_qkv``, whose output is laid out
``[Q(2048) | K(2048) | V(6144)]``.  Split contiguously across 4 chips, chip 0
receives rows 0..2559: all of Q, part of K, and *no V at all*.

:func:`qkv_shard_permutation` fixes this at load time by reordering the weight
rows into per-chip ``[q heads | k heads | v heads]`` blocks, so that a contiguous
shard is exactly the group of heads that chip should own.  The module then slices
Q/K/V using local widths.

Every other tensor shards contiguously without reordering, and they all agree
with this layout:

* ``in_proj_z`` (6144) and ``out_proj``'s input split by value head, giving chip
  ``c`` value heads ``12c..12c+11`` -- the same heads the permuted V block gives it.
* ``in_proj_a`` / ``in_proj_b`` (48) and the ``A_log`` / ``dt_bias`` buffers split
  one entry per value head, matching the same range.
* Attention's ``q_proj`` is per-head ``[query | gate]``, so a contiguous shard
  keeps each head together with its own gate.
"""

from __future__ import annotations

import numpy as np

import ttml
from ttml.modules import ColumnParallelLinear, LinearLayer, RowParallelLinear, Parameter

__all__ = [
    "tp_size",
    "make_column_linear",
    "make_row_linear",
    "make_sharded_parameter",
    "qkv_shard_permutation",
]


def tp_size(config) -> int:
    """Width of the tensor-parallel axis, or 1 when TP is off."""
    if not getattr(config, "use_tp", False):
        return 1
    return ttml.mesh().axis_size(getattr(config, "tp_axis_name", "tp"))


def _axis(config) -> str:
    return getattr(config, "tp_axis_name", "tp")


def make_column_linear(config, in_features: int, out_features: int, weight_init, *, gather_output: bool = False):
    """Column-parallel linear (output features sharded), or plain linear if TP is off."""
    if not getattr(config, "use_tp", False):
        return LinearLayer(in_features, out_features, False, weight_init=weight_init)
    return ColumnParallelLinear(
        in_features,
        out_features,
        False,
        weight_init=weight_init,
        gather_output=gather_output,
        axis_name=_axis(config),
    )


def make_row_linear(config, in_features: int, out_features: int, weight_init, *, input_is_parallel: bool = True):
    """Row-parallel linear (input features sharded, output all-reduced).

    ``input_is_parallel`` defaults to ``True`` because every use here consumes
    the sharded output of a column-parallel layer, so no scatter is needed.
    """
    if not getattr(config, "use_tp", False):
        return LinearLayer(in_features, out_features, False, weight_init=weight_init)
    return RowParallelLinear(
        in_features,
        out_features,
        False,
        weight_init=weight_init,
        input_is_parallel=input_is_parallel,
        axis_name=_axis(config),
    )


def make_sharded_parameter(config, init_fn, shape, *, shard_dim: int = 3):
    """A Parameter split across the TP axis along ``shard_dim``.

    Used for the DeltaNet's per-value-head ``A_log`` and ``dt_bias``, which have
    to follow the same value-head split as the projections around them.
    """
    if not getattr(config, "use_tp", False):
        return Parameter(init_fn(shape))
    mapper = ttml.mesh().axis_mapper(_axis(config), tdim=shard_dim)
    return Parameter(init_fn(shape, mapper=mapper))


# Which dim of each parameter the TP axis splits, keyed by the tail of the ttml
# parameter name. Weights are [1, 1, out, in]: column-parallel layers shard the
# output features (dim 2) and row-parallel layers shard the input features
# (dim 3). Per-value-head vectors are [1, 1, 1, H] and shard on dim 3.
#
# Anything absent from this table is replicated: the RMSNorm weights, the
# attention q_norm/k_norm (over head_dim, which is not sharded), the DeltaNet's
# output norm (over head_v_dim, likewise), and the embedding table.
_TP_SHARD_DIMS: dict[str, int] = {
    # --- DeltaNet ---
    "linear_attn/in_proj_qkv/weight": 2,
    "linear_attn/in_proj_z/weight": 2,
    "linear_attn/in_proj_a/weight": 2,
    "linear_attn/in_proj_b/weight": 2,
    "linear_attn/out_proj/weight": 3,
    "linear_attn/A_log": 3,
    "linear_attn/dt_bias": 3,
    # --- attention ---
    "self_attn/q_proj/weight": 2,
    "self_attn/k_proj/weight": 2,
    "self_attn/v_proj/weight": 2,
    "self_attn/o_proj/weight": 3,
    # --- MLP ---
    "mlp/gate_proj/weight": 2,
    "mlp/up_proj/weight": 2,
    "mlp/down_proj/weight": 3,
    # --- head ---
    "lm_head/weight": 2,
}


def tp_shard_dim(param_name: str) -> int | None:
    """The dim the TP axis splits for ``param_name``, or ``None`` if replicated.

    ``param_name`` is a full ttml parameter path such as
    ``"model/layers/3/self_attn/q_proj/weight"``; only the tail is matched.
    """
    for suffix, dim in _TP_SHARD_DIMS.items():
        if param_name.endswith(suffix):
            return dim
    # Conv taps are per-channel over the fused QKV output, so they follow it.
    if "/conv_tap_" in param_name:
        return 3
    return None


def qkv_shard_permutation(config, tp: int) -> np.ndarray:
    """Row permutation making a contiguous column shard of ``in_proj_qkv`` valid.

    Returns indices ``perm`` such that ``weight[perm]`` is ordered as::

        chip 0: q heads[0:4]   k heads[0:4]   v heads[0:12]
        chip 1: q heads[4:8]   k heads[4:8]   v heads[12:24]
        ...

    Applying this to the checkpoint's ``[10240, hidden]`` weight (and to the
    matching conv1d channels) lets ``ColumnParallelLinear`` hand each chip a
    self-consistent set of heads.  With ``tp == 1`` this is the identity, so the
    single-device path is unchanged.
    """
    n_k, n_v = config.linear_num_key_heads, config.linear_num_value_heads
    d_k, d_v = config.linear_key_head_dim, config.linear_value_head_dim
    key_dim = config.key_proj_dim

    if n_k % tp or n_v % tp:
        raise ValueError(f"key heads ({n_k}) and value heads ({n_v}) must both divide tp={tp}")

    k_per, v_per = n_k // tp, n_v // tp
    # Offsets of the Q, K and V blocks within the fused output.
    q_base, k_base, v_base = 0, key_dim, 2 * key_dim

    def head_rows(base: int, head: int, width: int) -> range:
        return range(base + head * width, base + (head + 1) * width)

    perm: list[int] = []
    for chip in range(tp):
        for head in range(chip * k_per, (chip + 1) * k_per):
            perm.extend(head_rows(q_base, head, d_k))
        for head in range(chip * k_per, (chip + 1) * k_per):
            perm.extend(head_rows(k_base, head, d_k))
        for head in range(chip * v_per, (chip + 1) * v_per):
            perm.extend(head_rows(v_base, head, d_v))

    if len(perm) != config.qkv_proj_dim:
        raise AssertionError(f"permutation covers {len(perm)} of {config.qkv_proj_dim} rows")
    return np.asarray(perm, dtype=np.int64)
