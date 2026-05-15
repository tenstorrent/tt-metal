# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Host/device DRAM mitigations for Devstral-2 large (~12k hidden) on **multi-chip** meshes.

Blackhole multi-device already needed interleaved DRAM for wide FFN/attention weight tilize and
tighter prefill matmul blocks (static L1 circular-buffer budget). Wormhole T3K (1×8) hits the same
class of tilize / wide-matmul L1 limits for this dense 123B stack; GPT-OSS 120B is MoE so active
weights per token differ and the same tilize pressure does not apply one-to-one.
"""

from __future__ import annotations

from typing import Any

from models.common.utility_functions import is_blackhole, is_wormhole_b0
from models.tt_transformers.tt.common import Mode

# Hidden sizes at or above this use the wide-model DRAM / minimal-matmul mitigations on
# multi-device non-Galaxy Wormhole (T3K) as well as Blackhole multi-chip.
_MIN_WIDE_DIM_FOR_MITIGATION = 8192


def devstral2_large_multi_device_dram_mitigation(mesh_device: Any, args: Any) -> bool:
    """
    True when we should prefer interleaved DRAM for weight upload and related FFN/attn paths.

    Matches prior Blackhole-only scope, extended to Wormhole B0 for T3K-style 1×N meshes.
    """
    if mesh_device is None or getattr(args, "is_galaxy", False):
        return False
    if mesh_device.get_num_devices() <= 1:
        return False
    dim = int(getattr(args, "dim", 0) or 0)
    if dim < _MIN_WIDE_DIM_FOR_MITIGATION:
        return False
    return bool(is_blackhole() or is_wormhole_b0())


def get_decode_mem_config_after_hidden_dim_concat(args, mode: Mode, prefetcher=None):
    """Width-sharded L1 layout for tensors whose last dim is full ``dim`` after TP concat on dim 3.

    ``get_residual_mem_config(DECODE)`` shards the **per-device** hidden width
    (``dim // num_devices``) using ``dim // grid.num_cores // num_devices`` per shard. Applying
    that spec to a **full** ``dim`` activation (e.g. after ``_all_gather_concat_hidden_dim``)
    makes ``interleaved_to_sharded`` infer ~``dim / shard_width`` width shards (e.g. 128) over a
    ``grid.num_cores`` of 32 and fails validation. Reuse decode ``get_mlp_input_mem_config``.
    For ``Mode.PREFILL`` (or Galaxy / prefetcher), returns ``get_residual_mem_config`` — only
    the non-Galaxy decode path needs the MLP full-width shard layout.
    """
    if mode != Mode.DECODE or prefetcher is not None or args.is_galaxy:
        return args.get_residual_mem_config(mode, prefetcher)
    return args.get_mlp_input_mem_config(mode, prefetcher)
