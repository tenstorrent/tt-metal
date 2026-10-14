# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill runner for the M3 stress test: the stock prefill_runner, launched once per pipeline stage by tt-run, with

- absolute layer placement: STRESS_LAYERS="37,38" builds this rank's logical layers 0..n-1 from those model layers
  (TtPrefillRuntimeConfig.layer_indices). KV slots, layer acks and the layer split stay logical.
- the KV caches' host zeros allocated in bf16 (fp32 doubles the host RAM peak of 4 ranks on one host).
- one "[pp rank N] DRAM ..." line after the KV allocation (per-chip DRAM use).
"""

import dataclasses
import os
import runpy

import torch
from loguru import logger

import models.demos.minimax_m3.tt.attention as attention
import models.demos.minimax_m3.tt.tt_prefill_runtime as rt
import ttnn

_LAYERS = [int(x) for x in os.environ.get("STRESS_LAYERS", "").split(",") if x.strip()]
_RANK = os.environ.get("TT_RUN_RANK", "?")
_runtime_init = rt.TtPrefillRuntime.__init__
_allocate_kv_caches = attention.allocate_kv_caches


def _runtime_init_absolute(self, mesh_device, hf_config, state_dict, config):
    if _LAYERS:
        assert len(_LAYERS) == config.num_layers, f"STRESS_LAYERS={_LAYERS} vs num_layers={config.num_layers}"
        config = dataclasses.replace(config, layer_indices=_LAYERS)
        logger.info(f"[pp rank {_RANK}] building model layers {_LAYERS} as logical 0..{len(_LAYERS) - 1}")
    _runtime_init(self, mesh_device, hf_config, state_dict, config)


def _log_dram(mesh_device):
    try:
        mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        gib = 1024**3
        logger.info(
            f"[pp rank {_RANK}] DRAM per chip after KV alloc: allocated "
            f"{mv.num_banks * mv.total_bytes_allocated_per_bank / gib:.2f} GiB, free "
            f"{mv.num_banks * mv.total_bytes_free_per_bank / gib:.2f} GiB, largest free per bank "
            f"{mv.largest_contiguous_bytes_free_per_bank / gib:.3f} GiB"
        )
    except Exception as e:  # diagnostics only
        logger.warning(f"[pp rank {_RANK}] DRAM report failed: {e!r}")


def _allocate_kv_caches_bf16_host(mesh_device, **kwargs):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        caches = _allocate_kv_caches(mesh_device, **kwargs)
    finally:
        torch.set_default_dtype(prev)
    _log_dram(mesh_device)
    return caches


rt.TtPrefillRuntime.__init__ = _runtime_init_absolute
attention.allocate_kv_caches = _allocate_kv_caches_bf16_host  # the adapter imports it at call time
runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__", alter_sys=True)
