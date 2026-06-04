# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE 5 Hz LM: sharded final RMSNorm before ``LMHead`` (prefill last-token path).

Stock ``tt_transformers`` runs prefill ``lm_head`` norm with ``in_sharded=False`` /
``out_sharded=False``, emitting DRAM interleaved activations and then calling
``ttnn.interleaved_to_sharded`` before the width-sharded ``LMHead`` matmul.

This module routes the final norm through the LMHead width-sharded grid:
direct reshard when input is already sharded (no DRAM s2i+i2s ping-pong), then
``rms_norm`` with ``in_sharded=True`` / ``out_sharded=True``.
"""

from __future__ import annotations

import functools
from typing import Any

import ttnn
from models.tt_transformers.tt.common import Mode

from .math_perf_env import (
    ace_step_build_prefill_block_sharded_norm_config,
    ace_step_lm_head_sharded_norm_enabled,
    ace_step_prefill_block_sharded_norm_enabled,
)


def _tensor_is_sharded(tensor: Any) -> bool:
    try:
        return bool(tensor.memory_config().is_sharded())
    except Exception:
        return False


def _is_lm_head_prefill_norm(dnorm: Any, mode: Mode, norm_config: dict | None) -> bool:
    if mode != Mode.PREFILL or norm_config is None:
        return False
    lm_cfg = dnorm.args.get_norm_config("lm_head", Mode.PREFILL, None)
    return norm_config.get("sharded_output_config") is lm_cfg.get("sharded_output_config")


def _shard_for_lm_head(x: Any, lm_head_input_mem_cfg: Any) -> Any:
    if _tensor_is_sharded(x):
        if x.memory_config() == lm_head_input_mem_cfg:
            return x
        return ttnn.to_memory_config(x, lm_head_input_mem_cfg)
    return ttnn.interleaved_to_sharded(x, lm_head_input_mem_cfg)


def _distributed_norm_prestep(
    dnorm: Any,
    x: Any,
    *,
    mode: Mode,
    norm_config: dict,
    target_sharded_mem_cfg: Any,
) -> Any:
    args = dnorm.args

    if _tensor_is_sharded(x):
        if args.is_multichip and not args.is_distributed_norm(mode):
            ag_key = dnorm.ag_config_key
            return ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=dnorm.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=args.model_config[ag_key]["num_links"]
                if ag_key and mode == "decode"
                else dnorm.tt_ccl.get_num_links(1),
                topology=args.ccl_topology(),
                memory_config=target_sharded_mem_cfg,
                barrier_semaphore=dnorm.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=args.model_config[ag_key]["chunks_per_sync"] if ag_key and mode == "decode" else 10,
                num_workers_per_link=args.model_config[ag_key]["num_workers_per_link"]
                if ag_key and mode == "decode"
                else 2,
                num_buffers_per_channel=2,
                subdevice_id=dnorm.prefetcher.worker_sub_device_id if dnorm.prefetcher is not None else None,
            )
        return _shard_for_lm_head(x, target_sharded_mem_cfg)

    input_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
    if args.is_multichip and not args.is_distributed_norm(mode):
        ag_key = dnorm.ag_config_key
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=dnorm.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=args.model_config[ag_key]["num_links"]
            if ag_key and mode == "decode"
            else dnorm.tt_ccl.get_num_links(1),
            topology=args.ccl_topology(),
            memory_config=input_mem_cfg,
            barrier_semaphore=dnorm.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=args.model_config[ag_key]["chunks_per_sync"] if ag_key and mode == "decode" else 10,
            num_workers_per_link=args.model_config[ag_key]["num_workers_per_link"]
            if ag_key and mode == "decode"
            else 2,
            num_buffers_per_channel=2,
            subdevice_id=dnorm.prefetcher.worker_sub_device_id if dnorm.prefetcher is not None else None,
        )
    return ttnn.to_memory_config(x, input_mem_cfg)


def _run_sharded_lm_head_norm(dnorm: Any, x: Any, *, mode: Mode, norm_config: dict) -> Any:
    prefetcher = None if mode == Mode.PREFILL else dnorm.prefetcher
    blk_cfg = (
        ace_step_build_prefill_block_sharded_norm_config(ttnn, x, dnorm.args.mesh_device)
        if ace_step_prefill_block_sharded_norm_enabled()
        else None
    )
    target_sharded_mem_cfg = (
        blk_cfg["sharded_output_config"]
        if blk_cfg is not None
        else dnorm.args.get_lm_head_input_mem_config(mode, prefetcher)
    )
    run_cfg = blk_cfg if blk_cfg is not None else norm_config
    x = _distributed_norm_prestep(
        dnorm,
        x,
        mode=mode,
        norm_config=norm_config,
        target_sharded_mem_cfg=target_sharded_mem_cfg,
    )
    if not _tensor_is_sharded(x):
        x = _shard_for_lm_head(x, target_sharded_mem_cfg)
    return dnorm.norm(
        x,
        mode=mode,
        in_sharded=True,
        out_sharded=True,
        norm_config=run_cfg,
    )


def _patch_distributed_norm_lm_head_sharded(dnorm: Any) -> None:
    orig_forward = dnorm.forward

    def forward(x, mode: Mode, norm_config=None):
        if ace_step_lm_head_sharded_norm_enabled() and _is_lm_head_prefill_norm(dnorm, mode, norm_config):
            return _run_sharded_lm_head_norm(dnorm, x, mode=mode, norm_config=norm_config)
        return orig_forward(x, mode, norm_config)

    dnorm.forward = forward  # type: ignore[method-assign]


def _patch_apply_norm_and_lm_head(tt_model: Any) -> None:
    def patched(x):
        x = tt_model.norm(
            x,
            mode=Mode.PREFILL,
            norm_config=tt_model.args.get_norm_config("lm_head", Mode.PREFILL, tt_model.prefetcher),
        )
        lm_head_input_mem_cfg = tt_model.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
        if lm_head_input_mem_cfg.is_sharded() and not _tensor_is_sharded(x):
            x = _shard_for_lm_head(x, lm_head_input_mem_cfg)
        logits = tt_model.lm_head(x)
        return ttnn.to_memory_config(logits, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_model._apply_norm_and_lm_head = patched  # type: ignore[method-assign]


def _patch_transformer_forward_i2s_guard(tt_model: Any) -> None:
    orig_forward = tt_model.forward
    orig_i2s = ttnn.interleaved_to_sharded

    @functools.wraps(orig_forward)
    def forward(*args, **kwargs):
        def guarded_i2s(tensor, mem_config, *i2s_args, **i2s_kwargs):
            if ace_step_lm_head_sharded_norm_enabled() and _tensor_is_sharded(tensor) and mem_config.is_sharded():
                if tensor.memory_config() == mem_config:
                    return tensor
                return ttnn.to_memory_config(tensor, mem_config)
            return orig_i2s(tensor, mem_config, *i2s_args, **i2s_kwargs)

        ttnn.interleaved_to_sharded = guarded_i2s
        try:
            return orig_forward(*args, **kwargs)
        finally:
            ttnn.interleaved_to_sharded = orig_i2s

    tt_model.forward = forward  # type: ignore[method-assign]


def ace_step_apply_lm_head_sharded_norm(tt_model: Any, model_args: Any) -> None:
    """Patch final ``lm_head`` RMSNorm to width-sharded output (prefill + ``_apply_norm_and_lm_head``)."""
    del model_args
    if not ace_step_lm_head_sharded_norm_enabled():
        return
    if hasattr(tt_model, "norm"):
        _patch_distributed_norm_lm_head_sharded(tt_model.norm)
    if hasattr(tt_model, "_apply_norm_and_lm_head"):
        _patch_apply_norm_and_lm_head(tt_model)
    if hasattr(tt_model, "forward"):
        _patch_transformer_forward_i2s_guard(tt_model)


__all__ = [
    "ace_step_apply_lm_head_sharded_norm",
]
