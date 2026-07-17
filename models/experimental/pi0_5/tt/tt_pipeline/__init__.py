# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone pi0.5 streamed-denoise port (tt_pipeline).

Self-contained sibling of ``tt/tt_bh_glx/`` -- imports with tt_symbiote NOT installed.
Public surface: the StageDenoise drop-in adapter, the pipeline builders + streamed driver,
the expert block, the stage, mesh carving, the euler schedule, and the split transport.
NO ``*TP4`` (VLM prefill) symbols.

On import, installs the probe-gated ttnn.fill_cache update_idx shim (idempotent; no-op if
native fill_cache already accepts update_idx).
"""
from __future__ import annotations

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

from .denoise_pipeline import (  # noqa: E402
    TTNNPi05DenoiseExpertBlock,
    TTNNPi05DenoisePipelineStage,
    TTNNPi05DenoiseStreamedPipeline,
    _install_fill_cache_shim,
    build_denoise_loop_pipeline,
    build_denoise_pipeline,
    build_expert_only_pipeline,
    build_n_stage_pipeline,
    build_single_stage_reference,
    euler_schedule,
    perf_action_horizon,
    perf_suffix_len,
)
from .mesh_carve import carve_four_submeshes, carve_n_submeshes  # noqa: E402

# Install the update_idx-capable fill_cache shim (probe-gated; load-bearing only for the
# static-KV reference/PCC builders -- the streamed concat-KV hot path never calls fill_cache).
_install_fill_cache_shim()


def __getattr__(name):
    if name == "StageDenoise":
        from .stage_denoise import StageDenoise

        return StageDenoise
    if name == "SplitSocketTransport":
        from ._transport import SplitSocketTransport

        return SplitSocketTransport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "StageDenoise",
    "build_denoise_loop_pipeline",
    "build_denoise_pipeline",
    "build_n_stage_pipeline",
    "build_single_stage_reference",
    "build_expert_only_pipeline",
    "TTNNPi05DenoiseStreamedPipeline",
    "TTNNPi05DenoiseExpertBlock",
    "TTNNPi05DenoisePipelineStage",
    "carve_n_submeshes",
    "carve_four_submeshes",
    "euler_schedule",
    "perf_action_horizon",
    "perf_suffix_len",
    "SplitSocketTransport",
]
