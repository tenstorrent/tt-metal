# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lightweight generator-path decode profile harness (correct server CCL path).

Uses ``Generator.prefill_forward_text`` + ``Generator.decode_forward`` with
``switch_mode("decode")`` — the same contract as ``test_qwen36_demo_generator_batch1``
and ``generator_vllm``.  Intended for Tracy capture with a reduced layer count:

    export QWEN36_N_LAYERS=4 QWEN36_GEN_DECODE_STEPS=2 \\
        QWEN36_GEN_HOST_SAMPLE=1 QWEN36_GEN_TRACE_HOSTSAMP=1 \\
        && python -m tracy -p -v -r -m --op-support-count 20000 pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_generator_profile.py -s
"""
from __future__ import annotations

import pytest

# Reuse the full generator demo harness (mesh fixture, model build, server flags).
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _DECODE_STEPS,
    _N_LAYERS,
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _PATTERN,
    _SNAPSHOT,
    _T_PREFILL,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    _load_prompt_for_isl,
    bh_glx_mesh,
    test_qwen36_demo_generator_batch1,
)


@pytest.mark.hardware
def test_decode_generator_profile_traced(bh_glx_mesh):
    """Run the generator demo with env-tunable N_LAYERS / decode steps (for Tracy)."""
    # Delegate to the proven generator demo; env vars control layer count + trace mode.
    # For PROFILING we run a reduced layer count -> output is intentionally incoherent, which
    # trips the demo's coherence assertion. Swallow it so the device-op capture completes and the
    # Tracy report (ops CSV) is written. The kernel timings are valid regardless of output text.
    import os as _os

    try:
        test_qwen36_demo_generator_batch1(bh_glx_mesh)
    except AssertionError as _e:
        if _os.environ.get("QWEN36_PROFILE_IGNORE_ASSERT", "1") == "1":
            print(f"[profile] swallowed coherence assertion (expected at reduced N_LAYERS): {str(_e)[:120]}")
        else:
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
