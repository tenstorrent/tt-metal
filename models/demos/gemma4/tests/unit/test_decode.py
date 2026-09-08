# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-step decode PCC for one Gemma4 decoder layer, with the KV cache under test.

``DECODE_STEPS`` decode steps run back to back on the *same* paged KV cache: step
``k`` reads what steps ``0…k-1`` wrote through ``paged_update_cache``. HF's
``DynamicCache`` accumulates in lockstep, so the two sides are compared at every
step. This is what ``test_layer_forward_decode`` does not cover — that test fills
the cache with ``torch.randn`` and takes a single step, so a KV write landing at
the wrong position, in the wrong block, or not at all would pass it.

Three variants:

* ``test_decode_multistep_kv_update`` — steps start at position 0 (cold cache).
* ``test_decode_multistep_kv_update_after_prefill`` — a prefill fills the cache
  first, then decode continues from ``prefill_len``, which is the demo's actual
  transition and the only version that exercises decode reading prefill-written
  pages. ``prefill_len=1536`` is there because the window is otherwise **inert**:
  below position 1024 a sliding layer masks nothing, so at 0 and 128 the
  ``sliding`` parametrization differs from ``global`` only in the RoPE base. At
  1536 the mask drops 522 keys and TT's ``sliding_window_size`` has to agree.
* ``test_decode_multistep_bounded_sliding_kv`` — the same run against the
  **bounded sliding** ring cache (``cache_position_modulo``), the layout the demo
  auto-enables above its long-context cutover. The two tests above always build the
  unbounded cache, so the ring's wrap arithmetic would otherwise be untested here.

Each step also verifies the K/V the device wrote at that position against HF's
cache, so a wrong-position write is reported as such instead of surfacing as a
vague output-PCC drift.

Run (T3K / 1x8, the production mesh for 31B and 12B)::

    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=1x8 \\
      pytest models/demos/gemma4/tests/unit/test_decode.py -k "1x8" -v
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

from ...tests.test_factory import get_pcc_threshold, parametrize_mesh_with_fabric, uses_ci_config_only_checkpoint
from .decoder_pcc_common import (
    DECODE_STEPS,
    KV_PCC_REQUIRED,
    PCC_BATCH_SIZE,
    build_decoder_pcc_context,
    check_pcc,
    compare_kv_cache,
    hf_forward_span,
    tt_decode_step,
    tt_prefill_chunk,
)

# Skipped at *collection* time, not inside the test body: the CI unit job points
# HF_MODEL at a checked-in config stub with no safetensors, and skipping per-test
# would still pay one mesh setup/teardown per parametrized node.
pytestmark = pytest.mark.skipif(
    uses_ci_config_only_checkpoint(),
    reason="Decoder PCC needs the checkpoint's real weights; HF_MODEL is a config-only stub",
)


def _run_multistep_decode(mesh_device, request, *, layer_type: str, prefill_len: int, bounded: bool = False) -> None:
    """``DECODE_STEPS`` decode steps from ``prefill_len``, comparing output and KV each step."""
    threshold = get_pcc_threshold(request)
    # Cover every position the run touches, rounded up to a whole page.
    max_seq_len = max(128, ((prefill_len + DECODE_STEPS + 63) // 64) * 64)
    ctx = build_decoder_pcc_context(mesh_device, layer_type, max_seq_len=max_seq_len, bounded=bounded)
    hf_cache = DynamicCache()

    if prefill_len:
        # Same tensor into both stacks, so decode starts from caches that agree.
        prefill_hidden = torch.randn(PCC_BATCH_SIZE, prefill_len, ctx.hidden_size, dtype=torch.float32)
        hf_prefill_out = hf_forward_span(ctx, prefill_hidden, start_pos=0, cache=hf_cache)
        tt_prefill_out = tt_prefill_chunk(ctx, mesh_device, prefill_hidden.to(torch.bfloat16), chunk_start=0)
        prefill_pass, _ = check_pcc(
            f"prefill seq_len={prefill_len} (decode seed)", hf_prefill_out, tt_prefill_out, threshold
        )
        assert prefill_pass, "Seed prefill already disagrees; the decode comparison below would be meaningless"

    logger.info(
        "Multi-step decode: layer={} ({}), kv={}, steps={}, positions {}..{}, batch={}, pcc>={}",
        ctx.layer_idx,
        layer_type,
        "bounded" if bounded else "unbounded",
        DECODE_STEPS,
        prefill_len,
        prefill_len + DECODE_STEPS - 1,
        PCC_BATCH_SIZE,
        threshold,
    )

    failures = []
    min_pcc = 1.0

    for step in range(DECODE_STEPS):
        position = prefill_len + step
        hidden = torch.randn(PCC_BATCH_SIZE, 1, ctx.hidden_size, dtype=torch.float32)

        hf_out = hf_forward_span(ctx, hidden, start_pos=position, cache=hf_cache)
        tt_out = tt_decode_step(ctx, mesh_device, hidden.to(torch.bfloat16), position=position)

        passing, pcc = check_pcc(f"decode step={step} pos={position}", hf_out, tt_out, threshold)
        min_pcc = min(min_pcc, pcc)
        if not passing:
            failures.append(f"output step={step} pos={position} pcc={pcc:.6f}")

        # The write this step just made must be readable at exactly `position`,
        # on every device, under that device's own KV heads.
        (tt_k, hf_k), (tt_v, hf_v) = compare_kv_cache(ctx, hf_cache, [position], written_through=position + 1)
        k_pass, k_pcc = check_pcc(f"  kv K step={step} pos={position}", hf_k, tt_k, KV_PCC_REQUIRED)
        v_pass, v_pcc = check_pcc(f"  kv V step={step} pos={position}", hf_v, tt_v, KV_PCC_REQUIRED)
        if not k_pass:
            failures.append(f"K cache step={step} pos={position} pcc={k_pcc:.6f}")
        if not v_pass:
            failures.append(f"V cache step={step} pos={position} pcc={v_pcc:.6f}")

    # Re-read every position the decode loop wrote: a later step must not have
    # clobbered an earlier one (same block, wrong row).
    written = [prefill_len + s for s in range(DECODE_STEPS)]
    (tt_k_all, hf_k_all), (tt_v_all, hf_v_all) = compare_kv_cache(
        ctx, hf_cache, written, written_through=prefill_len + DECODE_STEPS
    )
    hist_k_pass, hist_k_pcc = check_pcc("kv K all decode positions (final)", hf_k_all, tt_k_all, KV_PCC_REQUIRED)
    hist_v_pass, hist_v_pcc = check_pcc("kv V all decode positions (final)", hf_v_all, tt_v_all, KV_PCC_REQUIRED)
    if not hist_k_pass:
        failures.append(f"K cache history pcc={hist_k_pcc:.6f}")
    if not hist_v_pass:
        failures.append(f"V cache history pcc={hist_v_pcc:.6f}")

    logger.info("Multi-step decode min output PCC over {} steps: {:.6f}", DECODE_STEPS, min_pcc)
    assert not failures, (
        f"Multi-step decode failed for layer {ctx.layer_idx} ({layer_type}), "
        f"prefill_len={prefill_len}, bounded={bounded}, tp={ctx.tp}: " + "; ".join(failures)
    )


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
def test_decode_multistep_kv_update(layer_type, mesh_device, reset_seeds, request):
    """Decode from a cold cache: ``DECODE_STEPS`` steps at positions 0 … N-1."""
    _run_multistep_decode(mesh_device, request, layer_type=layer_type, prefill_len=0)


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "global"])
@pytest.mark.parametrize("prefill_len", [128, 1536], ids=["prefill128", "prefill1536"])
def test_decode_multistep_kv_update_after_prefill(layer_type, prefill_len, mesh_device, reset_seeds, request):
    """Prefill, then decode on: the demo's transition, over prefill-written pages."""
    _run_multistep_decode(mesh_device, request, layer_type=layer_type, prefill_len=prefill_len)


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("prefill_len", [1536], ids=["prefill1536"])
def test_decode_multistep_bounded_sliding_kv(prefill_len, mesh_device, reset_seeds, request):
    """The same multi-step decode against the **bounded sliding** KV ring.

    A sliding layer only ever reads the last ``sliding_window`` (1024) tokens, so
    above its long-context cutover the demo stops allocating one cache slot per
    position and switches to a 1024-token ring, wrapping every absolute position by
    ``cache_position_modulo`` before the page-table lookup. That wrap arithmetic is
    a different code path in all three paged ops, and the layout the demo actually
    runs at 128K+ — the tests above never touch it.

    ``prefill_len=1536`` is the point: it exceeds the 1024 window, so the seed
    prefill itself wraps (positions 1024–1535 land back on slots 0–511) and the ten
    decode steps then wrap on top of that. At any ``prefill_len`` below the window
    the ring never turns over and this would be the unbounded test with extra
    indirection.

    Sliding layers only — ``Gemma4Attention`` leaves the modulo unset on
    full-attention layers, so there is no bounded variant of one to test.

    Expected to agree with ``test_decode_multistep_kv_update_after_prefill``'s
    ``prefill1536-sliding`` case **digit for digit**, since ``reset_seeds`` gives
    both the same inputs and the ring is a storage change, not an arithmetic one.
    Measured on WH 1x8: all ten steps' PCCs match exactly on 31B and on 12B. If a
    future change makes them diverge, the wrap is doing something to the values.
    ``test_bounded_sliding_kv_cache_model.py`` asserts that parity directly at the
    attention level; this test's own gate is bounded-vs-HF.
    """
    _run_multistep_decode(mesh_device, request, layer_type="sliding_attention", prefill_len=prefill_len, bounded=True)
