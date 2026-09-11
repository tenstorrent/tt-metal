# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""FEASIBILITY PROBE: can the DFlash **verify** forward be trace-captured?

This is the one that matters. The verify forward is **94 % of a speculative step**
(README-DFLASH.md), the whole loop is 0.18x production traced decode, and the README's priority #1
is "trace the speculative step". The drafter — 6 % — is probed separately in
``tests/perf/test_dflash_drafter_trace_probe.py`` and is the *harder* of the two (its KV commit
lands at a 1-16 row offset and ``slice_write`` requires a multiple of TILE_HEIGHT).

WHY THIS IS NOT OBVIOUSLY IMPOSSIBLE, despite the docstrings saying so
---------------------------------------------------------------------
``capture_prefill_trace_bucket`` says "valid_len cannot be masked inside a trace, so a short prompt
would pad through the GDN recurrence and corrupt the decode state", and routes masked prefills to
the eager path. But read what the masking actually does — ``tt/gdn/fused_chunk.py``:

    _mt = torch.zeros(B, T, 1); _mt[:, :valid_len, :] = 1.0
    _m = ttnn.from_torch(_mt, ...)
    beta = ttnn.multiply(beta, _m); g = ttnn.multiply(g, _m)
    q = ttnn.multiply(q, _mq); k = ...; v = ...

Every op here is **fixed-shape**. ``valid_len`` changes the mask's CONTENTS, not any op's shape or
arguments. The only thing that blocks a capture is the inline ``ttnn.from_torch`` — a host write —
and this repo has already solved exactly that shape of problem twice: the vision splice is "a
trace-safe fixed-shape ``ttnn.where`` over persistent buffers", and SDPA takes ``chunk_start`` as a
device tensor because "host-int chunk_start compiles per position and can clobber parked trace".

So the hypothesis is: masked-bucket verify is trace-able once every per-step host write is moved to
a persistent buffer staged before ``execute_trace``. This file finds out which writes those are, by
attempting the capture and reporting what stops it — one blocker at a time, since each fix reveals
the next.

RESULT: **the masked verify forward CAPTURES** once every per-step host write is staged.

    warmed: masked bucket for a 16-token block, bucket=128, inputs STAGED
    VERIFY-FORWARD CAPTURE: SUCCEEDED

That disproves the premise the model's own docstrings encode three times over -- that a masked
bucket cannot be traced because "valid_len cannot be masked inside a trace". It can. The masking is
fixed-shape ``multiply``; only the mask CONTENTS vary, and contents can be staged.

**A SUCCESSFUL CAPTURE IS NOT A CORRECT REPLAY.** Nothing here has checked that replaying the trace
produces the eager forward's logits, GDN recurrent state, or paged KV. Two specific risks are still
untested: the padding K/V written to FUTURE positions by the fixed-width page table (argued
harmless because speculation rewrites them -- argued, not measured), and whether the staged buffers
are read at the right point of each replay. Do not wire this into TtTarget before that check.

It took EIGHT blockers, and the enumeration from reading source found six of them. Every extra one
surfaced only when the probe reached it, a level deeper than static reading went: the GDN mask
(fused_chunk.py), then the conv1d one-hot selector (ttnn_gated_deltanet.py), then ``ttnn.arange``
inside that selector's own replacement. Two confident in-repo comments were measured WRONG along
the way -- "padded writes would corrupt block 0" (they do not, see
tests/unit/test_paged_fill_cache_width.py) and conv_fir_wh.py's "Built ENTIRELY ON DEVICE ... that
also makes this trace-safe" (``arange`` uploads; that fork would fail the same way if traced).

EARLIER FAILURE, for the record:

    VERIFY-FORWARD CAPTURE: FAILED -> TT_FATAL fd_mesh_command_queue.cpp:789: !trace_id_.has_value()

Line 789 is "Writes are not supported during trace capture". **This is the good failure.** Contrast
the drafter probe, which dies on "Cannot load new binaries during trace capture" -- i.e. its shapes
change every step so its programs are never cached. This one got *past* every program-cache check
and stopped only on a host write, which means the masked verify forward **is shape-stable**. The
hypothesis above holds: what blocks it is writes, not shapes.

THE BLOCKER LIST -- note it was NOT complete on the first pass. Reading the top-level forward
found six writes; a seventh lives two levels down inside the GDN conv1d and only surfaced when the
probe printed frames. Enumerating by reading one function is not sufficient here; the probe is.

  7. ``_causal_conv1d_fir`` -- ``ttnn_gated_deltanet.py:199`` uploads a one-hot selector built with
     ``torch.zeros``. **The trace-safe version already exists in this repo**: the Wormhole fork
     ``conv_fir_wh.py:_causal_conv1d_fir_wh`` builds the identical selector with
     ``ttnn.arange``/``ttnn.eq`` -- "Built ENTIRELY ON DEVICE ... That also makes this trace-safe
     ... Verified bit-exact vs the torch.zeros fill". It is simply not reached on T3K:
     ``_use_wh = tpc.wh_9b_n300(model_args)`` gates it to the 9B on N300, a narrowing the file
     documents as deliberate and "Accepted per explicit instruction".

THE FIRST SIX (enumerated from ``_forward_prefill_chunk_masked_tp``, model.py:2272)

Six per-step host writes. Five are fixed-shape -- only their CONTENTS vary -- so each becomes a
persistent buffer plus a ``copy_host_to_device_tensor`` staged before ``execute_trace``, which is
exactly what ``prefill_traced_bucket_batched`` already does for its own inputs:

  1. ``tok``         token ids, [1, bucket] uint32                      fixed shape
  2. ``cos``/``sin`` [1,1,bucket,rope_w] bf16, contents vary with chunk_start   fixed shape
  3. ``full_pt``     page table, [1, NUM_BLOCKS] int32                  fixed, stage once
  5. ``csi_tensor``  [1] int32 holding chunk_start                      fixed shape
  6. GDN mask        ``fused_chunk.py`` ``_m``/``_mq``, [B,T,1]         fixed shape

  Note on 2: ``_rope_tp_cos_sin_dev`` already computes cos/sin on device, but its docstring says
  "for EAGER callers only" -- it slices the resident tables at a position-dependent offset, so it
  trades a host write for varying runtime args. Neither form is trace-safe; stage the buffer.

  4. ``chunk_pt``  -- THE ONE THAT NEEDS DESIGN. Width is ``blkN - blk0`` where
     ``blkN = num_blocks_in_seq(chunk_start + valid_len, 64)``, so the SHAPE varies with valid_len,
     and ``attention/tp.py:1182`` notes "chunk_start_idx (int) still sizes the page table
     host-side". A 128-row bucket spans at most ceil(128/64)+1 = 3 blocks whatever the alignment,
     so a fixed width with per-step contents is the fix. **RESOLVED, measured** in
     ``tests/unit/test_paged_fill_cache_width.py``: ``paged_fill_cache`` writes exactly
     ``min(S, page_len)`` rows at their correct positions and leaves surplus pages untouched, so a
     wider table does NOT corrupt block 0 (that model.py warning is about the batched path). Use a
     fixed width of ``bucket // block_size``; the rows past ``valid_len`` get padding K/V at FUTURE
     positions, which the next speculative step rewrites. Holds only while ``chunk_start`` stays
     block-aligned, which the target's whole-bucket anchor guarantees.

WHAT IT DOES NOT DO
-------------------
It does not attempt correctness of a replay. A capture that succeeds is necessary, not sufficient:
the GDN recurrent state and the paged-KV offset both advance per step, and a replay must be checked
against the eager path before any of this is believed.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_target_trace_probe.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
BLOCK = 16  # a speculative block: what the verify forward actually carries


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": TRACE_REGION,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_target_trace_probe(mesh_device, device_params, reset_seeds, ensure_gc):
    """Attempt to capture one masked-bucket verify forward; report the first thing that stops it."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    bucket = model._mask_bucket_for(BLOCK)
    ids = torch.randint(1000, 2000, (1, BLOCK), dtype=torch.int32)
    token_buf = torch.zeros(1, bucket, dtype=torch.int32)
    token_buf[:, :BLOCK] = ids

    # Capture the FORWARD only, not prefill_block_all_logits: that wrapper ends in ttnn.to_torch,
    # a device->host READ, which is as illegal inside a capture as a write. The traced verify will
    # need to leave its logits on device and read them after the replay; isolating the forward here
    # keeps this probe pointed at the input-staging question.
    model._reset_gdn_state_for_new_sequence()
    staged = model.alloc_verify_buffers(bucket, page_table)
    model.stage_verify_inputs(token_buf, BLOCK, chunk_start=0, page_table=page_table, bucket=bucket)

    def fwd():
        return model._forward_prefill_chunk_masked_tp(
            token_buf, BLOCK, 0, page_table, bucket, flex_sdpa=True, staged=staged
        )

    # Warm: the same call twice, so every program in the masked-bucket path is in the cache. A
    # capture that fails with "not yet in program cache" after this is telling us the step is not
    # shape-stable, not that it needs more warm-up (the lesson from the drafter probe).
    for _ in range(2):
        ttnn.deallocate(fwd())
    ttnn.synchronize_device(mesh_device)
    logger.info(f"warmed: masked bucket for a {BLOCK}-token block, bucket={bucket}, inputs STAGED")

    err = None
    tid = None
    try:
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = fwd()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        tid = None
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
    except Exception as e:  # noqa: BLE001 — the failure IS the result
        import traceback

        err = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        # The message alone does not say WHICH write. The frames do, and naming the exact
        # file:line of the offending ttnn.from_torch is the whole point of this probe.
        frames = [fr for fr in traceback.format_exc().splitlines() if fr.strip().startswith("File ")]
        for fr in frames[-6:]:
            logger.info(f"  frame: {fr.strip()}")
    finally:
        # ALWAYS close the capture. begin_trace_capture puts the command queue into capture mode;
        # if the body raises and we never end it, the queue stays there and the process HANGS in
        # device teardown -- 12 minutes of silence, then the next run blocks on a device lock it
        # reports as held by the corpse. Every wedged run in this port traced back to this.
        if tid is not None:
            try:
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            except Exception as e2:  # noqa: BLE001
                logger.warning(f"  could not close the capture: {type(e2).__name__}: {str(e2)[:120]}")

    logger.info("=" * 100)
    logger.info(f"VERIFY-FORWARD CAPTURE: {'FAILED -> ' + err if err else 'SUCCEEDED'}")
    logger.info("=" * 100)
    print(f"\n>>> target_capture_error={err}\n")
