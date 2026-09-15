# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-TAIL DECODE workload for Tracy / device-perf profiling of Qwen3.5/3.6.

Profiles the per-token work that is NOT inside a decoder layer: token embedding, the final
RMSNorm, the LM-head matmul (+ its logits all-gather on a vocab-sharded mesh), and greedy
sampling. Everything else in this directory profiles a layer; nothing profiled the tail, and the
tail is not small -- the LM head alone reads ``dim x vocab/tp`` of weights every single token,
which at the 27B's TP=8 shape is ~169 MB/device, i.e. the same order as a handful of decoder
layers put together.

WHY THE TAIL IS ITS OWN TEST
----------------------------
It has a completely different cost shape from a layer. A decoder layer at M=1 is a sequence of
modest matmuls plus two collectives; the tail is ONE enormous weight read (the LM head), one
tiny elementwise op (the norm), one gather-by-index (the embedding) and one reduction (argmax).
Only the LM head scales with vocab, and vocab is the one dimension no layer sees. Folding this
into the layer profilers would bury it: at ``n_layers=1`` it would dominate, and at the real 64
layers it would be 1/64th of a report that is 98% layer ops.

It is also the one place where the per-token cost does NOT amortize with batch: the LM-head
weight read is identical at B=1 and B=32 (M=1 vs M=32 against the same ``[dim, vocab/tp]``
matrix), so its per-token share drops 32x while every collective in it stays put. That crossover
is what the batch sweep below is for.

WHAT IS AND IS NOT INSIDE THE SIGNPOSTS
---------------------------------------
Model build, weight load and the token tensor are all outside, and a warmup iteration runs first
so nothing measured includes kernel compilation or first-touch allocation. The measured iteration
emits a nested signpost pair per STAGE -- ``embed_*``, ``norm_*``, ``lmhead_*``, ``sample_*`` --
inside one outer ``start``/``stop``, so each stage can be sliced out on its own::

    tt-perf-report --start-signpost lmhead_start --end-signpost lmhead_stop <csv>

Each stage syncs the device before its ``stop`` marker, because a signpost otherwise lands at
DISPATCH time and slices the wrong ops. That makes the op-to-op gap column meaningless here (it
already is in eager mode) -- read DEVICE KERNEL DURATION only.

``n_layers=1`` is deliberate and does not affect the numbers: the tail reads its own weights and
the layer count changes nothing about the embedding, the norm, the LM head or the sampler. It is
purely the cheapest build that yields a real ``Qwen36Model``, which is required because
``_embed`` / ``_final_norm_decode`` / ``_lm_head`` / ``_argmax_device`` are METHODS on the model,
not separable modules.

SAMPLING: BOTH PATHS, IN ONE CAPTURE
------------------------------------
Greedy decode has two implementations with very different tails, and this measures both off the
same matmul so the delta between them is directly readable.

``sample_shard`` is THE SERVED PATH. ``demo/text_demo.py`` defaults
``QWEN36_BATCHED_DECODE_MODE="shard"`` and sets ``model._ondev_argmax``, so both its B=1 and its
batched loop reduce each device's OWN ``[1,1,B,vocab/tp]`` shard to (argmax index, max value) on
device and read back two ``[num_devices, B]`` tensors. No full-vocab all-gather, no full logits
transfer. The stages below mirror ``_argmax_dev_b`` / ``_maxval_dev_b`` from that file op for op.

``lmhead_ag`` + ``sample_host`` are the LEGACY path (``_lm_head`` + ``Model._argmax_device``),
still reachable in eager mode and via ``QWEN36_BATCHED_DECODE_MODE=host``. They are profiled as
ADDITIONAL stages on the same logits rather than as an alternative run, because the interesting
number is what shard mode avoids -- text_demo's own comment says "sample" mode measured slower but
does not say by how much.

The framework ``SamplingGenerator`` (temperature/top-k/top-p) is out of scope here: it is only
reachable through the vLLM generator (``model.py`` touches it solely for ``max_batch_size``) and
consumes the same pre-gather shard, so its LM-head tail is the ``lmhead`` stage below.

Standalone Tracy capture::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      python -m tracy -p -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_model_tail_decode.py::test_profile_model_tail_decode[wormhole_b0-batch32-mesh_device0-device_params0]"

    Note the ``-m`` and the full node id with NO trailing pytest flags -- tracy parses argv with
    optparse and does not call disable_interspersed_args, so a trailing ``-v`` is taken as tracy's
    own verbose and ``-k`` fails as an unknown option.

Plain run (no profiler; sanity-checks the workload)::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_model_tail_decode.py -v -s
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

# Only the tail is profiled, so build the smallest model that still has one.
NUM_LAYERS = 1
DECODE_BATCH_SIZES = [1, 8, 32]
NUM_WARMUP_ITERS = 1


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

# Matches the other profilers and demo/text_demo.py so the capture reflects the served config;
# FABRIC_1D is required or the first CCL hangs and wedges the ETH cores.
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1024 * 1024 * 1024} if _MULTI else {}),
    }
]


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


class _TailPerfFixtures(NamedTuple):
    model: object
    batch_size: int
    tok: ttnn.Tensor
    per_shard: int
    maxval_r: int
    maxval_c: int


def _setup(mesh_device, batch_size: int) -> _TailPerfFixtures:
    """Build the model and the one decode-step token tensor. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=batch_size, max_seq_len=256, n_layers=NUM_LAYERS)

    # Decode ids arrive [B,1] and are flattened to [1,B] on the HOST -- that is the fused-tilize
    # precondition ttnn.embedding keys off (tp_common.decode_ids_for_embed), and reshaping [B,1]
    # on device instead would view the first padded row. Mirrors the real decode call sites.
    torch.manual_seed(0)
    tokens = torch.randint(0, 2000, (batch_size, 1), dtype=torch.int32)
    tok = ttnn.from_torch(
        tpc.decode_ids_for_embed(tokens),
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # Same reduction geometry text_demo.py's shard mode uses: reduce dim=-1 parallelizes over tile
    # ROWS, so a tall/narrow (R x 32) view of the shard lights ~R/32 cores instead of the 8 a
    # square-ish view would. Mirrored here so the profile reflects the served kernel, not a
    # differently-shaped one.
    per_shard = model.vocab_size // max(1, model.num_devices)
    maxval_c = 32
    maxval_r = (((per_shard + maxval_c - 1) // maxval_c) + 31) // 32 * 32
    logger.info(
        f"profiling model tail (decode), batch_size={batch_size}, mesh={MESH_SHAPE}, "
        f"vocab={model.vocab_size}, vocab_sharded_lm_head={model._lmhead_vocab_sharded}, "
        f"per_shard={per_shard}, maxval grid={maxval_r}x{maxval_c}"
    )
    return _TailPerfFixtures(
        model=model,
        batch_size=batch_size,
        tok=tok,
        per_shard=per_shard,
        maxval_r=maxval_r,
        maxval_c=maxval_c,
    )


def _sample_shard(f: _TailPerfFixtures, sharded_logits):
    """Per-shard greedy reduction: (local argmax index, local max value) per user, on device.

    Op-for-op mirror of ``_argmax_dev_b`` / ``_maxval_dev_b`` in ``demo/text_demo.py``'s shard mode.
    Deliberately duplicated rather than imported: those are closures inside the demo's decode loop,
    and this profiler must not depend on the demo's generation scaffolding. If the demo's reduction
    changes, this must change with it or the profile stops describing the served path.
    """
    Bn, R, C, per_shard = f.batch_size, f.maxval_r, f.maxval_c, f.per_shard

    # argmax leg: ttnn.argmax needs ROW_MAJOR; reduces the last dim per row, so Bn>1 is free.
    logits_rm = ttnn.to_layout(sharded_logits, ttnn.ROW_MAJOR_LAYOUT)
    idx = ttnn.argmax(logits_rm, dim=-1, keepdim=False)
    ttnn.deallocate(logits_rm)

    # max-value leg: view as (Bn, R1, C) and reduce twice. No pad on the shard when per_shard is
    # divisible by C (it is: 31,040 = 970*32); only the [Bn, R1] intermediate needs tile alignment
    # for the second reduce. Mirrors _maxval_dev / _maxval_dev_b after the 2026-08-21 change.
    R1 = -(-per_shard // C)
    if R1 * C != per_shard:
        padded = ttnn.pad(sharded_logits, [(0, 0), (0, 0), (0, 0), (0, R1 * C - per_shard)], value=-1e30)
        grid = ttnn.reshape(padded, (1, Bn, R1, C))
    else:
        padded, grid = None, ttnn.reshape(sharded_logits, (1, Bn, R1, C))
    part = ttnn.max(grid, dim=-1)
    part_row = ttnn.reshape(part, (1, 1, Bn, R1))
    if R != R1:
        part_pad = ttnn.pad(part_row, [(0, 0), (0, 0), (0, 0), (0, R - R1)], value=-1e30)
        ttnn.deallocate(part_row)
        part_row = part_pad
    val = ttnn.max(part_row, dim=-1)
    for t in (padded, grid, part, part_row):
        if t is not None:
            ttnn.deallocate(t)
    return idx, val


def _run_tail(mesh_device, f: _TailPerfFixtures, *, use_signpost: bool = False) -> None:
    """embed -> final norm -> LM head -> BOTH greedy tails (served shard, then legacy AG+host)."""
    m = f.model
    sp = None
    if use_signpost:
        from tracy import signpost as sp

        sp("start")

    def _stage(name, fn):
        if sp is not None:
            sp(f"{name}_start")
        out = fn()
        # Sync BEFORE the stop marker or the signpost lands at dispatch and slices the wrong ops.
        ttnn.synchronize_device(mesh_device)
        if sp is not None:
            sp(f"{name}_stop")
        return out

    x = _stage("embed", lambda: m._embed(f.tok))
    if m.num_devices > 1:
        # TP expects [1,1,B,dim_frac]; embd yields [B,1,dim_frac]. Metadata-only.
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))

    x = _stage("norm", lambda: m._final_norm_decode(x))

    # The matmul alone -- shared by BOTH tails, and by the vLLM sampler. This is the stage the
    # served path pays; `_lm_head` is this plus the all-gather timed separately below.
    logits = _stage("lmhead", lambda: ttnn.linear(x, m.lm_head_weight))
    ttnn.deallocate(x)

    # SERVED PATH (text_demo shard mode): reduce each device's own shard, read 2 tiny tensors.
    idx, val = _stage("sample_shard", lambda: _sample_shard(f, logits))
    ttnn.deallocate(idx)
    ttnn.deallocate(val)

    # LEGACY PATH, for the delta only: all-gather to full vocab, then argmax over all of it.
    if m._lmhead_vocab_sharded:
        from models.tt_transformers.tt.ccl import tt_all_gather

        # Same tuned kwargs `_lm_head` passes (wpl=4 / cps=25) -- an untuned AG here would
        # overstate what shard mode avoids.
        full = _stage(
            "lmhead_ag",
            lambda: tt_all_gather(
                logits,
                m.mesh_device,
                m.tt_ccl,
                cluster_axis=None,
                dim=len(logits.shape) - 1,
                topology=m.args.ccl_topology(),
                num_workers_per_link=4,
                chunks_per_sync=25,
            ),
        )
        ttnn.deallocate(logits)
        _stage("sample_host", lambda: m._argmax_device(full))
        ttnn.deallocate(full)
    else:
        ttnn.deallocate(logits)

    if sp is not None:
        sp("stop")


@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("batch_size", DECODE_BATCH_SIZES, ids=[f"batch{b}" for b in DECODE_BATCH_SIZES])
def test_profile_model_tail_decode(mesh_device, device_params, batch_size):
    """One decode step through the model tail: embedding, final norm, LM head, sampling."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, batch_size)

    for _ in range(NUM_WARMUP_ITERS):
        _run_tail(mesh_device, f)

    _run_tail(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.tok)
    logger.info(f"Tail profile workload complete: batch_size={batch_size}, signposts={'on' if use_signpost else 'off'}")
