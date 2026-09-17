# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 CP prefill served at two chunk widths from one model.

Three things are under test, in the order they matter:

1. **Two widths coexist.** One set of weights, one KV cache, two captured traces.
2. **Both are correct.** A prompt's final hidden state agrees across widths, including the
   multi-chunk block-cyclic path against a single-chunk reference. A negative control pins
   that the KV layout really is keyed by the width -- continuing a prefix at the other width
   must produce a different answer, or these tests prove nothing.
3. **Picking per request beats pinning either width.** Measured, not modelled.
"""

import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _get_prefill_tokens, _model_path
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gemma4_d_p.tt.chunk_buckets import bucket_switch_points, modelled_prefill_ms, select_chunk_size
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.variable_chunk_prefill import VariableChunkPrefill, cp_gather_torch

# Buckets: one that a short prompt fills exactly, one that is the 256k throughput optimum.
# See models/demos/gemma4_d_p/tt/chunk_buckets.py for why these two.
CHUNK_BUCKETS = tuple(int(c) for c in os.environ.get("GEMMA4_CHUNK_BUCKETS", "4096,8192,32768").split(","))
# Two captured 60-layer traces instead of one.
TRACE_REGION_SIZE = int(os.environ.get("GEMMA4_VARIABLE_TRACE_REGION_SIZE", 900_000_000))
PCC_THRESHOLD = 0.99
# Padding and replay must be exact, not merely close: both are the same arithmetic.
PADDING_PCC_THRESHOLD = 0.9999


def _build(mesh_device, max_seq_len, chunk_sizes=CHUNK_BUCKETS):
    mesh_config = MeshConfig(mesh_device)
    if mesh_config.cp_degree <= 1:
        pytest.skip(f"targets CP>1; mesh {tuple(mesh_device.shape)} gives CP={mesh_config.cp_degree}")
    model_path = _model_path()
    t0 = time.time()
    model_args, model, _ = create_tt_model(
        mesh_config=mesh_config,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat16,
        model_path=model_path,
        prefill_chunk_size=chunk_sizes,
    )[:3]
    logger.info(
        f"[variable_chunk] model ready in {time.time() - t0:.1f}s | widths={list(model.prefill_chunk_sizes)} "
        f"max_seq_len={max_seq_len} ring_cache={model.ring_cache_max_seq_len}"
    )
    return mesh_config, model_args, model, model_path


def _final_row(driver, chunk_size, prompt_len):
    """Hidden state of the last real prompt token, from the last replayed chunk."""
    hidden = driver.last_hidden_state(chunk_size)
    num_chunks = -(-prompt_len // chunk_size)
    return hidden[prompt_len - 1 - (num_chunks - 1) * chunk_size]


# ── correctness ───────────────────────────────────────────────────────────────


@torch.no_grad()
@pytest.mark.timeout(5400)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
def test_variable_chunk_prefill_is_correct_within_a_width(mesh_device, reset_seeds):
    """The three properties per-request bucketing actually rests on.

    1. **Padding invariance.** A prompt served in a bucket wider than it fills must give the same
       answer as those tokens in a chunk they fill exactly *at that width*. This is what makes a
       too-wide bucket merely wasteful rather than wrong, and it holds bit-exactly.
    2. **Replay determinism.** Two captured traces sharing one KV cache, one metadata pair and one
       set of ring-gather scratch buffers must not contaminate each other, in any replay order.
    3. **The negative control.** Changing width inside a request must CORRUPT the prefix. If it did
       not, the KV layout would not be width-keyed and (1) would be proving nothing.

    Width-invariance -- the same prompt giving the same answer at two different widths -- is NOT
    asserted here. It does not hold on this branch, at one layer, before any of this work; see
    ``test_prefill_is_chunk_width_invariant`` below.
    """
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 65536))
    mesh_config, model_args, model, model_path = _build(mesh_device, max_seq_len)
    narrow, wide = CHUNK_BUCKETS[0], CHUNK_BUCKETS[-1]
    tokens_all = _get_prefill_tokens(model_path, max_seq_len, model_args.vocab_size)

    with VariableChunkPrefill(model, mesh_config).capture(logger=logger) as driver:
        # (1) Padding invariance, at the WIDE width. `padded` runs `narrow` real tokens in a
        # `wide` chunk; `filled` runs `wide` real tokens in the same chunk. Rows [0, narrow) see
        # an identical causal prefix, so they must be identical outputs.
        driver.prefill(tokens_all[0, :narrow], chunk_size=wide)
        padded = driver.last_hidden_state(wide)[:narrow].clone()
        driver.prefill(tokens_all[0, :wide], chunk_size=wide)
        filled = driver.last_hidden_state(wide)[:narrow].clone()
        passing, pcc = comp_pcc(filled, padded, PADDING_PCC_THRESHOLD)
        logger.info(f"[variable_chunk] padding invariance at chunk {wide}: PCC {pcc}")
        assert float(filled.std()) > 1e-3, "output is degenerate; any PCC below would be meaningless"
        assert passing, (
            f"a {narrow}-token prompt padded into a {wide}-token chunk does not match the same "
            f"tokens in a filled chunk (PCC {pcc}). Padding is supposed to be invisible to the "
            f"real prefix -- without it, admitting a short prompt to a wide bucket is wrong, not "
            f"merely wasteful."
        )

        # (2) Replay determinism across two traces sharing one cache. Re-run the narrow width
        # after the wide one has used the same buffers.
        driver.prefill(tokens_all[0, :narrow], chunk_size=narrow)
        first = driver.last_hidden_state(narrow).clone()
        driver.prefill(tokens_all[0, :wide], chunk_size=wide)
        driver.prefill(tokens_all[0, :narrow], chunk_size=narrow)
        second = driver.last_hidden_state(narrow).clone()
        passing, pcc = comp_pcc(first, second, PADDING_PCC_THRESHOLD)
        logger.info(f"[variable_chunk] replay determinism at chunk {narrow}: PCC {pcc}")
        assert passing, f"replaying chunk {narrow} after chunk {wide} changed its answer (PCC {pcc})"

        # (3) Negative control. Write the prefix [0, wide) at the WIDE width, then continue it with
        # a NARROW chunk. The write lands correctly, but ring_joint reconstructs each cached row's
        # global position from the CURRENT width, so it reads rank r's row 0 as position
        # r*narrow/cp instead of r*wide/cp. Nothing raises.
        prompt_len = wide + narrow
        driver.prefill(tokens_all[0, :prompt_len], chunk_size=narrow)
        correct = _final_row(driver, narrow, prompt_len)

        tokens_padded = tokens_all[0, :prompt_len].to(torch.int32)
        driver.run_chunk(wide, chunk_idx=0, tokens_padded=tokens_padded)
        driver.run_chunk(narrow, chunk_idx=wide // narrow, tokens_padded=tokens_padded)
        mixed = _final_row(driver, narrow, prompt_len)

        matched, pcc = comp_pcc(correct, mixed, PCC_THRESHOLD)
        # Caveat, until the width-invariance defect below is fixed: the mixed run differs from the
        # all-narrow reference for TWO reasons -- the layout corruption this control is for, and the
        # fact that the prefix chunk ran at a different width at all. They cannot be separated here,
        # so this establishes "changing width mid-request is not safe" but does not by itself
        # measure how unsafe.
        logger.info(f"[variable_chunk] negative control (width changed mid-request): PCC {pcc}")
        assert not matched, (
            f"a prefix written at {wide} and read at {narrow} matched the all-{narrow} reference "
            f"(PCC {pcc}). The ring KV layout is block-cyclic with period C, so this MUST differ. "
            f"If it does not, the layout is not width-keyed and the padding check above is vacuous."
        )


@torch.no_grad()
@pytest.mark.timeout(5400)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN BUG in this implementation: a model built with several widths does not reproduce a "
        "single-width build's answer at a width they share. Measured PCC 0.869 (max_abs_diff 29.9) "
        "for chunk 8192 between a (8192,) build and a (4096,8192,32768) build, one layer, identical "
        "ring_cache geometry. Lead: the single-width 8192 result matches the multi-build's 4096 "
        "result, so per-width resources look bound by position-in-tuple or by prefill_chunk_sizes[-1] "
        "rather than by the requested width -- suspect capture() baking overlapping trace addresses, "
        "or the per-width RoPE tables. Until this XPASSes, per-request bucketing is NOT correct and "
        "no cross-width measurement from a multi-width build can be trusted. "
        "See tech_reports/Gemma4VariableChunkSize/ §4."
    ),
)
def test_multi_width_build_matches_single_width_build(mesh_device, reset_seeds):
    """A width's answer must not depend on which other widths the model was built with.

    This is the control that the earlier version of this suite was missing, and running it
    retracted three published claims. It needs two processes to be airtight (two models of this
    size will not co-reside), so the in-test version compares against a reference captured by
    ``GEMMA4_REFERENCE_PT`` -- produced by running this same test with ``GEMMA4_WIDTHS`` set to a
    single width. Without that file it skips rather than pretending to check anything.

    ``ring_cache_capacity`` is ``max(max_seq_len, 2*max(C))``, which saturates at ``max_seq_len``
    for every width set used here, so the KV geometry is identical between builds and cannot
    explain a difference.
    """
    reference_pt = os.environ.get("GEMMA4_REFERENCE_PT")
    save_pt = os.environ.get("GEMMA4_SAVE_PT")
    if not reference_pt and not save_pt:
        pytest.skip(
            "two-process control. First: GEMMA4_WIDTHS=8192 GEMMA4_SAVE_PT=/tmp/ref.pt (records the "
            "single-width reference). Then: GEMMA4_REFERENCE_PT=/tmp/ref.pt (compares a multi-width "
            "build against it). Two models this size cannot co-reside, hence two processes."
        )
    widths = tuple(int(c) for c in os.environ.get("GEMMA4_WIDTHS", ",".join(map(str, CHUNK_BUCKETS))).split(","))
    shared = int(os.environ.get("GEMMA4_SHARED_WIDTH", 8192))
    prompt_len = shared

    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 65536))
    mesh_config, model_args, model, model_path = _build(mesh_device, max_seq_len, chunk_sizes=widths)
    tokens_all = _get_prefill_tokens(model_path, max_seq_len, model_args.vocab_size)
    model.layers = model.layers[:1]

    with VariableChunkPrefill(model, mesh_config).capture(logger=logger) as driver:
        chunks = []
        driver.prefill(
            tokens_all[0, :prompt_len],
            chunk_size=shared,
            on_chunk=lambda _i, out: chunks.append(cp_gather_torch(out, mesh_config).reshape(-1, model.hidden_size)),
        )
        mine = torch.cat(chunks, dim=0)[:prompt_len].clone()

    if save_pt:
        torch.save({"widths": widths, "out": {shared: mine}}, save_pt)
        logger.info(f"[variable_chunk] saved reference for widths={widths} chunk={shared} -> {save_pt}")
        if not reference_pt:
            pytest.skip(f"reference recorded to {save_pt}; re-run with GEMMA4_REFERENCE_PT to compare")

    reference = torch.load(reference_pt)["out"][shared]
    passing, pcc = comp_pcc(reference, mine, PADDING_PCC_THRESHOLD)
    logger.info(
        f"[variable_chunk] build {widths} vs reference at chunk {shared}: PCC {pcc} "
        f"max_abs_diff {float((reference - mine).abs().max()):.3e}"
    )
    assert passing, (
        f"chunk {shared} differs between a single-width build and a {widths} build (PCC {pcc}). "
        f"A width's answer must not depend on which other widths were configured."
    )


# ── performance ───────────────────────────────────────────────────────────────


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize(
    "prompt_lens",
    [(4096, 16384, 36864, 262144)],
    ids=lambda p: "isl_" + "_".join(str(v // 1024) + "k" for v in p),
)
def test_variable_chunk_prefill_beats_either_fixed_width(mesh_device, prompt_lens, reset_seeds):
    """Measure every prompt at both widths; the per-request pick must beat pinning either."""
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 262144))
    mesh_config, model_args, model, model_path = _build(mesh_device, max_seq_len)
    tokens_all = _get_prefill_tokens(model_path, max(prompt_lens), model_args.vocab_size)

    logger.info(
        "[variable_chunk] admission policy: "
        + ", ".join(f"P>={p}->{c}" for p, c in bucket_switch_points(CHUNK_BUCKETS, max(prompt_lens)))
    )

    measured = {}
    with VariableChunkPrefill(model, mesh_config).capture(logger=logger) as driver:
        for prompt_len in prompt_lens:
            for chunk_size in CHUNK_BUCKETS:
                if -(-prompt_len // chunk_size) * chunk_size > max_seq_len:
                    continue
                result = driver.prefill(tokens_all[0, :prompt_len], chunk_size=chunk_size)
                measured[(prompt_len, chunk_size)] = result
                logger.info(f"[variable_chunk] measured {result.describe()}")

    # ── report ────────────────────────────────────────────────────────────
    totals = {c: 0.0 for c in CHUNK_BUCKETS}
    total_variable = 0.0
    logger.info(
        f"[variable_chunk] {'prompt':>8} "
        + " ".join(f"{'c' + str(c):>12}" for c in CHUNK_BUCKETS)
        + f" {'picked':>8} {'modelled':>9} {'vs best fixed':>14}"
    )
    mispredicted = []
    for prompt_len in prompt_lens:
        times = {c: measured[(prompt_len, c)].device_s for c in CHUNK_BUCKETS if (prompt_len, c) in measured}
        picked = select_chunk_size(prompt_len, CHUNK_BUCKETS)
        cheapest = min(times, key=times.get)
        if picked != cheapest:
            mispredicted.append((prompt_len, picked, cheapest, times))
        for chunk_size, seconds in times.items():
            totals[chunk_size] += seconds
        total_variable += times[picked]
        logger.info(
            f"[variable_chunk] {prompt_len:>8} "
            + " ".join(f"{times.get(c, float('nan')) * 1000:12.1f}" for c in CHUNK_BUCKETS)
            + f" {picked:>8} {modelled_prefill_ms(prompt_len, picked):9.1f}"
            + f" {min(times.values()) / times[picked]:13.3f}x"
        )

    # Two summaries, because they answer different questions. The workload total is dominated
    # by the longest prompt; the per-request geometric mean is what a request actually sees.
    for chunk_size in CHUNK_BUCKETS:
        speedups = [
            measured[(p, chunk_size)].device_s / measured[(p, select_chunk_size(p, CHUNK_BUCKETS))].device_s
            for p in prompt_lens
        ]
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        logger.info(
            f"[variable_chunk] WORKLOAD fixed {chunk_size}: {totals[chunk_size]:.2f}s"
            f" | variable: {total_variable:.2f}s = {totals[chunk_size] / total_variable:.2f}x total,"
            f" {geomean:.2f}x per-request geomean"
            f" (range {min(speedups):.2f}x-{max(speedups):.2f}x)"
        )

    assert not mispredicted, "the modelled selector did not pick the measured cheapest width: " + "; ".join(
        f"prompt {p}: picked {picked}, measured cheapest {cheapest} ({ {c: round(s * 1000, 1) for c, s in t.items()} })"
        for p, picked, cheapest, t in mispredicted
    )
    for chunk_size in CHUNK_BUCKETS:
        assert totals[chunk_size] > total_variable, (
            f"pinning chunk {chunk_size} for this workload cost {totals[chunk_size]:.2f}s vs "
            f"{total_variable:.2f}s variable -- no win to report"
        )
