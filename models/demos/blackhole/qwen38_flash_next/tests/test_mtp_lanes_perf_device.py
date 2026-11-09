# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Timing test: paged MTP lanes passes (8 lanes, K = 3) at a long context.

``test_lanes_perf`` ids ``ctx32k``, ``ctx128k`` and ``ctx256k``: the chain is opened with the id's allocated context on
``Qwen38LanePaging`` (block size 128), 8 lanes are admitted with distinct prompts and distinct block lists, ``WARM``
passes run untimed and ``PASSES`` passes are timed through ``Qwen38VLLMMTPLanesDriver.verify``.  The summary goes to the
log and to ``$QWEN38_PAGED_TEST_OUT/perf_<id>.json``.

Run:  QWEN38_PAGED_TEST_OUT=<dir> pytest <this file> -v -s -k "ctx32k"
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import _lanes_warm_hook
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import needle_prompts
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import construct_live_decode_diagnostic
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    Qwen38ChatChainError,
    Qwen38ChatSession,
    Qwen38TracedChain,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import prefill_form, prepare_vllm_construction
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38VLLMMTPLanesDriver, VerifyRow
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging, paged_table_width, validate_index_paging

LANES = 8
LIVE = 8
DRAFTS = 3
PAGE_SIZE = 128
WARM = 5
PASSES = 40
OUT_VARIABLE = "QWEN38_PAGED_TEST_OUT"
CONTEXTS = {
    "ctx32k": (32768, 16000),
    "ctx128k": (131072, 48000),
    "ctx256k": (262144, 60000),
}


def out_dir() -> Path:
    value = os.environ.get(OUT_VARIABLE)
    if not value:
        pytest.skip(f"set {OUT_VARIABLE} to the directory the test writes its results to")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_paged_chain(mesh_device, checkpoint, lanes_session, allocated_context: int, num_pages: int):
    """The chain on the lanes' paging at ``allocated_context``, the way the adapter's chain opener builds it."""

    prepared, allocated = prepare_vllm_construction(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, allocated_context - RESIDENT_CONTEXT_HEADROOM
    )
    assert allocated == allocated_context, allocated
    paging = validate_index_paging(
        Qwen38LanePaging(
            page_size=PAGE_SIZE, num_pages=num_pages, max_blocks=paged_table_width(allocated_context, PAGE_SIZE)
        ),
        allocated_context,
    )
    long_chunks, slab_rows = prefill_form()
    construction = construct_live_decode_diagnostic(
        prepared, mesh_device=mesh_device, collective_topology=ttnn.Topology.Linear, marker=logger.info
    )

    def warm_hook(chain) -> None:
        _lanes_warm_hook(chain.mesh, lanes_session, allocated_context, long_chunks or slab_rows is not None)(chain)

    logger.info(f"perf test: page table width {paging.max_blocks}, {num_pages} blocks of {PAGE_SIZE}")
    return Qwen38TracedChain.open(
        construction,
        marker=logger.info,
        chunked_prefill=True,
        long_chunks=long_chunks,
        slab_rows=slab_rows,
        paging=paging,
        sampling=False,
        mtp=DRAFTS,
        mtp_ple_early=False,
        warm_hook=warm_hook,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("ctx", list(CONTEXTS))
@pytest.mark.timeout(9000)
def test_lanes_perf(mesh_device, ctx):
    out = out_dir()
    allocated_context, prompt_len = CONTEXTS[ctx]
    checkpoint = checkpoint_directory()
    prompts, _tokenizer, _config = needle_prompts(checkpoint, LIVE, prompt_len)
    lengths = [len(p) for p in prompts]
    assert len({tuple(p) for p in prompts}) == LIVE, "lane prompts must differ"
    per_pass = DRAFTS + 1
    lane_block_counts = [-(-(n + (WARM + PASSES) * per_pass + DRAFTS + 8) // PAGE_SIZE) for n in lengths]
    num_pages = 1 + sum(lane_block_counts) + 16
    lane_blocks = []
    next_block = 1
    for count in lane_block_counts:
        lane_blocks.append(list(range(next_block, next_block + count)))
        next_block += count
    logger.info(f"perf {ctx}: prompt lengths {lengths}, blocks per lane {lane_block_counts}, pool {num_pages}")
    mesh_device.enable_program_cache()
    session = Qwen38LanesSession(lanes=LANES, drafts=DRAFTS, marker=logger.info)
    try:
        chain = open_paged_chain(mesh_device, checkpoint, session, allocated_context, num_pages)
    except Qwen38ChatChainError as error:
        if "refused" in str(error):
            session.close()
            pytest.skip(str(error))
        raise
    try:
        chat = Qwen38ChatSession(chain, None, allocated_context=allocated_context)
        chat.reset()
        session.capture(chain, chat)
        logger.info("lanes captured")
        driver = Qwen38VLLMMTPLanesDriver(session, LANES, DRAFTS, paged=True)
        admit_seconds = []
        for lane in range(LIVE):
            ids = lane_blocks[lane][: -(-lengths[lane] // PAGE_SIZE)]
            started = time.perf_counter()
            driver.admit(lane, prompts[lane], block_ids=ids)
            seconds = time.perf_counter() - started
            admit_seconds.append(seconds)
            logger.info(
                f"admit lane {lane}: {lengths[lane]} tokens in {seconds:.2f} s ({lengths[lane] / seconds:.0f} tok/s)"
            )

        def lane_row(lane: int) -> VerifyRow:
            run = driver.runs[lane]
            if run.fresh:
                position = run.position
                row = VerifyRow(lane, [run.pending] + [0] * DRAFTS, position, 0, run.settled)
            else:
                position = run.next_position
                row = VerifyRow(lane, list(run.next_tokens), position, DRAFTS, run.device_rows)
            need = min(position + DRAFTS, allocated_context - 1) // PAGE_SIZE + 1
            assert need <= len(lane_blocks[lane]), f"lane {lane} needs {need} blocks at position {position}"
            row.pages = lane_blocks[lane][:need]
            return row

        times_ms = []
        committed = 0
        for index in range(WARM + PASSES):
            rows = [lane_row(lane) for lane in range(LIVE)]
            started = time.perf_counter()
            driver.verify(rows)
            elapsed = time.perf_counter() - started
            record = session.lane_chain.records[-1]
            if index >= WARM:
                times_ms.append(elapsed * 1000.0)
                committed += sum(record.accepted[lane] + 1 for lane in range(LIVE))
            logger.info(
                f"pass {index} {'timed' if index >= WARM else 'warm'} {elapsed * 1000.0:.2f} ms "
                f"accepted {[record.accepted[lane] for lane in range(LIVE)]}"
            )
        mean = statistics.fmean(times_ms)
        median = statistics.median(times_ms)
        p90 = percentile(times_ms, 0.9)
        tpp = committed / (PASSES * LIVE)
        median_s = median / 1000.0
        admit_mean = statistics.fmean(admit_seconds)
        logger.info(
            f"PERF ctx={ctx} lanes=8 K=3 prompt={prompt_len} | pass mean {mean:.2f} ms median {median:.2f} ms "
            f"p90 {p90:.2f} ms | tokens/pass/lane {tpp:.3f} | per-lane {tpp / median_s:.1f} tok/s | "
            f"aggregate {8 * tpp / median_s:.1f} tok/s | admit mean {admit_mean:.2f} s"
        )
        (out / f"perf_{ctx}.json").write_text(
            json.dumps(
                {
                    "ctx": ctx,
                    "allocated_context": allocated_context,
                    "prompt_len": prompt_len,
                    "prompt_lengths": lengths,
                    "pass_ms": times_ms,
                    "mean_ms": mean,
                    "median_ms": median,
                    "p90_ms": p90,
                    "tokens_per_pass_per_lane": tpp,
                    "per_lane_tok_s": tpp / median_s,
                    "aggregate_tok_s": 8 * tpp / median_s,
                    "admit_seconds": admit_seconds,
                    "admit_mean_s": admit_mean,
                },
                indent=2,
            )
        )
    finally:
        session.close()
        chain.close()
