# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Diagnosis test: the MTP lanes on the chain's shared paged pools against the flat lanes, same prompts, K = 4.

``test_lanes_stream`` ids ``flat`` and ``paged``: 4 lanes, 2 live (prompts of ``needle_prompts``), ``PASSES`` passes
through ``Qwen38VLLMMTPLanesDriver.verify`` as the vLLM adapter calls it.  The paged case opens the chain with
``Qwen38LanePaging`` (block size 128, the adapter's page-table width), prepares the lanes on it, admits the lanes
with distinct block ids and gives every verify row the blocks that cover ``position + K``.  Tokens go to
``$QWEN38_PAGED_TEST_OUT/<id>.pt``; ``test_lanes_compare`` (no device) checks flat == paged bitwise.

Run:  QWEN38_PAGED_TEST_OUT=<dir> pytest <this file> -v -s -k "paged"
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import _lanes_warm_hook, open_mtp_chain
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import needle_prompts
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import construct_live_decode_diagnostic
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatSession, Qwen38TracedChain
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import prefill_form, prepare_vllm_construction
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38VLLMMTPLanesDriver, VerifyRow
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    Qwen38LanePaging,
    paged_table_width,
    validate_index_paging,
    verify_row_page_table_host,
)

DRAFTS = 4
LANES = 4
LIVE = 2
ALLOCATED_CONTEXT = 32768
PAGE_SIZE = 128
NUM_PAGES = 1026
PASSES = int(os.environ.get("QWEN38_PAGED_TEST_PASSES", "12"))
OUT_VARIABLE = "QWEN38_PAGED_TEST_OUT"
WARM_PROMPT = os.environ.get("QWEN38_PAGED_TEST_WARM") == "1"  # the adapter's 40-token warm-up prompt
BLOCKS_PER_LANE = 8
CHECK_TABLES = (
    os.environ.get("QWEN38_PAGED_TEST_CHECK") == "1"
)  # compare every device copy of the page tables to the host mirror


def out_dir() -> Path:
    value = os.environ.get(OUT_VARIABLE)
    if not value:
        pytest.skip(f"set {OUT_VARIABLE} to the directory the test writes its results to")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def lane_blocks(lane: int) -> list[int]:
    """Distinct blocks per lane, counted from 1 (block 0 is the scratch)."""

    return list(range(1 + lane * BLOCKS_PER_LANE, 1 + (lane + 1) * BLOCKS_PER_LANE))


def open_paged_chain(mesh_device, checkpoint, lanes_session):
    """``open_mtp_chain`` with the chain's paging, the way the adapter's chain opener builds it (lanes form)."""

    prepared, allocated_context = prepare_vllm_construction(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, ALLOCATED_CONTEXT - RESIDENT_CONTEXT_HEADROOM
    )
    assert allocated_context == ALLOCATED_CONTEXT, allocated_context
    paging = validate_index_paging(
        Qwen38LanePaging(
            page_size=PAGE_SIZE, num_pages=NUM_PAGES, max_blocks=paged_table_width(allocated_context, PAGE_SIZE)
        ),
        allocated_context,
    )
    long_chunks, slab_rows = prefill_form()
    construction = construct_live_decode_diagnostic(
        prepared, mesh_device=mesh_device, collective_topology=ttnn.Topology.Linear, marker=logger.info
    )

    def warm_hook(chain) -> None:
        _lanes_warm_hook(chain.mesh, lanes_session, allocated_context, long_chunks or slab_rows is not None)(chain)

    logger.info(f"paged lanes test: page table width {paging.max_blocks}, {NUM_PAGES} blocks of {PAGE_SIZE}")
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


def check_tables(session, mesh_device, label: str) -> None:
    """Log whether every device copy of the lane, verify row and draft row page tables equals the host mirror."""

    ttnn.synchronize_device(mesh_device)
    lane_host = mtp_lanes.lane_page_table_host(
        session.verify, [session.lane_pages.get(lane, []) for lane in range(session.lanes)], sorted(session.lane_pages)
    )
    tables = [
        (
            "lane table",
            session.verify.qsa_lane_constants.page_table,
            lane_host.reshape(1, 1, *lane_host.shape),
        ),
        (
            "verify row table",
            session.verify.qsa_lane_verify_constants.row_page_table,
            verify_row_page_table_host(
                lane_host, session.verify.qsa_lane_verify_constants.rows, session.verify.qsa_lane_verify_constants.lanes
            ),
        ),
    ]
    if session.draft is not None:
        constants = session.draft.qsa_lane_verify_constants
        tables.append(
            (
                "draft row table",
                constants.row_page_table,
                verify_row_page_table_host(lane_host, constants.rows, constants.lanes),
            )
        )
    bad = 0
    for name, tensor, want in tables:
        want = want.to(torch.int32)
        count = 0
        for device, local in enumerate(ttnn.get_device_tensors(tensor)):
            got = ttnn.to_torch(local).to(torch.int32).reshape(want.shape)
            for row, col in (got != want).nonzero().reshape(-1, want.dim())[:, -2:].tolist():
                count += 1
                if count <= 10:
                    logger.info(
                        f"table check {label}: {name} device {device} row {row} col {col} "
                        f"got {int(got[..., row, col].flatten()[0])} want {int(want[..., row, col].flatten()[0])}"
                    )
        if count:
            logger.info(f"table check {label}: {name} mismatches {count}")
        bad += count
    if not bad:
        logger.info(f"table check {label}: OK")


def pass_blocks(position: int) -> int:
    return min(position + DRAFTS, ALLOCATED_CONTEXT - 1) // PAGE_SIZE + 1


def lane_row(driver, lane: int, paged: bool) -> VerifyRow:
    run = driver.runs[lane]
    pages = None
    if run.fresh:
        position = run.position
        row = VerifyRow(lane, [run.pending] + [0] * DRAFTS, position, 0, run.settled)
    else:
        position = run.next_position
        row = VerifyRow(lane, list(run.next_tokens), position, DRAFTS, run.device_rows)
    if paged:
        assert (
            pass_blocks(position) <= BLOCKS_PER_LANE
        ), f"lane {lane} needs {pass_blocks(position)} blocks at position {position}, only {BLOCKS_PER_LANE} per lane"
        row.pages = lane_blocks(lane)[: pass_blocks(position)]
    return row


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("case", ["flat", "paged"])
@pytest.mark.timeout(5400)
def test_lanes_stream(mesh_device, case):
    out = out_dir()
    paged = case == "paged"
    checkpoint = checkpoint_directory()
    prompts, _tokenizer, _config = needle_prompts(checkpoint, LIVE, None)
    if WARM_PROMPT:
        from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import WARMUP_PROMPT_TOKEN_IDS

        prompts = [list(WARMUP_PROMPT_TOKEN_IDS)] * LIVE
    mesh_device.enable_program_cache()
    session = Qwen38LanesSession(lanes=LANES, drafts=DRAFTS, marker=logger.info)
    if paged:
        chain = open_paged_chain(mesh_device, checkpoint, session)
    else:
        chain = open_mtp_chain(
            mesh_device, checkpoint, ALLOCATED_CONTEXT - RESIDENT_CONTEXT_HEADROOM, DRAFTS, lanes_session=session
        )
    try:
        if CHECK_TABLES:
            sync_original = session.sync_page_tables

            def sync_checked(*args, **kwargs):
                written = sync_original(*args, **kwargs)
                if written:
                    check_tables(session, mesh_device, "after-upload")
                return written

            session.sync_page_tables = sync_checked
        chat = Qwen38ChatSession(chain, None, allocated_context=ALLOCATED_CONTEXT)
        chat.reset()
        session.capture(chain, chat)
        logger.info("lanes captured")
        driver = Qwen38VLLMMTPLanesDriver(session, LANES, DRAFTS, paged=paged)
        streams = {}
        for lane in range(LIVE):
            if paged:
                ids = lane_blocks(lane)[: -(-len(prompts[lane]) // PAGE_SIZE)]
                streams[lane] = [driver.admit(lane, prompts[lane], block_ids=ids)]
            else:
                streams[lane] = [driver.admit(lane, prompts[lane])]
            logger.info(f"admitted lane {lane}")
        records = []
        for index in range(PASSES):
            logger.info(f"pass {index} start")
            rows = [lane_row(driver, lane, paged) for lane in range(LIVE)]
            if CHECK_TABLES:
                check_tables(session, mesh_device, f"before-pass-{index}")
            by_lane = driver.verify(rows)
            record = session.lane_chain.records[-1]
            records.append(record)
            for lane in range(LIVE):
                streams[lane] += by_lane[lane][: record.accepted[lane] + 1]
            logger.info(f"pass {index} done accepted {[record.accepted[lane] for lane in range(LIVE)]}")
        torch.save(
            {
                "case": case,
                "streams": streams,
                "accepted": [list(r.accepted[:LIVE]) for r in records],
                "argmaxes": [[list(a) for a in r.argmaxes[:LIVE]] for r in records],
            },
            out / f"lanes_{case}{'_warm' if WARM_PROMPT else ''}.pt",
        )
    finally:
        session.close()
        chain.close()


def test_lanes_compare():
    out = out_dir()
    suffix = "_warm" if WARM_PROMPT else ""
    paths = [out / f"lanes_flat{suffix}.pt", out / f"lanes_paged{suffix}.pt"]
    if not all(p.exists() for p in paths):
        pytest.skip("run both test_lanes_stream cases first")
    flat, paged = (torch.load(p, weights_only=False) for p in paths)
    for lane in range(LIVE):
        assert flat["streams"][lane] == paged["streams"][lane], f"lane {lane}: flat vs paged tokens"
    assert flat["accepted"] == paged["accepted"]
    assert flat["argmaxes"] == paged["argmaxes"]
