# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of the MTP lane width buckets (``QWEN38_MTP_LANE_BUCKETS``) of ``Qwen38LanesSession``.

One session per process (one mesh open per case); the cases that need a comparison save a file and a no-device compare
test reads them, as ``test_batch_buckets_device.py`` does.

a. ``test_width_stream``  ids ``own{W}``, ``view{W}``, ``full8`` for W in (2, 4): the same W prompts in lanes 0..W-1,
   greedy, K = 3, ``PASSES`` traced passes through the lanes driver.  ``own{W}``: a W-lane session.  ``view{W}``: an
   8-lane session with buckets 2,4,8 (every pass must be a width-W pass: ``record.width == W``).  ``full8``: an 8-lane
   session with buckets off.  Each saves per pass the positions, accepted counts, argmaxes and alignment argmaxes of the
   live lanes, the committed token streams, and, for own / view, the final-mixer hidden rows ``[1,1,32,640]`` of one
   EAGER verify pass (``hidden_observer``) to ``$QWEN38_WIDTH_TEST_OUT/<id>.pt``.  The logits are not in the traces.
b. ``test_width_compare``  (no device): own{W} vs view{W}: equal tokens, accepts, argmaxes, alignment argmaxes and BIT-EQUAL
   hidden rows; view{W} vs full8: equal tokens.
c. ``test_width_switch_keeps_users``  (8-lane session, buckets 2,4,8): users 0 and 1 for 12 passes (width 2), a third
   admitted into lane 2 (width 4) for 12 passes, lane 2 released (width 2 again) for 12 passes.  Asserts the width of
   every pass, the trace set launched per pass (commit at the PREVIOUS pass's width), the device accept tile against the
   readbacks after every switch, no new program after the captures, the first tokens of every stream against the
   single-stream chain, and, with ``own2.pt`` saved, users 0 and 1 equal to their no-switch 2-lane run.  The admit into
   lane 2 after width-2 passes (the draft of those passes clobbered the lane's token blocks) is the case this test exists for.
d. ``test_width_step_times``  (informational): median ms per pass of the forced widths 2, 4, 8 with two live users.
e. ``test_width_memory``  free DRAM per bank and the per-width growth after the captures: no width dropped, free >= 150 MiB.

Run:  QWEN38_WIDTH_TEST_OUT=<dir> pytest models/demos/blackhole/qwen38_flash_next/tests/test_mtp_lane_widths_device.py -v -s -k "own2"
Env:  as demo/text_demo.py (HF_MODEL, TT_CACHE_PATH, TT_METAL_TRACE_ALLOC_TRACKING=1, MESH_DEVICE).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import open_mtp_chain, run_mtp_single
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_batch_buckets_device import needle_prompts
from models.demos.blackhole.qwen38_flash_next.tools import resident_decode
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38VLLMMTPLanesDriver, VerifyRow
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

DRAFTS = 3
ROWS = DRAFTS + 1
ALLOCATED_CONTEXT = 8192
PASSES = 36  # the stream length of every saved run (the switch test runs 12 + 12 + 12)
EXACT_PREFIX = 8
OUT_VARIABLE = "QWEN38_WIDTH_TEST_OUT"
BUCKET_VARIABLE = mtp_lanes.MTP_LANE_BUCKETS_VARIABLE
FREE_FLOOR = mtp_lanes.FREE_DRAM_FLOOR_BYTES_PER_BANK
NARROW = (2, 4)
STREAM_IDS = [f"own{w}" for w in NARROW] + [f"view{w}" for w in NARROW] + ["full8-2", "full8-4"]


def out_dir() -> Path:
    value = os.environ.get(OUT_VARIABLE)
    if not value:
        pytest.skip(f"set {OUT_VARIABLE} to the directory the test writes its results to")
    path = Path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Opened:
    """One lanes session over a fresh chain: ``session`` (the lanes session), ``driver`` (the vLLM adapter's lanes driver),
    ``chain`` and the single-stream ``chat`` session (the referee, also the prefill engine)."""

    def __init__(self, mesh_device, lanes: int, buckets: str | None, prompts_count: int, prompt_len: int | None = None):
        checkpoint = checkpoint_directory()
        if not os.environ.get("TT_CACHE_PATH"):
            pytest.fail("set TT_CACHE_PATH to the cache root (the book download goes to <root>/demo-context)")
        self.prompts, self.tokenizer, self.config = needle_prompts(checkpoint, prompts_count, prompt_len)
        environ = dict(os.environ)
        environ.pop(BUCKET_VARIABLE, None)
        if buckets is not None:
            environ[BUCKET_VARIABLE] = buckets
        self.lanes = lanes
        self.mesh = mesh_device
        mesh_device.enable_program_cache()
        self.session = Qwen38LanesSession(lanes=lanes, drafts=DRAFTS, marker=logger.info, environ=environ)
        max_seq_len = ALLOCATED_CONTEXT - RESIDENT_CONTEXT_HEADROOM
        self.chain = open_mtp_chain(mesh_device, checkpoint, max_seq_len, DRAFTS, lanes_session=self.session)
        self.chat = Qwen38ChatSession(self.chain, None, prefill_mode="chunked", allocated_context=ALLOCATED_CONTEXT)
        self.references: dict[int, list[int]] = {}

    def reference(self, user: int, tokens: int = 16) -> list[int]:
        """The prompt alone on the single-stream chain (before the lane capture)."""

        self.references[user] = run_mtp_single(self.chat, self.prompts[user], tokens)["tokens"]
        return self.references[user]

    def capture(self) -> None:
        self.chat.reset()
        self.session.capture(self.chain, self.chat)
        self.driver = Qwen38VLLMMTPLanesDriver(self.session, self.lanes, DRAFTS)
        self.programs = resident_decode.program_cache_count(self.mesh)

    def close(self) -> None:
        self.session.close()
        self.chain.close()


def admit(opened: Opened, lane: int, user: int, streams: dict[int, list[int]]) -> None:
    streams[user] = [opened.driver.admit(lane, opened.prompts[user])]


def lane_row(driver: Qwen38VLLMMTPLanesDriver, lane: int) -> VerifyRow:
    """The plugin's next row of a lane that takes everything the device took (the lane driver's no-resync case)."""

    run = driver.runs[lane]
    if run.fresh:
        return VerifyRow(lane, [run.pending] + [0] * DRAFTS, run.position, 0, run.settled)
    return VerifyRow(lane, list(run.next_tokens), run.next_position, DRAFTS, run.device_rows)


def run_passes(opened: Opened, lanes: list[int], passes: int, streams: dict[int, list[int]], users: dict[int, int]):
    """``passes`` lane passes over ``lanes``; appends each lane's committed ids to its user's stream; returns the records."""

    records = []
    session = opened.session
    for _ in range(passes):
        rows = [lane_row(opened.driver, lane) for lane in lanes]
        by_lane = opened.driver.verify(rows)
        record = session.lane_chain.records[-1]
        records.append(record)
        for lane in lanes:
            streams[users[lane]] += by_lane[lane][: record.accepted[lane] + 1]
    return records


def hidden_rows_of_one_eager_pass(opened: Opened, verify_state, width: int) -> torch.Tensor:
    """One eager verify pass on ``verify_state`` (the parent for own / full8, a width view for view{W}) with the
    final-mixer hidden rows read: ``[1,1,32,640]`` per device concatenated on the hidden axis, bf16."""

    model, parent, session = opened.session.model, opened.session.verify, opened.session
    seen = []

    def observe(hidden):
        seen.append(torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(hidden)], dim=3).to(torch.bfloat16))

    chain = session.lane_chain
    mtp_lanes.forward_commit_lanes(model, verify_state)
    mtp_lanes.write_lane_ple_rows(model, parent, chain.next_tokens)
    output = mtp_lanes.forward_verify_lanes(
        model, verify_state, catch_up=False, land_accept=True, hidden_observer=observe
    )
    ttnn.synchronize_device(opened.mesh)
    output.release_tensors()
    return seen[0][..., : width * ROWS, :].clone()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.parametrize("case", STREAM_IDS)
@pytest.mark.timeout(5400)
def test_width_stream(mesh_device, case):
    out = out_dir()
    kind, width = (
        (case.split("-")[0], int(case.split("-")[1])) if "-" in case else (case.rstrip("0123456789"), int(case[-1]))
    )
    if kind == "own":
        opened = Opened(mesh_device, width, None, width)
    elif kind == "view":
        opened = Opened(mesh_device, 8, "2,4,8", width)
    else:  # full8: eight lanes, no buckets, the same `width` prompts in lanes 0..width-1
        opened = Opened(mesh_device, 8, "off", width)
    try:
        opened.capture()
        streams: dict[int, list[int]] = {}
        users = {lane: lane for lane in range(width)}
        for lane in range(width):
            admit(opened, lane, lane, streams)
        records = run_passes(opened, list(range(width)), PASSES, streams, users)
        if kind == "view":
            assert [r.width for r in records] == [width] * PASSES, [r.width for r in records]
        if kind == "full8":
            assert {r.width for r in records} == {8}
        saved = {
            "case": case,
            "width": width,
            "streams": streams,
            "positions": [list(r.positions[:width]) for r in records],
            "accepted": [list(r.accepted[:width]) for r in records],
            "argmaxes": [[list(a) for a in r.argmaxes[:width]] for r in records],
            "alignment": [[list(a) for a in r.alignment_argmaxes[:width]] for r in records],
            "widths": [r.width for r in records],
        }
        if kind != "full8":
            session = opened.session
            state = session.verify if kind == "own" else session.lane_chain.width_sets[width].verify
            saved["hidden"] = hidden_rows_of_one_eager_pass(opened, state, width)
        torch.save(saved, out / f"{case}.pt")
        assert (
            resident_decode.program_cache_count(mesh_device) == opened.programs
        ), "a program was compiled while serving"
    finally:
        opened.close()


@pytest.mark.parametrize("width", NARROW)
def test_width_compare(width):
    out = out_dir()
    paths = {name: out / f"{name}{width}.pt" for name in ("own", "view")} | {"full8": out / f"full8-{width}.pt"}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"{missing} are not in {out}: run the test_width_stream cases first")
    own, view, full8 = (torch.load(paths[name], weights_only=False) for name in ("own", "view", "full8"))
    for user in range(width):
        assert own["streams"][user] == view["streams"][user], f"user {user}: own{width} vs view{width} tokens"
        assert view["streams"][user] == full8["streams"][user], f"user {user}: view{width} vs full8 tokens"
    for key in ("positions", "accepted", "argmaxes", "alignment"):
        assert own[key] == view[key], f"{key}: own{width} vs view{width}"
    assert torch.equal(own["hidden"], view["hidden"]), (
        f"final mixer hidden rows differ: max abs "
        f"{(own['hidden'].float() - view['hidden'].float()).abs().max():.6g}"
    )


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.timeout(5400)
def test_width_switch_keeps_users(mesh_device):
    opened = Opened(mesh_device, 8, "2,4,8", 3)
    try:
        for user in range(3):  # the single-stream chain's own tokens, before the lane capture
            opened.reference(user)
        opened.capture()
        session, driver = opened.session, opened.driver
        chain = session.lane_chain
        streams: dict[int, list[int]] = {}
        users = {0: 0, 1: 1, 2: 2}
        names = {}
        for width, width_set in chain.width_sets.items():
            for kind, tid in zip(
                ("verify", "commit", "draft"),
                (width_set.traces.verify_first, width_set.traces.commit, width_set.traces.draft),
            ):
                names[tid] = f"{kind}{width}"
        launched: list[str] = []
        for attr in ("replay", "enqueue"):
            base = getattr(chain, attr)
            if base is not None:
                setattr(
                    chain, attr, lambda tid, base=base: (launched.append(names.get(tid, f"chain:{tid}")), base(tid))[1]
                )

        def check_accept_tile(record, live):
            tile = ttnn.to_torch(ttnn.get_device_tensors(session.verify.accepted_lanes)[0]).reshape(-1)
            for lane in live:
                assert int(tile[lane]) == record.accepted[lane], (record.index, lane, tile.tolist(), record.accepted)

        widths: list[int] = []
        admit(opened, 0, 0, streams)
        admit(opened, 1, 1, streams)
        # phase 1: two users, width 2
        for _ in range(12):
            launched.clear()
            (record,) = run_passes(opened, [0, 1], 1, streams, users)
            widths.append(record.width)
            assert launched[-2:] == ["verify2", "draft2"] and launched[0] == (
                "commit8" if not widths[:-1] else "commit2"
            )
            check_accept_tile(record, [0, 1])
        # phase 2: the third request admitted into lane 2 AFTER width-2 passes: width 4
        admit(opened, 2, 2, streams)
        previous = widths[-1]
        for index in range(12):
            launched.clear()
            (record,) = run_passes(opened, [0, 1, 2], 1, streams, users)
            widths.append(record.width)
            assert launched[0] == f"commit{previous}" and launched[-2:] == ["verify4", "draft4"], launched
            previous = record.width
            check_accept_tile(record, [0, 1, 2])
        # phase 3: lane 2 released: width 2 again; the first pass still commits lane 2's last real pass at width 4
        driver.release(2)
        for index in range(12):
            launched.clear()
            (record,) = run_passes(opened, [0, 1], 1, streams, users)
            widths.append(record.width)
            assert launched[0] == ("commit4" if index == 0 else "commit2") and launched[-2:] == ["verify2", "draft2"]
            check_accept_tile(record, [0, 1])
        assert widths == [2] * 12 + [4] * 12 + [2] * 12, widths
        assert resident_decode.program_cache_count(mesh_device) == opened.programs
        for user in range(3):
            assert (
                streams[user][:EXACT_PREFIX] == opened.references[user][:EXACT_PREFIX]
            ), f"user {user} vs single stream"
        pass_keys = [opened.config["pass_key"]]
        decoded = opened.tokenizer.decode(streams[2], skip_special_tokens=True)
        logger.info(f"switch: widths {widths[:1]}..{widths[-1:]} third user {decoded[:120]!r} {pass_keys}")
        saved = out_dir() / "own2.pt"
        if saved.exists():
            own2 = torch.load(saved, weights_only=False)
            for user in (0, 1):  # the same positions: tokens of passes 0..35 of the no-switch 2-lane run
                assert streams[user] == own2["streams"][user], f"user {user}: switch run vs own2 (no switch)"
    finally:
        opened.close()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.timeout(5400)
def test_width_step_times(mesh_device):
    out = out_dir()
    opened = Opened(mesh_device, 8, "2,4,8", 2, 128)
    warm, timed = 8, 24
    try:
        opened.capture()
        streams: dict[int, list[int]] = {}
        users = {0: 0, 1: 1}
        admit(opened, 0, 0, streams)
        admit(opened, 1, 1, streams)
        run_passes(opened, [0, 1], 2, streams, users)
        opened.session.set_active([1, 1, 0, 0, 0, 0, 0, 0])
        chain = opened.session.lane_chain
        results = {}
        for width in (2, 4, 8):
            ms = []
            for index in range(warm + timed):
                started = time.perf_counter()
                chain.step(force_width=width)
                ms.append(1000.0 * (time.perf_counter() - started))
            results[width] = statistics.median(ms[warm:])
            logger.info(f"width {width}: median {results[width]:.2f} ms per pass (2 live users)")
        (out / "width_step_times.json").write_text(
            json.dumps({"median_ms": results}, indent=1) + "\n", encoding="utf-8"
        )
        assert results[2] <= results[8] * 1.05, results  # the narrow pass must not be slower than the full one
    finally:
        opened.close()


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "num_command_queues": 1}], indirect=True)
@pytest.mark.timeout(5400)
def test_width_memory(mesh_device):
    opened = Opened(mesh_device, 8, "2,4,8", 2, 128)
    try:
        opened.capture()
        summary = opened.session.summary()
        free = opened.session.free_dram_bytes_per_bank()
        logger.info(
            f"widths {summary['buckets']} dropped {summary['dropped_widths']} growth by width "
            f"{summary['dram_growth_bytes_per_bank_by_width']} free {free / 2**20:.1f} MiB/bank"
        )
        assert summary["dropped_widths"] == [] and summary["buckets"] == [2, 4, 8]
        assert free >= FREE_FLOOR, f"{free / 2**20:.1f} MiB/bank free after the width captures"
        assert set(summary["dram_growth_bytes_per_bank_by_width"]) == {"traces_w8", "traces_w4", "traces_w2"}
        # every width's first replay happens here, not while serving: two warm users, then the forced-width replays
        streams: dict[int, list[int]] = {}
        admit(opened, 0, 0, streams)
        admit(opened, 1, 1, streams)
        run_passes(opened, [0, 1], 1, streams, {0: 0, 1: 1})
        assert opened.session.warm_widths() == [8, 4, 2]
        assert resident_decode.program_cache_count(mesh_device) == opened.programs
    finally:
        opened.close()
