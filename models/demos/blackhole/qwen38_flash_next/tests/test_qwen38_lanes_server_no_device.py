# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The lanes server (``--lanes B``) without a device: the flag's start-up admission, the lanes DRAM admission, the
lanes session's host-side override and source pins."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lane_scheduler as scheduler_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_lanes_session as lanes_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes, mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan

SERVER_SOURCE = inspect.getsource(server)
SERVER_MAIN = SERVER_SOURCE[SERVER_SOURCE.index("def main() ->") :]


def _args(**overrides):
    base = dict(
        lanes=0,
        mtp=None,
        sampling=False,
        sampling_discriminator=False,
        agreement_reference=None,
        prefill_mode="chunked",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- the flag's admission at start


def test_lanes_switch_is_off_by_default_admits_the_one_tile_geometry_and_leaves_the_fused_set_alone() -> None:
    assert server.lanes_switches(_args()) is None
    assert server.lanes_switches(_args(lanes=4, mtp=4)) == {"lanes": 4, "rows": 5, "drafts": 4}  # no fused_off
    assert server.lanes_switches(_args(lanes=8, mtp=3))["rows"] == 4
    assert server.lanes_switches(_args(lanes=6, mtp=4))["lanes"] == 6
    # the lanes run the single stream's fused set: the switch sets no environment
    assert "os.environ[fused_module.OFF_ENV]" not in SERVER_SOURCE and "LANES_FUSED_OFF" not in SERVER_SOURCE
    assert "mtp_ple_early = False  # the lanes run the one-queue form" in SERVER_SOURCE


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"lanes": 4}, "needs --mtp"),
        ({"lanes": 4, "mtp": 4, "sampling": True}, "greedy requests only"),
        ({"lanes": 4, "mtp": 4, "sampling_discriminator": True}, "sampling-discriminator"),
        ({"lanes": 4, "mtp": 4, "agreement_reference": Path("x")}, "agreement-reference"),
        # B in 2..8, B x (k + 1) <= 32
        *(({"lanes": lanes, "mtp": drafts}, None) for lanes, drafts in ((1, 4), (9, 3), (7, 4), (8, 4))),
    ],
)
def test_lanes_switch_refuses_sampling_and_a_bad_geometry(expect_error, overrides, match) -> None:
    with expect_error(SystemExit, match=match):  # allow-pytest.raises: the flag's refusal names the rule
        server.lanes_switches(_args(**overrides))


def test_parser_takes_lanes_and_dropped_the_old_lane_service_flags(expect_error) -> None:
    parser = server._parser()
    options = {action.dest for action in parser._actions}
    assert "lanes" in options
    for old in ("lane_slots", "lane_verify_policy", "lane_prefill_chunk_budget", "lane_prefill_import_check"):
        assert old not in options, old
    argv = "--checkpoint c --component-cache-root c --routed-bf4-scratch-root c --model-io-cache-root c".split()
    argv += ["--phase-log", "p", "--lanes", "4", "--mtp", "4"]
    parsed = parser.parse_args(argv)
    assert parsed.lanes == 4 and parsed.mtp == 4 and parsed.sampling is False
    # the stall budget: the scheduler's default, a number of seconds, or off (whole admissions)
    assert parsed.lanes_stall_budget == scheduler_module.DEFAULT_STALL_BUDGET_SECONDS
    assert parser.parse_args(argv + ["--lanes-stall-budget", "0"]).lanes_stall_budget == 0.0
    assert parser.parse_args(argv + ["--lanes-stall-budget", "off"]).lanes_stall_budget is None
    assert parser.parse_args(argv + ["--lanes-stall-budget", "0.5"]).lanes_stall_budget == 0.5
    for bad in ("-1", "nan", "soon"):
        with expect_error(SystemExit):  # allow-pytest.raises: argparse refuses the value
            parser.parse_args(argv + ["--lanes-stall-budget", bad])
    # --long-chunks is accepted (the Hub manifests pass it) and is the chunked prefill's default
    assert parser.parse_args(argv + ["--long-chunks"]).long_chunks is True
    assert 'args.long_chunks = args.prefill_mode == "chunked"' in SERVER_MAIN
    assert '"--lanes-stall-budget needs --lanes' in SERVER_MAIN
    assert "stall_budget_seconds=args.lanes_stall_budget," in SERVER_MAIN


# --------------------------------------------------------------------------- the DRAM admission


LANES_ADMISSION_PINS = {
    # (lanes, drafts, context): required bytes per bank (the modeled terms with the slope correction and the margin)
    (4, 4, 32768): 374_109_150,
    (4, 4, 65536): 669_065_191,
    (4, 4, 131072): 1_258_977_272,
    (6, 4, 65536): 960_539_773,
    (8, 3, 32768): 681_696_777,
    (2, 4, 32768): 219_010_790,
}


@pytest.mark.parametrize("point", sorted(LANES_ADMISSION_PINS))
def test_lanes_admission_terms_are_pinned_per_point(point) -> None:
    lanes, drafts, context = point
    record = session_module.lanes_capacity_admission(context, lanes=lanes, drafts=drafts)
    assert record["required_free_bytes_per_bank"] == LANES_ADMISSION_PINS[point], record
    assert record["required_largest_contiguous_bytes_per_bank"] == 128 << 20
    assert record["one_tile"] and record["total_rows"] == lanes * (drafts + 1)
    assert record["free_bytes_source"] == "table_2026-09-04"
    assert set(record["lanes_growth_estimate_bytes_per_bank"]) == {"states", "traces"}
    assert set(record["lanes_growth_remainders_bytes_per_bank"]) == {
        "lane_states",
        "pagers",
        "moe_rows_instances",
        "fold_prefix_states",
        "traces",
    }
    assert record["lanes_growth_remainders_bytes_per_bank"]["fold_prefix_states"] == 0 and not record["gdn_rows_scan"]


def test_lanes_admission_estimate_covers_the_measured_state_and_decides_on_the_live_reading(expect_error) -> None:
    record = session_module.lanes_capacity_admission(32768, lanes=4, drafts=4)
    # the states term covers the measured 4 x 4 verify state with room for the pagers and the draft state, and stays
    # within a third of it (the model is not loose)
    measured = session_module.LANES_MEASURED_VERIFY_STATE_BYTES_PER_BANK_4X4_32K
    assert measured < record["lanes_growth_estimate_bytes_per_bank"]["states"] < measured * 1.33
    # the table's free side admits 4 x 4 at 32,768 and refuses it at 131,072; a live reading (from a device
    # measurement) admits 4 x 4 at 32,768 with room to spare
    assert record["fits"] and record["decided_by"]["shortfalls"] == []
    at_128k = session_module.lanes_capacity_admission(131072, lanes=4, drafts=4)
    assert not at_128k["fits"] and at_128k["decided_by"]["shortfalls"] == ["free_bytes_below_estimate"]
    live = {
        "num_banks": 8,
        "free_bytes_per_bank": 1_449_576_320 + 297_300_000,
        "largest_contiguous_bytes_free_per_bank": 1_434_522_432,
    }
    admitted = session_module.lanes_capacity_admission(
        32768, lanes=4, drafts=4, live=live, reserved_bytes_per_bank=80_000_000
    )
    assert admitted["fits"] and admitted["free_bytes_source"] == "measured_live"
    assert admitted["free_bytes_per_bank"] == live["free_bytes_per_bank"] - 80_000_000
    assert admitted["headroom_bytes_per_bank"] == admitted["free_bytes_per_bank"] - LANES_ADMISSION_PINS[(4, 4, 32768)]
    tight = dict(
        live,
        free_bytes_per_bank=LANES_ADMISSION_PINS[(4, 4, 32768)] + 1000,
        largest_contiguous_bytes_free_per_bank=100 << 20,
    )
    refused = session_module.lanes_capacity_admission(32768, lanes=4, drafts=4, live=tight)
    assert refused["decided_by"]["shortfalls"] == ["largest_contiguous_below_lane_kv"]
    for lanes, drafts in ((1, 4), (7, 4), (9, 3), (4, 6)):
        with expect_error(ValueError):  # allow-pytest.raises: the geometry
            session_module.lanes_capacity_admission(32768, lanes=lanes, drafts=drafts)
    with expect_error(ValueError, match="inconsistent"):  # allow-pytest.raises
        session_module.lanes_capacity_admission(
            32768, lanes=4, drafts=4, live={"free_bytes_per_bank": 1, "largest_contiguous_bytes_free_per_bank": 2}
        )


def test_lanes_admission_terms_cover_the_measured_128k_readings() -> None:
    """The modeled terms cover the measured 4 x 4 readings at 131,072: states within +5 %, pagers within +1 %."""

    record = session_module.lanes_capacity_admission(131072, lanes=4, drafts=4)
    remainders = record["lanes_growth_remainders_bytes_per_bank"]
    verify_state = session_module.LANES_MEASURED_VERIFY_STATE_BYTES_PER_BANK_4X4_128K
    assert verify_state < remainders["lane_states"] < verify_state * 1.05
    pagers = session_module.LANES_MEASURED_PAGERS_BYTES_PER_BANK_128K
    assert pagers * 0.99 < remainders["pagers"] < pagers * 1.01
    assert remainders["traces"] > session_module.LANES_MEASURED_TRACES_BYTES_PER_BANK
    traces = session_module.LANES_TRACES_BYTES_PER_BANK
    assert remainders["traces"] == traces == (51_000_000 + 47_000_000) // 8 + 7_400_000 == 19_650_000, traces
    # the per-lane slope from 32k to 128k, measured against the model's
    slope_measured = (
        (verify_state - session_module.LANES_MEASURED_VERIFY_STATE_BYTES_PER_BANK_4X4_32K) / 4 / (131072 - 32768)
    )
    per_lane = lambda context: (  # noqa: E731
        session_module.lanes_image_bytes_per_device(context)
        + session_module.lanes_mtp_extra_bytes_per_device(context, 5)
    )
    slope_modeled = (per_lane(131072) - per_lane(32768)) / (131072 - 32768) / 8
    assert abs(slope_measured - slope_modeled) / slope_measured < 0.01, (slope_measured, slope_modeled)
    # the 128k point admits on the measured live reading with the served extras reserved
    live = {
        "num_banks": 8,
        "free_bytes_per_bank": session_module.LANES_MEASURED_FREE_AFTER_CAPTURES_BYTES_PER_BANK_4X4_128K + 917_362_880,
        "largest_contiguous_bytes_free_per_bank": 373_234_752 + 917_362_880,
    }
    admitted = session_module.lanes_capacity_admission(
        131072, lanes=4, drafts=4, live=live, reserved_bytes_per_bank=session_module.LANES_SERVED_EXTRAS_BYTES_PER_BANK
    )
    assert admitted["fits"] and 0 < admitted["headroom_bytes_per_bank"] < 200_000_000
    # 6 x 4 at 128k does not fit the same reading (as measured)
    six = session_module.lanes_capacity_admission(
        131072, lanes=6, drafts=4, live=live, reserved_bytes_per_bank=session_module.LANES_SERVED_EXTRAS_BYTES_PER_BANK
    )
    assert not six["fits"] and six["decided_by"]["shortfalls"] == ["free_bytes_below_estimate"]


def test_lanes_admission_adds_the_fold_prefix_states_when_the_fold_serves_the_lanes() -> None:
    """The fold's lanes form keeps B x R prefix states per GDN layer (the B = 1 rule at the lanes' row count)."""

    assert session_module.lanes_fold_prefix_states_bytes_per_bank(20) == 70_778_880
    for rows in (10, 20, 30, 32):  # the fold module's own rule per device over the banks
        assert (
            session_module.lanes_fold_prefix_states_bytes_per_bank(rows)
            == gdn_rows_scan.prefix_states_bytes(rows, 36) // 8
        )
    assert session_module.lanes_fold_prefix_states_bytes_per_bank(
        5
    ) == session_module.fold_prefix_states_bytes_per_bank(4)
    off = session_module.lanes_capacity_admission(32768, lanes=4, drafts=4)
    on = session_module.lanes_capacity_admission(32768, lanes=4, drafts=4, gdn_rows_scan=True)
    assert on["gdn_rows_scan"] and on["lanes_growth_remainders_bytes_per_bank"]["fold_prefix_states"] == 70_778_880
    assert on["required_free_bytes_per_bank"] - off["required_free_bytes_per_bank"] == -(-70_778_880 * 110 // 100)
    assert on["required_free_bytes_per_bank"] == 451_965_918
    assert "gdn_rows_scan" in on["decided_by"]["required_side"]


def test_lanes_admission_grows_with_lanes_and_context() -> None:
    at = lambda lanes, context: (  # noqa: E731
        session_module.lanes_capacity_admission(context, lanes=lanes, drafts=4)["required_free_bytes_per_bank"]
    )
    assert at(2, 32768) < at(4, 32768) < at(6, 32768)
    assert at(4, 32768) < at(4, 65536) < at(4, 131072)
    per_lane_32k = session_module.lanes_image_bytes_per_device(32768) + session_module.lanes_mtp_extra_bytes_per_device(
        32768, 5
    )
    assert per_lane_32k == 516_045_568  # 457,503,744 + 58,541,824 per device (the capacity reader's figures)
    assert session_module.lanes_pager_bytes_per_device(32768) == 191_214_592


# --------------------------------------------------------------------------- the session's host override


class _FakePositions:
    def __init__(self, positions):
        self.positions = list(positions)
        self.written = []

    def write(self, positions):
        self.positions = list(positions)
        self.written.append(list(positions))


class _FakePLE:
    def __init__(self, contexts):
        self.token_contexts = tuple(contexts)
        self.validated = 0

    def validate(self):
        self.validated += 1


def test_override_commit_rewinds_the_position_and_takes_the_kept_rows_context(expect_error) -> None:
    lanes = 4
    positions = _FakePositions([100, 200, 300, 400])
    ple = _FakePLE([(1, 1), (2, 2), (3, 3), (4, 4)])
    verify = SimpleNamespace(
        rows=5,
        positions=positions,
        layers={lanes_module.PLE_CHECKPOINT_LAYER: SimpleNamespace(ple=ple)},
        pass_contexts=[tuple((lane, c) for c in range(6)) for lane in range(lanes)],
    )
    session = lanes_module.Qwen38LanesSession.__new__(lanes_module.Qwen38LanesSession)
    session.verify = verify
    # the pass started at 200 on lane 1 and the device advanced it by 4 (a = 3); the host keeps 2 rows
    record = SimpleNamespace(positions=(100, 200, 300, 400), accepted=(0, 3, 1, 4))
    positions.positions = [101, 204, 302, 405]
    session.override_commit(1, record, 2)
    assert positions.positions == [101, 202, 302, 405] and positions.written == [[101, 202, 302, 405]]
    assert ple.token_contexts == ((1, 1), (1, 2), (3, 3), (4, 4)) and ple.validated == 1
    for rows in (0, 6):
        with expect_error(ValueError):  # allow-pytest.raises: 1 .. R rows
            session.override_commit(1, record, rows)


def test_set_blocks_keeps_the_device_assembled_blocks_and_set_active_writes_on_change_only() -> None:
    writes = []
    chain = SimpleNamespace(
        next_tokens=((1, 2, 3, 4, 5), (6, 7, 8, 9, 10)),
        active=[1, 0],
        set_next_tokens=lambda blocks: writes.append(("tokens", [list(b) for b in blocks])),
        set_active=lambda mask: writes.append(("active", list(mask))),
    )
    session = lanes_module.Qwen38LanesSession.__new__(lanes_module.Qwen38LanesSession)
    session.lane_chain = chain
    session.set_blocks([None, [11, 12, 13, 14, 15]])
    assert writes == [("tokens", [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])]
    session.set_active([1, 0])
    assert len(writes) == 1  # unchanged: nothing written
    session.set_active([1, 1])
    assert writes[-1] == ("active", [1, 1])


# --------------------------------------------------------------------------- source pins


def test_lane_body_gr_read_is_the_single_streams_read_rows_call() -> None:
    """Both verify bodies read the residual through ``read_rows(residual, flat_views=True)``, the module's one fused
    read."""

    lanes, single = inspect.getsource(mtp_lanes), inspect.getsource(mtp_v2)
    for call in (
        "layer.attention_gr.read_rows(residual, flat_views=True)",
        "layer.mlp_gr.read_rows(residual, flat_views=True)",
    ):
        assert call in lanes and call in single, call
    gr = inspect.getsource(gr_module)
    read_rows = gr[gr.index("    def read_rows(") :]
    assert 'if flat_views and getattr(self, "_read_fused", None) is not None:' in read_rows
    assert "return self._read_fused(self, residual_rows)" in read_rows
    init = gr[gr.index("        self._read_fused = None") : gr.index("        self._write_fused = None")]
    assert 'if fused_kernels.enabled("gr_recip_last"):' in init and 'name = "gr_recip_last"' in init
    # the lane GDN rows dispatch through the fold's registry step as the single stream's forward_rows does
    gdn = inspect.getsource(gdn_module)
    lanes_body = gdn[gdn.index("    def forward_rows_lanes(") : gdn.index("    def commit_rows_lanes(")]
    assert (
        "scan_buffers" in lanes_body or "rows_body_scan_lanes" in lanes_body or "gdn_rows_scan" in lanes_body
    ), lanes_body[-600:]


def test_lanes_request_path_calls_no_device_method_of_the_session() -> None:
    """``_serve_lanes`` uses the session for its template only (the host-side decoder); every device call is the
    driver thread's (the scheduler's run over the lanes session)."""

    tree = ast.parse(SERVER_SOURCE)
    handler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38ChatHandler")
    serve_lanes = next(
        node for node in handler.body if isinstance(node, ast.FunctionDef) and node.name == "_serve_lanes"
    )
    session_attributes = {
        node.attr
        for node in ast.walk(serve_lanes)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "session"
    }
    assert session_attributes == {
        "template",
        "chain",
    }, session_attributes  # chain: the program-cache count for the ledger
    # an image request rides on the ticket (the decoded images and the prompt's rotary positions): the driver thread
    # runs the tower inside the admission; the reply carries the admission's vision record
    serve_lanes_source = ast.unparse(serve_lanes)
    assert "images=list(request.get('images') or ())" in serve_lanes_source
    assert "vision_positions=request.get('vision_positions')" in serve_lanes_source
    assert "{'vision': ticket.vision}" in serve_lanes_source
    main = SERVER_MAIN
    for line in (
        "lanes_session.prepare(opened_chain)",
        "warm_hook=warm_hook,",
        "lanes_session.capture(chain, session)",
        "lanes_session.tower = server.vision_prompt_for",  # the resident tower's request path, for the driver thread
        "if sum(growth.values()) > estimate:",  # the hard gate: measured growth against the estimate
        'target=_run_lanes_driver, args=(scheduler, lanes_session, server), name="lane-driver"',
        'report["acceptance_lanes"] = replay_acceptance_lanes(',
        "lanes_session.close()",
    ):
        assert line in main, line
    assert main.index("lanes_session.close()") < main.index("chain.close()")
    assert main.index("lanes_session.capture(chain, session)") > main.index('report["acceptance"] = replay_acceptance(')
    # READY names the lanes and the mode
    ready = main[main.index("ready = {") : main.index("ready_marker.write_text")]
    for key in ('"mode": summary["mode"]', '"lanes": (', '"greedy_only": True', '"acceptance": ('):
        assert key in ready, key
    assert '"mode": "chat_server_single_trace_chain" if lanes_switch is None else "chat_server_mtp_lanes"' in main
    assert '+ ("" if lanes_switch is None else f"-lanes{lanes_switch[\'lanes\']}")' in main


def test_replay_acceptance_lanes_compares_every_stream_with_the_single_stream_replay() -> None:
    source = inspect.getsource(server.replay_acceptance_lanes)
    assert "scheduler.run(lanes_session, forever=False)" in source
    assert "stall_budget_seconds=stall_budget_seconds" in source  # the served budget drives the pin's replay too
    assert '"equals_single_stream": single is not None and actual == list(single)' in source
    assert "if require_gate and not (gate_pass and equals_single_stream):" in source
    assert '"schema": "qwen38-chat-server-acceptance-lanes/v1"' in source


def test_scheduler_and_session_modules_split_the_device_from_the_rules() -> None:
    scheduler_source = inspect.getsource(scheduler_module)
    assert "import ttnn" not in scheduler_source and "mtp_lanes" not in scheduler_source
    session_source = inspect.getsource(lanes_module)
    for call in (
        "mtp_lanes.allocate_lane_verify_state(",
        "mtp_lanes.allocate_lane_draft_state(model, self.verify)",
        "mtp_lanes.import_lane_state(",
        "mtp_lanes.capture_verify_lanes(",
        "mtp_lanes.capture_commit_lanes(model, self.verify, guard=guard)",
        "mtp_lanes.capture_draft_lanes(model, self.verify, self.draft, output, guard=guard)",
        "TraceAllocationTracker.verify_before_replay(mesh, trace_id)",
        "mtp_lanes.write_lane_accepted(self.model, self.verify, list(counts))",
        "self.lane_chain.set_active([0] * self.lanes)",
        # the admission's segments and the scheduler's yield point between them
        "between_chunks=None if between is None else between_chunks,",
        "event_rows=None if between is None else ADMISSION_EVENT_ROWS,",
        'between("prefilled", len(ids), len(ids))',
        'between("evicted", len(ids), len(ids))',
        '"prefill": (prefilled - started) / 1e9 - spent,',
        # an image prompt: the tower segment first, the vision inputs through the chunk driver, the shift at import
        "vision, vision_record = self.tower(ids, ticket.vision_positions, ticket.images)",
        'between("tower", len(ids), len(ids))',
        "rope_shift = 0 if vision is None else min(vision.positions.shift_at(position), position & ~3)",
        "rope_shift=rope_shift,",
        'seconds = {"tower": tower_seconds, **seconds}',
        "vision=vision_record,",
    ):
        assert call in session_source, call
    assert lanes_module.ADMISSION_EVENT_ROWS == 128
    assert lanes_module.ADMISSION_SEGMENTS == ("tower", "chunks", "prefilled", "evicted")
    # the chunk driver takes the vision inputs as the single stream passes them (the positional ``vision``)
    assert "                vision,\n                between_chunks=" in session_source
    # the scheduler's between: a pass once the admission work since the last pass reached the budget
    assert (
        "if budget is None or self.active_count == 0 or now - self.lanes_progressed_at < budget:" in scheduler_source
        and "self._pass(device, boundary)" in scheduler_source
    )
    # the import warm-up covers every residue variant for every lane before the captures
    assert "for shift in range(IMPORT_RESIDUES):" in session_source and lanes_module.IMPORT_RESIDUES == 4
    assert lanes_module.EAGER_PASSES == 2
    # the geometry helpers agree (the scheduler's: its own test)
    assert mtp_lanes._validate_lane_geometry(4, 4) == (4, 5)
    assert mtp_v2.SUPPORTED_DRAFTS == (3, 4, 5)
