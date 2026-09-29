# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vision tower's residency rules without a device: the two ladders and the pad-up rule, the refusal above the
stock maximum, the padding's work bound, the DRAM admission against the measured line readings (the tower fits beside
the 32k chain; a synthetic lanes/128k reading decides on the numbers), the prewarm plan and its READY cost model
against the measured buckets, the ladder comparison's rule, the residency object's warm hook on a fake mesh (admit
before load; a shortfall leaves the process text-only with the reason every image refusal carries; a fit loads then
prewarms every bucket ascending; release), the builder's ``enable_vision`` wiring, and the source pins: the tower
always builds its attention windows (one program set per row count) and the served entry passes the bucket's rows."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import vision_residency as residency
from models.demos.blackhole.qwen38_flash_next.ttnn.vision_layout import padded_rows

ROOT = Path(__file__).resolve().parents[1]
VISION_SOURCE = ROOT / "ttnn" / "vision.py"
BUILDER_SOURCE = ROOT / "ttnn" / "builder.py"

# The line, beside the 32k chunked chain, 2026-09-29 (stage-1 note): the warm hook's reading before the tower.
LINE_32K_BEFORE_TOWER = {
    "free_bytes_per_bank": 1_733_694_464,
    "largest_contiguous_bytes_free_per_bank": 1_733_608_960,
    "num_banks": 8,
}
LINE_32K_MEASURED_TOWER_BYTES_PER_BANK = 125_231_104
LINE_32K_MEASURED_PEAK_65536_PER_BANK = 141_557_760
# The lanes B=4 / 128k server's headroom of record (WAVE-F): 305 MB per bank after its captures.
LANES_128K_AFTER_CAPTURES = {
    "free_bytes_per_bank": 305_000_000,
    "largest_contiguous_bytes_free_per_bank": 290_000_000,
    "num_banks": 8,
}


def test_both_ladders_are_powers_of_two_to_the_stock_maximum_and_the_eight_is_the_ladder_of_record() -> None:
    eight, four = residency.EIGHT_BUCKET_LADDER, residency.FOUR_BUCKET_LADDER
    assert eight == (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536) and four == (1024, 4096, 16384, 65536)
    for ladder in (eight, four):
        assert all(b & (b - 1) == 0 and b % 32 == 0 for b in ladder)
        assert ladder[-1] == residency.STOCK_MAXIMUM_PATCHES == 4096 * 4096 // 16 // 16
    assert set(four) < set(eight)
    assert (
        residency.VISION_ROW_BUCKETS == eight
    )  # the measured READY costs picked eight (43.7 vs 33.5 s, under 30 s apart)


@pytest.mark.parametrize(
    "patches, eight_rows, four_rows",
    [
        (1, 512, 1024),
        (256, 512, 1024),
        (396, 512, 1024),
        (512, 512, 1024),
        (513, 1024, 1024),
        (960, 1024, 1024),
        (1024, 1024, 1024),
        (1025, 2048, 4096),
        (4096, 4096, 4096),
        (4097, 8192, 16384),
        (16384, 16384, 16384),
        (16385, 32768, 65536),
        (65535, 65536, 65536),
        (65536, 65536, 65536),
    ],
)
def test_pad_up_rule_takes_the_smallest_bucket_that_holds_the_image(patches, eight_rows, four_rows) -> None:
    assert residency.bucket_rows(patches, residency.EIGHT_BUCKET_LADDER) == eight_rows
    assert residency.bucket_rows(patches, residency.FOUR_BUCKET_LADDER) == four_rows
    assert residency.bucket_rows(patches) == eight_rows  # the ladder of record
    assert min(eight_rows, four_rows) >= padded_rows(patches)  # never below the tile padding the tower needs anyway


def test_images_above_the_largest_bucket_are_refused_with_the_reason(expect_error) -> None:
    for ladder in (residency.EIGHT_BUCKET_LADDER, residency.FOUR_BUCKET_LADDER):
        with expect_error(ValueError, match="stock maximum"):
            residency.bucket_rows(65537, ladder)
    for bad in (0, -1, True, 2.0):
        with expect_error(ValueError):
            residency.bucket_rows(bad)


def test_padding_costs_at_most_twice_the_rows_on_eight_buckets_and_four_times_on_four() -> None:
    # From the processor's minimum image (65,536 px = 256 patches) up: the eight-bucket ladder pads at most 2x the rows,
    # the four-bucket ladder at most 4x; the padded patches are windowed out of attention, so the attention work grows
    # by the pad window's square only (at most 2x / 10x of the image's own).  Below the smallest bucket (an
    # under-minimum image the processor would upscale) the floor is a fixed cost, pinned separately.
    worst = {}
    for name, ladder in (("eight", residency.EIGHT_BUCKET_LADDER), ("four", residency.FOUR_BUCKET_LADDER)):
        linear = attention = 0.0
        for patches in list(range(256, 4096, 4)) + [65533, 65536, 32769, 16385, 8193, 4097, 1025]:
            cost = residency.padding_cost(patches, ladder)
            assert cost["rows"] == residency.bucket_rows(patches, ladder) and cost["pad_rows"] == cost["rows"] - patches
            linear, attention = max(linear, cost["linear_work_factor"]), max(attention, cost["attention_work_factor"])
        worst[name] = (linear, attention)
    assert worst["eight"][0] <= 2.0 and worst["eight"][1] <= 2.0
    assert worst["four"][0] <= 4.0 and worst["four"][1] <= 10.0
    assert (
        residency.padding_cost(1024)["linear_work_factor"]
        == 1.0
        == residency.padding_cost(1024)["attention_work_factor"]
    )
    assert residency.padding_cost(4, residency.EIGHT_BUCKET_LADDER)["rows"] == 512
    assert residency.padding_cost(4)["rows"] == 512 and residency.padding_cost(4)["linear_work_factor"] == 128.0


def test_admission_fits_the_line_reading_and_its_terms_cover_the_measurements() -> None:
    admission = residency.vision_capacity_admission(
        dies=4, live=LINE_32K_BEFORE_TOWER, reserved_bytes_per_bank=12_000_000
    )
    assert admission["fits"] and admission["decided_by"]["shortfalls"] == []
    # the modeled weights term is within 0.3 % of the measured residency on the line
    weights = admission["resident_weights_bytes_per_bank"]
    assert abs(weights - LINE_32K_MEASURED_TOWER_BYTES_PER_BANK) / LINE_32K_MEASURED_TOWER_BYTES_PER_BANK < 0.003
    # the peak term (18 KiB per row, 65,536 rows) covers the measured 65,536-row peak beside the chain
    assert admission["peak_activation_bytes_per_bank_largest_bucket"] >= LINE_32K_MEASURED_PEAK_65536_PER_BANK
    assert admission["required_free_bytes_per_bank"] == weights + -(
        -admission["peak_activation_bytes_per_bank_largest_bucket"] * 110 // 100
    )
    # the largest single buffer is the largest bucket's fused qkv activation, over the banks
    assert admission["required_largest_contiguous_bytes_per_bank"] == -(-65536 * 3 * 16 * 96 * 2 // 8)
    assert (
        admission["headroom_bytes_per_bank"]
        == LINE_32K_BEFORE_TOWER["free_bytes_per_bank"] - 12_000_000 - admission["required_free_bytes_per_bank"]
    )
    assert set(admission) >= {
        "required_free_bytes_per_bank",
        "required_largest_contiguous_bytes_per_bank",
        "headroom_bytes_per_bank",
        "decided_by",
        "fits",
        "buckets",
    }


def test_admission_decides_the_lanes_128k_form_on_the_numbers() -> None:
    admission = residency.vision_capacity_admission(dies=4, live=LANES_128K_AFTER_CAPTURES)
    # 125 MB of weights + 151 MB of peak x 1.1 = 291 MB against 305 MB free: the largest bucket fits with ~14 MB spare
    assert admission["fits"], admission["decided_by"]
    assert 0 < admission["headroom_bytes_per_bank"] < 30_000_000
    smaller = residency.vision_capacity_admission(
        dies=4, live=LANES_128K_AFTER_CAPTURES, buckets=residency.VISION_ROW_BUCKETS[:-1]
    )
    assert smaller["headroom_bytes_per_bank"] > admission["headroom_bytes_per_bank"]
    tight = dict(
        LANES_128K_AFTER_CAPTURES, free_bytes_per_bank=280_000_000, largest_contiguous_bytes_free_per_bank=280_000_000
    )
    refused = residency.vision_capacity_admission(dies=4, live=tight)
    assert not refused["fits"] and refused["decided_by"]["shortfalls"] == ["free_bytes_below_estimate"]
    fragmented = dict(LINE_32K_BEFORE_TOWER, largest_contiguous_bytes_free_per_bank=50_000_000)
    assert residency.vision_capacity_admission(dies=4, live=fragmented)["decided_by"]["shortfalls"] == [
        "largest_contiguous_below_largest_buffer"
    ]


def test_admission_refuses_inconsistent_inputs(expect_error) -> None:
    with expect_error(ValueError, match="banks"):
        residency.vision_capacity_admission(dies=4, live=dict(LINE_32K_BEFORE_TOWER, num_banks=12))
    with expect_error(ValueError, match="inconsistent"):
        residency.vision_capacity_admission(
            dies=4, live=dict(LINE_32K_BEFORE_TOWER, largest_contiguous_bytes_free_per_bank=10**12)
        )
    with expect_error(ValueError):
        residency.vision_capacity_admission(dies=0, live=LINE_32K_BEFORE_TOWER)
    with expect_error(ValueError):
        residency.vision_capacity_admission(dies=4, live=LINE_32K_BEFORE_TOWER, reserved_bytes_per_bank=-1)


def test_prewarm_plan_names_every_bucket_once_ascending_with_a_ready_cost_near_the_measured_ones() -> None:
    eight = residency.prewarm_plan(residency.EIGHT_BUCKET_LADDER)
    four = residency.prewarm_plan(residency.FOUR_BUCKET_LADDER)
    assert [row["rows"] for row in eight["buckets"]] == list(residency.EIGHT_BUCKET_LADDER) and eight["count"] == 8
    assert [row["rows"] for row in four["buckets"]] == list(residency.FOUR_BUCKET_LADDER) and four["count"] == 4
    for row in eight["buckets"]:
        measured = residency.PREWARM_SECONDS_MEASURED.get(row["rows"])
        if measured is not None:
            assert row["ready_seconds_measured"] == measured
            assert abs(row["ready_seconds_modeled"] - measured) / measured < 0.35  # the model is within the load band
    # MODELED: about two minutes for the eight buckets, about one for the four (the compiles dominate)
    assert 100 < eight["ready_seconds_modeled_total"] < 180 and 50 < four["ready_seconds_modeled_total"] < 100
    comparison = residency.ladder_comparison()  # the measured steady-state totals of record
    assert comparison["measured"] and comparison["pick"] == "eight"
    assert comparison["ready_seconds_difference"] == pytest.approx(43.65 - 33.47)
    assert comparison["eight"]["ready_seconds"] == residency.PREWARM_READY_SECONDS_MEASURED["eight"] == 43.65
    assert comparison["four"]["ready_seconds"] == residency.PREWARM_READY_SECONDS_MEASURED["four"] == 33.47
    assert comparison["eight"]["ready_seconds_cold_modeled"] == eight["ready_seconds_modeled_total"]
    assert residency.PREWARM_READY_SECONDS_COLD_MEASURED["eight"] > comparison["eight"]["ready_seconds"]
    # the steady state per bucket is dominated by the 65,536-row bucket (its two forwards): the rest cost under 14 s
    steady = residency.PREWARM_SECONDS_STEADY_MEASURED
    assert set(steady) == set(residency.EIGHT_BUCKET_LADDER) and steady[65536] > sum(
        v for k, v in steady.items() if k != 65536
    )
    assert (
        residency.ladder_comparison(100.0, 80.0)["pick"] == "eight"
    )  # a measured difference under the rule picks eight
    assert residency.ladder_comparison(120.0, 80.0)["pick"] == "four"
    # the warm forward law reproduces the measured warm device times within 25 %
    for rows_, measured in ((1024, 0.028), (4096, 0.122), (16384, 0.92), (65536, 12.4)):
        assert abs(residency.warm_forward_seconds_modeled(rows_) - measured) / measured < 0.25


class _FakeTower:
    """Stands in for ttnn.vision.VisionTower inside the residency's warm hook."""

    instances: list["_FakeTower"] = []

    def __init__(self, mesh_device, state_dict, config):
        self.loaded = False
        self.freed = False
        self.prewarmed: list[int] = []
        self.images: list[tuple[int, int]] = []
        _FakeTower.instances.append(self)

    def load(self):
        self.loaded = True
        return 1

    def free(self):
        self.freed = True

    def prewarm_rows(self, rows):
        assert self.loaded
        self.prewarmed.append(rows)
        return {"total_s": 0.0}

    def run_image(self, pixel_patches, grid_thw, *, rows, **kwargs):
        self.images.append((int(pixel_patches.shape[0]), rows))
        return SimpleNamespace(rows=rows)

    def resident_bytes_per_bank(self):
        return 124_939_776


def _residency(monkeypatch, live, **kwargs):
    import models.demos.blackhole.qwen38_flash_next.ttnn.vision as vision_module

    monkeypatch.setattr(vision_module, "VisionTower", _FakeTower)
    _FakeTower.instances.clear()
    mesh = SimpleNamespace(shape=(1, 4))
    return residency.Qwen38VisionResidency(mesh_device=mesh, state_dict={}, dram_view=lambda: dict(live), **kwargs)


def test_warm_hook_admits_then_loads_then_prewarms_every_bucket_ascending(expect_error, monkeypatch) -> None:
    pytest.importorskip("ttnn")
    r = _residency(monkeypatch, LINE_32K_BEFORE_TOWER, reserved_bytes_per_bank=12_000_000)
    assert not r.tower_resident and r.dies == 4
    r.vision_warm_hook(chain=None)
    tower = _FakeTower.instances[-1]
    assert r.tower_resident and tower.loaded and tower.prewarmed == list(residency.VISION_ROW_BUCKETS)
    assert r.admission["fits"] and r.load_seconds is not None
    assert len(r.prewarm_records) == len(residency.VISION_ROW_BUCKETS)
    summary = r.vision_summary()
    assert summary["resident"] and summary["shortfall"] is None and summary["shortfall_bytes_per_bank"] is None
    assert summary["largest_bucket_rows"] == 65536 and summary["resident_bytes_per_bank"] == 124_939_776
    assert [row["rows"] for row in summary["prewarm"]] == list(residency.VISION_ROW_BUCKETS)
    assert summary["ladders"]["pick"] in ("four", "eight")
    # the served entry runs an image in its bucket; the refusal reason is None while it would run
    assert r.refusal_reason(960) is None
    r.run_image_in_bucket(SimpleNamespace(shape=(960, 1536)), None)
    assert tower.images == [(960, 1024)]
    assert "stock maximum" in r.refusal_reason(65537)
    with expect_error(residency.Qwen38VisionResidencyError, match="stock maximum"):
        r.run_image_in_bucket(SimpleNamespace(shape=(65537, 1536)), None)
    r.release_vision()
    assert tower.freed and not r.tower_resident and r.vision_summary()["resident_bytes_per_bank"] is None


def test_warm_hook_leaves_the_process_text_only_on_a_shortfall_with_the_reason(expect_error, monkeypatch) -> None:
    pytest.importorskip("ttnn")
    tight = dict(
        LANES_128K_AFTER_CAPTURES, free_bytes_per_bank=200_000_000, largest_contiguous_bytes_free_per_bank=200_000_000
    )
    r = _residency(monkeypatch, tight)
    r.vision_warm_hook(chain=None)  # no raise: the server starts text-only
    assert not r.tower_resident and _FakeTower.instances == [] and not r.admission["fits"]
    summary = r.vision_summary()
    assert summary["resident"] is False and summary["shortfall_bytes_per_bank"] > 0
    assert "not resident" in summary["shortfall"] and "B per bank short" in summary["shortfall"]
    assert summary["shortfall_bytes_per_bank"] == r.admission["required_free_bytes_per_bank"] - 200_000_000
    # every image request is refused with that reason (the server's HTTP 400 text)
    assert r.refusal_reason(64) == summary["shortfall"]
    with expect_error(residency.Qwen38VisionResidencyError, match="B per bank short"):
        r.run_image_in_bucket(SimpleNamespace(shape=(64, 1536)), None)
    fragmented = dict(LINE_32K_BEFORE_TOWER, largest_contiguous_bytes_free_per_bank=50_000_000)
    r2 = _residency(monkeypatch, fragmented)
    r2.vision_warm_hook(chain=None)
    assert not r2.tower_resident and "largest contiguous block" in r2.vision_summary()["shortfall"]
    assert residency.shortfall_reason({"fits": True}) is None


def test_compose_warm_hooks_runs_in_order_and_is_none_without_hooks() -> None:
    order: list[str] = []
    hook = residency.compose_warm_hooks(
        None, lambda chain: order.append("lanes"), None, lambda chain: order.append("vision")
    )
    hook("chain")
    assert order == ["lanes", "vision"]
    assert residency.compose_warm_hooks(None, None) is None


def test_source_pins_the_tower_always_windows_and_the_served_entry_passes_the_bucket() -> None:
    source = VISION_SOURCE.read_text(encoding="utf-8")
    prepare = ast.unparse(
        next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    )
    assert "windows = torch.tensor(boundaries, dtype=torch.int32)" in prepare and "windows = None" not in prepare
    assert "if rows < patches or rows % TILE" in prepare
    residency_source = inspect.getsource(residency.Qwen38VisionResidency.run_image_in_bucket)
    assert "rows = self.rows_for(" in residency_source and "rows=rows" in residency_source
    hook = inspect.getsource(residency.Qwen38VisionResidency.vision_warm_hook)
    assert (
        hook.index("self.vision_admission()")
        < hook.index("VisionTower(")
        < hook.index(".load()")
        < hook.index("self.vision_prewarm()")
    )
    assert "raise" not in hook  # a shortfall never refuses the start
    prewarm = ast.unparse(
        next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "prewarm_rows")
    )
    # both window forms of a bucket compile: the filling image ([0, rows]) and a padded one ([0, rows - 4, rows])
    assert "for patches in (rows, rows - 4):" in prewarm and "rows=rows" in prewarm
    assert "ttnn.deallocate(output.features)" in prewarm


def test_builder_enable_vision_reads_the_tower_once_and_hands_out_the_warm_hook() -> None:
    source = BUILDER_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    builder = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNBuilder")
    enable = ast.unparse(next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "enable_vision"))
    assert "self.checkpoint.vision_state_dict()" in enable and "Qwen38VisionResidency(" in enable
    assert "already enabled" in enable and "self._assert_resident_builder_usable()" in enable
    assert "vision_residency: Qwen38VisionResidency | None = None" in source
    # no other builder method touches the tower: enabling is the only entry, the chain's warm hook the only device act
    others = [n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name != "enable_vision"]
    word = re.compile(r"(?<![a-z])vision(?![a-z])", re.I)  # the word, not checkpoint_re-vision
    assert not any(word.search(ast.unparse(n)) for n in others)


# -- the served path (tools/qwen38_chat_server.py): source pins and the parser with an image part --------------------

SERVER_SOURCE = ROOT / "tools" / "qwen38_chat_server.py"


def _server_function(name: str) -> str:
    tree = ast.parse(SERVER_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"no function {name} in the server")


def test_server_collects_and_decodes_image_parts_before_the_device() -> None:
    parse = _server_function("parse_chat_request")
    assert "protocol.normalize_messages(document.get('messages'), images=image_parts)" in parse
    assert "vision_inputs.decode_image(part.data, part.detail)" in parse
    assert "code='invalid_image'" in parse
    assert "'images': images" in parse and "'raw_messages': document.get('messages')" in parse


def test_server_refuses_or_renders_with_the_grids_then_runs_the_tower_on_the_device_thread() -> None:
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    handler = source[
        source.index("prompt_ids = protocol.render_prompt(") - 800 : source.index("**vision_inputs_kw,") + 40
    ]
    assert handler.index('refusal = self.server.vision_refusal(request["images"])') < handler.index(
        "prompt_ids = protocol.render_prompt("
    )
    assert 'code="vision_unavailable"' in handler
    assert 'image_grids=[image.grid for image in request["images"]]' in handler
    assert 'request["vision_positions"] = mrope.mrope_positions(' in handler
    assert handler.index("self.server.vision_prompt_for(") < handler.index("completion = session.complete(")
    assert 'vision_inputs_kw = {"vision": vision_prompt}' in handler and "**vision_inputs_kw," in handler
    refusal = _server_function("vision_refusal")
    # the lanes serve images since the follow-ups (the tower inside the lane admission): no lanes clause
    assert "self.lanes" not in refusal and "--lanes" not in refusal
    assert "self.vision is None" in refusal and "self.vision.refusal_reason(grid.t * grid.h * grid.w)" in refusal
    prompt_for = _server_function("vision_prompt_for")
    assert "vision_inputs.pixel_patches(image)" in prompt_for
    assert "self.vision.run_image_in_bucket(patches, grid_thw)" in prompt_for
    assert "features_to_torch(output).to(torch.bfloat16)" in prompt_for
    assert "vision_splice.Qwen38VisionPrompt(" in prompt_for and "prompt.validate_prompt(prompt_ids)" in prompt_for
    assert "vision_inputs.request_digest(images)" in prompt_for


def test_server_makes_the_residency_in_the_composed_warm_hook_and_records_it() -> None:
    source = SERVER_SOURCE.read_text(encoding="utf-8")
    hook = source[
        source.index("def vision_warm_hook(opened_chain)") : source.index(
            "warm_hook = compose_warm_hooks(warm_hook, vision_warm_hook)"
        )
    ]
    assert "opened_chain.construction.builder.enable_vision(" in hook
    assert "hardware_profiles.symmetric_mesh_dram_memory(mesh, hardware_profile.route)" in hook
    assert (
        "RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND" in hook and "LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES" in hook
    )
    assert 'opened_chain.mtp.admission["mtp_growth_estimate_bytes_per_bank"]["traces"]' in hook
    assert "residency.vision_warm_hook(opened_chain)" in hook
    assert 'summary["vision"] = residency.vision_summary()' in hook
    # the hook composes with the lanes' (theirs first), the residency reaches the HTTP server and the report
    assert source.index("lanes_session.prepare(opened_chain)") < source.index("def vision_warm_hook(opened_chain)")
    assert 'server.vision = vision_state["residency"]' in source
    assert '"vision": summary["vision"],' in source  # report["chain"]
    assert '"vision": None if self.server.vision is None else _vision_health(self.server.vision),' in source
    health = _server_function("_vision_health")
    for key in ("resident", "shortfall", "shortfall_bytes_per_bank", "buckets", "resident_bytes_per_bank"):
        assert f'"{key}"' in health or f"'{key}'" in health
    # the table fallback carries the modeled tower terms (the live decision stays the warm hook's)
    assert "vision_resident_bytes_per_bank=-(-vision_resident_layout(mesh_size=4)" in source
    assert (
        "vision_peak_activation_bytes_per_bank=" in source
        and "VISION_ROW_BUCKETS[-1] * PEAK_ACTIVATION_BYTES_PER_ROW_PER_DIE" in source
    )
    assert 'extension["vision"] = vision_record' in source and "if vision_record is not None:" in source


def test_parser_takes_an_image_part_and_refuses_undecodable_bytes() -> None:
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server_module

    pytest.importorskip("PIL")
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), (10, 20, 30)).save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    document = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "hi"}],
            }
        ],
        "max_tokens": 8,
    }
    request = server_module.parse_chat_request(document, sampling_available=False)
    assert len(request["images"]) == 1 and request["images"][0].grid.as_tuple() == (1, 32, 32)
    assert request["images"][0].merged_tokens == 256 and request["raw_messages"] is document["messages"]
    bad = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}
        ],
        "max_tokens": 8,
    }
    with pytest.raises(  # allow-pytest.raises: inspect the captured exception object
        server_module.Qwen38ChatRequestRejected
    ) as caught:  # allow-pytest.raises: inspect the captured exception object
        server_module.parse_chat_request(bad, sampling_available=False)
    assert caught.value.code == "invalid_image" and "messages[0].content[0].image_url" in str(caught.value)
    text_only = server_module.parse_chat_request(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}, sampling_available=False
    )
    assert text_only["images"] == []
