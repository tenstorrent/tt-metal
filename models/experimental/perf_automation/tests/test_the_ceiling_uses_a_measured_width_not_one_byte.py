# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A ceiling built from one byte per parameter, pinned before the measurement that would fix it.

WHAT THE REPORT PRINTED, 2026-08-16:

    DECODE — per token │ THEORETICAL │ MEASURED
      memory ← binds   │  7.05 ms    │  6.11 ms
                       │ 512.0 GB/s  │ 590.9 GB/s

590.9 GB/s on a 512 GB/s part. Not a fast model -- an impossible ceiling, and the reader cannot act
on either number. It came from two separate defects.

FIRST: THE WIDTH WAS A CONSTANT. `params x _BYTES_PER_PARAM` with _BYTES_PER_PARAM = 1.0, i.e. one
byte per parameter whatever the model is served at. That is right only for a 1-byte format: bf16 is
2.0, bf4 is 0.5625, and this model's measured mix is 1.3228. It published voxtral at 141.8 tok/s/u
against a true ~55, and survived review because gemma-3's bf8 (1.0625) sits within 6% of 1.0 and
looked correct. The checkpoint declares `dominant_dtype` at phase "before" with no device needed, so
a real width was available the whole time.

SECOND: THE PLACEHOLDER OUTRANKED THE MEASUREMENT. active_bytes is a write-once anchor, deliberately,
so the report and the stop gate can never score one run against two ceilings -- and it is written at
phase "before", when only the checkpoint exists. Correct on both counts. But it also outranked the
DEVICE CENSUS, which walks the built model and measures 1.72 GB actually resident, and only exists
afterwards. So the census was computed, written to perf_target_inputs.json, and never read.

THE RULE. A placeholder anchor is superseded exactly once, by a measurement of the same quantity,
before optimisation begins. Recognised by arithmetic -- params x 1.0 is exact, so an anchor within
half a percent of the parameter count is that rule and no other. Every anchor derived any other way
stays pinned, and the value never drifts during the run.

    active_bytes 3.61 GB -> 1.72 GB      decode floor 7.05 ms -> 3.36 ms
    measured 6.11 ms = 55% of peak, which is a number worth reading.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_FACTS = {
    "total_params": 3611483136,
    "dominant_dtype": "bfloat16",
    "weight_bytes": 9356474312,
    "device_weight_bytes": 1718081696,
    "device_census_complete": True,
    "bytes_per_param": 1.3228,
}
_PLACEHOLDER = 3611483136  # params x 1.0, exactly what the ledger pinned


def _pre_census(**over):
    mf = dict(_FACTS)
    mf.pop("bytes_per_param", None)
    mf.pop("device_weight_bytes", None)
    mf["device_census_complete"] = False
    mf.update(over)
    return mf


# ------------------------------------------------------------------ the width


def test_the_measured_width_is_used_when_the_census_has_run():
    from agent.perf_target import simple_active_bytes

    assert simple_active_bytes(_FACTS) == round(3611483136 * 1.3228)


def test_before_the_census_the_declared_dtype_decides_not_one_byte():
    """bf16 is 2 bytes. The old rule said 1 and was wrong by half."""
    from agent.perf_target import simple_active_bytes

    got = simple_active_bytes(_pre_census())
    assert got == 3611483136 * 2, got
    assert got != 3611483136, "the 1-byte placeholder is back"


def test_a_four_bit_model_is_not_given_a_one_byte_width():
    """The constant was wrong in BOTH directions; bf4 would have been overstated ~1.8x."""
    from agent.perf_target import simple_active_bytes

    got = simple_active_bytes(_pre_census(dominant_dtype="bfloat4_b"))
    assert got < 3611483136, "a sub-byte dtype still priced at one byte per parameter"


def test_no_declared_dtype_falls_back_to_the_checkpoints_own_total():
    """Last real evidence available without a device. Never a made-up width."""
    from agent.perf_target import simple_active_bytes

    mf = _pre_census()
    mf.pop("dominant_dtype")
    mf.pop("torch_dtype", None)
    assert simple_active_bytes(mf) == 9356474312


def test_the_one_byte_convention_is_gone_entirely():
    """Not demoted -- removed. A width is a property of the model, so it comes FROM the model or not
    at all: the census measures it, the checkpoint declares it, or the checkpoint's byte total states
    it. With none of those, 0 means "no ceiling", which the caller renders as a missing roofline
    rather than a number a reader would act on.

    The old rule (params x 1.0) was a 2026-07-29 team decision, justified as conservative because TT
    models are typically served under a byte per parameter. bf16 inverts that: voxtral streams 2
    B/param and was handed a ceiling ABOVE what the hardware permits. And the width is not even fixed
    for one model -- a dtype rung moves it mid-run, bf16 -> bf8 -> bf4 -- so no constant can stand in.
    """
    from agent.perf_target import simple_active_bytes

    assert simple_active_bytes({"total_params": 8_000_000_000}) == 0, "the constant still answers"

    src = (_PA / "agent" / "perf_target.py").read_text()
    i = src.index("def simple_active_bytes(")
    body = src[i : src.index("\ndef ", i + 1)].split('"""', 2)[-1]
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "_BYTES_PER_PARAM" not in code, "the byte rule still multiplies by the retired constant"


def test_the_constant_survives_only_to_recognise_an_old_anchor():
    """The ledger stores a value, not the rule behind it, so arithmetic is the only way to tell a
    placeholder anchor from a measured one -- that is the constant's one remaining job."""
    src = (_PA / "agent" / "perf_target.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    uses = [ln.strip() for ln in code.splitlines() if "_BYTES_PER_PARAM" in ln]
    # one definition, one use, and the use is inside the placeholder recogniser
    assert len(uses) == 2, uses
    assert uses[0].startswith("_BYTES_PER_PARAM = ")
    i = code.index("def _anchor_is_placeholder(")
    assert uses[1] in code[i : code.index("\ndef ", i + 1)], "the constant is used outside the recogniser"


# ------------------------------------------------------------------ the anchor


def test_a_placeholder_anchor_is_recognised():
    from agent.perf_target import _anchor_is_placeholder

    assert _anchor_is_placeholder(_PLACEHOLDER, _FACTS) is True


def test_an_anchor_from_real_evidence_is_left_alone():
    """Only the placeholder is overridable. A census or checkpoint anchor stays exactly as pinned."""
    from agent.perf_target import _anchor_is_placeholder

    assert _anchor_is_placeholder(1718081696, _FACTS) is False
    assert _anchor_is_placeholder(9356474312, _FACTS) is False


def test_the_census_supersedes_the_placeholder_anchor():
    """THE BUG: 3.61 GB pinned, 1.72 GB measured, measurement ignored."""
    from agent.perf_target import compute_target

    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=_PLACEHOLDER)
    # params x the measured WIDTH, not the census's byte total: the total is 1.718 GB here only
    # because that census was depth-capped to 2 layers. See
    # test_the_ceiling_uses_the_measured_width.py -- the same census at 62 layers totals 7.043 GB,
    # and a divisor that moves 4.1x with the capture depth is not a property of the model.
    assert t.active_bytes == 4777269892, t.active_bytes


def test_the_resulting_floor_is_physically_possible():
    """6.11 ms measured against a 512 GB/s part must not imply 590.9 GB/s."""
    from agent.perf_target import compute_target

    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=_PLACEHOLDER)
    floor_ms = 1e3 * t.active_bytes / 512e9
    # 6.11 ms was itself a capped-depth decode. Measured at FULL depth the stage takes 17.86 ms, and
    # THAT is what a floor has to sit under. The invariant is unchanged -- a floor above the
    # measurement implies a bandwidth above peak -- only the measurement it is checked against.
    measured_ms = 17.86
    assert floor_ms < measured_ms, "measured still beats the floor: %.2f ms" % floor_ms
    assert 1e-9 * t.active_bytes / (measured_ms * 1e-3) < 512, "implied bandwidth still exceeds peak"


def test_a_non_placeholder_anchor_still_wins_over_the_census():
    """The write-once guarantee survives: this replaces a guess, it does not open the anchor up."""
    from agent.perf_target import compute_target

    t = compute_target(_FACTS, {"dram_bw_gbps": 512.0}, bytes_per_unit=7000000000)
    assert t.active_bytes == 7000000000


def test_an_incomplete_census_never_supersedes():
    """Too few bytes reads as too HIGH a ceiling -- the direction that ends a run early."""
    from agent.perf_target import compute_target

    mf = dict(_FACTS, device_census_complete=False)
    t = compute_target(mf, {"dram_bw_gbps": 512.0}, bytes_per_unit=_PLACEHOLDER)
    assert t.active_bytes == _PLACEHOLDER


# ------------------------------------------------------- and it must not move once optimisation starts


def test_the_census_is_frozen_after_the_first_complete_measurement(tmp_path, monkeypatch):
    """THE DRIFT THIS WOULD OTHERWISE CAUSE. _persist_device_weight_bytes is called from
    _run_full_pipeline_ms, so it fires on every gate -- every iteration. The census measures the model
    AS BUILT, so a dtype rung (bf16 -> bf8) halves the resident weights and would drag the ceiling
    down with it: the run would chase a target that recedes as it improves."""
    import json as _json

    from cc_optimize import perf_mcp

    box = tmp_path / "box"
    box.mkdir()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setenv("PERF_MCP_BOARD_STATE_DIR", str(box))
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT_STATED", True)
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path)
    target = tmp_path / "perf_target_inputs.json"
    target.write_text(_json.dumps({"total_params": 3611483136}))

    perf_mcp._persist_device_weight_bytes(1718081696, True, 1.3228)
    assert _json.loads(target.read_text())["device_weight_bytes"] == 1718081696

    # a dtype win halves the resident weights on a later iteration
    perf_mcp._persist_device_weight_bytes(859040848, True, 0.6614)
    doc = _json.loads(target.read_text())
    assert doc["device_weight_bytes"] == 1718081696, "the ceiling moved mid-run"
    assert doc["bytes_per_param"] == 1.3228


def test_an_incomplete_census_may_still_be_replaced(tmp_path, monkeypatch):
    """Not an answer yet: a dtype the census has no width for must not freeze a partial figure."""
    import json as _json

    from cc_optimize import perf_mcp

    box = tmp_path / "box"
    box.mkdir()
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(box))
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT_STATED", True)
    monkeypatch.setattr(perf_mcp, "_MODEL_ROOT", tmp_path)
    target = tmp_path / "perf_target_inputs.json"
    target.write_text(_json.dumps({"total_params": 3611483136}))

    perf_mcp._persist_device_weight_bytes(900000000, False, 0.0)
    perf_mcp._persist_device_weight_bytes(1718081696, True, 1.3228)
    assert _json.loads(target.read_text())["device_weight_bytes"] == 1718081696
