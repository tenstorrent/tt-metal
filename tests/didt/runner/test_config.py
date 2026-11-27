# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.didt.runner.config import ConfigError, load

SHIPPED = Path(__file__).parents[1] / "didt_runner_config.json"

STEP = {
    "key": "nd:x",
    "runner": "pytest",
    "ids": ["tests/didt/x.py::t[p]"],
    "opts": ["--timeout=0"],
    "env": {},
    "timeout": None,
    "check": "nd_report",
    "fallbacks": ["throttle"],
    "note": "",
}
BASE = {
    "version": 1,
    "hang_secs": 120,
    "reset": {"cmd": "tt-smi -glx_reset_auto", "tries": 3, "timeout": 600},
    "smi": {"cmd": "tt-smi -s --snapshot_no_tty", "timeout": 120},
    "collect": {"cmd": "pytest --collect-only -q -p no:benchmark", "timeout": 600},
    "fallbacks": {"throttle": {"env": {"TT_MM_THROTTLE_PERF": "5"}, "opts": []}},
    "steps": [STEP],
}


def _load_errors(path) -> list[str]:
    # The repo's expect_error fixture lives in a conftest; these tests run with --noconftest.
    try:
        load(path)
    except ConfigError as e:
        return e.errors
    pytest.fail(f"{path} loaded without a ConfigError")


def _errors(tmp_path, cfg) -> list[str]:
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    return _load_errors(path)


def _with_step(**changes):
    cfg = copy.deepcopy(BASE)
    cfg["steps"][0].update(changes)
    return cfg


def test_shipped_config_steps():
    c = load(SHIPPED)
    assert [s.key for s in c.steps] == [
        "nd:minimal",
        "nd:ff1",
        "nd:lm-head",
        "nd:duty-cycle-bursty-5",
        "nd:run-to-run",
        "nd:core-sweep",
        "nd:mcast-ack",
        "nd:mcast-ack-duty",
    ]
    assert all([f.name for f in s.fallbacks] == ["throttle", "pin_core"] for s in c.steps)
    assert c.steps[0].fallbacks[0].env == {"TT_MM_THROTTLE_PERF": "5"}
    pin = c.steps[0].fallbacks[1]
    assert pin.kind == "pin_core" and pin.hang_reset == {"cmd": "tt-smi -r", "timeout": 600}


def test_shipped_pin_ids():
    pins = {s.key: s.pin_id for s in load(SHIPPED).steps}
    assert pins == {
        "nd:minimal": "tests/didt/test_minimal_matmul.py::test_specific_chip_minimal_matmul"
        "[blackhole-bf16_HiFi2-galaxy-galaxy_chip_0]",
        "nd:ff1": "tests/didt/test_ff1_matmul.py::test_specific_chip_ff1_matmul[blackhole-galaxy_chip_0-without_gelu]",
        "nd:lm-head": "tests/didt/test_lm_head_matmul.py::test_specific_chip_lm_head_matmul"
        "[blackhole-galaxy-logical_chip_0_]",
        "nd:duty-cycle-bursty-5": "tests/didt/test_duty_cycle.py::test_specific_chip_duty_cycle"
        "[blackhole-duty-5-rep-1000x-galaxy-galaxy_chip_0]",
        "nd:run-to-run": None,
        "nd:core-sweep": None,
        "nd:mcast-ack": "tests/didt/test_mm_mcast_write_ack_failure.py::test_mm_mcast_write_ack_failure_scan"
        "[blackhole-1x1]",
        "nd:mcast-ack-duty": "tests/didt/test_mm_mcast_write_ack_failure.py::test_mm_mcast_write_ack_failure_duty"
        "[blackhole-1x1]",
    }


def test_shipped_config_matches_spec_table():
    c = {s.key: s for s in load(SHIPPED).steps}
    m = c["nd:minimal"]
    assert m.ids == ["tests/didt/test_minimal_matmul.py::test_minimal_matmul[blackhole-bf16_HiFi2-galaxy]"]
    assert m.opts == ["--didt-workload-iterations=1000", "--determinism-check-interval=50", "--timeout=0"]
    assert m.timeout is None and m.check == "nd_report" and m.runner == "pytest"
    assert c["nd:duty-cycle-bursty-5"].env == {
        "TT_MESH_GRAPH_DESC_PATH": "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"
    }
    assert c["nd:run-to-run"].env == {"TT_DET_SWEEP_ITERS": "10"}
    assert c["nd:core-sweep"].env == {"TT_SWEEP_ITERS": "10", "TT_SWEEP_MODE": "grid", "TT_SWEEP_GRID": "11x10"}
    ack, duty = c["nd:mcast-ack"], c["nd:mcast-ack-duty"]
    assert ack.runner == duty.runner == "script"
    assert ack.cmd == duty.cmd == "python tests/didt/test_mm_mcast_write_ack_failure.py"
    assert ack.env == {"MATMUL_ITERS": "10", "MATMUL_WORKLOAD": "wkv_b1"}
    assert duty.env == {"MATMUL_ITERS": "10", "MATMUL_WORKLOAD": "duty"}
    assert ack.timeout == duty.timeout == 1800 and ack.check == duty.check == "write_ack"
    assert duty.ids == [
        "tests/didt/test_mm_mcast_write_ack_failure.py::test_mm_mcast_write_ack_failure_duty[blackhole-1x1]"
    ]


def test_config_identity(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(BASE))
    c = load(path)
    assert c.path == path and c.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert c.hang_secs == 120 and c.reset["tries"] == 3 and c.collect["cmd"].startswith("pytest --collect-only")


def test_unknown_field(tmp_path):
    assert any("nd:x: unknown field 'iters'" in e for e in _errors(tmp_path, _with_step(iters=5)))


def test_unknown_top_level_field(tmp_path):
    assert any("unknown field 'hang_sec'" in e for e in _errors(tmp_path, {**BASE, "hang_sec": 5}))


def test_missing_field(tmp_path):
    cfg = copy.deepcopy(BASE)
    del cfg["steps"][0]["check"]
    assert any("nd:x: missing field 'check'" in e for e in _errors(tmp_path, cfg))


def test_duplicate_key(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["steps"].append(copy.deepcopy(STEP))
    assert any("duplicate key 'nd:x'" in e for e in _errors(tmp_path, cfg))


def test_unknown_check(tmp_path):
    assert any("unknown check 'nd'" in e for e in _errors(tmp_path, _with_step(check="nd")))


def test_unknown_runner(tmp_path):
    assert any("unknown runner 'bash'" in e for e in _errors(tmp_path, _with_step(runner="bash")))


def test_unknown_fallback(tmp_path):
    assert any("unknown fallback 'core_cap'" in e for e in _errors(tmp_path, _with_step(fallbacks=["core_cap"])))


def test_k_in_opts(tmp_path):
    assert any("'-k' in opts" in e for e in _errors(tmp_path, _with_step(opts=["-k", "galaxy"])))


def test_m_in_opts(tmp_path):
    assert any("'-m' in opts" in e for e in _errors(tmp_path, _with_step(opts=["-m=slow"])))


def test_id_without_colons(tmp_path):
    assert any("id without '::'" in e for e in _errors(tmp_path, _with_step(ids=["tests/didt/x.py"])))


def test_non_string_env_value(tmp_path):
    assert any("env 'N' must be a string" in e for e in _errors(tmp_path, _with_step(env={"N": 10})))


def test_non_string_fallback_env_value(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["fallbacks"]["throttle"]["env"]["TT_MM_THROTTLE_PERF"] = 5
    assert any("env 'TT_MM_THROTTLE_PERF' must be a string" in e for e in _errors(tmp_path, cfg))


def test_script_without_cmd(tmp_path):
    assert any("script runner needs 'cmd'" in e for e in _errors(tmp_path, _with_step(runner="script")))


def test_cmd_on_pytest_runner(tmp_path):
    assert any("'cmd' is for the script runner" in e for e in _errors(tmp_path, _with_step(cmd="python x.py")))


def test_all_errors_reported_together(tmp_path):
    assert len(_errors(tmp_path, _with_step(check="nd", runner="bash"))) == 2


def test_bad_json(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{")
    assert len(_load_errors(path)) == 1


def test_unknown_field_in_reset(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["reset"]["retries"] = 3
    assert any("reset: unknown field 'retries'" in e for e in _errors(tmp_path, cfg))


PIN = {"kind": "pin_core", "hang_reset": {"cmd": "tt-smi -r", "timeout": 600}}


def test_pin_core_fallback_loads(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["fallbacks"]["pin_core"] = PIN
    cfg["steps"][0].update(fallbacks=["throttle", "pin_core"], pin_id="tests/didt/x.py::t_chip[p]")
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    step = load(path).steps[0]
    assert step.pin_id == "tests/didt/x.py::t_chip[p]"
    assert [(f.name, f.kind) for f in step.fallbacks] == [("throttle", "env"), ("pin_core", "pin_core")]


def test_unknown_fallback_kind(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["fallbacks"]["pin"] = {"kind": "per_core"}
    assert any("fallback pin: unknown kind 'per_core'" in e for e in _errors(tmp_path, cfg))


def test_pin_core_fallback_fields(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg["fallbacks"]["pin_core"] = {**PIN, "env": {}, "max_devices": 2}
    errors = _errors(tmp_path, cfg)
    assert any("fallback pin_core: unknown field 'env'" in e for e in errors)
    assert any("fallback pin_core: unknown field 'max_devices'" in e for e in errors)


def test_pin_id_without_colons(tmp_path):
    assert any("pin_id without '::'" in e for e in _errors(tmp_path, _with_step(pin_id="tests/didt/x.py")))
