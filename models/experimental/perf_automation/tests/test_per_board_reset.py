import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "ccrun_reset_ut",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py"),
)
run = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run)


def _stdout(pairs):
    return json.dumps({"device_info": [{"board_info": {"board_id": b, "bus_id": u}} for b, u in pairs]})


_P300C = [("A", "0000:01:00.0"), ("A", "0000:02:00.0"), ("B", "0000:03:00.0"), ("B", "0000:04:00.0")]
_N300 = [("A", "0000:01:00.0"), ("A", "N/A")]


class _R:
    def __init__(self, out="", rc=0):
        self.stdout = out
        self.returncode = rc


def test_read_topology_p300c_whole_board(monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: _R(_stdout(_P300C)))
    assert run._read_board_topology() == {"0": [0, 1], "1": [0, 1], "2": [2, 3], "3": [2, 3]}


def test_read_topology_n300_local_only(monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: _R(_stdout(_N300)))
    assert run._read_board_topology() == {"0": [0], "1": [0]}


def test_board_reset_targets_union_whole_boards(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "_BOARD_MAP_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(run, "_read_board_topology", lambda: {"0": [0, 1], "1": [0, 1], "2": [2, 3], "3": [2, 3]})
    assert run._board_reset_targets([0]) == "0,1"
    assert run._board_reset_targets([2]) == "2,3"
    assert run._board_reset_targets([0, 2]) == "0,1,2,3"


def test_board_reset_targets_legacy_int_cache(tmp_path, monkeypatch):
    f = tmp_path / "m.json"
    f.write_text('{"0": 0, "1": 0}')
    monkeypatch.setattr(run, "_BOARD_MAP_FILE", f)
    assert run._board_reset_targets([0, 1]) == "0"


def test_reset_chip_list_per_board_then_fallback(monkeypatch):
    monkeypatch.setattr(run, "_board_reset_targets", lambda req: "0,1" if req == [0] else None)
    assert run._reset_chip_list("0") == "0,1"
    assert run._reset_chip_list("single") == "0,1"
    assert run._reset_chip_list("all") == ""
    assert run._reset_chip_list("") == ""
    monkeypatch.setattr(run, "_board_reset_targets", lambda req: None)
    assert run._reset_chip_list("3") == ""


def test_reset_devices_prefers_per_board_over_full_enum(monkeypatch):
    import agent.probes as P

    monkeypatch.setattr(P, "_reset_arg_sets", lambda: [["-r", "0,1,2,3"]])
    monkeypatch.setattr(P, "_GALAXY_HOST", False, raising=False)
    monkeypatch.setattr(run, "_reset_chip_list", lambda dev: "0,1")
    monkeypatch.setattr(run.shutil, "which", lambda _x: __file__)
    seen = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: seen.append(cmd) or _R("", 0))
    run._reset_devices("0")
    assert seen and seen[0][1:] == ["-r", "0,1"]


def test_reset_devices_full_enum_fallback_when_no_per_board(monkeypatch):
    import agent.probes as P

    monkeypatch.setattr(P, "_reset_arg_sets", lambda: [["-r", "0,1,2,3"]])
    monkeypatch.setattr(P, "_GALAXY_HOST", False, raising=False)
    monkeypatch.setattr(run, "_reset_chip_list", lambda dev: "")
    monkeypatch.setattr(run.shutil, "which", lambda _x: __file__)
    seen = []
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **k: seen.append(cmd) or _R("", 0))
    run._reset_devices("0")
    assert seen and seen[0][1:] == ["-r", "0,1,2,3"]
