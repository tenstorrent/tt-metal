"""F297: a test declaring BATCH_COLD = True runs in a fresh process and first after the determinism check (a build
counter in a held process found the previous run's programs cached: PB.9C's test_no_rebuild 153 builds -> 0)."""

import json

from models.demos.common.bringup.core import batch as B


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_cold_mark_orders_and_requests_a_fresh_process(tmp_path, monkeypatch):
    cold = _write(tmp_path, "test_cold.py", "BATCH_SHARED = True\nBATCH_COLD = True\n")
    warm = _write(tmp_path, "test_warm.py", "BATCH_SHARED = True\n")
    assert B.cold({"args": [cold]}) and not B.cold({"args": [warm]})
    entries = [{"args": [warm], "env": {}}, {"args": [cold], "env": {}}]
    entries.sort(key=lambda e: -2 if B.phase(e) == -1 else (-1 if B.cold(e) else 0))
    assert entries[0]["args"] == [cold]
    seen = {}
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"entries": [{"args": [warm], "env": {}, "name": "a"}, {"args": [cold], "env": {}, "name": "b"}]})
    )

    class R:
        @staticmethod
        def submit_plan(d, p, env, fresh=False):
            seen["fresh"] = fresh
            return {"rc": 0}

        @staticmethod
        def describe(r):
            return ""

    import models.demos.common.bringup as pkg

    monkeypatch.setattr(pkg, "resident", R, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "models.demos.common.bringup.resident", R)
    B._drive_resident(tmp_path, plan, {}, lambda s: None)
    assert seen["fresh"] is True
