# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP activation wiring (run.py route decision). _decide_parallelism_route turns the discovered
manifest + model weight files into a route and exports TT_PERF_TP_REGIME=1 automatically when the
model does not fit on one chip, or when it fits under a latency metric on a mesh. A throughput metric,
a missing capacity fact, or any error leaves the regime off."""
import importlib.util
from pathlib import Path

_RUN_PY = Path(__file__).parent.parent / "cc_optimize" / "run.py"
_spec = importlib.util.spec_from_file_location("cc_run_under_test", _RUN_PY)
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)

BH = {"env": {"arch": "blackhole", "mesh_chips": 4}, "model_config": {"num_attention_heads": 16, "hidden_size": 2048}}


def _mk_weights(tmp_path, gb):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\0")
    import os as _os

    _os.truncate(f, int(gb * 1024**3))
    return tmp_path


def test_fits_throughput_metric_leaves_regime_off(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_TP_REGIME", raising=False)
    run._decide_parallelism_route(_mk_weights(tmp_path, 4), BH, metric="fps")
    import os

    assert os.environ.get("TT_PERF_TP_REGIME") != "1"


def test_fits_latency_metric_auto_enables_regime(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_TP_REGIME", raising=False)
    run._decide_parallelism_route(_mk_weights(tmp_path, 4), BH, metric="device_ms")
    import os

    assert os.environ.get("TT_PERF_TP_REGIME") == "1"


def test_too_big_enables_regime_automatically(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_TP_REGIME", raising=False)
    run._decide_parallelism_route(_mk_weights(tmp_path, 40), BH)
    import os

    assert os.environ.get("TT_PERF_TP_REGIME") == "1"


def test_missing_capacity_fact_is_failsafe_off(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_TP_REGIME", raising=False)
    unknown = {"env": {"arch": "nonexistent-arch", "mesh_chips": 4}, "model_config": {}}
    run._decide_parallelism_route(_mk_weights(tmp_path, 40), unknown)
    import os

    assert os.environ.get("TT_PERF_TP_REGIME") != "1"


def test_route_fires_when_agent_not_on_syspath(tmp_path, monkeypatch):
    monkeypatch.delenv("TT_PERF_TP_REGIME", raising=False)
    import os
    import subprocess
    import sys

    repo_root = str(Path(__file__).resolve().parents[4])
    f = tmp_path / "w.safetensors"
    f.write_bytes(b"\0")
    os.truncate(f, 40 * 1024**3)
    snippet = (
        "import importlib.util, os, sys, json\n"
        f"spec=importlib.util.spec_from_file_location('r', {str(_RUN_PY)!r})\n"
        "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "manifest={'env':{'arch':'blackhole','mesh_chips':4},'model_config':{'num_attention_heads':16,'hidden_size':2048}}\n"
        f"m._decide_parallelism_route({str(tmp_path)!r}, manifest, repo_root={repo_root!r})\n"
        "print('REGIME='+str(os.environ.get('TT_PERF_TP_REGIME')))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, cwd="/tmp", env={"PATH": os.environ["PATH"]}
    )
    assert "REGIME=1" in out.stdout
    assert "No module named 'agent'" not in (out.stdout + out.stderr)


def test_weight_bytes_and_dims_from_hf_cache_real():
    import pytest

    dd = "/home/ttuser/tt-metal/models/demos/wormhole/bge_m3"
    if not Path(dd).is_dir() or run._resolve_model_id(dd) is None:
        pytest.skip("bge_m3 model dir / HF cache not present")
    wb = run._model_weight_bytes(dd)
    dims = run._hf_cache_dims(run._resolve_model_id(dd))
    assert wb > 1_000_000_000
    assert int(dims["hidden_size"]) == 1024 and int(dims["num_attention_heads"]) == 16


def test_chip_count_from_devices_spec():
    assert run._chip_count("0,1,2,3") == 4
    assert run._chip_count("0,1") == 2
    assert run._chip_count("single") == 1


def test_resolve_model_id_prefers_valid_hint(tmp_path):
    dd = "/home/ttuser/tt-metal/models/demos/wormhole/bge_m3"
    import pytest

    if run._resolve_model_id(dd) is None:
        pytest.skip("bge_m3 HF cache not present")
    assert run._resolve_model_id(str(tmp_path), hint="BAAI/bge-m3") == "BAAI/bge-m3"
    assert run._resolve_model_id(str(tmp_path), hint="not/a-real-model") is None
