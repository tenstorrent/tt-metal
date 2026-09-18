# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for Kimi CI cache capacity and copy-process cancellation."""

import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "models/demos/deepseek_v3_d_p/scripts/run_kimi_prefill_ci.py"
spec = importlib.util.spec_from_file_location("kimi_ci", SCRIPT)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


@pytest.fixture
def cache(tmp_path):
    source = tmp_path / "source"
    weights = source / ci.CACHE_SUBDIR
    (weights / "nested").mkdir(parents=True)
    (weights / "nested/weight.tensorbin").write_bytes(bytes(range(256)) * 17)
    (weights / "other.tensorbin").write_bytes(b"second weight")
    local = tmp_path / "local"
    local.mkdir()
    return source, weights, local


@pytest.mark.parametrize("free_bytes", [361_900_000_000, 466_523_435_008])
def test_rejects_ci_worker_capacity_before_copying(cache, monkeypatch, free_bytes):
    _, weights, local = cache
    real_stat = Path.stat

    def measured_size(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if path.name == "weight.tensorbin":
            return SimpleNamespace(st_size=585_583_704_032, st_mode=result.st_mode)
        return result

    monkeypatch.setattr(Path, "stat", measured_size)
    monkeypatch.setattr(ci.shutil, "disk_usage", lambda _: SimpleNamespace(free=free_bytes))
    with pytest.raises(OSError, match="Provision a worker-local volume") as error:
        ci.check_cache(weights, local)
    assert str(free_bytes) in str(error.value)
    assert "--cache-dir" in str(error.value)
    assert list(local.iterdir()) == []


def test_missing_configured_volume_does_not_fall_back(cache):
    _, weights, local = cache
    with pytest.raises(OSError, match="No temporary-directory fallback"):
        ci.check_cache(weights, local / "missing")
    assert list(local.iterdir()) == []


def test_requires_headroom(cache, monkeypatch):
    _, weights, local = cache
    size = sum(path.stat().st_size for path in weights.rglob("*") if path.is_file())
    monkeypatch.setattr(ci.shutil, "disk_usage", lambda _: SimpleNamespace(free=size))
    with pytest.raises(OSError, match="Cache setup"):
        ci.check_cache(weights, local)
    monkeypatch.setattr(ci.shutil, "disk_usage", lambda _: SimpleNamespace(free=size + ci.CACHE_HEADROOM))
    assert len(ci.check_cache(weights, local)) == 2


def test_setup_check_uses_configured_volume_without_staging(cache, monkeypatch):
    source, _, local = cache
    monkeypatch.setenv(ci.CACHE_ENV, str(source))
    monkeypatch.setenv(ci.CACHE_DIR_ENV, str(local))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check-cache"])
    monkeypatch.setattr(ci.shutil, "disk_usage", lambda _: SimpleNamespace(free=ci.CACHE_HEADROOM * 2))
    assert ci.main() == 0
    assert list(local.iterdir()) == []


@pytest.mark.parametrize("exit_code", [0, 7])
def test_staging_contents_exit_status_and_cleanup(cache, tmp_path, exit_code):
    source, weights, local = cache
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "pytest.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "root = Path(os.environ['TT_KIMI_PREFILL_TTNN_CACHE'])\n"
        "assert (root / 'kimi_k2_7_bh_32dev/8x4/nested/weight.tensorbin').read_bytes() == bytes(range(256)) * 17\n"
        "assert sys.argv[1:] == ['test_model.py', '-k', 'traced']\n"
        "Path(os.environ['RECEIPT']).write_text(str(root))\n"
        "sys.exit(int(os.environ['CHILD_RC']))\n"
    )
    receipt = tmp_path / "receipt"
    env = dict(
        os.environ,
        TT_KIMI_PREFILL_TTNN_CACHE=str(source),
        PYTHONPATH=str(fake),
        RECEIPT=str(receipt),
        CHILD_RC=str(exit_code),
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--cache-dir", str(local), "--", "test_model.py", "-k", "traced"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == exit_code, result.stderr
    assert Path(receipt.read_text()).parent == local
    assert list(local.iterdir()) == []
    assert (weights / "nested/weight.tensorbin").read_bytes() == bytes(range(256)) * 17


def test_sigterm_stops_active_copy_threads_before_cleanup(cache, tmp_path):
    source, weights, local = cache
    # Spawn reimports this driver in the copy process, so both copy threads really
    # block inside copyfile. Cancelling only queued futures would hang this test.
    driver = tmp_path / "slow_copy.py"
    driver.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(SCRIPT.parent)!r})\n"
        "import run_kimi_prefill_ci as ci\n"
        "def slow_copy(source, target):\n"
        "    Path(target).write_bytes(b'partial')\n"
        "    (Path(os.environ['MARKERS']) / Path(source).name).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "ci.shutil.copyfile = slow_copy\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(ci.main())\n"
    )
    markers = tmp_path / "markers"
    markers.mkdir()
    env = dict(os.environ, TT_KIMI_PREFILL_TTNN_CACHE=str(source), MARKERS=str(markers))
    process = subprocess.Popen(
        [sys.executable, str(driver), "--workers", "2", "--cache-dir", str(local), "--", "must_not_run.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while len(list(markers.iterdir())) < 2 and time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.05)
        assert len(list(markers.iterdir())) == 2, "Copy workers did not both start"
        copy_pid = int(next(markers.iterdir()).read_text())
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 143, (stdout, stderr)
        with pytest.raises(ProcessLookupError):
            os.kill(copy_pid, 0)
        assert list(local.iterdir()) == []
        assert (weights / "nested/weight.tensorbin").read_bytes() == bytes(range(256)) * 17
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
