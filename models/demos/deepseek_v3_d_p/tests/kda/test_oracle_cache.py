# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts of the CPU oracle cache shared by worktrees (utils/oracle_cache.py); no device.

Covers: where oracles and weight caches live, that distinct oracles never share a key, that concurrent producers
in separate processes publish one correct entry, and that readers never see a partially written entry.
"""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.kda.cases import KDA_CASES, build_kda_case, kda_weight_cache_dir
from models.demos.deepseek_v3_d_p.tests.kda.decay_extremes import DECAY_EXTREME_CASES, _cache_path
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import cpu_reference_cache_path
from models.demos.deepseek_v3_d_p.tests.kda.text_input import text_input_cache_path
from models.demos.deepseek_v3_d_p.utils import oracle_cache
from models.demos.deepseek_v3_d_p.utils.oracle_cache import SHARED_ORACLE_CACHE_ENV, oracle_cache_root, publish_once

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_PAYLOAD = bytes(range(256)) * 4096  # 1 MiB, recognizable content

# A producer process: publish `path` with _PAYLOAD after an optional delay inside produce(); records each produce()
# call in `log`. With `partial`, it writes half the entry, reports it and waits to be killed.
_PRODUCER = """
import os, sys, time
from pathlib import Path
from models.demos.deepseek_v3_d_p.utils.oracle_cache import publish_once
path, log, start_at, mode = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
payload = bytes(range(256)) * 4096
def produce():
    with open(log, "a") as stream:
        stream.write(f"{os.getpid()}\\n")
    time.sleep(2.0)
    return payload
def write(value, file):
    if mode != "partial":
        file.write_bytes(value)
        return
    with open(file, "wb") as stream:
        stream.write(value[: len(value) // 2])
        stream.flush()
    print("partial", flush=True)
    time.sleep(600)
time.sleep(max(0.0, start_at - time.time()))
value, produced = publish_once(path, produce, write, lambda file: file.read_bytes())
print(f"produced={produced} value_ok={value == payload}", flush=True)
"""


def _producer(path: Path, log: Path, start_at: float, mode: str = "complete") -> subprocess.Popen:
    environment = {**os.environ, "PYTHONPATH": str(_REPOSITORY_ROOT)}
    return subprocess.Popen(
        [sys.executable, "-c", _PRODUCER, str(path), str(log), str(start_at), mode],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _leftovers(directory: Path, entry: Path) -> list[Path]:
    """Files other than the entry and its persistent lock file."""
    return [file for file in directory.iterdir() if file not in (entry, entry.with_name(f"{entry.name}.lock"))]


def test_oracles_use_the_shared_root_and_weight_caches_stay_per_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared, checkout = tmp_path / "shared", tmp_path / "checkout_model_cache"
    monkeypatch.setenv(SHARED_ORACLE_CACHE_ENV, str(shared))
    monkeypatch.setattr(ttnn.CONFIG, "model_cache_path", checkout)
    spec = KDA_CASES["kimi_k3-synthetic-mesh2x4-tpaxis1-T1280"]
    case = build_kda_case(spec)

    assert shared in cpu_reference_cache_path(case.weights, case.chunk_valid_hidden(0), None).parents
    assert shared in text_input_cache_path("kimi_k3", 1, 1280).parents
    assert shared in _cache_path(next(iter(DECAY_EXTREME_CASES.values()))).parents
    assert checkout in kda_weight_cache_dir(case.weights, spec.mesh_shape, spec.tensor_parallel_axis).parents


def test_root_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    host_local = tmp_path / "localdev"
    monkeypatch.setattr(oracle_cache, "_HOST_LOCAL_ROOT", host_local)
    monkeypatch.setattr(oracle_cache.getpass, "getuser", lambda: "someone")
    monkeypatch.setattr(ttnn.CONFIG, "model_cache_path", tmp_path / "model_cache")
    monkeypatch.delenv(SHARED_ORACLE_CACHE_ENV, raising=False)

    assert oracle_cache_root() == tmp_path / "model_cache"  # no host-local area: the checkout's model cache
    (host_local / "someone").mkdir(parents=True)
    assert oracle_cache_root() == host_local / "someone" / ".cache" / "tt-linear-layers-shared"
    monkeypatch.setenv(SHARED_ORACLE_CACHE_ENV, str(tmp_path / "explicit"))
    assert oracle_cache_root() == tmp_path / "explicit"


def test_reference_keys_collide_only_for_identical_oracles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Across every registered random-input case, one key <=> one oracle identity (weights, config, input)."""
    monkeypatch.setenv(SHARED_ORACLE_CACHE_ENV, str(tmp_path))
    identities_by_key: dict[Path, set] = {}
    keys_by_identity: dict[tuple, set] = {}
    for spec in (spec for spec in KDA_CASES.values() if spec.inputs == "randn"):
        case = build_kda_case(spec)
        hidden = case.chunk_valid_hidden(0)
        digest = hashlib.sha256(memoryview(hidden.contiguous().view(torch.uint8).numpy())).hexdigest()
        identity = (case.weights.model, case.weights.layer_idx, case.weights.identity, case.config, digest)
        key = cpu_reference_cache_path(case.weights, hidden, None)
        identities_by_key.setdefault(key, set()).add(identity)
        keys_by_identity.setdefault(identity, set()).add(key)
    assert all(len(identities) == 1 for identities in identities_by_key.values())
    assert all(len(keys) == 1 for keys in keys_by_identity.values())
    # Mesh placement does not change the CPU oracle, so cases differing only in mesh share it.
    assert len(identities_by_key) < sum(1 for spec in KDA_CASES.values() if spec.inputs == "randn")

    decay_paths = {_cache_path(case) for case in DECAY_EXTREME_CASES.values()}
    assert len(decay_paths) == len(DECAY_EXTREME_CASES)
    text_inputs = {
        (spec.model, spec.weight_source().layer_idx, spec.chunk_tokens * len(spec.chunk_valid_tokens))
        for spec in KDA_CASES.values()
        if spec.inputs == "text"
    }
    assert len({text_input_cache_path(*key) for key in text_inputs}) == len(text_inputs)


def test_concurrent_producers_in_separate_processes_publish_one_correct_entry(tmp_path: Path) -> None:
    entry, log = tmp_path / "oracle.bin", tmp_path / "produce_calls.log"
    start_at = time.time() + 3.0  # both processes enter publish_once together, after interpreter start-up
    producers = [_producer(entry, log, start_at) for _ in range(2)]
    results = [producer.communicate(timeout=120) for producer in producers]

    assert all(producer.returncode == 0 for producer in producers), results
    outputs = sorted(stdout.strip().splitlines()[-1] for stdout, _ in results)
    assert outputs == ["produced=False value_ok=True", "produced=True value_ok=True"], results
    assert len(log.read_text().splitlines()) == 1  # produce() ran once
    assert entry.read_bytes() == _PAYLOAD
    assert [file.name for file in _leftovers(tmp_path, entry) if file != log] == []


def test_partially_written_entry_is_invisible_and_a_killed_producer_releases_the_entry(tmp_path: Path) -> None:
    entry, log = tmp_path / "oracle.bin", tmp_path / "produce_calls.log"
    producer = _producer(entry, log, time.time(), mode="partial")
    try:
        assert producer.stdout.readline().strip() == "partial", producer.stderr.read()
        # Mid-write: the bytes exist only under a hidden temporary name; readers see no entry.
        temporaries = [file for file in _leftovers(tmp_path, entry) if file != log]
        assert len(temporaries) == 1 and temporaries[0].name.startswith(f".{entry.name}.")
        assert temporaries[0].stat().st_size == len(_PAYLOAD) // 2
        assert not entry.exists()
    finally:
        producer.send_signal(signal.SIGKILL)
        producer.wait(timeout=30)

    # The dead producer's lock is released by the kernel; the next producer publishes the complete entry.
    value, produced = publish_once(entry, lambda: _PAYLOAD, lambda data, file: file.write_bytes(data), Path.read_bytes)
    assert produced and value == _PAYLOAD and entry.read_bytes() == _PAYLOAD
    value, produced = publish_once(entry, lambda: pytest.fail("must not recompute"), None, Path.read_bytes)
    assert not produced and value == _PAYLOAD


def test_failed_producer_publishes_nothing(tmp_path: Path, expect_error) -> None:
    entry = tmp_path / "oracle.bin"

    def failing_write(data: bytes, file: Path) -> None:
        file.write_bytes(data[:10])
        raise RuntimeError("serialization failed")

    with expect_error(RuntimeError, "serialization failed"):
        publish_once(entry, lambda: _PAYLOAD, failing_write, Path.read_bytes)
    assert not entry.exists() and _leftovers(tmp_path, entry) == []
