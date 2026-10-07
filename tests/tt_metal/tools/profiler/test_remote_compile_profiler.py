# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Requires a profiler-enabled build, jit_compile_server, and one reserved device."""

import csv
import os
from pathlib import Path
import re
import socket
import subprocess
import time

import pytest


def _zone_records(root, profiler_dir):
    kernel = root / "tests/tt_metal/tools/profiler/kernels/remote_compile_zones.cpp"
    expected = {
        (match.group(1), str(kernel), str(number))
        for number, line in enumerate(kernel.read_text().splitlines(), 1)
        if (match := re.search(r'DeviceZoneScopedN\("([^" ]+)"\)', line))
    }
    assert len(expected) == 2
    logs = profiler_dir / ".logs"
    locations = "\n".join(path.read_text() for path in logs.glob("*zone_src_locations.log"))
    for name, source, line in expected:
        assert f"{name},{source},{line},KERNEL_PROFILER" in locations, f"Missing source metadata: {name}"
    with (logs / "profile_log_device.csv").open() as handle:
        next(handle)  # Architecture/frequency header precedes the CSV header.
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    actual = {
        (row["zone name"], row["source file"], row["source line"], row["type"])
        for row in rows
        if row["zone name"].startswith("REMOTE-COMPILE-")
    }
    assert actual == {(*record, event) for record in expected for event in ("ZONE_START", "ZONE_END")}
    return actual


@pytest.mark.parametrize("profiling", [True, False], ids=["profiling", "no-profiling"])
@pytest.mark.parametrize("preprocess", [False, True], ids=["server-cache", "client-cache"])
def test_remote_compile_preserves_profiler_zones(tmp_path, preprocess, profiling):
    root = Path(os.environ.get("TT_METAL_HOME", Path(__file__).resolve().parents[4]))
    server_bin = root / "build/tools/jit_compile_server"
    client_bin = root / "build/test/tt_metal/tools/profiler/test_remote_compile_zones"
    assert server_bin.is_file(), "Build jit_compile_server first"
    assert client_bin.is_file(), "Build profiler_test_remote_compile_zones first"

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        endpoint = f"127.0.0.1:{listener.getsockname()[1]}"

    env = os.environ.copy()
    env.pop("TT_METAL_JIT_SERVER_ENDPOINTS", None)
    env.pop("TT_METAL_JIT_PREPROCESS", None)
    env.update(
        TT_METAL_HOME=str(root),
        TT_METAL_JIT_SERVER_ENDPOINT=endpoint,
        TT_METAL_JIT_SERVER_CACHE_ROOT=str(tmp_path / "server-cache"),
        TT_METAL_CACHE=str(tmp_path / "client-cache"),
        TT_METAL_JIT_SERVER_ENABLE="1",
    )
    env.pop("TT_METAL_DEVICE_PROFILER", None)
    if profiling:
        env["TT_METAL_DEVICE_PROFILER"] = "1"
    if preprocess:
        env["TT_METAL_JIT_PREPROCESS"] = "1"

    reference = None
    if profiling:
        local_env = env.copy()
        local_env.pop("TT_METAL_JIT_SERVER_ENABLE")
        local_env.pop("TT_METAL_JIT_SERVER_ENDPOINT")
        local_env.pop("TT_METAL_JIT_PREPROCESS", None)
        local_env["TT_METAL_CACHE"] = str(tmp_path / "local-cache")
        local_env["TT_METAL_PROFILER_DIR"] = str(tmp_path / "local")
        with (tmp_path / "local.log").open("w") as output:
            subprocess.run(
                [str(client_bin)],
                cwd=root,
                env=local_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
        reference = _zone_records(root, tmp_path / "local")

    server_log = tmp_path / "server.log"
    with server_log.open("w") as output:
        server = subprocess.Popen([str(server_bin)], cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while "JIT compile server listening" not in server_log.read_text():
                assert server.poll() is None, server_log.read_text()
                assert time.monotonic() < deadline, "JIT server startup timed out"
                time.sleep(0.1)

            phases = ("cold", "warm", "missing-metadata") if preprocess and profiling else ("cold", "warm")
            for phase in phases:
                if phase == "missing-metadata":
                    # Simulate upgrading a client cache written before profiler diagnostics were transported.
                    # Keep the valid ELF and full-dephash files: only metadata absence should force the RPC.
                    cache = tmp_path / "client-cache"
                    metadata_files = list(cache.rglob("remote_profiler.o.log"))
                    assert metadata_files and list(cache.rglob("*.elf"))
                    for path in metadata_files:
                        path.unlink()
                # Each process starts with empty profiler logs, but shares the compilation caches.
                # This prevents locally compiled firmware or a previous run's superset log masking missing metadata.
                profiler_dir = tmp_path / phase
                env["TT_METAL_PROFILER_DIR"] = str(profiler_dir)
                with (tmp_path / f"{phase}.log").open("w") as client_log:
                    subprocess.run(
                        [str(client_bin)],
                        cwd=root,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=300,
                    )
                if profiling:
                    assert _zone_records(root, profiler_dir) == reference, phase
                else:
                    assert not (profiler_dir / ".logs/profile_log_device.csv").exists()
                    metadata = list((tmp_path / "client-cache").rglob("remote_profiler.o.log"))
                    assert all(not path.read_text() for path in metadata), "Unexpected profiler diagnostics"

                compiled_requests = server_log.read_text().count("compile remote_compile_zones/")
                if phase == "cold":
                    assert compiled_requests > 0, "No remote kernel compilations occurred"
                    cold_requests = compiled_requests
                elif phase == "missing-metadata":
                    assert compiled_requests > cold_requests, "Missing profiler metadata did not refresh the ELF cache"
                elif preprocess:
                    assert compiled_requests == cold_requests, "Expected warmed client ELF-cache reuse"
                else:
                    assert compiled_requests > cold_requests, "Expected requests to warmed server cache"
                    assert "miss=0 link=no" in server_log.read_text(), "No server object-cache hit occurred"
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
