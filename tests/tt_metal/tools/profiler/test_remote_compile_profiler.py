# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Requires a profiler-enabled build, jit_compile_server, and one reserved device."""

import os
from pathlib import Path
import socket
import subprocess
import time

import pytest


@pytest.mark.parametrize("preprocess", [False, True], ids=["server-cache", "client-cache"])
def test_remote_compile_preserves_profiler_zones(tmp_path, preprocess):
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
        TT_METAL_DEVICE_PROFILER="1",
        TT_METAL_JIT_SERVER_ENABLE="1",
    )
    if preprocess:
        env["TT_METAL_JIT_PREPROCESS"] = "1"

    server_log = tmp_path / "server.log"
    with server_log.open("w") as output:
        server = subprocess.Popen([str(server_bin)], cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while "JIT compile server listening" not in server_log.read_text():
                assert server.poll() is None, server_log.read_text()
                assert time.monotonic() < deadline, "JIT server startup timed out"
                time.sleep(0.1)

            phases = ("cold", "warm", "missing-metadata") if preprocess else ("cold", "warm")
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
                logs = profiler_dir / ".logs"
                locations = "\n".join(path.read_text() for path in logs.glob("*zone_src_locations.log"))
                csv = (logs / "profile_log_device.csv").read_text()
                for name in ("REMOTE-COMPILE-INNER", "REMOTE-COMPILE-OUTER"):
                    assert name in locations, f"{phase}: missing source metadata for {name}"
                    assert name in csv, f"{phase}: actual device markers did not resolve to {name}"
                assert "remote_compile_zones.cpp" in locations

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
