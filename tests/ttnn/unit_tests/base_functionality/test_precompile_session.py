# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# ttnn.graph.CompileSession's build sink, on a mock device. Each case runs in a subprocess of its own: the mock
# cluster descriptor and the kernel cache are fixed when a process first opens a device. No hardware is needed.

import glob
import json
import os
import socket
import subprocess
import sys
import textwrap
import time

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
_DESCRIPTOR = os.path.join(
    _REPO, "tt_metal", "third_party", "umd", "tests", "cluster_descriptor_examples", "wormhole_N150.yaml"
)

_PRELUDE = """
import json, torch, ttnn

def chain(device, shape=(1, 1, 64, 128)):
    a = ttnn.from_torch(torch.rand(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.mul(ttnn.add(ttnn.gelu(a), a), a)

def stats(session):
    s = session.stats()
    return {k: getattr(s, k) for k in dir(s) if not k.startswith("_")}
"""


def _run(cache, body, **extra_env):
    env = dict(os.environ, TT_METAL_MOCK_CLUSTER_DESC_PATH=_DESCRIPTOR, TT_METAL_CACHE=str(cache))
    for name in ("TT_METAL_JIT_SERVER_ENABLE", "TT_METAL_JIT_PREPROCESS", "TT_METAL_FORCE_JIT_COMPILE"):
        env.pop(name, None)
    env.update(extra_env)
    script = _PRELUDE + textwrap.dedent(body)
    proc = subprocess.run(
        [sys.executable, "-c", script], cwd=_REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.returncode == 0, proc.stdout[-4000:]
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stdout[-4000:]
    return json.loads(lines[-1][len("RESULT ") :])


def _kernel_binaries(cache):
    return {p for depth in ("", "*", "*/*") for p in glob.glob(os.path.join(str(cache), depth, "kernels", "*", "*"))}


def _object_files(cache):
    return [p for d in _kernel_binaries(cache) for p in glob.glob(os.path.join(d, "**", "*.o"), recursive=True)]


def _ordinary_run(cache, **extra_env):
    _run(
        cache,
        """
        device = ttnn.open_device(device_id=0)
        chain(device)
        ttnn.close_device(device)
        print("RESULT {}")
        """,
        **extra_env,
    )


@pytest.fixture
def jit_server(tmp_path_factory):
    """A JIT compile server on this host, on a free local port; yields its endpoint."""
    binary = os.path.join(_REPO, "build", "tools", "jit_compile_server")
    if not os.path.exists(binary):
        pytest.skip(f"no JIT compile server at {binary}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    endpoint = f"127.0.0.1:{port}"
    log = open(tmp_path_factory.mktemp("jit_server") / "server.log", "w")
    env = dict(
        os.environ,
        TT_METAL_JIT_SERVER_ENDPOINT=endpoint,
        TT_METAL_JIT_SERVER_CACHE_ROOT=str(tmp_path_factory.mktemp("jit_server_cache")),
    )
    server = subprocess.Popen([binary], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert server.poll() is None, f"JIT compile server exited with {server.returncode}"
                assert time.monotonic() < deadline, "JIT compile server did not start listening"
                time.sleep(0.2)
        yield endpoint
    finally:
        server.terminate()
        server.wait(timeout=30)
        log.close()


@pytest.fixture(autouse=True)
def _descriptor():
    if not os.path.exists(_DESCRIPTOR):
        pytest.skip(f"no mock cluster descriptor at {_DESCRIPTOR}")


def test_sink_builds_what_a_normal_compile_builds(tmp_path):
    # The sink queues every kernel of the programs a mock run compiles, and the workers build exactly the binaries
    # a normal compile of the same run builds: a second, ordinary run finds them all and builds nothing new.
    result = _run(
        tmp_path,
        """
        device = ttnn.open_device(device_id=0)
        session = ttnn.graph.CompileSession(workers=2)
        session.set_build_sink(True)
        chain(device)
        session.set_build_sink(False)
        session.wait()
        session.close()
        ttnn.close_device(device)
        print("RESULT " + json.dumps(stats(session)))
        """,
    )
    assert result["programs_added"] == 3
    assert result["kernels_queued"] > 0
    assert result["kernels_failed"] == 0
    assert result["kernels_built"] + result["kernels_preempted"] == result["kernels_queued"]
    # A program whose kernels were all queued by earlier programs adds nothing to build.
    assert 1 <= result["programs_built"] <= result["programs_added"]
    built = _kernel_binaries(tmp_path)
    assert len(built) == result["kernels_queued"]

    _ordinary_run(tmp_path)
    assert _kernel_binaries(tmp_path) == built


def test_builds_survive_device_close_and_reopen(tmp_path):
    # Workers keep building after the device that queued the kernels is closed and while it is open again.
    result = _run(
        tmp_path,
        """
        session = ttnn.graph.CompileSession(workers=1)
        device = ttnn.open_device(device_id=0)
        session.set_build_sink(True)
        chain(device, (1, 1, 64, 64))
        session.set_build_sink(False)
        ttnn.close_device(device)
        device = ttnn.open_device(device_id=0)
        session.wait()
        session.close()
        ttnn.close_device(device)
        print("RESULT " + json.dumps(stats(session)))
        """,
    )
    assert result["kernels_queued"] > 0
    assert result["kernels_failed"] == 0
    assert result["kernels_built"] + result["kernels_preempted"] == result["kernels_queued"]


def test_close_drops_queued_kernels(tmp_path):
    result = _run(
        tmp_path,
        """
        device = ttnn.open_device(device_id=0)
        session = ttnn.graph.CompileSession(workers=1)
        session.set_build_sink(True)
        a = ttnn.from_torch(torch.rand((1, 1, 32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
        for op in (ttnn.cos, ttnn.sin, ttnn.tan, ttnn.tanh, ttnn.erf, ttnn.erfc):
            op(a)
        session.set_build_sink(False)
        session.close()
        ttnn.close_device(device)
        print("RESULT " + json.dumps(stats(session)))
        """,
    )
    assert result["kernels_running"] == 0 and result["kernels_pending"] == 0
    accounted = (
        result["kernels_built"] + result["kernels_preempted"] + result["kernels_failed"] + result["kernels_dropped"]
    )
    assert accounted == result["kernels_queued"]


def test_sink_builds_through_the_jit_server(tmp_path, jit_server):
    # The session sends each program's kernels to the compile server as one batch. With
    # TT_METAL_JIT_PREPROCESS it leaves the reuse sidecars, so an ordinary local run reuses every server-built ELF:
    # it compiles nothing, so no object file appears.
    result = _run(
        tmp_path,
        f"""
        device = ttnn.open_device(device_id=0)
        session = ttnn.graph.CompileSession(
            workers=2, remote_endpoints=["{jit_server}"], remote_min_kernels=0
        )
        session.set_build_sink(True)
        chain(device)
        session.set_build_sink(False)
        session.wait()
        session.close()
        ttnn.close_device(device)
        print("RESULT " + json.dumps(stats(session)))
        """,
        TT_METAL_JIT_PREPROCESS="1",
    )
    assert result["kernels_queued"] > 0
    assert result["kernels_failed"] == 0 and result["kernels_remote_fallback"] == 0
    assert result["kernels_remote"] == result["kernels_built"] == result["kernels_queued"]
    built = _kernel_binaries(tmp_path)
    assert len(built) == result["kernels_queued"]
    assert _object_files(tmp_path) == []

    _ordinary_run(tmp_path)
    assert _kernel_binaries(tmp_path) == built
    assert _object_files(tmp_path) == []


def test_unreachable_jit_server_falls_back_to_local_builds(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        refused = f"127.0.0.1:{s.getsockname()[1]}"  # closed again once the block ends: connections are refused
    result = _run(
        tmp_path,
        f"""
        device = ttnn.open_device(device_id=0)
        session = ttnn.graph.CompileSession(
            workers=1, remote_endpoints=["{refused}"], remote_min_kernels=0
        )
        session.set_build_sink(True)
        chain(device)
        session.set_build_sink(False)
        session.wait()
        session.close()
        ttnn.close_device(device)
        print("RESULT " + json.dumps(stats(session)))
        """,
    )
    assert result["kernels_queued"] > 0
    assert result["kernels_failed"] == 0 and result["kernels_remote"] == 0
    assert result["kernels_remote_fallback"] >= 1
    assert result["kernels_built"] + result["kernels_preempted"] == result["kernels_queued"]
    assert len(_kernel_binaries(tmp_path)) == result["kernels_queued"]


def test_ordinary_compile_through_the_jit_server(tmp_path, jit_server):
    # The runtime's own remote compile (TT_METAL_JIT_SERVER_ENABLE) goes through the same build path as the session:
    # the run builds every kernel on the server, and loads them, and builds the same binaries as a local run.
    _ordinary_run(
        tmp_path / "remote",
        TT_METAL_JIT_SERVER_ENABLE="1",
        TT_METAL_JIT_SERVER_ENDPOINTS=jit_server,
        TT_METAL_JIT_PREPROCESS="1",
    )
    _ordinary_run(tmp_path / "local")
    remote = {os.path.relpath(p, tmp_path / "remote") for p in _kernel_binaries(tmp_path / "remote")}
    local = {os.path.relpath(p, tmp_path / "local") for p in _kernel_binaries(tmp_path / "local")}
    assert remote and remote == local
    assert _object_files(tmp_path / "remote") == []
