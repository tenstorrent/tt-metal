# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# ttnn.graph.CompileSession's build sink, on a mock device. Each case runs in a subprocess of its own: the mock
# cluster descriptor and the kernel cache are fixed when a process first opens a device. No hardware is needed.

import glob
import json
import os
import subprocess
import sys
import textwrap

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


def _run(cache, body):
    env = dict(os.environ, TT_METAL_MOCK_CLUSTER_DESC_PATH=_DESCRIPTOR, TT_METAL_CACHE=str(cache))
    for name in ("TT_METAL_JIT_SERVER_ENABLE", "TT_METAL_FORCE_JIT_COMPILE"):
        env.pop(name, None)
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
    built = _kernel_binaries(tmp_path)
    assert len(built) == result["kernels_queued"]

    _run(
        tmp_path,
        """
        device = ttnn.open_device(device_id=0)
        chain(device)
        ttnn.close_device(device)
        print("RESULT {}")
        """,
    )
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
