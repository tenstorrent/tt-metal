# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-d-gen's real engine as the runner tests' scheduler: find the build, build the driver command, end the runner.

The runner tests drive tt-metal's prefill runner with tt-d-gen's own engine (dgen_prefill_driver.py) when a built
tt-d-gen is found, and fall back to tt-metal's prefill_producer only when none is. ttnn (this tree, its own Python)
and tt_engine (tt-d-gen, Python 3.12) cannot share a process, so the driver is a subprocess with tt-d-gen's
environment, as in serving. The two sides share only the runner's H2D stream service (/dev/shm descriptor + sockets),
the layer-ack counter channel /tt_prefill_layer_acks_<service_id> (the runner's router owns it) and the 12-byte
header {slot, start, end}.

Finding the build (find_build):
  repo     BRINGUP_SERVER_REPO, else spec serving.server_repo, else /localdev/$USER/tt-d-gen
  module   <repo>/bindings/python/tt_engine/_tt_engine*.so (./build_dgen.sh --bindings --blaze)
  python   BRINGUP_DGEN_PYTHON, else the first of <repo>-build/venv312/bin/python, <repo>/.venv/bin/python,
           <repo>/build-bindings/venv/bin/python that imports tt_engine with the environment below
  env      ours minus PYTHONPATH / VIRTUAL_ENV / TT_METAL_HOME / TT_METAL_RUNTIME_ROOT / LD_LIBRARY_PATH, plus
           TT_METAL_HOME = TT_METAL_RUNTIME_ROOT = <repo>/third_party/tt-blaze/tt-metal, PYTHONPATH =
           <repo>/bindings/python
BRINGUP_DGEN=0 forces the producer fallback; BRINGUP_DGEN=1 makes a missing build a failure instead of a fallback.

The engine never sends the runner's shutdown sentinel (metadata -1, -1, -1; prefill_runner.py _is_shutdown_sentinel):
after the driver exits 0, `python -m models.demos.common.bringup.testing.dgen_engine shutdown <service_id>` (this
tree's Python) connects to the same H2D service as the next connector (the socket state lives in SHM,
hd_socket_connector_state.hpp) and sends it, as prefill_producer does with PREFILL_SEND_SHUTDOWN=1.
"""

from __future__ import annotations

import getpass
import glob
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

DRIVER = Path(__file__).resolve().with_name("dgen_prefill_driver.py")
_DROP = (
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "VIRTUAL_ENV_PROMPT",
    "TT_METAL_HOME",
    "TT_METAL_RUNTIME_ROOT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHON_ENV_DIR",
)


@dataclass
class Build:
    repo: Path
    python: str
    env_set: dict = field(default_factory=dict)  # variables to set for the driver
    sha: str = ""

    def env(self, base: dict | None = None) -> dict:
        e = {k: v for k, v in (base if base is not None else os.environ).items() if k not in _DROP}
        e.update(self.env_set)
        return e

    def argv(self, *args: str) -> list[str]:
        """The driver command; `env -u ... K=V` so it can run inside a shell whose environment is ours."""
        cmd = ["env"]
        for k in _DROP:
            cmd += ["-u", k]
        cmd += [f"{k}={v}" for k, v in self.env_set.items()]
        return cmd + [self.python, str(DRIVER), *args]

    def describe(self) -> str:
        return f"tt-d-gen engine ({self.repo} @ {self.sha[:11]}, {self.python})"


def server_repo(spec=None) -> Path:
    r = os.environ.get("BRINGUP_SERVER_REPO")
    if not r and spec is not None:
        try:
            r = spec.get("serving.server_repo")
        except Exception:
            r = None
    return Path(r or f"/localdev/{getpass.getuser()}/tt-d-gen")


def find_build(spec=None) -> tuple[Build | None, str]:
    """(build, "") when a built tt_engine imports, else (None, why not)."""
    if os.environ.get("BRINGUP_DGEN") == "0":
        return None, "BRINGUP_DGEN=0"
    repo = server_repo(spec)
    if not repo.is_dir():
        return None, f"no tt-d-gen checkout at {repo}"
    so = glob.glob(str(repo / "bindings/python/tt_engine/_tt_engine*.so"))
    if not so:
        return None, f"tt_engine not built in {repo} (./build_dgen.sh --bindings --blaze)"
    metal = repo / "third_party/tt-blaze/tt-metal"
    env_set = {"PYTHONPATH": str(repo / "bindings/python")}
    if metal.is_dir():
        env_set.update(TT_METAL_HOME=str(metal), TT_METAL_RUNTIME_ROOT=str(metal))
    cands = [os.environ.get("BRINGUP_DGEN_PYTHON", "")] + [
        str(repo.parent / f"{repo.name}-build/venv312/bin/python"),
        str(repo / ".venv/bin/python"),
        str(repo / "build-bindings/venv/bin/python"),
    ]
    tried = []
    for py in [c for c in cands if c]:
        if not os.access(py, os.X_OK):
            continue
        b = Build(repo, py, env_set)
        r = subprocess.run(
            b.argv()[:-1] + ["-c", "import tt_engine as te; te.device_prefill_pipeline"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode == 0:
            s = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
            b.sha = s.stdout.strip()
            return b, ""
        tried.append(f"{py}: {(r.stderr or r.stdout).strip().splitlines()[-1:] or ['exit ' + str(r.returncode)]}")
    return None, f"no Python that imports tt_engine from {repo} (set BRINGUP_DGEN_PYTHON); tried {tried}"


def driver_shell(build: Build, plan: str, out: str, rc_file: str, service_id: str) -> list[str]:
    """sh command: the engine driver, then (only when it exits 0) the shutdown sentinel from this tree's Python.
    <rc_file> gets the driver's exit code, or the sender's when the driver passed, so a watcher sees either fail."""
    drv = " ".join(shlex.quote(a) for a in build.argv(plan, out))
    snd = " ".join(
        shlex.quote(a)
        for a in (sys.executable, "-m", "models.demos.common.bringup.testing.dgen_engine", "shutdown", service_id)
    )
    q = shlex.quote(rc_file)
    return ["/bin/sh", "-c", f"{drv}; rc=$?; if [ $rc -eq 0 ]; then {snd}; rc=$?; fi; echo $rc > {q}"]


def send_shutdown(service_id: str, timeout_s: float = 60.0) -> None:
    """The runner's shutdown sentinel: one chunk of 1s with metadata (-1, -1, -1), then a barrier."""
    import struct

    import numpy as np

    import ttnn

    svc = ttnn.H2DStreamService.connect(service_id, timeout_ms=int(timeout_s * 1000))
    nbytes = svc.payload_size_bytes()
    sp = int(os.environ.get("PREFILL_SP", "1"))
    payload = np.ones(nbytes // 4, dtype=np.uint32).reshape(sp, 1, -1)
    meta = struct.pack("<3i", -1, -1, -1)
    svc.forward_to_tensor_bytes(payload, metadata=meta)
    svc.barrier()
    print(f"[dgen shutdown] sentinel (-1, -1, -1) sent to {service_id!r}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "shutdown":
        send_shutdown(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "find":
        b, why = find_build()
        print(b.describe() if b else f"not found: {why}")
    else:
        sys.exit("usage: python -m models.demos.common.bringup.testing.dgen_engine shutdown <service_id> | find")
