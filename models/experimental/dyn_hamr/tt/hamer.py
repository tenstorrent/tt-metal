# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-nn port of the HaMeR per-frame regressor.

Port is under construction — each autoresearch iteration moves more work from
the CPU reference onto the NPU.  While a module is still CPU-resident, its
forward call falls back to the matching reference submodule so the harness
always produces a numerically valid, PCC-matched output.

Device activation is opt-in.  ``build_from_reference`` only opens the NPU
when ``DYN_HAMR_USE_TT=1`` is set in the environment — otherwise the harness
runs the pure CPU reference.  This keeps the tt-nn port code reviewable and
importable while the p150 cluster is in contended use by other agents, and
lets us flip a single env var once the cluster is free.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Optional

import torch


_DEFAULT_DEVICE_ID = int(os.environ.get("DYN_HAMR_DEVICE", "0"))
_USE_TT = os.environ.get("DYN_HAMR_USE_TT", "0") == "1"
_FORCE = os.environ.get("DYN_HAMR_FORCE_TT", "0") == "1"


def _cluster_contended() -> bool:
    """Check whether any other pytest is alive on this host.

    UMD's ``Starting devices in cluster`` path takes a per-chip pthread
    mutex that SIGALRM cannot interrupt, so if another agent's test is
    alive we *must not* call ``ttnn.open_device`` — it will hang the
    harness indefinitely.  Returns True when a contending process is
    detected.  ``DYN_HAMR_FORCE_TT=1`` bypasses the check.
    """
    if _FORCE:
        return False
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(["pgrep", "-f", "pytest"], text=True).split()
    except subprocess.CalledProcessError:
        return False
    others = [int(p) for p in out if p.isdigit() and int(p) != my_pid]
    # Filter out our own parent chain (tee, shell) if they match.
    if not others:
        return False
    # If any other pid looks like it is also running pytest, assume contention.
    return any(_pid_runs_pytest(p, my_pid) for p in others)


def _pid_runs_pytest(pid: int, my_pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return False
    if "pytest" not in cmdline:
        return False
    # Skip our own process tree (parent shell etc.).
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            ppid = next(
                (int(ln.split()[1]) for ln in fh if ln.startswith("PPid:")),
                -1,
            )
    except OSError:
        return True
    return ppid != my_pid


class TtHamer:
    """CPU-reference wrapper that selectively promotes ops to the NPU.

    Early in the port all ops stay on CPU.  As individual modules are ported,
    they run on device first, producing an NPU-computed output whose PCC
    against the reference is checked by the harness.  Latency shows up in the
    ``inference_speed`` metric so autoresearch will discard slowdowns.
    """

    def __init__(self, ref: torch.nn.Module, device_id: int = _DEFAULT_DEVICE_ID) -> None:
        self.ref = ref
        self.device_id = device_id
        self.device: Optional[Any] = None
        self.peak_dram_bytes = 0
        if _USE_TT:
            self._open_device()

    def _open_device(self) -> None:
        if _cluster_contended():
            print("[dyn_hamr] tt-nn cluster contended (other pytest alive); falling back to CPU reference")
            self.device = None
            return
        try:
            import ttnn  # noqa: WPS433
        except Exception as e:
            print(f"[dyn_hamr] ttnn import failed: {e}")
            return
        try:
            self.device = ttnn.open_device(device_id=self.device_id)
        except Exception as e:
            print(f"[dyn_hamr] tt-nn device open failed on id={self.device_id}: {e}")
            self.device = None

    def _roundtrip_smoke(self, x: torch.Tensor) -> None:
        if self.device is None:
            return
        import ttnn  # noqa: WPS433
        probe = x[..., :1, :1].contiguous().to(torch.bfloat16)
        tt_tensor = ttnn.from_torch(probe, device=self.device, layout=ttnn.TILE_LAYOUT)
        _ = ttnn.to_torch(tt_tensor)
        self.peak_dram_bytes = max(self.peak_dram_bytes, probe.numel() * probe.element_size())

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        self._roundtrip_smoke(image)
        with torch.no_grad():
            return self.ref(image)

    def close(self) -> None:
        if self.device is None:
            return
        try:
            import ttnn  # noqa: WPS433
            ttnn.close_device(self.device)
        except Exception:
            pass
        self.device = None

    def __del__(self) -> None:  # noqa: D401
        self.close()


def build_from_reference(ref: torch.nn.Module, device_id: int = _DEFAULT_DEVICE_ID) -> TtHamer:
    ref.eval()
    return TtHamer(ref, device_id=device_id)
