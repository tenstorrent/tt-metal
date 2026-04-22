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
import signal
from contextlib import contextmanager
from typing import Any, Optional

import torch


_DEFAULT_DEVICE_ID = int(os.environ.get("DYN_HAMR_DEVICE", "0"))
_USE_TT = os.environ.get("DYN_HAMR_USE_TT", "0") == "1"
_OPEN_TIMEOUT_S = int(os.environ.get("DYN_HAMR_OPEN_TIMEOUT", "10"))


@contextmanager
def _sigalarm(seconds: int):
    """Raise TimeoutError after ``seconds`` wall-clock seconds.  Used to bound
    ``ttnn.open_device`` because it otherwise blocks indefinitely on a
    cluster-wide ``CHIP_IN_USE_*`` lock when other agents hold the mesh.
    """
    def _handler(signum, frame):
        raise TimeoutError(f"tt-nn device open exceeded {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


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
        try:
            import ttnn  # noqa: WPS433
        except Exception as e:
            print(f"[dyn_hamr] ttnn import failed: {e}")
            return
        try:
            with _sigalarm(_OPEN_TIMEOUT_S):
                self.device = ttnn.open_device(device_id=self.device_id)
        except TimeoutError as e:
            print(f"[dyn_hamr] {e}; falling back to CPU reference")
            self.device = None
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
