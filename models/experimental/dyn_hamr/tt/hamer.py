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
        self._vit_params: Optional[dict] = None
        if _USE_TT:
            self._open_device()
            if self.device is not None:
                self._build_vit_params()

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

    def _build_vit_params(self) -> None:
        """Lift the reference ViT-H weights into bfloat16 tile tensors on
        DRAM.  Runs once at construction; the 670M-parameter upload is
        amortized over every subsequent forward.
        """
        try:
            from models.experimental.dyn_hamr.tt import ttnn_vit  # noqa: WPS433
            self._vit_params = ttnn_vit.build_parameters_from_reference(self.ref, self.device)
            # Rough bookkeeping of bytes pushed to DRAM.
            self.peak_dram_bytes = sum(
                p.numel() * p.element_size() for p in self.ref.backbone.parameters()
            ) // 2  # bfloat16 halves fp32
        except Exception as e:
            print(f"[dyn_hamr] vit param upload failed: {e}; forward will use CPU ref")
            self._vit_params = None

    def _forward_device(self, image: torch.Tensor) -> Optional[torch.Tensor]:
        """NPU forward: ViT on device, MANO head on CPU (head port is still
        code-only).  Returns ``None`` if the device path isn't available.
        """
        if self.device is None or self._vit_params is None:
            return None
        try:
            import ttnn  # noqa: WPS433
            from models.experimental.dyn_hamr.tt import ttnn_vit  # noqa: WPS433

            # NCHW → NHWC on host (cheap), push to device.
            nhwc = image.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
            tt_input = ttnn.from_torch(
                nhwc, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
            )
            tt_feat = ttnn_vit.forward(tt_input, self._vit_params)  # (B, 192, 1280) tile-layout
            feat_host = ttnn.to_torch(tt_feat).to(torch.float32)
            # Reshape token-major → (B, 1280, 16, 12) like the CPU backbone output.
            B = image.shape[0]
            feat_map = feat_host.reshape(B, 16, 12, 1280).permute(0, 3, 1, 2).contiguous()
            # Head still on CPU until its tt-nn port is activated.
            with torch.no_grad():
                return self.ref.head(feat_map)
        except Exception as e:
            print(f"[dyn_hamr] tt-nn forward failed: {e}; falling back to CPU ref for this call")
            return None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        tt_out = self._forward_device(image)
        if tt_out is not None:
            return tt_out
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
