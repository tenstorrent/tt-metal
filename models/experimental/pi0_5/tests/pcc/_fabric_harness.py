# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Reset/retry harness for FABRIC_1D ethernet-core stalls (plan iter-3 §9, SC5).

The 4-chip Blackhole box exhibits recurring "active ethernet core timed out" stalls under
repeated FABRIC_1D opens. This harness (test-tree ONLY -- zero tt_pipeline/ change, zero
tt_bh_glx/ change) wraps mesh opens with a tt-smi reset + retry, and pairs every
FABRIC_1D-on-open with a DISABLED-on-close in a finally (a leaked FABRIC_1D is the proximate
cause of the next open's timeout).

Fabric hygiene (see harness docstrings):
  (a) ALWAYS pair set_fabric_config(FABRIC_1D) with ...(DISABLED) in a finally.
  (b) ALWAYS Pipeline.release_all() / stage.close() before close_mesh_device.
  (c) tt-smi -r + settle between distinct multi-chip invocations.
  (d) NEVER run the multi-chip path under tracy.

ZERO tt_symbiote imports.
"""
from __future__ import annotations

import subprocess
import time

import pytest
import ttnn

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

_TRACE_REGION = 134_217_728


def reset_board(settle: float = 5.0):
    """tt-smi -r and let the fabric settle. Clean-skips if tt-smi is unavailable."""
    try:
        subprocess.run(["tt-smi", "-r"], check=False, timeout=180)
    except FileNotFoundError:
        # Fall back to the .tenstorrent-venv tt-smi if not on PATH.
        try:
            subprocess.run(["/home/ttuser/.tenstorrent-venv/bin/tt-smi", "-r"], check=False, timeout=180)
        except FileNotFoundError:
            pytest.skip("tt-smi not available")
    time.sleep(settle)


def open_parent_with_retry(n, *, retries=2, l1_small_size=24576, trace_region_size=_TRACE_REGION):
    """Open an (1,n) FABRIC_1D parent mesh; reset+retry on ethernet-core stalls."""
    last = None
    for _ in range(retries + 1):
        if ttnn.get_num_devices() < n:
            pytest.skip(f"need >={n} chips, have {ttnn.get_num_devices()}")
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, n),
                l1_small_size=l1_small_size,
                trace_region_size=trace_region_size,
            )
        except Exception as e:  # noqa: BLE001  match "active ethernet core"/"timed out"/"FABRIC"
            last = e
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            reset_board()
    raise last


def close_parent(parent):
    """Close the parent mesh and ALWAYS drop the fabric config (hygiene (a))."""
    try:
        ttnn.close_mesh_device(parent)
    finally:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
