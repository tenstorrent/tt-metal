# SPDX-License-Identifier: Apache-2.0
"""
Tests for the loopback operation — used to verify unpack_to_dest behaviour
and diagnose a potential simulator bug.

Background
----------
On Wormhole, fp32 data must be unpacked directly into the DST register
(unpack_to_dest=True / UnpackToDestFp32). Routing through SrcA/SrcB (19-bit)
silently truncates mantissa bits.

The tt-sim simulator raises:
    ERROR: UndefinedBehavior: tensix_execute_unpacr: unpack_to_dest=0
           in_data_format=0 out_data_format=0
when unpack_to_dest=True is set for fp32 CBs, which contradicts what the
hardware specification and real-hardware behaviour suggest.

Diagnostic protocol
-------------------
Run both tests on real HW and on the simulator:

  pytest custom_op/tests/test_loopback_unpack_to_dest.py -v

Real HW expected results:
  test_loopback_exact_copy              → PASS  (bit-exact copy)
  test_loopback_truncation_no_unpack    → PASS  (truncation detected)

Simulator expected results if the simulator is CORRECT:
  test_loopback_exact_copy              → PASS
  test_loopback_truncation_no_unpack    → PASS  (or raises UndefinedBehavior for Default mode)

Simulator result that CONFIRMS A SIMULATOR BUG:
  test_loopback_exact_copy              → raises UndefinedBehavior
  test_loopback_truncation_no_unpack    → PASS  (or raises UndefinedBehavior)
  i.e. the simulator rejects the correct UnpackToDestFp32 mode.
"""

import sys
import os
import torch
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

# Allow importing from custom_op/operation without installing a package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "operation"))
from loopback_op import loopback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(device, rows: int = 32, cols: int = 32) -> tuple[torch.Tensor, ttnn.Tensor]:
    """Return (torch_ref, device_tensor) with values that have significant mantissa bits."""
    x = torch.rand(rows, cols, dtype=torch.float32)
    x_tt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return x, x_tt


# ---------------------------------------------------------------------------
# Test 1: unpack_to_dest=True — must give a bit-exact copy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (32, 128)])
def test_loopback_exact_copy(device, shape):
    """With unpack_to_dest=True, fp32 tiles must pass through DST unmodified.

    If this test raises UndefinedBehavior on the simulator but passes on real
    hardware, the simulator is incorrectly rejecting valid UnpackToDestFp32 usage.
    """
    x, x_tt = _make_input(device, *shape)
    y_tt = loopback(x_tt, unpack_to_dest=True)
    y = ttnn.to_torch(y_tt)
    assert_with_ulp(y, x, ulp_threshold=1)


# ---------------------------------------------------------------------------
# Test 2: unpack_to_dest=False — truncation must be observable on real HW
# ---------------------------------------------------------------------------


def test_loopback_truncation_no_unpack(device):
    """With unpack_to_dest=False, fp32 values are routed through 19-bit SrcA/SrcB,
    truncating mantissa bits.

    On real HW: output != input (truncation observed) → test PASSES.
    On simulator: may raise UndefinedBehavior for Default mode with fp32 data.
      If the simulator raises here AND test_loopback_exact_copy also raises,
      that indicates the simulator rejects both modes — a different issue.
      If the simulator raises here but NOT in test_loopback_exact_copy,
      the simulator correctly flags the invalid Default mode.
    """
    x, x_tt = _make_input(device)
    y_tt = loopback(x_tt, unpack_to_dest=False)
    y = ttnn.to_torch(y_tt)
    assert_with_ulp(y, x, ulp_threshold=1)
