# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The sequence-parallel linears on a (1, 4) ``tp`` mesh, ring and line.

The (1, 2) mesh of test_sequence_parallel.py is a line, so this is the only tt-train exercise of the fused
ops' ring schedules (T=4, the smallest even ring). Composed must be bitwise the collective + linear sequence
it replaces; Fused must agree with Composed in ULP, in both topologies.

Run it in its own pytest process: tt-metal sizes the SystemMesh once per process from the first MGD it
opens, so after a (1, 2)-mesh module the (1, 4) open fails and this module skips.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import ttnn
import ttml

SPLinearImpl = ttml.ops.distributed.SPLinearImpl
from sp_linear_testlib import (
    assert_bitwise_equal,
    assert_within_ulp,
    column_linear_reference,
    column_operands,
    forward_backward,
    row_linear_reference,
    row_operands,
    sp_linear_impl,
)

TP_AXIS_SIZE = 4
MESH_SHAPE = (1, TP_AXIS_SIZE)
MAX_ULP = 2.0  # as test_sequence_parallel.py
SEQ_LEN = 32 * TP_AXIS_SIZE * 2  # two tiles of sequence per rank
IN_FEATURES, OUT_FEATURES = 128, 256  # tile-aligned after sharding across four ranks

_MGD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "mgd")
_MGD = {
    "ring": os.path.join(_MGD_DIR, "bh_galaxy_1_4_ring_ring.textproto"),
    "line": os.path.join(_MGD_DIR, "bh_galaxy_1_4_line_line.textproto"),
}


def _close_quietly() -> None:
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="module", params=["ring", "line"])
def tp4_mesh(request):
    """A ``[1, 4]`` mesh with axes ``("dp", "tp")`` wired as a ring or a line along tp (the fabric config is
    inferred from the MGD's dim_types). Skips if the descriptor is not for this machine, if the caller already
    points TT_MESH_GRAPH_DESC_PATH elsewhere, or if four devices are unavailable."""
    mgd = os.path.realpath(_MGD[request.param])
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous and os.path.realpath(previous) != mgd:
        pytest.skip(f"TT_MESH_GRAPH_DESC_PATH points at {previous}; this module needs {mgd}")
    try:
        arch = ttnn.get_arch_name().lower()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot detect the device architecture: {e}")
    if "blackhole" not in arch:
        pytest.skip(f"{os.path.basename(mgd)} describes a Blackhole galaxy, this is {arch}")

    os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    _close_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(MESH_SHAPE, ("dp", "tp")))
    except Exception as e:  # noqa: BLE001
        _close_quietly()
        _restore(previous)
        pytest.skip(f"needs a {list(MESH_SHAPE)} 'tp' mesh wired as a {request.param}: {e}")

    yield request.param

    _close_quietly()
    _restore(previous)


def _restore(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


@pytest.fixture(autouse=True)
def default_impl_afterwards():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.ops.distributed.set_sp_linear_impl(SPLinearImpl.FUSED)  # the default


@pytest.mark.requires_device
class TestSPLinearOps1x4:
    @pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "no_bias"])
    @pytest.mark.parametrize("batch", [1, 2])
    def test_column_parallel(self, tp4_mesh, batch, has_bias):
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(100 + 10 * batch + has_bias)
        operands, grad_out = column_operands(rng, batch, has_bias, SEQ_LEN, IN_FEATURES, OUT_FEATURES)
        label = f"1x4 {tp4_mesh} column batch={batch} bias={has_bias}"

        expected = forward_backward(column_linear_reference, operands, grad_out, axis)
        with sp_linear_impl("composed"):
            composed = forward_backward(ttml.ops.distributed.sp_column_parallel_linear, operands, grad_out, axis)
        with sp_linear_impl("fused"):
            fused = forward_backward(ttml.ops.distributed.sp_column_parallel_linear, operands, grad_out, axis)

        assert composed["out"].shape == (batch, TP_AXIS_SIZE, SEQ_LEN, OUT_FEATURES // TP_AXIS_SIZE)
        assert_bitwise_equal(composed, expected, f"{label} composed")
        assert_within_ulp(fused, composed, f"{label} fused", MAX_ULP)

    @pytest.mark.parametrize("batch", [1, 2])
    def test_row_parallel(self, tp4_mesh, batch):
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(120 + batch)
        operands, grad_out = row_operands(rng, batch, SEQ_LEN, IN_FEATURES, OUT_FEATURES)
        label = f"1x4 {tp4_mesh} row batch={batch}"

        expected = forward_backward(row_linear_reference, operands, grad_out, axis)
        with sp_linear_impl("composed"):
            composed = forward_backward(ttml.ops.distributed.sp_row_parallel_linear, operands, grad_out, axis)
        with sp_linear_impl("fused"):
            fused = forward_backward(ttml.ops.distributed.sp_row_parallel_linear, operands, grad_out, axis)

        assert composed["out"].shape == (batch, TP_AXIS_SIZE, SEQ_LEN // TP_AXIS_SIZE, OUT_FEATURES)
        assert_bitwise_equal(composed, expected, f"{label} composed")
        assert_within_ulp(fused, composed, f"{label} fused", MAX_ULP)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
