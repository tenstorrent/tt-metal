# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only check of the Blackhole no-prefetcher prefill reduce-scatter layout. No device is opened.

TT_CCL.line_reduce_scatter must give Ring ttnn.reduce_scatter one (batch, channel) unit per
batched-prefill user. In one flat unit, a user's summation order depends on its row offset, so
identical users get different results. test_reduce_scatter_slot_invariance_bh.py checks on device
that one unit per user is slot-invariant; this test checks that TT_CCL builds that layout.
"""

import pytest
import torch

import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL

RING_SIZE = 4
WIDTH = 8 * RING_SIZE


@pytest.fixture(autouse=True)
def ensure_devices():
    """Skip the parent Galaxy device-count check: this test opens no device."""


def _bh_no_prefetcher_prefill_ccl():
    ccl = TT_CCL.__new__(TT_CCL)
    ccl.mode = "prefill"
    ccl.use_prefetcher = False
    ccl.model_config = {"CCL_TOPOLOGY": ttnn.Topology.Ring}
    ccl.worker_sub_device_id = None
    ccl.gather_idx = [0, 0]
    ccl.num_cbs = 2
    return ccl


@pytest.mark.parametrize(
    "input_shape,batch_size,reduce_scatter_shape",
    [
        # Batched prefill: 32 users of 128 rows arrive flat, and each needs its own unit.
        ((1, 1, 32 * 128, WIDTH), 32, (1, 32, 128, WIDTH)),
        # Single-user prefill stays one flat unit.
        ((1, 1, 4096, WIDTH), 1, (1, 1, 4096, WIDTH)),
        # Single-user prefill in 1024-row chunks, as the MLP sends it, is flattened to one unit.
        ((1, 2, 1024, WIDTH), 1, (1, 1, 2048, WIDTH)),
    ],
    ids=["batched_32_users", "single_user", "single_user_chunked"],
)
def test_prefill_reduce_scatter_unit_per_user(monkeypatch, input_shape, batch_size, reduce_scatter_shape):
    reduce_scatter_inputs = []

    def reduce_scatter(input_tensor, dim, **kwargs):
        reduce_scatter_inputs.append(tuple(input_tensor.shape))
        # Keep one device's slice of the scattered width; rows pass through in order.
        return input_tensor[..., : input_tensor.shape[-1] // RING_SIZE].clone()

    monkeypatch.setattr(ttnn, "reshape", lambda tensor, shape: tensor.reshape(shape))
    monkeypatch.setattr(ttnn, "reduce_scatter", reduce_scatter)

    x = torch.arange(torch.Size(input_shape).numel(), dtype=torch.float32).reshape(input_shape)
    out = _bh_no_prefetcher_prefill_ccl().line_reduce_scatter(
        x, memory_config=None, cluster_axis=1, dim=3, batch_size=batch_size
    )

    assert reduce_scatter_inputs == [reduce_scatter_shape]
    # The caller gets its own layout back, with every row in place.
    assert torch.equal(out, x[..., : WIDTH // RING_SIZE])
