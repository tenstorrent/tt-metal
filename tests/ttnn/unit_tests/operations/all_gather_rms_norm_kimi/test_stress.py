# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Back-to-back calls with DIFFERENT inputs each call, checked call by call.

The golden suite repeats identical inputs, so a call that read a landing buffer before (or after) the right call's
data landed would still pass there. Here every call's input differs, calls are enqueued back to back without host
synchronization (outputs read only after all are queued), and every output is checked bit-exactly against a fresh
single call of the same input."""

import os

import pytest
import torch
import ttnn

from eval.golden_tests.all_gather_rms_norm_kimi.helpers import (
    check_output,
    create_ttnn_gamma,
    create_ttnn_input_tensor,
    gathered_shape,
    pytorch_all_gather_rms_norm_kimi,
    random_gamma,
    random_stacked,
)
from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi, default_compute_kernel_config

N = int(os.environ.get("AGRN_STRESS_N", "12"))


@pytest.mark.parametrize("shape", [(1, 1, 640, 384)])
def test_back_to_back_distinct(shape, mesh_device):
    G = tuple(mesh_device.shape)[1]
    out_shape = gathered_shape(shape, G)
    gamma = random_gamma(out_shape[-1], torch.bfloat16, seed=7)
    tg = create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    xs = [
        random_stacked(mesh_device, shape, torch.bfloat16, seed=100 + i, scale=1.0 + i, offset=0.1 * i)
        for i in range(N)
    ]
    txs = [create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for x in xs]
    cfg = default_compute_kernel_config()
    outs = [all_gather_rms_norm_kimi(t, gamma=tg, cluster_axis=1, compute_kernel_config=cfg) for t in txs]
    bad = []
    for i, (x, o) in enumerate(zip(xs, outs)):
        expected = pytorch_all_gather_rms_norm_kimi(x, gamma, 1, 1e-6)
        try:
            check_output(
                o,
                expected,
                shape=out_shape,
                dtype=ttnn.bfloat16,
                fp32_dest_acc_en=True,
                expected_layout=ttnn.TILE_LAYOUT,
                group=G,
            )
        except AssertionError as e:
            bad.append(i)
            print(f"STRESS call {i}: {e}")
            # which slices/rows of which chip are wrong
            W = out_shape[-1]
            w = W // G
            for d, t in enumerate(ttnn.get_device_tensors(o)):
                a = ttnn.to_torch(t).float()[0, 0]
                e_ = expected[0, d][0, 0].float()
                for p in range(G):
                    for r in range(0, a.shape[0], 32):
                        blk = (
                            (a[r : r + 32, p * w : (p + 1) * w] - e_[r : r + 32, p * w : (p + 1) * w])
                            .abs()
                            .max()
                            .item()
                        )
                        if blk > 0.5:
                            print(
                                f"  chip {d} slice {p} tile-row {r // 32} cols-err {[(c // 32) for c in range(0, w, 32) if (a[r:r+32, p*w+c:p*w+c+32] - e_[r:r+32, p*w+c:p*w+c+32]).abs().max() > 0.5]}"
                            )
    assert not bad, f"calls {bad} wrong"
    # bit-exact against isolated calls, and every chip of the group identical
    for i in range(N):
        ref = all_gather_rms_norm_kimi(txs[i], gamma=tg, cluster_axis=1, compute_kernel_config=cfg)
        ttnn.synchronize_device(mesh_device)
        a = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(outs[i])]
        b = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(ref)]
        for d in range(len(a)):
            assert torch.equal(a[d], b[d]), f"call {i} chip {d}: back-to-back output differs from an isolated call"
            assert torch.equal(a[d], a[0]), f"call {i}: chip {d} differs from chip 0"
