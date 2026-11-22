# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""E8: tensor-prefetcher linear (tt/prefetch.py) matches ttnn.linear and replays in a trace.

    flock /tmp/tt-card.lock pytest -s models/demos/inkling2_mini/tests/test_prefetch.py
"""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.prefetch import PrefetchedLinear

# (K, N) of the attention weights: w_k / w_v, w_o, w_qgrs
SHAPES = [(2304, 1024), (3072, 2304), (2304, 6560)]


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 << 20}], indirect=True)
def test_prefetched_linear(device):
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("tensor prefetcher unsupported on this device")
    torch.manual_seed(0)
    pf = PrefetchedLinear(device, recv_rows=3, ring_kb=768)
    cases = []
    for k, n in SHAPES:
        w = torch.randn(1, 1, k, n) * 0.02
        x = torch.randn(1, 1, 1, k)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ref = ttnn.to_torch(
            ttnn.linear(
                tx,
                ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device),
                compute_kernel_config=pf.ckc,
            )
        )
        cases.append((tx, pf.weight(w), ref, n))

    pf.start()
    try:
        eager = [ttnn.to_torch(pf(tx, w)) for tx, w, _, _ in cases]
        for (_, _, ref, n), out in zip(cases, eager):
            assert out.shape[-1] == n
            assert _pcc(ref[..., :1, :], out[..., :1, :]) > 0.9999

        tid = ttnn.begin_trace_capture(device, cq_id=0)
        traced = [pf(tx, w) for tx, w, _, _ in cases]
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for _ in range(2):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            for out_t, out_e in zip(traced, eager):
                assert torch.equal(ttnn.to_torch(out_t), out_e)
        ttnn.release_trace(device, tid)
    finally:
        pf.stop()
