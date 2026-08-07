# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Perf harness: softcap / situ_glu (composite) on Kimi-K3 FFN activation shapes.
    python -m tracy -p -r -o <outdir> -m pytest <this file>::test_situ_perf -k <bf16|bfp8>
"""

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


WIDTHS = [2048, 18432]
TOKENS = [32, 256, 1024, 5120]  # 5120 = tile-aligned 5k
DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
BETA1 = 4.0
BETA2 = 25.0
WARMUP = 2
ITERS = 3


def _run(tag, fn, device):
    for _ in range(WARMUP):
        ttnn.synchronize_device(device)
        fn()
    ttnn.synchronize_device(device)
    signpost(header=f"{tag}_begin")
    for _ in range(ITERS):
        fn()
    ttnn.synchronize_device(device)
    signpost(header=f"{tag}_end")


@pytest.mark.parametrize("dname", list(DTYPES))
def test_situ_perf(dname, device):
    dt = DTYPES[dname]
    torch.manual_seed(0)
    for w in WIDTHS:
        for t in TOKENS:
            gate = ttnn.from_torch(
                torch.empty([t, w], dtype=torch.bfloat16).uniform_(-30.0, 30.0),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            up = ttnn.from_torch(
                torch.empty([t, w], dtype=torch.bfloat16).uniform_(-30.0, 30.0),
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            _run(f"softcap_{dname}_W{w}_T{t}", lambda: ttnn.softcap(gate, BETA2), device)
            _run(f"situ_glu_{dname}_W{w}_T{t}", lambda: ttnn.situ_glu(gate, up, BETA1, BETA2), device)
            gate.deallocate()
            up.deallocate()
