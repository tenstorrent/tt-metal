# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measured per-chip DRAM of the routed-expert weights on a real 4x8 mesh: free DRAM is probed by allocating 64 MB filler tensors per chip until the allocator
refuses (before and after building N real layers' expert weights, for the ND-sharded (unified-op) and the plain interleaved layout); also prints the allocator's
memory view when available. Extrapolation to 47 layers is flagged in the output. Env: MIMO_NL (4), MIMO_LAYER0 (1)."""
import os

import pytest

import ttnn
from models.demos.blackhole.mimo_v26.tt.moe_prefill import build_expert_weights
from models.demos.blackhole.mimo_v26.tt.moe_weights import load_layer_experts

NL = int(os.environ.get("MIMO_NL", "4"))
L0 = int(os.environ.get("MIMO_LAYER0", "1"))
FILL_MB = 64


def free_mb(md):
    fill = []
    try:
        while True:
            fill.append(
                ttnn.empty(
                    [1, 1, 32, FILL_MB * 1024 * 1024 // (32 * 2)],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=md,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            if len(fill) > 1000:
                break
    except Exception:
        pass
    n = len(fill)
    for f in fill:
        ttnn.deallocate(f)
    return n * FILL_MB


def view(md):
    try:
        v = ttnn.get_memory_view(md.get_devices()[0], ttnn.BufferType.DRAM)
        return {k: getattr(v, k) for k in dir(v) if not k.startswith("_") and isinstance(getattr(v, k), (int, float))}
    except Exception as ex:
        return f"n/a ({type(ex).__name__})"


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True
)
@pytest.mark.timeout(3600)
def test_moe_dram(mesh_device):
    md = mesh_device
    base = free_mb(md)
    print(
        f"RESULT free DRAM per chip before weights (64 MB probe granularity): {base} MB; allocator view {view(md)}",
        flush=True,
    )
    Ws = {L: load_layer_experts(L) for L in range(L0, L0 + NL)}
    for nd in (True, False):
        held = []
        for L in range(L0, L0 + NL):
            held.append(build_expert_weights(md, lambda e, L=L: Ws[L][e], nd_sharded=nd))
        after = free_mb(md)
        used = base - after
        print(
            f"RESULT expert weights {'ND-sharded (unified op)' if nd else 'interleaved'}: {NL} real layers use {used} MB per chip (+-64 MB) = {used / NL:.0f} MB/layer; "
            f"analytic bfp8 {3 * 4096 * 2048 * 1088 / 1024 * 8 / 2**20:.0f} MB/layer; EXTRAPOLATED x47 layers = {used / NL * 47 / 1024:.1f} GB (extrapolation, linear in layers); allocator view {view(md)}",
            flush=True,
        )
        for ws in held:
            for lst in ws:
                for t in lst:
                    ttnn.deallocate(t)
        del held
        print(f"RESULT after freeing: free {free_mb(md)} MB (baseline {base})", flush=True)
