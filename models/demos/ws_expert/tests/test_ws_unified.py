# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The deepseek_prefill unified routed expert op (unified_routed_expert_moe) as the shared expert, for comparison with
ws_shared.py: one chip, one expert with count T, K 3840, N 2944, x row-major bf16 (tilized to bf8 inside the op).
WS_WARM warm-up + WS_MEAS measured calls, wall time synchronize to synchronize; PCC on WSU_TOKENS sampled tokens
against torch (the device's weight quantization, bfp8 x, bf16 h)."""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, pcc, quantize
from models.demos.ws_expert.tests.test_ws_lane import first_visible

WARM = int(os.environ.get("WS_WARM", "20"))
MEAS = int(os.environ.get("WS_MEAS", "5"))


def weight_memory_config(mesh, n_tiles, nd_sharded):
    """DRAM interleaved, or the ND-sharded placement the op reads a core's N slice from in one request."""
    if not nd_sharded:
        return ttnn.DRAM_MEMORY_CONFIG
    gx = ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID.x
    per_core_n = (n_tiles + gx - 1) // gx
    dg = mesh.dram_grid_size()
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=ttnn.Shape([32, per_core_n * 32]),
            grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))]),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize(
    "prec,nd",
    [("bfp4_lofi", True), ("bfp4_lofi", False), ("bfp8_lofi", True)],
    ids=["bfp4_lofi_nd", "bfp4_lofi_il", "bfp8_lofi_nd"],
)
def test_ws_unified_shared(mesh_device, prec, nd):
    T, H, I = int(os.environ.get("WSU_T", "31250")), 3840, 2944
    TA = -(-T // 32) * 32
    # (the op runs at LoFi only)
    w_dt, fid = {"bfp4_lofi": (ttnn.bfloat4_b, "LoFi"), "bfp8_lofi": (ttnn.bfloat8_b, "LoFi")}[prec]
    torch.manual_seed(0)
    x = torch.zeros(TA, H, dtype=torch.bfloat16)
    x[:T] = torch.randn(T, H).to(torch.bfloat16)
    w1 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
    w3 = (torch.randn(H, I) / H**0.5).to(torch.bfloat16)
    w2 = (torch.randn(I, H) / I**0.5).to(torch.bfloat16)
    dev = lambda t, n: ttnn.to_device(
        ttnn.from_torch(t.float(), dtype=w_dt, layout=ttnn.TILE_LAYOUT),
        mesh_device,
        memory_config=weight_memory_config(mesh_device, n // 32, nd),
    )
    gate, up, down = [dev(w1, I)], [dev(w3, I)], [dev(w2, H)]
    u32 = lambda v: ttnn.from_torch(
        torch.tensor(v, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )
    offs, counts, gidx = u32([0]), u32([T]), u32([0])
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    ck = ttnn.WormholeComputeKernelConfig(  # (as TtRoutedExpert; the op requires math_approx_mode False)
        math_fidelity=getattr(ttnn.MathFidelity, fid),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    clk = aiclk_mhz(first_visible())
    wall = []
    for i in range(WARM + MEAS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        y = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x_t, offs, counts, gidx, gate, up, down, max_dispatched_tokens_per_expert=TA, compute_kernel_config=ck
        )
        ttnn.synchronize_device(mesh_device)
        if i >= WARM:
            wall.append((time.perf_counter() - t0) * 1e3)
        if i < WARM + MEAS - 1:
            ttnn.deallocate(y)
    ntok = int(os.environ.get("WSU_TOKENS", "256"))
    toks = torch.linspace(0, T - 1, ntok).round().long()
    got = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(TA, H)[toks]
    X = quantize(x[toks].float(), ttnn.bfloat8_b)
    q = lambda w: quantize(w.float(), w_dt)
    ref = quantize(torch.nn.functional.silu(X @ q(w1)) * (X @ q(w3)), ttnn.bfloat16) @ q(w2)
    p = pcc(got, ref)
    fl = 2 * T * H * I * 3
    wm = statistics.median(wall)
    sol = 110 * 4096 * clk * 1e6 / (1 if fid == "LoFi" else 2)
    gx, gy = ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID.x, ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID.y
    logger.info(
        f"WSU unified_routed_expert_moe as the shared expert, T {T}, {prec}, weights {'ND-sharded' if nd else 'interleaved'}"
        f", grid {gx} x {gy}, {WARM} warm-up + {MEAS} measured calls, aiclk {clk} MHz: wall ms median {wm:.3f} (min "
        f"{min(wall):.3f}); {fl / 1e12:.2f} TFLOP: util {100 * fl / sol / (wm / 1e3):.1f}% of the 110-core {fid} SOL "
        f"({fl / sol * 1e3:.2f} ms), {100 * fl * 110 / (gx * gy) / sol / (wm / 1e3):.1f}% of its {gx * gy} cores; "
        f"pcc {p:.6f} over {len(toks)} tokens; output {y.dtype} {y.layout} {list(y.shape)}"
    )
    assert p > 0.99
