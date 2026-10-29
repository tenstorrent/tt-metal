"""moe_tail reader v2 (DSV41_MOE_TAIL_V2=1, L1->L1 NoC face scatter) vs v1: bit-exact on synthetic routed data, T=1..16, 3 seeds,
including the padding rows of the output tile; DSV41_TAIL_PROF_T=4,8,16 additionally runs each variant 4x for the device profiler."""
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.moe_tail import make_col_tensor, moe_tail

K, H, E = 6, 5120, 256
ROWS, COLS = 4, 8


def _mk(md, T, seed):
    g = torch.Generator().manual_seed(seed)
    comb = torch.randn(ROWS * K, COLS * T, H, generator=g).to(torch.bfloat16)
    sc = torch.rand(ROWS * T, 1, 1, K, generator=g).to(torch.bfloat16)
    idx = torch.randint(0, E, (ROWS * T, 1, 1, K), generator=g).to(torch.int32)
    mp = torch.randint(0, ROWS * COLS, (1, E), generator=g).to(torch.int32)
    kw = dict(device=md, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t_comb = ttnn.from_torch(
        comb, dtype=ttnn.bfloat16, mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(ROWS, COLS)), **kw
    )
    rep = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(ROWS, COLS))
    t_sc = ttnn.from_torch(sc, dtype=ttnn.bfloat16, mesh_mapper=rep, **kw)
    t_idx = ttnn.from_torch(idx, dtype=ttnn.uint16, mesh_mapper=rep, **kw)
    t_map = ttnn.from_torch(
        mp,
        device=md,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    return t_comb, t_sc, t_idx, t_map


def _run(md, args, col, v2):
    os.environ["DSV41_MOE_TAIL_V2"] = "1" if v2 else "0"
    orig = ttnn.allocate_tensor_on_device

    def alloc32(shape, *a, **k):  # full 32-row logical tile so the padding rows are visible
        return orig(ttnn.Shape([1, 1, 32, H]), *a, **k)

    ttnn.allocate_tensor_on_device = alloc32
    try:
        o = moe_tail(*args, col)
    finally:
        ttnn.allocate_tensor_on_device = orig
    ttnn.synchronize_device(md)
    return o


def _full(md, o, T):
    """raw 32-row tile content incl. padding rows (view of the tile tensor as [1,1,32,H])."""
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=(ROWS, COLS))
    return ttnn.to_torch(o, mesh_composer=comp)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
def test_moe_tail_v2(mesh_device):
    md = mesh_device
    col = make_col_tensor(md)
    prof_ts = [int(v) for v in os.environ.get("DSV41_TAIL_PROF_T", "").split(",") if v]
    ts = [
        int(v)
        for v in os.environ.get("DSV41_TAIL_TS", ",".join(str(i) for i in list(range(1, 17)) + [24, 32])).split(",")
    ]
    seeds = [int(v) for v in os.environ.get("DSV41_TAIL_SEEDS", "0,1,2").split(",")]
    bad = 0
    for T in ts:
        for seed in seeds:
            args = _mk(md, T, seed)
            a = _full(md, _run(md, args, col, False), T)
            b = _full(md, _run(md, args, col, True), T)
            same = torch.equal(a.view(torch.int16), b.view(torch.int16))
            nz = float((a.float().abs() > 0).float().mean())
            print(f"TAILV2 T={T} seed={seed} bit_equal={same} nonzero_frac={nz:.3f}", flush=True)
            bad += 0 if same else 1
            if not same:
                d = (a.view(torch.int16) != b.view(torch.int16)).reshape(-1, 32, H)
                print(
                    "  diff rows",
                    d.any(-1).any(0).nonzero().flatten().tolist(),
                    "frac in real rows",
                    float(d[:, :T].float().mean()),
                    "pad rows",
                    float(d[:, T:].float().mean()) if T < 32 else 0,
                    flush=True,
                )
                bb = b.view(torch.int16).reshape(-1, 32, H // 32, 32)[:, :T]
                aa = a.view(torch.int16).reshape(-1, 32, H // 32, 32)[:, :T]
                print(
                    "  v2 hi half == v1 lo half:",
                    float((bb[..., 16:] == aa[..., :16]).float().mean()),
                    "v2 hi == v1 hi:",
                    float((bb[..., 16:] == aa[..., 16:]).float().mean()),
                    "lo ok:",
                    float((bb[..., :16] == aa[..., :16]).float().mean()),
                    flush=True,
                )
                print(
                    "  diff col%32",
                    d.any(0).any(0).reshape(-1, 32).any(0).nonzero().flatten().tolist(),
                    "dev0 frac",
                    float(d[0].float().mean()),
                    flush=True,
                )
            for a_ in args:
                ttnn.deallocate(a_)
    assert bad == 0
    for T in prof_ts:
        args = _mk(md, T, 7)
        for v2 in (False, True):
            for _ in range(4):
                _run(md, args, col, v2)
        print(f"TAILV2 PROF T={T} done (order: 4x v1 then 4x v2)", flush=True)
    if prof_ts:
        ttnn.ReadDeviceProfiler(md)
