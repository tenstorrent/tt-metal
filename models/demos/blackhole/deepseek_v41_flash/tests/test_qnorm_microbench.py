# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Result (2026-10-14, .44, T=16): sharded 8-core q-norm = 5.1 us alone but 10 us with slice + reshard + reshard back (single core: 16.0); kv-norm no gain (9.4 -> 9.0).
In the layer (L3 op table) the kernel time drops 5.7 us but two extra ops add ~4 us of gaps: traced 984.9 -> 987.6 us, i.e. NOT a speed-up; not merged.
test_qnorm_sharded_pcc imports DSV41Attention._q_norm, which exists only in the reverted experiment (re-apply from the notes to run it).

Traced micro-benchmark of the attention q-norm / kv-norm variants (decode, T=16 rows per device): the baseline slice + single-core rms_norm
against sharded multi-core rms_norm. Prints QN lines: name = us per call (20 calls in one trace)."""

import time

import pytest
import torch

import ttnn

T, QL, HD = 16, 1280, 512
REP = 20


def traced_us(mesh, fn, n=30):
    def body():
        for _ in range(REP):
            fn()

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n / REP
    ttnn.release_trace(mesh, tid)
    return dt * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 100_000_000}, id="noring")],
    indirect=True,
)
@torch.no_grad()
def test_qnorm_microbench(mesh_device):
    md = mesh_device
    up = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(
        t.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    y = up(torch.randn(1, 1, T, QL + HD))
    gq = up(torch.rand(1, 1, 1, QL) + 0.5)
    gk = up(torch.rand(1, 1, 1, HD) + 0.5)
    eps = 1e-20
    res = {}

    res["q_base"] = traced_us(
        md, lambda: ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, QL]), weight=gq, epsilon=eps)
    )
    res["k_base"] = traced_us(
        md, lambda: ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, QL], [1, 1, T, QL + HD]), weight=gk, epsilon=eps)
    )
    res["slice_q"] = traced_us(md, lambda: ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, QL]))
    yq = ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, QL])
    res["q_norm_only"] = traced_us(md, lambda: ttnn.rms_norm(yq, weight=gq, epsilon=eps))
    l1 = ttnn.L1_MEMORY_CONFIG
    yql1 = ttnn.to_memory_config(yq, l1)
    res["q_norm_l1_in_out"] = traced_us(md, lambda: ttnn.rms_norm(yql1, weight=gq, epsilon=eps, memory_config=l1))

    # sharded multi-core variants
    for n in (5, 8, 10, 20):
        sw = QL // n
        if QL % n or sw % 32:
            continue
        grid = ttnn.CoreGrid(y=1, x=n) if n <= 8 else ttnn.CoreGrid(y=n // 10, x=10) if n % 10 == 0 else None
        if grid is None:
            continue
        crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(8, 8), row_wise=True)
        smc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (32, sw), ttnn.ShardOrientation.ROW_MAJOR),
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(min(n, 8), (n + 7) // 8),
            subblock_w=min(sw // 32, 4) if (sw // 32) % min(sw // 32, 4) == 0 else 1,
            block_h=1,
            block_w=sw // 32,
            inplace=False,
        )
        try:
            ys = ttnn.to_memory_config(yq, smc)
            res[f"q_sharded{n}_op"] = traced_us(
                md, lambda: ttnn.rms_norm(ys, weight=gq, epsilon=eps, program_config=pc, memory_config=smc)
            )
            res[f"q_sharded{n}_slice+reshard+op+back"] = traced_us(
                md,
                lambda: ttnn.to_memory_config(
                    ttnn.rms_norm(
                        ttnn.to_memory_config(ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, QL]), smc),
                        weight=gq,
                        epsilon=eps,
                        program_config=pc,
                        memory_config=smc,
                    ),
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        except Exception as e:  # noqa: BLE001
            res[f"q_sharded{n}"] = f"ERR {str(e)[:200]}"

    # kv-norm sharded variants
    ykv = ttnn.slice(y, [0, 0, 0, QL], [1, 1, T, QL + HD])
    for n in (2, 4, 8):
        sw = HD // n
        crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(8, 8), row_wise=True)
        smc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (32, sw), ttnn.ShardOrientation.ROW_MAJOR),
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(n, 1),
            subblock_w=1,
            block_h=1,
            block_w=sw // 32,
            inplace=False,
        )
        try:
            ys = ttnn.to_memory_config(ykv, smc)
            res[f"k_sharded{n}_op"] = traced_us(
                md, lambda: ttnn.rms_norm(ys, weight=gk, epsilon=eps, program_config=pc, memory_config=smc)
            )
            res[f"k_sharded{n}_full"] = traced_us(
                md,
                lambda: ttnn.to_memory_config(
                    ttnn.rms_norm(
                        ttnn.to_memory_config(ttnn.slice(y, [0, 0, 0, QL], [1, 1, T, QL + HD]), smc),
                        weight=gk,
                        epsilon=eps,
                        program_config=pc,
                        memory_config=smc,
                    ),
                    ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        except Exception as e:  # noqa: BLE001
            res[f"k_sharded{n}"] = f"ERR {str(e)[:200]}"
    for k, v in res.items():
        print(f"QN {k} = {v if isinstance(v, str) else f'{v:.2f} us'}", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 100_000_000}, id="noring")],
    indirect=True,
)
@torch.no_grad()
def test_qnorm_sharded_pcc(mesh_device):
    """The 8-core sharded q-norm of DSV41Attention._q_norm against the single-core rms_norm it replaces, and against a torch reference."""
    from models.demos.blackhole.deepseek_v41_flash.tt.attention import Q_LORA as QL_
    from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention

    md = mesh_device
    up = lambda t: ttnn.from_torch(
        t.to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    h = torch.randn(1, 1, T, QL_ + HD) * 3
    g = torch.rand(1, 1, 1, QL_) + 0.5
    y, gq = up(h), up(g)
    # build only what _q_norm touches
    self = DSV41Attention.__new__(DSV41Attention)
    self.T, self.eps, self.q_norm = T, 1e-20, gq
    n = 8
    self._qn_cores = n
    self._qn_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(8, 8), row_wise=True),
            (32, QL_ // n),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    self._qn_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(n, 1), subblock_w=1, block_h=1, block_w=QL_ // n // 32, inplace=False
    )
    got = ttnn.to_torch(ttnn.get_device_tensors(DSV41Attention._q_norm(self, y))[0]).float()[0, 0, :T]
    base = ttnn.to_torch(
        ttnn.get_device_tensors(ttnn.rms_norm(ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, QL_]), weight=gq, epsilon=1e-20))[0]
    ).float()[0, 0, :T]
    x = h[0, 0, :, :QL_].to(torch.bfloat16).float()
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-20) * g[0, 0].to(torch.bfloat16).float()
    p = lambda a, b: float(torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1])
    print(
        f"QN pcc sharded vs single-core = {p(got, base):.6f}, sharded vs torch = {p(got, ref):.6f}, single-core vs torch = {p(base, ref):.6f}, "
        f"max|sharded-single| = {float((got - base).abs().max()):.4g}, bit-exact = {bool(torch.equal(got, base))}",
        flush=True,
    )
    assert p(got, base) > 0.9999
