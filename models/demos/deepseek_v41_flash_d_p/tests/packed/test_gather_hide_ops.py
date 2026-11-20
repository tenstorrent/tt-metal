# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.3H: the gather sub-device of PACKED_GATHER_HIDE=on (tt/gather_hide.Split) on the model's mesh, mesh only (no
model), smallest shape first: the history's full-mesh high_bw_all_gather of 320 B records on sub-device 1 (the grid's
free column) while sub-device 0 runs a chain of local ops on the matmul grid, against the same gather serial on the
default manager and against the host (every chip: owner q's rows at q R). Twice (the second region on the cached
programs), then host-timed (median of 5, synchronized around): serial gather, serial gather + compute, the region.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=8192 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_gather_hide_ops.py"
"""

import os
import statistics
import time

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"
ROWS = [int(v) for v in os.environ.get("PB3H_ROWS", "128,2176").split(",")]  # an owner's records (1 / 17 periods)


def _sync(mesh):
    import ttnn

    ttnn.synchronize_device(mesh)


@mesh_parametrize(timeout_s=900)
def test_gather_hide_ops(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import ccl, cores
    from models.demos.deepseek_v41_flash_d_p.tt.gather_hide import Split

    mesh = mesh_device
    cores.init_caps(mesh)
    split = Split.of(mesh)
    if split is None:
        pytest.skip("no free grid column beside the matmul grid (no core cap on this box): the serial path only")
    sp, tp = tuple(mesh.shape)
    n = sp * tp
    print(f"\n[PB.3H] compute {split.compute} gather {split.gather}")
    s0, s1 = ccl._sems(mesh, None)
    g = torch.Generator().manual_seed(11)
    # (the compute block: local ops on the matmul grid, ~a layer's attention phase in length)
    xa = ttnn.from_torch(
        torch.randn(1, 1, 2048, 4096, generator=g),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def compute():
        y = xa
        for _ in range(6):
            z = ttnn.multiply(y, 1.0009765625, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=cores.sub_cores(y))
            if y is not xa:
                ttnn.deallocate(y)
            y = z
        return y

    for R in ROWS:
        src = torch.randint(0, 256, (1, 1, n * R, 320), dtype=torch.uint8, generator=g)
        x = ttnn.from_torch(
            src,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
        )
        outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, n * R, 320]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            for _ in range(3)
        ]

        def gather(out, sub=None):
            ttnn.experimental.high_bw_all_gather(
                x,
                2,
                out,
                cluster_axis=None,
                num_links=ccl.AG_LINKS,
                ready_semaphore=s0,
                data_valid_semaphore=s1,
                **(sub or {}),
            )

        gather(outs[0])
        y0 = compute()
        ys = []
        for k in (1, 2):
            ys.append(split.region(lambda sub, o=outs[k]: gather(o, sub), compute, "attention"))
        _sync(mesh)
        want = src.reshape(n * R, 320)
        bad = []
        for k, out in enumerate(outs):
            for q, d in enumerate(ttnn.get_device_tensors(out)):
                if not torch.equal(ttnn.to_torch(d).reshape(n * R, 320), want):
                    bad.append((k, q))
        yr = ttnn.to_torch(ttnn.get_device_tensors(y0)[0])
        ybad = [k for k, y in enumerate(ys) if not torch.equal(ttnn.to_torch(ttnn.get_device_tensors(y)[0]), yr)]
        for y in [y0] + ys:
            ttnn.deallocate(y)
        print(
            f"[PB.3H] R {R}: gather outputs wrong (call, chip): {bad[:8]} ({len(bad)}); region compute != serial: {ybad}"
        )
        assert not bad and not ybad

        def timed(fn):
            ts = []
            for k in range(2 + 5):
                _sync(mesh)
                t0 = time.perf_counter()
                out = fn()
                _sync(mesh)
                if k >= 2:
                    ts.append((time.perf_counter() - t0) * 1e3)
                if isinstance(out, ttnn.Tensor):
                    ttnn.deallocate(out)
            return statistics.median(ts)

        t_g = timed(lambda: gather(outs[0]))
        t_c = timed(compute)
        t_gc = timed(lambda: (gather(outs[0]), compute())[1])
        t_rg = timed(lambda: split.region(lambda sub: gather(outs[1], sub), lambda: None, "attention"))
        t_r = timed(lambda: split.region(lambda sub: gather(outs[1], sub), compute, "attention"))
        print(
            f"[PB.3H] R {R} ({n * R * 320 / 2**20:.1f} MiB a chip): ms serial gather {t_g:.3f}, compute {t_c:.3f}, "
            f"serial both {t_gc:.3f}; region gather alone (gather column) {t_rg:.3f}, region gather || compute "
            f"{t_r:.3f}: hidden {t_gc - t_r:.3f}"
        )
        for t in outs + [x]:
            ttnn.deallocate(t)
    ttnn.deallocate(xa)
