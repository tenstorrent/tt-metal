# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.HG: the owned history send (tt/owned_gather.OwnedSend) on the model's mesh, mesh only (no model), smallest shape
first. Every chip holds its own 320 B records of every segment (a different random buffer a segment); one call sends
segment k's rows [lo_k, hi_k) of every chip only to the chips the map gives k, into owner q's rows q R + [lo, hi) of
one history buffer. Checked against the host on every chip: the destination rows equal the owners' records, every
other row keeps its fill (nothing else written). Layouts: 4 x 8 chips (4 x 2k), 4 x 7 chips + 4 idle chips on segment
0 (4 x 1600s), 1 + 31, six mixed (1-chip segments), a tail range [lo, hi) mid-slot; one call inside a gather_hide
region (sub-device 1 beside a compute chain). No program after the first call of a geometry (the map, ranges and
buffers are values). Then host-timed (median of 5, synchronized around) at R = 8192 (a 1M slot): the owned send of 4
x 8 chips vs four whole-mesh high_bw_all_gathers (today: one a segment).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=8192 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_owned_send_ops.py"
"""

import os
import statistics
import time

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"
ROWS = [int(v) for v in os.environ.get("PBHG_ROWS", "64,2176").split(",")]
TIME_ROWS = int(os.environ.get("PBHG_TIME_ROWS", "8192"))
FILL = 0xA5


def _layouts(n: int) -> list:
    """(name, chips a segment, idle chips' segment) layouts of the n chips (contiguous chip-linear ranges)."""
    out = [("4seg", [n // 4] * 4)]
    if n >= 32:
        out += [("4x7+idle", [7, 7, 7, 7]), ("1+31", [1, n - 1]), ("six", [1, 2, 8, 1, 7, 13])]
    elif n >= 4:
        out += [("1+rest", [1, n - 1])]
    return out


def _dest(n: int, sizes: list) -> list:
    dest, q = [], 0
    for k, m in enumerate(sizes):
        dest += [k] * m
        q += m
    dest += [0] * (n - q)  # (idle chips: segment 0's history, as the sdpa's lane map reads)
    return dest


@mesh_parametrize(timeout_s=1200)
def test_owned_send_ops(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import ccl, cores
    from models.demos.deepseek_v41_flash_d_p.tt.gather_hide import Split
    from models.demos.deepseek_v41_flash_d_p.tt.owned_gather import sender

    mesh = mesh_device
    cores.init_caps(mesh)
    sp, tp = tuple(mesh.shape)
    n = sp * tp
    send = sender(mesh)
    print(f"\n[PB.HG] mesh {sp}x{tp}, cores {[(c.x, c.y) for c in send.cores]}, links {send.L}")
    g = torch.Generator().manual_seed(21)

    def upload(t, mapper):
        return ttnn.from_torch(
            t,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def check(out, srcs, entries, dest, R, tag):
        bad = []
        for d, t in enumerate(ttnn.get_device_tensors(out)):
            got = ttnn.to_torch(t).reshape(n, R, 320)
            want = torch.full((n, R, 320), FILL, dtype=torch.uint8)
            k = dest[d]
            if k is not None:
                lo, hi = entries[k][1], entries[k][2]
                want[:, lo:hi] = srcs[k].reshape(n, R, 320)[:, lo:hi]
            if not torch.equal(got, want):
                rows = (got != want).any(-1).nonzero()
                bad.append((d, int(rows.shape[0]), rows[:2].tolist()))
        print(f"[PB.HG] {tag}: chips wrong {len(bad)} {bad[:4]}")
        return bad

    split = Split.of(mesh)
    fails = []
    for R in ROWS:
        srcs = [torch.randint(0, 256, (1, 1, n * R, 320), dtype=torch.uint8, generator=g) for _ in range(6)]
        locs = [upload(s, ttnn.ShardTensorToMesh(mesh, dim=2)) for s in srcs]
        fill = torch.full((1, 1, n * R, 320), FILL, dtype=torch.uint8)
        out = upload(fill, ttnn.ReplicateTensorToMesh(mesh))
        host_fill = ttnn.from_torch(
            fill, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )

        def refill(t):
            ttnn.copy_host_to_device_tensor(host_fill, t)

        n0 = None
        for name, sizes in _layouts(n):
            for rng in ("full", "tail"):
                dest = _dest(n, sizes)
                if rng == "full":
                    entries = [(locs[k], 0, R) for k in range(len(sizes))]
                else:  # (a tail: rows mid-slot, a different range a segment)
                    entries = []
                    for k in range(len(sizes)):
                        lo = (R // 3) // 32 * 32 + 32 * (k % 2)
                        entries.append((locs[k], lo, min(R, lo + R // 4 + 32 * (k % 3))))
                refill(out)
                send(out, R, 0, entries, dest)
                ttnn.synchronize_device(mesh)
                if n0 is None:
                    n0 = mesh.num_program_cache_entries()
                fails += [(R, name, rng, b) for b in check(out, srcs, entries, dest, R, f"R {R} {name} {rng}")]
        if split is not None:  # (inside a region, on the gather sub-device beside a compute chain)
            dest = _dest(n, _layouts(n)[0][1])
            entries = [(locs[k], 0, R) for k in range(4)]
            refill(out)
            xa = ttnn.from_torch(
                torch.randn(1, 1, 2048, 4096, generator=g),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

            def compute(_):
                y = xa
                for _ in range(6):
                    z = ttnn.multiply(
                        y, 1.0009765625, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=cores.sub_cores(y)
                    )
                    if y is not xa:
                        ttnn.deallocate(y)
                    y = z
                return y

            y = split.region([(0, lambda sub: send(out, R, 0, entries, dest))], [compute], "attention")
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(y)
            fails += [(R, "region", b) for b in check(out, srcs, entries, dest, R, f"R {R} region")]
        builds = mesh.num_program_cache_entries() - n0
        print(f"[PB.HG] R {R}: program-cache entries after the first call +{builds} (other layouts / ranges)")
        for t in locs:
            ttnn.deallocate(t)
        ttnn.deallocate(out)
    assert not fails, fails[:8]

    # host-timed at a 1M slot's rows: the owned send (4 segments) vs today's four whole-mesh gathers, serial (whole
    # grid) and on the gather column (a region beside a one-op compute step, as the model runs them)
    R = TIME_ROWS
    srcs = [torch.randint(0, 256, (1, 1, n * R, 320), dtype=torch.uint8, generator=g) for _ in range(4)]
    locs = [upload(s_, ttnn.ShardTensorToMesh(mesh, dim=2)) for s_ in srcs]
    outs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, n * R, 320]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(5)
    ]
    s0, s1 = ccl._sems(mesh, None)
    dest = _dest(n, _layouts(n)[0][1])
    entries = [(locs[k], 0, R) for k in range(4)]
    print(f"[PB.HG] fabric payload {send.payload} B: {send.payload // 320} rows a packet")

    def owned(sub=None):
        send(outs[0], R, 0, entries, dest)

    def today(sub=None):
        for k in range(4):
            ttnn.experimental.high_bw_all_gather(
                locs[k],
                2,
                outs[1 + k],
                cluster_axis=None,
                num_links=ccl.AG_LINKS,
                ready_semaphore=s0,
                data_valid_semaphore=s1,
                **(sub or {}),
            )

    tiny = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def step(_):
        return ttnn.multiply(tiny, 1.0, sub_core_grids=cores.sub_cores(tiny))

    res = {}
    for where in ("serial", "column"):
        if where == "column" and split is None:
            continue
        for name, fn in (("owned", owned), ("today4", today)):

            def run():
                if where == "serial":
                    fn()
                else:
                    ttnn.deallocate(split.region([(0, fn)], [step], "attention"))

            run()
            ttnn.synchronize_device(mesh)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                run()
                ttnn.synchronize_device(mesh)
                ts.append((time.perf_counter() - t0) * 1e3)
            res[(where, name)] = statistics.median(ts)
    mb = n * R * 320 / 2**20
    for where in ("serial", "column"):
        if (where, "owned") in res:
            print(
                f"[PB.HG] R {R} ({mb:.1f} MiB a history a chip), {where}: owned 4 x {n // 4} chips "
                f"{res[(where, 'owned')]:.2f} ms, today 4 x high_bw_all_gather {res[(where, 'today4')]:.2f} ms "
                f"(host-timed, median of 5)"
            )
    assert not check(outs[0], srcs, entries, dest, R, f"R {R} timed (every row a destination row)")
