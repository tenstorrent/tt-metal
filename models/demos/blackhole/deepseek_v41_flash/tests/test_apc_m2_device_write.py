# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""APC_CHUNKED_PREFILL_PLAN.md milestone M2, the largest unknown: can the host write ONE device's shard of a persistent mesh tensor (the per-slot carried prefill state:
halo, kv table, key FIFO) outside a trace, bit-exactly, without touching the other devices, while a captured trace keeps reading the same buffer? How long does it take?

Model-free (no weights): persistent bf16 tensors shaped like the carried state, replicated over the (4, 8) mesh and then given DIFFERENT contents per device by the SPMD program in
real use. Variants measured:
  A. ``ttnn.copy_host_to_device_tensor(host_shard, ttnn.get_device_tensors(t)[i])`` per device shard (no read, no mask),
  B. the masked idiom of the model (``ttnn.where`` / multiply-add with a per-row 0/1 mask, whole-mesh upload of a replicated host tensor),
  C. read-back of one device shard (``ttnn.to_torch(get_device_tensors(t)[i])``) for the snapshot side.
Also: the buffer address is unchanged and a captured trace that reads the tensor sees the new values.

    pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_m2_device_write.py -s
Env: DSV41_M2_ROWS_K (rows of the big tensor in thousands, default 16 -> 16384 x 512 bf16 = 16 MiB per device)
"""

import os
import time

import pytest
import torch

import ttnn

K_ROWS = int(os.environ.get("DSV41_M2_ROWS_K", "16"))


def _addr(t):
    try:
        return [d.buffer_address() for d in ttnn.get_device_tensors(t)][:3]
    except (
        Exception
    ):  # noqa: BLE001  (no per-device buffer_address in this build: the persistent tensor object is the same)
        return [t.buffer_address()]


def same(x, y):
    return x.numel() == y.numel() and bool(
        torch.equal(x.reshape(-1).view(torch.int16), y.reshape(-1).view(torch.int16))
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 50_000_000}, id="plain")],
    indirect=True,
)
@torch.no_grad()
def test_per_device_shard_write(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    log = lambda m: print(m, flush=True)
    N = K_ROWS * 1024
    shape = [1, 1, N, 512]
    g = torch.Generator().manual_seed(7)
    mk = lambda: torch.randn(*shape, generator=g).to(torch.bfloat16)

    t = ttnn.from_torch(
        torch.zeros(*shape),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    addr0 = _addr(t)
    devs = ttnn.get_device_tensors(t)
    assert len(devs) == rows * cols

    def read_all():
        return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]

    # ---- A. per-device-shard host write: does NOT work as a single-device write ---------------------------------------------------------------------
    data = {k: mk() for k in (0, 9, 31)}  # (mesh row 0, col 0), (row 1, col 1), (row 3, col 7)
    for k, x in data.items():
        host = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)  # single-device host tensor
        ttnn.copy_host_to_device_tensor(host, ttnn.get_device_tensors(t)[k])
        ttnn.synchronize_device(md)
    back = read_all()
    lab = []
    for k in range(rows * cols):
        if float(back[k].float().abs().max()) == 0.0:
            lab.append("0")
        else:
            m_ = [j for j, x in data.items() if same(back[k], x)]
            lab.append(f"d{m_[0]}" if m_ else "?")
    per_dev = all(same(back[k], data[k]) for k in data) and all(l == "0" for k, l in enumerate(lab) if k not in data)
    log(
        f"A: copy_host_to_device_tensor(host, get_device_tensors(t)[k]) writes ONE device only: {per_dev}. Devices written in order {list(data)}; content afterwards (0 = zeros, dK = the data written for device K): "
        f"{sorted(set(lab))} -> a device-tensor handle addresses the WHOLE mesh buffer: the last write is broadcast to all {rows * cols} devices"
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            torch.zeros(*shape),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        ),
        t,
    )

    # read-back of one shard (snapshot side)
    ttnn.synchronize_device(md)
    t0 = time.perf_counter()
    for _ in range(3):
        ttnn.to_torch(ttnn.get_device_tensors(t)[9])
    log(
        f"C: read-back of one device shard ({N * 512 * 2 / 2**20:.0f} MiB): {1000 * (time.perf_counter() - t0) / 3:.1f} ms"
    )

    # ---- B. the masked idiom (whole-mesh upload sharded over the mesh rows + per-row 0/1 mask, ``gate_user_state``) --------------------------------
    # state of every device first (distinct per row so that "untouched" is meaningful): written with a row-sharded upload
    shard = ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols))
    base = torch.stack([mk()[0] for _ in range(rows)])  # [rows, 1, N, 512]
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(base, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard), t
    )
    before = read_all()
    assert all(same(before[r * cols + c], base[r]) for r in range(rows) for c in range(cols))
    full = torch.zeros(rows, 1, N, 512, dtype=torch.bfloat16)
    full[2] = mk()[0]  # only mesh row 2 carries real data; the other rows' content is a don't-care (zeros)
    mask = torch.zeros(rows, 1, 1, 1)
    mask[2] = 1.0
    mask_d = ttnn.from_torch(mask, device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard)
    addr_b0 = _addr(t)
    for rep in range(3):  # (the first pass compiles the programs; the last is the warm time)
        t0 = time.perf_counter()
        new_d = ttnn.from_torch(
            full,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        ttnn.synchronize_device(md)
        t_up = time.perf_counter() - t0
        t0 = time.perf_counter()
        upd = ttnn.where(mask_d, new_d, t)
        ttnn.copy(upd, t)
        ttnn.synchronize_device(md)
        t_mask = time.perf_counter() - t0
        ttnn.deallocate(upd)
        ttnn.deallocate(new_d)
    after = read_all()
    sel_ok = all(same(after[2 * cols + c], full[2]) for c in range(cols))
    keep_ok = all(same(after[k], before[k]) for k in range(rows * cols) if k // cols != 2)
    log(
        f"B: masked write (tile bf16, {N * 512 * 2 / 2**20:.0f} MiB per device) of one mesh row: bit-exact {sel_ok}, other rows untouched {keep_ok}, buffer address unchanged {addr_b0 == _addr(t)}; "
        f"upload of the whole-mesh new tensor {1000 * t_up:.1f} ms, where+copy {1000 * t_mask:.1f} ms"
    )
    assert sel_ok and keep_ok
    # a captured trace that reads the tensor sees a masked write made after the capture
    ttnn.synchronize_device(md)
    _ = ttnn.add(t, 0.0)
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    out = ttnn.add(t, 0.0)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    full2 = torch.zeros(rows, 1, N, 512, dtype=torch.bfloat16)
    full2[1] = mk()[0]
    mask2 = torch.zeros(rows, 1, 1, 1)
    mask2[1] = 1.0
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(mask2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=shard), mask_d
    )
    new_d = ttnn.from_torch(
        full2,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    upd = ttnn.where(mask_d, new_d, t)
    ttnn.copy(upd, t)
    ttnn.synchronize_device(md)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
    seen = same(ttnn.to_torch(ttnn.get_device_tensors(out)[1 * cols + 3]), full2[1])
    log(f"B: a captured trace reading the tensor sees the masked write made after the capture: {seen}")
    ttnn.release_trace(md, tid)
    assert seen

    # ---- D. row scatter into a row-major table (the pool / kv-table path: only the valid rows move, the other mesh rows are SKIP) ------------------------
    from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P

    tab = ttnn.from_torch(
        torch.zeros(rows, 1, N, 512, dtype=torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    base_t = torch.stack([mk()[0] for _ in range(rows)])
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(base_t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=shard), tab
    )
    before_t = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(tab)]
    for n_valid in (1024, 1024, 6144, 6144, N // 2, N // 2):  # (each size twice: compile, then warm)
        n_valid = min(n_valid, N)
        ids = torch.full((rows, 1, n_valid), 0xFFFFFFFF, dtype=torch.int64)
        ids[3, 0] = torch.arange(n_valid) + 17  # mesh row 3, rows [17, 17 + n_valid)
        data_t = torch.zeros(rows, 1, n_valid, 512, dtype=torch.bfloat16)
        data_t[3, 0] = torch.randn(n_valid, 512, generator=g).to(torch.bfloat16)
        t0 = time.perf_counter()
        d_ids = ttnn.from_torch(
            ids.to(torch.int32),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        d_src = ttnn.from_torch(
            data_t,
            device=md,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard,
        )
        P.paged_scatter_rows(tab, d_src, d_ids)
        ttnn.synchronize_device(md)
        t_sc = time.perf_counter() - t0
        ttnn.deallocate(d_ids)
        ttnn.deallocate(d_src)
        after_t = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(tab)]
        ok = all(same(after_t[3 * cols + c].reshape(-1, 512)[17 : 17 + n_valid], data_t[3, 0]) for c in range(cols))
        keep = all(
            same(after_t[k].reshape(-1, 512)[:17], before_t[k].reshape(-1, 512)[:17])
            and same(after_t[k].reshape(-1, 512)[17 + n_valid :], before_t[k].reshape(-1, 512)[17 + n_valid :])
            for k in range(rows * cols)
            if k // cols == 3
        ) and all(same(after_t[k], before_t[k]) for k in range(rows * cols) if k // cols != 3)
        log(
            f"D: paged_scatter_rows of {n_valid} rows ({n_valid * 1024 / 2**20:.1f} MiB) into one mesh row of a RM table: bit-exact {ok}, everything else untouched {keep}; upload + scatter {1000 * t_sc:.1f} ms"
        )
        assert ok and keep
        before_t = after_t
