# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cost of the hidden-dim sharded embedding (DSV41_EMBED_SHARD) by variant, 4x8 mesh, traced us per call. Prints EDB lines.

Variants (T = tokens per mesh row): rep (replicated table, tile / row-major output) | local lookup only (tile / rm) | + sync ttnn.all_gather | + all_gather_async (default / lean semaphores, tile / row-major).
Bit-identity of every gathered variant against the replicated lookup on all 32 devices."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_embedding_dshard import (
    COLS,
    DIM,
    ROWS,
    VOCAB,
    dram_mib,
    load_table,
    traced_us,
    upload,
)
from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

MESH = [pytest.param((4, 8), id="4x8")]
DP = [
    pytest.param(
        {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000},
        id="ring",
    )
]


@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
@pytest.mark.parametrize("device_params", DP, indirect=True)
@torch.no_grad()
def test_embedding_dshard_bench(mesh_device):
    md = mesh_device
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    ccl_lean.make_lean(mc)
    ccl_lean.make_lean_sems(ccl)
    w = load_table()
    rep = ttnn.from_torch(
        w,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    sh = ttnn.from_torch(
        w,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(ROWS, COLS)),
    )
    bf = ttnn.bfloat16

    def emb(table, tok, lay):
        return ttnn.embedding(tok, table, layout=lay, dtype=bf)

    def ag_sync(x):
        return ttnn.all_gather(x, dim=len(x.shape) - 1, cluster_axis=1, num_links=1, topology=ttnn.Topology.Ring)

    def ag_async(x, lean):  # x [1,1,T,640] (dim 3)
        return mc.allgather(x, ccl, axis=1, dim=3, lean_ag=lean)

    def rm4(x, T):  # [T,1,640] rm -> [1,1,T,640] rm
        return ttnn.reshape(x, [1, 1, T, x.shape[-1]])

    g = torch.Generator().manual_seed(0)
    for n in (1, 16, 24, 2048):
        T = n
        ids = torch.randint(0, VOCAB, (ROWS * n,), generator=g)
        tok = upload(md, ids)
        reps = 10 if n <= 24 else 3
        ref = [ttnn.to_torch(x).reshape(-1, DIM) for x in ttnn.get_device_tensors(emb(rep, tok, ttnn.ROW_MAJOR_LAYOUT))]
        variants = {
            "rep_tile": lambda: emb(rep, tok, ttnn.TILE_LAYOUT),
            "rep_rm": lambda: emb(rep, tok, ttnn.ROW_MAJOR_LAYOUT),
            "local_tile": lambda: emb(sh, tok, ttnn.TILE_LAYOUT),
            "local_rm": lambda: emb(sh, tok, ttnn.ROW_MAJOR_LAYOUT),
            "sync_tile": lambda: ag_sync(emb(sh, tok, ttnn.TILE_LAYOUT)),
            "async_rm_default": lambda: ag_async(rm4(emb(sh, tok, ttnn.ROW_MAJOR_LAYOUT), T), False),
            "async_rm_lean": lambda: ag_async(rm4(emb(sh, tok, ttnn.ROW_MAJOR_LAYOUT), T), True),
            "async_tile_lean": lambda: ag_async(
                ttnn.reshape(emb(sh, tok, ttnn.TILE_LAYOUT), [1, 1, T, DIM // COLS]), True
            ),
        }
        out = {}
        for name, fn in variants.items():
            try:
                y = fn()
                ttnn.synchronize_device(md)
                ok = ""
                if name.startswith(("sync", "async")):
                    got = [ttnn.to_torch(x).reshape(-1, DIM) for x in ttnn.get_device_tensors(y)]
                    ok = " identical=" + str(all(torch.equal(a, b) for a, b in zip(ref, got)))
                out[name] = (traced_us(md, fn, reps=reps), ok)
            except Exception as e:  # noqa: BLE001
                out[name] = (float("nan"), f" FAILED {type(e).__name__}: {str(e)[:120]}")
        print(
            f"EDB n/row={n}: " + " | ".join(f"{k} {v[0]:.1f}{v[1]}" for k, v in out.items()),
            flush=True,
        )


@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
@pytest.mark.parametrize("device_params", DP, indirect=True)
@torch.no_grad()
def test_embedding_streams_bench(mesh_device):
    """what DSV41DeviceEmbedding.forward returns ([T,1,4,D] fp32 streams) by variant: S0 current (tile lookup, typecast, repeat), S1 row-major lookup + concat + to_layout, S2 = S1 on the sharded table + lean all_gather_async."""
    md = mesh_device
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    ccl_lean.make_lean(mc)
    ccl_lean.make_lean_sems(ccl)
    w = load_table()
    mk = lambda mapper: ttnn.from_torch(
        w,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    rep = mk(ttnn.ReplicateTensorToMesh(md))
    sh = mk(ttnn.ShardTensor2dMesh(md, dims=(None, 1), mesh_shape=(ROWS, COLS)))
    bf, f32 = ttnn.bfloat16, ttnn.float32

    def s0(tok, T):
        e = ttnn.embedding(tok, rep, layout=ttnn.TILE_LAYOUT, dtype=bf)
        return ttnn.repeat(ttnn.typecast(ttnn.reshape(e, [T, 1, 1, DIM]), f32), [1, 1, 4, 1])

    def s1(tok, T):
        e = ttnn.reshape(ttnn.embedding(tok, rep, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=bf), [T, 1, 1, DIM])
        return ttnn.to_layout(ttnn.concat([e, e, e, e], dim=2), ttnn.TILE_LAYOUT, dtype=f32)

    def s2(tok, T):
        e = ttnn.reshape(ttnn.embedding(tok, sh, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=bf), [1, 1, T, DIM // COLS])
        e = ttnn.reshape(mc.allgather(e, ccl, axis=1, dim=3, lean_ag=True), [T, 1, 1, DIM])
        return ttnn.to_layout(ttnn.concat([e, e, e, e], dim=2), ttnn.TILE_LAYOUT, dtype=f32)

    def rmE(tok, T):
        return ttnn.reshape(ttnn.embedding(tok, rep, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=bf), [T, 1, 1, DIM])

    def s1b(tok, T):  # tilize to fp32 once, repeat in tile layout
        return ttnn.repeat(ttnn.to_layout(rmE(tok, T), ttnn.TILE_LAYOUT, dtype=f32), [1, 1, 4, 1])

    def s1c(tok, T):  # repeat row-major, then tilize
        return ttnn.to_layout(ttnn.repeat(rmE(tok, T), [1, 1, 4, 1]), ttnn.TILE_LAYOUT, dtype=f32)

    def s1e(tok, T):  # fp32 row-major first, concat, tilize
        e = ttnn.typecast(rmE(tok, T), f32)
        return ttnn.to_layout(ttnn.concat([e, e, e, e], dim=2), ttnn.TILE_LAYOUT)

    def s1f(tok, T):  # one lookup of 4T rows (index repeat) -> [T,1,4,D]
        t4 = ttnn.reshape(ttnn.repeat(ttnn.reshape(tok, [1, T, 1, 1]), [1, 1, 4, 1]), [T * 4, 1]) if False else None
        raise RuntimeError("skipped")

    g = torch.Generator().manual_seed(1)
    for n in (1, 16, 24, 2048):
        tok = upload(md, torch.randint(0, VOCAB, (ROWS * n,), generator=g))
        ref = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(s0(tok, n))]
        res = {}
        for name, fn in (
            ("S0_current", s0),
            ("S1_rm_rep", s1),
            ("S1b_tile_f32_repeat", s1b),
            ("S1c_rm_repeat", s1c),
            ("S1e_f32_rm_concat", s1e),
            ("S2_rm_sharded_lean", s2),
        ):
            try:
                y = fn(tok, n)
                ttnn.synchronize_device(md)
                got = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(y)]
                same = all(torch.equal(a, b) for a, b in zip(ref, got))
                res[name] = f"{traced_us(md, lambda: fn(tok, n), reps=10 if n <= 24 else 2):.1f} identical={same}"
            except Exception as e:  # noqa: BLE001
                res[name] = f"FAILED {type(e).__name__}: {str(e)[:100]}"
        print(f"EDB streams n/row={n}: " + " | ".join(f"{k} {v}" for k, v in res.items()), flush=True)


@pytest.mark.parametrize("mesh_device", MESH, indirect=True)
@pytest.mark.parametrize("device_params", DP, indirect=True)
@torch.no_grad()
def test_embedding_class(mesh_device, monkeypatch):
    """the real DSV41DeviceEmbedding.forward in three modes (default, DSV41_EMBED_RM=1, DSV41_EMBED_SHARD=1 + set_ccl): bit-identical streams, traced us, DRAM."""
    from models.demos.blackhole.deepseek_v41_flash.tt import device_head as DH

    md = mesh_device
    mc, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    ccl_lean.make_lean(mc)
    ccl_lean.make_lean_sems(ccl)
    w = load_table()
    embs = {}
    for name, env in (("default", {}), ("rm", {"DSV41_EMBED_RM": "1"}), ("shard", {"DSV41_EMBED_SHARD": "1"})):
        for k in ("DSV41_EMBED_RM", "DSV41_EMBED_SHARD"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        d0 = dram_mib(md)
        e = DH.DSV41DeviceEmbedding(md, w, users_per_row=4)
        if hasattr(e.weight, "set_ccl"):
            e.weight.set_ccl(mc, ccl)
        print(f"EDB class {name}: table DRAM {dram_mib(md) - d0:.1f} MiB/bank", flush=True)
        embs[name] = (e, dict(env))
    g = torch.Generator().manual_seed(2)
    for n in (1, 16, 24, 2048):
        tok = upload(md, torch.randint(0, VOCAB, (ROWS * n,), generator=g))
        ref = None
        line = []
        for name, (e, env) in embs.items():
            for k in ("DSV41_EMBED_RM", "DSV41_EMBED_SHARD"):
                monkeypatch.delenv(k, raising=False)
            for k, v in env.items():
                monkeypatch.setenv(k, v)
            s, _ = e.forward(tok)
            ttnn.synchronize_device(md)
            got = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(s)]
            if ref is None:
                ref = got
            same = all(torch.equal(a, b) for a, b in zip(ref, got))
            line.append(
                f"{name} {traced_us(md, lambda: e.forward(tok), reps=10 if n <= 24 else 2):.1f} identical={same}"
            )
            assert same
        print(f"EDB class n/row={n}: " + " | ".join(line), flush=True)
