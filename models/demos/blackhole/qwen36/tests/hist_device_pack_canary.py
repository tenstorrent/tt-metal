# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Write-footprint check of the device-side conv-history repack (gdn/tp.py pack_hist_device + fill_cache into the
[B, Nv*4, 32, 32] view of conv_hist_packed): does repacking EVERY slot, unsynced and repeatedly, touch any byte outside
the packed buffer's own row? Canary tensors (page-table-like int32, position-like int32, KV-like bf16, plus plain bf16
tiles) are allocated before and after the packed buffer and after the first repacks (post-capture allocations), and every
canary and every packed row is read back and compared after each repetition. Also exercises the served mode-2 path
(taps = a B=1 scratch [1, 1, C] per tap) and the slice path (taps = row `slot` of the batched conv_states).

Geometry via HIST_* like hist_device_pack_scratch.py; TP=8 serving geometry: HIST_NV=6 HIST_NK=2 HIST_C=1280 HIST_MESH=1x8
(the per-device shapes are what matter; HIST_MESH=1x4 with TT_VISIBLE_DEVICES=2,3,4,5 checks the same per-device chain on
four chips). HIST_REPS (5) repetitions of the all-slot repack."""
import os
import time
import types

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, hist_pack_consts, pack_head_tiles_host

_e = lambda n, d: int(os.environ.get(n) or d)
NV, NK, DK, DV, K, B = (
    _e("HIST_NV", 12),
    _e("HIST_NK", 4),
    _e("HIST_DK", 128),
    _e("HIST_DV", 128),
    _e("HIST_K", 4),
    _e("HIST_B", 32),
)
C = _e("HIST_C", 2 * NK * DK + NV * DV)
REPS = _e("HIST_REPS", 5)
MESH_SHAPE = tuple(int(v) for v in os.environ.get("HIST_MESH", "1x4").lower().split("x"))


def shard(mesh, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def read(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def host_pack(rows, parity):  # rows [n_dev, K, C] -> [n_dev, NV, 4, 32, 32]
    return torch.stack(
        [pack_head_tiles_host([rows[d, j] for j in range(K)], NV, NK, DK, DV, parity) for d in range(rows.shape[0])]
    )


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), l1_small_size=24576)
    mesh.enable_program_cache()
    n = mesh.get_num_devices()
    logger.info(f"mesh {MESH_SHAPE} ({n} devices) Nv={NV} Nk={NK} C={C} B={B} K={K} reps={REPS}")
    torch.manual_seed(1)
    consts = hist_pack_consts(mesh, NV, NK, DK, DV, C, build=True)
    canaries = []  # (name, tensor, host copy)

    def canary(name, host, dtype, layout):
        t = shard(mesh, host, dtype, layout)
        canaries.append((name, t, read(mesh, t).clone()))
        logger.info(f"canary {name}: shape {tuple(t.shape)} addr 0x{t.buffer_address():x}")

    canary("pt_before", torch.randint(0, 4096, (n * 32, 4096), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    canary("pos_before", torch.randint(0, 200000, (n * 32, 32), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    canary("kv_before", torch.randn(n * 64, 1, 64, 256), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    hist_host = torch.randn(n * B, NV, 4, 32, 32).to(torch.bfloat16)
    hist = shard(mesh, hist_host)
    logger.info(f"conv_hist_packed: shape {tuple(hist.shape)} addr 0x{hist.buffer_address():x}")
    canary("tiles_after", torch.randn(n * 4, 24, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    canary("pt_after", torch.randint(0, 4096, (n * 32, 4096), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    canary("pos_after", torch.randint(0, 200000, (n * 32, 32), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    conv_host = torch.randn(n, B, C).to(torch.bfloat16)
    convs = [shard(mesh, (conv_host * (1.0 + 0.25 * j)).to(torch.bfloat16).reshape(n, B, C)) for j in range(K)]
    conv_rows = [read(mesh, c).view(n, B, C) for c in convs]
    scratch_host = torch.randn(n, 1, C).to(torch.bfloat16)
    scratch = [shard(mesh, (scratch_host * (1.0 + 0.5 * j)).to(torch.bfloat16).reshape(n, 1, C)) for j in range(K)]
    scratch_rows = [read(mesh, s).view(n, 1, C) for s in scratch]
    layer = types.SimpleNamespace(
        mesh=mesh,
        Nv=NV,
        Nk=NK,
        Dk=DK,
        Dv=DV,
        qkv_dim_tp=C,
        K=K,
        conv_states=convs,
        conv_hist_packed=hist,
        _hist_packed_valid=False,
        _decode_fused_conv=True,
    )
    for name in ("_slice_along", "_hist_pack_consts", "_sync_conv_hist_packed_device"):
        setattr(layer, name, types.MethodType(getattr(TPGatedDeltaNet, name), layer))
    expect = hist_host.view(n, B, NV, 4, 32, 32).clone()
    failures = 0
    for rep in range(REPS):
        t0 = time.perf_counter()
        for slot in range(B):
            if rep % 2 == 0:  # served mode-2 path: the B=1 scratch taps
                layer._sync_conv_hist_packed_device(slot, taps=scratch)
                expect[:, slot] = host_pack(torch.stack([scratch_rows[j][:, 0] for j in range(K)], dim=1), slot & 1)
            else:  # row-slice path
                layer._sync_conv_hist_packed_device(slot)
                expect[:, slot] = host_pack(torch.stack([conv_rows[j][:, slot] for j in range(K)], dim=1), slot & 1)
        if rep == 0:  # post-first-repack allocations, as buffers allocated after the warm-up would sit
            canary("late_tiles", torch.randn(n * 2, 24, 32, 32), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            canary(
                "late_pt", torch.randint(0, 4096, (n * 32, 4096), dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
            )
        ttnn.synchronize_device(mesh)
        dt = time.perf_counter() - t0
        got = read(mesh, hist).view(n, B, NV, 4, 32, 32)
        ok_rows = torch.equal(got, expect)
        bad = (
            []
            if ok_rows
            else [
                (s, int((got[:, s] != expect[:, s]).sum()))
                for s in range(B)
                if not torch.equal(got[:, s], expect[:, s])
            ]
        )
        cana_bad = []
        for name, t, ref in canaries:
            cur = read(mesh, t)
            if not torch.equal(cur, ref):
                cana_bad.append((name, int((cur != ref).sum())))
        taps_ok = all(torch.equal(read(mesh, convs[j]).view(n, B, C), conv_rows[j]) for j in range(K)) and all(
            torch.equal(read(mesh, scratch[j]).view(n, 1, C), scratch_rows[j]) for j in range(K)
        )
        failures += (not ok_rows) + len(cana_bad) + (not taps_ok)
        logger.info(
            f"rep {rep} ({'scratch taps' if rep % 2 == 0 else 'row slices'}, {B} slots in {1e3*dt:.0f} ms): "
            f"packed rows {'OK' if ok_rows else 'BAD ' + str(bad[:6])}; canaries {'OK' if not cana_bad else 'CORRUPTED ' + str(cana_bad)}; "
            f"taps intact {'OK' if taps_ok else 'BAD'}"
        )
    logger.info(f"RESULT: {'PASS' if failures == 0 else f'FAIL ({failures})'}")
    ttnn.close_mesh_device(mesh)
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
