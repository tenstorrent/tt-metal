# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-side build of one decode slot's packed conv-history tiles (gdn/tp.py pack_hist_device): random tap rows are
packed on the host with pack_head_tiles_host (the model's reference) for both parities and compared BITWISE with the
device chain (row-broadcast multiply by the channel-spread mask, one HiFi4/fp32-acc 0/1 selection matmul), including
the ttnn.fill_cache write into a [B, Nv*4, 32, 32] view of the packed buffer and the row-slice source path; the layer
method TPGatedDeltaNet._sync_conv_hist_packed_device(slot) (taps=None -> row `slot` of the batched conv_states) on a
model-free stand-in for both Bmax=32 and Bmax=1 batched buffers (at Bmax=1 a full-range ttnn.slice returns its INPUT, so
the method must not free the live taps -- checked by reading them back); then times the per-layer chain.
Model-free.

Geometry is parametric via environment variables (defaults = the TP=4 per-device geometry this test was first
validated at):
    HIST_NV (12)  value heads per device      HIST_NK (4)  key heads per device
    HIST_C  (2*NK*DK + NV*DV = 2560)          per-device qkv width (override only if the model pads it)
    HIST_DK (128) HIST_DV (128) HIST_K (4) HIST_B (32) HIST_LAYERS (48)
    HIST_MESH ("1x4")  mesh shape opened by the test (rows x cols); the taps are sharded over all its devices.
  TP=4 (chips 2-5):  TT_VISIBLE_DEVICES=2,3,4,5 python <this file>
  TP=8 (1x8 mesh, the P150x8 serving geometry Nv=6, Nk=2, C=1280):
                     HIST_NV=6 HIST_NK=2 HIST_C=1280 HIST_MESH=1x8 python <this file>
  (Bmax=1 stand-in:  HIST_B=1 ... -- section 3a needs B >= 2 and is skipped at B=1.)"""
import os
import time
import types

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.gdn.tp import (
    TPGatedDeltaNet,
    hist_pack_consts,
    pack_head_tiles_host,
    pack_hist_device,
    release_hist_pack_consts,
)


# Qwen3.8-27B GDN: 48 value heads / 16 key heads, 128-dim heads, 4 taps. Per device: TP=4 -> Nv=12, Nk=4, C=2560
# (the defaults); TP=8 -> Nv=6, Nk=2, C=1280 (HIST_NV=6 HIST_NK=2 HIST_C=1280 HIST_MESH=1x8).
def _env_int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else int(default)


NV = _env_int("HIST_NV", 12)
NK = _env_int("HIST_NK", 4)
DK = _env_int("HIST_DK", 128)
DV = _env_int("HIST_DV", 128)
K = _env_int("HIST_K", 4)
B = _env_int("HIST_B", 32)
C = _env_int("HIST_C", 2 * NK * DK + NV * DV)
N_LAYERS = _env_int("HIST_LAYERS", 48)
_mesh_rows, _mesh_cols = (int(v) for v in os.environ.get("HIST_MESH", "1x4").lower().split("x"))
MESH_SHAPE = (_mesh_rows, _mesh_cols)
assert NV % NK == 0, f"HIST_NV={NV} must be a multiple of HIST_NK={NK}"
assert C >= 2 * NK * DK + NV * DV and C % 32 == 0, f"HIST_C={C} must be >= 2*NK*DK+NV*DV and tile aligned"
assert B >= 1, "HIST_B must be >= 1"


def shard(mesh, t):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def read(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def host_pack(taps_host, parity):
    """taps_host [n_dev, K, C] -> [n_dev, Nv, 4, 32, 32] via the model's host reference."""
    return torch.stack(
        [
            pack_head_tiles_host([taps_host[d, j] for j in range(K)], NV, NK, DK, DV, parity)
            for d in range(taps_host.shape[0])
        ]
    )


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*MESH_SHAPE), l1_small_size=24576)
    mesh.enable_program_cache()
    n_dev = mesh.get_num_devices()
    logger.info(
        f"geometry: mesh={MESH_SHAPE[0]}x{MESH_SHAPE[1]} ({n_dev} devices) Nv={NV} Nk={NK} Dk={DK} Dv={DV} K={K} "
        f"C={C} B={B} layers={N_LAYERS}"
    )
    torch.manual_seed(0)
    failures = 0
    t0 = time.perf_counter()
    consts = hist_pack_consts(mesh, NV, NK, DK, DV, C, build=True)
    logger.info(f"consts built in {1e3 * (time.perf_counter() - t0):.1f} ms")

    # --- 1. scratch-style taps: K tensors [1, 1, C] per device (the B=1 prefill scratch conv_states) ---
    taps_host = torch.randn(n_dev, K, C) * 3.0
    taps_host[0, 0, :64] = 0.0  # zeros, negatives, large magnitudes, tiny values
    taps_host[1 % n_dev, 2 % K, 100:140] = -1e-3 * torch.arange(40)
    _hi = min(2100, C)
    taps_host[2 % n_dev, (K - 1), _hi - 100 : _hi] = 1e4 * torch.randn(100)
    taps_host = taps_host.to(torch.bfloat16)
    taps_dev = [shard(mesh, taps_host[:, j].reshape(n_dev, 1, C)) for j in range(K)]
    for parity in (0, 1):
        ref = host_pack(taps_host, parity)
        for fused in (False, True):
            packed = pack_hist_device(consts, taps_dev, parity, fused=fused)
            got = read(mesh, packed).view(n_dev, NV, 4, 32, 32)
            ok = torch.equal(got, ref)
            failures += not ok
            nbad = int((got != ref).sum())
            logger.info(
                f"scratch taps parity={parity} fused={fused}: {'OK' if ok else f'BAD ({nbad} elems differ)'} shape={tuple(packed.shape)}"
            )
            ttnn.deallocate(packed)

    # --- 2. fill_cache write into a [B, Nv, 4, 32, 32] packed buffer (other rows untouched) + row-slice source path ---
    hist_host = torch.randn(n_dev * B, NV, 4, 32, 32).to(torch.bfloat16)
    hist = shard(mesh, hist_host)
    conv_host = torch.randn(n_dev, B, C).to(torch.bfloat16)  # K batched taps, [1, B, C] per device as in the model
    convs = [shard(mesh, (conv_host * (1.0 + 0.25 * j)).to(torch.bfloat16).reshape(n_dev, B, C)) for j in range(K)]
    conv_rows = [read(mesh, c).view(n_dev, B, C) for c in convs]  # exact per-device rows as the device holds them
    expect = hist_host.view(n_dev, B, NV, 4, 32, 32).clone()
    _slot_cases = [(0, True), (1, True), (5, False), (B - 2, False), (B - 1, True), (2, False)]
    _slot_cases = [(min(max(sl, 0), B - 1), fu) for sl, fu in _slot_cases]
    for slot, fused in _slot_cases:
        taps = [ttnn.slice(convs[j], (0, slot, 0), (1, slot + 1, C)) for j in range(K)]  # [1, 1, C]
        packed = pack_hist_device(consts, taps, slot & 1, fused=fused)
        for t in taps:
            ttnn.deallocate(t)
        src = ttnn.reshape(packed, (1, NV * 4, 32, 32))
        dst = ttnn.reshape(hist, (B, NV * 4, 32, 32))
        ttnn.fill_cache(dst, src, slot)
        ttnn.deallocate(packed)
        rows_host = torch.stack([conv_rows[j][:, slot] for j in range(K)], dim=1)  # [n_dev, K, C]
        expect[:, slot] = host_pack(rows_host, slot & 1)
        got = read(mesh, hist).view(n_dev, B, NV, 4, 32, 32)
        ok = torch.equal(got, expect)
        failures += not ok
        logger.info(
            f"row-slice source + fill_cache slot={slot} (parity {slot & 1}) fused={fused}: {'OK' if ok else 'BAD'}"
        )

    # --- 3. the layer method itself, taps=None (row `slot` of the batched conv_states), on a model-free stand-in ---
    def stand_in(convs_, hist_):
        o = types.SimpleNamespace(
            mesh=mesh,
            Nv=NV,
            Nk=NK,
            Dk=DK,
            Dv=DV,
            qkv_dim_tp=C,
            K=K,
            conv_states=convs_,
            conv_hist_packed=hist_,
            _hist_packed_valid=False,
            _decode_fused_conv=True,
        )
        for name in ("_slice_along", "_hist_pack_consts", "_sync_conv_hist_packed_device", "warmup_hist_device_pack"):
            setattr(o, name, types.MethodType(getattr(TPGatedDeltaNet, name), o))
        return o

    # 3a. Bmax >= 2: the sliced rows are distinct buffers; the method frees them and leaves conv_states intact.
    # (At HIST_B=1 the batched buffers ARE the B=1 case of 3b; skip.)
    layer = stand_in(convs, hist)
    for slot in (min(3, B - 1), B - 1, 0) if B >= 2 else ():
        layer._sync_conv_hist_packed_device(slot)
        rows_host = torch.stack([conv_rows[j][:, slot] for j in range(K)], dim=1)
        expect[:, slot] = host_pack(rows_host, slot & 1)
        got = read(mesh, hist).view(n_dev, B, NV, 4, 32, 32)
        ok = torch.equal(got, expect)
        taps_ok = all(torch.equal(read(mesh, convs[j]).view(n_dev, B, C), conv_rows[j]) for j in range(K))
        failures += (not ok) + (not taps_ok)
        logger.info(
            f"method taps=None Bmax={B} slot={slot}: packed {'OK' if ok else 'BAD'}, conv_states intact {'OK' if taps_ok else 'BAD'}"
        )

    # 3b. Bmax = 1 (write_slot / pd_transfer.import_gdn_slot / warmup on a B=1 build): a full-range ttnn.slice of the
    # [1, 1, C] tap returns the input buffer itself -- the method must use the taps directly and never free them.
    conv1_host = torch.randn(n_dev, 1, C).to(torch.bfloat16)
    convs1 = [shard(mesh, (conv1_host * (1.0 + 0.5 * j)).to(torch.bfloat16).reshape(n_dev, 1, C)) for j in range(K)]
    conv1_rows = [read(mesh, c).view(n_dev, 1, C) for c in convs1]
    full = ttnn.slice(convs1[0], (0, 0, 0), (1, 1, C))
    same_buf = full.buffer_address() == convs1[0].buffer_address()
    logger.info(
        f"full-range ttnn.slice of a [1,1,C] tap returns the input buffer: {same_buf} (the hazard the guard is for)"
    )
    if not same_buf:
        ttnn.deallocate(full)
    hist1 = shard(mesh, torch.randn(n_dev * 1, NV, 4, 32, 32).to(torch.bfloat16))
    layer1 = stand_in(convs1, hist1)
    for rep in range(3):  # repeated repacks: a freed tap would be reallocated over by now and read back wrong / fault
        assert layer1.warmup_hist_device_pack() is True  # the warm-up hook itself goes through slot 0 with taps=None
        layer1._sync_conv_hist_packed_device(0)
        got1 = read(mesh, hist1).view(n_dev, 1, NV, 4, 32, 32)
        expect1 = host_pack(torch.stack([conv1_rows[j][:, 0] for j in range(K)], dim=1), 0).unsqueeze(1)
        ok = torch.equal(got1, expect1)
        taps_ok = all(torch.equal(read(mesh, convs1[j]).view(n_dev, 1, C), conv1_rows[j]) for j in range(K))
        failures += (not ok) + (not taps_ok)
        logger.info(
            f"method taps=None Bmax=1 rep={rep}: packed {'OK' if ok else 'BAD'}, conv_states intact {'OK' if taps_ok else 'BAD'}"
        )
    # the taps must still be live, distinct buffers (nothing freed by the method)
    scratch = shard(mesh, torch.zeros(n_dev, 1, C).to(torch.bfloat16))  # would land on a freed tap's address
    taps_ok = all(torch.equal(read(mesh, convs1[j]).view(n_dev, 1, C), conv1_rows[j]) for j in range(K))
    failures += not taps_ok
    logger.info(f"Bmax=1 conv_states intact after a fresh allocation: {'OK' if taps_ok else 'BAD'}")
    ttnn.deallocate(scratch)
    for c in convs1:
        ttnn.deallocate(c)
    ttnn.deallocate(hist1)

    # --- 4. timing: one slot repack per GDN layer (constants shared), warm program cache ---
    for fused in (False, True):
        for rep in range(2):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for li in range(N_LAYERS):
                slot = li % B
                packed = pack_hist_device(consts, taps_dev, slot & 1, fused=fused)
                src = ttnn.reshape(packed, (1, NV * 4, 32, 32))
                dst = ttnn.reshape(hist, (B, NV * 4, 32, 32))
                ttnn.fill_cache(dst, src, slot)
                ttnn.deallocate(packed)
            t1 = time.perf_counter()
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
        logger.info(
            f"timing fused={fused}: {N_LAYERS} layers dispatch {1e3 * (t1 - t0):.1f} ms, +sync {1e3 * (t2 - t0):.1f} ms "
            f"({1e3 * (t2 - t0) / N_LAYERS:.2f} ms/layer)"
        )
    # host reference cost for comparison (pack only, no device I/O)
    t0 = time.perf_counter()
    for li in range(N_LAYERS):
        host_pack(taps_host, li & 1)
    logger.info(
        f"host pack_head_tiles reference: {1e3 * (time.perf_counter() - t0) / N_LAYERS:.2f} ms/layer (pack only)"
    )

    for c in convs:
        ttnn.deallocate(c)
    for t in taps_dev:
        ttnn.deallocate(t)
    ttnn.deallocate(hist)
    release_hist_pack_consts()
    ttnn.close_mesh_device(mesh)
    logger.info(f"RESULT: {'ALL EXACT' if failures == 0 else f'{failures} FAILURES'}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
