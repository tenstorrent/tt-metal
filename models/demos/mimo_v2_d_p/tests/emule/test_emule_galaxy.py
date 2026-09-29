# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional (not perf) checks of the MiMo-V2 d_p collectives / attention on an EMULATED BH Galaxy (tt-emule, pure
CPU, no hardware): 32 mock chips (blackhole_galaxy.yaml), opened as an 8x4 mesh (SP8 on rows x TP4 on cols).
Tiny shapes -- the emulator runs every RISC as a CPU fiber.

* test_hbw_all_gather: ttnn.experimental.high_bw_all_gather over cluster_axis 0 (8 rows) and 1 (4 cols), row-major
  x / uint16 indices / tile x, bit-exact vs the host.
* test_ring_sdpa_ksplit: ring joint SDPA over the 8-long SP axis (block-cyclic cache, causal, GA shapes qk 192 / v 128)
  with ring_k_split 1 and 2 (the split merged inside the op by ttnn.transformer.sdpa_k_split_merge) vs an fp32 host
  reference.

* test_rs_ag_axis: ttnn.reduce_scatter + ttnn.all_gather on axis 0 / 1 (the MoE block's > 2-row send-back and its
  > 2-column TP all-reduce).

Results (2026-09-29, tt-metal mstaletovic/mimo-v2-dp ab3dc8c5127): all 4 pass in 3m54s (hbw all-gather bit-exact on
both axes; ring SDPA 8x4 C256 ctx 6144 split 1 / 2 / 3 PCC 0.99995 / 0.99997 / 0.99998 on every chip); the all-gather
MoE block (tests/perf/test_moe_ag_mesh.py, MIMO_AG_MESH=8x4 MIMO_AG_SEQ=32) min PCC 0.999993.
A full real-weights layer (tests/unit/test_decoder_layer.py -k L1-SWA, MIMO_MESH=8x4, the harvested descriptor
below, MIMO_TTNN_CACHE=0) loads and runs attention, and found the Galaxy bugs fixed on this branch (flat_routed_expert
laid out for the p150 11 x 10 grid only -- a Galaxy chip is 12 x 10 with DRAM readers in columns 0, 1, 7, 8 --;
3 CCL links where a Galaxy has 2); it then stops in the flat expert's gate/up compute (se3_compute.cpp) with a
SIGSEGV inside tt-emule's __emule_unpack_tile_to (same with bf8 weights): an emulator gap in that kernel's CB access,
still open. tt-emule-blaze.patch has the emulator shims the flat expert kernels needed to compile
(cb_pages_reservable_at_back, ncrisc_noc_nonposted_writes_sent / _flushed, SFPU_BINARY_INIT_FN, sfpu_binary_init,
5-argument fast_tilize_block / tilize_block honoring the tile indices).

Setup: tt-emule-blaze at 88787b9 (it provides tt-emule::runtime; tt-emule itself is header-only now) with
tt-emule-blaze.patch (here) applied, and tt-metal (plus the emule marshaller commit 56dd501dd1f) built with
    cmake -B build_emule -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake \\
      -DCMAKE_BUILD_TYPE=Release -DTT_METAL_USE_EMULE=ON -DTT_EMULE_PATH=<tt-emule-blaze> -DWITH_PYTHON_BINDINGS=ON \\
      -DENABLE_TRACY=OFF -DENABLE_DISTRIBUTED=ON -DCMAKE_INSTALL_PREFIX=$PWD/build_emule && cmake --build build_emule
    ln -sfn $PWD/build_emule/ttnn/_ttnn.so ttnn/ttnn/_ttnn.so  (+ build_emule/lib symlinks, tt-emule BUILD_GUIDE.md) From the tt-metal root (python_env = a tt-metal venv; env -u PYTHONPATH keeps another checkout out):

    E=<tt-emule-blaze checkout>; R=$PWD
    env -u PYTHONPATH TT_METAL_HOME=$R TT_METAL_RUNTIME_ROOT=$R \\
      PYTHONPATH=$R/ttnn:$R/tools:$R/build_emule/lib:$R LD_LIBRARY_PATH=$R/build_emule/lib \\
      TT_METAL_EMULE_MODE=1 TT_METAL_SLOW_DISPATCH_MODE=1 EMULE_FABRIC8=1 TT_EMULE_FIBER_WORKERS=24 \\
      TT_METAL_MOCK_CLUSTER_DESC_PATH=$E/cluster_descriptors/blackhole_galaxy.yaml \\
      TT_EMULE_JIT_CACHE_DIR=$R/.emule-jit-cache \\
      python_env/bin/python -m pytest -p no:cacheprovider -p no:faulthandler -sv \\
        models/demos/mimo_v2_d_p/tests/emule/test_emule_galaxy.py

The same environment runs the MESH_PARAMS tests on 8x4 with MIMO_MESH=8x4 (tests/mesh.py), e.g.
tests/unit/test_decoder_layer.py, and tests/perf/test_moe_ag_mesh.py with MIMO_AG_MESH=8x4 MIMO_AG_SEQ=32
MIMO_CCL_ITERS=0. -p no:faulthandler: a kernel SIGSEGV otherwise hangs the process in the faulthandler dump.

Knobs: MIMO_EMULE_MESH (8x4), MIMO_EMULE_FABRIC (2d | torus_x | torus_y | torus_xy; default 2d, the fabric the model
opens on the QuietBox), MIMO_EMULE_SEQ (rows per chip for the gathers, 32), MIMO_EMULE_NQ / MIMO_EMULE_NKV (q / kv
heads over the whole TP axis, 16 / 4), MIMO_EMULE_CHUNK_LOCAL (q tokens per chip, 256), MIMO_EMULE_CHUNKS (chunks of
context incl. the current one, 3), MIMO_EMULE_KSPLIT (1,2), MIMO_EMULE_K_CHUNK (128),
MIMO_EMULE_SDPA_GRID (4x2 | full), MIMO_HBW_LINKS (links, default: op discovers).
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    torus_x_device_params,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import MiMoKVCache, _cache_mem
from models.demos.mimo_v2_d_p.tt.attention.sdpa import ring_attention, ring_program_config
from models.demos.mimo_v2_d_p.tt.ccl import CCLManager

_FABRICS = {
    "2d": fabric2d_device_params,
    "torus_x": torus_x_device_params,
    "torus_y": torus_y_device_params,
    "torus_xy": torus_xy_device_params,
}
_MESH = tuple(int(v) for v in os.environ.get("MIMO_EMULE_MESH", "8x4").split("x"))
_FAB = os.environ.get("MIMO_EMULE_FABRIC", "2d")
MESH_PARAMS = pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(_MESH, _FABRICS[_FAB](), id=f"{_MESH[0]}x{_MESH[1]}-{_FAB}")],
    indirect=["mesh_device", "device_params"],
)
LINKS = int(os.environ["MIMO_HBW_LINKS"]) if os.environ.get("MIMO_HBW_LINKS") else None


@pytest.fixture(autouse=True)
def _emule_only():
    assert os.environ.get("TT_METAL_EMULE_MODE") == "1", "emulator-only test: set TT_METAL_EMULE_MODE=1"


def _stats(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (
        torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
        ((a - b).norm() / b.norm()).item(),
        (a.norm() / b.norm()).item(),
    )


@pytest.mark.timeout(7200)
@MESH_PARAMS
def test_hbw_all_gather(mesh_device, device_params):
    rows, cols = tuple(mesh_device.shape)
    S = int(os.environ.get("MIMO_EMULE_SEQ", "32"))
    failures = []
    for axis in (0, 1):
        n = (rows, cols)[axis]
        if n == 1:
            continue
        # per chip (r, c) its S-row slice of a [rows * S] (axis 0) or [cols * S] (axis 1) tensor: dims=(2, None)
        # shards dim 2 over the rows (replicated over the cols), dims=(None, 2) over the cols
        shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(2, None) if axis == 0 else (None, 2))
        cases = [
            ("x_rm", torch.randn(1, 1, n * S, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            ("idx", torch.randint(0, 256, (1, 1, n * S, 8)), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
            ("x_tile", torch.randn(1, 1, n * S, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ]
        for name, t, dt, lay in cases:
            x = ttnn.from_torch(
                t, device=mesh_device, layout=lay, dtype=dt, mesh_mapper=shard, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            out = ttnn.from_torch(
                torch.zeros_like(t),
                device=mesh_device,
                layout=lay,
                dtype=dt,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ref = t if dt != ttnn.bfloat16 else t.bfloat16()
            t0 = time.perf_counter()
            try:
                ttnn.experimental.high_bw_all_gather(x, dim=2, output_tensor=out, cluster_axis=axis, num_links=LINKS)
                ttnn.synchronize_device(mesh_device)
            except Exception as e:  # noqa: BLE001 -- report every case, fail at the end
                failures.append(f"axis{axis} {name}: {str(e).splitlines()[0][:400]}")
                logger.error(f"HBW_FAIL axis{axis} {name}: {e}")
                continue
            wall = time.perf_counter() - t0
            bad = [
                i
                for i, d in enumerate(ttnn.get_device_tensors(out))
                if not torch.equal(ttnn.to_torch(d).to(ref.dtype), ref)
            ]
            if bad:
                failures.append(f"axis{axis} {name}: mismatch on devices {bad}")
            logger.info(f"HBW axis{axis} {name} S{S}: {'OK' if not bad else f'BAD {bad}'} ({wall:.1f} s emulated)")
            for tt_t in (x, out):
                ttnn.deallocate(tt_t)
    assert not failures, failures


@pytest.mark.timeout(14400)
@MESH_PARAMS
def test_ring_sdpa_ksplit(mesh_device, device_params):
    torch.manual_seed(0)
    sp, tp = tuple(mesh_device.shape)
    NQ, NKV = int(os.environ.get("MIMO_EMULE_NQ", "16")), int(os.environ.get("MIMO_EMULE_NKV", "4"))
    DK, DV = 192, 128
    chunk_local = int(os.environ.get("MIMO_EMULE_CHUNK_LOCAL", "256"))
    n_chunks = int(os.environ.get("MIMO_EMULE_CHUNKS", "3"))
    splits = [int(s) for s in os.environ.get("MIMO_EMULE_KSPLIT", "1,2").split(",")]
    chunk = chunk_local * sp
    max_seq = n_chunks * chunk
    kv_actual = max_seq - chunk  # the current chunk is the last one; the prefix is (n_chunks - 1) chunks
    nkv_l = NKV // tp
    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ccl = CCLManager(mesh_device, num_links=int(os.environ.get("MIMO_NUM_LINKS", "1")), topology=sp_topo)

    def cache(d):  # the same random cache on every chip
        return ttnn.from_torch(
            torch.randn(1, nkv_l, max_seq // sp, d),
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_cache_mem(mesh_device, d),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    kv = MiMoKVCache(cache(DK), cache(DV), 1, 1, max_seq, sp, nkv_l, DK, DV)
    q_host = torch.randn(1, NQ, chunk, DK)
    q = ttnn.from_torch(
        q_host,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(2, 1)),
    )
    scale = DK**-0.5
    # k chunk: the op needs a K chunk per partition in every ring iteration -> kv_actual / sp >= k_split * k_chunk
    k_chunk = int(os.environ.get("MIMO_EMULE_K_CHUNK", "128"))
    q_chunk = min(128, chunk_local)
    # SDPA grid (MIMO_EMULE_SDPA_GRID = XxY, "full" = the model's grid.x - 1 by grid.y). The K split needs >= 2 work
    # units ((head, Q chunk) x k_split) per core, so tiny shapes need a small grid: 4x2 by default.
    g = os.environ.get("MIMO_EMULE_SDPA_GRID", "4x2")
    grid = None if g == "full" else tuple(int(v) for v in g.split("x"))
    assert kv_actual // sp >= max(splits) * k_chunk, (kv_actual // sp, max(splits), k_chunk)

    # fp32 host reference, every chip: chip (r, c) holds q rows r * chunk_local .. of the chunk (block-cyclic: chunk
    # position i -> global kv_actual + r * chunk_local + i) and q heads c * NQ / tp ..; keys at global position p live
    # on cache row (p // chunk) * chunk_local + p % chunk_local of chip p % chunk // chunk_local (same random cache on
    # every chip -> row index only)
    k_all = ttnn.to_torch(ttnn.get_device_tensors(kv.k)[0]).float()[0]
    v_all = ttnn.to_torch(ttnn.get_device_tensors(kv.v)[0]).float()[0]
    n_keys = kv_actual + chunk
    pos = torch.arange(n_keys)
    krow = (pos // chunk) * chunk_local + pos % chunk_local
    nq_l, g = NQ // tp, (NQ // tp) // nkv_l
    ref = torch.empty(sp, tp, nq_l, chunk_local, DV)
    for r in range(sp):
        qpos = kv_actual + r * chunk_local + torch.arange(chunk_local)
        mask = pos[None, :] <= qpos[:, None]
        for c in range(tp):
            qh = q_host[0, c * nq_l : (c + 1) * nq_l, r * chunk_local : (r + 1) * chunk_local].bfloat16().double()
            for h in range(nq_l):
                sc = (qh[h] @ k_all[h // g, krow].double().T) * scale
                sc = sc.masked_fill(~mask, float("-inf"))
                ref[r, c, h] = (torch.softmax(sc, -1) @ v_all[h // g, krow].double()).float()

    outs, failures = {}, []
    for s in splits:
        pc = ring_program_config(mesh_device, q_chunk=q_chunk, k_chunk=k_chunk, k_split=s)
        if grid is not None:  # a smaller SDPA grid (the CCL column stays at the device grid's last column)
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=pc.exp_approx_mode,
                ring_k_split=s,
            )
        t0 = time.perf_counter()
        try:
            o = ring_attention(
                q,
                kv,
                kv_actual=kv_actual,
                logical_n=n_keys,
                window=None,
                sink=None,
                layer_slot=0,
                mesh_device=mesh_device,
                ccl_manager=ccl,
                sp_axis=0,
                scale=scale,
                program_config=pc,
                k_split=s,
            )
            ttnn.synchronize_device(mesh_device)
        except Exception as e:  # noqa: BLE001
            failures.append(f"k_split {s}: {str(e).splitlines()[0][:400]}")
            logger.error(f"SDPA_FAIL k_split {s}: {e}")
            continue
        wall = time.perf_counter() - t0
        got = torch.stack([ttnn.to_torch(t).float()[0] for t in ttnn.get_device_tensors(o)]).reshape(
            sp, tp, nq_l, chunk_local, DV
        )
        outs[s] = got
        worst = min(_stats(got[r, c], ref[r, c])[0] for r in range(sp) for c in range(tp))
        p, rel, nr = _stats(got, ref)
        logger.info(
            f"SDPA k_split {s} mesh {sp}x{tp} C{chunk_local} ctx{n_keys}: PCC {p:.6f} (worst chip {worst:.6f}) "
            f"rel {rel:.2e} norm ratio {nr:.5f} ({wall:.1f} s emulated)"
        )
        if worst < 0.99:
            failures.append(f"k_split {s}: worst chip PCC {worst:.6f}")
        o.deallocate(True)
    for s in splits[1:]:
        if s in outs and splits[0] in outs:
            p, rel, nr = _stats(outs[s], outs[splits[0]])
            logger.info(f"SDPA k_split {s} vs {splits[0]}: PCC {p:.6f} rel {rel:.2e} norm ratio {nr:.5f}")
    assert not failures, failures


@pytest.mark.timeout(7200)
@MESH_PARAMS
@pytest.mark.parametrize("axis", [0, 1])
def test_rs_ag_axis(mesh_device, device_params, axis):
    """ttnn.reduce_scatter + ttnn.all_gather on one mesh axis (the MoE block's > 2-row send-back uses axis 0, its
    > 2-column TP all-reduce "rsag" axis 1), bf16 tiles [1, 1, S, 4096] per chip, vs host sums."""
    rows, cols = tuple(mesh_device.shape)
    S = int(os.environ.get("MIMO_EMULE_SEQ", "32"))
    n = (rows, cols)[axis]
    topo = per_axis_topology(device_params["fabric_config"])[axis]
    R = S * n if axis == 0 else S  # rows per chip: the scattered dim must split into whole tiles
    x_host = torch.randn(rows, cols, R, 4096)
    x = ttnn.from_torch(
        x_host,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    dim = 3 if axis == 1 else 2
    t0 = time.perf_counter()
    rs = ttnn.reduce_scatter(x, dim=dim, cluster_axis=axis, topology=topo)
    ttnn.synchronize_device(mesh_device)
    t1 = time.perf_counter()
    ag = ttnn.all_gather(rs, dim=dim, cluster_axis=axis, topology=topo)
    ttnn.synchronize_device(mesh_device)
    t2 = time.perf_counter()
    xb = x_host.bfloat16().float()
    ref = xb.sum(dim=axis, keepdim=True)  # summed over the axis, per orthogonal index
    worst = 1.0
    for i, d in enumerate(ttnn.get_device_tensors(ag)):
        r, c = divmod(i, cols)
        want = ref[0 if axis == 0 else r, 0 if axis == 1 else c]
        got = ttnn.to_torch(d).float().reshape(R, 4096)
        worst = min(worst, _stats(got, want)[0])
    logger.info(
        f"RSAG axis{axis} ({n} chips, {topo}): reduce_scatter {t1 - t0:.1f} s + all_gather {t2 - t1:.1f} s "
        f"emulated, worst chip PCC {worst:.6f}"
    )
    assert worst > 0.999, worst
