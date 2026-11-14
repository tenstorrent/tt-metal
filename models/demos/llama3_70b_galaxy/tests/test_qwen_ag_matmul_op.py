# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""custom_ops/qwen_ag_matmul against the stock pair of the context-parallel prefill (qwen_cp_prefill.py):
_row_ag (all_gather_async along dim 2 over the mesh rows) + ttnn.experimental.minimal_matmul with the model's config.

test_qwen_ag_matmul_plan: host only; every device's block lists cover each gathered row once, equal list lengths.
test_qwen_ag_matmul_check_ring: host only; the 8-row FABRIC_1D_RING check raises ValueError otherwise.
test_qwen_ag_matmul_release_on_error: host only; a prefill that raises still frees the fused ops' words.
test_qwen_ag_matmul: on the 8x4 Galaxy (FABRIC_1D_RING), per kind (QKV: N=1280, FF1: N=3200 with copy_local) and
    sequence length:
    1. matmul_port (the 6x10 matmul with the planned block lists, no fabric) on the stock gathered buffer:
       torch.equal to the stock minimal_matmul on all 32 devices.
    2. qwen_ag_matmul on zero-filled gathered / output buffers, twice with fresh buffers (descriptor cache hit):
       the output torch.equal to the stock pair on all 32 devices, every gathered slot the op writes (all slots with
       copy_local, else the remote ones) torch.equal to the stock gather.
    3. With QAGMM_TIME=1 only: pipelined wall time per call (2 warm-ups, REPS x BATCH back-to-back calls, one sync per
       batch, median) of the stock pair and of the fused op. Printed as QAGMM_TIME lines; never asserted.
    Inputs come from a fixed seed (SEED), so a mismatch reproduces.
    QAGMM_SEQS / QAGMM_KINDS (comma lists) narrow the sweep; QAGMM_WAVES (e.g. 8,8,8,8,8; ';' separates sets)
    forces the wave sizes; QAGMM_V2=1 tests qwen_ag_matmul_v2 (7x10 grid, senders in the in0 receivers) instead;
    QAGMM_PROBES=1 (with QAGMM_TIME=1) adds timing probes (port, nowait, and for v1 nomm / agonly).

Run on the Wormhole Galaxy (8x4 mesh):
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_ag_matmul_op.py -s
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tt import qwen_cp_prefill as cp
from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_ag_matmul import (
    check_plan,
    check_ring,
    matmul_port,
    plan,
    plan_v2,
    qwen_ag_matmul,
    qwen_ag_matmul_v2,
    resident_fits,
    v2_waves,
)

ROWS, COLS = 8, 4
SEQS = (16384, 24576, 32768, 40960)
FABRIC_PARAMS = [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": True, "worker_l1_size": 1345000}]
K = 5120
SEED = 0


def _kinds(margs):
    lofi16 = cp.lofi_fp32_config(margs.compute_kernel_config_hifi2, "QWEN_CP_ATTN_MM_LOFI", fp16=True)
    return {
        "qkv": dict(N=1280, config=cp.QKV_FP16_CFG, ckc=lofi16, copy_local=False),
        "ff1": dict(N=3200, config=cp.FF13_CFG, ckc=margs.compute_kernel_config_lofi, copy_local=True),
    }


@pytest.mark.host_only
def test_qwen_ag_matmul_plan():
    """Slot coverage: on every device, each gathered row (every slot, own and remote) is in exactly one block of one
    column; resident plans start every column on its own slot; the QKV shape is resident."""
    assert resident_fits(K, 1280, cp.QKV_FP16_CFG) and resident_fits(K, 1280, cp.QKV_CFG)
    assert not resident_fits(K, 3200, cp.FF13_CFG)
    for S in SEQS + (17408, 20480, 28672, 36864):
        for N, resident in ((1280, True), (1280, False), (3200, False)):
            p = plan(S // 32 // 32, 16, N, resident=resident)
            check_plan(p)
            logger.info(f"S={S} N={N} resident={resident} waves {p.waves} blocks/column {[len(c) for c in p.cols[0]]}")
        # v2 (7x10): column x = a piece of the own slot, then every wave of remote slot (r + 1 + x) % 8
        T = S // 32 // 32
        for resident in (False, True):
            for r in range(ROWS):
                waves, _, cols = plan_v2(T, r, waves=v2_waves(T, resident))
                cover = [0] * (ROWS * T)
                for x, col in enumerate(cols):
                    assert col[0][2] == 0xFFFF and all(m0 // T == (r + 1 + x) % ROWS for m0, _, _ in col[1:])
                    for m0, rows, _ in col:
                        for m in range(m0, m0 + rows):
                            cover[m] += 1
                assert all(v == 1 for v in cover), (S, r, resident)


@pytest.mark.host_only
def test_qwen_ag_matmul_check_ring(monkeypatch, expect_error):
    """Host only: check_ring passes on 8 rows + FABRIC_1D_RING and raises ValueError on 4 rows or on FABRIC_1D."""

    class _Mesh:
        def __init__(self, shape):
            self.shape = shape

    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: ttnn.FabricConfig.FABRIC_1D_RING)
    check_ring(_Mesh((ROWS, COLS)))
    with expect_error(ValueError, "mesh rows"):
        check_ring(_Mesh((COLS, ROWS)))
    monkeypatch.setattr(ttnn, "get_fabric_config", lambda: ttnn.FabricConfig.FABRIC_1D)
    with expect_error(ValueError, "FABRIC_1D_RING"):
        check_ring(_Mesh((ROWS, COLS)))


@pytest.mark.host_only
def test_qwen_ag_matmul_release_on_error(monkeypatch, expect_error):
    """Host only: model_prefill and layer_major_prefill free the fused ops' words when a layer raises."""
    freed = []

    def boom(*a, **k):
        raise RuntimeError("layer failed")

    class _H:
        shape = (1, 1, 32768 // (ROWS * COLS), 1280)

        def deallocate(self, *a):
            pass

    class _Layer:
        attention = feed_forward = None

    class _Model:
        tt_ccl, mesh_device, layers = None, "mesh", [_Layer()]

    monkeypatch.setattr(cp, "get_context", lambda *a: None)
    monkeypatch.setattr(cp, "entry", lambda *a: _H())
    monkeypatch.setattr(cp, "gather_layer_weights", lambda *a: None)
    monkeypatch.setattr(cp, "decoder_prefill", boom)
    monkeypatch.setattr(cp, "release_fused_ag", freed.append)
    with expect_error(RuntimeError, "layer failed"):
        cp.model_prefill(_Model(), _H(), None)
    with expect_error(RuntimeError, "layer failed"):
        cp.layer_major_prefill(_Model(), [dict(h=_H(), user_id=0, page_table=None, last_idx=0)], None, None)
    assert freed == ["mesh", "mesh"]


def _host(t):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _zeros(mesh, shape):
    return ttnn.from_torch(
        torch.zeros(shape),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _timeit(mesh, run, reps, batch):
    for _ in range(2):
        for t in run():
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    out = []
    for _ in range(reps):
        held = []
        t0 = time.perf_counter()
        for _ in range(batch):
            held.append(run())
        ttnn.synchronize_device(mesh)
        out.append((time.perf_counter() - t0) * 1e3 / batch)
        for ts in held:
            for t in ts:
                ttnn.deallocate(t)
    return statistics.median(out), out


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("device_params", FABRIC_PARAMS, indirect=True)
def test_qwen_ag_matmul(mesh_device):
    from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
    from models.demos.llama3_70b_galaxy.tt.model_config import LlamaOptimizations
    from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
    from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs

    mesh = mesh_device
    seqs = [int(s) for s in os.environ.get("QAGMM_SEQS", ",".join(map(str, SEQS))).split(",")]
    kinds = os.environ.get("QAGMM_KINDS", "qkv,ff1").split(",")
    wave_sets = [None]
    if os.environ.get("QAGMM_WAVES"):
        wave_sets = [tuple(int(w) for w in ws.split(",")) for ws in os.environ["QAGMM_WAVES"].split(";")]
    res_env = os.environ.get("QAGMM_RESIDENT")  # unset: the op's choice (resident when the K x N block fits)
    split = int(os.environ.get("QAGMM_SPLIT", "0"))
    reps, batch = int(os.environ.get("QAGMM_REPS", "3")), int(os.environ.get("QAGMM_BATCH", "8"))
    S_max = max(seqs)
    margs = TtQwenModelArgs(mesh, max_batch_size=32, max_seq_len=S_max, optimizations=LlamaOptimizations.performance)
    mc = margs.get_model_config()
    pf = TtLlamaPrefetcherSetup(mesh, n_tensors=0, n_layers=1, mode="prefill", save_tensor_addresses=True, is_qwen=True)
    mesh.set_sub_device_stall_group([pf.worker_sub_device_id])
    ccl = TT_CCL(mesh, margs, pf.worker_sub_device_id, mode="prefill", is_qwen=True)
    KINDS = _kinds(margs)
    g = torch.Generator().manual_seed(SEED)
    logger.info(f"QAGMM inputs seed {SEED}")
    failures = []
    for kind in kinds:
        kd = KINDS[kind]
        N = kd["N"]
        wh = torch.randn(ROWS, 1, K, N, generator=g).to(torch.bfloat16)
        w = ttnn.from_torch(
            wh,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, None), mesh_shape=(ROWS, COLS)),
        )
        mm_kw = dict(config=kd["config"], compute_kernel_config=kd["ckc"], dtype=ttnn.bfloat8_b)
        v2 = os.environ.get("QAGMM_V2") == "1"

        def fused(x, w, copy_local, waves=None, probe=None, gathered=None, out=None, resident=None, split=0, **kw):
            if v2:
                return qwen_ag_matmul_v2(
                    x,
                    w,
                    copy_local=copy_local,
                    gathered=gathered,
                    out=out,
                    resident=resident,
                    probe=probe,
                    waves=waves,
                    **kw,
                )
            return qwen_ag_matmul(
                x,
                w,
                copy_local=copy_local,
                waves=waves,
                probe=probe,
                gathered=gathered,
                out=out,
                resident=resident,
                split=split,
                **kw,
            )

        rk = dict(resident=None if res_env is None else res_env == "1", split=split)
        if rk["resident"] is None:
            rk["resident"] = resident_fits(K, N, kd["config"])

        def stock(x, S):
            xa = cp._row_ag(ccl, mc, x, S)
            o = ttnn.experimental.minimal_matmul(
                input_tensor=xa, weight_tensor=w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **mm_kw
            )
            return xa, o

        for S, waves in [(S, ws) for S in seqs for ws in wave_sets]:
            T = S // 32 // 32
            p = plan(T, kd["config"].M_block_size, N, waves, **rk)
            check_plan(p)
            # the waves each op runs: v2 takes v2_waves(T, resident) when waves is None, v1 and matmul_port take plan()'s
            vw = waves or v2_waves(T, rk["resident"])
            tag = f"{kind} S={S} waves {vw if v2 else p.waves} {rk} v2={int(v2)}"
            port_tag = f"{kind} S={S} waves {p.waves} {rk} (6x10 port)"
            xs = []
            for rep in range(2):
                xh = torch.randn(ROWS, COLS, S // 32, K, generator=g).to(torch.bfloat16)
                xs.append(
                    ttnn.from_torch(
                        xh,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(ROWS, COLS)),
                    )
                )
            for rep, x in enumerate(xs):
                xa, ref = stock(x, S)
                xa_h, ref_h = _host(xa), _host(ref)
                if rep == 0:
                    port = matmul_port(xa, w, waves=waves, out=_zeros(mesh, (1, 1, S // 4, N)), **mm_kw, **rk)
                    eq = sum(torch.equal(a, b) for a, b in zip(_host(port), ref_h))
                    logger.info(f"QAGMM port {port_tag}: torch.equal on {eq}/32 devices")
                    if eq != 32:
                        failures.append(f"port {port_tag}: torch.equal on {eq}/32 devices")
                    ttnn.deallocate(port)
                gath = _zeros(mesh, (1, 1, S // 4, K))
                out = _zeros(mesh, (1, 1, S // 4, N))
                t0 = time.perf_counter()
                o, gg = fused(x, w, copy_local=kd["copy_local"], waves=waves, gathered=gath, out=out, **mm_kw, **rk)
                ttnn.synchronize_device(mesh)
                dt = time.perf_counter() - t0
                o_h, g_h = _host(o), _host(gg)
                eq_o = sum(torch.equal(a, b) for a, b in zip(o_h, ref_h))
                bad_slots = 0
                for d in range(ROWS * COLS):
                    r = d // COLS
                    a = g_h[d].reshape(ROWS, S // 32, K)
                    b = xa_h[d].reshape(ROWS, S // 32, K)
                    for s in range(ROWS):
                        if s == r and not kd["copy_local"]:
                            continue
                        bad_slots += not torch.equal(a[s], b[s])
                logger.info(
                    f"QAGMM fused {tag} call {rep} ({dt:.2f} s): out torch.equal on {eq_o}/32 devices, "
                    f"gathered bad slots {bad_slots}"
                )
                if eq_o != 32:
                    for d in range(ROWS * COLS):
                        if not torch.equal(o_h[d], ref_h[d]):
                            diff = (o_h[d] != ref_h[d]).reshape(S // 4 // 32, 32, N // 32, 32).any(-1).any(1)
                            rows_bad = diff.any(-1).nonzero().flatten().tolist()
                            logger.info(f"QAGMM dev {d}: bad tile rows {rows_bad[:40]} ({len(rows_bad)})")
                            break
                if eq_o != 32 or bad_slots:
                    failures.append(f"fused {tag} call {rep}: out equal on {eq_o}/32 devices, {bad_slots} bad slots")
                for t in (xa, ref, o, gg):
                    ttnn.deallocate(t)
            x = xs[0]
            if os.environ.get("QAGMM_TIME") != "1":
                for t in xs:
                    ttnn.deallocate(t)
                continue
            m_s, r_s = _timeit(mesh, lambda: stock(x, S), reps, batch)
            m_f, r_f = _timeit(
                mesh, lambda: fused(x, w, copy_local=kd["copy_local"], waves=waves, **mm_kw, **rk), reps, batch
            )
            ttnn.synchronize_device(mesh)
            held = []
            t0 = time.perf_counter()
            for _ in range(batch):
                held.append(fused(x, w, copy_local=kd["copy_local"], waves=waves, **mm_kw, **rk))
            t_host = (time.perf_counter() - t0) * 1e3 / batch
            ttnn.synchronize_device(mesh)
            for ts in held:
                for t in ts:
                    ttnn.deallocate(t)
            logger.info(f"QAGMM_HOST {tag}: fused host enqueue {t_host:.3f} ms/call")
            if os.environ.get("QAGMM_PROBES") == "1":
                xa0 = cp._row_ag(ccl, mc, x, S)
                gx = 7 if v2 else 6
                m_p, _ = _timeit(mesh, lambda: [matmul_port(xa0, w, waves=waves, gx=gx, **mm_kw, **rk)], reps, batch)
                ttnn.deallocate(xa0)
                res = {f"port {gx}x10": m_p}
                for probe in ("nowait",) if v2 else ("nowait", "nomm", "agonly"):
                    res[probe], _ = _timeit(
                        mesh,
                        lambda: fused(x, w, copy_local=kd["copy_local"], waves=waves, probe=probe, **mm_kw, **rk),
                        reps,
                        batch,
                    )
                logger.info(f"QAGMM_PROBE {tag}: " + ", ".join(f"{k} {v:.3f} ms" for k, v in res.items()))
            logger.info(
                f"QAGMM_TIME {tag}: stock pair {m_s:.3f} ms {['%.3f' % v for v in r_s]}, fused {m_f:.3f} ms "
                f"{['%.3f' % v for v in r_f]}"
            )
            for t in xs:
                ttnn.deallocate(t)
        ttnn.deallocate(w)
    assert not failures, "qwen_ag_matmul mismatches:\n" + "\n".join(failures)
