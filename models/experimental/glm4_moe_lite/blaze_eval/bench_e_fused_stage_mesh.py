# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark E — a real two-matmul fused stage with an L1-only boundary.

    BLAZE_DSM_WORKERS_PER_BANK=6 timeout 1800 python bench_e_fused_stage.py

THE QUESTION
------------
Benchmark D showed the INPUT half of blaze's boundary pair collapses from 12.4 us to 0.2 us when
the activation arrives by Mcast from L1 instead of by TileRowReplicate from a 32x32 TILE DRAM
tensor. That measured the consumer side with the producer's contribution assumed away.

This bench closes the loop. It builds the composition end to end —

    DSM1 -> GatherRowToDRAM(write_to_dram=False) -> Mcast -> DSM2

— against a control that runs the same two matmuls in the same program with no data dependency
and no boundary op at all. The difference is the entire fused-stage boundary, producer side
included, and it is gated on a PCC that can only pass if the round trip delivered the right bytes
to all 48 cores.

Shapes: matmul 1 is benchmark A's concatenated q_a+kv_a (K=2048, N_pad=1536); matmul 2 consumes
its full 1536-column output (K2=1536, N_pad2=1536). At W=6 each of the 48 workers owns exactly
32 output columns = one 1x32 page, so the gather moves 48 x 64 B into one core and the mcast
broadcasts 3 KB back out.

VARIANTS
--------
  P1     DSM1 alone, native act -> L1 out            core only
  P2     DSM2 alone, native act -> L1 out            core only
  BASE   DSM1 + DSM2 in one program, NO dependency   two cores' worth of arithmetic, no boundary
  FUSED  DSM1 -> gather(L1) -> Mcast -> DSM2         the fused stage
  U1     DSM1 shipped: model act -> DRAM out         what blaze runs today for matmul 1
  U2     DSM2 shipped: model act -> DRAM out         what blaze runs today for matmul 2

  fused-stage boundary  = FUSED - BASE
  unfused stage cost    = U1 + U2      (two programs, two DRAM boundary pairs)
  saving from fusing    = U1 + U2 - FUSED

BASE is the right control rather than P1+P2 because it pays one dispatch, not two, so it isolates
the boundary rather than the dispatch overhead.

WHAT WOULD MAKE THIS UNTRUSTWORTHY
-----------------------------------
  1. FUSED's PCC gate below 0.99 -- the round trip delivered wrong bytes.
  2. BASE's PCC gate failing on either output -- the control is not running both matmuls.
  3. FUSED - BASE coming out negative by more than measurement noise: that would mean BASE is not
     a valid control (e.g. its two independent matmuls overlap in a way the dependent chain cannot).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_guard  # noqa: E402

bench_guard.preflight("bench_e")

import torch  # noqa: E402
import ttnn  # noqa: E402
from blaze.models.blaze_tests_namespace import register_blaze_tests_namespace  # noqa: E402

register_blaze_tests_namespace()

from blaze.blaze_op import Risc  # noqa: E402
from blaze.fused_program import FusedProgram  # noqa: E402
from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul  # noqa: E402
from blaze.ops.dram_streaming_matmul.common import dram_bank_worker_cores  # noqa: E402
from blaze.ops.gather_row_to_dram import GatherRowToDRAM  # noqa: E402
from blaze.ops.mcast import Mcast, McastGridConfig  # noqa: E402
from blaze.ops.tile_row_replicate import TileRowReplicate  # noqa: E402
from blaze_tests.micro_ops.common.test_dram_streaming_matmul import (  # noqa: E402
    _make_act_tensor,
    _make_output_tensor,
    _make_weights_tensor,
    _pad_to_dram_banks,
    _roundtrip_weights,
)

BUDGET_S = float(os.environ.get("BENCH_E_BUDGET_S", "1500"))
_T0 = time.time()

K1, N, M = 2048, 1344, 32
PCC_FLOOR = 0.99
W = max(1, int(os.environ.get("BLAZE_DSM_WORKERS_PER_BANK", "1")))
SUBBLOCK_K = 2


def budget(phase: str) -> None:
    if time.time() - _T0 > BUDGET_S:
        raise SystemExit(f"RESULT ABORT: wall budget {BUDGET_S:.0f}s exceeded before {phase}")
    bench_guard.checkpoint(f"bench_e/{phase}")


# MESH port: the single-chip version of this bench passes; the model (4x8 mesh of 32) hangs in
# the Q stage. mesh_qkv_a_ab.py already proved the mcast-FREE chain on 32 chips, so Mcast on a
# mesh is the one untested variable. This reproduces it in ~1 min instead of a 10-min model run.
_parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
d = _parent.create_submesh(ttnn.MeshShape(4, 8))
print(f"RESULT mesh n={d.get_num_devices()}", flush=True)


def _mesh_to_torch(t):
    """to_torch on a mesh tensor needs a single shard; tensors here are replicated."""
    shards = ttnn.get_device_tensors(t)
    return _ttnn_to_torch(shards[0] if shards else t)


_ttnn_to_torch = ttnn.to_torch
ttnn.to_torch = _mesh_to_torch
banks = d.dram_grid_size().x
n_pad = _pad_to_dram_banks(N, 32, 32 * banks * W)
K2 = n_pad  # matmul 2 consumes matmul 1's full padded output
n_pad2 = _pad_to_dram_banks(N, 32, 32 * banks * W)
worker_list, worker_grid = dram_bank_worker_cores(d)
ncores = len(worker_list)

_probe = FusedProgram(kernel=None, device=d, name="bench_e__probe")
sender_core = _probe.sender
assert (sender_core.x, sender_core.y) not in {
    (c.x, c.y) for c in worker_list
}, f"mcast sender {sender_core} is also a DSM worker at W={W}"

print(
    f"RESULT config: K1={K1} N_pad={n_pad} K2={K2} N_pad2={n_pad2} banks={banks} W={W} "
    f"cores={ncores} gather_receiver=mcast_sender={(sender_core.x, sender_core.y)}",
    flush=True,
)

torch.manual_seed(47)
row = torch.randn(1, 1, 1, K1).bfloat16().float()
w1 = torch.randn(1, 1, K1, n_pad).bfloat16().float()
w2 = torch.randn(1, 1, K2, n_pad2).bfloat16().float()

# Intermediate rounded to bf16: that is what the device carries between the two matmuls, whether
# it travels through L1 or through DRAM, so the golden must round too or PCC measures rounding.
mid = (row.reshape(1, K1) @ _roundtrip_weights(w1, ttnn.bfloat8_b).reshape(K1, n_pad)).bfloat16().float()
golden1 = mid[:, :N]
golden2 = (mid @ _roundtrip_weights(w2, ttnn.bfloat8_b).reshape(K2, n_pad2))[:, :N]

b_w1 = _make_weights_tensor(d, w1, k=K1, n_padded=n_pad, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
b_w2 = _make_weights_tensor(d, w2, k=K2, n_padded=n_pad2, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)


def model_tile_tensor(r, k):
    t = torch.zeros(1, 1, M, k, dtype=torch.bfloat16).float()
    t[..., :1, :] = r
    return ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


act1_native = _make_act_tensor(d, row, m=1, k=K1, tile_h=1, tile_w=32, compute_core_grid=worker_grid, num_cores=ncores)
# matmul 2's standalone activation is matmul 1's exact (bf16) output, so ONE golden gates both the
# control and the fused chain: if the L1 round trip corrupts anything, FUSED misses and BASE does not.
act2_native = _make_act_tensor(
    d, mid.reshape(1, 1, 1, K2), m=1, k=K2, tile_h=1, tile_w=32, compute_core_grid=worker_grid, num_cores=ncores
)
act1_model = model_tile_tensor(row, K1)
act2_model = model_tile_tensor(mid.reshape(1, 1, 1, K2), K2)


def l1_out(width):
    return _make_output_tensor(
        d, m=1, n_padded=width, tile_h=1, tile_w=32, compute_core_grid=worker_grid, per_core_N=width // ncores
    )


def dram_out(width):
    return ttnn.from_torch(
        torch.zeros(1, 1, ttnn.TILE_SIZE, width),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=d,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def new_prog(name):
    return FusedProgram(
        kernel=None,
        device=d,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name=name,
    )


def mm(f, act, weights, out, prefix):
    return DRAMStreamingMatmul.emit(
        f,
        act,
        weights,
        index=None,
        bias=None,
        out=out,
        prefix=prefix,
        fp32_dest_acc_en=True,
        subblock_k=SUBBLOCK_K,
        fused_activation=None,
        index_offset=0,
        wait_for_out=False,
        pop_index=False,
        pop_act=True,
    )


def build_single(*, act, weights, out_tensor, gather, replicate_from, k, name):
    f = new_prog(name)
    a = (
        TileRowReplicate.emit(f, replicate_from, prefix=f"{name}__in", cores=worker_grid, num_tile_cols=k // 32, row=0)
        if replicate_from is not None
        else act
    )
    o = mm(f, a, weights, None if gather else out_tensor, f"{name}__mm")
    if gather:
        GatherRowToDRAM.emit(f, o, out_tensor, prefix=f"{name}__out", receiver=sender_core)
    return f


def build_base(out1, out2):
    """Both matmuls, one program, no data dependency and no boundary op."""
    f = new_prog("bench_e_base")
    mm(f, act1_native, b_w1, out1, "bench_e_base__mm1")
    mm(f, act2_native, b_w2, out2, "bench_e_base__mm2")
    return f


def build_fused(out2, gather_dst):
    """DSM1 -> L1 gather onto the sender core -> Mcast to every worker -> DSM2. No DRAM."""
    f = new_prog("bench_e_fused")
    o1 = mm(f, act1_native, b_w1, None, "bench_e_fused__mm1")
    # The gather's receiver must BE the mcast sender: Mcast takes its sender from the program-level
    # grid config (f.sender), not from an argument, so the row has to be gathered onto that core.
    staged = GatherRowToDRAM.emit(
        f, o1, gather_dst, prefix="bench_e_fused__gather", receiver=sender_core, write_to_dram=False
    )
    act = Mcast.emit(
        f,
        staged,
        prefix="bench_e_fused__mcast",
        receiver_risc=Risc.DM0,
        mcast_grid_config=McastGridConfig(
            receiving_core_range_set=worker_grid, acknowledging_core_range_set=worker_grid
        ),
    )
    mm(f, act, b_w2, out2, "bench_e_fused__mm2")
    return f


res: dict[str, float] = {}
gates: dict[str, bool] = {}


def run_variant(key, label, prog, checks):
    """checks: list of (tensor, golden, width). Gate every output, then time."""
    prog.run()
    firsts = [ttnn.to_torch(t).clone() for t, _, _ in checks]
    prog.run()
    for i, (t, _, _) in enumerate(checks):
        if not torch.allclose(firsts[i], ttnn.to_torch(t)):
            print(f"RESULT {key} REUSE-FAIL: repeated run() diverges", flush=True)
            return
    ok = True
    for i, (t, g, w) in enumerate(checks):
        got = ttnn.to_torch(t)[..., :1, :w].reshape(1, w)
        ok &= bench_guard.gate(f"{key} out{i+1}", bench_guard.pcc(g, got), PCC_FLOOR)
    gates[key] = ok
    us = bench_guard.measured(d, prog.run, label=key)
    res[key] = us
    print(f"RESULT {key:6s} {label} = {us:.1f} us", flush=True)


for key, label, kw in [
    (
        "P1",
        "DSM1 alone, native act -> L1 out        (core only)",
        dict(act=act1_native, weights=b_w1, gather=False, replicate_from=None, k=K1),
    ),
    (
        "P2",
        "DSM2 alone, native act -> L1 out        (core only)",
        dict(act=act2_native, weights=b_w2, gather=False, replicate_from=None, k=K2),
    ),
    (
        "U1",
        "DSM1 shipped, model act -> DRAM out     (today)",
        dict(act=None, weights=b_w1, gather=True, replicate_from=act1_model, k=K1),
    ),
    (
        "U2",
        "DSM2 shipped, model act -> DRAM out     (today)",
        dict(act=None, weights=b_w2, gather=True, replicate_from=act2_model, k=K2),
    ),
]:
    budget(key)
    try:
        width = n_pad if key in ("P1", "U1") else n_pad2
        g = golden1 if key in ("P1", "U1") else golden2
        out_t = dram_out(width) if kw["gather"] else l1_out(width)
        prog = build_single(out_tensor=out_t, name=f"bench_e_{key.lower()}", **kw)
        run_variant(key, label, prog, [(out_t, g, N)])
    except Exception as e:
        print(f"RESULT {key} FAIL: {str(e).split(chr(10))[0][:200]}", flush=True)

budget("BASE")
try:
    o1, o2 = l1_out(n_pad), l1_out(n_pad2)
    run_variant(
        "BASE",
        "DSM1 + DSM2, one program, no boundary   (control)",
        build_base(o1, o2),
        [(o1, golden1, N), (o2, golden2, N)],
    )
except Exception as e:
    print(f"RESULT BASE FAIL: {str(e).split(chr(10))[0][:200]}", flush=True)

budget("FUSED")
try:
    o2 = l1_out(n_pad2)
    run_variant(
        "FUSED",
        "DSM1 -> gather(L1) -> Mcast -> DSM2     (fused stage)",
        build_fused(o2, dram_out(n_pad)),
        [(o2, golden2, N)],
    )
except Exception as e:
    print(f"RESULT FUSED FAIL: {str(e).split(chr(10))[0][:200]}", flush=True)

# ---- MULTI-PROGRAM PROBE: the model builds 47 of these, one per layer, and hangs. Every
# single-stage config passes (1 chip, 32 chips, trace, no trace), so "more than one program" is the
# last untested difference. blaze_ops.py records the shared-scratch arena wedging the SECOND
# _build_q_stage_program call, so N=2 is the interesting point and N>2 separates "the second
# specifically" from cumulative L1/CB exhaustion.
print("RESULT ---- multi-program probe ----", flush=True)
for _n in range(2, 6):
    try:
        print(f"RESULT building program #{_n} ...", flush=True)
        _o = l1_out(n_pad2)
        _prog = build_fused(_o, dram_out(n_pad))
        print(f"RESULT   #{_n} built, running ...", flush=True)
        _prog.run()
        print(f"RESULT   #{_n} RAN OK", flush=True)
    except Exception as _e:
        print(f"RESULT   #{_n} FAIL: {str(_e).split(chr(10))[0][:180]}", flush=True)
        break

# ---- ttnn running the same two matmuls, L1 in/out, 1D mcast over interleaved DRAM weights.
# Without this the "the fused stage beats ttnn" claim rests on scaling benchmark A's 10.5 us by a
# byte ratio, which is an estimate. Measure it.
budget("ttnn")
try:
    from models.experimental.glm4_moe_lite.tt.linear_helpers import compute_1d_prog_cfg

    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    def ttnn_leg(act_row, w_full, k, width, tag):
        a = ttnn.from_torch(
            act_row.reshape(1, 1, 1, k),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=d,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        wt_ = ttnn.from_torch(
            w_full[..., :N],
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=d,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pc = compute_1d_prog_cfg(d, wt_, 1)

        def fn():
            return ttnn.linear(
                a, wt_, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=ckc
            )

        gates[f"ttnn{tag}"] = bench_guard.gate(
            f"ttnn{tag}",
            bench_guard.pcc(golden1 if tag == "1" else golden2, ttnn.to_torch(fn())[..., :1, :N].reshape(1, N)),
            PCC_FLOOR,
        )
        return bench_guard.measured(d, fn, label=f"ttnn{tag}")

    t1 = ttnn_leg(row, w1, K1, n_pad, "1")
    t2 = ttnn_leg(mid, w2, K2, n_pad2, "2")
    res["TTNN"] = t1 + t2
    print(f"RESULT TTNN   two ttnn.linear calls, L1 in/out = {t1:.1f} + {t2:.1f} = {t1+t2:.1f} us", flush=True)
except Exception as e:
    print(f"RESULT TTNN FAIL: {str(e).split(chr(10))[0][:200]}", flush=True)

print("RESULT " + "-" * 72, flush=True)
if "FUSED" in res and "BASE" in res:
    bnd = res["FUSED"] - res["BASE"]
    print(f"RESULT fused-stage L1 boundary  = {bnd:6.1f} us  (FUSED - BASE)", flush=True)
    print(f"RESULT   vs DRAM boundary pair  =   17.2 us  (benchmark D, W=6)", flush=True)
    if "U1" in res and "U2" in res:
        unfused = res["U1"] + res["U2"]
        print(
            f"RESULT unfused two programs     = {unfused:6.1f} us  (U1 {res['U1']:.1f} + U2 {res['U2']:.1f})",
            flush=True,
        )
        print(f"RESULT fused one program        = {res['FUSED']:6.1f} us", flush=True)
        print(
            f"RESULT saving from fusing       = {unfused - res['FUSED']:6.1f} us "
            f"({(unfused - res['FUSED'])/unfused*100:.0f}% of the unfused cost)",
            flush=True,
        )
    if "TTNN" in res:
        x = res["TTNN"]
        r = res["BASE"] / x
        print(f"RESULT ttnn same two matmuls    = {x:6.1f} us", flush=True)
        print(f"RESULT blaze arithmetic ratio r = {r:.3f}  (BASE / ttnn)", flush=True)
        print(
            f"RESULT fused stage vs ttnn      = {res['FUSED']/x:.3f}x  "
            f"({'blaze WINS' if res['FUSED'] < x else 'ttnn wins'})",
            flush=True,
        )
        print(
            f"RESULT unfused vs ttnn          = {res['U1']+res['U2']:.1f}/{x:.1f} = " f"{(res['U1']+res['U2'])/x:.3f}x",
            flush=True,
        )
        if r < 1.0:
            print(f"RESULT break-even X at B={bnd:.1f} us (L1)   = {bnd/(1-r):6.1f} us", flush=True)
            print(f"RESULT break-even X at B=17.2 us (DRAM) = {17.2/(1-r):6.1f} us", flush=True)
            print(f"RESULT   against a GLM layer matmul budget of 207.8 us", flush=True)

if gates and not all(gates.values()):
    print("RESULT OVERALL: a correctness gate FAILED -- do not quote any timing above.", flush=True)
print(f"RESULT elapsed {time.time()-_T0:.0f}s  free: {bench_guard.free_gib('/'):.2f} GiB on /", flush=True)
ttnn.close_device(d)
