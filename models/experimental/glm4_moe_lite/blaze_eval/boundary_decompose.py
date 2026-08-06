# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Split GLMQKVAProjection's model-boundary cost between its input and output halves.

native_boundary_ab.py says the fused op is 0.40x against ttnn once it takes the model's native
32x32 TILE activation and writes DRAM outputs -- 117 us against 47 us. This attributes that cost.

MEASURED (single BH device, profiler on, builds hoisted):

    A  core only                36.5 us
    B  A + TileRowReplicate     76.2 us     input  boundary = 39.7 us
    C  A + 2x GatherRowToDRAM   77.0 us     output boundary = 40.6 us
    D  both                    117.1 us     additivity holds: B+C-A = 116.8

THE VERDICT DOES NOT DEPEND ON THE CORE. The two boundary halves alone are 39.7 + 40.6 = 80.3 us,
which already exceeds the entire ttnn path at 47.4 us. Even with a hypothetically free matmul this
cluster loses as a drop-in, so no amount of core tuning rescues it.

Keeping outputs in L1 and adapting the downstream consumers -- the one escape hatch this script
was written to find -- gets to B = 76.2 us, still only 0.62x. Only removing BOTH boundaries, i.e.
a fully blaze-native neighbourhood where the activation arrives already replicated, reaches
A = 36.5 us for 1.30x, worth ~0.5 ms of a 33.2 ms token (1.6%).

Note A = 36.5 us contradicts the 9.5 us this evaluation's 4.76x headline rested on, for what is
the same configuration. Unresolved; candidates are profiler env (a misordered set_profiler_env
undercounts silently) and the op's new default subblock_k=2. The sweep at the bottom would settle
it but is opt-in -- see there for why, and note the conclusion above is insensitive to it.

The op branches on tile shape, which is what makes the split measurable without editing it:
a (32,32) activation inserts TileRowReplicate, and (32,32) outputs insert GatherRowToDRAM.
Feeding blaze-native (1,32) tensors on either side removes that half.

    A  native act,  L1 out     pure matmul core, no boundary at all
    B  model act,   L1 out     A + TileRowReplicate           -> input cost  = B - A
    C  native act,  DRAM out   A + 2x GatherRowToDRAM         -> output cost = C - A
    D  model act,   DRAM out   both -- what integration pays  (should be ~B + C - A)

Which half dominates decides whether any integration can win: if it is the output gather, the
downstream consumers could take blaze's L1-sharded outputs directly and skip it. If it is the
input replicate, the cluster is dead as a drop-in.
"""

import os
import sys

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")
import ab_harness

# MUST precede the ttnn import: _measure() returns None on zero profiler samples rather than
# raising, so a misordered import degrades silently into "no result".
ab_harness.set_profiler_env()

import torch  # noqa: E402
import ttnn  # noqa: E402
from blaze.models.blaze_tests_namespace import register_blaze_tests_namespace  # noqa: E402

register_blaze_tests_namespace()
from blaze.fused_program import FusedProgram  # noqa: E402
from blaze.ops.dram_streaming_matmul.common import dram_bank_worker_cores  # noqa: E402
from blaze.ops.glm_qkv_a_projection import GLMQKVAProjection  # noqa: E402
from blaze_tests.micro_ops.common.test_dram_streaming_matmul import (  # noqa: E402
    _make_act_tensor,
    _make_output_tensor,
    _make_weights_tensor,
    _pad_to_dram_banks,
)

d = ttnn.open_device(device_id=0)
K, NQ, NKV, M = 2048, 768, 576, 32
banks = d.dram_grid_size().x
nq, nkv = _pad_to_dram_banks(NQ, 32, 32 * banks), _pad_to_dram_banks(NKV, 32, 32 * banks)
_, worker_grid = dram_bank_worker_cores(d)

torch.manual_seed(47)
row = torch.randn(1, 1, 1, K).bfloat16().float()
model_act_t = torch.zeros(1, 1, M, K, dtype=torch.bfloat16).float()
model_act_t[..., :1, :] = row
wq = torch.randn(1, 1, K, nq).bfloat16().float()
wkv = torch.randn(1, 1, K, nkv).bfloat16().float()

b_wq = _make_weights_tensor(d, wq, k=K, n_padded=nq, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
b_wkv = _make_weights_tensor(d, wkv, k=K, n_padded=nkv, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)

# (32,32) TILE DRAM -- exactly what the model holds at the q_kv_a call site.
act_model = ttnn.from_torch(
    model_act_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
# (1,32) L1 replicated -- blaze's own activation, no conversion needed.
act_native = _make_act_tensor(d, row, m=1, k=K, tile_h=1, tile_w=32, compute_core_grid=worker_grid, num_cores=banks)

mk_l1 = lambda n: _make_output_tensor(
    d, m=1, n_padded=n, tile_h=1, tile_w=32, compute_core_grid=worker_grid, per_core_N=n // banks
)
mk_dram = lambda n: ttnn.from_torch(
    torch.zeros(1, 1, 1, n),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=d,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)


def build(act, outs, subblock_k=None):
    """Build once, outside any timed callable -- rebuilding measures host composition."""
    q_out, kv_out = outs
    f = FusedProgram(
        kernel=None,
        device=d,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="decomp",
    )
    GLMQKVAProjection.emit(
        f,
        act,
        b_wq,
        b_wkv,
        q_a_out=q_out,
        kv_a_out=kv_out,
        prefix="qkv_a",
        fp32_dest_acc_en=True,
        q_a_subblock_k=subblock_k,
        kv_a_subblock_k=subblock_k,
    )
    return f


VARIANTS = [
    ("A  native act -> L1  out  (core only)", act_native, "l1"),
    ("B  model  act -> L1  out  (+TileRowReplicate)", act_model, "l1"),
    ("C  native act -> DRAM out (+2x GatherRowToDRAM)", act_native, "dram"),
    ("D  model  act -> DRAM out (both = integration)", act_model, "dram"),
]

res = {}
for label, act, kind in VARIANTS:
    try:
        outs = (mk_l1(nq), mk_l1(nkv)) if kind == "l1" else (mk_dram(nq), mk_dram(nkv))
        prog = build(act, outs)
        us, _ = ab_harness._measure(d, lambda p=prog: p.run(), warmup=2, iters=5)
        res[label[0]] = us
        print(f"RESULT {label} = {us} us", flush=True)
    except Exception as e:
        print(f"RESULT {label} = FAIL: {str(e).split(chr(10))[0][:130]}", flush=True)

if all(k in res and res[k] for k in "ABCD"):
    a, b, c, dd = (res[k] for k in "ABCD")
    print(f"RESULT input  TileRowReplicate cost = {b-a:.1f} us  (B-A)", flush=True)
    print(f"RESULT output 2x GatherRowToDRAM    = {c-a:.1f} us  (C-A)", flush=True)
    print(f"RESULT additivity check: B+C-A = {b+c-a:.1f} vs D = {dd:.1f} us", flush=True)
    print(f"RESULT best case if outputs stay in L1 (B) vs ttnn 47.4 us = {47.4/b:.2f}x", flush=True)
# The core (variant A) measures 36.5 us, but this evaluation's headline 4.76x rested on 9.5 us for
# the same configuration. This sweep tests whether the op's new default subblock_k=2 explains the
# gap (its comment says 1 deadlocks on a MESH, and this is a single device).
#
# OPT-IN: each subblock_k is a fresh JIT compile, and the four together ran past 15 minutes of
# Galaxy time without finishing. It is also not on the critical path -- see the module docstring:
# the boundary halves alone already exceed the whole ttnn path, so the verdict does not depend on
# what the core costs. Set GLM_SWEEP_SUBBLOCK_K=1 and allow >20 min if you want the answer.
for sk in (1, 2, 4, 8) if os.environ.get("GLM_SWEEP_SUBBLOCK_K") == "1" else ():
    try:
        prog = build(act_native, (mk_l1(nq), mk_l1(nkv)), subblock_k=sk)
        us, _ = ab_harness._measure(d, lambda p=prog: p.run(), warmup=2, iters=5)
        print(f"RESULT core (native->L1) subblock_k={sk} = {us} us", flush=True)
    except Exception as e:
        print(f"RESULT core subblock_k={sk} = FAIL: {str(e).split(chr(10))[0][:110]}", flush=True)

ttnn.close_device(d)
