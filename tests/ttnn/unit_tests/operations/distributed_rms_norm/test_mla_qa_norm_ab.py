# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLA q_a latent norm (models/demos/deepseek_v3_d_p/tt/mla/mla.py _q_a_latent), two orders, end to end:

  A (the model today):  high_bw_all_gather(x_shard) -> rms_norm(gathered latent)
  B (distributed):      distributed_rms_norm(x_shard) -> high_bw_all_gather(normed shard)

x_shard is the reduce-scattered q_a latent, (640, q_lora / 4) per chip at the production 5120-token chunk over SP 8.
Both orders use the same high_bw_all_gather call into a persistent output and the model's norm config (HiFi2,
fp32 dest off, ROW_MAJOR gamma sticks). `with_mrs` prepends the real producer, matmul_reduce_scatter_prefill of
the (640, 7168 / 4) hidden slice against the K-sharded bf8 q_a_proj, so the norm starts with the chip skew a
preceding CCL leaves.

Measured traced, ITERS pipelines back to back, device profiler per program. Per chip: each op's device time, the
pipeline span (first op start -> last op end) and the back-to-back period (pipeline launch to next launch), median
over iterations and REPEATS trace replays; reported as the max over chips.

    scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/distributed_rms_norm/test_mla_qa_norm_ab.py -s
"""

import os
import statistics as st

from tests.ttnn.unit_tests.operations.distributed_rms_norm import drms_ccl_perf as ccl_perf

for _k, _v in ccl_perf.PROFILER_ENV.items():  # before the mesh opens
    os.environ.setdefault(_k, _v)

import pytest  # noqa: E402
import torch  # noqa: E402
import ttnn  # noqa: E402

from tests.ttnn.unit_tests.operations.distributed_rms_norm.drms_mesh import (  # noqa: E402
    TP,
    close_ring_mesh,
    open_ring_mesh,
    pcc,
    replicated,
    rms_norm_ref,
    width_sharded,
)

ITERS = int(os.environ.get("AB_ITERS", "100"))
REPEATS = int(os.environ.get("AB_REPEATS", "5"))
M = 640  # PREFILL_CHUNK_TOKENS_PER_CHIP
HIDDEN = 7168
EPS = 1e-5
LINKS = 2


@pytest.fixture(scope="module")
def ring_mesh():
    mesh = open_ring_mesh(1 << 26)
    yield mesh
    close_ring_mesh(mesh)


def _sticks(g, mesh, shard):
    sticks = g.reshape(1, 1, -1, 32)
    if shard:
        return ttnn.from_torch(
            sticks,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(1, TP), dims=(None, 2)),
        )
    return replicated(sticks, mesh, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)


def _measure(mesh, pipe, n_ops):
    chips = ccl_perf.chips_of(mesh)
    pipe()
    ttnn.synchronize_device(mesh)
    ccl_perf._read(mesh, chips)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(ITERS):
        pipe()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
    ccl_perf._read(mesh, chips)
    per_chip = {c: {"ops": [[] for _ in range(n_ops)], "span": [], "period": []} for c in chips}
    for _ in range(REPEATS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        per = ccl_perf._read(mesh, chips)
        for c in chips:
            ps = per[c]
            assert len(ps) == n_ops * ITERS, f"chip {c}: {len(ps)} programs, expected {n_ops * ITERS}"
            npc = st.median(d / (e - s) for s, e, d in ps if e > s)
            r = per_chip[c]
            for k in range(ITERS):
                prog = ps[n_ops * k : n_ops * (k + 1)]
                for j, p in enumerate(prog):
                    r["ops"][j].append(p[2] / 1e3)
                r["span"].append((prog[-1][1] - prog[0][0]) * npc / 1e3)
                if k + 1 < ITERS:
                    r["period"].append((ps[n_ops * (k + 1)][0] - prog[0][0]) * npc / 1e3)
    ttnn.release_trace(mesh, tid)
    med = {
        c: {"ops": [st.median(o) for o in r["ops"]], "span": st.median(r["span"]), "period": st.median(r["period"])}
        for c, r in per_chip.items()
    }
    return {
        "ops": [max(m["ops"][j] for m in med.values()) for j in range(n_ops)],
        "span": max(m["span"] for m in med.values()),
        "period": max(m["period"] for m in med.values()),
        "period_per_chip": [round(m["period"], 2) for m in med.values()],
    }


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("with_mrs", [False, True], ids=["norm_and_gather", "with_mrs"])
@pytest.mark.parametrize("fp32_dest", [False, True], ids=["fp32dest0", "fp32dest1"])
@pytest.mark.parametrize("q_lora", [1536, 2048], ids=["qlora1536", "qlora2048"])
def test_mla_qa_norm_ab(ring_mesh, q_lora, fp32_dest, with_mrs):
    mesh = ring_mesh
    torch.manual_seed(0)
    W = q_lora
    Wl = W // TP
    gamma = (1 + 0.1 * torch.randn(W)).to(torch.bfloat16)

    if with_mrs:
        hidden = (torch.randn(1, 1, M, HIDDEN) * 0.5).to(torch.bfloat16)
        w_qa = (torch.randn(HIDDEN, W) / HIDDEN**0.5).to(torch.bfloat16)
        t_hidden = width_sharded(hidden, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        t_wqa = ttnn.from_torch(
            w_qa,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(1, TP), dims=(None, 0)),
        )
        w_q = ttnn.to_torch(ttnn.from_torch(w_qa, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        latent = (hidden.float() @ w_q).reshape(1, 1, M, W)

        def producer():
            return ttnn.experimental.matmul_reduce_scatter_prefill(
                t_hidden, t_wqa, cluster_axis=1, scatter_dim=-1, topology=ttnn.Topology.Ring, num_links=LINKS
            )

    else:
        latent = torch.randn(1, 1, M, W)
        t_latent = width_sharded(latent.to(torch.bfloat16), mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)

        def producer():
            return t_latent

    ref = rms_norm_ref(latent, gamma, EPS)
    g_full = _sticks(gamma, mesh, shard=False)
    g_shard = _sticks(gamma, mesh, shard=True)
    out_a = replicated(torch.zeros(1, 1, M, W), mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    out_b = replicated(torch.zeros(1, 1, M, W), mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    model_ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest,
        packer_l1_acc=True,
    )
    drms_ckc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )

    def gather(src, out):
        return ttnn.experimental.high_bw_all_gather(src, dim=3, output_tensor=out, cluster_axis=1, num_links=LINKS)

    def pipe_a():
        gather(producer(), out_a)
        return ttnn.rms_norm(
            out_a, weight=g_full, epsilon=EPS, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=model_ckc
        )

    def pipe_b():
        normed = ttnn.distributed_rms_norm(
            producer(),
            cluster_axis=1,
            gamma=g_shard,
            epsilon=EPS,
            topology=ttnn.Topology.Ring,
            num_links=LINKS,
            compute_kernel_config=drms_ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return gather(normed, out_b)

    n_ops = 3 if with_mrs else 2
    names = (["mrs"] if with_mrs else []) + ["op1", "op2"]
    res = {}
    for label, pipe, opn in (
        ("A all_gather -> rms_norm", pipe_a, ["all_gather", "rms_norm"]),
        ("B drms -> all_gather", pipe_b, ["drms", "all_gather"]),
    ):
        out = pipe()
        ttnn.synchronize_device(mesh)
        for i, t in enumerate(ttnn.get_device_tensors(out)):
            p = pcc(ttnn.to_torch(t), ref)
            assert p > 0.999, f"{label} chip {i}: pcc {p}"
        r = _measure(mesh, pipe, n_ops)
        res[label] = r
        ops = ", ".join(f"{n} {v:.2f}" for n, v in zip((["mrs"] if with_mrs else []) + opn, r["ops"]))
        print(
            f"\nAB q_lora={W} (per chip {M}x{Wl}) fp32_dest={int(fp32_dest)} with_mrs={int(with_mrs)} | {label}: "
            f"ops [{ops}] us, span {r['span']:.2f} us, period {r['period']:.2f} us (per chip {r['period_per_chip']})"
        )
    a, b = res.values()
    print(
        f"AB SUMMARY q_lora={W} fp32_dest={int(fp32_dest)} with_mrs={int(with_mrs)}: "
        f"period A {a['period']:.2f} -> B {b['period']:.2f} us ({b['period'] - a['period']:+.2f}, "
        f"{100 * (b['period'] / a['period'] - 1):+.1f}%); span A {a['span']:.2f} -> B {b['span']:.2f} us"
    )
