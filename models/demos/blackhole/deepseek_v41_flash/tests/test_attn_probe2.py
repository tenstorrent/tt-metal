# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 2: op-variant timings inside the real attention object (random weights, ratio 2). Prints 'P2 name us'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_profile import make_weights
from models.demos.blackhole.deepseek_v41_flash.tt import attention as A
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_probe2(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    rows, cols = tuple(md.shape)
    T, B, S = 4, rows * 4, 24
    freqs = torch.polar(torch.ones(256, 32), torch.rand(256, 32))
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    attn = A.DSV41Attention(md, mesh_4x8(), ccl, make_weights(), freqs, users_per_row=T, max_seq=256)
    attn.load_window(torch.randn(B, S, 512))
    st = attn.step_inputs(torch.full((B,), S))
    shard = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
    x = ttnn.from_torch(
        torch.randn(1, 1, B, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    ckc, lofi = attn.ckc, ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    rep = ttnn.ReplicateTensorToMesh(md)

    def run(name, fn):
        try:
            print(f"P2 {name:48s} {chain_ms(md, fn) * 1e3:7.1f} us", flush=True)
        except Exception as e:
            print(f"P2 {name:48s} FAIL {str(e).splitlines()[0][:100]!r}", flush=True)

    qa = attn._lin(x, attn.wq_a, "QA")
    kvraw = attn._lin(x, attn.wkv, "KV")
    # --- rms_norm variants
    gq_tile = ttnn.from_torch(
        torch.ones(1, 1, 1, 1280).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    gk_tile = ttnn.from_torch(
        torch.ones(1, 1, 1, 512).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    run("rms_q 1280 RM gamma (current)", lambda: ttnn.rms_norm(qa, weight=attn.q_norm, epsilon=1e-20))
    run("rms_q 1280 tile gamma", lambda: ttnn.rms_norm(qa, weight=gq_tile, epsilon=1e-20))
    run("rms_q 1280 no gamma", lambda: ttnn.rms_norm(qa, epsilon=1e-20))
    run("rms_q 1280 lofi ckc", lambda: ttnn.rms_norm(qa, weight=attn.q_norm, epsilon=1e-20, compute_kernel_config=lofi))
    run("rms_kv 512 RM gamma (current)", lambda: ttnn.rms_norm(kvraw, weight=attn.kv_norm, epsilon=1e-20))
    run("rms_kv 512 tile gamma", lambda: ttnn.rms_norm(kvraw, weight=gk_tile, epsilon=1e-20))
    run(
        "rms_q 1280 tile gamma lofi",
        lambda: ttnn.rms_norm(qa, weight=gq_tile, epsilon=1e-20, compute_kernel_config=lofi),
    )
    # --- fused wq_a|wkv matmul + slices
    wcat = ttnn.from_torch(
        torch.randn(1, 1, 5120, 1792).to(torch.bfloat16) * 0.02,
        device=md,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    run(
        "fused wq_a|wkv N=1792 cg2x8",
        lambda: ttnn.linear(x, wcat, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=2, x=8)),
    )
    run(
        "fused wq_a|wkv N=1792 1d64",
        lambda: ttnn.linear(x, wcat, compute_kernel_config=ckc, program_config=A._cfg_1d(5120, 1792, 64)),
    )
    y = ttnn.linear(x, wcat, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=2, x=8))
    run("slice [..:1280] of [T,1792]", lambda: ttnn.slice(y, [0, 0, 0, 0], [1, 1, T, 1280]))
    run("slice [1280:] of [T,1792]", lambda: ttnn.slice(y, [0, 0, 0, 1280], [1, 1, T, 1792]))
    # --- wq_b variants
    qr = ttnn.rms_norm(qa, weight=attn.q_norm, epsilon=1e-20)
    run("wq_b 1d32 (current)", lambda: attn._lin(qr, attn.wq_b, "QB"))
    run("wq_b default", lambda: ttnn.linear(qr, attn.wq_b, compute_kernel_config=ckc))
    run(
        "wq_b 1d64",
        lambda: ttnn.linear(qr, attn.wq_b, compute_kernel_config=ckc, program_config=A._cfg_1d(1280, 4096, 64)),
    )
    run(
        "wq_b 1d16",
        lambda: ttnn.linear(qr, attn.wq_b, compute_kernel_config=ckc, program_config=A._cfg_1d(1280, 4096, 16)),
    )
    run(
        "wq_b 1d32 HiFi2",
        lambda: ttnn.linear(
            qr,
            attn.wq_b,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                md.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
            program_config=A._cfg_1d(1280, 4096, 32),
        ),
    )
    wqb2 = ttnn.from_torch(
        torch.randn(1, 1, 1280, 8192).to(torch.bfloat16) * 0.02,
        device=md,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=rep,
    )
    run(
        "wq_b|sw fused N=8192 1d64",
        lambda: ttnn.linear(qr, wqb2, compute_kernel_config=ckc, program_config=A._cfg_1d(1280, 8192, 64)),
    )
    run(
        "wq_b|sw fused N=8192 1d32",
        lambda: ttnn.linear(qr, wqb2, compute_kernel_config=ckc, program_config=A._cfg_1d(1280, 8192, 32)),
    )
    # --- q rope variants
    q = attn._lin(qr, attn.wq_b, "QB")
    qs = attn._lin(qr, attn.wq_b_sw, "QB")
    run("q: mul + addcmul", lambda: ttnn.addcmul(ttnn.multiply(q, st["Cq"]), qs, st["Sq"]))
    run("q: mul", lambda: ttnn.multiply(q, st["Cq"]))
    run("q: mul,mul,add", lambda: ttnn.add(ttnn.multiply(q, st["Cq"]), ttnn.multiply(qs, st["Sq"])))
    # --- o path
    cat = ttnn.concat([q, kvraw, kvraw], dim=3)
    qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
        cat, num_heads=8, num_kv_heads=1, memory_config=attn._ucfg
    )
    run(
        "nlp_create (DRAM input)",
        lambda: ttnn.experimental.nlp_create_qkv_heads_decode(
            cat, num_heads=8, num_kv_heads=1, memory_config=attn._ucfg
        ),
    )
    prog = attn._sdpa_cfg(128)
    o = ttnn.transformer.scaled_dot_product_attention_decode(
        qh,
        attn.cache,
        attn.cache,
        cur_pos_tensor=st["pos"],
        sliding_window_size=128,
        attention_sink=attn.sinks,
        scale=attn.scale,
        program_config=prog,
        compute_kernel_config=ckc,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    run("o@Pf (default cfg, DRAM o)", lambda: ttnn.linear(o, attn.Pf, compute_kernel_config=ckc))
    run("o@Pf HiFi2 no fp32acc", lambda: ttnn.linear(o, attn.Pf, compute_kernel_config=lofi))
    osw = ttnn.linear(o, attn.Pf, compute_kernel_config=ckc)
    run("o*Ch (DRAM out)", lambda: ttnn.multiply(o, st["Ch"]))
    run("o*Ch (sharded out)", lambda: ttnn.multiply(o, st["Ch"], memory_config=attn._ucfg))
    oc = ttnn.multiply(o, st["Ch"])
    run("addcmul DRAM out", lambda: ttnn.addcmul(oc, osw, st["nSh"]))
    run("addcmul sharded out", lambda: ttnn.addcmul(oc, osw, st["nSh"], memory_config=attn._ucfg))
    run("rope_inv full (current)", lambda: attn._rope_inv(o, st))
    oi = attn._rope_inv(o, st)
    oi_d = ttnn.to_memory_config(oi, ttnn.DRAM_MEMORY_CONFIG)
    run("rope_inv + DRAM (extra op)", lambda: ttnn.to_memory_config(attn._rope_inv(o, st), ttnn.DRAM_MEMORY_CONFIG))
    run("nlp_concat_heads_decode (from sharded)", lambda: ttnn.experimental.nlp_concat_heads_decode(oi, num_heads=8))
    c = ttnn.experimental.nlp_concat_heads_decode(oi, num_heads=8)
    c_d = ttnn.to_memory_config(c, ttnn.DRAM_MEMORY_CONFIG)
    run("concat out -> DRAM", lambda: ttnn.to_memory_config(c, ttnn.DRAM_MEMORY_CONFIG))
    run("wo_a on sharded c, cg2x8 (current)", lambda: attn._lin(c, attn.wo_a, "OA"))
    run("wo_a on sharded c, default", lambda: ttnn.linear(c, attn.wo_a, compute_kernel_config=ckc))
    run("wo_a on DRAM c, cg2x8", lambda: attn._lin(c_d, attn.wo_a, "OA"))
    run(
        "wo_a on DRAM c, 1d32",
        lambda: ttnn.linear(c_d, attn.wo_a, compute_kernel_config=ckc, program_config=A._cfg_1d(4096, 1024, 32)),
    )
    run(
        "wo_a on DRAM c, 1d16",
        lambda: ttnn.linear(c_d, attn.wo_a, compute_kernel_config=ckc, program_config=A._cfg_1d(4096, 1024, 16)),
    )
    # alt: transposes instead of nlp ops
    run("transpose(1,2) o -> [1,8,T,512]", lambda: ttnn.transpose(o, 1, 2))
    a1 = attn._lin(c_d, attn.wo_a, "OA")
    run("wo_b 1d32 (current)", lambda: attn._lin(a1, attn.wo_b, "OB"))
    run("wo_b cg2x8", lambda: ttnn.linear(a1, attn.wo_b, compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=2, x=8)))
    run(
        "wo_b 1d64",
        lambda: ttnn.linear(a1, attn.wo_b, compute_kernel_config=ckc, program_config=A._cfg_1d(1024, 5120, 64)),
    )
    part = attn._lin(a1, attn.wo_b, "OB")

    # --- all-reduce variants (never deallocating the input)
    def rs_ag(t, links=2, mem=ttnn.DRAM_MEMORY_CONFIG):
        sc = ttnn.experimental.reduce_scatter_minimal_async(
            t,
            dim=3,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            num_links=links,
            memory_config=mem,
            topology=ccl.topology,
            cluster_axis=1,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )
        return ttnn.experimental.all_gather_async(
            sc,
            dim=3,
            cluster_axis=1,
            mesh_device=md,
            topology=ccl.topology,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=links,
            memory_config=mem,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        )

    run("allreduce RS+AG links=2 (current)", lambda: rs_ag(part))
    run("allreduce RS+AG links=1", lambda: rs_ag(part, 1))
    run("allreduce RS+AG links=4", lambda: rs_ag(part, 4))
    run("allreduce RS+AG L1 out", lambda: rs_ag(part, 2, ttnn.L1_MEMORY_CONFIG))
    run(
        "ttnn.all_reduce ring l2",
        lambda: ttnn.all_reduce(part, cluster_axis=1, num_links=2, topology=ttnn.Topology.Ring),
    )
    run(
        "ttnn.all_reduce linear l2",
        lambda: ttnn.all_reduce(part, cluster_axis=1, num_links=2, topology=ttnn.Topology.Linear),
    )
    sc = ttnn.experimental.reduce_scatter_minimal_async(
        part,
        dim=3,
        multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ccl.topology,
        cluster_axis=1,
        barrier_semaphore=ccl.get_barrier_semaphore(),
    )
    run(
        "RS only",
        lambda: ttnn.experimental.reduce_scatter_minimal_async(
            part,
            dim=3,
            multi_device_global_semaphore=ccl.get_rs_ping_pong_semaphore(),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl.topology,
            cluster_axis=1,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        ),
    )
    run(
        "AG only (640 -> 5120)",
        lambda: ttnn.experimental.all_gather_async(
            sc,
            dim=3,
            cluster_axis=1,
            mesh_device=md,
            topology=ccl.topology,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        ),
    )
    # AG of the 1024-wide latent instead (gather-first variant)
    run(
        "AG a1 (1024 -> 8192)",
        lambda: ttnn.experimental.all_gather_async(
            a1,
            dim=3,
            cluster_axis=1,
            mesh_device=md,
            topology=ccl.topology,
            multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=ccl.get_barrier_semaphore(),
        ),
    )
    # --- cache write / small ops
    run("paged_update_cache k", lambda: attn._write_cache(attn.cache, kh, st["pos"]))
    # --- kv rope variants
    kvn = ttnn.rms_norm(kvraw, weight=attn.kv_norm, epsilon=1e-20)
    run("kv rope_rows (mm+mul+addcmul)", lambda: attn._rope_rows(kvn, st["Cr"], st["Sr"]))
    run("kv P-matmul only", lambda: attn._lin(kvn, attn.Pf, "SW"))
    run("kv mul only", lambda: ttnn.multiply(kvn, st["Cr"]))
