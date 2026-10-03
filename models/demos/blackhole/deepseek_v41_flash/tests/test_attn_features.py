# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Feature probe for the fused attention path (replicated data, device 0 compared against torch). Prints 'FT name: ...'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms

T, H, D = 4, 8, 512


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rep_up(md, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def user_cfg(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=ttnn.num_cores_to_corerangeset(T, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


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
def test_features(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    xq = torch.randn(1, 1, T, 5120).to(torch.bfloat16)
    x_t = rep_up(md, xq)

    def run(name, fn):
        try:
            print(f"FT {name}: {fn()}", flush=True)
        except Exception as e:
            print(f"FT {name}: FAIL {str(e)[:300]!r}", flush=True)

    # F2: nlp_create_qkv_heads_decode on an interleaved DRAM input
    state = {}

    def f2():
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            x_t, num_heads=H, num_kv_heads=1, memory_config=user_cfg((32, D))
        )
        state["q"], state["k"] = q, k
        qd = dev0(q)  # [1,T,8(pad?),512]
        exp = xq[0, 0, :, : H * D].reshape(T, H, D).float()
        kexp = xq[0, 0, :, H * D : H * D + D].float()
        return f"q shape {tuple(q.shape)} pcc {pcc(qd[0, :, :H], exp):.5f}; k shape {tuple(k.shape)} pcc {pcc(dev0(k)[0, :, 0], kexp):.5f} mem {q.memory_config().memory_layout}"

    run("nlp_create_qkv_heads_decode", f2)
    qs = state.get("q")

    # F1: SDPA non-causal with mask over a [T,1,256,512] combined cache
    def f1(q_in, label):
        S = 256
        K = torch.randn(T, 1, S, D).to(torch.bfloat16)
        valid = torch.zeros(T, 1, 1, S, dtype=torch.bool)
        valid[..., :40] = True
        valid[..., 128:150] = True
        mask = torch.where(valid, 0.0, -1e9).expand(T, 1, H, S).contiguous()
        sink = torch.randn(H)
        scale = D**-0.5
        sinks = torch.zeros(32, 32)
        sinks[:H, 0] = sink / scale
        Kt = rep_up(md, K)
        mt = rep_up(md, mask.to(torch.bfloat16))
        st = rep_up(md, sinks.to(torch.bfloat16))
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 1), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q_in,
            Kt,
            Kt,
            is_causal=False,
            attn_mask=mt,
            attention_sink=st,
            scale=scale,
            program_config=prog,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        state["sdpa_o"] = o
        qd = dev0(q_in)[0, :, :H].float()  # [T,H,D]
        sc = torch.einsum("thd,tsd->ths", qd, K[:, 0].float()) * scale + mask[:, 0]
        full = torch.cat([sc, (sink).reshape(1, H, 1).expand(T, H, 1)], -1)
        p = torch.softmax(full, -1)[..., :S]
        ref = torch.einsum("ths,tsd->thd", p, K[:, 0].float())
        od = dev0(o)[0, :, :H]
        return f"{label} out shape {tuple(o.shape)} pcc {pcc(od, ref):.5f}"

    if qs is not None:
        run("sdpa_mask sharded q", lambda: f1(qs, "sharded"))
        qi = ttnn.to_memory_config(qs, ttnn.DRAM_MEMORY_CONFIG)
        run("sdpa_mask dram q", lambda: f1(qi, "dram"))
        # chain time
        K0 = rep_up(md, torch.randn(T, 1, 256, D).to(torch.bfloat16))
        mt0 = rep_up(md, torch.zeros(T, 1, H, 256).to(torch.bfloat16))
        st0 = rep_up(md, torch.zeros(32, 32).to(torch.bfloat16))
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(4, 1), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        for kc in (64, 128, 256):
            prog = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(4, 1),
                q_chunk_size=0,
                k_chunk_size=kc,
                exp_approx_mode=False,
            )
            for gx in (4, 8):
                prog = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, 1 if gx == 4 else 4),
                    q_chunk_size=0,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                )
                run(
                    f"sdpa_mask time kc={kc} grid={gx}",
                    lambda: f"{chain_ms(md, lambda: ttnn.transformer.scaled_dot_product_attention_decode(qi, K0, K0, is_causal=False, attn_mask=mt0, attention_sink=st0, scale=D**-0.5, program_config=prog, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3:.1f} us",
                )

    # F3: paged_update_cache straight from the nlp k output
    def f3():
        cache = rep_up(md, torch.zeros(T, 1, 256, D).to(torch.bfloat16))
        idx = ttnn.from_torch(
            torch.tensor([3, 5, 7, 9], dtype=torch.int32),
            device=md,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        ttnn.experimental.paged_update_cache(cache, state["k"], update_idxs_tensor=idx, page_table=None)
        c = dev0(cache)
        kexp = xq[0, 0, :, H * D : H * D + D].float()
        got = torch.stack([c[t, 0, [3, 5, 7, 9][t]] for t in range(T)])
        return f"pcc {pcc(got, kexp):.5f}, other rows zero: {float(c[0,0,0].abs().max()) == 0.0}"

    run("paged_update_cache from nlp k", f3)

    # F3b: unpadded [1,T,1,512] tile tensor -> sharded without pad
    def f3b():
        row = rep_up(md, torch.randn(1, T, 1, D).to(torch.bfloat16))
        sh = ttnn.to_memory_config(row, user_cfg((32, D)))
        cache = rep_up(md, torch.zeros(T, 1, 256, D).to(torch.bfloat16))
        idx = ttnn.from_torch(
            torch.tensor([3, 5, 7, 9], dtype=torch.int32),
            device=md,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        ttnn.experimental.paged_update_cache(cache, sh, update_idxs_tensor=idx, page_table=None)
        c = dev0(cache)
        got = torch.stack([c[t, 0, [3, 5, 7, 9][t]] for t in range(T)])
        return f"pcc {pcc(got, dev0(row)[0, :, 0]):.5f}"

    run("to_memory_config w/o pad + update", f3b)

    # F4: addcmul with broadcast
    def f4():
        a = torch.randn(1, T, 32, D).to(torch.bfloat16)
        b = torch.randn(1, T, 32, D).to(torch.bfloat16)
        c = torch.randn(1, T, 1, D).to(torch.bfloat16)
        r = ttnn.addcmul(rep_up(md, a), rep_up(md, b), rep_up(md, c), value=1.0)
        ref = a.float() + b.float() * c.float()
        return f"pcc {pcc(dev0(r), ref):.5f} shape {tuple(r.shape)}"

    run("addcmul bcast", f4)

    # F5: transposes / reshape layouts
    def f5():
        a = torch.randn(1, 1, T, D).to(torch.bfloat16)
        r = ttnn.transpose(rep_up(md, a), 1, 2)
        return (
            f"transpose(1,2) [1,1,T,512]->{tuple(r.shape)} pcc {pcc(dev0(r)[0, :, 0], a[0, 0]):.5f} t={chain_ms(md, lambda: ttnn.transpose(rep_up_a, 1, 2)) * 1e3:.1f}us"
            if False
            else "skipped"
        )

    a_t = rep_up(md, torch.randn(1, 1, T, D).to(torch.bfloat16))

    def f5b():
        r = ttnn.transpose(a_t, 1, 2)
        exp = dev0(a_t)[0, 0, :T]
        return f"[1,1,T,512]->{tuple(r.shape)} pcc {pcc(dev0(r)[0, :, 0], exp):.5f} {chain_ms(md, lambda: ttnn.transpose(a_t, 1, 2)) * 1e3:.1f}us"

    run("transpose(1,2) kv", f5b)

    def f5c():
        r = ttnn.reshape(a_t, [1, T, 1, D])
        exp = dev0(a_t)[0, 0, :T]
        return f"reshape ->{tuple(r.shape)} pcc {pcc(dev0(r)[0, :, 0], exp):.5f} {chain_ms(md, lambda: ttnn.reshape(a_t, [1, T, 1, D])) * 1e3:.1f}us"

    run("reshape kv tile", f5c)

    def f5d():
        hq = rep_up(md, torch.randn(1, H, T, D).to(torch.bfloat16))
        r = ttnn.transpose(hq, 1, 2)
        exp = dev0(hq)[0].permute(1, 0, 2)  # [T,H,D]
        return f"[1,8,T,512]->{tuple(r.shape)} pcc {pcc(dev0(r)[0, :, :H], exp):.5f} {chain_ms(md, lambda: ttnn.transpose(hq, 1, 2)) * 1e3:.1f}us"

    run("transpose(1,2) heads", f5d)

    # F6: concat heads decode from the SDPA output
    def f6():
        o = state["sdpa_o"]  # [1,T,32?,512] DRAM
        sh = ttnn.to_memory_config(o, user_cfg((32, D)))
        c = ttnn.experimental.nlp_concat_heads_decode(sh, num_heads=H)
        ref = dev0(o)[0, :, :H].reshape(T, H * D)
        got = dev0(c).reshape(-1, H * D)[:T]
        return f"out {tuple(c.shape)} {c.memory_config().memory_layout} pcc {pcc(got, ref):.5f}"

    if "sdpa_o" in state:
        run("nlp_concat_heads_decode", f6)

    # F7: batched per-head wq_b matmul with broadcast in0 -> heads as batch
    def f7():
        qr = torch.randn(1, 1, T, 1280).to(torch.bfloat16)
        W = (torch.randn(1, H, 1280, D) * 0.02).to(torch.bfloat16)
        r = ttnn.matmul(rep_up(md, qr), rep_up(md, W), compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=4, x=8))
        ref = torch.einsum("tk,hkd->htd", qr[0, 0].float(), W[0].float())
        return f"shape {tuple(r.shape)} pcc {pcc(dev0(r)[0], ref):.5f}"

    run("batched head matmul bcast in0", f7)
