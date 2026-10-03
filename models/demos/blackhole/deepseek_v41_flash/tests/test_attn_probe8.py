# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe 8: 9-head trick (kv as head 0 of the q tile) feasibility. Prints 'P8 ...'."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_features import dev0, pcc, rep_up, user_cfg

T, D = 4, 512


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
def test_probe8(mesh_device):
    md = mesh_device
    torch.manual_seed(1)
    ckc = ttnn.init_device_compute_kernel_config(
        md.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def run(name, fn):
        try:
            print(f"P8 {name}: {fn()}", flush=True)
        except Exception as e:
            print(f"P8 {name}: FAIL {str(e).splitlines()[0][:200]!r}", flush=True)

    # (a) paged_update_cache from an interleaved DRAM tensor
    def fa():
        cache = rep_up(md, torch.zeros(T, 1, 256, D).to(torch.bfloat16))
        row = rep_up(md, torch.randn(1, T, 32, D).to(torch.bfloat16))
        idx = ttnn.from_torch(
            torch.tensor([3, 5, 7, 9], dtype=torch.int32),
            device=md,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        ttnn.experimental.paged_update_cache(cache, row, update_idxs_tensor=idx, page_table=None)
        c = dev0(cache)
        got = torch.stack([c[t, 0, [3, 5, 7, 9][t]] for t in range(T)])
        return f"pcc {pcc(got, dev0(row)[0, :, 0]):.5f}"

    run("paged_update_cache from DRAM-interleaved", fa)

    # (b) 9 heads
    x = torch.randn(1, 1, T, 11 * D).to(torch.bfloat16)
    state = {}

    def fb():
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            rep_up(md, x), num_heads=9, num_kv_heads=1, memory_config=user_cfg((32, D))
        )
        state["q"] = q
        exp = x[0, 0, :, : 9 * D].reshape(T, 9, D).float()
        return f"q {tuple(q.shape)} pcc {pcc(dev0(q)[0, :, :9], exp):.5f}; k pcc {pcc(dev0(k)[0, :, 0], x[0, 0, :, 9 * D : 10 * D].float()):.5f}"

    run("nlp_create 9 heads", fb)

    # (c) SDPA with 9 logical heads, sinks shifted
    def fc():
        H = 9
        S = 256
        K = torch.randn(T, 1, S, D).to(torch.bfloat16)
        valid = torch.zeros(T, 1, 1, S, dtype=torch.bool)
        valid[..., :40] = True
        valid[..., 128:150] = True
        mask = torch.where(valid, 0.0, -1e9).expand(T, 1, H, S).contiguous()
        sink = torch.randn(H)
        sink[0] = 0
        scale = D**-0.5
        sinks = torch.zeros(32, 32)
        sinks[:H, 0] = sink / scale
        q = ttnn.to_memory_config(state["q"], ttnn.DRAM_MEMORY_CONFIG)
        Kt, mt, st = rep_up(md, K), rep_up(md, mask.to(torch.bfloat16)), rep_up(md, sinks.to(torch.bfloat16))
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 1),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
            max_cores_per_head_batch=2,
        )
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
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
        qd = dev0(q)[0, :, :H].float()
        sc = torch.einsum("thd,tsd->ths", qd, K[:, 0].float()) * scale + mask[:, 0]
        p = torch.softmax(torch.cat([sc, sink.reshape(1, H, 1).expand(T, H, 1)], -1), -1)[..., :S]
        ref = torch.einsum("ths,tsd->thd", p, K[:, 0].float())
        state["o"] = o
        return f"out {tuple(o.shape)} pcc {pcc(dev0(o)[0, :, :H], ref):.5f}"

    run("sdpa 9 heads", fc)

    def fd():
        o = state["o"]
        sh = ttnn.to_memory_config(o, user_cfg((32, D)))
        c = ttnn.experimental.nlp_concat_heads_decode(sh, num_heads=9)
        ref = dev0(o)[0, :, :9].reshape(T, 9 * D)
        return f"out {tuple(c.shape)} pcc {pcc(dev0(c).reshape(-1, 9 * D)[:T], ref):.5f}"

    run("nlp_concat 9 heads", fd)
