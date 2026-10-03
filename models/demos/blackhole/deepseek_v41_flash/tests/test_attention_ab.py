# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A/B: optimised tt/attention.py vs the baseline kept in tests/attention_ref_impl.py (and the torch reference), real
layer weights, several decode steps (state evolution incl. compressor group completion), plus in-trace timing.
Prints 'AB ...' lines."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tests import attention_ref_impl as OLD
from models.demos.blackhole.deepseek_v41_flash.tests.test_attn_probe_matmul import chain_ms
from models.demos.blackhole.deepseek_v41_flash.tt import attention as NEW
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager


def build(mod, md, mesh_config, ccl, blk, ratio, per_row, snap, source=None):
    weights = R.dequantized_attention_weights(blk)
    if ratio == 0:
        a = mod.DSV41Attention(md, mesh_config, ccl, weights, blk.attn.freqs_cis, users_per_row=per_row, max_seq=256)
        a.load_window(snap["window"])
        return a
    comp = blk.attn.compressor
    if source is None:
        cw = {"wkv": comp.wkv.weight.data.float(), "norm": comp.norm.weight.data.float()}
        if ratio > 1:
            cw["wgate"] = comp.wgate.weight.data.float()
        a = mod.DSV41CompressedAttention(
            md, mesh_config, ccl, weights, blk.attn.freqs_cis, ratio, cw, users_per_row=per_row, max_comp=128
        )
        a.load_state(snap["window"], snap["comp"], snap["kv_state"], snap["score_state"])
    else:
        a = mod.DSV41CompressedAttention(
            md,
            mesh_config,
            ccl,
            weights,
            blk.attn.freqs_cis,
            ratio,
            None,
            users_per_row=per_row,
            max_comp=128,
            source=source,
        )
        a.load_state(snap["window"], None, None, None)
    return a


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
@pytest.mark.parametrize("layer_id,S,steps", [(0, 24, 3), (2, 23, 4), (20, 24, 3), (2, 130, 3), (20, 120, 4)])
@torch.no_grad()
def test_attention_ab(mesh_device, layer_id, S, steps):
    torch.manual_seed(0)
    ref_kernels.FAKE_QUANT = False
    md = mesh_device
    rows, cols = tuple(md.shape)
    per_row, B = 4, rows * 4
    blk = R.build_layer(layer_id, max_batch_size=B, max_seq_len=256)
    ratio = blk.attn.compress_ratio

    def block_input(tok):
        h, pm = R.embed_tokens(tok)
        return blk.attn_norm(blk.hc_pre(h, pm))

    x_pre = block_input(torch.randint(1000, 100000, (B, S)))
    xs = [block_input(torch.randint(1000, 100000, (B, 1))) for _ in range(steps)]
    blk.attn(x_pre, 0)
    comp = blk.attn.compressor
    snap = dict(window=blk.attn.window_kv_cache.clone().float(), comp=None, kv_state=None, score_state=None)
    if ratio:
        snap.update(
            comp=blk.attn.compress_kv_cache[:, : S // ratio].clone().float(),
            kv_state=comp.kv_state.clone() if ratio > 1 else None,
            score_state=comp.score_state.clone() if ratio > 1 else None,
        )
    else:
        snap["window"] = snap["window"][:, :S]
    refs = [
        blk.attn(xs[i], S + i).float().reshape(B, 5120) for i in range(steps)
    ]  # sequential: caches evolve in the reference

    mesh_config = mesh_4x8()
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    old = build(OLD, md, mesh_config, ccl, blk, ratio, per_row, snap)
    new = build(NEW, md, mesh_config, ccl, blk, ratio, per_row, snap)
    shard = ttnn.ShardTensor2dMesh(md, dims=(2, None), mesh_shape=(rows, cols))
    to_dev = lambda t: ttnn.from_torch(
        t.reshape(1, 1, B, 5120).to(torch.bfloat16),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=shard,
    )
    to_host = lambda t: torch.cat(
        [ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).reshape(-1, 5120) for r in range(rows)]
    ).float()[:B]
    worst = 1.0
    for i in range(steps):
        x = to_dev(xs[i])
        o_old = to_host(old.forward(x, old.step_inputs(torch.full((B,), S + i))))
        st = new.step_inputs(torch.full((B,), S + i))
        o_new = to_host(new.forward(x, st))
        p_new, p_old, p_nn = R.pcc(o_new, refs[i]), R.pcc(o_old, refs[i]), R.pcc(o_new, o_old)
        print(
            f"AB layer {layer_id} step {i} (pos {S + i}): new-vs-torch {p_new:.5f}  old-vs-torch {p_old:.5f}  new-vs-old {p_nn:.5f}",
            flush=True,
        )
        worst = min(worst, p_new)
    st_o, st_n = old.step_inputs(torch.full((B,), S + steps)), new.step_inputs(torch.full((B,), S + steps))
    x = to_dev(xs[0])
    t_old = chain_ms(md, lambda: old.forward(x, st_o))
    t_new = chain_ms(md, lambda: new.forward(x, st_n))
    print(f"AB layer {layer_id} timing: old {t_old * 1e3:.1f} us  new {t_new * 1e3:.1f} us", flush=True)
    ref_kernels.FAKE_QUANT = True
    assert worst > 0.99
