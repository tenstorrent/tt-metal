# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: the text decode-step calls with the shapes/layouts the Quasar config uses on the emulator's 2x1 grid.

Shapes come from progress.log of the ttsim WH 2x1 run: norm [1,1,32,2560] width-sharded L1, fused QKV [1,1,1,6144],
q/k heads height-sharded L1, bf16 paged KV cache [8 blocks, 8 kv heads, 32, 128], decode position 78.
Run with TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=3,2 on craq-sim.
"""
import math

import torch
import ttnn
from models.tt_transformers.tt.common import get_rot_transformation_mat

NH, NKV, HD, DIM, BLOCKS, BS, POS = 32, 8, 128, 2560, 8, 32, 78


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def report(name, fn):
    try:
        print(f"{name:28s} {fn()}", flush=True)
    except Exception as e:
        lines = [ln.strip() for ln in str(e).splitlines() if ln.strip() and not ln.strip().startswith("---")]
        print(f"{name:28s} FAIL: {' | '.join(lines[:2])[:200]}", flush=True)


def hs_l1(rows, cols):
    """Height-sharded L1 on core (0,0), one [rows, cols] shard (decode batch 1 padded to a tile)."""
    return ttnn.create_sharded_memory_config(
        shape=(rows, cols),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    print(f"grid={g.x}x{g.y}", flush=True)
    up = lambda t, mc=ttnn.DRAM_MEMORY_CONFIG, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
        t, dtype=dt, layout=layout, device=dev, memory_config=mc
    )
    bf = lambda t: t.bfloat16().float()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )

    # 1. Decode RMSNorm, width-sharded over the 2 cores.
    def norm():
        x, w = torch.randn(1, 1, 32, DIM), torch.randn(DIM)
        ws = ttnn.create_sharded_memory_config(
            shape=(32, DIM // (g.x * g.y)),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))}),
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y),
            subblock_w=1,
            block_h=1,
            block_w=DIM // 32 // (g.x * g.y),
            inplace=False,
        )
        out = ttnn.rms_norm(
            up(x, ws),
            epsilon=1e-6,
            weight=up(w.reshape(1, 1, DIM // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT),
            program_config=prog,
            memory_config=ws,
            compute_kernel_config=ckc,
        )
        ref = bf(x) * torch.rsqrt(bf(x).pow(2).mean(-1, keepdim=True) + 1e-6) * bf(w)
        return f"pcc={pcc(ttnn.to_torch(out).float(), ref):.5f}"

    report("rms_norm (width-sharded)", norm)

    # 2. Split fused QKV [1,1,1,6144] into heads.
    qkv = torch.randn(1, 1, 1, (NH + 2 * NKV) * HD)

    def heads():
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            up(qkv, ttnn.L1_MEMORY_CONFIG),
            num_heads=NH,
            num_kv_heads=NKV,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        got = [ttnn.to_torch(t).float()[:, :1] for t in (q, k, v)]
        refs = [bf(qkv)[..., : NH * HD].reshape(1, 1, NH, HD)]
        refs.append(bf(qkv)[..., NH * HD : (NH + NKV) * HD].reshape(1, 1, NKV, HD))
        refs.append(bf(qkv)[..., (NH + NKV) * HD :].reshape(1, 1, NKV, HD))
        return " ".join(f"{n}={pcc(a[..., : r.shape[2], :], r):.5f}" for n, a, r in zip("qkv", got, refs))

    report("nlp_create_qkv_heads_decode", heads)

    # 3. Decode rotary on the height-sharded q heads.
    def rope():
        x = torch.randn(1, 1, 32, HD)
        ang = torch.randn(1, 1, 1, HD // 2)
        cos = torch.stack([ang.cos(), ang.cos()], -1).reshape(1, 1, 1, HD)
        sin = torch.stack([ang.sin(), ang.sin()], -1).reshape(1, 1, 1, HD)
        trans = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE)
        out = ttnn.experimental.rotary_embedding_llama(
            up(x, hs_l1(32, HD)),
            up(cos, hs_l1(32, HD)),
            up(sin, hs_l1(32, HD)),
            up(trans, hs_l1(32, 32)),
            is_decode_mode=True,
        )
        rot = torch.stack([-bf(x)[..., 1::2], bf(x)[..., 0::2]], -1).reshape(x.shape)
        ref = bf(x) * bf(cos) + rot * bf(sin)
        return f"pcc={pcc(ttnn.to_torch(out).float(), ref):.5f}"

    report("rotary_embedding_llama decode", rope)

    # 4. Paged KV cache update at position POS, then 5. decode SDPA over the paged cache.
    page_table = torch.randperm(BLOCKS).reshape(1, BLOCKS).to(torch.int32)
    k_cache, v_cache = torch.randn(BLOCKS, NKV, BS, HD), torch.randn(BLOCKS, NKV, BS, HD)
    tk, tv = up(k_cache), up(v_cache)
    tpt = up(page_table, dt=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tpos = up(torch.tensor([POS], dtype=torch.int32), dt=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k_new = torch.randn(1, 1, NKV, HD)

    def update():
        ttnn.experimental.paged_update_cache(
            tk,
            up(torch.nn.functional.pad(k_new, (0, 0, 0, 32 - NKV)), hs_l1(32, HD)),
            update_idxs_tensor=tpos,
            page_table=tpt,
        )
        k_cache[int(page_table[0, POS // BS]), :, POS % BS, :] = k_new[0, 0]
        got = ttnn.to_torch(tk).float()
        return f"pcc={pcc(got, bf(k_cache)):.5f} exact_row={torch.equal(got[int(page_table[0, POS // BS]), :, POS % BS], bf(k_new)[0, 0])}"

    report("paged_update_cache (bf16)", update)

    def sdpa():
        q = torch.randn(1, 1, NH, HD)
        prog = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y), q_chunk_size=0, k_chunk_size=0, exp_approx_mode=False
        )
        out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
            up(torch.nn.functional.pad(q, (0, 0, 0, 32 - NH)) if NH < 32 else q, hs_l1(32, HD)),
            tk,
            tv,
            page_table_tensor=tpt,
            cur_pos_tensor=tpos,
            scale=1 / math.sqrt(HD),
            program_config=prog,
            compute_kernel_config=ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # as the model calls it; GQA rejects a sharded output
        )
        blocks = page_table[0].tolist()
        keys = torch.cat([bf(k_cache)[b] for b in blocks], dim=1)[:, : POS + 1]  # [NKV, POS+1, HD]
        vals = torch.cat([bf(v_cache)[b] for b in blocks], dim=1)[:, : POS + 1]
        rep = NH // NKV
        att = torch.softmax(
            torch.einsum("hd,hsd->hs", bf(q)[0, 0], keys.repeat_interleave(rep, 0)) / math.sqrt(HD), dim=-1
        )
        ref = torch.einsum("hs,hsd->hd", att, vals.repeat_interleave(rep, 0))
        got = ttnn.to_torch(out).float().reshape(-1, HD)[:NH]
        return f"pcc={pcc(got, ref):.5f}"

    report("decode SDPA (experimental)", sdpa)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
