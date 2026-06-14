# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Op-level profile of ONE SigLIP encoder block (forward_bs path) on a single
chip, to find the real hotspot (matmul vs SDPA vs reshard vs LN).

Times each region with device syncs. Run:
    source models/experimental/pi0_5/local_env.sh
    python_env/bin/python models/experimental/pi0_5/tests/perf/bench_siglip_block.py
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.ttnn_siglip import (
    SigLIPBlockTTNN,
    _SIGLIP_BS_GRID,
    _make_bs_memcfg,
)

WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))
ITERS = int(os.environ.get("BENCH_ITERS", "30"))
CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream")
BS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
SEQ = 256


def main():
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = Pi0_5WeightLoader(CKPT)
    vw = loader.categorized_weights["vlm_vision"]

    # extract layer-0 weights (vision_slice._layer_weights style)
    from models.experimental.pi0_5.tt.tt_bh_glx.vision_slice import _layer_weights

    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        block = SigLIPBlockTTNN(cfg, _layer_weights(vw, 0), dev)
        gx, gy = _SIGLIP_BS_GRID
        M = BS * SEQ
        H = cfg.hidden_size

        hidden = torch.randn(1, 1, M, H)
        INTER_PAD = 4608
        # Mirror SigLIPVisionTowerTTNN._get_bs_memcfgs EXACTLY: batch folded into M,
        # b arg = 1, total_m = BS*SEQ, qkv width = 144*32, attn width = 48*32.
        bs_hidden = _make_bs_memcfg(1, M, H, gx, gy)
        bs_qkv = _make_bs_memcfg(1, M, 144 * 32, gx, gy)
        bs_attn = _make_bs_memcfg(1, M, 48 * 32, gx, gy)
        bs_inter = _make_bs_memcfg(1, M, INTER_PAD, gx, gy)
        h_bs = ttnn.to_memory_config(
            ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev),
            bs_hidden,
            dtype=ttnn.bfloat16,
        )

        # time full block forward_bs
        def block_fwd():
            return block.forward_bs(h_bs, bs_hidden, bs_qkv, bs_attn, bs_inter, n_batch=BS, n_seq=SEQ)

        # time just attention.forward_bs and mlp.forward_bs separately
        normed = block._sharded_layer_norm_bs(h_bs, block.ln1_weight, block.ln1_bias, bs_hidden)

        def attn_fwd():
            return block.attention.forward_bs(normed, bs_hidden, bs_qkv, bs_attn, n_batch=BS, n_seq=SEQ)

        def mlp_fwd():
            return block.mlp.forward_bs(normed, bs_hidden, bs_inter)

        def ln_fwd():
            return block._sharded_layer_norm_bs(h_bs, block.ln1_weight, block.ln1_bias, bs_hidden)

        def t(fn):
            ts = []
            for i in range(WARMUP + ITERS):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                fn()
                ttnn.synchronize_device(dev)
                if i >= WARMUP:
                    ts.append((time.perf_counter() - t0) * 1e3)
            return statistics.mean(ts)

        print(f"\n=== SigLIP block op profile  M={M} grid={gx}x{gy} warmup={WARMUP} iters={ITERS} ===")
        print(f"  full block   : {t(block_fwd):7.4f} ms")
        print(f"  layernorm x1 : {t(ln_fwd):7.4f} ms")
        print(f"  attention    : {t(attn_fwd):7.4f} ms")
        print(f"  mlp          : {t(mlp_fwd):7.4f} ms")
        print(f"  (block has 2 LN + 1 attn + 1 mlp + 2 residual adds)")
        # --- attention-internal breakdown ---
        att = block.attention
        from models.experimental.pi0_5.tt.ttnn_siglip import _build_bs_matmul_pcfg, _SIGLIP_BS_GRID as _G

        gx2, gy2 = _G
        m_tiles = M // 32
        hidden_t = H // 32
        qkv_pcfg = _build_bs_matmul_pcfg(m_tiles, hidden_t, att._qkv_n_tiles, gx2, gy2, dst_budget=4)

        def _qkv():
            return ttnn.linear(
                normed,
                att.wqkv,
                bias=att.bqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=bs_qkv,
                compute_kernel_config=att.compute_kernel_config_hifi4,
                program_config=qkv_pcfg,
            )

        xqkv0 = _qkv()

        def _reshard_in():
            x = ttnn.sharded_to_interleaved(xqkv0, memory_config=ttnn.L1_MEMORY_CONFIG)
            return ttnn.reshape(x, (BS, 1, SEQ, int(xqkv0.shape[-1])))

        xr = _reshard_in()

        def _heads():
            return ttnn.experimental.nlp_create_qkv_heads(
                xr,
                num_heads=att.num_heads,
                num_kv_heads=att.num_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        qh, kh, vh = _heads()
        from models.experimental.pi0_5.tt.ttnn_common import sdpa_prefill_chunk_sizes, get_sdpa_exp_approx_mode

        qc, kc = sdpa_prefill_chunk_sizes(SEQ, SEQ)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=att.grid_size,
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=get_sdpa_exp_approx_mode(SEQ),
        )

        def _sdpa():
            return ttnn.transformer.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                is_causal=False,
                scale=att.scale,
                program_config=sdpa_cfg,
                compute_kernel_config=att.compute_kernel_config_sdpa,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        print(f"  -- attention internals --")
        print(f"  qkv matmul   : {t(_qkv):7.4f} ms")
        print(f"  reshard->heads-in: {t(_reshard_in):7.4f} ms")
        print(f"  create_heads : {t(_heads):7.4f} ms")
        print(f"  SDPA         : {t(_sdpa):7.4f} ms")
        # --- residual add timing (BS add over M x H) ---
        attn_out = block.attention.forward_bs(normed, bs_hidden, bs_qkv, bs_attn, n_batch=BS, n_seq=SEQ)

        def _resid_add():
            return ttnn.add(h_bs, attn_out, memory_config=bs_hidden)

        print(f"  residual add : {t(_resid_add):7.4f} ms (x2 per block)")
        # --- full block minus a no-op: re-measure full + sum check ---
        parts = 2 * t(ln_fwd) + t(attn_fwd) + t(mlp_fwd) + 2 * t(_resid_add)
        print(f"  sum-of-parts : {parts:7.4f} ms")

        # --- residual add variants ---
        def _add_default():
            return ttnn.add(h_bs, attn_out, memory_config=bs_hidden)

        def _add_bf8():
            return ttnn.add(h_bs, attn_out, memory_config=bs_hidden, dtype=ttnn.bfloat8_b)

        # in-place: write back into h_bs buffer
        def _add_inplace():
            ttnn.add(h_bs, attn_out, memory_config=bs_hidden, output_tensor=h_bs)
            return h_bs

        print(f"  add default  : {t(_add_default):7.4f} ms  dtype_in={h_bs.dtype}")
        try:
            print(f"  add bf8 out  : {t(_add_bf8):7.4f} ms")
        except Exception as e:
            print(f"  add bf8 out  : ERR {repr(e)[:50]}")
        try:
            print(f"  add inplace  : {t(_add_inplace):7.4f} ms")
        except Exception as e:
            print(f"  add inplace  : ERR {repr(e)[:50]}")
        # --- fused residual+LN vs separate add then LN ---
        pc_ln = block._get_bs_ln_pcfg(int(h_bs.shape[0]), int(h_bs.shape[-2]))

        def _sep_add_then_ln():
            hh = ttnn.add(h_bs, attn_out, memory_config=bs_hidden)
            return ttnn.layer_norm(
                hh,
                weight=block.ln2_weight,
                bias=block.ln2_bias,
                epsilon=block.config.layer_norm_eps,
                memory_config=bs_hidden,
                program_config=pc_ln,
            )

        def _fused_resid_ln():
            return ttnn.layer_norm(
                h_bs,
                residual_input_tensor=attn_out,
                weight=block.ln2_weight,
                bias=block.ln2_bias,
                epsilon=block.config.layer_norm_eps,
                memory_config=bs_hidden,
                program_config=pc_ln,
            )

        try:
            print(f"  sep add+LN   : {t(_sep_add_then_ln):7.4f} ms")
        except Exception as e:
            print(f"  sep add+LN   : ERR {repr(e)[:60]}")
        try:
            r = _fused_resid_ln()
            print(f"  fused res+LN : {t(_fused_resid_ln):7.4f} ms  out_shape={tuple(r.shape)}")
        except Exception as e:
            print(f"  fused res+LN : ERR {repr(e)[:80]}")

    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
