# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fully-traced e2e pipeline assembly: vision -> prefill -> denoise -> actions,
all three stages traced (FABRIC_1D, collinear layout), host-bounce cross-stage
hand-offs. First milestone: runs e2e and produces finite actions of the right
shape (numerical-vs-torch with upstream masks is a follow-up; here mask=None).

Stages (each reuses the validated prototype logic):
  vision : real images -> eager embed -> traced 27-block core -> eager post_ln+projector
  prefix : reshape vision feats + host lang-embed + concat  (host-side)
  prefill: traced 18 VLM blocks (snake) -> per-layer KV
  KVmig  : host-bounce KV -> denoise mesh (chip c = layers 3c..3c+2)
  denoise: traced Euler loop (suffix embed + KV cross-attn + final proj) -> actions
"""

import os
import sys
import time

# Bake in the validated production perf flags (single source of truth:
# _bench_runs/pi05_production.env, 97.2% LIBERO). setdefault => an explicit
# value in the caller's environment still wins. Set BEFORE the pi0_5 imports
# so the building blocks (ttnn_gemma/siglip/common) read them at import time.
_PROD_PERF_FLAGS = {
    # matmul / kernel tuning (no PCC impact)
    "PI0_EXPERT_MM_LOFI": "1",
    "PI0_ROPE_TABLES_L1": "1",
    "PI0_MM_SWEEP_V2": "1",
    "PI0_DENOISE_MM_TUNE": "1",
    "PI0_PREFILL_MM_TUNE": "1",
    # attention / mask handling
    "PI0_UPSTREAM_MASKS": "1",
    "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT": "1",
    "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT": "1",
    "PI0_MQA_HEAD_SPLIT": "1",
    "PI0_SDPA_DENOISE_K_FORCE": "96",
    # VLM single-pass at bs=3 (training-spec correct)
    "PI0_NUM_CAMERAS": "3",
    "PI0_VLM_CHUNK_SIZE": "1024",
    "PI0_VLM_MLP_BF8_OUT": "1",
    "PI0_VLM_MLP_MINIMAL": "1",
    "PI0_VLM_MINIMAL_CFG": "4,8,8,1,8",
    "PI0_SIGLIP_USE_FOLD": "1",
    # denoise steps (5 = perf-tuned default, 97.2% LIBERO)
    "PI05_NUM_DENOISE_STEPS": "5",
}
for _k, _v in _PROD_PERF_FLAGS.items():
    os.environ.setdefault(_k, _v)

import torch
import ttnn

from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.ttnn_common import get_ln_weight_memory_config, tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    GemmaBlockTTNN,
    ada_rms_norm_no_gate_ttnn,
    precompute_freqs_cis_meta_format,
)
from models.experimental.pi0_5.tt.ttnn_pi0_5_model import _MASK_VAL, _precompute_rope_table_torch
from models.experimental.pi0_5.tt.ttnn_siglip import MultiModalProjectorTTNN, SigLIPBlockTTNN
from models.experimental.pi0_5.tt.tt_bh_glx import stages
from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import _inject_adarms_weights_to_submesh
from models.experimental.pi0_5.tt.tt_bh_glx.suffix_slice import SuffixSlice
from models.experimental.pi0_5.tt.tt_bh_glx.vision_slice import SigLIPEmbedSlice, _layer_weights
from models.experimental.pi0_5.tt.tt_bh_glx.vlm_slice import _load_block_weights_to_submesh

CKPT = os.environ.get(
    "PI05_CHECKPOINT_DIR",
    "/home/tt-admin/sdawle/tt-metal/models/experimental/pi0_5/weights/pi05_libero_upstream",
)
N_CAMS = int(os.environ["PI0_NUM_CAMERAS"])  # 3 -> prefix = 3*256 + 256 lang = 1024 (the VLM block)
LANG_LEN = 256
N_STEPS = int(os.environ["PI05_NUM_DENOISE_STEPS"])
SEED = 42
SV_ATTRS = [
    ("ln1_weight", ""),
    ("ln1_bias", ""),
    ("ln2_weight", ""),
    ("ln2_bias", ""),
    ("wqkv", "attention"),
    ("bqkv", "attention"),
    ("wo", "attention"),
    ("bo", "attention"),
    ("fc1_weight", "mlp"),
    ("fc1_bias", "mlp"),
    ("fc2_weight", "mlp"),
    ("fc2_bias", "mlp"),
]


def _sv_get(b, n, o):
    return getattr(b if o == "" else getattr(b, o), n, None)


def _sv_set(b, n, o, v):
    setattr(b if o == "" else getattr(b, o), n, v)


def _shard(stacked, dt, lay, mesh):
    return ttnn.from_torch(
        stacked,
        dtype=dt,
        layout=lay,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def _repl(t, mesh, dt=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t,
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=mc,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def main():
    def log(m):
        print(f"[e2e] {m}", flush=True)

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CKPT), num_denoising_steps=N_STEPS)
    scfg, vcfg, ecfg = cfg.siglip_config, cfg.vlm_config, cfg.expert_config
    ah = cfg.action_horizon
    ah_pad = ((ah + 31) // 32) * 32
    adim = cfg.action_dim
    npatch = (scfg.image_size // scfg.patch_size) ** 2
    prefix_len = npatch * N_CAMS + LANG_LEN
    prefix_pad = ((prefix_len + 31) // 32) * 32
    log(f"ah={ah} ah_pad={ah_pad} adim={adim} prefix_len={prefix_len} prefix_pad={prefix_pad}")

    loader = Pi0_5WeightLoader(CKPT)
    vw, pw = loader.categorized_weights["vlm_vision"], loader.categorized_weights["vlm_projector"]
    lw_all = loader.categorized_weights["vlm_language"]
    ew = loader.categorized_weights["action_expert"]
    pi0proj = loader.categorized_weights["pi0_projections"]

    torch.manual_seed(SEED)
    images = torch.randn(N_CAMS, 3, scfg.image_size, scfg.image_size, dtype=torch.float32)
    lang_tokens = torch.randint(0, 256000, (1, LANG_LEN), dtype=torch.int64)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []

    def carve(shape, off):
        sm = parent.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(*off))
        subs.append(sm)
        return sm

    try:
        embed_chip = carve((1, 1), (6, 3))
        vblk = carve((1, 3), (6, 0))  # vision 27-block core
        prefill = carve((6, 3), (0, 0))  # 18 VLM blocks (snake)
        denoise = carve((6, 1), (0, 3))  # expert + Euler
        scratch = carve((1, 1), (7, 0))

        # ===================== VISION =====================
        embed = SigLIPEmbedSlice(scfg, vw, embed_chip)
        img_m = ttnn.from_torch(
            images,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=embed_chip,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        emb_torch = ttnn.to_torch(embed.forward(img_m))  # (N_CAMS,256,1152)
        # traced 27-block core on vblk (chip (0,c)=layers 9c..9c+8)
        vlt = [_layer_weights(vw, L) for L in range(27)]
        vpl, vmeta = [], {}
        for L in range(27):
            b = SigLIPBlockTTNN(scfg, vlt[L], scratch)
            d = {}
            for n, o in SV_ATTRS:
                t = _sv_get(b, n, o)
                if t is None:
                    continue
                d[(n, o)] = ttnn.to_torch(t)
                vmeta.setdefault((n, o), (t.dtype, t.layout))
                ttnn.deallocate(t)
            vpl.append(d)

        def vblock(local):
            layers = [c * 9 + local for c in range(3)]
            blk = SigLIPBlockTTNN(scfg, vlt[layers[0]], vblk)
            for key in vmeta:
                _sv_set(blk, key[0], key[1], _shard(torch.stack([vpl[L][key] for L in layers], 0), *vmeta[key], vblk))
            return blk

        vblocks = [vblock(L) for L in range(9)]
        emb_v = _repl(emb_torch, vblk)

        def vchain(h):
            cur = h
            for c in range(3):
                for L in range(9):
                    cur = vblocks[L].forward(cur)
                if c < 2:
                    cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(0, c),
                        ttnn.MeshCoordinate(0, c + 1),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur

        vchain(emb_v)
        ttnn.synchronize_device(vblk)
        vtid = ttnn.begin_trace_capture(vblk, cq_id=0)
        vbo = vchain(emb_v)
        ttnn.end_trace_capture(vblk, vtid, cq_id=0)
        ttnn.execute_trace(vblk, vtid, cq_id=0, blocking=True)
        # chip (0,2) holds the last 9 SigLIP layers; ConcatMeshToTensor interleaves
        # device-batch, so chip 2's lane is rows [2*N_CAMS : 3*N_CAMS] -> (N_CAMS,256,1152).
        vblk_out = ttnn.to_torch(vbo, mesh_composer=ttnn.ConcatMeshToTensor(vblk, dim=0))[2 * N_CAMS : 3 * N_CAMS]
        # post_ln + projector (eager) on embed_chip
        post_w = vw.get("post_layernorm.weight") or vw.get("vision_model.post_layernorm.weight")
        post_b = vw.get("post_layernorm.bias") or vw.get("vision_model.post_layernorm.bias")
        plw = tensor_1d_to_2d_ttnn(post_w, embed_chip, dtype=ttnn.bfloat16, memory_config=get_ln_weight_memory_config())
        plb = (
            tensor_1d_to_2d_ttnn(post_b, embed_chip, dtype=ttnn.bfloat16, memory_config=get_ln_weight_memory_config())
            if post_b is not None
            else None
        )
        projector = MultiModalProjectorTTNN(pw, embed_chip)
        vblk_m = ttnn.from_torch(
            vblk_out.reshape(N_CAMS, npatch, scfg.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=embed_chip,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        feats = ttnn.to_torch(
            projector.forward(
                ttnn.layer_norm(
                    vblk_m, weight=plw, bias=plb, epsilon=scfg.layer_norm_eps, memory_config=ttnn.L1_MEMORY_CONFIG
                )
            )
        )
        log(f"VISION done: feats {tuple(feats.shape)} finite={torch.isfinite(feats).all().item()}")

        # ===================== PREFIX (host) =====================
        embed_tok = lw_all.get("model.embed_tokens.weight")
        if embed_tok is None:
            embed_tok = lw_all["lm_head.weight"]  # tied embeddings
        scale = float(vcfg.width) ** 0.5
        lang_emb = (embed_tok[lang_tokens[0].long()] * scale).unsqueeze(0)  # (1, LANG_LEN, 2048)
        vis_flat = feats.reshape(1, N_CAMS * npatch, vcfg.width).float()
        prefix = torch.cat([vis_flat, lang_emb.float()], dim=1)  # (1, prefix_len, 2048)
        if prefix_pad > prefix_len:
            prefix = torch.nn.functional.pad(prefix, (0, 0, 0, prefix_pad - prefix_len))
        log(f"PREFIX built {tuple(prefix.shape)}")

        # ===================== PREFILL (traced snake + KV) =====================
        snake = stages.prefill_snake_order(6, 3)

        def c2l(r, c):
            return 3 * r + (c if r % 2 == 0 else 2 - c)

        def snake_col_e2e(r, j):  # prefill column holding layer 3r+j (inverse of c2l)
            return j if r % 2 == 0 else 2 - j

        ppl, pmeta = [], {}
        for L in range(18):
            bw = _load_block_weights_to_submesh(lw_all, L, scratch)
            d = {}
            for k, v in bw.items():
                d[k] = ttnn.to_torch(v)
                pmeta.setdefault(k, (v.dtype, v.layout))
                ttnn.deallocate(v)
            ppl.append(d)
        pshard = {}
        for k in ppl[0]:
            order = [ppl[c2l(i // 3, i % 3)][k] for i in range(18)]
            pshard[k] = _shard(torch.stack(order, 0), *pmeta[k], prefill)
        pblock = GemmaBlockTTNN(vcfg, pshard, 0, prefill, None, None)
        pcos, psin = precompute_freqs_cis_meta_format(vcfg.head_dim, prefix_pad, prefill, base=vcfg.rope_base)
        prefix_m = _repl(prefix.to(torch.float32), prefill)

        def pchain(h):
            cur, kvs = h, []
            for k in range(18):
                cur, nkv = pblock.forward(cur, pcos, psin, None, None, None, True)
                kvs.append(nkv)
                if k < 17:
                    cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(*snake[k]),
                        ttnn.MeshCoordinate(*snake[k + 1]),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur, kvs

        pchain(prefix_m)
        ttnn.synchronize_device(prefill)
        ptid = ttnn.begin_trace_capture(prefill, cq_id=0)
        _, pkvs = pchain(prefix_m)
        ttnn.end_trace_capture(prefill, ptid, cq_id=0)
        ttnn.execute_trace(prefill, ptid, cq_id=0, blocking=True)
        log(f"PREFILL done: 18 KV, K[0] {tuple(pkvs[0][0].shape)}")

        # ===================== KV migration -> denoise (on-device p2p + socket) =====================
        # Per row r, in-mesh point_to_point gathers layer-(3r+j) KV onto the col-2
        # chip (r,2), then ONE adjacent FABRIC_1D socket (r,2)->(r,3) lands it on
        # denoise chip r as past_k[j] (chip r = layer 3r+j). Fully on-device,
        # replacing the 36x full-mesh-gather host bounce (~314ms; see PI05_E2E_TIMING).
        # FABRIC_1D sockets are adjacent-only, hence the p2p-to-col-2 first; pkvs and
        # past_k share per-chip shape (1,1,P,hd) so shards move directly. Expert
        # cross-attn concat needs bf8_b, so normalize the per-layer KV first.
        _SOCK_PAGE = 4096
        _row_sock = {}

        def _sock_for_row(r):
            if r not in _row_sock:
                conn = ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(r, 2), ttnn.CoreCoord(0, 0)),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(r, 0), ttnn.CoreCoord(0, 1)),
                )
                mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, _SOCK_PAGE * 4)
                _row_sock[r] = ttnn.create_socket_pair(prefill, denoise, ttnn.SocketConfig([conn], mem))
            return _row_sock[r]

        def _kv_norm(side):
            # bf16 DRAM-interleaved TILE per-layer KV on prefill: p2p does NOT
            # support bf8_b ("datum for bfp8 is invalid"), and DRAM-interleaved is
            # the p2p-safe layout. pkvs[k][side] per-chip = (1,1,P,hd). Typecast to
            # bf8_b happens AFTER the socket, on the denoise side.
            out = []
            for k in range(18):
                t = pkvs[k][side]
                if t.dtype != ttnn.bfloat16:
                    t = ttnn.typecast(t, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                out.append(ttnn.to_memory_config(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG))
            return out

        def _recv_bufs():
            return [
                _shard(torch.zeros(6, 1, prefix_pad, ecfg.head_dim), ttnn.bfloat16, ttnn.TILE_LAYOUT, denoise)
                for _ in range(3)
            ]

        def migrate_side(side, out):
            layers = _kv_norm(side)
            for j in range(3):
                for r in range(6):
                    sc = snake_col_e2e(r, j)
                    if sc != 2:  # arg order: (input, FROM_source, TO_dest)
                        ttnn.point_to_point(
                            layers[3 * r + j],
                            ttnn.MeshCoordinate(r, sc),
                            ttnn.MeshCoordinate(r, 2),
                            topology=ttnn.Topology.Linear,
                            output_tensor=layers[3 * r + j],
                        )
            ttnn.synchronize_device(prefill)
            for j in range(3):
                for r in range(6):
                    s_sock, r_sock = _sock_for_row(r)
                    ttnn.experimental.send_direct_async(layers[3 * r + j], s_sock)
                    ttnn.experimental.recv_direct_async(out[j], r_sock)
            ttnn.synchronize_device(denoise)

        past_k, past_v = _recv_bufs(), _recv_bufs()
        migrate_side(0, past_k)
        migrate_side(1, past_v)
        # expert cross-attn concat needs bf8_b (matches per-step k_rope dtype)
        past_k = [ttnn.typecast(t, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in past_k]
        past_v = [ttnn.typecast(t, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in past_v]
        log("KV migrated to denoise (on-device p2p+socket, 3 layers/chip)")

        # ===================== steady-state host-bounce timing (opt-in) =====================
        # PI05_E2E_TIMING=1 measures the PURE per-inference transfer cost of the two
        # cross-stage host bounces, with all ops already built+JIT-warmed (so this
        # excludes one-time projector construction / first-call compilation). N reps,
        # report mean. This is the latency the socket+p2p redesign would remove.
        if os.environ.get("PI05_E2E_TIMING", "").lower() in ("1", "true", "yes", "on"):
            N = 10

            def _bench(fn):
                fn()  # warm
                t0 = time.perf_counter()
                for _ in range(N):
                    fn()
                return 1e3 * (time.perf_counter() - t0) / N

            def _v_d2h():
                ttnn.execute_trace(vblk, vtid, cq_id=0, blocking=True)
                ttnn.to_torch(vbo, mesh_composer=ttnn.ConcatMeshToTensor(vblk, dim=0))[2 * N_CAMS : 3 * N_CAMS]

            def _prefix_h2d():
                _repl(prefix.to(torch.float32), prefill)

            def _kv_host():  # OLD host-bounce: 36 full-mesh gathers + 6 reshards
                kk = [
                    ttnn.to_torch(pkvs[k][0], mesh_composer=ttnn.ConcatMeshToTensor(prefill, dim=0))[
                        snake[k][0] * 3 + snake[k][1]
                    ]
                    for k in range(18)
                ]
                vv = [
                    ttnn.to_torch(pkvs[k][1], mesh_composer=ttnn.ConcatMeshToTensor(prefill, dim=0))[
                        snake[k][0] * 3 + snake[k][1]
                    ]
                    for k in range(18)
                ]
                for j in range(3):
                    ly = [c * 3 + j for c in range(6)]
                    _shard(
                        torch.cat([kk[L].reshape(1, 1, -1, ecfg.head_dim) for L in ly], 0),
                        ttnn.bfloat8_b,
                        ttnn.TILE_LAYOUT,
                        denoise,
                    )
                    _shard(
                        torch.cat([vv[L].reshape(1, 1, -1, ecfg.head_dim) for L in ly], 0),
                        ttnn.bfloat8_b,
                        ttnn.TILE_LAYOUT,
                        denoise,
                    )

            def _kv_socket():  # NEW on-device p2p + adjacent socket
                migrate_side(0, _recv_bufs())
                migrate_side(1, _recv_bufs())

            v_d2h = _bench(_v_d2h)
            prefix_h2d = _bench(_prefix_h2d)
            kv_host = _bench(_kv_host)
            kv_sock = _bench(_kv_socket)
            log(
                f"TIMING vision->prefix host-bounce: d2h={v_d2h:.2f}ms + h2d={prefix_h2d:.2f}ms = {v_d2h+prefix_h2d:.2f}ms"
            )
            log(
                f"TIMING KV migration: host-bounce={kv_host:.2f}ms  on-device p2p+socket={kv_sock:.2f}ms  speedup={kv_host/kv_sock:.1f}x"
            )

        # ===================== DENOISE (traced Euler) =====================
        elt = []
        for L in range(18):
            bw = _load_block_weights_to_submesh(ew, L, scratch)
            _inject_adarms_weights_to_submesh(bw, ew, L, scratch)
            d = {}
            for k, v in bw.items():
                d[k] = ttnn.to_torch(v)
                ttnn.deallocate(v)
            elt.append((d, {k: (None) for k in d}))
        # rebuild meta from a fresh load (dtype/layout)
        emeta = {}
        bw0 = _load_block_weights_to_submesh(ew, 0, scratch)
        _inject_adarms_weights_to_submesh(bw0, ew, 0, scratch)
        for k, v in bw0.items():
            emeta[k] = (v.dtype, v.layout)
            ttnn.deallocate(v)

        def eblock(local):
            layers = [c * 3 + local for c in range(6)]
            sh = {k: _shard(torch.stack([elt[L][0][k] for L in layers], 0), *emeta[k], denoise) for k in elt[0][0]}
            return AdaRMSGemmaBlockTTNN(ecfg, sh, local, denoise, None, None)

        eblocks = [eblock(L) for L in range(3)]
        # Upstream-compat denoise artifacts (all-real masks here):
        #  - suffix RoPE is OFFSET by prefix_real_count (not sequential from 0)
        #  - expert cross-attn mask blocks the padded suffix rows/cols beyond
        #    action_horizon (and any prefix padding; none here since all-real).
        prefix_real_count = prefix_len  # all img+lang tokens real
        cos_exp, sin_exp = _precompute_rope_table_torch(ecfg.head_dim, cfg.max_seq_len)
        spos = (torch.arange(ah_pad, dtype=torch.int64) + prefix_real_count).clamp(max=cfg.max_seq_len - 1)
        ecos = _repl(cos_exp[spos].unsqueeze(0).unsqueeze(0), denoise)
        esin = _repl(sin_exp[spos].unsqueeze(0).unsqueeze(0), denoise)
        kv_total = prefix_pad + ah_pad
        em = torch.zeros(ah_pad, kv_total, dtype=torch.bfloat16)
        if prefix_pad > prefix_len:
            em[:, prefix_len:prefix_pad] = _MASK_VAL
        if ah_pad > ah:
            em[:, prefix_pad + ah : kv_total] = _MASK_VAL  # padded suffix KV cols
            em[ah:ah_pad, :] = _MASK_VAL  # padded suffix query rows
        emask = _repl(em.unsqueeze(0).unsqueeze(0), denoise)
        scfg_s = SuffixConfig(action_dim=adim, action_horizon=ah, expert_width=ecfg.width, pi05=True)
        suffix = SuffixSlice(scfg_s, pi0proj, denoise)
        hw = _repl(ew["model.norm.dense.weight"].T.contiguous(), denoise)
        hb = (
            _repl(ew["model.norm.dense.bias"].reshape(1, -1), denoise)
            if ew.get("model.norm.dense.bias") is not None
            else None
        )
        dg = denoise.compute_with_storage_grid_size()
        hgrid = ttnn.CoreGrid(y=dg.y, x=dg.x)
        ts = [1.0 - i / N_STEPS for i in range(N_STEPS + 1)]
        dts = [ts[i + 1] - ts[i] for i in range(N_STEPS)]
        conds = [
            suffix.embed_adarms_cond(_repl(torch.tensor([ts[i]], dtype=torch.float32), denoise)) for i in range(N_STEPS)
        ]

        torch.manual_seed(SEED + 1)
        noise = torch.zeros(1, ah_pad, adim, dtype=torch.float32)
        noise[:, :ah, :] = torch.randn(1, ah, adim)
        x_t = _repl(noise, denoise, dt=ttnn.float32, mc=ttnn.L1_MEMORY_CONFIG)

        def euler():
            for i in range(N_STEPS):
                cond = conds[i]
                cur = suffix.embed_actions(ttnn.typecast(x_t, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG))
                for c in range(6):
                    for L in range(3):
                        cur, _ = eblocks[L].forward(cur, ecos, esin, cond, emask, None, (past_k[L], past_v[L]), False)
                    if c < 5:
                        cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                        cur = ttnn.point_to_point(
                            cur,
                            ttnn.MeshCoordinate(c, 0),
                            ttnn.MeshCoordinate(c + 1, 0),
                            topology=ttnn.Topology.Linear,
                            output_tensor=cur,
                        )
                normed = ada_rms_norm_no_gate_ttnn(cur, cond, hw, hb, ecfg.rms_norm_eps, hgrid)
                vel = suffix.project_output(normed)
                vel = ttnn.to_memory_config(ttnn.to_layout(vel, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                vel = ttnn.point_to_point(
                    vel,
                    ttnn.MeshCoordinate(5, 0),
                    ttnn.MeshCoordinate(0, 0),
                    topology=ttnn.Topology.Linear,
                    output_tensor=vel,
                )
                v32 = ttnn.mul(
                    ttnn.typecast(vel, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG),
                    dts[i],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn.add(x_t, v32, output_tensor=x_t)

        euler()  # warmup
        nh = ttnn.from_torch(
            noise, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(denoise)
        )
        ttnn.copy_host_to_device_tensor(nh, x_t)
        dtid = ttnn.begin_trace_capture(denoise, cq_id=0)
        euler()
        ttnn.end_trace_capture(denoise, dtid, cq_id=0)
        ttnn.copy_host_to_device_tensor(nh, x_t)
        ttnn.execute_trace(denoise, dtid, cq_id=0, blocking=True)
        actions = ttnn.to_torch(x_t, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))[0][:ah, :]
        log(
            f"DENOISE done: actions {tuple(actions.shape)} finite={torch.isfinite(actions).all().item()} "
            f"mean={actions.float().mean():.4f} std={actions.float().std():.4f}"
        )

        # ===================== numerical validation vs torch (opt-in) =====================
        # PI05_E2E_PCC=1 runs the torch Pi0_5Model.sample_actions reference with
        # matched inputs/seed and reports PCC. Same noise contract as the eager
        # PCC test: reseed SEED+1 right before sample_actions (its first randn IS
        # the denoise noise -> identical to x_t above). Inputs match because the
        # per-camera list flattens to the same RNG draw as randn(N_CAMS,...).
        if os.environ.get("PI05_E2E_PCC", "").lower() in ("1", "true", "yes", "on"):
            from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model as TorchPi0_5Model

            t_images = [images[i : i + 1] for i in range(N_CAMS)]
            t_img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
            t_lang_masks = torch.ones(1, LANG_LEN, dtype=torch.bool)
            torch.manual_seed(SEED)
            ref = TorchPi0_5Model(cfg, loader)
            with torch.no_grad():
                torch.manual_seed(SEED + 1)
                ref_actions = ref.sample_actions(
                    images=t_images,
                    img_masks=t_img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=t_lang_masks,
                    state=None,
                )[0].float()
            a, b = actions.flatten().float(), ref_actions.flatten().float()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            mae = (actions.float() - ref_actions).abs().mean().item()
            log(
                f"PCC vs torch: {pcc:.6f}  MAE={mae:.5f}  ref(mean={ref_actions.mean():.4f} std={ref_actions.std():.4f})"
            )
            log(f"PCC {'>=' if pcc >= 0.95 else '<'} 0.95 target")

        log("SUCCESS — fully-traced e2e pipeline ran (vision+prefill+denoise all traced)")
    finally:
        for sm in reversed(subs):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
