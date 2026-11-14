# prefill-roof phase 3 (3): per-op traced us of the indexer non-scan ops of one 512-row prefill chunk (layer 1, T=3072 keys), stock vs HY4_PF_OPT=all. Env VARIANTS 'off;all'. Prints IX json lines.
import json
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tests.pf_roof_ab import CKPT, GOLD, N, tr
from models.demos.blackhole.hy4_preview.tests.run_attn_device import load_sd
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload
from models.demos.blackhole.hy4_preview.tt.hy4_pf_opt import PfOpt, plinear
from models.demos.gpt_oss.config import mesh_1x8
from models.demos.gpt_oss.tt.ccl import CCLManager

cfg = Hy4Config.from_checkpoint(CKPT)
g = torch.load(f"{GOLD}/L1.pt")
conv = g["conv"]
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=800_000_000)
try:
    ccl = CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    sd = load_sd(1)
    for v in os.environ.get("VARIANTS", "off;all").split(";"):
        pf = PfOpt.parse("" if v == "off" else v)
        a = Hy4Attention(md, mesh_1x8(), ccl, cfg, sd, 1, kv_fmt="bf16", rope_interleave=conv["rope_interleave"], idx_rope_interleave=conv["idx_rope_interleave"], wdtype=ttnn.bfloat8_b, pf_opt=pf)
        for p0 in range(0, 3072, N):
            h = upload(md, g["h"][p0 : p0 + N].reshape(1, 1, N, -1).to(torch.bfloat16))
            a.prefill_chunk(h, p0, None)
        ix = a.indexer
        P0 = 2560
        tabs = a.rope_tabs(P0, N)
        icos, isin = tabs[2], tabs[3]
        qa = a._rms(a.mesh_config.allgather(plinear(pf, h, a.w_q_a, a.ckc_pf), a.ccl, axis=1, dim=3), a.q_a_norm)
        R = {"variant": v}
        R["wq_b_linear"] = tr(md, lambda: plinear(pf, qa, ix.w_q, ix.ckc_pf))
        q = plinear(pf, qa, ix.w_q, ix.ckc_pf)
        R["heads_split"] = tr(md, lambda: ttnn.experimental.nlp_create_qkv_heads(q, num_heads=32, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(q, num_heads=32, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        R["rope_q"] = tr(md, lambda: ix._rot(qh, icos, isin))
        R["w_proj_linear"] = tr(md, lambda: plinear(pf, h, ix.w_p, ix.ckc_pf))
        R["wk_linear"] = tr(md, lambda: plinear(pf, h, ix.w_k, ix.ckc_pf))
        k = plinear(pf, h, ix.w_k, ix.ckc_pf)
        R["k_layer_norm"] = tr(md, lambda: ttnn.layer_norm(k, weight=ix.k_norm_w, bias=ix.k_norm_b, epsilon=ix.ln_eps))
        kn = ttnn.layer_norm(k, weight=ix.k_norm_w, bias=ix.k_norm_b, epsilon=ix.ln_eps)
        R["rope_k"] = tr(md, lambda: ix._rot(kn, icos, isin))
        w = plinear(pf, h, ix.w_p, ix.ckc_pf)
        T = ix.length
        pc = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(T, 256), head_group_size=1)
        qr = ix._rot(qh, icos, isin)
        R["score_op"] = tr(md, lambda: ttnn.experimental.indexer_score_dsa(qr, ix.keys, w, chunk_start_idx=P0, program_config=pc, compute_kernel_config=ix.ckc_score, kv_len=T))
        sc = ttnn.experimental.indexer_score_dsa(qr, ix.keys, w, chunk_start_idx=P0, program_config=pc, compute_kernel_config=ix.ckc_score, kv_len=T)
        if pf.mask1:
            ix._causal_mask1(ttnn.clone(sc), P0, N, T)
            R["causal_fix"] = tr(md, lambda: ix._causal_mask1(ttnn.clone(sc), P0, N, T))
        else:
            R["causal_fix"] = tr(md, lambda: ix._causal_fix(ttnn.clone(sc), P0, N, T))
        R["clone_overhead"] = tr(md, lambda: ttnn.clone(sc))
        R["topk"] = tr(md, lambda: ttnn.experimental.topk_large_indices(sc, k=2048, valid_length=T))
        print("IX " + json.dumps(R), flush=True)
        del a
finally:
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
