# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""layer-other: MEASURED per-layer decode work of Hy4 EXCLUDING the routed experts and the sparse-attention kernel, on ONE 1x8 mesh row (slot), traced,
cumulative-prefix method: the layer's decode ops are a list of steps; a trace of steps[:k] x REP back to back is timed, the per-step cost is the difference of consecutive prefixes.

Env: LAYER (full-indexer layer id, default 1; shared layer id, default 2), KIND=full|shared, S (query rows = users per mesh row: 1 = B4, 2 = B8), WDT=bf16|bfp8 (attention
projection weights), OUT (json path), REP, NREP.
Stubs (reported in RESULT.md): sparse kernel + q all_gather + a2a + merge of decode-merge (replaced by o_lat := q_abs, plus a measured [1,S,32,512] -> [1,8,S,512] layout glue),
indexer score + top-k scan (not run; idx-1m numbers), routed experts + their reduce (y := the shared-expert partial), KV / key cache write = update op on a preallocated cache.
"""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tt.hy4_attention import Hy4Attention
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

CKPT = os.environ.get("HY4_CKPT", "/mnt/tt-data/Hy4-Preview")
KIND = os.environ.get("KIND", "shared")
LAYER = int(os.environ.get("LAYER", "2" if KIND == "shared" else "1"))
S = int(os.environ.get("S", "1"))
WDT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("WDT", "bf16")]
TUNE = (
    os.environ.get("TUNE", "0") == "1"
)  # DSV4-style 1D-mcast program configs for the M=1-tile projections (attn-bfp4 sweep)
RING = os.environ.get("RING", "0") == "1"  # FABRIC_1D_RING + Ring topology + 2 links
LEAN = os.environ.get("LEAN", "0") == "1"  # DSV4 lean CCLs (one worker per link)
L1 = os.environ.get("L1", "0") == "1"  # activations / matmul outputs in L1
FUSEPROJ = (
    os.environ.get("FUSEPROJ", "0") == "1"
)  # q_a | kv_a | gate [| idx_wk | idx_wp] as ONE linear on a concatenated weight + slices
ROPE1 = (
    os.environ.get("ROPE1", "0") == "1"
)  # every rope as ONE rotary_embedding_llama op (needs the rope columns of the weights permuted to half-split pairing offline: timing only here)
RMSSH = os.environ.get("RMSSH", "0") == "1"  # q_a rms_norm width-sharded over 16 cores (probe: 34 -> 11 us)
MC = ttnn.L1_MEMORY_CONFIG if L1 else None
REP = int(os.environ.get("REP", "5"))
NREP = int(os.environ.get("NREP", "9"))
OUT = os.environ.get("OUT", "/tmp/other.json")
HID = 6144
POS = int(os.environ.get("POS", "5000"))


def cfg_1d(K, N, max_cores, act=None):
    """DSV4 attention.py _cfg_1d: 1D multicast program config for one tile-row activation, N split over <= max_cores cores."""
    Kt, Nt = K // 32, N // 32
    pcn = -(-Nt // max_cores)
    n = -(-Nt // pcn)
    gx, gy = min(n, 8), -(-n // 8)
    sw = max(d for d in range(1, min(pcn, 4) + 1) if pcn % d == 0)
    ibw = max(d for d in range(1, 9) if Kt % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sw,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


CAPS = {
    "q_a": (6144, 256, 64),
    "q_b": (2048, 2048, 32),
    "kv_a": (6144, 576, 32),
    "gate": (6144, 2048, 32),
    "o_proj": (2048, 6144, 32),
    "idx_wq_b": (2048, 4096, 32),
    "idx_wk": (6144, 128, 64),
    "idx_wp": (6144, 32, 32),
}


def lin(x, w, name, ckc, act=None):
    """ttnn.linear; with TUNE=1 the tuned 1D program config of ``name`` (shapes per device)."""
    if not TUNE or S > 4:
        return ttnn.linear(x, w, compute_kernel_config=ckc, activation=act, memory_config=MC)
    K, N, cap = CAPS[name]
    pc = cfg_1d(K, N, cap, ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID) if act == "sigmoid" else None)
    return ttnn.linear(x, w, compute_kernel_config=ckc, program_config=pc, memory_config=MC)


def P(*a):
    print("OTHER", *a, flush=True)


def load_sd(layer):
    from safetensors import safe_open

    wm = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    pre = f"model.layers.{layer}."
    sd = {}
    for k, f in wm.items():
        if k.startswith(pre + "self_attn."):
            with safe_open(f"{CKPT}/{f}", "pt") as fh:
                sd[k[len(pre) :]] = fh.get_tensor(k)
    return sd


def rep_up(md, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def free(t):
    try:
        if isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t)
        elif isinstance(t, (tuple, list)):
            for x in t:
                free(x)
    except Exception:
        pass


class Writer:
    """cache row write at index POS: tries paged_update_cache, update_cache, fill_cache; the first that works is used (logged)."""

    def __init__(self, md, width):
        self.md, self.w = md, width
        self.cache = rep_up(md, torch.zeros(S, 1, 8192, width))
        self.idx = ttnn.from_torch(
            torch.full((S,), POS, dtype=torch.int32),
            device=md,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(md),
        )
        self.mc = ttnn.create_sharded_memory_config(
            shape=(32, width),
            core_grid=ttnn.CoreGrid(y=1, x=S),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.variant = None

    def _a(self, row):  # row [1,1,S,W] tile
        r = ttnn.reshape(row, [1, S, 1, self.w])
        r = ttnn.to_memory_config(r, self.mc)
        ttnn.experimental.paged_update_cache(self.cache, r, update_idxs_tensor=self.idx, page_table=None)
        ttnn.deallocate(r)

    def _b(self, row):
        ttnn.fill_cache(self.cache, row, 0)

    def write(self, row):
        if self.variant is None:
            for name, fn in (("paged_update_cache", self._a), ("fill_cache_proxy", self._b)):
                try:
                    fn(row)
                    ttnn.synchronize_device(self.md)
                    self.variant = name
                    P("cache write variant", name, "width", self.w)
                    break
                except Exception as e:
                    P("cache write variant", name, "FAILED", str(e)[:200])
            if self.variant is None:
                self.variant = "none"
            return
        {"paged_update_cache": self._a, "fill_cache_proxy": self._b, "none": lambda r: None}[self.variant](row)


def build(md):
    from models.demos.blackhole.hy4_preview.tt import hy4_dense as D
    from models.demos.blackhole.hy4_preview.tt.hy4_ihc import Hy4IHC
    from models.demos.blackhole.hy4_preview.tt.hy4_router import Hy4Gate
    from models.demos.gpt_oss.config import mesh_1x8
    from models.demos.gpt_oss.tt.ccl import CCLManager

    cfg = Hy4Config.from_checkpoint(CKPT)
    mc = mesh_1x8()
    ccl = CCLManager(md, num_links=2 if RING else 1, topology=ttnn.Topology.Ring if RING else ttnn.Topology.Linear)
    if LEAN:
        from models.demos.blackhole.deepseek_v41_flash.tt import ccl_lean

        ccl_lean.make_lean(mc)
        ccl_lean.make_lean_sems(ccl)
    sd = load_sd(LAYER)
    attn = Hy4Attention(
        md, mc, ccl, cfg, sd, LAYER, kv_fmt="bf16", rope_interleave=True, idx_rope_interleave=False, wdtype=WDT
    )
    full = attn.has_indexer
    assert full == (KIND == "full"), (KIND, LAYER, full)
    g = torch.Generator().manual_seed(0)
    rnd = lambda *s, sc=0.02: torch.randn(*s, generator=g) * sc
    ihc_a = Hy4IHC(md, rnd(8, 4 * HID, sc=0.01), rnd(8), torch.tensor([0.1, 0.1]), dim=HID, hc_eps=1e-6, norm_eps=1e-5)
    ihc_m = Hy4IHC(md, rnd(8, 4 * HID, sc=0.01), rnd(8), torch.tensor([0.1, 0.1]), dim=HID, hc_eps=1e-6, norm_eps=1e-5)
    w_in = rep_up(md, torch.ones(1, 1, 1, HID), ttnn.float32)
    w_post = rep_up(md, torch.ones(1, 1, 1, HID), ttnn.float32)
    sh = D.make_shared(
        md,
        rnd(1, 1, HID, 2048).to(torch.bfloat16),
        rnd(1, 1, HID, 2048).to(torch.bfloat16),
        rnd(1, 1, 2048, HID).to(torch.bfloat16),
        decode=True,
    )
    gate = Hy4Gate(md, rnd(256, HID, sc=0.05), rnd(256, sc=0.03))
    pos = [POS + i for i in range(S)]
    cos, sin = attn.rope.tables(pos)
    icos, isin = attn.idx_rope.tables(pos) if full else (None, None)
    x = rep_up(md, rnd(S, 1, 4, HID, sc=1.0), ttnn.float32, ttnn.TILE_LAYOUT)
    # stand-in for the decode-merge output (per column [1,S,32,512], rows 0..7 = the column's heads)
    o_dm = rep_up(md, rnd(1, S, 32, 512, sc=0.1))
    extra = {}
    if FUSEPROJ:
        hw = attn.host_w
        parts = []
        for c in range(8):
            p_ = [
                hw["q_a"].tensor[..., c * 256 : (c + 1) * 256],
                hw["kv_a"].tensor,
                hw["gate"].tensor[..., c * 2048 : (c + 1) * 2048],
            ]
            if full:
                p_ += [hw["idx_wk"].tensor, hw["idx_wp"].tensor]
            parts.append(torch.cat(p_, dim=-1))
        Wcat = torch.cat(parts, dim=-1)
        Nloc = Wcat.shape[-1] // 8
        extra["wf"] = upload(md, Wcat, WDT, ttnn.TILE_LAYOUT, shard_dim=3)
        extra["Nloc"] = Nloc
        extra["offs"] = [0, 256, 256 + 576, 256 + 576 + 2048] + ([256 + 576 + 2048 + 128, Nloc] if full else [Nloc])
        extra["pc"] = cfg_1d(6144, Nloc, 32)
    if ROPE1:
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        extra["tm"] = rep_up(md, get_rot_transformation_mat())
        if full:
            extra["icos128"] = ttnn.concat([icos, rep_up(md, torch.ones(1, 1, S, 64))], dim=3)
            extra["isin128"] = ttnn.concat([isin, rep_up(md, torch.zeros(1, 1, S, 64))], dim=3)
    if RMSSH:
        extra["rms_mc"] = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreGrid(y=2, x=8),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # gamma of the q_a norm in the [64,32] RM layout the sharded op wants (host weight = checkpoint q_a_norm)
        extra["rms_g"] = upload(
            md, attn.host_w["q_a_norm"].tensor.reshape(1, 1, 64, 32).float(), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
        )
        extra["rms_pc"] = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 2), subblock_w=4, block_h=1, block_w=4, inplace=False
        )
    kvw = Writer(md, 576)
    kw = Writer(md, 128) if full else None
    return dict(
        cfg=cfg,
        attn=attn,
        full=full,
        ihc_a=ihc_a,
        ihc_m=ihc_m,
        w_in=w_in,
        w_post=w_post,
        sh=sh,
        gate=gate,
        cos=cos,
        sin=sin,
        icos=icos,
        isin=isin,
        x=x,
        o_dm=o_dm,
        kvw=kvw,
        kw=kw,
        **extra,
    )


def steps(B):
    """[(name, group, fn(s))]: s = state dict (tensors produced so far); persistent inputs live in B."""
    a, cfg = B["attn"], B["cfg"]
    ckc = a.ckc
    hc = a.hc
    ix = a.indexer
    St = []
    llama = lambda x, cos, sin: ttnn.experimental.rotary_embedding_llama(
        x, cos, sin, B["tm"], is_decode_mode=False, compute_kernel_config=ckc
    )
    rope = (lambda x, cos, sin: llama(x, cos, sin)) if ROPE1 else (lambda x, cos, sin: a.rope.apply(x, cos, sin, ckc))
    ccl_kw = {"memory_config": MC} if L1 else {}

    def add(name, group):
        def deco(f):
            St.append((name, group, f))
            return f

        return deco

    @add("ihc_attn_mixes", "iHC attn")
    def _(s):
        s["pre"], s["post"], s["comb"] = B["ihc_a"].mixes(B["x"])

    @add("ihc_attn_collapse_norm (input_layernorm)", "iHC attn")
    def _(s):
        s["h"] = B["ihc_a"].collapse_norm(B["x"], s["pre"], B["w_in"], 1e-5)

    if FUSEPROJ:

        @add("FUSED h-proj linear (q_a|kv_a|gate" + ("|idx_wk|idx_wp)" if B["full"] else ")"), "q/kv proj")
        def _(s):
            s["fp"] = ttnn.linear(s["h"], B["wf"], compute_kernel_config=ckc, program_config=B["pc"], memory_config=MC)

        @add("FUSED h-proj slices", "q/kv proj")
        def _(s):
            o = B["offs"]
            sl = [ttnn.slice(s["fp"], [0, 0, 0, o[i]], [1, 1, S, o[i + 1]]) for i in range(len(o) - 1)]
            s["qa_c"], s["kv"], s["g_raw"] = sl[0], sl[1], sl[2]
            if B["full"]:
                s["ik"], s["iw"] = sl[3], sl[4]

        @add("gate sigmoid", "q/kv proj")
        def _(s):
            s["g"] = ttnn.sigmoid(s["g_raw"])

    else:

        @add("q_a linear", "q/kv proj")
        def _(s):
            s["qa_c"] = lin(s["h"], a.w_q_a, "q_a", ckc)

    @add("q_a all_gather", "q/kv proj")
    def _(s):
        s["qa_g"] = a.mesh_config.allgather(s["qa_c"], a.ccl, axis=1, dim=3, **ccl_kw)

    @add("q_a rms_norm", "q/kv proj")
    def _(s):
        if RMSSH:
            xs_ = ttnn.to_memory_config(s["qa_g"], B["rms_mc"])
            o = ttnn.rms_norm(
                xs_,
                weight=B["rms_g"],
                epsilon=cfg.rms_norm_eps,
                program_config=B["rms_pc"],
                memory_config=B["rms_mc"],
                compute_kernel_config=ckc,
            )
            s["qa"] = ttnn.to_memory_config(o, ttnn.L1_MEMORY_CONFIG)
        else:
            s["qa"] = a._rms(s["qa_g"], a.q_a_norm)

    if not FUSEPROJ:

        @add("kv_a linear", "q/kv proj")
        def _(s):
            s["kv"] = lin(s["h"], a.w_kv_a, "kv_a", ckc)

    @add("kv latent slice + rms_norm", "kv norm/rope/cache")
    def _(s):
        s["lat"] = a._rms(ttnn.slice(s["kv"], [0, 0, 0, 0], [1, 1, S, cfg.kv_lora_rank]), a.kv_a_norm)

    @add("k_pe slice + rope (matmul,mul,mul,add)", "kv norm/rope/cache")
    def _(s):
        s["k_pe"] = rope(ttnn.slice(s["kv"], [0, 0, 0, cfg.kv_lora_rank], [1, 1, S, cfg.kv_a_out]), B["cos"], B["sin"])

    @add("kv concat + cache write", "kv norm/rope/cache")
    def _(s):
        s["kvrow"] = ttnn.concat([s["lat"], s["k_pe"]], dim=3)
        B["kvw"].write(s["kvrow"])

    if B["full"]:
        if not FUSEPROJ:

            @add("idx wk linear", "indexer proj")
            def _(s):
                s["ik"] = lin(s["h"], ix.w_k, "idx_wk", ix.ckc)

        @add("idx k layer_norm", "indexer proj")
        def _(s):
            s["ik"] = ttnn.layer_norm(s["ik"], weight=ix.k_norm_w, bias=ix.k_norm_b, epsilon=ix.ln_eps)

        @add("idx k rope (slice,rope,slice,concat)", "indexer proj")
        def _(s):
            s["ik"] = (
                llama(s["ik"], B["icos128"], B["isin128"]) if ROPE1 else ix._rot_first(s["ik"], B["icos"], B["isin"])
            )

        @add("idx key cache write", "indexer proj")
        def _(s):
            B["kw"].write(s["ik"])

        @add("idx wq_b linear", "indexer proj")
        def _(s):
            s["iq"] = lin(s["qa"], ix.w_q, "idx_wq_b", ix.ckc)

        @add("idx q heads split", "indexer proj")
        def _(s):
            s["iq"], _a, _b = ttnn.experimental.nlp_create_qkv_heads(
                s["iq"],
                num_heads=ix.nh,
                num_kv_heads=0,
                transpose_k_heads=False,
                memory_config=MC or ttnn.DRAM_MEMORY_CONFIG,
            )

        @add("idx q rope", "indexer proj")
        def _(s):
            s["iq"] = (
                llama(s["iq"], B["icos128"], B["isin128"]) if ROPE1 else ix._rot_first(s["iq"], B["icos"], B["isin"])
            )

        if not FUSEPROJ:

            @add("idx weights_proj linear", "indexer proj")
            def _(s):
                s["iw"] = lin(s["h"], ix.w_p, "idx_wp", ix.ckc)

    @add("q_b linear", "q path")
    def _(s):
        s["q"] = lin(s["qa"], a.w_q_b, "q_b", ckc)

    @add("q heads split", "q path")
    def _(s):
        s["q_nope"], s["q_pe"] = ttnn.experimental.nlp_create_q_heads_split(
            s["q"], num_heads=hc, split_head_dim=cfg.qk_nope_head_dim, memory_config=MC or ttnn.DRAM_MEMORY_CONFIG
        )

    @add("absorb: q_nope x w_uk matmul", "q path")
    def _(s):
        s["q_abs"] = ttnn.matmul(s["q_nope"], a.w_uk, compute_kernel_config=ckc, memory_config=MC)

    @add("q_pe rope", "q path")
    def _(s):
        s["q_pe"] = rope(s["q_pe"], B["cos"], B["sin"])

    @add("qfull concat [q_abs|q_pe]", "q path")
    def _(s):
        s["qfull"] = ttnn.concat([s["q_abs"], s["q_pe"]], dim=3)

    @add("glue: DM out [1,S,32,512] -> [1,8,S,512] (slice+permute)", "DM->o glue")
    def _(s):
        o8 = ttnn.slice(B["o_dm"], [0, 0, 0, 0], [1, S, hc, cfg.kv_lora_rank])
        s["o_lat"] = ttnn.permute(o8, (0, 2, 1, 3))

    @add("absorb: o_lat x w_uv matmul", "o path")
    def _(s):
        s["o"] = ttnn.matmul(s["o_lat"], a.w_uv_t, compute_kernel_config=ckc, memory_config=MC)

    @add("concat heads", "o path")
    def _(s):
        s["o"] = ttnn.experimental.nlp_concat_heads(s["o"], memory_config=MC or ttnn.DRAM_MEMORY_CONFIG)

    if not FUSEPROJ:

        @add("gate linear + sigmoid", "o path")
        def _(s):
            s["g"] = lin(s["h"], a.w_gate, "gate", ckc, "sigmoid")

    @add("o * gate", "o path")
    def _(s):
        s["og"] = ttnn.multiply(s["o"], s["g"])

    @add("o_proj linear", "o path")
    def _(s):
        s["part"] = lin(s["og"], a.w_o, "o_proj", ckc)

    @add("o_proj all_reduce (8 cols)", "o path")
    def _(s):
        s["y"] = a.mesh_config.allreduce(s["part"], a.ccl, axis=1, **ccl_kw)

    @add("ihc_attn_expand", "iHC attn")
    def _(s):
        s["x2"] = B["ihc_a"].expand(s["y"], B["x"], s["post"], s["comb"])

    @add("ihc_mlp_mixes", "iHC mlp")
    def _(s):
        s["pre_m"], s["post_m"], s["comb_m"] = B["ihc_m"].mixes(s["x2"])

    @add("ihc_mlp_collapse_norm_rm (post_attention_layernorm)", "iHC mlp")
    def _(s):
        s["h2"], s["h2_tok"] = B["ihc_m"].collapse_norm_rm(s["x2"], s["pre_m"], B["w_post"], 1e-5)

    @add("shared expert partial (TP8 bfp8)", "shared expert")
    def _(s):
        s["sh"] = B["sh"].forward(s["h2"])

    @add("router (matmul+sigmoid+bias+top8)", "router")
    def _(s):
        s["rw"], s["ri"] = B["gate"].forward(s["h2"])

    @add("ihc_mlp_expand (y := shared partial stand-in)", "iHC mlp")
    def _(s):
        s["xo"] = B["ihc_m"].expand(s["sh"], s["x2"], s["post_m"], s["comb_m"])

    return St


def traced(md, body, n=NREP):
    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(md, tid)
    return ts


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING if RING else ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384, trace_region_size=200_000_000)
    res = {
        "rmssh": RMSSH,
        "l1": L1,
        "fuseproj": FUSEPROJ,
        "rope1": ROPE1,
        "tune": TUNE,
        "ring": RING,
        "lean": LEAN,
        "kind": KIND,
        "layer": LAYER,
        "S": S,
        "wdt": os.environ.get("WDT", "bf16"),
        "rep": REP,
    }
    try:
        t0 = time.time()
        B = build(md)
        P(f"built {KIND} layer {LAYER} S={S} in {time.time()-t0:.0f}s")
        St = steps(B)
        s = {}
        for name, grp, f in St:  # eager warm-up of everything (weights/norm caches outside a trace)
            f(s)
        ttnn.synchronize_device(md)
        for k_, v in list(s.items()):
            free(v)
        P("warm-up ok")

        def body_k(k):
            def body():
                for _ in range(REP):
                    s = {}
                    for name, grp, f in St[:k]:
                        f(s)
                    for v in s.values():
                        free(v)

            return body

        cum = [0.0]
        raw = []
        for k in range(1, len(St) + 1):
            ts = traced(md, body_k(k))
            m = statistics.median(ts) / REP
            raw.append((m, min(ts) / REP, max(ts) / REP))
            P(
                f"prefix {k:2d} {St[k-1][0]:60s} cum {m:8.1f} us (min {min(ts)/REP:.1f} max {max(ts)/REP:.1f})  step {m-cum[-1]:7.1f}"
            )
            cum.append(m)

        def body_chain():
            orig = B["x"]
            for _ in range(REP):
                s = {}
                for name, grp, f in St:
                    f(s)
                nx = s.pop("xo")
                for v in s.values():
                    free(v)
                if B["x"] is not orig:
                    free(B["x"])
                B["x"] = nx
            if B["x"] is not orig:
                free(B["x"])
            B["x"] = orig

        tsc = traced(md, body_chain)
        res["chain_us"] = statistics.median(tsc) / REP
        P(
            f"CHAIN (x fed forward layer to layer, {REP} layers per trace): {res['chain_us']:.1f} us/layer vs prefix total {cum[-1]:.1f}"
        )
        res["steps"] = [
            {"name": St[i][0], "group": St[i][1], "cum_us": cum[i + 1], "step_us": cum[i + 1] - cum[i]}
            for i in range(len(St))
        ]
        res["total_us"] = cum[-1]
        res["cache_write_variant"] = [B["kvw"].variant, B["kw"].variant if B["kw"] else None]
        gs = {}
        for st in res["steps"]:
            gs[st["group"]] = gs.get(st["group"], 0.0) + st["step_us"]
        res["groups"] = gs
        P("GROUPS", json.dumps({k: round(v, 1) for k, v in gs.items()}), "TOTAL", round(cum[-1], 1))
    finally:
        json.dump(res, open(OUT, "w"), indent=1)
        ttnn.close_mesh_device(md)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
