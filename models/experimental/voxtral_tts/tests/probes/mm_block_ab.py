"""Do sweep_mm.py's isolated matmul wins survive in the WHOLE BLOCK? 6.43's rule, applied.

sweep_mm.py found the ttnn heuristic collapses on deep reductions -- 144-147 GB/s at Kt=128/288
against 352 at Kt=96 -- and that a tuned in0_block_w recovers the full ~350 GB/s:

    wqkv  Kt= 96   71.4 -> 58.0 us   (1.23x)     1D 13x10 ibw=4
    wo    Kt=128   92.7 -> 39.8 us   (2.33x)     1D 13x10 ibw=8
    w1/w3 Kt= 96   85.4 -> 83.0 us   (1.03x)     1D 12x6  ibw=2
    w2    Kt=288  205.2 -> 84.6 us   (2.43x)     1D 13x10 ibw=8

Isolated that is ~192 us per layer-pass, ~9 ms/frame over 47 passes. But 6.43 is the whole reason
this file exists: the Wo config won its isolated sweep on the N150 and LOST inside the block,
because a lone matmul in a tight loop has the entire DRAM system and every core to itself, while
the same matmul in a 17-op layer does not. So nothing here is decided until it is measured on the
real block. Arms are cumulative and interleaved, shipped is entered twice as a noise floor.

Both blocks share dims and both are ONE tile of rows, so one set of configs serves all 8 sites.
w1 carries activation="silu", which must move into the program config's fused_activation --
passing both is not allowed.
"""
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as fref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM, FM_HEAD_DIM, FM_N_HEADS, FM_N_KV_HEADS, HEAD_DIM, N_ACOUSTIC_CODEBOOK, N_HEADS,
    N_KV_HEADS)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
CC, _L1 = gpt.COMPUTE_CONFIG, gpt._L1
ROUNDS, REPS, FRAMES = 11, 30, 30
SILU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)


def prg(gx, gy, ibw, pcn, act=None):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=ibw, out_subblock_h=1,
        out_subblock_w=next((s for s in (4, 2, 1) if pcn % s == 0), 1),
        per_core_M=1, per_core_N=pcn, fuse_batch=True, fused_activation=act, mcast_in0=True)


# Two candidate sets. MIXED is each shape's isolated winner and pins TWO grids. UNIFORM puts
# every matmul on one 12x6 grid -- within 1.7% of the isolated winner for all four shapes, and it
# tests whether ALL's extra 1.7 ms comes from grid CONSISTENCY across the layer rather than from
# any individual config (each of which measured neutral-or-worse alone).
SETS = {
    "mixed": dict(qkv=prg(13, 10, 4, 2), wo=prg(13, 10, 8, 1), w13=prg(12, 6, 2, 4),
                  w1=prg(12, 6, 2, 4, SILU), w2=prg(13, 10, 8, 1)),
    "uni12x6": dict(qkv=prg(12, 6, 2, 3), wo=prg(12, 6, 4, 2), w13=prg(12, 6, 2, 4),
                    w1=prg(12, 6, 2, 4, SILU), w2=prg(12, 6, 4, 2)),
    "uni13x10": dict(qkv=prg(13, 10, 4, 2), wo=prg(13, 10, 8, 1), w13=prg(13, 10, 4, 3),
                     w1=prg(13, 10, 4, 3, SILU), w2=prg(13, 10, 8, 1)),
}
CUR = SETS["mixed"]
P_QKV = P_WO = P_W13 = P_W1 = P_W2 = None

_b1_mlp, _b1_step = gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step
_b2_block = flow.TtVoxtralFlow._block


def use(name):
    global P_QKV, P_WO, P_W13, P_W1, P_W2
    c = SETS[name]
    P_QKV, P_WO, P_W13, P_W1, P_W2 = c["qkv"], c["wo"], c["w13"], c["w1"], c["w2"]


def _w1(h, w, mc, mode):
    """mode 0 = ships (activation kwarg, no config); 1 = config WITH fused silu;
    2 = config WITHOUT fusion, silu as its own op -- the control that separates the two."""
    if mode == 1:
        return ttnn.linear(h, w["w1"], program_config=P_W1, compute_kernel_config=CC,
                           memory_config=mc)
    if mode == 2:
        return ttnn.silu(ttnn.linear(h, w["w1"], program_config=P_W13,
                                     compute_kernel_config=CC, memory_config=mc))
    return ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=CC,
                       memory_config=mc)


def make_b1(qkv_p, wo_p, w1_p, w3_p, w2_p):
    def _mlp(self, x, h, w, mc):
        g = _w1(h, w, mc, w1_p)
        k3 = {"program_config": P_W13} if w3_p else {}
        k2 = {"program_config": P_W2} if w2_p else {}
        u = ttnn.multiply_(g, ttnn.linear(h, w["w3"], compute_kernel_config=CC,
                                          memory_config=mc, **k3))
        return ttnn.add_(x, ttnn.linear(u, w["w2"], compute_kernel_config=CC,
                                        memory_config=mc, **k2))

    def _layer_step(self, x, w, cos, sin, cache, pos_t):
        kq = {"program_config": P_QKV} if qkv_p else {}
        qkv = ttnn.linear(self._norm(x, w["an"]), w["wqkv"], compute_kernel_config=CC, **kq)
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, gpt._QKV_WIDTH]), gpt._QKV_SHARD)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        qh = ttnn.experimental.rotary_embedding_hf(qh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=CC)
        kh = ttnn.experimental.rotary_embedding_hf(kh, cos, sin, is_decode_mode=True,
                                                   compute_kernel_config=CC)
        ttnn.experimental.paged_update_cache(cache[0], kh, update_idxs_tensor=pos_t)
        ttnn.experimental.paged_update_cache(cache[1], vh, update_idxs_tensor=pos_t)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh, cache[0], cache[1], cur_pos_tensor=pos_t, scale=gpt.SCALE,
            compute_kernel_config=CC, program_config=gpt._SDPA_PRG)
        a = ttnn.reshape(o, [1, 1, gpt.Q_WIDTH])
        kw = {"program_config": P_WO} if wo_p else {}
        x = ttnn.add_(x, ttnn.linear(a, w["wo"], compute_kernel_config=CC,
                                     memory_config=_L1, **kw))
        return self._mlp(x, self._norm(x, w["fn"]), w, _L1)
    return _mlp, _layer_step


def make_b2(qkv_p, wo_p, w1_p, w3_p, w2_p):
    def _block(self, x, w, B):
        h = self._norm(x, w["an"])
        kq = {"program_config": P_QKV} if qkv_p else {}
        qkv = ttnn.linear(h, w["wqkv"], compute_kernel_config=CC, **kq)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [B, 1, 3, flow._QKV_WIDTH]), num_heads=FM_N_HEADS,
            num_kv_heads=FM_N_KV_HEADS, transpose_k_heads=False, memory_config=flow._L1)
        a = ttnn.transformer.scaled_dot_product_attention(
            qh, kh, vh, is_causal=False, scale=1.0, compute_kernel_config=CC)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        kw = {"program_config": P_WO} if wo_p else {}
        x = ttnn.add_(x, ttnn.linear(a, w["wo"], compute_kernel_config=CC,
                                     memory_config=flow._L1, **kw))
        h = self._norm(x, w["fn"])
        g = _w1(h, w, flow._L1, w1_p)
        k3 = {"program_config": P_W13} if w3_p else {}
        k2 = {"program_config": P_W2} if w2_p else {}
        u = ttnn.multiply_(g, ttnn.linear(h, w["w3"], compute_kernel_config=CC,
                                          memory_config=flow._L1, **k3))
        return ttnn.add_(x, ttnn.linear(u, w["w2"], compute_kernel_config=CC,
                                        memory_config=flow._L1, **k2))
    return _block


def main():
    dev = open_device()
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
        wf = fref.load_flow_state()
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
        embeds = pref.build_inputs_embeds(
            torch.tensor(case["ids"], dtype=torch.long), pref.load_voice(case["voice"]), pipe.wb)
        h = pipe.backbone.prefill_last(embeds)[:, 0]
        bb, gen = pipe.backbone, pipe.flow
        pos = bb.pos
        frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
        sem = gen.semantic_code(h)
        torch.manual_seed(0)
        x0 = torch.randn(1, N_ACOUSTIC_CODEBOOK)
        ref = fref.decode_frame(sem, h, wf, x_0=x0)

        cosb, sinb = gpt.rope_tables(1, offset=pos)
        up = lambda t: ttnn.from_torch(t.contiguous(), dtype=bb.dtype, layout=ttnn.TILE_LAYOUT,
                                       device=dev)
        cos = ttnn.to_memory_config(up(cosb.reshape(1, 1, 1, HEAD_DIM)), gpt._ROPE_SHARD)
        sin = ttnn.to_memory_config(up(sinb.reshape(1, 1, 1, HEAD_DIM)), gpt._ROPE_SHARD)
        pos_t = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=dev)
        xin = up(bref.embed_frame(pipe.wb, frames[0]).reshape(1, 1, DIM))
        layers = list(zip(bb.layers, bb.caches))

        def step26():
            x = ttnn.clone(xin)
            for lw, cache in layers:
                x = bb._layer_step(x, lw, cos, sin, cache, pos_t)
            return bb._norm(x, bb.norm)

        def t_b1():
            step26(); ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(REPS):
                step26()
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) / REPS * 1e3

        def t_b2():
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(FRAMES):
                c = gen.decode_frame(sem, h, x_0=x0)
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) / FRAMES * 1e3, c

        # (qkv, wo, w1, w3, w2). w1: 0 ships, 1 config+fused silu, 2 config + separate silu.
        ARMS = [("shipped", ("mixed", (0, 0, 0, 0, 0))),
                ("shipped#ctl", ("mixed", (0, 0, 0, 0, 0))),
                ("w1 only  [mixed]", ("mixed", (0, 0, 1, 0, 0))),
                ("w1 only  [uni12x6]", ("uni12x6", (0, 0, 1, 0, 0))),
                ("ALL      [mixed]", ("mixed", (1, 1, 1, 1, 1))),
                ("ALL      [uni12x6]", ("uni12x6", (1, 1, 1, 1, 1))),
                ("ALL      [uni13x10]", ("uni13x10", (1, 1, 1, 1, 1)))]

        def install(sf):
            name, f = sf
            use(name)
            if f == (0, 0, 0, 0, 0):
                gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step = _b1_mlp, _b1_step
                flow.TtVoxtralFlow._block = _b2_block
            else:
                m, s = make_b1(*f)
                gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step = m, s
                flow.TtVoxtralFlow._block = make_b2(*f)

        ok, outs = [], {}
        for lbl, f in ARMS:
            try:
                install(f)
                step26(); _, c = t_b2()
                ok.append((lbl, f)); outs[lbl] = c
                print(f"  {lbl:>20} builds, codes vs fp32 ref {int((c[0]!=ref[0]).sum())}/36",
                      flush=True)
            except Exception as e:
                print(f"  {lbl:>12} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:70]}")
        r1 = {l: [] for l, _ in ok}
        r2 = {l: [] for l, _ in ok}
        for r in range(ROUNDS):
            for lbl, f in (ok if r % 2 == 0 else ok[::-1]):
                install(f)
                r1[lbl].append(t_b1())
                r2[lbl].append(t_b2()[0])
        m1 = {l: sum(v) / len(v) for l, v in r1.items()}
        m2 = {l: sum(v) / len(v) for l, v in r2.items()}
        base = outs["shipped"]
        print(f"\n{'arm':>20} {'B1 ms':>7} {'B2 ms':>7} {'total':>7} {'vs ship':>9} "
              f"{'spread':>7} {'!=ship':>7} {'!=fp32':>7}")
        tot = {l: m1[l] + m2[l] for l, _ in ok}
        for lbl, _ in sorted(ok, key=lambda a: tot[a[0]]):
            print(f"{lbl:>20} {m1[lbl]:>7.2f} {m2[lbl]:>7.2f} {tot[lbl]:>7.2f} "
                  f"{tot['shipped']-tot[lbl]:>+9.2f} "
                  f"{max(r1[lbl])-min(r1[lbl]):>7.3f} "
                  f"{int((outs[lbl][0]!=base[0]).sum()):>4}/36 "
                  f"{int((outs[lbl][0]!=ref[0]).sum()):>4}/36")
        print(f"\nnoise floor: {abs(tot['shipped']-tot['shipped#ctl']):.3f} ms")
        print("silu_fusion.py: activation= is NOT fused (+14.9us); fused_activation is (+2.7).")
        print("isolated sweep predicted ~9 ms/frame; 6.43 says isolated does not decide.")
    finally:
        gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step = _b1_mlp, _b1_step
        flow.TtVoxtralFlow._block = _b2_block
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
