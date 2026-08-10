"""Where did the 1.918 ms go? Block A/B says -1.918 ms/step; end-to-end says 0.

The generator harness is repeatable to 0.390 ms over three identical runs, so "no change" at
37.47 -> 37.54 ms/frame is a real measurement, not noise. Either the block A/B is measuring
something the real loop does not do, or the saving is being given back elsewhere.

This times the ACTUAL per-frame work -- Block 1 step, Block 2 frame, and the two together -- with
the arms INTERLEAVED in one session, which is the construction the block A/B used and the
generator does not. If Block 1 shows -1.9 here but the pair does not, the saving is being absorbed
by Block 2 or by the boundary between them.
"""
import json, os, time
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import DIM, N_ACOUSTIC_CODEBOOK
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
CC, _L1, P = gpt.COMPUTE_CONFIG, gpt._L1, gpt.DECODE_PRG
ROUNDS, REPS = 9, 25
_mlp_bias, _step_bias = gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step


def _mlp_add(self, x, h, w, mc, prg=None):
    prg = prg or {}
    g = (ttnn.linear(h, w["w1"], program_config=prg["w1"], compute_kernel_config=CC,
                     memory_config=mc) if prg else
         ttnn.linear(h, w["w1"], activation="silu", compute_kernel_config=CC, memory_config=mc))
    u = ttnn.multiply_(g, ttnn.linear(h, w["w3"], compute_kernel_config=CC, memory_config=mc,
                                      **gpt._pc(prg, "w3")))
    return ttnn.add_(x, ttnn.linear(u, w["w2"], compute_kernel_config=CC, memory_config=mc,
                                    **gpt._pc(prg, "w2")))


def _step_add(self, x, w, cos, sin, cache, pos_t):
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import N_HEADS, N_KV_HEADS
    qkv = ttnn.linear(self._norm(x, w["an"]), w["wqkv"], program_config=P["wqkv"],
                      compute_kernel_config=CC)
    qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, gpt._QKV_WIDTH]), gpt._QKV_SHARD)
    qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(qkv, num_heads=N_HEADS,
                                                              num_kv_heads=N_KV_HEADS)
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
    x = ttnn.add_(x, ttnn.linear(a, w["wo"], program_config=P["wo"], compute_kernel_config=CC,
                                 memory_config=_L1))
    return self._mlp(x, self._norm(x, w["fn"]), w, _L1, P)


def main():
    dev = open_device()
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
        embeds = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                          pref.load_voice(case["voice"]), pipe.wb)
        h = pipe.backbone.prefill_last(embeds)[:, 0]
        frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
        emb = bref.embed_frame(pipe.wb, frames[0]).reshape(1, 1, DIM)
        sem = pipe.flow.semantic_code(h)
        torch.manual_seed(0); x0 = torch.randn(1, N_ACOUSTIC_CODEBOOK)
        ARMS = [("bias  (HEAD)", _mlp_bias, _step_bias), ("bias#ctl", _mlp_bias, _step_bias),
                ("add_  (before)", _mlp_add, _step_add)]
        res = {l: {"b1": [], "b2": [], "both": []} for l,_,_ in ARMS}
        for r in range(ROUNDS):
            for lbl, m, s in (ARMS if r % 2 == 0 else ARMS[::-1]):
                gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step = m, s
                for key, fn in (("b1", lambda: pipe.backbone.step(emb)),
                                ("b2", lambda: pipe.flow.decode_frame(sem, h, x_0=x0)),
                                ("both", lambda: (pipe.backbone.step(emb),
                                                  pipe.flow.decode_frame(sem, h, x_0=x0)))):
                    pipe.backbone.pos = 60
                    fn(); ttnn.synchronize_device(dev)
                    pipe.backbone.pos = 60
                    t0 = time.perf_counter()
                    for _ in range(REPS): fn()
                    ttnn.synchronize_device(dev)
                    res[lbl][key].append((time.perf_counter()-t0)/REPS*1e3)
        print(f"\n  {'arm':>15} {'Block 1':>9} {'Block 2':>9} {'both':>9}")
        mean = lambda l,k: sum(res[l][k])/len(res[l][k])
        for lbl,_,_ in ARMS:
            print(f"  {lbl:>15} {mean(lbl,'b1'):>9.3f} {mean(lbl,'b2'):>9.3f} {mean(lbl,'both'):>9.3f}")
        print(f"\n  bias vs add_:  Block 1 {mean('add_  (before)','b1')-mean('bias  (HEAD)','b1'):+.3f}"
              f"   both {mean('add_  (before)','both')-mean('bias  (HEAD)','both'):+.3f}")
        print(f"  noise floor:   Block 1 {abs(mean('bias  (HEAD)','b1')-mean('bias#ctl','b1')):.3f}"
              f"   both {abs(mean('bias  (HEAD)','both')-mean('bias#ctl','both')):.3f}")
    finally:
        gpt.TtVoxtralGPT._mlp, gpt.TtVoxtralGPT._layer_step = _mlp_bias, _step_bias
        ttnn.close_device(dev)


main()
