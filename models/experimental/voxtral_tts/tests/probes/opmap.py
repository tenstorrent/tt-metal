"""Per-op time and overhead for both blocks, on current HEAD.

6.45's map predates the program configs (6.52), residual-as-bias (6.62) and tracing (6.65), so
every number in it is stale. This re-measures in situ: each ttnn call inside the real
`_layer_step` / `_block` is timed with a synchronize on both sides, so the attribution is right
even though serialising them inflates the total.

OVERHEAD is the branch's floor method (6.41): an op that only moves weights cannot beat
bytes / 367 GB/s, so overhead = measured - that floor. For ops with no weights the floor is
essentially zero and the whole time is overhead.

TWO THINGS TO READ CAREFULLY:
  * these are SERIALISED measurements. The shipped path pipelines, so the column sums exceed the
    real block time -- use the shares, not the totals.
  * the shipped path is TRACED (6.65), so the per-op DISPATCH counted here is not paid in
    production. The traced block time is printed underneath for exactly that comparison.
"""
import json, os, time
from collections import defaultdict
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM, HEAD_DIM, N_ACOUSTIC_CODEBOOK)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline, open_device
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
CEIL = 367e9
W = {ttnn.bfloat8_b: 1.0625, ttnn.bfloat4_b: 0.5625, ttnn.bfloat16: 2.0, ttnn.float32: 4.0,
     ttnn.uint32: 4.0, ttnn.int32: 4.0}

T = defaultdict(lambda: [0.0, 0, 0.0])     # name -> [seconds, calls, weight bytes]
ON = [False]
_saved = []


def wrap(mod, name, dev):
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        return
    def w(*a, **k):
        if not ON[0]:
            return fn(*a, **k)
        # weight bytes = the largest tensor argument, which for a linear IS the weight
        wb = 0.0
        for t in list(a) + list(k.values()):
            if isinstance(t, ttnn.Tensor):
                try: wb = max(wb, t.volume() * W.get(t.dtype, 2.0))
                except Exception: pass
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn(*a, **k)
        ttnn.synchronize_device(dev)
        e = T[name]; e[0] += time.perf_counter() - t0; e[1] += 1; e[2] += wb
        return r
    setattr(mod, name, w); _saved.append((mod, name, fn))


def report(title, calls_per_frame, traced_ms):
    tot = sum(v[0] for v in T.values())
    print(f"\n=== {title} ===")
    print(f"  {'op':<34} {'calls':>6} {'us/call':>8} {'MB/call':>8} {'floor us':>9} "
          f"{'overhead':>9} {'share':>6}")
    for name, (sec, n, wb) in sorted(T.items(), key=lambda kv: -kv[1][0]):
        us = sec / n * 1e6
        mb = wb / n / 1e6
        fl = (wb / n) / CEIL * 1e6
        print(f"  {name:<34} {n:>6} {us:>8.1f} {mb:>8.2f} {fl:>9.1f} {us-fl:>9.1f} "
              f"{sec/tot*100:>5.1f}%")
    print(f"  {'':<34} {'':>6} {'':>8} {'':>8} {'':>9} {'serialised total':>18} "
          f"{tot*1e3/calls_per_frame:.2f} ms")
    print(f"  the same work TRACED and pipelined: {traced_ms:.2f} ms/frame")
    T.clear()


def main():
    dev = open_device(trace_region_size=0)
    try:
        pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
        bb, fl = pipe.backbone, pipe.flow
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
        e = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                     pref.load_voice(case["voice"]), pipe.wb)
        h = bb.prefill_last(e)[:, 0]
        frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
        for m, names in ((ttnn, ["linear", "matmul", "add", "add_", "multiply", "multiply_",
                                 "concat", "typecast", "slice", "permute", "reshape", "clone",
                                 "rms_norm", "to_memory_config", "zeros_like"]),
                         (ttnn.experimental, ["nlp_create_qkv_heads",
                                              "nlp_create_qkv_heads_decode",
                                              "rotary_embedding_hf", "paged_update_cache"]),
                         (ttnn.transformer, ["scaled_dot_product_attention",
                                             "scaled_dot_product_attention_decode"])):
            for n in names:
                wrap(m, n, dev)

        # ---------------- Block 1: one full 26-layer decode step ----------------
        x = bref.embed_frame(pipe.wb, frames[0]).reshape(1, 1, DIM)
        bb.step(x)                                    # warm
        ON[0] = True
        bb.step(x)
        ON[0] = False
        report("BLOCK 1 -- one decode step, 26 layers", 1, 15.9)

        # ---------------- Block 2: one full frame (7 Euler steps) ----------------
        sem = fl.semantic_code(h)
        torch.manual_seed(0); x0 = torch.randn(1, N_ACOUSTIC_CODEBOOK)
        fl.decode_frame(sem, h, x_0=x0)               # warm
        ON[0] = True
        fl.decode_frame(sem, h, x_0=x0)
        ON[0] = False
        report("BLOCK 2 -- one frame, 7 Euler steps x 3 layers", 1, 15.0)
    finally:
        for m, n, f in _saved:
            setattr(m, n, f)
        ttnn.close_device(dev)


main()
