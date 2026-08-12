"""Traced marginal cost of `_trunk`'s sequence build -- ONBOARDING 7's "most likely next win".

That entry rests on 6.36, an EAGER op map on the N150: lines 174+176 (a 3-way concat and the
reshape to folded rows) at 101.0 and 106.0 us x7 = 1.449 ms/frame combined, "the largest genuinely
untouched item". Two things make it worth re-measuring rather than acting on:

  * it is a WORMHOLE number, never re-taken on this chip;
  * 6.67 measured `concat` at 144.7 us eager and **2.6 us traced** -- a 98%-launch-cost ghost --
    because 6.65 traced the frame. An eager map ranks by launch cost, which is not what ships.

6.72 is the cautionary tale in the other direction: 6.68 closed a line of enquiry using a traced
cost measured on the WRONG OP, and 1.8 ms/frame sat unclaimed. So measure these two, at their real
shapes, inside the trace, by the injection method 6.67/6.72 used -- and read the slope, not the map.

`_trunk` runs 7x a frame (once per Euler step), not 21x like `_block`.
"""
import json, os, time
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import FM_INPUT_DIM
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
REPS, ROUNDS = 30, 5
B = 2                                   # the CFG-doubled batch _solve hands _trunk
L1 = flow._L1
CALLS = 7                               # _trunk invocations per frame

dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=250 * 1024 * 1024)
try:
    pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
    fl = pipe.flow
    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
    e = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                 pref.load_voice(case["voice"]), pipe.wb)
    h = pipe.backbone.prefill_last(e)[:, 0]

    # the three [B,1,3072] projections exactly as _solve builds them
    hh = fl._up(fl._cfg_input(1, h))
    p2 = ttnn.reshape(ttnn.linear(hh, fl.proj["llm_projection"],
                                  compute_kernel_config=flow.COMPUTE_CONFIG),
                      [B, 1, FM_INPUT_DIM])
    p1s, _ = fl._schedule(B, flow.N_DECODING_STEPS)
    p1 = p1s[0]
    x0 = fl._up(torch.zeros(B, 1, flow.N_ACOUSTIC_CODEBOOK), ttnn.float32)
    p0 = ttnn.linear(ttnn.typecast(x0, fl.dtype), fl.proj["input_projection"],
                     compute_kernel_config=flow.COMPUTE_CONFIG)

    EX = {"k": 0, "which": None}

    def graph():
        # memory_config on the concat is load-bearing -- [flow-22]: it puts the residual stream in
        # L1 so _block's add_ inherits it. Injected copies keep it so the arm stays faithful.
        seq = ttnn.concat([p0, p1, p2], dim=1, memory_config=L1)
        for _ in range(EX["k"] if EX["which"] == "concat" else 0):
            seq = ttnn.concat([p0, p1, p2], dim=1, memory_config=L1)
        folded = ttnn.reshape(seq, [1, B * 3, FM_INPUT_DIM])
        for _ in range(EX["k"] if EX["which"] == "reshape" else 0):
            folded = ttnn.reshape(seq, [1, B * 3, FM_INPUT_DIM])
        # one real block, so the pair is measured with the neighbours it actually has
        return fl._block(folded, fl.layers[0], B)

    def timed():
        graph(); ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try: graph()
        finally: ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        out = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(REPS): ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            out.append((time.perf_counter() - t0) / REPS * 1e6)
        ttnn.release_trace(dev, tid)
        return min(out)

    EX["which"], EX["k"] = None, 0
    base = timed()
    print(f"\n  base graph (concat + reshape + one _block), traced: {base:.1f} us\n")
    print(f"  {'op':<34} {'+2':>8} {'+4':>8} {'traced us':>10} {'ms/frame @7':>12} {'6.36 eager':>11}")
    for which, label, eager in (("concat", "concat([p0,p1,p2], dim=1)", 101.0),
                                ("reshape", "reshape -> [1,B*3,3072]", 106.0)):
        EX["which"] = which
        EX["k"] = 2; t2 = timed()
        EX["k"] = 4; t4 = timed()
        per = (t4 - base) / 4
        print(f"  {label:<34} {t2-base:>8.1f} {t4-base:>8.1f} {per:>10.1f} "
              f"{per*CALLS/1000:>12.3f} {eager*CALLS/1000:>10.3f}ms")
    print("\n  ONBOARDING 7 claims 1.449 ms/frame combined, from an EAGER N150 map (6.36).")
    print("  If both are single-digit us traced, that lead is a ghost and 7 should say so.")
finally:
    ttnn.close_device(dev)
