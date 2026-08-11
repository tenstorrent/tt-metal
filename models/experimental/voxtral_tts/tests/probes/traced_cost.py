"""What do rms_norm and concat cost INSIDE the trace? The eager map cannot say.

The op map is eager and serialised: rms_norm reads 101 us and concat 145, but 2.4-2.6x of the
serialised total is per-op launch cost that 6.65's trace already removes. Attacking either on the
strength of an eager number would be 6.63's mistake again.

Direct measurement instead: inject K EXTRA copies of the op into the traced graph and read the
slope. The slope is the op's marginal cost in exactly the context that ships. If it is near zero,
neither is worth touching and the eager map was pointing at ghosts.
"""
import json, os, time
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import N_ACOUSTIC_CODEBOOK
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
REPS, ROUNDS = 30, 5
dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=250*1024*1024)
tids = []
try:
    pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
    fl = pipe.flow
    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
    e = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                 pref.load_voice(case["voice"]), pipe.wb)
    h = pipe.backbone.prefill_last(e)[:, 0]
    torch.manual_seed(0)
    xd = fl._up(torch.randn(1, 1, N_ACOUSTIC_CODEBOOK), ttnn.float32)
    hd = fl._up(fl._cfg_input(1, h))

    _norm0 = flow.TtVoxtralFlow._norm
    EXTRA = {"norm": 0, "concat": 0}

    def norm_k(self, x, g):
        r = _norm0(self, x, g)
        for _ in range(EXTRA["norm"]):
            r = _norm0(self, r, g)                    # extra rms_norms on the same shape
        return r

    def solve_with_extra(*a, **k):
        out = fl._solve(*a, **k)
        for _ in range(EXTRA["concat"]):
            out = ttnn.slice(ttnn.concat([out, out], dim=0), [0, 0, 0],
                             [1, 1, N_ACOUSTIC_CODEBOOK])
        return out

    flow.TtVoxtralFlow._norm = norm_k

    def timed(nrm, cat):
        EXTRA["norm"], EXTRA["concat"] = nrm, cat
        solve_with_extra(xd, hd, 1, flow.N_DECODING_STEPS, flow.CFG_ALPHA)
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        try:
            solve_with_extra(xd, hd, 1, flow.N_DECODING_STEPS, flow.CFG_ALPHA)
        finally:
            ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        out = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(REPS):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            out.append((time.perf_counter() - t0) / REPS * 1e3)
        ttnn.release_trace(dev, tid)
        return sum(out) / len(out)

    print(f"  Block 2 has 49 rms_norm and 14 concat calls per frame (eager map).\n")
    print(f"  {'extra per _norm call':>21} {'total extra norms':>18} {'ms/frame':>10} {'slope us/op':>12}")
    base = None
    for k in (0, 1, 2):
        ms = timed(k, 0)
        if base is None: base = ms
        n = k * 49
        print(f"  {k:>21} {n:>18} {ms:>10.3f} "
              f"{((ms-base)*1000/n if n else 0):>12.1f}")
    print(f"\n  {'extra concats/frame':>21} {'':>18} {'ms/frame':>10} {'slope us/op':>12}")
    base2 = None
    for k in (0, 4, 8):
        ms = timed(0, k)
        if base2 is None: base2 = ms
        print(f"  {k:>21} {'':>18} {ms:>10.3f} "
              f"{((ms-base2)*1000/k if k else 0):>12.1f}")
    print("\n  eager map said rms_norm 101 us/call and concat 145 us/call.")
finally:
    flow.TtVoxtralFlow._norm = _norm0
    for t in tids:
        try: ttnn.release_trace(dev, t)
        except Exception: pass
    ttnn.close_device(dev)
