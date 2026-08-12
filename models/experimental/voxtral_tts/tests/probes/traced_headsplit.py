"""Traced marginal cost of BLOCK 2's head split -- the measurement 6.68 should have made.

6.68 closed the stale-rejection sweep and called 6.45 "reinforced, by a wider margin than when it
was made", on the strength of a fused head split costing **6.2 us traced**. That number is
`nlp_create_qkv_heads_DECODE`, Block 1's variant, which is what traced_ops.py measures. 6.45 is
about BLOCK 2, and Block 2 calls the NON-decode `nlp_create_qkv_heads` on a [2,1,3,6144] input.
They are different ops at different shapes and there is no reason for them to cost the same.

So: same method as traced_ops.py -- inject K extra copies into the TRACED graph, read the slope --
pointed at the op 6.45 is actually about, plus the 9-op hand-roll it replaced.

THE THIRD ARM IS THE POINT, and 6.31 is why it exists. The hand-rolled form must be measured with
its outputs in L1, like-for-like with the fused op's `memory_config=_L1`. 6.30/6.31 timed slices
and permutes at the DEFAULT memory config (DRAM) against a fused op given L1, read a 1.086x that
did not reproduce, and [flow-10] measured that same difference worth 2.5 ms/frame DOWNSTREAM,
because q/k/v then stay in L1 for the four ops that consume them. A DRAM-output hand-roll is not
the cheaper arm, it is a different computation with a bill that lands somewhere else.
"""
import json, os, time
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    FM_HEAD_DIM, FM_INPUT_DIM, FM_N_HEADS, FM_N_KV_HEADS)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
REPS, ROUNDS = 30, 5
B = 2                                    # the CFG-doubled batch _trunk hands _block
QW = flow._QKV_WIDTH
L1 = flow._L1

dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=250 * 1024 * 1024)
try:
    pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
    fl = pipe.flow
    w0 = fl.layers[0]

    # a real residual stream: prefill a real prompt, build the CFG pair exactly as _solve does
    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
    e = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                 pref.load_voice(case["voice"]), pipe.wb)
    h = pipe.backbone.prefill_last(e)[:, 0]                      # [1,3072]
    hh = fl._up(fl._cfg_input(1, h))                             # [2,3072] cond++uncond
    p2 = ttnn.reshape(ttnn.linear(hh, fl.proj["llm_projection"],
                                  compute_kernel_config=flow.COMPUTE_CONFIG),
                      [B, 1, FM_INPUT_DIM])
    p1s, _ = fl._schedule(B, flow.N_DECODING_STEPS)
    x0 = fl._up(torch.zeros(B, 1, flow.N_ACOUSTIC_CODEBOOK), ttnn.float32)
    p0 = ttnn.linear(ttnn.typecast(x0, fl.dtype), fl.proj["input_projection"],
                     compute_kernel_config=flow.COMPUTE_CONFIG)
    # born in L1, per [flow-22] -- load-bearing, and the in-place adds inherit it
    xin = ttnn.reshape(ttnn.concat([p0, p1s[0], p2], dim=1, memory_config=L1),
                       [1, B * 3, FM_INPUT_DIM])

    def fused(qkv):
        return ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, [B, 1, 3, QW]), num_heads=FM_N_HEADS,
            num_kv_heads=FM_N_KV_HEADS, transpose_k_heads=False, memory_config=L1)

    def hand(qkv, mc):
        """The 9-op form 6.45 deleted. `mc=None` reproduces the default-memory-config arm."""
        t = ttnn.reshape(qkv, [B, 1, 3, QW])
        kw = {} if mc is None else {"memory_config": mc}

        def take(lo, hi, nh):
            s = ttnn.slice(t, [0, 0, 0, lo], [B, 1, 3, hi], **kw)
            return ttnn.permute(ttnn.reshape(s, [B, 3, nh, FM_HEAD_DIM]), (0, 2, 1, 3), **kw)

        qw_, kw_ = flow._Q_WIDTH, flow._KV_WIDTH
        return (take(0, qw_, FM_N_HEADS), take(qw_, qw_ + kw_, FM_N_KV_HEADS),
                take(qw_ + kw_, QW, FM_N_KV_HEADS))

    EX = {"k": 0, "which": None}

    def graph():
        x = ttnn.clone(xin)
        h_ = fl._norm(x, w0["an"])
        qkv = ttnn.linear(h_, w0["wqkv"], program_config=gpt.DECODE_PRG["wqkv"],
                          compute_kernel_config=flow.COMPUTE_CONFIG)
        qh, kh, vh = fused(qkv)
        for _ in range(EX["k"]):
            if EX["which"] == "fused":
                qh, kh, vh = fused(qkv)
            elif EX["which"] == "hand_l1":
                qh, kh, vh = hand(qkv, L1)
            elif EX["which"] == "hand_default":
                qh, kh, vh = hand(qkv, None)
        a = ttnn.transformer.scaled_dot_product_attention(
            qh, kh, vh, is_causal=False, scale=1.0, compute_kernel_config=flow.COMPUTE_CONFIG)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), [1, B * 3, FM_N_HEADS * FM_HEAD_DIM])
        return ttnn.add_(x, ttnn.linear(a, w0["wo"], program_config=gpt.DECODE_PRG["wo"],
                                        compute_kernel_config=flow.COMPUTE_CONFIG,
                                        memory_config=L1))

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
        return min(out)          # min over rounds: least host contamination

    EX["which"], EX["k"] = None, 0
    base = timed()
    print(f"\n  one Block 2 attention half (norm+wqkv+split+sdpa+wo), traced: {base:.1f} us")
    print(f"  Block 2 runs _block 21x a frame (3 layers x 7 Euler steps)\n")
    print(f"  {'arm':<34} {'+2':>8} {'+4':>8} {'us/split':>10} {'ms/frame @21':>13}")
    rows = {}
    for which, label in (("fused", "nlp_create_qkv_heads (ships)"),
                         ("hand_l1", "hand-rolled 9-op, outputs L1"),
                         ("hand_default", "hand-rolled 9-op, DEFAULT mc")):
        EX["which"] = which
        EX["k"] = 2; t2 = timed()
        EX["k"] = 4; t4 = timed()
        per = (t4 - base) / 4
        rows[which] = per
        print(f"  {label:<34} {t2-base:>8.1f} {t4-base:>8.1f} {per:>10.1f} {per*21/1000:>13.3f}")

    d = (rows["fused"] - rows["hand_l1"]) * 21 / 1000
    print(f"\n  fused - hand(L1) = {d:+.3f} ms/frame  "
          f"({'hand-rolled wins -- 6.45 INVERTS' if d > 0 else '6.45 stands'})")
    print("  6.68 quoted 6.2 us for nlp_create_qkv_heads_DECODE (Block 1). This is the other op.")
    print("  A slope is a SCREEN (6.63). Decide on --tier audio ms_per_frame, paired, same tree.")
finally:
    ttnn.close_device(dev)
