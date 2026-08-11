"""Traced marginal cost of the ops behind 6.44 / 6.45 / 6.28, before spending effort on any of them.

The eager map ranks by launch cost and got concat vs rms_norm exactly backwards (6.67). So measure
each candidate the same way that settled those: inject K extra copies into the TRACED graph and
read the slope. That is the op's cost in the path that actually ships.

Candidates and what a win would mean:
  nlp_create_qkv_heads / _decode  -- 6.45 replaced a 9-op hand-rolled split with this fused op
                                     BECAUSE ops cost 67.7 us each. If the fused op is expensive in
                                     DEVICE time, that trade may have inverted now ops are cheap.
  paged_update_cache              -- 6.44 kept TWO of these over one fused write. The fused one lost
                                     0.687 ms/step, which cannot be launch (it is FEWER ops), so it
                                     lost on device work and tracing should not rescue it.
  to_memory_config                -- 6.67's sharded norm ends by UNSHARDING. 6.28 wanted to feed the
                                     sharded form straight into a DRAM-sharded matmul; its premise
                                     died when 6.39 removed the sharded norm, and 6.67 revived it.
"""
import json, os, time
import torch, ttnn
from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM, HEAD_DIM, N_HEADS, N_KV_HEADS)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import TtVoxtralPipeline
HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
REPS, ROUNDS = 30, 5
dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=250*1024*1024)
try:
    pipe = TtVoxtralPipeline(dev, max_seq_len=2048)
    bb = pipe.backbone
    case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][2]
    e = pref.build_inputs_embeds(torch.tensor(case["ids"], dtype=torch.long),
                                 pref.load_voice(case["voice"]), pipe.wb)
    bb.prefill_last(e); pos = bb.pos
    frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
    up = lambda t, d=None: ttnn.from_torch(t.contiguous(), dtype=d or bb.dtype,
                                           layout=ttnn.TILE_LAYOUT, device=dev)
    cb, sb = gpt.rope_tables(1, offset=pos)
    cos = ttnn.to_memory_config(up(cb.reshape(1,1,1,HEAD_DIM)), gpt._ROPE_SHARD)
    sin = ttnn.to_memory_config(up(sb.reshape(1,1,1,HEAD_DIM)), gpt._ROPE_SHARD)
    pos_t = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=dev)
    xin = up(bref.embed_frame(pipe.wb, frames[0]).reshape(1, 1, DIM))
    w0, c0 = bb.layers[0], bb.caches[0]

    EX = {"k": 0, "which": None}

    def graph():
        h = bb._norm(ttnn.clone(xin), w0["an"])
        qkv = ttnn.linear(h, w0["wqkv"], program_config=gpt.DECODE_PRG["wqkv"],
                          compute_kernel_config=gpt.COMPUTE_CONFIG)
        qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1,1,1,gpt._QKV_WIDTH]), gpt._QKV_SHARD)
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        for _ in range(EX["k"] if EX["which"] == "heads" else 0):
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
                qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
        qh = ttnn.experimental.rotary_embedding_hf(qh, cos, sin, is_decode_mode=True,
                                                  compute_kernel_config=gpt.COMPUTE_CONFIG)
        kh = ttnn.experimental.rotary_embedding_hf(kh, cos, sin, is_decode_mode=True,
                                                  compute_kernel_config=gpt.COMPUTE_CONFIG)
        ttnn.experimental.paged_update_cache(c0[0], kh, update_idxs_tensor=pos_t)
        ttnn.experimental.paged_update_cache(c0[1], vh, update_idxs_tensor=pos_t)
        for _ in range(EX["k"] if EX["which"] == "cache" else 0):
            ttnn.experimental.paged_update_cache(c0[1], vh, update_idxs_tensor=pos_t)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh, c0[0], c0[1], cur_pos_tensor=pos_t, scale=gpt.SCALE,
            compute_kernel_config=gpt.COMPUTE_CONFIG, program_config=gpt._SDPA_PRG)
        for _ in range(EX["k"] if EX["which"] == "sdpa" else 0):
            o = ttnn.transformer.scaled_dot_product_attention_decode(
                qh, c0[0], c0[1], cur_pos_tensor=pos_t, scale=gpt.SCALE,
                compute_kernel_config=gpt.COMPUTE_CONFIG, program_config=gpt._SDPA_PRG)
        r = ttnn.reshape(o, [1, 1, gpt.Q_WIDTH])
        for _ in range(EX["k"] if EX["which"] == "unshard" else 0):
            r = ttnn.to_memory_config(ttnn.to_memory_config(r, gpt._NORM_SHARD), gpt._L1)
        return r

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
            out.append((time.perf_counter()-t0)/REPS*1e6)
        ttnn.release_trace(dev, tid)
        return sum(out)/len(out)

    EX["which"] = None; base = timed()
    print(f"  one Block 1 layer, traced: {base:.1f} us\n")
    print(f"  {'op':<32} {'+2 copies':>10} {'+4 copies':>10} {'traced us/op':>14}")
    for which, label in (("heads", "nlp_create_qkv_heads_decode"),
                         ("cache", "paged_update_cache"),
                         ("sdpa", "sdpa_decode"),
                         ("unshard", "to_memory_config pair (reshard)")):
        EX["which"] = which
        EX["k"] = 2; t2 = timed()
        EX["k"] = 4; t4 = timed()
        print(f"  {label:<32} {t2-base:>10.1f} {t4-base:>10.1f} {(t4-base)/4:>14.1f}")
    print("\n  eager map: heads 71.3 us, paged_update_cache 38.2, sdpa_decode 66.3")
finally:
    ttnn.close_device(dev)
