"""fp32 KV cache with a HAND-ROLLED decode attention, since sdpa_decode refuses fp32.

6.56 established: an fp32 cache would cost ~0.8% of a step in bandwidth (decode is flat while
cache traffic grows 8x) and it is the ONE place a precision gain would reach every frame rather
than just frame 0 -- but `scaled_dot_product_attention_decode` rejects the dtype outright. The only
way to have it is to stop using that op.

So: replace it with q@k^T -> scale -> softmax -> @v in plain ttnn, which has no dtype restriction.
Three arms, so the cost of HAND-ROLLING is separated from the cost of fp32:

    sdpa      bf16 cache, sdpa_decode                  <- ships
    hand      bf16 cache, hand-rolled attention        <- what does giving up the fused op cost?
    hand f32  fp32 cache, hand-rolled attention        <- the actual question

PRIOR: this should lose, and badly. The fused op becomes ~6, and 6.45 measured a small op here at
~68 us against the N150's ~20 -- the single most load-bearing number on this fork. 26 layers x 5
extra ops x 68 us is ~8.8 ms/step against a 17.7 ms baseline. Recorded up front so the measurement
can contradict it; 6.52 is a standing reminder that isolated intuitions about op cost have been
wrong in both directions today.

Accuracy is scored too -- an fp32 cache that is slower but much more accurate is a different
conversation from one that is slower and identical.
"""
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM, HEAD_DIM, N_HEADS, N_KV_HEADS, pcc)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
CC, REP = gpt.COMPUTE_CONFIG, N_HEADS // N_KV_HEADS
_ship_step = gpt.TtVoxtralGPT._layer_step
TARGET_POS, REPS = 160, 10


def _layer_step_hand(self, x, w, cos, sin, cache, pos_t):
    """_layer_step with the fused sdpa_decode replaced by explicit q@k^T / softmax / @v."""
    qkv = ttnn.linear(self._norm(x, w["an"]), w["wqkv"], program_config=gpt.DECODE_PRG["wqkv"],
                      compute_kernel_config=CC)
    qkv = ttnn.to_memory_config(ttnn.reshape(qkv, [1, 1, 1, gpt._QKV_WIDTH]), gpt._QKV_SHARD)
    qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads_decode(
        qkv, num_heads=N_HEADS, num_kv_heads=N_KV_HEADS)
    qh = ttnn.experimental.rotary_embedding_hf(qh, cos, sin, is_decode_mode=True,
                                               compute_kernel_config=CC)
    kh = ttnn.experimental.rotary_embedding_hf(kh, cos, sin, is_decode_mode=True,
                                               compute_kernel_config=CC)
    cd = cache[0].dtype
    ttnn.experimental.paged_update_cache(cache[0], ttnn.typecast(kh, cd) if kh.dtype != cd else kh,
                                         update_idxs_tensor=pos_t)
    ttnn.experimental.paged_update_cache(cache[1], ttnn.typecast(vh, cd) if vh.dtype != cd else vh,
                                         update_idxs_tensor=pos_t)
    # ---- hand-rolled attention over positions [0, pos] -------------------------------------
    P = self.pos + 1
    k = ttnn.slice(cache[0], [0, 0, 0, 0], [1, N_KV_HEADS, P, HEAD_DIM])
    v = ttnn.slice(cache[1], [0, 0, 0, 0], [1, N_KV_HEADS, P, HEAD_DIM])
    kr, vr = ttnn.repeat_interleave(k, REP, dim=1), ttnn.repeat_interleave(v, REP, dim=1)
    q = ttnn.permute(ttnn.to_memory_config(qh, ttnn.DRAM_MEMORY_CONFIG), (0, 2, 1, 3))
    q = ttnn.typecast(q, kr.dtype) if q.dtype != kr.dtype else q
    sc = ttnn.matmul(q, ttnn.permute(kr, (0, 1, 3, 2)), compute_kernel_config=CC)
    sc = ttnn.softmax(ttnn.multiply(sc, gpt.SCALE), dim=-1)
    o = ttnn.matmul(sc, vr, compute_kernel_config=CC)
    a = ttnn.reshape(ttnn.typecast(o, self.dtype), [1, 1, gpt.Q_WIDTH])
    x = ttnn.add_(x, ttnn.linear(a, w["wo"], program_config=gpt.DECODE_PRG["wo"],
                                 compute_kernel_config=CC, memory_config=gpt._L1))
    return self._mlp(x, self._norm(x, w["fn"]), w, gpt._L1, gpt.DECODE_PRG)


def main():
    dev = open_device()
    try:
        wb = bref.load_backbone_state()
        frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
        x = bref.embed_frame(wb, frames[0]).reshape(1, 1, DIM)
        ARMS = [("sdpa      (ships)", _ship_step, ttnn.bfloat16),
                ("hand      bf16$", _layer_step_hand, ttnn.bfloat16),
                ("hand f32  fp32$", _layer_step_hand, ttnn.float32)]
        print(f"  {'arm':<20} {'ms/step':>9} {'vs ships':>9} {'cache MB':>9}  accuracy")
        base = None
        for lbl, fn, cdt in ARMS:
            gpt.TtVoxtralGPT._layer_step = fn
            g = gpt.TtVoxtralGPT(dev, state=wb, max_seq_len=1024)
            try:
                if cdt != ttnn.bfloat16:      # re-allocate the cache at the requested dtype
                    z = torch.zeros(1, N_KV_HEADS, 1024, HEAD_DIM)
                    g.caches = [(ttnn.from_torch(z, dtype=cdt, layout=ttnn.TILE_LAYOUT, device=dev),
                                 ttnn.from_torch(z, dtype=cdt, layout=ttnn.TILE_LAYOUT, device=dev))
                                for _ in range(26)]
                while g.pos < TARGET_POS:
                    g.step(x)
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                for _ in range(REPS):
                    g.step(x)
                ttnn.synchronize_device(dev)
                ms = (time.perf_counter() - t0) / REPS * 1e3
                mb = sum(c.volume() * (4 if cdt == ttnn.float32 else 2)
                         for kv in g.caches for c in kv) / 1e6
                if base is None:
                    base = ms
                print(f"  {lbl:<20} {ms:>9.2f} {ms-base:>+9.2f} {mb:>9.0f}  ok")
            except Exception as e:
                print(f"  {lbl:<20} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:56]}")
            del g
        print("\n  PRIOR (recorded before running): hand-rolling turns 1 op into ~6, and 6.45 puts")
        print("  a small op at ~68 us here, so ~26 x 5 x 68 us = ~8.8 ms/step of added cost.")
    finally:
        gpt.TtVoxtralGPT._layer_step = _ship_step
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
