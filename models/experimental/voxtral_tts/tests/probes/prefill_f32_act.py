"""fp32 ACTIVATIONS through prefill compute, with an explicit typecast at the KV-cache boundary.

6.56 measured fp32 prefill activations at 0.70% -> 0.29% relative error, then found the config
cannot run decode: fp32 activations make K/V fp32, sdpa_decode rejects an fp32 cache, and forcing
the cache to bf16 trips fill_cache's "same dtype" assert. The fix is one explicit typecast of K and
V to the cache dtype. This builds that and asks three things 6.56 could not:

  1. DOES IT RUN? prefill AND a decode step, with the cache at bf16 throughout.
  2. HOW MUCH SURVIVES the typecast? The last-position h never goes through the cache, so it keeps
     the full fp32 benefit. The cache does not.
  3. IS THE BENEFIT REALLY ONLY FRAME 0? 6.56 claimed it collapses there. That claim is probably
     WRONG and this is the test: a cache computed in fp32 and rounded ONCE at the boundary should
     be more accurate than one computed in bf16 at every intermediate step, which would help every
     later frame's attention over the prompt too. Measured as PCC(h_dev, h_ref) at decode steps
     0..3, which is exactly where 6.54 found the divergence lives.

Decode is untouched -- same bf16 activations, same bf16 cache, so no per-frame cost at all beyond
prefill's own (+0.06 ms/frame amortised, 6.56).
"""
import json
import os
import time

import torch
import ttnn

from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as bref
from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as pref
from models.experimental.voxtral_tts.reference.voxtral_common_ref import (
    DIM, HEAD_DIM, N_ACOUSTIC_CODEBOOK, NORM_EPS, pcc)
from models.experimental.voxtral_tts.tt import ttnn_voxtral_gpt as gpt
from models.experimental.voxtral_tts.tt import ttnn_voxtral_flow as flow
from models.experimental.voxtral_tts.tt.ttnn_voxtral_pipeline import open_device

HERE = os.path.join(os.environ["TT_METAL_HOME"], "models/experimental/voxtral_tts")
_ship_layer, _ship_prefill = gpt.TtVoxtralGPT._layer, gpt.TtVoxtralGPT.prefill


def _layer_cast(self, x, w, S, cos, sin, mask, cache=None):
    """_layer, but K and V are cast to the CACHE's dtype before fill_cache.

    Without this: fp32 activations -> fp32 K/V -> either an fp32 cache (sdpa_decode: "Unsupported
    data type DataType::FLOAT32") or a dtype mismatch (fill_cache: "same dtype"). Both are 6.56.
    """
    qh, kh, vh = self._qkv(x, w, S, cos, sin)
    if cache is not None:
        ttnn.fill_cache(cache[0], ttnn.typecast(kh, self.dtype), 0)
        ttnn.fill_cache(cache[1], ttnn.typecast(vh, self.dtype), 0)
    a = self._attend(qh, kh, vh, S, mask)
    x = ttnn.add(x, ttnn.linear(a, w["wo"], compute_kernel_config=gpt.COMPUTE_CONFIG))
    return self._mlp(x, self._norm(x, w["fn"]), w, ttnn.DRAM_MEMORY_CONFIG)


def prefill_f32(self, embeds, apply_final_norm=True, last_only=False):
    """prefill with fp32 ACTIVATIONS. Weights are untouched (6.56: fp32 weights buy nothing)."""
    S = embeds.shape[1]
    Sp = (S + gpt.PREFILL_MULTIPLE - 1) // gpt.PREFILL_MULTIPLE * gpt.PREFILL_MULTIPLE
    if self.caches and Sp > self.max_seq_len:
        raise ValueError(f"prompt pads to {Sp} but the KV cache holds {self.max_seq_len}")
    if Sp != S:
        embeds = torch.cat([embeds, embeds.new_zeros(1, Sp - S, DIM)], dim=1)
    cosb, sinb = gpt.rope_tables(Sp)
    up = lambda t, d: ttnn.from_torch(t.contiguous(), dtype=d, layout=ttnn.TILE_LAYOUT,
                                      device=self.device)
    F = ttnn.float32
    cos, sin = up(cosb.reshape(1, 1, Sp, HEAD_DIM), F), up(sinb.reshape(1, 1, Sp, HEAD_DIM), F)
    m = torch.full((Sp, Sp), float("-inf")).triu(1).reshape(1, 1, Sp, Sp)
    mask = up(m, ttnn.bfloat16)          # kept bf16: it is an additive -inf mask, not a value
    x = up(embeds.reshape(1, Sp, DIM), F)
    for i, w in enumerate(self.layers):
        x = self._layer(x, w, Sp, cos, sin, mask, self.caches[i] if self.caches else None)
    self.pos = S
    if last_only:
        x = ttnn.slice(x, [0, S - 1, 0], [1, S, DIM])
    if apply_final_norm:
        x = ttnn.rms_norm(x, weight=self.norm, epsilon=NORM_EPS,
                          compute_kernel_config=gpt.COMPUTE_CONFIG)
    return ttnn.to_torch(x).float().reshape(1, 1 if last_only else Sp, DIM)[:, :None if last_only
                                                                            else S]


def main():
    dev = open_device()
    try:
        wb = bref.load_backbone_state()
        case = json.load(open(os.path.join(HERE, "tests", "prompt_fixture.json")))["cases"][0]
        real = pref.build_inputs_embeds(
            torch.tensor(case["ids"], dtype=torch.long), pref.load_voice(case["voice"]), wb)
        ref_pre = bref.reference_forward(real, wb)[:, -1:]
        frames = torch.load(os.path.join(HERE, "tests", "real_frames_fixture.pt")).long()
        fl = flow.TtVoxtralFlow(dev)

        print(f"  {'arm':<16} {'prefill PCC':>12} {'rel':>7} {'s':>6}  "
              f"PCC(h) at decode step 0..3")
        for lbl, pf, ly in (("bf16 SHIPS", _ship_prefill, _ship_layer),
                            ("fp32 act", prefill_f32, _layer_cast)):
            gpt.TtVoxtralGPT.prefill, gpt.TtVoxtralGPT._layer = pf, ly
            g = gpt.TtVoxtralGPT(dev, state=wb, max_seq_len=512)
            try:
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                h_dev = g.prefill(real, last_only=True)
                ttnn.synchronize_device(dev)
                dt = time.perf_counter() - t0
                p = pcc(h_dev, ref_pre)
                rel = ((h_dev - ref_pre).abs().max() / ref_pre.abs().max()).item() * 100
                # THE question: does a more accurate prompt cache help LATER frames?
                rd = bref.IncrementalBackbone(wb)
                h_ref = rd.prefill(real)
                pccs = []
                hd = h_dev
                for i in range(4):
                    pccs.append(pcc(hd, h_ref))
                    emb = bref.embed_frame(wb, frames[i])
                    h_ref = rd.step(emb)
                    hd = g.step(emb.reshape(1, 1, DIM)).reshape(1, 1, DIM)
                print(f"  {lbl:<16} {p:>12.6f} {rel:>6.2f}% {dt:>6.2f}  "
                      + "  ".join(f"{v:.6f}" for v in pccs))
            except Exception as e:
                print(f"  {lbl:<16} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:60]}")
            del g
        print("\n  6.56 claimed the benefit 'collapses to frame 0'. Steps 1-3 above test that:")
        print("  if the fp32 row stays ahead there, the prompt CACHE is better too and the claim")
        print("  was wrong. Decode itself is unchanged in both arms -- bf16 activations, bf16 cache.")
    finally:
        gpt.TtVoxtralGPT.prefill, gpt.TtVoxtralGPT._layer = _ship_prefill, _ship_layer
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
