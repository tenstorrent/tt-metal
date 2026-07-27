# SPDX-License-Identifier: Apache-2.0
"""Warm per-block perf table for the XTTS-v2 TTNN pipeline. Each block is built once, run a
few times to compile+cache kernels (program cache), then timed over N warm iterations.
Inputs come from a real pipeline work dir. TT-only (the coqui CPU phases are excluded)."""
import os, time, torch, ttnn
import torch.nn.functional as F

from models.experimental.xtts_v2.tt.ttnn_xtts_speaker import TTNNSpeakerEncoder, preprocess_speaker_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import (
    TTNNConditioningEncoder,
    TTNNPerceiver,
    preprocess_encoder_parameters,
    preprocess_perceiver_parameters,
    LATENTS,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder
from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import TTNNHifiganGenerator, preprocess_hifigan_parameters

# WORK is a pipeline work dir produced by run_pipeline.sh (holds the captured block inputs).
WORK = os.environ.get("XTTS_WORK", "./xtts_pipeline_out/work")
_ckpt_dir = os.environ.get("XTTS_CKPT_DIR")
if _ckpt_dir:
    os.environ.setdefault("XTTS_CKPT", os.path.join(_ckpt_dir, "model.pth"))


def timeit(fn, n, warm=3):
    for _ in range(warm):  # compile + warm the program cache
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n * 1000.0  # ms


dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=200_000_000)
res = {}
try:
    dev.enable_program_cache()

    # --- Block 2: ResNet speaker encoder ---
    logmel = torch.load(f"{WORK}/speaker_logmel.pt").float()
    spk = TTNNSpeakerEncoder(dev, preprocess_speaker_parameters(dev))
    logmel_tt = ttnn.from_torch(logmel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def spk_fwd():
        e = spk(logmel_tt)
        ttnn.to_torch(e[0] if isinstance(e, tuple) else e)

    res["Block2 speaker encoder"] = (timeit(spk_fwd, 20), "ms/call")

    # --- Block 1: conditioning encoder + Perceiver ---
    mel = torch.load(f"{WORK}/cond_mel_in.pt").float()
    T = mel.shape[2]
    S = ((T + 31) // 32) * 32
    mel_f = F.pad(mel.permute(0, 2, 1).contiguous(), (0, 0, 0, S - T))
    enc = TTNNConditioningEncoder(dev, preprocess_encoder_parameters(dev, dtype=ttnn.float32), t_real=T, s_pad=S)
    perc = TTNNPerceiver(dev, preprocess_perceiver_parameters(dev, dtype=ttnn.float32))
    km = torch.zeros(1, 1, 1, LATENTS + S)
    km[:, :, :, LATENTS + T :] = -1e9
    km_tt = ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    mel_tt = ttnn.from_torch(mel_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    def cond_fwd():
        ttnn.to_torch(perc(enc(mel_tt), km_tt))

    res["Block1 cond + perceiver"] = (timeit(cond_fwd, 20), "ms/call")

    # --- Block 3: GPT traced decode (warm ms/token, position advancing like a real decode) ---
    p = preprocess_gpt_parameters(dev, dtype=ttnn.bfloat16)
    prefix = torch.load(f"{WORK}/prefix_emb.pt").float()  # real prompt prefix [1,P,1024]
    Pn = prefix.shape[1]
    max_seq = ((Pn + 1 + 605 + 63) // 64) * 64  # prefill(P) + START(1) + up to 605 codes
    dec = TTNNGPTTracedDecoder(dev, p, max_seq=max_seq)

    # one-shot parallel prefill of the whole prompt (fills KV cache 0..P-1 in a single pass)
    def gpt_prefill():
        dec.reset_caches()
        dec.prefill(prefix)

    res["Block3 GPT prefill"] = (timeit(gpt_prefill, 10, warm=2), f"ms/call (P={Pn}, one-shot)")

    dec.reset_caches()
    dec.prefill(prefix)  # leave a valid prefilled cache in place before capturing the decode graph
    dec.capture()
    x = torch.randn(1, 1, 1024)
    posc = [Pn]  # decode positions advance past the prefill region, like a real run

    def gpt_step():
        emb_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn.to_torch(dec.step_device(emb_dev, Pn + (posc[0] - Pn) % 600))
        posc[0] += 1

    res["Block3 GPT decode"] = (timeit(gpt_step, 100), "ms/token")

    # --- Block 4: HiFi-GAN vocoder (fp32) ---
    gl = torch.load(f"{WORK}/gpt_latents_tt.pt").float()
    g = torch.load(f"{WORK}/speaker_embedding_tt.pt").float()
    AR_COMP, HOP, ISR, OSR = 1024, 256, 22050, 24000
    z = F.interpolate(gl.transpose(1, 2), scale_factor=AR_COMP / HOP, mode="linear")
    z = F.interpolate(z, scale_factor=OSR / ISR, mode="linear")
    Lz = z.shape[-1]
    voc = TTNNHifiganGenerator(dev, preprocess_hifigan_parameters(dev))
    z_tt = ttnn.from_torch(
        z.permute(0, 2, 1).reshape(1, 1, Lz, 1024), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    g_tt = ttnn.from_torch(g.reshape(1, 512), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    def voc_fwd():
        ttnn.to_torch(voc(z_tt, g_tt))

    res["Block4 HiFi-GAN vocoder"] = (timeit(voc_fwd, 10, warm=2), f"ms/call (out {Lz*256} samples)")

    print("\n==================== WARM per-block perf (single N150, bf16 decode) ====================")
    for k, (v, unit) in res.items():
        print(f"  {k:28s} {v:8.2f} {unit}")
finally:
    ttnn.close_device(dev)
