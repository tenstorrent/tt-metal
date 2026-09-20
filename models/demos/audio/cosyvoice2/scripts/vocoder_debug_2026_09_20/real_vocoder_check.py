"""Vocoder isolation test (technique borrowed from CosyVoice1 reference PR
#52540): take our flow decoder's mel output for the exact test sample and
feed it into the REAL, official, unmodified CosyVoice2 HiFTGenerator +
ConvRNNF0Predictor classes (fetched fresh from FunAudioLLM/CosyVoice, run
completely outside our own TT/torch port) with the real hift.pt weights. If
this sounds clean, the bug is downstream of the flow decoder (our vocoder/F0
predictor). If it's still noisy/garbled, the bug is upstream (the flow
decoder's mel itself).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/user/tt-metal")
sys.path.insert(0, "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad/real_cosyvoice_pkg")

SCRATCH = "/tmp/claude-1000/-home-user-tt-metal/135d13b5-798c-4cd3-ac97-1e2502d3ba47/scratchpad"

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict

print("=== loading real hift.pt + our flow decoder's mel for the exact test sample ===")
hift_sd = load_checkpoint_file("hift.pt")
mel = torch.from_numpy(np.load(f"{SCRATCH}/stage1_v3_mel.npy"))  # [1, T, 80], channels-last (this port's convention)
print(f"mel: {mel.shape}, mean={mel.mean().item():.4f} std={mel.std().item():.4f}")

from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
from cosyvoice.hifigan.generator import HiFTGenerator

print("=== building the REAL, unmodified HiFTGenerator + ConvRNNF0Predictor ===")
f0_predictor = ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512)
generator = HiFTGenerator(
    in_channels=80,
    base_channels=512,
    nb_harmonics=8,
    sampling_rate=24000,
    nsf_alpha=0.1,
    nsf_sigma=0.003,
    nsf_voiced_threshold=10,
    upsample_rates=[8, 5, 3],
    upsample_kernel_sizes=[16, 11, 7],
    istft_params={"n_fft": 16, "hop_len": 4},
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    source_resblock_kernel_sizes=[7, 7, 11],
    source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    lrelu_slope=0.1,
    audio_limit=0.99,
    f0_predictor=f0_predictor,
)

print("=== loading real hift.pt weights into the REAL classes (strict) ===")
missing, unexpected = generator.load_state_dict(hift_sd, strict=False)
print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
if missing:
    print("  missing (first 20):", missing[:20])
if unexpected:
    print("  unexpected (first 20):", unexpected[:20])
generator.eval()

print("\n=== real HiFTGenerator.inference() on our flow decoder's real mel ===")
# real HiFTGenerator.inference expects channel-first mel [B, 80, T] (its own
# convention, confirmed from the cached real source's docstrings/usage in
# cosyvoice/cli/model.py) -- our port stores mel channels-last [B, T, 80].
mel_cf = mel.transpose(1, 2).contiguous()
with torch.no_grad():
    wav, _ = generator.inference(speech_feat=mel_cf)
wav = wav.squeeze().float()
print(f"real-vocoder waveform: {wav.shape}, mean={wav.mean().item():.5f} std={wav.std().item():.5f}")

import numpy as np
from scipy.io import wavfile

wav_np = wav.numpy()
wavfile.write(f"{SCRATCH}/stage1_synth_v3_REALVOCODER.wav", 24000, (np.clip(wav_np, -1, 1) * 32767).astype(np.int16))
print(f"wrote {SCRATCH}/stage1_synth_v3_REALVOCODER.wav")

# quick numeric comparison against our own TT vocoder's waveform for this SAME mel
our_wav = np.load(f"{SCRATCH}/stage1_v3_mel.npy")  # placeholder, replaced below
try:
    import soundfile as sf

    our_wav, our_sr = sf.read(f"{SCRATCH}/stage1_synth_v3_ourvocoder.wav")
except Exception:
    our_sr, our_wav = wavfile.read(f"{SCRATCH}/stage1_synth_v3_ourvocoder.wav")
    our_wav = our_wav.astype(np.float32) / 32767.0

n = min(len(our_wav), len(wav_np))
corr = np.corrcoef(our_wav[:n], wav_np[:n])[0, 1]
print(f"\nsample-domain correlation (our TT vocoder vs real official vocoder, SAME mel, first {n} samples): {corr:.4f}")
print("(note: even a CORRECT vocoder can have low sample-domain correlation vs ours due to phase/ISTFT differences --")
print(" the real test is what it SOUNDS like, not this correlation number. Listen to both files.)")
