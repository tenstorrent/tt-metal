"""Experiments A and B: is F0 predictor error what makes the TT vocoder sound wrong?

All runs: SAME real mel (stage1_v4_mel.npy, 464 frames), SAME fixed harmonic noise draw N1.

  OFFICIAL  real FunAudioLLM HiFTGenerator (randn_like patched to N1)   -- audible ground truth
  TORCH     our torch reference, torch f0, N1                            -- proven == OFFICIAL upstream
  TT_OWN    TT generator fp32, its own (device) f0, N1                   -- current production path
  A         TT generator fp32, TORCH f0 injected, N1                     -- removes F0 error from the TT path
  B         torch reference, TT device f0 injected, N1                   -- adds ONLY F0 error to a correct vocoder
  TORCH_N2  torch reference, torch f0, DIFFERENT noise N2                -- "two correct vocoders differ by this much"

Distances are phase-insensitive (log-mel L1 in dB) vs TORCH, per mel frame, so they can be lined up
with per-frame F0 error. Wavs written to WAVDIR for listening.
"""
import os
import sys

import numpy as np
import torch
from scipy.io import wavfile

import ttnn

SCR = os.environ.get("COSYVOICE2_DEBUG_OUT", "/tmp/cosyvoice2_debug")  # scratch outputs; never the repo
WAVDIR = f"{SCR}/wavs"
os.makedirs(WAVDIR, exist_ok=True)
DBG = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{DBG}/real_cosyvoice_pkg")

from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file, sub_state_dict
from models.demos.audio.cosyvoice2.tt.hifigan.f0_predictor import TorchConvRNNF0PredictorRef
from models.demos.audio.cosyvoice2.tt.hifigan.generator import (
    TorchHiFTDecodeRef,
    TorchHiFTGeneratorInferenceRef,
    TtHiFTDecoder,
    TtHiFTGenerator,
)

SR, UP = 24000, 480
mel = torch.from_numpy(np.load(f"{DBG}/stage1_v4_mel.npy")).float()  # [1,T,80]
T = mel.shape[1]
L = T * UP
g1 = torch.Generator().manual_seed(1234)
g2 = torch.Generator().manual_seed(5678)
N1 = torch.randn(1, L, 9, generator=g1)
N2 = torch.randn(1, L, 9, generator=g2)

hift_sd = load_checkpoint_file("hift.pt")
decode_ref = TorchHiFTDecodeRef.from_checkpoint(hift_sd)
f0_ref = TorchConvRNNF0PredictorRef.from_checkpoint(sub_state_dict(hift_sd, "f0_predictor."))
hift_ref = TorchHiFTGeneratorInferenceRef(
    decode_ref, f0_ref, hift_sd["m_source.l_linear.weight"], hift_sd["m_source.l_linear.bias"]
)

out = {}


def wav1d(x):
    return x.detach().reshape(-1).float()


# ---- OFFICIAL (real class), noise patched to N1 -------------------------------------------------
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor
from cosyvoice.hifigan.generator import HiFTGenerator

off = HiFTGenerator(
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
    resblock_dilation_sizes=[[1, 3, 5]] * 3,
    source_resblock_kernel_sizes=[7, 7, 11],
    source_resblock_dilation_sizes=[[1, 3, 5]] * 3,
    lrelu_slope=0.1,
    audio_limit=0.99,
    f0_predictor=ConvRNNF0Predictor(num_class=1, in_channels=80, cond_channels=512),
)
missing, unexpected = off.load_state_dict(hift_sd, strict=False)
print("official load: missing", len(missing), "unexpected", len(unexpected))
off.eval()
_orig_randn_like = torch.randn_like


def _patched(t, *a, **k):
    return N1.clone() if tuple(t.shape) == tuple(N1.shape) else _orig_randn_like(t, *a, **k)


torch.randn_like = _patched
with torch.no_grad():
    out["OFFICIAL"], _ = off.inference(speech_feat=mel.transpose(1, 2).contiguous())
torch.randn_like = _orig_randn_like
out["OFFICIAL"] = wav1d(out["OFFICIAL"])

# ---- TORCH reference (torch f0), N1 and N2 ------------------------------------------------------
with torch.no_grad():
    out["TORCH"] = wav1d(hift_ref.inference(mel, sine_noise=N1))
    out["TORCH_N2"] = wav1d(hift_ref.inference(mel, sine_noise=N2))
    f0_torch = f0_ref(mel.transpose(1, 2)).reshape(1, -1).clone()  # [1,T] Hz

# ---- device runs -------------------------------------------------------------------------------
dev = ttnn.open_device(device_id=0, l1_small_size=65536)
try:
    dec = TtHiFTDecoder(dev, decode_ref, dtype=ttnn.float32)
    tt_gen = TtHiFTGenerator(dev, hift_ref, dec, dtype=ttnn.float32)
    mel_dev = ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    # TT_OWN, capturing the device f0 that this very run produced
    real_pred = tt_gen.f0_predictor
    captured = {}

    class Recorder:
        def __call__(self, m, frames, batch_size=1):
            r = real_pred(m, frames, batch_size)
            captured["f0"] = ttnn.to_torch(r).reshape(1, -1).float().clone()
            return r

    tt_gen.f0_predictor = Recorder()
    out["TT_OWN"] = wav1d(ttnn.to_torch(tt_gen.inference(mel_dev, T, 1, sine_noise=N1)))
    f0_tt = captured["f0"]

    # A: TT generator, torch f0 injected
    class Inject:
        def __init__(self, f0):
            self.f0 = f0

        def __call__(self, m, frames, batch_size=1):
            return ttnn.from_torch(self.f0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    tt_gen.f0_predictor = Inject(f0_torch)
    out["A_TT_torchf0"] = wav1d(ttnn.to_torch(tt_gen.inference(mel_dev, T, 1, sine_noise=N1)))
finally:
    ttnn.close_device(dev)

# B: torch reference with the TT device f0 injected
hift_ref.f0_predictor_ref = lambda mel_cf: f0_tt
with torch.no_grad():
    out["B_torch_ttf0"] = wav1d(hift_ref.inference(mel, sine_noise=N1))

# ---- save wavs ---------------------------------------------------------------------------------
for k, v in out.items():
    wavfile.write(f"{WAVDIR}/{k}.wav", SR, (np.clip(v.numpy(), -1, 1) * 32767).astype(np.int16))
np.save(f"{SCR}/f0_torch_gen.npy", f0_torch.numpy())
np.save(f"{SCR}/f0_tt_gen.npy", f0_tt.numpy())
print("f0 in-generator TT vs torch: max|d|", (f0_tt - f0_torch).abs().max().item())

# ---- metrics -----------------------------------------------------------------------------------
import librosa

melfb = torch.from_numpy(librosa.filters.mel(sr=SR, n_fft=1024, n_mels=80)).float()
win = torch.hann_window(1024)


def logmel(x):
    sp = torch.stft(x, 1024, hop_length=UP, win_length=1024, window=win, center=True, return_complex=True).abs() ** 2
    return 10 * torch.log10((melfb @ sp).clamp(min=1e-10))[:, :T]  # [80, T]


ref = out["TORCH"]
LM = {k: logmel(v[:L]) for k, v in out.items()}
df0 = (f0_tt - f0_torch).abs().reshape(-1).numpy()
voiced = (f0_torch.reshape(-1) > 10).numpy()


def pcc(a, b):
    return float(np.corrcoef(a.numpy()[:L], b.numpy()[:L])[0, 1])


print(
    f"\n{'run':14s} {'LSD dB (all)':>13s} {'voiced':>8s} {'unvoiced':>9s} {'p95 frame':>10s} {'max frame':>10s} {'sample PCC':>11s}"
)
per_frame = {}
for k in out:
    if k == "TORCH":
        continue
    d = (LM[k] - LM["TORCH"]).abs().mean(0).numpy()  # per-frame dB
    per_frame[k] = d
    print(
        f"{k:14s} {d.mean():13.3f} {d[voiced].mean():8.3f} {d[~voiced].mean():9.3f} {np.percentile(d,95):10.3f} {d.max():10.3f} {pcc(out[k], ref):11.4f}"
    )

# do the worst frames line up with the F0 error?
from scipy.stats import spearmanr

for k in ("TT_OWN", "B_torch_ttf0"):
    d = per_frame[k]
    r = spearmanr(d[voiced], df0[voiced]).correlation
    top = np.argsort(-d)[:8]
    print(f"\n[{k}] Spearman(per-frame LSD, |df0|) over voiced frames = {r:.3f}")
    print(
        "  worst-8 frames (frame, LSD dB, |df0| Hz, f0_torch):",
        [(int(i), round(float(d[i]), 1), round(float(df0[i]), 2), round(float(f0_torch[0, i]), 1)) for i in top],
    )
np.save(f"{SCR}/per_frame_lsd.npy", np.stack([per_frame[k] for k in per_frame]))
print("\nper-frame rows:", list(per_frame))
print("wavs in", WAVDIR)
