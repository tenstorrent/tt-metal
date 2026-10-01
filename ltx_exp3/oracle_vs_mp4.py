"""Decode an audio latent with the torch oracle and compare to the audio track of an mp4 (same gen)."""
import sys, os, numpy as np, torch, av

sys.path.insert(0, "/home/rsalman/tt-metal")
from models.tt_dit.tests.models.ltx.test_audio_ltx import _decode_audio_reference

lat_p, mp4_p = sys.argv[1], sys.argv[2]
lat = torch.load(lat_p).float()
ref = _decode_audio_reference(os.environ["LTX_CHECKPOINT"], lat, 153, fps=25.0).waveform.float()
au = np.concatenate([f.to_ndarray() for f in av.open(mp4_p).decode(audio=0)], axis=-1).astype(np.float64)
a = ref.numpy().astype(np.float64)
# AAC adds encoder delay (~1-2k samples): align by cross-correlation of the mono mixes before comparing.
ma, mb = a.mean(0), au.mean(0)
L = min(len(ma), len(mb))
ma, mb = ma[:L], mb[:L]
xc = np.fft.irfft(np.fft.rfft(mb, 2 * L) * np.conj(np.fft.rfft(ma, 2 * L)))
lags = np.r_[np.arange(0, 6000), np.arange(-6000, 0)]
lag = int(lags[np.argmax(np.r_[xc[:6000], xc[-6000:]])])
if lag >= 0:
    a2, b2 = a[..., : L - lag], au[..., lag:L]
else:
    a2, b2 = a[..., -lag:L], au[..., : L + lag]
pcc = np.corrcoef(a2.ravel(), b2.ravel())[0, 1]
print(
    f"oracle({os.path.basename(lat_p)}) vs mp4({os.path.basename(mp4_p)}): lag={lag} PCC={pcc:.4f} rms {np.sqrt((a2**2).mean()):.4f}/{np.sqrt((b2**2).mean()):.4f}"
)
