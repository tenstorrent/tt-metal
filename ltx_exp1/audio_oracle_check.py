"""CPU-only cross-check: our torch audio-decode oracle (diffusers-based, test_audio_ltx._decode_audio_reference)
vs Lightricks' ltx-core decode (ref_driver audio.pt), both fed the SAME reference s2 audio latent.
Run in the tt-metal venv: python ltx_exp1/audio_oracle_check.py ltx_exp1/ref_153f25_seed10"""
import sys, os, json, torch

sys.path.insert(0, "/home/rsalman/tt-metal")
from models.tt_dit.tests.models.ltx.test_audio_ltx import _decode_audio_reference, _psnr

ref = sys.argv[1]
meta = json.load(open(os.path.join(ref, "meta.json")))
lat = torch.load(os.path.join(ref, "s2_audio.pt")).detach().float()
ours = _decode_audio_reference(os.environ["LTX_CHECKPOINT"], lat, meta["num_frames"], fps=meta["fps"])
torch.save(
    {"waveform": ours.waveform.float().cpu(), "sampling_rate": ours.sampling_rate}, os.path.join(ref, "oracle_audio.pt")
)
theirs = torch.load(os.path.join(ref, "audio.pt"))["waveform"].float()
a, b = ours.waveform.float(), theirs
if a.dim() == 1:
    a = a.unsqueeze(0)
if b.dim() == 1:
    b = b.unsqueeze(0)
n = min(a.shape[-1], b.shape[-1])
print(
    f"ours {tuple(ours.waveform.shape)} @ {ours.sampling_rate}  theirs {tuple(theirs.shape)} @ {torch.load(os.path.join(ref,'audio.pt'))['sampling_rate']}"
)
a, b = a[..., :n], b[..., :n]
pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
print(
    f"oracle vs ltx-core decode: PCC={pcc:.5f}  PSNR={_psnr(b, a):.2f} dB  rms {a.pow(2).mean().sqrt():.4f}/{b.pow(2).mean().sqrt():.4f}"
)
