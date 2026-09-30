"""ltx-core audio decode of the reference s2 audio latent with the mel decoder in bf16 (their default) and fp32.
Run in ~/ltx2-ref-venv. Compares both against our torch oracle (oracle_audio.pt) and each other."""
import sys, os, torch

ref = sys.argv[1]
from ltx_pipelines.utils.blocks import AudioDecoder
from ltx_pipelines.utils.model_paths import ModelPaths
import ltx_pipelines.utils.blocks as blocks

_orig = blocks.vae_decode_audio
blocks.vae_decode_audio = lambda latent, decoder, vocoder: _orig(latent, decoder, vocoder.float())
ckpt = "/home/rsalman/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/5948be4ced3a4493d1f836df64378ff136ddb770/ltx-2.3-22b-distilled-1.1.safetensors"
tok = torch.load(os.path.join(ref, "s2_audio.pt")).detach().float()  # (1, T, c*f)
lat = tok.reshape(1, tok.shape[1], 8, 16).permute(0, 2, 1, 3).contiguous()  # b t (c f) -> b c t f


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


out = {}
with torch.inference_mode():
    for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        dec = AudioDecoder(ModelPaths.from_monolith(ckpt).audio_vae(), dt, torch.device("cpu"))
        a = dec(lat.to(dt))
        out[name] = a.waveform.float()
        print(name, tuple(a.waveform.shape), a.sampling_rate)
        torch.save(
            {"waveform": out[name], "sampling_rate": a.sampling_rate}, os.path.join(ref, f"ltxcore_audio_dec{name}.pt")
        )
ours = torch.load(os.path.join(ref, "oracle_audio.pt"))["waveform"].float()
theirs = torch.load(os.path.join(ref, "audio.pt"))["waveform"].float()
print(f"ltx-core bf16 (rerun) vs pipeline audio.pt : PCC={pcc(out['bf16'], theirs):.5f}")
print(f"ltx-core bf16 vs ltx-core fp32 decoder    : PCC={pcc(out['bf16'], out['fp32']):.5f}")
print(f"our oracle vs ltx-core bf16 decoder       : PCC={pcc(ours, out['bf16']):.5f}")
print(f"our oracle vs ltx-core fp32 decoder       : PCC={pcc(ours, out['fp32']):.5f}")
