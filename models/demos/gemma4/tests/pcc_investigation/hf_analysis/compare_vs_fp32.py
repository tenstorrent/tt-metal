# Every bf16 implementation of Gemma-4-26B-A4B against the fp32 Hugging Face run, same tokens, same positions.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import os, torch
D = f"{DATA}"
R = f"{D}/gemma-4-26B-A4B-it.refpt"
ref_ids = torch.load(R)["reference_tokens"][0, :1023]
fp32 = torch.load(f"{D}/book_logits_hf_fp32.pt").float()
s = slice(511, 1011)
cands = [("HF 5.12.1 bf16 sdpa (the reference)", R + ".logits.pt"),
         ("HF 5.12.1 bf16 eager", f"{D}/book_logits_hf_eager.pt"),
         ("HF 5.18.0 bf16 sdpa", R + ".logits-5.18.pt"),
         ("Google JAX bf16", R + ".logits-google.pt"),
         ("Tenstorrent chips, unmodified", f"{D}/tt-unmodified-logits.pt"),
         ("Tenstorrent chips, optimized", f"{D}/tt-optimized-logits.pt")]
real = ref_ids[512:1012]
print(f"{'implementation':40s} {'mean PCC vs fp32':>16s} {'<0.99':>6s} {'same top word':>13s} {'guesses real word':>17s}")
print(f"{'HF fp32 itself':40s} {'1.0000':>16s} {'0':>6s} {'100.0%':>13s} {(fp32[s].argmax(-1)==real).float().mean()*100:16.1f}%")
for name, path in cands:
    if not os.path.exists(path):
        print(f"{name:40s} (not run yet)"); continue
    x = torch.load(path)
    x = (x["logits"] if isinstance(x, dict) else x).float()
    xs = x[-500:] if x.shape[0] == 500 else x[s]   # chip files hold exactly positions 511-1010
    p = torch.stack([torch.corrcoef(torch.stack((xs[i], fp32[s][i])))[0, 1] for i in range(500)])
    print(f"{name:40s} {p.mean():16.4f} {(p<0.99).sum().item():6d} {(xs.argmax(-1)==fp32[s].argmax(-1)).float().mean()*100:12.1f}% {(xs.argmax(-1)==real).float().mean()*100:16.1f}%")
