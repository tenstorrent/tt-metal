# Is Google's own Gemma 4 implementation as sensitive as Hugging Face's? Same 0.2% embedding nudge.
# Weights: Google's checkpoint converted to bf16 .npy (google_ckpt_to_bf16_npy.py), remapped with gemma's own
# _CheckpointTree.as_nested(remove_mm=True) exactly as gm.ckpts.load_params does after restoring.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import glob, os, numpy as np, torch, jax, jax.numpy as jnp, ml_dtypes
from gemma import gm
from gemma.gm.ckpts import _checkpoint as C
from gemma.gm.nn.gemma4 import _modules as GM

SRC = f"{MODELS}/gemma4-26b-a4b-it-google-bf16npy"
D = f"{DATA}"
tree = {}
for f in sorted(glob.glob(f"{SRC}/*.npy")):
    keys = os.path.basename(f)[:-4].split(".")
    d = tree
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    a = np.load(f)
    if a.dtype.kind == 'V':  # bf16 saved by ml_dtypes reloads as raw 2-byte values
        a = a.view(ml_dtypes.bfloat16)
    d[keys[-1]] = jnp.asarray(a)
params = C._CheckpointTree(tree=tree).as_nested(remove_mm=True).tree
print("params leaves", len(jax.tree.leaves(params)), flush=True)

NOISE = {"on": False}
_enc = GM.Embedder.encode
def enc(self, x):
    y = _enc(self, x)
    if NOISE["on"]:
        g = np.random.RandomState(0).randn(*y.shape).astype(np.float32)
        y = (y.astype(jnp.float32) * (1 + 2.0 ** -9 * jnp.asarray(g))).astype(y.dtype)
    return y
GM.Embedder.encode = enc

NTOK = 512
ids = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][0, :NTOK].numpy()
model = gm.nn.Gemma4_26B_A4B()
def run():
    return torch.from_numpy(np.asarray(model.apply({"params": params}, tokens=jnp.asarray(ids[None], dtype=jnp.int32), return_last_only=False).logits[0], dtype=np.float32))
s = slice(NTOK // 2, NTOK)
def cmp(x, y):
    p = torch.stack([torch.corrcoef(torch.stack((x[i], y[i])))[0, 1] for i in range(NTOK // 2, NTOK)])
    return f"mean PCC {p.mean():.5f}, <0.99 {(p<0.99).sum().item()}/{NTOK//2}, same top word {(x[s].argmax(-1)==y[s].argmax(-1)).float().mean()*100:.1f}%"
a = run()
earlier = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt.logits-google.pt").float()[:NTOK]
print("VALIDATE this run vs earlier Google run (normal loader):", cmp(a, earlier), "| max abs diff", (a - earlier).abs().max().item(), flush=True)
NOISE["on"] = True
c = run()
print("GOOGLE reference vs 0.2% embedding nudge:", cmp(a, c), flush=True)
