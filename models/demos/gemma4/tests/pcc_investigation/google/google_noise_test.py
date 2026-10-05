# Is Google's own Gemma 4 implementation as sensitive as Hugging Face's? Same 0.2% embedding nudge.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import collections, numpy as np, torch, jax, jax.numpy as jnp
from gemma import gm
from gemma.gm.ckpts import _checkpoint as C
from gemma.gm.nn.gemma4 import _modules as GM
_orig = C._as_shape_dtype_struct
C._as_shape_dtype_struct = lambda t: jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, jnp.bfloat16, sharding=s.sharding) if jnp.issubdtype(s.dtype, jnp.floating) else s, _orig(t))
params = gm.ckpts.load_params(f"{MODELS}/gemma4-26b-a4b-it-google", text_only=True)
NOISE = {"on": False}
_enc = GM.Embedder.encode
def enc(self, x):
    y = _enc(self, x)
    if NOISE["on"]:
        g = np.random.RandomState(0).randn(*y.shape).astype(np.float32)
        y = (y.astype(jnp.float32) * (1 + 2.0 ** -9 * jnp.asarray(g))).astype(y.dtype)
    return y
GM.Embedder.encode = enc
D = f"{DATA}"
ids = torch.load(f"{D}/gemma-4-26B-A4B-it.refpt")["reference_tokens"][0, :1023].numpy()
model = gm.nn.Gemma4_26B_A4B()
def run():
    return torch.from_numpy(np.asarray(model.apply({"params": params}, tokens=jnp.asarray(ids[None], dtype=jnp.int32), return_last_only=False).logits[0], dtype=np.float32))
a = run(); NOISE["on"] = True; c = run()
s = slice(511, 1011)
def cmp(x, y):
    p = torch.stack([torch.corrcoef(torch.stack((x[i], y[i])))[0, 1] for i in range(511, 1011)])
    return f"mean PCC {p.mean():.5f}, <0.99 {(p<0.99).sum().item()}/500, same top word {(x[s].argmax(-1)==y[s].argmax(-1)).float().mean()*100:.1f}%"
print("GOOGLE reference vs 0.2% embedding nudge:", cmp(a, c), flush=True)
real = torch.from_numpy(ids[1:]).long()
print(f"GOOGLE guesses real next word {(a[:-1].argmax(-1)==real).float().mean()*100:.1f}%; dtype in embedder output check done", flush=True)
