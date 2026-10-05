import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import jax, jax.numpy as jnp, numpy as np, torch
from gemma import gm
from gemma.gm.ckpts import _checkpoint as C
_orig = C._as_shape_dtype_struct
C._as_shape_dtype_struct = lambda t: jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, jnp.bfloat16, sharding=s.sharding) if jnp.issubdtype(s.dtype, jnp.floating) else s, _orig(t))
params = gm.ckpts.load_params(f"{MODELS}/gemma4-26b-a4b-it-google", text_only=True)
model = gm.nn.Gemma4_26B_A4B()
d = torch.load(f"{DATA}/chat_ids.pt")
lg = np.asarray(model.apply({"params": params}, tokens=jnp.asarray(d["ids"].numpy()[None], dtype=jnp.int32), return_last_only=False).logits[0], dtype=np.float32)
g = torch.from_numpy(lg); h = torch.load(f"{DATA}/chat_logits_hf.pt")
ids = d["ids"]; p0 = d["prompt_len"] - 1
s = slice(p0, len(ids) - 1)
pcc = torch.stack([torch.corrcoef(torch.stack((g[i], h[i])))[0, 1] for i in range(s.start, s.stop)])
real = ids[s.start + 1: s.stop + 1]
print(f"CHAT google vs HF over {len(pcc)} answer positions: mean PCC {pcc.mean():.4f}, min {pcc.min():.4f}, <0.99 {(pcc<0.99).sum().item()}, same top word {(g[s].argmax(-1)==h[s].argmax(-1)).float().mean()*100:.1f}%")
print(f"CHAT real answer word guessed: google {(g[s].argmax(-1)==real).float().mean()*100:.1f}%, HF {(h[s].argmax(-1)==real).float().mean()*100:.1f}%")
sampler = gm.text.ChatSampler(model=model, params=params, multi_turn=False)
print("CHAT google sampler answer:", repr(sampler.chat("What's the capital of France?", max_new_tokens=24)))
