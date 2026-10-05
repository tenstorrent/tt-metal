import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import jax, jax.numpy as jnp, numpy as np, torch
from gemma import gm
from gemma.gm.ckpts import _checkpoint as C
from transformers import AutoTokenizer
from safetensors import safe_open
import glob, json
CKPT = f"{MODELS}/gemma4-26b-a4b-it-google"; HFP = f"{MODELS}/gemma-4-26B-A4B-it"
_orig = C._as_shape_dtype_struct
C._as_shape_dtype_struct = lambda t: jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, jnp.bfloat16, sharding=s.sharding) if jnp.issubdtype(s.dtype, jnp.floating) else s, _orig(t))
params = gm.ckpts.load_params(CKPT, text_only=True)
# 1. tokenizer ids
htok = AutoTokenizer.from_pretrained(HFP)
text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of"
try:
    gtok = gm.text.Gemma4Tokenizer(); gids = gtok.encode(text); print("TOKENS google", gids[:12], "hf", htok.encode(text)[:12], "equal:", gids == htok.encode(text))
except Exception as e:
    print("TOKENS google tokenizer unavailable:", type(e).__name__, str(e)[:120])
# 2. weights: per-expert scale and router scale of layer 0, and embedding row 818
idx = json.load(open(f"{HFP}/model.safetensors.index.json"))["weight_map"]
def hf(name):
    with safe_open(f"{HFP}/{idx[name]}", "pt") as f: return f.get_tensor(name).float()
L0 = params["layer_0"]["mlp"]
pre = "model.language_model.layers.0.router."
for gk, hk in (("per_expert_scale", pre + "per_expert_scale"), ("router_scale", pre + "scale")):
    g = torch.from_numpy(np.asarray(L0[gk], dtype=np.float32)).reshape(-1); h = hf(hk).reshape(-1)
    print(f"WEIGHT layer0 {gk}: shapes {tuple(g.shape)} {tuple(h.shape)}, max abs diff {(g-h).abs().max():.4g}, google[:4] {g[:4].tolist()} hf[:4] {h[:4].tolist()}")
ge = torch.from_numpy(np.asarray(params["embedder"]["input_embedding"][818], dtype=np.float32)); he = hf("model.language_model.embed_tokens.weight")[818]
print(f"WEIGHT embedding row 818: max abs diff {(ge-he).abs().max():.4g}")
# 3. famous sentence
ids = [2] + htok.encode(text)
lg = np.asarray(gm.nn.Gemma4_26B_A4B().apply({"params": params}, tokens=jnp.asarray([ids], dtype=jnp.int32), return_last_only=False).logits[0], dtype=np.float32)
hits = sum(int(lg[i-1].argmax()) == ids[i] for i in range(1, len(ids)))
print(f"SENTENCE google next-word top-1 {hits}/{len(ids)-1}; after full text:", [htok.decode([int(t)]) for t in np.argsort(-lg[-1])[:5]])
