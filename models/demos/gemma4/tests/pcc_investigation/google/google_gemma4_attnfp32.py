# Google DeepMind's own Gemma 4 implementation (gemma 4.0.1, JAX, CPU) on the same 1023 book tokens
# the Hugging Face runs used. Weights: Google's checkpoint gs://gemma-data/checkpoints/gemma4-26b-a4b-it,
# restored as bfloat16. Compared against the saved Hugging Face Transformers 5.12.1 (sdpa) logits.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import collections
import jax
import jax.numpy as jnp
import numpy as np
import torch
from gemma import gm
from gemma.gm.ckpts import _checkpoint as C

CKPT = f"{MODELS}/gemma4-26b-a4b-it-google"
R = f"{DATA}/gemma-4-26B-A4B-it.refpt"

# Restore floating-point weights as bfloat16 (the HF runs are bf16; fp32 would need ~96 GB of RAM).
_orig = C._as_shape_dtype_struct
def _as_bf16(tree):
    out = _orig(tree)
    return jax.tree.map(
        lambda s: jax.ShapeDtypeStruct(s.shape, jnp.bfloat16, sharding=s.sharding)
        if jnp.issubdtype(s.dtype, jnp.floating) else s, out)
C._as_shape_dtype_struct = _as_bf16


# Self-test: do ONLY the attention softmax (gemma4/_modules.py) in fp32, then round back to bf16,
# like Hugging Face's eager attention. Every other softmax (e.g. the router) is unchanged.
import inspect
_softmax = jax.nn.softmax
_hits=[0]; _miss=[]
def _softmax_attn_fp32(x, *a, **k):
    f = inspect.currentframe().f_back
    from_attention = False
    for _ in range(8):
        if f is None: break
        if f.f_code.co_filename.endswith("gemma4/_modules.py"): from_attention = True; break
        f = f.f_back
    if from_attention and x.dtype == jnp.bfloat16:
        _hits[0] += 1
        return _softmax(x.astype(jnp.float32), *a, **k).astype(jnp.bfloat16)
    _miss.append((x.dtype, inspect.currentframe().f_back.f_code.co_filename.split("site-packages/")[-1]))
    return _softmax(x, *a, **k)
jax.nn.softmax = _softmax_attn_fp32

params = gm.ckpts.load_params(CKPT, text_only=True)
leaves = jax.tree.leaves(params)
print("restored", len(leaves), "arrays; dtypes", collections.Counter(str(x.dtype) for x in leaves), flush=True)

ids = torch.load(R)["reference_tokens"][0, :1023].numpy()
model = gm.nn.Gemma4_26B_A4B()
out = model.apply({"params": params}, tokens=jnp.asarray(ids[None, :], dtype=jnp.int32), return_last_only=False)
g = torch.from_numpy(np.asarray(out.logits[0], dtype=np.float32))  # [1023, vocab]
torch.save(g.to(torch.bfloat16), R + ".logits-google-attnfp32.pt")
import collections; print("GOOGLE patched attention softmax calls:", _hits[0], "other softmax calls:", collections.Counter(_miss).most_common(4))

hf = torch.load(R + ".logits.pt").float()
real = torch.from_numpy(ids[1:]).long()
s = slice(511, 1011)
pcc = torch.stack([torch.corrcoef(torch.stack((g[i], hf[i])))[0, 1] for i in range(1023)])
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{MODELS}/gemma-4-26B-A4B-it")
top = g[s].argmax(-1)
conf = torch.softmax(g[s], -1).max(-1).values
print(f"GOOGLE real next word guessed: {(g[:-1].argmax(-1) == real).float().mean()*100:.1f}% "
      f"(HF 5.12.1: {(hf[:-1].argmax(-1) == real).float().mean()*100:.1f}%)")
print(f"GOOGLE vs HF 5.12.1, positions 511-1010: mean PCC {pcc[s].mean():.4f}, <0.99 {(pcc[s]<0.99).sum().item()}/500, "
      f"same top word {(top == hf[s].argmax(-1)).float().mean()*100:.1f}%; all positions mean PCC {pcc.mean():.4f}")
print(f"GOOGLE mean top-1 probability {conf.mean():.3f}; most common top guesses:",
      [(tok.decode([t]), n) for t, n in collections.Counter(top.tolist()).most_common(6)])
