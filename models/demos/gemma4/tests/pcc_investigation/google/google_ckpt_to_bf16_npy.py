# Convert Google's Gemma 4 26B-A4B orbax checkpoint (fp32) to bf16 .npy files, a few top-level entries
# at a time (orbax partial_restore), so peak memory stays far below the ~100 GB of a full load.
# The output tree is checked against the model's expected parameter tree (shapes) before saving.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import os, sys, time
import numpy as np, jax, jax.numpy as jnp, ml_dtypes
import orbax.checkpoint as ocp
from gemma import gm

SRC = f"{MODELS}/gemma4-26b-a4b-it-google"
DST = f"{MODELS}/gemma4-26b-a4b-it-google-bf16npy"
os.makedirs(DST, exist_ok=True)

ck = ocp.PyTreeCheckpointer()
meta = ck.metadata(SRC)
meta = getattr(meta, "item_metadata", None) or meta
meta = getattr(meta, "tree", meta)

# expected text-only parameter tree of the model (abstract: no memory)
model = gm.nn.Gemma4_26B_A4B()
expected = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), tokens=jnp.zeros((1, 8), jnp.int32)))["params"]


def flat(tree, prefix=()):
    for k, v in tree.items():
        if isinstance(v, dict):
            yield from flat(v, prefix + (k,))
        else:
            yield prefix + (k,), v


exp_flat = dict(flat(expected))
ckpt_flat = dict(flat(meta))
missing = [k for k in exp_flat if k not in ckpt_flat]
print(f"expected leaves {len(exp_flat)}, checkpoint leaves {len(ckpt_flat)}, expected-but-missing {len(missing)} {missing[:3]}", flush=True)
bad = [(k, tuple(exp_flat[k].shape), tuple(ckpt_flat[k].shape)) for k in exp_flat if k in ckpt_flat and tuple(exp_flat[k].shape) != tuple(ckpt_flat[k].shape)]
print(f"shape mismatches {len(bad)} {bad[:3]}", flush=True)
# names differ (e.g. .../gating_einsum/w on disk); gemma remaps after restore, so save checkpoint names as-is

top = sorted({k[0] for k in ckpt_flat})
chunks = [top[i:i + 3] for i in range(0, len(top), 3)]
for chunk in chunks:
    t = time.time()
    sub = {}
    for k, v in ckpt_flat.items():
        if k[0] in chunk:
            d = sub
            for p in k[:-1]:
                d = d.setdefault(p, {})
            d[k[-1]] = k
    dev = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    is_key = lambda x: isinstance(x, tuple)
    dt = lambda k: jnp.bfloat16 if jnp.issubdtype(ckpt_flat[k].dtype, jnp.floating) else ckpt_flat[k].dtype
    item = jax.tree.map(lambda k: jax.ShapeDtypeStruct(tuple(ckpt_flat[k].shape), dt(k), sharding=dev), sub, is_leaf=is_key)
    rargs = jax.tree.map(lambda k: ocp.ArrayRestoreArgs(dtype=dt(k), sharding=dev), sub, is_leaf=is_key)
    out = ck.restore(SRC, args=ocp.args.PyTreeRestore(item=item, restore_args=rargs, partial_restore=True))
    n = 0
    for k, arr in flat(out):
        arr = np.asarray(arr)
        assert arr.shape == tuple(ckpt_flat[k].shape), (k, arr.shape)
        if np.issubdtype(arr.dtype, np.floating) or arr.dtype == ml_dtypes.bfloat16:
            arr = arr.astype(ml_dtypes.bfloat16)
        np.save(os.path.join(DST, ".".join(k) + ".npy"), arr)
        n += 1
    del out
    print(f"chunk {chunk}: {n} arrays saved in {time.time()-t:.0f}s", flush=True)
print("CONVERTED", len(ckpt_flat), "arrays to", DST, flush=True)
