import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import resource, time, jax, jax.numpy as jnp, numpy as np
from gemma import gm
from gemma.gm.ckpts import _checkpoint as C
_orig = C._as_shape_dtype_struct
C._as_shape_dtype_struct = lambda t: jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, jnp.bfloat16, sharding=s.sharding) if jnp.issubdtype(s.dtype, jnp.floating) else s, _orig(t))
mx = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
cur = lambda: int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e9
t = time.time(); params = gm.ckpts.load_params(f"{MODELS}/gemma4-26b-a4b-it-google", text_only=True)
print(f"PROBE after load: current {cur():.1f} GB, peak {mx():.1f} GB, {time.time()-t:.0f}s", flush=True)
ids = np.arange(2, 2 + 64)[None].astype(np.int32)
out = gm.nn.Gemma4_26B_A4B().apply({"params": params}, tokens=jnp.asarray(ids), return_last_only=False)
print(f"PROBE after 64-token forward: current {cur():.1f} GB, peak {mx():.1f} GB", flush=True)
