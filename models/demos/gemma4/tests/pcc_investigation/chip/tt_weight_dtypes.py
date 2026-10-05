# Build the model exactly as tt_token_accuracy.py does and print the on-device dtype of key weights.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc, math, os
from pathlib import Path
import torch, ttnn
from models.tt_transformers.tests.optimizer_weight_cache import RunCache
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.demos.gemma4.tt.precision import Gemma4Precision
MODEL = f"{MODELS}/gemma-4-26B-A4B-it"
print("DTYPE overrides file says:", Gemma4Precision.load(MODEL, (1, 4)), flush=True)
cache = RunCache(Path(f"{REPO}/generated/optimizer_cache"), "gemma4-dtypes-"); os.environ["TT_CACHE_PATH"] = cache.path
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
try:
    pac = PagedAttentionConfig(block_size=32, max_num_blocks=32)
    lc = resolve_gemma4_demo_long_context(1024, md, MODEL, paged_attention=True)
    gen, kv, _ = cache.build(lambda: Gemma4Generator.from_pretrained(mesh_device=md, model_path=MODEL, max_batch_size=1, max_seq_len=1024,
        paged_attention_config=pac, bounded_sliding_kv_cache=lc["bounded_sliding"]), loaders=[(Gemma4ModelArgs, "load_state_dict")])
    m = gen.model[0]; L = m.layers[0]
    def dt(x):
        x = getattr(x, "weight", x)
        return str(getattr(x, "dtype", type(x).__name__))
    w = L.self_attn.weights
    print("DTYPE attention wqkv:", dt(w.wqkv), "o_proj:", dt(w.o_proj), flush=True)
    print("DTYPE experts gate_proj:", dt(L.moe.experts.weights.gate_proj) if hasattr(L.moe, "experts") else "?", flush=True)
    print("DTYPE router proj:", dt(L.moe.router.proj_weight) if hasattr(L.moe, "router") else "?", "lm_head:", dt(m.lm_head_weight), "embedding:", dt(m.embedding_weight), flush=True)
finally:
    gen = kv = None; gc.collect(); ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED); cache.cleanup()
