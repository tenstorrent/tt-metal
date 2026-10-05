# Build Gemma4 with N layers, decode 16 book tokens from position 0, save logits. Run once per mode.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc, math, os, sys
from pathlib import Path
import torch, ttnn
from models.tt_transformers.tests.optimizer_weight_cache import RunCache
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.tt_transformers.tt.common import PagedAttentionConfig
N, label = int(sys.argv[1]), sys.argv[2]
MODEL = f"{MODELS}/gemma-4-26B-A4B-it"; D = Path(f"{DATA}")
NPOS = int(sys.argv[3]); ids = torch.load(D / "gemma-4-26B-A4B-it.refpt")["reference_tokens"][0, :NPOS]
cache = RunCache(Path(f"{REPO}/generated/optimizer_cache"), f"gemma4-bisect-{label}-"); os.environ["TT_CACHE_PATH"] = cache.path
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
try:
    pac = PagedAttentionConfig(block_size=32, max_num_blocks=32)
    lc = resolve_gemma4_demo_long_context(1024, md, MODEL, paged_attention=True)
    gen, kv, _ = cache.build(lambda: Gemma4Generator.from_pretrained(mesh_device=md, model_path=MODEL, max_batch_size=1, max_seq_len=1024,
        num_layers=N, paged_attention_config=pac, bounded_sliding_kv_cache=lc["bounded_sliding"]), loaders=[(Gemma4ModelArgs, "load_state_dict")])
    pt = torch.arange(32, dtype=torch.int32).reshape(1, 32)
    out = []
    for pos in range(NPOS):
        o = gen.decode_forward(ids[pos].reshape(1, 1).long(), torch.tensor([pos], dtype=torch.int64), page_table=pt, kv_cache=kv, enable_trace=False, sampling_params=None, reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False)
        o = o[0] if isinstance(o, (tuple, list)) else o
        out.append(o.float().reshape(-1, o.shape[-1])[0, :262144])
    torch.save(torch.stack(out), D / f"bisect-{N}-{label}-{NPOS}.pt"); print("BISECT saved", N, label)
finally:
    gen = kv = None; gc.collect(); ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED); cache.cleanup()
