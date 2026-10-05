# Per-layer and per-operation PCC of the Tenstorrent Gemma-4-26B-A4B, decode path only, against the Hugging Face
# fp32 reference from hf_per_layer_ref.py. Same model setup as tt_decode_only_accuracy.py (Gemma4Generator, paged KV,
# untraced): every token from position 0 to 511 goes through decode one at a time (teacher forced). This is the path
# the fp32-intermediate switches (GEMMA4_FP32_ACTIVATIONS etc.) cover; prefill is never used.
#   decode_accumulated: normal run, each layer gets the chip's own previous output
#   decode_isolated:    each layer's input is replaced by HF's exact input for that layer and position
# Per layer: PCC (mean over the 512 positions) of layer input, attention output, shared-MLP output, expert output,
# layer output; and how many of the router's 8 chosen experts match HF.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc, json, math, os, sys, time
from pathlib import Path
import torch, ttnn

LABEL = sys.argv[1] if len(sys.argv) > 1 else "tree"
D = Path(f"{DATA}")
MODEL = f"{MODELS}/gemma-4-26B-A4B-it"
MAX_SEQ_LEN, PAGE_BLOCK_SIZE = 1024, 32
ref = torch.load(D / "hf_per_layer_ref_512.pt")
ids = ref["ids"]
S = ids.shape[0]

from models.demos.gemma4.tt import layer as L, router as R, shared_mlp as SM
from models.demos.gemma4.tt.attention import Gemma4Attention
from models.demos.gemma4.tt.experts import Gemma4Experts

MODE = {"isolated": False, "pos": 0}
cur = {"layer": None}
rec = {}


def host(t):
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return x.reshape(-1, x.shape[-1])


def put(kind, t):
    rec.setdefault(kind, {})[cur["layer"]] = host(t)[:1]  # row 0 = the one user


_layer_call = L.Gemma4DecoderLayer.__call__
def layer_call(self, hidden_states, *a, **k):
    cur["layer"] = self.layer_idx
    if MODE["isolated"]:
        shape = list(hidden_states.shape)
        x = torch.zeros(shape)
        x.reshape(-1, shape[-1])[0] = ref["in"][self.layer_idx][MODE["pos"]]  # HF's exact input to this layer
        x = x if hidden_states.dtype == ttnn.float32 else x.to(torch.bfloat16)
        new = ttnn.from_torch(x, device=self_mesh[0], dtype=hidden_states.dtype, layout=hidden_states.layout,
                              mesh_mapper=ttnn.ReplicateTensorToMesh(self_mesh[0]), memory_config=hidden_states.memory_config())
        hidden_states = new
    put("in", hidden_states)
    out = _layer_call(self, hidden_states, *a, **k)
    put("out", out)
    return out
L.Gemma4DecoderLayer.__call__ = layer_call

for cls, kind in ((Gemma4Attention, "attn"), (SM.SharedMLP, "mlp"), (Gemma4Experts, "experts")):
    orig = cls.__call__
    def wrapped(self, *a, _orig=orig, _kind=kind, **k):
        out = _orig(self, *a, **k)
        put(_kind, out)
        return out
    cls.__call__ = wrapped

_router_call = R.Gemma4Router.__call__
def router_call(self, hidden_states):
    out = _router_call(self, hidden_states)
    put("routing", out)
    return out
R.Gemma4Router.__call__ = router_call


def pcc_rows(a, b):
    a, b = a.double(), b.double()
    a = a - a.mean(-1, keepdim=True); b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)


def score(title):
    out = []
    for i in range(len(rec["out"])):
        r = {"layer": i}
        for kind in ("in", "attn", "mlp", "experts", "out"):
            p = pcc_rows(rec[kind][i], ref[kind][i])
            r[kind] = round(p.mean().item(), 5)
            r[kind + "_min"] = round(p.min().item(), 4)
        tt_sets = (rec["routing"][i][:, :128] != 0)
        hf_sets = torch.zeros_like(tt_sets); hf_sets.scatter_(1, ref["idx"][i], True)
        overlap = (tt_sets & hf_sets).sum(-1).float()
        r["experts_same_8"] = round((overlap == 8).float().mean().item() * 100, 1)
        r["experts_overlap_of_8"] = round(overlap.mean().item(), 3)
        out.append(r)
        print(f"{title} " + json.dumps(r), flush=True)
    return out


os.environ["TT_CACHE_PATH"] = str(D / f"tt_cache_perlayer_dec_{LABEL}")
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
self_mesh = [md]
results = {}
generator = kv = None
try:
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
    from models.tt_transformers.tt.common import PagedAttentionConfig

    pac = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE))
    lc = resolve_gemma4_demo_long_context(MAX_SEQ_LEN, md, MODEL, paged_attention=True)
    generator, kv, _ = Gemma4Generator.from_pretrained(mesh_device=md, model_path=MODEL, max_batch_size=1, max_seq_len=MAX_SEQ_LEN,
                                                       paged_attention_config=pac, bounded_sliding_kv_cache=lc["bounded_sliding"])
    gc.collect()
    page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, pac.max_num_blocks)
    for mode, iso in (("decode_accumulated", False), ("decode_isolated", True)):
        MODE["isolated"] = iso
        steps = []
        t = time.time()
        for pos in range(S):
            rec.clear(); MODE["pos"] = pos
            generator.decode_forward(ids[pos].reshape(1, 1).long(), torch.tensor([pos], dtype=torch.int64), page_table=page_table,
                                     kv_cache=kv, enable_trace=False, sampling_params=None)
            steps.append({k: dict(v) for k, v in rec.items()})
            if pos % 128 == 127:
                print(f"{mode} position {pos} at {time.time()-t:.0f}s", flush=True)
        rec.clear()
        for kind in steps[0]:
            rec[kind] = {i: torch.cat([s[kind][i] for s in steps]) for i in steps[0][kind]}
        steps = None
        print(f"{mode} ran in {time.time()-t:.0f}s", flush=True)
        results[mode] = score(mode.upper())
        json.dump(results, open(D / f"tt_per_layer_pcc_decode_{LABEL}.json", "w"), indent=1)
    print("DONE", flush=True)
finally:
    generator = kv = None; gc.collect()
    ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
