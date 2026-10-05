# Per-layer and per-operation PCC of the Tenstorrent Gemma-4-26B-A4B (this tree) against the Hugging Face
# fp32 reference from hf_per_layer_ref.py, on the first 512 book tokens, QuietBox 2 (1x4).
#   A. prefill, accumulated: normal run, each layer gets the chip's own previous output
#   B. prefill, isolated:    each layer's input is replaced by HF's exact input for that layer
#   C. decode, accumulated:  prefill 480 tokens, then decode tokens 480..511 one at a time (teacher forced)
# Per layer: PCC (mean over positions) of layer input, attention output, shared-MLP output, expert output,
# layer output; and how many of the router's 8 chosen experts match HF.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc, json, os, sys, time
from pathlib import Path
import torch, ttnn

LABEL = sys.argv[1] if len(sys.argv) > 1 else "tree"
D = Path(f"{DATA}")
MODEL = f"{MODELS}/gemma-4-26B-A4B-it"
ref = torch.load(D / "hf_per_layer_ref_512.pt")
ids = ref["ids"]
S = ids.shape[0]

from models.demos.gemma4.tt import layer as L, router as R, shared_mlp as SM
from models.demos.gemma4.tt.attention import Gemma4Attention
from models.demos.gemma4.tt.experts import Gemma4Experts
from models.demos.gemma4.tt.common import create_tt_model

MODE = {"isolated": False, "rows": None}  # rows: slice of reference positions this call covers
cur = {"layer": None}
rec = {}


def host(t):
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return x.reshape(-1, x.shape[-1])


def put(kind, t):
    n = MODE["rows"].stop - MODE["rows"].start
    rec.setdefault(kind, {})[cur["layer"]] = host(t)[:n]


_layer_call = L.Gemma4DecoderLayer.__call__
def layer_call(self, hidden_states, *a, **k):
    cur["layer"] = self.layer_idx
    if MODE["isolated"]:
        x = ref["in"][self.layer_idx][MODE["rows"]]  # HF's exact input to this layer
        x = x.reshape(1, 1, x.shape[0], x.shape[1])
        x = x if hidden_states.dtype == ttnn.float32 else x.to(torch.bfloat16)  # same dtype the layer would get
        hidden_states = ttnn.from_torch(x, device=self_mesh[0], dtype=hidden_states.dtype, layout=ttnn.TILE_LAYOUT,
                                        mesh_mapper=ttnn.ReplicateTensorToMesh(self_mesh[0]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
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


def score(rows, title):
    out = []
    for i in range(len(rec["out"])):
        r = {"layer": i}
        for kind in ("in", "attn", "mlp", "experts", "out"):
            p = pcc_rows(rec[kind][i], ref[kind][i][rows])
            r[kind] = round(p.mean().item(), 5)
            r[kind + "_min"] = round(p.min().item(), 4)
        tt_sets = (rec["routing"][i] != 0)
        hf_idx = ref["idx"][i][rows]
        hf_sets = torch.zeros_like(tt_sets); hf_sets.scatter_(1, hf_idx, True)
        overlap = (tt_sets & hf_sets).sum(-1).float()
        r["experts_same_8"] = round((overlap == 8).float().mean().item() * 100, 1)
        r["experts_overlap_of_8"] = round(overlap.mean().item(), 3)
        out.append(r)
        print(f"{title} " + json.dumps(r), flush=True)
    return out


os.environ["TT_CACHE_PATH"] = str(D / f"tt_cache_perlayer_{LABEL}")
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
self_mesh = [md]
results = {}
try:
    model_args, model, kv, _sd = create_tt_model(mesh_device=md, max_batch_size=1, max_seq_len=1024, model_path=MODEL, create_kv_cache=True)
    _sd = None; gc.collect()
    rep = ttnn.ReplicateTensorToMesh(md)

    def prefill(n):
        tok = ttnn.from_torch(ids[:n].reshape(1, n).to(torch.int32), device=md, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh_mapper=rep)
        emb = ttnn.to_layout(ttnn.reshape(model.embed_tokens(tok), (1, 1, n, model_args.hidden_size)), ttnn.TILE_LAYOUT)
        return model.ttnn_prefill_forward(emb, page_table=None, kv_cache=kv, input_ids_torch=ids[:n].reshape(1, n), embeds_torch=None)

    for mode, iso in (("prefill_accumulated", False), ("prefill_isolated", True)):
        rec.clear(); MODE.update(isolated=iso, rows=slice(0, S))
        t = time.time(); prefill(S).deallocate(True)
        print(f"{mode} ran in {time.time()-t:.0f}s", flush=True)
        results[mode] = score(slice(0, S), mode.upper())

    # C. decode: prefill 480 tokens, then decode 480..511 one token at a time (teacher forced)
    P0 = S - 32  # multiple of 32 (prefill length must be tile aligned for K/V)
    MODE.update(isolated=False, rows=slice(0, P0)); rec.clear(); prefill(P0).deallocate(True)
    steps = []
    for pos in range(P0, S):
        rec.clear(); MODE.update(rows=slice(pos, pos + 1))
        inp = model.prepare_inputs_decode(ids[pos:pos + 1], torch.tensor([pos]), page_table=None)
        logits, _ = model.ttnn_decode_forward(x=inp[0], current_pos=inp[1], rot_mat_idxs=inp[2], page_table=inp[3], kv_cache=kv)
        steps.append({k: dict(v) for k, v in rec.items()})
    rec.clear()
    for kind in steps[0]:
        rec[kind] = {i: torch.cat([s[kind][i] for s in steps]) for i in steps[0][kind]}
    results["decode_accumulated"] = score(slice(P0, S), "DECODE_ACCUMULATED")
    json.dump(results, open(D / f"tt_per_layer_pcc_{LABEL}.json", "w"), indent=1)
    print("DONE", flush=True)
finally:
    model = kv = None; gc.collect()
    ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
