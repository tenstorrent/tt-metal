# SPDX-License-Identifier: Apache-2.0
"""Emit the EXACT ttnn call sequence a phi decode executes, for one policy.

Wraps a curated list of ttnn entry points, logs name + shape/layout/memory-config of the
first tensor argument and of the result, runs ONE decode, writes a .py-shaped trace file.
Run twice (rope off / rope on) and diff the two files in split view.
"""
import importlib.util, os, sys, io, json
import torch, ttnn

sys.argv = [sys.argv[0]]
MD = "microsoft_phi_3_5_mini_instruct"
ROPE = os.environ.get("TRACE_ROPE", "")
CORES = os.environ.get("TRACE_NORM_CORES", "0")
OUT = os.environ["TRACE_OUT"]
os.environ.update({"CHALLENGER_MODEL_DIR": MD, "CHALLENGER_DECODE_BATCH": "32",
                   "CHALLENGER_REQUESTED_DECODE_BATCH": "32",
                   "CHALLENGER_ADVISOR_ROPE_L1": ROPE,
                   "CHALLENGER_ADVISOR_NORM_CORES": CORES,
                   "CHALLENGER_HARNESS_SCOPE": "trace"})

LOG = []
DEPTH = {"n": 0}

def desc(t):
    if not isinstance(t, ttnn.Tensor):
        return None
    try:
        mc = t.memory_config()
        buf = str(mc.buffer_type).split(".")[-1].lower()
        ml = str(mc.memory_layout).split(".")[-1].lower()
        shard = ""
        try:
            ss = mc.shard_spec
            if ss is not None:
                g = ss.grid
                ncores = g.num_cores() if hasattr(g, "num_cores") else "?"
                shard = f", shard={tuple(ss.shape)}, cores={ncores}"
        except Exception:
            pass
        lay = str(t.layout).split(".")[-1]
        return f"{tuple(t.shape)} {t.dtype.name if hasattr(t.dtype,'name') else t.dtype} {lay} {buf}/{ml}{shard}"
    except Exception:
        return "<tensor>"

TARGETS = [
    (ttnn, "to_memory_config"), (ttnn, "to_layout"), (ttnn, "typecast"), (ttnn, "reshape"),
    (ttnn, "embedding"), (ttnn, "linear"), (ttnn, "matmul"), (ttnn, "add"), (ttnn, "multiply"),
    (ttnn, "concat"), (ttnn, "permute"), (ttnn, "transpose"), (ttnn, "silu"), (ttnn, "neg"),
    (ttnn, "rms_norm"), (ttnn, "layer_norm"), (ttnn, "slice"), (ttnn, "sharded_to_interleaved"),
    (ttnn, "interleaved_to_sharded"), (ttnn, "tilize"), (ttnn, "untilize"),
    (ttnn, "tilize_with_val_padding"), (ttnn, "untilize_with_unpadding"),
]
for mod, name in (("experimental","rotary_embedding_llama"), ("experimental","rotary_embedding_hf"),
                  ("experimental","nlp_create_qkv_heads_decode"), ("experimental","nlp_concat_heads_decode"),
                  ("experimental","paged_update_cache"),
                  ("transformer","paged_scaled_dot_product_attention_decode")):
    m = getattr(ttnn, mod, None)
    if m is not None and hasattr(m, name):
        TARGETS.append((m, f"{name}"))

def wrap(mod, name):
    orig = getattr(mod, name)
    label = (getattr(mod, "__name__", "ttnn").split(".")[-1] + "." if mod is not ttnn else "ttnn.") + name
    def inner(*a, **k):
        ins = [desc(x) for x in a if isinstance(x, ttnn.Tensor)]
        r = orig(*a, **k)
        outs = []
        rr = r if isinstance(r, (tuple, list)) else [r]
        for x in rr:
            d = desc(x)
            if d: outs.append(d)
        kw = {kk: (str(vv).split(".")[-1] if "MemoryConfig" not in type(vv).__name__ else "<memcfg>")
              for kk, vv in k.items() if kk in ("dim","memory_config","layout","dtype","num_heads","num_kv_heads","keepdim","epsilon")}
        LOG.append({"op": label, "in": ins, "out": outs, "kwargs": kw})
        return r
    try:
        setattr(mod, name, inner)
    except Exception:
        pass

for mod, name in TARGETS:
    if hasattr(mod, name): wrap(mod, name)

spec = importlib.util.spec_from_file_location(
    "h", f"models/autoports/{MD}/doc/advisor_challenger/scripts/harness.py")
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
POL = json.load(open(f"models/autoports/{MD}/doc/advisor_challenger/incumbent.json"))["shipped_policy"]
dev = ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=64*1024*1024)
try:
    state = h.build(dev, POL)
    LOG.clear()                      # drop construction, keep only the decode
    h.decode(state)
finally:
    ttnn.close_mesh_device(dev)

with open(OUT, "w") as f:
    f.write(f"# EXECUTED ttnn CALL SEQUENCE - one phi-3.5-mini decode layer, batch 32\n")
    f.write(f"# CHALLENGER_ADVISOR_ROPE_L1={ROPE!r}   CHALLENGER_ADVISOR_NORM_CORES={CORES}\n")
    f.write(f"# {len(LOG)} traced ttnn calls\n\n")
    for i, e in enumerate(LOG, 1):
        kw = ("  # " + ", ".join(f"{k}={v}" for k,v in e["kwargs"].items())) if e["kwargs"] else ""
        f.write(f"[{i:3d}] {e['op']}({kw.strip('# ') if False else ''})\n")
        for s in e["in"]:  f.write(f"        in   {s}\n")
        for s in e["out"]: f.write(f"        out  {s}\n")
        if e["kwargs"]:    f.write(f"        args {e['kwargs']}\n")
print("wrote", OUT, len(LOG), "calls")
