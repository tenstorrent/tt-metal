"""Do per-request and batched prefill agree, at a k where batched works?"""
import argparse, os, torch, ttnn
from pathlib import Path
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.common.utility_functions import comp_pcc

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=32)
p.add_argument("--active", type=int, default=7)
p.add_argument("--length", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--bar", type=float, default=0.999)
a = p.parse_args()

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
try:
    g = build_generator(Path("models/autoports/qwen_qwen3_6_27b"), mesh,
                        num_layers=a.layers, max_context=512, batch=a.batch)
    v = g.model.vocab_size
    tokens = torch.zeros((a.batch, a.length), dtype=torch.long)
    lens = [0] * a.batch
    for s in range(a.active):
        tokens[s, :a.length] = (torch.arange(a.length) * 7 + s * 101 + 3) % v
        lens[s] = a.length
    out = {}
    for mode in ("0", "1"):
        os.environ["QWEN36_PREFILL_PER_REQUEST"] = mode
        g.model.reset_cache()
        g._slots_requiring_prefill = set(range(a.batch))
        out[mode] = g.prefill_forward(tokens, page_table=g._page_table,
                                      kv_cache=g.kv_cache, prompt_lens=lens).clone()
        print(f"  mode per_request={mode}: shape {tuple(out[mode].shape)}", flush=True)
    fails = []
    for s in range(a.active):
        ok, pcc = comp_pcc(out["0"][s].float(), out["1"][s].float(), a.bar)
        am = int(out["0"][s].argmax()) == int(out["1"][s].argmax())
        print(f"  slot {s}: pcc={pcc} argmax_match={am}", flush=True)
        if not (ok and am):
            fails.append(s)
    print("EQUIV OK" if not fails else f"EQUIV FAILED slots {fails}")
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
