"""CPU: the device dataflow (torch model hy4_attn_ref.attention_dataflow) on REAL layer weights vs the oracle golden (L{layer}.pt). Usage: python check_dataflow_vs_golden.py LAYER [NTOK]"""
import sys

import torch

from models.demos.blackhole.hy4_preview.config import Hy4Config
from models.demos.blackhole.hy4_preview.tt.hy4_attn_ref import attention_dataflow

GOLD = "/mnt/tt-data/ssinghal/hy4-logs/layer-attn/golden"
torch.set_num_threads(16)
L = int(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3072
cfg = Hy4Config.from_checkpoint()
g = torch.load(f"{GOLD}/L{L}.pt")
import json

from safetensors import safe_open

ck = "/mnt/tt-data/Hy4-Preview"
wm = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
sd = {}
for k, f in wm.items():
    if k.startswith(f"model.layers.{L}.self_attn."):
        with safe_open(f"{ck}/{f}", "pt") as fh:
            sd[k[len(f"model.layers.{L}.") :]] = fh.get_tensor(k).float()
h = g["h"][:n]
idx = g["idx"][:n]
allowed = torch.zeros(n, n + 1, dtype=torch.bool)
allowed.scatter_(1, torch.where(idx >= 0, idx, torch.full_like(idx, n)), True)
allowed = allowed[:, :n] & torch.ones(n, n, dtype=torch.bool).tril()
out, inter = attention_dataflow(cfg, sd, h, allowed, rope_interleave=g["conv"]["rope_interleave"])
ref = g["out"][:n]
pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
print(
    f"layer {L} dataflow vs oracle golden: pcc {pcc:.7f} max abs {(out-ref).abs().max():.3e} (|ref| max {ref.abs().max():.3e})"
)
rows = torch.cat([g["lat"][:n], g["k_pe"][:n]], -1)
print("cache rows max abs diff", (inter["rows"] - rows).abs().max().item())
