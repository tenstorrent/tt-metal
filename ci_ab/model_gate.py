# Scratch (not for merge): TTMoEGate per model config on one chip, outputs saved for model_gate_cmp.py.
import os, sys
from pathlib import Path
import torch
import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig

OUT = sys.argv[1]
CFG = Path("/work/models/common/modules/moe/configs")
NAMES = os.environ.get("GATE_CFGS", "").split()
SEEDS = int(os.environ.get("GATE_SEEDS", "8"))
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
res = {}
try:
    for name in NAMES:
        cfg = TTMoEGateConfig.from_yaml((CFG / f"{name}.yaml").read_text())
        n, k, hidden = cfg.num_routed_experts, cfg.select_experts_k, cfg.hidden_size
        batch = cfg.batch_per_device or 32
        for stim in ("test", "router"):
            for seed in range(SEEDS):
                g = torch.Generator().manual_seed(1000 * seed + 7)
                if stim == "test":
                    x = (2 * torch.rand((batch, hidden), generator=g) - 1).to(torch.bfloat16)
                    w = ((2 * torch.rand((hidden, n), generator=g) - 1) * 0.1).to(torch.bfloat16)
                    b = (2 * torch.rand((n,), generator=g) - 1) if cfg.score_correction_bias else None
                    pb = (2 * torch.rand((n,), generator=g) - 1) if cfg.gate_proj_bias else None
                else:
                    # RMS-normed hidden states and a router whose logits have unit-scale spread, a small correction bias.
                    x = torch.randn((batch, hidden), generator=g).to(torch.bfloat16)
                    w = (torch.randn((hidden, n), generator=g) / hidden**0.5).to(torch.bfloat16)
                    b = (0.02 * torch.randn((n,), generator=g)) if cfg.score_correction_bias else None
                    pb = (0.1 * torch.randn((n,), generator=g)) if cfg.gate_proj_bias else None
                gate = TTMoEGate(mesh, cfg, torch_gate_weight=w, torch_gate_bias=b, torch_gate_proj_bias=pb)
                tx = ttnn.from_torch(x.reshape(1, 1, batch, hidden), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                     device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                     mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                ts, ti = gate.forward(tx)
                comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
                ds = ttnn.to_torch(ts, mesh_composer=comp).reshape(-1, k)[:batch].float()
                di = ttnn.to_torch(ti, mesh_composer=comp).reshape(-1, k)[:batch].to(torch.int64)
                res[(name, stim, seed)] = dict(x=x, w=w, b=b, pb=pb, ds=ds, di=di, k=k, n=n, cfg=name)
                del gate
            print("GATE CFG", name, stim, "done", flush=True)
finally:
    torch.save(res, OUT)
    print("GATE DONE", len(res), flush=True)
    ttnn.close_mesh_device(mesh)
