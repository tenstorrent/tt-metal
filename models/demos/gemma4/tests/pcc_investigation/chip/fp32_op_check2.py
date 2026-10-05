import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch, ttnn, traceback
torch.manual_seed(0)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
rep = ttnn.ReplicateTensorToMesh(md)
HI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, device=md, dtype=dt, layout=lay, mesh_mapper=rep)
host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
pcc = lambda a, b: float(torch.corrcoef(torch.stack((a.reshape(-1).double(), b.reshape(-1).double())))[0, 1])
s = torch.randn(1, 1, 32, 128) * 3
ref = torch.softmax(s, -1)
for dt in (ttnn.bfloat16, ttnn.float32):
    for name, kw in (("default", {}), ("numeric_stable", {"numeric_stable": True}), ("HiFi4 no-approx", {"compute_kernel_config": HI}), ("both", {"numeric_stable": True, "compute_kernel_config": HI})):
        try:
            y = host(ttnn.softmax(dev(s, dt), dim=-1, **kw))
            err = (y - ref).abs().max().item()
            print(f"SOFTMAX {('bf16' if dt == ttnn.bfloat16 else 'fp32')} {name:16s} PCC {pcc(y, ref):.7f} max abs err {err:.2e}")
        except Exception as e:
            print(f"SOFTMAX {name} ERROR {str(e).splitlines()[0][:150]}")
# sparse matmul exactly as experts/decode.py builds it (batch_size=1, decode)
import math, sys
sys.path.insert(0, ".")
from models.demos.gemma4.tt.experts.decode import _build_sparse_matmul_config
E, I, H = 128, 192, 2816
We = (torch.randn(1, E, H, I) / 53).to(torch.bfloat16).float()
x = torch.randn(1, 1, 1, H)
idx = torch.randperm(E)[:8]
rw = torch.zeros(1, 1, 1, E); rw[0, 0, 0, idx] = torch.rand(8)
ref = torch.stack([x[0, 0, 0] @ We[0, e] for e in idx.tolist()])
for dt in (ttnn.bfloat16, ttnn.float32):
    try:
        sp = ttnn.to_layout(dev(rw, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.sparse_matmul(dev(x, dt), dev(We, ttnn.bfloat16), sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG,
                                 output_tile=ttnn.Tile([32, 32]), program_config=_build_sparse_matmul_config(1, I), dtype=dt, compute_kernel_config=HI)
        y = host(out)
        y = y.reshape(-1, E, y.shape[-2] if y.dim() > 2 else 1, y.shape[-1])[0, :, 0, :I]
        print(f"SPARSE {('bf16' if dt == ttnn.bfloat16 else 'fp32')} out shape {tuple(out.shape)} dtype {out.dtype} PCC {pcc(y[idx], ref):.7f}")
    except Exception as e:
        print("SPARSE ERROR", str(e)[:600].replace("\n", " | "))
ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
