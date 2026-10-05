# ttnn.transformer.scaled_dot_product_attention_decode vs torch, Gemma4 sliding-layer shapes on one chip's share:
# 4 query heads, 2 KV heads, head_dim 256, scale 1.0, positions up to cur_pos.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch, ttnn
torch.manual_seed(0)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
rep = ttnn.ReplicateTensorToMesh(md)
dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, device=md, dtype=dt, layout=lay, mesh_mapper=rep)
host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
NH, NKV, HD, S = 4, 2, 256, 1024
cfgs = {
    "default (none passed)": None,
    "HiFi4, fp32 acc": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False),
    "HiFi4, bf16 acc": ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False),
}
grid = md.compute_with_storage_grid_size()
pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y), q_chunk_size=32, k_chunk_size=64, exp_approx_mode=False)
k = (torch.randn(1, NKV, S, HD) * 0.5).to(torch.bfloat16).float(); v = torch.randn(1, NKV, S, HD).to(torch.bfloat16).float()
q = (torch.randn(1, 1, NH, HD) * 0.1).to(torch.bfloat16).float()
kt, vt = dev(k, ttnn.bfloat16), dev(v, ttnn.bfloat16); qt = dev(q, ttnn.bfloat16)
for cur in (10, 63, 64, 65, 100, 200, 500):
    # torch reference: GQA, attend to positions 0..cur inclusive
    kk = k[0, :, : cur + 1].repeat_interleave(NH // NKV, 0); vv = v[0, :, : cur + 1].repeat_interleave(NH // NKV, 0)
    att = torch.softmax(torch.einsum("hd,hsd->hs", q[0, 0], kk), -1); ref = torch.einsum("hs,hsd->hd", att, vv)
    row = []
    for name, ck in cfgs.items():
        pos = ttnn.from_torch(torch.tensor([cur], dtype=torch.int32), device=md, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
        kw = {} if ck is None else {"compute_kernel_config": ck}
        try:
            o = host(ttnn.transformer.scaled_dot_product_attention_decode(qt, kt, vt, cur_pos_tensor=pos, scale=1.0, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kw))[0, 0, :NH]
            p = float(torch.corrcoef(torch.stack((o.reshape(-1), ref.reshape(-1))))[0, 1]); row.append(f"{name}: PCC {p:.6f}")
        except Exception as e:
            row.append(f"{name}: ERROR {str(e).splitlines()[0][:80]}")
    print(f"SDPA cur_pos {cur:4d} | " + " | ".join(row), flush=True)
ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
