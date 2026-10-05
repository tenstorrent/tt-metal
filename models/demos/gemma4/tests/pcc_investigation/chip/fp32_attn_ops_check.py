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
dev = lambda t, dt, lay=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t, device=md, dtype=dt, layout=lay, mesh_mapper=rep, memory_config=mc)
host = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
HI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False)
NKV, HD, BS, NB = 2, 256, 32, 32
# 1. paged_update_cache into an fp32 paged cache
cache = dev(torch.zeros(NB, NKV, BS, HD), ttnn.float32)
pt = ttnn.from_torch(torch.arange(NB, dtype=torch.int32).reshape(1, NB), device=md, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
newk = torch.randn(1, 1, NKV, HD)  # [1, batch, nkv, hd] like decode tt_k
try:
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    shard = ttnn.create_sharded_memory_config(shape=(32, HD), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    kt = ttnn.to_memory_config(dev(newk, ttnn.float32), shard)
    pos = ttnn.from_torch(torch.tensor([70], dtype=torch.int32), device=md, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
    ttnn.experimental.paged_update_cache(cache, kt, update_idxs_tensor=pos, page_table=pt, block_size=BS, num_kv_heads=NKV)
    got = host(cache)[70 // BS, :, 70 % BS, :]
    print("PAGED_UPDATE fp32 cache: max abs err", (got - newk[0, 0]).abs().max().item())
except Exception as e:
    print("PAGED_UPDATE fp32 ERROR", str(e).splitlines()[0][:200])
# 2. batched fp32 matmul q[1,nkv,G,hd] x k^T[1,nkv,hd,S]
G, S = 2, 1024
q = torch.randn(1, NKV, 32, HD); K = torch.randn(1, NKV, S, HD)
try:
    sc = ttnn.matmul(dev(q, ttnn.float32), ttnn.transpose(dev(K, ttnn.float32), -2, -1), compute_kernel_config=HI)
    ref = q @ K.transpose(-2, -1)
    y = host(sc); print("MATMUL q.kT fp32 PCC", float(torch.corrcoef(torch.stack((y.reshape(-1), ref.reshape(-1))))[0, 1]), "max rel err", ((y - ref).abs().max() / ref.abs().max()).item())
except Exception as e:
    print("MATMUL ERROR", str(e).splitlines()[0][:200])
# 3. mask from a device position: arange > cur -> -inf
try:
    ar = dev(torch.arange(S, dtype=torch.float32).reshape(1, 1, 1, S).expand(1, 1, 32, S).contiguous(), ttnn.float32)
    cur = ttnn.typecast(ttnn.to_layout(ttnn.from_torch(torch.full((1, 1, 32, 32), 70, dtype=torch.int32), device=md, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep), ttnn.TILE_LAYOUT), ttnn.float32)
    m = ttnn.mul(ttnn.gt(ar, ttnn.repeat(ttnn.slice(cur, [0, 0, 0, 0], [1, 1, 32, 1]), ttnn.Shape([1, 1, 1, S]))), -1e9)
    mm = host(m)[0, 0, 0]; print("MASK: positions masked", int((mm < 0).sum()), "first masked", int((mm < 0).nonzero()[0]))
except Exception as e:
    print("MASK ERROR", str(e).splitlines()[0][:200])
# 4. softmax fp32 over 1024 with HiFi4
s = torch.randn(1, 2, 32, S) * 2
try:
    y = host(ttnn.softmax(dev(s, ttnn.float32), dim=-1, compute_kernel_config=HI, numeric_stable=True)); ref = torch.softmax(s, -1)
    print("SOFTMAX 1024 fp32: max abs err", (y - ref).abs().max().item(), "PCC", float(torch.corrcoef(torch.stack((y.reshape(-1), ref.reshape(-1))))[0, 1]))
except Exception as e:
    print("SOFTMAX ERROR", str(e).splitlines()[0][:200])
ttnn.close_mesh_device(md); ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
