# Each ttnn op the Gemma4 decode path uses, run on the QB2 mesh with bf16 and with fp32 activations
# (weights bf16), compared against a torch fp32 reference on the same inputs. Prints PCC per op.
import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import torch
import ttnn
from models.demos.gemma4.tt import fp32_mode  # noqa: F401  (module import check)

torch.manual_seed(0)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
rep = ttnn.ReplicateTensorToMesh(md)
HI = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)


def dev(t, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, device=md, dtype=dtype, layout=layout, mesh_mapper=rep)


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def pcc(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


def report(name, ref, run):
    out = []
    for dt in (ttnn.bfloat16, ttnn.float32):
        try:
            y = run(dt)
            y = host(y)[tuple(slice(0, s) for s in ref.shape)].reshape(ref.shape)
            out.append(f"{'bf16' if dt == ttnn.bfloat16 else 'fp32'} PCC {pcc(y, ref):.6f} out-dtype-ok")
        except Exception as e:  # noqa: BLE001
            out.append(f"{'bf16' if dt == ttnn.bfloat16 else 'fp32'} ERROR {type(e).__name__}: {str(e).splitlines()[0][:90]}")
    print(f"OP {name:34s} | " + " | ".join(out), flush=True)


x = torch.randn(1, 1, 32, 2816)
w = (torch.randn(2816, 2048) / 53).to(torch.bfloat16).float()
report("linear 2816->2048 (HiFi4 fp32acc)", x @ w, lambda dt: ttnn.linear(dev(x, dt), dev(w, ttnn.bfloat16), compute_kernel_config=HI))
g = torch.randn(2816).to(torch.bfloat16).float()
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * g
report("rms_norm (weight)", ref, lambda dt: ttnn.rms_norm(dev(x, dt), weight=dev(g.reshape(1, 1, -1, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), epsilon=1e-6, compute_kernel_config=HI))
report("gelu (accurate)", torch.nn.functional.gelu(x), lambda dt: ttnn.gelu(dev(x, dt), variant=ttnn.GeluVariant.Accurate))
y = torch.randn(1, 1, 32, 2816)
report("mul", x * y, lambda dt: ttnn.mul(dev(x, dt), dev(y, dt)))
report("add", x + y, lambda dt: ttnn.add(dev(x, dt), dev(y, dt)))
s = torch.randn(1, 1, 32, 128)
report("softmax over 128", torch.softmax(s, -1), lambda dt: ttnn.softmax(dev(s, dt), dim=-1))
report("tanh", torch.tanh(x / 30) * 30, lambda dt: ttnn.mul(ttnn.tanh(ttnn.mul(dev(x, dt), 1 / 30)), 30.0))
# topk on probabilities: compare values (sorted)
p = torch.softmax(torch.randn(1, 1, 32, 128), -1)
report("topk(8) values", p.topk(8, -1).values, lambda dt: ttnn.topk(dev(p, dt), k=8, dim=-1)[0])
def topk_idx(dt):
    v, i = ttnn.topk(dev(p, dt), k=8, dim=-1)
    got = ttnn.to_torch(ttnn.get_device_tensors(i)[0]).long()[..., :8]
    want = p.topk(8, -1).indices
    return ttnn.from_torch((torch.sort(got, -1).values == torch.sort(want, -1).values).float(), device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
report("topk(8) indices exact (1.0=all)", torch.ones(1, 1, 32, 8), topk_idx)
# sparse_matmul expert gate: x[1,1,32,2816] x W[1,E,2816,192], 8 active experts
E, I = 128, 192
We = (torch.randn(1, E, 2816, I) / 53).to(torch.bfloat16).float()
active = torch.zeros(1, 1, 32, E); idx = torch.randperm(E)[:8]; active[0, 0, 0, idx] = 1.0
x1 = torch.zeros(1, 1, 32, 2816); x1[0, 0, 0] = x[0, 0, 0]
ref = torch.zeros(1, E, I)
for e in idx.tolist():
    ref[0, e] = x1[0, 0, 0] @ We[0, e]
def sm(dt):
    sp = ttnn.from_torch(active, device=md, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(6, 1), in0_block_w=1, out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=1, per_core_M=1, per_core_N=1, fuse_batch=False, fused_activation=None, mcast_in0=True)
    out = ttnn.sparse_matmul(dev(x1, dt), dev(We, ttnn.bfloat16), sparsity=sp, nnz=8, memory_config=ttnn.L1_MEMORY_CONFIG, output_tile=ttnn.Tile([32, 32]), program_config=cfg, dtype=dt, compute_kernel_config=HI)
    t = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(1, 1, E, 32, I)[:, :, :, 0, :]  # row 0 of each expert
    return ttnn.from_torch(t.reshape(1, E, I), device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
report("sparse_matmul experts (8 of 128)", ref[:, idx], lambda dt: ttnn.from_torch(host(sm(dt))[:, idx], device=md, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep))
# all_reduce across the 4 chips (each holds x): sum = 4x
sh = ttnn.ShardTensor2dMesh(md, md.shape, dims=(None, 0))
xs = torch.randn(4, 1, 32, 2816)
def ar(dt):
    t = ttnn.from_torch(xs, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensor2dMesh(md, md.shape, dims=(None, 0)))
    return ttnn.all_reduce(t, cluster_axis=1)
report("all_reduce (sum of 4 chips)", xs.sum(0, keepdim=True), ar)
# typecast fp32<->bf16 roundtrip + embedding hi/lo trick
emb = torch.randn(1024, 256)
hi = emb.to(torch.bfloat16); lo = (emb - hi.float()).to(torch.bfloat16)
pos = torch.zeros(1, 32, dtype=torch.int32); pos[0, 0] = 517
def rg(dt):
    ti = ttnn.from_torch(pos, device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
    a = ttnn.embedding(ti, dev(hi, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.TILE_LAYOUT)
    if dt == ttnn.bfloat16:
        return a
    b = ttnn.embedding(ti, dev(lo, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.TILE_LAYOUT)
    return ttnn.add(ttnn.typecast(a, ttnn.float32), ttnn.typecast(b, ttnn.float32))
report("rope table gather (bf16 | hi+lo)", emb[517].reshape(1, 1, 256), lambda dt: ttnn.reshape(rg(dt), (1, 32, 256))[:, 0:1, :])
ttnn.close_mesh_device(md)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
