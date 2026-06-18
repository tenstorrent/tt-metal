# SPDX-License-Identifier: Apache-2.0
"""TP MLP on a 1x8 line (Megatron-style: col-parallel gate/up, row-parallel down + all_reduce).
Validates against a host reference. Template for the rest of the 8-chip TP port."""
import torch
import ttnn

TP = 8


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    H, I = 5120, 17408
    torch.manual_seed(0)
    Wg = torch.randn(I, H) * 0.02   # gate_proj.weight [out=I, in=H]
    Wu = torch.randn(I, H) * 0.02   # up_proj.weight
    Wd = torch.randn(H, I) * 0.02   # down_proj.weight [out=H, in=I]
    x = torch.randn(1, 1, 1, H) * 0.5

    # host reference (matches TtMLP: linear(x, W.T))
    g = torch.nn.functional.silu(x @ Wg.T)        # [..,I]
    u = x @ Wu.T
    ref = (g * u) @ Wd.T                            # [..,H]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, TP))
    try:
        shape = (1, TP)
        # weights, sharded across the 8-device (col) axis
        gate_w = ttnn.from_torch(Wg.T.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                 device=md, mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=shape))
        up_w = ttnn.from_torch(Wu.T.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                               device=md, mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 3), mesh_shape=shape))
        # down: row-parallel → shard the contraction dim (I = dim 2)
        down_w = ttnn.from_torch(Wd.T.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT,
                                 device=md, mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(None, 2), mesh_shape=shape))
        # x replicated to all devices
        xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=md, mesh_mapper=ttnn.ReplicateTensorToMesh(md))

        gate = ttnn.silu(ttnn.linear(xt, gate_w))   # [..,I/4] per chip
        up = ttnn.linear(xt, up_w)
        hidden = ttnn.mul(gate, up)                  # [..,I/4] per chip
        out_partial = ttnn.linear(hidden, down_w)    # [..,H] partial sum per chip
        out = ttnn.all_reduce(out_partial, cluster_axis=1, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(md)

        # all_reduce yields the same full tensor on every device; take device 0
        out_t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0))[0:1].float()
        print(f"[shapes] ref={tuple(ref.shape)} out={tuple(out_t.shape)}", flush=True)
        print(f"[PCC] TP MLP vs host ref = {pcc(ref, out_t):.5f}", flush=True)
        print("PASS" if pcc(ref, out_t) > 0.98 else "FAIL", flush=True)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
