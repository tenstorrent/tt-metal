"""Which conv shapes are wrong under T-sharding? Standalone Conv1dViaConv3d, sharded vs not.

stage_bisect.py put the first divergence at `conv_pre` (-191 dB, values ~1e9x the reference), and
halo_correctness.py then showed `_partition_t` and `_t_neighbor_pad` are both exactly right at that
shape -- so the conv is being handed correct input and producing garbage.

This drops the decoder entirely and tests `Conv1dViaConv3d` on its own: same weights, one instance
sharded and one not, compared on the same input. Sweeping shape says whether the fault is universal or
specific to the big-C convs that trip the DRAM auto-slice fallback (the decode logs 18 of those, and the
factor=8 run added "could not find valid slice configuration ... on output dimension 160" on top).

  all shapes wrong            -> the sharded conv path is broken outright
  only large C_in wrong       -> the auto-slice / C-chunking fallback is wrong under sharding
  only kernel > 1 wrong       -> the halo is fine standalone but the conv mis-consumes it
"""

import os

import torch

import ttnn
from models.tt_dit.layers.audio_ops import Conv1dViaConv3d, _partition_t
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

FACTOR = int(os.environ.get("T_FACTOR", "8"))
MESH_AXIS = int(os.environ.get("MESH_AXIS", "1"))
T = int(os.environ.get("NUM_T", "256"))

# (C_in, C_out, kernel, dilation) -- conv_pre is (2048, 1024, 7, 1); the rest walk C down and kernel
# up so the failure boundary is visible rather than inferred from one point.
CASES = [
    (2048, 1024, 7, 1),  # conv_pre itself
    (256, 256, 7, 1),
    (64, 64, 7, 1),
    (64, 64, 3, 1),
    (64, 64, 1, 1),  # kernel 1 needs no halo at all
    (64, 64, 3, 5),  # dilated, halo 5
    (1024, 512, 7, 1),
]


def psnr(a, b):
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    if mse == 0:
        return float("inf")
    return 10.0 * float(torch.log10(a.abs().max() ** 2 / mse))


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    pc = ParallelFactor(factor=FACTOR, mesh_axis=MESH_AXIS)
    ccl = CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    print(f"factor={FACTOR} axis={MESH_AXIS} T={T} local_T={T // FACTOR}\n", flush=True)
    print(f"{'C_in':>6} {'C_out':>6} {'k':>3} {'dil':>4} {'ref shape':>20} {'sharded shape':>20} {'PSNR dB':>9}")
    print("-" * 76)

    for c_in, c_out, k, dil in CASES:
        torch.manual_seed(0)
        w = torch.randn(c_out, c_in, k) * (1.0 / (c_in * k) ** 0.5)
        x = torch.randn(2, T, c_in) * 0.3

        def make(parallel_config, ccl_manager):
            conv = Conv1dViaConv3d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=k,
                stride=1,
                dilation=dil,
                bias=False,
                padding_mode="zeros",
                mesh_device=d,
                dtype=ttnn.float32,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            conv.load_torch_state_dict({"weight": w})
            return conv

        try:
            ref_conv = make(None, None)
            xr = ttnn.from_torch(x, device=d, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
            ref = ttnn.to_torch(ttnn.get_device_tensors(ref_conv(xr))[0]).float()

            sh_conv = make(pc, ccl)
            xs = ttnn.from_torch(x, device=d, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32)
            xs = ttnn.to_layout(_partition_t(ttnn.to_layout(xs, ttnn.TILE_LAYOUT), pc), ttnn.ROW_MAJOR_LAYOUT)
            out = sh_conv(xs)
            parts = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(out)[:FACTOR]]
            got = torch.cat(parts, dim=1)

            p = psnr(ref, got)
            flag = "" if (p is not None and p > 40) else "  <== WRONG"
            print(
                f"{c_in:>6} {c_out:>6} {k:>3} {dil:>4} {str(tuple(ref.shape)):>20} "
                f"{str(tuple(got.shape)):>20} {('n/a' if p is None else f'{p:8.1f}')}{flag}",
                flush=True,
            )
            if p is not None and p <= 40:
                print(
                    f"       ref absmax {ref.abs().max():.4g}  sharded absmax {got.abs().max():.4g}  "
                    f"(a huge sharded absmax means uninitialized memory, not arithmetic)"
                )
        except Exception as exc:
            print(f"{c_in:>6} {c_out:>6} {k:>3} {dil:>4}  FAILED: {str(exc)[:150]}", flush=True)
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
