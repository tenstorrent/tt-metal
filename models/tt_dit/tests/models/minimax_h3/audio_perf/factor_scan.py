"""Clean T-sharding scaling curve: one mesh axis, one T, factor 1..8.

Needed because the numbers item 2 has been reasoned from are not comparable with each other. The
t_parallel test runs factor=4 on mesh axis 0 (the 4-wide axis) and factor=8 on axis 1 (the 8-wide one),
so fitting `t = F + D/factor` across those two points mixes an axis change into the factor change --
and the factor=8 point is the KNOWN_BROKEN one (PSNR -4.0 dB), so its timing may not even be work.

Here everything varies except the factor:

  * one mesh axis for the whole scan -- but see the constraint below: only factor == the axis length
    actually runs, so "one axis, many factors" is not achievable on a 4x8 mesh
  * factor=1 still runs *on the 32-chip mesh*, so mesh overhead is in the baseline rather than
    confounded with it -- that isolates sharding from "being on a mesh at all"
  * fusion off throughout, since `_snake_conv_params` declines when factor > 1 and a fused factor=1
    baseline would be measuring a different graph

Fit `t = F + D/factor` on the result: F is what sharding cannot remove, D is what it divides.

**Measured constraint (2026-08-12): the T-shard factor must equal the mesh axis length.** On a 4x8
mesh, factor=2 and factor=4 on axis 1 (which is 8 wide) both die in `_partition_t` with
slice_device_operation.cpp:164, "height begin index aligned to tiles" -- the partition indexes by the
device's coordinate along the axis and assumes it covers the axis. factor=4 on axis 0 (4 wide) and
factor=8 on axis 1 (8 wide) both run. That is exactly why the test's FACTORS are [(1,1), (4,0), (8,1)]
and why a single-axis scaling curve is not obtainable here: the only comparable pairs differ in axis
too. Reaching 32 needs `AudioTParallelConfig`, which shards both axes.

Results, T=207, fusion off:
    single-device mesh, factor 1   1.0980 s
    32-chip mesh,       factor 1   1.4409 s   <- +343 ms just for being on the mesh
    32-chip mesh,       factor 4   0.9469 s   (axis 0)
    32-chip mesh,       factor 8   0.8950 s   (axis 1, KNOWN_BROKEN, PSNR -4.0 dB)
"""

import json
import os
import statistics
import time

import torch
from safetensors.torch import load_file

import ttnn
from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder
from models.tt_dit.parallel.config import ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

N = int(os.environ.get("BENCH_N", "3"))
MESH_AXIS = int(os.environ.get("MESH_AXIS", "1"))
FACTORS = [int(x) for x in os.environ.get("FACTORS", "1,2,4,8").split(",")]
T = int(os.environ.get("NUM_T", "207"))

wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}
converted = convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors")))

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536)
try:
    print(f"mesh {d.get_num_devices()} chips, axis={MESH_AXIS}, T={T}, fusion off", flush=True)
    rows = []
    for factor in FACTORS:
        pc = None if factor <= 1 else ParallelFactor(factor=factor, mesh_axis=MESH_AXIS)
        ccl = None if pc is None else CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
        dec = MiniMaxH3AudioDecoder(
            latent_channels=cfg["latent_channels"],
            latent_dim=cfg["latent_dim"],
            decoder_dim=cfg["decoder_dim"],
            decoder_rates=tuple(cfg["decoder_rates"]),
            decoder_kernel_sizes=tuple(cfg["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(cfg["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(x) for x in cfg["resblock_dilation_sizes"]),
            mesh_device=d,
            parallel_config=pc,
            ccl_manager=ccl,
        )
        dec.load_torch_state_dict(converted, strict=False)
        torch.manual_seed(2)
        lat = torch.randn(2, cfg["latent_channels"], T) * 0.1
        try:
            ref = dec(lat)
            ttnn.synchronize_device(d)
            s = []
            for _ in range(N):
                t0 = time.perf_counter()
                dec(lat)
                ttnn.synchronize_device(d)
                s.append(time.perf_counter() - t0)
            m = statistics.median(s)
            rows.append((factor, m))
            print(f"SCAN factor={factor:>2} axis={MESH_AXIS}  median {m:.4f}s  min {min(s):.4f}", flush=True)
        except Exception as exc:
            print(f"SCAN factor={factor:>2} axis={MESH_AXIS}  FAILED: {str(exc)[:200]}", flush=True)
        del dec

    if len(rows) >= 2:
        # Least squares on t = F + D*(1/factor).
        xs = [1.0 / f for f, _ in rows]
        ys = [m for _, m in rows]
        n = len(rows)
        sx, sy = sum(xs), sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        D = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        F = (sy - D * sx) / n
        print(f"\nfit  t = {F * 1e3:.1f} ms + {D * 1e3:.1f} ms / factor")
        print(f"  F (sharding cannot remove) {F * 1e3:.1f} ms")
        print(f"  D (sharding divides)       {D * 1e3:.1f} ms")
        for f in (8, 32):
            print(f"  projected factor {f:>2}: {(F + D / f) * 1e3:.1f} ms")
        print("\n  item 1 single-chip reference: 260 ms fixed + 838 ms data (unfused 1.0980 s)")
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
