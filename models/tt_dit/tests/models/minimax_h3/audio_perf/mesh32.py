"""The real 32-chip config: shard T across BOTH mesh axes, so no chip repeats work.

Everything measured so far used `ParallelFactor(factor=8, mesh_axis=1)` on a 4x8 mesh, which shards T
across the 8-wide axis only and *replicates* along the 4-wide one. halo_correctness.py printed the
proof -- "replication groups: [4, 4, 4, 4, 4, 4, 4, 4]" -- so 283 ms was achieved with 8-way
parallelism and 24 of the 32 chips doing redundant work.

`AudioTParallelConfig(axis0, axis1)` shards both axes; its `factor` is the product, 4 * 8 = 32. The
halo (`_t_neighbor_pad`), the partition (`_partition_t`) and the closing gather (`_all_gather_t`) all
have two-axis branches, so the path exists -- it is simply less exercised than the single-axis one.

Reports PSNR against the *unsharded* decode on the same mesh, so a fast wrong answer fails here rather
than being reported as a speedup, plus untraced/traced timing.

Fit from the traced single-axis points predicted 191-281 ms at factor 32.
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
from models.tt_dit.parallel.config import AudioTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

N = int(os.environ.get("BENCH_N", "3"))
T = int(os.environ.get("NUM_T", "207"))
AX0 = int(os.environ.get("AX0_FACTOR", "4"))
AX1 = int(os.environ.get("AX1_FACTOR", "8"))

wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}
converted = convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors")))


def build(d, pc, ccl):
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
    return dec


def psnr(a, b):
    a, b = a.float(), b.float()
    if a.shape != b.shape:
        return None
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * float(torch.log10(a.abs().max() ** 2 / mse))


def med(fn):
    fn()
    ttnn.synchronize_device(d)
    s = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(d)
        s.append(time.perf_counter() - t0)
    return statistics.median(s)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
d = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=65536, trace_region_size=450_000_000)
try:
    pc = AudioTParallelConfig(
        axis0=ParallelFactor(factor=AX0, mesh_axis=0),
        axis1=ParallelFactor(factor=AX1, mesh_axis=1),
    )
    print(f"mesh 4x8 = {d.get_num_devices()} chips; axis0 factor {AX0} x axis1 factor {AX1} = {pc.factor}", flush=True)
    print(
        f"T={T}, so T pads to a multiple of {32 * pc.factor} and each chip holds "
        f"{(((T + 32 * pc.factor - 1) // (32 * pc.factor)) * 32 * pc.factor) // pc.factor} rows",
        flush=True,
    )

    torch.manual_seed(2)
    lat = torch.randn(2, cfg["latent_channels"], T) * 0.1

    ref_dec = build(d, None, None)
    ref = torch.as_tensor(ref_dec(lat)).float()
    ref_s = med(lambda: ref_dec(lat))
    print(f"REF   unsharded on mesh: {ref_s:.4f}s", flush=True)

    ccl = CCLManager(d, num_links=1, topology=ttnn.Topology.Linear)
    dec = build(d, pc, ccl)
    try:
        got = torch.as_tensor(dec(lat)).float()
        p = psnr(ref, got)
        untraced = med(lambda: dec(lat))
        dec(lat, traced=True)
        tracers = type(dec.decoder)._forward_device._tracers_keyed.get(dec.decoder, {})
        traced = med(lambda: dec(lat, traced=True))
        print(
            f"MESH32 factor={pc.factor}: untraced {untraced:.4f}s | traced {traced:.4f}s "
            f"-> {untraced / traced:.2f}x  PSNR vs unsharded {p:.1f} dB  (tracers={len(tracers)})",
            flush=True,
        )
        print(f"  vs the 8-way traced result (0.2860 s): {0.2860 / traced:.2f}x")
        print(f"  {'CORRECT' if p is not None and p > 40 else 'WRONG -- a fast wrong answer is not a result'}")
        dec.release_trace()
    except Exception as exc:
        print(f"MESH32 factor={pc.factor} FAILED: {str(exc)[:400]}", flush=True)
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
