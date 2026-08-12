"""Is the mesh's extra cost dispatch? Trace the decode on the 32-chip mesh, sharded and not.

Item 1 measured trace at 1.04x on a *single* device and concluded host dispatch was not the lever.
factor_scan.py then measured factor=1 on the 32-chip mesh at 1.4409 s against 1.0980 s on a
single-device mesh -- +343 ms for being on the mesh with no sharding at all. Per-op dispatch has to
reach 32 devices instead of 1, so the natural reading is that the mesh reintroduces exactly the
host-dispatch cost that tracing removes, and that trace is a big lever on a mesh even though it is
nearly nothing on one chip. That would also explain `vocoder_ltx.Vocoder`'s "~70% host-bound"
docstring, which item 1 found false for a single device.

If traced factor=8 lands near `260 + 838/8` = ~365 ms, the route to <=300 ms is
mesh + T-shard + trace, and the remaining gap is item 3's job.
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
FACTORS = [int(x) for x in os.environ.get("FACTORS", "1,8").split(",")]
T = int(os.environ.get("NUM_T", "207"))

wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}
converted = convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors")))


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
    print(f"mesh {d.get_num_devices()} chips, axis={MESH_AXIS}, T={T}, fusion off", flush=True)
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
            plain = dec(lat)
            untraced = med(lambda: dec(lat))
            traced_out = dec(lat, traced=True)
            # A trace that silently fell through to the untraced path would report a flattering ratio,
            # so confirm a tracer was actually captured before believing the number.
            tracers = type(dec.decoder)._forward_device._tracers_keyed.get(dec.decoder, {})
            tr = med(lambda: dec(lat, traced=True))
            p, t = torch.as_tensor(plain).float(), torch.as_tensor(traced_out).float()
            mse = float(((p - t) ** 2).mean())
            psnr = float("inf") if mse == 0 else 10.0 * torch.log10(p.abs().max() ** 2 / mse).item()
            print(
                f"TRACE factor={factor:>2}: untraced {untraced:.4f}s | traced {tr:.4f}s "
                f"-> {untraced / tr:.2f}x  (tracers={len(tracers)}, traced-vs-plain PSNR {psnr:.1f} dB)",
                flush=True,
            )
            dec.release_trace()
        except Exception as exc:
            print(f"TRACE factor={factor:>2} FAILED: {str(exc)[:300]}", flush=True)
        del dec
    print("\nsingle-device references (item 1): untraced 1.0980 s unfused, 0.9304 s fused;")
    print("trace on ONE chip was 1.04x. mesh factor=1 untraced was 1.4409 s.")
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
