# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: FLUX.2 klein's VAE encoder alone on the WH 2x4 preset, encoding as Flux2Pipeline._encode_reference
does. Warms at 1024x1024, then per reference size of KLEIN_SIZES (HxW, comma-separated): the cache entries, kernels
and DRAM the first encode adds, the misses by device op, and the encode's wall time (median of 3).
KLEIN_OUT_DIR gets summary.txt.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import Counter
from pathlib import Path

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

import ttnn

from ....models.vae.vae import flat_to_row_major
from ....models.vae.vae_flux2 import Flux2VaeEncoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....utils.tensor import fast_device_to_host, from_torch
from .device_params import line_params_flux2_klein


def _dram(mesh_device: ttnn.MeshDevice) -> int:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _misses(graph: list) -> Counter:
    out: Counter = Counter()
    stack: list = []
    for node in graph:
        params = (node.get("params") if isinstance(node, dict) else None) or {}
        if node.get("node_type") == "function_start":
            stack.append(params.get("name", "?"))
        elif node.get("node_type") == "function_end" and stack:
            stack.pop()
        if str(params.get("program_cache_hit")).lower() == "false":
            out[params.get("name", "?") + " in " + " > ".join(stack[-4:])] += 1
    return out


def _kernels() -> int:
    root = Path(os.environ.get("TT_METAL_CACHE", Path.home() / ".cache"))
    return sum(1 for d in root.glob("tt-metal-cache*/kernels/*/*"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein]],
    ids=["2x4", "2x2"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_enc_entries(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    out_dir = Path(os.environ.get("KLEIN_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("KLEIN_SIZES", "1024x1184").split(",")]

    torch_vae = AutoencoderKLFlux2.from_pretrained(
        model_location_generator("black-forest-labs/FLUX.2-klein-4B"), subfolder="vae"
    )
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel = Flux2VaeParallelConfig.from_axes(mesh_device, h_axis=1, w_axis=0)
    if os.environ.get("KLEIN_TILES") == "off":
        Flux2VaeEncoder.fixed_slice_hw_per_level = None
    encoder = Flux2VaeEncoder(device=mesh_device, parallel_config=parallel, ccl_manager=ccl)
    encoder.load_torch_state_dict(torch_vae.state_dict())
    encoder.warm_sdpa([(1024, 1024), *sizes])

    def encode(h: int, w: int, *, capture: bool = False) -> Counter:
        pixels = torch.rand(1, h, w, 3) * 2 - 1
        tt_pixels = from_torch(
            pixels.to(torch.bfloat16),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[None, parallel.h_parallel.mesh_axis, parallel.w_parallel.mesh_axis, None],
        )
        if capture:
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            out = encoder.forward(tt_pixels)
            out = flat_to_row_major(
                out,
                encoder.output_hw((tt_pixels.shape[1], tt_pixels.shape[2])),
                **({"runtime_shapes": encoder.runtime_shapes} if hasattr(encoder, "runtime_shapes") else {}),
            )
            fast_device_to_host(out, mesh_device, encoder.output_concat_dims(), ccl_manager=ccl)
        finally:
            graph = ttnn.graph.end_graph_capture() if capture else []
        ttnn.synchronize_device(mesh_device)
        return _misses(graph)

    import traceback

    orig_reshape = ttnn.reshape
    reshape_log = []

    def reshape(x, *args, **kwargs):
        e0 = mesh_device.num_program_cache_entries()
        out = orig_reshape(x, *args, **kwargs)
        if mesh_device.num_program_cache_entries() > e0:
            site = [f"{f.filename.split('tt_dit/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[-4:-1]]
            reshape_log.append(f"  reshape {list(x.shape)} {x.layout} {x.memory_config().memory_layout} -> {args} {site}")
        return out

    ttnn.reshape = reshape
    orig_treshape = ttnn.Tensor.reshape

    def treshape(x, *args, **kwargs):
        e0 = mesh_device.num_program_cache_entries()
        out = orig_treshape(x, *args, **kwargs)
        if mesh_device.num_program_cache_entries() > e0:
            site = [f"{f.filename.split('tt_dit/')[-1]}:{f.lineno}" for f in traceback.extract_stack()[-4:-1]]
            reshape_log.append(f"  Tensor.reshape {list(x.shape)} {x.layout} -> {args} {site}")
        return out

    ttnn.Tensor.reshape = treshape
    from ....layers import conv2d as conv_mod

    orig_cforward = conv_mod.Conv2d.forward

    def cforward(self, x, /, **kwargs):
        e0 = mesh_device.num_program_cache_entries()
        out = orig_cforward(self, x, **kwargs)
        if mesh_device.num_program_cache_entries() > e0:
            reshape_log.append(
                f"  conv {list(x.shape)} {x.layout} {self.in_channels}->{self.out_channels} k{self.kernel_size} "
                f"s{self.stride} slice {self.fixed_slice_hw} -> {list(out.shape)}"
            )
        return out

    conv_mod.Conv2d.forward = cforward
    timing = {"on": False, "conv": Counter(), "n": Counter()}

    def tforward(self, x, /, **kwargs):
        if not timing["on"]:
            return cforward(self, x, **kwargs)
        ttnn.synchronize_device(mesh_device)
        t = time.perf_counter()
        out = cforward(self, x, **kwargs)
        ttnn.synchronize_device(mesh_device)
        key = f"{self.in_channels}->{self.out_channels} s{self.stride[0]} in {x.shape[1]}x{x.shape[2]}"
        timing["conv"][key] += time.perf_counter() - t
        timing["n"][key] += 1
        return out

    conv_mod.Conv2d.forward = tforward
    orig_prep = ttnn.prepare_conv_weights
    import hashlib

    def prep(**kwargs):
        out = orig_prep(**kwargs)
        t = ttnn.to_torch(ttnn.get_device_tensors(out)[0]) if out.storage_type() == ttnn.StorageType.DEVICE else ttnn.to_torch(out)
        digest = hashlib.sha1(t.float().numpy().tobytes()).hexdigest()[:12]
        reshape_log.append(
            f"  prep {kwargs['in_channels']}->{kwargs['out_channels']} at {kwargs['input_height']}x{kwargs['input_width']}: "
            f"{list(t.shape)} {out.layout} {digest}"
        )
        return out

    ttnn.prepare_conv_weights = prep
    lines = reshape_log
    e_build, d_build = mesh_device.num_program_cache_entries(), _dram(mesh_device)
    encode(1024, 1024)
    lines.append(
        f"first encode: {mesh_device.num_program_cache_entries() - e_build} entries, "
        f"{(_dram(mesh_device) - d_build) / 2**20 / mesh_device.get_num_devices():.1f} MiB DRAM per chip kept; "
        f"after build: {e_build} entries, {d_build / 2**20 / mesh_device.get_num_devices():.1f} MiB DRAM per chip"
    )
    encode(1024, 1024)
    for h, w in [(1024, 1024), *sizes]:
        e0, d0, k0 = mesh_device.num_program_cache_entries(), _dram(mesh_device), _kernels()
        misses = encode(h, w, capture=True)
        entries, dram, kernels = mesh_device.num_program_cache_entries() - e0, _dram(mesh_device) - d0, _kernels() - k0
        times = []
        for _ in range(3):
            t = time.perf_counter()
            encode(h, w)
            times.append(time.perf_counter() - t)
        lines.append(
            f"{h}x{w}: {entries} new entries, {kernels} kernels, {dram / 2**20:.2f} MiB DRAM kept (all chips); "
            f"encode {statistics.median(times) * 1e3:.1f} ms median"
        )
        lines.append("  misses: " + ", ".join(f"{k} {v}" for k, v in misses.most_common()))
        timing.update(on=True, conv=Counter(), n=Counter())
        t = time.perf_counter()
        encode(h, w)
        total = time.perf_counter() - t
        timing["on"] = False
        conv_total = sum(timing["conv"].values())
        lines.append(f"  synced encode {total * 1e3:.1f} ms, convs {conv_total * 1e3:.1f} ms, rest {(total - conv_total) * 1e3:.1f} ms")
        for k, v in timing["conv"].items():
            lines.append(f"    {k} x{timing['n'][k]}: {v * 1e3:.1f} ms")
        print("\n".join(lines[-2:]), flush=True)
        (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
