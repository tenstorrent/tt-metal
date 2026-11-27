# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: FLUX.2 klein's VAE decoder alone on the WH 2x4 preset (rows over mesh axis 1, columns over
axis 0), with the Klein VAE weights. Warms at 1024x1024, then per size of KLEIN_SIZES (HxW,
comma-separated):

- the program-cache misses of each conv2d call of the first decode, by device op (graph capture),
- the cache entries and the DRAM the first decode keeps (output freed),
- the host time of the ttnn.conv2d calls (synchronized before each call, not after),
- the decode's wall time, median of KLEIN_RUNS (default 5) after one warm run.

KLEIN_TILES sets the decoder's fixed 2D conv slices per level (HxW per chip, comma-separated, latent level first), or
"off" for conv2d's own slicing.
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

from ....layers import conv2d as conv_mod
from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from .device_params import line_params_flux2_klein


def _dram(mesh_device: ttnn.MeshDevice) -> int:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _misses(graph: list) -> Counter:
    out: Counter = Counter()
    for node in graph:
        params = (node.get("params") if isinstance(node, dict) else None) or {}
        if str(params.get("program_cache_hit")).lower() == "false":
            out[params.get("name", "?")] += 1
    return out


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein]],
    ids=["2x4", "2x2"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_conv_entries(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    out_dir = Path(os.environ.get("KLEIN_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("KLEIN_SIZES", "1024x1184").split(",")]
    runs = int(os.environ.get("KLEIN_RUNS", "5"))

    torch_vae = AutoencoderKLFlux2.from_pretrained(
        model_location_generator("black-forest-labs/FLUX.2-klein-4B"), subfolder="vae"
    )
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    tiles = os.environ.get("KLEIN_TILES")
    if tiles == "off":
        Flux2VaeDecoder.fixed_slice_hw_per_level = None
    elif tiles:
        Flux2VaeDecoder.fixed_slice_hw_per_level = tuple(tuple(int(v) for v in t.split("x")) for t in tiles.split(","))
    print(f"fixed slices per level: {Flux2VaeDecoder.fixed_slice_hw_per_level}")
    decoder = Flux2VaeDecoder(
        out_channels=3,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        z_channels=32,
        device=mesh_device,
        parallel_config=Flux2VaeParallelConfig.from_axes(mesh_device, h_axis=1, w_axis=0),
        ccl_manager=ccl,
    )
    decoder.load_torch_state_dict(torch_vae.state_dict())
    decoder.warm_sdpa([(1024, 1024), *sizes])

    state = {"capture": False, "per_call": [], "conv_s": 0.0}
    orig_forward = conv_mod.Conv2d.forward
    orig_conv2d = ttnn.conv2d

    def conv2d(*args, **kwargs):
        ttnn.synchronize_device(mesh_device)
        t = time.perf_counter()
        out = orig_conv2d(*args, **kwargs)
        state["conv_s"] += time.perf_counter() - t
        return out

    profile = os.environ.get("KLEIN_PROFILE") == "1"
    device_ns: dict[str, Counter] = {"conv": Counter(), "other": Counter()}

    def read_profiler(kind: str) -> None:
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
        for chip, programs in ttnn.get_latest_programs_perf_data().items():
            for p in programs:
                device_ns[kind][chip] += p.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration

    def forward(self, x, /, **kwargs):
        if profile and state.get("profiling"):
            read_profiler("other")
            out = orig_forward(self, x, **kwargs)
            read_profiler("conv")
            return out
        if not state["capture"]:
            return orig_forward(self, x, **kwargs)
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            out = orig_forward(self, x, **kwargs)
        finally:
            graph = ttnn.graph.end_graph_capture()
        state["per_call"].append(
            ("x".join(str(v) for v in x.shape), self.in_channels, self.out_channels, _misses(graph))
        )
        return out

    conv_mod.Conv2d.forward = forward
    ttnn.conv2d = conv2d

    def decode(h: int, w: int) -> None:
        z = torch.randn(1, (h // 16) * (w // 16), 128)
        tt_z = decoder.preprocess_and_unpatchify(z, height=h // 8, width=w // 8)
        out = decoder.forward(tt_z)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(out)
        ttnn.deallocate(tt_z)

    lines = [f"fixed slices per level: {Flux2VaeDecoder.fixed_slice_hw_per_level}"]
    try:
        decode(1024, 1024)
        decode(1024, 1024)
        for h, w in [(1024, 1024), *sizes]:
            e0, d0 = mesh_device.num_program_cache_entries(), _dram(mesh_device)
            state.update(capture=True, per_call=[], conv_s=0.0)
            if os.environ.get("KLEIN_ALL_MISSES") == "1":
                state["capture"] = False
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                try:
                    decode(h, w)
                finally:
                    graph = ttnn.graph.end_graph_capture()
                stack, all_misses = [], Counter()
                for node in graph:
                    params = (node.get("params") if isinstance(node, dict) else None) or {}
                    if node.get("node_type") == "function_start":
                        stack.append(params.get("name", "?"))
                    elif node.get("node_type") == "function_end" and stack:
                        stack.pop()
                    if str(params.get("program_cache_hit")).lower() == "false":
                        all_misses[params.get("name", "?") + " in " + " > ".join(stack[-4:])] += 1
                lines.append(f"{h}x{w} all misses: " + ", ".join(f"{k} {v}" for k, v in all_misses.most_common()))
            else:
                decode(h, w)
            state["capture"] = False
            entries, dram = mesh_device.num_program_cache_entries() - e0, _dram(mesh_device) - d0
            first_conv_s = state["conv_s"]
            decode(h, w)
            times, conv_s = [], []
            for _ in range(runs):
                state["conv_s"] = 0.0
                ttnn.synchronize_device(mesh_device)
                t = time.perf_counter()
                decode(h, w)
                times.append(time.perf_counter() - t)
                conv_s.append(state["conv_s"])
            if profile:
                # Device kernel time of one decode (after the timed runs), per chip: the conv2d calls' programs and
                # the rest.
                read_profiler("other")
                for c in device_ns.values():
                    c.clear()
                state["profiling"] = True
                decode(h, w)
                read_profiler("other")
                state["profiling"] = False
                conv_ms = [v / 1e6 for v in device_ns["conv"].values()]
                other_ms = [v / 1e6 for v in device_ns["other"].values()]
                total_ms = [a + b for a, b in zip(conv_ms, other_ms)]
                lines.append(
                    f"{h}x{w} device: conv2d {statistics.mean(conv_ms):.2f} ms mean per chip (max {max(conv_ms):.2f}), "
                    f"decoder {statistics.mean(total_ms):.2f} ms mean (max {max(total_ms):.2f})"
                )
            by_op: Counter = Counter()
            for _, _, _, m in state["per_call"]:
                by_op.update(m)
            lines.append(
                f"{h}x{w}: {entries} new entries, {dram / 2**20:.2f} MiB DRAM kept (all chips), "
                f"conv2d misses {sum(by_op.values())}; decode {statistics.median(times) * 1e3:.1f} ms median "
                f"(runs {[round(v * 1e3, 1) for v in times]}); conv2d host {statistics.median(conv_s) * 1e3:.1f} ms "
                f"per decode over {len(state['per_call'])} calls (first run {first_conv_s * 1e3:.1f} ms)"
            )
            lines.append("  by op: " + ", ".join(f"{k} {v}" for k, v in by_op.most_common()))
            for shape, cin, cout, m in state["per_call"]:
                if m:
                    lines.append(f"  conv {shape} {cin}->{cout}: " + ", ".join(f"{k} {v}" for k, v in m.most_common()))
            print("\n".join(lines[-4:]), flush=True)
            (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    finally:
        conv_mod.Conv2d.forward = orig_forward
        ttnn.conv2d = orig_conv2d
    print("\n".join(lines))
