# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced vs eager conv VAE decode on a 2x4 submesh at 544x960/145f, in one process.

On 2x4 this gives each chip the same shard as 1088x1920/145f on the production 4x8. The decoder is built
with LTX_VIDEO_VAE_TRACE=1 and toggled the way the pipeline does it: eager while ``_vae_traced`` is False
(warmup + timed decodes), then marked warm so the next call captures and later calls replay. Prints AB
timing lines and the trace size (TRACE bank allocation), saves $AB_OUT_DIR/yuv_{eager,traced}.pt and
asserts both outputs are identical.
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.tests.models.ltx.test_vae_ltx_fold_time_pad_ab import NF, TIMED_DECODES, H, W, _latent

TRACE_REGION = int(os.environ.get("LTX_TRACE_REGION", "500000000"))


def _trace_bytes(mesh_device) -> int:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
    return view.total_bytes_allocated_per_bank * view.num_banks


def _timed(dec, mesh_device, lat, n):
    times = []
    out = None
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = dec(lat, output_type="yuv")
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
    return torch.as_tensor(out).clone(), times


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_trace_ab(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    os.environ["LTX_VIDEO_VAE_TRACE"] = "1"
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )

    torch.manual_seed(42)
    tdec = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=_require_diffusers_ltx_vae(),
    ).eval()
    state = tdec.state_dict()
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=2, mesh_axis=0),
        width_parallel=ParallelFactor(factor=4, mesh_axis=1),
    )
    dec = vae_ltx.LTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        num_frames=NF,
        height=H,
        width=W,
        mesh_device=mesh_device,
        parallel_config=pc,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2),
    )
    dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))
    assert dec.trace_decode and not dec._vae_traced

    lat = _latent()
    dec(lat, output_type="yuv")  # warmup: kernel compile + program cache
    eager, t_eager = _timed(dec, mesh_device, lat, TIMED_DECODES)
    print(f"AB arm=eager decode_s={' '.join(f'{t:.4f}' for t in t_eager)} min={min(t_eager):.4f}")

    trace0 = _trace_bytes(mesh_device)
    dec._vae_traced = True
    captured, t_capture = _timed(dec, mesh_device, lat, 1)  # prep run + capture + first execute
    trace_bytes = _trace_bytes(mesh_device) - trace0
    print(
        f"AB trace_bytes={trace_bytes} ({trace_bytes / 1e6:.1f} MB) region={TRACE_REGION} capture_s={t_capture[0]:.4f}"
    )
    assert dec._decode_tracer is not None and dec._decode_tracer._trace_ids is not None
    traced, t_traced = _timed(dec, mesh_device, lat, TIMED_DECODES)
    print(f"AB arm=traced decode_s={' '.join(f'{t:.4f}' for t in t_traced)} min={min(t_traced):.4f}")

    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(eager, os.path.join(out_dir, "yuv_eager.pt"))
        torch.save(traced, os.path.join(out_dir, "yuv_traced.pt"))
    for name, out in (("capture", captured), ("traced", traced)):
        same = out.shape == eager.shape and torch.equal(out, eager)
        diff = (out.int() - eager.int()).abs().max().item() if out.shape == eager.shape else "shape"
        print(f"AB_CMP {name}_vs_eager identical={same} max_abs_diff={diff} shape={tuple(out.shape)} dtype={out.dtype}")
    dec.release_trace()
    assert torch.equal(captured, eager) and torch.equal(traced, eager)
