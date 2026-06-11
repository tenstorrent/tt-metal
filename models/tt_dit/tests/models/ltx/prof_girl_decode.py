# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op device-time + FPU-util profile of the *real* girl-clip ``decode_audio`` — real
checkpoint weights, real latent fixture, eager. The full audio path (mel-VAE → vocoder →
BWE) emits >1000 device zones, which overflows the tracy 1000-zone/device buffer and drops
ops. To avoid that we drain the buffer after every block (``AMPBlock1`` for the vocoder and
BWE generator — both are ``Vocoder`` — and ``ResnetBlock`` for the mel-VAE). The flush lives
here, not in the model, because the block forwards run inside the trace-captured region and a
profiler readback is not a traceable op.

Run under:
    python -m tracy -p -r -m pytest <this>::test_prof_girl_decode -k bh_2x4sp1tp0
"""

import os

import numpy as np
import pytest
import torch

import ttnn
from models.tt_dit.layers.audio_ops import Conv1dViaConv3d, Conv2dViaConv3d, ConvTranspose1dViaConv3d
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import default_ltx_checkpoint, default_ltx_gemma
from models.tt_dit.utils.test import line_params


def _flush_after(cls, mesh):
    orig = cls.forward

    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        ttnn.ReadDeviceProfiler(mesh)  # no-op without the profiler
        return out

    cls.forward = forward


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params",
    [[(2, 4), (2, 4), line_params], [(4, 8), (4, 8), line_params]],
    ids=["bh_2x4sp1tp0", "bh_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_prof_girl_decode(mesh_device, mesh_shape, device_params):
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(*mesh_shape))

    # Flush per conv wrapper (every conv in mel-VAE / vocoder / BWE goes through one of these).
    # Per-block was borderline — a window occasionally exceeded the 1000-zone buffer and dropped an op.
    _flush_after(Conv1dViaConv3d, mesh)  # _AlignedOutConv1d inherits this forward
    _flush_after(Conv2dViaConv3d, mesh)  # mel-VAE
    _flush_after(ConvTranspose1dViaConv3d, mesh)  # upsample inner-convs

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),  # lazy shell; never loaded by decode_audio
        sp_axis=1,
        tp_axis=0,
        num_links=2,
        dynamic_load=(mesh_shape != (4, 8)),
        topology=ttnn.Topology.Linear,
        is_fsdp=False,
        run_warmup=False,
        traced=False,
        num_frames=num_frames,
        height=1088,
        width=1920,
    )

    lat = os.environ.get("AUDIO_LATENT") or os.path.join(os.path.dirname(__file__), "fixtures", "girl_audio_latent.npy")
    latent = torch.from_numpy(np.load(lat)).float()

    # Single decode (cold: weight load + compile is host-side, so per-op DEVICE times are
    # valid). One decode halves the captured zone count vs cold+warm → faster post-processing.
    pipeline.decode_audio(latent, num_frames, fps=24.0)
    ttnn.ReadDeviceProfiler(mesh)
    pipeline.release_traces()
