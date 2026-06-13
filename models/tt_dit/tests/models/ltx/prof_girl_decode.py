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
    [[(1, 4), (1, 4), line_params], [(2, 4), (2, 4), line_params], [(4, 8), (4, 8), line_params]],
    ids=["bh_1x4sp1tp0", "bh_2x4sp1tp0", "bh_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_prof_girl_decode(mesh_device, mesh_shape, device_params):
    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(*mesh_shape))

    # Per-conv ReadDeviceProfiler flushes bound the device zone buffer, but on a 32-chip mesh each
    # is a full mesh sync (poisons OP-TO-OP LATENCY and writes a huge device log). Set
    # LTX_PROF_NOFLUSH=1 and rely on a large --op-support-count for a clean streaming capture.
    if os.environ.get("LTX_PROF_NOFLUSH") != "1":
        _flush_after(Conv1dViaConv3d, mesh)  # _AlignedOutConv1d inherits this forward
        _flush_after(Conv2dViaConv3d, mesh)  # mel-VAE
        _flush_after(ConvTranspose1dViaConv3d, mesh)  # upsample inner-convs

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    # PROF_TRACED=1 captures the TRACED vocoder graph instead of eager. Used to settle the
    # dispatch paradox: compare the per-op device-active sum eager-vs-traced at constant op
    # count. If traced RAISES device-active, the wall-vs-device gap is device-side (per-op
    # sync / scheduling) and the durable fix is op-count reduction, not host-dispatch removal.
    prof_traced = os.environ.get("PROF_TRACED") == "1"
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
        traced=prof_traced,
        audio_only=True,  # skip the 22B transformer/VAE build+prime decode_audio never uses
        num_frames=num_frames,
        height=1088,
        width=1920,
    )

    lat = os.environ.get("AUDIO_LATENT") or os.path.join(os.path.dirname(__file__), "fixtures", "girl_audio_latent.npy")
    latent = torch.from_numpy(np.load(lat)).float()

    # PROF_WARM=1 (default): an untimed cold decode does the one-time device-side mesh-workload
    # assembly first, then ReadDeviceProfiler drains those zones, so the profiled decode below
    # captures WARM steady-state op times + clean OP-TO-OP gaps (not assembly overhead).
    # PROF_WARM=0 profiles the cold decode (per-op DEVICE times still valid; fewer zones to drain).
    if os.environ.get("PROF_WARM", "1") != "0":
        pipeline.decode_audio(latent, num_frames, fps=24.0)
        ttnn.ReadDeviceProfiler(mesh)  # discard cold-assembly zones
    pipeline.decode_audio(latent, num_frames, fps=24.0)
    ttnn.ReadDeviceProfiler(mesh)
    pipeline.release_traces()
