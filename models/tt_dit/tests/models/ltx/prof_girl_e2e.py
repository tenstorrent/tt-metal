# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-op device-time profile of the *full* girl-clip AV e2e — real checkpoint, real
DEFAULT_LTX_PROMPT, eager. The complete pipeline (Gemma encode → stage-1 denoise → latent
upsample → stage-2 denoise → VAE video decode → audio decode) emits far more device zones
than the profiler's ~12000-marker/core (1000-program) buffer holds, so a single readback at
teardown drops thousands of markers. We drain the buffer mid-run at every marker-heavy unit:

  - after every transformer block (``LTXTransformerBlock``) — one denoise STEP alone emits
    >10000 device programs/core, so draining only per step overflows; and a buffer big enough
    to hold a whole step is too large to read back within the per-op timeout. The block is the
    right granularity: its ~hundreds of programs fit the small default buffer, so each readback
    stays cheap. (Plus one drain per ``inner_step`` for the adaln/proj tail after the loop.)
  - after every ``LTXCausalConv3d`` (VAE video decoder + latent upsampler),
  - after every audio conv (``Conv1dViaConv3d``/``Conv2dViaConv3d``/``ConvTranspose1dViaConv3d``
    — mel-VAE, vocoder, BWE), and after the (cache-skippable) Gemma encode.

Each drain does ``ttnn.synchronize_device`` BEFORE ``ttnn.ReadDeviceProfiler``: reading
mid-flight truncates an open CCL/halo zone and aborts the dump with
``TT_FATAL: End marker found without a corresponding start marker`` (see prof_vocoder_forward).
The sync quiesces the mesh so every marker pair is closed before the read.

The flushes live here, not in the pipeline/model, because a profiler readback is not a
traceable op. That forces EAGER (LTX_TRACED=0): under trace the denoise is replayed as one
captured region that cannot be drained mid-run — exactly what overflowed before.

Run under (``--timeout=0`` lifts pytest.ini's 300 s per-test cap; the e2e is much longer):
    python -m tracy -p -r -m pytest <this>::test_prof_girl_e2e -k bh_2x4sp1tp0 -s --timeout=0
"""

import os

import pytest

import ttnn
from models.tt_dit.layers.audio_ops import Conv1dViaConv3d, Conv2dViaConv3d, ConvTranspose1dViaConv3d
from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerBlock, LTXTransformerModel
from models.tt_dit.models.vae.vae_ltx import LTXCausalConv3d
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.tests.models.ltx.test_pipeline_ltx_distilled import line_trace_params
from models.tt_dit.utils.ltx import DEFAULT_LTX_PROMPT, default_ltx_checkpoint, default_ltx_gemma


def _drain(mesh):
    ttnn.synchronize_device(mesh)  # quiesce so no open CCL/halo zone is read mid-flight
    ttnn.ReadDeviceProfiler(mesh)  # no-op without the profiler


def _flush_after(cls, method, mesh):
    orig = getattr(cls, method)

    def wrapped(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        _drain(mesh)
        return out

    setattr(cls, method, wrapped)


def _flush_instance_after(obj, method, mesh):
    orig = getattr(obj, method)

    def wrapped(*args, **kwargs):
        out = orig(*args, **kwargs)
        _drain(mesh)
        return out

    setattr(obj, method, wrapped)


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_prof_girl_e2e(mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp):
    parent_mesh = mesh_device
    mesh = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    seed = int(os.environ.get("SEED", "10"))
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)
    output_path = os.environ.get("OUTPUT_PATH", os.path.join("tmp", "ltx_av_girl.mp4"))

    # Mid-run drains bound the buffer only when the ops run EAGER; a captured trace replays
    # without hitting these Python forwards, so LTX_TRACED must stay 0. Each _drain syncs first.
    _flush_after(LTXTransformerBlock, "forward", mesh)  # each DiT block; one step alone overflows
    _flush_after(LTXTransformerModel, "inner_step", mesh)  # per-step adaln/proj tail after the loop
    _flush_after(LTXCausalConv3d, "forward", mesh)  # VAE video decoder + latent upsampler
    _flush_after(Conv1dViaConv3d, "forward", mesh)  # audio vocoder/BWE (_AlignedOutConv1d inherits)
    _flush_after(Conv2dViaConv3d, "forward", mesh)  # mel-VAE
    _flush_after(ConvTranspose1dViaConv3d, "forward", mesh)  # vocoder upsample inner-convs

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=False,  # eager needs no trace warmup
        traced=False,  # MUST be eager so the mid-run drains above actually fire
        num_frames=num_frames,
        height=height,
        width=width,
    )

    # A single Gemma encode forward is itself marker-heavy (skipped entirely on a cached
    # prompt); drain after it so its zones don't pile onto the first denoise step's budget.
    _flush_instance_after(pipeline, "encode_prompts", mesh)

    # PROF_WARM=1 runs an untimed generate first, then drains, so the profiled pass captures
    # warm steady-state op times. Default off: a full e2e generate is minutes long, and per-op
    # DEVICE times are valid cold once the JIT cache is warm — two passes risks the timeout.
    if os.environ.get("PROF_WARM", "0") != "0":
        pipeline.generate(prompt, output_path=output_path, num_frames=num_frames, height=height, width=width, seed=seed)
        _drain(mesh)  # discard warm-up zones

    pipeline.generate(prompt, output_path=output_path, num_frames=num_frames, height=height, width=width, seed=seed)
    _drain(mesh)
    pipeline.release_traces()
