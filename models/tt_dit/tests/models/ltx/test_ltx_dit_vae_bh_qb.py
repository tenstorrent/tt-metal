# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BH QB (4-chip 2x2) smoke test: load real 22B LTX weights via FSDP and run the
full DiT denoise + VAE decode using dummy (zero) prompt embeddings. Bypasses the
gated Gemma text encoder (gemma_path=None) so it needs only the LTX checkpoint."""
import os

import pytest

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXPipeline
from models.tt_dit.utils.test import line_params


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [[(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True]],
    ids=["2x2sp0tp1"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_dit_vae_warmup(mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp):
    sub = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    ckpt = os.environ.get("LTX_CHECKPOINT", "Lightricks/LTX-2.3:ltx-2.3-22b-dev.safetensors")
    nf = int(os.environ.get("NUM_FRAMES", "25"))
    h = int(os.environ.get("HEIGHT", "256"))
    w = int(os.environ.get("WIDTH", "256"))
    LTXPipeline.create_pipeline(
        mesh_device=sub,
        checkpoint_name=ckpt,
        gemma_path=None,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        is_fsdp=is_fsdp,
        topology=topology,
        num_frames=nf,
        height=h,
        width=w,
        run_warmup=True,
    )
    print("DIT+VAE WARMUP OK", flush=True)
