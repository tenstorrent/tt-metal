# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Manual bring-up hook for Wan2.2 TI2V-5B on single BH Galaxy.

    pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py -k bh_4x8
"""

import os

import pytest

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ti2v_5b(mesh_device, mesh_shape, topology):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B manual bring-up is targeting BH Galaxy first")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=704,  # 704/16=44 even; 720/16=45 odd breaks patch_size=2 patchify
        width=1280,
        num_frames=21 if os.environ.get("WAN5B_SMOKE") else 121,
        run_warmup=True,
    )
    assert pipeline.transformer_2 is None
    assert pipeline.transformer.dim == 3072
    assert pipeline.transformer.ffn_dim == 14336
