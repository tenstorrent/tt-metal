# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One arm of the #82 SFPU-hoist A/B: traced LTX AV block 0 on a 2x4 submesh, Linear, sp1/tp0.

F,H,W = 10,34,60 gives 5100 video tokens per chip, close to the 4x8 S2 block (4845). The arm is set by
the runtime root (TT_METAL_RUNTIME_ROOT) the job points at; this file only times and dumps.
Logs AB_BLOCK arm=<AB_ARM> ms=<min lap>; saves video/audio to $AB_OUT_DIR/blk_<AB_ARM>.pt.
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.tests.models.ltx.test_transformer_ltx import _build_block_trace_setup
from models.tt_dit.utils.tracing import Tracer

F, H, W = 10, 34, 60


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 * 1024 * 1024, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_block_sfpu_ab(mesh_device, device_params, reset_seeds):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    arm = os.environ.get("AB_ARM", "x")
    sp_axis, tp_axis = 1, 0
    tt_block, kw, video_n, audio_n = _build_block_trace_setup(
        mesh_device=mesh_device,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=2,
        topology=ttnn.Topology.Linear,
        F=F,
        H=H,
        W=W,
        checkpoint_variant="fast",
    )
    tracer = Tracer(tt_block.forward, device=mesh_device, prep_run=False, clone_prep_inputs=False)
    tt_block(**kw)
    ttnn.synchronize_device(mesh_device)
    tracer(**kw, traced=True)
    ttnn.synchronize_device(mesh_device)

    n_replay = int(os.environ.get("LTX_BLOCK_REPLAYS", "10"))
    laps = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n_replay):
            result = tracer(**kw, traced=True)
        ttnn.synchronize_device(mesh_device)
        laps.append((time.perf_counter() - t0) * 1000 / n_replay)
    print(f"AB_BLOCK arm={arm} ms={min(laps):.3f} laps={','.join(f'{x:.3f}' for x in laps)}")

    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 3
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    v = ttnn.to_torch(result[0], mesh_composer=composer).squeeze(0)[:, :video_n, :]
    a = ttnn.to_torch(result[1], mesh_composer=composer).squeeze(0)[:, :audio_n, :]
    assert torch.isfinite(v.float()).all() and torch.isfinite(a.float()).all(), "NaN/Inf in block output"
    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        torch.save({"video": v, "audio": a}, os.path.join(out_dir, f"blk_{arm}.pt"))
    tracer.release_trace()
