# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Per-layer TP x SP sweep for the LTX-2.3 denoiser on a BH Galaxy 4x8 mesh.

Drives a single video-only ``LTXTransformerBlock`` forward under Tracy so the
per-op device kernel durations can be bucketed into matmul+TP-CCL vs ring
attention vs overhead. TP*SP = 32 is fixed; every row reshapes the same 32
physical chips into a different (rows, cols) and picks which axis carries SP.

Construction (block, random scaled weights, inputs) is reused verbatim from
``test_transformer_ltx`` so op sizes/mem-configs match production. PCC is not
checked here — device timing is value-independent.

Run one config under the profiler, e.g.:
    python3 -m tracy -p -r -o generated/profiler/ltx_tpsp/tp4_sp8 \
        -a device_kernel_duration -t 5000 \
        -m "pytest models/tt_dit/tests/models/ltx/test_perf_ltx_layer_tpsp.py -k 'tp4_sp8 and stage_2'"
"""

import pytest
import torch

import ttnn
from tracy import signpost
from models.tt_dit.utils.tensor import bf16_tensor, bf16_tensor_2dshard
from models.tt_dit.utils.mochi import get_rot_transformation_mat
from models.tt_dit.utils.test import line_params, ring_params

from models.tt_dit.tests.models.ltx.test_transformer_ltx import (
    DIM,
    NUM_HEADS,
    CTX_DIM,
    PROMPT_LEN,
    _sp_pad_len,
    _pad_seq_dim,
    _make_ccl_manager,
    _make_parallel_config,
    _make_tt_block,
    _make_diffusers_video_block,
    _scale_init_,
    _convert_diffusers_video_block_to_tt,
    _video_rope_freqs,
    _tt_rope,
)

# TP*SP=32 splits. mesh_shape=(rows=axis0, cols=axis1); SP on axis1 unless noted.
# num_links=2 and Ring mirror the production BH 4x8 LTX config (pipeline_ltx.py device_configs).
# altaxis rows carry the SAME (TP,SP) degrees but place SP on the other physical torus axis,
# isolating the 8x4-torus placement effect. SP=1 (tp32_sp1) is expected to fail: LTX video
# self-attention masks padded keys only via ring SDPA's logical_n, which needs sp>1.
_SWEEP = [
    pytest.param((1, 32), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="tp1_sp32"),
    pytest.param((2, 16), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="tp2_sp16"),
    pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="tp4_sp8"),
    pytest.param((8, 4), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="tp8_sp4"),
    pytest.param((16, 2), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="tp16_sp2"),
    pytest.param((32, 1), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="tp32_sp1"),
    # Axis-placement variants for the two balanced contenders: transpose the mesh and swap
    # sp/tp axes so the SP ring lands on the opposite physical axis.
    pytest.param((8, 4), 0, 1, 2, ring_params, ttnn.Topology.Ring, id="tp4_sp8_altaxis"),
    pytest.param((4, 8), 0, 1, 2, ring_params, ttnn.Topology.Ring, id="tp8_sp4_altaxis"),
]

_SHAPES = [
    pytest.param(19, 17, 30, id="stage_1"),  # F*H*W = 9690 real; SP-padded to 9728
    pytest.param(19, 34, 60, id="stage_2"),  # F*H*W = 38760 real; SP-padded to 38912
]


@torch.no_grad()
@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    _SWEEP,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("F", "H", "W"), _SHAPES)
def test_ltx_layer_tpsp(mesh_device, sp_axis, tp_axis, num_links, topology, F, H, W, reset_seeds):
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    video_N_real = F * H * W
    video_N = _sp_pad_len(video_N_real, sp_factor)
    assert video_N % (32 * sp_factor) == 0

    ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
    parallel_config = _make_parallel_config(mesh_device, sp_axis, tp_axis)
    tt_block = _make_tt_block(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=False,
        has_audio=False,
    )

    # Random scaled weights (default init overflows the AdaLN chains); numerics irrelevant to timing.
    dummy = _make_diffusers_video_block()
    _scale_init_(dummy)
    conv = _convert_diffusers_video_block_to_tt(dummy.state_dict(), num_heads=NUM_HEADS, head_dim=DIM // NUM_HEADS)
    tt_block.load_torch_state_dict({k: v.detach().clone() for k, v in conv.items()})
    del dummy

    x = torch.randn(1, video_N_real, DIM, dtype=torch.float32)
    context = torch.randn(1, PROMPT_LEN, CTX_DIM, dtype=torch.float32)
    temb = torch.randn(1, 1, 9 * DIM, dtype=torch.float32)
    prompt_temb = torch.randn(1, 1, 2 * DIM, dtype=torch.float32)

    spatial = _pad_seq_dim(x, video_N, dim=1).unsqueeze(0)
    tt_spatial = bf16_tensor_2dshard(spatial, device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
    tt_prompt = bf16_tensor(context.unsqueeze(0), device=mesh_device)
    tt_temb = bf16_tensor(
        temb.reshape(9, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device, mesh_axis=tp_axis, shard_dim=3
    )
    tt_prompt_temb = bf16_tensor(prompt_temb.reshape(2, DIM).unsqueeze(1).unsqueeze(1), device=mesh_device)
    tt_cos, tt_sin = _tt_rope(
        _video_rope_freqs, F, H, W, mesh_device=mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, pad_to=video_N
    )
    tt_trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    forward_kwargs = dict(
        video_1BND=tt_spatial,
        video_prompt=tt_prompt,
        video_temb=tt_temb,
        video_N=video_N_real,
        video_rope_cos=tt_cos,
        video_rope_sin=tt_sin,
        trans_mat=tt_trans_mat,
        video_prompt_temb=tt_prompt_temb,
    )

    for _ in range(2):
        tt_out = tt_block(**forward_kwargs)
    ttnn.synchronize_device(mesh_device)
    signpost("start")
    tt_out = tt_block(**forward_kwargs)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3
    tt_v = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
    ).squeeze(0)[:, :video_N_real, :]
    assert torch.isfinite(tt_v).all(), "output NaN/Inf"
