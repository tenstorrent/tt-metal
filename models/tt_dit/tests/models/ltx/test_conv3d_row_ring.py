# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TT_CONV3D_ROW_RING=1 must give the same bytes as the default reader, in halo mode on a 2x4 submesh.

Each arm runs in the same process: the factory reads the env when it builds the program, so the program
cache is cleared between arms. Both arms are trace-timed; the speedup is printed, not asserted.
"""

import pytest
import torch

import ttnn
from models.tt_dit.utils.conv3d import _BLOCKINGS, aligned_channels

from ..wan2_2.bruteforce_conv3d_sweep import (
    MATH_FIDELITY,
    TRACE_REGION_SIZE,
    HaloSpec,
    _compare_outputs,
    _halo_conv_inputs,
    _invoke,
    _trace_us,
)

# Same rows as _SWEEP_LAYERS_LTX25_544P_145F_HALO: padded per-device input dims, _BLOCKINGS key, logical H/W.
_LAYERS = [
    # (name,   C_in, C_out,  T,   H,  W,  key,            logical_hw)
    ("s3_res", 256, 256, 147, 38, 34, (147, 34, 30), (68, 120)),
    ("s2_res", 512, 512, 75, 38, 34, (75, 34, 30), (68, 120)),
    ("s4_res", 128, 128, 147, 74, 66, (147, 68, 60), (136, 240)),
]


def _all_outputs(args):
    o = _invoke(args)
    try:
        return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(o)]
    finally:
        ttnn.deallocate(o)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("layer_name, C_in, C_out, T, H, W, key, logical_hw", _LAYERS, ids=[l[0] for l in _LAYERS])
def test_conv3d_row_ring_bit_exact(
    mesh_device, device_params, monkeypatch, layer_name, C_in, C_out, T, H, W, key, logical_hw
):
    # A bare 2x4 on the galaxy fails fabric router sync, so open the system mesh and use a 2x4 submesh.
    device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    kernel_size = (3, 3, 3)
    cin, cout, t_blk, h_blk, w_blk = _BLOCKINGS[(4, 8, C_in, C_out, kernel_size, *key)]
    padded_cin = aligned_channels(C_in)
    mesh_mapper = ttnn.ReplicateTensorToMesh(device)

    torch.manual_seed(42)
    tt_input, padding, conv_kwargs = _halo_conv_inputs(
        device, HaloSpec(2, 4, *logical_hw), T, H, W, padded_cin, kernel_size, mesh_mapper
    )
    tt_w = ttnn.from_torch(
        torch.randn(C_out, padded_cin, *kernel_size, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
        mesh_mapper=mesh_mapper,
    )
    tt_w = ttnn.experimental.prepare_conv3d_weights(weight_tensor=tt_w, C_in_block=cin, device=device)
    tt_bias = ttnn.from_torch(
        torch.randn(1, C_out, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        pad_value=0,
        mesh_mapper=mesh_mapper,
    )
    cfg = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=t_blk,
        W_out_block=w_blk,
        H_out_block=h_blk,
        C_out_block=cout,
        C_in_block=cin,
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    )
    ckc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    args = (device, tt_input, tt_w, tt_bias, cfg, C_out, kernel_size, (1, 1, 1), padding, ckc, conv_kwargs)

    outs, us = {}, {}
    for arm in ("0", "1"):
        monkeypatch.setenv("TT_CONV3D_ROW_RING", arm)
        device.clear_program_cache()
        outs[arm] = _all_outputs(args)
        us[arm] = _trace_us(args)
    cmp = _compare_outputs(outs["0"], outs["1"])
    print(
        f"[row_ring] {layer_name} blocking={(cin, cout, t_blk, h_blk, w_blk)} off_us={us['0']:.1f} "
        f"on_us={us['1']:.1f} speedup={us['0'] / us['1']:.3f} identical={cmp['identical']} "
        f"max_abs_diff={cmp['max_abs_diff']} pcc={cmp['pcc']:.6f}"
    )
    assert cmp["identical"], cmp
