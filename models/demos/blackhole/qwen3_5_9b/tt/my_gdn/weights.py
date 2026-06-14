from dataclasses import dataclass

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.my_gdn.operations import conv1d_weight_taps


@dataclass(frozen=True)
class Qwen35GDNWeights:
    dt_bias: ttnn.Tensor
    neg_A_log_exp: ttnn.Tensor
    w_conv1d: ttnn.Tensor
    w_taps: list[ttnn.Tensor]
    w_norm: ttnn.Tensor
    wo: ttnn.Tensor
    wqkv: ttnn.Tensor
    wz: ttnn.Tensor
    wb: ttnn.Tensor
    wa: ttnn.Tensor


def load_gdn_weights(mesh_device, state_dict, args, dtype=ttnn.bfloat16) -> Qwen35GDNWeights:
    """
    ['dt_bias', 'A_log', 'conv1d.weight', 'norm.weight', 'out_proj.weight', 'in_proj_qkv.weight', 'in_proj_z.weight', 'in_proj_b.weight', 'in_proj_a.weight']
    """
    #
    # The HF state_dict is a flat dict of 1D tensors, all named like in_proj_{qkv,z,a,b} or out_proj or conv1d or A_log or dt_bias or norm. We need to reshape and pack these into the Qwen35GDNWeights structure, converting to ttnn.Tensor along the way.
    wqkv = ttnn.from_torch(
        state_dict["in_proj_qkv.weight"].T.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    wz = ttnn.from_torch(
        state_dict["in_proj_z.weight"].T.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    wa = ttnn.from_torch(
        state_dict["in_proj_a.weight"].T.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    wb = ttnn.from_torch(
        state_dict["in_proj_b.weight"].T.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    wo = ttnn.from_torch(
        state_dict["out_proj.weight"].T.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    w_conv1d = ttnn.from_torch(
        state_dict["conv1d.weight"],
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    w_taps = conv1d_weight_taps(w_conv1d, args.linear_conv_kernel_dim, mesh_device, dtype=dtype)
    neg_A_log_exp = ttnn.from_torch(
        -(state_dict["A_log"].float().exp()),
        device=mesh_device,
        dtype=ttnn.float32,  # A_log.float() in HF otherwise may become -inf
        layout=ttnn.TILE_LAYOUT,
    )
    dt_bias = ttnn.from_torch(
        state_dict["dt_bias"],
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )
    w_norm = ttnn.from_torch(
        state_dict["norm.weight"],
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
    )

    return Qwen35GDNWeights(
        dt_bias=dt_bias,
        neg_A_log_exp=neg_A_log_exp,
        w_conv1d=w_conv1d,
        w_taps=w_taps,
        w_norm=w_norm,
        wo=wo,
        wqkv=wqkv,
        wz=wz,
        wb=wb,
        wa=wa,
    )
