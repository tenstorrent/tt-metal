# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

_precision_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)

# global mesh cache
_mesh_composer = None
_mesh_mapper = None


def _get_mesh_utils(device):
    global _mesh_composer, _mesh_mapper
    if _mesh_composer is None and hasattr(device, "get_num_devices"):
        _mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
        _mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return _mesh_composer, _mesh_mapper


def _layer_norm(x, p, eps=1e-5):
    return ttnn.layer_norm(
        x,
        epsilon=eps,
        weight=p.weight,
        bias=p.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _self_attention(x, pos, p, device, num_heads=8):
    shape = x.shape
    b = shape[0] if len(shape) == 4 else shape[0]
    seq = shape[2] if len(shape) == 4 else shape[1]
    hidden = shape[-1]

    x_flat = ttnn.reshape(x, (b, seq, hidden))

    if pos is not None:
        pos_flat = ttnn.reshape(pos, (b, seq, hidden))
        x_pos = ttnn.add(x_flat, pos_flat, memory_config=ttnn.L1_MEMORY_CONFIG)
    else:
        x_pos = x_flat

    q = ttnn.linear(
        x_pos, p.q.weight, bias=p.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config
    )
    k = ttnn.linear(
        x_pos, p.k.weight, bias=p.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config
    )
    v = ttnn.linear(
        x_flat, p.v.weight, bias=p.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config
    )

    if pos is not None:
        ttnn.deallocate(x_pos)

    head_dim = hidden // num_heads

    q_trans = ttnn.transpose(ttnn.reshape(q, (b, seq, num_heads, head_dim)), 1, 2)
    k_trans = ttnn.transpose(ttnn.reshape(k, (b, seq, num_heads, head_dim)), 1, 2)
    v_trans = ttnn.transpose(ttnn.reshape(v, (b, seq, num_heads, head_dim)), 1, 2)

    out = ttnn.transformer.scaled_dot_product_attention(q_trans, k_trans, v_trans, is_causal=False)

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    out_trans = ttnn.transpose(out, 1, 2)
    out_reshaped = ttnn.reshape(out_trans, (b, 1, seq, hidden))

    out_proj = ttnn.linear(
        out_reshaped,
        p.out_proj.weight,
        bias=p.out_proj.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=_precision_config,
    )
    ttnn.deallocate(out)

    return out_proj


def decoder_layer(
    query_in,
    query_pos_tt,
    torch_layer,
    tt_params,
    memory_torch,
    ref_points,
    spatial_shapes,
    device,
    num_heads=8,
    valid_mask=None,
):
    mesh_composer, mesh_mapper = _get_mesh_utils(device)

    # 1. Self-attention
    sa_out = _self_attention(query_in, query_pos_tt, tt_params.self_attn, device, num_heads)

    add1 = ttnn.add(query_in, sa_out, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(sa_out)

    # 2. LayerNorm 1
    norm1 = _layer_norm(add1, tt_params.norm1)
    ttnn.deallocate(add1)

    # 3. PyTorch CPU Fallback
    query_torch = ttnn.to_torch(norm1, mesh_composer=mesh_composer)[0:1].view(1, 300, 256).float()
    query_pos_torch = ttnn.to_torch(query_pos_tt, mesh_composer=mesh_composer)[0:1].view(1, 300, 256).float()

    with torch.no_grad():
        ca_out = torch_layer.cross_attn(
            query=query_torch + query_pos_torch,
            reference_points=ref_points,
            value=memory_torch,
            value_spatial_shapes=spatial_shapes,
            value_mask=None,
        )

    ca_out_tt = ttnn.from_torch(
        ca_out.reshape(1, 1, 300, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )

    # 4. Residual + LayerNorm 2
    add2 = ttnn.add(norm1, ca_out_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(norm1)
    ttnn.deallocate(ca_out_tt)

    norm2 = _layer_norm(add2, tt_params.norm2)
    ttnn.deallocate(add2)

    # 5. FFN
    ffn1 = ttnn.linear(
        norm2,
        tt_params.linear1.weight,
        bias=tt_params.linear1.bias,
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=_precision_config,
    )
    ffn2 = ttnn.linear(
        ffn1,
        tt_params.linear2.weight,
        bias=tt_params.linear2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=_precision_config,
    )
    ttnn.deallocate(ffn1)

    # 6. Residual + LayerNorm 3
    add3 = ttnn.add(norm2, ffn2, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(norm2)
    ttnn.deallocate(ffn2)

    norm3 = _layer_norm(add3, tt_params.norm3)
    ttnn.deallocate(add3)

    return norm3


def run_decoder(
    query_tt,
    torch_decoder,
    tt_layer_params,
    memory_torch,
    ref_points,
    spatial_shapes,
    device,
    num_heads=8,
    valid_mask=None,
):
    from src.zoo.rtdetr.utils import inverse_sigmoid

    actual_decoder = torch_decoder.decoder if hasattr(torch_decoder, "decoder") else torch_decoder

    mesh_composer, mesh_mapper = _get_mesh_utils(device)

    if valid_mask is not None:
        memory_torch = valid_mask.to(memory_torch.dtype) * memory_torch

    if ref_points.dim() == 4:
        ref_points_detach = torch.sigmoid(ref_points[:, :, 0, :].clone())
    else:
        ref_points_detach = torch.sigmoid(ref_points.clone())

    n_levels = len(spatial_shapes)

    current_query = query_tt

    for i, (torch_layer, tt_params) in enumerate(zip(actual_decoder.layers, tt_layer_params)):
        with torch.no_grad():
            query_pos = torch_decoder.query_pos_head(ref_points_detach)

        query_pos_tt = ttnn.from_torch(
            query_pos.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

        ref_points_input = ref_points_detach.unsqueeze(2).expand(-1, -1, n_levels, -1)

        next_query = decoder_layer(
            current_query,
            query_pos_tt,
            torch_layer,
            tt_params,
            memory_torch,
            ref_points_input,
            spatial_shapes,
            device,
            num_heads,
            valid_mask=None,
        )

        query_torch_out = ttnn.to_torch(next_query, mesh_composer=mesh_composer)[0:1].view(1, 300, 256).float()

        with torch.no_grad():
            inter_ref_bbox = torch.sigmoid(
                torch_decoder.dec_bbox_head[i](query_torch_out) + inverse_sigmoid(ref_points_detach)
            )
        ref_points_detach = inter_ref_bbox

        # Cleanup loop tensors
        ttnn.deallocate(query_pos_tt)

        if i > 0:
            ttnn.deallocate(current_query)

        current_query = next_query

    return current_query, ref_points_detach
