# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

_precision_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, 
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True
)

def _layer_norm(x, p, eps=1e-5):
    return ttnn.layer_norm(
        x, epsilon=eps, weight=p.weight, bias=p.bias,
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

    q = ttnn.linear(x_pos, p.q.weight, bias=p.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    k = ttnn.linear(x_pos, p.k.weight, bias=p.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    v = ttnn.linear(x_flat, p.v.weight, bias=p.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)

    if pos is not None:
        ttnn.deallocate(x_pos)

    head_dim = hidden // num_heads

    q = ttnn.transpose(ttnn.reshape(q, (b, seq, num_heads, head_dim)), 1, 2)
    k = ttnn.transpose(ttnn.reshape(k, (b, seq, num_heads, head_dim)), 1, 2)
    v = ttnn.transpose(ttnn.reshape(v, (b, seq, num_heads, head_dim)), 1, 2)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)

    ttnn.deallocate(q)
    ttnn.deallocate(k)
    ttnn.deallocate(v)

    out = ttnn.transpose(out, 1, 2)
    out = ttnn.reshape(out, (b, 1, seq, hidden))

    out_proj = ttnn.linear(out, p.out_proj.weight, bias=p.out_proj.bias, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_precision_config)
    ttnn.deallocate(out)
    
    return out_proj


def decoder_layer(query_tt, query_pos_tt, torch_layer, tt_params,
                  memory, ref_points, spatial_shapes, device, num_heads=8,
                  valid_mask=None):
    """Single decoder layer: self-attn (TT) + cross-attn (torch) + FFN (TT). """
    
    # 1. Self-attention on TT device
    residual = query_tt
    sa_out = _self_attention(query_tt, query_pos_tt, tt_params.self_attn, device, num_heads)
    
    query_tt = _layer_norm(
        ttnn.add(residual, sa_out, memory_config=ttnn.L1_MEMORY_CONFIG),
        tt_params.norm1,
    )
    ttnn.deallocate(sa_out)  # Explicitly freeing SRAM memory

    # 2. Cross-attention on CPU (MSDeformableAttention)
    # Add query and pos on the device to save a full PCIe D2H transfer
    q_with_pos_tt = ttnn.add(query_tt, query_pos_tt, memory_config=ttnn.L1_MEMORY_CONFIG)

    # Transfer the single combined tensor, saving PCIe bandwidth
    q_with_pos_torch = ttnn.to_torch(
        q_with_pos_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, 'get_num_devices') else None
    )[0:1].view(1, 300, 256).float()  
    
    ttnn.deallocate(q_with_pos_tt)

    memory_torch = ttnn.to_torch(
        memory, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, 'get_num_devices') else None
    )[0:1].squeeze(1).float()

    if valid_mask is not None:
        memory_torch = valid_mask.to(memory_torch.dtype) * memory_torch

    with torch.no_grad():
        ca_out = torch_layer.cross_attn(
            query=q_with_pos_torch,         
            reference_points=ref_points, 
            value=memory_torch,
            value_spatial_shapes=spatial_shapes, 
            value_mask=None
        )
        
    ca_out_tt = ttnn.from_torch(
        ca_out.reshape(1, 1, 300, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, 'get_num_devices') else None,
    )

    residual = query_tt
    query_tt = _layer_norm(
        ttnn.add(residual, ca_out_tt, memory_config=ttnn.L1_MEMORY_CONFIG),
        tt_params.norm2,
    )
    ttnn.deallocate(ca_out_tt)

    # 3. FFN on TT device
    residual = query_tt
    ffn = ttnn.linear(query_tt, tt_params.linear1.weight, bias=tt_params.linear1.bias,
                      activation="relu", memory_config=ttnn.L1_MEMORY_CONFIG)
    ffn = ttnn.linear(ffn, tt_params.linear2.weight, bias=tt_params.linear2.bias,
                      memory_config=ttnn.L1_MEMORY_CONFIG)
                      
    query_tt = _layer_norm(
        ttnn.add(residual, ffn, memory_config=ttnn.L1_MEMORY_CONFIG),
        tt_params.norm3,
    )
    ttnn.deallocate(ffn)

    return query_tt


def run_decoder(query_tt, query_pos_tt, torch_decoder, tt_layer_params,
                memory, ref_points, spatial_shapes, device, num_heads=8,
                valid_mask=None):
    from src.zoo.rtdetr.utils import inverse_sigmoid
    
    actual_decoder = torch_decoder.decoder if hasattr(torch_decoder, 'decoder') else torch_decoder
    
    if ref_points.dim() == 4:
        ref_points_detach = ref_points[:, :, 0, :].clone()
    else:
        ref_points_detach = ref_points.clone()

    n_levels = spatial_shapes.shape[0]

    for i, (torch_layer, tt_params) in enumerate(zip(actual_decoder.layers, tt_layer_params)):
        
        with torch.no_grad():
            query_pos = torch_decoder.query_pos_head(ref_points_detach)
            
        query_pos_tt = ttnn.from_torch(
            query_pos.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if hasattr(device, 'get_num_devices') else None,
        )
        
        ref_points_input = ref_points_detach.unsqueeze(2).expand(-1, -1, n_levels, -1)

        query_tt = decoder_layer(
            query_tt, query_pos_tt, torch_layer, tt_params,
            memory, ref_points_input, spatial_shapes, device, num_heads,
            valid_mask=valid_mask
        )
        
        query_torch_out = ttnn.to_torch(
            query_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0) if hasattr(device, 'get_num_devices') else None
        )[0:1].view(1, 300, 256).float()
        
        with torch.no_grad():
            inter_ref_bbox = torch.sigmoid(
                torch_decoder.dec_bbox_head[i](query_torch_out) +
                inverse_sigmoid(ref_points_detach)
            )
        ref_points_detach = inter_ref_bbox

    return query_tt, ref_points_detach