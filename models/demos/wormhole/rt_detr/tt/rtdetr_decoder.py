# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RT-DETR decoder — hybrid TTNN/torch forward.

import torch
import ttnn


def _layer_norm(x, p, eps=1e-5):
    return ttnn.layer_norm(
        x, epsilon=eps, weight=p.weight, bias=p.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def _split_heads(x, num_heads, b, seq, hidden):
    head_dim = hidden // num_heads
    x = ttnn.reshape(x, (b, seq, num_heads, head_dim))
    return ttnn.transpose(x, 1, 2)


def _self_attention(x, pos, p, device, num_heads=8):
    """Self-attention with pos embed added to Q and K only."""
    x_pos = ttnn.add(x, pos, memory_config=ttnn.L1_MEMORY_CONFIG) if pos is not None else x

    if len(x.shape) == 4:
        b, _, seq, hidden = x.shape
    else:
        b, seq, hidden = x.shape

    q = ttnn.linear(x_pos, p.q.weight, bias=p.q.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    k = ttnn.linear(x_pos, p.k.weight, bias=p.k.bias, memory_config=ttnn.L1_MEMORY_CONFIG)
    v = ttnn.linear(x,     p.v.weight, bias=p.v.bias, memory_config=ttnn.L1_MEMORY_CONFIG)

    q = _split_heads(q, num_heads, b, seq, hidden)
    k = _split_heads(k, num_heads, b, seq, hidden)
    v = _split_heads(v, num_heads, b, seq, hidden)

    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
    out = ttnn.transpose(out, 1, 2)
    out = ttnn.reshape(out, (b, 1, seq, hidden))

    return ttnn.linear(out, p.out_proj.weight, bias=p.out_proj.bias,
                       memory_config=ttnn.L1_MEMORY_CONFIG)


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

    # 2. Cross-attention on CPU (MSDeformableAttention)
    # Pull both the query AND the positional embedding back to PyTorch
    query_torch = ttnn.to_torch(
        query_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
    )[0:1].view(1, 300, 256).float()  
    
    query_pos_torch = ttnn.to_torch(
        query_pos_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
    )[0:1].view(1, 300, 256).float()
    
    memory_torch = ttnn.to_torch(
        memory, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
    )[0:1].squeeze(1).float()

    if valid_mask is not None:
        memory_torch = valid_mask.to(memory_torch.dtype) * memory_torch

    # CRITICAL FIX: Add positional embedding before cross-attention
    q_with_pos = query_torch + query_pos_torch

    with torch.no_grad():
        ca_out = torch_layer.cross_attn(
            query=q_with_pos,               # <- Fixed!
            reference_points=ref_points, 
            value=memory_torch,
            value_spatial_shapes=spatial_shapes, 
            value_mask=None
        )
        
    ca_out_tt = ttnn.from_torch(
        ca_out.reshape(1, 1, 300, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )

    residual = query_tt
    query_tt = _layer_norm(
        ttnn.add(residual, ca_out_tt, memory_config=ttnn.L1_MEMORY_CONFIG),
        tt_params.norm2,
    )

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

    return query_tt


def run_decoder(query_tt, query_pos_tt, torch_decoder, tt_layer_params,
                memory, ref_points, spatial_shapes, device, num_heads=8,
                valid_mask=None):
    from src.zoo.rtdetr.utils import inverse_sigmoid
    
    # We need the parent torch_decoder to access the bbox_head and query_pos_head
    actual_decoder = torch_decoder.decoder if hasattr(torch_decoder, 'decoder') else torch_decoder
    
    # Guard: Ensure we start with the base [1, 300, 4] tensor.
    # If the hook accidentally caught the scaled [1, 300, 3, 4] tensor, strip the level dimension.
    if ref_points.dim() == 4:
        ref_points_detach = ref_points[:, :, 0, :].clone()
    else:
        ref_points_detach = ref_points.clone()

    n_levels = spatial_shapes.shape[0]  # This is 3

    for i, (torch_layer, tt_params) in enumerate(zip(actual_decoder.layers, tt_layer_params)):
        
        # 1. Update query_pos dynamically based on the current layer's ref_points
        with torch.no_grad():
            query_pos = torch_decoder.query_pos_head(ref_points_detach)
            
        query_pos_tt = ttnn.from_torch(
            query_pos.reshape(1, 1, 300, 256).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        
        # 2. Expand reference points for all 3 feature levels -> [1, 300, 3, 4]
        ref_points_input = ref_points_detach.unsqueeze(2).expand(-1, -1, n_levels, -1)

        # 3. Run the decoder layer (feeding the correctly expanded ref_points)
        query_tt = decoder_layer(
            query_tt, query_pos_tt, torch_layer, tt_params,
            memory, ref_points_input, spatial_shapes, device, num_heads,
            valid_mask=valid_mask
        )
        
        # 4. Predict new reference points for the NEXT layer
        query_torch_out = ttnn.to_torch(
            query_tt, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)
        )[0:1].view(1, 300, 256).float()
        
        with torch.no_grad():
            inter_ref_bbox = torch.sigmoid(
                torch_decoder.dec_bbox_head[i](query_torch_out) +
                inverse_sigmoid(ref_points_detach)
            )
        ref_points_detach = inter_ref_bbox

    # Return BOTH the final query and the final reference points
    return query_tt, ref_points_detach
    
def _print_shape(name, tensor):
    """Helper to safely print shapes of either TTNN or PyTorch tensors."""
    if tensor is None:
        print(f"[DEBUG] {name: <30} | None")
        return
    
    # Extract shape, handle ttnn.Shape vs standard tuples
    shape_str = str(tuple(tensor.shape))
    
    # Check if it's a torch tensor or ttnn tensor to print the type
    t_type = "Torch" if isinstance(tensor, torch.Tensor) else "TTNN "
    print(f"[DEBUG] {name: <30} | {t_type} | {shape_str}")


def decoder_layer_debug(query_tt, query_pos_tt, torch_layer, tt_params,
                  memory, ref_points, spatial_shapes, device, num_heads=8, layer_idx=0):
    """Debug version of a single decoder layer with shape printing."""
    print(f"\n{'='*15} LAYER {layer_idx} START {'='*15}")
    
    # 1. Self-attention on TT device

    residual = query_tt
    sa_out = _self_attention(query_tt, query_pos_tt, tt_params.self_attn, device, num_heads)

    query_tt = _layer_norm(
        ttnn.add(residual, sa_out, memory_config=ttnn.L1_MEMORY_CONFIG),
        tt_params.norm1,
    )

    # 2. Cross-attention on CPU (MSDeformableAttention)
    # convert to pytorch tensors
    query_torch = ttnn.to_torch(query_tt).squeeze(1).float()  
    
    memory_torch = ttnn.to_torch(memory).squeeze(1).float()

    with torch.no_grad():
        ca_out = torch_layer.cross_attn(
            query=query_torch, 
            reference_points=ref_points, 
            value=memory_torch,                  
            value_spatial_shapes=spatial_shapes, 
            value_mask=None
        )

    # Unsqueeze back to TTNN expected shape [B, 1, seq, hidden]
    ca_out_tt = ttnn.from_torch(
        ca_out.unsqueeze(1).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    residual = query_tt
    query_tt = _layer_norm(
        ttnn.add(residual, ca_out_tt, memory_config=ttnn.L1_MEMORY_CONFIG),
        tt_params.norm2,
    )

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

    return query_tt


def run_decoder_debug(query_tt, query_pos_tt, torch_decoder, tt_layer_params,
                memory, ref_points, spatial_shapes, device, num_heads=8):
    """Debug version to run all 6 decoder layers."""
    print(f"\n{'='*15} RUNNING DECODER DEBUG {'='*15}")
    
    for i, (torch_layer, tt_params) in enumerate(zip(torch_decoder.layers, tt_layer_params)):
        query_tt = decoder_layer_debug(
            query_tt, query_pos_tt, torch_layer, tt_params,
            memory, ref_points, spatial_shapes, device, num_heads, layer_idx=i
        )
    return query_tt