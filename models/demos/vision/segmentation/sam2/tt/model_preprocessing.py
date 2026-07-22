# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as _ns

import torch

import ttnn
from models.demos.vision.segmentation.sam2.tt.tt_hiera import HIERA_TINY_BLOCK_PLAN


def _to_conv_weight(torch_tensor, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


def _linear_params(weight, bias, device, dtype=ttnn.bfloat16):
    return _ns(
        weight=ttnn.from_torch(
            weight.T.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        bias=(
            None
            if bias is None
            else ttnn.from_torch(
                bias.reshape(1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ),
    )


def _linear_ns(layer, device, dtype=ttnn.bfloat16):
    return _linear_params(layer.weight.data, layer.bias.data, device, dtype)


def _conv_ns(layer, device=None, *, keep_tiled_bias=False):
    namespace = _ns(
        weight=_to_conv_weight(layer.weight.data),
        bias=_to_conv_weight(layer.bias.data.reshape(1, 1, 1, -1)),
    )
    if keep_tiled_bias:
        namespace.post_bias = ttnn.from_torch(
            layer.bias.data.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return namespace


def _fold_layer_norm_affine(linear_weight, linear_bias, norm_weight, norm_bias):
    weight_f32 = linear_weight.float()
    folded_weight = weight_f32 * norm_weight.float().unsqueeze(0)
    folded_bias = linear_bias.float() + weight_f32 @ norm_bias.float()
    return folded_weight.to(linear_weight.dtype), folded_bias.to(linear_bias.dtype)


def _compact_window_maps(height, width, window_size):
    raster = (
        torch.arange(height * width)
        .reshape(height // window_size, window_size, width // window_size, window_size)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )
    return raster.tolist(), torch.argsort(raster).tolist()


def _preprocess_hiera_window_indices(device):
    windows, local = torch.arange(25), torch.arange(224)
    row = (windows // 5)[:, None] * 14 + local[None, :] // 14
    col = (windows % 5)[:, None] * 14 + local[None, :] % 14
    valid = (local[None, :] < 196) & (row < 64) & (col < 64)
    gather = torch.where(valid, row * 64 + col, 4096).reshape(-1)
    flat_valid = valid.reshape(-1)
    raster = torch.empty(4096, dtype=torch.int64)
    raster[gather[flat_valid]] = torch.arange(gather.numel())[flat_valid]
    w14_padded224_mask = torch.zeros((25, 1, 224, 224), dtype=torch.bfloat16)
    w14_padded224_mask[:, :, :, 196:] = -10000.0
    w14_padded224_from_raster64, raster64_from_w14_padded224 = gather.tolist(), raster.tolist()
    w8_from_raster256, raster256_from_w8 = _compact_window_maps(256, 256, 8)
    _, raster128_from_w4 = _compact_window_maps(128, 128, 4)
    _, raster64_from_w2 = _compact_window_maps(64, 64, 2)
    raster32_from_w7 = [
        (row // 7 * 5 + col // 7) * 49 + row % 7 * 7 + col % 7 for row in range(32) for col in range(32)
    ]
    index_tensors = [
        ttnn.from_torch(
            torch.tensor(indices, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for indices in (
            w14_padded224_from_raster64,
            raster64_from_w14_padded224,
            raster32_from_w7,
            w8_from_raster256,
            raster256_from_w8,
            raster128_from_w4,
            raster64_from_w2,
        )
    ]
    return _ns(
        w14_padded224_from_raster64=index_tensors[0],
        raster64_from_w14_padded224=index_tensors[1],
        w14_padded224_mask=ttnn.from_torch(
            w14_padded224_mask,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        raster32_from_w7=index_tensors[2],
        w8_from_raster256=index_tensors[3],
        raster256_from_w8=index_tensors[4],
        raster128_from_w4=index_tensors[5],
        raster64_from_w2=index_tensors[6],
    )


def preprocess_hiera(ref_trunk, device):
    p = _ns()
    pe = ref_trunk.patch_embed.projection
    pe_weight = (
        torch.nn.functional.pad(pe.weight.data, (0, 1, 0, 1))
        .reshape(96, 3, 2, 4, 2, 4)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(96, 48, 2, 2)
    )

    p.patch_embed = _ns(
        proj=_ns(
            weight=_to_conv_weight(pe_weight),
            bias=_to_conv_weight(pe.bias.data.reshape(1, 1, 1, -1)),
        )
    )
    position = torch.nn.functional.interpolate(ref_trunk.pos_embed.data, size=(256, 256), mode="bicubic")
    position = position + ref_trunk.pos_embed_window.data.tile((1, 1, 32, 32))
    position = position.permute(0, 2, 3, 1).contiguous().reshape(1, 1, 256 * 256, 96)
    p.pos_tokens = ttnn.from_torch(
        position,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    p.window_indices = _preprocess_hiera_window_indices(device)
    blocks = []
    for blk, spec in zip(ref_trunk.blocks, HIERA_TINY_BLOCK_PLAN):
        attention_weight_dtype = spec.attention_weight_dtype or spec.weight_dtype
        explicit_norm1 = spec.explicit_norm1
        norm1_weight = blk.layer_norm1.weight.data
        norm1_bias = blk.layer_norm1.bias.data
        qkv_weight = blk.attn.qkv.weight.data
        qkv_bias = blk.attn.qkv.bias.data
        mlp0_weight = blk.mlp.proj_in.weight.data
        mlp0_bias = blk.mlp.proj_in.bias.data
        shortcut_weight = blk.proj.weight.data if blk.dim != blk.dim_out else None
        shortcut_bias = blk.proj.bias.data if blk.dim != blk.dim_out else None
        if not explicit_norm1:
            qkv_weight, qkv_bias = _fold_layer_norm_affine(
                qkv_weight,
                qkv_bias,
                norm1_weight,
                norm1_bias,
            )
        mlp0_weight, mlp0_bias = _fold_layer_norm_affine(
            mlp0_weight,
            mlp0_bias,
            blk.layer_norm2.weight.data,
            blk.layer_norm2.bias.data,
        )
        if shortcut_weight is not None and not explicit_norm1:
            shortcut_weight, shortcut_bias = _fold_layer_norm_affine(
                shortcut_weight,
                shortcut_bias,
                norm1_weight,
                norm1_bias,
            )
        if explicit_norm1:
            padding_values = torch.zeros_like(norm1_weight)
            norm1 = _ns(
                weight=ttnn.from_torch(
                    norm1_weight.reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                bias=ttnn.from_torch(
                    norm1_bias.reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        else:
            padding_values = torch.where(
                norm1_weight.abs() > 1e-12,
                -norm1_bias / norm1_weight,
                torch.zeros_like(norm1_bias),
            )
            norm1 = _ns(weight=None, bias=None)
        proj = None
        if shortcut_weight is not None:
            proj = _linear_params(shortcut_weight, shortcut_bias, device, spec.weight_dtype)
        mlp_layers = {
            "0": _linear_params(mlp0_weight, mlp0_bias, device, spec.weight_dtype),
            "1": _linear_ns(blk.mlp.proj_out, device, attention_weight_dtype),
        }
        blocks.append(
            _ns(
                norm1=norm1,
                norm2=_ns(weight=None, bias=None),
                input_pad=ttnn.from_torch(
                    padding_values.reshape(1, 1, 1, -1).repeat(1, 1, 32, 1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                attn=_ns(
                    qkv=_linear_params(qkv_weight, qkv_bias, device, attention_weight_dtype),
                    proj=_linear_ns(blk.attn.proj, device, attention_weight_dtype),
                ),
                proj=proj,
                mlp=_ns(layers=mlp_layers),
            )
        )
    p.blocks = blocks
    return p


def preprocess_fpn_neck(ref_neck, high_res_projection_convs):
    conv_s0, conv_s1 = high_res_projection_convs
    fold_by_index = {
        len(ref_neck.convs) - 1: conv_s0,
        len(ref_neck.convs) - 2: conv_s1,
    }
    convs = []
    for index, conv in enumerate(ref_neck.convs):
        weight = conv.weight.data
        bias = conv.bias.data
        if index in fold_by_index:
            second = fold_by_index[index]
            first_weight = weight[:, :, 0, 0].float()
            second_weight = second.weight.data[:, :, 0, 0].float()
            weight = (second_weight @ first_weight).to(weight.dtype)[:, :, None, None]
            bias = (second.bias.data.float() + second_weight @ bias.float()).to(bias.dtype)
        convs.append(
            _ns(
                conv=_ns(
                    weight=_to_conv_weight(weight),
                    bias=_to_conv_weight(bias.reshape(1, 1, 1, -1)),
                    out_channels=weight.shape[0],
                )
            )
        )
    return _ns(convs=convs)


def preprocess_prompt_encoder(prompt_encoder, device):
    def _mask_linear(layer):
        return _ns(
            weight=ttnn.from_torch(
                layer.weight.data.reshape(layer.out_channels, -1).T.contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            bias=ttnn.from_torch(
                layer.bias.data.reshape(1, -1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            kernel_size=layer.kernel_size[0],
        )

    height, width = prompt_encoder.image_embedding_size
    x = (torch.arange(width, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, dtype=torch.float32) + 0.5) / height
    coords = torch.stack((x[None].expand(height, width), y[:, None].expand(height, width)), dim=-1)
    projected = (coords * 2.0 - 1.0) @ prompt_encoder.shared_embedding.positional_embedding.data.float()
    dense_pe = torch.cat((torch.sin(2.0 * torch.pi * projected), torch.cos(2.0 * torch.pi * projected)), dim=-1)
    no_mask_dense = prompt_encoder.no_mask_embed.weight.data.reshape(1, 1, -1).expand(1, height * width, -1)

    return _ns(
        dense_pe_seq=ttnn.from_torch(
            dense_pe.reshape(1, height * width, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        no_mask_dense_seq=ttnn.from_torch(
            no_mask_dense,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        positional_embedding=prompt_encoder.shared_embedding.positional_embedding.data,
        point_embeddings=prompt_encoder.point_embed.weight.data,
        not_a_point_embed=prompt_encoder.not_a_point_embed.weight.data,
        mask_embed=_ns(
            conv1=_mask_linear(prompt_encoder.mask_embed.conv1),
            norm1=_ln_ns(prompt_encoder.mask_embed.layer_norm1, device, dtype=ttnn.float32),
            conv2=_mask_linear(prompt_encoder.mask_embed.conv2),
            norm2=_ln_ns(prompt_encoder.mask_embed.layer_norm2, device, dtype=ttnn.float32),
            conv3=_mask_linear(prompt_encoder.mask_embed.conv3),
        ),
    )


def _mlp_ns(module, device):
    layers = (module.proj_in, *module.layers, module.proj_out)
    return _ns(layers={str(index): _linear_ns(layer, device) for index, layer in enumerate(layers)})


def _batched_mlp_ns(modules, device):
    module_layers = [(module.proj_in, *module.layers, module.proj_out) for module in modules]
    layers = {}
    for index in range(len(module_layers[0])):
        current_layers = [layers_for_module[index] for layers_for_module in module_layers]
        weights = [layer.weight.data.T.contiguous() for layer in current_layers]
        biases = [layer.bias.data.reshape(1, -1) for layer in current_layers]
        weight = torch.stack(weights, dim=0)
        bias = torch.stack(biases, dim=0)
        layers[str(index)] = _ns(
            weight=ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            bias=ttnn.from_torch(
                bias,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
    return _ns(layers=layers)


def preprocess_mask_decoder(mask_decoder, prompt_encoder, device):
    transformer = mask_decoder.transformer
    output_tokens = torch.cat(
        (
            mask_decoder.obj_score_token.weight.data,
            mask_decoder.iou_token.weight.data,
            mask_decoder.mask_tokens.weight.data,
        )
    ).unsqueeze(0)
    no_prompt_sparse = prompt_encoder.not_a_point_embed.weight.data.reshape(1, 1, -1).expand(1, 2, -1)
    return _ns(
        output_tokens=ttnn.from_torch(
            output_tokens,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        no_prompt_decoder_tokens=ttnn.from_torch(
            torch.cat((output_tokens, no_prompt_sparse), dim=1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        output_upscaling=(
            _conv_ns(mask_decoder.upscale_conv1),
            _ln_ns(mask_decoder.upscale_layer_norm, device),
            _conv_ns(mask_decoder.upscale_conv2),
        ),
        multimask_output_hypernetwork_mlp=_batched_mlp_ns(mask_decoder.output_hypernetworks_mlps[1:], device),
        single_mask_output_hypernetwork_mlp=_mlp_ns(mask_decoder.output_hypernetworks_mlps[0], device),
        dynamic_multimask_via_stability=mask_decoder.dynamic_multimask_via_stability,
        dynamic_multimask_stability_delta=mask_decoder.dynamic_multimask_stability_delta,
        dynamic_multimask_stability_thresh=mask_decoder.dynamic_multimask_stability_thresh,
        iou_prediction_head=_mlp_ns(mask_decoder.iou_prediction_head, device),
        pred_obj_score_head=_mlp_ns(mask_decoder.pred_obj_score_head, device),
        transformer=_ns(
            layers=[
                _ns(
                    self_attn=_attn_ns(
                        layer.self_attn,
                        device,
                        pack_qkv=True,
                        qkv_position_correction=index != 0,
                        weight_dtype=ttnn.bfloat8_b,
                    ),
                    norm1=_ln_ns(layer.layer_norm1, device),
                    cross_attn_token_to_image=_attn_ns(
                        layer.cross_attn_token_to_image, device, pack_kv=True, weight_dtype=ttnn.bfloat8_b
                    ),
                    norm2=_ln_ns(layer.layer_norm2, device),
                    mlp=_mlp_ns(layer.mlp, device),
                    norm3=_ln_ns(layer.layer_norm3, device),
                    norm4=_ln_ns(layer.layer_norm4, device),
                    cross_attn_image_to_token=_attn_ns(
                        layer.cross_attn_image_to_token, device, pack_kv=True, weight_dtype=ttnn.bfloat8_b
                    ),
                )
                for index, layer in enumerate(transformer.layers)
            ],
            final_attn_token_to_image=_attn_ns(
                transformer.final_attn_token_to_image, device, pack_kv=True, weight_dtype=ttnn.bfloat8_b
            ),
            norm_final_attn=_ln_ns(transformer.layer_norm_final_attn, device),
        ),
    )


def _attn_ns(
    attn,
    device,
    *,
    pack_qkv=False,
    qkv_position_correction=False,
    pack_kv=False,
    weight_dtype=ttnn.bfloat16,
    include_out_proj=True,
):
    q, k, v, out = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
    q_weight, q_bias = q.weight.data, q.bias.data
    k_weight, k_bias = k.weight.data, k.bias.data
    v_weight, v_bias = v.weight.data, v.bias.data
    out_weight, out_bias = out.weight.data, out.bias.data
    num_heads = attn.num_attention_heads
    semantic_head_dim = attn.internal_dim // num_heads
    padded_head_dim = None

    if semantic_head_dim < 32 and 32 % semantic_head_dim == 0:
        padded_head_dim = 32
        padding = padded_head_dim - semantic_head_dim
        padded_internal_dim = num_heads * padded_head_dim

        def pad_projection(weight, bias):
            weight = torch.nn.functional.pad(
                weight.reshape(num_heads, semantic_head_dim, weight.shape[1]), (0, 0, 0, padding)
            ).reshape(padded_internal_dim, weight.shape[1])
            bias = torch.nn.functional.pad(bias.reshape(num_heads, semantic_head_dim), (0, padding)).reshape(-1)
            return weight, bias

        q_weight, q_bias = pad_projection(q_weight, q_bias)
        k_weight, k_bias = pad_projection(k_weight, k_bias)
        v_weight, v_bias = pad_projection(v_weight, v_bias)
        out_weight = torch.nn.functional.pad(
            out_weight.reshape(out_weight.shape[0], num_heads, semantic_head_dim), (0, padding)
        ).reshape(out_weight.shape[0], padded_internal_dim)
    q_proj = None if pack_qkv else _linear_params(q_weight, q_bias, device, weight_dtype)
    k_proj = None if pack_qkv or pack_kv else _linear_params(k_weight, k_bias, device, weight_dtype)
    qkv_proj = qkv_position_proj = None
    if pack_qkv:
        qkv_proj = _linear_params(
            torch.cat((q_weight, k_weight, v_weight)),
            torch.cat((q_bias, k_bias, v_bias)),
            device,
            weight_dtype,
        )
        if qkv_position_correction:
            qkv_position_proj = _linear_params(
                torch.cat((torch.zeros_like(q_weight), torch.zeros_like(k_weight), v_weight)),
                None,
                device,
                weight_dtype,
            )

    kv_proj = kv_position_proj = None
    if pack_kv:
        projected_head_dim = padded_head_dim or semantic_head_dim
        head_shape = (num_heads, projected_head_dim, k_weight.shape[1])
        k_weight_heads, v_weight_heads = k_weight.reshape(head_shape), v_weight.reshape(head_shape)
        k_bias_heads = k_bias.reshape(num_heads, projected_head_dim)
        v_bias_heads = v_bias.reshape(num_heads, projected_head_dim)
        kv_weight = torch.stack((k_weight_heads, v_weight_heads), dim=1).reshape(-1, k_weight.shape[1])
        kv_bias = torch.stack((k_bias_heads, v_bias_heads), dim=1).reshape(-1)
        kv_position_weight = torch.stack((torch.zeros_like(k_weight_heads), v_weight_heads), dim=1).reshape(
            -1, k_weight.shape[1]
        )
        kv_proj = _linear_params(kv_weight, kv_bias, device, weight_dtype)
        kv_position_proj = _linear_params(kv_position_weight, None, device, weight_dtype)

    return _ns(
        q_proj=q_proj,
        k_proj=k_proj,
        qkv_proj=qkv_proj,
        qkv_position_proj=qkv_position_proj,
        kv_proj=kv_proj,
        kv_position_proj=kv_position_proj,
        out_proj=_linear_params(out_weight, out_bias, device, weight_dtype) if include_out_proj else None,
        num_heads=num_heads,
        padded_head_dim=padded_head_dim,
        semantic_head_dim=semantic_head_dim,
    )


def _ln_ns(ln, device, dtype=ttnn.bfloat16):
    return _ns(
        weight=ttnn.from_torch(
            ln.weight.data,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        bias=ttnn.from_torch(
            ln.bias.data,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def preprocess_memory_encoder(memory_encoder, device):
    blocks = [
        _ns(conv=_conv_ns(layer.conv), norm=_ln_ns(layer.layer_norm, device))
        for layer in memory_encoder.mask_downsampler.layers
    ]
    fuser_layers = []
    for block in memory_encoder.memory_fuser.layers:
        scale = block.scale.data
        output_weight = block.pointwise_conv2.weight.data * scale[:, None]
        output_bias = block.pointwise_conv2.bias.data * scale
        fuser_layers.append(
            _ns(
                dwconv=_conv_ns(block.depthwise_conv, device, keep_tiled_bias=True),
                norm=_ln_ns(block.layer_norm, device),
                pwconv1=_linear_ns(block.pointwise_conv1, device),
                pwconv2=_linear_params(output_weight, output_bias, device),
            )
        )
    return _ns(
        mask_downsampler=_ns(blocks=blocks, final=_conv_ns(memory_encoder.mask_downsampler.final_conv)),
        pix_feat_proj=_conv_ns(memory_encoder.feature_projection),
        fuser=_ns(layers=fuser_layers),
        out_proj=_conv_ns(memory_encoder.projection),
    )


def preprocess_memory_attention(memory_attention, device):
    weight_dtype = ttnn.bfloat16
    layers = []
    for layer in memory_attention.layers:
        cross_attention = _attn_ns(layer.cross_attn_image, device, weight_dtype=weight_dtype, include_out_proj=False)
        value_weight = layer.cross_attn_image.v_proj.weight.data.float()
        value_bias = layer.cross_attn_image.v_proj.bias.data.float()
        output_weight = layer.cross_attn_image.o_proj.weight.data.float()
        output_bias = layer.cross_attn_image.o_proj.bias.data.float()
        latent_output_weight = (output_weight @ value_weight).to(layer.cross_attn_image.o_proj.weight.dtype)
        latent_output_bias = (output_bias + output_weight @ value_bias).to(layer.cross_attn_image.o_proj.bias.dtype)
        cross_attention.latent_out_proj = _linear_params(latent_output_weight, latent_output_bias, device, weight_dtype)
        layers.append(
            _ns(
                self_attn=_attn_ns(layer.self_attn, device, pack_qkv=True, weight_dtype=weight_dtype),
                cross_attn_image=cross_attention,
                norm1=_ln_ns(layer.layer_norm1, device),
                norm2=_ln_ns(layer.layer_norm2, device),
                norm3=_ln_ns(layer.layer_norm3, device),
                linear1=_linear_ns(layer.linear1, device, dtype=weight_dtype),
                linear2=_linear_ns(layer.linear2, device, dtype=weight_dtype),
            )
        )
    cross_attention_modules = [layer.cross_attn_image for layer in memory_attention.layers]
    return _ns(
        layers=layers,
        bank_k_proj=_linear_params(
            torch.cat([attention.k_proj.weight.data for attention in cross_attention_modules], dim=0),
            torch.cat([attention.k_proj.bias.data for attention in cross_attention_modules]),
            device,
            weight_dtype,
        ),
        norm=_ln_ns(memory_attention.layer_norm, device),
    )


def preprocess_sam2_video_encoder_parameters(model, encoder_device):
    return _ns(
        trunk=preprocess_hiera(model.vision_encoder.backbone, encoder_device),
        neck=preprocess_fpn_neck(
            model.vision_encoder.neck,
            high_res_projection_convs=(model.mask_decoder.conv_s0, model.mask_decoder.conv_s1),
        ),
    )


def preprocess_sam2_image_head_parameters(model, image_device):
    return _ns(
        sam_prompt_encoder=preprocess_prompt_encoder(model.prompt_encoder, image_device),
        sam_mask_decoder=preprocess_mask_decoder(model.mask_decoder, model.prompt_encoder, image_device),
        no_mem_embed_dev=ttnn.from_torch(
            model.no_memory_embedding.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=image_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )


def preprocess_sam2_video_tracker_parameters(model, tracker_device):
    positions = []
    for module in (model.vision_encoder.neck.position_encoding, model.memory_encoder.position_encoding):
        channels = module.num_position_features * 2
        position = module(torch.Size((1, channels, 64, 64)), torch.device("cpu"), torch.float32).to(torch.float32)
        positions.append(
            ttnn.from_torch(
                position,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=tracker_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
    return _ns(
        sam_prompt_encoder=preprocess_prompt_encoder(model.prompt_encoder, tracker_device),
        sam_mask_decoder=preprocess_mask_decoder(model.mask_decoder, model.prompt_encoder, tracker_device),
        memory_attention=preprocess_memory_attention(model.memory_attention, tracker_device),
        memory_encoder=preprocess_memory_encoder(model.memory_encoder, tracker_device),
        obj_ptr_proj=_mlp_ns(model.object_pointer_proj, tracker_device),
        vision_pos_dev=positions[0],
        memory_pos_dev=positions[1],
        no_mem_embed_dev=ttnn.from_torch(
            model.no_memory_embedding.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tracker_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        no_obj_ptr_dev=ttnn.from_torch(
            model.no_object_pointer.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tracker_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
        maskmem_tpos_enc_dev=ttnn.from_torch(
            model.memory_temporal_positional_encoding.data,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=tracker_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    )
