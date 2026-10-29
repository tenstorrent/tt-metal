# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BEVFormer's parameters for the TTNN port: each reference module's weights as its TTNN counterpart
takes them, on device, with whatever the forward would otherwise compute from the weights alone
already computed. ``create_bevformer_parameters`` builds the whole detector's.

Sections:
- Shared helpers: Linears, LayerNorms and FFNs
- Deformable attention and encoder
- Perception transformer
- Backbone and FPN
- Decoder
- Head
- Detector
- Frame inputs
"""

import math
from types import SimpleNamespace

import torch

import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters
from models.experimental.bevformer.model_config import (
    CENTERS_FIRST,
    GRID_DTYPE,
    WEIGHT_DTYPE,
    tt_fpn_kwargs,
    tt_resnet_kwargs,
)
from models.experimental.bevformer.tt.tt_common import is_pointwise
from models.experimental.bevformer.tt.tt_modulated_deform_conv import GRID_SAMPLE_C_CAP, grid_offset_order
from models.experimental.bevformer.tt.tt_resnet import TTResNet

# --- Shared helpers ---------------------------------------------------------------------------


def preprocess_linear_weight(weight, *, device, dtype=WEIGHT_DTYPE):
    """A Linear's weight transposed to ``(in, out)``, as ``ttnn.linear`` takes it."""
    return ttnn.from_torch(weight.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def preprocess_linear_bias(bias, *, device, dtype=WEIGHT_DTYPE):
    """A Linear's bias as a ``(1, out)`` row."""
    return ttnn.from_torch(bias.reshape((1, -1)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def linear_params(weight, bias, device, dtype):
    """A Linear's ``weight`` and ``bias`` as ``ttnn.linear`` takes them, on ``device``."""
    return SimpleNamespace(
        weight=preprocess_linear_weight(weight, dtype=dtype, device=device),
        bias=None if bias is None else preprocess_linear_bias(bias, dtype=dtype, device=device),
    )


def preprocess_layer_norm_parameters(layer_norm, *, device, dtype=WEIGHT_DTYPE):
    """A LayerNorm's weight and bias, and its ``eps``: ``ttnn.layer_norm`` defaults to 1e-12."""

    def upload(tensor):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return SimpleNamespace(
        eps=layer_norm.eps,
        weight=upload(layer_norm.weight),
        bias=None if layer_norm.bias is None else upload(layer_norm.bias),
    )


def preprocess_ffn_parameters(ffn, *, device, dtype=WEIGHT_DTYPE):
    """The encoder's and decoder's mmcv-style ``FFN``: ``layers`` is
    ``Sequential(Sequential(Linear, ReLU), Linear)``, so the two Linears are ``layers[0][0]`` and
    ``layers[1]``."""
    first, second = ffn.layers[0][0], ffn.layers[1]
    return SimpleNamespace(
        linear1=linear_params(first.weight, first.bias, device, dtype),
        linear2=linear_params(second.weight, second.bias, device, dtype),
    )


# --- Deformable attention and encoder ---------------------------------------------------------


def create_ms_deformable_attention_parameters(torch_model, *, device, dtype=WEIGHT_DTYPE):
    """``reference.ms_deformable_attention.MSDeformableAttention``'s value, offset, attention and
    output Linears, as TTMSDeformableAttention takes them."""
    return SimpleNamespace(
        **{
            name: linear_params(getattr(torch_model, name).weight, getattr(torch_model, name).bias, device, dtype)
            for name in ("value_proj", "sampling_offsets", "attention_weights", "output_proj")
        }
    )


def create_spatial_cross_attention_parameters(sca, device, dtype=WEIGHT_DTYPE):
    """``reference.spatial_cross_attention.SpatialCrossAttention`` as TTSpatialCrossAttention
    takes it: the merged-slots ``output_proj`` and the nested ``MSDeformableAttention3D``'s
    value, offset and attention Linears (it has no output projection)."""
    deform = sca.deformable_attention
    return SimpleNamespace(
        output_proj=linear_params(sca.output_proj.weight, sca.output_proj.bias, device, dtype),
        deformable_attention=SimpleNamespace(
            **{
                name: linear_params(getattr(deform, name).weight, getattr(deform, name).bias, device, dtype)
                for name in ("value_proj", "sampling_offsets", "attention_weights")
            }
        ),
    )


def create_temporal_self_attention_parameters(tsa, device, dtype=WEIGHT_DTYPE):
    """``reference.temporal_self_attention.TemporalSelfAttention`` as TTTemporalSelfAttention
    takes it. The offset and attention Linears emit channels head-major,
    (head, queue, level, point[, xy]); they are reordered queue-major, so each stacked map's
    channels are one contiguous half of the row."""
    heads, queue = tsa.num_heads, tsa.num_bev_queue
    per_head = tsa.num_levels * tsa.num_points

    def queue_major(linear, width):
        order = torch.arange(heads * queue * per_head * width).view(heads, queue, per_head * width)
        order = order.permute(1, 0, 2).reshape(-1)
        return linear_params(linear.weight[order], linear.bias[order], device, dtype)

    return SimpleNamespace(
        value_proj=linear_params(tsa.value_proj.weight, tsa.value_proj.bias, device, dtype),
        sampling_offsets=queue_major(tsa.sampling_offsets, 2),
        attention_weights=queue_major(tsa.attention_weights, 1),
        output_proj=linear_params(tsa.output_proj.weight, tsa.output_proj.bias, device, dtype),
    )


def create_bevformer_layer_parameters(layer, device, dtype=WEIGHT_DTYPE):
    """``reference.encoder.BEVFormerLayer`` as TTBEVFormerLayer takes it, whose FFN reads its
    Linears as dicts."""
    ffn = preprocess_ffn_parameters(layer.ffns[0], device=device, dtype=dtype)
    norm1, norm2, norm3 = (preprocess_layer_norm_parameters(norm, device=device, dtype=dtype) for norm in layer.norms)
    return SimpleNamespace(
        temporal_self_attention=create_temporal_self_attention_parameters(layer.attentions[0], device, dtype),
        spatial_cross_attention=create_spatial_cross_attention_parameters(layer.attentions[1], device, dtype),
        ffn=SimpleNamespace(linear1=vars(ffn.linear1), linear2=vars(ffn.linear2)),
        norm1=norm1,
        norm2=norm2,
        norm3=norm3,
    )


def create_bevformer_encoder_parameters(encoder, device, dtype=WEIGHT_DTYPE):
    """``reference.encoder.BEVFormerEncoder`` as TTBEVFormerEncoder takes it: ``layer_{i}`` per
    layer, and in ``config`` the constructor's arguments the encoder was built with, so the port
    cannot disagree with the model."""
    config = SimpleNamespace(
        num_layers=len(encoder.layers),
        embed_dims=encoder.embed_dims,
        num_heads=encoder.num_heads,
        num_levels=encoder.num_levels,
        # The layer gives every attention num_points; the self-attention takes its own.
        num_points=encoder.num_points,
        temporal_num_points=encoder.tsa_num_points,
        num_cams=encoder.num_cams,
        pc_range=list(encoder.pc_range),
        num_points_in_pillar=encoder.num_points_in_pillar,
        feedforward_channels=encoder.layers[0].ffns[0].layers[0][0].out_features,
    )
    return SimpleNamespace(
        config=config,
        **{
            f"layer_{i}": create_bevformer_layer_parameters(layer, device, dtype)
            for i, layer in enumerate(encoder.layers)
        },
    )


# --- Perception transformer --------------------------------------------------------------------


def create_perception_transformer_parameters(transformer, device, dtype=WEIGHT_DTYPE):
    """``reference.perception_transformer.PerceptionTransformer`` as TTPerceptionTransformer takes
    it: the encoder's parameters, the CAN-bus MLP, and per FPN level the camera embeddings plus the
    level's embedding, ``(1, num_cams, 1, C)``."""
    mlp = transformer.can_bus_mlp
    level_cams_embeds = [
        ttnn.from_torch(
            (transformer.cams_embeds + level_embed)[None, :, None, :],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        for level_embed in transformer.level_embeds
    ]
    return SimpleNamespace(
        config=SimpleNamespace(
            embed_dims=transformer.embed_dims,
            num_cams=transformer.num_cams,
            rotate_center=tuple(transformer.rotate_center),
        ),
        encoder=create_bevformer_encoder_parameters(transformer.encoder, device, dtype),
        can_bus_mlp=SimpleNamespace(
            linear1=linear_params(mlp[0].weight, mlp[0].bias, device, dtype),
            linear2=linear_params(mlp[2].weight, mlp[2].bias, device, dtype),
            norm=preprocess_layer_norm_parameters(mlp.norm, device=device, dtype=dtype),
        ),
        level_cams_embeds=level_cams_embeds,
    )


# --- Backbone and FPN -------------------------------------------------------------------------

# Each BatchNorm is folded into the conv before it. The backbone's conv input shapes are recorded
# from one reference forward; the FPN's follow from its input levels' shapes. The weights stay on
# the host: conv2d lays them out for its input on the device. A stride-1 1x1 conv whose activations
# are in DRAM runs as a matmul instead and gets a Linear's weight; the DCNs get their device op's
# matmul weight chunks and an offset conv that emits the offsets in its (x, y) order and grid units.


def _conv_parameters(weight, bias, linear_dtype, staging_dtype):
    """A conv's torch ``weight`` and ``bias`` as TTConv2D takes them: in ``staging_dtype`` for
    conv2d to lay out, or, with a ``linear_dtype``, a Linear's ``(in, out)`` weight in that dtype
    and ``(1, out)`` bias, laid out for ``ttnn.linear``."""
    if linear_dtype is not None:
        return {
            "weight": ttnn.from_torch(weight.flatten(1).T.contiguous(), dtype=linear_dtype, layout=ttnn.TILE_LAYOUT),
            "bias": ttnn.from_torch(bias.reshape((1, -1)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        }
    return {
        "weight": ttnn.from_torch(weight, dtype=staging_dtype),
        "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=staging_dtype),
    }


def _dcn_parameters(conv, bn, offset_args):
    """A DCNv2 conv and its offset conv as TTModulatedDeformConv2dPack takes them, ``bn`` folded in.

    The weight becomes the device DCN's matmul weight, one ``(K*K*c_chunk, C_out)`` block per
    chunk of ``GRID_SAMPLE_C_CAP`` input channels, rows kk-major as grid_sample emits its samples.
    The offset conv's offset rows are reordered to the device DCN's (x, y) order and scaled by
    ``2 / [W, H]`` to grid_sample's units; both are exact on the weights and drop a reorder and
    a multiply from every forward.
    """
    if conv.groups != 1 or conv.deform_groups != 1:
        raise ValueError(f"the device DCN takes groups 1 and deform_groups 1, got {conv.groups}, {conv.deform_groups}")
    weight, bias = (t.detach() for t in fold_batch_norm2d_into_conv2d(conv, bn))
    c_out, c_in, kernel_h, kernel_w = weight.shape
    c_chunk = min(c_in, GRID_SAMPLE_C_CAP)
    if c_in % c_chunk != 0:
        raise ValueError(f"C_in {c_in} is not a multiple of the grid_sample chunk {c_chunk}")
    weight_chunks = [
        ttnn.from_torch(
            weight[:, start : start + c_chunk].permute(2, 3, 1, 0).reshape(-1, c_out).contiguous(),
            dtype=WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
        )
        for start in range(0, c_in, c_chunk)
    ]

    order = grid_offset_order(kernel_h * kernel_w)
    offset_weight = conv.conv_offset.weight.detach()
    offset_bias = conv.conv_offset.bias.detach()
    scale = torch.ones(offset_weight.shape[0])
    scale[: len(order) : 2] = 2.0 / offset_args.input_width
    scale[1 : len(order) : 2] = 2.0 / offset_args.input_height
    offset_weight = torch.cat([offset_weight[order], offset_weight[len(order) :]]) * scale.view(-1, 1, 1, 1)
    offset_bias = torch.cat([offset_bias[order], offset_bias[len(order) :]]) * scale
    return {
        "weight_chunks": weight_chunks,
        "bias": ttnn.from_torch(bias.reshape((1, -1)), dtype=WEIGHT_DTYPE, layout=ttnn.TILE_LAYOUT),
        "conv_offset": _conv_parameters(offset_weight, offset_bias, None, ttnn.float32),
    }


def _resnet_preprocessor(model, resnet, conv_args, tt_kwargs):
    """``preprocess_model_parameters``' hook for ``resnet``, given its recorded ``conv_args`` and
    TTResNet's ``tt_kwargs``; the hook sees every submodule and answers for ``resnet`` only."""
    if model is not resnet:
        return {}
    dram_activation_stages = tt_kwargs["dram_activation_stages"]

    def conv_parameters(conv, bn, dram_activation):
        weight, bias = fold_batch_norm2d_into_conv2d(conv, bn)
        linear_dtype = tt_kwargs["weights_dtype"] if dram_activation and is_pointwise(conv) else None
        return _conv_parameters(weight, bias, linear_dtype, ttnn.float32)

    parameters = {"conv1": conv_parameters(model.conv1, model.bn1, False)}
    for layer_idx in range(1, 5):
        prefix = f"layer{layer_idx}"
        parameters[prefix] = {}
        for block_idx, block in enumerate(getattr(model, prefix)):
            dram_convs = TTResNet.block_dram_convs(layer_idx - 1, block_idx, dram_activation_stages)
            block_parameters = parameters[prefix][block_idx] = {}
            for conv_name in ["conv1", "conv2", "conv3"]:
                conv = getattr(block, conv_name)
                bn = getattr(block, f"bn{conv_name[-1]}")
                # A DCNv2 conv carries the conv that predicts its offsets and mask.
                if hasattr(conv, "conv_offset"):
                    offset_args = conv_args[prefix][block_idx][conv_name].conv_offset
                    block_parameters[conv_name] = _dcn_parameters(conv, bn, offset_args)
                else:
                    block_parameters[conv_name] = conv_parameters(conv, bn, conv_name in dram_convs)

            ds = getattr(block, "downsample", None)
            if ds is not None:
                assert (
                    isinstance(ds, torch.nn.Sequential) and len(ds) == 2
                ), f"expected a (conv, norm) downsample, got {ds}"
                block_parameters["downsample"] = conv_parameters(ds[0], ds[1], "downsample" in dram_convs)
    return {"res_model": parameters}


def create_resnet_parameters(model, input_tensor, tt_kwargs=None):
    """Preprocess backbone weights and record per-conv shapes from one reference forward, for
    ``TTResNet`` with ``tt_kwargs``, ``model_config.tt_resnet_kwargs()`` by default, which the
    result carries: which 1x1 convs run as matmuls decides their weights' layout.

    ``conv_args`` pins every conv's input height, width and batch to ``input_tensor``'s
    shape, so the TT backbone built from these parameters only accepts that shape.
    """
    tt_kwargs = tt_resnet_kwargs() if tt_kwargs is None else tt_kwargs
    conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=lambda module, name: _resnet_preprocessor(module, model, conv_args, tt_kwargs),
    )
    parameters.conv_args = conv_args
    parameters.tt_kwargs = tt_kwargs
    return parameters


def create_fpn_parameters(model, input_tensors, tt_kwargs=None):
    """Preprocess FPN weights and record every conv's geometry and input shape, for ``TTFPN``
    with ``tt_kwargs``, ``model_config.tt_fpn_kwargs()`` by default, which the result carries.

    ``input_tensors`` are the NCHW backbone outputs the FPN will run on. A conv reads the
    level it belongs to, except the extra output convs: the first reads the last level's
    output and each later one reads the previous extra conv's output.

    ``conv_args`` holds, per conv, the attributes ``TTConv2D`` reads, as
    ``create_resnet_parameters`` records them for the backbone.
    """
    level_shapes = [(t.shape[0], t.shape[2], t.shape[3]) for t in input_tensors]
    shapes = list(level_shapes)
    batch, height, width = level_shapes[-1]
    for fpn_conv in model.fpn_convs[len(level_shapes) :]:
        shapes.append((batch, height, width))
        conv = fpn_conv.conv
        height = (height + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        width = (width + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1

    def conv_args(conv_module, shape):
        conv = conv_module.conv
        batch, height, width = shape
        return SimpleNamespace(
            conv=SimpleNamespace(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                batch_size=batch,
                input_height=height,
                input_width=width,
            )
        )

    tt_kwargs = tt_fpn_kwargs() if tt_kwargs is None else tt_kwargs
    dram_activation_levels = tt_kwargs["dram_activation_levels"]

    def conv_weights(conv_module, dram_activation):
        conv = conv_module.conv
        linear_dtype = tt_kwargs["weights_dtype"] if dram_activation and is_pointwise(conv) else None
        return {"conv": _conv_parameters(conv.weight.detach(), conv.bias.detach(), linear_dtype, ttnn.bfloat16)}

    def preprocessor(module, name):
        if module is not model:
            return {}
        return {
            "fpn": {
                "lateral_convs": {
                    str(i): conv_weights(lateral, i in dram_activation_levels)
                    for i, lateral in enumerate(module.lateral_convs)
                },
                "fpn_convs": {
                    str(i): conv_weights(fpn_conv, i < len(level_shapes) and i in dram_activation_levels)
                    for i, fpn_conv in enumerate(module.fpn_convs)
                },
            }
        }

    parameters = preprocess_model_parameters(initialize_model=lambda: model, custom_preprocessor=preprocessor)
    parameters["conv_args"] = SimpleNamespace(
        lateral_convs=[conv_args(lateral, level_shapes[i]) for i, lateral in enumerate(model.lateral_convs)],
        fpn_convs=[conv_args(fpn_conv, shapes[i]) for i, fpn_conv in enumerate(model.fpn_convs)],
        add_extra_convs=model.add_extra_convs,
        relu_before_extra_convs=model.relu_before_extra_convs,
    )
    parameters["tt_kwargs"] = tt_kwargs
    return parameters


# --- Decoder ----------------------------------------------------------------------------------


def _self_attn_parameters(mha, device, dtype):
    """Q and K of ``nn.MultiheadAttention`` as one Linear, with Q pre-scaled by ``head_dim**-0.5``.

    Q and K both project ``query + query_pos``; V projects ``query`` alone.
    """
    head_dim = mha.embed_dim // mha.num_heads
    q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
    q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
    scale = head_dim**-0.5
    return SimpleNamespace(
        num_heads=mha.num_heads,
        qk_proj=linear_params(torch.cat([q_w * scale, k_w]), torch.cat([q_b * scale, k_b]), device, dtype),
        v_proj=linear_params(v_w, v_b, device, dtype),
        out_proj=linear_params(mha.out_proj.weight, mha.out_proj.bias, device, dtype),
    )


def _cross_attn_parameters(msda, device, dtype):
    """The decoder's single-level cross-attention as TTMSDeformableAttention takes it, with its config."""
    if msda.num_levels != 1:
        raise ValueError(f"the decoder cross-attention is single-level, got {msda.num_levels} levels")
    params = create_ms_deformable_attention_parameters(msda, device=device, dtype=dtype)
    params.config = msda.config
    return params


def _decoder_layer_parameters(layer, device, dtype):
    """``reference.decoder.DetrTransformerDecoderLayer`` as TTDetectionTransformerDecoder's layer
    takes it."""
    return SimpleNamespace(
        self_attn=_self_attn_parameters(layer.attentions[0].attn, device, dtype),
        cross_attn=_cross_attn_parameters(layer.attentions[1], device, dtype),
        ffn=preprocess_ffn_parameters(layer.ffns[0], device=device, dtype=dtype),
        norms=[preprocess_layer_norm_parameters(norm, device=device, dtype=dtype) for norm in layer.norms],
    )


def create_decoder_parameters(torch_model, device, dtype=WEIGHT_DTYPE):
    """Parameters for ``TTDetectionTransformerDecoder``."""
    return SimpleNamespace(layers=[_decoder_layer_parameters(layer, device, dtype) for layer in torch_model.layers])


def create_reg_branch_parameters(reg_branches, device, dtype=WEIGHT_DTYPE):
    """The three Linears of each ``Linear-ReLU-Linear-ReLU-Linear`` branch.

    The decoder runs each branch once per layer: it refines its reference points with the
    center channels of the box code and returns the whole code, which the head uses. The last
    Linear's rows go in ``CENTERS_FIRST`` order, the code order the decoder takes.
    """
    branches = []
    for branch in reg_branches:
        # Unpacking checks the three Linears TTDetectionTransformerDecoder._reg_branch runs.
        first, second, last = (m for m in branch if isinstance(m, torch.nn.Linear))
        branches.append(
            [linear_params(m.weight, m.bias, device, dtype) for m in (first, second)]
            + [linear_params(last.weight[CENTERS_FIRST], last.bias[CENTERS_FIRST], device, dtype)]
        )
    return branches


# --- Head -------------------------------------------------------------------------------------


def _cls_branch_parameters(branch, device, dtype):
    """``hidden``: the ``(linear, layer_norm)`` of each hidden block; ``out``: the output Linear."""
    linears = [m for m in branch if isinstance(m, torch.nn.Linear)]
    norms = [m for m in branch if isinstance(m, torch.nn.LayerNorm)]
    hidden = [
        (
            linear_params(linear.weight, linear.bias, device, dtype),
            preprocess_layer_norm_parameters(norm, device=device, dtype=dtype),
        )
        for linear, norm in zip(linears[:-1], norms, strict=True)
    ]
    return SimpleNamespace(hidden=hidden, out=linear_params(linears[-1].weight, linears[-1].bias, device, dtype))


@torch.no_grad()
def create_head_parameters(torch_model, device, dtype=WEIGHT_DTYPE):
    """Parameters for ``TTBEVFormerHead``. They carry the reference head's BEV shape and its
    ``pc_range`` as the corner ``pc_min`` and size ``pc_size`` the refined points scale to.

    The object queries, their positional embeddings and the initial reference points depend
    on the weights only, so they are computed here, the points in ``GRID_DTYPE`` as the decoder
    takes them. The reg branches run inside the decoder, which returns their box codes.
    """
    query_pos, query = torch.split(torch_model.query_embedding.weight, torch_model.embed_dims, dim=1)
    init_reference = torch_model.reference_points(query_pos).sigmoid()

    def upload(tensor, tensor_dtype=dtype):
        return ttnn.from_torch(tensor.unsqueeze(0), dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    pc_min = torch.tensor(torch_model.pc_range[:3])
    pc_size = torch.tensor(torch_model.pc_range[3:]) - pc_min
    return SimpleNamespace(
        bev_shape=(torch_model.bev_h, torch_model.bev_w),
        pc_min=upload(pc_min.view(1, 1, 3), GRID_DTYPE),
        pc_size=upload(pc_size.view(1, 1, 3), GRID_DTYPE),
        query=upload(query),
        query_pos=upload(query_pos),
        reference_points=upload(init_reference, GRID_DTYPE),
        decoder=create_decoder_parameters(torch_model.decoder, device, dtype),
        reg_branches=create_reg_branch_parameters(torch_model.reg_branches, device, dtype),
        cls_branches=[_cls_branch_parameters(branch, device, dtype) for branch in torch_model.cls_branches],
    )


# --- Detector ---------------------------------------------------------------------------------


@torch.no_grad()
def create_bevformer_parameters(model, img, device, dtype=WEIGHT_DTYPE):
    """The detector's parameters for images shaped like ``img`` ``(bs, num_cams, 3, H, W)``: the
    backbone and FPN pin every conv to that shape, so the TT detector only takes it. One reference
    forward of the backbone records the shapes, and the FPN's on its features the levels' ``(h, w)``.
    The BEV queries and their positional encoding are constants of the weights, so they are
    uploaded here, ``(1, bev_h * bev_w, C)``. The result's ``config`` and ``coder`` carry the
    reference's settings the TT detector and its coder are built with, so the two cannot disagree.
    ``dtype`` is the transformer's and the head's weights'; the backbone's and FPN's precision is
    ``model_config``'s."""
    images = img.flatten(0, 1)
    # create_resnet_parameters runs the backbone to record its conv shapes; its features come from
    # that same forward, as a second ResNet101 forward on the CPU takes minutes. The forward runs
    # under ttnn's tracer, whose tensor subclass is unwrapped here.
    recorded = []
    hook = model.img_backbone.register_forward_hook(lambda _module, _inputs, outputs: recorded.append(outputs))
    try:
        backbone = create_resnet_parameters(model.img_backbone, images)
    finally:
        hook.remove()
    features = [feature.as_subclass(torch.Tensor) for feature in recorded[-1]]
    levels = model.img_neck(list(features))

    def upload(tensor):
        return ttnn.from_torch(tensor[None], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return SimpleNamespace(
        config=SimpleNamespace(
            bev_h=model.bev_h,
            bev_w=model.bev_w,
            batch_size=img.shape[0],
            num_cams=img.shape[1],
            out_indices=tuple(model.img_backbone.out_indices),
            spatial_shapes=tuple(tuple(level.shape[-2:]) for level in levels),
        ),
        coder=SimpleNamespace(
            **{
                name: getattr(model.bbox_coder, name)
                for name in ("post_center_range", "max_num", "score_threshold", "num_classes")
            }
        ),
        backbone=backbone,
        neck=create_fpn_parameters(model.img_neck, features),
        transformer=create_perception_transformer_parameters(model.transformer, device, dtype),
        head=create_head_parameters(model.head, device, dtype),
        bev_queries=upload(model.bev_embedding.weight),
        bev_pos=upload(model.positional_encoding(model.bev_h, model.bev_w)),
    )


# --- Frame inputs -------------------------------------------------------------------------------

# Host-side inputs the perception transformer derives from the BEV grid and the CAN bus, in
# float32 as upstream computes them, once per frame into buffers the forward reads.


def bev_grid_length(pc_range, bev_h, bev_w):
    """A BEV cell's size in metres, (y, x): the point-cloud range over the grid, as upstream's head
    passes ``grid_length`` to ``get_bev_features``."""
    return ((pc_range[4] - pc_range[1]) / bev_h, (pc_range[3] - pc_range[0]) / bev_w)


def ego_shift(img_metas, bev_h, bev_w, grid_length):
    """The ego translation in BEV-grid fractions, ``(bs, 2)`` as (x, y): upstream's ``shift``, from
    each sample's relative CAN bus, with ``grid_length`` from :func:`bev_grid_length`."""
    shifts = []
    for meta in img_metas:
        can_bus = meta["can_bus"]
        delta_x, delta_y = float(can_bus[0]), float(can_bus[1])
        translation_length = math.hypot(delta_x, delta_y)
        bev_angle = math.radians(float(can_bus[-2]) / math.pi * 180 - math.degrees(math.atan2(delta_y, delta_x)))
        shift_y = translation_length * math.cos(bev_angle) / grid_length[0] / bev_h
        shift_x = translation_length * math.sin(bev_angle) / grid_length[1] / bev_w
        shifts.append([shift_x, shift_y])
    return torch.tensor(shifts, dtype=torch.float32)
