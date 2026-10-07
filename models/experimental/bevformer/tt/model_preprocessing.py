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
"""

from types import SimpleNamespace

import torch

import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters
from models.experimental.bevformer.model_config import GRID_DTYPE
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.resnet import ModulatedDeformConv2dPack, ResNet
from models.experimental.bevformer.tt.tt_modulated_deform_conv import grid_offset_order


DEFAULT_DTYPE = ttnn.bfloat16


# --- Shared helpers ---------------------------------------------------------------------------


def preprocess_linear_weight(weight, *, device, dtype=DEFAULT_DTYPE):
    """A Linear's weight transposed to ``(in, out)``, as ``ttnn.linear`` takes it."""
    return ttnn.from_torch(weight.T.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def preprocess_linear_bias(bias, *, device, dtype=DEFAULT_DTYPE):
    """A Linear's bias as a ``(1, out)`` row."""
    return ttnn.from_torch(bias.reshape((1, -1)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def linear_params(weight, bias, device, dtype):
    """A Linear's ``weight`` and ``bias`` as ``ttnn.linear`` takes them, on ``device``."""
    return SimpleNamespace(
        weight=preprocess_linear_weight(weight, dtype=dtype, device=device),
        bias=None if bias is None else preprocess_linear_bias(bias, dtype=dtype, device=device),
    )


def preprocess_layer_norm_parameters(layer_norm, *, device, dtype=DEFAULT_DTYPE):
    """A LayerNorm's weight and bias, and its ``eps``: ``ttnn.layer_norm`` defaults to 1e-12."""

    def upload(tensor):
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return SimpleNamespace(
        eps=layer_norm.eps,
        weight=upload(layer_norm.weight),
        bias=None if layer_norm.bias is None else upload(layer_norm.bias),
    )


def preprocess_ffn_parameters(ffn, *, device, dtype=DEFAULT_DTYPE):
    """The encoder's and decoder's mmcv-style ``FFN``: ``layers`` is
    ``Sequential(Sequential(Linear, ReLU), Linear)``, so the two Linears are ``layers[0][0]`` and
    ``layers[1]``."""
    first, second = ffn.layers[0][0], ffn.layers[1]
    return SimpleNamespace(
        linear1=linear_params(first.weight, first.bias, device, dtype),
        linear2=linear_params(second.weight, second.bias, device, dtype),
    )


# --- Deformable attention and encoder ---------------------------------------------------------


def create_ms_deformable_attention_parameters(torch_model, *, device, dtype=DEFAULT_DTYPE):
    """``reference.ms_deformable_attention.MSDeformableAttention``'s value, offset, attention and
    output Linears, as TTMSDeformableAttention takes them."""
    return SimpleNamespace(
        **{
            name: linear_params(getattr(torch_model, name).weight, getattr(torch_model, name).bias, device, dtype)
            for name in ("value_proj", "sampling_offsets", "attention_weights", "output_proj")
        }
    )


def create_spatial_cross_attention_parameters(sca, device, dtype=DEFAULT_DTYPE):
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


def create_temporal_self_attention_parameters(tsa, device, dtype=DEFAULT_DTYPE):
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


def create_bevformer_layer_parameters(layer, device, dtype=DEFAULT_DTYPE):
    """``reference.encoder.BEVFormerLayer`` as TTBEVFormerLayer takes it."""
    return SimpleNamespace(
        tsa=create_temporal_self_attention_parameters(layer.attentions[0], device, dtype),
        sca=create_spatial_cross_attention_parameters(layer.attentions[1], device, dtype),
        ffn=preprocess_ffn_parameters(layer.ffns[0], device=device, dtype=dtype),
        norms=[preprocess_layer_norm_parameters(norm, device=device, dtype=dtype) for norm in layer.norms],
    )


def create_bevformer_encoder_parameters(encoder, device, dtype=DEFAULT_DTYPE):
    """``reference.encoder.BEVFormerEncoder`` as TTBEVFormerEncoder takes it: one entry per layer,
    and the encoder's configuration, so the port cannot disagree with the model it was built from."""
    config = SimpleNamespace(
        **{
            name: getattr(encoder, name)
            for name in (
                "embed_dims",
                "num_heads",
                "num_levels",
                "num_points",
                "num_cams",
                "tsa_num_points",
                "num_points_in_pillar",
                "pc_range",
            )
        }
    )
    return SimpleNamespace(
        config=config, layers=[create_bevformer_layer_parameters(layer, device, dtype) for layer in encoder.layers]
    )


# --- Perception transformer --------------------------------------------------------------------


def create_perception_transformer_parameters(transformer, device, dtype=DEFAULT_DTYPE):
    """``reference.perception_transformer.PerceptionTransformer`` as TtPerceptionTransformer takes
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

# Each BatchNorm is folded into the conv before it, and every conv's input shape is recorded from one
# reference forward. The DCN offset rows are reordered to the (x, y) order the device deformable
# conv reads.


def _fold_batch_norm(weight, bias, bn):
    """``(weight, bias)`` of a conv with the eval-mode BatchNorm ``bn`` after it folded in."""
    scale = bn.running_var.add(bn.eps).rsqrt()
    shift = -bn.running_mean * scale
    if bn.affine:
        scale = scale * bn.weight
        shift = shift * bn.weight + bn.bias
    bias = torch.zeros_like(shift) if bias is None else bias.detach()
    return weight * scale.view(-1, 1, 1, 1), bias * scale + shift


def _resnet_preprocessor(model, name):
    """``preprocess_model_parameters``' hook for the ResNet: BatchNorm-folded conv weights, DCN
    offsets in the device's (x, y) order."""
    parameters = {}

    if isinstance(model, ResNet):
        parameters["res_model"] = {}

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            prefix = f"layer{layer_idx}"
            parameters["res_model"][prefix] = {}
            for block_idx, block in enumerate(layer):
                parameters["res_model"][prefix][block_idx] = {}

                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    if isinstance(conv, ModulatedDeformConv2dPack):
                        # The DCN conv consumes torch weights, with its BatchNorm folded in.
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        weight, bias = _fold_batch_norm(conv.weight.detach(), conv.bias, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {}
                        parameters["res_model"][prefix][block_idx][conv_name]["weight"] = weight
                        parameters["res_model"][prefix][block_idx][conv_name]["bias"] = bias
                        # The offset rows are reordered to the (x, y) order the device DCN takes;
                        # moving whole rows is exact.
                        order = grid_offset_order(conv.kernel_size[0] * conv.kernel_size[1])
                        offset_weight = conv.conv_offset.weight.detach()
                        offset_bias = conv.conv_offset.bias.detach()
                        offset_weight = torch.cat([offset_weight[order], offset_weight[len(order) :]])
                        offset_bias = torch.cat([offset_bias[order], offset_bias[len(order) :]])
                        parameters["res_model"][prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(offset_weight, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(offset_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }
                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    assert (
                        isinstance(ds, torch.nn.Sequential) and len(ds) == 2
                    ), f"expected a (conv, norm) downsample, got {ds}"
                    w, b = fold_batch_norm2d_into_conv2d(ds[0], ds[1])
                    parameters["res_model"][prefix][block_idx]["downsample"] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }
    return parameters


def create_resnet_parameters(model: ResNet, input_tensor, device=None):
    """Preprocess backbone weights and record per-conv shapes from one reference forward.

    ``conv_args`` pins every conv's input height, width and batch to ``input_tensor``'s
    shape, so the TT backbone built from these parameters only accepts that shape.
    """
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=_resnet_preprocessor,
        device=device,
    )
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    return parameters


def create_fpn_parameters(model: FPN, input_tensors):
    """Preprocess FPN weights and record every conv's geometry and input shape.

    ``input_tensors`` are the NCHW backbone outputs the FPN will run on. A conv reads the
    level it belongs to, except the extra output convs: the first reads the last level's
    output and each later one reads the previous extra conv's output.

    ``conv_args`` holds, per conv, the attributes ``TtnnConv2D`` reads, as
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

    def conv_weights(conv_module):
        conv = conv_module.conv
        return {
            "conv": {
                "weight": ttnn.from_torch(conv.weight, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
            }
        }

    def preprocessor(module, name):
        if not isinstance(module, FPN):
            return {}
        return {
            "fpn": {
                "lateral_convs": {str(i): conv_weights(lateral) for i, lateral in enumerate(module.lateral_convs)},
                "fpn_convs": {str(i): conv_weights(fpn_conv) for i, fpn_conv in enumerate(module.fpn_convs)},
            }
        }

    parameters = preprocess_model_parameters(initialize_model=lambda: model, custom_preprocessor=preprocessor)
    parameters["conv_args"] = SimpleNamespace(
        lateral_convs=[conv_args(lateral, level_shapes[i]) for i, lateral in enumerate(model.lateral_convs)],
        fpn_convs=[conv_args(fpn_conv, shapes[i]) for i, fpn_conv in enumerate(model.fpn_convs)],
        add_extra_convs=model.add_extra_convs,
        relu_before_extra_convs=model.relu_before_extra_convs,
    )
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
    """The reference module's parameters and config."""
    if msda.num_levels != 1:
        raise ValueError(f"the decoder cross-attention is single-level, got {msda.num_levels} levels")
    params = create_ms_deformable_attention_parameters(msda, device=device, dtype=dtype)
    params.config = msda.config
    return params


def _decoder_layer_parameters(layer, device, dtype):
    """``reference.decoder.DetrTransformerDecoderLayer`` as TtDetectionTransformerDecoder's layer
    takes it."""
    return SimpleNamespace(
        self_attn=_self_attn_parameters(layer.attentions[0].attn, device, dtype),
        cross_attn=_cross_attn_parameters(layer.attentions[1], device, dtype),
        ffn=preprocess_ffn_parameters(layer.ffns[0], device=device, dtype=dtype),
        norms=[preprocess_layer_norm_parameters(norm, device=device, dtype=dtype) for norm in layer.norms],
    )


def create_decoder_parameters(torch_model, device, dtype=DEFAULT_DTYPE):
    """Parameters for a single ``TtDetectionTransformerDecoder``.

    Single use: the decoder's constructor folds the BEV size into each layer's
    ``cross_attn.sampling_offsets`` and frees the original, so every decoder instance (and
    every BEV size) needs its own call.
    """
    return SimpleNamespace(layers=[_decoder_layer_parameters(layer, device, dtype) for layer in torch_model.layers])


def create_reg_branch_parameters(reg_branches, device, dtype=DEFAULT_DTYPE):
    """The three Linears of each ``Linear-ReLU-Linear-ReLU-Linear`` branch.

    The decoder runs each branch once per layer: it refines its reference points with the
    center channels of the box code and returns the whole code, which the head uses.
    """
    branches = []
    for branch in reg_branches:
        # Unpacking checks the three Linears TtDetectionTransformerDecoder._reg_branch runs.
        first, second, last = (m for m in branch if isinstance(m, torch.nn.Linear))
        branches.append([linear_params(m.weight, m.bias, device, dtype) for m in (first, second, last)])
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
def create_head_parameters(torch_model, device, dtype=DEFAULT_DTYPE):
    """Parameters for a single ``TtBEVFormerHead``; single use, as ``create_decoder_parameters`` is.
    They carry the reference head's BEV shape and ``pc_range``.

    The object queries, their positional embeddings and the initial reference points depend
    on the weights only, so they are computed here, the points in float32 as the decoder
    takes them. The reg branches run inside the decoder, which returns their box codes.
    """
    query_pos, query = torch.split(torch_model.query_embedding.weight, torch_model.embed_dims, dim=1)
    init_reference = torch_model.reference_points(query_pos).sigmoid()

    def upload(tensor, tensor_dtype=dtype):
        return ttnn.from_torch(tensor.unsqueeze(0), dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return SimpleNamespace(
        bev_shape=(torch_model.bev_h, torch_model.bev_w),
        pc_range=tuple(torch_model.pc_range),
        query=upload(query),
        query_pos=upload(query_pos),
        reference_points=upload(init_reference, GRID_DTYPE),
        decoder=create_decoder_parameters(torch_model.decoder, device, dtype),
        reg_branches=create_reg_branch_parameters(torch_model.reg_branches, device, dtype),
        cls_branches=[_cls_branch_parameters(branch, device, dtype) for branch in torch_model.cls_branches],
    )


# --- Detector ---------------------------------------------------------------------------------


@torch.no_grad()
def create_bevformer_parameters(model, img, device, dtype=DEFAULT_DTYPE):
    """The detector's parameters for images shaped like ``img`` ``(bs, num_cams, 3, H, W)``: the
    backbone and FPN pin every conv to that shape, so the TT detector only takes it. One reference
    forward of the backbone records the shapes, and the FPN's on its features the levels' ``(h, w)``.
    The BEV queries and their positional encoding are constants of the weights, so they are
    uploaded here, ``(1, bev_h * bev_w, C)``. The result's ``config`` carries the reference's
    settings the TT detector is built with, so the two cannot disagree. ``dtype`` is the transformer's
    and the head's weights'; the backbone's and FPN's precision is ``model_config``'s."""
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
        backbone=backbone,
        neck=create_fpn_parameters(model.img_neck, features),
        transformer=create_perception_transformer_parameters(model.transformer, device, dtype),
        head=create_head_parameters(model.head, device, dtype),
        bev_queries=upload(model.bev_embedding.weight),
        bev_pos=upload(model.positional_encoding(model.bev_h, model.bev_w)),
    )
