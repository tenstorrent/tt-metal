from ttnn.model_preprocessing import ParameterList, make_parameter_dict


def preprocess_resnet_conv_layer(torch_module, *_):
    """Preserve the convolution parameter and Frozen-BN buffers as Torch tensors."""
    return {
        "convolution": {"weight": torch_module.convolution.weight},
        "normalization": {
            "weight": torch_module.normalization.weight,
            "bias": torch_module.normalization.bias,
            "running_mean": torch_module.normalization.running_mean,
            "running_var": torch_module.normalization.running_var,
        },
    }


def preprocess_resnet_embeddings(torch_module, *_):
    return {
        "embedder": ParameterList(
            [make_parameter_dict(preprocess_resnet_conv_layer(layer)) for layer in torch_module.embedder]
        ),
    }


def preprocess_resnet_bottleneck(torch_module, *_):
    parameters = {
        "layer": ParameterList(
            [make_parameter_dict(preprocess_resnet_conv_layer(layer)) for layer in torch_module.layer]
        ),
    }

    projection = torch_module.shortcut if hasattr(torch_module.shortcut, "convolution") else None
    if projection is None:
        projection = next(
            (module for module in torch_module.shortcut.children() if hasattr(module, "convolution")),
            None,
        )
    if projection is not None:
        parameters["shortcut"] = make_parameter_dict(preprocess_resnet_conv_layer(projection))

    return parameters


def preprocess_resnet_stage(torch_module, *_):
    return {
        "layers": ParameterList(
            [make_parameter_dict(preprocess_resnet_bottleneck(layer)) for layer in torch_module.layers]
        ),
    }


def preprocess_resnet_encoder(torch_module, *_):
    return {
        "stages": ParameterList([make_parameter_dict(preprocess_resnet_stage(stage)) for stage in torch_module.stages]),
    }


def preprocess_resnet_backbone(torch_module, *_):
    return {
        "embedder": make_parameter_dict(preprocess_resnet_embeddings(torch_module.embedder)),
        "encoder": make_parameter_dict(preprocess_resnet_encoder(torch_module.encoder)),
    }


def preprocess_conv_encoder(torch_module, *_):
    return {
        "model": make_parameter_dict(preprocess_resnet_backbone(torch_module.model)),
    }


def preprocess_encoder_input_projection(torch_projection, *_):
    convolution = torch_projection[0]
    normalization = torch_projection[1]

    return {
        "convolution": {
            "weight": convolution.weight,
        },
        "normalization": {
            "weight": normalization.weight,
            "bias": normalization.bias,
            "running_mean": normalization.running_mean,
            "running_var": normalization.running_var,
        },
    }


def preprocess_rtdetr_conv_norm_layer(torch_module, *_):
    return {
        "convolution": {
            "weight": torch_module.conv.weight,
        },
        "normalization": {
            "weight": torch_module.norm.weight,
            "bias": torch_module.norm.bias,
            "running_mean": torch_module.norm.running_mean,
            "running_var": torch_module.norm.running_var,
        },
    }


def preprocess_rtdetr_mlp(torch_module, *_):
    return {
        "fc1": {
            "weight": torch_module.fc1.weight,
            "bias": torch_module.fc1.bias,
        },
        "fc2": {
            "weight": torch_module.fc2.weight,
            "bias": torch_module.fc2.bias,
        },
    }


def preprocess_rtdetr_self_attention(torch_module, *_):
    return {
        projection_name: {
            "weight": getattr(torch_module, projection_name).weight,
            "bias": getattr(torch_module, projection_name).bias,
        }
        for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }


def preprocess_rtdetr_encoder_layer(torch_module, *_):
    return {
        "self_attn": make_parameter_dict(preprocess_rtdetr_self_attention(torch_module.self_attn)),
        "self_attn_layer_norm": {
            "weight": torch_module.self_attn_layer_norm.weight,
            "bias": torch_module.self_attn_layer_norm.bias,
        },
        "mlp": make_parameter_dict(preprocess_rtdetr_mlp(torch_module.mlp)),
        "final_layer_norm": {
            "weight": torch_module.final_layer_norm.weight,
            "bias": torch_module.final_layer_norm.bias,
        },
    }


def preprocess_rtdetr_aifi_layer(torch_module, *_):
    return {
        "layers": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_encoder_layer(layer)) for layer in torch_module.layers]
        ),
    }


def preprocess_rtdetr_rep_vgg_block(torch_module, *_):
    return {
        "conv1": make_parameter_dict(preprocess_rtdetr_conv_norm_layer(torch_module.conv1)),
        "conv2": make_parameter_dict(preprocess_rtdetr_conv_norm_layer(torch_module.conv2)),
    }


def preprocess_rtdetr_csp_rep_layer(torch_module, *_):
    parameters = {
        "conv1": make_parameter_dict(preprocess_rtdetr_conv_norm_layer(torch_module.conv1)),
        "conv2": make_parameter_dict(preprocess_rtdetr_conv_norm_layer(torch_module.conv2)),
        "bottlenecks": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_rep_vgg_block(block)) for block in torch_module.bottlenecks]
        ),
    }

    if hasattr(torch_module.conv3, "conv"):
        parameters["conv3"] = make_parameter_dict(preprocess_rtdetr_conv_norm_layer(torch_module.conv3))

    return parameters


def preprocess_rtdetr_hybrid_encoder(torch_module, *_):
    return {
        "aifi": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_aifi_layer(layer)) for layer in torch_module.aifi]
        ),
        "lateral_convs": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_conv_norm_layer(layer)) for layer in torch_module.lateral_convs]
        ),
        "fpn_blocks": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_csp_rep_layer(layer)) for layer in torch_module.fpn_blocks]
        ),
        "downsample_convs": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_conv_norm_layer(layer)) for layer in torch_module.downsample_convs]
        ),
        "pan_blocks": ParameterList(
            [make_parameter_dict(preprocess_rtdetr_csp_rep_layer(layer)) for layer in torch_module.pan_blocks]
        ),
    }


def custom_preprocessor(torch_module, name):
    if torch_module.__class__.__name__ == "RTDetrHybridEncoder":
        return preprocess_rtdetr_hybrid_encoder(torch_module)

    if torch_module.__class__.__name__ == "RTDetrConvNormLayer":
        return preprocess_rtdetr_conv_norm_layer(torch_module)

    if name == "backbone":
        return preprocess_conv_encoder(torch_module)

    if name.startswith("encoder_input_proj."):
        return preprocess_encoder_input_projection(torch_module)

    if name.startswith("encoder.aifi."):
        return preprocess_rtdetr_aifi_layer(torch_module)

    return {}
