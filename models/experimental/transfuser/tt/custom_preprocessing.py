import torch
import ttnn

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d

from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.reference.bottleneck import Bottleneck
from models.experimental.transfuser.reference.stage import Stage
from models.experimental.transfuser.reference.common import Conv2d


# =========================
# Config / constants
# =========================

TT_DTYPE = ttnn.bfloat16
BIAS_SHAPE = (1, 1, 1, -1)

# Blocks per stage name
STAGE_STRUCTURE = {
    "layer1": ["b1", "b2"],
    "layer2": ["b1", "b2", "b3", "b4", "b5"],
    "layer3": ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10", "b11", "b12", "b13"],
    "layer4": ["b1"],
}


# =========================
# Small helpers
# =========================


def _get_or_create(root: dict, *keys: str) -> dict:
    """Ensure nested dictionaries exist and return the leaf dict."""
    d = root
    for k in keys:
        d = d.setdefault(k, {})
    return d


def _fold_and_pack_conv(conv_module, bn_module, *, mesh_mapper):
    """Fold BN into Conv, return (weight_ttnn, bias_ttnn) with standard dtype/shape."""
    weight, bias = fold_batch_norm2d_into_conv2d(conv_module, bn_module)
    w_t = ttnn.from_torch(weight, dtype=TT_DTYPE, mesh_mapper=mesh_mapper)
    b_t = ttnn.from_torch(bias.reshape(BIAS_SHAPE), dtype=TT_DTYPE, mesh_mapper=mesh_mapper)
    return w_t, b_t


def _pack_bias_only(bias_tensor, *, mesh_mapper):
    """Pack a bias tensor (1D) into TTNN with standard shape."""
    return ttnn.from_torch(bias_tensor.reshape(BIAS_SHAPE), dtype=TT_DTYPE, mesh_mapper=mesh_mapper)


def _pack_weight_only(weight_tensor, *, mesh_mapper):
    """Pack a weight tensor into TTNN."""
    return ttnn.from_torch(weight_tensor, dtype=TT_DTYPE, mesh_mapper=mesh_mapper)


def _handle_conv2d(parameters: dict, conv: Conv2d, *, mesh_mapper):
    """Generic handler for Conv2d with optional BN fused in."""
    if conv.norm is not None:
        w_t, b_t = _fold_and_pack_conv(conv, conv.norm, mesh_mapper=mesh_mapper)
    else:
        weight = conv.weight.clone().detach().contiguous()
        if conv.bias is not None:
            bias = conv.bias.clone().detach().contiguous()
        else:
            bias = torch.zeros(conv.out_channels, device=weight.device, dtype=weight.dtype)
        w_t = ttnn.from_torch(weight, mesh_mapper=mesh_mapper)
        b_t = _pack_bias_only(bias, mesh_mapper=mesh_mapper)

    parameters["weight"] = w_t
    parameters["bias"] = b_t


def _extract_conv_bn(ds):
    """Return (conv, bn) from a downsample-like module, or (None, None) if Identity/None."""
    if ds is None:
        return None, None
    # Identity → nothing to do
    if isinstance(ds, torch.nn.Identity):
        return None, None
    # Custom modules exposing .conv/.bn
    if hasattr(ds, "conv") and hasattr(ds, "bn"):
        return ds.conv, ds.bn
    # Torch Sequential(conv, bn)
    if isinstance(ds, torch.nn.Sequential):
        if len(ds) < 2:
            raise TypeError("Downsample Sequential must contain at least [conv, bn].")
        return ds[0], ds[1]
    # Anything indexable (very defensive)
    if hasattr(ds, "__getitem__"):
        try:
            return ds[0], ds[1]
        except Exception as e:
            raise TypeError(f"Unsupported downsample module type: {type(ds)}") from e
    # Unknown pattern
    raise TypeError(f"Unsupported downsample module type: {type(ds)}")


def _handle_bottleneck(dst: dict, block: Bottleneck, *, mesh_mapper):
    """Fill parameters for a ResNet-style Bottleneck block:
    conv1/conv2/conv3, SE(fc1/fc2 weights), optional downsample."""
    # conv1
    w, b = _fold_and_pack_conv(block.conv1.conv, block.conv1.bn, mesh_mapper=mesh_mapper)
    c1 = _get_or_create(dst, "conv1")
    c1["weight"], c1["bias"] = w, b

    # conv2 (grouped 3x3)
    w, b = _fold_and_pack_conv(block.conv2.conv, block.conv2.bn, mesh_mapper=mesh_mapper)
    c2 = _get_or_create(dst, "conv2")
    c2["weight"], c2["bias"] = w, b

    # conv3
    w, b = _fold_and_pack_conv(block.conv3.conv, block.conv3.bn, mesh_mapper=mesh_mapper)
    c3 = _get_or_create(dst, "conv3")
    c3["weight"], c3["bias"] = w, b

    # SE (weights only)
    se = _get_or_create(dst, "se")
    fc1 = _get_or_create(se, "fc1")
    fc2 = _get_or_create(se, "fc2")
    fc1["weight"] = _pack_weight_only(block.se.fc1.weight, mesh_mapper=mesh_mapper)
    fc2["weight"] = _pack_weight_only(block.se.fc2.weight, mesh_mapper=mesh_mapper)

    # Downsample (if present and not Identity)
    if hasattr(block, "downsample") and block.downsample is not None:
        conv_m, bn_m = _extract_conv_bn(block.downsample)
        if conv_m is not None and bn_m is not None:
            w, b = _fold_and_pack_conv(conv_m, bn_m, mesh_mapper=mesh_mapper)
            dsd = _get_or_create(dst, "downsample")
            dsd["weight"], dsd["bias"] = w, b
        # Identity/None → skip


def _handle_stage(dst: dict, stage_module, stage_name: str, *, mesh_mapper):
    """Process a full stage with multiple Bottleneck blocks, using STAGE_STRUCTURE."""
    blocks = STAGE_STRUCTURE.get(stage_name, [])
    stage_dst = _get_or_create(dst, stage_name)
    for block_name in blocks:
        if hasattr(stage_module, block_name):
            block = getattr(stage_module, block_name)
            _handle_bottleneck(_get_or_create(stage_dst, block_name), block, mesh_mapper=mesh_mapper)


# =========================
# Public API
# =========================


def preprocess_conv_parameter(parameter, *, dtype):
    # Kept for API compatibility (your original helper)
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(
    model, name, ttnn_module_args, convert_to_ttnn, custom_preprocessor_func=None, mesh_mapper=None
):
    """
    DRY, refactored preprocessor:
    - Handles Conv2d (with inline BN folding)
    - TransfuserBackbone (conv1 for image & lidar, layer1 pairs for both encoders)
    - Standalone Bottleneck
    - Stage (generic by stage name using STAGE_STRUCTURE)
    """
    parameters = {}

    # 1) Plain Conv2d wrapper (your common.Conv2d)
    if isinstance(model, Conv2d):
        _handle_conv2d(parameters, model, mesh_mapper=mesh_mapper)
        return parameters

    # 2) TransfuserBackbone (conv1s + layer1 for image/lidar encoders)
    if isinstance(model, TransfuserBackbone):
        # Image conv1
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "features"):
            img_features = model.image_encoder.features
            if hasattr(img_features, "conv1") and hasattr(img_features, "bn1"):
                img_conv1 = _get_or_create(parameters, "image_encoder", "features", "conv1")
                w, b = _fold_and_pack_conv(img_features.conv1, img_features.bn1, mesh_mapper=mesh_mapper)
                img_conv1["weight"], img_conv1["bias"] = w, b

        # Lidar conv1
        if hasattr(model, "lidar_encoder") and hasattr(model.lidar_encoder, "_model"):
            lidar = model.lidar_encoder._model
            if hasattr(lidar, "conv1") and hasattr(lidar, "bn1"):
                lid_conv1 = _get_or_create(parameters, "lidar_encoder", "_model", "conv1")
                w, b = _fold_and_pack_conv(lidar.conv1, lidar.bn1, mesh_mapper=mesh_mapper)
                lid_conv1["weight"], lid_conv1["bias"] = w, b

        # Image layer1
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "features"):
            img_features = model.image_encoder.features
            if hasattr(img_features, "layer1"):
                img_l1 = _get_or_create(parameters, "image_encoder", "features")
                _handle_stage(img_l1, img_features.layer1, "layer1", mesh_mapper=mesh_mapper)

        # Lidar layer1
        if hasattr(model, "lidar_encoder") and hasattr(model.lidar_encoder, "_model"):
            lidar = model.lidar_encoder._model
            if hasattr(lidar, "layer1"):
                lid_l1 = _get_or_create(parameters, "lidar_encoder", "_model")
                _handle_stage(lid_l1, lidar.layer1, "layer1", mesh_mapper=mesh_mapper)

        return parameters

    # 3) Standalone Bottleneck
    if isinstance(model, Bottleneck):
        _handle_bottleneck(parameters, model, mesh_mapper=mesh_mapper)
        return parameters

    # 4) Stage wrapper (expects model.stage_name and image_encoder.features.<stage_name>)
    if isinstance(model, Stage):
        stage_name = getattr(model, "stage_name", None)
        if stage_name is None:
            return parameters  # nothing to do

        # Choose a source to read from; original code pulled from image_encoder.features.<stage_name>
        if hasattr(model, "image_encoder") and hasattr(model.image_encoder, "features"):
            src_stage = getattr(model.image_encoder.features, stage_name, None)
            if src_stage is not None:
                _handle_stage(parameters, src_stage, stage_name, mesh_mapper=mesh_mapper)
        return parameters

    # Fallback (unknown module): return empty to let caller decide next steps
    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor
