# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Safely translate an upstream BEVFormer Python config to normalized JSON.

The source is parsed as an AST and a deliberately small expression subset is interpreted. It is
never imported or executed. Fields absent from the upstream file use the selected bundled
variant's defaults, including the camera order that upstream stores in its dataset converter.
"""

import argparse
import ast
import json
import operator
import re
from dataclasses import replace
from pathlib import Path

from .loader import load_builtin_config, read_source_bytes
from .schema import SUPPORTED_VARIANTS

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_CONVERTED_ASSIGNMENTS = {"model", "test_pipeline", "img_norm_cfg", "class_names", "point_cloud_range"}


def _safe_eval(node, environment):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_safe_eval(item, environment) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        return {
            _safe_eval(key, environment): _safe_eval(value, environment)
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_eval(node.operand, environment)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        return _BINARY_OPERATORS[type(node.op)](_safe_eval(node.left, environment), _safe_eval(node.right, environment))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        if len(node.args) > 1:
            raise ValueError("dict() with more than one positional argument is unsupported")
        result = {} if not node.args else dict(_safe_eval(node.args[0], environment))
        for keyword in node.keywords:
            value = _safe_eval(keyword.value, environment)
            if keyword.arg is None:
                if not isinstance(value, dict):
                    raise ValueError("dict() expansion must evaluate to a dictionary")
                result.update(value)
            else:
                result[keyword.arg] = value
        return result
    raise ValueError(f"unsupported expression {type(node).__name__}")


def _assignments(source: str) -> dict:
    tree = ast.parse(source)
    environment = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            environment[target.id] = _safe_eval(statement.value, environment)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            if target.id in _CONVERTED_ASSIGNMENTS:
                raise ValueError(f"could not statically read {target.id}") from error
            continue
    return environment


def variant_from_source(source: str) -> str | None:
    match = re.search(r"bevformer[_-](tiny|small|base)", source, flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _pipeline_stages(pipeline):
    if not isinstance(pipeline, (list, tuple)):
        raise ValueError("test_pipeline must be a list of transform dictionaries")
    for stage in pipeline:
        if not isinstance(stage, dict) or not isinstance(stage.get("type"), str):
            raise ValueError("test_pipeline must contain transform dictionaries with string types")
        yield stage
        if "transforms" in stage:
            yield from _pipeline_stages(stage["transforms"])


def _single_pipeline_stage(stages, stage_type):
    matches = [stage for stage in stages if stage["type"] == stage_type]
    if len(matches) > 1:
        raise ValueError(f"test_pipeline has multiple {stage_type} stages")
    return matches[0] if matches else None


def _inference_input_settings(values, config):
    pipeline = values.get("test_pipeline")
    if pipeline is None:
        return config.camera_image_size, config.image_scale, config.image_size_divisor
    stages = list(_pipeline_stages(pipeline))

    augmentation = _single_pipeline_stage(stages, "MultiScaleFlipAug3D")
    camera_image_size = config.camera_image_size
    if augmentation is not None:
        camera_image_size = augmentation.get("img_scale")
        if (
            not isinstance(camera_image_size, (list, tuple))
            or len(camera_image_size) != 2
            or not all(isinstance(size, int) and not isinstance(size, bool) and size > 0 for size in camera_image_size)
        ):
            raise ValueError("MultiScaleFlipAug3D.img_scale must be one positive (width, height) pair")
        camera_image_size = tuple(camera_image_size)
        if augmentation.get("pts_scale_ratio", 1) != 1:
            raise ValueError("MultiScaleFlipAug3D.pts_scale_ratio must be 1")
        if augmentation.get("flip", False) is not False:
            raise ValueError("MultiScaleFlipAug3D.flip must be false")

    scale_stage = _single_pipeline_stage(stages, "RandomScaleImageMultiViewImage")
    image_scale = 1.0
    if scale_stage is not None:
        scales = scale_stage.get("scales")
        if (
            not isinstance(scales, (list, tuple))
            or len(scales) != 1
            or not isinstance(scales[0], (int, float))
            or isinstance(scales[0], bool)
            or scales[0] <= 0
        ):
            raise ValueError("RandomScaleImageMultiViewImage.scales must contain one positive scale")
        image_scale = float(scales[0])

    pad_stage = _single_pipeline_stage(stages, "PadMultiViewImage")
    image_size_divisor = 1
    if pad_stage is not None:
        image_size_divisor = pad_stage.get("size_divisor")
        if not isinstance(image_size_divisor, int) or isinstance(image_size_divisor, bool) or image_size_divisor < 1:
            raise ValueError("PadMultiViewImage.size_divisor must be a positive integer")
        if pad_stage.get("size") is not None:
            raise ValueError("fixed PadMultiViewImage.size is unsupported")
        if pad_stage.get("pad_val", 0) != 0:
            raise ValueError("non-zero PadMultiViewImage.pad_val is unsupported")

    return camera_image_size, image_scale, image_size_divisor


def _validate_transformer_flags(transformer):
    for flag in ("rotate_prev_bev", "use_shift", "use_can_bus", "can_bus_norm", "use_cams_embeds"):
        if flag in transformer and transformer[flag] is not True:
            raise ValueError(f"transformer.{flag}={transformer[flag]!r} is unsupported; only True is supported")


def convert_upstream_config(source_text: str, variant: str):
    values = _assignments(source_text)
    config = load_builtin_config(variant)
    model = values.get("model")
    if not isinstance(model, dict):
        raise ValueError("upstream config has no statically readable model dictionary")
    backbone = model.get("img_backbone", {})
    neck = model.get("img_neck", {})
    head = model.get("pts_bbox_head", {})
    transformer = head.get("transformer", {})
    encoder = transformer.get("encoder", {})
    decoder = transformer.get("decoder", {})
    encoder_layer = encoder.get("transformerlayers", {})
    decoder_layer = decoder.get("transformerlayers", {})
    encoder_attentions = encoder_layer.get("attn_cfgs", [{}, {}])
    decoder_attentions = decoder_layer.get("attn_cfgs", [{}, {}])
    spatial = encoder_attentions[1].get("deformable_attention", {})
    temporal = encoder_attentions[0]
    decoder_cross = decoder_attentions[1]
    coder = head.get("bbox_coder", {})
    normalization = values.get("img_norm_cfg", {})
    class_names = tuple(values.get("class_names", config.class_names))
    point_cloud_range = tuple(values.get("point_cloud_range", config.point_cloud_range))
    camera_image_size, image_scale, image_size_divisor = _inference_input_settings(values, config)
    _validate_transformer_flags(transformer)

    converted = replace(
        config,
        name=f"bevformer_{variant}",
        class_names=class_names,
        camera_image_size=camera_image_size,
        image_scale=image_scale,
        image_mean=tuple(normalization.get("mean", config.image_mean)),
        image_std=tuple(normalization.get("std", config.image_std)),
        color_order="RGB" if normalization.get("to_rgb", config.color_order == "RGB") else "BGR",
        image_size_divisor=image_size_divisor,
        resnet_depth=backbone.get("depth", config.resnet_depth),
        resnet_style=backbone.get("style", config.resnet_style),
        with_dcn=bool(backbone.get("dcn", config.with_dcn)),
        out_indices=tuple(backbone.get("out_indices", config.out_indices)),
        num_levels=neck.get("num_outs", config.num_levels),
        bev_size=head.get("bev_h", config.bev_size),
        embed_dims=transformer.get("embed_dims", config.embed_dims),
        num_heads=decoder_attentions[0].get("num_heads", config.num_heads),
        feedforward_channels=encoder_layer.get("feedforward_channels", config.feedforward_channels),
        encoder_num_layers=encoder.get("num_layers", config.encoder_num_layers),
        decoder_num_layers=decoder.get("num_layers", config.decoder_num_layers),
        spatial_num_points=spatial.get("num_points", config.spatial_num_points),
        temporal_num_points=temporal.get("num_points", config.temporal_num_points),
        decoder_num_points=decoder_cross.get("num_points", config.decoder_num_points),
        num_points_in_pillar=encoder.get("num_points_in_pillar", config.num_points_in_pillar),
        num_query=head.get("num_query", config.num_query),
        point_cloud_range=point_cloud_range,
        rotate_center=tuple(transformer.get("rotate_center", config.rotate_center)),
        post_center_range=tuple(coder.get("post_center_range", config.post_center_range)),
        max_detections=coder.get("max_num", config.max_detections),
    )
    return type(config).from_dict(converted.to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert an upstream BEVFormer Python config without executing it")
    parser.add_argument("--input", required=True, help="Local path or HTTPS URL to an upstream .py config")
    parser.add_argument("--output", required=True, help="Normalized JSON output path")
    parser.add_argument("--variant", choices=SUPPORTED_VARIANTS, help="Override variant inference from the input name")
    parser.add_argument("--sha256", help="Expected SHA-256 of the input")
    parser.add_argument("--cache", help="Directory used to cache a remote input")
    parser.add_argument("--refresh", action="store_true", help="Refetch a cached remote input")
    args = parser.parse_args(argv)
    content, origin = read_source_bytes(
        args.input,
        sha256=args.sha256,
        cache_dir=args.cache,
        refresh=args.refresh,
    )
    variant = args.variant or variant_from_source(origin)
    if variant is None:
        raise ValueError("could not infer tiny/small/base from the input name; pass --variant")
    try:
        source_text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{origin}: upstream config must be UTF-8 Python source") from error
    config = convert_upstream_config(source_text, variant)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
