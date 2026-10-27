# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Schema for the portable subset of an upstream BEVFormer configuration."""

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = 1
SUPPORTED_VARIANTS = ("tiny", "small", "base")


def _tuple_of(value: Any, length: int, field: str, element_type):
    if not isinstance(value, list) or len(value) != length or not all(isinstance(item, element_type) for item in value):
        type_name = (
            "/".join(item.__name__ for item in element_type)
            if isinstance(element_type, tuple)
            else element_type.__name__
        )
        raise ValueError(f"{field} must be a {length}-element list of {type_name} values")
    return tuple(value)


def _mapping(raw: dict, field: str) -> dict:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _only(mapping: dict, allowed: set[str], context: str):
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"{context} has unsupported fields: {sorted(unknown)}")


@dataclass(frozen=True)
class ModelConfig:
    """The model, preprocessing and ordered-camera contract required for inference."""

    name: str
    variant: str
    class_names: tuple[str, ...]
    camera_order: tuple[str, ...]
    can_bus_dims: int
    camera_image_size: tuple[int, int]
    image_scale: float
    color_order: str
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    image_size_divisor: int
    resnet_depth: int
    resnet_style: str
    with_dcn: bool
    out_indices: tuple[int, ...]
    num_levels: int
    bev_size: int
    embed_dims: int
    num_heads: int
    feedforward_channels: int
    encoder_num_layers: int
    decoder_num_layers: int
    spatial_num_points: int
    temporal_num_points: int
    decoder_num_points: int
    num_points_in_pillar: int
    num_query: int
    point_cloud_range: tuple[float, float, float, float, float, float]
    rotate_center: tuple[float, float]
    post_center_range: tuple[float, float, float, float, float, float]
    max_detections: int

    def __post_init__(self):
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("class_names must not contain duplicates")
        if any(size < 1 for size in self.camera_image_size):
            raise ValueError("input.camera_image_size values must be positive")
        if any(value == 0 for value in self.image_std):
            raise ValueError("input.std values must be non-zero")
        if len(set(self.out_indices)) != len(self.out_indices) or any(
            index not in range(4) for index in self.out_indices
        ):
            raise ValueError("backbone.out_indices must contain unique ResNet stage indices from 0 through 3")
        if self.num_levels < len(self.out_indices):
            raise ValueError("fpn.num_levels must be at least the number of backbone.out_indices")
        if self.embed_dims % self.num_heads:
            raise ValueError("transformer.embed_dims must be divisible by transformer.num_heads")
        if any(low >= high for low, high in zip(self.point_cloud_range[:3], self.point_cloud_range[3:], strict=True)):
            raise ValueError("transformer.point_cloud_range minima must be below its maxima")
        if any(low >= high for low, high in zip(self.post_center_range[:3], self.post_center_range[3:], strict=True)):
            raise ValueError("box_coder.post_center_range minima must be below its maxima")

    @property
    def num_cameras(self) -> int:
        return len(self.camera_order)

    @property
    def raw_image_size(self) -> tuple[int, int]:
        return tuple(round(size * self.image_scale) for size in self.camera_image_size)

    def to_dict(self) -> dict:
        """Return the normalized, versioned JSON representation."""
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "variant": self.variant,
            "class_names": list(self.class_names),
            "input": {
                "camera_order": list(self.camera_order),
                "can_bus_dims": self.can_bus_dims,
                "camera_image_size": list(self.camera_image_size),
                "image_scale": self.image_scale,
                "color_order": self.color_order,
                "mean": list(self.image_mean),
                "std": list(self.image_std),
                "size_divisor": self.image_size_divisor,
            },
            "backbone": {
                "depth": self.resnet_depth,
                "style": self.resnet_style,
                "with_dcn": self.with_dcn,
                "out_indices": list(self.out_indices),
            },
            "fpn": {"num_levels": self.num_levels},
            "transformer": {
                "bev_size": self.bev_size,
                "embed_dims": self.embed_dims,
                "num_heads": self.num_heads,
                "feedforward_channels": self.feedforward_channels,
                "encoder_layers": self.encoder_num_layers,
                "decoder_layers": self.decoder_num_layers,
                "spatial_num_points": self.spatial_num_points,
                "temporal_num_points": self.temporal_num_points,
                "decoder_num_points": self.decoder_num_points,
                "num_points_in_pillar": self.num_points_in_pillar,
                "num_query": self.num_query,
                "point_cloud_range": list(self.point_cloud_range),
                "rotate_center": list(self.rotate_center),
            },
            "box_coder": {
                "post_center_range": list(self.post_center_range),
                "max_detections": self.max_detections,
            },
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        if not isinstance(raw, dict):
            raise ValueError("top-level config must be an object")
        _only(
            raw,
            {
                "schema_version",
                "name",
                "variant",
                "class_names",
                "input",
                "backbone",
                "fpn",
                "transformer",
                "box_coder",
            },
            "config",
        )
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

        input_config = _mapping(raw, "input")
        backbone = _mapping(raw, "backbone")
        fpn = _mapping(raw, "fpn")
        transformer = _mapping(raw, "transformer")
        box_coder = _mapping(raw, "box_coder")
        _only(
            input_config,
            {
                "camera_order",
                "can_bus_dims",
                "camera_image_size",
                "image_scale",
                "color_order",
                "mean",
                "std",
                "size_divisor",
            },
            "input",
        )
        _only(backbone, {"depth", "style", "with_dcn", "out_indices"}, "backbone")
        _only(fpn, {"num_levels"}, "fpn")
        _only(
            transformer,
            {
                "bev_size",
                "embed_dims",
                "num_heads",
                "feedforward_channels",
                "encoder_layers",
                "decoder_layers",
                "spatial_num_points",
                "temporal_num_points",
                "decoder_num_points",
                "num_points_in_pillar",
                "num_query",
                "point_cloud_range",
                "rotate_center",
            },
            "transformer",
        )
        _only(box_coder, {"post_center_range", "max_detections"}, "box_coder")

        variant = raw.get("variant")
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variant is {variant!r}, expected one of {SUPPORTED_VARIANTS}")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        class_names = raw.get("class_names")
        if (
            not isinstance(class_names, list)
            or not class_names
            or not all(isinstance(item, str) and item for item in class_names)
        ):
            raise ValueError("class_names must be a non-empty list of strings")
        camera_order = input_config.get("camera_order")
        if (
            not isinstance(camera_order, list)
            or not camera_order
            or not all(isinstance(item, str) and item for item in camera_order)
        ):
            raise ValueError("input.camera_order must be a non-empty list of strings")
        if len(set(camera_order)) != len(camera_order):
            raise ValueError("input.camera_order must not contain duplicates")

        integer_fields = {
            "input.can_bus_dims": input_config.get("can_bus_dims"),
            "input.size_divisor": input_config.get("size_divisor"),
            "backbone.depth": backbone.get("depth"),
            "fpn.num_levels": fpn.get("num_levels"),
            "transformer.bev_size": transformer.get("bev_size"),
            "transformer.embed_dims": transformer.get("embed_dims"),
            "transformer.num_heads": transformer.get("num_heads"),
            "transformer.feedforward_channels": transformer.get("feedforward_channels"),
            "transformer.encoder_layers": transformer.get("encoder_layers"),
            "transformer.decoder_layers": transformer.get("decoder_layers"),
            "transformer.spatial_num_points": transformer.get("spatial_num_points"),
            "transformer.temporal_num_points": transformer.get("temporal_num_points"),
            "transformer.decoder_num_points": transformer.get("decoder_num_points"),
            "transformer.num_points_in_pillar": transformer.get("num_points_in_pillar"),
            "transformer.num_query": transformer.get("num_query"),
            "box_coder.max_detections": box_coder.get("max_detections"),
        }
        for field, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        image_scale = input_config.get("image_scale")
        if not isinstance(image_scale, (int, float)) or isinstance(image_scale, bool) or image_scale <= 0:
            raise ValueError("input.image_scale must be positive")
        color_order = input_config.get("color_order")
        if color_order not in ("RGB", "BGR"):
            raise ValueError("input.color_order must be RGB or BGR")
        style = backbone.get("style")
        if style not in ("pytorch", "caffe"):
            raise ValueError("backbone.style must be pytorch or caffe")
        if not isinstance(backbone.get("with_dcn"), bool):
            raise ValueError("backbone.with_dcn must be a boolean")
        out_indices = backbone.get("out_indices")
        if (
            not isinstance(out_indices, list)
            or not out_indices
            or not all(isinstance(item, int) for item in out_indices)
        ):
            raise ValueError("backbone.out_indices must be a non-empty list of integers")

        return cls(
            name=name,
            variant=variant,
            class_names=tuple(class_names),
            camera_order=tuple(camera_order),
            can_bus_dims=input_config["can_bus_dims"],
            camera_image_size=_tuple_of(input_config.get("camera_image_size"), 2, "input.camera_image_size", int),
            image_scale=float(image_scale),
            color_order=color_order,
            image_mean=tuple(
                float(item) for item in _tuple_of(input_config.get("mean"), 3, "input.mean", (int, float))
            ),
            image_std=tuple(float(item) for item in _tuple_of(input_config.get("std"), 3, "input.std", (int, float))),
            image_size_divisor=input_config["size_divisor"],
            resnet_depth=backbone["depth"],
            resnet_style=style,
            with_dcn=backbone["with_dcn"],
            out_indices=tuple(out_indices),
            num_levels=fpn["num_levels"],
            bev_size=transformer["bev_size"],
            embed_dims=transformer["embed_dims"],
            num_heads=transformer["num_heads"],
            feedforward_channels=transformer["feedforward_channels"],
            encoder_num_layers=transformer["encoder_layers"],
            decoder_num_layers=transformer["decoder_layers"],
            spatial_num_points=transformer["spatial_num_points"],
            temporal_num_points=transformer["temporal_num_points"],
            decoder_num_points=transformer["decoder_num_points"],
            num_points_in_pillar=transformer["num_points_in_pillar"],
            num_query=transformer["num_query"],
            point_cloud_range=tuple(
                float(item)
                for item in _tuple_of(
                    transformer.get("point_cloud_range"), 6, "transformer.point_cloud_range", (int, float)
                )
            ),
            rotate_center=tuple(
                float(item)
                for item in _tuple_of(transformer.get("rotate_center"), 2, "transformer.rotate_center", (int, float))
            ),
            post_center_range=tuple(
                float(item)
                for item in _tuple_of(
                    box_coder.get("post_center_range"), 6, "box_coder.post_center_range", (int, float)
                )
            ),
            max_detections=box_coder["max_detections"],
        )
