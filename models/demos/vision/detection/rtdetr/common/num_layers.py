# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reduced-layer execution for kernel coverage runs.

Every stack of identical layers in RT-DETR runs the same kernels on the same shapes for each
repeat, so keeping a few repeats still exercises every kernel the full model launches. This is
meant for simulator bring-up, where each launch is expensive; a reduced model is not the
pretrained detector and its detections are meaningless.
"""

import os

import torch
from loguru import logger

NUM_LAYERS_ENV = "RTDETR_NUM_LAYERS"


def num_layers_from_env() -> int | None:
    value = os.environ.get(NUM_LAYERS_ENV)
    if value is None or value == "":
        return None
    num_layers = int(value)
    if num_layers < 1:
        raise ValueError(f"{NUM_LAYERS_ENV} must be at least 1, got {num_layers}")
    return num_layers


def reduce_layers(torch_model: torch.nn.Module, num_layers: int) -> None:
    """Keep `num_layers` repeats of every stack of identical layers, in place.

    Takes an RT-DETR or RT-DETRv2 object-detection model. Stacks reduced: decoder layers with
    their class and box heads, AIFI encoder layers, CSP RepVgg bottlenecks, and the repeated
    blocks of each ResNet stage. A ResNet stage keeps its first block in addition, since that
    block alone strides and projects the shortcut. The config is updated to match, so a TT model
    built from this model and its config has the same layers.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}")

    model = torch_model.model
    config = torch_model.config

    decoder = model.decoder
    decoder.layers = decoder.layers[:num_layers]
    decoder.class_embed = decoder.class_embed[:num_layers]
    decoder.bbox_embed = decoder.bbox_embed[:num_layers]
    if hasattr(torch_model, "class_embed"):
        # RT-DETRv2 also keeps the heads on the top-level module, shared with the decoder.
        torch_model.class_embed = decoder.class_embed
        torch_model.bbox_embed = decoder.bbox_embed
    config.decoder_layers = len(decoder.layers)

    for aifi in model.encoder.aifi:
        aifi.layers = aifi.layers[:num_layers]
    config.encoder_layers = min(config.encoder_layers, num_layers)

    for csp in [*model.encoder.fpn_blocks, *model.encoder.pan_blocks]:
        csp.bottlenecks = csp.bottlenecks[:num_layers]

    stages = model.backbone.model.encoder.stages
    for stage in stages:
        stage.layers = stage.layers[: 1 + num_layers]
    config.backbone_config.depths = [len(stage.layers) for stage in stages]


def load_reference_model(model_class, model_name: str, num_layers: int | None) -> torch.nn.Module:
    """Load the pretrained model in eval mode, reduced to `num_layers` repeats when given."""
    torch_model = model_class.from_pretrained(model_name).eval()
    if num_layers is not None:
        reduce_layers(torch_model, num_layers)
    return torch_model


def log_layer_counts(config, num_layers: int | None) -> None:
    reduced = f" ({NUM_LAYERS_ENV}={num_layers})" if num_layers is not None else ""
    logger.info(
        f"RT-DETR layers: backbone depths {list(config.backbone_config.depths)}, "
        f"{config.encoder_layers} AIFI encoder, {config.decoder_layers} decoder{reduced}"
    )
