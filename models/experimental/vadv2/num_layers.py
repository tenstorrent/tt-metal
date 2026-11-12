# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reduced-layer execution for kernel coverage runs.

Every stack of identical layers in VAD runs the same kernels on the same shapes for each repeat, so
keeping a few repeats still exercises every kernel the full model launches. This is meant for
simulator bring-up, where each launch is expensive; a reduced model is not the trained planner and
its outputs are meaningless.
"""

import os

from loguru import logger

NUM_LAYERS_ENV = "VAD_NUM_LAYERS"


def num_layers_from_env() -> int | None:
    value = os.environ.get(NUM_LAYERS_ENV)
    if value is None or value == "":
        return None
    num_layers = int(value)
    if num_layers < 1:
        raise ValueError(f"{NUM_LAYERS_ENV} must be at least 1, got {num_layers}")
    return num_layers


def reduce_layers(vad, num_layers: int) -> None:
    """Keep `num_layers` repeats of every stack of identical layers of a reference VAD, in place.

    Stacks reduced: the BEV encoder layers, the detection and map decoder layers with the class and
    box branches of each decoder layer, and the repeated blocks of each ResNet stage. A ResNet stage
    keeps its first block in addition, since that block alone strides and projects the shortcut. The
    motion and planning decoders have one layer already. Call it after loading the weights, which
    are loaded strictly.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}")

    backbone = vad.img_backbone
    for name in ("layer1", "layer2", "layer3", "layer4"):
        stage = getattr(backbone, name)
        setattr(backbone, name, stage[: 1 + num_layers])

    head = vad.pts_bbox_head
    encoder = head.transformer.encoder
    encoder.layers = encoder.layers[:num_layers]
    encoder.num_layers = len(encoder.layers)

    decoder = head.transformer.decoder
    decoder.layers = decoder.layers[:num_layers]
    head.cls_branches = head.cls_branches[: len(decoder.layers)]
    head.reg_branches = head.reg_branches[: len(decoder.layers)]

    map_decoder = head.transformer.map_decoder
    map_decoder.layers = map_decoder.layers[:num_layers]
    head.map_cls_branches = head.map_cls_branches[: len(map_decoder.layers)]
    head.map_reg_branches = head.map_reg_branches[: len(map_decoder.layers)]


def log_layer_counts(vad, num_layers: int | None) -> None:
    backbone = vad.img_backbone
    transformer = vad.pts_bbox_head.transformer
    reduced = f" ({NUM_LAYERS_ENV}={num_layers})" if num_layers is not None else ""
    logger.info(
        f"VAD layers: backbone blocks {[len(getattr(backbone, f'layer{i}')) for i in range(1, 5)]}, "
        f"{len(transformer.encoder.layers)} encoder, {len(transformer.decoder.layers)} decoder, "
        f"{len(transformer.map_decoder.layers)} map decoder{reduced}"
    )
