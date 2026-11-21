# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Reduced-layer execution for kernel coverage runs.

Every stack of identical layers in RetinaNet runs the same kernels on the same shapes for each
repeat, so keeping a few repeats still exercises every kernel the full model launches. This is
meant for simulator bring-up, where each launch is expensive; a reduced model is not the
pretrained detector and its detections are meaningless.
"""

import os

import torch
from loguru import logger

NUM_LAYERS_ENV = "RETINANET_NUM_LAYERS"


def num_layers_from_env() -> int | None:
    value = os.environ.get(NUM_LAYERS_ENV)
    if value is None or value == "":
        return None
    num_layers = int(value)
    if num_layers < 1:
        raise ValueError(f"{NUM_LAYERS_ENV} must be at least 1, got {num_layers}")
    return num_layers


def reduce_layers(retinanet: torch.nn.Module, num_layers: int) -> None:
    """Keep ``num_layers`` repeats of every stack of identical layers of a torchvision RetinaNet,
    in place: the blocks of each ResNet stage, and the conv blocks of the classification and
    regression heads. A ResNet stage keeps its first block in addition, since that block alone
    projects the shortcut (and, from layer2 on, strides). The TT model takes its layer counts from the parameters
    preprocessed from this model, so it has the same layers."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be at least 1, got {num_layers}")
    body = retinanet.backbone.body
    stage_end = 1 + num_layers
    body.layer1 = body.layer1[:stage_end]
    body.layer2 = body.layer2[:stage_end]
    body.layer3 = body.layer3[:stage_end]
    body.layer4 = body.layer4[:stage_end]
    for head in (retinanet.head.classification_head, retinanet.head.regression_head):
        head.conv = head.conv[:num_layers]


def load_reduced(retinanet: torch.nn.Module) -> torch.nn.Module:
    """``retinanet`` reduced to the layer count ``RETINANET_NUM_LAYERS`` sets, if it is set."""
    num_layers = num_layers_from_env()
    if num_layers is not None:
        reduce_layers(retinanet, num_layers)
    log_layer_counts(retinanet, num_layers)
    return retinanet


def log_layer_counts(retinanet: torch.nn.Module, num_layers: int | None) -> None:
    reduced = f" ({NUM_LAYERS_ENV}={num_layers})" if num_layers is not None else ""
    body = retinanet.backbone.body
    depths = [len(body.layer1), len(body.layer2), len(body.layer3), len(body.layer4)]
    logger.info(
        f"RetinaNet layers: backbone depths {depths}, head conv blocks "
        f"{len(retinanet.head.classification_head.conv)}{reduced}"
    )
