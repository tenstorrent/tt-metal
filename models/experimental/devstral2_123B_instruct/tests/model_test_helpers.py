# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for single-layer ``TtMinistral3Model`` tests (device perf profiling)."""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    load_ministral3_model_weights,
    require_model_weights,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL

NUM_LAYERS = 1


def mesh_device_param() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


class ModelTestFixtures(NamedTuple):
    text_cfg: Ministral3Config
    ref: Ministral3Model
    tt_model: TtMinistral3Model


def _shallow_config(text_cfg: Ministral3Config, num_layers: int = NUM_LAYERS) -> Ministral3Config:
    cfg_dict = text_cfg.to_dict()
    cfg_dict["num_hidden_layers"] = num_layers
    lt = cfg_dict.get("layer_types")
    if isinstance(lt, (list, tuple)) and len(lt) > num_layers:
        cfg_dict["layer_types"] = list(lt)[:num_layers]
    return Ministral3Config(**cfg_dict)


def input_ids_to_tt(input_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def current_pos_to_tt(positions: torch.Tensor, mesh_device) -> ttnn.Tensor:
    pos = positions.reshape(-1).to(torch.int32)
    return ttnn.from_torch(
        pos,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def setup_devstral_ministral3_partial_one_layer(mesh_device, *, max_seq_len: int) -> ModelTestFixtures:
    text_cfg = require_text_config()
    ref_cfg = _shallow_config(text_cfg, NUM_LAYERS)
    state_dict = require_model_weights(NUM_LAYERS)

    ref_cfg._attn_implementation = "eager"
    ref = Ministral3Model(ref_cfg).to(dtype=torch.bfloat16).eval()
    load_ministral3_model_weights(ref, state_dict)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMinistral3Model(args, mesh_device, state_dict, tt_ccl, num_layers=NUM_LAYERS)
    return ModelTestFixtures(text_cfg=ref_cfg, ref=ref, tt_model=tt_model)
