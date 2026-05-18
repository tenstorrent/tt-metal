# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: TT ``TtMinistral3Model`` vs HF ``Ministral3Model`` on Devstral-2-123B weights (partial depth).

Covers prefill (full sequence) and decode (prefill KV cache, then one token at ``prefill_seq_len``).
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests._devstral_weights import (
    load_hf_tensors_for_keys,
    load_ministral3_model_weights,
    load_text_config,
    model_prefill_weight_keys,
    replicated_tt_to_torch,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99


def _mesh_shape_from_env() -> tuple[int, int]:
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "P150x4": (1, 4),
        "T3K": (1, 8),
    }.get(os.environ.get("MESH_DEVICE", "P150x4"), (1, 4))


def _shallow_config(text_cfg, num_layers: int) -> Ministral3Config:
    """HF model with the same shapes as Devstral-2-123B but only ``num_layers`` blocks."""
    cfg_dict = text_cfg.to_dict()
    cfg_dict["num_hidden_layers"] = num_layers
    return Ministral3Config(**cfg_dict)


class _ModelPccFixtures(NamedTuple):
    text_cfg: object
    ref: Ministral3Model
    tt_model: TtMinistral3Model


def _load_model_pcc_fixtures(mesh_device, num_layers: int, *, max_seq_len: int) -> _ModelPccFixtures:
    try:
        text_cfg = load_text_config()
    except Exception as exc:
        pytest.skip(f"Could not load Devstral-2-123B HF config: {exc}")

    keys = model_prefill_weight_keys(num_layers)
    try:
        state_dict = load_hf_tensors_for_keys(keys)
    except Exception as exc:
        pytest.skip(f"Could not download Devstral-2-123B model weights ({num_layers} layer(s)): {exc}")

    ref_cfg = _shallow_config(text_cfg, num_layers)
    ref = Ministral3Model(ref_cfg).to(dtype=torch.bfloat16).eval()
    load_ministral3_model_weights(ref, state_dict)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMinistral3Model(args, mesh_device, state_dict, tt_ccl, num_layers=num_layers)
    return _ModelPccFixtures(text_cfg=text_cfg, ref=ref, tt_model=tt_model)


def _assert_pcc(ref_out: torch.Tensor, tt_torch: torch.Tensor) -> None:
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC: {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("seq_len", [32])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_full_model_prefill_pcc_real_weights(mesh_device, seq_len, num_layers):
    fixtures = _load_model_pcc_fixtures(mesh_device, num_layers, max_seq_len=max(512, seq_len))
    text_cfg = fixtures.text_cfg
    ref = fixtures.ref
    tt_model = fixtures.tt_model

    input_ids = torch.randint(0, text_cfg.vocab_size, (1, seq_len), dtype=torch.long)
    ref_out = ref(input_ids=input_ids).last_hidden_state

    tt_out = tt_model(input_ids, mode="prefill", start_pos=0)
    tt_torch = replicated_tt_to_torch(tt_out, reshape=(1, seq_len, text_cfg.hidden_size))

    _assert_pcc(ref_out, tt_torch)


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize("prefill_seq_len", [32])
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE}],
    indirect=True,
)
def test_full_model_decode_pcc_real_weights(mesh_device, prefill_seq_len, num_layers):
    decode_pos = prefill_seq_len
    fixtures = _load_model_pcc_fixtures(mesh_device, num_layers, max_seq_len=max(512, decode_pos + 1))
    text_cfg = fixtures.text_cfg
    ref = fixtures.ref
    tt_model = fixtures.tt_model

    input_ids = torch.randint(0, text_cfg.vocab_size, (1, prefill_seq_len + 1), dtype=torch.long)
    input_ids_prefill = input_ids[:, :prefill_seq_len]
    input_ids_decode = input_ids[:, decode_pos : decode_pos + 1]

    past = DynamicCache()
    ref(input_ids=input_ids_prefill, past_key_values=past, use_cache=True)
    ref_out = ref(
        input_ids=input_ids_decode,
        past_key_values=past,
        use_cache=True,
    ).last_hidden_state

    tt_model(input_ids_prefill, mode="prefill", start_pos=0)
    current_pos_host = torch.tensor([decode_pos], dtype=torch.long)
    tt_out = tt_model(
        input_ids_decode,
        mode="decode",
        current_pos_host=current_pos_host,
    )
    tt_torch = replicated_tt_to_torch(tt_out)
    # Decode activations may carry a tile-padded batch dim; keep the first user slot.
    if tt_torch.shape[2] > 1:
        tt_torch = tt_torch[:, :, :1, :]
    tt_torch = tt_torch.reshape(1, 1, text_cfg.hidden_size)

    _assert_pcc(ref_out, tt_torch)
