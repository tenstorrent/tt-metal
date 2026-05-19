# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC: Hugging Face ``Ministral3Model`` vs ``TtMinistral3Model`` (Devstral-2 large stack, real Hub weights).

Loads **partial** tensors from the Hub (``model.embed_tokens``, ``model.norm``, ``model.layers.0`` only)
and compares hidden states to a HF reference built from the same tensors.

- **Prefill test:** full-sequence prefill PCC at ``seq_len=128`` (one decoder layer).
- **Decode test:** prefill 128 tokens to fill KV cache, then decode one token at index 128; the HF
  reference is the last position of a single forward over 129 tokens (equivalent under causal attention).

``Devstral2Args`` / ``TtMinistral3Model`` are configured for **one decoder layer** so the tests never
load the full 123B checkpoint into host RAM.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstral2_large.tests._devstral_weights import (
    load_hf_tensors_for_keys,
    load_ministral3_model_weights,
    load_text_config,
    model_prefill_weight_keys,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99
NUM_LAYERS = 1


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 4))


def _shallow_config(text_cfg: Ministral3Config, num_layers: int = NUM_LAYERS) -> Ministral3Config:
    cfg_dict = text_cfg.to_dict()
    cfg_dict["num_hidden_layers"] = num_layers
    lt = cfg_dict.get("layer_types")
    if isinstance(lt, (list, tuple)) and len(lt) > num_layers:
        cfg_dict["layer_types"] = list(lt)[:num_layers]
    return Ministral3Config(**cfg_dict)


def _tt_hidden_to_torch_ref_shape(
    tt_out: ttnn.Tensor,
    mesh_device,
    hidden_size: int,
    ref_shape: torch.Size,
) -> torch.Tensor:
    """Convert TT hidden states to torch, slicing tile-padded seq/batch dims to match ``ref_shape``."""
    out_last = int(tt_out.shape[-1])
    if out_last == hidden_size:
        tt_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0])
    else:
        tt_torch = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    if tt_torch.ndim == 4:
        tt_torch = tt_torch[0:1]
    tt_torch = tt_torch[..., : ref_shape[-2], :]
    return tt_torch.reshape(ref_shape)


class _ModelPccFixtures(NamedTuple):
    text_cfg: Ministral3Config
    ref: Ministral3Model
    tt_model: TtMinistral3Model


def _setup_devstral_ministral3_partial_one_layer(
    mesh_device,
    *,
    max_seq_len: int,
) -> _ModelPccFixtures:
    try:
        text_cfg = load_text_config()
    except Exception as exc:
        pytest.skip(f"Could not load Devstral-2-123B HF config: {exc}")

    ref_cfg = _shallow_config(text_cfg, NUM_LAYERS)
    keys = model_prefill_weight_keys(NUM_LAYERS)
    try:
        state_dict = load_hf_tensors_for_keys(keys)
    except Exception as exc:
        pytest.skip(f"Could not download Devstral-2-123B model weights ({NUM_LAYERS} layer(s)): {exc}")

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
    return _ModelPccFixtures(text_cfg=ref_cfg, ref=ref, tt_model=tt_model)


def _assert_pcc(ref_out: torch.Tensor, tt_torch: torch.Tensor, *, label: str) -> None:
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC ({label}): {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_ministral3_model_pcc_devstral2_large_partial_weights_one_layer_prefill(
    mesh_device,
    seq_len,
    batch_size,
):
    fixtures = _setup_devstral_ministral3_partial_one_layer(mesh_device, max_seq_len=max(512, seq_len))
    text_cfg = fixtures.text_cfg
    ref = fixtures.ref
    tt_model = fixtures.tt_model

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size, seq_len), dtype=torch.long, generator=gen)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    inputs_embeds = ref.embed_tokens(input_ids)
    causal_mask = create_causal_mask(
        config=text_cfg,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )

    ref_out = ref(
        input_ids=input_ids,
        attention_mask=causal_mask,
        position_ids=position_ids,
        use_cache=False,
    ).last_hidden_state

    tt_out = tt_model(input_ids, mode="prefill", start_pos=0)
    tt_torch = _tt_hidden_to_torch_ref_shape(tt_out, mesh_device, text_cfg.hidden_size, ref_out.shape)

    _assert_pcc(ref_out, tt_torch, label="Ministral3Model partial Hub weights, prefill")


@torch.no_grad()
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
@pytest.mark.parametrize("batch_size", (1,))
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 30000000,
            "num_command_queues": 1,
            "l1_small_size": DEVSTRAL2_LARGE_L1_SMALL_SIZE,
        }
    ],
    indirect=True,
)
def test_ministral3_model_pcc_devstral2_large_partial_weights_one_layer_decode(
    mesh_device,
    batch_size,
):
    """Prefill 128 tokens (KV fill), then one decode step vs HF last position over 129 tokens."""
    prefill_seq_len = 128
    decode_pos = prefill_seq_len

    fixtures = _setup_devstral_ministral3_partial_one_layer(mesh_device, max_seq_len=max(512, decode_pos + 1))
    text_cfg = fixtures.text_cfg
    ref = fixtures.ref
    tt_model = fixtures.tt_model

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    total_len = prefill_seq_len + 1
    input_ids_full = torch.randint(0, text_cfg.vocab_size, (batch_size, total_len), dtype=torch.long, generator=gen)
    input_ids_prefill = input_ids_full[:, :prefill_seq_len]
    input_ids_decode = input_ids_full[:, prefill_seq_len : prefill_seq_len + 1]

    position_ids_full = torch.arange(total_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    inputs_embeds_full = ref.embed_tokens(input_ids_full)
    causal_mask_full = create_causal_mask(
        config=text_cfg,
        inputs_embeds=inputs_embeds_full,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids_full,
    )
    ref_decode = ref(
        input_ids=input_ids_full,
        attention_mask=causal_mask_full,
        position_ids=position_ids_full,
        use_cache=False,
    ).last_hidden_state[:, -1:, :]

    tt_model(input_ids_prefill, mode="prefill", start_pos=0)
    current_pos_host = torch.tensor([decode_pos], dtype=torch.long)
    tt_out = tt_model(
        input_ids_decode,
        mode="decode",
        current_pos_host=current_pos_host,
    )
    tt_torch = _tt_hidden_to_torch_ref_shape(tt_out, mesh_device, text_cfg.hidden_size, ref_decode.shape)

    _assert_pcc(ref_decode, tt_torch, label="Ministral3Model partial Hub weights, decode")
