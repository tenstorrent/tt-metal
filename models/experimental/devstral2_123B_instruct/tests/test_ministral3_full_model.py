# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC (decode only): Hugging Face ``Ministral3Model`` vs ``TtMinistral3Model``.

The HF reference is loaded with the same path as ``reference/devstral2_123b_inference.py``
(``AutoModelForCausalLM`` + ``FineGrainedFP8Config``, FP8 dequant compat patch, disk offload).
TT weights are bf16 tensors extracted from that loaded model (not shard download via
``_devstral_weights``).

- **Decode test:** prefill 128 tokens, then decode at position 128 (**129 tokens** total for
  one HF ``ref()`` forward), matching ``test_ministral3_single_layer.py``.

**Note:** The HF reference forward stays short (129 tokens for decode PCC). The TT model uses
``DEVSTRAL2_TEST_MAX_SEQ_LEN`` (98304) for KV/RoPE, matching other Devstral PCC tests.

This test loads the full checkpoint through Transformers; mark accordingly and only run on
machines with sufficient DRAM/disk offload space and a populated HF cache (or ``HF_TOKEN``).
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
from models.experimental.devstral2_123B_instruct.reference.hf_reference_loader import (
    extract_backbone_bf16_state_dict,
    load_devstral2_causal_lm,
    load_devstral2_text_config,
    prepare_ministral3_backbone_for_pcc,
)
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import DEVSTRAL2_TEST_MAX_SEQ_LEN
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3Model
from models.experimental.devstral2_123B_instruct.tt.weight_loading import resolve_weight_cache_path
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def _input_ids_to_tt(input_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Upload token indices ``[batch, seq]`` for ``ttnn.embedding`` on device."""
    return ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _current_pos_to_tt(positions: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """Upload decode position indices ``[batch]`` as int32 on device."""
    pos = positions.reshape(-1).to(torch.int32)
    return ttnn.from_torch(
        pos,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


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


def _assert_pcc(ref_out: torch.Tensor, tt_torch: torch.Tensor, *, label: str) -> None:
    passing, msg = comp_pcc(ref_out, tt_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_torch))
    logger.info(f"PCC ({label}): {msg}")
    assert passing, f"PCC below {PCC_REQUIRED}: {msg}"


class _FullModelFixtures(NamedTuple):
    text_cfg: Ministral3Config
    ref: Ministral3Model
    tt_model: TtMinistral3Model
    num_layers: int


def _skip_hf_load_failure(exc: BaseException) -> None:
    pytest.skip(
        "Could not load Devstral-2-123B via Hugging Face "
        "(set HF_TOKEN if gated, ensure offload disk space, or pre-cache weights). "
        f"Error: {exc}"
    )


def _weight_cache_path(mesh_device, text_cfg: Ministral3Config, num_layers: int) -> str:
    """Path to ``…/layers_{N}/seq_{DEVSTRAL2_TEST_MAX_SEQ_LEN}/`` for on-disk TT weight caches."""
    cache_args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=DEVSTRAL2_TEST_MAX_SEQ_LEN,
        max_batch_size=1,
    )
    path = resolve_weight_cache_path(None, cache_args, num_layers=num_layers)
    assert path is not None
    return path


def _setup_full_model(
    mesh_device,
    *,
    max_seq_len: int,
    reuse_weight_cache: bool = False,
) -> _FullModelFixtures:
    """Build HF reference (inference load path) and TT model from extracted bf16 weights."""
    try:
        text_cfg = load_devstral2_text_config()
    except Exception as exc:
        _skip_hf_load_failure(exc)

    num_layers = text_cfg.num_hidden_layers
    weight_cache_path = _weight_cache_path(mesh_device, text_cfg, num_layers) if reuse_weight_cache else None
    logger.info(
        f"Loading full model: {num_layers} decoder layers "
        f"(max_seq_len={max_seq_len}, weight_cache_path={weight_cache_path})"
    )

    try:
        causal_lm = load_devstral2_causal_lm()
    except Exception as exc:
        _skip_hf_load_failure(exc)

    ref = prepare_ministral3_backbone_for_pcc(causal_lm)
    state_dict = extract_backbone_bf16_state_dict(causal_lm, num_layers)

    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMinistral3Model(
        args,
        mesh_device,
        state_dict,
        tt_ccl,
        weight_cache_path=weight_cache_path,
    )
    return _FullModelFixtures(text_cfg=text_cfg, ref=ref, tt_model=tt_model, num_layers=num_layers)


@torch.no_grad()
@pytest.mark.slow
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
@pytest.mark.timeout(3600)
def test_ministral3_model_pcc_devstral2_123B_instruct_full_weights_all_layers_decode(
    mesh_device,
    batch_size,
):
    """Prefill 128 tokens (KV fill), then one decode step; PCC vs HF last position over 129 tokens."""
    prefill_seq_len = 128
    decode_pos = prefill_seq_len
    tt_max_seq_len = max(DEVSTRAL2_TEST_MAX_SEQ_LEN, decode_pos + 1)

    fixtures = _setup_full_model(
        mesh_device,
        max_seq_len=tt_max_seq_len,
        reuse_weight_cache=True,
    )
    text_cfg = fixtures.text_cfg
    ref = fixtures.ref
    tt_model = fixtures.tt_model

    ref_device = next(ref.parameters()).device

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    total_len = prefill_seq_len + 1
    input_ids_full = torch.randint(0, text_cfg.vocab_size, (batch_size, total_len), dtype=torch.long, generator=gen)
    input_ids_prefill = input_ids_full[:, :prefill_seq_len]
    input_ids_decode = input_ids_full[:, prefill_seq_len : prefill_seq_len + 1]

    position_ids_full = torch.arange(total_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    input_ids_full = input_ids_full.to(ref_device)
    position_ids_full = position_ids_full.to(ref_device)
    inputs_embeds_full = ref.embed_tokens(input_ids_full)
    causal_mask_full = create_causal_mask(
        config=text_cfg,
        inputs_embeds=inputs_embeds_full,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids_full,
    )
    ref_decode = (
        ref(
            input_ids=input_ids_full,
            attention_mask=causal_mask_full,
            position_ids=position_ids_full,
            use_cache=False,
        )
        .last_hidden_state[:, -1:, :]
        .cpu()
    )

    tt_model(_input_ids_to_tt(input_ids_prefill, mesh_device), mode="prefill", start_pos=0)
    current_pos_tt = _current_pos_to_tt(torch.tensor([decode_pos], dtype=torch.long), mesh_device)
    tt_out = tt_model(
        _input_ids_to_tt(input_ids_decode, mesh_device),
        mode="decode",
        current_pos=current_pos_tt,
    )
    tt_torch = _tt_hidden_to_torch_ref_shape(tt_out, mesh_device, text_cfg.hidden_size, ref_decode.shape)

    _assert_pcc(ref_decode, tt_torch, label=f"Full model ({fixtures.num_layers} layers), decode")
