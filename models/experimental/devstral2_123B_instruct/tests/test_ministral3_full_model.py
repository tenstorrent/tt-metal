# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC (decode only): Hugging Face ``Ministral3Model`` vs ``TtMinistral3Model``.

Downloads all decoder-layer weights from the Hub (embed_tokens, norm, and every
``layers.<i>`` block) and compares one decode step after a short prefill fill.

- **Decode test:** prefill 128 tokens, then decode at position 128 (**129 tokens** total for
  one HF ``ref()`` forward), matching ``test_ministral3_single_layer.py``.

**Note:** The HF reference model is built and run with this same ``MAX_SEQ_LEN`` budget (128).
Do not raise it here to match ``DEVSTRAL2_TEST_MAX_SEQ_LEN`` — loading/running the full
123B reference at a larger sequence length will OOM on typical host DRAM.

This test loads the full 123B checkpoint's tensors; mark accordingly and only run on
machines with sufficient DRAM and a populated HF cache (or set ``HF_TOKEN`` for gated access).
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
from models.experimental.devstral2_123B_instruct.tests._devstral_weights import (
    load_ministral3_model_weights,
    require_model_weights,
    require_text_config,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_123B_instruct.tt.tt_ministral3_model import TtMinistral3Model
from models.experimental.devstral2_123B_instruct.tt.weight_loading import resolve_weight_cache_path
from models.tt_transformers.tt.ccl import TT_CCL

PCC_REQUIRED = 0.99
# On-disk layer-weight cache built with ``max_seq_len=128`` (``…/seq_128/``).
TT_WEIGHT_CACHE_SEQ_LEN = 128


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


def _seq_128_weight_cache_path(mesh_device, text_cfg: Ministral3Config, num_layers: int) -> str:
    """Path to ``…/layers_{N}/seq_128/`` (layer weights already cached there)."""
    args_128 = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=TT_WEIGHT_CACHE_SEQ_LEN,
        max_batch_size=1,
    )
    path = resolve_weight_cache_path(None, args_128, num_layers=num_layers)
    assert path is not None
    return path


def _setup_full_model(
    mesh_device,
    *,
    max_seq_len: int,
    reuse_seq_128_weight_cache: bool = False,
) -> _FullModelFixtures:
    """Build HF reference and TT model with all decoder layers from the Hub checkpoint."""
    text_cfg = require_text_config()
    num_layers = text_cfg.num_hidden_layers
    weight_cache_path = (
        _seq_128_weight_cache_path(mesh_device, text_cfg, num_layers) if reuse_seq_128_weight_cache else None
    )
    logger.info(
        f"Loading full model: {num_layers} decoder layers "
        f"(max_seq_len={max_seq_len}, weight_cache_path={weight_cache_path})"
    )

    state_dict = require_model_weights(num_layers)

    ref_cfg = Ministral3Config(**text_cfg.to_dict())
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
    tt_model = TtMinistral3Model(
        args,
        mesh_device,
        state_dict,
        tt_ccl,
        weight_cache_path=weight_cache_path,
    )
    return _FullModelFixtures(text_cfg=ref_cfg, ref=ref, tt_model=tt_model, num_layers=num_layers)


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
@pytest.mark.timeout(0)
def test_ministral3_model_pcc_devstral2_123B_instruct_full_weights_all_layers_decode(
    mesh_device,
    batch_size,
):
    """Prefill 128 tokens (KV fill), then one decode step; PCC vs HF last position over 129 tokens."""
    prefill_seq_len = 128
    decode_pos = prefill_seq_len
    tt_max_seq_len = decode_pos + 1

    fixtures = _setup_full_model(
        mesh_device,
        max_seq_len=tt_max_seq_len,
        reuse_seq_128_weight_cache=True,
    )
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

    tt_model(_input_ids_to_tt(input_ids_prefill, mesh_device), mode="prefill", start_pos=0)
    current_pos_tt = _current_pos_to_tt(torch.tensor([decode_pos], dtype=torch.long), mesh_device)
    tt_out = tt_model(
        _input_ids_to_tt(input_ids_decode, mesh_device),
        mode="decode",
        current_pos=current_pos_tt,
    )
    tt_torch = _tt_hidden_to_torch_ref_shape(tt_out, mesh_device, text_cfg.hidden_size, ref_decode.shape)

    _assert_pcc(ref_decode, tt_torch, label=f"Full model ({fixtures.num_layers} layers), decode")
