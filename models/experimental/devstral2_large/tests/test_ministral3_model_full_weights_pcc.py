# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full-model TT performance (all Hub layers, random inputs, no HF reference).

Loads the full Devstral-2 / Ministral3 checkpoint onto device and times:
- **Prefill:** one forward over ``prefill_seq_len`` random token ids.
- **Decode:** KV fill prefill, then ``num_decode_tokens`` single-token decode steps.

Hub weights are required (set ``HF_TOKEN`` if gated). Activations are random integers in
``[0, vocab_size)`` — no PCC or reference forward.

Environment overrides:
- ``DEVSTRAL2_PREFILL_SEQ_LEN`` (default 128)
- ``DEVSTRAL2_NUM_DECODE_TOKENS`` (default 20)
- ``DEVSTRAL2_NUM_LAYERS`` (empty = all ``num_hidden_layers`` from config)
"""

from __future__ import annotations

import os
import time
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.devstral2_large.tests._devstral_weights import (
    require_model_weights,
    require_text_config,
)
from models.experimental.devstral2_large.tt.model_args import (
    DEVSTRAL2_LARGE_L1_SMALL_SIZE,
    Devstral2Args,
)
from models.experimental.devstral2_large.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL


def _mesh_device_param():
    return {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
    }.get(os.environ.get("MESH_DEVICE"), (1, 8))


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    return int(raw) if raw else default


def _env_num_layers(default: int) -> int:
    raw = os.environ.get("DEVSTRAL2_NUM_LAYERS", "")
    return int(raw) if raw else default


def _input_ids_to_tt(input_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return ttnn.from_torch(
        input_ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _current_pos_to_tt(positions: torch.Tensor, mesh_device) -> ttnn.Tensor:
    pos = positions.reshape(-1).to(torch.int32)
    return ttnn.from_torch(
        pos,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class _TtFullModel(NamedTuple):
    args: Devstral2Args
    tt_model: TtMinistral3Model
    num_layers: int
    vocab_size: int


def _setup_tt_full_model(mesh_device, *, max_seq_len: int) -> _TtFullModel:
    """Build ``TtMinistral3Model`` with all (or env-overridden) decoder layers; no HF reference."""
    text_cfg = require_text_config()
    num_layers = _env_num_layers(int(text_cfg.num_hidden_layers))
    logger.info(f"Loading TT model: {num_layers} / {text_cfg.num_hidden_layers} decoder layers")

    state_dict = require_model_weights(num_layers)
    args = Devstral2Args.from_hf_config(
        text_cfg,
        mesh_shape=tuple(mesh_device.shape),
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )
    tt_ccl = TT_CCL(mesh_device)
    tt_model = TtMinistral3Model(args, mesh_device, state_dict, tt_ccl)
    return _TtFullModel(args=args, tt_model=tt_model, num_layers=num_layers, vocab_size=text_cfg.vocab_size)


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
def test_ministral3_model_full_weights_all_layers_decode_perf(mesh_device, batch_size):
    """Prefill random ids (KV fill), then time ``num_decode_tokens`` decode steps (no HF reference)."""
    prefill_seq_len = _env_int("DEVSTRAL2_PREFILL_SEQ_LEN", 128)
    num_decode_tokens = _env_int("DEVSTRAL2_NUM_DECODE_TOKENS", 20)
    decode_pos_start = prefill_seq_len

    fixtures = _setup_tt_full_model(
        mesh_device,
        max_seq_len=max(512, decode_pos_start + num_decode_tokens),
    )

    torch.manual_seed(42)
    gen = torch.Generator(device="cpu").manual_seed(42)
    input_ids_prefill = torch.randint(
        0,
        fixtures.vocab_size,
        (batch_size, prefill_seq_len),
        dtype=torch.long,
        generator=gen,
    )
    decode_ids = torch.randint(
        0,
        fixtures.vocab_size,
        (batch_size, num_decode_tokens),
        dtype=torch.long,
        generator=gen,
    )

    tt_prefill_ids = _input_ids_to_tt(input_ids_prefill, mesh_device)
    ttnn.synchronize_device(mesh_device)
    t_prefill0 = time.perf_counter()
    prefill_out = fixtures.tt_model(tt_prefill_ids, mode="prefill", start_pos=0)
    ttnn.synchronize_device(mesh_device)
    prefill_s = time.perf_counter() - t_prefill0
    prefill_out.deallocate(True)
    logger.info(f"Prefill (KV fill): {prefill_s * 1000:.1f} ms for {prefill_seq_len} tokens")

    t_decode0 = time.perf_counter()
    for step in range(num_decode_tokens):
        pos = decode_pos_start + step
        current_pos_tt = _current_pos_to_tt(torch.tensor([pos], dtype=torch.long), mesh_device)
        tt_out = fixtures.tt_model(
            _input_ids_to_tt(decode_ids[:, step : step + 1], mesh_device),
            mode="decode",
            current_pos=current_pos_tt,
        )
        tt_out.deallocate(True)
    ttnn.synchronize_device(mesh_device)
    decode_s = time.perf_counter() - t_decode0

    logger.info(
        f"Decode perf ({fixtures.num_layers} layers, mesh={tuple(mesh_device.shape)}): "
        f"{num_decode_tokens} steps in {decode_s * 1000:.1f} ms "
        f"({num_decode_tokens / decode_s:.2f} tok/s)"
    )
