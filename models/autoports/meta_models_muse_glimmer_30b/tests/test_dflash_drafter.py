# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC parity for the TTNN DFlash drafter against the real HF drafter.

Goldens come from ``reference_dflash.py`` (real weights, real ``transformers``
classes).  Regenerate them with::

    python models/autoports/meta_models_muse_glimmer_30b/tests/reference_dflash.py

``ctx4096`` deliberately exceeds the 2048 sliding window, so it is the only case
that exercises the mask's lower bound; the shorter cases all fit inside the
window and would pass even with the mask disabled.
"""

from __future__ import annotations

import gc

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference_dflash as R
from models.autoports.meta_models_muse_glimmer_30b.tt.dflash_drafter import (
    DFlashDrafter,
    bidirectional_sliding_mask,
    config_from_hf,
)
from models.common.utility_functions import comp_pcc

PCC_THRESHOLD = 0.99
CONTEXT_LENS = (1, 16, 128, 2048, 4096)


@pytest.fixture(scope="session")
def mesh_device():
    if ttnn.get_num_devices() < 1:  # pragma: no cover - no hardware
        pytest.skip("no Tenstorrent device available")
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)
        gc.collect()


@pytest.fixture(scope="session")
def goldens():
    path = R.golden_path()
    if not path.exists():  # pragma: no cover - goldens not generated
        pytest.skip(f"missing {path}; run reference_dflash.py first")
    return torch.load(path, weights_only=False)


@pytest.fixture(scope="session")
def drafter(mesh_device):
    return DFlashDrafter.from_state_dict(
        R.draft_state_dict(),
        hf_config=R.draft_config(),
        mesh_device=mesh_device,
        weight_dtype=ttnn.bfloat16,
        activation_dtype=ttnn.bfloat16,
    )


def _to_device_hidden(mesh_device, tensor: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor.reshape(1, 1, *tensor.shape[-2:]).to(torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def test_goldens_match_current_checkpoint(goldens):
    """A stale golden set must fail loudly rather than silently grade the wrong model."""
    assert goldens["_fingerprint"] == R.config_fingerprint()


def test_mask_is_bidirectional_not_causal():
    """The window must see its own future; a causal mask silently costs acceptance rate."""
    config = config_from_hf(R.draft_config())
    positions = torch.arange(10)
    mask = bidirectional_sliding_mask(positions[-4:], positions, config.sliding_window, torch.float32)[0, 0]
    blocked = torch.finfo(torch.float32).min
    # Query 0 of the window (absolute position 6) must attend to position 9, which is *ahead* of it.
    assert mask[0, 9] == 0.0, "window queries must attend bidirectionally within the block"
    assert (mask == blocked).sum() == 0, "nothing should be masked when everything fits the window"


def test_mask_lower_bound_applies_past_the_window():
    config = config_from_hf(R.draft_config())
    window = config.sliding_window
    total = window + 64
    positions = torch.arange(total)
    mask = bidirectional_sliding_mask(positions[-4:], positions, window, torch.float32)[0, 0]
    blocked = torch.finfo(torch.float32).min
    q_pos = int(positions[-4])
    # kv_idx > q_idx - window is the exact HF condition.
    assert mask[0, q_pos - window] == blocked
    assert mask[0, q_pos - window + 1] == 0.0


@pytest.mark.parametrize("context_len", CONTEXT_LENS)
def test_encoder_projection_pcc(mesh_device, drafter, goldens, context_len):
    """``fc`` + ``output_norm_enc`` in isolation, so a projection bug is not blamed on attention."""
    case = goldens[f"ctx{context_len}"]
    inputs = R.synthetic_inputs(context_len=context_len)
    context = _to_device_hidden(mesh_device, inputs["context_hidden_states"])
    out = drafter.project_context(context)
    actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()
    expected = case["outputs"]["encoder_out"].float()
    passed, message = comp_pcc(expected, actual.reshape(expected.shape), PCC_THRESHOLD)
    logger.info(f"encoder ctx={context_len}: {message}")
    assert passed, message


@pytest.mark.parametrize("context_len", CONTEXT_LENS)
def test_drafter_end_to_end_pcc(mesh_device, drafter, goldens, context_len):
    case = goldens[f"ctx{context_len}"]
    config = drafter.config
    inputs = R.synthetic_inputs(context_len=context_len)
    noise = _to_device_hidden(mesh_device, inputs["noise_embeds"])
    context = _to_device_hidden(mesh_device, inputs["context_hidden_states"])
    position_ids = torch.arange(context_len + config.block_size)

    out = drafter(noise, context, position_ids=position_ids)
    actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()
    expected = case["outputs"]["last_hidden_state"].float()

    passed, message = comp_pcc(expected, actual.reshape(expected.shape), PCC_THRESHOLD)
    logger.info(f"drafter ctx={context_len}: {message}")
    assert passed, message
