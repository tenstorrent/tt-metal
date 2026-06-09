# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy argmax parity: on-device global combine vs host ``torch.argmax`` on the full logits row."""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import _make_tt_model
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    mesh_default_device,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import _read_int_scalar


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        raise pytest.skip.Exception(str(e))
    except Exception as e:
        raise pytest.skip.Exception(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def _random_sharded_decode_logits(mesh_device: ttnn.Device, tt_model, *, seed: int) -> ttnn.Tensor:
    torch.manual_seed(seed)
    h = ttnn.from_torch(
        torch.randn(1, 1, tt_model.hidden_size, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logits_tt = tt_model._lm_head_sharded(h)
    ttnn.deallocate(h)
    return logits_tt


def _device_global_argmax(tt_model, logits_tt: ttnn.Tensor) -> int:
    token_tt = tt_model._ondevice_global_argmax_token(logits_tt)
    token_id = _read_int_scalar(token_tt)
    ttnn.deallocate(token_tt)
    return token_id


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_ondevice_global_argmax_matches_host_torch(mesh_device, device_params, reset_seeds):
    weights_dir = _weights_dir_or_skip()
    hf_model, _, _ = load_hf_model_and_processor(weights_dir)
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)

        for seed in range(8):
            logits_tt = _random_sharded_decode_logits(mesh_device, tt_model, seed=seed)
            host_row = tt_model._logits_row_to_host(logits_tt, 1, sharded=True)
            host_id = int(host_row[0].argmax().item())
            device_id = _device_global_argmax(tt_model, logits_tt)
            ttnn.deallocate(logits_tt)
            logger.info(f"argmax seed {seed}: host={host_id} device={device_id}")
            assert device_id == host_id, f"seed {seed}: device {device_id} != host {host_id}"
