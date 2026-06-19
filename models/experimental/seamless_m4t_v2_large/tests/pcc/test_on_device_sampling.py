# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device sampling parity: greedy (temperature=0) matches chunked global argmax."""

from __future__ import annotations

import pytest
import ttnn
from loguru import logger

from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_greedy_argmax_parity import (
    _device_global_argmax,
    _random_sharded_decode_logits,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.test_seamless_m4t_v2_model import _make_tt_model
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    MESH_DEVICE_PARAMETRIZE_TEXT,
    mesh_default_device,
)


def _weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_on_device_sampling_greedy_matches_global_argmax(mesh_device, device_params, reset_seeds):
    weights_dir = _weights_dir_or_skip()
    hf_model, _, _ = load_hf_model_and_processor(weights_dir)
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        if not tt_model._supports_on_device_sampling:
            pytest.skip("On-device sampling not supported on this mesh")

        sampler = tt_model._on_device_sampler
        assert sampler is not None
        sampler.configure(do_sample=False, temperature=0.0, top_k=1, top_p=1.0, seed=42)

        for seed in range(4):
            logits_tt = _random_sharded_decode_logits(mesh_device, tt_model, seed=seed)
            argmax_id = _device_global_argmax(tt_model, logits_tt)
            sample_id = sampler.sample_and_read(logits_tt, dec_len=1)
            ttnn.deallocate(logits_tt)
            logger.info(f"on-device sampling seed {seed}: argmax={argmax_id} sample={sample_id}")
            assert sample_id == argmax_id, f"seed {seed}: sample {sample_id} != argmax {argmax_id}"


@pytest.mark.parametrize(*MESH_DEVICE_PARAMETRIZE_TEXT, indirect=["mesh_device", "device_params"])
def test_on_device_sampling_traced_decode_greedy(mesh_device, device_params, reset_seeds):
    """Post-trace TTSampling greedy matches chunked argmax on a traced KV decode step."""
    weights_dir = _weights_dir_or_skip()
    hf_model, _, _ = load_hf_model_and_processor(weights_dir)
    cfg = hf_model.config
    t2u_cfg = hf_model.t2u_model.config

    with mesh_default_device(mesh_device):
        tt_model = _make_tt_model(mesh_device, hf_model, cfg, t2u_cfg)
        if not tt_model._supports_on_device_sampling:
            pytest.skip("On-device sampling not supported on this mesh")

        sampler = tt_model._on_device_sampler
        assert sampler is not None
        sampler.configure(do_sample=False, temperature=0.0, top_k=1, top_p=1.0, seed=7)

        logits_tt = _random_sharded_decode_logits(mesh_device, tt_model, seed=99)
        argmax_id = _device_global_argmax(tt_model, logits_tt)
        sample_id = tt_model._read_token_from_traced_logits(logits_tt, dec_len=1)
        ttnn.deallocate(logits_tt)
        assert sample_id == argmax_id
