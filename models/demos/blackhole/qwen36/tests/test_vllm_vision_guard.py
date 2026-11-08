# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DEVICE test of the serving vision-bucket guard + max_pixels-driven warmup (Qwen36ForCausalLM, TP=8).

(a) default warmup (no max_pixels): an over-size grid (1x86x128 = 11008 patches) raises ValueError at once, no hang;
(b) simulated vLLM config max_pixels=1048576 is read by _read_vllm_max_pixels; warmup covers its bucket and a 1 MP
    image (4096 patches) runs; (c) max_pixels=3e6 appends a new bucket (12288) to the warmup and the 11008-patch
    grid then runs instead of being rejected.

    pytest models/demos/blackhole/qwen36/tests/test_vllm_vision_guard.py -svq
"""
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        "trace_region_size": 1024 * 1024 * 1024,
    }
]


def _kwargs(model, grid):
    vc = model.vision_args.hf_config.vision_config
    t, h, w = grid
    pd = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2
    return dict(
        pixel_values=[[torch.zeros(t * h * w, pd)]], image_grid_thw=[[torch.tensor([t, h, w], dtype=torch.int32)]]
    )


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_vision_guard(mesh_device, reset_seeds, ensure_gc, expect_error):
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(os.environ["HF_MODEL"])
    gen = Qwen36ForCausalLM.initialize_vllm_model(hf_config, mesh_device, max_batch_size=32, max_seq_len=16384)
    model = gen.model[0]

    # (a) no max_pixels -> default grids (max bucket 8192); over-size item is refused host-side.
    assert gen._vllm_mm_max_pixels is None
    gen.warmup_vision()
    assert gen._vision_warmed_max_bucket == 8192
    with expect_error(ValueError, "max_pixels"):
        gen._compute_vision_tokens(model, _kwargs(model, (1, 86, 128)))

    # (b) simulated vLLM config
    for cfg in (
        SimpleNamespace(
            model_config=SimpleNamespace(mm_processor_kwargs={"max_pixels": 1048576}, multimodal_config=None)
        ),
        SimpleNamespace(
            model_config=SimpleNamespace(
                mm_processor_kwargs=None,
                multimodal_config=SimpleNamespace(mm_processor_kwargs={"max_pixels": 1048576}),
            )
        ),
    ):
        with mock.patch("vllm.config.get_current_vllm_config", return_value=cfg):
            assert Qwen36ForCausalLM._read_vllm_max_pixels() == 1048576
    gen._vllm_mm_max_pixels = 1048576
    gen.warmup_vision()
    assert gen._vision_warmed_max_bucket == 8192
    ttnn.deallocate(gen._compute_vision_tokens(model, _kwargs(model, (1, 64, 64))))  # 1 MP image: 4096 patches

    # (c) larger max_pixels appends a bucket; the formerly-hanging 2048x1365 grid now runs.
    gen._vllm_mm_max_pixels = 3_000_000
    gen.warmup_vision()
    assert gen._vision_warmed_max_bucket == 12288
    ttnn.deallocate(gen._compute_vision_tokens(model, _kwargs(model, (1, 86, 128))))
    with expect_error(ValueError, "max_pixels"):
        gen._compute_vision_tokens(model, _kwargs(model, (1, 128, 128)))
