# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The MTP adapter in lanes mode (4 lanes, K=4, block size 128, 1026 blocks) through the plugin's warm-up calls, in-process.

Run:  pytest models/demos/blackhole/qwen38_flash_next/tests/test_vllm_mtp_lanes_device.py -s
Env:  MESH_DEVICE=P150x8, HF_MODEL, TT_CACHE_PATH, TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0, QWEN38_VLLM_MTP_K=4
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.checkpoint import checkpoint_directory
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import Qwen38MTPForCausalLM
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE

MAX_SEQ_LEN = 32768 - RESIDENT_CONTEXT_HEADROOM
LANES = 4
BLOCK_SIZE = 128
NUM_BLOCKS = 1026


def step(name: str) -> None:
    logger.info(f"F17 step: {name}")


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_mtp_lanes_warmup(mesh_device, monkeypatch):
    import ttnn

    try:
        checkpoint: Path = checkpoint_directory()
    except FileNotFoundError as error:
        pytest.fail(str(error))
    if not os.environ.get("QWEN38_VLLM_MTP_K"):
        monkeypatch.setenv("QWEN38_VLLM_MTP_K", "4")
    mesh_device.enable_program_cache()

    step("initialize_vllm_model")
    model = Qwen38MTPForCausalLM.initialize_vllm_model(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_batch_size=LANES, max_seq_len=MAX_SEQ_LEN
    )
    step("initialize_vllm_model done")
    try:
        step("allocate_kv_cache")
        model.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, 128), ttnn.bfloat16, 12)
        step("allocate_kv_cache done")

        width = -(-MAX_SEQ_LEN // BLOCK_SIZE)
        prefill_kwargs = dict(kv_cache=None, can_sample_on_device=False)
        decode_kwargs = dict(kv_cache=None, max_batch_size=LANES, num_blocks=width, can_sample_on_device=True)

        step("warmup_model_prefill eager")
        model.warmup_model_prefill(enable_trace=False, **prefill_kwargs)
        step("warmup_model_prefill eager done")
        step("warmup_model_decode eager")
        model.warmup_model_decode(enable_trace=False, **decode_kwargs)
        step("warmup_model_decode eager done")

        if hasattr(model, "already_warmed_up_prefill"):
            model.already_warmed_up_prefill = False

        step("warmup_model_prefill traced")
        model.warmup_model_prefill(enable_trace=True, **prefill_kwargs)
        step("warmup_model_prefill traced done")
        step("warmup_model_decode traced")
        model.warmup_model_decode(enable_trace=True, **decode_kwargs)
        step("warmup_model_decode traced done")
    finally:
        step("release_persistent_capture")
        model.release_persistent_capture()
        step("release_persistent_capture done")
