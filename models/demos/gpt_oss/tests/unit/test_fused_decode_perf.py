# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 device decode step of the fused decode path with the real gpt-oss-20b weights.

The model is built like demo/text_demo.py (Generator, paged KV cache, decode trace). A 128-token prompt is prefilled,
the decode trace is captured and warmed up, then windows of back-to-back trace replays are timed with one device
synchronization per window (no token readback); the decode step is the mean window time per token. It must not exceed
the target of its mesh (the fused path measured on a quiet Blackhole host, plus about 3 %):
  1x4 (QuietBox 2): 2.5 ms with BFP4 routed-expert decode weights, 3.2 ms with BFP8 (GPT_OSS_FUSED_EXPERT_DTYPE)
  1x1 (one P150):   9.6 ms (routed experts streamed from the model's BFP8 ND-sharded expert weights)

Needs the real weights (HF_MODEL = the checkpoint; TT_CACHE_PATH for the converted weights). Run on a quiet host:
    HF_MODEL=openai/gpt-oss-20b pytest models/demos/gpt_oss/tests/unit/test_fused_decode_perf.py -k 1x1
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling import SamplingParams

from ..test_factory import TestFactory, parametrize_mesh_with_fabric

ISL = 128
WARMUP = 32
WINDOW = 16
WINDOWS = 16
MAX_SEQ_LEN = 4096
TARGET_MS = {((1, 4), "bfp4"): 2.5, ((1, 4), "bfp8"): 3.2, ((1, 1), "bfp8"): 9.6}


def _real_weights():
    """HF_MODEL names a checkpoint (a hub id or a directory with weights), not a config-only directory."""
    from pathlib import Path

    model = os.environ.get("HF_MODEL", "")
    return bool(model) and (not Path(model).is_dir() or any(Path(model).glob("*.safetensors")))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Needs the real weights and a quiet Blackhole host")
@pytest.mark.skipif(not _real_weights(), reason="HF_MODEL must name the real gpt-oss-20b checkpoint")
@parametrize_mesh_with_fabric([(1, 4), (1, 1)])
def test_fused_decode_step(mesh_device, device_params, reset_seeds):
    from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
    from models.demos.gpt_oss.tt.fused_decode.config import expert_decode_weight_dtype_name
    from models.tt_transformers.tt.generator import Generator

    if not ttnn.device.is_blackhole(mesh_device):
        pytest.skip("The fused decode path runs on Blackhole only")
    mesh_shape = tuple(mesh_device.shape)
    dtype = "bfp8" if mesh_shape == (1, 1) else expert_decode_weight_dtype_name()
    target = TARGET_MS[(mesh_shape, dtype)]

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    model_args, model, page_table, kv_cache, tokenizer, _, _ = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params={"page_block_size": 64, "page_max_num_blocks_per_dp": MAX_SEQ_LEN // 64},
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=None,
    )
    assert model[0].fused_decode, f"the fused decode path is not engaged on mesh {mesh_shape}"
    generator = Generator(model, model_args, mesh_device, tokenizer=tokenizer)
    greedy = SamplingParams(
        temperature=[0.0] * 32, top_k=[1] * 32, top_p=[1.0] * 32, enable_log_probs=[False] * 32, num_logprobs=[0] * 32
    )

    def step(tok, pos, first=False):
        """One traced decode step, enqueued only (the next input token stays on device)."""
        generator.decode_forward(
            torch.tensor([tok]),
            torch.tensor([pos]),
            enable_trace=True,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=greedy,
            reload_inputs=first,
            reload_page_table=False,
            reload_sampling_params=True,
            reset_sampling_state=first,
            read_from_device=False,
        )

    ids = torch.randint(1000, 199000, (1, ISL), generator=torch.Generator().manual_seed(ISL))
    logits = generator.prefill_forward_text(
        ids, page_table=page_table, kv_cache=kv_cache, prompt_lens=[ISL], enable_trace=False, warmup_prefill=False
    )
    tok, pos = int(torch.argmax(logits, dim=-1).view(-1)[0]), ISL
    step(tok, pos, first=True)  # captures the decode trace
    pos += 1
    for _ in range(WARMUP):
        step(tok, pos)
        pos += 1
    window_ms = []
    for _ in range(WINDOWS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(WINDOW):
            step(tok, pos)
            pos += 1
        ttnn.synchronize_device(mesh_device)
        window_ms.append((time.perf_counter() - t0) * 1e3 / WINDOW)
    mean_ms = statistics.mean(window_ms)
    result = {"mesh": list(mesh_shape), "expert_dtype": dtype, "isl": ISL, "decode_step_ms": mean_ms}
    result.update({"window_ms": window_ms, "target_ms": target, "load_avg": os.getloadavg()[0]})
    logger.info(f"fused decode step {json.dumps(result)}")
    if os.environ.get("GPT_OSS_PERF_OUT"):
        with open(os.environ["GPT_OSS_PERF_OUT"], "w") as f:
            json.dump(result, f, indent=1)
    assert mean_ms <= target, f"decode step {mean_ms:.3f} ms > target {target} ms (mesh {mesh_shape}, {dtype} experts)"
