# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-step op profile for the roofline split (matmul / CCL / other / gaps).

Builds the generator, prefills a short prompt, warms the traced decode path,
then wraps ``N_STEPS`` traced decode steps in Tracy ``start`` / ``stop``
signposts so ``ops_perf_results_*.csv`` carries exactly those steps. The
companion ``decode_roofline_report.py`` splits the device time per step into
matmuls, collectives, everything else, and op-to-op gaps, and reports the
step against the DRAM-bandwidth floor.

    export TT_METAL_DEVICE_PROFILER=1
    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=0,1,2,3 \\
    python -m tracy -p -r -v -m pytest \\
        models/demos/gemma4/tests/unit/test_decode_roofline.py -k "batch_1" -s
    python tools/tracy/process_ops_logs.py --date
    python models/demos/gemma4/tests/unit/decode_roofline_report.py <ops_perf_results.csv> --steps 10
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo import (
    _batch_page_params,
    _create_tt_page_table,
    _install_hybrid_page_tables,
    _resolve_bounded_sliding,
)
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

PREFILL_LEN = 128
WARM_STEPS = 4
N_STEPS = int(os.environ.get("GEMMA4_ROOFLINE_STEPS", "10"))


def _signpost(name):
    try:
        from tracy import signpost
    except ImportError:  # not under python -m tracy: still a valid timing run
        return
    signpost(name)


@pytest.fixture
def model_path():
    return os.environ.get("HF_MODEL", "google/gemma-4-31B-it")


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 512_000_000})
@pytest.mark.parametrize("batch_size", [1, 32], ids=["batch_1", "batch_32"])
def test_decode_roofline(mesh_device, model_path, batch_size):
    skip_if_config_only_checkpoint()
    max_seq_len = 4096
    page_params = _batch_page_params(batch_size, PREFILL_LEN, WARM_STEPS + N_STEPS + 2)
    paged_attention_config = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]),
        max_num_blocks=int(page_params["page_max_num_blocks"]),
    )
    page_table = _create_tt_page_table(batch_size, paged_attention_config)
    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    if bounded_sliding:
        _install_hybrid_page_tables(
            generator.model[0],
            generator.model_args[0],
            batch_size=batch_size,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
        )

    prompt = tokenizer.encode("It was the best of times, it was the worst of times, " * 12, add_special_tokens=True)
    tokens = torch.tensor(prompt[:PREFILL_LEN]).unsqueeze(0).repeat(batch_size, 1)
    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[PREFILL_LEN] * batch_size,
        warmup_prefill=False,
        enable_trace=False,
    )
    tok = torch.argmax(out.reshape(batch_size, -1).float(), dim=-1, keepdim=True)
    current_pos = torch.full((batch_size,), PREFILL_LEN, dtype=torch.int32)

    def step(t):
        o = generator.decode_forward(
            t,
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            enable_trace=True,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        logits = o[0] if isinstance(o, tuple) else o
        return torch.argmax(logits.reshape(batch_size, -1).float(), dim=-1, keepdim=True)

    for _ in range(WARM_STEPS):  # trace capture + warm replays
        tok = step(tok)
        current_pos += 1
    ttnn.synchronize_device(mesh_device)

    _signpost("start")
    for _ in range(N_STEPS):
        tok = step(tok)
        current_pos += 1
    ttnn.synchronize_device(mesh_device)
    _signpost("stop")
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:  # profiler off: nothing to flush
        pass
    logger.info(f"{os.path.basename(model_path)} batch={batch_size}: {N_STEPS} traced decode steps between signposts")
