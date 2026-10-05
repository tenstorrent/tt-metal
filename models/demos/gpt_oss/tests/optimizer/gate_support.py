# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the gpt-oss-20b optimizer gates (test_optimizer_pcc.py, test_optimizer_perf.py).

Both gates build the model exactly as demo/text_demo.py does for its prefill_128 case on a 1x4 Blackhole mesh
(QuietBox 2): prepare_gpt_oss_generator_args with batch 1, data parallel 1, paged attention (64-token blocks,
4096-token context), bfloat8_b default dtype, then the tt_transformers Generator.

Weight cache: create_tt_model skips the HF weight load once the ttnn cache is marked complete. A change that
asks for a cache file the cache does not hold (a new dtype or layout for some weight) then fails in
ttnn.as_tensor(None, ...). build_generator catches that once, frees what was built, and rebuilds with the HF
weights loaded (GPT_OSS_FORCE_MODEL_LOAD=1), which writes the missing files. It prints GATE_WEIGHT_RELOAD when it
does.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.gpt_oss.tests.test_factory import L1_SMALL_SIZE
from models.demos.utils.trace_region_sizes import TRACE_MODEL_KEY_PARAM

MESH_SHAPE = (1, 4)
MAX_SEQ_LEN = 4 * 1024
PAGE_PARAMS = {"page_block_size": 64, "page_max_num_blocks_per_dp": 4 * 1024 // 64}
HF_MODEL_ID = "openai/gpt-oss-20b"
MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[5]
PROMPTS = MODEL_DIR / "demo" / "sample_prompts" / "input_data_questions_prefill_128.json"


def device_params() -> dict:
    """The demo's device_params for a 1x4 mesh; the optimizer may raise the trace region (TT_PERF_TRACE_REGION)."""
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "l1_small_size": L1_SMALL_SIZE}
    region = int(os.environ.get("TT_PERF_TRACE_REGION") or 0)
    if region > 0:
        params["trace_region_size"] = region
    else:
        params[TRACE_MODEL_KEY_PARAM] = "gpt-oss-120b"  # the key demo/text_demo.py and test_factory use
    return params


def _build(mesh_device, state_dict):
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
    from models.tt_transformers.tt.generator import Generator

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    torch.manual_seed(0)  # create_tt_page_table permutes blocks with torch.randperm
    model_args, model, page_table, tt_kv_cache, tokenizer, processor, _ = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=MAX_SEQ_LEN,
        page_params=PAGE_PARAMS,
        paged_attention=True,
        mesh_config=mesh_config,
        state_dict=state_dict,
        users_row_sharded=False,
    )
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    return generator, model_args, model, page_table, tt_kv_cache, tokenizer


def build_generator(mesh_device, state_dict=None):
    """(generator, model_args, model, page_table, kv_cache, tokenizer), retrying once with a forced HF load."""
    try:
        return _build(mesh_device, state_dict)
    except Exception as error:  # noqa: BLE001
        if os.environ.get("GPT_OSS_FORCE_MODEL_LOAD") == "1":
            raise
        print(
            f"GATE_WEIGHT_RELOAD build from the warm weight cache failed ({type(error).__name__}: "
            f"{str(error)[:300]}); rebuilding with the HF weights loaded",
            flush=True,
        )
    gc.collect()
    ttnn.synchronize_device(mesh_device)
    os.environ["GPT_OSS_FORCE_MODEL_LOAD"] = "1"
    try:
        return _build(mesh_device, None)
    finally:
        os.environ.pop("GPT_OSS_FORCE_MODEL_LOAD", None)


def greedy_sampling_params():
    """The demo's on-device greedy sampling parameters (temperature 0 -> top_k 1, top_p 1.0)."""
    from models.common.sampling import SamplingParams

    n = 32
    return SamplingParams(
        temperature=[0] * n, top_k=[1] * n, top_p=[1.0] * n, enable_log_probs=[False] * n, num_logprobs=[0] * n
    )


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(MODEL_DIR), capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def tree_diff_hash() -> str:
    """Hash of the uncommitted diff, so candidates measured on the same HEAD can be told apart."""
    try:
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return ""
    if not diff:
        return "clean"
    import hashlib

    return hashlib.sha1(diff).hexdigest()[:12]


def append_history(env_name: str, record: dict) -> None:
    """One JSON line per gate run in the file named by env_name (unset: no history)."""
    path = os.environ.get(env_name)
    if not path:
        return
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tree": git_head(), "diff": tree_diff_hash(), **record}
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as error:
        logger.warning(f"history {path} not written: {error}")
