# Temporary: isolate the DSpark + Rust host runtime prompt-replay hang. Not for commit.
import contextlib
import faulthandler
import os
import sys
import threading
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _DEFAULT_TEXT,
    _build_rope,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import stage_engram_tables
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

MODE = os.environ.get("DBG_MODE", "drain")  # drain | plain | python | nodspark


@torch.no_grad()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "num_command_queues": 2, "l1_small_size": 16384}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["tp4_8stage"])
def test_dbg(mesh_device):
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    stage_engram_tables(loader, config.engram_layer_ids)
    prompt = _tokenize_chat(tokenizer, _DEFAULT_TEXT)[:32]
    max_seq = _traced_max_seq(config, 512, config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)
    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=7, verify_rows=1
        )
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head)
        if MODE != "nodspark":
            model.prepare_spec_decode(rope["main"])
            if model.use_prefetcher:
                stack.enter_context(tensor_prefetcher_session(model.dspark.device))
        model.activate_session(model.open_session())
        if MODE != "nodspark":
            model.capture_spec_traces(prompt[0], 0)
        else:
            model.decode_traced(prompt[0], 0)
        if MODE != "python":
            model.start_host_runtime()
        if MODE == "drain":
            original = model._execute_traces

            def patched(step):
                pos, sid, held, image = step
                if model._host is None or model.dspark is None or held:
                    return original(step)
                original((pos, sid, True, image))
                model._host.drain()
                model.spec_release_writes(1)

            model._execute_traces = patched
        done = threading.Event()

        def watchdog():
            if not done.wait(240):
                logger.error(f"[dbg] HANG in mode {MODE}; stats {model.host_runtime_stats()}")
                faulthandler.dump_traceback(file=sys.stderr)
                os._exit(3)

        threading.Thread(target=watchdog, daemon=True).start()
        t0 = time.perf_counter()
        out = model.decode_prompt_traced(prompt, 0)
        logger.info(
            f"[dbg] mode {MODE}: prompt of {len(prompt)} in {time.perf_counter() - t0:.2f}s, "
            f"next {int(out.reshape(-1).float().argmax())}, stats {model.host_runtime_stats()}"
        )
        tok = int(out.reshape(-1).float().argmax())
        for i in range(16):
            out = model.decode_traced(tok, len(prompt) + i)
            tok = int(out.reshape(-1).float().argmax())
        done.set()
        logger.info(f"[dbg] mode {MODE}: 16 steps ok, stats {model.host_runtime_stats()}")
