# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Several whole model instances on one persistent MoE, each with its DSpark drafter on its own row.

``launch_instances_server`` launches the TP12 MoE (rows 5-7, layers 0-39 plus the drafter's stages) for
``DEEPSEEK_V41_INSTANCES`` instances (default 5); instance j is a ``DeepSeekV41Model(drafter_on_stage=True)`` on mesh
row 4 - j, the server's requester j: attention, Engram, norms, ``lm_head`` and drafter on the one TP4 row, every FFN
on the server, one-row decode steps and 4-row verify blocks (and 4-row draft FFNs) mixed on its one launch.

Every user asks the same prompt (``DEEPSEEK_V4_SPEC_PROMPT``, default GSM8K's three problems) and decodes, greedy,
``DEEPSEEK_V4_SPEC_TOKENS`` tokens (default 300, capped to the block verify's dense range). Instance 0 alone gives the
reference tokens of plain one-row decode and of DSpark (3 drafts + 1 in one 4-row step). Then for each user count
of ``DEEPSEEK_V41_USERS`` (default 1,2,3,4,5) instances 0 .. n-1 run at once, one thread each, first DSpark, then plain
decode; every user must get exactly the reference tokens (the block path and the server's reduction are
deterministic, so other instances' requests interleaving must change nothing). The test ends with one table: per user
count, DSpark tok/s/u (mean and slowest user), total tok/s, tokens per round, ms per round and its verify / draft /
fixup split, and plain decode's tok/s/u and total.

Logs stay at INFO with one line per user count; ``DEEPSEEK_V4_SPEC_VERBOSE=1`` adds every DSpark round,
``DEEPSEEK_V41_PRINT_OUTPUT=1`` prints the DSpark text, and ``TT_LOGGER_LEVEL=error`` quiets ttnn's C++ warnings.

With ``DEEPSEEK_V41_RUST_HOST=1`` (the default) every instance runs on its own Rust host runtime
(``start_host_runtime``), as the server does.

    TT_LOGGER_LEVEL=error DEEPSEEK_V41_INSTANCES=5 pytest -s models/experimental/deepseek_v41_flash/tests/decode/test_instances_dspark.py
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_dspark_spec import _PROMPTS
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _available_bytes,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tt import host_rs
from models.experimental.deepseek_v41_flash.tt import spec as spec_module
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.decode import persistent_moe as pmoe
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    ENGRAM_RAM_MARGIN,
    engram_ram_needed,
    stage_engram_tables,
)
from models.experimental.deepseek_v41_flash.tt.spec import DSparkSpeculator
from models.experimental.deepseek_v4_flash.encoding_dsv4 import eos_token
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

INSTANCES = int(os.environ.get("DEEPSEEK_V41_INSTANCES", "5"))
# DEEPSEEK_V4_SPEC_VERBOSE=1: every DSpark round's log line; DEEPSEEK_V41_PRINT_OUTPUT=1: the DSpark text;
# LOGURU_LEVEL (default INFO here) for the rest.
_VERBOSE = os.environ.get("DEEPSEEK_V4_SPEC_VERBOSE", "0") not in ("0", "", "false", "False")
_PRINT_OUTPUT = os.environ.get("DEEPSEEK_V41_PRINT_OUTPUT", "0") not in ("0", "", "false", "False")
# The users decoding at once in each run of the sweep (each at most INSTANCES).
USERS = [int(n) for n in os.environ.get("DEEPSEEK_V41_USERS", "1,2,3,4,5").split(",") if n.strip()]
# Every user's prompt: a name in test_dspark_spec's _PROMPTS (default GSM8K's three problems) or the text itself.
PROMPT = os.environ.get("DEEPSEEK_V4_SPEC_PROMPT", "gsm8k")
# Bisect switches for concurrent runs: DEEPSEEK_V41_MIXED_PROMPTS=1 gives odd instances the demo prompt instead (the
# users then decode different text); DEEPSEEK_V41_STAGGER_MS starts user j j * that many ms after user 0.
MIXED_PROMPTS = os.environ.get("DEEPSEEK_V41_MIXED_PROMPTS", "0") not in ("0", "", "false", "False")
STAGGER_MS = float(os.environ.get("DEEPSEEK_V41_STAGGER_MS", "0"))
# DEEPSEEK_V41_PLAIN_FIRST=1: each user count runs plain decode at once before DSpark (default DSpark first).
PLAIN_FIRST = os.environ.get("DEEPSEEK_V41_PLAIN_FIRST", "0") not in ("0", "", "false", "False")
ROWS = 4


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "16384")),
        }
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["tp4_instances_persistent_moe12"])
def test_instances_dspark(mesh_device, monkeypatch):
    # spec.py logs every round by default and loguru's default sink is DEBUG: this test keeps to its summary lines.
    monkeypatch.setattr(spec_module, "_VERBOSE", _VERBOSE)
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LOGURU_LEVEL", "INFO"))
    if not 1 <= INSTANCES <= len(pmoe.INSTANCE_ROWS):
        pytest.skip(f"DEEPSEEK_V41_INSTANCES={INSTANCES}: 1 to {len(pmoe.INSTANCE_ROWS)}")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    stage_engram_tables(loader, config.engram_layer_ids)
    nbytes = engram_ram_needed(loader, config.engram_layer_ids)
    if _available_bytes() < ENGRAM_RAM_MARGIN * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    texts = [_PROMPTS.get(PROMPT, PROMPT), _PROMPTS["demo"]] if MIXED_PROMPTS else [_PROMPTS.get(PROMPT, PROMPT)]
    prompts = [_tokenize_chat(tokenizer, texts[j % len(texts)]) for j in range(INSTANCES)]
    dense_limit = config.index_topk - 2 * ROWS
    n_tokens = min(int(os.environ.get("DEEPSEEK_V4_SPEC_TOKENS", "300")), dense_limit - max(map(len, prompts)))
    max_seq = _traced_max_seq(config, max(map(len, prompts)) + n_tokens + 2 * ROWS, config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)
    num_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )

    with contextlib.ExitStack() as stack:
        t0 = time.perf_counter()
        server = pmoe.launch_instances_server(
            mesh_device, config, loader, WeightCache(CACHE_DIR) if CACHE_DIR else None, num_layers, INSTANCES
        )
        stack.callback(server.stop)  # last: after every model's shutdown
        logger.info(f"[test] server for {INSTANCES} instances up in {time.perf_counter() - t0:.0f}s")
        models = []
        for j in range(INSTANCES):
            t0 = time.perf_counter()
            model, lm_head, _, _ = _construct_model(
                mesh_device,
                stack,
                loader=loader,
                config=config,
                num_stages=1,
                persistent_moe=True,
                verify_rows=ROWS,
                moe_server=server,
                moe_requester=j,
                drafter_on_stage=True,
            )
            model.prepare_static_decode(rope, max_seq, lm_head=lm_head, device_sampling=True)
            model.device_sampler.set_sampling(0.0, 0)
            model.prepare_spec_decode(rope["main"])
            model.activate_session(model.open_session())
            # One instance at a time: a capture picks the server's request mode, which the instances share.
            model.capture_spec_traces(prompts[j][0], 0)
            if host_rs.enabled():
                model.start_host_runtime()
            models.append(model)
            logger.info(
                f"[test] instance {j} on mesh row {pmoe.INSTANCE_ROWS[j]}: built and captured in "
                f"{time.perf_counter() - t0:.0f}s (host runtime {'on' if model.host_runtime_running else 'off'})"
            )

        def token_of(out) -> int:
            return int(out.reshape(-1)[0])

        def plain(j: int) -> dict:
            """Plain one-row greedy decode of instance j's prompt from position 0."""
            model, prompt = models[j], prompts[j]
            got = [token_of(model.decode_prompt_traced(prompt, 0))]
            t0 = time.perf_counter()
            while len(got) < n_tokens and got[-1] != eos_id:
                got.append(token_of(model.decode_traced(got[-1], len(prompt) + len(got) - 1)))
            return {"tokens": got, "tps": (len(got) - 1) / (time.perf_counter() - t0)}

        def dspark(j: int) -> dict:
            """DSpark (3 drafts + 1 in one 4-row step) on instance j's prompt from position 0."""
            spec = DSparkSpeculator(models[j], ROWS - 1, parallel_verify=True, temperature=0.0)
            got = [spec.prefill(prompts[j])]
            t0 = time.perf_counter()
            if got[-1] != eos_id:
                got += spec.generate(n_tokens - 1, stop_ids=(eos_id,))
            seconds = time.perf_counter() - t0
            spec.finish()
            stats, rounds = spec.stats, max(len(spec.stats.accepted), 1)
            return {
                "tokens": got[:n_tokens],
                "tps": (len(got) - 1) / seconds,
                "per_round": stats.tokens / rounds,
                "ms_round": 1000 * stats.seconds / rounds,
                "phases": {name: 1000 * t / rounds for name, t in stats.phases.items()},
            }

        def concurrently(kind, n: int) -> list:
            """``kind`` (plain or dspark) on instances 0 .. n-1 at once, one thread each; their results."""
            results: list = [None] * n
            errors: list = []
            start = threading.Barrier(n)

            def run(j: int) -> None:
                try:
                    start.wait()
                    if STAGGER_MS:
                        time.sleep(j * STAGGER_MS / 1000)
                    results[j] = kind(j)
                except BaseException as e:  # noqa: BLE001 - re-raised below
                    errors.append(e)

            threads = [threading.Thread(target=run, args=(j,), name=f"instance-{j}") for j in range(n)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
            return results

        def first_diff(a: list[int], b: list[int]) -> int:
            return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))

        # References: each instance alone, one row at a time (each its own prompt's tokens). A fault confined to one row
        # shows here, before any concurrent run (where it would stall every instance on the shared MoE alike).
        ref_plain, ref_spec = [], []
        for j in range(INSTANCES):
            row = pmoe.INSTANCE_ROWS[j]
            got_plain, got_spec = plain(j), dspark(j)
            for name, got, refs in (("plain", got_plain, ref_plain), ("DSpark", got_spec, ref_spec)):
                same = next((r for k, r in enumerate(refs) if prompts[k] == prompts[j]), None)
                if same is not None:
                    diff = first_diff(got["tokens"], same)
                    assert got["tokens"] == same, f"instance {j} (row {row}) alone, {name}: differs at {diff}"
                refs.append(got["tokens"])
            logger.info(
                f"[test] instance {j} (row {row}) alone: plain {got_plain['tps']:.1f}, DSpark {got_spec['tps']:.1f} "
                f"tok/s/u; DSpark first differs from plain decode at {first_diff(got_spec['tokens'], got_plain['tokens'])}"
            )
        if _PRINT_OUTPUT:
            print(f"\n===== DSpark output =====\n{tokenizer.decode(ref_spec[0])}\n", flush=True)

        rows = []
        for n in USERS:
            if n > INSTANCES:
                logger.warning(f"[test] {n} users need {n} instances; built {INSTANCES}")
                continue
            if PLAIN_FIRST:
                plain_runs = concurrently(plain, n)
                logger.info(f"[test] {n} user(s): plain decode at once ok")
                spec_runs = concurrently(dspark, n)
            else:
                spec_runs = concurrently(dspark, n)
                plain_runs = concurrently(plain, n)
            for j in range(n):
                got, ref = spec_runs[j]["tokens"], ref_spec[j]
                assert (
                    got == ref
                ), f"{n} users: instance {j} DSpark differs from the 1-user run at {first_diff(got, ref)}"
                got, ref = plain_runs[j]["tokens"], ref_plain[j]
                assert got == ref, f"{n} users: instance {j} plain decode differs from alone at {first_diff(got, ref)}"
            spec_tps = [r["tps"] for r in spec_runs]
            plain_tps = [r["tps"] for r in plain_runs]
            phases = {name: sum(r["phases"].get(name, 0.0) for r in spec_runs) / n for name in spec_runs[0]["phases"]}
            rows.append(
                (
                    n,
                    sum(spec_tps) / n,
                    min(spec_tps),
                    sum(spec_tps),
                    sum(r["per_round"] for r in spec_runs) / n,
                    sum(r["ms_round"] for r in spec_runs) / n,
                    phases,
                    sum(plain_tps) / n,
                    sum(plain_tps),
                )
            )
            logger.info(
                f"[test] {n} user(s) done: DSpark {sum(spec_tps) / n:.1f}, plain {sum(plain_tps) / n:.1f} tok/s/u"
            )

        prompt_name = PROMPT if PROMPT in _PROMPTS else "custom"
        lines = [
            f"DSpark vs plain decode, {prompt_name} prompt, {n_tokens} tokens per user, one user per instance (greedy)",
            f"{'users':>5} | {'DSpark tok/s/u':>14} {'(min)':>7} | {'total':>7} | {'tok/round':>9} | "
            f"{'ms/round':>8} {'verify':>6} {'draft':>6} {'fixup':>6} | {'plain tok/s/u':>13} | {'total':>7}",
        ]
        for n, mean, low, total, per_round, ms, phases, p_mean, p_total in rows:
            lines.append(
                f"{n:>5} | {mean:>14.1f} {low:>7.1f} | {total:>7.1f} | {per_round:>9.2f} | {ms:>8.1f} "
                f"{phases.get('verify', 0):>6.1f} {phases.get('draft', 0):>6.1f} {phases.get('fixup', 0):>6.1f} | "
                f"{p_mean:>13.1f} | {p_total:>7.1f}"
            )
        print("\n" + "\n".join(lines) + "\n", flush=True)
