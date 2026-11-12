# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Eight V4.1 decode sessions generate interleaved, fully pipelined, and each gets the logits it gets alone.

Eight prompts are decoded greedily on eight sessions, round-robin one step per session, by three host
stages that never wait on one another except through the data they hand over:

* the posting thread (this one) seats each step's session and queues its trace replay, which the
  model's replay thread issues to the command queues;
* a packet thread pushes each step's packet into the H2D socket as soon as its token is known (a
  prompt token, or the session's previous greedy pick);
* an output thread drains the D2H socket in step order and hands every pick back to the packet thread.

A session's next token depends only on its own previous step, so the eight sessions keep eight steps
in flight, one per pipeline stage. Tokens/s per user and in total are logged with every user's text.
Two sessions are then rewound and decoded alone with the same tokens: every position's logits must
match the interleaved run's, since the sessions share the block pools and the traces but none reads
another's rows (with ``DEEPSEEK_V4_DEVICE_SAMPLING=1``, whose steps send back only the greedy pick, the picks
must match instead).

The build and the step are ``test_full_model_decode_demo.py``'s: the same model, prefetcher and layout (pinned by
that demo's asserts), Engram tables staged in tmpfs, every session's blocks grown for the whole run before the
first replay (a later growth rewrites the page tables, which drains the pipeline), and on-device sampling with
``DEEPSEEK_V4_DEVICE_SAMPLING=1``. Like ``demo/server.py`` it then runs on the Rust host runtime (trace replays,
packets, Engram rows and output reads on native threads; ``DEEPSEEK_V41_RUST_HOST=0`` keeps the Python ones) with
the server's GIL switch interval, so its total tok/s is what the device gives the server's eight decoding users
without the server's scheduler or HTTP layer.

Run (32-chip Galaxy, ttnn venv, from the repo root)::

    DEEPSEEK_V4_DEVICE_SAMPLING=1 DEEPSEEK_V4_MAX_NEW_TOKENS=256 \\
    pytest models/experimental/deepseek_v41_flash/tests/decode/test_multi_session_decode.py

``DEEPSEEK_V4_MAX_NEW_TOKENS`` sets the tokens generated per user (default 64).
"""

from __future__ import annotations

import contextlib
import os
import queue
import sys
import threading
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _DEFAULT_TEXT,
    _DEVICE_SAMPLING,
    _L1_SMALL_SIZE,
    _PAGE_BLOCK_SIZE,
    _SEED,
    _assert_decode_parallelism,
    _assert_prefetch_rings,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _stage_engram_or_skip,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tt import host_rs
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_TEXTS = (
    _DEFAULT_TEXT,
    "Write a short poem about the sea, then explain which images in it are the strongest and why.",
    "Explain how a hash table handles collisions, and compare chaining with open addressing.",
    "What causes the seasons on Earth? Answer in three short paragraphs.",
    "Give me a recipe for a simple vegetarian curry, with a shopping list.",
    "Summarize the plot of Romeo and Juliet for a ten-year-old.",
    "List five practical tips for writing readable Python code, each with a one-line example.",
    "Why is the sky blue? Explain it first simply, then with the physics.",
)
_NUM_SESSIONS = len(_TEXTS)
_MAX_NEW_TOKENS = int(os.environ.get("DEEPSEEK_V4_MAX_NEW_TOKENS", "64"))
# Sessions whose logits are kept and re-decoded alone.
_CHECKED = (0, _NUM_SESSIONS - 1)
_MAX_IN_FLIGHT = 2 * _NUM_SESSIONS
_PCC = 0.9999


def _pick(out: torch.Tensor) -> int:
    """The greedy pick of one step's output: the device's token (int32, token first), or the logits' argmax."""
    out = out.reshape(-1)
    return int(out[0]) if out.dtype == torch.int32 else int(out.argmax())


def _decode_alone(model, sid: int, ids: list[int]) -> list[torch.Tensor]:
    """``ids`` at positions ``0 ..`` on session ``sid`` by itself, one blocking step at a time."""
    model.activate_session(sid)
    return [model.decode_traced(token, pos).reshape(-1) for pos, token in enumerate(ids)]


def _poll(get, errors: list):
    """``get(timeout=...)`` until it yields, raising if another stage failed while this one waited."""
    while True:
        if errors:
            raise RuntimeError("another decode stage failed") from errors[0]
        try:
            return get(timeout=1.0)
        except queue.Empty:
            continue


def _generate_interleaved(model, prompts: dict[int, list[int]], max_new: int, keep: set[int]) -> dict:
    """Greedy-generate ``max_new`` tokens for every session in ``prompts`` (sid -> prompt ids), the
    sessions round-robin one step at a time, with posting, packet writes and output reads on three threads.

    A session takes ``len(prompt) + max_new - 1`` steps: its prompt, then each pick but the last. Returns
    per session the generated ids, each step's output arrival time, and (for ``keep``) each step's logits.
    """
    schedule = [
        (sid, pos)
        for pos in range(max(len(ids) for ids in prompts.values()) + max_new - 1)
        for sid, ids in prompts.items()
        if pos < len(ids) + max_new - 1
    ]
    posted: queue.Queue = queue.Queue()
    written: queue.Queue = queue.Queue()
    picks = {sid: queue.Queue() for sid in prompts}
    in_flight = threading.Semaphore(_MAX_IN_FLIGHT)
    errors: list[BaseException] = []
    generated = {sid: [] for sid in prompts}
    arrivals = {sid: [] for sid in prompts}
    logits = {sid: [] for sid in keep}

    def write_packets() -> None:
        try:
            while (step := _poll(posted.get, errors)) is not None:
                sid, pos = step
                ids = prompts[sid]
                token = ids[pos] if pos < len(ids) else _poll(picks[sid].get, errors)
                model.write_step_packet(token, pos)
                written.put(step)
        except BaseException as e:
            errors.append(e)
        finally:
            written.put(None)

    def read_outputs() -> None:
        try:
            while (step := _poll(written.get, errors)) is not None:
                sid, pos = step
                out = model.read_decoded_output().reshape(-1)
                arrivals[sid].append(time.perf_counter())
                in_flight.release()
                if sid in keep:
                    logits[sid].append(out)
                if pos >= len(prompts[sid]) - 1:
                    token = _pick(out)
                    generated[sid].append(token)
                    picks[sid].put(token)
        except BaseException as e:
            errors.append(e)

    stages = [
        threading.Thread(target=write_packets, name="decode-packets", daemon=True),
        threading.Thread(target=read_outputs, name="decode-outputs", daemon=True),
    ]
    for stage in stages:
        stage.start()
    start = time.perf_counter()
    try:
        for step in schedule:
            while not in_flight.acquire(timeout=1.0):
                if errors:
                    raise RuntimeError("a decode stage failed") from errors[0]
            sid, pos = step
            model.activate_session(sid)
            model.replay_traced(pos)
            posted.put(step)
    finally:
        posted.put(None)
        for stage in stages:
            stage.join()
    if errors:
        raise RuntimeError("a decode stage failed") from errors[0]
    return {"start": start, "steps": len(schedule), "generated": generated, "arrivals": arrivals, "logits": logits}


def _log_throughput(run: dict, prompts: dict[int, list[int]], max_new: int) -> None:
    """Per user: time to the first pick and the generation rate after it. In total: the users' rates
    summed, and every decode step (prompt and generation) over the whole run's wall time."""
    total = 0.0
    for sid, ids in prompts.items():
        stamps = run["arrivals"][sid][len(ids) - 1 :]
        first = stamps[0] - run["start"]
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        total += rate
        logger.info(
            f"user {sid}: {len(ids)} prompt + {max_new} generated tokens, first token at {first:.2f} s, {rate:.2f} tok/s"
        )
    wall = max(stamps[-1] for stamps in run["arrivals"].values()) - run["start"]
    logger.info(
        f"total: {total:.2f} tok/s generated across {len(prompts)} users; "
        f"{run['steps'] / wall:.2f} decode steps/s overall ({run['steps']} steps in {wall:.2f} s)"
    )


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": _L1_SMALL_SIZE,
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_8stage"])
def test_interleaved_sessions_match_solo(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    from models.experimental.deepseek_v4_flash.demo.server import _GIL_SWITCH_SECONDS

    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    prompt_ids = [_tokenize_chat(tokenizer, text) for text in _TEXTS]
    _stage_engram_or_skip(loader, config)
    if host_rs.enabled():
        host_rs.build()
    with contextlib.ExitStack() as prefetcher:
        model, lm_head, loader, config = _construct_model(mesh_device, prefetcher, loader=loader, config=config)
        _assert_decode_parallelism(model, model.tp_size)
        _assert_prefetch_rings(model)
        max_seq = _traced_max_seq(config, max(len(ids) for ids in prompt_ids) + _MAX_NEW_TOKENS, model.num_layers)
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq),
            max_seq,
            lm_head=lm_head,
            num_sessions=_NUM_SESSIONS,
            total_tokens=_NUM_SESSIONS * max_seq,
            block_size=_PAGE_BLOCK_SIZE,
            device_sampling=_DEVICE_SAMPLING,
        )
        if model.device_sampler is not None:
            model.device_sampler.set_sampling(0.0, _SEED)
        logger.info(f"sampling: {'device' if _DEVICE_SAMPLING else 'host'}, greedy")
        sids = [model.open_session() for _ in prompt_ids]
        # The first step compiles and captures every variant; its writes are dropped by the reset.
        model.activate_session(sids[0])
        model.decode_traced(prompt_ids[0][0], 0)
        model.reset_session(sids[0])
        # Every session's blocks for its whole run, before any replay: growing one mid-run rewrites the page
        # tables, which first waits for every posted replay to be dispatched.
        for sid, ids in zip(sids, prompt_ids):
            model.ensure_session_capacity(len(ids) + _MAX_NEW_TOKENS - 2, sid)
        if host_rs.enabled():
            model.start_host_runtime()
        logger.info(f"host runtime: {'Rust' if model.host_runtime_running else 'Python'}")
        switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(_GIL_SWITCH_SECONDS)
        prefetcher.callback(sys.setswitchinterval, switch_interval)

        prompts = dict(zip(sids, prompt_ids))
        checked = {sids[i] for i in _CHECKED}
        run = _generate_interleaved(model, prompts, _MAX_NEW_TOKENS, checked)
        logger.info(f"interleaved run done; pool usage {model.session_usage()}")
        for sid, text in zip(sids, _TEXTS):
            generated = run["generated"][sid]
            if tokenizer.eos_token_id in generated:
                generated = generated[: generated.index(tokenizer.eos_token_id)]
            logger.info(f"user {sid} PROMPT    : {text!r}")
            logger.info(f"user {sid} GENERATED : {tokenizer.decode(generated)!r}")
        _log_throughput(run, prompts, _MAX_NEW_TOKENS)

        for sid in sorted(checked):
            fed = prompts[sid] + run["generated"][sid][:-1]
            model.reset_session(sid)
            solo = _decode_alone(model, sid, fed)
            for pos, (want, got) in enumerate(zip(solo, run["logits"][sid])):
                if got.dtype == torch.int32:
                    assert _pick(want) == _pick(got), f"user {sid} position {pos}: device pick differs from solo"
                    continue
                passing, pcc = comp_pcc(want.float(), got.float(), _PCC)
                assert passing, f"user {sid} position {pos}: interleaved logits PCC {pcc} vs solo"
                assert int(want.argmax()) == int(got.argmax()), f"user {sid} position {pos}: argmax differs"
            logger.info(f"user {sid}: {len(fed)} interleaved positions match the solo run")
