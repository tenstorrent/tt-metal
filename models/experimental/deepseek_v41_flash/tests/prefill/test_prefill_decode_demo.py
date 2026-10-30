# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end V4.1 demo: prefill real prompts with the traced prefill model, then generate with the traced decode model.

The V4.1 twin of ``models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_decode_demo.py`` (read its
docstring for the flow; this one follows it step for step). Each prompt's 128-aligned part is prefilled by
:class:`~models.experimental.deepseek_v41_flash.tt.prefill.model.DeepSeekV41PrefillModel` (traced, the stages
pipelined), its per-layer state committed into the decode caches on device
(:meth:`~models.experimental.deepseek_v41_flash.tt.model.DeepSeekV41Model.commit_prefill_state`: every layer's sliding
ring, and each compressed group's ``comp_kv`` / ``idx_key_cache`` on every stage that selects from it), the ragged tail
(``len % 128`` tokens) replayed through traced decode, and the answer generated.

* ``test_prefill_decode_demo``: the first ``DEEPSEEK_V4_DEMO_COUNT`` (20) ``length == "short"`` LongBench questions
  that fit the context (or exactly ``DEEPSEEK_V4_LONGBENCH_INDICES``), back to back; each verdict and a running tally
  are printed as it finishes, and a correctness table at the end.
* ``test_prefill_decode_longbench_known_correct``: an accuracy regression over the LongBench questions
  ``test_prefill_decode_demo`` answered correctly (``_KNOWN_CORRECT_LONGBENCH``, or ``DEEPSEEK_V4_LONGBENCH_INDICES``);
  it fails unless every one is still answered correctly, and skips while the list is empty.
* ``test_prefill_decode_text``: one chat prompt, ``DEEPSEEK_V4_E2E_TEXT`` or the decode demo's 128-token prompt (a
  quick bring-up; with ``DEEPSEEK_V4_DECODE_LAYERS=N`` the text is gibberish, the flow is not).

What differs from V4:

* Placement. The prefill layers share the decode layers' routed experts, so both models use one placement: the layers
  split evenly into contiguous runs (V4's ``plan_layer_placement`` with a pipeline group size of 1), five on each of
  the ``DEEPSEEK_V4_NUM_STAGES`` (default eight) ``1 x 4`` stages. Traced prefill sends a selection on when a stage
  edge splits an index group.
* Context. Prompt + new tokens must fit ``DEEPSEEK_V4_E2E_CONTEXT`` (16384 + ``DEEPSEEK_V4_MAX_NEW_TOKENS``); a
  LongBench question that does not is skipped. Past ``candidate_topk_blocks * candidate_block_size`` (16384) compressed entries prefill and decode both
  select two-level, so a larger context with ``DEEPSEEK_V4_E2E_COMPARE=1`` checks the two-level prefill against
  decode.
* The CSA lightning-indexer and overlap state of V4 do not exist: V4.1's index sources always select, from the index
  keys the commit writes.
* Engram layers 1 and 14 read host n-gram tables (~101 GB each, skipped when that RAM is not free). Prefill and decode
  share one :class:`~models.experimental.deepseek_v41_flash.tt.engram_lookup.EngramHostLookup`, so decode hashes the
  tail and the generated tokens with the prompt's history.

Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v41_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_demo

The known-correct questions::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v41_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_longbench_known_correct

A quick bring-up (layers 0-2: a window layer, an Engram layer and a ratio-2 kv + index source, one per stage)::

    DEEPSEEK_V4_DECODE_LAYERS=3 DEEPSEEK_V4_MAX_NEW_TOKENS=32 DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v41_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_text

Knobs (environment), as in the V4 demo: ``DEEPSEEK_V4_MAX_NEW_TOKENS`` (16384), ``DEEPSEEK_V4_E2E_CHUNK`` (1024 = prefill
chunk), ``DEEPSEEK_V4_E2E_TEXT``, ``DEEPSEEK_V4_DECODE_LAYERS``, ``DEEPSEEK_V4_E2E_MAX_INPUT``,
``DEEPSEEK_V4_MAX_INPUT_PREFILL_CHUNKS``, ``DEEPSEEK_V4_E2E_COMPARE`` (1: also run each prompt through decode only,
before its prefill, and compare the next-token logits -- the check that the commit is right),
``DEEPSEEK_V4_PREFILL_MAX_LEN``, ``DEEPSEEK_V4_TRACE_REGION_SIZE``,
``DEEPSEEK_V4_L1_SMALL_SIZE``, ``DEEPSEEK_V4_PREFETCHER``, ``DEEPSEEK_V4_GCB_HEADROOM``, ``DEEPSEEK_V4_LONGBENCH``,
``DEEPSEEK_V4_DEMO_COUNT``, ``DEEPSEEK_V4_LONGBENCH_INDICES``, ``DEEPSEEK_V4_E2E_CONTEXT`` (16384 +
``DEEPSEEK_V4_MAX_NEW_TOKENS``, prompt + new tokens), and ``DEEPSEEK_V4_NUM_STAGES`` (8; how many ``1 x 4``
submeshes are carved from the full 8 x 4 mesh. ``DEEPSEEK_V4_PREFILL_STAGES`` still applies when the former is unset).
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from typing import Callable, Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _DEFAULT_TEXT,
    _ENGRAM_RAM_MARGIN,
    _PAGE_BLOCK_SIZE,
    _InterruptFlag,
    _assert_decode_parallelism,
    _available_bytes,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _TP_SIZE,
)
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    load_config,
    pipeline_stage_count,
)
from models.experimental.deepseek_v41_flash.tt.engram_lookup import engram_ram_needed
from models.experimental.deepseek_v41_flash.tt.model import DeepSeekV41Model
from models.experimental.deepseek_v41_flash.tt.prefill.model import DeepSeekV41PrefillModel
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import _Progress, _env_int
from models.experimental.deepseek_v4_flash.tests.prefill.test_prefill_decode_demo import (
    _DECODE_REPLAY_AHEAD,
    _DEVICE_PARAMS,
    _LONGBENCH_FILE,
    _Prompt,
    _Result,
    _compare_logits,
    _extract_choice,
    _pick_longbench,
    _prefill_one,
    _print_correctness,
    _summary,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT, PrefillAttentionState
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# The full Galaxy stays open. DEEPSEEK_V4_NUM_STAGES (default 8) is how many 1x4 submeshes decode and
# prefill share; the other rows stay idle.
_NUM_STAGES = pipeline_stage_count()
_MESH = (8, _TP_SIZE)
# LongBench indices of the ``length == "short"`` questions test_prefill_decode_demo answered correctly. Empty until a
# full-model V4.1 run records them (V4's list does not carry over: those passages are past V4.1's 16384-token context).
_KNOWN_CORRECT_LONGBENCH: list[int] = []


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.skipif(not _LONGBENCH_FILE.is_file(), reason=f"LongBench file not found at {_LONGBENCH_FILE}")
@pytest.mark.timeout(86400)  # 20 long passages, prefill + decode each
@torch.no_grad()
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"], ids=["fabric_2d"])
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_decode_demo(mesh_device, reset_seeds) -> None:
    """The first ``DEEPSEEK_V4_DEMO_COUNT`` (20) ``length == "short"`` questions of ``data.json`` that fit the context,
    in file order, answered back to back by the two models (each loaded and captured once)."""
    from transformers import AutoTokenizer

    count = _env_int("DEEPSEEK_V4_DEMO_COUNT", 20)
    indices = os.environ.get("DEEPSEEK_V4_LONGBENCH_INDICES")

    def make_prompts(tokenizer, max_prompt_tokens: int) -> list[_Prompt]:
        chosen = [int(i) for i in indices.split(",")] if indices else None
        return _pick_longbench(tokenizer, _LONGBENCH_FILE, count, None, max_prompt_tokens, chosen)

    results = _run_with_progress(mesh_device, AutoTokenizer, make_prompts)
    assert all(r.generated for r in results)
    _print_correctness(results)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)  # weight conversion on a cold cache + one prompt
@torch.no_grad()
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"], ids=["fabric_2d"])
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_decode_text(mesh_device, reset_seeds) -> None:
    """One chat prompt (``DEEPSEEK_V4_E2E_TEXT``, default the decode demo's 128-token prompt), prefilled and answered."""
    from transformers import AutoTokenizer

    text = os.environ.get("DEEPSEEK_V4_E2E_TEXT", _DEFAULT_TEXT)

    def make_prompts(tokenizer, max_prompt_tokens: int) -> list[_Prompt]:
        ids = _tokenize_chat(tokenizer, text)
        if len(ids) > max_prompt_tokens:
            raise ValueError(f"the prompt has {len(ids)} tokens, more than the {max_prompt_tokens} that fit")
        return [_Prompt("text", ids)]

    results = _run_with_progress(mesh_device, AutoTokenizer, make_prompts)
    assert all(r.generated for r in results)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.skipif(not _LONGBENCH_FILE.is_file(), reason=f"LongBench file not found at {_LONGBENCH_FILE}")
@pytest.mark.timeout(86400)  # long passages, prefill + decode each
@torch.no_grad()
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"], ids=["fabric_2d"])
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_decode_longbench_known_correct(mesh_device, reset_seeds) -> None:
    """Accuracy regression: the ``_KNOWN_CORRECT_LONGBENCH`` questions (or ``DEEPSEEK_V4_LONGBENCH_INDICES``), which
    ``test_prefill_decode_demo`` answered correctly, must all still be answered correctly."""
    from transformers import AutoTokenizer

    indices = os.environ.get("DEEPSEEK_V4_LONGBENCH_INDICES")
    known = [int(i) for i in indices.split(",")] if indices else _KNOWN_CORRECT_LONGBENCH
    if not known:
        pytest.skip(
            "no known-correct V4.1 LongBench questions yet: run test_prefill_decode_demo, then put the indices it "
            "answered correctly in _KNOWN_CORRECT_LONGBENCH (or pass them in DEEPSEEK_V4_LONGBENCH_INDICES)"
        )

    def make_prompts(tokenizer, max_prompt_tokens: int) -> list[_Prompt]:
        return _pick_longbench(tokenizer, _LONGBENCH_FILE, len(known), None, max_prompt_tokens, known)

    results = _run_with_progress(mesh_device, AutoTokenizer, make_prompts)
    correct = _print_correctness(results)
    assert all(r.generated for r in results)
    assert len(results) == len(known), (
        f"only {len(results)} of {len(known)} questions were run (a prompt that, with DEEPSEEK_V4_MAX_NEW_TOKENS new "
        "tokens, does not fit the context was skipped)"
    )
    wrong = [f"{r.name} (expected {r.expected}, model {r.choice or '?'})" for r in results if r.choice != r.expected]
    assert not wrong, f"{correct}/{len(results)} correct; wrong: {', '.join(wrong)}"


def _run_with_progress(mesh_device, AutoTokenizer, make_prompts) -> list[_Result]:
    progress = _Progress(interval=0.0, stall=0.0)  # never entered: step logging only, no heartbeat thread
    progress.verbose = False
    # Room above the GCB rings for decode's own buffers, so releasing the rings frees one block of L1 for prefill.
    os.environ.setdefault("DEEPSEEK_V4_GCB_HEADROOM", "16384")
    # The decode model's prefetcher session spans everything (as in the decode demo).
    with contextlib.ExitStack() as prefetcher:
        return _run(mesh_device, progress, prefetcher, AutoTokenizer, make_prompts)


# --------------------------------------------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------------------------------------------- #
def _run(
    mesh_device,
    progress: _Progress,
    prefetcher: contextlib.ExitStack,
    AutoTokenizer,
    make_prompts: Callable[..., list[_Prompt]],
) -> list[_Result]:
    max_new = _env_int("DEEPSEEK_V4_MAX_NEW_TOKENS", 16384)
    chunk_size = _env_int("DEEPSEEK_V4_E2E_CHUNK", 1024)
    compare = os.environ.get("DEEPSEEK_V4_E2E_COMPARE", "0") == "1"
    if chunk_size <= 0 or chunk_size % ALIGNMENT:
        raise ValueError(f"DEEPSEEK_V4_E2E_CHUNK={chunk_size} must be a positive multiple of {ALIGNMENT}")

    # --- tokenizer, config, prompts ----------------------------------------------------------------- #
    progress.step("[1/6] tokenizer, config and prompts")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers), config.num_hidden_layers)
    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    if engram_ids:
        need = _ENGRAM_RAM_MARGIN * engram_ram_needed(loader, engram_ids)
        if _available_bytes() < need:
            pytest.skip(f"Engram tables for layers {engram_ids} need {need / 1e9:.0f} GB of free memory with headroom")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    context_limit = _env_int(
        "DEEPSEEK_V4_E2E_CONTEXT", config.candidate_topk_blocks * config.candidate_block_size + max_new
    )
    if max_new >= context_limit:
        raise ValueError(f"DEEPSEEK_V4_MAX_NEW_TOKENS={max_new} leaves no room for a prompt in {context_limit} tokens")
    prompts = make_prompts(tokenizer, context_limit - max_new)
    max_input = _env_int("DEEPSEEK_V4_E2E_MAX_INPUT", 0)
    if max_input > 0:
        for prompt in prompts:
            if len(prompt.ids) > max_input:
                logger.info(f"{prompt.name}: truncated to its first {max_input} of {len(prompt.ids)} tokens")
                prompt.ids = prompt.ids[:max_input]
    max_prefill_chunks = _env_int("DEEPSEEK_V4_MAX_INPUT_PREFILL_CHUNKS", 0)
    if max_prefill_chunks < 0:
        raise ValueError(f"DEEPSEEK_V4_MAX_INPUT_PREFILL_CHUNKS={max_prefill_chunks} must be >= 0")
    if max_prefill_chunks:
        keep = max_prefill_chunks * chunk_size
        for prompt in prompts:
            if len(prompt.ids) > keep:
                logger.info(
                    f"{prompt.name}: truncated to its first {max_prefill_chunks} prefill chunk(s), {keep} of "
                    f"{len(prompt.ids)} tokens"
                )
                prompt.ids = prompt.ids[:keep]
    # DEEPSEEK_V4_PREFILL_MAX_LEN (0 = unset): the prefill buffer / trace size. It caps what a prompt prefills (the
    # excess goes through decode) and sizes the buffers even when every prompt is shorter.
    prefill_max_len = _env_int("DEEPSEEK_V4_PREFILL_MAX_LEN", 0) // ALIGNMENT * ALIGNMENT
    results = []
    for prompt in prompts:
        real_len = len(prompt.ids)
        aligned = real_len // ALIGNMENT * ALIGNMENT
        if prefill_max_len:
            aligned = min(aligned, prefill_max_len)
        results.append(_Result(prompt.name, real_len, aligned, expected=prompt.expected))
        logger.info(
            f"{prompt.name}: {real_len} tokens = {aligned} prefilled ({math.ceil(aligned / chunk_size)} chunk(s) of up "
            f"to {chunk_size}) + {real_len - aligned} replayed through decode; up to {max_new} new tokens"
        )
        logger.info(f"  starts : {tokenizer.decode(prompt.ids[:24])!r}")
        logger.info(f"  ends   : {tokenizer.decode(prompt.ids[-80:])!r}")

    # --- the full mesh stays open; only ``_NUM_STAGES`` rows become submeshes, shared by both models ---- #
    system_config = load_system_config(mesh_device=mesh_device)
    set_active_system_config(system_config)
    mesh_rows = tuple(mesh_device.shape)[0]
    if not 1 <= _NUM_STAGES <= mesh_rows:
        raise ValueError(f"DEEPSEEK_V4_NUM_STAGES={_NUM_STAGES} must be in [1, {mesh_rows}] (mesh rows)")
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)) for i in range(_NUM_STAGES)
    ]
    placement = plan_layer_placement(num_layers, _NUM_STAGES, 1)
    spans = []
    for k in dict.fromkeys(placement):
        owned = [li for li, stage in enumerate(placement) if stage == k]
        spans.append(f"stage {k}: layers {owned[0]}-{owned[-1]}")
    logger.info(f"{num_layers}/{config.num_hidden_layers} layers over {len(spans)} stage(s): " + ", ".join(spans))
    needed = max(r.tokens for r in results) + max_new  # decode is sized once, for the longest conversation
    max_seq = _traced_max_seq(config, needed, num_layers)
    if max_seq < needed:
        raise ValueError(f"{needed} tokens needed but decode is capped at {max_seq}")
    rope = _build_rope(config, loader.snapshot_dir, max_seq)

    # --- the decode model: static state and session, before anything of prefill exists ------------------ #
    # Everything allocated here, in DRAM and in L1, is live while prefill is captured, so prefill keeps off it.
    use_prefetcher = os.environ.get("DEEPSEEK_V4_PREFETCHER", "1") == "1"
    progress.step(
        f"[2/6] decode model ({len(spans)} x TP{_TP_SIZE} stages, prefetcher {'on' if use_prefetcher else 'off'})"
    )
    t0 = time.perf_counter()
    decode, lm_head, loader, config = _construct_model(
        mesh_device,
        prefetcher,
        tp_size=_TP_SIZE,
        system_config=system_config,
        loader=loader,
        config=config,
        num_stages=_NUM_STAGES,
        submeshes=submeshes,
        use_prefetcher=use_prefetcher,
        layer_submesh_ids=placement,
    )
    _assert_decode_parallelism(decode, _TP_SIZE, _NUM_STAGES)
    assert decode.num_layers == num_layers, f"decode built {decode.num_layers} layers, prefill {num_layers}"
    logger.info(f"decode model: {num_layers} layers built in {time.perf_counter() - t0:.1f}s")
    decode.prepare_static_decode(
        rope, max_seq, lm_head=lm_head, num_sessions=1, total_tokens=max_seq, block_size=_PAGE_BLOCK_SIZE
    )
    if decode.context_limit is not None and decode.context_limit < needed:
        raise ValueError(f"decode context capped at {decode.context_limit} < the {needed} tokens needed")
    sid = decode.open_session()
    decode.activate_session(sid)

    # --- decode compile, then prefill compile and capture, with decode's GCBs and L1 tensors released ----- #
    prefill = None
    build_seconds = prepare_seconds = 0.0
    num_stages = len(spans)
    max_len = prefill_max_len or max(r.prefilled for r in results)
    # Prefill gathers its RoPE rows for every position up to ``max_len``; decode's table only spans its own ``max_seq``.
    prefill_rope = rope if max_len <= max_seq else _build_rope(config, loader.snapshot_dir, max_len)
    if max_len:
        # Decode's ops keep buffers of their own (global semaphores, cached constants) that no snapshot can reach;
        # compiling first allocates them before prefill lays out its memory, so its replays keep off them.
        progress.step("[3/6] decode compile (every variant), before anything of prefill exists")
        t0 = time.perf_counter()
        decode.compile_traces(pad_id, 0)
        logger.info(f"decode compile: {time.perf_counter() - t0:.1f}s")
        progress.step(
            "[3/6] prefill model on the decode stages (sharing the routed experts, embedding, lm_head, Engram)"
        )
        # Stops the prefetcher and frees the GCBs; allowed because no decode trace exists yet.
        decode.release_prefetch_buffers()
        t0 = time.perf_counter()
        prefill = decode.build_prefill(
            prefill_rope,
            lm_head=lm_head,
            cache=WeightCache(CACHE_DIR) if CACHE_DIR else None,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            progress=progress,
        )
        prefill.synchronize("uploads")
        build_seconds = time.perf_counter() - t0
        logger.info(f"prefill model built in {build_seconds:.1f}s")
        progress.step(
            f"[3/6] prefill compile for prompts of up to {max_len} tokens: persistent buffers, H2D / D2D / D2H "
            f"sockets, compile the {chunk_size}-token chunk step (once; not timed)"
        )
        t0 = time.perf_counter()
        prefill.compile_traced_prefill(max_len, chunk_size)
        prepare_seconds = time.perf_counter() - t0
        logger.info(f"traced prefill compiled in {prepare_seconds:.1f}s")
        # Unwinds before the decode model's shutdown (LIFO).
        prefetcher.callback(prefill.release_traced_prefill)

        # --- prefill capture, before decode's: stopping the prefetcher would drop the decode traces' requests -- #
        progress.step("[4/6] prefill capture: one trace per stage (decode's L1 still released)")
        t0 = time.perf_counter()
        prefill.capture_traced_prefill()
        capture_seconds = time.perf_counter() - t0
        prepare_seconds += capture_seconds
        logger.info(f"traced prefill captured in {capture_seconds:.1f}s")
        # The GCBs (config pages in L1_SMALL) and the L1 tensors come back under prefill's buffers; restarts the
        # prefetcher, which then runs until teardown.
        decode.restore_prefetch_buffers()

    # --- decode capture: a throw-away step; its scratch cache writes are overwritten by every commit ------ #
    progress.step("[4/6] throw-away decode step: compiles against the rebuilt rings and captures the decode traces")
    t0 = time.perf_counter()
    decode.decode_traced(pad_id, 0)
    logger.info(f"trace capture + first step: {time.perf_counter() - t0:.1f}s")
    if prefill is not None:
        # Decode's L1 tensors, and what its capture allocated, sit under prefill's buffers, so its replays
        # overwrite them.
        decode.snapshot_resident_state()

    # --- every prompt: prefill execute, then decode execute, in the one session -------------------------- #
    for k, (prompt, result) in enumerate(zip(prompts, results)):
        reference = None
        if compare and result.prefilled:
            reference = _reference_logits(decode, sid, prompt, result.prefilled, f"[5/6] {prompt.name}", progress)
        row = states = None
        if result.prefilled:
            tag = f"[5/6] prefill {k + 1}/{len(prompts)} {prompt.name}"
            row, states = _prefill_one(
                prefill,
                prompt,
                result,
                tag,
                progress,
                tokenizer,
                chunk_size,
                num_stages,
                build_seconds,
                prepare_seconds,
            )
            decode.reload_resident_state()
        tag = f"[6/6] decode {k + 1}/{len(prompts)} {prompt.name}"
        _decode_one(
            decode,
            prefill,
            sid,
            prompt,
            result,
            row,
            states,
            reference,
            tag,
            progress,
            tokenizer,
            eos,
            max_new,
        )
        assert all(0 <= t < config.vocab_size for t in result.generated)
        if result.expected is not None:
            graded = [r for r in results[: k + 1] if r.expected is not None]
            correct = sum(r.choice == r.expected for r in graded)
            verdict = "correct" if result.choice == result.expected else "WRONG"
            print(
                f"\n[{k + 1}/{len(prompts)}] {result.name}: expected {result.expected}, model {result.choice or '?'} "
                f"-> {verdict}  (running: {correct}/{len(graded)} correct)",
                flush=True,
            )
    logger.info(f"pool usage: {decode.session_usage()}")
    _summary(results, prepare_seconds, decode.use_prefetcher)
    progress.step("done")
    return results


def _reference_logits(
    decode: DeepSeekV41Model, sid: int, prompt: _Prompt, aligned: int, tag: str, progress: _Progress
) -> torch.Tensor:
    """``DEEPSEEK_V4_E2E_COMPARE=1``: decode-only logits at the last aligned position, to compare with prefill's.

    Run before the prompt's prefill, so no prefill state is alive while decode replays.
    """
    progress.step(f"{tag}: reference: {aligned} prompt tokens through decode only (DEEPSEEK_V4_E2E_COMPARE=1)")
    decode.reset_session(sid)
    t0 = time.perf_counter()

    def alive(i: int, _out) -> None:
        if (i + 1) % 512 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            logger.info(f"{tag}: reference {i + 1}/{aligned} ({rate:.1f} tok/s)")

    return decode.decode_prompt_traced(prompt.ids[:aligned], 0, on_output=alive).reshape(-1).float()


def _decode_one(
    decode: DeepSeekV41Model,
    prefill: Optional[DeepSeekV41PrefillModel],
    sid: int,
    prompt: _Prompt,
    result: _Result,
    row: Optional[torch.Tensor],
    states: Optional[list[PrefillAttentionState]],
    reference: Optional[torch.Tensor],
    tag: str,
    progress: _Progress,
    tokenizer,
    eos: set[int],
    max_new: int,
) -> None:
    """Rewind the session, commit one prompt's prefill state, replay its tail and generate its answer."""
    aligned, real_len = result.prefilled, result.tokens
    decode.reset_session(sid)
    next_id = None
    if aligned:
        if reference is not None:
            _compare_logits(tokenizer, row.float(), reference[: row.numel()])

        # Committed on device (a group whose kv source is on another stage goes through the host), then freed
        # before the next decode_traced: the states were allocated under the live traces.
        progress.step(f"{tag}: commit: prefill state -> decode caches")
        t0 = time.perf_counter()
        committed = decode.commit_prefill_state(states, sid, progress=lambda m: progress(m, important=False))
        prefill.free_traced_states(states)
        assert committed == aligned
        logger.info(f"commit of {committed} tokens: {time.perf_counter() - t0:.2f}s")
        if aligned == real_len:
            next_id = int(row.argmax())
    else:
        logger.warning(f"{prompt.name} is shorter than one 128-token block: nothing to prefill, decode replays it all")

    # --- the ragged tail (or the whole short prompt) through decode --------------------------------- #
    progress.step(f"{tag}: replaying {real_len - aligned} prompt token(s) through decode")
    t0 = time.perf_counter()
    if real_len > aligned:
        logits = decode.decode_prompt_traced(prompt.ids[aligned:real_len], aligned).reshape(1, -1).float()
        next_id = int(logits[0].argmax())
        logger.info(f"tail replay: {real_len - aligned} tokens in {time.perf_counter() - t0:.2f}s")
    assert next_id is not None
    logger.info(f"first generated token: {next_id} {tokenizer.decode([next_id])!r}")

    # --- generation ------------------------------------------------------------------------------- #
    progress.step(f"{tag}: decode: up to {max_new} tokens")
    generated = [next_id]
    step_times: list[float] = []
    # As in the decode demo: the traces are posted a window ahead, so each step is only its packet and its output.
    positions = [real_len + step - 1 for step in range(1, max_new)]
    if positions:
        decode.ensure_session_capacity(positions[-1])
    posted = fed = read = 0  # positions handed to replay_traced_ahead / packets written / outputs read
    interrupt = _InterruptFlag()
    interrupt.install()
    try:
        for step, pos in enumerate(positions, start=1):
            if interrupt.hit:
                logger.info("interrupted; unwinding the posted replays")
                raise KeyboardInterrupt
            if generated[-1] in eos:
                logger.info("hit EOS; stopping")
                break
            want = min(len(positions), step - 1 + _DECODE_REPLAY_AHEAD)
            if want > posted:
                decode.replay_traced_ahead(positions[posted:want])
                posted = want
            t0 = time.perf_counter()
            decode.write_step_packet(generated[-1], pos)
            fed += 1
            logits = decode.read_decoded_output().reshape(1, -1).float()
            read += 1
            generated.append(int(logits[0].argmax()))
            step_times.append(time.perf_counter() - t0)
            print(tokenizer.decode([generated[-1]]), end="", flush=True)
            if step % 32 == 0:
                recent = step_times[-32:]
                logger.info(
                    f"decode {step}/{max_new}: {len(recent) / sum(recent):.2f} tok/s (last {len(recent)} tokens)"
                )
    finally:
        # Every posted replay must be fed and read before the next prefill or session reset; their logits are
        # not part of the reply.
        try:
            for i in range(read, posted):
                if i >= fed:
                    decode.write_step_packet(generated[-1], positions[i])
                    fed += 1
                decode.read_decoded_output()
                read += 1
        finally:
            interrupt.restore()
    if step_times:
        result.decode_tps = len(step_times) / sum(step_times)
        logger.info(
            f"decode: {len(generated)} tokens, {result.decode_tps:.2f} tok/s steady (first step {step_times[0]:.2f}s)"
        )
    result.generated = generated
    result.text = tokenizer.decode(generated)
    logger.info(f"PROMPT (last 600 chars):\n{tokenizer.decode(prompt.ids)[-600:]}")
    logger.info(f"GENERATED ({len(generated)} tokens):\n{result.text}")
    if prompt.expected is not None:
        result.choice = _extract_choice(result.text)
        verdict = "correct" if result.choice == prompt.expected else "WRONG"
        logger.info(f"{prompt.name}: expected {prompt.expected}, model {result.choice} -> {verdict}")
