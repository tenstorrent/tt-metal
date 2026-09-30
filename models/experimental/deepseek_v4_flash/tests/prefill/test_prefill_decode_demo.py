# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: prefill real prompts with the prefill model, then generate with the traced decode model.

Each prompt is prefilled in 128-aligned chunks by :class:`DeepSeekV4PrefillModel`; its per-layer attention state is
then committed into the decode model's own buffers on device (:meth:`DeepSeekV4Model.commit_prefill_state`: same
ring / compressed KV / CSA overlap window / paged HCA pool that decode fills itself), the prompt's ragged tail
(``len % 128`` tokens) is fed through ``decode_traced`` one token at a time, and generation continues with
``decode_traced`` exactly like ``tests/decode/test_full_model_decode_demo.py``.

* ``test_prefill_decode_demo``: a single ``length == "short"`` LongBench question (seeded random pick, or the
  first of ``DEEPSEEK_V4_LONGBENCH_INDICES``), indexer on; its generated letter is logged against ``answer``.
* ``test_prefill_decode_longbench``: ``DEEPSEEK_V4_LONGBENCH_COUNT`` (8) questions picked at random (seeded) from
  the ``length == "short"`` multiple-choice items of ``~/smanoj/data.json``, every one answered by the same two
  models, each loaded once. It reports each question's generated letter against ``answer`` and the accuracy. The
  passages are far past 2048 tokens, so the CSA lightning indexer is on; it runs inside the prefill traces.

The prefill is traced and pipelined by default, as in ``test_full_model_prefill_demo.py``
(:class:`~models.experimental.deepseek_v4_flash.tt.model.TracedPrefill`): one trace per stage replayed for every
chunk, the replays posted ahead from a thread, each chunk's packet pushed over an H2D socket, the streams handed on
over device-to-device sockets and the logits streamed back over a D2H socket, so the 8 stages work on 8 consecutive
chunks at once. Every prompt's prefill perf table and summary (that demo's) are logged after its prefill, and a
closing table covers every prompt's prefill and decode. ``DEEPSEEK_V4_PREFILL_TRACED=0`` prefills eagerly instead.

Chips (Galaxy 8 x 4): eight ``1 x 4`` TP4 pipeline stages, one per mesh row, all 32 chips. Prefill runs first on
the same stages (one layer per decode layer's submesh, with its own weights), and is freed before the decode model
is built there, so the two never hold weights at the same time (the decode model's resident L1 -- its prefetch
buffers -- leaves prefill's ops no room). So the models are loaded once each, in two phases, rather than
alternating per prompt:

1. tokenizer, config and the prompts;
2. the prefill model, built on the 8 stages before anything of decode exists;
3. (traced) once, on otherwise empty devices: persistent buffers sized for the longest prompt, sockets, the
   chunk step compiled and captured, one trace per stage; not timed. Then per prompt: its 128-aligned part
   prefilled by replaying those traces (timed) and its per-layer states parked on the host. The traces are
   released after the last prompt;
4. the prefill model freed, the decode model built on the same submeshes, its static state prepared (sized for the
   longest prompt), a session opened and one throw-away decode step run, which captures the decode traces;
5. per prompt: the session rewound, the prompt's states uploaded and committed into the decode buffers on device,
   the ragged tail replayed through decode, and up to ``DEEPSEEK_V4_MAX_NEW_TOKENS`` tokens generated.

Only one prompt's state is on device at a time; the parked states cost host memory (roughly
``prompt tokens x 25 KB`` for the full model).

Limits: ``test_prefill_decode_demo`` leaves the lightning indexer off, so the conversation (prompt + generated
tokens) must stay below ``index_topk * 4 = 2048`` tokens. The longbench test turns the indexer on and prefills
the whole passage.

Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_demo

The longbench questions (indexer on; the prefill traces replayed)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v4_flash/tests/prefill/test_prefill_decode_demo.py::test_prefill_decode_longbench

Knobs (environment): ``DEEPSEEK_V4_E2E_PROMPT_LEN`` (1000 tokens; not a multiple of 128 on purpose, so the tail path
runs), ``DEEPSEEK_V4_MAX_NEW_TOKENS`` (128), ``DEEPSEEK_V4_E2E_CHUNK`` (1024 = prefill chunk),
``DEEPSEEK_V4_E2E_TEXT`` (a plain user message instead of the book prompt; ``decode_demo`` = the prompt of
``tests/decode/test_full_model_decode_demo.py``, exactly 128 tokens),
``DEEPSEEK_V4_DECODE_LAYERS`` (bring-up: first N layers in both models; the text is then gibberish, the flow is not),
``DEEPSEEK_V4_PREFILL_PROMPT`` (another prompt file), ``DEEPSEEK_V4_PREFILL_HEARTBEAT`` / ``_STALL_SECS``,
``DEEPSEEK_V4_PREFILL_TRACED`` (1, the default: the traced, pipelined prefill; 0: eager),
``DEEPSEEK_V4_E2E_MAX_INPUT`` (0 = off: keep only a prompt's first N tokens),
``DEEPSEEK_V4_E2E_TRACE_CHECK`` (1: prefill the first prompt traced, then eagerly, compare, and stop),
``DEEPSEEK_V4_E2E_COMPARE`` (1: also run each prompt through decode only and compare the next-token logits),
``DEEPSEEK_V4_TRACE_REGION_SIZE`` (bytes to reserve for the captured traces; unset keeps the ttnn default -- set it,
e.g. 500000000, if a capture reports the trace region too small).
LongBench: ``DEEPSEEK_V4_LONGBENCH`` (the question file), ``DEEPSEEK_V4_LONGBENCH_COUNT`` (8),
``DEEPSEEK_V4_LONGBENCH_SEED`` (0), ``DEEPSEEK_V4_LONGBENCH_MAX_TOKENS`` (65536: a question whose prompt plus the new
tokens is longer is skipped and another drawn), ``DEEPSEEK_V4_LONGBENCH_INDICES`` (comma-separated question indices
instead of a random draw, e.g. ``1`` for the one question the test used to run),
``DEEPSEEK_V4_LONGBENCH_MIN_CORRECT`` (0: the test fails if fewer answers are right).
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import (
    _CACHE_DIR,
    _DEFAULT_MODEL_DIR,
    _PAGE_BLOCK_SIZE,
    _assert_decode_parallelism,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _TP_SIZE,
    _Progress,
    _env_int,
    _report,
)
from models.experimental.deepseek_v4_flash.tt.decode.paged_cache import round_context
from models.experimental.deepseek_v4_flash.tt.model import (
    DeepSeekV4PrefillModel,
    plan_layer_placement,
    prefill_bias_slots,
)
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT, PrefillAttentionState
from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_expert_provider, checkpoint_weights
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_MAX_CONTEXT = 2048  # index_topk * CSA rate: the dense CSA / no-indexer-state limit of the state commit
_NUM_STAGES = 8  # 1 x 4 TP4 stages, one per Galaxy row: decode and prefill share all 32 chips
_LONGBENCH_FILE = Path(os.path.expanduser(os.environ.get("DEEPSEEK_V4_LONGBENCH", "~/smanoj/data.json")))
_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "num_command_queues": 2,
    # Like the prefill demo: only reserve a trace region when asked (the ttnn default otherwise).
    **(
        {"trace_region_size": int(os.environ["DEEPSEEK_V4_TRACE_REGION_SIZE"])}
        if os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE")
        else {}
    ),
}
# The tensors of a PrefillAttentionState, parked on the host between the prefill and the decode phase.
_STATE_TENSORS = (
    "kv_tail",
    "compressed_kv",
    "csa_prev_kv",
    "csa_prev_gate",
    "idx_keys",
    "idx_prev_kv",
    "idx_prev_gate",
)


@dataclass
class _Prompt:
    name: str
    ids: list[int]
    expected: Optional[str] = None  # the LongBench answer letter, when there is one


@dataclass
class _Result:
    """One prompt's outcome and perf, filled in as the run goes."""

    name: str
    tokens: int
    prefilled: int  # the 128-aligned part; the rest goes through decode
    expected: Optional[str] = None
    prefill_seconds: float = 0.0
    pipelined_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    generated: list = field(default_factory=list)
    text: str = ""
    choice: Optional[str] = None


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.skipif(not _LONGBENCH_FILE.is_file(), reason=f"LongBench file not found at {_LONGBENCH_FILE}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"], ids=["fabric_2d"])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_decode_demo(mesh_device, reset_seeds) -> None:
    """One random (seeded) ``length == "short"`` question of ``data.json``, answered once."""
    from transformers import AutoTokenizer

    seed = _env_int("DEEPSEEK_V4_LONGBENCH_SEED", 0)
    max_tokens = _env_int("DEEPSEEK_V4_LONGBENCH_MAX_TOKENS", 65536)
    indices = os.environ.get("DEEPSEEK_V4_LONGBENCH_INDICES")

    def make_prompts(tokenizer, max_new: int) -> list[_Prompt]:
        chosen = [int(indices.split(",")[0])] if indices else None
        return _pick_longbench(tokenizer, _LONGBENCH_FILE, 1, seed, max_tokens - max_new, chosen)

    results = _run_with_progress(mesh_device, AutoTokenizer, make_prompts, lightning_indexer=True)
    assert len(results) == 1 and results[0].generated
    result = results[0]
    logger.info(f"{result.name}: expected {result.expected}, model {result.choice}")


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.skipif(not _LONGBENCH_FILE.is_file(), reason=f"LongBench file not found at {_LONGBENCH_FILE}")
@pytest.mark.timeout(86400)  # 8 long passages, prefill + decode each
@torch.no_grad()
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"], ids=["fabric_2d"])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_prefill_decode_longbench(mesh_device, reset_seeds) -> None:
    """Answer ``DEEPSEEK_V4_LONGBENCH_COUNT`` random short questions of ``data.json``, the models loaded once."""
    from transformers import AutoTokenizer

    count = _env_int("DEEPSEEK_V4_LONGBENCH_COUNT", 8)
    seed = _env_int("DEEPSEEK_V4_LONGBENCH_SEED", 0)
    max_tokens = _env_int("DEEPSEEK_V4_LONGBENCH_MAX_TOKENS", 65536)
    indices = os.environ.get("DEEPSEEK_V4_LONGBENCH_INDICES")
    min_correct = _env_int("DEEPSEEK_V4_LONGBENCH_MIN_CORRECT", 0)

    def make_prompts(tokenizer, max_new: int) -> list[_Prompt]:
        chosen = [int(i) for i in indices.split(",")] if indices else None
        return _pick_longbench(tokenizer, _LONGBENCH_FILE, count, seed, max_tokens - max_new, chosen)

    results = _run_with_progress(mesh_device, AutoTokenizer, make_prompts, lightning_indexer=True)
    correct = sum(r.choice == r.expected for r in results)
    logger.info(f"longbench: {correct}/{len(results)} correct ({correct / len(results):.0%}), seed {seed}")
    assert all(r.generated for r in results)
    assert correct >= min_correct, f"{correct}/{len(results)} correct < DEEPSEEK_V4_LONGBENCH_MIN_CORRECT={min_correct}"


def _run_with_progress(mesh_device, AutoTokenizer, make_prompts, **kwargs) -> list[_Result]:
    progress = _Progress(interval=0.0, stall=0.0)  # never entered: step logging only, no heartbeat thread
    progress.verbose = False
    os.environ["DEEPSEEK_V4_DSPARK"] = "0"  # MTP disabled for now (read by DeepSeekV4Model.__init__): no link on row 2
    # The decode model's prefetcher session spans everything (as in the decode demo).
    with contextlib.ExitStack() as prefetcher:
        return _run(mesh_device, progress, prefetcher, AutoTokenizer, make_prompts, **kwargs)


# --------------------------------------------------------------------------------------------------------------- #
# LongBench
# --------------------------------------------------------------------------------------------------------------- #
def _longbench_prompt(entry: dict) -> str:
    """The LongBench-v2 zero-shot user turn: the passage, then the question and four choices."""
    return (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
        f"{entry['context']}\n"
        "</text>\n\n"
        f"What is the correct answer to this question: {entry['question']}\n"
        "Choices:\n"
        f"(A) {entry['choice_A']}\n"
        f"(B) {entry['choice_B']}\n"
        f"(C) {entry['choice_C']}\n"
        f"(D) {entry['choice_D']}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def _pick_longbench(
    tokenizer, path: Path, count: int, seed: int, max_prompt_tokens: int, indices: Optional[list[int]] = None
) -> list[_Prompt]:
    """``count`` short questions of ``path`` drawn at random with ``seed`` (or exactly ``indices``), as prompts.

    A drawn question whose prompt is longer than ``max_prompt_tokens`` is skipped and the next one drawn.
    """
    data = json.loads(path.read_text())
    if indices is None:
        pool = [i for i, entry in enumerate(data) if entry.get("length") == "short"]
        random.Random(seed).shuffle(pool)
    else:
        for i in indices:
            if not 0 <= i < len(data):
                raise IndexError(f"{path} has {len(data)} questions, index {i} is out of range")
            if data[i].get("length") != "short":
                raise ValueError(f"longbench[{i}] is a {data[i].get('length')!r} question, not a 'short' one")
        pool, count = list(indices), len(indices)
    logger.info(
        f"{path.name}: {len(data)} questions; drawing {count} of {len(pool)} "
        + (f"short ones with seed {seed}" if indices is None else f"given: {indices}")
    )
    prompts = []
    for i in pool:
        entry = data[i]
        ids = _tokenize_chat(tokenizer, _longbench_prompt(entry))
        if len(ids) > max_prompt_tokens:
            logger.info(f"longbench[{i}] skipped: {len(ids)} prompt tokens > {max_prompt_tokens}")
            continue
        answer = str(entry["answer"]).strip()
        logger.info(
            f"longbench[{i}] {entry['_id']} ({entry['domain']} / {entry['sub_domain']}, {entry['length']}, "
            f"{len(ids)} tokens, answer {answer}): {entry['question']}"
        )
        prompts.append(_Prompt(f"longbench[{i}]", ids, answer))
        if len(prompts) == count:
            break
    if not prompts:
        raise ValueError(f"no question of {path} fits {max_prompt_tokens} prompt tokens")
    if len(prompts) < count:
        logger.warning(f"only {len(prompts)} of {count} questions fit {max_prompt_tokens} prompt tokens")
    return prompts


def _extract_choice(generated: str) -> str | None:
    """The letter the model committed to, using the LongBench-v2 answer patterns first."""
    text = generated.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


# --------------------------------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------------------------------- #
def _compare_logits(tokenizer, prefill_row: torch.Tensor, decode_row: torch.Tensor) -> None:
    """Log how far prefill's next-token logits (position T-1) are from decode-only's at the same position."""
    a, b = prefill_row.reshape(-1), decode_row.reshape(-1)
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    logger.info(
        f"logits compare: PCC {pcc:.5f}, max|diff| {(a - b).abs().max():.3f}, mean|diff| {(a - b).abs().mean():.4f}"
    )
    for name, row in (("prefill", a), ("decode ", b)):
        top = row.topk(5)
        logger.info(
            f"  {name} top-5: "
            + ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
        )
    ta, tb = set(a.topk(10).indices.tolist()), set(b.topk(10).indices.tolist())
    logger.info(f"  top-10 overlap {len(ta & tb)}/10; argmax match: {int(a.argmax()) == int(b.argmax())}")
    ia, ib = int(a.argmax()), int(b.argmax())
    logger.info(
        f"  prefill's gap top1-top2 {float(a.topk(2).values.diff().abs()):.3f}; decode's {float(b.topk(2).values.diff().abs()):.3f}; "
        f"prefill logit of decode's argmax: {a[ib]:.2f} vs its own max {a[ia]:.2f}"
    )


def _eos_ids(config) -> set[int]:
    eos = config.eos_token_id
    return {int(eos)} if isinstance(eos, int) else {int(e) for e in (eos or [])}


def _park_states(states: list[PrefillAttentionState]) -> list[dict]:
    """Copy a prefill's per-layer states to the host, with where each tensor lives, so the device can be reused."""
    parked = []
    for state in states:
        entry: dict = {"seq_len": state.seq_len}
        for name in _STATE_TENSORS:
            tensor = getattr(state, name)
            if tensor is not None:
                entry[name] = (ttnn.from_device(tensor), tensor.memory_config(), tensor.device())
        parked.append(entry)
    return parked


def _upload_states(parked: list[dict]) -> list[PrefillAttentionState]:
    """:func:`_park_states` undone: the states back on the submeshes they came from (the decode layers' own)."""
    states = []
    for entry in parked:
        state = PrefillAttentionState(seq_len=entry["seq_len"])
        for name in _STATE_TENSORS:
            if name in entry:
                host, memory_config, device = entry[name]
                setattr(state, name, ttnn.to_device(host, device, memory_config=memory_config))
        states.append(state)
    return states


def _free_states(states: list[PrefillAttentionState]) -> None:
    """Deallocate the uploaded states once committed, before decode replays any trace over the device again."""
    for state in states:
        for name in _STATE_TENSORS:
            tensor = getattr(state, name)
            if tensor is not None:
                ttnn.deallocate(tensor)
                setattr(state, name, None)


def _shards(host: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(host)]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.equal(a, b) else float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _check_traced_against_eager(prefill, prompt, result, traced_row, traced_parked, chunk_size, tokenizer) -> None:
    """``DEEPSEEK_V4_E2E_TRACE_CHECK=1``: prefill the first prompt again eagerly (the traces are released by now, so
    it may allocate) and log, per layer and state tensor, how far the traced run's result is from the eager one."""
    aligned = result.prefilled
    traced = prefill._traced
    last_start = (aligned - 1) // chunk_size * chunk_size
    sent = traced._packet(torch.tensor(prompt.ids[last_start:aligned]), last_start).reshape(-1)
    for stage in traced.stages:
        for rank, shard in enumerate(_shards(ttnn.from_device(stage.pkt))):
            got = shard.reshape(-1).to(torch.int64)
            wrong = (got != sent.to(torch.int64)).nonzero().flatten()
            logger.info(
                f"[trace check] stage {stage.index} rank {rank} packet: {wrong.numel()}/{sent.numel()} int32 slots "
                "differ" + (f", slots {int(wrong[0])}..{int(wrong[-1])}" if wrong.numel() else "")
            )
    if aligned == chunk_size:  # one chunk: stage k's received streams must be the eager output of its first layer - 1
        outs = {}

        def on_layer(li, out, dev):
            outs[li] = _shards(ttnn.from_device(out))[0].float()

        ids = torch.tensor(prompt.ids[:aligned], dtype=torch.long).unsqueeze(0)
        ttnn.deallocate(prefill._stack(prefill._host_ids(ids), prefill.new_state(), on_layer=on_layer))
        for stage in traced.stages[1:]:
            before = stage.layers[0] - 1
            got = _shards(ttnn.from_device(stage.io.streams_in))[0].float().reshape(outs[before].shape)
            rows = (got - outs[before]).abs().reshape(aligned, -1).amax(dim=-1)
            bad = (rows > 0.05 * outs[before].abs().max()).nonzero().flatten()
            logger.info(
                f"[trace check] stage {stage.index} streams_in vs eager layer {before} out: PCC "
                f"{_pcc(got, outs[before]):.5f}, {bad.numel()}/{aligned} token rows off (first {bad[:6].tolist()}, "
                f"last {bad[-3:].tolist()}); nonfinite {int((~torch.isfinite(got)).sum())}"
            )
    logger.info(f"[trace check] eager prefill of {prompt.name}'s {aligned} tokens, to compare with the traced run")
    ids = torch.tensor(prompt.ids[:aligned], dtype=torch.long).unsqueeze(0)
    logits, states = prefill.prefill(ids, chunk_size=chunk_size)
    eager_row = prefill.to_host(logits, prefill.head_device).reshape(-1).float()
    eager_parked = _park_states(states)
    del states, logits
    gc.collect()
    _log_top5(tokenizer, "[trace check] eager  next token", eager_row)
    _log_top5(tokenizer, "[trace check] traced next token", traced_row.float())
    logger.info(f"[trace check] logits PCC {_pcc(traced_row.float(), eager_row):.5f}")
    for li, (t_entry, e_entry) in enumerate(zip(traced_parked, eager_parked)):
        parts = []
        for name in _STATE_TENSORS:
            if (name in t_entry) != (name in e_entry):
                parts.append(f"{name}: traced {'has' if name in t_entry else 'lacks'} it, eager does not")
                continue
            if name not in t_entry:
                continue
            ts, es = _shards(t_entry[name][0]), _shards(e_entry[name][0])
            if [t.shape for t in ts] != [e.shape for e in es]:
                parts.append(f"{name}: shape {tuple(ts[0].shape)} vs {tuple(es[0].shape)}")
                continue
            pccs = [_pcc(t, e) for t, e in zip(ts, es)]
            diff = max(float((t - e).abs().max()) for t, e in zip(ts, es))
            zero = all(float(t.abs().max()) == 0 for t in ts)
            parts.append(
                f"{name} {min(pccs):.4f}{' ZERO' if zero else ''} (max|d| {diff:.3g}; |traced| max "
                f"{max(float(t.abs().max()) for t in ts):.3g}, |eager| max {max(float(e.abs().max()) for e in es):.3g})"
            )
            if min(pccs) < 0.99:
                logger.info(
                    f"[trace check] layer {li:2d} {name} per rank: PCC {[round(p, 4) for p in pccs]}, "
                    f"|traced| max {[float(t.abs().max()) for t in ts]}"
                )
            if min(pccs) < 0.99 and ts[0].dim() >= 2:
                t, e = ts[0].reshape(-1, ts[0].shape[-1]), es[0].reshape(-1, es[0].shape[-1])
                bad = ((t - e).abs().amax(dim=-1) > 0.5 * e.abs().amax(dim=-1).clamp(min=1e-3)).nonzero().flatten()
                logger.info(
                    f"[trace check] layer {li:2d} {name}: {bad.numel()}/{t.shape[0]} rows off; "
                    f"first bad rows {bad[:12].tolist()}, last {bad[-4:].tolist()}"
                )
        logger.info(f"[trace check] layer {li:2d}: " + ", ".join(parts))


def _log_top5(tokenizer, what: str, row: torch.Tensor) -> None:
    top = row.topk(5)
    logger.info(
        f"{what} (top 5): "
        + ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
    )


# --------------------------------------------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------------------------------------------- #
def _run(
    mesh_device,
    progress: _Progress,
    prefetcher: contextlib.ExitStack,
    AutoTokenizer,
    make_prompts: Callable[..., list[_Prompt]],
    *,
    lightning_indexer: bool | None = None,
) -> list[_Result]:
    max_new = _env_int("DEEPSEEK_V4_MAX_NEW_TOKENS", 128)
    chunk_size = _env_int("DEEPSEEK_V4_E2E_CHUNK", 1024)
    traced = os.environ.get("DEEPSEEK_V4_PREFILL_TRACED", "1") == "1"
    compare = os.environ.get("DEEPSEEK_V4_E2E_COMPARE", "0") == "1"
    if chunk_size <= 0 or chunk_size % ALIGNMENT:
        raise ValueError(f"DEEPSEEK_V4_E2E_CHUNK={chunk_size} must be a positive multiple of {ALIGNMENT}")

    # --- tokenizer, config, prompts ----------------------------------------------------------------- #
    progress.step("[1/5] tokenizer, config and prompts")
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id
    if isinstance(pad_id, list):
        pad_id = pad_id[0]
    prompts = make_prompts(tokenizer, max_new)
    max_input = _env_int("DEEPSEEK_V4_E2E_MAX_INPUT", 0)
    if max_input > 0:
        for prompt in prompts:
            if len(prompt.ids) > max_input:
                logger.info(f"{prompt.name}: truncated to its first {max_input} of {len(prompt.ids)} tokens")
                prompt.ids = prompt.ids[:max_input]
    if "DEEPSEEK_V4_PREFILL_INDEXER" in os.environ:
        indexer_on = os.environ["DEEPSEEK_V4_PREFILL_INDEXER"] == "1"
    else:
        indexer_on = bool(lightning_indexer)
    results = []
    for prompt in prompts:
        real_len = len(prompt.ids)
        if not indexer_on and real_len + max_new >= _MAX_CONTEXT:
            raise ValueError(
                f"{prompt.name}: prompt ({real_len}) + max new tokens ({max_new}) must stay below {_MAX_CONTEXT}: the "
                "state commit does not carry the CSA indexer's key cache (lower DEEPSEEK_V4_E2E_PROMPT_LEN / "
                "DEEPSEEK_V4_MAX_NEW_TOKENS)"
            )
        aligned = real_len // ALIGNMENT * ALIGNMENT
        results.append(_Result(prompt.name, real_len, aligned, expected=prompt.expected))
        logger.info(
            f"{prompt.name}: {real_len} tokens = {aligned} prefilled ({math.ceil(aligned / chunk_size)} chunk(s) of up "
            f"to {chunk_size}) + {real_len - aligned} replayed through decode; up to {max_new} new tokens"
        )
        logger.info(f"  starts : {tokenizer.decode(prompt.ids[:24])!r}")
        logger.info(f"  ends   : {tokenizer.decode(prompt.ids[-80:])!r}")

    # --- the 8 x TP4 stages (all 32 chips), shared by the prefill model and then the decode model ------- #
    system_config = load_system_config(mesh_device=mesh_device)
    set_active_system_config(system_config)
    num_layers = min(
        int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", config.num_hidden_layers)), config.num_hidden_layers
    )
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)) for i in range(_NUM_STAGES)
    ]
    placement = plan_layer_placement(num_layers, _NUM_STAGES, max(0, system_config.pipeline.group_size))
    needed = max(r.tokens for r in results) + max_new  # decode is sized once, for the longest conversation
    max_seq = round_context(_traced_max_seq(config, needed), set(config.compress_rates.values()), _PAGE_BLOCK_SIZE)
    rope = _build_rope(config, max_seq)

    # --- phase 1: the prefill model, loaded once; every prompt prefilled, its states parked on the host ---- #
    # The traces are captured once, for the longest prompt, on devices that hold nothing else of decode; every
    # prompt replays them, and they are released before decode allocates anything.
    rows: list[Optional[torch.Tensor]] = [None] * len(prompts)
    parked: list[Optional[list[dict]]] = [None] * len(prompts)
    bias_slots = None
    prepare_seconds = 0.0
    if any(r.prefilled for r in results):
        progress.step(f"[2/5] prefill model on {_NUM_STAGES} x TP{_TP_SIZE} stages (its own routed experts)")
        cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
        t0 = time.perf_counter()
        layer_devices = [submeshes[k] for k in placement]
        prefill = DeepSeekV4PrefillModel(
            config,
            checkpoint_weights(loader, config, num_layers),
            layer_devices[0],
            rope,
            expert_provider=checkpoint_expert_provider(loader),
            num_layers=num_layers,
            cache=cache,
            weight_dtype=_ATTENTION_WEIGHT_DTYPE,
            expert_dtype=system_config.decode.ttnn_weight_dtype,
            tp_size=_TP_SIZE,
            layer_devices=layer_devices,
            dense_csa=True,
            # CSA lightning indexer. Traced prefill runs it too.
            lightning_indexer=indexer_on,
            progress=progress,
        )
        prefill.synchronize("uploads")
        build_seconds = time.perf_counter() - t0
        logger.info(f"prefill model built in {build_seconds:.1f}s")
        num_stages = len(dict.fromkeys(placement))
        if traced:
            max_len = max(r.prefilled for r in results)
            progress.step(
                f"[3/5] traced prefill plan for prompts of up to {max_len} tokens: persistent buffers, H2D / D2D / "
                f"D2H sockets, compile the {chunk_size}-token chunk step, capture one trace per stage (once; not timed)"
            )
            t0 = time.perf_counter()
            prefill.prepare_traced_prefill(max_len, chunk_size)
            prepare_seconds = time.perf_counter() - t0
            logger.info(f"traced prefill prepared in {prepare_seconds:.1f}s")
        for k, (prompt, result) in enumerate(zip(prompts, results)):
            if result.prefilled:
                tag = f"[3/5] prefill {k + 1}/{len(prompts)} {prompt.name}"
                rows[k], parked[k] = _prefill_one(
                    prefill,
                    prompt,
                    result,
                    tag,
                    progress,
                    tokenizer,
                    traced,
                    chunk_size,
                    num_stages,
                    build_seconds,
                    prepare_seconds,
                )
        # Keep only the parked states; every prefill tensor goes before decode allocates.
        prefill.release_traced_prefill()  # a no-op when eager
        if traced and os.environ.get("DEEPSEEK_V4_E2E_TRACE_CHECK", "0") == "1" and results[0].prefilled:
            _check_traced_against_eager(prefill, prompts[0], results[0], rows[0], parked[0], chunk_size, tokenizer)
            pytest.skip("DEEPSEEK_V4_E2E_TRACE_CHECK=1: traced vs eager prefill compared; decode not run")
        bias_slots = prefill_bias_slots(prefill)  # the one thing the commit needs of the prefill model
        prefill.synchronize("before release")
        del prefill
        gc.collect()

    # --- phase 2: the decode model on the same stages, loaded once; static state, session, trace capture -- #
    progress.step(f"[4/5] decode model ({_NUM_STAGES} x TP{_TP_SIZE} stages) - a warm weight cache just uploads")
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
    )
    _assert_decode_parallelism(decode, _TP_SIZE, _NUM_STAGES)
    assert decode.num_layers == num_layers, f"decode built {decode.num_layers} layers, prefill {num_layers}"
    assert list(decode.layer_submesh_ids) == list(placement), "decode and prefill placed the layers differently"
    logger.info(f"decode model: {num_layers} layers built in {time.perf_counter() - t0:.1f}s")
    decode.prepare_static_decode(
        rope, max_seq, lm_head=lm_head, num_sessions=1, total_tokens=max_seq, block_size=_PAGE_BLOCK_SIZE
    )
    if decode.context_limit is not None and decode.context_limit < needed:
        raise ValueError(f"decode context capped at {decode.context_limit} < the {needed} tokens needed")
    sid = decode.open_session()
    decode.activate_session(sid)
    # A throw-away step captures the traces; its scratch cache writes are overwritten by every prompt's commit.
    progress.step("[4/5] throw-away decode step: captures the decode traces")
    t0 = time.perf_counter()
    decode.decode_traced(pad_id, 0)
    logger.info(f"trace capture + first step: {time.perf_counter() - t0:.1f}s")

    # --- every prompt through decode, one after another in the one session ---------------------------- #
    eos = _eos_ids(config)
    for k, (prompt, result) in enumerate(zip(prompts, results)):
        tag = f"[5/5] decode {k + 1}/{len(prompts)} {prompt.name}"
        _decode_one(
            decode, sid, prompt, result, rows[k], parked[k], bias_slots, tag, progress, tokenizer, eos, max_new, compare
        )
        parked[k] = rows[k] = None
        assert all(0 <= t < config.vocab_size for t in result.generated)
    logger.info(f"pool usage: {decode.session_usage()}")
    _summary(results, traced, prepare_seconds)
    progress.step("done")
    return results


def _prefill_one(
    prefill: DeepSeekV4PrefillModel,
    prompt: _Prompt,
    result: _Result,
    tag: str,
    progress: _Progress,
    tokenizer,
    traced: bool,
    chunk_size: int,
    num_stages: int,
    build_seconds: float,
    prepare_seconds: float,
) -> tuple[torch.Tensor, list[dict]]:
    """Prefill one prompt's aligned part; returns its next-token logits row (host) and its states, parked on host."""
    aligned = result.prefilled
    num_chunks = math.ceil(aligned / chunk_size)
    how = f"through {num_stages} pipelined stage(s), traced" if traced else "eager, includes compilation"
    progress.step(f"{tag}: prefill of {aligned} tokens, {num_chunks} chunk(s) of up to {chunk_size} ({how}; timed)")
    ids = torch.tensor(prompt.ids[:aligned], dtype=torch.long).unsqueeze(0)
    chunk_times: list[tuple[int, int, float]] = []

    def on_chunk(index: int, start: int, end: int, seconds: float) -> None:
        chunk_times.append((start, end, seconds))
        tokens = end - start
        # Traced, ``seconds`` is the time since the previous chunk's logits arrived (the first: the pipeline fill).
        fill = " (pipeline fill)" if traced and index == 0 and num_chunks > 1 else ""
        logger.info(
            f"prefill chunk {index + 1}/{num_chunks} [{start}, {end}): {tokens} tokens, +{seconds:.3f}s{fill}, "
            f"{tokens / seconds:.1f} tok/s"
        )

    t0 = time.perf_counter()
    if traced:
        logits, states = prefill.prefill_traced(ids, on_chunk=on_chunk)
    else:
        logits, states = prefill.prefill(ids, chunk_size=chunk_size, on_chunk=on_chunk)
    result.prefill_seconds = time.perf_counter() - t0
    if traced:
        _report(chunk_times, num_stages, result.prefill_seconds, build_seconds, prepare_seconds)
        steady = chunk_times[1:] or chunk_times
        result.pipelined_tps = sum(e - s for s, e, _ in steady) / sum(sec for _, _, sec in steady)
    else:
        logger.info(
            f"prefill: {aligned} tokens in {result.prefill_seconds:.2f}s "
            f"({result.prefill_seconds / aligned * 1000:.2f} ms/token, {aligned / result.prefill_seconds:.1f} tok/s; "
            "eager, includes first-run compilation)"
        )
    # The traced prefill's logits come back on the host (D2H socket); the eager one's are on the last stage.
    row = logits.reshape(-1) if traced else prefill.to_host(logits, prefill.head_device).reshape(-1)
    assert torch.isfinite(row).all(), f"{prompt.name}: non-finite prefill logits"
    _log_top5(tokenizer, f"prefill's own next token after position {aligned - 1}", row)

    # The traced states live in the persistent buffers, which the next prompt's run overwrites.
    progress.step(f"{tag}: parking the per-layer states on the host")
    t0 = time.perf_counter()
    parked = _park_states(states)
    if traced:
        prefill.free_traced_states(states)  # allocated under the live traces: gone before the next replay
    del states, logits
    gc.collect()
    logger.info(f"states parked in {time.perf_counter() - t0:.1f}s")
    return row, parked


def _decode_one(
    decode,
    sid: int,
    prompt: _Prompt,
    result: _Result,
    row: Optional[torch.Tensor],
    parked: Optional[list[dict]],
    bias_slots,
    tag: str,
    progress: _Progress,
    tokenizer,
    eos: set[int],
    max_new: int,
    compare: bool,
) -> None:
    """Rewind the session, commit one prompt's prefill state, replay its tail and generate its answer."""
    aligned, real_len = result.prefilled, result.tokens
    # The previous prompt's (or the capture step's) state must not leak into this one.
    decode.reset_session(sid)
    decode.reset_static_caches()
    next_id = None
    if aligned:
        if compare:  # decode-only logits at the last aligned position, to compare with prefill's (numerics check)
            progress.step(f"{tag}: reference: {aligned} prompt tokens through decode only (DEEPSEEK_V4_E2E_COMPARE=1)")
            t0 = time.perf_counter()

            def alive(i: int, _out) -> None:
                if (i + 1) % 512 == 0:
                    rate = (i + 1) / (time.perf_counter() - t0)
                    logger.info(f"{tag}: reference {i + 1}/{aligned} ({rate:.1f} tok/s)")

            out = decode.decode_prompt_traced(prompt.ids[:aligned], 0, on_output=alive)
            _compare_logits(tokenizer, row.float(), out.reshape(-1).float()[: row.numel()])
            decode.reset_session(sid)  # back to the state the commit expects
            decode.reset_static_caches()

        # The states go up, are committed (device side: prefill and decode share every stage) and are freed
        # before the next decode_traced, so no allocation outlives the commit under the live decode traces.
        progress.step(f"{tag}: commit: prefill state -> decode buffers, on device")
        t0 = time.perf_counter()
        states = _upload_states(parked)
        committed = decode.commit_prefill_state(
            states, sid, progress=lambda m: progress(m, important=False), bias_slots=bias_slots
        )
        _free_states(states)
        del states
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
    for step in range(1, max_new):
        if generated[-1] in eos:
            logger.info("hit EOS; stopping")
            break
        t0 = time.perf_counter()
        logits = decode.decode_traced(generated[-1], real_len + step - 1).reshape(1, -1).float()
        generated.append(int(logits[0].argmax()))
        step_times.append(time.perf_counter() - t0)
        print(tokenizer.decode([generated[-1]]), end="", flush=True)
        if step % 32 == 0:
            recent = step_times[-32:]
            logger.info(f"decode {step}/{max_new}: {len(recent) / sum(recent):.2f} tok/s (last {len(recent)} tokens)")
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


def _summary(results: list[_Result], traced: bool, prepare_seconds: float) -> None:
    """One line per prompt: its prefill and decode perf (and answer), then the totals."""
    mode = "traced, pipelined" if traced else "eager"
    lines = [
        "",
        f"=== end-to-end perf ({len(results)} prompt(s); prefill {mode}) ===",
        f"{'prompt':<18} {'tokens':>7} {'prefill s':>10} {'tok/s':>8} {'piped t/s':>10} "
        f"{'decode t/s':>10} {'gen':>4} {'answer':>9}",
    ]
    for r in results:
        tps = f"{r.prefilled / r.prefill_seconds:.1f}" if r.prefill_seconds else "-"
        piped = f"{r.pipelined_tps:.1f}" if r.pipelined_tps else "-"
        dec = f"{r.decode_tps:.2f}" if r.decode_tps else "-"
        answer = "" if r.expected is None else f"{r.choice or '?'}/{r.expected}{'' if r.choice == r.expected else ' x'}"
        lines.append(
            f"{r.name:<18} {r.tokens:>7} {r.prefill_seconds:>10.2f} {tps:>8} {piped:>10} "
            f"{dec:>10} {len(r.generated):>4} {answer:>9}"
        )
    prefilled = sum(r.prefilled for r in results)
    prefill_seconds = sum(r.prefill_seconds for r in results)
    if prefill_seconds:
        lines.append(
            f"prefill total          : {prefilled} tokens in {prefill_seconds:.2f}s = "
            f"{prefilled / prefill_seconds:.1f} tok/s (plus {prepare_seconds:.1f}s of untimed trace preparation, "
            "once for all prompts)"
        )
    decode_rates = [r.decode_tps for r in results if r.decode_tps]
    if decode_rates:
        lines.append(f"decode, mean           : {sum(decode_rates) / len(decode_rates):.2f} tok/s steady")
    graded = [r for r in results if r.expected is not None]
    if graded:
        correct = sum(r.choice == r.expected for r in graded)
        lines.append(f"answers                : {correct}/{len(graded)} correct")
    logger.info("\n".join(lines))
