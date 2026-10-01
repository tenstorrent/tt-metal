# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Long-prompt prefill demo on the full ttnn ``DeepSeekV4PrefillModel`` (the prefill twin of
``tests/decode/test_full_model_decode_demo.py``).

Opens the full 8 x 4 Galaxy mesh, carves ``1 x 4`` tensor-parallel submeshes out of it (rows 0..N-1 of
column 0) and builds the real 43-layer checkpoint on them as N pipeline stages, eight by default (contiguous
layer spans, 5-6 layers a stage, the embedding with the first stage and the head with the last), tokenizes a
*real* prompt -- by default the 128K-token long-context prompt of
``models/tt_transformers/demo/sample_prompts/input_data_long_128k.json`` -- and prefills it chunk by chunk.

The prefill is *traced and pipelined* (:class:`~models.experimental.deepseek_v4_flash.tt.model.TracedPrefill`),
the way the decode demo runs decode:

* every stage's chunk step is captured once as a trace, and a replay thread posts the ``execute_trace`` of
  every stage for chunks well ahead of the host, as the decode demo posts its replays ahead;
* the host's only input per chunk is one packet (token ids and positions) pushed into an H2D socket on stage 0,
  the stages hand the residual streams on over device-to-device sockets, and the last stage streams each
  chunk's last-token logits back over a D2H socket;
* so the stages overlap: with N stages, N consecutive chunks are in flight at once, one in each stage, and
  once the pipeline is full a chunk completes every ``slowest stage`` seconds rather than every
  ``sum of the stages`` seconds.

It reports, per chunk, the time between consecutive chunks' logits arriving (the first includes filling the
pipeline), and the whole prefill, then the first token the model would generate.

The prompt file holds an instruction and the URL of a book (Frankenstein, Project Gutenberg). As in the
tt_transformers demos, the prompt is the book in a markdown fence followed by the instruction, wrapped in
the V4 chat template. The book is ~112K tokens, so to reach a prompt of exactly ``DEEPSEEK_V4_PREFILL_LEN``
tokens (131072 = 128K by default) the book's opening is repeated after its end; a longer target is
never truncated below the instruction, which always closes the prompt.

Caveats to keep in mind when reading the output:

* CSA layers run *without* the lightning indexer (``dense_csa``): each query attends to every visible
  compressed entry instead of the indexer's top ``index_topk``. Timings are those of this dense
  implementation (more work than the sparse one will be), and the generated token is not the model's exact
  answer beyond ``index_topk * 4 = 2048`` tokens of context.
* The prefill state is not handed to the decode path here: this measures prefill and stops at the first
  token.
* Compilation is off the clock: preparing the traced prefill compiles the chunk step and captures one trace per
  stage before the timed run, which is replay only. The traces serve any prompt up to the prepared length. Each
  layer attends over its whole compressed-KV buffer (sized for that length, the rows not yet filled masked), not
  the exact entry count, so early chunks do more attention work than the eager prefill's; the last chunk is
  padded to the chunk size.

Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v4_flash/tests/prefill/test_full_model_prefill_demo.py

Knobs (environment):

* ``DEEPSEEK_V4_PREFILL_LEN``     -- prompt length in tokens, a multiple of 128 (default 131072),
* ``DEEPSEEK_V4_PREFILL_CHUNK``   -- chunk size in tokens, a multiple of 128 (default 1024),
* ``DEEPSEEK_V4_PREFILL_LAYERS``  -- build only the first N layers, for bring-up (default all 43),
* ``DEEPSEEK_V4_PREFILL_STAGES``  -- number of ``1 x 4`` pipeline stages, 1 to 8 (default 8; that many chunks
  run at once, so more stages mean more overlap; capped at the layer count),
* ``DEEPSEEK_V4_PREFILL_HEARTBEAT`` -- seconds between ``[heartbeat]`` lines saying what it is doing (default 30),
* ``DEEPSEEK_V4_PREFILL_STALL_SECS`` -- warn when one step lasts longer than this (default 600),
* ``DEEPSEEK_V4_PREFILL_PROMPT``  -- another prompt file (a JSON list; the first entry is used),
* ``DEEPSEEK_V4_TRACE_REGION_SIZE`` -- bytes to reserve for the captured traces (unset keeps the ttnn default; set
  it, e.g. 500000000, if the capture reports the trace region too small),
* ``DEEPSEEK_V4_CACHE_DIR``       -- the converted-weight tile cache (shared with the decode demo's routed
  experts).

A quick bring-up: ``DEEPSEEK_V4_PREFILL_LAYERS=8 DEEPSEEK_V4_PREFILL_LEN=4096`` (one layer per stage).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.encoding_dsv4 import encode_messages
from models.experimental.deepseek_v4_flash.tests.decode.test_full_model_decode_demo import (
    _CACHE_DIR,
    _DEFAULT_MODEL_DIR,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.model import DeepSeekV4PrefillModel
from models.experimental.deepseek_v4_flash.tt.prefill.weights import checkpoint_expert_provider, checkpoint_weights
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_TP_SIZE = 4
_MODELS_DIR = Path(__file__).resolve().parents[4]
_DEFAULT_PROMPT_FILE = _MODELS_DIR / "tt_transformers/demo/sample_prompts/input_data_long_128k.json"
_CONTEXT_CACHE = _MODELS_DIR / "tt_transformers/demo/context_cache"
_ATTENTION_WEIGHT_DTYPE = ttnn.bfloat8_b
_SPLIT_SENTINEL = "@@USER-CONTENT@@"


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _load_context(url: str) -> str:
    """The text behind ``url``, cached on disk under the same key the tt_transformers demos use."""
    _CONTEXT_CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = _CONTEXT_CACHE / hashlib.md5(url.encode()).hexdigest()
    if cache_file.exists():
        return cache_file.read_text()
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    cache_file.write_text(response.text)
    logger.info(f"downloaded and cached context: {url}")
    return response.text


def _build_prompt_ids(tokenizer, entry: dict, target_len: int) -> tuple[list[int], dict]:
    """A chat-templated prompt of exactly ``target_len`` tokens from a prompt-file entry.

    ``entry`` holds ``prompt`` (the instruction) and optionally ``context`` (a URL). The user turn is
    ``"```" + context + "```\\n\\n" + prompt`` like the tt_transformers demos. The context is trimmed, or
    repeated from its start, so the whole prompt is ``target_len`` tokens; the instruction and the chat
    template's tail are always kept.
    """
    chat = encode_messages([{"role": "user", "content": _SPLIT_SENTINEL}], "chat")
    prefix, suffix = chat.split(_SPLIT_SENTINEL)
    encode = lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"])  # noqa: E731

    if "context" not in entry:
        ids = encode(prefix + entry["prompt"] + suffix)
        return ids, {"prefix": len(ids), "context": 0, "suffix": 0, "book": 0, "repeated": 0}

    head = encode(prefix + "```")
    tail = encode("```\n\n" + entry["prompt"] + suffix)
    book = encode(_load_context(entry["context"]))
    budget = target_len - len(head) - len(tail)
    if budget <= 0:
        raise ValueError(f"target length {target_len} leaves no room for the prompt around the context")
    context = (book * math.ceil(budget / len(book)))[:budget]
    info = {
        "prefix": len(head),
        "suffix": len(tail),
        "book": len(book),
        "context": budget,
        "repeated": max(0, budget - len(book)),
    }
    return head + context + tail, info


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _report(
    chunk_times: list[tuple[int, int, float]],
    num_stages: int,
    wall_seconds: float,
    build_seconds: float,
    prepare_seconds: float,
) -> None:
    """The per-chunk table and the whole-prefill summary (all via the logger, so ``-s`` shows them).

    ``chunk_times`` holds ``(start, end, interval)``: the time between this chunk's logits and the previous
    chunk's arriving. The first interval is the pipeline fill (the first chunk through every stage); after it, an
    interval is the pipelined time per chunk, set by the slowest stage.
    """
    total_tokens = chunk_times[-1][1]
    fill_seconds = chunk_times[0][2]
    steady = chunk_times[1:] or chunk_times
    per_chunk_ms = [seconds / (end - start) * 1000 for start, end, seconds in steady]

    lines = ["", f"{'chunk':>6} {'tokens':>15} {'interval (s)':>12} {'ms/token':>10} {'tok/s':>10} {'cum (s)':>10}"]
    cumulative = 0.0
    for i, (start, end, seconds) in enumerate(chunk_times):
        cumulative += seconds
        note = "  (pipeline fill)" if i == 0 and len(chunk_times) > 1 else ""
        lines.append(
            f"{i + 1:>6} {f'{start}-{end}':>15} {seconds:>12.3f} {seconds / (end - start) * 1000:>10.3f} "
            f"{(end - start) / seconds:>10.1f} {cumulative:>10.2f}{note}"
        )
    logger.info("\n".join(lines))

    seconds_sorted = sorted(seconds for _, _, seconds in steady)
    steady_tokens = sum(end - start for start, end, _ in steady)
    steady_seconds = sum(seconds for _, _, seconds in steady)
    quarter = max(1, len(steady) // 4)
    first_q = statistics.mean(per_chunk_ms[:quarter])
    last_q = statistics.mean(per_chunk_ms[-quarter:])
    logger.info(
        "\n".join(
            [
                "",
                "=== prefill summary (traced, pipelined) ===",
                f"prompt tokens          : {total_tokens}",
                f"chunks                 : {len(chunk_times)} of up to "
                f"{max(end - start for start, end, _ in chunk_times)} tokens "
                f"through {num_stages} stage(s), up to {num_stages} in flight",
                f"total prefill time     : {wall_seconds:.2f} s ({wall_seconds / 60:.1f} min)",
                f"average per token      : {wall_seconds / total_tokens * 1000:.3f} ms/token",
                f"throughput             : {total_tokens / wall_seconds:.1f} tokens/s",
                f"pipeline fill          : {fill_seconds:.3f} s (first chunk's logits)",
                f"pipelined throughput   : {steady_tokens / steady_seconds:.1f} tokens/s (after the fill)",
                f"chunk interval (s)     : min {seconds_sorted[0]:.3f} | "
                f"median {statistics.median(seconds_sorted):.3f} | p90 {_percentile(seconds_sorted, 0.9):.3f} | "
                f"max {seconds_sorted[-1]:.3f} | last {chunk_times[-1][2]:.3f}",
                f"ms/token, first quarter: {first_q:.3f}  ->  last quarter: {last_q:.3f} "
                f"(x{last_q / first_q:.2f} as the context grows)",
                f"model build time       : {build_seconds:.1f} s",
                f"trace prepare time     : {prepare_seconds:.1f} s (compile every variant + capture; not counted)",
            ]
        )
    )


class _Progress:
    """What the run is doing right now, reported so a slow step can be told from a hung one.

    The model calls this with a message before each fine-grained step (see ``progress`` in
    :class:`DeepSeekV4PrefillModel`); it keeps the latest message and, in a background thread, logs a
    *heartbeat* every ``interval`` seconds: total elapsed time, how long the current phase has lasted, host
    memory, and the phase itself. A healthy run shows the phase changing; a hang shows one phase with a
    growing timer, and after ``stall`` seconds in the same phase a warning says so. Per-layer messages are
    logged only while ``verbose`` (e.g. the build and the compile runs); ``important`` ones (build steps,
    chunk starts, trace captures) always are.
    """

    def __init__(self, interval: float, stall: float):
        self.interval = interval
        self.stall = stall
        self.verbose = True
        self.phase = "starting"
        self.started = self.phase_started = time.perf_counter()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name="prefill-heartbeat", daemon=True)

    def __call__(self, message: str, important: bool = False) -> None:
        self.phase = message
        self.phase_started = time.perf_counter()
        if important or self.verbose:
            logger.info(f"[+{self.phase_started - self.started:7.1f}s] {message}")

    def step(self, message: str) -> None:
        """A numbered top-level step of the demo (always logged)."""
        self(message, important=True)

    @staticmethod
    def _rss_gb() -> float:
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        except (OSError, ValueError):
            return float("nan")

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            now = time.perf_counter()
            in_phase = now - self.phase_started
            logger.info(
                f"[heartbeat] alive: {now - self.started:.0f}s since start | {in_phase:.0f}s in this step | "
                f"host RSS {self._rss_gb():.1f} GB | now: {self.phase}"
            )
            if in_phase > self.stall:
                logger.warning(
                    f"[heartbeat] NO PROGRESS for {in_phase:.0f}s (> {self.stall:.0f}s) in: {self.phase}. "
                    "It may be hung: check the device with tt-triage (the attached process is not killed by it) "
                    "before restarting."
                )

    def __enter__(self) -> "_Progress":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)  # weight conversion on a cold cache + a 128K-token prefill
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            # Like the decode demos: only reserve a trace region when asked (the ttnn default otherwise).
            **(
                {"trace_region_size": int(os.environ["DEEPSEEK_V4_TRACE_REGION_SIZE"])}
                if os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE")
                else {}
            ),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_full_model_prefill_demo(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    with progress:
        _run_demo(mesh_device, progress, AutoTokenizer, DeepseekV4Config)


def _run_demo(mesh_device, progress: _Progress, AutoTokenizer, DeepseekV4Config) -> None:
    target_len = _env_int("DEEPSEEK_V4_PREFILL_LEN", 131072)
    chunk_size = _env_int("DEEPSEEK_V4_PREFILL_CHUNK", 1024)
    for name, value in (("DEEPSEEK_V4_PREFILL_LEN", target_len), ("DEEPSEEK_V4_PREFILL_CHUNK", chunk_size)):
        if value <= 0 or value % ALIGNMENT:
            raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
    logger.info(
        f"settings: prompt {target_len} tokens, chunk {chunk_size}, traced + pipelined, "
        f"heartbeat every {progress.interval:.0f}s, stall warning after {progress.stall:.0f}s, per-layer "
        "messages for the build and the trace compile"
    )

    progress.step("[1/7] reading the checkpoint index, the config and the tokenizer")
    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_PREFILL_LAYERS", config.num_hidden_layers), config.num_hidden_layers)
    logger.info(f"checkpoint {loader.snapshot_dir}: {num_layers}/{config.num_hidden_layers} layers will be built")

    # --- the prompt ------------------------------------------------------------------------------ #
    progress.step(f"[2/7] building the {target_len}-token prompt (downloads the book on first use, tokenizes it)")
    prompt_file = Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE))
    entry = json.loads(prompt_file.read_text())[0]
    prompt_ids, info = _build_prompt_ids(tokenizer, entry, target_len)
    assert len(prompt_ids) % ALIGNMENT == 0, f"prompt of {len(prompt_ids)} tokens is not a multiple of {ALIGNMENT}"
    repeated = f", its first {info['repeated']} repeated to fill" if info["repeated"] else ""
    logger.info(
        f"prompt file {prompt_file.name}: {len(prompt_ids)} tokens = {info['prefix']} template/fence + "
        f"{info['context']} context ({info['book']} in the book{repeated}) + {info['suffix']} instruction"
    )
    logger.info(f"prompt starts : {tokenizer.decode(prompt_ids[:24])!r}")
    logger.info(f"prompt ends   : {tokenizer.decode(prompt_ids[-90:])!r}")
    num_chunks = math.ceil(len(prompt_ids) / chunk_size)
    logger.info(f"prefill plan: {num_chunks} chunks of {chunk_size} tokens")

    # --- the mesh: the full Galaxy is open; each pipeline stage is a 1 x TP submesh of it -------- #
    progress.step("[3/7] system profile and submeshes")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert mesh_cols >= _TP_SIZE, f"a 1 x {_TP_SIZE} submesh does not fit in the {mesh_rows} x {mesh_cols} mesh"
    num_stages = _env_int("DEEPSEEK_V4_PREFILL_STAGES", 8)
    if not 1 <= num_stages <= mesh_rows:
        raise ValueError(f"DEEPSEEK_V4_PREFILL_STAGES={num_stages} must be in [1, {mesh_rows}] (mesh rows)")
    num_stages = min(num_stages, num_layers)  # a stage owns at least one layer
    logger.info(f"opened a {mesh_rows} x {mesh_cols} mesh; {num_stages} stage(s) of 1 x {_TP_SIZE}")
    submeshes = []
    for i in range(num_stages):
        submeshes.append(mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)))
        logger.info(f"created submesh for stage {i}: 1 x {_TP_SIZE} at mesh row {i}, column 0")
    placement = plan_layer_placement(num_layers, num_stages, 1)  # contiguous spans, one per stage
    layer_devices = [submeshes[k] for k in placement]
    spans = []
    for k in dict.fromkeys(placement):
        owned = [li for li, stage in enumerate(placement) if stage == k]
        spans.append(f"stage {k}: layers {owned[0]}-{owned[-1]}")
    logger.info(
        f"{num_layers}/{config.num_hidden_layers} layers over {num_stages} stages x TP{_TP_SIZE}: " + ", ".join(spans)
    )

    progress.step(f"[4/7] RoPE tables for {len(prompt_ids)} positions")
    t0 = time.perf_counter()
    rope = _build_rope(config, len(prompt_ids))
    logger.info(f"RoPE tables built in {time.perf_counter() - t0:.1f}s")

    # --- the model ------------------------------------------------------------------------------- #
    progress.step(
        f"[5/7] building the model: {num_layers} layers (a cold weight cache reads + quantizes the checkpoint: "
        "slow; a warm one just uploads)"
    )
    cache = WeightCache(os.path.join(_CACHE_DIR, os.path.basename(_DEFAULT_MODEL_DIR))) if _CACHE_DIR else None
    logger.info(f"weight cache: {cache.path if cache else 'disabled'}")
    t0 = time.perf_counter()
    model = DeepSeekV4PrefillModel(
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
        progress=progress,
    )
    progress.step("model built; waiting for every device to finish its uploads")
    model.synchronize("uploads")
    build_seconds = time.perf_counter() - t0
    logger.info(f"built DeepSeekV4PrefillModel ({model.num_layers} layers) in {build_seconds:.1f}s")

    ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

    # --- traces: allocate the persistent state and sockets, compile the chunk step, capture ------- #
    # Before anything else allocates on the devices: allocating on a device that holds a trace is unsafe.
    progress.step(
        f"[6/7] preparing the traced prefill for {len(prompt_ids)} tokens: persistent buffers, H2D / D2D / D2H "
        "sockets, compile the chunk step, capture one trace per stage (not timed)"
    )
    t0 = time.perf_counter()
    model.prepare_traced_prefill(len(prompt_ids), chunk_size)
    prepare_seconds = time.perf_counter() - t0
    logger.info(f"traced prefill prepared in {prepare_seconds:.1f}s")

    # --- the measured prefill: trace replays, the stages pipelined -------------------------------- #
    progress.step(
        f"[7/7] prefill: {num_chunks} chunks of {chunk_size} tokens through {num_stages} pipelined stage(s) (timed)"
    )
    chunk_times: list[tuple[int, int, float]] = []
    elapsed = 0.0

    def on_chunk(index: int, start: int, end: int, seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds
        chunk_times.append((start, end, seconds))
        recent = [s for _, _, s in chunk_times[1:][-8:]] or [seconds]
        remaining = (num_chunks - index - 1) * statistics.mean(recent)
        fill = " (pipeline fill)" if index == 0 and num_chunks > 1 else ""
        logger.info(
            f"prefill chunk {index + 1}/{num_chunks} tokens [{start}, {end}) logits in: +{seconds:.3f} s{fill} | "
            f"{seconds / (end - start) * 1000:.3f} ms/token | {(end - start) / seconds:.1f} tok/s | "
            f"{end}/{len(prompt_ids)} tokens, elapsed {elapsed:.1f} s, ~{remaining:.0f} s to go"
        )

    # The replay runs no per-layer Python, so only the chunk lines (and the heartbeat) remain.
    progress.verbose = False
    t0 = time.perf_counter()
    logits, states = model.prefill_traced(ids, on_chunk=on_chunk)
    wall_seconds = time.perf_counter() - t0
    progress.step("prefill finished")
    _report(chunk_times, num_stages, wall_seconds, build_seconds, prepare_seconds)

    # --- the first generated token --------------------------------------------------------------- #
    assert len(chunk_times) == num_chunks
    assert all(state.seq_len == len(prompt_ids) for state in states)
    row = logits.reshape(-1)  # already on the host: the last stage streams it over the D2H socket
    assert row.numel() == config.vocab_size and torch.isfinite(row).all(), "non-finite logits"
    top = row.topk(5)
    logger.info(
        "first generated token (top 5): "
        + ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
    )
    model.release_traced_prefill()
    progress.step("done")
