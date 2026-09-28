# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared scaffolding for the Phase 2 performance baseline.

Three numbers are reported separately and never combined, because each one bounds a different
optimization:

  weight upload plus first forward   compilation and program-cache population, paid once
  steady-state model latency         the backbone on an input already on device
  request latency                    tokenize, upload, backbone, pool, read back

The gap between the second and the third is host work and transfers; the gap between the second
and the device kernel total reported by test_nomic_device_perf.py is dispatch. Collapsing them
into one figure hides which of trace, sharding or precision is worth doing.

Shapes are fixed rather than swept. T = B * S is what the expert bank scales with, and 8x512
sits just past the 3520-token bound in tt/experts.py, so it takes the multi-pass branch that
8x384 does not. Keeping both makes the cost of that split visible.
"""

from __future__ import annotations

import statistics
import time

import ttnn

from models.experimental.nomic_embed_text_v2_moe.tt.model import TtNomicBertModel
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import TtModelConfig

# (batch, seqlen). 8x512 is the headline: T = 4096, the longest the tokenizer emits at this
# checkpoint's 512-token truncation limit, and the shape the expert chunking splits.
BENCHMARK_SHAPES = [(1, 128), (8, 384), (8, 512)]
HEADLINE_SHAPE = (8, 512)

WARMUP_ITERATIONS = 2
MEASURED_ITERATIONS = 20

# Passages repeated past the tokenizer 512-token truncation limit, so every row is exactly 512
# real tokens and the request figure is comparable with test_model_latency at 8x512.
BENCHMARK_TEXTS = [
    "Retrieval augmented generation combines a dense retriever with a generative language model, "
    "letting the system ground its answers in documents it fetches at query time. " * 16,
    "Les modeles de recuperation multilingues encodent des phrases de langues differentes dans un "
    "espace vectoriel commun, ce qui permet une recherche translingue. " * 16,
    "Die Hauptstadt von Deutschland ist Berlin, eine Stadt mit einer langen und bewegten "
    "Geschichte, deren Museen und Viertel viele Besucher anziehen. " * 16,
    "El procesamiento del lenguaje natural ha avanzado gracias a los transformadores, que "
    "sustituyeron a las redes recurrentes en casi todas las tareas. " * 16,
    "Sparse mixture of experts layers activate only a subset of the parameters for each input "
    "token, decoupling parameter count from per-token computation. " * 16,
    "Tokenization choices influence downstream retrieval quality more than most practitioners "
    "expect, because a vocabulary tuned for one language fragments another. " * 16,
    "Machine learning systems benefit from careful evaluation on held out data collected under "
    "the same conditions as production traffic. " * 16,
    "Vector databases index embeddings with approximate nearest neighbour structures, trading "
    "recall for query latencies that stay flat as the collection grows. " * 16,
]

_MODEL_CACHE = {}


def build_model(device, config, state_dict) -> tuple[TtNomicBertModel, float]:
    """Build the model once per device, returning it and the seconds the first build took.

    Memoized on the device because the `device` fixture is module-scoped under
    use_module_device while fixtures depending on it are not, so every test in a module would
    otherwise re-upload 951 MB of weights. The returned time is meaningful only on the first
    call; later callers get 0.0 and should not report it.
    """
    key = id(device)
    if key not in _MODEL_CACHE:
        start = time.perf_counter()
        model = TtNomicBertModel(device, config, TtModelConfig.from_device(device), state_dict)
        ttnn.synchronize_device(device)
        _MODEL_CACHE[key] = (model, time.perf_counter() - start)
        return _MODEL_CACHE[key]
    return _MODEL_CACHE[key][0], 0.0


def dram_allocated_bytes(device) -> int:
    """Bytes currently allocated across all DRAM banks.

    A live snapshot, not a high-water mark: ttnn exposes no peak counter, so a transient freed
    before the sample is invisible. Sampled at weight load and at forward exit these bracket
    residency without capturing the peak inside the expert bank.
    """
    view = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


def measure(run, device, warmup: int = WARMUP_ITERATIONS, iterations: int = MEASURED_ITERATIONS):
    """Time `run` after warming it, returning latency statistics in milliseconds.

    `run` takes no arguments and returns whatever it allocates; the caller's `release` is not
    needed because every caller here deallocates inside `run`. The first call is timed
    separately and returned as first_ms: it carries kernel compilation and program-cache
    population, which is a one-time cost and would otherwise skew the mean by seconds.
    """
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    run()
    ttnn.synchronize_device(device)
    first_ms = (time.perf_counter() - start) * 1000

    for _ in range(warmup - 1):
        run()
    ttnn.synchronize_device(device)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        run()
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - start) * 1000)

    return {
        "first_ms": first_ms,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": statistics.quantiles(samples, n=20)[18],
    }


def report(logger, label: str, batch: int, seqlen: int, stats: dict, extra: dict = None):
    """Log one benchmark row. Kept in one place so every measurement prints the same fields."""
    tokens = batch * seqlen
    median_s = stats["median_ms"] / 1000
    logger.info(
        f"\n{label}  B={batch} S={seqlen} T={tokens}"
        f"\n  first (compile + run) {stats['first_ms']:10.2f} ms"
        f"\n  steady median         {stats['median_ms']:10.3f} ms"
        f"\n  steady min            {stats['min_ms']:10.3f} ms"
        f"\n  steady mean           {stats['mean_ms']:10.3f} ms"
        f"\n  steady p95            {stats['p95_ms']:10.3f} ms"
        f"\n  throughput            {tokens / median_s:10.0f} tokens/s"
        f"\n  throughput            {batch / median_s:10.1f} sequences/s"
        + "".join(f"\n  {name:<21} {value:>10}" for name, value in (extra or {}).items())
    )
