# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 REAL-WEIGHTS prefill: throughput + KV-cache PCC on the galaxy (SP=8 × TP=4 + EP=32).

Builds the full 60-layer model with real bf16 weights via TtPrefillRuntime, runs prefill over the
golden trace's prompt (one-shot or chunked), measures throughput over a few iterations, then
PCC-checks every layer's post-RoPE K / raw V / MSA index_k against the golden trace from
scripts/generate_golden_kv_cache.py.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json + kv_cache/layer_N.safetensors)   [required in single-run;
                      in multi-run optional: seeds the initial capacity and the default PREFILL_GOLDEN_ROOT]
  PREFILL_CHUNKED     "1" -> chunked prefill (chunk loop, cache-read path); "0" -> one-shot  [default 0]
  PREFILL_CHUNK_SIZE  chunk size in tokens (chunked mode)                                  [default 5120]
  PREFILL_TPS_ITERS   prefill repetitions for the throughput measurement (less noise)      [default 1]
  PREFILL_SKIP_PCC    "1" -> perf only: skip the per-layer golden KV PCC (no kv_cache/ needed) [default 0]
  PREFILL_EXPECTED_TPS  whole-sequence tok/s baseline; when set, assert measured median is
                        within +/- PREFILL_PERF_MARGIN of this value                           [default: unset]
  PREFILL_PERF_MARGIN fraction tolerance around PREFILL_EXPECTED_TPS (e.g. 0.05 = +/-5%)     [default 0.05]
  PREFILL_REQUIRE_HIGH_POWER  "1" -> require is_high_power() (>=130W TDP via tt-smi); skip
                        otherwise. Used by the Blaze perf job; leave unset for accuracy/KV PCC   [default 0]
  PREFILL_STANDALONE_CHUNKED_PCC  min K/V/index_k PCC gate (fail below this)                 [default 0.88]
  PREFILL_NUM_LAYERS  build/run only the first N decoder layers (faster partial-model runs; also auto-sets
                      M3_LOAD_NLAYERS so only those layers' weight shards are read)          [default: all]
  EXPERT_DTYPE        MoE routed-expert weight dtype: "bf4" or "bf8" (cache holds both)      [default bf4]
  HF_MODEL            real MiniMax-M3 weights dir (read by ModelArgs)

Multi-run mode (load weights + compile ONCE, then run many specs against the resident model):
  PREFILL_RUNS        where run specs come from:
                        <file>        one spec per line, run in order, then exit (batch)
                        -             read specs from stdin line by line (interactive)
                        fifo:<path>   create the FIFO if needed and SERVE: block on it, run every
                                      line written to it, reopen on EOF; a line "quit" exits
                      A spec is whitespace-separated key=value pairs; `#` starts a comment anywhere on the
                      line (so paths may not contain `#`); a `quit` line ends any source, batch files too:
                        trace=<dir|name>   golden trace dir, or a name under PREFILL_GOLDEN_ROOT  [required]
                        iters=N            PREFILL_TPS_ITERS for this run                [default: env]
                        skip_pcc=0|1       PREFILL_SKIP_PCC for this run                 [default: env]
                        expected_tps=X perf_margin=F pcc_threshold=F                    [default: env]
                        isl=N              run N tokens: the trace's REAL tokens tiled cyclically to N
                                           (truncated if N < trace). Tiling keeps MoE routing / MSA block
                                           selection realistic; zero-padding would collapse both. KV PCC
                                           still checks the first min(N, trace) tokens (causal: the tail
                                           cannot touch them).                       [default: trace length]
                        capacity=N         KV-cache capacity for this run (tokens; rounded up to a chunk
                                           multiple)                          [default: PREFILL_MAX_SEQ_LEN or fit]
                        label=<str>        tag for the log lines / summary
  PREFILL_GOLDEN_ROOT dir that bare trace names resolve under   [default: dirname(PREFILL_TRACE_DIR)]
  PREFILL_MAX_SEQ_LEN when set, the FIXED KV-cache capacity (tokens) for every run without its own
                      capacity=; when unset each run FITS the cache to its trace (padded length), as a
                      standalone run would. [default: unset -> fit]
  PREFILL_RESULTS_JSONL  append one JSON line per run (perf + min PCC + status)         [default: unset]
  PREFILL_ISL         default isl= for every run (single-run mode too)                   [default: trace length]
  PREFILL_WARMUP_ITERS untimed whole-sequence passes before the timed iterations of each run. Programs are
                      keyed by cache capacity and chunk kind (first / cache-read / ragged), not by ISL, and
                      compile() warms those three; iteration 0 still carries host-side first-call costs
                      (e.g. the per-actual_isl MoE padding-config upload), so 1 is a good default for perf
                                                                                                   [default 0]
  PREFILL_COMPILE     "0" -> skip runtime.compile()'s three-variant warm-up entirely (then iteration 0 pays
                      the program-cache misses)                                                    [default 1]
  The chunk size (PREFILL_CHUNK_SIZE) is FIXED for the process: it is baked into the MoE dispatch
  buffers at build time. Multi-run mode is therefore chunked-only (PREFILL_CHUNKED=0 is rejected); a
  one-shot run is just a trace whose padded length equals the chunk size. The cache capacity is NOT
  fixed: when a run needs a different capacity the KV cache is re-allocated, the indexed RoPE rebuilt
  and the JIT buckets re-warmed (TtPrefillRuntime.reconfigure_capacity) — seconds to minutes, no
  weight reload. Fitting matters: the dense layers' ring-joint SDPA gathers the whole cache shard per
  chunk, so an over-sized cache costs real time (MSA layers are bounded by the valid prefix).

  # serve: weights load once, then from ANY shell on the node:
  PREFILL_TRACE_DIR=$GOLDEN/longbook_56320 PREFILL_RUNS=fifo:/tmp/m3_pcc.fifo \
    python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py
  echo "trace=longbook_10240 iters=3"                        > /tmp/m3_pcc.fifo
  echo "trace=longbook_56320 iters=5 skip_pcc=1 label=perf"  > /tmp/m3_pcc.fifo
  echo quit                                                   > /tmp/m3_pcc.fifo

Run (after weights are present on disk):
  cd <your tt-metal checkout>
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
  source python_env/bin/activate
  # Real bf16 weights + the tilized per-tensor cache both live here (the cache dir is derived from
  # HF_MODEL, so a complete cache means the ~869GB bf16 source is never read):
  export HF_MODEL=/mnt/models/MiniMaxAI/MiniMax-M3-ref
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  # chunked over the 10240-token golden (two 5120 chunks, no pad tail), 5 timed iterations:
  PREFILL_CHUNKED=1 PREFILL_TPS_ITERS=5 \
    PREFILL_TRACE_DIR=$HF_MODEL/golden/longbook_10240 \
    python3 models/demos/minimax_m3/tests/galaxy_prefill_kv_pcc.py
"""

import json
import math
import os
import resource
import stat
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE
from models.demos.minimax_m3.utils.fabric_env import ccl_topology_from_env, fabric_config_from_env


def _raise_nproc_limit():
    """tt-metal JIT-compiles device kernels in parallel, and each target is its own chain of
    short-lived processes (g++/cc1plus/as to compile; g++/collect2/lto-wrapper/lto1/as/ld to link),
    so the live process count runs to roughly a dozen times the build's parallelism. A low
    RLIMIT_NPROC (e.g. a 512 soft default) makes clone3 fail with EAGAIN mid-build, which gcc
    reports as "posix_spawn: Operation not permitted" and aborts the kernel link. Raise the soft
    limit to the hard limit (allowed without privileges) so the build never starves."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-pcc] raised RLIMIT_NPROC soft {soft} -> {hard} (tt-metal JIT build spawns many procs)")
        except (ValueError, OSError) as e:
            print(
                f"[prefill-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}); if the kernel build "
                f"fails with 'posix_spawn: Operation not permitted', run `ulimit -u 1048576` first: {e}",
                file=sys.stderr,
            )


# MSA (sparse) layers 3-59 pick the top-TOPK_BLOCKS of BLOCK_SIZE-token blocks; topk_large_indices
# aborts unless at least that many blocks exist, so the sparse path floors the sequence here. Keep in
# sync with TOPK_BLOCKS * BLOCK_SIZE in models/demos/minimax_m3/tt/attention/msa.py.
MSA_MIN_TOKENS = 16 * 128  # = 2048


class GateFailure(AssertionError):
    """A measurement gate (perf band / KV PCC threshold / capacity fit) failed. Subclasses AssertionError so
    external callers that caught the old assert keep working; the multi-run loop tells it apart from a real
    runtime invariant (which is reported as ERROR with a traceback, not FAIL)."""


def plan(n_tokens, chunk_size, chunked):
    """Resolve (n_chunks, chunk, total). chunked: full chunk_size chunks (tail padded). one-shot:
    a single chunk == total, padded up to a multiple of 1024 and at least MSA_MIN_TOKENS (MSA needs
    S%1024==0 and S>=2048; 1024 is also a multiple of sp=8 so the SP shard stays tile-aligned)."""
    if chunked:
        chunk = chunk_size
        n_chunks = max(1, math.ceil(n_tokens / chunk))
    else:
        chunk = max(MSA_MIN_TOKENS, math.ceil(n_tokens / 1024) * 1024)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


def check_kv_pcc(runtime, kv_cache, golden_dir, n_tokens, num_layers, hf_config, threshold=None):
    """Per-layer K / V / index_k PCC: device cache vs the golden trace. The device stores K / index_k
    Meta-RoPE swizzled over the rotary slice; the golden is HF half-split, so permute the golden's
    rotary slice (identity tail) before comparing. V is raw (no swizzle).

    The slot is read back ONCE (``read_slot_kv``: a device-side slot slice plus one mesh compose per
    cache), bounded to the first ceil(n_tokens / chunk) chunks — the block-cyclic layout keeps them in the
    first rows of every chip, so a 55k check inside a 1M-capacity cache reads 55k, not 1M (which would be
    hundreds of GB on host). Each layer is un-rotated on host; reading per layer instead would re-copy
    the packed cache num_layers times over PCIe.

    Fails if overall min PCC is below PREFILL_STANDALONE_CHUNKED_PCC (default 0.88), matching
    models/demos/minimax_m3/tt/runners/prefill_kv_validation.py.
    """
    from safetensors import safe_open

    from models.common.utility_functions import comp_pcc
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block

    if threshold is None:
        threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88"))
    head_dim = hf_config.head_dim
    rotary_dim = getattr(hf_config, "rotary_dim", head_dim)
    half = rotary_dim // 2
    src = list(range(head_dim))
    for m in range(rotary_dim):
        src[m] = half * (m % 2) + (m // 2)
    src = torch.tensor(src, dtype=torch.long)

    kv_dir = Path(golden_dir) / "kv_cache"
    logger.info(f"[kv-pcc] per-layer K / V / index_k vs golden ({golden_dir}):")
    mins = {"k": 1.0, "v": 1.0, "index_k": 1.0}
    assert kv_cache.max_seq_len == runtime.config.max_seq_len, (
        f"KV cache capacity {kv_cache.max_seq_len} != runtime capacity {runtime.config.max_seq_len}: the un-rotation "
        f"below would decode the wrong block-cyclic layout (stale handle after a re-target?)"
    )
    k_blk, v_blk, ik_blk = runtime.read_slot_kv(kv_cache, 0, n_tokens)  # block-cyclic; un-rotated per layer below
    sp, chunk, seq = runtime.config.sp_factor, runtime.config.chunk_size, runtime.read_seq_len(n_tokens)
    for L in range(num_layers):
        dev_k = naturalize_kv_block(k_blk[L], n_tokens, sp, chunk, seq).unsqueeze(0)
        dev_v = naturalize_kv_block(v_blk[L], n_tokens, sp, chunk, seq).unsqueeze(0)
        dev_ik = naturalize_kv_block(ik_blk[L], n_tokens, sp, chunk, seq).unsqueeze(0)
        with safe_open(str(kv_dir / f"layer_{L}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g_k = h.get_tensor(f"key_cache_layer_{L}").float()[:, :, :n_tokens, :][..., src]  # HF -> Meta
            g_v = h.get_tensor(f"value_cache_layer_{L}").float()[:, :, :n_tokens, :]
            has_ik = f"index_k_cache_layer_{L}" in keys
            g_ik = h.get_tensor(f"index_k_cache_layer_{L}").float()[:, :, :n_tokens, :][..., src] if has_ik else None

        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        line = f"  layer {L:>2}: K={pcc_k:.5f} V={pcc_v:.5f}"
        if has_ik:
            pcc_ik = float(comp_pcc(g_ik, dev_ik, 0.0)[1])
            mins["index_k"] = min(mins["index_k"], pcc_ik)
            line += f" index_k={pcc_ik:.5f}"
        logger.info(line)

    min_pcc = min(mins.values())
    logger.info(
        f"[kv-pcc] min PCC across {num_layers} layers: "
        f"K={mins['k']:.5f} V={mins['v']:.5f} index_k={mins['index_k']:.5f} "
        f"(overall {min_pcc:.5f}, threshold {threshold})"
    )
    if min_pcc < threshold:
        raise GateFailure(f"KV-cache PCC {min_pcc:.5f} < threshold {threshold}")
    return mins


@dataclass
class RunSpec:
    """One prefill measurement against the resident model: which golden trace, how many timed
    iterations, and the perf / PCC gates. In single-run mode there is exactly one, built from the env;
    in multi-run mode (PREFILL_RUNS) one per spec line, with the env as the per-field default."""

    trace_dir: str
    tps_iters: int = 1
    skip_pcc: bool = False
    expected_tps: float | None = None
    perf_margin: float = 0.05
    pcc_threshold: float | None = None  # None -> PREFILL_STANDALONE_CHUNKED_PCC (0.88)
    capacity: int | None = (
        None  # KV-cache tokens for this run; None -> fit the padded length (from_env pre-fills PREFILL_MAX_SEQ_LEN)
    )
    isl: int | None = None  # tokens to prefill (trace tiled cyclically / truncated); None -> the trace length
    label: str = ""

    @classmethod
    def from_env(cls, trace_dir: str) -> "RunSpec":
        exp = os.environ.get("PREFILL_EXPECTED_TPS")
        thr = os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC")
        return cls(
            trace_dir=trace_dir,
            tps_iters=int(os.getenv("PREFILL_TPS_ITERS", "1")),
            skip_pcc=os.environ.get("PREFILL_SKIP_PCC") == "1",
            expected_tps=float(exp) if exp is not None else None,
            perf_margin=float(os.environ.get("PREFILL_PERF_MARGIN", "0.05")),
            pcc_threshold=float(thr) if thr is not None else None,
            capacity=int(cap) if (cap := os.environ.get("PREFILL_MAX_SEQ_LEN")) else None,
            isl=int(isl) if (isl := os.environ.get("PREFILL_ISL")) else None,
        )

    @classmethod
    def parse(cls, line: str, golden_root: str | None) -> "RunSpec":
        """``key=value`` pairs separated by whitespace. ``trace`` is a dir, or a bare name resolved under
        ``golden_root``. Every other key defaults to the env-derived value (see from_env)."""
        kv = {}
        for tok in line.split():
            if "=" not in tok:
                raise ValueError(f"bad token {tok!r} (want key=value)")
            k, v = tok.split("=", 1)
            kv[k] = v
        if "trace" not in kv:
            raise ValueError("spec needs trace=<dir|name>")
        trace = kv.pop("trace")
        if not os.path.isdir(trace) and golden_root and os.path.isdir(os.path.join(golden_root, trace)):
            trace = os.path.join(golden_root, trace)
        if not os.path.isfile(os.path.join(trace, "metadata.json")):
            raise ValueError(f"trace {trace!r}: no metadata.json (PREFILL_GOLDEN_ROOT={golden_root})")
        spec = cls.from_env(trace)
        for k, v in kv.items():
            if k == "iters":
                spec.tps_iters = int(v)
            elif k == "skip_pcc":
                spec.skip_pcc = v == "1"
            elif k == "expected_tps":
                spec.expected_tps = float(v)
            elif k == "perf_margin":
                spec.perf_margin = float(v)
            elif k == "pcc_threshold":
                spec.pcc_threshold = float(v)
            elif k == "capacity":
                spec.capacity = int(v)
            elif k == "isl":
                spec.isl = int(v)
            elif k == "label":
                spec.label = v
            else:
                raise ValueError(f"unknown key {k!r}")
        if spec.tps_iters < 1:
            raise ValueError(f"iters={spec.tps_iters}: need >= 1")
        if spec.isl is not None and spec.isl < 1:
            raise ValueError(f"isl={spec.isl}: need >= 1")
        if spec.capacity is not None and spec.capacity < 1:
            raise ValueError(f"capacity={spec.capacity}: need >= 1")
        return spec

    @property
    def name(self) -> str:
        base = self.label or os.path.basename(os.path.normpath(self.trace_dir))
        return base if self.label or self.isl is None else f"{base}@{self.isl}"


def load_trace_tokens(trace_dir: str) -> list:
    return list(json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"])


def tile_tokens(token_ids: list, isl: int) -> list:
    """The trace's real tokens repeated cyclically to exactly ``isl`` tokens (truncated when isl < len).
    Real text keeps the MoE router's expert distribution and the MSA top-k block selection realistic;
    a run of identical pad tokens would route every token to the same experts (EP collapse) and pick
    the same blocks, measuring a pathological load instead of long-context prefill."""
    assert token_ids, "empty trace"
    reps = math.ceil(isl / len(token_ids))
    return (token_ids * reps)[:isl]


def ensure_fifo(source: str) -> str:
    """Create the FIFO of a ``fifo:<path>`` source if missing and check it IS a FIFO. Called from main()
    BEFORE the 10+ minute model build: an `echo spec > path` issued before the server got there would
    otherwise create a regular file and the assert would kill the server after the build."""
    path = source[len("fifo:") :]
    if not os.path.exists(path):
        os.mkfifo(path)
    if not stat.S_ISFIFO(os.stat(path).st_mode):
        raise SystemExit(f"ERROR: PREFILL_RUNS={source}: {path} exists and is not a FIFO (remove it first)")
    return path


def iter_run_specs(source: str, golden_root: str | None):
    """Yield (raw_line, RunSpec-or-ValueError) from the PREFILL_RUNS source. ``<file>``: every line, then
    stop. ``-``: stdin until EOF. ``fifo:<path>``: create the FIFO if missing, then serve — block on it,
    yield each line written, reopen after every writer EOF, stop on a ``quit`` / ``exit`` line."""

    def parse_lines(fh):
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line in ("quit", "exit"):
                yield line, None
                return
            try:
                yield line, RunSpec.parse(line, golden_root)
            except ValueError as e:
                yield line, e

    if source == "-":
        print("[prefill-pcc] multi-run: reading specs from stdin (EOF ends)", flush=True)
        yield from parse_lines(sys.stdin)
        return
    if source.startswith("fifo:"):
        path = ensure_fifo(source)
        print(
            f"[prefill-pcc] multi-run: SERVING on {path} — weights stay resident.\n"
            f"    echo 'trace=<dir|name> [isl=N] [capacity=N] [iters=N] [skip_pcc=1] [expected_tps=X] "
            f"[perf_margin=F] [pcc_threshold=F] [label=..]' > {path}\n"
            f"    echo quit > {path}",
            flush=True,
        )
        while True:
            with open(path) as fh:  # blocks until a writer opens; EOF when it closes -> reopen
                for line, item in parse_lines(fh):
                    yield line, item
                    if item is None:
                        return
        return
    with open(source) as fh:
        yield from parse_lines(fh)


def run_one(runtime, state: dict, mesh, spec: RunSpec, num_layers, hf_config) -> dict:
    """Prefill ``spec.trace_dir`` into slot 0 of the resident model, time it, gate it, PCC it. The chunk
    size is the runtime's (fixed at build). The KV-cache capacity is per run — ``spec.capacity`` or the
    trace's padded length — and when it differs from the resident one the cache in ``state["kv_cache"]``
    is re-allocated, the runtime re-targeted (reconfigure_capacity) and the JIT buckets re-warmed."""
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches

    chunk = runtime.config.chunk_size
    trace_ids = load_trace_tokens(spec.trace_dir)
    n_trace = len(trace_ids)
    n_tokens = spec.isl or n_trace  # tokens actually prefilled (the "isl")
    token_ids = tile_tokens(trace_ids, n_tokens) if n_tokens != n_trace else trace_ids
    n_pcc = min(n_tokens, n_trace)  # golden covers the trace only; causality keeps the tiled tail off it
    n_chunks = max(1, math.ceil(n_tokens / chunk))
    total = n_chunks * chunk
    capacity = math.ceil((spec.capacity or total) / chunk) * chunk
    if total > capacity:
        raise GateFailure(
            f"trace {spec.trace_dir} pads to {total} tokens ({n_chunks} x {chunk}) but the requested capacity "
            f"is {capacity} (capacity= / PREFILL_MAX_SEQ_LEN); drop it or raise it"
        )
    tps_iters = spec.tps_iters
    warmup_iters = int(os.getenv("PREFILL_WARMUP_ITERS", "0"))
    print(
        f"[prefill-pcc] === run '{spec.name}': golden={spec.trace_dir} isl={n_tokens} (trace {n_trace}"
        f"{', TILED cyclically' if n_tokens > n_trace else (', truncated' if n_tokens < n_trace else '')}) "
        f"chunk={chunk} n_chunks={n_chunks} total={total} capacity={capacity} "
        f"warmup_iters={warmup_iters} tps_iters={tps_iters} skip_pcc={spec.skip_pcc}",
        flush=True,
    )
    recompile_s = 0.0
    if capacity != runtime.config.max_seq_len or state["kv_cache"] is None:
        t0 = time.perf_counter()
        warm = (
            "re-warming the 3 chunk program variants"
            if os.getenv("PREFILL_COMPILE", "1") != "0"
            else "no warm-up (PREFILL_COMPILE=0)"
        )
        print(
            f"[prefill-pcc] capacity {runtime.config.max_seq_len} -> {capacity}: re-allocating the KV cache, "
            f"rebuilding indexed RoPE, {warm} (weights stay resident) ...",
            flush=True,
        )
        # Free first (the new cache may not fit next to the old one), and drop the handle BEFORE anything
        # that can raise: a failed re-target must leave state["kv_cache"] None, so the next spec
        # re-allocates instead of feeding a dead handle into prefill_chunk.
        if state["kv_cache"] is not None:
            state["kv_cache"].deallocate()
            state["kv_cache"] = None
        runtime.reconfigure_capacity(capacity)  # on failure keeps the old capacity + rope
        state["kv_cache"] = allocate_kv_caches(
            mesh, num_layers=num_layers, max_seq_len=capacity, num_users=1, head_dim=hf_config.head_dim
        )
        if os.getenv("PREFILL_COMPILE", "1") != "0":
            runtime.compile(state["kv_cache"])
        recompile_s = time.perf_counter() - t0
        print(f"[prefill-pcc] re-targeted at capacity {capacity} in {recompile_s:.1f} s", flush=True)
    kv_cache = state["kv_cache"]
    if n_tokens < MSA_MIN_TOKENS:
        print(
            f"[prefill-pcc] WARNING: prompt is only {n_tokens} tokens (< MSA floor {MSA_MIN_TOKENS}); padded to "
            f"{total}. KV PCC over the {n_tokens} real tokens stays valid; tok/s is pad-dominated — ignore it.",
            flush=True,
        )

    # --- throughput. Each iteration re-fills slot 0 (valid for the PCC check after the loop) and
    # times two distinct full-prefill passes, each with syncs placed so it pays for no extra barrier:
    #   WHOLE SEQUENCE  all n_chunks cold from an empty cache, ONE sync at the very end
    #                   -> time to prefill the whole prompt         (e.g. 55k @ 0 cache)
    #   LAST CHUNK      pre-fill chunks 0..n-2, sync ONCE (barrier, NOT timed), then time only the
    #                   final chunk against that accumulated cache   (e.g. 5k @ 50k cache)
    padded = token_ids + [0] * (total - n_tokens)
    a_last = (n_chunks - 1) * chunk  # context tokens already in cache ahead of the final chunk
    last_len = min(a_last + chunk, total) - a_last  # width of the final chunk (incl pad)

    def prefill_chunk(c):
        a = c * chunk
        inp = runtime.make_chunk_input(padded[a : a + chunk])
        runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=a, actual_end=min(a + chunk, n_tokens))

    def run_whole():  # cold, no mid-loop syncs — one barrier at the end
        for c in range(n_chunks):
            prefill_chunk(c)
        ttnn.synchronize_device(mesh)

    def run_last_chunk():  # returns the timed last-chunk wall seconds
        for c in range(n_chunks - 1):
            prefill_chunk(c)
        ttnn.synchronize_device(mesh)  # barrier before the timed region — NOT counted
        t0 = time.perf_counter()
        prefill_chunk(n_chunks - 1)
        ttnn.synchronize_device(mesh)
        return time.perf_counter() - t0

    for i in range(warmup_iters):  # untimed: absorbs the per-ISL JIT (see PREFILL_WARMUP_ITERS)
        t0 = time.perf_counter()
        run_whole()
        print(f"[prefill-pcc] warmup {i}: whole {(time.perf_counter() - t0) * 1000:.1f} ms (not counted)", flush=True)

    whole_times, last_times = [], []
    for i in range(tps_iters):
        t0 = time.perf_counter()
        run_whole()
        whole_times.append(time.perf_counter() - t0)
        last_times.append(run_last_chunk())
        print(
            f"[prefill-pcc] iter {i}: whole {whole_times[-1] * 1000:.1f} ms  last_chunk {last_times[-1] * 1000:.1f} ms",
            flush=True,
        )

    w = statistics.median(whole_times)
    whole_tps = n_tokens / w
    print(
        f"[prefill-pcc] WHOLE SEQUENCE over {tps_iters} iters: {n_tokens} tok @ 0 cache, "
        f"median {whole_tps:.1f} tok/s (real prompt), {total / w:.1f} tok/s (processed); "
        f"wall median {w * 1000:.1f} ms [min {min(whole_times) * 1000:.1f}, max {max(whole_times) * 1000:.1f}]",
        flush=True,
    )
    lc = statistics.median(last_times)
    last_tps = last_len / lc  # processed (padded) tokens, the historical basis of this line
    last_real = n_tokens - a_last  # real tokens in the final chunk (== last_len unless the trace is ragged)
    print(
        f"[prefill-pcc] LAST CHUNK over {tps_iters} iters: {last_len} tok @ {a_last} cache, "
        f"median {last_tps:.1f} tok/s (processed), {last_real / lc:.1f} tok/s ({last_real} real); "
        f"wall median {lc * 1000:.1f} ms [min {min(last_times) * 1000:.1f}, max {max(last_times) * 1000:.1f}]",
        flush=True,
    )
    result = {
        "label": spec.name,
        "trace": spec.trace_dir,
        "n_tokens": n_tokens,
        "trace_tokens": n_trace,
        "chunk": chunk,
        "n_chunks": n_chunks,
        "capacity": capacity,
        "recompile_s": recompile_s,
        "warmup_iters": warmup_iters,
        "tps_iters": tps_iters,
        "whole_ms": w * 1000,
        "whole_tps": whole_tps,
        "last_chunk_ms": lc * 1000,
        "last_chunk_tps": last_tps,
        "last_chunk_real_tokens": last_real,
        "min_pcc": None,
    }

    # --- perf gate: whole-sequence tok/s vs expected_tps +/- perf_margin ---
    if spec.expected_tps is not None:
        low = spec.expected_tps * (1.0 - spec.perf_margin)
        high = spec.expected_tps * (1.0 + spec.perf_margin)
        print(
            f"[prefill-pcc] PERF GATE: measured {whole_tps:.1f} tok/s vs baseline "
            f"{spec.expected_tps:.1f} +/- {spec.perf_margin * 100:.1f}% band [{low:.1f}, {high:.1f}]",
            flush=True,
        )
        if not (low <= whole_tps <= high):
            raise GateFailure(
                f"whole-sequence throughput {whole_tps:.1f} tok/s outside baseline "
                f"{spec.expected_tps:.1f} tok/s +/- {spec.perf_margin * 100:.1f}% band [{low:.1f}, {high:.1f}]"
            )

    # --- accuracy: per-layer KV PCC vs golden (skipped in perf-only mode; synthetic
    # traces carry only metadata.json, so there is no golden KV cache to compare against) ---
    if spec.skip_pcc:
        print("[prefill-pcc] skip_pcc -> skipping per-layer KV PCC", flush=True)
    else:
        if n_pcc != n_tokens:
            print(f"[prefill-pcc] isl={n_tokens} != trace {n_trace}: KV PCC over the first {n_pcc} tokens", flush=True)
        mins = check_kv_pcc(runtime, kv_cache, spec.trace_dir, n_pcc, num_layers, hf_config, spec.pcc_threshold)
        result["min_pcc"] = min(mins.values())
    return result


def _fmt_result(r: dict) -> str:
    pcc = f"{r['min_pcc']:.5f}" if r.get("min_pcc") is not None else "-"
    if r.get("status") != "ok":
        return f"{r['label']:<28} {r.get('status', '?'):<6} {r.get('error', '')}"
    return (
        f"{r['label']:<28} ok     {r['n_tokens']:>6} tok cap {r['capacity']:>6}  whole {r['whole_ms']:>9.1f} ms {r['whole_tps']:>8.1f} tok/s  "
        f"last {r['last_chunk_ms']:>8.1f} ms {r['last_chunk_tps']:>8.1f} tok/s  minPCC {pcc}"
    )


def main():
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    _raise_nproc_limit()  # tt-metal parallel kernel JIT needs a high process limit (see fn docstring)

    # Perf CI only: same guard as Kimi/GLM no_pcc / ring-joint SDPA (pytest skipif is_high_power).
    # Accuracy/KV-PCC leaves PREFILL_REQUIRE_HIGH_POWER unset so bh_sc1 hosts are fine.
    if os.getenv("PREFILL_REQUIRE_HIGH_POWER", "0") == "1":
        from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_tdp_limit_max, is_high_power

        if not is_high_power():
            tdp = get_tdp_limit_max()
            print(
                f"[prefill-pcc] SKIP: PREFILL_REQUIRE_HIGH_POWER=1 but host is not high-power "
                f"(TDP_LIMIT_MAX={tdp}; need >=130W). Guards exabox.tenstorrent.com/power=14kw.",
                flush=True,
            )
            return 0

    golden_dir = os.environ.get("PREFILL_TRACE_DIR")
    runs_source = os.environ.get("PREFILL_RUNS")
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", "5120"))
    golden_root = os.environ.get("PREFILL_GOLDEN_ROOT") or (
        os.path.dirname(os.path.normpath(golden_dir)) if golden_dir else None
    )

    # --- resolve the process-wide (chunk, capacity) and the run list -----------------------------
    # Single-run: exactly the historical behaviour (one-shot or chunked, capacity == padded length).
    # Multi-run: chunked only. The capacity chosen here is only the INITIAL one (first build + warm-up):
    # run_one re-targets the resident model whenever a spec needs a different capacity. Default it to
    # the first known trace so the common case (PREFILL_TRACE_DIR == the first spec) needs no re-warm.
    batch_specs = None  # only for a regular runs file (known up front)
    if runs_source is None:
        if not golden_dir:
            print("ERROR: set PREFILL_TRACE_DIR to a golden trace dir (or PREFILL_RUNS for multi-run)", file=sys.stderr)
            return 1
        n_tokens = int(os.environ.get("PREFILL_ISL") or len(load_trace_tokens(golden_dir)))
        n_chunks, chunk, capacity = plan(n_tokens, chunk_size, chunked)
        print(
            f"[prefill-pcc] golden={golden_dir} n_tokens={n_tokens} "
            f"mode={'chunked' if chunked else 'one-shot'} chunk={chunk} n_chunks={n_chunks} total={capacity} "
            f"tps_iters={os.getenv('PREFILL_TPS_ITERS', '1')}",
            flush=True,
        )
        if not chunked and n_tokens < MSA_MIN_TOKENS:
            bang = "!" * 80
            print(
                f"\n{bang}\n"
                f"[prefill-pcc] WARNING: prompt is only {n_tokens} tokens, below the MSA sparse floor of\n"
                f"  {MSA_MIN_TOKENS} (TOPK_BLOCKS*BLOCK_SIZE = 16*128). Layers 3-59 select the top-16 of\n"
                f"  128-token blocks, and topk_large_indices aborts with fewer than 16 blocks. PADDING the\n"
                f"  sequence {n_tokens} -> {capacity} tokens (token 0) so the sparse path can run.\n"
                f"  * ACCURACY IS STILL VALID: KV PCC compares only the first {n_tokens} real tokens, and\n"
                f"    causal masking keeps the trailing pad (positionally future) from touching them.\n"
                f"  * THROUGHPUT IS NOT: tok/s below is dominated by {capacity - n_tokens} pad tokens — ignore it\n"
                f"    for tiny prompts and measure perf on a >= {MSA_MIN_TOKENS}-token trace instead.\n"
                f"{bang}\n",
                flush=True,
            )
    else:
        if os.getenv("PREFILL_CHUNKED", "1") != "1":
            print(
                "ERROR: PREFILL_RUNS (multi-run) is chunked-only: the chunk size is baked into the resident "
                "model (MoE dispatch buffers, indexed RoPE, JIT warm-up). Unset PREFILL_CHUNKED=0; a one-shot "
                "run is a trace whose padded length equals PREFILL_CHUNK_SIZE.",
                file=sys.stderr,
            )
            return 1
        chunk = chunk_size
        known_totals = []
        if golden_dir:
            n0 = int(os.environ.get("PREFILL_ISL") or len(load_trace_tokens(golden_dir)))
            known_totals.append(plan(n0, chunk, True)[2])
        if runs_source.startswith("fifo:"):
            ensure_fifo(runs_source)  # fail fast, before the build; the serve loop re-checks
        if runs_source != "-" and not runs_source.startswith("fifo:"):
            batch_specs = []
            for line, item in iter_run_specs(runs_source, golden_root):
                if item is None:
                    break
                if isinstance(item, Exception):
                    print(f"ERROR: PREFILL_RUNS line {line!r}: {item}", file=sys.stderr)
                    return 1
                batch_specs.append(item)
                fit = plan(item.isl or len(load_trace_tokens(item.trace_dir)), chunk, True)[2]
                known_totals.append(max(fit, math.ceil(item.capacity / chunk) * chunk) if item.capacity else fit)
            if not batch_specs:
                print(f"ERROR: no run specs in {runs_source}", file=sys.stderr)
                return 1
        cap_env = os.environ.get("PREFILL_MAX_SEQ_LEN")
        if cap_env:
            capacity = math.ceil(int(cap_env) / chunk) * chunk
        elif known_totals:
            capacity = known_totals[0]  # PREFILL_TRACE_DIR if set, else the first batch spec
        else:
            capacity = chunk  # nothing known yet (stdin / fifo): warm one bucket, first spec re-targets
        print(
            f"[prefill-pcc] multi-run: source={runs_source} chunk={chunk} initial capacity={capacity} "
            f"({capacity // chunk} chunks) per-run capacity={'FIXED ' + cap_env if cap_env else 'fit to trace'} "
            f"golden_root={golden_root} specs={'stream' if batch_specs is None else len(batch_specs)}",
            flush=True,
        )
    if chunk < MSA_MIN_TOKENS and (runs_source is not None or chunked):
        print(
            f"[prefill-pcc] WARNING: chunked chunk={chunk} < MSA floor {MSA_MIN_TOKENS}; the first chunk "
            f"(cache empty) has < 16 blocks and MSA topk will abort. Use PREFILL_CHUNK_SIZE >= {MSA_MIN_TOKENS}.",
            flush=True,
        )

    rows, cols = 8, 4  # SP=8 (rows), TP=4 (cols), EP=32

    # M3_FABRIC / M3_CCL_TOPOLOGY (utils/fabric_env.py): fabric config and legacy-CCL topology. Defaults
    # match the production runner (1d, linear). 1d_ring / 2d_torus_xy need the torus_xy mesh graph
    # descriptor (the wrapper scripts pick it); measurements in PR #55668.
    ccl_topology = ccl_topology_from_env()
    ttnn.set_fabric_config(fabric_config_from_env())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), l1_small_size=L1_SMALL_SIZE)
    print(
        f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()} "
        f"fabric={ttnn.get_fabric_config()} ccl_topology={ccl_topology}",
        flush=True,
    )
    results = []
    try:
        model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        nl_override = os.getenv("PREFILL_NUM_LAYERS")
        if nl_override:
            num_layers = int(nl_override)
            hf_config.num_hidden_layers = num_layers  # build/run only the first N decoder layers
            # Only read the safetensors shards holding layers 0..N-1 (+ embed/norm/lm_head) — skips most of
            # the NFS source read. The full model's first N layers are causal-identical to the truncated
            # model's, so the golden's first N layers are still the correct reference.
            os.environ.setdefault("M3_LOAD_NLAYERS", str(num_layers))
            print(
                f"[prefill-pcc] PREFILL_NUM_LAYERS={num_layers}: first {num_layers} layers only "
                f"(M3_LOAD_NLAYERS={os.environ['M3_LOAD_NLAYERS']})",
                flush=True,
            )

        # Weight loading. The bf16 source backbone is ~869GB — larger than host RAM here — so reading
        # it every run thrashes the page cache for >1h. Every weight module already loads its tilized
        # tensor from a per-tensor .tensorbin cache via ttnn.as_tensor(cache_file_name=); on a cache hit
        # the source tensor is ignored. So once the cache is populated we pass an EMPTY state_dict and
        # never touch the source — DeepSeek's state_dict={} + check_cache_complete trick.
        #   M3_FORCE_LOAD_WEIGHTS=1  force the source read (to (re)populate the cache / first run)
        #   M3_WEIGHTS_FROM_CACHE=1  force cache-only even if the completeness check is unsure
        cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        # EXPERT_DTYPE selects the MoE routed-expert weight dtype (bf4 default / bf8). The tilized
        # cache holds both, so either stays on the fast cache path. Same knob name as the generate
        # harnesses (galaxy_generate_m3*.py). It feeds TtPrefillRuntimeConfig.expert_weight_dtype below.
        expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
        print(
            f"[prefill-pcc] expert_dtype={expert_dtype} (EXPERT_DTYPE={os.getenv('EXPERT_DTYPE', 'bf4')})", flush=True
        )
        force_load = os.getenv("M3_FORCE_LOAD_WEIGHTS") == "1"
        cache_complete = weight_cache_is_complete(cache_path, hf_config, num_layers, expert_dtype)
        cache_only = not force_load and (os.getenv("M3_WEIGHTS_FROM_CACHE") == "1" or cache_complete)
        if cache_only:
            print(
                "[prefill-pcc] tilized weight cache complete -> loading from cache, "
                "skipping the ~869GB bf16 source read",
                flush=True,
            )
            state_dict = {}
        else:
            bang = "!" * 80
            print(
                f"\n{bang}\n"
                f"[prefill-pcc] WARNING: warm tilized weight cache NOT available at\n"
                f"  {cache_path}\n"
                f"  (complete={cache_complete}, M3_FORCE_LOAD_WEIGHTS={force_load}).\n"
                f"  Falling back to the ~869GB bf16 source read + cache build — expect multi-hour\n"
                f"  wall time; CI timeouts (even 2h) will usually fire. Prefill tensor_cache_bfp8_*\n"
                f"  under HF_MODEL / TT_CACHE_PATH on this host before relying on this job.\n"
                f"{bang}\n",
                flush=True,
            )
            print("[prefill-pcc] loading real bf16 weights + EP placement (slow: bf16 source read) ...", flush=True)
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)
        cfg = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=capacity,
            mesh_shape=(rows, cols),
            chunk_size=chunk,
            num_users=1,
            expert_weight_dtype=expert_dtype,
            weight_cache_path=cache_path,
            topology=ccl_topology,
        )
        t_build = time.perf_counter()
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
        del state_dict
        print(f"[prefill-pcc] model built in {time.perf_counter() - t_build:.1f} s", flush=True)

        # The runtime is stateless w.r.t. the cache (engine-owned model): allocate it here and pass it
        # into every runtime call (compile / prefill_chunk / gather_layer), mirroring the prefill engine.
        # Held in a dict because run_one re-allocates it when a spec needs a different capacity.
        state = {
            "kv_cache": allocate_kv_caches(
                mesh, num_layers=num_layers, max_seq_len=capacity, num_users=1, head_dim=hf_config.head_dim
            )
        }

        if os.getenv("PREFILL_COMPILE", "1") != "0":
            print(f"[prefill-pcc] compiling ({num_layers}L, SP=8 × TP=4 + EP=32, capacity {capacity}) ...", flush=True)
            runtime.compile(state["kv_cache"])
        else:
            print("[prefill-pcc] PREFILL_COMPILE=0 -> skipping the per-bucket warm-up sweep", flush=True)

        results_jsonl = os.environ.get("PREFILL_RESULTS_JSONL")

        def do_run(spec: RunSpec) -> dict:
            t0 = time.perf_counter()
            try:
                r = run_one(runtime, state, mesh, spec, num_layers, hf_config)
                r["status"] = "ok"
            except GateFailure as e:  # perf band / PCC threshold / capacity fit: a measurement verdict, not a crash
                r = {"label": spec.name, "trace": spec.trace_dir, "status": "FAIL", "error": str(e)}
                print(f"[prefill-pcc] run '{spec.name}' FAILED: {e}", flush=True)
            except Exception as e:  # anything else is a real error; keep the resident model alive for the next spec
                r = {
                    "label": spec.name,
                    "trace": spec.trace_dir,
                    "status": "ERROR",
                    "error": f"{type(e).__name__}: {e}",
                }
                traceback.print_exc()
                print(f"[prefill-pcc] run '{spec.name}' ERROR: {e}", flush=True)
            r["wall_s"] = time.perf_counter() - t0
            print(f"[prefill-pcc] RESULT {_fmt_result(r)}", flush=True)
            if results_jsonl:
                with open(results_jsonl, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            results.append(r)
            return r

        clean_exit = False  # a streaming session that ended on quit/EOF/Ctrl-C with no runs is not a failure
        if runs_source is None:
            spec = RunSpec.from_env(golden_dir)
            spec.capacity = capacity  # single-run: exactly the historical capacity (padded length); PREFILL_MAX_SEQ_LEN is multi-run only
            do_run(spec)
        elif batch_specs is not None:
            for spec in batch_specs:
                do_run(spec)
        else:
            try:
                for line, item in iter_run_specs(runs_source, golden_root):
                    if item is None:
                        print("[prefill-pcc] quit received", flush=True)
                        break
                    if isinstance(item, Exception):
                        print(f"[prefill-pcc] ignoring bad spec {line!r}: {item}", flush=True)
                        continue
                    do_run(item)
                    if runs_source.startswith("fifo:"):
                        print(f"[prefill-pcc] idle — waiting for the next spec on {runs_source[5:]}", flush=True)
                clean_exit = True
            except KeyboardInterrupt:
                print("[prefill-pcc] interrupted — summarising the runs done so far", flush=True)
                clean_exit = True

        if len(results) > 1 or runs_source is not None:
            print("[prefill-pcc] ===== SUMMARY =====", flush=True)
            for r in results:
                print(f"[prefill-pcc]   {_fmt_result(r)}", flush=True)
        print("[prefill-pcc] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    ok = all(r["status"] == "ok" for r in results)
    return 0 if ok and (results or clean_exit) else 1


if __name__ == "__main__":
    sys.exit(main())
