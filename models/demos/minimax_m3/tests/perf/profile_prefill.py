# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 REAL-WEIGHTS chunked-prefill zone profiler: per-zone device time for a dense and a sparse layer.

Measures ONE chunk attending an already-populated KV cache — the "5k attended to 25k / 55k" case.
Structure (mirrors galaxy_prefill_kv_pcc.py's LAST CHUNK measurement):

  1. build the real 60-layer model (SP=8 x TP=4 + EP=32) from the tilized weight cache
  2. runtime.compile()          -> WARMUP: JIT-compiles every op, populates the program cache
  3. pre-fill chunks 0..n-2     -> fills the cache to PROFILE_CACHE tokens (NOT profiled)
  4. profile the FINAL chunk    -> zones on, one signposted region per zone, per-layer profiler reads

Only step 4 is inside the zone markers, so the report is exactly "one 5k chunk against an N-token
cache". Layers 0-2 are dense (dense attention + dense MLP), layers 3-59 are sparse (MSA + MoE), so a
single chunk profiles both classes; the report separates them by the layer tag.

What you get per zone: summed DEVICE KERNEL DURATION [ns] per device (with the across-device skew),
op count, bytes moved (from the CSV's input/output shapes + dtypes) and the implied GB/s. Parse with
    python3 models/demos/minimax_m3/tests/perf/parse_zone_perf.py <ops_perf_results_*.csv> --html report.html

Zone list — dense layer: input_norm, attn/{qkv_proj,split_heads,qk_norm,rope,kv_write,
ring_joint_sdpa,concat_heads,o_proj,ccl_out_allreduce}, post_attn_norm, mlp/{gate_up_proj,swiglu,
down_proj,tp_allreduce}. Sparse layer: the same front end plus attn/{index_branch,index_k_write,
ag_kv,ag_index_k,indexer,sparse_sdpa} and mlp/{shared_expert,router_topk,routing_setup,dispatch,
experts_mm,combine,reduce_ws_rs,tp_allgather,add_shared}. The MSA cache read is the ag_kv + ag_index_k
gathers (high_bw_all_gather straight from the cache slot); there is no separate cache_read zone.

Tokens come from a REAL golden trace's metadata.json (tiled to length, exactly like
scripts/run_prefill_perf.sh's make_trace): MoE expert routing is content-dependent, so random token ids would
give an unrealistically uniform expert load and mis-measure dispatch / experts_mm / combine.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json with token_ids) — tokens are tiled to the
                      required length; no kv_cache/ needed                          [required]
  PROFILE_CHUNK       tokens per chunk (the profiled chunk's width)                   [default 5120]
  PROFILE_CACHE       tokens already in the cache before the profiled chunk; rounded
                      DOWN to a multiple of PROFILE_CHUNK                            [default 25600]
  PROFILE_NUM_LAYERS  build/run only the first N layers (keep >=4 to cover both
                      classes; also sets M3_LOAD_NLAYERS)                            [default: all 60]
  PROFILE_LAYER_IDS   explicit global layer indices, e.g. "0,3" = one dense + one sparse. The fastest
                      way to cover both classes; overrides PROFILE_NUM_LAYERS. Cache-only.
  PROFILE_READ_EVERY  call ttnn.ReadDeviceProfiler every N layers (<1000 ops/read!)   [default 1]
  PROFILE_N_REAL      real tokens in the profiled chunk; the rest is pad (actual_end < chunk end)
                      [default: PROFILE_CHUNK]
  PROFILE_SKIP_COMPILE "1" -> skip runtime.compile()'s sweep over every cache bucket and warm only the
                      profiled chunk. Keeps the tracy capture small at deep caches   [default 0]
  PROFILE_SKIP_PREFIX "1" -> skip the prefix fill and attend a ZEROED cache. Shapes (and op costs)
                      are identical but MoE routing is not representative — bring-up only  [default 0]
  PROFILE_PREFIX_QUIET "1" -> real prefix, small capture: only the last two forwards reach the capture.
                      Unset TTNN_OP_PROFILER (no op records) and mute the zones until the profiled
                      forward; drain the device profiler once per un-profiled forward instead of per layer,
                      each drain analysed and released (TT_METAL_PROFILER_MID_RUN_DUMP); no per-core device
                      log (TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES), no device zones in the tracy file
                      (TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY); delete the prefix's rows of
                      .logs/cpp_device_perf_report.csv, then run one recorded (still un-zoned) warm forward
                      before the profiled one, so op-metadata serialisation stays out of its op-to-op gaps.
                      Needs the C++ post-process (the tracy default); excludes --collect-noc-traces. Also mutes
                      the real-time profiler's Tracy lanes for the whole run (unregisters its program callback):
                      they add one dynamic source location per program execution (Program_<runtime_id>, ~400 per
                      7-layer forward), and tracy-capture aborts at 32K of them ("Too many source locations"),
                      i.e. after ~80 forwards. The ops report does not use those lanes             [default 0]
  PROFILE_PREFIX_READ_EVERY  with PROFILE_PREFIX_QUIET: drain every N un-profiled forwards; 0 = never: the
                      device buffer overflows and drops the later prefix markers, and the one drain before
                      the recorded warm forward reads at most one buffer. Every marker read becomes a host
                      zone in the .tracy (readDeviceMarkerData), so 0 keeps a deep prefix's capture the size
                      of a shallow one. Size TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT to hold the last two
                      forwards (the recorded warm-up + the profiled one), no more       [default 1]
  PROFILE_PROGRESS_EVERY  N > 0: log a line every N un-profiled forwards (watchdog heartbeat)  [default 0]
  PROFILE_WARM_ITERS  warm forwards of the profiled chunk / composition before it is profiled
                      (PROFILE_SKIP_COMPILE=1 or PROFILE_SEGMENTS)                              [default 2]
  PROFILE_WARM_POINT  N > 0: repeat the process's 2nd forward (the first cache-read one) N more times
                      before going deeper — the warm point that avoids the W=8192 "slow mode"    [default 0]
  PROFILE_SEGMENTS    profile ONE packed forward (TtPrefillRuntime.prefill_segments, 2048-token segments)
                      instead of one chunk, e.g. "prose@141312:2048,code@0:2048". Syntax of budget_packed.py's
                      BUDGET_COMPOS entries: comma-separated entries map to slots 0..; "+" keeps consecutive
                      segments in one slot; an optional "X@" prefix picks the tokens: an integer = stream X of
                      the default input, a name = that PROFILE_INPUTS input from its start (stream 0),
                      "name.s" = both.
                      Each slot's history before its first segment is filled with real tokens through packed
                      forwards of that slot (padded with cold segments of a scratch slot), as budget_packed
                      does. Overrides PROFILE_CHUNK / PROFILE_CACHE / PROFILE_N_REAL; never runs compile()
  PROFILE_INPUTS      named token sources for PROFILE_SEGMENTS: "prose=<metadata.json>;code=<metadata.json>"
                      (a directory means its metadata.json). "default" = PREFILL_TRACE_DIR. Token stream s at
                      position p of an input is token_ids[(p + 7919*s) % len], as in budget_packed.py
  PROFILE_STAGES      intra-galaxy pipeline depth: 1 (whole 8x4 galaxy), 2 ((4,4) sub-meshes, EP16) or
                      4 ((2,4) sub-meshes, EP8). The galaxy is opened whole and stage PROFILE_STAGE's
                      sub-mesh is carved out of it, so one process profiles one stage           [default 1]
  PROFILE_MESH        SPxTP sub-mesh shape, e.g. 4x2; overrides PROFILE_STAGES with (8/SP)*(4/TP) and carves
                      create_submeshes(MeshShape(SP, TP))[PROFILE_STAGE]. TP != 4 needs the multi-head KV cache
                      of m3_budget_study/results_ops/tools/profile_4x2.py (run through it)       [default unset]
  PROFILE_PARENT_MESH RxC mesh to open instead of the 8x4 galaxy, e.g. 4x4 on the middle-rows sub-torus
                      (TT_VISIBLE_DEVICES + a matching TT_MESH_GRAPH_DESC_PATH). The sub-mesh is then
                      create_submeshes(PROFILE_MESH)[PROFILE_SUBMESH]; PROFILE_STAGE still picks the layers
                      of the 8x4 stage split                                                     [default 8x4]
  PROFILE_SUBMESH     sub-mesh index inside PROFILE_PARENT_MESH                   [default PROFILE_STAGE]
  PROFILE_STAGE       which stage to profile, 0..PROFILE_STAGES-1. Stage k owns global layers
                      [k*60/S, (k+1)*60/S); PROFILE_LAYER_IDS must fall inside that range and
                      PROFILE_NUM_LAYERS takes the first N of it                             [default 0]
  M3_FABRIC           fabric config: 1d | 1d_ring | 2d | 2d_torus_xy (utils/fabric_env.py)     [default 1d]
  M3_CCL_TOPOLOGY     legacy-CCL topology: linear | ring (ring needs a ring/torus fabric)   [default linear]
  M3_MOE_TOPOLOGY     MoE axis-0 dispatch / v1 combine topology: linear | ring             [default linear]
  M3_MOE_COMBINE      MoE combine: v1 | v2 (combine_fabric2d; needs M3_FABRIC=2d_torus_xy) [default v1]
  M3_MOE_DISPATCH     MoE dispatch: v1 | v2 (dispatch_fabric2d; needs M3_FABRIC=2d_torus_xy) [default v1]
  M3_MOE_LOAD_STATS   1 = log an M3_MOE_LOAD per-expert load line per MoE layer (host sync)  [default off]
  M3_MOE_LOAD_STATS_FILE  also append the raw per-expert counts there as JSON lines           [default unset]
  M3_MOE_W_NDSHARD    1 = routed-expert weights DRAM ND-sharded, 0 = DRAM-interleaved       [default 0]
  M3_MOE_HYBRID_THRESHOLD  T > 0: experts with <= T tokens run moe_fused_swiglu (M3 measured 128) [default 0 = off]
  EXPERT_DTYPE        MoE routed-expert weight dtype: "bf4" or "bf8"                  [default bf4]
  HF_MODEL            real MiniMax-M3 weights dir (read by ModelArgs)
  M3_PROFILE_ZONES    set to 1 by this script before the model is imported

Prefer the wrapper, which handles the venv, tt-smi -glx_reset, trace synthesis and logging the same
way run_prefill_perf.sh does:

  ./models/demos/minimax_m3/scripts/run_prefill_profile.sh                    # both 5k@25k and 5k@55k
  PROFILE_CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh

Manual equivalent:
  cd $TT_METAL_HOME && source python_env/bin/activate && export PYTHONPATH=$TT_METAL_HOME
  export HF_MODEL=/mnt/weka/model-weights/llm/minimax/MiniMax-M3
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  PROFILE_CACHE=25600 PREFILL_TRACE_DIR=<golden> \
    python3 -m tracy -v -r -p models/demos/minimax_m3/tests/perf/profile_prefill.py

Add --collect-noc-traces to the tracy invocation for measured DRAM BW UTIL (%) / NOC UTIL (%) per op
(requires tt-npe installed); the parser picks those columns up automatically when present.

Smoke test without a device (chunk math + token tiling only, no model build):
  PROFILE_DRY_RUN=1 PREFILL_TRACE_DIR=<golden> python3 .../profile_prefill.py
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

# Zones are read at import time by utils/profiler_utils, and the model modules import it, so the flag
# must be set before anything under models.demos.minimax_m3.tt is imported.
os.environ.setdefault("M3_PROFILE_ZONES", "1")
# The programmatic per-program perf API (ttnn.get_latest_programs_perf_data) needs these; harmless when
# unused, and they make mid-run ReadDeviceProfiler calls actually flush.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")

# PROFILE_PREFIX_QUIET: `python -m tracy -r` sets TTNN_OP_PROFILER=1, which makes every enqueued op send its
# metadata to the capture. The C++ side latches the flag the first time an op sees it set, so it is removed
# here, before any op runs, and restored right before the profiled forward. The rtoptions below are read when
# the device opens, so they must be set before that too. Without MID_RUN_DUMP a ReadDeviceProfiler only moves
# the device markers into host RAM and everything is post-processed at close (the whole prefix at once);
# with it, every read is analysed into cpp_device_perf_report.csv and released.
PREFIX_QUIET = os.getenv("PROFILE_PREFIX_QUIET", "0") == "1"
_OP_PROFILER_ENV = os.environ.get("TTNN_OP_PROFILER")
if PREFIX_QUIET:
    os.environ.pop("TTNN_OP_PROFILER", None)
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES", "1")
    os.environ.setdefault("TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE  # noqa: E402
from models.demos.minimax_m3.utils.fabric_env import (  # noqa: E402
    ccl_topology_from_env,
    fabric_config_from_env,
    set_fabric_config_from_env,
)


def _raise_nproc_limit():
    """tt-metal JIT-compiles device kernels in parallel and each target spawns its own chain of
    short-lived processes (g++, cc1plus/lto1, as, ld); a low RLIMIT_NPROC makes clone3 fail
    mid-build ("posix_spawn: Operation not permitted"). Raise the soft limit to the hard limit.
    Copied from galaxy_prefill_kv_pcc.py."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[zone-prof] raised RLIMIT_NPROC soft {soft} -> {hard}")
        except (ValueError, OSError) as e:
            print(f"[zone-prof] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr)


# MSA layers pick the top-16 of 128-token blocks; topk_large_indices aborts with fewer than 16 blocks,
# so a chunk must cover at least this many tokens. Keep in sync with tt/attention/msa.py.
MSA_MIN_TOKENS = 16 * 128  # 2048

# sparse_attention_freq marks layers 0-2 dense and 3-59 sparse (tt/layer.py).
FIRST_SPARSE_LAYER = 3


def load_tokens(n: int):
    """Read PREFILL_TRACE_DIR/metadata.json's token_ids and tile them to exactly `n` tokens.

    Real tokens, not random ones: MoE routing is content-dependent, so the expert load imbalance (and
    with it the dispatch / experts_mm / combine cost) is only realistic with real text. Tiling matches
    scripts/run_prefill_perf.sh's make_trace, so a profile is comparable to the perf sweep's numbers.
    """
    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not trace_dir:
        raise SystemExit(
            "ERROR: set PREFILL_TRACE_DIR to a golden trace dir (a metadata.json with token_ids).\n"
            "       Use models/demos/minimax_m3/scripts/run_prefill_profile.sh, which synthesizes\n"
            "       one the same way run_prefill_perf.sh does."
        )
    src = json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"]
    assert src, f"source trace {trace_dir} has no tokens"
    print(f"[zone-prof] tokens: {len(src)} real tokens from {trace_dir}, tiled to {n}", flush=True)
    return [src[i % len(src)] for i in range(n)]


def plan(chunk: int, cache: int):
    """Resolve the chunk schedule for "one `chunk`-token chunk attending `cache` cached tokens".

    Returns (n_chunks, cache_aligned, total). The cache depth is rounded DOWN to a whole number of
    chunks (the runtime fills the cache one chunk at a time, so a partial prefix is not reachable),
    and `total` is the cache capacity the KV cache must be allocated for.
    """
    assert chunk % 1024 == 0, f"chunk ({chunk}) must be a multiple of 1024 (MSA needs S%1024==0)"
    assert chunk >= MSA_MIN_TOKENS, f"chunk ({chunk}) below the MSA floor {MSA_MIN_TOKENS}"
    n_prefix = cache // chunk
    cache_aligned = n_prefix * chunk
    n_chunks = n_prefix + 1
    return n_chunks, cache_aligned, n_chunks * chunk


SEG = 2048  # packed-forward segment size (budget_packed.py)
STREAM_STRIDE = 7919  # token-stream offset, as in budget_packed.py


def load_inputs():
    """PROFILE_INPUTS ("name=path;...") + "default" (PREFILL_TRACE_DIR) -> {name: token_ids}."""
    specs = {}
    if os.getenv("PREFILL_TRACE_DIR"):
        specs["default"] = os.environ["PREFILL_TRACE_DIR"]
    for item in os.getenv("PROFILE_INPUTS", "").split(";"):
        if item.strip():
            name, _, path = item.partition("=")
            specs[name.strip()] = path.strip()
    inputs = {}
    for name, path in specs.items():
        p = Path(path)
        p = p / "metadata.json" if p.is_dir() else p
        inputs[name] = json.load(open(p))["token_ids"]
        assert inputs[name], f"input {name} ({p}) has no tokens"
        print(f"[zone-prof] input {name}: {len(inputs[name])} tokens from {p}", flush=True)
    return inputs


def parse_segments(spec, inputs):
    """'prose@141312:2048,code@0:2048' -> [(slot, input, stream, h, n), ...] in forward order.

    Same grammar as budget_packed.parse_compo, with the "X@" token selector extended to input names."""
    segs = []
    for slot, entry in enumerate(spec.split(",")):
        name, stream = "default", slot
        if "@" in entry:
            sel, entry = entry.split("@", 1)
            head, _, tail = sel.partition(".")
            if head.isdigit():
                stream = int(head)
            else:
                name = head
                stream = int(tail) if tail else 0  # a named input reads as the document itself
        assert name in inputs, f"PROFILE_SEGMENTS input {name!r} unknown (have {sorted(inputs)}; see PROFILE_INPUTS)"
        for part in entry.split("+"):
            h, _, n = part.partition(":")
            h, n = int(h), int(n or SEG)
            assert h % SEG == 0 and 0 < n <= SEG, f"bad segment {part!r} (h % {SEG} == 0, 0 < n <= {SEG})"
            segs.append((slot, name, stream, h, n))
    return segs


def segment_tokens(inputs, name, stream, p):
    src = inputs[name]
    return [src[(p + STREAM_STRIDE * stream + i) % len(src)] for i in range(SEG)]


def mute_realtime_profiler_tracy():
    """Unregister the real-time profiler's Tracy callback(s) (see PROFILE_PREFIX_QUIET).

    Callback handles are sequential from 0 and, in this process, only the runtime's own Tracy handler registers
    one (when the mesh opens), so every handle below a fresh probe's belongs to it."""
    probe = ttnn.device.RegisterProgramRealtimeProfilerCallback(lambda batch: None)
    ttnn.device.UnregisterProgramRealtimeProfilerCallback(probe)
    for handle in range(probe):
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    print(f"[zone-prof] real-time profiler Tracy lanes muted ({probe} callback(s) unregistered)", flush=True)


def drop_device_perf_report():
    """Delete the C++ per-program report the un-profiled forwards appended to. The runtime re-creates it
    (with its header) on the next profiler read, so the report tracy -r joins holds only what follows."""
    root = os.environ.get("TT_METAL_PROFILER_DIR") or os.path.join(
        os.environ.get("TT_METAL_HOME", "."), "generated/profiler"
    )
    path = Path(root) / ".logs" / "cpp_device_perf_report.csv"
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        print(f"[zone-prof] dropped {size / 2**20:.1f} MiB of un-profiled rows: {path}", flush=True)
    else:
        print(f"[zone-prof] WARNING: no {path} to drop (TT_METAL_PROFILER_CPP_POST_PROCESS unset?)", flush=True)


def build_runtime(
    mesh, chunk, total, num_layers_override, layer_ids=None, stages=1, stage=0, segment_size=None, num_users=1
):
    """Build the real-weights model + KV cache for pipeline stage `stage` of `stages` on `mesh` (the
    whole galaxy or one carved sub-mesh). Returns (runtime, kv_cache, hf_config, global_layer_indices).
    segment_size / num_users: packed forwards (PROFILE_SEGMENTS)."""
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL; the tilized-cache path is keyed by mesh.shape
    hf_config = model_args.hf_config
    total_layers = hf_config.num_hidden_layers
    assert total_layers % stages == 0, f"{total_layers} layers do not split evenly into {stages} stages"
    per_stage = total_layers // stages
    stage_first, stage_end = stage * per_stage, (stage + 1) * per_stage
    is_last_stage = stage == stages - 1
    first_layer_idx = stage_first
    num_layers = per_stage
    if layer_ids:
        # Explicit layer selection, e.g. [0, 3] = one dense + one sparse. The layers keep their real
        # global indices (so weights, cache keys and the dense/sparse decision are the real ones) but
        # are stacked back to back, which makes a 2-layer run cover both classes. Needs the tilized
        # cache: a non-contiguous index may not live in the shards M3_LOAD_NLAYERS would read.
        outside = [i for i in layer_ids if not stage_first <= i < stage_end]
        assert (
            not outside
        ), f"PROFILE_LAYER_IDS {outside} lie outside stage {stage}/{stages}'s layers [{stage_first}, {stage_end})"
        num_layers = len(layer_ids)
        os.environ.setdefault("M3_WEIGHTS_FROM_CACHE", "1")
        print(f"[zone-prof] PROFILE_LAYER_IDS={layer_ids}: building global layers {layer_ids}", flush=True)
    elif num_layers_override:
        num_layers = int(num_layers_override)
        assert num_layers <= per_stage, f"PROFILE_NUM_LAYERS={num_layers} exceeds the {per_stage} layers of a stage"
        os.environ["M3_LOAD_NLAYERS"] = str(num_layers)
        os.environ["M3_LOAD_LAYER_START"] = str(first_layer_idx)
        print(
            f"[zone-prof] PROFILE_NUM_LAYERS={num_layers}: global layers "
            f"[{first_layer_idx}, {first_layer_idx + num_layers}) only",
            flush=True,
        )
        if first_layer_idx + num_layers <= FIRST_SPARSE_LAYER:
            print(
                f"[zone-prof] WARNING: layers [{first_layer_idx}, {first_layer_idx + num_layers}) cover no sparse "
                f"layer (layers 0-{FIRST_SPARSE_LAYER - 1} are dense) — use >={FIRST_SPARSE_LAYER + 1} to profile "
                "both classes.",
                flush=True,
            )
    hf_config.num_hidden_layers = num_layers
    global_layer_indices = list(layer_ids or range(first_layer_idx, first_layer_idx + num_layers))
    if stages > 1:
        print(
            f"[zone-prof] stage {stage}/{stages}: mesh {tuple(mesh.shape)}, stage layers [{stage_first}, {stage_end}), "
            f"building {global_layer_indices}",
            flush=True,
        )

    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
    # Real bf16 source is ~869GB; every weight module loads its tilized tensor from the per-tensor cache
    # via ttnn.as_tensor(cache_file_name=), so on a complete cache we pass an EMPTY state_dict and never
    # read the source. Same trick as galaxy_prefill_kv_pcc.py / DeepSeek.
    force_load = os.getenv("M3_FORCE_LOAD_WEIGHTS") == "1"
    cache_only = not force_load and (
        os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
        or weight_cache_is_complete(
            cache_path,
            hf_config,
            num_layers,
            expert_dtype,
            first_layer_idx=first_layer_idx,
            is_first_rank=True,
            is_last_rank=is_last_stage,
        )
    )
    if cache_only:
        print("[zone-prof] tilized weight cache complete -> loading from cache", flush=True)
        state_dict = {}
    else:
        print("[zone-prof] loading real bf16 weights (slow: ~869GB source read) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)

    # Every stage embeds its own tokens (is_first_rank=True): there is no upstream stage to hand over a
    # hidden state, and the zero placeholder a real middle rank warms up with would collapse the MoE
    # router onto one expert per layer. The embedding runs outside the layer zones, so the per-layer
    # report is unaffected. The tail (final norm + LM head) follows the real stage layout.
    cfg = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=total,
        mesh_shape=tuple(mesh.shape),
        chunk_size=chunk,
        segment_size=segment_size,
        num_users=num_users,
        expert_weight_dtype=expert_dtype,
        weight_cache_path=cache_path,
        first_layer_idx=first_layer_idx,
        layer_indices=layer_ids,
        topology=ccl_topology_from_env(),
        is_first_rank=True,
        is_last_rank=is_last_stage,
    )
    runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
    del state_dict

    kv_cache = allocate_kv_caches(
        mesh, num_layers=num_layers, max_seq_len=total, num_users=num_users, head_dim=hf_config.head_dim
    )
    return runtime, kv_cache, hf_config, global_layer_indices


def parent_mesh_from_env(stage):
    """(PROFILE_PARENT_MESH shape, sub-mesh index): the mesh opened and which create_submeshes tile to use."""
    env = os.getenv("PROFILE_PARENT_MESH", "").strip().lower()
    shape = tuple(int(x) for x in env.split("x")) if env else (8, 4)
    return shape, int(os.getenv("PROFILE_SUBMESH", str(stage)))


def main():
    _raise_nproc_limit()

    chunk = int(os.getenv("PROFILE_CHUNK", "5120"))
    cache_req = int(os.getenv("PROFILE_CACHE", "25600"))
    read_every = int(os.getenv("PROFILE_READ_EVERY", "1"))
    num_layers_override = os.getenv("PROFILE_NUM_LAYERS")
    layer_ids = [int(x) for x in os.getenv("PROFILE_LAYER_IDS", "").split(",") if x.strip()] or None
    stages = int(os.getenv("PROFILE_STAGES", "1"))
    stage = int(os.getenv("PROFILE_STAGE", "0"))
    assert stages in (1, 2, 4), f"PROFILE_STAGES must be 1, 2 or 4 (got {stages})"
    sub_shape = (8 // stages, 4)
    mesh_env = os.getenv("PROFILE_MESH", "").strip().lower()
    if mesh_env:
        sub_shape = tuple(int(x) for x in mesh_env.split("x"))
        assert 8 % sub_shape[0] == 0 and 4 % sub_shape[1] == 0, f"PROFILE_MESH={mesh_env} does not tile the 8x4 galaxy"
        stages = (8 // sub_shape[0]) * (4 // sub_shape[1])
    assert 0 <= stage < stages, f"PROFILE_STAGE={stage} out of range for {stages} stages"
    parent_shape, submesh_idx = parent_mesh_from_env(stage)
    fabric_config = fabric_config_from_env()
    warm_iters = int(os.getenv("PROFILE_WARM_ITERS", "2"))
    warm_point = int(os.getenv("PROFILE_WARM_POINT", "0"))
    prefix_read_every = max(0, int(os.getenv("PROFILE_PREFIX_READ_EVERY", "1")))
    progress_every = int(os.getenv("PROFILE_PROGRESS_EVERY", "0"))
    skip_prefix = os.getenv("PROFILE_SKIP_PREFIX") == "1"
    skip_compile = os.getenv("PROFILE_SKIP_COMPILE") == "1"

    seg_spec = os.getenv("PROFILE_SEGMENTS", "").strip()
    packed = bool(seg_spec)
    if packed:
        inputs = load_inputs()
        segs = parse_segments(seg_spec, inputs)
        n_slots = max(s for s, *_ in segs) + 1
        scratch = n_slots  # cold filler segments of the fill forwards
        chunk = len(segs) * SEG
        total = -(-max(h + SEG for *_, h, _ in segs) // SEG) * SEG
        cache = max(h for *_, h, _ in segs)
        fills = {}  # slot -> (input, stream, h) of its first segment: history [0, h) is filled before
        for slot, name, stream, h, _ in segs:
            fills.setdefault(slot, (name, stream, h))
        n_fill = sum(-(-h // SEG) for _, _, h in fills.values())
        print(
            f"[zone-prof] PROFILING one packed forward: {len(segs)} x {SEG} = {chunk} tokens, {n_slots} slot(s) + 1 "
            f"scratch, capacity {total}; segments (slot, input, stream, h, n) = {segs}; history fill {n_fill} "
            f"segments ({'skipped: PROFILE_SKIP_PREFIX=1' if skip_prefix else 'real tokens'})",
            flush=True,
        )
    else:
        n_chunks, cache, total = plan(chunk, cache_req)
        print(
            f"[zone-prof] PROFILING one {chunk}-token chunk attending {cache} cached tokens "
            f"({n_chunks} chunks total, cache capacity {total})"
            + (f"  [requested cache {cache_req} -> aligned down to {cache}]" if cache != cache_req else ""),
            flush=True,
        )
    if PREFIX_QUIET:
        print(
            "[zone-prof] PROFILE_PREFIX_QUIET=1: op records + zones off until the profiled forward, "
            + (
                f"one profiler drain per {prefix_read_every} un-profiled forward(s)"
                if prefix_read_every
                else "no profiler drain in the un-profiled forwards (device buffer overflows, prefix markers dropped)"
            )
            + ", no device log / tracy device zones",
            flush=True,
        )
        if os.getenv("TT_METAL_DEVICE_PROFILER_NOC_EVENTS") == "1":
            raise SystemExit("ERROR: PROFILE_PREFIX_QUIET=1 disables the device log files NOC traces are written with")
    if os.getenv("PROFILE_DRY_RUN") == "1":
        if not packed:
            load_tokens(total)
        print("[zone-prof] PROFILE_DRY_RUN=1 -> chunk math + tokens only, exiting before device open", flush=True)
        return 0

    # TODO(profiling): the pipeline runner's intra-galaxy bindings use 2D fabric (PREFILL_FABRIC_MODE=2d);
    # 1d is the default here so stage captures compare like-for-like with the whole-galaxy baseline.
    # 2d / 2d_torus_xy are wired through but not yet validated on a carved sub-mesh (torus also needs the
    # matching *_torus_xy mesh graph descriptor). M3_CCL_TOPOLOGY=Ring puts the legacy CCLs on the ring
    # (measured in PR #55668); high_bw_all_gather derives its own from the fabric.
    set_fabric_config_from_env(fabric_config)
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(*parent_shape), l1_small_size=L1_SMALL_SIZE)
    print(
        f"[zone-prof] galaxy opened {tuple(galaxy.shape)} ndev={galaxy.get_num_devices()} fabric={fabric_config} "
        f"ccl_topology={ccl_topology_from_env()}",
        flush=True,
    )
    mesh = galaxy
    try:
        from models.demos.minimax_m3.utils.profiler_utils import (
            COARSE,
            ZONES_ENABLED,
            read_profiler,
            set_zones_active,
            zone,
        )

        if stages > 1 and tuple(sub_shape) != tuple(parent_shape):
            # Row-major tiles of the grid (row blocks for the default (8/S, 4) shape), in the same order the
            # pipeline bindings assign stages.
            mesh = galaxy.create_submeshes(ttnn.MeshShape(*sub_shape))[submesh_idx]
            print(
                f"[zone-prof] stage {stage}/{stages} sub-mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()} "
                f"(tile {submesh_idx} of {tuple(parent_shape)})",
                flush=True,
            )
        sp, tp = tuple(mesh.shape)

        runtime, kv_cache, hf_config, global_layer_indices = build_runtime(
            mesh,
            chunk,
            total,
            num_layers_override,
            layer_ids,
            stages=stages,
            stage=stage,
            segment_size=SEG if packed else None,
            num_users=n_slots + 1 if packed else 1,
        )
        num_layers = len(global_layer_indices)
        if PREFIX_QUIET:
            mute_realtime_profiler_tracy()  # after every mesh / sub-mesh is open, before the first forward

        # Per-layer ReadDeviceProfiler for the UN-profiled phases only (warmup + prefix). The device
        # profiler buffer must be drained or it overflows and the next phase's data is dropped — but a
        # drain is a blocking device sync + PCIe pull, and it lands in the trace as a multi-second
        # OP TO OP LATENCY on the next op. Draining inside the profiled chunk therefore destroys the
        # one measurement that explains where wall-clock goes (kernel time is unaffected, the gaps are
        # not). So: drain freely before the chunk, go silent during it, flush once after.
        #
        # That means the profiled chunk's ops must all fit in the buffer at once
        # (num_layers x ~72 ops). Size it with TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT — the default is
        # only 1000 (tt_metal/impl/profiler/profiler_state_manager.cpp).
        #
        # PROFILE_PREFIX_QUIET drains once per un-profiled forward instead (see unprofiled() below): a
        # forward is ~num_layers x 75 programs, far below the 20000 the wrapper sizes the buffer for.
        read_in_chunk = os.getenv("PROFILE_READ_IN_CHUNK", "0") == "1"
        state = {"reads": 0, "in_chunk": False, "fwd": 0}

        def on_layer_complete(layer_idx):
            if state["in_chunk"] and not read_in_chunk:
                return
            if PREFIX_QUIET and not state["in_chunk"]:
                return
            if read_every > 0 and (layer_idx + 1) % read_every == 0:
                read_profiler(mesh)
                state["reads"] += 1

        runtime._on_layer_complete = on_layer_complete

        def unprofiled(fn):
            """Run one un-profiled forward; apply PROFILE_WARM_POINT and the quiet-mode drain."""
            fn()
            state["fwd"] += 1
            if state["fwd"] == 2 and warm_point > 0:
                # The process's first cache-read forward: repeat it (same tokens, same KV positions, so
                # the cache is unchanged) before any deeper one — the budget harness's warm point.
                for _ in range(warm_point):
                    fn()
                print(f"[zone-prof] warm point: forward #2 repeated {warm_point}x", flush=True)
            if PREFIX_QUIET and prefix_read_every > 0 and state["fwd"] % prefix_read_every == 0:
                read_profiler(mesh)
                state["reads"] += 1
            if progress_every > 0 and state["fwd"] % progress_every == 0:
                print(f"[zone-prof] {state['fwd']} un-profiled forwards done", flush=True)

        if PREFIX_QUIET:
            set_zones_active(False)

        # --- 1. WARMUP: JIT-compiles every op and populates the program cache. Its ops land in the CSV
        # too, but outside the `profiled_chunk` zone, so the parser drops them.
        print(f"[zone-prof] warmup / compile ({num_layers}L, SP={sp} x TP={tp} + EP={sp * tp}) ...", flush=True)
        t0 = time.perf_counter()
        if packed and not skip_compile:
            print("[zone-prof] PROFILE_SEGMENTS: runtime.compile() is chunk-path only; warming the composition instead")
        elif not skip_compile:
            runtime.compile(kv_cache)
            if PREFIX_QUIET:
                read_profiler(mesh)
                state["reads"] += 1
        print(f"[zone-prof] warmup done in {(time.perf_counter()-t0):.1f}s", flush=True)

        if packed:

            def run_segments(group):
                """group: [(slot, input, stream, h, n)] -> one prefill_segments forward."""
                inp = runtime.make_segments_input([segment_tokens(inputs, nm, st, h) for _, nm, st, h, _ in group])
                out = runtime.prefill_segments(inp, kv_cache, [(slot, h, n) for slot, _, _, h, n in group])
                if out is not None:
                    out.deallocate(True)

            B = len(segs)
            profiled = lambda: run_segments(segs)
            warm = profiled
            if skip_prefix:
                print(
                    f"[zone-prof] PROFILE_SKIP_PREFIX=1 -> skipping the {n_fill}-segment history fill; attention "
                    "reads a ZEROED cache (shapes real, MoE routing NOT representative)",
                    flush=True,
                )
            elif n_fill:
                print(f"[zone-prof] filling history: {n_fill} segments in forwards of {B} ...", flush=True)
                t0 = time.perf_counter()
                fill_fwds = 0
                for slot, (name, stream, h) in fills.items():
                    todo = [(slot, name, stream, p, SEG) for p in range(0, h, SEG)]
                    while todo:
                        group, todo = todo[:B], todo[B:]
                        group += [(scratch, "default" if "default" in inputs else name, 0, 0, SEG)] * (B - len(group))
                        unprofiled(lambda g=group: run_segments(g))
                        fill_fwds += 1
                ttnn.synchronize_device(mesh)
                print(
                    f"[zone-prof] history filled in {(time.perf_counter()-t0):.1f}s ({fill_fwds} forwards)", flush=True
                )
            n_warm = warm_iters
        else:
            tokens = load_tokens(total)

            n_real = int(os.getenv("PROFILE_N_REAL", str(chunk)))
            assert 0 < n_real <= chunk, f"PROFILE_N_REAL={n_real} must be in (0, {chunk}]"

            def prefill_chunk(c, n=chunk):
                a = c * chunk
                inp = runtime.make_chunk_input(tokens[a : a + chunk])
                out = runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=a, actual_end=a + n)
                if out is not None:  # a non-last stage returns the hidden state meant for the next stage
                    out.deallocate(True)

            profiled = lambda: prefill_chunk(n_chunks - 1, n_real)
            warm = lambda: prefill_chunk(n_chunks - 1)

            # --- 2. fill the cache to `cache` tokens. Not inside the `profiled_chunk` zone, so these ops are
            # excluded from the report; synced before the profiled chunk so it pays for no leftover barrier.
            if skip_prefix:
                # FAST/APPROXIMATE: run the profiled chunk at actual_start=`cache` against a still-ZEROED
                # cache. Shapes (and therefore every op's cost) are identical, but the attention outputs are
                # garbage, so the hidden states feeding the MoE router are unrealistic -> the expert load
                # imbalance (dispatch / experts_mm / combine) is NOT representative. Use for bring-up only.
                print(
                    f"[zone-prof] PROFILE_SKIP_PREFIX=1 -> skipping the {n_chunks-1}-chunk prefix fill; "
                    f"attention reads a ZEROED cache (shapes real, MoE routing NOT representative)",
                    flush=True,
                )
            elif n_chunks > 1:
                print(f"[zone-prof] pre-filling {n_chunks-1} chunks -> {cache} cached tokens ...", flush=True)
                t0 = time.perf_counter()
                for c in range(n_chunks - 1):
                    unprofiled(lambda c=c: prefill_chunk(c))
                ttnn.synchronize_device(mesh)
                print(f"[zone-prof] prefix filled in {(time.perf_counter()-t0):.1f}s", flush=True)
            n_warm = warm_iters if skip_compile else 0

        # No bucket sweep: warm the profiled forward's own programs instead (twice, like the timing harness).
        for _ in range(n_warm):
            unprofiled(warm)
        if n_warm:
            ttnn.synchronize_device(mesh)
        if not packed and n_real < chunk:
            # A short chunk builds a different MoE padding config: warm it here, not inside the profile.
            unprofiled(profiled)
            ttnn.synchronize_device(mesh)

        if PREFIX_QUIET:
            # Everything before this line is out of the report: drain it and drop the per-program rows it
            # left in the C++ report. Then turn op records back on for ONE more un-profiled forward of the
            # profiled programs: the first time an op is recorded its metadata is serialised in full (slow
            # host work), later records reuse it, so this keeps that cost out of the profiled forward's
            # op-to-op gaps. Its ops and device rows are both kept (tracy -r needs device rows for every
            # recorded op) and it runs outside the profiled_chunk zone, so the parser drops it.
            ttnn.synchronize_device(mesh)
            read_profiler(mesh)
            state["reads"] += 1
            drop_device_perf_report()
            if _OP_PROFILER_ENV is not None:
                os.environ["TTNN_OP_PROFILER"] = _OP_PROFILER_ENV
            unprofiled(profiled)
            ttnn.synchronize_device(mesh)
            set_zones_active(True)

        # --- 3. the profiled forward, bracketed by the `profiled_chunk` zone. Everything the parser
        # reports is nested under it, which is what separates it from warmup + prefix.
        read_note = (
            "per-layer reads INSIDE the chunk — op-to-op latency will be meaningless"
            if read_in_chunk
            else "no reads inside the chunk — op-to-op latency is clean"
        )
        what = f"packed {len(segs)}x{SEG} forward" if packed else f"final chunk: {chunk} tok @ {cache} cache"
        print(
            f"[zone-prof] profiling the {what} (zones {'ON' if ZONES_ENABLED else 'OFF'}, {read_note}) ...", flush=True
        )
        prefix_reads = state["reads"]
        state["in_chunk"] = True
        t0 = time.perf_counter()
        with zone("profiled_chunk", COARSE):
            profiled()
            ttnn.synchronize_device(mesh)
        wall = time.perf_counter() - t0
        state["in_chunk"] = False
        read_profiler(mesh)  # single flush of the whole profiled chunk
        chunk_reads = state["reads"] - prefix_reads

        head = (
            f"PROFILED FORWARD: packed {len(segs)} x {SEG} = {chunk} tok, segments {[(s, nm, h, n) for s, nm, _, h, n in segs]}"
            if packed
            else f"PROFILED CHUNK: {chunk} tok @ {cache} cache"
        )
        print(
            f"\n[zone-prof] {head}, {num_layers} layers "
            f"(stage {stage}/{stages}, mesh {sp}x{tp}, fabric {fabric_config})\n"
            f"  wall-clock: {wall*1e3:.1f} ms  ({chunk_reads} profiler reads inside the chunk, "
            f"{prefix_reads} before it)\n"
            f"  device-kernel time per zone: parse the ops CSV with\n"
            f"    python3 models/demos/minimax_m3/tests/perf/parse_zone_perf.py "
            f"<generated/profiler/reports/*/ops_perf_results_*.csv> --html zones.html",
            flush=True,
        )
        print("[zone-prof] DONE", flush=True)
    finally:
        # Sub-mesh first: closing the parent runs a final profiler read on the parent's command queue,
        # and MeshDevice::close then refuses to close a mesh whose child still holds an in-use queue.
        for sub in galaxy.get_submeshes():
            ttnn.close_mesh_device(sub)
        ttnn.close_mesh_device(galaxy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
