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
  PROFILE_SKIP_PREFIX "1" -> skip the prefix fill and attend a ZEROED cache. Shapes (and op costs)
                      are identical but MoE routing is not representative — bring-up only  [default 0]
  PROFILE_STAGES      intra-galaxy pipeline depth: 1 (whole 8x4 galaxy), 2 ((4,4) sub-meshes, EP16) or
                      4 ((2,4) sub-meshes, EP8). The galaxy is opened whole and stage PROFILE_STAGE's
                      sub-mesh is carved out of it, so one process profiles one stage           [default 1]
  PROFILE_STAGE       which stage to profile, 0..PROFILE_STAGES-1. Stage k owns global layers
                      [k*60/S, (k+1)*60/S); PROFILE_LAYER_IDS must fall inside that range and
                      PROFILE_NUM_LAYERS takes the first N of it                             [default 0]
  M3_FABRIC           fabric config: 1d | 1d_ring | 2d | 2d_torus_xy (utils/fabric_env.py)     [default 1d]
  M3_CCL_TOPOLOGY     legacy-CCL topology: linear | ring (ring needs a ring/torus fabric)   [default linear]
  EXPERT_DTYPE        MoE routed-expert weight dtype: "bf4" or "bf8"                  [default bf4]
  HF_MODEL            real MiniMax-M3 weights dir (read by ModelArgs)
  M3_PROFILE_ZONES    set to 1 by this script before the model is imported

Prefer the wrapper, which handles the venv, tt-smi -glx_reset, trace synthesis and logging the same
way run_prefill_perf.sh does:

  ./models/demos/minimax_m3/scripts/run_prefill_profile.sh                    # both 5k@25k and 5k@55k
  PROFILE_CACHE=25600 ./models/demos/minimax_m3/scripts/run_prefill_profile.sh

Manual equivalent:
  cd $TT_METAL_HOME && source python_env/bin/activate && export PYTHONPATH=$TT_METAL_HOME
  export HF_MODEL=/mnt/models/MiniMaxAI/MiniMax-M3-ref
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

from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.demos.minimax_m3.tt.ccl import L1_SMALL_SIZE  # noqa: E402
from models.demos.minimax_m3.utils.fabric_env import ccl_topology_from_env, fabric_config_from_env  # noqa: E402


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


def build_runtime(mesh, chunk, total, num_layers_override, layer_ids=None, stages=1, stage=0):
    """Build the real-weights model + KV cache for pipeline stage `stage` of `stages` on `mesh` (the
    whole galaxy or one carved sub-mesh). Returns (runtime, kv_cache, hf_config, global_layer_indices)."""
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
        num_users=1,
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
        mesh, num_layers=num_layers, max_seq_len=total, num_users=1, head_dim=hf_config.head_dim
    )
    return runtime, kv_cache, hf_config, global_layer_indices


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
    assert 0 <= stage < stages, f"PROFILE_STAGE={stage} out of range for {stages} stages"
    fabric_config = fabric_config_from_env()

    n_chunks, cache, total = plan(chunk, cache_req)
    print(
        f"[zone-prof] PROFILING one {chunk}-token chunk attending {cache} cached tokens "
        f"({n_chunks} chunks total, cache capacity {total})"
        + (f"  [requested cache {cache_req} -> aligned down to {cache}]" if cache != cache_req else ""),
        flush=True,
    )
    if os.getenv("PROFILE_DRY_RUN") == "1":
        load_tokens(total)
        print("[zone-prof] PROFILE_DRY_RUN=1 -> chunk math + tokens only, exiting before device open", flush=True)
        return 0

    # TODO(profiling): the pipeline runner's intra-galaxy bindings use 2D fabric (PREFILL_FABRIC_MODE=2d);
    # 1d is the default here so stage captures compare like-for-like with the whole-galaxy baseline.
    # 2d / 2d_torus_xy are wired through but not yet validated on a carved sub-mesh (torus also needs the
    # matching *_torus_xy mesh graph descriptor). M3_CCL_TOPOLOGY=Ring puts the legacy CCLs on the ring
    # (measured in PR #55668); high_bw_all_gather derives its own from the fabric.
    ttnn.set_fabric_config(fabric_config)
    galaxy = ttnn.open_mesh_device(ttnn.MeshShape(8, 4), l1_small_size=L1_SMALL_SIZE)
    print(
        f"[zone-prof] galaxy opened {tuple(galaxy.shape)} ndev={galaxy.get_num_devices()} fabric={fabric_config} "
        f"ccl_topology={ccl_topology_from_env()}",
        flush=True,
    )
    mesh = galaxy
    try:
        from models.demos.minimax_m3.utils.profiler_utils import COARSE, ZONES_ENABLED, read_profiler, zone

        if stages > 1:
            # Contiguous row-block sub-meshes, in the same order the pipeline bindings assign stages.
            mesh = galaxy.create_submeshes(ttnn.MeshShape(8 // stages, 4))[stage]
            print(
                f"[zone-prof] stage {stage}/{stages} sub-mesh {tuple(mesh.shape)} ndev={mesh.get_num_devices()}",
                flush=True,
            )
        sp, tp = tuple(mesh.shape)

        runtime, kv_cache, hf_config, global_layer_indices = build_runtime(
            mesh, chunk, total, num_layers_override, layer_ids, stages=stages, stage=stage
        )
        num_layers = len(global_layer_indices)

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
        read_in_chunk = os.getenv("PROFILE_READ_IN_CHUNK", "0") == "1"
        state = {"reads": 0, "in_chunk": False}

        def on_layer_complete(layer_idx):
            if state["in_chunk"] and not read_in_chunk:
                return
            if read_every > 0 and (layer_idx + 1) % read_every == 0:
                read_profiler(mesh)
                state["reads"] += 1

        runtime._on_layer_complete = on_layer_complete

        # --- 1. WARMUP: JIT-compiles every op and populates the program cache. Its ops land in the CSV
        # too, but outside the `profiled_chunk` zone, so the parser drops them.
        print(f"[zone-prof] warmup / compile ({num_layers}L, SP={sp} x TP={tp} + EP={sp * tp}) ...", flush=True)
        t0 = time.perf_counter()
        runtime.compile(kv_cache)
        print(f"[zone-prof] warmup done in {(time.perf_counter()-t0):.1f}s", flush=True)

        tokens = load_tokens(total)

        def prefill_chunk(c):
            a = c * chunk
            inp = runtime.make_chunk_input(tokens[a : a + chunk])
            out = runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=a, actual_end=a + chunk)
            if out is not None:  # a non-last stage returns the hidden state meant for the next stage
                out.deallocate(True)

        # --- 2. fill the cache to `cache` tokens. Not inside the `profiled_chunk` zone, so these ops are
        # excluded from the report; synced before the profiled chunk so it pays for no leftover barrier.
        skip_prefix = os.getenv("PROFILE_SKIP_PREFIX") == "1"
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
                prefill_chunk(c)
            ttnn.synchronize_device(mesh)
            print(f"[zone-prof] prefix filled in {(time.perf_counter()-t0):.1f}s", flush=True)

        # --- 3. the profiled chunk, bracketed by the `profiled_chunk` zone. Everything the parser
        # reports is nested under it, which is what separates this chunk from warmup + prefix.
        read_note = (
            "per-layer reads INSIDE the chunk — op-to-op latency will be meaningless"
            if read_in_chunk
            else "no reads inside the chunk — op-to-op latency is clean"
        )
        print(
            f"[zone-prof] profiling the final chunk: {chunk} tok @ {cache} cache "
            f"(zones {'ON' if ZONES_ENABLED else 'OFF'}, {read_note}) ...",
            flush=True,
        )
        prefix_reads = state["reads"]
        state["in_chunk"] = True
        t0 = time.perf_counter()
        with zone("profiled_chunk", COARSE):
            prefill_chunk(n_chunks - 1)
            ttnn.synchronize_device(mesh)
        wall = time.perf_counter() - t0
        state["in_chunk"] = False
        read_profiler(mesh)  # single flush of the whole profiled chunk
        chunk_reads = state["reads"] - prefix_reads

        print(
            f"\n[zone-prof] PROFILED CHUNK: {chunk} tok @ {cache} cache, {num_layers} layers "
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
