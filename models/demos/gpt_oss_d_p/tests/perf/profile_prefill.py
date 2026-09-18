# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS REAL-WEIGHTS chunked-prefill zone profiler: per-zone device time for a sliding and a full
attention layer. Sibling of ``minimax_m3/tests/perf/profile_prefill.py``; the zone tooling itself is
shared (models/demos/common/prefill/profiling).

Measures ONE chunk attending an already-populated KV cache — the "8k attended to 24k" case.
Structure (mirrors galaxy_prefill_kv_pcc.py's LAST CHUNK measurement):

  1. build the real model (SP=4 x TP=8 + EP=32) from real weights or the tilized cache
  2. runtime.compile()          -> WARMUP: JIT-compiles every op, populates the program cache
  3. pre-fill chunks 0..n-2     -> fills the cache to PROFILE_CACHE tokens (NOT profiled; the device
                                   profiler is drained after every layer and every chunk)
  4. profile the FINAL chunk    -> zones on, one signposted region per zone, NO profiler reads inside
                                   the chunk; one flush after its closing marker

Only step 4 is inside the zone markers, so the report is exactly "one 8k chunk against an N-token
cache". GPT-OSS alternates sliding-window (even) and full-causal (odd) attention layers, so ANY two
consecutive layers cover both classes; the report separates them by the layer tag.

PROFILE_CACHE=0 profiles the ONE-SHOT path instead: a single chunk with no cache, which takes the
all-gather + SDPA + reduce-scatter fallback rather than the cache-backed ring SDPA. Profiling both
tells you what the chunked ring path costs relative to one-shot — the genuine ~2x warm overhead that
#52000 tracks (its original ~16x was a first-compile artifact).

What you get per zone: summed DEVICE KERNEL DURATION [ns] per device (with the across-device skew),
op count, bytes moved (from the CSV's input/output shapes + dtypes) and the implied GB/s. Render with
    python3 models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py <ops_perf_results_*.csv>

Zone list — both classes share it (only the attention core differs): input_norm, attn/{qkv_proj,
split_heads,rope,kv_write,ring_joint_sdpa | ag_qkv,sdpa,sdpa_reduce_scatter,concat_heads,o_proj,
ccl_out_allreduce}, residual_attn, post_attn_norm, mlp/{router_topk,routing_setup,dispatch,
experts_mm,combine,moe_reduce,tp_allgather}, residual_mlp.

Tokens come from a REAL golden trace's metadata.json (tiled to length): MoE expert routing is
content-dependent, so random token ids would give an unrealistically uniform expert load and
mis-measure dispatch / experts_mm / combine.

Env:
  PREFILL_TRACE_DIR   golden trace dir (metadata.json with token_ids) — tokens are tiled to the
                      required length; no kv_cache/ needed                          [required]
  PROFILE_CHUNK       tokens per chunk (the profiled chunk's width)                   [default 8192]
  PROFILE_CACHE       tokens already in the cache before the profiled chunk; rounded
                      DOWN to a multiple of PROFILE_CHUNK. 0 = one-shot path         [default 24576]
  PROFILE_NUM_LAYERS  build/run only the first N layers (>=2 covers both classes)   [default: all 36]
  PROFILE_READ_EVERY  drain the device profiler every N layers outside the chunk       [default 1]
  PROFILE_READ_IN_CHUNK "1" -> also drain per layer INSIDE the profiled chunk. Perturbs the inter-op
                      gaps; use it to check whether drains change the CCL device times     [default 0]
  PROFILE_SKIP_PREFIX "1" -> skip the prefix fill and attend a ZEROED cache. Shapes (and op costs)
                      are identical but MoE routing is not representative — bring-up only  [default 0]
  PREFILL_TOPOLOGY    "ring" (default; torus descriptor + FABRIC_1D_RING) or "linear"
  EXPERT_DTYPE        MoE routed-expert weight dtype: "bf4" or "bf8"                  [default bf4]
  KV_CACHE_DTYPE      KV-cache storage dtype: "bf8" or "bf16"                         [default bf8]
  GPT_OSS_WEIGHTS_FROM_CACHE  "1" -> empty state_dict, load tilized weights from the TTNN cache
  HF_MODEL            real gpt-oss-120b weights dir (read by ModelArgs)
  GPTOSS_PROFILE_ZONES / TT_METAL_DEVICE_PROFILER  set to 1 by main() before the model is imported

Prefer the wrapper, which handles the venv, results folders and logging:

  ./models/demos/gpt_oss_d_p/scripts/run_prefill_profile.sh
  CACHE=24576 ./models/demos/gpt_oss_d_p/scripts/run_prefill_profile.sh

Manual equivalent:
  cd $TT_METAL_HOME && source python_env/bin/activate && export PYTHONPATH=$TT_METAL_HOME
  export HF_MODEL=/path/to/gpt-oss-120b
  export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_torus_xy_graph_descriptor.textproto
  export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000   # the profiled chunk must fit the device buffer
  PROFILE_CACHE=24576 PREFILL_TRACE_DIR=<golden> \
    python3 -m tracy -v -r -p --check-exit-code models/demos/gpt_oss_d_p/tests/perf/profile_prefill.py

Add --collect-noc-traces to the tracy invocation for measured DRAM BW UTIL (%) / NOC UTIL (%) per op
(requires tt-npe installed); the parser picks those columns up automatically when present.

Smoke test without a device (chunk math + token tiling only, no model build):
  PROFILE_DRY_RUN=1 PREFILL_TRACE_DIR=<golden> python3 .../profile_prefill.py
"""

import json
import os
import sys
import time
from pathlib import Path

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import raise_nproc_limit
from models.demos.gpt_oss_d_p.tests.galaxy_prefill_kv_pcc import COLS, ROWS, chunk_alignment

# chunk/sp feeds the MoE routing setup (64 Tensix cores) and build_indexed_rope (TILE_SIZE * sp); both
# are satisfied by chunk_alignment(sp) = 256 tokens at sp=4. Same math as galaxy_prefill_kv_pcc.plan().
CHUNK_ALIGN = chunk_alignment(ROWS)


def load_tokens(n: int):
    """Read PREFILL_TRACE_DIR/metadata.json's token_ids and tile them to exactly `n` tokens.

    Real tokens, not random ones: MoE routing is content-dependent, so the expert load imbalance (and
    with it the dispatch / experts_mm / combine cost) is only realistic with real text.
    """
    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not trace_dir:
        raise SystemExit(
            "ERROR: set PREFILL_TRACE_DIR to a golden trace dir (a metadata.json with token_ids).\n"
            "       Use models/demos/gpt_oss_d_p/scripts/run_prefill_profile.sh, which points it at one."
        )
    src = json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"]
    assert src, f"source trace {trace_dir} has no tokens"
    print(f"[zone-prof] tokens: {len(src)} real tokens from {trace_dir}, tiled to {n}", flush=True)
    return [src[i % len(src)] for i in range(n)]


def plan(chunk: int, cache: int):
    """Resolve the chunk schedule for "one `chunk`-token chunk attending `cache` cached tokens".

    Returns (n_chunks, cache_aligned, total). The cache depth is rounded DOWN to a whole number of
    chunks (the runtime fills the cache one chunk at a time, so a partial prefix is not reachable),
    and `total` is the cache capacity the KV cache must be allocated for. cache=0 gives one-shot:
    total == chunk, which routes attention down the all-gather fallback instead of the cache-backed
    ring (see attention/prefill.py use_cache_backed_ring).
    """
    assert chunk % CHUNK_ALIGN == 0, (
        f"chunk ({chunk}) must be a multiple of {CHUNK_ALIGN} "
        f"(MoE routing needs chunk/sp to split across its cores at sp={ROWS})"
    )
    n_prefix = cache // chunk
    cache_aligned = n_prefix * chunk
    n_chunks = n_prefix + 1
    return n_chunks, cache_aligned, n_chunks * chunk


def build_runtime(mesh, chunk, total, num_layers_override, topology):
    """Build the real-weights model (runtime owns its KV cache). Returns (runtime, hf_config, n_layers)."""
    from models.demos.gpt_oss_d_p.tt.model_config import ModelArgs
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    model_args = ModelArgs(mesh_device=mesh)  # HF_MODEL
    hf_config = model_args.hf_config
    num_layers = hf_config.num_hidden_layers
    if num_layers_override:
        num_layers = int(num_layers_override)
        hf_config.num_hidden_layers = num_layers
        print(f"[zone-prof] PROFILE_NUM_LAYERS={num_layers}: first {num_layers} layers only", flush=True)
        if num_layers < 2:
            print(
                "[zone-prof] WARNING: 1 layer covers only the sliding class (layers alternate "
                "sliding/full, even=sliding) — use >=2 to profile both.",
                flush=True,
            )

    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    kv_cache_dtype = ttnn.bfloat16 if os.getenv("KV_CACHE_DTYPE", "bf8") == "bf16" else ttnn.bfloat8_b
    cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)

    # On a complete tilized cache, pass an EMPTY state_dict and skip the safetensors read entirely
    # (same trick as galaxy_prefill_kv_pcc.py). Requires a prior real-weights build to have populated
    # the cache AND the MoE bias sidecar (tt/mlp.py fails loud if the sidecar is missing).
    if os.getenv("GPT_OSS_WEIGHTS_FROM_CACHE") == "1":
        print("[zone-prof] GPT_OSS_WEIGHTS_FROM_CACHE=1 -> empty state_dict (load tilized cache)", flush=True)
        state_dict = {}
    else:
        print("[zone-prof] loading real bf16 weights (slow: safetensors read) ...", flush=True)
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)

    cfg = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=total,
        mesh_shape=(ROWS, COLS),
        default_chunk_size=chunk,
        num_users=1,
        expert_weight_dtype=expert_dtype,
        cache_dtype=kv_cache_dtype,
        weight_cache_path=cache_path,
        owns_kv_cache=True,
        topology=topology,
    )
    runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
    del state_dict
    return runtime, hf_config, num_layers


def cache_traffic_note(hf_config, num_layers, cache, chunk, kv_bytes_per_elem):
    """Log the expected ring cache-read traffic so the measured GB/s is immediately interpretable.

    The cache-backed RingJointSDPA reads the accumulated K/V prefix (cached_len + this chunk) once per
    layer; each chip holds 1 KV head (TP=8 over 8 KV heads) x head_dim columns of the SP-sharded rows.
    """
    rows_read = cache + chunk
    per_chip_rows = rows_read // ROWS
    per_layer = 2 * per_chip_rows * hf_config.head_dim * kv_bytes_per_elem  # K + V
    logger.info(
        f"[zone-prof] expected ring cache-read traffic: {rows_read} K/V rows "
        f"({per_chip_rows}/chip after SP), {per_layer / 2**20:.1f} MiB per layer per chip, "
        f"x {num_layers} layers = {per_layer * num_layers / 2**20:.0f} MiB per chunk per chip"
    )


def main():
    # The zone flag is read when utils/profiler_utils is imported, and the model modules import it —
    # so it goes on before anything under models.demos.gpt_oss_d_p.tt is imported (all of those imports
    # are inside functions below). The device-profiler flag makes mid-run ReadDeviceProfiler calls
    # actually flush. Both live HERE, not at module import: importing this module (the unit tests do,
    # for plan()) must not arm the profiler for the rest of the process.
    os.environ.setdefault("GPTOSS_PROFILE_ZONES", "1")
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    raise_nproc_limit("zone-prof")

    chunk = int(os.getenv("PROFILE_CHUNK", "8192"))
    cache_req = int(os.getenv("PROFILE_CACHE", "24576"))
    read_every = int(os.getenv("PROFILE_READ_EVERY", "1"))
    read_in_chunk = os.getenv("PROFILE_READ_IN_CHUNK", "0") == "1"
    num_layers_override = os.getenv("PROFILE_NUM_LAYERS")
    linear = os.getenv("PREFILL_TOPOLOGY", "ring") == "linear"

    n_chunks, cache, total = plan(chunk, cache_req)
    mode = "one-shot (all-gather fallback)" if cache == 0 else "chunked (cache-backed ring SDPA)"
    print(
        f"[zone-prof] PROFILING one {chunk}-token chunk attending {cache} cached tokens — {mode} "
        f"({n_chunks} chunks total, cache capacity {total})"
        + (f"  [requested cache {cache_req} -> aligned down to {cache}]" if cache != cache_req else ""),
        flush=True,
    )
    if os.getenv("PROFILE_DRY_RUN") == "1":
        load_tokens(total)
        print("[zone-prof] PROFILE_DRY_RUN=1 -> chunk math + tokens only, exiting before device open", flush=True)
        return 0

    # Ring collectives need the cyclic torus route (torus mesh descriptor + FABRIC_1D_RING);
    # PREFILL_TOPOLOGY=linear keeps plain FABRIC_1D for pods without wraparound.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if linear else ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(ROWS, COLS))
    print(f"[zone-prof] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        from models.demos.gpt_oss_d_p.utils.profiler_utils import COARSE, ZONES_ENABLED, read_profiler, zone

        topology = ttnn.Topology.Linear if linear else ttnn.Topology.Ring
        runtime, hf_config, num_layers = build_runtime(mesh, chunk, total, num_layers_override, topology)
        kv_b = 2.0 if os.getenv("KV_CACHE_DTYPE", "bf8") == "bf16" else 1.0625
        if cache > 0:
            cache_traffic_note(hf_config, num_layers, cache, chunk, kv_b)

        # The device profiler buffer holds TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT programs per device
        # (default 1000) and drops everything past that, so it must be drained regularly — but a drain
        # is a blocking device sync + PCIe pull that lands in the trace as a multi-second OP TO OP
        # LATENCY on the next op. Draining inside the profiled chunk therefore destroys the one
        # measurement that explains where wall-clock goes (kernel time is unaffected, the gaps are
        # not). So: drain freely before the chunk (after compile, after every prefix layer via the
        # runtime's layer-completion sink, and after every prefix chunk), go silent during it, flush
        # once after. That means the profiled chunk's ops must all fit in the buffer at once
        # (num_layers x ~45 ops): size it with TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT.
        #
        # PROFILE_READ_IN_CHUNK=1 drains per layer inside the chunk too — an experiment, not a mode: a
        # drain is mesh-wide, so on a ring it can change what a CCL's device time includes (waiting for
        # peers). Comparing its totals against a silent run measures that effect.
        state = {"reads": 0, "in_chunk": False}

        def drain():
            read_profiler(mesh)
            state["reads"] += 1

        def layer_sink(global_layer_idx, request_id):
            if state["in_chunk"] and not read_in_chunk:
                return
            if read_every > 0 and (global_layer_idx + 1) % read_every == 0:
                drain()

        # --- 1. WARMUP: JIT-compiles every op and populates the program cache. Its ops land in the CSV
        # too, but outside the `profiled_chunk` zone, so the parser drops them. The sink can only be
        # registered after compile(), so the warmup itself is covered by the drain right after it.
        print(f"[zone-prof] warmup / compile ({num_layers}L, SP={ROWS} x TP={COLS} + EP=32) ...", flush=True)
        t0 = time.perf_counter()
        runtime.compile()
        drain()
        runtime.set_layer_completion_sink(layer_sink)
        print(f"[zone-prof] warmup done in {(time.perf_counter() - t0):.1f}s", flush=True)

        tokens = load_tokens(total)

        def prefill_chunk(c):
            a = c * chunk
            inp = runtime.make_chunk_input(tokens[a : a + chunk])
            runtime.prefill_chunk(inp, slot_id=0, actual_start=a, actual_end=a + chunk)

        # --- 2. fill the cache to `cache` tokens. Not inside the `profiled_chunk` zone, so these ops are
        # excluded from the report; synced and drained before the profiled chunk so it pays for no
        # leftover barrier and starts with an empty profiler buffer.
        skip_prefix = os.getenv("PROFILE_SKIP_PREFIX") == "1"
        if skip_prefix and n_chunks > 1:
            # FAST/APPROXIMATE: run the profiled chunk at actual_start=`cache` against a still-ZEROED
            # cache. Shapes (and therefore every op's cost) are identical, but the attention outputs are
            # garbage, so the hidden states feeding the MoE router are unrealistic -> the expert load
            # imbalance (dispatch / experts_mm / combine) is NOT representative. Use for bring-up only.
            print(
                f"[zone-prof] PROFILE_SKIP_PREFIX=1 -> skipping the {n_chunks - 1}-chunk prefix fill; "
                f"attention reads a ZEROED cache (shapes real, MoE routing NOT representative)",
                flush=True,
            )
        elif n_chunks > 1:
            print(f"[zone-prof] pre-filling {n_chunks - 1} chunks -> {cache} cached tokens ...", flush=True)
            t0 = time.perf_counter()
            for c in range(n_chunks - 1):
                prefill_chunk(c)
                ttnn.synchronize_device(mesh)
                drain()
            print(f"[zone-prof] prefix filled in {(time.perf_counter() - t0):.1f}s", flush=True)

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
        chunk_reads = state["reads"] - prefix_reads
        drain()  # single flush of the whole profiled chunk, after its closing marker

        print(
            f"\n[zone-prof] PROFILED CHUNK: {chunk} tok @ {cache} cache, {num_layers} layers\n"
            f"  wall-clock: {wall * 1e3:.1f} ms  ({chunk_reads} profiler reads inside the chunk, "
            f"{prefix_reads} before it)\n"
            f"  device-kernel time per zone: render the ops CSV with\n"
            f"    python3 models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py "
            f"<generated/profiler/reports/*/ops_perf_results_*.csv>",
            flush=True,
        )
        print("[zone-prof] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
