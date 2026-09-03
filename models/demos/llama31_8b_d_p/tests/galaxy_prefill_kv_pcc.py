# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B REAL-WEIGHTS prefill on the `(4,8)` Blackhole galaxy: gates `G-MESH-KV` and `G-RACE`.

Mirrors `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:78` (`main`), minus every MoE knob
(`EXPERT_DTYPE`, the `num_cores` routing alignment in its `plan`) and plus the two things P8 needs
that the template does not have:

* **`PREFILL_RUNS`** — repeat the whole prefill N times in **one process, on one `CCLManager`**, and
  print a SHA-256 of the full read-back KV after each. That is `G-RACE`: the recipe asks for three
  runs bit-identical, and reusing one `CCLManager` across runs is precisely the semaphore-reuse
  condition the gate exists to falsify (`R-013`). Re-running in three separate processes would
  *hide* the bug it is looking for.
* **`PREFILL_KV_HASH_ONLY`** — skip the golden comparison (which reads 129 MB of fp32 safetensors per
  run) when only the hashes are wanted.

Two modes, both required by `G-MESH-KV` (`BRINGUP_RECIPE.md:858`):

| mode | cache size | attention core | what it proves |
|---|---|---|---|
| one-shot (`PREFILL_CHUNKED=0`) | `max_seq_len == chunk` | the SP **bootstrap** — all-gather Q/K/V -> plain causal SDPA -> reduce-scatter -> `x 1/sp` | the SP data layout and the whole TP stack, independent of the cache being readable |
| chunked (`PREFILL_CHUNKED=1`) | `max_seq_len == n_chunks * chunk` | the **ring cache-read** on every chunk, chunk 0 included | `tt/attention/dense_sp.py` |

The mode selects itself from the cache size, exactly as upstream (`DEC-021`): the ring op needs Q
strictly shorter than the per-chip cache shard, so a one-shot request whose cache is one chunk long
has no ring to take (`tt/attention/prefill.py`).

Env:
  `PREFILL_TRACE_DIR`        golden trace dir (`metadata.json` + `kv_cache/layer_N.safetensors`)  [required]
  `PREFILL_CHUNKED`          `1` -> chunked (ring cache-read); `0` -> one-shot (bootstrap)        [0]
  `PREFILL_CHUNK_SIZE`       chunk size in tokens; must satisfy `chunk % (32*sp) == 0`            [512]
  `PREFILL_RUNS`             prefill repetitions, each hashed — `G-RACE` uses 3                   [1]
  `PREFILL_NUM_LAYERS`       build/run only the first N decoder layers                            [all 32]
  `PREFILL_TOPOLOGY`         `ring` or `linear`                                                   [ring]
  `PREFILL_KV_HASH_ONLY`     `1` -> hashes only, no golden read                                   [0]
  `LLAMA_WEIGHTS_FROM_CACHE` `1` -> empty state_dict; load the tilized weight cache (`R-017`)     [0]
  `LLAMA_KV_PCC_MIN`         fail the run when min KV PCC drops below this                        [unset]
  `HF_MODEL`                 the real checkpoint directory                                        [required]

Run::

    export HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct
    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
    PREFILL_TRACE_DIR=/home/mstojkovic/llama31_8b_golden/p7_s512 \\
      python3 models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py

Auto-SKIPs (exit 0) without the 32-device galaxy or without a golden trace, so it is safe in a suite
that runs on smaller hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import ttnn

# The (4,8) galaxy = 32 devices: TP=8 (cols, one KV head per chip) x SP=4 (rows) — DEC-002.
GALAXY_NUM_DEVICES = 32
ROWS, COLS = 4, 8


def _raise_nproc_limit():
    """Raise `RLIMIT_NPROC` to the hard limit so tt-metal's parallel kernel JIT (a burst of g++/make
    processes) does not starve with `EAGAIN` mid-build. Copied from the template's harness."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[prefill-pcc] raised RLIMIT_NPROC soft {soft} -> {hard}", flush=True)
        except (ValueError, OSError) as exc:
            print(f"[prefill-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {exc}", file=sys.stderr)


def plan(n_tokens, chunk_size, chunked, sp):
    """Resolve `(n_chunks, chunk, total)`.

    `chunk` is rounded up to a multiple of `TILE_SIZE * sp` (128 at SP=4), which is
    `build_indexed_rope`'s constraint (`tt/rope.py`) and the engine doc's `CHUNK % (SP*32) == 0`.
    One-shot uses a single chunk covering the whole padded prompt, which makes
    `max_seq_len == chunk` and therefore selects the bootstrap (`DEC-021`). Chunked keeps the
    requested chunk size and pads the tail.
    """
    align = ttnn.TILE_SIZE * sp
    if chunked:
        chunk = -(-chunk_size // align) * align
        n_chunks = max(1, -(-n_tokens // chunk))
    else:
        chunk = max(align, -(-n_tokens // align) * align)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


def _kv_hash(runtime, kv_cache, *, slot_id, n_tokens, chunk_size, num_layers):
    """SHA-256 over every layer's read-back K and V, in layer order.

    `G-RACE` compares this across runs. It hashes the **fp32 read-back**, not the raw device bytes,
    so it is exactly the product the gates score; a semaphore reused while in flight perturbs a
    collective's partial sums and changes it.
    """
    digest = hashlib.sha256()
    for local in range(num_layers):
        k, v = runtime.gather_layer(
            slot_id=slot_id, layer_idx=local, n_tokens=n_tokens, kv_cache=kv_cache, chunk_size=chunk_size
        )
        digest.update(k.contiguous().numpy().tobytes())
        digest.update(v.contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> int:  # noqa: C901
    _raise_nproc_limit()

    golden_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not golden_dir:
        print("[prefill-pcc] SKIP: set PREFILL_TRACE_DIR to a golden trace dir", flush=True)
        return 0
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(
            f"[prefill-pcc] SKIP: needs the ({ROWS},{COLS}) galaxy ({GALAXY_NUM_DEVICES} devices) for "
            f"TP={COLS} x SP={ROWS}; have {ttnn.get_num_devices()}",
            flush=True,
        )
        return 0

    with open(Path(golden_dir) / "metadata.json") as handle:
        token_ids = list(json.load(handle)["token_ids"])
    n_tokens = len(token_ids)
    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", "512"))
    runs = int(os.getenv("PREFILL_RUNS", "1"))
    hash_only = os.getenv("PREFILL_KV_HASH_ONLY", "0") == "1"
    linear = os.getenv("PREFILL_TOPOLOGY", "ring") == "linear"

    n_chunks, chunk, total = plan(n_tokens, chunk_size, chunked, ROWS)
    print(
        f"[prefill-pcc] golden={golden_dir} n_tokens={n_tokens} "
        f"mode={'chunked' if chunked else 'one-shot'} chunk={chunk} n_chunks={n_chunks} total={total} "
        f"runs={runs} topology={'linear' if linear else 'ring'}",
        flush=True,
    )
    print(
        "[prefill-pcc] attention core: "
        + (
            "SP ring cache-read on every chunk (tt/attention/dense_sp.py)"
            if chunked
            else "SP bootstrap (all-gather Q/K/V -> causal SDPA -> reduce-scatter -> x1/sp), because "
            "max_seq_len == chunk leaves the ring op no room (DEC-021)"
        ),
        flush=True,
    )

    from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache
    from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs
    from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    # Ring collectives need the cyclic torus route: FABRIC_1D_RING + the torus descriptor
    # (TT_MESH_GRAPH_DESC_PATH). DEC-020 / DEC-081.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if linear else ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(ROWS, COLS))
    print(f"[prefill-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        model_args = ModelArgs(mesh_device=mesh)  # reads HF_MODEL
        hf_config = model_args.hf_config
        num_layers = hf_config.num_hidden_layers
        override = os.getenv("PREFILL_NUM_LAYERS")
        if override:
            num_layers = int(override)
            print(f"[prefill-pcc] PREFILL_NUM_LAYERS={num_layers}: first {num_layers} layers only", flush=True)

        weight_dtype = ttnn.bfloat8_b
        cache_path = model_args.weight_cache_path(weight_dtype)
        print(f"[prefill-pcc] weight cache {cache_path}", flush=True)
        if os.getenv("LLAMA_WEIGHTS_FROM_CACHE") == "1":
            print("[prefill-pcc] LLAMA_WEIGHTS_FROM_CACHE=1 -> empty state_dict (load the tilized cache)", flush=True)
            state_dict = {}
        else:
            print("[prefill-pcc] loading the real checkpoint (safetensors read) ...", flush=True)
            state_dict = ModelArgs.load_state_dict(model_args.weights_path)

        config = TtPrefillRuntimeConfig(
            num_layers=num_layers,
            max_seq_len=total,
            mesh_shape=(ROWS, COLS),
            default_chunk_size=chunk,
            num_users=1,
            cache_dtype=ttnn.bfloat8_b,
            weight_dtype=weight_dtype,
            weight_cache_path=cache_path,
            topology=ttnn.Topology.Linear if linear else ttnn.Topology.Ring,
            # The engine owns the cache (DEC-055); this harness plays the engine and allocates below.
            owns_kv_cache=False,
            sequence_parallel=True,
        )
        runtime = TtPrefillRuntime(mesh, hf_config, state_dict, config)
        del state_dict

        kv_cache = allocate_kv_cache(
            mesh,
            num_layers=num_layers,
            max_seq_len=total,
            sp_axis=config.sp_axis,
            num_users=config.num_users,
            head_dim=hf_config.head_dim,
            cache_dtype=config.cache_dtype,
        )
        print(f"[prefill-pcc] compiling ({num_layers}L, SP={ROWS} x TP={COLS}) ...", flush=True)
        runtime.compile(kv_cache)

        padded = token_ids + [0] * (total - n_tokens)

        def run_once():
            for c in range(n_chunks):
                start = c * chunk
                chunk_input = runtime.make_chunk_input(padded[start : start + chunk], chunk)
                runtime.prefill_chunk(
                    chunk_input,
                    kv_cache,
                    slot_id=0,
                    actual_start=start,
                    actual_end=min(start + chunk, n_tokens),
                    chunk_size=chunk,
                )
            ttnn.synchronize_device(mesh)

        times, hashes = [], []
        for i in range(runs):
            t0 = time.perf_counter()
            run_once()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            digest = _kv_hash(
                runtime,
                kv_cache,
                slot_id=0,
                n_tokens=min(n_tokens, total) - (min(n_tokens, total) % ttnn.TILE_SIZE),
                chunk_size=chunk,
                num_layers=num_layers,
            )
            hashes.append(digest)
            print(
                f"[prefill-pcc] run {i}: {elapsed * 1000:.1f} ms  {n_tokens / elapsed:.1f} tok/s (real)  "
                f"{total / elapsed:.1f} tok/s (incl pad)  KV sha256={digest}",
                flush=True,
            )
        median = statistics.median(times)
        print(
            f"[prefill-pcc] THROUGHPUT over {runs} run(s): median {n_tokens / median:.1f} tok/s (real), "
            f"{total / median:.1f} tok/s (processed); wall median {median * 1000:.1f} ms",
            flush=True,
        )

        # --- G-RACE: every run's KV must be bit-identical ---
        identical = len(set(hashes)) == 1
        print(
            f"[prefill-pcc] G-RACE: {runs} run(s), {len(set(hashes))} distinct KV hash(es) -> "
            f"{'BIT-IDENTICAL' if identical else 'NON-DETERMINISTIC'}",
            flush=True,
        )
        for i, digest in enumerate(hashes):
            print(f"[prefill-pcc]   run {i} sha256 = {digest}", flush=True)
        if runs > 1 and not identical:
            print(
                "[prefill-pcc] FAIL: the same prefill produced different KV across runs on one "
                "CCLManager. A semaphore is being reused while a collective still holds it "
                "(R-013): check the ping-pong depth in tt/ccl.py and reset_global_semaphores.",
                flush=True,
            )
            return 1

        # --- G-SEMAPHORE, at full depth on real weights: the four lists are still their constants
        # after `runs * n_chunks * num_layers * 2` all-reduces plus the ring attention.
        ccl = runtime.ccl_manager
        print(
            f"[prefill-pcc] semaphores rs/ag/barrier/ring = "
            f"{(len(ccl.rs_ping_pong_semaphores), len(ccl.ag_ping_pong_semaphores), len(ccl.barrier_semaphore), len(ccl.ring_attention_ccl_semaphore_handles))} "
            f"after {runs * n_chunks * num_layers * 2} all-reduces (expect (6, 4, 2, 2)); "
            f"ring-gather buffers allocated = {len(ccl._ring_gather_buffers)}",
            flush=True,
        )

        # --- G-MESH-KV: per-layer K/V PCC vs the fp32 golden ---
        if hash_only:
            print("[prefill-pcc] PREFILL_KV_HASH_ONLY=1: skipping the golden comparison", flush=True)
            print("[prefill-pcc] DONE", flush=True)
            return 0
        min_pcc = runtime.kv_cache_pcc_check(
            kv_cache, slot_id=0, n_chunks=n_chunks, trace_dir=golden_dir, chunk_size=chunk, real_len=n_tokens
        )
        print(f"[prefill-pcc] min KV PCC across {num_layers} layers = {min_pcc:.5f}", flush=True)
        floor = os.environ.get("LLAMA_KV_PCC_MIN")
        if floor is not None and min_pcc < float(floor):
            print(f"[prefill-pcc] FAIL: min KV PCC {min_pcc:.5f} < LLAMA_KV_PCC_MIN={floor}", flush=True)
            return 1
        print("[prefill-pcc] DONE", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
