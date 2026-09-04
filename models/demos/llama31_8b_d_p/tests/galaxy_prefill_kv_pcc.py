# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full-model KV cache vs the fp32 golden on the target `(4, 8)` mesh. Gates: `G-MESH-KV`, `G-RACE`.

**HF anchor:** none directly — this is the deployment path end to end. It builds the real 32-layer
model through **`TtPrefillRuntime`** (the engine's own handle on it), runs prefill over the golden
trace's prompt at SP=4 x TP=8, reads every layer's K and V back out of the block-cyclic cache and
scores them against the fp32 golden.

**This is the first time `TtPrefillRuntime` has ever been instantiated** (`07_RISKS.md` R-029): the
`tp == num_key_value_heads` equality forbids `(1,1)`, so `G-RUNTIME` could only audit it
*statically* and no line of its happy path had executed. Driving `G-MESH-KV` through the runtime
rather than through `tt/model.py` directly is what closes that (`DEC-084`) — a harness that
reimplemented the chunk loop would have left the deployment object untested at the exact moment its
first real run became possible.

**Template:** `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py` (env-driven script, auto-skip
off a galaxy, throughput then per-layer PCC). Differences, all forced:

* **no `owns_kv_cache`.** The engine owns the cache (`DEC-062`), so the harness allocates it the way
  the adapter will and passes it in — the template's runtime holds its own
  (`models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:12-14`).
* **no `kv_cache_pcc_check` on the runtime.** The template puts the scoring *inside* the runtime;
  here it lives in this file, because a runtime that can grade itself is a runtime the engine ships
  with a test harness attached.
* **`FABRIC_1D` + `Topology.Linear`, not the ring fabric.** Measured, not chosen — see
  `tests/test_factory.py`'s `_FABRIC_NAME` / `_TOPOLOGY_NAME` and `G-FABRIC-MATRIX`.

## The cache read-back, which is the part that can be silently wrong

Device `(r, c)` holds **KV head `c`** (proved bit-exactly by `G-KV-TP8`) and, at local row `lr`,
**global position** `(lr // chunk_local) * chunk_global + r * chunk_local + (lr % chunk_local)` —
the block-cyclic layout, mirroring the writer kernel exactly
(`models/demos/deepseek_v3_d_p/tt/mla/utils.py:88-92`, which states the same formula, and
`block_cyclic_reorder:65-80`, which is how `tt/rope.py::build_indexed_rope` reorders the RoPE tables
to match). The period is `chunk_local = chunk_global // sp`, so **the read-back depends on the chunk
size the run used** — a one-shot run and a chunked run of the same prompt lay the same tokens out
differently, and reading one with the other's period would produce a plausible, wrong PCC.

`G-MESH-KV` therefore runs **more than one chunk size** (`BRINGUP_RECIPE.md:1849-1851`), which is
also what makes the layout claim falsifiable: a read-back with the wrong period cannot score well at
both.

## `G-RACE`

`PREFILL_RACE_ITERS=3` runs the whole harness **three times in one process on one `CCLManager`** and
requires the per-layer PCC tables to be **bit-identical**, logging all three SHA-256 hashes
(`BRINGUP_RECIPE.md:1841-1845`). Non-determinism here means a semaphore is being reused while in
flight. Note the scope of a pass, as the recipe insists: a few hundred collectives is not hundreds
of thousands, and it says nothing about multi-user slots.

Run:
    export TT_METAL_HOME=$PWD PYTHONPATH=$PWD HF_MODEL=/home/mstojkovic/models/Llama-3.1-8B-Instruct
    export TT_CACHE_PATH=$HOME/.cache/llama31_8b_d_p
    export PREFILL_TRACE_DIR=/home/mstojkovic/prefill_traces/llama31_8b_d_p/s1024
    python models/demos/llama31_8b_d_p/tests/galaxy_prefill_kv_pcc.py            # one-shot
    PREFILL_CHUNKED=1 PREFILL_CHUNK_SIZE=512 python .../galaxy_prefill_kv_pcc.py # chunked
    PREFILL_RACE_ITERS=3 python .../galaxy_prefill_kv_pcc.py                     # G-RACE

Env:
    PREFILL_TRACE_DIR    golden trace dir (metadata.json + kv_cache/layer_N.safetensors)  [required]
    PREFILL_CHUNKED      "1" -> chunked (the SP ring core); "0" -> one chunk              [default 0]
    PREFILL_CHUNK_SIZE   global chunk size in tokens, chunked mode only                   [default 512]
    PREFILL_NUM_LAYERS   build only the first N decoder layers                            [default all]
    PREFILL_RACE_ITERS   repeat the whole measurement N times in one process              [default 1]
    PREFILL_KV_PCC_MIN_K / _MIN_V   gate thresholds; unset -> report only
    TT_CACHE_PATH        weight-cache root (never the checkpoint dir — DEC-048)           [required]
"""

import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

import ttnn

_HERE = Path(__file__).resolve()
if str(_HERE.parents[3]) not in sys.path:
    sys.path.insert(0, str(_HERE.parents[3]))

from models.common.utility_functions import comp_pcc  # noqa: E402
from models.demos.llama31_8b_d_p.tests.test_factory import (  # noqa: E402
    GALAXY_MESH_SHAPE,
    prefill_fabric_config,
    prefill_topology,
)
from models.demos.llama31_8b_d_p.tt.attention.kv_cache import allocate_kv_cache  # noqa: E402
from models.demos.llama31_8b_d_p.tt.attention.prefill import select_attention_core  # noqa: E402
from models.demos.llama31_8b_d_p.tt.config import derive_head_dim  # noqa: E402
from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs  # noqa: E402
from models.demos.llama31_8b_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig  # noqa: E402

GALAXY_NUM_DEVICES = GALAXY_MESH_SHAPE[0] * GALAXY_MESH_SHAPE[1]
DEFAULT_CHUNK_SIZE = 512

# Carried from `G-CHUNK` / `G-KV-TP8` rather than picked here (`BRINGUP_RECIPE.md:1802-1804`): a
# threshold chosen against this measurement could not fail.
GOLDEN_K_THRESHOLD = 0.99
GOLDEN_V_THRESHOLD = 0.98

_RAW_DIR = _HERE.parents[1] / "bringup_log" / "raw"


def _raise_nproc_limit():
    """Raise `RLIMIT_NPROC` to the hard limit so the parallel kernel JIT does not hit `EAGAIN`.

    Same rationale as `models/demos/gpt_oss_d_p/tests/galaxy_prefill_kv_pcc.py:46-56`: a 32-device
    first run forks a burst of compiler processes.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
            print(f"[kv-pcc] raised RLIMIT_NPROC soft {soft} -> {hard}", flush=True)
        except (ValueError, OSError) as e:
            print(f"[kv-pcc] WARNING: could not raise RLIMIT_NPROC (soft={soft}): {e}", file=sys.stderr, flush=True)


# =============================================================================================
# geometry
# =============================================================================================
def plan(n_tokens, chunk_size, chunked, sp):
    """`(n_chunks, chunk_global, total)`, aligned so every downstream constraint holds.

    * `chunk_global % (TILE_SIZE * sp) == 0` — the per-chip chunk must be tile-aligned
      (`tt/rope.py::build_indexed_rope`, `tt_prefill_runtime.py::TtPrefillRuntimeConfig`);
    * `total % chunk_global == 0` — `resolve_chunk_sizes` requires every chunk size to divide
      `max_seq_len`, because `build_indexed_rope` tiles the cache by `chunk_global // sp`.

    One-shot means **one chunk covering the whole prompt**, so `total == chunk_global` and the
    attention core is the SP bootstrap; chunked means `total > chunk_global` and the core is the
    ring (`tt/attention/prefill.py::select_attention_core`). Which one ran is asserted, not assumed
    — Appendix B's last row is exactly this mix-up.
    """
    align = ttnn.TILE_SIZE * sp
    if chunked:
        chunk = ((chunk_size + align - 1) // align) * align
        n_chunks = max(1, (n_tokens + chunk - 1) // chunk)
    else:
        chunk = max(align, ((n_tokens + align - 1) // align) * align)
        n_chunks = 1
    return n_chunks, chunk, n_chunks * chunk


def cache_row_to_global_position(local_row, row, *, chunk_local, chunk_global):
    """The block-cyclic map, in one place. See the module docstring."""
    return (local_row // chunk_local) * chunk_global + row * chunk_local + (local_row % chunk_local)


def read_kv_cache(kv_cache, *, num_layers, n_kv, head_dim, chunk_global, total, n_tokens, mesh_shape):
    """The device cache -> `{layer: (k, v)}`, each `[1, n_kv, n_tokens, head_dim]`, **Meta** order.

    Head `c` comes from mesh column `c` (`G-KV-TP8`, bit-exact) and the sequence is un-block-cyclic'd
    by `cache_row_to_global_position`. Only the first `n_tokens` global positions are returned; the
    pad tail is dropped rather than scored.
    """
    rows, cols = mesh_shape
    sp = rows
    chunk_local = chunk_global // sp
    seq_local = total // sp
    assert seq_local % chunk_local == 0, f"seq_local {seq_local} must be a whole number of {chunk_local}-blocks"

    per_device_k = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(kv_cache.k)]
    per_device_v = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(kv_cache.v)]

    # position -> (row, local_row), built once and reused for every layer and both caches.
    index = {}
    for row in range(sp):
        for local_row in range(seq_local):
            pos = cache_row_to_global_position(local_row, row, chunk_local=chunk_local, chunk_global=chunk_global)
            if pos < n_tokens:
                index[pos] = (row, local_row)
    missing = sorted(set(range(n_tokens)) - set(index))
    assert not missing, f"the block-cyclic map does not cover global positions {missing[:8]} (+{len(missing) - 8} more)"

    rows_by_pos = torch.tensor([index[p][0] for p in range(n_tokens)])
    locals_by_pos = torch.tensor([index[p][1] for p in range(n_tokens)])

    out = {}
    for layer in range(num_layers):
        k = torch.empty(1, n_kv, n_tokens, head_dim)
        v = torch.empty(1, n_kv, n_tokens, head_dim)
        for c in range(n_kv):
            for r in range(sp):
                dev = r * cols + c
                sel = rows_by_pos == r
                lr = locals_by_pos[sel]
                k[0, c, sel, :] = per_device_k[dev][layer, 0][lr, :]
                v[0, c, sel, :] = per_device_v[dev][layer, 0][lr, :]
        out[layer] = (k, v)
    return out


# =============================================================================================
# the golden
# =============================================================================================
def meta_head_index(head_dim):
    """`src` such that `meta = hf[..., src]` — `reverse_permute`'s head-dim map.

    Duplicated from `tests/unit/test_decoder_layer_vs_ref.py::_meta_head_index` on purpose: this
    file is a **script** and must not import a pytest module (`DEC-085`). The two are asserted equal
    by `tests/unit/test_chunked_attention_ring.py`, so they cannot drift silently.
    """
    half = head_dim // 2
    return torch.tensor([half * (m % 2) + (m // 2) for m in range(head_dim)], dtype=torch.long)


def load_golden(trace_dir, num_layers, head_dim, n_tokens):
    """`{layer: (k_meta, v)}` from the fp32 trace, K permuted **HF -> Meta** over `head_dim`.

    The permutation is applied to the golden, never to the device tensor, and it happens **before**
    any quantiser touches it — `bfloat8_b`'s shared exponent is per 16-element block of the last dim,
    so `permute(quantise(x)) != quantise(permute(x))` (recipe §2.2.3a). Nothing here quantises, so
    the hazard is only latent, but the ordering is kept so a floor computed from these tensors
    inherits it.
    """
    from safetensors import safe_open

    src = meta_head_index(head_dim)
    golden = {}
    for layer in range(num_layers):
        path = Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"
        with safe_open(str(path), framework="pt", device="cpu") as h:
            k = h.get_tensor(f"key_cache_layer_{layer}").float()[:, :, :n_tokens, :]
            v = h.get_tensor(f"value_cache_layer_{layer}").float()[:, :, :n_tokens, :]
        golden[layer] = (k[..., src], v)
    return golden


def score(read_back, golden, num_layers):
    """`{"k": {layer: pcc}, "v": {...}}` — per-layer PCC of the device cache against the golden."""
    out = {"k": {}, "v": {}}
    for layer in range(num_layers):
        got_k, got_v = read_back[layer]
        ref_k, ref_v = golden[layer]
        _, pcc_k = comp_pcc(ref_k, got_k, 0.0)
        _, pcc_v = comp_pcc(ref_v, got_v, 0.0)
        out["k"][layer], out["v"][layer] = float(pcc_k), float(pcc_v)
    return out


def table_hash(scores):
    """SHA-256 of the per-layer PCC table, at full float repr. `G-RACE`'s comparand."""
    payload = json.dumps(
        {name: {str(k): repr(v) for k, v in sorted(vals.items())} for name, vals in sorted(scores.items())},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# =============================================================================================
# one measurement
# =============================================================================================
def build_runtime(mesh, hf, state_dict, *, num_layers, chunk_global, total, cache_path, ccl_manager=None):
    """The deployment runtime. `sequence_parallel=True` is what makes the SP cores reachable."""
    config = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=total,
        chunk_size=chunk_global,
        mesh_shape=GALAXY_MESH_SHAPE,
        num_users=1,
        weight_cache_path=cache_path,
        sequence_parallel=True,
        topology=prefill_topology(),
    )
    return TtPrefillRuntime(mesh, hf, state_dict, config, ccl_manager=ccl_manager)


def allocate_engine_cache(mesh, *, num_layers, total, head_dim):
    """The cache the **engine** owns, allocated the way the P10 adapter will (`DEC-062`)."""
    return allocate_kv_cache(
        mesh,
        num_layers=num_layers,
        max_seq_len=total,
        sp_axis=0,
        num_users=1,
        head_dim=head_dim,
        cache_dtype=ttnn.bfloat8_b,  # `DEC-021`
    )


def run_prefill(runtime, kv_cache, padded_tokens, *, n_chunks, chunk_global, n_tokens):
    """Feed every chunk in order. Returns the wall time."""
    t0 = time.perf_counter()
    for chunk in range(n_chunks):
        start = chunk * chunk_global
        runtime.prefill_chunk(
            runtime.make_chunk_input(padded_tokens[start : start + chunk_global], chunk_global),
            kv_cache,
            slot_id=0,
            actual_start=start,
            actual_end=min(start + chunk_global, n_tokens),
            chunk_size=chunk_global,
        )
    ttnn.synchronize_device(runtime.mesh_device)
    return time.perf_counter() - t0


def measure(runtime, mesh, hf, *, num_layers, n_chunks, chunk_global, total, n_tokens, padded, golden, head_dim):
    """Allocate a **fresh** cache, prefill, read back, score. Returns `(scores, seconds)`."""
    kv_cache = allocate_engine_cache(mesh, num_layers=num_layers, total=total, head_dim=head_dim)
    seconds = run_prefill(runtime, kv_cache, padded, n_chunks=n_chunks, chunk_global=chunk_global, n_tokens=n_tokens)
    read_back = read_kv_cache(
        kv_cache,
        num_layers=num_layers,
        n_kv=hf["num_key_value_heads"],
        head_dim=head_dim,
        chunk_global=chunk_global,
        total=total,
        n_tokens=n_tokens,
        mesh_shape=GALAXY_MESH_SHAPE,
    )
    kv_cache.k.deallocate(True)
    kv_cache.v.deallocate(True)
    return score(read_back, golden, num_layers), seconds


def expected_core(chunked, total, chunk_global):
    """Which attention core this configuration selects. Asserted, because Appendix B's last row."""
    return "sp_ring" if (chunked and total > chunk_global) else "sp_bootstrap"


# =============================================================================================
# main
# =============================================================================================
def main():
    _raise_nproc_limit()

    trace_dir = os.environ.get("PREFILL_TRACE_DIR")
    if not trace_dir:
        print("[kv-pcc] SKIP: set PREFILL_TRACE_DIR to a golden trace dir", flush=True)
        return 0
    if ttnn.get_num_devices() < GALAXY_NUM_DEVICES:
        print(
            f"[kv-pcc] SKIP: needs the {GALAXY_MESH_SHAPE} galaxy ({GALAXY_NUM_DEVICES} devices) for "
            f"SP=4 x TP=8; have {ttnn.get_num_devices()}",
            flush=True,
        )
        return 0

    metadata = json.load(open(Path(trace_dir) / "metadata.json"))
    token_ids = list(metadata["token_ids"])
    n_tokens = len(token_ids)
    assert metadata["dtype"] == "float32", f"the golden must be fp32, this trace is {metadata['dtype']!r} (DEC-059)"

    chunked = os.getenv("PREFILL_CHUNKED", "0") == "1"
    chunk_size = int(os.getenv("PREFILL_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
    race_iters = int(os.getenv("PREFILL_RACE_ITERS", "1"))
    num_layers_env = os.getenv("PREFILL_NUM_LAYERS")

    n_chunks, chunk_global, total = plan(n_tokens, chunk_size, chunked, GALAXY_MESH_SHAPE[0])
    core = expected_core(chunked, total, chunk_global)
    print(
        f"[kv-pcc] golden={trace_dir} n_tokens={n_tokens} mode={'chunked' if chunked else 'one-shot'} "
        f"chunk_global={chunk_global} chunk_local={chunk_global // GALAXY_MESH_SHAPE[0]} "
        f"n_chunks={n_chunks} total={total} race_iters={race_iters} "
        f"topology={prefill_topology()} fabric={prefill_fabric_config()} expected_core={core}",
        flush=True,
    )

    ttnn.set_fabric_config(prefill_fabric_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*GALAXY_MESH_SHAPE))
    print(f"[kv-pcc] mesh opened {tuple(mesh.shape)} ndev={mesh.get_num_devices()}", flush=True)
    try:
        args = ModelArgs(mesh, max_seq_len=total)
        hf = args.hf_config
        num_layers = int(num_layers_env) if num_layers_env else hf["num_hidden_layers"]
        head_dim = derive_head_dim(hf)
        cache_path = args.weight_cache_path(ttnn.bfloat8_b)
        print(f"[kv-pcc] layers={num_layers} head_dim={head_dim} weight_cache={cache_path}", flush=True)

        golden = load_golden(trace_dir, num_layers, head_dim, n_tokens)
        print(f"[kv-pcc] golden loaded: {num_layers} layers, {n_tokens} tokens, fp32", flush=True)

        print("[kv-pcc] loading real bf16 weights from the safetensors shards ...", flush=True)
        state_dict = ModelArgs.load_state_dict(args.model_path)
        runtime = build_runtime(
            mesh,
            hf,
            state_dict,
            num_layers=num_layers,
            chunk_global=chunk_global,
            total=total,
            cache_path=cache_path,
        )
        del state_dict

        # `compile()` runs zero-token chunks and would pollute a measured cache, so it gets a
        # throwaway one. Running it at all is deliberate: it is the engine's own warm-up call and
        # this is its first execution anywhere (`R-029`).
        print("[kv-pcc] compile(): warming the kernels on a throwaway cache ...", flush=True)
        scratch = allocate_engine_cache(mesh, num_layers=num_layers, total=total, head_dim=head_dim)

        # Which core actually runs, asserted rather than assumed — and asserted BEFORE the
        # expensive part. Appendix B's last row is "everything passes but the numbers look too good
        # | you measured the SP bootstrap because max_seq_len == chunk_size", and the two cores
        # differ only in a condition on those two numbers
        # (`tt/attention/prefill.py::select_attention_core`).
        actual_core = select_attention_core(
            runtime.model.attention_config,
            runtime.mesh_config,
            scratch,
            seq_len=chunk_global // GALAXY_MESH_SHAPE[0],
            cached_len=0,
        )
        assert actual_core == core, (
            f"this configuration selects the {actual_core!r} attention core, not the {core!r} one "
            f"this run believes it is measuring (chunked={chunked}, total={total}, "
            f"chunk_global={chunk_global})"
        )
        print(f"[kv-pcc] attention core for chunk 0: {actual_core}", flush=True)

        t0 = time.perf_counter()
        runtime.compile(scratch)
        print(f"[kv-pcc] compile() done in {time.perf_counter() - t0:.1f}s", flush=True)
        scratch.k.deallocate(True)
        scratch.v.deallocate(True)

        padded = token_ids + [0] * (total - n_tokens)
        runs = []
        for iteration in range(race_iters):
            scores, seconds = measure(
                runtime,
                mesh,
                hf,
                num_layers=num_layers,
                n_chunks=n_chunks,
                chunk_global=chunk_global,
                total=total,
                n_tokens=n_tokens,
                padded=padded,
                golden=golden,
                head_dim=head_dim,
            )
            digest = table_hash(scores)
            runs.append((scores, digest))
            print(
                f"[kv-pcc] run {iteration}: {seconds * 1000:.1f} ms, {n_tokens / seconds:.1f} tok/s, "
                f"min K = {min(scores['k'].values()):.7f}, min V = {min(scores['v'].values()):.7f}, "
                f"table sha256 = {digest}",
                flush=True,
            )

        scores = runs[0][0]
        for layer in range(num_layers):
            print(f"[kv-pcc] L{layer:>2}: K={scores['k'][layer]:.7f} V={scores['v'][layer]:.7f}", flush=True)
        min_k = min(scores["k"].values())
        min_v = min(scores["v"].values())
        argmin_k = min(scores["k"], key=scores["k"].get)
        argmin_v = min(scores["v"], key=scores["v"].get)
        print(
            f"[kv-pcc] MIN over {num_layers} layers: K = {min_k:.7f} (L{argmin_k}), "
            f"V = {min_v:.7f} (L{argmin_v}); thresholds K >= {GOLDEN_K_THRESHOLD} / V >= {GOLDEN_V_THRESHOLD}",
            flush=True,
        )

        # --- G-RACE -------------------------------------------------------------------------
        race_ok = True
        if race_iters > 1:
            digests = [d for _, d in runs]
            race_ok = len(set(digests)) == 1
            for i, d in enumerate(digests):
                print(f"[kv-pcc] G-RACE run {i} hash: {d}", flush=True)
            print(
                f"[kv-pcc] G-RACE: {race_iters} runs in ONE process on ONE CCLManager -> "
                f"{len(set(digests))} distinct hash(es) — {'BIT-IDENTICAL' if race_ok else 'NON-DETERMINISTIC'}",
                flush=True,
            )
            if not race_ok:
                for i in range(1, race_iters):
                    diff = [
                        (name, layer)
                        for name in ("k", "v")
                        for layer in range(num_layers)
                        if runs[i][0][name][layer] != runs[0][0][name][layer]
                    ]
                    print(f"[kv-pcc] G-RACE run {i} differs from run 0 at {diff[:12]}", flush=True)

        _RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _RAW_DIR / (
            f"G-MESH-KV_{'chunked' if chunked else 'oneshot'}_c{chunk_global}"
            f"{f'_race{race_iters}' if race_iters > 1 else ''}_per_layer_pcc.json"
        )
        with open(out_path, "w") as f:
            json.dump(
                {
                    "trace_dir": trace_dir,
                    "mesh": list(GALAXY_MESH_SHAPE),
                    "sp": GALAXY_MESH_SHAPE[0],
                    "tp": GALAXY_MESH_SHAPE[1],
                    "topology": str(prefill_topology()),
                    "fabric": str(prefill_fabric_config()),
                    "mode": "chunked" if chunked else "one-shot",
                    "attention_core": core,
                    "chunk_global": chunk_global,
                    "chunk_local": chunk_global // GALAXY_MESH_SHAPE[0],
                    "n_chunks": n_chunks,
                    "n_tokens": n_tokens,
                    "total": total,
                    "num_layers": num_layers,
                    "cache_dtype": "bfloat8_b",
                    "weight_dtype": "bfloat8_b",
                    "activation_dtype": "bfloat16",
                    "min_k": min_k,
                    "min_v": min_v,
                    "argmin_k": argmin_k,
                    "argmin_v": argmin_v,
                    "race_iters": race_iters,
                    "race_hashes": [d for _, d in runs],
                    "race_bit_identical": race_ok,
                    "per_layer": {name: {str(k): v for k, v in sorted(vals.items())} for name, vals in scores.items()},
                },
                f,
                indent=2,
            )
        print(f"[kv-pcc] per-layer table written to {out_path}", flush=True)

        failed = []
        gate_k = os.environ.get("PREFILL_KV_PCC_MIN_K")
        gate_v = os.environ.get("PREFILL_KV_PCC_MIN_V")
        if gate_k is not None and min_k < float(gate_k):
            failed.append(f"min K {min_k:.7f} < PREFILL_KV_PCC_MIN_K={gate_k}")
        if gate_v is not None and min_v < float(gate_v):
            failed.append(f"min V {min_v:.7f} < PREFILL_KV_PCC_MIN_V={gate_v}")
        if not race_ok:
            failed.append(f"G-RACE: {race_iters} runs produced more than one hash")
        if failed:
            for reason in failed:
                print(f"[kv-pcc] FAIL: {reason}", flush=True)
            return 1
        print("[kv-pcc] DONE", flush=True)
    finally:
        mesh.quiesce_devices()
        ttnn.close_mesh_device(mesh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
