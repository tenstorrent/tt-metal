# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 prefill DEVICE-perf breakdown on the real SP=8 × TP=4 + EP=32 galaxy layout.

Mirrors the inner/outer + tracy-signpost pattern of test_ep_moe_perf.py, but drives the full
decoder stack via ``TtPrefillRuntime`` (the same builder the KV-cache PCC harness uses), so it
loads real weights FAST from the tilized cache and runs the genuine per-layer op mix.

The real 60-layer model overflows the tracy per-op CSV, so profile a FEW layers (PERF_NUM_LAYERS,
default 5 -> layers 0-2 dense + 3-4 sparse/MoE). To ISOLATE dense vs sparse without subtraction, the
inner emits a tracy signpost ``L{i}`` after every layer (via the runtime's ``on_layer_complete``
seam); the outer buckets each device op into its layer, then averages the dense layers (0-2) and the
sparse/MoE layers (>=3) separately and projects the full 3-dense + 57-sparse model.

Two pieces:
- INNER  ``test_prefill_fwd``: build (from cache) -> compile/warmup -> ONE forward with per-layer
  signposts. Run *under* the tracy device profiler.
- OUTER  ``test_prefill_device_perf``: spawns the inner under Tracy, buckets the ops CSV per layer,
  merges the 32 device rows per op (collectives averaged, others max = critical path), and prints
  per-dense-layer + per-sparse-layer per-op tables, a CCL/matmul/SDPA/MoE category rollup, and the
  full-model projection.

Scenario: one-shot single chunk (PERF_SEQ tokens, empty cache -> first-chunk / nocache attention).

Env: PERF_NUM_LAYERS (default 5), PERF_SEQ (default 5120; >=2048, %1024), EXPERT_DTYPE (bf4|bf8),
     HF_MODEL (real weights dir), M3_WEIGHTS_FROM_CACHE=1 to force the cache path. Set
     TT_MESH_GRAPH_DESC_PATH to the stock single_bh_galaxy ([8,4]) descriptor.

Run:
  source python_env/bin/activate
  export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
  export HF_MODEL=/mnt/models/MiniMaxAI/MiniMax-M3-ref
  export TT_MESH_GRAPH_DESC_PATH=$PWD/tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto
  export EXPERT_DTYPE=bf4 M3_WEIGHTS_FROM_CACHE=1
  pytest models/demos/minimax_m3/tests/perf/test_prefill_perf.py::test_prefill_device_perf -s
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn

_THIS = "models/demos/minimax_m3/tests/perf/test_prefill_perf.py"
MSA_MIN_TOKENS = 16 * 128  # topk_blocks * block_size — MSA needs at least this many tokens
N_DENSE = 3  # M3 schedule: layers 0-2 dense, 3-59 sparse/MoE
N_LAYERS_FULL = 60


def _build_runtime_from_cache(mesh, num_layers, max_seq_len, chunk_size, expert_dtype):
    """Build TtPrefillRuntime on (8,4) SP=8×TP=4, loading real weights from the tilized cache.

    Mirrors galaxy_prefill_kv_pcc.py: on a complete cache pass an EMPTY state_dict and never touch
    the ~869GB bf16 source (M3_WEIGHTS_FROM_CACHE=1 forces it; M3_FORCE_LOAD_WEIGHTS=1 repopulates)."""
    from models.demos.minimax_m3.tt.model_config import ModelArgs
    from models.demos.minimax_m3.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.minimax_m3.tt.weight_cache import weight_cache_is_complete

    model_args = ModelArgs(mesh_device=mesh)  # reads HF_MODEL
    hf_config = model_args.hf_config
    hf_config.num_hidden_layers = num_layers  # TtPrefillRuntime also pins this; keep them equal
    os.environ.setdefault("M3_LOAD_NLAYERS", str(num_layers))  # only matters on a source read

    cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
    force_load = os.getenv("M3_FORCE_LOAD_WEIGHTS") == "1"
    cache_only = not force_load and (
        os.getenv("M3_WEIGHTS_FROM_CACHE") == "1"
        or weight_cache_is_complete(cache_path, hf_config, num_layers, expert_dtype)
    )
    if cache_only:
        logger.info(f"[prefill-perf] tilized cache complete at {cache_path} -> empty state_dict (no bf16 read)")
        state_dict = {}
    else:
        logger.info("[prefill-perf] loading real bf16 weights from source (slow) ...")
        state_dict = ModelArgs.load_state_dict(model_args.weights_path)

    cfg = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        mesh_shape=(8, 4),
        chunk_size=chunk_size,  # == max_seq_len for one-shot; < it for the cache-read scenario
        num_users=1,
        expert_weight_dtype=expert_dtype,
        weight_cache_path=cache_path,
    )
    runtime = TtPrefillRuntime(mesh, hf_config, state_dict, cfg)
    del state_dict
    return runtime, hf_config


@pytest.mark.timeout(0)  # build-from-cache + compile + prefix-fill + measured forward exceeds the 300s default
def test_prefill_fwd():
    """INNER: one signposted (per-layer) prefill forward. Under Tracy.

    Scenario is set by PERF_CACHED_LEN:
      0            -> a PERF_SEQ chunk over an EMPTY cache (first-chunk / nocache attention path).
      >0 (e.g.10240) -> a PERF_SEQ chunk attending to a PERF_CACHED_LEN prefix already in cache
                      (cache-read path). The prefix is pre-filled first; then the EXACT measured
                      cache-read chunk is warmed up once (so the timed pass hits the program cache),
                      and only that final chunk carries the per-layer signposts."""
    from models.demos.minimax_m3.tt.attention import allocate_kv_caches

    rows, cols = 8, 4
    num_layers = int(os.getenv("PERF_NUM_LAYERS", "5"))
    chunk = int(os.getenv("PERF_SEQ", "5120"))
    cached_len = int(os.getenv("PERF_CACHED_LEN", "0"))
    expert_dtype = ttnn.bfloat8_b if os.getenv("EXPERT_DTYPE", "bf4") == "bf8" else ttnn.bfloat4_b
    assert chunk % 1024 == 0 and chunk >= MSA_MIN_TOKENS, f"PERF_SEQ={chunk} must be %1024 and >= {MSA_MIN_TOKENS}"
    assert cached_len % chunk == 0, f"PERF_CACHED_LEN={cached_len} must be a multiple of PERF_SEQ={chunk}"
    assert num_layers > N_DENSE, f"PERF_NUM_LAYERS={num_layers} must be > {N_DENSE} to include a sparse layer"
    max_seq_len = cached_len + chunk
    n_prefix = cached_len // chunk  # chunks to pre-fill before the measured (cache-read) chunk
    torch.manual_seed(0)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)  # SP=8 -> 1D fabric (matches the PCC harness)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        runtime, hf_config = _build_runtime_from_cache(mesh, num_layers, max_seq_len, chunk, expert_dtype)
        kv_cache = allocate_kv_caches(
            mesh, num_layers=num_layers, max_seq_len=max_seq_len, num_users=1, head_dim=hf_config.head_dim
        )
        logger.info(
            f"[prefill-perf] compiling ({num_layers}L, SP=8×TP=4+EP=32, chunk={chunk}, cached_len={cached_len}) ..."
        )
        runtime.compile(kv_cache)  # warmup: compiles the nocache (start=0) path

        def _run_chunk(a):  # one chunk at absolute cache offset a; no signposts unless registered
            inp = runtime.make_chunk_input([0] * chunk)  # perf: token content is irrelevant to device time
            runtime.prefill_chunk(inp, kv_cache, slot_id=0, actual_start=a, actual_end=min(a + chunk, max_seq_len))

        runtime._on_layer_complete = None
        for c in range(n_prefix):  # populate the prefix (also JIT-compiles the cache-read path)
            _run_chunk(c * chunk)
        if cached_len > 0:
            _run_chunk(cached_len)  # warm up the EXACT measured cache-read chunk (hit the program cache)
        ttnn.synchronize_device(mesh)

        # Per-layer signposts: the model calls on_layer_complete(i) after layer i (host-dispatch order),
        # so "L{i}" bounds each layer's device ops. Registered ONLY for the measured chunk.
        runtime._on_layer_complete = lambda i: signpost(header=f"L{i}")
        signpost(header="start")
        _run_chunk(cached_len)
        ttnn.synchronize_device(mesh)
        signpost(header="stop")
        logger.info(f"[prefill-perf] signposted chunk done: {num_layers}L chunk={chunk} @ cached_len={cached_len}")
    finally:
        ttnn.close_mesh_device(mesh)


# --- op-code categorization for the bottleneck rollup (first match wins) ---
_CATEGORIES = [
    ("CCL collective", ("AllGather", "ReduceScatter", "AllReduce", "AllBroadcast")),
    ("MoE dispatch/combine", ("Dispatch", "Combine", "PostCombineReduce")),
    ("Attention (SDPA/indexer)", ("SDPA", "ScaledDotProduct", "RingJoint", "sparse_sdpa", "Indexer", "indexer")),
    ("TopK", ("TopK", "topk", "LargeIndices")),
    ("Expert FFN", ("RoutedExpertFfn", "ExpertFfn")),
    ("Matmul", ("Matmul", "Linear")),
    ("Norm", ("RMSNorm", "LayerNorm", "Norm")),
    ("Embedding", ("Embedding",)),
]


def _categorize(op_code: str) -> str:
    for name, needles in _CATEGORIES:
        if any(n in op_code for n in needles):
            return name
    return "Other"


def _short(op_code: str) -> str:
    """Strip the MeshDeviceOperationAdapter<...> / namespace noise for a readable table."""
    s = str(op_code)
    if "::" in s:
        s = s.split("::")[-1]
    return s.rstrip(">").strip()


def _print_table(title, per_op_ns, total_ns):
    logger.info(f"\n{'='*84}\n{title}\n{'='*84}\n  {'op':<44}{'us':>12}{'% ':>7}")
    logger.info(f"  {'-'*82}")
    for op, ns in sorted(per_op_ns.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {op[:44]:<44}{ns/1e3:>12.1f}{100*ns/total_ns:>7.1f}")
    cats = {}
    for op, ns in per_op_ns.items():
        cats[_categorize(op)] = cats.get(_categorize(op), 0.0) + ns
    logger.info(f"  {'-'*82}\n  {'CATEGORY':<44}{'us':>12}{'% ':>7}")
    for c, ns in sorted(cats.items(), key=lambda kv: -kv[1]):
        logger.info(f"  {c:<44}{ns/1e3:>12.1f}{100*ns/total_ns:>7.1f}")
    logger.info(f"  {'-'*82}\n  {'TOTAL':<44}{total_ns/1e3:>12.1f}   ({total_ns/1e6:.3f} ms)")


@pytest.mark.timeout(0)
def test_prefill_device_perf():
    """OUTER: profile the inner forward, bucket ops per layer, print per-dense / per-sparse breakdown."""
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename

    from models.perf.device_perf_utils import run_device_perf
    from models.tt_transformers.tests.test_utils import merge_device_rows

    num_layers = int(os.getenv("PERF_NUM_LAYERS", "5"))
    total = int(os.getenv("PERF_SEQ", "5120"))
    cached_len = int(os.getenv("PERF_CACHED_LEN", "0"))
    scen = f"chunk={total} @ cached_len={cached_len} ({'cache-read' if cached_len else 'empty cache'})"
    subdir = "minimax_prefill"

    # The cache-read scenario dispatches several forwards (prefix fill + warmup + measured) x layers,
    # which overruns the default profiler op reservation (1333) and aborts the device readback. Bump it.
    run_device_perf(
        command=f"pytest {_THIS}::test_prefill_fwd -s",
        subdir=subdir,
        num_iterations=1,
        cols=["DEVICE KERNEL"],
        batch_size=1,
        has_signposts=True,
        op_support_count=int(os.getenv("PERF_OP_SUPPORT", "8000")),
    )

    fn = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(fn, low_memory=False)
    dur = "DEVICE KERNEL DURATION [ns]"

    # In host-dispatch order: a device op belongs to layer k = number of "L*" signposts before it.
    # "start"/"stop" bound the timed forward (drops warmup + one-time weight tilize).
    sp = df["OP TYPE"] == "signpost"
    is_start, is_stop = sp & (df["OP CODE"] == "start"), sp & (df["OP CODE"] == "stop")
    is_layer_sp = sp & df["OP CODE"].astype(str).str.match(r"^L\d+$")
    assert is_start.any() and is_stop.any(), f"start/stop signposts not found in {fn}"
    assert is_layer_sp.any(), f"per-layer L* signposts not found in {fn}"
    in_region = (is_start.astype(int) - is_stop.astype(int)).cumsum() > 0
    layer_col = is_layer_sp.cumsum()  # 0 before L0, 1 after L0, ... == layer index of following ops

    dev = df[in_region & (df["OP TYPE"] == "tt_dnn_device")].copy()
    dev["layer"] = layer_col[dev.index]
    assert not dev.empty, f"no device ops between signposts in {fn}"

    # Merge the 32 device rows per op WITHIN each layer, then sum per op-code per layer.
    per_layer_op = {}  # layer -> {op_short: ns}
    for lyr, g in dev.groupby("layer"):
        m = merge_device_rows(g)
        m[dur] = m[dur].astype(float)
        s = m.groupby(m["OP CODE"].map(_short))[dur].sum()
        per_layer_op[int(lyr)] = s.to_dict()

    layers = sorted(per_layer_op)
    dense = [l for l in layers if l < N_DENSE]
    sparse = [l for l in layers if l >= N_DENSE]
    logger.info(
        f"\n[prefill-perf] captured layers={layers} (dense={dense}, sparse={sparse}); "
        f"per-layer totals us: " + ", ".join(f"L{l}={sum(per_layer_op[l].values())/1e3:.0f}" for l in layers)
    )
    assert dense and sparse, f"need both dense and sparse layers; got dense={dense} sparse={sparse}"

    def _avg(layer_set):
        ops = set().union(*(per_layer_op[l].keys() for l in layer_set))
        return {op: sum(per_layer_op[l].get(op, 0.0) for l in layer_set) / len(layer_set) for op in ops}

    dense_op, sparse_op = _avg(dense), _avg(sparse)
    dense_ns, sparse_ns = sum(dense_op.values()), sum(sparse_op.values())

    _print_table(f"[prefill-perf] PER DENSE LAYER (avg of {dense}, {scen})", dense_op, dense_ns)
    _print_table(f"[prefill-perf] PER SPARSE/MoE LAYER (avg of {sparse}, {scen})", sparse_op, sparse_ns)

    full_ns = N_DENSE * dense_ns + (N_LAYERS_FULL - N_DENSE) * sparse_ns
    logger.info(
        f"\n{'#'*84}\n[prefill-perf] FULL-MODEL PROJECTION  ({scen})\n{'#'*84}\n"
        f"  per dense layer : {dense_ns/1e3:>10.1f} us\n"
        f"  per sparse layer: {sparse_ns/1e3:>10.1f} us   ({sparse_ns/dense_ns:.1f}x a dense layer)\n"
        f"  full model      : {N_DENSE}×dense + {N_LAYERS_FULL-N_DENSE}×sparse = "
        f"{full_ns/1e6:.1f} ms device-kernel (critical-path sum; overlap not modeled)\n"
        f"  prefill floor   : {total / (full_ns/1e9):,.0f} tokens/s\n{'#'*84}"
    )
