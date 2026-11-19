# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where one Hy4-preview decode step spends its time and its memory on the 8x4 Galaxy, by stage
(tenstorrent/tt-metal#65884; the numbers and what follows from them are in ../PERF.md).

The target is one Galaxy, 8 users with 1M tokens of context each, 75 tokens per second per user:
one step of the batch of 8 in 13.3 ms, 0.171 ms per layer over 78 layers (TARGET_LAYER_MS).

What a case builds: the first three decoder layers with checkpoint weights from the decode weight
cache (BFP8 weights, BFP4 routed experts), which are the three kinds of block the model has

    layer 0   dense FFN + full indexer        (1 of 78)
    layer 1   MoE + full indexer              (20 of 78)
    layer 2   MoE + shared indexer            (57 of 78)

with the embedding and the head, over a decode cache of ``users`` x ``context`` positions that
holds these three layers only, zero-initialised (the time of an op does not depend on the values;
at one million positions for 8 users the three latent stripes and two index-key stripes are 17.8 GB
over the Galaxy). Every user steps at the last positions of the cache, so a full layer scores the
whole stripe. No prefill runs here.

Clocks, and which tool gives which number
-----------------------------------------
* traced, per stage: the stage captured as a trace of its own (ttnn.begin_trace_capture) and
  replayed 20 times back to back with ttnn.execute_trace, one synchronize at the end. This is the
  stage's time on the device: its programs dispatched by the chips' own dispatch cores, the fabric
  inside its collectives, and no host work per op. It is the number the 0.171 ms is compared with.
* traced, whole layer: TtHy4DecoderLayer.forward as the model runs it (the attention looping
  over the users) captured and replayed the same way; the replay's output is compared with the
  untraced forward's.
* untraced staged wall: host time of a stage from its first ttnn call to the end of a
  synchronize, an idle synchronize taken off; ``issue`` is the part until the last call returned
  (the host building and enqueueing the programs).
* untraced, whole layer: the same forward issued repeatedly with one synchronize at the end. What
  a layer costs as the model stands.
* ttnn calls: the ttnn operations a stage issues, counted on the host (OpCounter).
All with the program cache warm (every stage runs before it is measured).

Tools that do not give a number on this host: the realtime program profiler
(ttnn.device.RegisterProgramRealtimeProfilerCallback, kernel time per program per chip) is not
active, because the 32 chips sit in identity IOMMU domains (kernel command line ``iommu=pt``) and
the profiler's device-to-host socket needs a translated one
(tt_metal/distributed/realtime_profiler_manager.cpp); ``realtime_profiler_active`` in the results
records it. The Tracy device profiler (TT_METAL_DEVICE_PROFILER=1) rebuilds every kernel, the
fabric routers included, and was not tried on a board that a failed fabric start takes down.
test_op_floor measures what stands in for kernel time: the cost of a traced program that has
next to nothing to compute.

The attention sublayer takes one user per call (tt/attention_decode.py), so its stages are
measured for one user of the cache (user 0) and a batch of 8 is 8 such calls; every other stage
takes the users as rows and is measured with all of them. The queued and traced layer times of an
8-user case are the real loop over the 8 users.

The stages are the modules' own methods called in the order the layer's forward calls them, so a
stage boundary is a method boundary; the two places where a method is cut finer (the indexer's
selection into scoring / local top-k / merge, the MoE into router / dispatch / experts / combine /
shared expert) repeat the lines of TtHy4Indexer._select and TtHy4MoE.forward here. Each case
checks that the staged chain gives the layer's own output.

Jobs (one case per job; the weights load in seconds from the weight cache):
    tools/run_device_tests.sh perf_<case> tests/test_perf_decode.py -k "test_decode_stages and <case>"
    tools/run_device_tests.sh perf_floor tests/test_perf_decode.py -k test_op_floor
    tools/run_device_tests.sh perf_next tests/test_perf_decode.py -k test_traced_layer_follows_step_inputs
    tools/run_device_tests.sh perf_selection tests/test_perf_decode.py -k test_selection_ops
(with HY4_DECODE_WEIGHT_CACHE naming the decode weight cache; test_selection_ops takes a minute,
its labelled top-k being 0.26 to 0.52 s a call).
``128K-8users`` and the two ``one-million`` cases read stripes of 32,768 rows or more: they run
last and alone. Every case writes its numbers to $HY4_PERF_RESULTS/<case>.json (default
/mnt/data/zni/hy4/logs/perf) and logs them as "MEASURED perf ..." lines.
"""

import gc
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.attention import rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import NEG
from models.experimental.hy4_preview.tests.common import GALAXY_MESH, HY4_HF_MODEL_ENV
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.attention_decode import begin_decode_step
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.decoder_layer import streams_to_torch
from models.experimental.hy4_preview.tt.ihc import ihc_expand
from models.experimental.hy4_preview.tt.indexer import Picks
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, KvCacheConfig, TtHy4KvCache, append_index_table
from models.experimental.hy4_preview.tt.model import WEIGHT_CACHE_ENV, Checkpoint, ModelConfig, TtHy4Model

WEIGHT_CACHE = os.environ.get(WEIGHT_CACHE_ENV, "/mnt/data/zni/hy4/decode-cache/weights")
RESULTS = Path(os.environ.get("HY4_PERF_RESULTS", "/mnt/data/zni/hy4/logs/perf"))
TARGET_USERS = 8
TARGET_LAYER_MS = 1e3 / 75 / 78  # 8 users x 1M context at 75 tokens per second per user
# DRAM kept for trace buffers on every chip; a traced layer of 8 users is some 2000 programs.
TRACE_REGION_BYTES = 128 << 20
LAYERS = 3
KINDS = {0: "dense FFN + full indexer", 1: "MoE + full indexer", 2: "MoE + shared indexer"}
# How many of the model's 78 layers are of each kind.
KIND_COUNTS = {0: 1, 1: 20, 2: 57}

STAGED_REPEATS = 5
QUEUED_REPEATS = 8
TRACED_REPEATS = 20
TRACED_STAGE_REPEATS = 20
FLOOR_CALLS = 32

# id -> (positions per user, users of the cache)
CASES = {
    "4K-1user": (4096, 1),
    "4K-8users": (4096, 8),
    "128K-1user": (131072, 1),
    "128K-8users": (131072, 8),
    "one-million-1user": (1048576, 1),
    "one-million-8users": (1048576, 8),
}

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION_BYTES), GALAXY_MESH)],
    indirect=True,
    ids=["8x4-torus"],
)


def _checkpoint():
    root = os.environ.get(HY4_HF_MODEL_ENV)
    if not root or not os.path.isfile(os.path.join(root, "model.safetensors.index.json")):
        pytest.skip(f"{HY4_HF_MODEL_ENV} does not point at the checkpoint")
    return Checkpoint(root)


# ---------------------------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------------------------
class OpCounter:
    """Counts the ttnn operations called while it is open, by name: every FastOperation of the
    ttnn namespaces the decode modules call is wrapped for that time. A call is what the host
    issues; a composite (an all-reduce, a repeat) is one call and several device programs."""

    def __init__(self):
        self.counts = {}
        self._saved = []

    def __enter__(self):
        kind = type(ttnn.matmul)
        for module in (ttnn, ttnn.experimental, ttnn.kv_cache):
            for name in dir(module):
                op = getattr(module, name)
                if isinstance(op, kind):
                    self._saved.append((module, name, op))
                    setattr(module, name, self._counting(name, op))
        return self

    def _counting(self, name, op):
        def call(*args, **kwargs):
            self.counts[name] = self.counts.get(name, 0) + 1
            return op(*args, **kwargs)

        return call

    def __exit__(self, *exc):
        for module, name, op in self._saved:
            setattr(module, name, op)
        self._saved = []

    def take(self) -> dict:
        out, self.counts = self.counts, {}
        return out


def _sync(mesh_device) -> float:
    ttnn.synchronize_device(mesh_device)
    return time.perf_counter()


def _idle_sync_ms(mesh_device, repeats: int = 20) -> float:
    start = _sync(mesh_device)
    for _ in range(repeats):
        ttnn.synchronize_device(mesh_device)
    return 1e3 * (time.perf_counter() - start) / repeats


def _replay_ms(mesh_device, trace_id, repeats: int) -> float:
    """ms per replay of a captured trace: ``repeats`` replays enqueued back to back, one
    synchronize at the end (a blocking replay each would add a synchronize, 0.1 ms, to each)."""
    start = _sync(mesh_device)
    for _ in range(repeats):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    return 1e3 * (_sync(mesh_device) - start) / repeats


def _capture(mesh_device, fn):
    """Capture ``fn()`` as a trace. Returns (trace id or None, what fn returned, error or None).
    A capture that fails is ended and released, and reported instead of raised: what a capture
    refuses is a result of this test."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    error = result = None
    try:
        result = fn()
    except Exception as failure:
        error = f"{type(failure).__name__}: {str(failure)[:600]}"
    try:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    except Exception as failure:
        return None, None, f"{error or ''} | end_trace_capture: {str(failure)[:300]}"
    if error is not None:
        ttnn.release_trace(mesh_device, trace_id)
        return None, None, error
    return trace_id, result, None


def _measure_stages(mesh_device, stages, state: dict, idle_ms: float) -> dict:
    """``stages`` is [(name, fn(state))] in the order they run; a name starting with "_" is glue
    between stages (a slice that hands one user's row on): run, not reported. Three passes over
    the chain, each leaving ``state`` as a plain run would:

    1. once, to compile;
    2. STAGED_REPEATS times untraced with a synchronize after every stage: ``issue_ms`` (until the
       stage's last ttnn call returned) and ``wall_ms`` (to the end of the synchronize, an idle
       synchronize taken off), medians;
    3. once under an OpCounter: ``calls`` (ttnn operations the stage issues) and ``ops`` by name.
    """
    for _, fn in stages:
        fn(state)
    _sync(mesh_device)

    runs = {name: [] for name, _ in stages}
    for _ in range(STAGED_REPEATS):
        for name, fn in stages:
            start = _sync(mesh_device)
            fn(state)
            issued = time.perf_counter()
            done = _sync(mesh_device)
            runs[name].append((1e3 * (issued - start), 1e3 * (done - start)))
    out = {}
    for name, rows in runs.items():
        if not name.startswith("_"):
            out[name] = {
                "issue_ms": statistics.median(row[0] for row in rows),
                "wall_ms": statistics.median(row[1] for row in rows) - idle_ms,
                "traced_ms": None,
            }

    with OpCounter() as counter:
        for name, fn in stages:
            fn(state)
            ops = counter.take()
            if name in out:
                out[name]["calls"] = sum(ops.values())
                out[name]["ops"] = ops
    _sync(mesh_device)
    return out


def _trace_stages(mesh_device, stages, state: dict, rows: dict) -> str | None:
    """Every stage captured as a trace of its own, replayed and released before the next stage is
    captured: ``rows[name]["traced_ms"]``, the stage's time on the device with no host work per
    op. Returns the error of the first stage that does not capture (the pass ends there), else None.

    The values in ``state`` are not a plain run's afterwards: a stage that frees one of its inputs
    (a collective does) reads freed memory from its second replay on. The time of a program does
    not depend on the values; nothing computed here is used again.
    """
    for name, fn in stages:
        if name not in rows or rows[name]["calls"] == 0:
            fn(state)  # glue, or nothing to capture
            continue
        trace_id, _, error = _capture(mesh_device, lambda: fn(state))
        if trace_id is None:
            rows[name]["trace_error"] = error
            return f"{name}: {error}"
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            rows[name]["traced_ms"] = _replay_ms(mesh_device, trace_id, TRACED_STAGE_REPEATS)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
    return None


# ---------------------------------------------------------------------------------------------
# The stages of a layer and of the step around the layers
# ---------------------------------------------------------------------------------------------
def _local_top(indexer, masked):
    """TtHy4Indexer._select, first part: every chip's own top-k candidates of its masked scores."""
    c = indexer.cfg
    slots = masked.shape[-1]
    slot_position = indexer.slot_positions(slots)
    if slots <= c.topk:
        return masked, slot_position
    _, _, cand_slots = indexer._top(masked, c.topk, indexer.refine_passes)
    return ttnn.gather(masked, dim=-1, index=cand_slots), ttnn.gather(slot_position, dim=-1, index=cand_slots)


def _merge(indexer, cand, cand_position) -> Picks:
    """TtHy4Indexer._select, second part, for one query: the candidates of all chips gathered on
    every chip and the top-k of them."""
    c, chips = indexer.cfg, indexer.num_chips
    K = cand.shape[-1]
    both = indexer._gather_all(ttnn.concat([cand, cand_position], dim=-1))
    both = ttnn.reshape(both, [1, 1, chips, 2 * K])
    scores = ttnn.reshape(ttnn.slice(both, [0, 0, 0, 0], [1, 1, chips, K]), [1, 1, 1, chips * K])
    picked = ttnn.reshape(ttnn.slice(both, [0, 0, 0, K], [1, 1, chips, 2 * K]), [1, 1, 1, chips * K])
    if chips * K > c.topk:
        _, _, index = indexer._top(scores, c.topk, indexer.refine_passes)
        scores = ttnn.gather(scores, dim=-1, index=index)
        picked = ttnn.gather(picked, dim=-1, index=index)
    valid = ttnn.gt(scores, 0.5 * NEG)
    valid = valid if valid.dtype == ttnn.float32 else ttnn.typecast(valid, ttnn.float32)
    return Picks(positions=picked, valid=valid, num_queries=1)


def _layer_stages(model: TtHy4Model, index: int, step, users: int) -> list:
    """One decoder layer as [(stage, fn(state))]. ``state["streams"]`` is the layer's input
    ([1, 1, users, 4 * hidden / 4] fp32); the result is ``state["streams_out"]``. The attention
    stages run for ``step.user`` alone."""
    layer, decode, cache, plan = model.layers[index], model.decodes[index], model.cache, model.plan
    attention, indexer, mlp = decode.attention, decode.indexer, layer.mlp
    user = step.user

    def ihc_pre_attention(s):
        s["h"], s["post"] = layer.hc_attn.hc_pre(s["streams"])

    def norm_attention(s):
        s["x"] = layer.input_layernorm(s["h"])

    def one_user(s):
        x = s["x"]
        s["xu"] = x if users == 1 else ttnn.slice(x, [0, 0, user, 0], [1, 1, user + 1, x.shape[-1]])

    def latent_projections(s):
        s["q_resid"], k_pass, k_rope = attention.latent(s["xu"], step.cos, step.sin)
        s["latent"] = cache.pack(k_pass, k_rope)

    def indexer_projections(s):
        s["index_key"], s["index_q"], s["index_w"] = indexer(s["xu"], s["q_resid"], step.cos, step.sin)

    def cache_append(s):
        key = decode._per_user(s["index_key"], step) if indexer is not None else None
        cache.write_decode(index, step.indices, latent=decode._per_user(s["latent"], step), index_key=key)

    def indexer_scoring(s):
        stripes = cache.stripe(index, INDEX_KEY, user=user, length=step.position + 1)
        s["masked"] = indexer.masked_scores(s["index_q"], s["index_w"], step.query_position, stripes)

    def local_top_k(s):
        s["cand"], s["cand_position"] = _local_top(indexer, s["masked"])

    def merge(s):
        picks = _merge(indexer, s["cand"], s["cand_position"])
        decode.index_cache = model.index_caches[user]
        decode.index_cache.store(index, picks)
        s["picks"] = picks

    def locate(s):
        if step.selection is not None:
            step.selection.deallocate()
        step.selection = decode.locate(s["picks"], step)

    def row_gather(s):
        s["gathered"] = decode.gather(step.selection)

    def collect(s):
        s["selected"] = decode.collect(s["gathered"])

    def query_projections(s):
        q, s["gate"] = decode.query_gate(s["xu"], s["q_resid"])
        s["q"] = attention.queries(q, step.cos, step.sin)

    def attention_core(s):
        core = decode.absorbed_core if decode.core == "absorb" else decode.expanded_core
        s["attn"] = core(s["q"], s["selected"], step.selection.mask)

    def output_projection(s):
        s["out_user"] = ttnn.typecast(attention.output(s["attn"], s["gate"]), layer.stream_dtype)

    def all_users(s):
        # every row gets the measured user's output: the stages after attention take all users
        s["out"] = s["out_user"] if users == 1 else ttnn.concat([s["out_user"]] * users, dim=2)

    def ihc_post_attention(s):
        s["mid"] = layer.hc_attn.hc_post(s["out"], s["streams"], s["post"])

    def ihc_pre_mlp(s):
        s["h"], s["post"] = layer.hc_mlp.hc_pre(s["mid"])

    def norm_mlp(s):
        s["x"] = layer.post_attention_layernorm(s["h"])

    def ffn_gate_up(s):
        s["act"] = mlp.activation(s["x"])

    def ffn_down(s):
        s["out"] = ttnn.typecast(mlp.down(s["act"]), layer.stream_dtype)

    def moe_input_gather(s):
        s["x_whole"] = mlp.whole(s["x"])

    def router(s):
        s["weights"] = mlp.route(s["x_whole"])

    def dispatch(s):
        s["chip_weights"] = mlp.chip_weights(s["weights"])

    def experts(s):
        s["partial"] = mlp.experts(s["x_whole"], s["chip_weights"])

    def combine(s):
        total = plan.all_reduce(plan.all_reduce(s["partial"], plan.head_axis), plan.hidden_axis)
        s["routed"] = plan.partition(total, dim=3, axis=plan.hidden_axis)

    def shared_expert(s):
        s["out"] = ttnn.typecast(ttnn.add(s["routed"], mlp.shared(s["x"])), layer.stream_dtype)

    def ihc_post_mlp(s):
        s["streams_out"] = layer.hc_mlp.hc_post(s["out"], s["mid"], s["post"])

    stages = [
        ("iHC pre-mix (attention)", ihc_pre_attention),
        ("norm (attention)", norm_attention),
        ("_one user", one_user),
        ("latent projections (q_a, kv_a, norms, RoPE)", latent_projections),
    ]
    if indexer is not None:
        stages.append(("indexer projections (key, queries, weights)", indexer_projections))
    stages.append(("cache append", cache_append))
    if indexer is not None:
        stages += [
            ("indexer scoring", indexer_scoring),
            ("local top-k", local_top_k),
            ("merge (all-gather, top-k)", merge),
            ("locate", locate),
        ]
    stages += [
        ("row gather", row_gather),
        ("collect", collect),
        ("query and gate projections (q_b, gate, RoPE)", query_projections),
        ("attention core", attention_core),
        ("output projection (gate, o_proj)", output_projection),
        ("_all users", all_users),
        ("iHC post-mix (attention)", ihc_post_attention),
        ("iHC pre-mix (MLP)", ihc_pre_mlp),
        ("norm (MLP)", norm_mlp),
    ]
    if layer.dense:
        stages += [("dense FFN gate/up", ffn_gate_up), ("dense FFN down", ffn_down)]
    else:
        stages += [
            ("MoE input all-gather", moe_input_gather),
            ("router (scores, top-8, weights)", router),
            ("dispatch (weights to chips)", dispatch),
            ("experts", experts),
            ("combine", combine),
            ("shared expert", shared_expert),
        ]
    stages.append(("iHC post-mix (MLP)", ihc_post_mlp))
    return stages


# The attention stages, which one call does for one user; the others take the users as rows.
PER_USER_STAGES = (
    "latent projections (q_a, kv_a, norms, RoPE)",
    "indexer projections (key, queries, weights)",
    "cache append",
    "indexer scoring",
    "local top-k",
    "merge (all-gather, top-k)",
    "locate",
    "row gather",
    "collect",
    "query and gate projections (q_b, gate, RoPE)",
    "attention core",
    "output projection (gate, o_proj)",
)


def _step_stages(model: TtHy4Model) -> list:
    """What a step does around the layers: ``state["tokens"]`` (device ids) -> ``state["streams"]``,
    and ``state["streams_out"]`` -> ``state["next"]``."""
    n = model.cfg.decoder.ihc.n
    head = model.lm_head

    def embedding(s):
        s["embedded"] = model.embedding(model.embedding.table_rows(s["tokens"]))

    def expand(s):
        s["streams"] = ihc_expand(s["embedded"], n)

    def head_readout(s):
        s["readout"] = model.hc_head(s["streams_out"])

    def final_norm(s):
        s["normed"] = head.norm(head.whole(s["readout"]))

    def lm_head(s):
        s["logits"] = head.logits(s["normed"])

    def greedy(s):
        s["next"] = model.greedy(s["logits"])

    return [
        ("embedding (lookup, spread)", embedding),
        ("iHC expand", expand),
        ("head readout (iHC head)", head_readout),
        ("final norm (all-gather, norm)", final_norm),
        ("LM head", lm_head),
        ("greedy (top-k per chip, all-gather, top-1)", greedy),
    ]


def _streams_row(plan, x, n: int, row: int) -> torch.Tensor:
    return streams_to_torch(plan, x, n)[row]


def _queued(mesh_device, fn, repeats: int = QUEUED_REPEATS) -> float:
    """ms per call of ``fn`` issued ``repeats`` times back to back, one synchronize at the end."""
    start = _sync(mesh_device)
    for _ in range(repeats):
        fn()
    return 1e3 * (_sync(mesh_device) - start) / repeats


def _traced(mesh_device, fn, read) -> dict:
    """Capture ``fn`` as a trace and replay it. Returns {"captured", "error", "replay_ms"
    (replays back to back, one synchronize at the end), "blocking_ms" (median of blocking
    replays), "output": read(what fn returned) after the last replay}."""
    trace_id, result, error = _capture(mesh_device, fn)
    out = {"captured": trace_id is not None, "error": error}
    if trace_id is None:
        return out
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        blocking = []
        for _ in range(TRACED_REPEATS):
            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            blocking.append(1e3 * (time.perf_counter() - start))
        out["blocking_ms"] = statistics.median(blocking)
        out["blocking_min_ms"] = min(blocking)
        out["replay_ms"] = _replay_ms(mesh_device, trace_id, TRACED_REPEATS)
        out["output"] = read(result)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return out


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------------------------
@galaxy
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.timeout(0)
def test_decode_stages(mesh_device, case):
    """Time and memory of a decode step by stage, for the three kinds of layer, at one context
    length and one number of users (module docstring). Asserts only that the staged chain and the
    traced replay reproduce the layer's own output; the numbers are the result."""
    context, users = CASES[case]
    ckpt = _checkpoint()
    plan = MeshPlan(mesh_device)
    cfg = ModelConfig(num_layers=LAYERS)
    n = cfg.decoder.ihc.n
    full = tuple(layer for layer in range(LAYERS) if cfg.is_full(layer))
    cache_cfg = KvCacheConfig(
        num_layers=LAYERS, indexer_layers=full, max_users=users, max_context=context, dtype="bfp8"
    )
    tokens_cached = users * cache_cfg.capacity
    result = {
        "case": case,
        "context": context,
        "users": users,
        "slots_per_chip": cache_cfg.slots,
        "target_layer_ms": TARGET_LAYER_MS,
        "kinds": KINDS,
        "layers": {},
    }

    def allocated():
        ttnn.synchronize_device(mesh_device)
        return pb.dram(mesh_device)["allocated_gb"]

    memory = {"total_gb_per_chip": pb.dram(mesh_device)["total_gb"], "open": allocated()}
    # the bytes of one latent stripe alone, where it is cheap to allocate one more
    if tokens_cached <= 8 * 131072:
        before = allocated()
        probe = TtHy4KvCache(mesh_device, cache_cfg, layers=[LAYERS - 1])
        memory["latent_stripe_gb_per_chip"] = allocated() - before
        probe.deallocate()
        del probe
    before = allocated()
    start = time.perf_counter()
    cache = TtHy4KvCache(mesh_device, cache_cfg)
    memory["cache_gb_per_chip"] = allocated() - before
    memory["cache_allocate_s"] = time.perf_counter() - start
    memory["cache_formula_gb_per_chip"] = cache_cfg.bytes_per_chip() / 1e9
    chips = mesh_device.get_num_devices()
    if "latent_stripe_gb_per_chip" in memory:
        latent = memory["latent_stripe_gb_per_chip"]
        index_key = (memory["cache_gb_per_chip"] - LAYERS * latent) / len(full)
        memory["latent_bytes_per_token_per_layer"] = latent * 1e9 * chips / tokens_cached
        memory["index_key_bytes_per_token_per_layer"] = index_key * 1e9 * chips / tokens_cached
    memory["cache_bytes_per_token_3_layers"] = memory["cache_gb_per_chip"] * 1e9 * chips / tokens_cached

    marks = [("cache", allocated())]
    start = time.perf_counter()
    model = TtHy4Model(
        plan, cfg, cache, ckpt, weight_cache=WEIGHT_CACHE, progress=lambda name: marks.append((name, allocated()))
    )
    memory["load_s"] = time.perf_counter() - start
    memory["weights_gb_per_chip"] = {name: now - marks[i][1] for i, (name, now) in enumerate(marks[1:])}
    result["memory"] = memory
    logger.info(
        f"MEASURED perf {case} memory per chip (GB of {memory['total_gb_per_chip']:.2f}): cache of {users} user(s) x "
        f"{cache_cfg.capacity} positions x {LAYERS} layers ({len(full)} with index keys) {memory['cache_gb_per_chip']:.4f} "
        f"(formula {memory['cache_formula_gb_per_chip']:.4f}), "
        + (
            f"latent {memory['latent_bytes_per_token_per_layer']:.1f} B and index key "
            f"{memory['index_key_bytes_per_token_per_layer']:.1f} B per token per layer over the Galaxy, "
            if "latent_bytes_per_token_per_layer" in memory
            else ""
        )
        + "weights "
        + ", ".join(f"{name} {gb:.4f}" for name, gb in memory["weights_gb_per_chip"].items())
        + f"; cache allocated in {memory['cache_allocate_s']:.1f} s, weights loaded in {memory['load_s']:.1f} s"
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    trace = os.environ.get("HY4_PERF_TRACE", "1") != "0"
    # the realtime program profiler would give kernel time per program; it is off on a host
    # whose chips sit in identity IOMMU domains
    result["realtime_profiler_active"] = bool(ttnn.device.IsProgramRealtimeProfilerActive())

    def save():
        (RESULTS / f"{case}.json").write_text(json.dumps(result, indent=1, default=str))

    def log_stages(where, rows, per_user_note):
        for name, row in rows.items():
            note = " [one user]" if per_user_note and name in PER_USER_STAGES else ""
            logger.info(
                f"MEASURED perf {case} {where} stage | {name}{note}: traced {_fmt(row['traced_ms'])} ms, untraced "
                f"staged wall {_fmt(row['wall_ms'])} ms (host issue {_fmt(row['issue_ms'])}), {row['calls']} ttnn calls"
            )

    position = cache_cfg.capacity - 64
    attention_cfg = cfg.decoder.attention
    steps = []
    try:
        idle_ms = _idle_sync_ms(mesh_device)
        result["idle_sync_ms"] = idle_ms
        result["position"] = position

        # ---- the step's host inputs --------------------------------------------------------------
        token_ids = torch.arange(users) * 1009 + 17
        uploads = []
        for _ in range(STAGED_REPEATS + 1):
            for step in steps:
                step.end()
            start = _sync(mesh_device)
            tokens = model.embedding.host_token_ids(token_ids)
            steps = [begin_decode_step(plan, cache, attention_cfg, position, user=user) for user in range(users)]
            uploads.append(1e3 * (_sync(mesh_device) - start) - idle_ms)
        result["host_inputs_ms"] = statistics.median(uploads[1:])

        # ---- the layers as the model runs them: inputs and reference outputs -----------------------
        state = {"tokens": tokens}
        step_stages = _step_stages(model)
        for _, fn in step_stages[:2]:
            fn(state)
        streams = [state["streams"]]
        for layer in model.layers:
            streams.append(layer(streams[-1], steps=steps))
        state["streams_out"] = streams[-1]
        for _, fn in step_stages[2:]:
            fn(state)
        _sync(mesh_device)

        # ---- untraced: around the layers, then each kind of layer --------------------------------
        result["step"] = _measure_stages(mesh_device, step_stages, state, idle_ms)
        forwards, layer_states, layer_stages = {}, {}, {}
        for index in range(LAYERS):
            entry = result["layers"][index] = {"kind": KINDS[index]}
            layer_states[index] = layer_state = {"streams": streams[index]}
            layer_stages[index] = stages = _layer_stages(model, index, steps[0], users)
            entry["stages"] = _measure_stages(mesh_device, stages, layer_state, idle_ms)
            got = _streams_row(plan, layer_state["streams_out"], n, 0)
            want = _streams_row(plan, streams[index + 1], n, 0)
            entry["staged_vs_forward_max_abs"] = float((got - want).abs().max())
            scale = float(want.abs().max())
            assert torch.allclose(
                got, want, rtol=0, atol=1e-3 * scale
            ), f"layer {index}: the staged chain is not the layer's forward (max diff {entry['staged_vs_forward_max_abs']})"

            def forward(index=index):
                return model.layers[index](streams[index], steps=steps)

            forwards[index] = forward
            forward()  # leaves this layer's selection in the step for the layer above
            entry["queued_ms"] = _queued(mesh_device, forward)
            logger.info(
                f"MEASURED perf {case} layer {index} ({KINDS[index]}), context {context}, {users} user(s) | untraced, "
                f"issued back to back: {entry['queued_ms']:.2f} ms per layer "
                f"({entry['queued_ms'] / TARGET_LAYER_MS:.0f}x the {TARGET_LAYER_MS:.3f} ms budget)"
            )
            save()

        # ---- a whole step of the three layers, as the model runs it ------------------------------
        def whole_step():
            logits = model.decode_step(token_ids, [position] * users)
            return model.greedy(logits)

        whole_step()
        result["whole_step_ms"] = _queued(mesh_device, whole_step, repeats=4)
        result["l1_small"] = pb.l1_small(mesh_device)
        logger.info(
            f"MEASURED perf {case} whole step of {LAYERS} layers with embedding, head and greedy, {users} user(s), "
            f"untraced: {result['whole_step_ms']:.2f} ms; host inputs of a step {result['host_inputs_ms']:.2f} ms; "
            f"idle synchronize {idle_ms:.3f} ms; L1_SMALL in use {result['l1_small']}"
        )
        save()

        # ---- traced: stage by stage, then every layer whole --------------------------------------
        failed = None
        if trace:
            failed = _trace_stages(mesh_device, step_stages, state, result["step"])
        for index in range(LAYERS):
            if not trace or failed:
                break
            entry = result["layers"][index]
            for below in range(index + 1):  # the step holds the selection this layer's stages read
                want = _streams_row(plan, forwards[below](), n, 0)
            failed = _trace_stages(mesh_device, layer_stages[index], layer_states[index], entry["stages"])
            if failed:
                break
            want = _streams_row(plan, forwards[index](), n, 0)
            traced = _traced(mesh_device, forwards[index], lambda out: _streams_row(plan, out, n, 0))
            got = traced.pop("output", None)
            if got is not None:
                traced["vs_untraced_max_abs"] = float((got - want).abs().max())
                traced["vs_untraced_scale"] = float(want.abs().max())
            entry["traced"] = traced
            if not traced["captured"]:
                failed = f"layer {index}: {traced['error']}"
        result["trace_failed"] = failed
        save()

        # ---- the tables --------------------------------------------------------------------------
        log_stages("step", result["step"], False)
        for index in range(LAYERS):
            entry = result["layers"][index]
            rows = entry["stages"]
            log_stages(f"layer {index} ({KINDS[index]})", rows, users > 1)

            def total(key, names):
                # a stage that issues nothing (the local top-k of a stripe no longer than the top-k) is zero
                values = [0.0 if rows[name]["calls"] == 0 else rows[name][key] for name in names]
                return None if any(v is None for v in values) else sum(values)

            once = [name for name in rows if name not in PER_USER_STAGES]
            per = [name for name in rows if name in PER_USER_STAGES]
            entry["sum"] = sums = {}
            for key in ("traced_ms", "wall_ms", "issue_ms", "calls"):
                a, b = total(key, once), total(key, per)
                sums[key] = {
                    "batched_stages": a,
                    "attention_one_user": b,
                    "layer_all_users": None if a is None or b is None else a + users * b,
                }
            traced = entry.get("traced", {})
            logger.info(
                f"MEASURED perf {case} layer {index} ({KINDS[index]}), context {context}, {users} user(s) | sums of "
                f"the stages: traced {_fmt(sums['traced_ms']['layer_all_users'])} ms (stages that take the users as "
                f"rows {_fmt(sums['traced_ms']['batched_stages'])} + {users} x attention of one user "
                f"{_fmt(sums['traced_ms']['attention_one_user'])}), untraced staged wall "
                f"{_fmt(sums['wall_ms']['layer_all_users'])} ms, host issue {_fmt(sums['issue_ms']['layer_all_users'])} ms, "
                f"{sums['calls']['layer_all_users']} ttnn calls | whole layer untraced {entry['queued_ms']:.2f} ms, "
                f"TRACED {_fmt(traced.get('replay_ms'))} ms replayed back to back and {_fmt(traced.get('blocking_ms'))} ms "
                f"blocking; budget {TARGET_LAYER_MS:.3f} ms"
                + (
                    f" ({traced['replay_ms'] / TARGET_LAYER_MS:.0f}x); traced output vs untraced max |diff| "
                    f"{traced.get('vs_untraced_max_abs')} on values up to {traced.get('vs_untraced_scale')}"
                    if traced.get("replay_ms")
                    else ""
                )
            )
        result["dram_end_gb_per_chip"] = allocated()
        save()
        assert not failed, f"a trace capture failed: {failed}"
        for index in range(LAYERS):
            traced = result["layers"][index].get("traced")
            if traced:
                assert traced["vs_untraced_max_abs"] <= 1e-3 * traced["vs_untraced_scale"], (
                    f"layer {index}: the traced replay is not the layer's forward "
                    f"(max diff {traced['vs_untraced_max_abs']})"
                )
    finally:
        for step in steps:
            try:
                step.end()
            except Exception:  # a tensor a failed capture already freed
                pass
        cache.deallocate()
        del model
        gc.collect()


@galaxy
@pytest.mark.timeout(0)
def test_op_floor(mesh_device):
    """What one device program costs on the 32 chips when it has next to nothing to compute, and
    what one collective costs on each mesh axis: the per-op floors a decode step is built from.
    Each op is issued FLOOR_CALLS times, untraced (one synchronize at the end) and as one trace
    replayed. Nothing is asserted; the numbers go to $HY4_PERF_RESULTS/op_floor.json."""
    dram = ttnn.DRAM_MEMORY_CONFIG

    def up(shape, dtype):
        return ttnn.from_torch(torch.randn(shape), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=dtype)

    tile, other = up((1, 1, 32, 32), ttnn.bfloat16), up((1, 1, 32, 32), ttnn.bfloat16)
    row = up((1, 1, 1, 1536), ttnn.float32)
    rows8 = up((1, 1, 8, 1536), ttnn.float32)
    weight = up((1, 1, 1536, 32), ttnn.bfloat16)
    picked = up((1, 1, 2048, 576), ttnn.bfloat16)
    ops = {
        "add, [32 x 32] bf16": lambda: ttnn.add(tile, other),
        "multiply by a scalar, [32 x 32] bf16": lambda: ttnn.multiply(tile, 0.5),
        "typecast bf16 -> fp32, [32 x 32]": lambda: ttnn.typecast(tile, ttnn.float32),
        "slice of 16 columns, [32 x 32] bf16": lambda: ttnn.slice(tile, [0, 0, 0, 0], [1, 1, 32, 16]),
        "concat of two, [32 x 32] bf16": lambda: ttnn.concat([tile, other], dim=-1),
        "add, [1 x 1536] fp32": lambda: ttnn.add(row, row),
        "add, [8 x 1536] fp32": lambda: ttnn.add(rows8, rows8),
        "matmul, [1 x 1536] fp32 x [1536 x 32] bf16": lambda: ttnn.matmul(row, weight),
        "all_reduce over mesh axis 1 (4 chips, hidden), [32 x 32] bf16": lambda: ttnn.all_reduce(
            tile, cluster_axis=1, memory_config=dram
        ),
        "all_reduce over mesh axis 0 (8 chips, heads), [32 x 32] bf16": lambda: ttnn.all_reduce(
            tile, cluster_axis=0, memory_config=dram
        ),
        "all_gather over mesh axis 1 (4 chips), [32 x 32] bf16": lambda: ttnn.all_gather(
            tile, dim=3, cluster_axis=1, memory_config=dram
        ),
        "all_gather over mesh axis 0 (8 chips), [32 x 32] bf16": lambda: ttnn.all_gather(
            tile, dim=3, cluster_axis=0, memory_config=dram
        ),
        "all_reduce over mesh axis 1 (4 chips), [8 x 1536] fp32": lambda: ttnn.all_reduce(
            rows8, cluster_axis=1, memory_config=dram
        ),
        "all_reduce over mesh axis 1 (4 chips), [2048 x 576] bf16": lambda: ttnn.all_reduce(
            picked, cluster_axis=1, memory_config=dram
        ),
        "all_reduce over mesh axis 0 (8 chips), [2048 x 576] bf16": lambda: ttnn.all_reduce(
            picked, cluster_axis=0, memory_config=dram
        ),
    }
    result = {"calls": FLOOR_CALLS, "idle_sync_ms": _idle_sync_ms(mesh_device), "ops": {}}
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name, fn in ops.items():

        def many(fn=fn):
            for _ in range(FLOOR_CALLS):
                fn()

        fn()
        _sync(mesh_device)
        entry = result["ops"][name] = {"untraced_ms": _queued(mesh_device, many, repeats=4) / FLOOR_CALLS}
        trace_id, _, error = _capture(mesh_device, many)
        if trace_id is None:
            entry["trace_error"] = error
            logger.info(f"MEASURED perf floor | {name}: untraced {entry['untraced_ms']:.4f} ms, trace FAILED: {error}")
            (RESULTS / "op_floor.json").write_text(json.dumps(result, indent=1))
            break
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            entry["traced_ms"] = _replay_ms(mesh_device, trace_id, 8) / FLOOR_CALLS
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        logger.info(
            f"MEASURED perf floor | {name}: untraced {entry['untraced_ms']:.4f} ms per call, traced "
            f"{entry['traced_ms']:.4f} ms per call"
        )
        (RESULTS / "op_floor.json").write_text(json.dumps(result, indent=1))
    assert all("trace_error" not in entry for entry in result["ops"].values())


def _rewrite_step(plan, cache, attention_cfg, step, position: int):
    """Put ``position`` into a DecodeStep's four device inputs in place (the tensors a captured
    trace reads): cos, sin, the append indices and the query position. These are the host uploads
    of begin_decode_step, written into the tensors that exist instead of new ones."""
    mesh_device = plan.mesh_device
    cos, sin = frames.rope_tables(*rope_cos_sin(torch.tensor([position]), attention_cfg))

    def replicated(t, dtype):
        return ttnn.from_torch(
            t.float(), layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )

    ttnn.copy_host_to_device_tensor(replicated(cos.reshape(1, 1, 1, -1), ttnn.bfloat16), step.cos)
    ttnn.copy_host_to_device_tensor(replicated(sin.reshape(1, 1, 1, -1), ttnn.bfloat16), step.sin)
    ttnn.copy_host_to_device_tensor(
        replicated(torch.tensor([float(position)]).reshape(1, 1, 1, 1), ttnn.float32), step.query_position
    )
    users = cache.cfg.max_users
    table = append_index_table(
        cache.layout, [position if u == step.user else -1 for u in range(users)], cache.cfg.capacity
    )
    shards = [
        ttnn.from_torch(table[chip].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        for chip in range(table.shape[0])
    ]
    ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, mesh_device.shape), step.indices)


@galaxy
@pytest.mark.timeout(0)
def test_traced_layer_follows_step_inputs(mesh_device):
    """A traced layer replayed for the next token. The trace of a full MoE layer is captured at
    position P; the step's four inputs are then rewritten in place for P + 1 and the same trace is
    replayed. What it writes into the cache (the latent row and the index key of P + 1) and what
    it returns must be what an untraced step at P + 1 writes and returns. So the per-step tables
    do not break a trace as long as they live in tensors that exist before the capture; the
    uploads themselves cannot be inside one (the command queue refuses a write while it records).
    Both positions are in the same round of 4096, so the captured stripe length still fits."""
    ckpt = _checkpoint()
    plan = MeshPlan(mesh_device)
    cfg = ModelConfig(num_layers=LAYERS)
    n = cfg.decoder.ihc.n
    full = tuple(layer for layer in range(LAYERS) if cfg.is_full(layer))
    cache = TtHy4KvCache(
        mesh_device, KvCacheConfig(num_layers=LAYERS, indexer_layers=full, max_users=1, max_context=4096, dtype="bfp8")
    )
    model = TtHy4Model(plan, cfg, cache, ckpt, weight_cache=WEIGHT_CACHE)
    attention_cfg = cfg.decoder.attention
    index, position = 1, 1000
    g = torch.Generator().manual_seed(7)
    for layer in range(LAYERS):  # a context to select from and attend over
        keys = torch.randn(position, cache.cfg.index_key_dim, generator=g) if layer in full else None
        cache.write_prefill_host(layer, 0, torch.randn(position, cache.cfg.latent_dim, generator=g), keys)
    steps = []
    try:
        steps = [begin_decode_step(plan, cache, attention_cfg, position)]
        rows = model.embedding.host_table_rows(torch.tensor([17]))
        x = model.layers[0](ihc_expand(model.embedding(rows), n), steps=steps)

        def forward():
            return model.layers[index](x, steps=steps)

        def read(out):
            return _streams_row(plan, out, n, 0)

        def cached(at):
            return (
                cache.to_torch(index, 0, at + 1)[at].clone(),
                cache.to_torch(index, 0, at + 1, INDEX_KEY)[at].clone(),
            )

        forward()
        at_p = read(forward())
        trace_id, out, error = _capture(mesh_device, forward)
        assert trace_id is not None, error
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            traced_p = read(out)
            rewrites = []
            for target in (position + 1, position, position + 1):
                start = _sync(mesh_device)
                _rewrite_step(plan, cache, attention_cfg, steps[0], target)
                rewrites.append(1e3 * (_sync(mesh_device) - start))
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            traced_next = read(out)
            traced_latent, traced_key = cached(position + 1)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        steps[0].end()
        steps = [begin_decode_step(plan, cache, attention_cfg, position + 1)]
        want_next = read(forward())
        want_latent, want_key = cached(position + 1)
        scale = float(want_next.abs().max())
        diffs = {
            "output at P, traced vs untraced": float((traced_p - at_p).abs().max()),
            "output at P + 1, traced vs untraced": float((traced_next - want_next).abs().max()),
            "latent row P + 1": float((traced_latent - want_latent).abs().max()),
            "index key P + 1": float((traced_key - want_key).abs().max()),
        }
        moved = float((want_next - at_p).abs().max())
        logger.info(
            f"MEASURED perf traced layer {index} ({KINDS[index]}) at the next position: max |diff| {diffs} on outputs up "
            f"to {scale:.4f}; the output moved by {moved:.4f} from P to P + 1; rewriting the step's four inputs in "
            f"place took {statistics.median(rewrites):.3f} ms (one user, with a synchronize)"
        )
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "trace_next_position.json").write_text(
            json.dumps({"diffs": diffs, "scale": scale, "moved": moved, "rewrite_ms": rewrites}, indent=1)
        )
        assert moved > 1e-3 * scale, "the two positions give the same output: the check shows nothing"
        assert float(want_latent.abs().max()) > 0, "the untraced step wrote no row at P + 1"
        for name, diff in diffs.items():
            assert diff <= 1e-3 * scale, f"{name}: {diff}"
    finally:
        for step in steps:
            step.end()
        cache.deallocate()
        del model
        gc.collect()


# (rows, width): one user's and eight users' scores of a chip's stripe at 1M (32,768 slots), and
# the candidates of the 32 chips in the merge (65,536).
TOPK_SHAPES = ((1, 32768), (8, 32768), (1, 65536), (8, 65536))


@galaxy
@pytest.mark.timeout(0)
def test_selection_ops(mesh_device):
    """The two ops the selection's top-2048 is made of (TtHy4Indexer._top and the gathers after
    it), each alone, traced, with the users as rows of one call ([1, 1, rows, width]; one row and
    eight rows sit in the same row of tiles): ttnn.topk on the bf16 scores, the same with a tensor
    of labels riding along (``indices_tensor``), and ttnn.gather of the fp32 scores at the picked
    columns. Nothing is asserted; the numbers go to $HY4_PERF_RESULTS/selection_ops.json."""
    result = {}
    RESULTS.mkdir(parents=True, exist_ok=True)
    repeats = 4

    def timed(name, fn):
        def many():
            for _ in range(repeats):
                fn()

        many()
        _sync(mesh_device)
        untraced = _queued(mesh_device, many, repeats=2) / repeats
        trace_id, _, error = _capture(mesh_device, many)
        assert trace_id is not None, error
        try:
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            traced = _replay_ms(mesh_device, trace_id, 4) / repeats
        finally:
            ttnn.release_trace(mesh_device, trace_id)
        result[name] = {"untraced_ms": untraced, "traced_ms": traced}
        logger.info(f"MEASURED perf selection op | {name}: untraced {untraced:.3f} ms, traced {traced:.3f} ms per call")
        (RESULTS / "selection_ops.json").write_text(json.dumps(result, indent=1))

    for rows, width in TOPK_SHAPES:
        scores = torch.randn(1, 1, rows, width)
        x16 = ttnn.from_torch(scores, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16)
        x32 = ttnn.from_torch(scores, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.float32)
        labels = ttnn.from_torch(
            torch.arange(width, dtype=torch.int32).expand(1, 1, rows, width).contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        shape = f"[{rows} x {width}]"
        timed(f"topk 2048 of bf16 {shape}", lambda: ttnn.topk(x16, k=2048, dim=-1, largest=True, sorted=True))
        timed(
            f"topk 2048 of bf16 {shape} with uint32 labels",
            lambda: ttnn.topk(x16, k=2048, dim=-1, largest=True, sorted=True, indices_tensor=labels),
        )
        _, picked = ttnn.topk(x16, k=2048, dim=-1, largest=True, sorted=True)
        timed(f"gather 2048 columns of fp32 {shape}", lambda: ttnn.gather(x32, dim=-1, index=picked))
        for t in (x16, x32, labels, picked):
            t.deallocate(True)
