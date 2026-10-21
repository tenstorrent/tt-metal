# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op / per-module cost of ONE long-context prefill chunk on a mesh block (the 1 M-token run's unit of work).

The resident mesh model (``tt/mesh_model.py``) is built for ``GLM53_PROF_LAYERS`` (default 0-21: all three layer
kinds, fits a (2,4) block in bfp8), the DSA layers' latent cache + pooled-key index are seeded synthetically to
``GLM53_PROF_PREFIX`` positions (``TtDSA.prefill_state`` on realistic-statistics rows, the ``test_cache_1m`` method:
no attention output, so seeding costs seconds; the KDA carries start from zeros, which the chunk's cost does not
depend on), then ONE chunk of ``GLM53_PROF_CHUNK`` real prompt tokens (the needle prompt of
``test_mesh_long_context``) is prefilled at ``start = prefix`` through every layer and the head, three times:

* ``compile``: the first pass (kernel compiles);
* ``free``: un-instrumented, one ``synchronize_device`` per layer (the per-layer wall table, the chunk time);
* ``attributed``: every module call bracketed by a device synchronization (``_ModuleTimer``): exclusive wall time
  per (layer kind, module) for KDA / DSA (stems, index scorer + top-k, kv copy, sparse_sdpa, absorb, o_proj) / MoE
  (router, experts, shared) / dense MLP / mHC pre + post blocks / norms / collectives / embedding / head. The syncs
  serialize dispatch, so the attributed chunk is slower than the free one; the shares are what it measures.

``GLM53_PROF_TRACY=1`` (run under ``python -m tracy -r -p -v --no-web-server -m pytest ...``) adds Tracy signposts
(``PERF_PREFILL`` .. ``PERF_PREFILL_END`` around the free pass, ``L<idx>_<kind>`` per layer and the module labels
nested inside) and drains the device profiler buffers (``ReadDeviceProfiler``) after every layer and every
``GLM53_PROF_DRAIN_BLOCKS`` routed-expert blocks (a 22-layer chunk issues ~60 K device ops, far beyond the
per-core marker buffers; a profiler drain inside the chunk does not change the kernels' durations).
``tests/perf_prefill_report.py <ops csv> --out <dir>`` then attributes every op of the window by its innermost
signpost and writes the tables of ``doc/mesh/prefill_profile.md``. Profile one layer of each kind
(``GLM53_PROF_LAYERS=0,3,4``) under Tracy: every layer of a kind issues the same op sequence.

    export TT_VISIBLE_DEVICES=25,24,29,28,21,20,17,16 MESH_DEVICE=P150x8 GLM53_MESH_SHAPE=2,4 \\
        TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto
    GLM53_PROF_LAYERS=0-21 pytest models/demos/glm53_flash/tests/test_mesh_prefill_profile.py -s --timeout 1500
    GLM53_PROF_LAYERS=0,3,4 GLM53_PROF_TRACY=1 python -m tracy -r -p -v --no-web-server -m pytest ... -s

Knobs: GLM53_PROF_LAYERS (0-21), GLM53_PROF_PREFIX (65536), GLM53_PROF_CHUNK (8192), GLM53_PROF_BUILD_ONLY=1
(build the model, i.e. write the tensorbin cache, and stop), GLM53_PROF_RESULTS (JSON lines file), GLM53_MESH_CACHE
(tensorbin cache root, default /home/ttuser/ashai/glm53/_cache/mesh), GLM53_MODEL_EXPERT_DTYPE / _DOWN_DTYPE (bfp8),
GLM53_PROF_SEED_CHUNK (8192), GLM53_PROF_PASSES (compile,free,attributed). The cache is single-slot
(``GLM53_PARTITION_USERS=0``, ``max_batch=1``: the 1 M run's configuration).
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_cache_1m import _attn_keys, _x_stats, synthetic_rows
from models.demos.glm53_flash.tests.test_perf_layer import _log_dram, _read_profiler, signpost
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel

os.environ.setdefault("GLM53_PARTITION_USERS", "0")

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
PREFIX = int(os.environ.get("GLM53_PROF_PREFIX", "65536"))
CHUNK = int(os.environ.get("GLM53_PROF_CHUNK", "8192"))
SEED_CHUNK = int(os.environ.get("GLM53_PROF_SEED_CHUNK", "8192"))
TRACY = os.environ.get("GLM53_PROF_TRACY", "0") == "1"
DRAIN_BLOCKS = int(os.environ.get("GLM53_PROF_DRAIN_BLOCKS", "16"))
BUILD_ONLY = os.environ.get("GLM53_PROF_BUILD_ONLY", "0") == "1"
PASSES = os.environ.get("GLM53_PROF_PASSES", "compile,free,attributed").split(",")
RESULTS = os.environ.get("GLM53_PROF_RESULTS")
_DT = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
EXPERT_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8")]
EXPERT_DOWN_DTYPE = _DT[os.environ.get("GLM53_MODEL_EXPERT_DOWN_DTYPE", "bfp8")]
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
TILE = 32


def _layers() -> list[int]:
    spec = os.environ.get("GLM53_PROF_LAYERS", "0-21")
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def layer_kind(cfg: GLM53FlashConfig, i: int) -> str:
    return ("KDA" if cfg.is_kda(i) else "DSA") + "+" + ("MoE" if cfg.is_moe(i) else "dense")


# ----------------------------------------------------------------------------------------- attribution
class _ModuleTimer:
    """Exclusive wall time per (context, label) with a device synchronization at every label boundary, plus
    Tracy signposts (``<LABEL>_BEGIN`` / ``<LABEL>_END``) when ``signposts``. ``wrap(obj, name, label)`` replaces
    a bound method / class attribute; ``restore`` puts the originals back. Time between two boundaries is
    charged to the innermost open label (``other`` outside every label)."""

    def __init__(self, device, *, sync: bool = True, signposts: bool = False):
        self.device = device
        self.sync = sync
        self.signposts = signposts
        self.context = "-"
        self.excl: dict[tuple[str, str], float] = {}
        self.incl: dict[tuple[str, str], float] = {}
        self.calls: dict[tuple[str, str], int] = {}
        self._stack: list[tuple[str, float]] = []
        self._t: float | None = None
        self._undo = []
        self.enabled = False

    def _now(self) -> float:
        if self.sync:
            ttnn.synchronize_device(self.device)
        return time.perf_counter()

    def _charge(self, now: float):
        if self._t is not None:
            label = self._stack[-1][0] if self._stack else "other"
            key = (self.context, label)
            self.excl[key] = self.excl.get(key, 0.0) + (now - self._t)
        self._t = now

    def begin(self, label: str):
        if not self.enabled:
            return
        now = self._now()
        self._charge(now)
        self._stack.append((label, now))
        if self.signposts:
            signpost(f"{label}_BEGIN")

    def end(self, label: str):
        if not self.enabled:
            return
        now = self._now()
        self._charge(now)
        if self.signposts:
            signpost(f"{label}_END")
        if self._stack and self._stack[-1][0] == label:
            _, t0 = self._stack.pop()
            key = (self.context, label)
            self.incl[key] = self.incl.get(key, 0.0) + (now - t0)
            self.calls[key] = self.calls.get(key, 0) + 1

    def start(self, context: str):
        self.context = context
        self.enabled = True
        self._stack.clear()
        self._t = self._now()

    def stop(self):
        now = self._now()
        self._charge(now)
        self.enabled = False
        self._t = None

    def wrap(self, obj, name: str, label: str, *, label_fn=None):
        orig = getattr(obj, name)
        timer = self

        def wrapped(*args, **kwargs):
            lab = label_fn(*args, **kwargs) if label_fn is not None else label
            timer.begin(lab)
            try:
                return orig(*args, **kwargs)
            finally:
                timer.end(lab)

        setattr(obj, name, wrapped)
        self._undo.append((obj, name, orig))

    def restore(self):
        for obj, name, orig in reversed(self._undo):
            setattr(obj, name, orig)
        self._undo.clear()

    def table(self) -> list[tuple[str, str, float, float, int]]:
        keys = sorted(set(self.excl) | set(self.incl))
        return [
            (c, l, self.excl.get((c, l), 0.0), self.incl.get((c, l), 0.0), self.calls.get((c, l), 0)) for c, l in keys
        ]


def _install_labels(model: GLM53MeshModel, timer: _ModuleTimer):
    """Module labels (innermost wins): the mHC pre/post blocks, the norms, the attention (with the DSA stages),
    the MLP (router / experts / shared), the collectives, the embedding and the head."""

    for dec in model.layers:
        layer = dec.layer
        for hc, tag in ((layer.attn_hc, "ATTN_HC"), (layer.ffn_hc, "FFN_HC")):
            if hasattr(hc, "hc_pre"):
                timer.wrap(hc, "hc_pre", f"{tag}_PRE")
                timer.wrap(hc, "hc_post", f"{tag}_POST")
            else:  # TtMHCFast / TtMHCSplit: one label for the whole wrap minus the sublayer
                timer.wrap(hc, "forward", tag)
        timer.wrap(layer, "_norm", "NORM")
        if layer.is_kda:
            timer.wrap(layer.attn, "prefill_forward", "KDA")
        else:
            attn = layer.attn
            timer.wrap(attn, "prefill_forward", "DSA")
            timer.wrap(attn, "_q_resid", "DSA_STEM_Q")
            timer.wrap(attn, "_q", "DSA_STEM_Q")
            timer.wrap(attn, "_latent", "DSA_STEM_KV")
            timer.wrap(attn, "_dense_attention", "DSA_DENSE_SDPA")
            timer.wrap(attn, "_absorb_q", "DSA_ABSORB_Q")
            timer.wrap(attn, "_o_lat_to_heads", "DSA_ABSORB_O")
            timer.wrap(attn, "_o_proj", "DSA_O_PROJ")
            timer.wrap(attn.indexer, "maintain_prefill", "DSA_INDEX_MAINTAIN")
            timer.wrap(attn.indexer, "queries_and_weights", "DSA_INDEX_QUERIES")
            timer.wrap(attn.indexer, "select", "DSA_INDEX_SELECT")
            timer.wrap(attn.indexer, "_score_prefill_rows", "DSA_INDEX_SCORER")
            timer.wrap(dec.cache, "fill", "DSA_CACHE_FILL")
            timer.wrap(dec.cache, "user_rows_rm", "DSA_KV_COPY")
        mlp = layer.mlp
        if hasattr(mlp, "router"):
            timer.wrap(mlp, "forward", "MOE")
            timer.wrap(mlp.router, "forward", "MOE_ROUTER")
            timer.wrap(mlp.experts, "forward", "MOE_EXPERTS")
            timer.wrap(mlp.shared, "forward", "MOE_SHARED")
        else:
            timer.wrap(mlp, "forward", "DENSE_MLP")
    timer.wrap(model.embedding, "forward", "EMBEDDING")
    if model.head is not None:
        timer.wrap(model.head, "last_token_logits", "HEAD")
    timer.wrap(ttnn.transformer, "sparse_sdpa", "DSA_SPARSE_SDPA")
    for name, label in (
        ("all_reduce_cols", "CCL_ALL_REDUCE_COLS"),
        ("all_gather_rows", "CCL_ALL_GATHER_ROWS"),
        ("reduce_scatter_rows", "CCL_REDUCE_SCATTER_ROWS"),
        ("all_gather_cols", "CCL_ALL_GATHER_COLS"),
    ):
        timer.wrap(GLMMesh, name, label)


class _Drainer:
    """Profiler drains inside the chunk (Tracy runs): every ``every`` routed-expert blocks."""

    def __init__(self, device, every: int):
        self.device, self.every, self.n = device, every, 0
        self._undo = []

    def install(self):
        from models.demos.glm53_flash.tt.moe import TtRoutedExperts

        drainer = self
        for name in ("_block", "_block_stream", "_block_dispatch", "_block_dense"):
            orig = getattr(TtRoutedExperts, name)

            def wrapped(self_, *a, _orig=orig, **k):
                out = _orig(self_, *a, **k)
                drainer.n += 1
                if drainer.every and drainer.n % drainer.every == 0:
                    _read_profiler(drainer.device)
                return out

            setattr(TtRoutedExperts, name, wrapped)
            self._undo.append((TtRoutedExperts, name, orig))

    def restore(self):
        for obj, name, orig in reversed(self._undo):
            setattr(obj, name, orig)


# ----------------------------------------------------------------------------------------- the chunk
def seed_prefix(model: GLM53MeshModel, cfg: GLM53FlashConfig, prefix: int, seed_chunk: int):
    """Seed every DSA layer's latent pages + pooled-key index to ``prefix`` positions of user 0 through
    ``TtDSA.prefill_state`` (replicated rows on every chip: the SP rows of this model fill the replicated cache
    identically), then arm every layer's chunk bookkeeping at ``prefix``."""
    snapshot = default_snapshot_dir()
    raw3 = load_layer_raw(snapshot, 3, shard_headers(snapshot), name_filter=_attn_keys)
    mu, sigma = _x_stats(raw3, cfg)
    dsa = [dec for dec in model.layers if not dec.is_kda]
    t0 = time.perf_counter()
    filled, ci = 0, 0
    while filled < prefix:
        T = min(seed_chunk, prefix - filled)
        x_host = synthetic_rows(mu, sigma, ci, T).reshape(1, 1, T, -1)
        x_dev = model.mesh.upload(x_host, ttnn.bfloat16)
        for dec in dsa:
            dec.layer.attn.prefill_state(x_dev, dec.cache, 0, T, start=filled)
        ttnn.deallocate(x_dev)
        filled += T
        ci += 1
    ttnn.synchronize_device(model.device)
    for dec in model.layers:
        dec.next_start[0] = prefix
    logger.info(f"seeded {len(dsa)} DSA layers to {prefix} positions in {time.perf_counter() - t0:.1f} s")


def run_chunk(model: GLM53MeshModel, ids: torch.Tensor, start: int, timer: _ModuleTimer | None, on_layer=None):
    """``GLM53MeshModel.prefill`` with per-layer hooks: returns (streams, logits_last, owner_row, per-layer seconds)."""
    S = int(ids.numel())
    unit = TILE * model.mesh.rows
    T_pad = -(-S // unit) * unit
    host = model.embedding.tokens_to_host(ids, pad_to=T_pad)
    for dec in model.layers:
        dec.next_start[0] = start
    if timer is not None:
        timer.start("EMBED")
    if model.mesh.rows > 1:
        ids_dev = model.mesh.upload(host, ttnn.uint32, rows=3, layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        ids_dev = ttnn.from_torch(host, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model.device)
    streams = model.embedding(ids_dev)
    ttnn.deallocate(ids_dev)
    per_layer = []
    for dec in model.layers:
        kind = layer_kind(model.cfg, dec.layer_idx)
        if timer is not None:
            timer.stop()
            timer.start(kind)
        if TRACY:
            signpost(f"L{dec.layer_idx}_{kind}_BEGIN")
        ttnn.synchronize_device(model.device)
        t0 = time.perf_counter()
        out = dec.prefill_forward(streams, user=0, seq_len=S, start=start)
        ttnn.synchronize_device(model.device)
        per_layer.append((dec.layer_idx, kind, time.perf_counter() - t0))
        if TRACY:
            signpost(f"L{dec.layer_idx}_{kind}_END")
            _read_profiler(model.device)
        ttnn.deallocate(streams)
        streams = out
        if on_layer is not None:
            on_layer(dec.layer_idx, kind, per_layer[-1][2])
    if timer is not None:
        timer.stop()
        timer.start("HEAD")
    T_loc = T_pad // model.mesh.rows
    owner_row, local_last = divmod(S - 1, T_loc)
    logits = model.head.last_token_logits(streams, local_last + 1) if model.head is not None else None
    ttnn.synchronize_device(model.device)
    if timer is not None:
        timer.stop()
    return streams, logits, owner_row, per_layer


def _record(**kw):
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(kw) + "\n")


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mesh_prefill_chunk_profile(mesh_device):
    from transformers import AutoTokenizer

    # imported here: that module defaults GLM53_PARTITION_USERS to 1 at import time; this test's default is 0
    from models.demos.glm53_flash.tests.test_mesh_long_context import build_needle_ids

    layers = _layers()
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    rows, cols = MESH
    assert PREFIX % TILE == 0 and CHUNK % (TILE * rows) == 0, (PREFIX, CHUNK, rows)
    max_seq_len = PREFIX + CHUNK + 256
    _log_dram(mesh_device, "before build")
    t0 = time.perf_counter()
    model = GLM53MeshModel(
        mesh_device,
        cfg,
        layers,
        max_batch=1,
        max_seq_len=max_seq_len,
        weight_cache_path=CACHE_ROOT,
        expert_dtype=EXPERT_DTYPE,
        expert_down_dtype=EXPERT_DOWN_DTYPE,
    )
    build_s = time.perf_counter() - t0
    kinds = [layer_kind(cfg, i) for i in layers]
    logger.info(
        f"model: mesh {MESH} layers {layers[0]}..{layers[-1]} ({len(layers)}: "
        + ", ".join(f"{k} x{kinds.count(k)}" for k in OrderedDict.fromkeys(kinds))
        + f"), max_seq_len {max_seq_len}, built in {build_s:.0f} s "
        f"(per layer {' '.join(f'{i}:{model.build_times[i]:.0f}s' for i in layers)})"
    )
    _log_dram(mesh_device, "after build")
    if BUILD_ONLY:
        return

    # ---- the chunk's tokens: real prompt text (the router sees realistic expert distributions)
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    ids, _ = build_needle_ids(tok, CHUNK + 512, 0.5)
    assert len(ids) >= CHUNK, len(ids)
    ids = torch.tensor(ids[:CHUNK], dtype=torch.long)

    # ---- the prefix: the DSA caches / indices seeded to PREFIX positions
    seed_prefix(model, cfg, PREFIX, SEED_CHUNK)
    _log_dram(mesh_device, f"after seeding {PREFIX} positions")

    if TRACY:
        drainer = _Drainer(mesh_device, DRAIN_BLOCKS)
        drainer.install()
    results = {}
    timer = _ModuleTimer(mesh_device, sync=True, signposts=TRACY)
    out_streams = {}
    for variant in PASSES:
        model.reset_kda_states()  # every pass continues the same (zero) KDA carries: identical outputs across passes
        if variant == "attributed":
            _install_labels(model, timer)
        if TRACY:
            _read_profiler(mesh_device)
            signpost("PERF_PREFILL" if variant == "free" else f"PASS_{variant.upper()}")
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        streams, logits, owner, per_layer = run_chunk(model, ids, PREFIX, timer if variant == "attributed" else None)
        first = model.first_token(logits, owner) if logits is not None else None
        wall = time.perf_counter() - t0
        if TRACY:
            signpost("PERF_PREFILL_END" if variant == "free" else f"PASS_{variant.upper()}_END")
            _read_profiler(mesh_device)
        if variant == "attributed":
            timer.restore()
        results[variant] = {"wall_s": wall, "per_layer": per_layer, "first_token": first}
        out_streams[variant] = model.mesh.to_host(streams, rows=2).float()[0, 0, :CHUNK]
        ttnn.deallocate(streams)
        if logits is not None:
            ttnn.deallocate(logits)
        by_kind: dict[str, list[float]] = {}
        for _, kind, sec in per_layer:
            by_kind.setdefault(kind, []).append(sec)
        logger.info(
            f"PASS {variant}: chunk [{PREFIX}, {PREFIX + CHUNK}) of {CHUNK} tokens through {len(layers)} layers + head: "
            f"{wall:.2f} s ({CHUNK / wall:.0f} tok/s); first token {first} {tok.decode([first])!r}; per kind "
            + "; ".join(
                f"{k}: {sum(v):.2f} s over {len(v)} layers ({1e3 * sum(v) / len(v):.0f} ms each)"
                for k, v in by_kind.items()
            )
        )
        logger.info("    per layer: " + " ".join(f"L{i}({k}) {1e3 * s:.0f}ms" for i, k, s in per_layer))
        _log_dram(mesh_device, f"after the {variant} pass")
    if TRACY:
        drainer.restore()
    if "free" in out_streams and "attributed" in out_streams:
        same = torch.equal(out_streams["free"], out_streams["attributed"])
        logger.info(f"free vs attributed output streams: {'identical' if same else 'DIFFERENT'}")
    assert all(torch.isfinite(v).all() for v in out_streams.values()), "non-finite output streams"

    # ---- the attribution table (exclusive wall per module, per layer kind)
    if "attributed" in results:
        rows_t = timer.table()
        tot = {}
        for c, l, ex, _, _ in rows_t:
            tot[c] = tot.get(c, 0.0) + ex
        lines = ["| layer kind | module | exclusive s | inclusive s | calls | % of kind |", "|---|---|---|---|---|---|"]
        for c, l, ex, inc, n in sorted(rows_t, key=lambda r: (r[0], -r[2])):
            lines.append(f"| {c} | {l} | {ex:.3f} | {inc:.3f} | {n} | {100 * ex / max(tot[c], 1e-9):.1f} % |")
        logger.info(
            "attributed pass (sync at every module boundary; exclusive = time charged to the innermost open label):\n"
            + "\n".join(lines)
        )
        results["attribution"] = [
            {"kind": c, "label": l, "exclusive_s": ex, "inclusive_s": inc, "calls": n} for c, l, ex, inc, n in rows_t
        ]
    # ---- the projection: 1 M tokens in chunks of CHUNK on this block (prefix-independent part flat, the
    # prefix-dependent DSA stages (kv copy, scorer, top-k) linear in the prefix, measured at PREFIX)
    if "free" in results:
        free = results["free"]["wall_s"]
        dep = 0.0
        if "attribution" in results:
            dep = sum(
                r["exclusive_s"]
                for r in results["attribution"]
                if r["label"] in ("DSA_KV_COPY", "DSA_INDEX_SCORER", "DSA_INDEX_SELECT")
            )
        n_chunks = (1 << 20) // CHUNK
        indep = max(free - dep, 0.0)
        proj = sum(indep + dep * ((i + 1) * CHUNK / max(PREFIX, 1)) for i in range(n_chunks))
        logger.info(
            f"PROJECTION 1 M tokens in {n_chunks} chunks of {CHUNK} on {MESH} with {len(layers)} layers: {proj:.0f} s "
            f"(free chunk {free:.2f} s at prefix {PREFIX}, of which prefix-dependent {dep:.2f} s scaled linearly; "
            f"x 45/{len(layers)} layers ~ {proj * 45 / len(layers):.0f} s)"
        )
        results["projection_1m_s"] = proj
    _record(
        mesh=MESH,
        layers=layers,
        prefix=PREFIX,
        chunk=CHUNK,
        build_s=build_s,
        expert_dtype=os.environ.get("GLM53_MODEL_EXPERT_DTYPE", "bfp8"),
        env={k: v for k, v in os.environ.items() if k.startswith("GLM53_")},
        **{k: v for k, v in results.items()},
    )
