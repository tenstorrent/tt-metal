# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Long-context indexer baseline (bead F10, phase 1): measured time of ``TtV41Indexer`` at chunk start P.

One 5120-token chunk at start P on LoudBox 2x4 with synthetic weights and index keys, for the three indexer roles:
layer 2 (ratio-2 index source), 20 (ratio-1 candidate source) and 24 (ratio-1 candidate index source, masked by
layer 20's published candidates). Warm (program cache hot) measurements, host wall time with device synchronizes:

* ``stage_ms``: the module's stages as the attention calls them (``scores``, ``candidates`` / candidate mask, top-k),
  synchronized only at stage boundaries (dispatch pipelined inside a stage);
* ``ops``: every ttnn call synchronized individually (serialized; sums exceed the stage times), with the bytes each
  output occupies per chip; outputs at least one score row wide (``[S/(sp*tp), >= T]``) are the [S, T] DRAM passes;
* ``score_configs_ms``: ``indexer_score_dsa`` alone on the module's own inputs under alternative program configs;
* ``model``: ``utils/v41_perf_model.py`` indexer ops (B9-B12) for the same workload.

Logged as ``V41_INDEXER_PERF`` JSON lines, also appended to ``$V41_INDEXER_PERF_OUT`` (default
``generated/v41_indexer_perf.log``) so results survive a shared runner log being overwritten; no pass/fail bar
(the bead sets its targets from these numbers).
"""

import json
import os
import time
import traceback
from collections import defaultdict

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41ChunkTables
from models.demos.deepseek_v3_d_p.tt.v41.indexer import TtV41Indexer
from models.demos.deepseek_v3_d_p.utils import v41_perf_model as M

CHUNK = 5120
LAYERS = (2, 20, 24)
TIMED_ITERS = 2
TTNN_OPS = (
    "from_torch",
    "to_layout",
    "slice",
    "pad",
    "add",
    "multiply",
    "subtract",
    "maximum",
    "minimum",
    "max",
    "where",
    "eq",
    "gt",
    "gtz",
    "typecast",
    "bitcast",
    "bitwise_and",
    "abs",
    "sign",
    "floor",
    "rsub",
    "reshape",
    "full",
    "scatter",
    "repeat_interleave",
    "linear",
    "concat",
    "mesh_partition",
)
EXPERIMENTAL_OPS = (
    "indexer_score_dsa",
    "topk_large_indices",
    "rotary_embedding_llama",
    "nlp_create_qkv_heads",
    "all_gather_async",
    "reduce_scatter_minimal_async",
)
ELEMENT_BYTES = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.uint32: 4,
    ttnn.int32: 4,
    ttnn.uint16: 2,
    ttnn.bfloat8_b: 1088 / 1024,
}
SCORE_CONFIGS = {
    "default_q32_k32_h1": dict(q_chunk_size=32, k_chunk_size=32, head_group_size=1),
    "q32_k256_h0": dict(q_chunk_size=32, k_chunk_size=256, head_group_size=0),
    "q64_k320_h0": dict(q_chunk_size=64, k_chunk_size=320, head_group_size=0),
}


OUT = os.environ.get("V41_INDEXER_PERF_OUT", "generated/v41_indexer_perf.log")


def _log(msg: str) -> None:
    logger.info(msg)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def _per_chip_shape(t) -> list[int]:
    return list(ttnn.get_device_tensors(t)[0].shape)


def _bytes(t) -> float:
    n = 1
    for d in _per_chip_shape(t):
        n *= d
    return n * ELEMENT_BYTES.get(t.dtype, 2)


class _OpProfiler:
    """Replaces ttnn callables with device-synchronized timers for the duration of a ``with`` block; nested ttnn
    calls inside a timed call are not timed separately. Records (stage, op, ms, output per-chip shape, bytes)."""

    def __init__(self, mesh_device):
        self.mesh, self.stage, self.calls, self._depth, self._saved = mesh_device, "", [], 0, []
        self.captured = {}

    def _wrap(self, fn, name):
        def timed(*args, **kwargs):
            if self._depth:
                return fn(*args, **kwargs)
            self._depth += 1
            try:
                ttnn.synchronize_device(self.mesh)
                start = time.perf_counter()
                out = fn(*args, **kwargs)
                ttnn.synchronize_device(self.mesh)
                ms = (time.perf_counter() - start) * 1e3
            finally:
                self._depth -= 1
            if name == "indexer_score_dsa":
                self.captured[name] = (args, kwargs)
            first = out[0] if isinstance(out, (tuple, list)) else out
            shape = _per_chip_shape(first) if isinstance(first, ttnn.Tensor) else []
            nbytes = _bytes(first) if isinstance(first, ttnn.Tensor) else 0.0
            self.calls.append((self.stage, name, ms, shape, nbytes))
            return out

        return timed

    def __enter__(self):
        for owner, names in ((ttnn, TTNN_OPS), (ttnn.experimental, EXPERIMENTAL_OPS)):
            for name in names:
                fn = getattr(owner, name)
                self._saved.append((owner, name, fn))
                setattr(owner, name, self._wrap(fn, name))
        return self

    def __exit__(self, *exc):
        for owner, name, fn in reversed(self._saved):
            setattr(owner, name, fn)
        self._saved.clear()


def _model_ms(layer: int, start: int) -> dict:
    ops = M.block_ops(layer, M.Workload(chunk=CHUNK, start=start), M.LOUDBOX_2X4)
    ops = [o for o in ops if o.node in ("B9", "B10", "B11", "B12")]
    e = M.compose(ops)
    return {
        "ops": {o.name: {"compute_ms": o.compute_ns / 1e6, "dram_ms": o.dram_ns / 1e6} for o in ops},
        "optimistic_ms": e.optimistic_ns / 1e6,
        "conservative_ms": e.conservative_ns / 1e6,
    }


def _run(indexer, x, qr, index_k, tables, start, candidate_mask, timer=None):
    """Indexer forward split into its stages (same calls as ``TtV41Indexer.forward``) -> (idx, published, ms)."""
    ms = {}

    def stage(name, fn, *args):
        if timer is not None:
            timer.stage = name
        ttnn.synchronize_device(indexer.mesh_device)
        t0 = time.perf_counter()
        out = fn(*args)
        ttnn.synchronize_device(indexer.mesh_device)
        ms[name] = (time.perf_counter() - t0) * 1e3
        return out

    score, visible = stage("scores", indexer.scores, x, qr, index_k, tables, start, CHUNK)
    q_rows = score.shape[2]
    published = None
    if indexer.is_candidate_source:
        published = stage("candidates", indexer.candidates, score, tables, start, q_rows, visible)
    elif indexer.uses_candidates:
        score = stage(
            "candidate_mask",
            lambda s: ttnn.to_layout(
                ttnn.add(ttnn.to_layout(s, ttnn.TILE_LAYOUT), candidate_mask), ttnn.ROW_MAJOR_LAYOUT
            ),
            score,
        )
    k = min(C.INDEX_TOPK, visible)
    idx = stage("topk", lambda s: ttnn.experimental.topk_large_indices(s, k=max(16, -(-k // 16) * 16)), score)
    ms["total"] = sum(ms.values())
    return idx, published, ms, score, visible


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("start", [16384, 131072, 262144, 1048576], ids=lambda p: f"P{p}")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_indexer_perf(mesh_device, device_params, start):
    shape, (sp, tp) = tuple(mesh_device.shape), tuple(mesh_device.shape)
    g = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()

    def up(t, dims, layout=ttnn.TILE_LAYOUT):
        mapper = (
            ttnn.ReplicateTensorToMesh(mesh_device)
            if dims is None
            else ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims)
        )
        return ttnn.from_torch(t, device=mesh_device, dtype=ttnn.bfloat16, layout=layout, mesh_mapper=mapper)

    x = up(torch.randn(1, 1, CHUNK, C.EMB_SIZE, generator=g).to(torch.bfloat16), (2, 3))
    qr = up(torch.randn(1, 1, CHUNK, C.Q_LORA_RANK, generator=g).to(torch.bfloat16), (2, None))
    weights = {
        "wq_b": torch.randn(C.INDEX_N_HEADS * C.INDEX_HEAD_DIM, C.Q_LORA_RANK, generator=g) * C.Q_LORA_RANK**-0.5,
        "weights_proj": torch.randn(C.INDEX_N_HEADS, C.EMB_SIZE, generator=g) * C.EMB_SIZE**-0.5,
    }
    index_k = {}
    for ratio in sorted({C.compress_ratio(l) for l in LAYERS}):
        rows = -(-(start + CHUNK) // ratio // 32) * 32
        index_k[ratio] = up(
            torch.randn(1, 1, rows, C.INDEX_HEAD_DIM, generator=g).to(torch.bfloat16), None, ttnn.ROW_MAJOR_LAYOUT
        )
    _log(f"P={start}: inputs uploaded {time.perf_counter() - t0:.1f}s")

    try:
        _measure(mesh_device, sp, tp, x, qr, weights, index_k, start, t0)
    except Exception:
        _log(f"P={start}: FAILED\n{traceback.format_exc()}")
        raise


def _measure(mesh_device, sp, tp, x, qr, weights, index_k, start, t0):
    candidate_mask = None
    tables = V41ChunkTables(mesh_device, C, start + CHUNK, CHUNK, list(LAYERS))  # position tables up to this chunk
    for layer in LAYERS:
        _log(f"P={start} L{layer}: start")
        ratio = C.compress_ratio(layer)
        indexer = TtV41Indexer(mesh_device, C, layer, weights)
        mask = candidate_mask if indexer.uses_candidates else None
        t1 = time.perf_counter()
        _, published, warm_ms, _, _ = _run(indexer, x, qr, index_k[ratio], tables, start, mask)
        _log(f"P={start} L{layer}: warm-up {time.perf_counter() - t1:.1f}s {warm_ms}")
        stage_ms = defaultdict(list)
        for _ in range(TIMED_ITERS):
            _, _, ms, _, _ = _run(indexer, x, qr, index_k[ratio], tables, start, mask)
            for key, value in ms.items():
                stage_ms[key].append(value)
        _log(f"P={start} L{layer}: stages {dict(stage_ms)}")

        with _OpProfiler(mesh_device) as prof:
            _, _, _, score, visible = _run(indexer, x, qr, index_k[ratio], tables, start, mask, timer=prof)
        q_rows = CHUNK // (sp * tp)
        width = score.shape[-1]
        per_op = defaultdict(lambda: {"calls": 0, "ms": 0.0, "bytes": 0.0})
        st_passes, st_bytes = 0, 0.0
        for stage, name, ms, out_shape, nbytes in prof.calls:
            key = f"{stage}.{name}"
            per_op[key]["calls"] += 1
            per_op[key]["ms"] += ms
            per_op[key]["bytes"] += nbytes
            if len(out_shape) >= 2 and out_shape[-2] == q_rows and out_shape[-1] >= width - 64:
                st_passes += 1
                st_bytes += nbytes
        _log(f"P={start} L{layer}: op profile {len(prof.calls)} calls, [S,T] outputs {st_passes}")

        score_cfg_ms = {}
        args, kwargs = prof.captured["indexer_score_dsa"]
        for cfg_name, cfg in SCORE_CONFIGS.items():
            call = dict(kwargs, program_config=ttnn.IndexerScoreProgramConfig(**cfg))
            try:
                ttnn.experimental.indexer_score_dsa(*args, **call)  # compile
                ttnn.synchronize_device(mesh_device)
                t2 = time.perf_counter()
                ttnn.experimental.indexer_score_dsa(*args, **call)
                ttnn.synchronize_device(mesh_device)
                score_cfg_ms[cfg_name] = (time.perf_counter() - t2) * 1e3
            except RuntimeError as err:  # a config the op rejects (e.g. L1) is recorded, not fatal
                score_cfg_ms[cfg_name] = f"rejected: {str(err).splitlines()[0][:160]}"
        _log(f"P={start} L{layer}: score configs {score_cfg_ms}")

        record = {
            "start": start,
            "layer": layer,
            "ratio": ratio,
            "visible": visible,
            "score_width": width,
            "q_rows_per_chip": q_rows,
            "stage_ms": {k: min(v) for k, v in stage_ms.items()},
            "stage_ms_all": dict(stage_ms),
            "ops": dict(sorted(per_op.items(), key=lambda kv: -kv[1]["ms"])),
            "st_output_passes": st_passes,
            "st_output_bytes_per_chip": st_bytes,
            "score_unit_bytes_per_chip": q_rows * width * 2,
            "score_configs_ms": score_cfg_ms,
            "model": _model_ms(layer, start),
        }
        _log("V41_INDEXER_PERF " + json.dumps(record))
        if indexer.is_candidate_source:
            candidate_mask = published
        del score
    _log(f"P={start}: done {time.perf_counter() - t0:.1f}s")
