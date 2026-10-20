# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced decode performance of one decoder layer on a (rows, cols) chip block, with Tracy signposts.

Same pattern as ``test_perf_layer.py`` (compile/warm-up, drain the device profiler, signpost, one
measured traced replay, signpost), built on the mesh: ``FunctionalDecoder.from_state_dict`` on the
``(rows, cols)`` mesh device, ``B`` user slots per row (``rows * B`` users), the decode streams and
positions row-sharded. The mesh has no prefill path, so a DSA layer's latent cache is seeded with
random rows (``GLM53_PERF_SEED_ONLY`` semantics of the single-chip test: the decode profile does not
depend on the cache content, and no SDPA prefill kernel is compiled, so the Tracy zone-hash collision
of ``compute_streaming.hpp`` never triggers). After the signposted window, ``GLM53_PERF_WALL_STEPS``
unsignposted replays give the steady-state wall time (skipped under the device profiler: un-drained replays
overflow the per-core marker buffers and crash the Tracy capture; the profiler-free run is the wall number),
then ONE eager decode step runs inside a
second signpost window (``PERF_EAGER`` .. ``PERF_EAGER_END``) with nested per-module signposts
(``ATTN`` / ``NORM`` / ``ROUTER`` / ``EXPERTS`` / ``SHARED`` / ``MOE`` and the three collectives); the
eager step issues the same ops in the same order as the captured graph, so
``tests/perf_mesh_report.py`` can label every op of the traced window by index.

Collect on the rows-4-5 (2,4) block (one layer x users-per-row point per run, device profiler only):

    export TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 MESH_DEVICE=P150x8 \
        TT_MESH_GRAPH_DESC_PATH=$W/tt_metal/fabric/mesh_graph_descriptors/p150_x8_mesh_graph_descriptor.textproto
    GLM53_PERF_LAYER=4 GLM53_PERF_B=4 GLM53_PERF_DECODE_STEPS=1 python -m tracy -r -p -v -m pytest \
        models/demos/glm53_flash/tests/test_perf_mesh.py -s --timeout 3000
    python models/demos/glm53_flash/tests/perf_mesh_report.py <ops_perf_results_*.csv> --out <dir> --rows 2 --cols 4

The ops CSV carries every chip (``DEVICE ID`` = local chip id, row-major mesh order on this block);
the report script splits it per device and runs ``tt-perf-report --start-signpost PERF_DECODE
--end-signpost PERF_DECODE_END`` on each.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests import test_perf_layer as _single  # module import: pytest must not collect its test
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder, host_streams_to_device
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

signpost = _single.signpost
_read_profiler = _single._read_profiler
_log_dram = _single._log_dram

ROWS, COLS = (int(v) for v in os.environ.get("GLM53_PERF_MESH_SHAPE", "2,4").split(","))
LAYER = int(os.environ.get("GLM53_PERF_LAYER", "4"))
B = int(os.environ.get("GLM53_PERF_B", "4"))  # users per row (design points: 4 and 32)
POS = int(os.environ.get("GLM53_PERF_POS", "1024"))  # decode position (the single-chip table: after a 1024 prefill)
DECODE_STEPS = int(os.environ.get("GLM53_PERF_DECODE_STEPS", "1"))  # signposted replays (1 => rows == ops)
WALL_STEPS = int(os.environ.get("GLM53_PERF_WALL_STEPS", "10"))  # unsignposted replays for the wall time
# under the device profiler the un-drained wall replays overflow the per-core marker buffers (12000 entries;
# a layer-3 B=32 step leaves ~5k markers on some cores: 3 replays already dropped markers and 10 crashed the
# Tracy capture, no CSV), so the wall loop is skipped there; the profiler-free run is the wall number anyway
if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
    WALL_STEPS = 0
EAGER_PASS = os.environ.get("GLM53_PERF_EAGER", "1") == "1"  # the labelled eager step after the window
TRACE_MB = int(os.environ.get("GLM53_PERF_TRACE_MB", "60"))
_DTYPES = _single._DTYPES
EXPERT_DTYPE, EXPERT_DOWN_DTYPE = _single.EXPERT_DTYPE, _single.EXPERT_DOWN_DTYPE
DUMP = Path("bringup/references/glm53_flash/T4097")


class _Labels:
    """Wrap bound methods / class methods with BEGIN/END signposts for the eager attribution pass."""

    def __init__(self):
        self._undo = []

    def wrap(self, obj, name: str, label: str):
        orig = getattr(obj, name)

        def wrapped(*args, **kwargs):
            signpost(f"{label}_BEGIN")
            try:
                return orig(*args, **kwargs)
            finally:
                signpost(f"{label}_END")

        setattr(obj, name, wrapped)
        self._undo.append((obj, name, orig))

    def restore(self):
        for obj, name, orig in reversed(self._undo):
            setattr(obj, name, orig)
        self._undo.clear()


def _label_modules(dec: FunctionalDecoder) -> _Labels:
    lab = _Labels()
    layer = dec.layer
    lab.wrap(layer.attn, "decode_forward", "ATTN")
    lab.wrap(layer, "_norm", "NORM")
    mlp = layer.mlp
    if hasattr(mlp, "router"):  # MoE layer
        lab.wrap(mlp.router, "forward", "ROUTER")
        lab.wrap(mlp.experts, "forward", "EXPERTS")
        lab.wrap(mlp.shared, "forward", "SHARED")
        lab.wrap(mlp, "forward", "MOE")
    else:
        lab.wrap(mlp, "forward", "DENSE_MLP")
    # the collectives are class-level (frozen dataclass instances)
    for name, label in (
        ("all_reduce_cols", "CCL_ALL_REDUCE"),
        ("all_gather_rows", "CCL_ALL_GATHER"),
        ("reduce_scatter_rows", "CCL_REDUCE_SCATTER"),
        ("all_gather_cols", "CCL_ALL_GATHER_COLS"),
    ):
        lab.wrap(GLMMesh, name, label)
    return lab


@pytest.mark.timeout(3000)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(ROWS, COLS, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_mesh_layer_perf(mesh_device):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    pre = torch.load(DUMP / f"layer{LAYER}_prefill.pt")
    U = ROWS * B
    try:
        ids = list(mesh_device.get_device_ids())
    except Exception:  # pragma: no cover
        ids = []
    logger.info(f"mesh {ROWS}x{COLS} device ids (row-major): {ids}")
    dec = FunctionalDecoder.from_state_dict(
        raw,
        hf_config=cfg,
        layer_idx=LAYER,
        mesh_device=mesh_device,
        max_batch=B,
        max_seq_len=POS + 64,
        expert_dtype=_DTYPES[EXPERT_DTYPE],
        expert_down_dtype=_DTYPES[EXPERT_DOWN_DTYPE],
    )
    mesh = dec.mesh
    assert (mesh.rows, mesh.cols) == (ROWS, COLS) and dec.total_users == U
    kind = f"{'KDA' if dec.layer.is_kda else 'DSA'}+{'MoE' if dec.layer.is_moe else 'dense'}"
    if dec.layer.is_moe:
        kind += f" experts gate/up {EXPERT_DTYPE} down {EXPERT_DOWN_DTYPE}"
    _log_dram(mesh_device, f"layer {LAYER} ({kind}) built on {ROWS}x{COLS}, B={B}/row, max_seq_len={POS + 64}")
    if dec.cache is not None:  # DSA: seed every slot's latent pages with random rows (replicated on every chip)
        g = torch.Generator().manual_seed(0)
        rows = -(-POS // 32) * 32
        latent = mesh.upload(torch.randn(1, 1, rows, cfg.kv_lora_rank, generator=g) * 0.5, ttnn.bfloat16)
        for u in range(B):
            dec.cache.fill(latent, user=u)
        ttnn.deallocate(latent)
        logger.info(f"layer {LAYER} ({kind}) seed-only: {B} slots x {rows} random latent rows per chip")

    # ---- decode inputs: U distinct real token rows of the dump, row-sharded over the mesh rows
    x_in = pre["x_in"][:, POS : POS + U]  # [1, U, 4, 4096]
    x_dev = host_streams_to_device(x_in, mesh_device, pad_to=U, mesh=mesh)
    dp = dec.make_decode_positions([POS] * U)

    # ---- capture trace (warm-up inside), one warm replay, drain, then the signposted window
    trace_id, out_dev, warm = dec.capture_decode_trace(x_dev, dp)
    ttnn.deallocate(warm)
    dec.replay_decode(trace_id)
    ttnn.synchronize_device(mesh_device)
    _read_profiler(mesh_device)
    signpost("PERF_DECODE")
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        dec.replay_decode(trace_id, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t_decode = (time.perf_counter() - t0) / DECODE_STEPS
    signpost("PERF_DECODE_END")
    _read_profiler(mesh_device)

    # ---- steady-state wall time (unsignposted)
    timings = [("signposted", DECODE_STEPS, t_decode)]
    if WALL_STEPS > 0:
        t0 = time.perf_counter()
        for _ in range(WALL_STEPS):
            dec.replay_decode(trace_id, blocking=False)
        ttnn.synchronize_device(mesh_device)
        timings.append(("steady-state", WALL_STEPS, (time.perf_counter() - t0) / WALL_STEPS))
    for label, n, t in timings:
        logger.info(
            f"layer {LAYER} ({kind}) mesh {ROWS}x{COLS} traced decode B={B}/row ({U} users, position {POS}), "
            f"{label} {n} steps: {t * 1e3:.2f} ms/step wall; single-layer rate {1 / t:.0f} tok/s/user, "
            f"{U / t:.0f} tok/s aggregate; x45 layers -> {1 / (45 * t):.1f} tok/s/user, {U / (45 * t):.0f} tok/s aggregate"
        )

    # ---- one eager step with per-module signposts (op order == the captured graph's)
    if EAGER_PASS:
        _read_profiler(mesh_device)
        labels = _label_modules(dec)
        try:
            signpost("PERF_EAGER")
            y = dec.decode_forward(x_dev, positions=dp)
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_EAGER_END")
        finally:
            labels.restore()
        _read_profiler(mesh_device)
        ttnn.deallocate(y)
    ttnn.release_trace(mesh_device, trace_id)
