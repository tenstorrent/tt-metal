# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The full-indexer layers chained into the Stack's step (program.py Stack.set_indexer, PLAN.md section 53), on the mock
Galaxy: every program of a step is assembled and compiled (nothing runs), audited, and the L1 need, the number of
programs and of program-cache entries and the host-write registry are logged.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>
    HY4_MOCK=1 HY4_MOCK_LAYERS=78 pytest <this file> -k L78

Random weights and placeholder expert streams (no file is read); the layers' kinds follow the checkpoint's schedule
(full-indexer layers 0, 1, 5, 9, ...: projections/prepare.py is_full). Lines start with HY4_MF.
"""

import os

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.kv_store import device as kv_device
from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.model.tests.test_mock import galaxy, random_site
from models.experimental.hy4_preview.fused.moe import layout as ML
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.fused.projections.tests.harness import random_reference, tables
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL

DEPTHS = [6] + ([int(os.environ["HY4_MOCK_LAYERS"])] if os.environ.get("HY4_MOCK_LAYERS") else [])
FRAMES = int(os.environ.get("HY4_MF_FRAMES", "8"))  # frames of 8,192 positions a row holds
APPEND = os.environ.get("HY4_MF_APPEND", "nearest")


def stores(mesh_device, layers, full_layers, frames=FRAMES, users=8):
    """(latent store, key store, their geometries): uninitialised, slot-major, ``frames`` frames a slot."""
    lat = kv_frames.Kind(
        "latent", layers, kv_frames.LATENT.width, kv_frames.LATENT.shard_bytes, kv_frames.LATENT.row_stride
    )
    idx = kv_frames.Kind("index", full_layers, kv_frames.INDEX.width, kv_frames.INDEX.shard_bytes, 0)
    g_lat, g_idx = kv_frames.Geometry.contiguous(lat, frames), kv_frames.Geometry.contiguous(idx, frames)
    return kv_device.allocate(mesh_device, g_lat, users), kv_device.allocate(mesh_device, g_idx, users), g_lat, g_idx


def build_random_full(mesh_device, layers, small_streams=True, extra_sites=(), extra_gammas=(), **kw):
    """A Stack of ``layers`` layers with random parameters in the full-indexer form: the checkpoint's schedule of
    full layers, the append in every layer, the stores in the model's striping."""
    g = torch.Generator().manual_seed(1)
    full = [n for n in range(layers) if PP.is_full(n)]
    st = Stack(mesh_device, attn_kwargs=dict(append=APPEND), **kw)
    kv, keys, g_lat, _ = stores(mesh_device, layers, len(full))
    st.set_indexer(keys, len(full))
    logger.info(
        f"HY4_MF mock: the full set, the scan, the merge and the cut made: L1 tensors {st.l1_bytes()} B per core"
    )
    st.set_sites(
        [random_site(g) for _ in range(2 * layers)] + list(extra_sites),
        [torch.ones(A.HIDDEN) for _ in range(2 * layers)] + list(extra_gammas),
    )
    small = lambda: ML.to_device_stream(mesh_device, torch.zeros(64, 256))
    for n in range(layers):
        is_full = n in full
        ref, _, indexer = random_reference(n, is_full)
        lw = L.layer_weights(ref.weights(), None, indexer.weights() if is_full else None)
        sink = torch.randn(64, generator=g)
        dense = n == 0
        st.add_layer(
            n,
            lw,
            sink,
            2 * n,
            2 * n + 1,
            cache_layer=n,
            dense=dense,
            router_weight=None if dense else torch.randn(256, A.HIDDEN, generator=g) * 0.01,
            router_bias=None if dense else torch.zeros(256),
            streams=None if dense or not small_streams else (small(), small()),
            moe_layer=n,
        )
        if n < 6 or n % 12 == 5:
            logger.info(
                f"HY4_MF mock: layer {n} ({'full' if is_full else 'shared'}) added, L1 tensors {st.l1_bytes()} B per core"
            )
    cfg = SL.SparseMlaConfig(rows_per_slot=FRAMES * kv_frames.FRAME_ROWS, num_layers=layers)
    assert tuple(kv.shape) == g_lat.tensor_shape(cfg.users)
    st.set_cache(cfg, kv)
    _, _, cos, sin = tables()
    st.set_rope(cos, sin)
    st.set_positions([0, 1, 127, 128, 4095, 4096, 8191, 0xFFFFFFFF])
    st.set_contexts([1, 2, 128, 129, 4096, 4097, 8192, 0])
    st.set_streams(torch.randn(8, 4, A.HIDDEN, generator=g))
    return st


@galaxy
@pytest.mark.parametrize("split", ["tp32", "tp8g4"])
@pytest.mark.parametrize("layers", DEPTHS, ids=lambda n: f"L{n}")
def test_mock(mesh_device, layers, split):
    """The whole step compiles and fits, with either expert layout of the MoE layers (Stack(mlp_split=))."""
    if not is_mock():
        pytest.skip("the mock's compile and size check")
    st = build_random_full(mesh_device, layers, mlp_split=split)
    full = [s.key for s in st.layers if s.full]
    programs = st.step_programs(audit=True)
    names, entries = [], []
    for name, io, desc in programs:
        Stack.launch([(name, io, desc)])
        names.append(name)
        entries.append(mesh_device.num_program_cache_entries())
    kinds = {}
    for name in names:
        kinds[name.split("[")[0]] = kinds.get(name.split("[")[0], 0) + 1
    logger.info(
        f"HY4_MF mock {layers} layers, full-indexer layers {full}: {len(programs)} programs compiled (3 N + 3 F + 1 = "
        f"{3 * layers + 3 * len(full) + 1}): {kinds}; program-cache entries after the step {entries[-1]}; the first layers' order "
        f"{names[:14]}"
    )
    assert len(programs) == 3 * layers + 3 * len(full) + 1
    rows = []
    for n, state in enumerate(st.layers):
        rows += st.cb_audit(state, post=n > 0)
    multi = [r for r in rows if len(r[5]) > 1 or len(r[6]) > 1]
    logger.info(
        f"HY4_MF mock {layers} layers: cb_audit over root / head / other chip of P1, P2 and MLP of every layer (the appends' and "
        f"the key core's buffers in it; scan and merge: no buffer id twice on a core): {len(rows)} (core, buffer) rows, "
        f"{len(multi)} with two producers or consumers; feed_audit rows {sum(len(a[3]) for a in st.audits)} all fed"
    )
    assert not multi
    shared = next(s for s in reversed(st.layers) if not s.full)
    last_full = next(s for s in reversed(st.layers) if s.full)
    for label, state in (("a shared layer", shared), ("a full layer", last_full)):
        report = st.l1_report(state)
        core = report["core"]
        logger.info(
            f"HY4_MF mock {layers} layers L1 ({label}, {state.key}): persistent tensors {report['tensors']} of {core} B per core; "
            f"largest free block {report['largest_free']}"
        )
        for key, value in report.items():
            if isinstance(key, tuple):
                cb, worst, kernels, cbs = value
                logger.info(
                    f"HY4_MF mock {layers} layers L1 {label} {key[0]} chip {key[1]}: {kernels} kernels, {cbs} CBs; largest CB set "
                    f"{cb} B on core {worst}; with the tensors {cb + report['tensors']} of {core} B ({core - cb - report['tensors']} free)"
                )
                assert cb + report["tensors"] <= core, f"{split}: {key[0]} on chip {key[1]} does not fit a core's L1"
    # the registry and the restore: every L1 tensor the host wrote (the full set's, the scan's, the merge's in it)
    hw = st.hw
    hw.mark_per_step(st.a.rope)
    logger.info(f"HY4_MF mock {layers} layers registry: {hw.summary()}")
    from models.experimental.hy4_preview.fused.model import reserve as RS

    per_core = sorted(
        (
            (e.bytes() // max(len(RS.cores_of(e.tensor) or [1]), 1), len(RS.cores_of(e.tensor) or [1]), e.where)
            for e in hw.live().values()
        ),
        reverse=True,
    )
    logger.info(
        f"HY4_MF mock {layers} layers registry, every host-written L1 tensor as (B a core, cores, written at): {per_core}; their "
        f"sum {sum(v[0] for v in per_core)} B a core of the {st.l1_bytes()} allocated"
    )
    plan = hw.plan()
    logger.info(
        f"HY4_MF mock {layers} layers restore plan: zero {len(plan['zero'])} tensors or ranges ({plan['bytes']['zero']} B a chip), "
        f"store {len(plan['store'])} ({plan['bytes']['store']} B a chip), {plan['semaphores']} semaphores; the largest zeroed: "
        f"{sorted(((z[4], z[1]) for z in plan['zero']), reverse=True)[:5]}; stored: {sorted(((x[2], x[1]) for x in plan['store']), reverse=True)[:8]}"
    )
    restore = st.make_restore()
    logger.info(f"HY4_MF mock {layers} layers {restore.summary()}")
    restore.launch()
