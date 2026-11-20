# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fused/model/program.py on the mock Galaxy: every program of a step of one and of six layers is assembled and
compiled (nothing runs), audited, and its L1 need and the number of program-cache entries are logged.

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file>

Lines start with HY4_MK11.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.layer_attn import program as A
from models.experimental.hy4_preview.fused.model.program import Stack, is_mock
from models.experimental.hy4_preview.fused.moe import layout as ML
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections.tests.harness import random_reference, tables
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.fused.sparse_mla.tests import harness as hs
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
TRACE_REGION = 64 << 20
galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)


def random_site(g):
    return (torch.randn(8, 4 * A.HIDDEN, generator=g) * 0.02, torch.randn(8, generator=g), torch.tensor([0.01, 0.005]))


def build_random(
    mesh_device, layers, dense_first=True, small_streams=True, moe_layer0=0, extra_sites=(), extra_gammas=(), **kw
):
    """A Stack of ``layers`` layers with random parameters (layer 0 the dense FFN when dense_first). small_streams:
    placeholder expert streams (the mock only: a board run reads the picked experts, so it takes the prepared streams
    of checkpoint layer moe_layer0 + n). extra_sites / extra_gammas: further entries of the site table behind the
    layers' 2 x ``layers`` (the head's: ends_fused.head_site)."""
    g = torch.Generator().manual_seed(1)
    st = Stack(mesh_device, **kw)
    st.set_sites(
        [random_site(g) for _ in range(2 * layers)] + list(extra_sites),
        [torch.ones(A.HIDDEN) for _ in range(2 * layers)] + list(extra_gammas),
    )
    small = lambda: ML.to_device_stream(mesh_device, torch.zeros(64, 256))
    for n in range(layers):
        ref, _, _ = random_reference(n, False)
        lw = L.layer_weights(ref.weights(), None)
        sink = torch.randn(64, generator=g)
        dense = dense_first and n == 0
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
            moe_layer=moe_layer0 + n,
        )
        logger.info(f"HY4_MK11 mock: layer {n} added, L1 tensors {st.l1_bytes()} B per core")
    cfg = SL.SparseMlaConfig(rows_per_slot=64, num_layers=layers)
    latent, rope = hs.random_latents(cfg.users * layers, cfg.rows_per_slot, cfg, seed=3)
    packed = hs.pack_rows(mesh_device, latent[:, None], rope[:, None])
    kv = ttnn.to_memory_config(packed, hs.kv_memory_config(mesh_device, cfg))
    ttnn.deallocate(packed)
    sel = hs.make_selection(mesh_device, cfg, [hs.spread_ids(cfg, 8, seed=20 + u) for u in range(cfg.users)])
    st.set_cache(cfg, kv, sel)
    _, _, cos, sin = tables()
    st.set_rope(cos, sin)
    st.set_streams(torch.randn(8, 4, A.HIDDEN, generator=g))
    return st


@galaxy
@pytest.mark.parametrize("layers", [1, 2, 6], ids=["one", "two", "six"])
def test_mock(mesh_device, layers):
    if not is_mock():
        pytest.skip("the mock's compile and size check")
    st = build_random(mesh_device, layers, dense_first=layers > 1)
    programs = st.step_programs(audit=True)
    entries = []
    for name, io, desc in programs:
        Stack.launch([(name, io, desc)])
        entries.append((name, mesh_device.num_program_cache_entries()))
    logger.info(
        f"HY4_MK11 mock {layers} layer(s): {len(programs)} programs compiled; program-cache entries after each: {entries}"
    )
    rows = []
    for n, state in enumerate(st.layers):
        rows += st.cb_audit(state, post=n > 0)
    multi = [r for r in rows if len(r[5]) > 1 or len(r[6]) > 1]
    logger.info(
        f"HY4_MK11 mock {layers} layer(s): cb_audit over root / head / other chip of P1, P2 and MLP of every layer: "
        f"{len(rows)} (core, buffer) rows, {len(multi)} with two producers or consumers; feed_audit rows "
        f"{sum(len(a[3]) for a in st.audits)} all fed"
    )
    report = st.l1_report(st.layers[-1])
    core = report["core"]
    logger.info(
        f"HY4_MK11 mock {layers} layer(s) L1: persistent tensors {report['tensors']} of {core} B per core; largest free block {report['largest_free']}"
    )
    for key, value in report.items():
        if isinstance(key, tuple):
            cb, worst, kernels, cbs = value
            logger.info(
                f"HY4_MK11 mock {layers} layer(s) L1 {key[0]} chip {key[1]}: {kernels} kernels, {cbs} CBs; largest CB set "
                f"{cb} B on core {worst}; with the tensors {cb + report['tensors']} of {core} B ({core - cb - report['tensors']} free)"
            )
