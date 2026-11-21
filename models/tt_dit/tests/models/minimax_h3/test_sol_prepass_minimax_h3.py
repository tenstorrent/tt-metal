# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sol-Attn routing pre-pass on the 4x8 BH mesh: the ttnn-op pre-pass (sol_prepass.SolPrepass) and the fused one
(sol_prepass_fused.SolPrepassFused) against their torch mirror (sol_prepass.torch_reference): pooled Q/K/V, block
mask and column bias.

    pytest models/tt_dit/tests/models/minimax_h3/test_sol_prepass_minimax_h3.py -k 4x8
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import (
    NEG,
    SolGeometry,
    SolPrepass,
    build_constants,
    sink_block_mask,
    torch_reference,
)
from ....models.transformers.minimax_h3.sol_prepass_fused import SolPrepassFused, torch_reference_chunk
from ....parallel.manager import CCLManager
from ....utils.tensor import from_torch
from .common import GALAXY_RING, _ring_8k

H = 14
D = 128
SLOW = pytest.mark.skipif(os.environ.get("CI") == "true", reason="production-size shapes - skip on CI")


def _structured_qkv(S: int, rows: int, seed: int = 1):
    """Q/K that share a few 'topics' so the routing is non-trivial; bf16-representable fp32."""
    torch.manual_seed(seed)
    topics = torch.randn(8, D)
    tid = torch.randint(0, 8, (S * rows,))
    q = (topics[tid] + 0.7 * torch.randn(S * rows, D)).view(S, 1, rows, D).expand(S, H, rows, D).clone()
    q += 0.3 * torch.randn(S, H, rows, D)
    k = (topics[tid] + 0.7 * torch.randn(S * rows, D)).view(S, 1, rows, D).expand(S, H, rows, D).clone()
    k += 0.3 * torch.randn(S, H, rows, D)
    v = torch.randn(S, H, rows, D)
    return tuple(x.bfloat16().float() for x in (q, k, v))


def _upload(mesh_device, sp_axis, x):
    return from_torch(
        x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None], layout=ttnn.TILE_LAYOUT
    )


def _to_host(mesh_device, sp_axis, tp_axis, t):
    dims = [None, None]
    dims[sp_axis] = 0
    dims[tp_axis] = 1
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )


def _check_bias(dev_bias, ref_bias, geo, *, marker_min, live_atol=None):
    """The column bias: tile rows identical, the kept-block marker agreeing with the mirror; returns the agreement."""
    S = geo.shards
    real = dev_bias[:, :H].float().view(S, H, geo.nqc_pad, 32, geo.n_blocks_pad)[:, :, : geo.nqc]
    assert torch.equal(real[:, :, :, 0], real[:, :, :, 17]), "bias tile rows must be identical"
    b0 = real[:, :, :, 0]
    ref = ref_bias[:, :, : geo.nqc]
    kept_dev, kept_ref = b0 < NEG / 2, ref < NEG / 2
    marker_agree = (kept_dev == kept_ref).float().mean().item()
    logger.info(f"bias kept-marker agreement {marker_agree:.5f}")
    assert marker_agree >= marker_min, marker_agree
    if live_atol is not None:
        live = ~kept_ref & ~kept_dev
        torch.testing.assert_close(b0[live], ref[live].bfloat16().float(), atol=live_atol, rtol=1e-2)
    return marker_agree


def _check_fused(
    mesh_device,
    sp_axis,
    tp_axis,
    ccl_manager,
    *,
    geo,
    tau,
    mode,
    sink_tokens,
    q,
    k,
    v,
    compare_ttnn=True,
    partial_prefix=False,
):
    S = geo.shards
    scale = D**-0.5
    consts = build_constants(
        geo, band=1, sink_tokens=sink_tokens, force_first_chunk=True, scale=scale, partial_prefix=partial_prefix
    )
    ref = torch_reference(q, k, v, geo, consts, tau=tau, scale=scale)
    compare_ttnn = compare_ttnn and mode != "compact"
    if mode == "chunk":
        refc = torch_reference_chunk(ref, consts)
        ref_mask, ref_bias = refc["block_mask"], refc["bias"]
        cols = torch.cat([torch.arange(s * geo.nkc, s * geo.nkc + geo.nkc) for s in range(S)])
    else:
        ref_mask, ref_bias = ref["block_mask"], ref["bias"]
        cols = torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(S)])

    tq, tk, tv = (_upload(mesh_device, sp_axis, x) for x in (q, k, v))
    pre = SolPrepassFused(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
        tau=tau,
        mode=mode,
        partial_prefix=partial_prefix,
    )
    pre.prepare(geo, sink_tokens, scale)

    pooled = pre.pool(tq, tk, tv)
    ttnn.synchronize_device(mesh_device)
    qb = _to_host(mesh_device, sp_axis, tp_axis, pooled["q_bar"])[:, :H].float()
    torch.testing.assert_close(qb, ref["q_bar"].bfloat16().float(), atol=3e-2, rtol=3e-2)
    kcl = _to_host(mesh_device, sp_axis, tp_axis, pooled["kc_local"])[:, :H].float()
    kc_ref = ref["pooled_k"].view(H, S, geo.nb_pad, D).permute(1, 0, 2, 3)
    torch.testing.assert_close(kcl, kc_ref.bfloat16().float(), atol=3e-2, rtol=3e-2)
    for t in pooled.values():
        ttnn.deallocate(t)

    out = pre(tq, tk, tv, scale=scale, report_density=True)
    ttnn.synchronize_device(mesh_device)
    pk = _to_host(mesh_device, sp_axis, tp_axis, out["pooled_k"])[0, :H].float()
    torch.testing.assert_close(pk, ref["pooled_k"].bfloat16().float(), atol=3e-2, rtol=3e-2)
    pv = _to_host(mesh_device, sp_axis, tp_axis, out["pooled_v"])[0, :H].float()
    torch.testing.assert_close(pv, ref["pooled_v"].bfloat16().float(), atol=3e-2, rtol=3e-2)

    dev_mask = _to_host(mesh_device, sp_axis, tp_axis, out["block_mask"])[:, :H].float()
    assert dev_mask.shape == ref_mask.shape, (dev_mask.shape, ref_mask.shape)
    agree_all = (dev_mask == ref_mask).float().mean().item()
    agree_real = (dev_mask[:, :, : geo.nqc][..., cols] == ref_mask[:, :, : geo.nqc][..., cols]).float().mean().item()
    logger.info(f"mask vs mirror: agreement all {agree_all:.5f} real {agree_real:.5f}")
    assert set(dev_mask.unique().tolist()) <= {0.0, 1.0}
    if mode == "chunk":
        assert torch.all(dev_mask[..., S * geo.nkc :] == 0), "mask padding words must be zero"

    marker_agree = 1.0
    if out.get("bias") is not None:
        marker_agree = _check_bias(
            _to_host(mesh_device, sp_axis, tp_axis, out["bias"]), ref_bias, geo, marker_min=0.985
        )

    result = {"agree_all": agree_all, "agree_real": agree_real, "marker_agree": marker_agree}
    if compare_ttnn:
        pre_ttnn = SolPrepass(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
            tau=tau,
            mode=mode,
            partial_prefix=partial_prefix,
        )
        pre_ttnn.prepare(geo, sink_tokens, scale)
        out2 = pre_ttnn(tq, tk, tv, scale=scale, report_density=True)
        ttnn.synchronize_device(mesh_device)
        m2 = _to_host(mesh_device, sp_axis, tp_axis, out2["block_mask"])[:, :H].float()
        result["agree_ttnn_ref"] = (
            (m2[:, :, : geo.nqc][..., cols] == ref_mask[:, :, : geo.nqc][..., cols]).float().mean().item()
        )
    for t in out.values():
        if t is not None:
            ttnn.deallocate(t)
    return result


@GALAXY_RING
@pytest.mark.parametrize(
    ("rows", "logical_n", "q_chunk", "k_chunk"),
    [(1184, 9000, 256, 512), (1184, 9472, 256, 256), (1184, 7000, 256, 512)],
    ids=["mid_shard_cut", "k256", "two_padding_shards"],
)
@pytest.mark.parametrize("mode", ["chunk", "gather", "compact"])
def test_sol_prepass_fused(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, q_chunk, k_chunk, mode, reset_seeds
):
    """Fused pre-pass vs the torch mirror, and at least as close to it as the ttnn-op pre-pass."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    q, k, v = _structured_qkv(S, rows)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    r = _check_fused(
        mesh_device, sp_axis, tp_axis, ccl_manager, geo=geo, tau=1.0, mode=mode, sink_tokens=300, q=q, k=k, v=v
    )
    bar = max(0.985, r["agree_ttnn_ref"] - 0.002) if "agree_ttnn_ref" in r else 0.985
    assert r["agree_real"] >= bar, (r, bar)


def _partial_blocks(geo):
    """[nbg] bool: valid blocks shorter than a block (a shard's short last block, the logical_n tail)."""
    lens = geo.block_lengths().reshape(-1)
    return (lens > 0) & (lens < geo.block)


@GALAXY_RING
@pytest.mark.parametrize(
    ("rows", "logical_n", "sink_tokens"),
    [(1184, 9000, 300), (1184, 7000, 1400)],
    ids=["mid_shard_cut", "sink_two_shards"],
)
@pytest.mark.parametrize("mode", ["gather", "compact"])
def test_sol_prepass_fused_partial_prefix(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, sink_tokens, mode, reset_seeds
):
    """partial_prefix (MINIMAX_H3_SOL_PREFIX_PARTIAL): the sink Q blocks keep every valid block, partial ones
    included; the other Q blocks keep full blocks only. Fused and ttnn-op pre-pass vs the torch mirror."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=64, k_chunk=512)
    q, k, v = _structured_qkv(S, rows)
    scale = D**-0.5
    consts = build_constants(
        geo, band=1, sink_tokens=sink_tokens, force_first_chunk=True, scale=scale, partial_prefix=True
    )
    ref = torch_reference(q, k, v, geo, consts, tau=1.0, scale=scale)["block_mask"]
    part = _partial_blocks(geo)
    sink_q = torch.zeros(S, geo.nqc_pad, dtype=torch.bool)
    sink_q[:, : geo.nb] = sink_block_mask(geo, sink_tokens)[:, : geo.nb]
    assert bool(part.any()) and bool(ref[..., part][sink_q[:, None].expand(-1, H, -1)].all())
    assert not bool(ref[..., part][~sink_q[:, None].expand(-1, H, -1)].any())
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    r = _check_fused(
        mesh_device,
        sp_axis,
        tp_axis,
        ccl_manager,
        geo=geo,
        tau=1.0,
        mode=mode,
        sink_tokens=sink_tokens,
        q=q,
        k=k,
        v=v,
        partial_prefix=True,
    )
    bar = max(0.985, r["agree_ttnn_ref"] - 0.002) if "agree_ttnn_ref" in r else 0.985
    assert r["agree_real"] >= bar, (r, bar)


@SLOW
@GALAXY_RING
@pytest.mark.parametrize(
    ("rows", "logical_n", "q_chunk", "k_chunk"),
    [
        (4736, 37800, 320, 384),
        (9184, 73425, 256, 512),
        (13664, 109101, 256, 512),
    ],
    ids=["5s", "10s", "15s"],
)
def test_sol_prepass_fused_production_geometry(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, q_chunk, k_chunk, reset_seeds
):
    """Production geometries, random data: the fused pre-pass vs the torch mirror and the ttnn-op pre-pass."""
    S = tuple(mesh_device.shape)[sp_axis]
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    sink_tokens = {13664: 1245, 9184: 849, 4736: 453}[rows]
    torch.manual_seed(0)
    q, k, v = (torch.randn(S, H, rows, D).bfloat16().float() for _ in range(3))
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    r = _check_fused(
        mesh_device,
        sp_axis,
        tp_axis,
        ccl_manager,
        geo=geo,
        tau=1.0,
        mode="chunk",
        sink_tokens=sink_tokens,
        q=q,
        k=k,
        v=v,
    )
    assert r["agree_real"] >= max(0.985, r.get("agree_ttnn_ref", 1.0) - 0.002), r


# Runtime prompt length: one padded geometry (bucket), three logical lengths with their own sink and prefix-spread block
# map, the production pre-pass configuration (resident: the compact gather-form mask, no bias, head weights).
_RTLEN_ROWS, _RTLEN_K_CHUNK = 1184, 512
_RTLEN_SINKS = (300, 453, 200)


def _rtlen_cases(S: int):
    from ....models.transformers.minimax_h3.prefix_spread import prefix_spread_block_map

    full = S * _RTLEN_ROWS
    out = []
    for n, sink in zip((full, full - 472, (S - 1) * _RTLEN_ROWS - 40), _RTLEN_SINKS):
        geo = SolGeometry(shards=S, rows=_RTLEN_ROWS, logical_n=n, q_chunk=64, k_chunk=_RTLEN_K_CHUNK)
        bmap = prefix_spread_block_map(rows=_RTLEN_ROWS, shards=S, logical_n=n, prefix_rows=sink)
        out.append((geo, sink, bmap))
    return out


def _rtlen_prepass(mesh_device, sp_axis, ccl_manager):
    return SolPrepassFused(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sp_axis=sp_axis,
        tau=1.2,
        mode="compact",
        emit_bias=False,
        head_weights=True,
    )


def _rtlen_outputs(mesh_device, sp_axis, tp_axis, out) -> dict:
    names = ("block_mask", "pooled_k", "pooled_v", "head_weights")
    return {name: _to_host(mesh_device, sp_axis, tp_axis, out[name]).float() for name in names if name in out}


_RING_8K_TRACE_SMALL = {**_ring_8k, "trace_region_size": 64 * 1024 * 1024}


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [pytest.param((4, 8), 1, 0, 2, _RING_8K_TRACE_SMALL, ttnn.Topology.Ring, False, id="4x8_trace")],
    indirect=["mesh_device", "device_params"],
)
def test_sol_prepass_fused_runtime_len(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds, monkeypatch
):
    """Three logical lengths of one padded geometry, each with its own sink and block map: the persistent pre-pass
    (one state, constants rewritten in place) builds its programs once and gives the same bits as a fresh pre-pass per
    length; a trace captured at one length replays at the others after only `prepare()`."""
    S = tuple(mesh_device.shape)[sp_axis]
    scale = D**-0.5
    cases = _rtlen_cases(S)
    q, k, v = _structured_qkv(S, _RTLEN_ROWS, seed=5)
    tq, tk, tv = (_upload(mesh_device, sp_axis, x) for x in (q, k, v))
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    monkeypatch.setattr(SolPrepassFused, "keep_across_generations", False)

    refs = []
    for geo, sink, bmap in cases:
        SolPrepassFused.new_generation()  # drops the shared state: a fresh pre-pass per length
        pre = _rtlen_prepass(mesh_device, sp_axis, ccl_manager)
        pre.block_map = bmap
        pre.prepare(geo, sink, scale)
        refs.append(_rtlen_outputs(mesh_device, sp_axis, tp_axis, pre(tq, tk, tv, scale=scale)))
        ttnn.synchronize_device(mesh_device)

    monkeypatch.setattr(SolPrepassFused, "keep_across_generations", True)
    SolPrepassFused._shared.clear()
    pre = _rtlen_prepass(mesh_device, sp_axis, ccl_manager)
    added = []
    for i, (geo, sink, bmap) in enumerate(cases):
        SolPrepassFused.new_generation()  # a new request, as MiniMaxH3Attention.sol_logical_n's setter does
        pre.block_map = bmap
        before = mesh_device.num_program_cache_entries()
        pre.prepare(geo, sink, scale)
        got = _rtlen_outputs(mesh_device, sp_axis, tp_axis, pre(tq, tk, tv, scale=scale))
        ttnn.synchronize_device(mesh_device)
        added.append(mesh_device.num_program_cache_entries() - before)
        for name, ref in refs[i].items():
            same = torch.equal(got[name], ref)
            logger.info(f"[prepass runtime_len] logical_n {geo.logical_n} {name}: bit-identical {same}")
            assert same, f"logical_n {geo.logical_n}: {name} differs from the fresh per-length pre-pass"
    assert len(SolPrepassFused._shared) == 1, f"{len(SolPrepassFused._shared)} states for one padded geometry"
    logger.info(f"[prepass runtime_len] program cache entries added per length: {added}")
    assert all(a == 0 for a in added[1:]), f"program cache entries added per length {added}: expected 0 after the first"

    # Trace: captured at the first length, replayed at the others after prepare() rewrote the constants.
    geo, sink, bmap = cases[0]
    pre.block_map = bmap
    pre.prepare(geo, sink, scale)
    pre(tq, tk, tv, scale=scale)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = pre(tq, tk, tv, scale=scale)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    try:
        for i in (1, 2, 0, 2):
            geo, sink, bmap = cases[i]
            pre.block_map = bmap
            pre.prepare(geo, sink, scale)
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            got = _rtlen_outputs(mesh_device, sp_axis, tp_axis, out)
            for name, ref in refs[i].items():
                same = torch.equal(got[name], ref)
                logger.info(
                    f"[prepass runtime_len trace] replay at logical_n {geo.logical_n} {name}: bit-identical {same}"
                )
                assert same, f"trace replay at logical_n {geo.logical_n}: {name} differs"
    finally:
        ttnn.release_trace(mesh_device, tid)
