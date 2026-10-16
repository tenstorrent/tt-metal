# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device pre-pass (sol_prepass.SolPrepass) vs its torch mirror and vs the Sol-Attn reference routing."""

from __future__ import annotations

import os
import sys

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.sol_prepass import (
    NEG,
    SolGeometry,
    SolPrepass,
    build_constants,
    torch_reference,
)
from ....parallel.manager import CCLManager
from ....utils.tensor import from_torch
from .common import GALAXY_RING

sys.path.insert(0, os.environ.get("SOL_ATTN_WORK", "/home/losullivan/sol_attn_work"))  # /data copy on Slurm nodes
import sol_ref as R  # noqa: E402


@GALAXY_RING
@pytest.mark.parametrize(("rows", "logical_n", "q_chunk", "k_chunk"), [(1184, 9000, 256, 512), (1184, 9472, 256, 256)])
@pytest.mark.parametrize("tau", [0.5, 1.0])
def test_sol_prepass(
    mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, rows, logical_n, q_chunk, k_chunk, tau, reset_seeds
):
    S = tuple(mesh_device.shape)[sp_axis]
    H = 14
    D = 128
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=k_chunk)
    scale = D**-0.5
    torch.manual_seed(1)
    # structured data so routing is non-trivial: a few "topics" shared by Q and K
    topics = torch.randn(8, D)
    tid = torch.randint(0, 8, (S * rows,))
    q = (topics[tid] + 0.7 * torch.randn(S * rows, D)).view(S, 1, rows, D).expand(S, H, rows, D).clone()
    q += 0.3 * torch.randn(S, H, rows, D)
    k = (topics[tid] + 0.7 * torch.randn(S * rows, D)).view(S, 1, rows, D).expand(S, H, rows, D).clone()
    k += 0.3 * torch.randn(S, H, rows, D)
    v = torch.randn(S, H, rows, D)
    qb, kb, vb = (x.bfloat16().float() for x in (q, k, v))
    sink_tokens = 300

    consts = build_constants(geo, band=1, sink_tokens=sink_tokens, force_first_chunk=True, scale=scale)
    ref = torch_reference(qb, kb, vb, geo, consts, tau=tau, scale=scale)

    # ---- cross-check the torch mirror against sol_ref (q-chunk x 64-block granularity, kernel rules)
    rgeo = R.Geometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=q_chunk, k_chunk=64)
    qg = qb.permute(1, 0, 2, 3).reshape(H, S * rows, D)
    kg = kb.permute(1, 0, 2, 3).reshape(H, S * rows, D)
    rr = R.route(
        qg,
        kg,
        rgeo,
        tau=tau,
        scale=scale,
        band=1,
        sink_tokens=sink_tokens,
        tt_kernel_rules=True,
        dense_prefix_queries=True,
    )
    nqc, nb = geo.nqc, geo.nb
    cols = torch.cat(
        [torch.arange(s_ * geo.nb_pad, s_ * geo.nb_pad + nb) for s_ in range(S)]
    )  # padded -> real block cols
    mirror = ref["block_mask"][:, :, :nqc][..., cols].permute(1, 0, 2, 3).reshape(H, S * nqc, S * nb)
    expect = rr["chunk_mask"]
    agree = (mirror.bool() == expect).float().mean().item()
    logger.info(
        f"torch mirror vs sol_ref block mask agreement {agree:.5f}; density mirror {mirror.mean():.4f} ref {expect.float().mean():.4f}"
    )
    assert agree > 0.999, agree

    # ---- device pre-pass
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    pre = SolPrepass(mesh_device=mesh_device, ccl_manager=ccl_manager, sp_axis=sp_axis, tau=tau)
    pre.prepare(geo, sink_tokens, scale)

    def upload(x):  # [S,H,rows,D] -> per-device [1,H,rows,D], sharded on SP, replicated on TP
        return from_torch(
            x, device=mesh_device, dtype=ttnn.bfloat16, mesh_axes=[sp_axis, None, None, None], layout=ttnn.TILE_LAYOUT
        )

    tq, tk, tv = upload(qb), upload(kb), upload(vb)
    out = pre(tq, tk, tv, scale=scale, report_density=True)
    ttnn.synchronize_device(mesh_device)

    def to_host(t, dims):
        return ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
        )

    cat_dims = [None, None]
    cat_dims[sp_axis] = 0  # stack shards on a new leading dim via dim 0 (batch) of [1,H,...]
    cat_dims[tp_axis] = 1
    dev_mask = to_host(out["block_mask"], cat_dims)  # [S, 4*H, nqc_pad, nbg] uint32 (TP replicas repeated)
    dev_mask = dev_mask[:, :H].float()
    agree = (dev_mask == ref["block_mask"]).float().mean().item()
    logger.info(f"device block mask vs mirror agreement {agree:.5f}; device density {pre.last_density:.4f}")
    assert agree > 0.995, agree

    pk = to_host(out["pooled_k"], cat_dims)[0, :H].float()  # replicated across shards after all-gather
    torch.testing.assert_close(pk, ref["pooled_k"].bfloat16().float(), atol=3e-2, rtol=3e-2)
    pv = to_host(out["pooled_v"], cat_dims)[0, :H].float()
    torch.testing.assert_close(pv, ref["pooled_v"].bfloat16().float(), atol=3e-2, rtol=3e-2)

    bias = to_host(out["bias"], cat_dims)[:, :H].float()  # [S,H,nqc_pad*32,nbg]
    bias_rows = bias.view(S, H, geo.nqc_pad, 32, geo.n_blocks_pad)
    assert torch.equal(bias_rows[:, :, :, 0], bias_rows[:, :, :, 17]), "bias tile rows must be identical"
    b0 = bias_rows[:, :, :, 0]
    kept_dev = b0 < NEG / 2
    kept_ref = ref["bias"] < NEG / 2
    logger.info(f"bias kept-marker agreement {(kept_dev == kept_ref).float().mean():.5f}")
    assert (kept_dev == kept_ref).float().mean() > 0.995
    live = ~kept_ref & ~kept_dev  # where device and mirror routing agree the block is approximated
    torch.testing.assert_close(b0[live], ref["bias"][live].bfloat16().float(), atol=0.3, rtol=1e-2)
    logger.info("prepass OK")
