# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Capture a Sol pre-pass block mask + pooled K/V for the sparse ring SDPA perf benchmark (dev only).

Run on a checkout of sosborne/sol-attn-cleanup (needs models/tt_dit/models/transformers/minimax_h3/sol_prepass.py),
on the 4x8 BH Galaxy, with this file copied in (it is a pytest test so the repo's mesh_device fixture opens the mesh):

    pytest dev/capture_masks.py -s -k 4x8
    CAPTURE_TAU=2.0 CAPTURE_OUT=masks_5s_tau2.pt pytest dev/capture_masks.py -s -k 4x8
    CAPTURE_ACTS=acts.pt pytest dev/capture_masks.py -s -k 4x8      # real activations instead of structured inputs

Env: CAPTURE_ROWS (4736), CAPTURE_LOGICAL_N (37888), CAPTURE_TAU (1.0), CAPTURE_SINK (849),
CAPTURE_OUT (sol_masks_r<rows>_tau<tau>.pt), CAPTURE_ACTS (optional .pt dict {"q","k","v"}, each
[1, H, N, D] post-RoPE with all H = 14 * TP heads and N <= SP * rows tokens; zero-padded to SP * rows).

Output (torch.save): {"block_mask": [S, H, nb_pad, S*nb_pad] int32 (SP gathered on dim 0, heads over TP on dim 1),
"pooled_k", "pooled_v": [1, H, S*nb_pad, D] float, "rows", "logical_n", "tau", "density"}.
"""

import os

import torch
from loguru import logger

import ttnn
from models.tt_dit.models.transformers.minimax_h3.sol_prepass import SolGeometry, SolPrepass
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.minimax_h3.common import GALAXY_RING

HEADS_PER_DEVICE, D, BLOCK = 14, 128, 64


def _structured_qkv(H, n, seed=0):
    # Same structure as the cleanup branch's _qkv: randn plus a positional ramp, so scores concentrate near the diagonal.
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn((1, H, n, D), generator=g) * s for s in (0.5, 0.5, 1.0))
    pos = torch.linspace(-2.0, 2.0, n).view(1, 1, n, 1)
    return q + pos, k + pos, v


@GALAXY_RING
def test_capture_masks(mesh_device, sp_axis, tp_axis, num_links, topology, is_fsdp, reset_seeds):
    rows = int(os.environ.get("CAPTURE_ROWS", 4736))
    logical_n = int(os.environ.get("CAPTURE_LOGICAL_N", 37888))
    tau = float(os.environ.get("CAPTURE_TAU", 1.0))
    sink = int(os.environ.get("CAPTURE_SINK", 849))
    out_path = os.environ.get("CAPTURE_OUT", f"sol_masks_r{rows}_tau{tau:g}.pt")
    shape = tuple(mesh_device.shape)
    S, TP = shape[sp_axis], shape[tp_axis]
    H, n = HEADS_PER_DEVICE * TP, S * rows
    scale = D**-0.5

    if os.environ.get("CAPTURE_ACTS"):
        acts = torch.load(os.environ["CAPTURE_ACTS"])
        q, k, v = (torch.nn.functional.pad(acts[x].float(), (0, 0, 0, n - acts[x].shape[2])) for x in "qkv")
        assert q.shape == (1, H, n, D), q.shape
    else:
        q, k, v = _structured_qkv(H, n)

    # Sequence on SP (dim 2), heads on TP (dim 1): the sparse op's layout (14 heads per device).
    in_dims = [None, None]
    in_dims[sp_axis], in_dims[tp_axis] = 2, 1
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=in_dims)
    tq, tk, tv = (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=mapper)
        for x in (q, k, v)
    )

    # q_chunk = 64: one mask row per 64-token Q block, so the mask is [.., nb_pad, S*nb_pad].
    geo = SolGeometry(shards=S, rows=rows, logical_n=logical_n, q_chunk=BLOCK, k_chunk=384)
    ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    pre = SolPrepass(mesh_device=mesh_device, ccl_manager=ccl, sp_axis=sp_axis, tau=tau, mode="compact", emit_bias=False)
    pre.prepare(geo, sink, scale)
    sol = pre(tq, tk, tv, scale=scale)
    ttnn.synchronize_device(mesh_device)

    # Device (sp, tp) -> dim 0 = sp, dim 1 = tp: [S, H, nb_pad, S*nb_pad]; pooled K/V are replicated over SP.
    out_dims = [None, None]
    out_dims[sp_axis], out_dims[tp_axis] = 0, 1
    comp = ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_dims, mesh_shape=shape)
    mask = ttnn.to_torch(sol["block_mask"], mesh_composer=comp).to(torch.int32)
    pk, pv = (ttnn.to_torch(sol[x], mesh_composer=comp)[:1].float() for x in ("pooled_k", "pooled_v"))
    assert mask.shape == (S, H, geo.nb_pad, S * geo.nb_pad), mask.shape

    cols = torch.cat([torch.arange(s * geo.nb_pad, s * geo.nb_pad + geo.nb) for s in range(S)])
    density = float(mask[:, :, : geo.nb][..., cols].float().mean())
    out = dict(block_mask=mask, pooled_k=pk, pooled_v=pv, rows=rows, logical_n=logical_n, tau=tau, density=density)
    torch.save(out, out_path)
    logger.info(f"saved {out_path}: mask {tuple(mask.shape)} pooled {tuple(pk.shape)} density {density:.4f}")
