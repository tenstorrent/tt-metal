# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Is the fused AG-matmul's gathered activation reusable by a second matmul?

`all_gather_minimal_matmul_async` gathers along K and already materializes the gathered
activation: `compute_output_specs` gives slot 0 the shape `in0 * ring_size`, and the caller
even supplies the buffer (`persistent_output_buffer`). If that buffer held a valid all-gather,
`to_gate_logits` could read it instead of paying a second gather -- a plumbing change, no kernel.

The suspect is `READ_FROM_LOCAL_INPUT`: `read_in0_block_sync` sources the device's OWN K-slice
from the unsharded input (in3) straight into the compute CB, so nothing ever writes that slice
into the gather buffer. The remote slices land there (the fabric relay writes through the in0
address-gen); the local one may be a hole -- and the buffer is `torch.empty`, so a hole reads as
uninitialized garbage rather than zeros.

This measures each (device, K-slice) cell of the gather buffer against the true all-gather, and
checks the matmul output itself, which must stay correct either way (the local data reaches the
compute CB regardless of whether it is also written back).
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import bf16_tensor
from models.tt_dit.utils.test import ring_params

VIDEO_DIM = 4096


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return float("nan")
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    if denom == 0:
        return float("nan")
    return float((va @ vb) / denom)


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {**ring_params, "trace_region_size": 90000000}, id="ring_bh_4x8")],
    indirect=True,
)
def test_agmm_gathered_out(mesh_device: ttnn.MeshDevice) -> None:
    sp_axis, tp_axis = 1, 0
    tp, sp = tuple(mesh_device.shape)[tp_axis], tuple(mesh_device.shape)[sp_axis]
    assert (tp, sp) == (4, 8), f"expected 4x8 with tp on axis0, got tp={tp} sp={sp}"

    ccl = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    pc = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=sp, mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tp, mesh_axis=tp_axis),
    )
    d_loc = VIDEO_DIM // tp  # 1024

    # N=256 is the structural probe; N=1216 is the production S1 row count, where M blocks are
    # partial per core -- the local-slice write (if any) must not depend on that.
    for rows in (256, 1216):
        torch.manual_seed(0)
        x_full = torch.randn(1, 1, rows, VIDEO_DIM)  # the logical, already-gathered activation
        x = bf16_tensor(x_full, mesh_device, mesh_axis=tp_axis, shard_dim=3)  # device r holds cols [r*1024, ...]
        x_ref = x_full.to(torch.bfloat16).float()  # bf16-rounded, so the compare is not a dtype artifact

        w = torch.randn(VIDEO_DIM, VIDEO_DIM) * 0.02
        lin = ColParallelLinear(
            VIDEO_DIM, VIDEO_DIM, bias=False, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl
        )
        lin.load_torch_state_dict({"weight": w})

        # The gather buffer is handed to the op by the caller, so we can read it back afterwards.
        # get_ag_ping_pong_buffer alternates slots on every call, so allocate the pair first and
        # then read the index the op is about to consume.
        key = ("ag", tuple(x.shape), 3, tp_axis, x.get_dtype())
        ccl.get_ag_ping_pong_buffer(x.shape, 3, tp_axis, dtype=x.get_dtype())
        slot = ccl._ping_pong_buffer_indices[key]

        out = lin(x, parallel_config=pc)
        ttnn.synchronize_device(mesh_device)
        ag_buf = ccl._ping_pong_buffer_cache[key][slot]

        # A device's gather buffer should equal the full activation, K-slice by K-slice.
        ag_shards = ttnn.get_device_tensors(ag_buf)
        local_pccs, remote_pccs = [], []
        for dev_i, shard in enumerate(ag_shards):
            r = dev_i // sp  # mesh is (tp, sp): axis0 index is the TP rank
            got = ttnn.to_torch(shard).float()
            for s in range(tp):
                cell = _pcc(got[..., s * d_loc : (s + 1) * d_loc], x_ref[..., s * d_loc : (s + 1) * d_loc])
                (local_pccs if s == r else remote_pccs).append(cell)
                if dev_i < tp:  # one representative device per TP rank is enough to print
                    logger.info(f"AGBUF rows={rows} dev={dev_i} tp_rank={r} k_slice={s} pcc={cell:.6f}")

        worst_remote = min(remote_pccs)
        worst_local = min(local_pccs)
        best_local = max(local_pccs)
        logger.info(
            f"AGBUF_SUMMARY rows={rows} remote_slices_min_pcc={worst_remote:.6f} "
            f"local_slice_min_pcc={worst_local:.6f} local_slice_max_pcc={best_local:.6f}"
        )

        # The matmul output must be correct regardless: the local K-slice reaches the compute CB
        # via in3 whether or not it is also written back to the gather buffer.
        out_shards = ttnn.get_device_tensors(out)
        mm_pccs = []
        for dev_i, shard in enumerate(out_shards):
            r = dev_i // sp
            got = ttnn.to_torch(shard).float()
            exp = x_ref[0, 0] @ w[r * d_loc : (r + 1) * d_loc, :].T.float()
            mm_pccs.append(_pcc(got[0, 0], exp))
        logger.info(f"AGBUF_SUMMARY rows={rows} matmul_out_min_pcc={min(mm_pccs):.6f}")

        ttnn.deallocate(out)

    logger.info("AGBUF_DONE")
