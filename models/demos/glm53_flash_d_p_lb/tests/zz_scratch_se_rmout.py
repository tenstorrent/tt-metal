# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scratch: the folded shared expert's down projection with a row-major output (1D multicast, untilize_out) vs torch,
into a fresh output and into a preallocated one."""

import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S_ = spec()
pytestmark = device_timeout(S_)


@mesh_parametrize
def test_se_rmout(mesh_device):
    import ttnn
    from models.demos.glm53_flash_d_p.tt.shared_tp_row import TtSharedTpRow

    M, H, I = 1280, 4096, 2048
    g = torch.Generator().manual_seed(0)
    w = [
        torch.randn(I, H, generator=g) * 0.02,
        torch.randn(I, H, generator=g) * 0.02,
        torch.randn(H, I, generator=g) * 0.02,
    ]
    se = TtSharedTpRow(mesh_device, *w, axis=0)
    h = torch.randn(M, se.per, generator=g)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    hd = ttnn.from_torch(
        h.view(1, 1, M, se.per), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    ref = h @ w[2][:, : se.per].T  # (chip (0, *): slice 0)
    cfg = se._down_rm_config(M, se.per, H)
    logger.info(f"config {cfg}")
    for name, out in (
        ("fresh", None),
        (
            "prealloc",
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, M, H]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
            ),
        ),
    ):
        o = ttnn.matmul(
            hd,
            se.w_down,
            dtype=ttnn.bfloat16,
            program_config=cfg,
            compute_kernel_config=se.cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            optional_output_tensor=out,
        )
        got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(M, H).float()
        rel = ((got - ref).norm() / ref.norm()).item()
        logger.info(
            f"RMOUT {name}: layout {o.layout} shape {o.shape} rel {rel:.4f} finite {bool(torch.isfinite(got).all())} row0 {got[0, :4].tolist()} ref {ref[0, :4].tolist()}"
        )
