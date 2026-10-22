# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""prefill_id_translate_single.cpp vs prefill_id_translate.cpp: the same set of entries expressed as FIFO slots (entry + off) and as single-trace slots (scratch / history) must translate to the same pool rows."""

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_paged as PG
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_single as PS


def _rep(md, t, dtype, layout):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 100000000},
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_translate_single(mesh_device):
    md = mesh_device
    g = torch.Generator().manual_seed(5)
    C, U, MAXP = 512, 1, 8192
    pt = torch.randint(0, 8000, (U, MAXP), generator=g, dtype=torch.int64).to(torch.int32)
    pt_d = _rep(md, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    for R, src_off in ((1, 192), (2, 0)):
        Cc = C // R
        for s0 in (0, 512 * 100, 512 * 250, 512 * 500, 512 * 2000):
            nh = s0 // R
            L = 1048576 // R
            off_f = L - (s0 + C) // R
            hist = nh
            tot = nh + Cc  # entries visible to the last row
            j = torch.randint(0, tot, (U, 1, C, 512), generator=g, dtype=torch.int64)
            fifo = (j + off_f).to(torch.int32)
            single = torch.where(j >= nh, j - nh, j + Cc).to(torch.int32)
            offf_d = _rep(md, torch.tensor([[off_f] + [0] * 31], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            offs_d = _rep(md, torch.tensor([[nh] + [0] * 31], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            a = (
                ttnn.to_torch(
                    ttnn.get_device_tensors(
                        PG.translate_ids(
                            _rep(md, fifo, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                            pt_d,
                            offf_d,
                            ratio=R,
                            src_off=src_off,
                            C=C,
                        )
                    )[0]
                ).long()
                & 0xFFFFFFFF
            )
            b = (
                ttnn.to_torch(
                    ttnn.get_device_tensors(
                        PS.translate_ids_single(
                            _rep(md, single, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                            pt_d,
                            offs_d,
                            ratio=R,
                            src_off=src_off,
                            C=C,
                            ccur=Cc,
                        )
                    )[0]
                ).long()
                & 0xFFFFFFFF
            )
            print(
                f"TRANSLATE R={R} s0={s0}: equal {bool(torch.equal(a, b))}, mismatching {int((a != b).sum())} of {a.numel()}",
                flush=True,
            )
    print("TRANSLATE DONE", flush=True)
