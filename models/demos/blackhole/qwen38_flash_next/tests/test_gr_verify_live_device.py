"""Device test of the verify's live-row GR read (QWEN38_GR_VERIFY_LIVE_ROWS): a 32-row residual tile with R live rows (the rest zero)
through front v3 + gate v3 at R rows (``live_rows``), against today's 32-row read (default front at 32 rows + the 32-row gate):
valid rows of the block and the injection bit for bit, the block's dead rows exact zeros, the output shapes the 32-row ones.
Run: QWEN38_FUSED_DEVICE_TEST=1 MESH_DEVICE=P150x8 TT_VISIBLE_DEVICES=... pytest this_file -s"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.tests import test_gr_front_device as F
from models.demos.blackhole.qwen38_flash_next.ttnn import decode_matmul as dm
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_recip_last as grl
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_recip_last import front3

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the TP=8 mesh")
STEPS = int(os.environ.get("GL_STEPS", "6"))
TP = MESH_SHAPE[1]


def bits(x):
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def residual32(mesh, live, seed):
    g = torch.Generator().manual_seed(seed)
    scale = torch.exp2(torch.randint(-6, 7, (1, F.B, 1, 1), generator=g).float())
    x = torch.zeros(1, F.B, 32, F.HIDDEN)
    x[:, :, :live] = torch.randn(1, F.B, live, F.HIDDEN, generator=g) * scale
    x[:, 3, :live] = 0.5
    return ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=F.DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
    )


def host_rows(outs):
    return [
        (ttnn.to_torch(b).clone(), ttnn.to_torch(i).clone())
        for b, i in zip(*(ttnn.get_device_tensors(t) for t in outs))
    ]


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [F.DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("live", [2, 4, 5])
def test_live_rows_read(mesh_device, live):
    mesh = mesh_device
    mesh.enable_program_cache()
    g = torch.Generator().manual_seed(11)
    up = ttnn.from_torch(
        (torch.randn(1, 1, F.PARTIAL, TP * F.FLAT, generator=g) * 0.05).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=dm.dram_sharded_weight_memory_config(mesh, F.PARTIAL, F.FLAT),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 3)),
    )
    for step in range(STEPS):
        _, gam, down = F.make_inputs(mesh, 1, 500 + step)
        res = residual32(mesh, live, 900 + 10 * live + step)
        stats, norm, parts = grl.stats_recip_down_gather(res, gam, down)  # today: the default front at 32 rows
        ttnn.deallocate(stats)
        ref = host_rows(gr_read.low_rank_gate(parts, norm, up, matmul="chain"))
        ttnn.deallocate(norm), ttnn.deallocate(parts)
        stats, norm, parts = front3.stats_recip_down_gather(res, gam, down, live_rows=live)
        ttnn.deallocate(stats)
        out = gate3.low_rank_gate3(parts, norm, up, matmul="chain", out_rows=32)
        assert tuple(out[0].shape) == (1, 1, 32, 320) and tuple(out[1].shape) == (1, 1, 32, 4), (
            out[0].shape,
            out[1].shape,
        )
        got = host_rows(out)
        for t in (*out, norm, parts, res, gam, down):
            ttnn.deallocate(t)
        for d, ((rb, ri), (gb, gi)) in enumerate(zip(ref, got)):
            assert torch.equal(
                bits(rb[:, :, :live]), bits(gb[:, :, :live])
            ), f"live={live} step {step} dev {d}: block valid rows differ"
            assert torch.equal(
                bits(ri[:, :, :live]), bits(gi[:, :, :live])
            ), f"live={live} step {step} dev {d}: injection valid rows differ"
            assert not gb[:, :, live:].any(), f"live={live} step {step} dev {d}: block dead rows not zero"
            assert not gi[:, :, live:].any(), f"live={live} step {step} dev {d}: injection dead rows not zero"
            assert not rb[:, :, live:].any(), "reference block dead rows not zero"
    logger.info(f"live={live}: valid rows bit-identical to the 32-row read over {STEPS} steps, dead rows zero")


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [F.DEVICE_PARAMS], indirect=True)
def test_mixed_row_counts_share_the_landing_buffer(mesh_device):
    """Front v3 calls of different row counts back to back (one shared landing buffer with the live-row knob on): valid rows of
    every call bit-identical to the default front."""

    mesh = mesh_device
    mesh.enable_program_cache()
    front3.prepare(mesh)
    for step, rows in enumerate([1, 2, 4, 5, 4, 1, 5, 2, 5, 5, 1, 4] * 2):
        res, gam, down = F.make_inputs(mesh, rows, 40 + step)
        ref = F.valid_rows(grl.stats_recip_down_gather(res, gam, down), rows)
        got = F.valid_rows(front3.stats_recip_down_gather(res, gam, down), rows)
        for d, (a, b) in enumerate(zip(ref, got)):
            for what, x, y in zip(("stats", "normalized", "partials"), a, b):
                if what != "stats":
                    assert torch.equal(bits(x), bits(y)), f"step {step} rows {rows} dev {d}: {what} differs"
        for t in (res, gam, down):
            ttnn.deallocate(t)
