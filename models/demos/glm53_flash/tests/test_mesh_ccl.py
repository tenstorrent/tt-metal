# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 groundwork: the mesh placement + collective contract of ``tt/mesh.py`` on a chip block.

Checks, against torch, the three patterns every multi-chip sublayer uses (design section 5):
  * TP over columns: column-parallel W1, row-parallel W2, ``all_reduce_cols`` -> replicated result,
    with the rows carrying independent (DP) tokens;
  * the hybrid-MoE row axis: ``all_gather_rows`` of the row activations and ``reduce_scatter_rows``
    of per-row partials;
  * the same collectives captured in a trace after one eager warm-up and replayed with new inputs.

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_mesh_ccl.py -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.mesh import GLMMesh, interleave_blocks, mesh_device_params

ROWS, COLS = 2, 4
H, I = 4096, 2048  # hidden, intermediate (per-TP slice 512)


def _mesh(mesh_device) -> GLMMesh:
    m = GLMMesh.from_device(mesh_device)
    assert (m.rows, m.cols) == (ROWS, COLS), (m.rows, m.cols)
    return m


def _tp_swiglu_host(x, w13, w2, cols):
    """Per-row torch reference of the clamped SwiGLU with the fused [gate|up] layout of the chips."""
    gu = x @ w13
    gate, up = gu[..., :I], gu[..., I:]
    return (torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)) @ w2


@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_tp_cols_dp_rows(mesh_device):
    m = _mesh(mesh_device)
    torch.manual_seed(0)
    B = 32
    x = torch.randn(1, 1, ROWS * B, H, dtype=torch.bfloat16) * 0.5  # row r owns tokens [r*B, (r+1)*B)
    w_gate = torch.randn(H, I, dtype=torch.bfloat16) * 0.02
    w_up = torch.randn(H, I, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(I, H, dtype=torch.bfloat16) * 0.02
    # chip c keeps [gate_c | up_c]: interleave so a plain column split of dim -1 hands each chip its pair
    w13_host = interleave_blocks([w_gate, w_up], COLS, dim=-1)
    tt_x = m.upload(x, ttnn.bfloat16, rows=2)
    tt_w13 = m.upload(w13_host, ttnn.bfloat8_b, cols=-1)
    tt_w2 = m.upload(w2, ttnn.bfloat8_b, rows=None, cols=-2)
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    I_loc = I // COLS

    def layer(tt_x):
        gu = ttnn.linear(tt_x, tt_w13, dtype=ttnn.bfloat16, compute_kernel_config=compute)
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, B, I_loc])
        up = ttnn.slice(gu, [0, 0, 0, I_loc], [1, 1, B, 2 * I_loc])
        gate = ttnn.clamp(gate, min=None, max=10.0)
        up = ttnn.clamp(up, min=-10.0, max=10.0)
        h = ttnn.multiply(ttnn.silu(gate), up)
        part = ttnn.linear(h, tt_w2, dtype=ttnn.bfloat16, compute_kernel_config=compute)  # TP partial
        return m.all_reduce_cols(part)

    ref = _tp_swiglu_host(x.float(), torch.cat([w_gate, w_up], -1).float(), w2.float(), COLS)
    for it in range(3):
        y = layer(tt_x)
        shards = [ttnn.to_torch(d).float() for d in m.device_tensors(y)]
        for r in range(ROWS):
            for c in range(COLS):
                p = pcc(ref[:, :, r * B : (r + 1) * B], shards[r * COLS + c])
                assert p > 0.995, (it, r, c, p)
                if c > 0:  # replicated across the TP group
                    assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (it, r, c)
        logger.info(
            f"iter {it}: TP4 swiglu + all_reduce_cols vs torch: worst pcc "
            f"{min(pcc(ref[:, :, r * B:(r + 1) * B], shards[r * COLS]) for r in range(ROWS)):.6f}; "
            f"rows carry different tokens: {not torch.equal(shards[0], shards[COLS])}"
        )


@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_rows_gather_and_reduce_scatter(mesh_device):
    m = _mesh(mesh_device)
    torch.manual_seed(1)
    B = 32
    x = torch.randn(1, 1, ROWS * B, H, dtype=torch.bfloat16)
    tt_x = m.upload(x, ttnn.bfloat16, rows=2)
    g = m.all_gather_rows(tt_x, dim=2)  # every chip: [1,1,ROWS*B,H] in row order
    for i, d in enumerate(m.device_tensors(g)):
        assert torch.equal(ttnn.to_torch(d), x), i
    logger.info(f"all_gather_rows: all {ROWS * COLS} chips hold the {ROWS * B} tokens in row order (byte-exact)")
    # fp32 routing weights gather the same way
    w = torch.randn(1, 1, ROWS * B, 288, dtype=torch.float32)
    gw = m.all_gather_rows(m.upload(w, ttnn.float32, rows=2), dim=2)
    for i, d in enumerate(m.device_tensors(gw)):
        assert torch.equal(ttnn.to_torch(d), w), i
    logger.info("all_gather_rows fp32 [T,288]: byte-exact")
    # reduce_scatter over rows: partials P_r [ROWS*B, H] per row chip -> row r gets sum_r' P_r'[r*B:(r+1)*B]
    parts = torch.randn(ROWS, 1, 1, ROWS * B, H, dtype=torch.bfloat16)
    tt_p = ttnn.from_torch(
        parts.reshape(ROWS, 1, ROWS * B, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=m.shard_rows(0),
    )  # row r holds parts[r] (replicated over cols)
    rs = m.reduce_scatter_rows(tt_p, dim=2)
    ref = parts.float().sum(0)[0]  # [1, ROWS*B, H]
    worst = 1.0
    for i, d in enumerate(m.device_tensors(rs)):
        r = i // COLS
        got = ttnn.to_torch(d).float().reshape(ROWS * B // ROWS * 1, -1) if False else ttnn.to_torch(d).float()
        assert got.shape[-2] == B, got.shape
        p = pcc(ref[:, r * B : (r + 1) * B].reshape(1, B, H), got.reshape(1, B, H))
        worst = min(worst, p)
        assert p > 0.999, (i, p)
    logger.info(f"reduce_scatter_rows: worst pcc vs torch {worst:.6f}")


@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS, trace_region_size=20_000_000)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
def test_collectives_in_trace(mesh_device):
    m = _mesh(mesh_device)
    torch.manual_seed(2)
    B = 32
    x0 = torch.randn(1, 1, ROWS * B, H, dtype=torch.bfloat16)
    tt_x = m.upload(x0, ttnn.bfloat16, rows=2)  # persistent input

    def graph(t):
        g = m.all_gather_rows(t, dim=2)  # [1,1,ROWS*B,H]
        s = m.reduce_scatter_rows(g, dim=2)  # = ROWS * x_row (every row holds the same gathered tensor)
        return m.all_reduce_cols(s)  # = COLS * ROWS * x_row

    def expect(x):
        return x.float() * ROWS * COLS

    y = graph(tt_x)  # eager warm-up compiles every collective before the capture
    for i, d in enumerate(m.device_tensors(y)):
        r = i // COLS
        assert pcc(expect(x0)[:, :, r * B : (r + 1) * B], ttnn.to_torch(d).float()) > 0.9999, i
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    y_tr = graph(tt_x)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for step in range(3):
        xs = torch.randn(1, 1, ROWS * B, H, dtype=torch.bfloat16)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(xs, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=m.shard_rows(2)), tt_x
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        worst = 1.0
        for i, d in enumerate(m.device_tensors(y_tr)):
            r = i // COLS
            worst = min(worst, pcc(expect(xs)[:, :, r * B : (r + 1) * B], ttnn.to_torch(d).float()))
        assert worst > 0.9999, (step, worst)
        logger.info(f"trace replay {step}: gather->reduce_scatter->all_reduce on new inputs, worst pcc {worst:.6f}")
    ttnn.release_trace(mesh_device, tid)
