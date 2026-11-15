# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""persistent_moe_request on DSpark verify blocks (``rows`` = 4) with the hyper-connection post step fused.

Same setup as ``models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent.py`` (three random layers
on the 3x4 MoE submesh at Galaxy row 5, TP4 requesters at rows 2..4; needs the whole 8x4 Galaxy and the V4.1 config).
:class:`PersistentMoE` is built with ``rows=ROWS`` and ``fuse_hc=True``.

- ``test_persistent_moe_request_rows_hc_post``: each block first as a plain request (output: the block's MoE rows in
  ``tile_rows`` order, checked per row against torch), then again with the HC post step fused -- streams
  ``(rows x 32)``-tiled ``[1, 1, hc * rows, D]`` (row ``h * rows + j``), per-row ``post`` ``[1, rows, hc, 1]`` and
  ``comb`` ``[1, rows, hc, hc]`` -- checked per row against ``post_j[h] * x_j + sum_g comb_j[g, h] * streams[g, j]``
  on the plain output ``x``. The HC round's request rows are produced on device (an L1 copy moved to DRAM by a
  device op), not uploaded, and must give the plain round's output bit for bit.
- ``test_persistent_moe_request_rows_hc_post_perf``: one requester, traced back to back, per block (``MOE_ROWS``
  rows, ``MOE_PERF_FUSE_HC=0`` for plain requests). For the HC post step's own time capture zones with
  ``TT_METAL_STREAMING_PROFILER=1 TT_METAL_STREAMING_PROFILER_ZONE_CSV=<csv>`` and read them with
  ``models/experimental/deepseek_v41_flash/tools/moe_requester_zones.py <csv>`` (``tail`` = the requester's work
  once the partials are in).
"""

import gc
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_moe_persistent import (
    PARENT_MESH,
    PCC,
    REQUEST_ROWS,
    STEPS,
    _load_layers,
    _pcc,
    _reference,
    _setup,
    _single_senders,
    _snapshot,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.decode.persistent_moe import PersistentMoE, tile_rows, untile_rows

ROWS = int(os.environ.get("MOE_ROWS", 4))
HC = 4
# The HC post step against torch's bf16 math on the plain output: bf16 rounding of the products and sums only.
HC_PCC = 0.9999
MIXES = (1.0, 0.0, 0.3, 0.6, 1.0, 0.15)


def _block(hidden, mix, generator):
    base = torch.randn(hidden, generator=generator)
    rows = []
    for _ in range(ROWS):
        row = base * (1.0 - mix) + torch.randn(hidden, generator=generator) * mix
        rows.append(row / row.norm() * hidden**0.5)
    return torch.stack(rows).to(torch.bfloat16).float()


def _hc_inputs(mesh, moe_op, generator):
    """Host ``(streams [hc, rows, D], post [rows, hc], comb [rows, hc, hc])`` and their device tensors."""
    hidden = moe_op.hidden
    streams = torch.randn(HC, ROWS, hidden, generator=generator).to(torch.bfloat16)
    post = (2 * torch.rand(ROWS, HC, generator=generator)).to(torch.bfloat16)
    comb = torch.softmax(torch.randn(ROWS, HC, HC, generator=generator), dim=-1).to(torch.bfloat16)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    tt_streams = moe_op.block_streams(mesh, streams if ROWS > 1 else streams[:, 0])
    tt_post = ttnn.from_torch(
        post.reshape(1, ROWS, HC, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
    )
    tt_comb = ttnn.from_torch(
        comb.reshape(1, ROWS, HC, HC), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
    )
    return (streams, post, comb), dict(streams=tt_streams, post=tt_post, comb=tt_comb)


def _device_row(moe_op, requester, block):
    """``block``'s request row produced on device: uploaded to an L1 row, then moved into a fresh
    ``[1, 1, 1, rows * D]`` ROW_MAJOR DRAM-interleaved one-page row by a device op."""
    mesh = moe_op.request_meshes[requester]
    row = tile_rows(block) if ROWS > 1 else block.reshape(-1)
    l1 = ttnn.from_torch(
        row.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    dram = ttnn.to_memory_config(l1, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(l1)
    return dram


def _expected(x, streams, post, comb):
    """``[hc, rows, D]``: per row j, ``post_j[h] * x_j + sum_g comb_j[g, h] * streams[g, j]`` in fp32."""
    mix = torch.einsum("jgh,gjd->hjd", comb.float(), streams.float())
    return post.float().T.unsqueeze(-1) * x.unsqueeze(0) + mix


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_persistent_moe_request_rows_hc_post(mesh_device):
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    layers, moes = _load_layers(moe_mesh, config)
    num_requesters = len(request_meshes)
    steps = STEPS[: 2 * num_requesters]

    g = torch.Generator().manual_seed(0)
    blocks = [_block(config.hidden_size, MIXES[s % len(MIXES)], g) for s in range(len(steps))]
    references = [
        [_reference(x, *layers[layer], config, moes[layer].experts) for x in blocks[s]] for s, layer in enumerate(steps)
    ]
    moe_op = PersistentMoE(
        moe_mesh,
        request_meshes,
        moes,
        socket_core,
        request_core,
        _single_senders(REQUEST_ROWS),
        fuse_hc=True,
        rows=ROWS,
    )
    plain, flat_plain, hc_outs, hc_host = [], [], [], []
    try:
        # Plain requests (uploaded rows): the block's MoE rows.
        for first in range(0, len(steps), num_requesters):
            pending = []
            for s in range(first, min(first + num_requesters, len(steps))):
                moe_op.upload(s % num_requesters, blocks[s])
                pending.append(moe_op.request(s % num_requesters, steps[s]))
            for out in pending:
                flat = PersistentMoE.row_of(out)
                flat_plain.append(flat)
                plain.append(untile_rows(flat.float(), ROWS) if ROWS > 1 else flat.float().reshape(1, -1))
                ttnn.deallocate(out)
        # The same blocks with the HC post step fused, their request rows made on device.
        for first in range(0, len(steps), num_requesters):
            pending = []
            for s in range(first, min(first + num_requesters, len(steps))):
                j = s % num_requesters
                host, dev = _hc_inputs(moe_op.request_meshes[j], moe_op, g)
                row = _device_row(moe_op, j, blocks[s])
                pending.append((moe_op.request(j, steps[s], row, **dev), row, dev))
                hc_host.append(host)
            for out, row, dev in pending:
                hc_outs.append([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)])
                ttnn.deallocate(out)
                ttnn.deallocate(row)
                for t in dev.values():
                    ttnn.deallocate(t)
        # A plain request on a device-made row must match the uploaded one bit for bit.
        same_bits = []
        for s in range(num_requesters):
            row = _device_row(moe_op, s, blocks[s])
            out = moe_op.request(s, steps[s], row)
            same_bits.append(torch.equal(PersistentMoE.row_of(out), flat_plain[s]))
            ttnn.deallocate(out)
            ttnn.deallocate(row)
    finally:
        moe_op.stop()

    report, failed = [], False
    for s, layer in enumerate(steps):
        moe_pccs = [max(_pcc(c, plain[s][r]) for c in references[s][r][0]) for r in range(ROWS)]
        streams, post, comb = hc_host[s]
        expected = _expected(plain[s], streams, post, comb)  # [hc, rows, D]
        hc_pccs, errs = [], []
        for got in hc_outs[s]:
            got = got.reshape(HC, ROWS, -1)
            for r in range(ROWS):
                hc_pccs.append(_pcc(expected[:, r], got[:, r]))
                errs.append((expected[:, r] - got[:, r]).abs().max().item())
        failed |= min(moe_pccs) < PCC or min(hc_pccs) < HC_PCC
        report.append(
            f"step {s} layer {layer}: MoE per-row PCC vs torch {' '.join(f'{p:.5f}' for p in moe_pccs)}; HC post "
            f"per-row PCC (all chips) min {min(hc_pccs):.6f}, max |err| {max(errs):.3e}"
        )
    logger.info("\n" + "\n".join(report) + f"\ndevice-made request rows match uploaded ones bit for bit: {same_bits}")
    assert not failed, "\n".join(report)
    assert all(same_bits), f"device-made request rows gave other outputs: {same_bits}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 32 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_persistent_moe_request_rows_hc_post_perf(mesh_device):
    """Per-block latency, one requester (Galaxy row 4), traced and replayed back to back with no host I/O, at each
    ``MOE_ROWS_MIXES``; ``MOE_PERF_FUSE_HC=0`` sends plain requests. Each traced layer's output must match its
    eager run's bit for bit."""
    moe_mesh, request_meshes, config, socket_core, request_core = _setup(mesh_device)
    _, moes = _load_layers(moe_mesh, config)
    hidden = config.hidden_size
    warmup = int(os.environ.get("MOE_PERF_WARMUP", 20))
    steps = int(os.environ.get("MOE_PERF_STEPS", 200))
    mixes = [float(m) for m in os.environ.get("MOE_ROWS_MIXES", "1,0.3,0").split(",")]
    fuse = os.environ.get("MOE_PERF_FUSE_HC", "1") == "1"
    order = [STEPS[i % len(STEPS)] for i in range(warmup + steps)]
    mesh = request_meshes[-1]
    results = []
    for mix in mixes:
        g = torch.Generator().manual_seed(1)
        blocks = [_block(hidden, mix, g) for _ in order]
        moe_op = PersistentMoE(
            moe_mesh, [mesh], moes, socket_core, request_core, _single_senders(REQUEST_ROWS[-1:]), True, ROWS
        )
        name = f"rows {ROWS} mix {mix:.2f}, HC post {'fused' if fuse else 'off'}"
        hc = _hc_inputs(mesh, moe_op, g)[1] if fuse else {}
        try:
            eager = {}
            for i, layer in enumerate(order[: len(STEPS)]):
                moe_op.upload(0, blocks[i])
                out = moe_op.request(0, layer, **hc)
                eager.setdefault(layer, (i, ttnn.to_torch(ttnn.get_device_tensors(out)[0])))
                ttnn.deallocate(out)
            inputs, tids, outs = {}, {}, {}
            for layer, (first, _) in eager.items():
                inputs[layer] = moe_op.request_input(0)
                moe_op.upload(0, blocks[first], inputs[layer])
                tids[layer] = ttnn.begin_trace_capture(mesh, cq_id=0)
                outs[layer] = moe_op.request(0, layer, inputs[layer], **hc)
                ttnn.end_trace_capture(mesh, tids[layer], cq_id=0)
            ttnn.synchronize_device(mesh)

            def replay(layers):
                for layer in layers:
                    ttnn.execute_trace(mesh, tids[layer], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)

            replay(order[:warmup])
            start = time.perf_counter()
            replay(order[warmup:])
            per_block = (time.perf_counter() - start) / steps
            results.append(f"{name:<36} traced, device only: {per_block * 1e6:8.1f} us/block")
            mismatched = [
                layer
                for layer, (_, ref) in eager.items()
                if not all(torch.equal(ttnn.to_torch(t), ref) for t in ttnn.get_device_tensors(outs[layer]))
            ]
            for layer in tids:
                ttnn.release_trace(mesh, tids[layer])
            for t in list(outs.values()) + list(inputs.values()) + list(hc.values()):
                ttnn.deallocate(t)
        finally:
            moe_op.stop()
        assert not mismatched, f"{name}: traced outputs of layers {mismatched} differ from eager"
        del moe_op
        gc.collect()
        ttnn.synchronize_device(moe_mesh)
    logger.info(f"persistent_moe_request, {ROWS}-row blocks, 1 requester:\n" + "\n".join(results))
