# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Accuracy floor of the sparse op at production lengths, against the dense op on the same inputs. Dev branch only.

The 10 s keep-all case (pcc 0.9983-0.9984 vs the block-sparse reference) and the 15 s tau=1 bench case (0.99857)
miss the tests' 0.999 gate. This tells a bf16 accumulation limit from a kernel bug:

  sparse vs its fp64 block-sparse reference   (what the gate measures)
  dense  vs an fp64 exact dense reference     (the dense op's own floor at this length)
  per (head, shard, 64-row Q block) pcc of sparse vs its reference, the worst blocks and the max |error| location:
  a uniform spread points to precision, hot spots (a head, a core's blocks, a ring position) to a bug.

Both references cover the same slice (first / last head of the first / last shard). Inputs are randn (bf16-rounded).

    pytest dev/accuracy_case.py -s
"""

import pytest
import torch
from loguru import logger

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa import (
    BLOCK,
    KEEP_ALL,
    Case,
    Geometry,
    _density,
    _pcc,
    make_mask,
)

S = MESH_CONFIG.sp_size
CASES = {  # id: (rows per device, dense chunks, mask density)
    "s05_keepall": (4736, (320, 384), KEEP_ALL),
    "s10_keepall": (9184, (256, 512), KEEP_ALL),
    "s10_d015": (9184, (256, 512), 0.15),
    "s15_keepall": (13664, (256, 512), KEEP_ALL),
    "s15_d017": (13664, (256, 512), 0.17),
}


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


def dense_reference(q, k, v, logical_n, scale, heads, shards, rows):
    """fp64 exact attention over the logical_n real tokens, for the given heads / shards (NaN elsewhere)."""
    out = torch.full(q.shape, float("nan"))
    for h in heads:
        kh, vh = k[0, h, :logical_n].double(), v[0, h, :logical_n].double()
        for s in shards:
            for r0 in range(0, rows, 512):
                sl = slice(s * rows + r0, s * rows + min(r0 + 512, rows))
                w = torch.softmax((q[0, h, sl].double() @ kh.T) * scale, dim=1)
                out[0, h, sl] = (w @ vh).float()
    return out


def block_pccs(ref, out, heads, shards, rows, logical_n):
    """[(pcc, head, shard, q block)] per 64-row Q block of the computed slice (rows past logical_n dropped)."""
    res = []
    for h in heads:
        for s in shards:
            for qb in range(-(-rows // BLOCK)):
                r0, r1 = s * rows + qb * BLOCK, min(s * rows + (qb + 1) * BLOCK, s * rows + rows, logical_n)
                if r1 > r0:
                    res.append((_pcc(ref[0, h, r0:r1], out[0, h, r0:r1]), h, s, qb))
    return res


def summarize(tag, name, ref, out, heads, shards, rows, logical_n):
    per = block_pccs(ref, out, heads, shards, rows, logical_n)
    vals = sorted(p for p, *_ in per)
    q = lambda f: vals[min(len(vals) - 1, int(f * (len(vals) - 1)))]
    worst = sorted(per)[:5]
    err = (ref - out).abs()
    err = torch.where(torch.isnan(err), torch.zeros_like(err), err)
    flat = int(err.argmax())
    _, H, N, Dh = err.shape
    h, row, col = flat // (N * Dh), (flat // Dh) % N, flat % Dh
    logger.info(
        f"[{tag}] {name}: per-Q-block pcc min {vals[0]:.5f} p10 {q(0.1):.5f} median {q(0.5):.5f} max {vals[-1]:.5f}; "
        f"worst (pcc, head, shard, block) {[(round(p, 5), hh, ss, bb) for p, hh, ss, bb in worst]}; "
        f"max |err| {err.max().item():.4f} at head {h} shard {row // rows} block {(row % rows) // BLOCK} "
        f"row-in-block {(row % rows) % BLOCK} col {col}; |ref| rms {ref[~torch.isnan(ref)].pow(2).mean().sqrt():.4f}"
    )
    per_shard = {s: _pcc(ref[0, :, s * rows : (s + 1) * rows], out[0, :, s * rows : (s + 1) * rows]) for s in shards}
    logger.info(f"[{tag}] {name}: pcc per shard {({s: round(p, 5) for s, p in per_shard.items()})}")


@pytest.mark.parametrize("case_id", list(CASES))
def test_accuracy_floor(runtime, case_id):
    rows, dense_chunks, density = CASES[case_id]
    geo = Geometry(rows, rows * S, 256, dense_chunks=dense_chunks)
    case = Case(runtime, geo)
    mask = make_mask(geo, case.S, case.H, density=density, seed=21)
    q, k, v = case.inputs(21)
    r = case.run(q, k, v, mask, want_ref="slice")
    heads, shards = [0, case.H - 1], [0, case.S - 1]
    n = geo.logical_n
    dref = dense_reference(q, k, v, n, case.scale, heads, shards, rows)[:, :, :n]
    logger.info(
        f"[{case_id}] block density {_density(mask, geo, case.S):.3f}: sparse vs block-sparse ref "
        f"{_pcc(r['ref'], r['sparse']):.5f}; dense vs exact ref {_pcc(dref, r['dense']):.5f}; sparse vs exact ref "
        f"{_pcc(dref, r['sparse']):.5f}; sparse vs dense {_pcc(r['dense'], r['sparse']):.5f}"
    )
    summarize(case_id, "sparse vs block-sparse ref", r["ref"], r["sparse"], heads, shards, rows, n)
    summarize(case_id, "dense vs exact ref", dref, r["dense"], heads, shards, rows, n)
