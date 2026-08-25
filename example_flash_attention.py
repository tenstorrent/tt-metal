# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Runs unified_kernels/example_flash_attention.cpp: one attention head, one core.

64 queries against 256 keys, streamed in 4 chunks. The scores are never all resident:
each chunk updates a running maximum, a running denominator and a running output, and
the last two are corrected by exp(old_max - new_max) whenever the maximum moves. That
is the online softmax, and the running state lives in circular buffers.

Queries are pre-scaled by 1/sqrt(head_dim) and the keys are stored transposed, so the
kernel is a plain matmul in both places.

The second case is the one that matters: it makes a late chunk's keys 20x larger, so the
running maximum jumps near the end and every partial result before it has to be rescaled.
Uniform random keys barely move the maximum and would pass even with the correction gone.

    export TT_METAL_HOME=$PWD
    source python_env/bin/activate
    python example_flash_attention.py
"""

import sys

import torch
from loguru import logger

import ttnn
from unified_harness import make_cb, single_core, unified_program

KERNEL = "unified_kernels/example_flash_attention.cpp"
TILE = 32
KEY_CHUNKS = 4
SQ, SK, DT = 2, 2, 2

N_QUERIES = SQ * TILE
N_KEYS = KEY_CHUNKS * SK * TILE
HEAD_DIM = DT * TILE

CB_Q, CB_K, CB_V, CB_ONES, CB_SCALER = 0, 1, 2, 3, 4
CB_SCORES, CB_CHUNK_MAX, CB_PROB, CB_CHUNK_SUM = 5, 6, 7, 8
CB_NEW_MAX, CB_CORRECTION, CB_RESCALED, CB_WEIGHTED_V, CB_RECIPROCAL = 9, 10, 11, 12, 13
CB_OUT, CB_MAX, CB_SUM, CB_ACC = 16, 20, 21, 22

CB_PAGES = {
    CB_Q: SQ * DT,
    CB_K: DT * SK,
    CB_V: SK * DT,
    CB_ONES: SK,
    CB_SCALER: 1,
    CB_SCORES: SQ * SK,
    CB_CHUNK_MAX: SQ,
    CB_PROB: SQ * SK,
    CB_CHUNK_SUM: SQ,
    CB_NEW_MAX: SQ,
    CB_CORRECTION: SQ,
    CB_RESCALED: SQ * DT,
    CB_WEIGHTED_V: SQ * DT,
    CB_RECIPROCAL: SQ,
    CB_OUT: SQ * DT,
    CB_MAX: 2 * SQ,
    CB_SUM: 2 * SQ,
    CB_ACC: 2 * SQ * DT,
}


def to_device(device, t):
    return ttnn.from_torch(
        t.reshape(1, 1, *t.shape).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def attention(device, q, k, v):
    q_scaled = (q.to(torch.float32) / HEAD_DIM**0.5).to(torch.bfloat16)
    k_transposed = torch.cat([k[j * SK * TILE : (j + 1) * SK * TILE].T for j in range(KEY_CHUNKS)])
    ones = torch.ones([SK * TILE, TILE], dtype=torch.bfloat16)

    tq = to_device(device, q_scaled)
    tk = to_device(device, k_transposed)
    tv = to_device(device, v)
    tones = to_device(device, ones)
    tout = to_device(device, torch.full([N_QUERIES, HEAD_DIM], float("nan")))
    tensors = (tq, tk, tv, tones, tout)

    core_ranges, cores = single_core()

    compile_time_args = []
    for t in tensors:
        compile_time_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    program = unified_program(
        kernel_source=KERNEL,
        core_ranges=core_ranges,
        cores=cores,
        cbs=[make_cb(index, core_ranges, num_pages=pages) for index, pages in CB_PAGES.items()],
        compile_time_args=compile_time_args,
        runtime_args=[t.buffer_address() for t in tensors],
    )

    out = ttnn.generic_op(list(tensors), program)
    return ttnn.to_torch(out).to(torch.float32)[0, 0]


def reference(q, k, v):
    qf, kf, vf = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    return torch.softmax(qf @ kf.T / HEAD_DIM**0.5, dim=-1) @ vf


def main():
    torch.manual_seed(0)
    q = (torch.rand([N_QUERIES, HEAD_DIM]) - 0.5).to(torch.bfloat16)
    k = (torch.rand([N_KEYS, HEAD_DIM]) - 0.5).to(torch.bfloat16)
    v = (torch.rand([N_KEYS, HEAD_DIM]) - 0.5).to(torch.bfloat16)

    dominant = k.clone()
    late = KEY_CHUNKS - 1
    rows = slice(late * SK * TILE, (late + 1) * SK * TILE)
    dominant[rows] = (dominant[rows].to(torch.float32) * 20.0).to(torch.bfloat16)

    device = ttnn.open_device(device_id=0)
    failed = []
    try:
        for label, keys in (("uniform keys", k), (f"chunk {late} keys 20x larger", dominant)):
            got = attention(device, q, keys, v)
            error = ((got - reference(q, keys, v)).norm() / reference(q, keys, v).norm()).item()
            ok = error <= 0.02
            logger.info(
                f"{N_QUERIES} queries, {N_KEYS} keys in {KEY_CHUNKS} chunks, {label}: "
                f"relative error = {error:.5f}   {'ok' if ok else 'FAIL'}"
            )
            if not ok:
                failed.append(label)
    finally:
        ttnn.close_device(device)

    if failed:
        logger.error(f"FAIL: {failed}")
        return 1
    logger.info("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
