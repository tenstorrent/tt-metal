# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Is ``rotary_embedding_hf(is_decode_mode=True)`` the same rotation as the
shipped ``ttnn.experimental.rotary_embedding(..., token_index)``?

This is the one op swap stage 05 makes inside the decoder layer, and it is made
for a *correctness* reason rather than a speed one: the shipped spelling takes
the position as a Python int compile-time argument, so a captured decode trace
rotates every later token at the position it was captured at. The replacement
reads a per-user cos/sin pair gathered on device from a position **tensor**.

Both are HF ``rotate_half``, so if they agree here the KV-cache channel
convention, prefill and every weight are untouched -- which is exactly what
stage 04's rejected ``rotary_embedding_llama`` lever could not offer. To be
precise about *why* that lever was rejected: not traceability. Its nanobind
signature (``rotary_embedding_llama_nanobind.cpp:38-44``) takes tensors only and
no position argument, so it is perfectly trace-replayable; stage 04's wiring
hoisted the cos/sin gather onto the first eager call, which is what baked the
position in there. It was rejected on **channel convention** -- PCC 0.1933
against a prefill-primed KV cache versus 0.99997 against a fresh one, plus a
bfloat8_b requantisation ``max|diff|`` of 3.125e-01.

Coverage. Decode is where ``rotary_embedding_hf`` earns its place, and its
distinguishing feature is a *per-user* position: each row of the ``[1, batch]``
index tensor gathers its own cos/sin row, which a Python-int ``token_index``
cannot express at all. So the sweep runs several batch sizes with **distinct
positions per user**, not just ``BATCH = 1``, and reaches into the advertised
262144-token context rather than stopping at 4095.

Prints max|diff| and PCC per position, and the trace-slope cost of each.
"""

from __future__ import annotations

import sys
import time

import torch

import ttnn

sys.path.insert(0, ".")
from models.common.utility_functions import comp_pcc  # noqa: E402

HEAD_DIM = 128
N_HEADS = 8  # per die, TP=4
N_KV = 1
BATCH = 1  # the batch the published trace slopes are measured at
#: Batch sizes swept for correctness. 32 is the model's hard ceiling
#: (``nlp_create_qkv_heads_decode_device_operation.cpp:51``).
BATCHES = [1, 4, 8, 32]
#: Reaches 262143 -- the last position of the advertised context -- rather than
#: stopping at 4095. Includes both sides of the 8192 default rope-cache length.
POSITIONS = [0, 1, 31, 32, 33, 127, 1000, 4095, 8191, 8192, 65535, 131071, 262143]
CAPACITY = 262144
THETA = 1e7


def head_shard(rows, cols, batch):
    gx = min(batch, 8)
    while batch % gx:
        gx -= 1
    gy = batch // gx
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))}),
            [rows, cols],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def rope_tables(capacity):
    inv = 1.0 / (THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM))
    ang = torch.outer(torch.arange(capacity, dtype=torch.float64), inv)
    ang = torch.cat([ang, ang], dim=-1)
    return ang.cos().float(), ang.sin().float()


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=20_000_000)
    try:
        cos_t, sin_t = rope_tables(CAPACITY)
        cos_dev = ttnn.from_torch(
            cos_t.reshape(1, 1, CAPACITY, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        sin_dev = ttnn.from_torch(
            sin_t.reshape(1, 1, CAPACITY, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        def gather_pair(positions, batch):
            """The shipped decode gather: one cos/sin row per user, on device."""
            idx = ttnn.from_torch(
                torch.tensor(positions, dtype=torch.int32).reshape(1, batch),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
            )
            shard = head_shard(32, HEAD_DIM, batch)
            pair = []
            for table in (cos_dev, sin_dev):
                g = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT)
                g = ttnn.unsqueeze_to_4D(g)
                g = ttnn.transpose(g, 1, 2)
                g = g[:, :batch, :, :]
                pair.append(ttnn.interleaved_to_sharded(g, shard))
            return pair, shard

        torch.manual_seed(0)
        for n_heads in (N_HEADS, N_KV):
            print(f"\n=== n_heads={n_heads} ===")
            host = torch.randn(1, BATCH, n_heads, HEAD_DIM) * 0.5
            for pos in POSITIONS:
                # --- shipped spelling: int token_index, DRAM interleaved
                x = ttnn.from_torch(
                    host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ref = ttnn.experimental.rotary_embedding(x, cos_dev, sin_dev, pos)
                ref_t = ttnn.to_torch(ref)[:, :, :n_heads, :].float()

                # --- stage-05 spelling: device position gather + rotary_embedding_hf
                pair, shard = gather_pair([pos] * BATCH, BATCH)
                xs = ttnn.to_memory_config(x, shard)
                out = ttnn.experimental.rotary_embedding_hf(xs, pair[0], pair[1], is_decode_mode=True)
                out_t = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))[:, :, :n_heads, :].float()

                diff = (ref_t - out_t).abs().max().item()
                ok, pcc = comp_pcc(ref_t, out_t, 0.999)
                print(f"  pos {pos:>6}: max|diff|={diff:.3e}  {pcc}")

        # --- per-user distinct positions, which is the whole point of the swap
        #
        # ``rotary_embedding(..., token_index)`` takes ONE Python int for the
        # whole tensor, so a batch of users at different decode positions is not
        # expressible in it at all. The reference here is therefore the shipped
        # op run once per user at that user's own position and stitched back
        # together; the stage-05 op does the entire batch in one call from a
        # ``[1, batch]`` index tensor.
        for n_heads in (N_HEADS, N_KV):
            print(f"\n=== per-user positions, n_heads={n_heads} ===")
            for batch in BATCHES:
                positions = [POSITIONS[i % len(POSITIONS)] for i in range(batch)]
                host = torch.randn(1, batch, n_heads, HEAD_DIM) * 0.5

                rows = []
                for user, pos in enumerate(positions):
                    xu = ttnn.from_torch(
                        host[:, user : user + 1],
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    rows.append(ttnn.to_torch(ttnn.experimental.rotary_embedding(xu, cos_dev, sin_dev, pos)))
                ref_t = torch.cat(rows, dim=1)[:, :, :n_heads, :].float()

                x = ttnn.from_torch(
                    host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                pair, shard = gather_pair(positions, batch)
                xs = ttnn.to_memory_config(x, shard)
                out = ttnn.experimental.rotary_embedding_hf(xs, pair[0], pair[1], is_decode_mode=True)
                out_t = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))[:, :, :n_heads, :].float()

                diff = (ref_t - out_t).abs().max().item()
                ok, pcc = comp_pcc(ref_t, out_t, 0.999)
                distinct = len(set(positions))
                print(
                    f"  batch {batch:>3} ({distinct:>2} distinct positions, "
                    f"max {max(positions)}): max|diff|={diff:.3e}  {pcc}"
                )

        # --- trace slope: cost of each spelling at the shipped decode shape
        host = torch.randn(1, BATCH, N_HEADS, HEAD_DIM) * 0.5
        x = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        idx = ttnn.from_torch(
            torch.zeros((1, BATCH), dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        shard = head_shard(32, HEAD_DIM, BATCH)

        def shipped():
            return ttnn.experimental.rotary_embedding(x, cos_dev, sin_dev, 5)

        def gather():
            pair = []
            for table in (cos_dev, sin_dev):
                g = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT)
                g = ttnn.unsqueeze_to_4D(g)
                g = ttnn.transpose(g, 1, 2)
                g = g[:, :BATCH, :, :]
                pair.append(ttnn.interleaved_to_sharded(g, shard))
            return pair

        def staged(pair):
            xs = ttnn.to_memory_config(x, shard)
            return ttnn.experimental.rotary_embedding_hf(xs, pair[0], pair[1], is_decode_mode=True)

        pair = gather()
        for name, fn in (
            ("rotary_embedding (int pos)", shipped),
            ("rotary_embedding_hf + reshard", lambda: staged(pair)),
        ):
            fn()
            ttnn.synchronize_device(device)
            reps = 50
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(reps):
                fn()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            dt = (time.perf_counter() - t0) / reps * 1e6
            ttnn.release_trace(device, tid)
            print(f"trace slope {name}: {dt:.2f} us")

        # The gather cost scales with the cos/sin table length, because
        # ``ttnn.embedding`` indexes into the whole table -- so it is reported at
        # both the shipped default (``rope_cache_len`` 8192) and at the full
        # advertised context, which is what ``ensure_rope_capacity`` grows to.
        print("\ncos/sin device gather (2x embedding+transpose+slice+i2s), trace slope:")
        for label, capacity in (("rope_cache_len 8192", 8192), (f"rope_cache_len {CAPACITY}", CAPACITY)):
            if capacity == CAPACITY:
                tables = (cos_dev, sin_dev)
            else:
                c, s = rope_tables(capacity)
                tables = tuple(
                    ttnn.from_torch(
                        t.reshape(1, 1, capacity, HEAD_DIM),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    for t in (c, s)
                )

            def gather_at(tables=tables):
                pair = []
                for table in tables:
                    g = ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT)
                    g = ttnn.unsqueeze_to_4D(g)
                    g = ttnn.transpose(g, 1, 2)
                    g = g[:, :BATCH, :, :]
                    pair.append(ttnn.interleaved_to_sharded(g, shard))
                return pair

            gather_at()
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(50):
                gather_at()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            print(f"  {label}: {(time.perf_counter() - t0) / 50 * 1e6:.2f} us")
            ttnn.release_trace(device, tid)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
