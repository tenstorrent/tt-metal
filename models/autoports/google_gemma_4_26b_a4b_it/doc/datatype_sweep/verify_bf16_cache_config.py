# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-only call lowering/allocation probe; no TTNN import or device execution."""

import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[2] / "tt" / "optimized_decoder.py"


class Tensor:
    def __init__(self, shape, dtype="bfloat16"):
        self.shape = tuple(shape)
        self.padded_shape = self.shape
        self.dtype = dtype

    def deallocate(self, force):
        pass


records = []


def sdpa(query, k, v, page_table, **kwargs):
    config = kwargs["program_config"]
    q_chunk, k_chunk = config.q_chunk_size, config.k_chunk_size
    q_rows, head_dim = query.shape[-2:]
    q_tiles, k_tiles, dim_tiles = q_chunk // 32, k_chunk // 32, head_dim // 32
    chunks = q_rows // q_chunk
    cores = config.compute_with_storage_grid_size[0] * config.compute_with_storage_grid_size[1]
    total = query.shape[0] * query.shape[1] * chunks
    if chunks % 2 == 0:
        base = (total // 2 // cores) * 2
        extra = 2 if (total // 2) % cores else 0
    else:
        base = total // cores
        extra = bool(total % cores)
    q_buffers = 2 if base + extra > 1 else 1
    # Legacy causal, FP32 accumulation: sdpa_program_factory.cpp502-512,781-889.
    cache_tile_bytes = 2048 if k.dtype == "bfloat16" else 1088
    cb_bytes = (
        q_tiles * dim_tiles * q_buffers * 2048
        + 2 * (k_tiles * dim_tiles * 2 * cache_tile_bytes)
        + 2 * 2048
        + 2 * 2048
        + (page_table.shape[-1] * 4 + 63) // 64 * 64
        + q_tiles * k_tiles * 4096
        + 3 * q_tiles * dim_tiles * 2048
        + 3 * q_tiles * 2048
        + 2 * q_tiles * 4096
    )
    query_end = kwargs["chunk_start_idx"] + q_rows
    read_end = query_end + (-query_end) % k_chunk
    records.append(
        dict(
            q_chunk=q_chunk,
            k_chunk=k_chunk,
            q_rows=q_rows,
            base_offset=kwargs["chunk_start_idx"],
            q_buffers=q_buffers,
            cb_bytes=cb_bytes,
            cb_end=111616 + cb_bytes,
            read_end=read_end,
            capacity=page_table.shape[-1] * k.shape[-2],
        )
    )
    return Tensor(query.shape)


def concat(tensors, dim):
    shape = list(tensors[0].shape)
    shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return Tensor(shape)


ttnn = SimpleNamespace(
    bfloat16="bfloat16",
    SDPAProgramConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    PagedCacheGeometryOverride=lambda **kwargs: SimpleNamespace(**kwargs),
    slice=lambda tensor, start, end: Tensor([b - a for a, b in zip(start, end)], tensor.dtype),
    pad=lambda tensor, pairs, value: Tensor([d + a + b for d, (a, b) in zip(tensor.shape, pairs)], tensor.dtype),
    concat=concat,
    transformer=SimpleNamespace(chunked_scaled_dot_product_attention=sdpa),
)
module = ast.parse(SOURCE.read_text())
class_node = next(
    node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "ConfiguredChunkedPrefillAttention"
)
namespace = dict(
    os=os, ttnn=ttnn, PREFILL_CHUNK_SIZE=8192, effective_block_size=lambda cache, head_dim, heads: cache.shape[-2]
)
exec(compile(ast.Module(body=[class_node], type_ignores=[]), str(SOURCE), "exec"), namespace)
Attention = namespace["ConfiguredChunkedPrefillAttention"]
results = []
for dtype in ("bfloat16", "bfloat8_b"):
    for name, q_rows, offset, capacity, users in (
        ("short_tail_1025", 32, 1024, 1152, 1),
        ("full_second_chunk", 1024, 1024, 4224, 1),
        ("tail_4097", 32, 4096, 4224, 1),
        ("page_boundary_tail", 32, 4096, 4224, 1),
        ("page_boundary_aligned", 128, 4096, 4224, 1),
        ("max_context_final_chunk", 1024, 261120, 262144, 1),
        ("batched_page_table", 1024, 1024, 8192, 32),
        ("internal_multichunk", 16384, 8192, 32768, 1),
    ):
        records.clear()
        attention = Attention(512, "compute")
        cache = Tensor((capacity // 32 * users, 1, 32, 512), dtype)
        output = attention(
            Tensor((1, 4, q_rows, 512)),
            cache,
            cache,
            Tensor((users, capacity // 32), "int32"),
            users - 1,
            512,
            base_offset=offset,
            num_kv_heads=1,
        )
        assert output.shape == (1, 4, q_rows, 512)
        assert all(record["read_end"] <= record["capacity"] for record in records)
        results.append(dict(dtype=dtype, case=name, calls=list(records)))

path = Path(sys.argv[1])
path.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
if len(sys.argv) > 2:
    baseline = json.loads(Path(sys.argv[2]).read_text())
    assert [x for x in results if x["dtype"] == "bfloat8_b"] == [x for x in baseline if x["dtype"] == "bfloat8_b"]
    assert all(call["cb_end"] <= 1572864 for result in results for call in result["calls"])
    print("PASS: 16 exact-call config cases; all allocations and rounded reads fit; BFP8 records unchanged.")
else:
    failing = [
        x["case"] for x in results if x["dtype"] == "bfloat16" and any(c["cb_end"] > 1572864 for c in x["calls"])
    ]
    assert "full_second_chunk" in failing
    assert "short_tail_1025" not in failing
    print("EXPECTED BF16 allocation failures:", failing)
