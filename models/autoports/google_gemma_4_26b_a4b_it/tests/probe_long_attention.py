# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-input localization of full-context paged decode attention operations.

This diagnostic uses the model's actual head geometry and page contract. CPU
readbacks are intentional instrumentation outside a traced model invocation.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import Gemma4TextConfig

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.precise_attention import PrecisePagedAttention
from models.demos.gemma4.tt.attention import Gemma4AttentionConfig
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


def compare(expected, actual):
    shape = list(actual.shape)
    exact = torch.equal(expected, actual)
    stride = max(1, actual.numel() // 1048576)
    expected, actual = expected.flatten()[::stride], actual.flatten()[::stride]
    expected, actual = expected.float(), actual.float()
    delta = actual - expected
    left, right = expected.flatten().double(), actual.flatten().double()
    left, right = left - left.mean(), right - right.mean()
    denominator = left.norm() * right.norm()
    return dict(
        shape=shape,
        exact=exact,
        statistics_stride=stride,
        pcc=float((left @ right) / denominator) if denominator else None,
        max_error=float(delta.abs().max()),
        rms_error=float(delta.square().mean().sqrt()),
        expected_rms=float(expected.square().mean().sqrt()),
        actual_rms=float(actual.square().mean().sqrt()),
        finite=bool(actual.isfinite().all()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=262144)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--native-full-gather", action="store_true")
    parser.add_argument("--identity-full-gather", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(123)
    torch.set_num_threads(8)
    raw = json.loads(Path(__file__).with_name("config.json").read_text())
    hf = Gemma4TextConfig(**raw.get("text_config", raw))
    cfg = Gemma4AttentionConfig(Gemma4ModelArgs.from_hf_config(hf), args.layer)
    assert not cfg.is_sliding, "This diagnostic compares full attention over every page"
    block = 32
    assert args.length % block == 0
    pages = args.length // block
    table = torch.randperm(pages).int()[None]
    caches = [torch.randn(pages, cfg.num_key_value_heads, block, cfg.head_dim).bfloat16() for _ in range(2)]
    query = torch.randn(1, 1, cfg.num_attention_heads, cfg.head_dim) / cfg.head_dim**0.5

    def logical(cache):
        return cache[table[0].long()].permute(1, 0, 2, 3).reshape(1, cfg.num_key_value_heads, args.length, cfg.head_dim)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(
        length=args.length,
        layer=args.layer,
        native_full_gather=args.native_full_gather,
        identity_full_gather=args.identity_full_gather,
        stages=[],
    )
    originals = {}

    def read(value):
        return ttnn.to_torch(value)

    def record(name, expected, actual, **extra):
        item = dict(operation=name, **compare(expected, actual), **extra)
        report["stages"].append(item)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(item, flush=True)

    def wrap(name, cpu):
        original = getattr(ttnn, name)
        originals[name] = original

        def observed(*inputs, **kwargs):
            output = original(*inputs, **kwargs)
            expected = cpu(*inputs, **kwargs)
            actual = read(output)
            record(name, expected, actual)
            if name == "gather":
                if args.identity_full_gather:
                    assert not cfg.is_sliding and torch.equal(read(kwargs["index"]).long(), torch.arange(pages)[None])
                    return inputs[0]
                assert torch.equal(expected, actual), "Stop before invalid page addresses reach cache embedding"
            return output

        setattr(ttnn, name, observed)

    try:

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout)

        operation = PrecisePagedAttention(mesh, cfg, hf.max_position_embeddings)
        if args.native_full_gather:
            identity_ids = device(torch.arange(pages, dtype=torch.int32)[None], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            cache_rows = operation.cache_row_indices

            def native_cache_rows(physical):
                physical = ttnn.gather(physical, dim=1, index=identity_ids)
                return cache_rows(physical)

            operation.cache_row_indices = native_cache_rows
        q = device(query, ttnn.float32)
        k, v = [device(cache) for cache in caches]
        pt = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = device(torch.tensor([args.length - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def gather_cpu(value, *, dim, index, **kwargs):
            ids = read(index).long()
            record("page_ids", torch.arange(pages)[None], ids)
            return torch.gather(read(value), dim, ids)

        def embedding_cpu(index, weights, **kwargs):
            rows = read(index).long()
            expected_rows = (
                table.long()[..., None] * (cfg.num_key_value_heads * block)
                + torch.arange(cfg.num_key_value_heads * block)
            ).reshape(1, -1)
            record("cache_rows", expected_rows, rows)
            return read(weights)[rows]

        matrix_index = 0

        def matmul_cpu(a, b, **kwargs):
            nonlocal matrix_index
            left, right = read(a).float(), read(b).float()
            head = matrix_index // 2
            cache_index = matrix_index % 2
            record(
                "logical_key" if cache_index == 0 else "logical_value",
                logical(caches[cache_index])[:, head : head + 1],
                right,
            )
            matrix_index += 1
            if kwargs.get("transpose_a"):
                left = left.transpose(-2, -1)
            if kwargs.get("transpose_b"):
                right = right.transpose(-2, -1)
            return left @ right

        wrap("gather", gather_cpu)
        wrap("embedding", embedding_cpu)
        wrap("matmul", matmul_cpu)
        wrap("max", lambda value, **kw: torch.amax(read(value), dim=kw["dim"], keepdim=kw["keepdim"]))
        wrap("sum", lambda value, **kw: torch.sum(read(value), dim=kw["dim"], keepdim=kw["keepdim"]))
        wrap("div", lambda a, b, **kw: read(a) / read(b))
        wrap("exp", lambda value, **kw: torch.exp(read(value)))
        wrap("where", lambda condition, a, b, **kw: torch.where(read(condition).bool(), read(a), b))
        output = operation(q, k, v, cur_pos_tensor=pos, page_table_tensor=pt)
        actual = read(output)
        for name, original in originals.items():
            setattr(ttnn, name, original)
        originals.clear()

        keys, values = [logical(cache).float() for cache in caches]
        group = cfg.num_key_value_groups
        expected = torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    query[:, :, head * group : (head + 1) * group].transpose(1, 2),
                    keys[:, head : head + 1],
                    values[:, head : head + 1],
                    scale=1.0,
                ).transpose(1, 2)
                for head in range(cfg.num_key_value_heads)
            ],
            dim=2,
        )
        record("attention_end_to_end", expected, actual)
    finally:
        for name, original in originals.items():
            setattr(ttnn, name, original)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
