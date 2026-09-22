# pplx-embed-v1-0.6B model-local custom ops

`ttnn.generic_op` kernels plus Python wrappers that live **inside this demo
directory**. Nothing under `ttnn/` or `models/tt_transformers/` is modified, so
these optimizations survive upstream rebases.

This replaces an earlier in-tree approach that patched the
`nlp_create_qkv_heads` / `nlp_concat_heads` program factories directly. Those
hooks were lost when upstream migrated both ops to the Metal 2.0 descriptor API
(`KernelSpec` / `DFBBinding` / named compile-time args), deleting the
`CreateKernel` + `SetRuntimeArgs` path they attached to. The same head-split
idea was independently carried into
`models/demos/wormhole/bge_m3/tt/custom_ops` as model-local `generic_op`
kernels; this package follows that layout, with the kernels adapted back for
pplx-embed's GQA geometry.

## What's here

| Op | Replaces | Work split |
|---|---|---|
| `fused_qkv_heads` | `ttnn.experimental.nlp_create_qkv_heads` | `(batch, seq_tile, head_group)` |
| `fused_concat_heads` | `ttnn.experimental.nlp_concat_heads` | `(batch, seq_tile, head_group)` |

The stock ops split work by `(batch, seq_tile)` only. At bs=1 / ISL=512 that is
16 work units on a 130-core Blackhole grid — roughly 12% occupancy. Adding a
head-group axis raises this to `16 * head_groups` units:

- QKV: `head_groups = num_kv_heads = 8` → **128 units**, each moving 8 Q + 4 K + 4 V tiles.
- Concat: `head_groups = num_heads = 16` → **256 units**, each moving 4 tiles.

Both are pure tile-copy reorders, so output is **bit-identical** to the stock
ops — only the dispatch pattern changes.

## Geometry

pplx-embed-v1-0.6B is GQA: 16 Q heads over 8 KV heads, `head_dim` 128
(4 tiles). The fused QKV row is `(16 + 2*8) * 128 = 4096` wide = 128 tiles.

## How it's wired in

`tt/attention.py` swaps the two `ttnn.experimental` entry points for the
duration of the parent `forward_prefill`, then restores them — the same
technique the file already uses to force `is_causal=False` on SDPA. Gated on
the flags the demo already sets:

    QWEN_NLP_CREATE_HEADS_HEAD_SPLIT=1
    QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT=1

Each wrapper calls `supported(...)` first and falls back to the stock op when
the fast path can't express the call — sharded input or output, transposed K
heads, indivisible head counts, non-tile-aligned shapes. Leaving the flags on
is therefore always safe.

## Correctness

Validated bit-identical against the stock ops across bf16 and bfp8, DRAM and L1
outputs, and batched (`B=2`) as well as single-batch shapes — 16/16 tensors
exactly equal, PCC 1.0.

## Conventions

- Kernel paths in `op.py` are relative to `TT_METAL_HOME`.
- `io_tensors` order in the `generic_op` call binds buffer addresses to the
  kernels' `TensorAccessor` compile-time args — inputs first, then outputs.
- Kernels use the Device 2.0 data-movement API (`Noc`, `CircularBuffer`,
  `TensorAccessor`) with positional `get_arg_val` / `get_compile_time_arg_val`,
  which is what the `generic_op` path expects.
