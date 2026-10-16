# pplx-embed-v1-4B model-local custom ops

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

- QKV: `head_groups = num_kv_heads = 8` → **128 units**, each moving 16 Q + 4 K + 4 V tiles.
- Concat: `head_groups = num_heads = 32` → **512 units**, each moving 4 tiles.

Override either with `QWEN_HEADSPLIT_GROUPS_{QKV,CONCAT}`. Swept at bs=32 and
flat (558.0-561.0 ms across 8/32, 4/16, 2/8, 8/8, 1/4, 4/32), so the defaults
are already at the optimum for this shape.

Both are pure tile-copy reorders, so output is **bit-identical** to the stock
ops — only the dispatch pattern changes.

## Geometry

pplx-embed-v1-4B is GQA: **32 Q heads over 8 KV heads** (4 Q per KV),
`head_dim` 128 (4 tiles), `dim` 2560, 36 layers. The fused QKV row is
`(32 + 2*8) * 128 = 6144` wide = 192 tiles.

Note this differs from the 0.6B sibling (16 Q / 8 KV, `dim` 1024), which is why
the concat head-split is a win here and a small loss there: 32 heads at 4 tiles
each give it enough work per unit. Measured bs=1: QKV split -1.5 ms, concat
split a further -1.2 ms; on 0.6B the concat split cost +0.2 ms and is default
off.

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

## Measured and rejected

**Rotary DEST_TO_SRCA fusion.** Not ported. The pre-rebase branch carried a
rewrite of `rotary_embedding_llama`'s compute kernel collapsing its 4 ACQ/REL
phases into 2 by reusing DST, removing two intermediate CB round-trips.
Upstream has since rewritten that kernel onto `ckl::eltwise_chain`, which does
expose `ckl::DestReuseBinary<..., DEST_TO_SRCA>`, so the fusion is
expressible — but re-homing it here means reimplementing the whole op as a
`generic_op` (reader, compute, writer, ~8 CBs), because the program factory
selects the compute kernel by fixed path. Unlike the head-splits it is
unconditional, so a mis-port corrupts every run rather than a flag-gated path.

Size, from the profiler rather than a stub: rotary is **1.66 ms of 25.9 ms at
bs=1 (6.7%)** and **28.0 ms of ~502 ms at bs=32 (5.6%)**. The fusion removes
two of four CB round-trips, so realistically ~0.5 ms at bs=1 and ~10 ms at
bs=32.

An earlier note here claimed stubbing the op out was worth only ~0.2-0.25 ms
and used that to dismiss the fusion. That measurement was wrong: the stub
returned `ttnn.clone(input)`, a full tensor copy, so the delta measured
*rotary minus clone*, not rotary. Use the profiler figures above.

## Conventions

- Kernel paths in `op.py` are relative to `TT_METAL_HOME`.
- `io_tensors` order in the `generic_op` call binds buffer addresses to the
  kernels' `TensorAccessor` compile-time args — inputs first, then outputs.
- Kernels use the Device 2.0 data-movement API (`Noc`, `CircularBuffer`,
  `TensorAccessor`) with positional `get_arg_val` / `get_compile_time_arg_val`,
  which is what the `generic_op` path expects.
