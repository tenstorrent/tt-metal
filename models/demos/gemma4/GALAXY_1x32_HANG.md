# Gemma4 1x32 hang on BH Galaxy

## Issue

`ttnn.all_gather` silently falls back to `composite_all_gather` when a row-major
input page is not 64 B aligned. That composite path is `ttnn::all_broadcast` +
`concat`, and it **deadlocks at 32 devices**.

Gemma4-31B has `hidden_size = 5376`. The per-device shard is `5376 / TP`, and
whether the page happens to be 64 B aligned is what decides the code path:

| TP | shard width | row-major bf16 page | 64 B aligned | path | result |
|----|-------------|---------------------|--------------|------|--------|
| 4  | 1344 | 2688 B | yes (42x64) | native | works |
| 8  | 672  | 1344 B | yes (21x64) | native | works |
| 16 | 336  | 672 B  | no          | composite | would hang |
| 32 | 168  | 336 B  | no (5x64+16)| composite | **hangs** |

So 1x4 works and 1x32 hangs for a purely arithmetic reason. It is **not** a
topology problem: the 1x32 logical mesh is a snake over the physical 8x4 system
mesh, but an aligned all_gather completes fine on that exact mesh.

First collective to hit it is `Gemma4Model.embed_tokens()` in
[tt/model.py](tt/model.py) — column-parallel `ttnn.embedding` (ROW_MAJOR output)
followed by `ccl_allgather(dim=3)`. The model hangs there, in prefill warmup,
before the first layer runs.

Log line that identifies it:

```
Using slower composite all_gather: row-major input page (336 B) is not a multiple of the 64 B page alignment
```

## Reproducing

```bash
cd $TT_METAL_HOME   # tt-metal checkout
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
source python_env/bin/activate

PYTEST_TIMEOUT=1800 \
HF_HOME=/localdev/svuckovic/huggingface \
HF_MODEL=google/gemma-4-31B-it \
TT_CACHE_PATH=/localdev/svuckovic/huggingface/tt_cache/google--gemma-4-31B-it \
GEMMA4_BOUNDED_SLIDING=1 \
pytest "models/demos/gemma4/demo/text_demo.py::test_demo[blackhole-prefill_4096-1x32]"
```

Add these to abort in ~5 min instead of hanging until the pytest timeout:

```bash
export TT_METAL_OPERATION_TIMEOUT_SECONDS=300
```

The passing comparison is the same command with `1x8` in place of `1x32`.

**A hang wedges ethernet cores.** Reset before the next run, or you get
misleading `Timed out while waiting for active ethernet core` errors instead of
the real behaviour:

```bash
python_env/bin/tt-smi -r
```

## Fix

In `embed_tokens()`, convert to `TILE_LAYOUT` *before* `ccl_allgather` instead of
after. Tile pages are always aligned, so the native path is used. This is free —
every caller of `embed_tokens()` already does `ttnn.to_layout(..., TILE_LAYOUT)`
on the following line.

Caveat: this clears the collective that currently hangs. Other TP=32 collectives
could hit the same fallback; they have not all been checked.
