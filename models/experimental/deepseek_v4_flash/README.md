# DeepSeek-V4-Flash Decode Demo

Autoregressive decode demo for DeepSeek-V4-Flash on Tenstorrent hardware via TT-NN.
It builds the full ttnn `DeepSeekV4Model`, “prefills” a chat prompt by replaying
decode once per prompt token (seeding sliding K/V + compressor caches), then
generates new tokens greedily (`S = 1`). All weights are loaded on device as
`bfloat4_b`.

Entry point: [`tests/test_full_model_decode_demo.py`](tests/test_full_model_decode_demo.py).

## Prerequisites

- Cloned [tt-metal](https://github.com/tenstorrent/tt-metal) and a working
  TT-Metalium / TT-NN install — see [`INSTALLING.md`](../../../INSTALLING.md)
- Python venv activated:
  ```bash
  ./create_venv.sh
  source python_env/bin/activate
  ```
- Tenstorrent device visible (`tt-smi`)
- The demo opens the mesh with `FABRIC_2D` and `num_command_queues=2`

## Download weights

Download the checkpoint from Hugging Face:

**https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark**

```bash
# Example: Hugging Face CLI into the default cache layout the demo expects
hf download deepseek-ai/DeepSeek-V4-Flash-DSpark
```

The demo looks for weights under:

```text
~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark
```

(HF cache `snapshots/<hash>/` layout is resolved automatically.) If the
checkpoint is missing, the test is skipped.

## Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEEK_V4_DECODE_LAYERS` | all layers (43) | Cap layer count. The full bf4 stack does not fit a single Blackhole 32 GB; start with a small `N` (e.g. `4`) for bringup. |
| `DEEPSEEK_V4_CACHE_DIR` | `../cache` | Directory for converted ttnn weight tiles. Reuse across runs to avoid redoing the slow bf4 conversion. |
| `DEEPSEEK_V4_MAX_NEW_TOKENS` | `1024` | Max tokens to generate after the prompt. |
| `DEEPSEEK_V4_TRACED_DECODE` | `1` (on) | Set to `0` / `false` to use eager host-bound decode instead of captured ttnn traces. |
| `DEEPSEEK_V4_POOL_EVERY_STEP` | `0` (off) | Re-pool the CSA/HCA compressors on *every* decode step instead of only on the steps that close a window. Slower, and only useful as an A/B reference — the two are bit-identical. See below. |
| `DEEPSEEK_V4_SDPA_CAUSAL` | `1` (on) | Bound CSA/HCA attention with a causal `cur_pos` instead of an additive mask once the sliding ring is full. Set to `0` to force the mask everywhere. See below. |

### Compressor pooling schedule

A CSA/HCA compressor emits a new compressed entry once every `compress_rate`
tokens (CSA 4, HCA 128), and the additive block-bias only ever exposes entries
`w < (pos+1)//compress_rate` — a value that does not change between two window
closures. The pool itself runs over the whole fixed `max_seq`-sized buffer, so
its cost scales with `max_seq` rather than with the current position. Running it
only on the steps that close a window is therefore bit-identical and
`compress_rate` times cheaper. Measured on 4 layers, 2048 generated tokens:
71.0 tok/s (`DEEPSEEK_V4_POOL_EVERY_STEP=1`) vs 84.0 tok/s, same token stream.

A ttnn trace is a flat op sequence and cannot branch on the device-side
position, so the traced path bakes the schedule into the capture: one trace
variant per *window phase*, picked on host per step. Because the rates divide
one another (4 | 128), an HCA closure always coincides with a CSA closure, so
there are three phases — pool nothing / CSA / CSA+HCA — not four. Variants are
deduplicated per submesh, so a submesh whose layers are all `sliding_attention`
is still captured exactly once.

**This costs trace memory**: a submesh hosting both a CSA and an HCA layer now
holds three traces instead of one. If capture fails or hangs on the full
43-layer stack, raise `trace_region_size` in the test's `device_params`, or set
`DEEPSEEK_V4_POOL_EVERY_STEP=1` to collapse back to a single trace.

### Causal SDPA instead of an additive mask

A CSA/HCA layer's KV axis is `[sliding 0..127 | compressor 0..max_seq/cr)`, and
the valid set at position `pos` is sliding slot `i <= pos` plus compressor window
`j < (pos+1)//cr`. Once the ring is full (`pos + 1 >= sliding_window`) every
sliding slot is valid, so the union is the contiguous prefix
`[0, sliding_window + (pos+1)//cr)` — which a single SDPA-decode `cur_pos`
describes exactly. That matters because an additive mask is *data*, not control
flow: the kernel sets `cur_pos` to the end of the buffer in non-causal mode and
walks every chunk regardless. In causal mode it derives its chunk range from the
position and skips the rest, so attention cost tracks the actual position rather
than `max_seq`. It also drops the per-step per-layer head-broadcast of the mask
row. The kernel generates a partial mask for the final chunk, so this is exact
even mid-chunk, not rounded to `k_chunk_size`.

Note the `-1` in `sliding_window + (pos+1)//cr - 1`: `(pos+1)//cr` is the *count*
of closed windows and `cur_pos` is inclusive. Using `sliding_window + pos//cr`
instead agrees only at window boundaries and otherwise exposes the still-open
window, which is a silent accuracy loss rather than an error.

Below the sliding window the valid set has a hole (slots `pos+1 .. 127` are
unwritten), which no single `cur_pos` can express, so those steps keep the mask.
The op rejects an `attn_mask` in causal mode, so the two are exclusive branches
and the traced path needs a variant per (SDPA mode, window phase) pair: five with
the default rates (three causal phases plus the two phases reachable below the
window — the HCA closure first lands at `pos == 127`, i.e. exactly at the switch,
so `CSA+HCA` is causal-only). Same trace-memory caveat as above;
`DEEPSEEK_V4_SDPA_CAUSAL=0` collapses it back to one family.

Measured on the SDPA-decode op alone at the model's shapes (64 heads,
`head_dim` 512, MQA, `k_chunk_size` 32) with the valid extent held at 136:

| `Skv` | implied `max_seq` | masked | causal | speedup |
| --- | --- | --- | --- | --- |
| 640 | 2048 | 187 µs | 187 µs | 1.00x |
| 2176 | 8192 | 216 µs | 197 µs | 1.10x |
| 8320 | 32768 | 722 µs | 198 µs | 3.65x |

Causal is flat in `Skv` (work follows the position); masked grows with the
buffer. **End-to-end this is currently a wash**, and it is worth knowing why: at
`max_seq` 8192 the per-op saving is only ~20 µs, and on a 4-layer stack (2 of
which are compressor layers) that is well under 1% of a ~15 ms step. Measured
69-76 tok/s in both modes at `max_seq` 8192, and 62-70 tok/s in both at 12288.
Reaching the 3.65x regime needs `max_seq` ~32k, which is blocked by the
compressor pooling path — see below. So treat the causal path as removing the
`max_seq` sensitivity in attention, a prerequisite for long contexts, rather than
as a speedup at today's runnable sizes.

### What still limits `max_seq`

`DeepSeekV4*Compressor._pool` recomputes *every* compressed window from the whole
fixed cache capacity each time it fires, rather than pooling only the window that
just closed. So its output is `max_seq // compress_rate` rows tall, and several
per-row-scaled assumptions downstream break as `max_seq` grows:

- `DeepSeekV4RMSNorm(sharded=True)` width-shards into L1 giving every core the
  *full* height, so L1 use grows with the row count: ~2.8 MB against a 1.5 MB
  budget by `max_seq` 32k. Now fixed — the shard is only applied while the tensor
  is a single tile-row (the single-token decode activations it was written for);
  taller tensors take the interleaved path, which is also measurably faster past
  one tile-row (58 µs vs 88 µs at 512 rows).
- `_apply_rope` shards one tile-row per core, so it caps at
  `110 cores x 32 = 3520` rows — `fused_partial_rope` requires exactly one
  tile-row per core, so this is an op-level contract, not a config choice. With
  CSA's `compress_rate` 4 that puts the ceiling at `max_seq` ~14k
  (`Target number of cores 257 is greater than total number of available cores
  110`).

The durable fix for all of these is incremental pooling: compute just the newly
closed window, RoPE that single row, and write it at index `(pos+1)//cr - 1`.
That is `O(1)` per closure instead of `O(max_seq)`, matches the HF reference's
`torch.cat` semantics, and makes both row-count limits above disappear.

## Run the demo

From the tt-metal repo root, with the venv active:

```
pytest -s models/experimental/deepseek_v4_flash/tests/test_full_model_decode_demo.py
```

## Notes

- The first run converts every expert weight to `bfloat4_b` and can take a long
  time; set `DEEPSEEK_V4_CACHE_DIR` so later runs reuse the tile cache.
- Decode throughput is logged every 10 generated tokens (and once at the end).
- Generation stops on EOS or when the precomputed RoPE span is exhausted.
- Status: experimental bringup path — layer-capped runs are the supported way
  to exercise the demo on limited DRAM.
