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
