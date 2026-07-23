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
