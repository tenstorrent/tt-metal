# Serving the resident Qwen3.6 / 3.8-27B through vLLM

One user on one 1 x 4 Blackhole mesh (QuietBox 2: 2x P300c), vLLM 0.26.0 with vllm-tt-plugin (upstream
da7c018). The adapter is `qwen36_resident/generator_vllm.py`; the plugin finds it through the bundle in
`qwen36_resident/vllm_models` (`EXTRA_MODELS_DIR`).

## Launch

```bash
HF_MODEL=<checkpoint snapshot dir> VLLM_VENV=<venv with vllm + vllm-tt-plugin> \
  qwen36_resident/serve/serve_local.sh plain     # or: mtp
```

`HF_MODEL` should be the local snapshot directory: with `HF_HUB_OFFLINE=1` a repo id resolves only when the cache
holds `refs/main`. The served model name is `Qwen/Qwen3.8-27B` (`SERVED_MODEL_NAME`). Building the model takes
about 6 minutes before the server answers. Extra arguments go to `vllm serve`.

| profile | model class | sampling |
| --- | --- | --- |
| plain | `Qwen36ResidentForCausalLM` | any; greedy and top_k <= 32 drawn on device (full speed), others on the host |
| mtp | `Qwen36ResidentMTPForCausalLM` (`TT_MODEL_CLASS_OVERRIDES`) | greedy only (HTTP sampling controls are ignored); 4 tokens per step |

Both: `--max-model-len 32768 --max-num-seqs 1 --block-size 64 --async-scheduling`, additional config
`{"tt": {"trace_region_size": 134217728, "sample_on_device_mode": "decode_only"}}`, and
`TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0` (without it the large host -> device uploads can stall for 10-40 min in
the driver's `ioctl_pin_pages`).

## Checks

1. `python -m qwen36_resident.serve.greedy_ref --out OUT/greedy_ref.json` (no vLLM, device to itself): greedy text
   of the prompts in `prompts.py`, the argmax of each step's logits, and the on-device greedy feed beside it.
2. `python -m qwen36_resident.serve.mtp_ref --out OUT/mtp_ref.json --plain OUT/greedy_ref.json`: the MTP model's
   verify-step greedy and its speculative decode.
3. Start a server, then `qwen36_resident/serve/run_checks.sh plain|mtp OUT` (parity and speed; see the script).
   Run the plain profile first: the mtp checks compare with its served text.

`chip_telemetry.sh` logs every chip's AICLK, temperature and power from sysfs beside a running server.

## What the checks showed (2026-10-29, Qwen3.8-27B, QuietBox 2)

- plain, greedy: served tokens equal the direct greedy decode, 5 prompts x 128 tokens, token for token.
- mtp: served tokens equal the MTP verify-step greedy (5 x 128); they differ from the plain profile's greedy
  text where its top two logits are within one bf16 step (4 of 5 prompts, gaps 0.0625-0.125).
- one user, 128 generated tokens: TPOT 11.55 / 11.62 / 11.85 / 13.08 ms at 128 / 2048 / 8192 / 32000 prompt tokens,
  TTFT 0.24 / 0.54 / 1.24 / 4.77 s; mtp 107-132 tok/s on chat prompts.
- sustained decode (about 80 s back to back) drops one chip's AICLK from 1350 to 800-1100 MHz and the decode to
  60-75 tok/s; it recovers as soon as the load stops.
