# Qwen3.5 / Qwen3.6 / Qwen3.8 on Blackhole

End-to-end text demo: `demo/text_demo.py`.

## Setup

```bash
export HF_MODEL=Qwen/Qwen3.6-27B   # also: Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B, Qwen/Qwen3.6-35B-A3B, Qwen/Qwen3.8-27B
export MESH_DEVICE=P150x4          # P150 | P150x4 | P150x8
# optional
export HF_HUB_OFFLINE=1               # use the local HF cache only
```

| Device | `MESH_DEVICE` | Models |
| ------ | ------------- | ------ |
| 1x P150 | `P150` | Qwen3.5-9B |
| 4x P150 | `P150x4` | Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B |
| 8x P150 | `P150x8` | Qwen3.6-27B |

## Run the e2e demo

```bash
# Single P150 (9B): single-user only
MESH_DEVICE=P150 HF_MODEL=Qwen/Qwen3.5-9B \
  pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"

# P150x4 / P150x8: single user, any context length
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128"   # also traced_4k ... traced_64k, traced_128k, traced_256k
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced"       # all single-user lengths

# P150x4 / P150x8: batched
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "spec_128_b4"    # also spec_128_b2/b8, spec_4k_b4/b8, spec_8k_b8, spec_16k_b8
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "batched_128_b32" # also batched_*_b8/b32, batched_8k_b8 ... batched_64k_b8

# Determinism (run twice, outputs must match) and accuracy vs HF reference (top-1/top-5)
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "determinism_128"
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "accuracy_512"
```

Prompts of 8k and longer download a public-domain text on first run (cached in `demo/sample_prompts/.context_cache`).

## What runs

| Case | Device | Decode |
| ---- | ------ | ------ |
| `traced_*`, `determinism_128` | `P150` | plain decode (no speculation) |
| `traced_*`, `determinism_128` | `P150x4` / `P150x8` | MTP speculative decode, K=11 (prompt <= 4k) or K=7 (longer) |
| `spec_*_b2/b4/b8`, `batched_*_b8` | `P150x4` / `P150x8` | MTP speculative decode, K by batch: B=2 -> 11, B=4 -> 7, B=8 -> 3 |
| `batched_*_b32` (any B > 8) | `P150x4` / `P150x8` | plain batched decode |
| `accuracy_512` | any | plain decode (teacher forced) |

Plain decode is also used on any device when `QWEN36_SPEC=0`, the checkpoint has no MTP head, `QWEN35_REP_PENALTY` / `QWEN35_NO_REPEAT_NGRAM` is set, or `QWEN35_PRESENCE_PENALTY > 0` with greedy decoding. Greedy output matches plain decode except at near-ties (top-2 logits within bf16 rounding). Batched cases use one identical prompt for all users.

## Options

| Env var | Default | Effect |
| ------- | ------- | ------ |
| `QWEN36_SPEC` | `1` | `0` forces plain decode (baseline) |
| `QWEN36_SPEC_DRAFT_LEN` | auto (see above) | Override K (3, 7 or 11; batch * (K+1) must be <= 32) |
| `QWEN36_SPEC_BATCH_DISTINCT` | `0` | `1`: each batched user gets a different prompt (skips the identical-output check) |
| `QWEN35_TEMP` | `0` (greedy) | Temperature; > 0 samples (speculation stays on) |
| `QWEN35_TOP_K` / `QWEN35_TOP_P` | `0` / `1.0` | Top-k / top-p when sampling |
| `QWEN35_SEED` | random | Sampling seed for speculative decode |
| `QWEN35_PRESENCE_PENALTY` | `0` | Presence penalty (speculation stays on only if `QWEN35_TEMP > 0`) |
| `QWEN35_REP_PENALTY` / `QWEN35_NO_REPEAT_NGRAM` | `1.0` / `0` | Repetition controls (force plain decode) |
| `QWEN36_MTP` | `1` | `0` skips loading the MTP head (no speculation) |
| `QWEN_SDPA_BF8` | `0` | `1`: bf8 KV cache for SDPA (less memory, slightly lower precision) |
