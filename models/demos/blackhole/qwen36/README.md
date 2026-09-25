# Qwen3.5 / Qwen3.6 on Blackhole and Wormhole

This directory implements Tenstorrent inference for the hybrid
**Gated DeltaNet + Gated Full Attention** Qwen3.5/3.6 family. A single code path serves four
checkpoints. Everything runs on **Blackhole** (P150); the sparse-MoE **Qwen3.6-35B-A3B**
additionally runs on a **Wormhole LoudBox** (4 × n300), as a `(1, 4)` mesh (`N150x4`):

| Model            | `HF_MODEL`             | Arch      | Mesh / `MESH_DEVICE` | Parallelism            |
| ---------------- | ---------------------- | --------- | -------------------- | ---------------------- |
| Qwen3.5-9B       | `Qwen/Qwen3.5-9B`      | Blackhole | single P150 — `P150` | single device          |
| Qwen3.5-27B      | `Qwen/Qwen3.5-27B`     | Blackhole | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | Blackhole | P150x4 — `P150x4`    | 4-way tensor parallel  |
| Qwen3.6-27B      | `Qwen/Qwen3.6-27B`     | Blackhole | P150x8 — `P150x8`    | 8-way tensor parallel  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | Blackhole | P150x4 — `P150x4`    | 4-way TP + sparse MoE  |
| Qwen3.6-35B-A3B  | `Qwen/Qwen3.6-35B-A3B` | Wormhole  | WH LoudBox — `N150x4` | 4-way TP + sparse MoE |

The same `(1, 4)` TP code path serves `P150x4` and `N150x4`. See
[Running on the Wormhole LoudBox](#running-on-the-wormhole-loudbox).

> **Wormhole is for the 35B-A3B only.** The 9B / 27B checkpoints are Blackhole-only;
> `demo/text_demo.py` skips them on Wormhole.

The **35B-A3B** is the sparse Mixture-of-Experts member of the family (`qwen3_5_moe`:
256 routed experts, top-8, plus a gated shared expert on every layer). Every layer's
dense SwiGLU MLP is replaced by the sparse MoE block in `tt/moe/`; dispatch is
config-driven (`args.is_moe_layer`), so on the dense 9B/27B `num_experts == 0` and the
dense MLP path is byte-for-byte unchanged.

- The **9B** runs on a **single Blackhole P150** device. It uses the validated
  single-device forward path (no collectives).
- The **27B** variants (both Qwen3.5-27B and Qwen3.6-27B) run on a **P150x4**
  (a `(1, 4)` Blackhole mesh) using **4-way tensor parallelism (TP)**. The TP
  path needs `FABRIC_1D` for the cross-device collectives (all-reduce /
  reduce-scatter) and a trace region for the captured chunk-outer prefill trace.
- **Qwen3.6-27B additionally runs at TP=8** on a `(1, 8)` mesh (`P150x8`).
  Because it has only **4 KV heads**, TP=8 cannot give each device its own head:
  each head is instead **replicated across the device pair holding its GQA query
  group** (devices 0-1 share KV head 0, 2-3 head 1, and so on), so
  `n_local_kv_heads` is 1 at both TP=4 and TP=8 and the whole runtime KV path is
  unchanged. See `tp_common.replicate_kv_weight` and
  `ModelArgs.SUPPORTS_KV_REPLICATION`.

Everything model-specific (hybrid layer dispatch, DeltaNet head/conv dims,
partial rotary factor, vocab, layer count) is read from the parsed HF config, so
the single code base adapts to each checkpoint. The device count alone
(`num_devices > 1`) switches between the single-device and TP code paths — see
`tt/model_config.py` and `tt/tp_common.py`.

## Architecture

Assembly: `tok_embeddings → N × Qwen36DecoderLayer → RMSNorm → LM Head`.

Each model interleaves two attention block types (read from the HF
`layer_types`): **Gated DeltaNet** (linear-attention, recurrent + causal conv
state) layers and **Gated Full Attention** (paged KV cache) layers. The 9B has
32 layers (24 DeltaNet + 8 full-attention). Qwen3.5 uses zero-centered RMSNorm
everywhere and **partial** RoPE (only a fraction of each head is rotated).

## Environment setup

Before running **any** test, export the two environment variables that select
the checkpoint and the device mesh.

**9B (single P150):**

```bash
export HF_MODEL=Qwen/Qwen3.5-9B
export MESH_DEVICE=P150
```

**27B (P150x4):**

```bash
# Qwen3.6-27B
export HF_MODEL=Qwen/Qwen3.6-27B
export MESH_DEVICE=P150x4

# …or Qwen3.5-27B
export HF_MODEL=Qwen/Qwen3.5-27B
export MESH_DEVICE=P150x4
```

**35B-A3B (P150x4, sparse MoE):**

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=P150x4
```

**35B-A3B (n150x4 — Wormhole, sparse MoE):**

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=N150x4
```

`HF_MODEL` is the single source of truth for the checkpoint — it may be a Hugging
Face hub id (resolved via `snapshot_download`) or a local checkpoint directory.
`MESH_DEVICE` selects the mesh shape for `demo/text_demo.py` and the `*_tp` tests:
`P150` → `(1,1)`, `P150x4` / `N150x4` → `(1,4)`, `P150x8` → `(1,8)`.

Optional flags:

```bash
# Run SDPA in BF8 (faster; slightly lower precision).
export QWEN_SDPA_BF8=1
```

## Running on the Wormhole LoudBox

A Wormhole LoudBox holds four n300 cards (8 Wormhole chips). `MESH_DEVICE=N150x4` opens four of
those chips as a `(1, 4)` mesh and runs the **Qwen3.6-35B-A3B** through the same TP code path as
`P150x4`. No other checkpoint in this directory is supported on Wormhole.

```bash
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=N150x4
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not 128k" --timeout=5000
```

### Supported range

Batch-1 prefill runs up to and including the `traced_128k` case (a 103,351-token prompt). The
`traced_256k` case and batched cases with prompts longer than 128 tokens have not been run on
Wormhole.

### Validated results

Measured on the Wormhole LoudBox, `MESH_DEVICE=N150x4`, `HF_MODEL=Qwen/Qwen3.6-35B-A3B`, traced,
model load 12–18 s from a warm weight cache:

```bash
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s --timeout=5000 \
    -k "traced_128 or traced_4k or traced_8k or traced_64k or traced_128k or batched_128_b8 or batched_128_b32"
```

| Case | Prompt tokens | TTFT | Decode | Result |
| ---- | ------------- | ---- | ------ | ------ |
| `traced_128`      | 128     | 0.47 s   | 25.46 tok/s | PASSED |
| `traced_4k`       | 2,642   | 3.00 s   | 25.27 tok/s | PASSED |
| `traced_8k`       | 8,192   | 7.86 s   | 24.99 tok/s | PASSED |
| `traced_64k`      | 65,536  | 71.24 s  | 23.13 tok/s | PASSED |
| `traced_128k`     | 103,351 | 120.89 s | 21.51 tok/s | PASSED |
| `batched_128_b8`  | 128 × 8  | 16.69 s | 20.12 tok/s/user (161.0 aggregate) | PASSED |
| `batched_128_b32` | 128 × 32 | 7.59 s  | 9.37 tok/s/user (299.9 aggregate)  | PASSED |

`traced_4k` uses a 2,642-token prompt, and `traced_128k` runs the whole Frankenstein text, which
is 103,351 tokens (its KV block budget is sized for 128k).

PCC on the same mesh with the 35B-A3B checkpoint. `test_moe_tp.py` (6/6), `test_gdn_tp.py`
(17/17), `test_attention_tp.py` (7/7), `test_rope_tp.py` (2/2) and the `test_model_tp.py` cases
below all pass; `test_mlp_tp` skips on a MoE checkpoint, which has no dense MLP. See
[Running the tests on Wormhole](#running-the-tests-on-wormhole--35b-a3b-n150x4) for the commands.

| Case | PCC vs reference |
| ---- | ---------------- |
| MoE decode (torch)                         | 0.98747 |
| MoE decode, batch 8 (torch)                | 0.98884 |
| MoE prefill, seq 32 / 256 / 512 (torch)    | 0.98949 / 0.99016 / 0.98999 |
| Model contract, prefill logits (8 layers)  | 0.99845 |
| Long prefill T=2304, chunked vs single-pass | 0.99745 |
| Traced vs eager chunked prefill, T=4096 / 4352 | 0.99823 / 0.99928 |

## Running the demo and measuring performance

### Text demo (`demo/text_demo.py`)

`demo/text_demo.py` is a single parametrized test, `test_demo_text`, that runs prefill + decode on a
real prompt and prints TTFT and decode throughput. Every case is traced: the prefill (chunk-outer,
2048-token chunks) and decode forward passes are captured as device traces and replayed — the path
vLLM serves. The same command serves every checkpoint and arch; only `HF_MODEL` / `MESH_DEVICE`
change (see [Environment setup](#environment-setup)).

| Case id | ISL | Generated tokens | Batch | Notes |
| ------- | --- | ---------------- | ----- | ----- |
| `traced_128` | 128 | 50 | 1 | |
| `traced_4k` / `traced_8k` | 4k / 8k | 100 | 1 | `traced_4k`'s Frankenstein prompt is 2,642 tokens |
| `traced_16k` / `traced_32k` | 16k / 32k | 100 | 1 | |
| `traced_64k` | 64k | 500 | 1 | |
| `traced_128k` | 128k | 100 | 1 | the whole Frankenstein text: 103,351 tokens |
| `traced_256k` | 256k | 100 | 1 | 900 s timeout |
| `determinism_128` | 128 | 50 | 1 | runs twice, asserts identical output |
| `batched_128_b8` / `_b32` | 128 | 50 | 8 / 32 | multi-device only |
| `batched_256_b8` / `_b32` | 256 | 50 | 8 / 32 | multi-device only |
| `batched_4k_b8` / `_b32` | 4k | 50 | 8 / 32 | per-user prefill; multi-device only |
| `batched_8k_b8` … `batched_64k_b8` | 8k – 64k | 50 | 8 | per-user prefill; 64k has a 900 s timeout |

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=Qwen/Qwen3.6-35B-A3B
export MESH_DEVICE=N150x4          # Wormhole LoudBox; P150x4 on Blackhole

# One case. Note "traced_128" alone also matches traced_128k.
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced_128 and not 128k" --timeout=5000

# The Wormhole validation set from the results table above
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s --timeout=5000 \
    -k "traced_128 or traced_4k or traced_8k or traced_64k or traced_128k or batched_128_b8 or batched_128_b32"

# Every traced single-user ISL, and the determinism check
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "traced" --timeout=5000
pytest models/demos/blackhole/qwen36/demo/text_demo.py -v -s -k "determinism_128" --timeout=5000
```

**Reading the output.** Each case logs its prompt length and one result line:

```
Prompt: 2642 tokens (block budget: 72 blocks x 64 = 4608 tokens)
[TP 4-dev] ttft=3.00s decode=25.27 tok/s
[TP 4-dev B=8] ttft=16.69s per-user-decode=20.12 tok/s aggregate=161.0 tok/s
```

followed by the generated text. The first case in a session also logs a one-time
`prefill chunk-trace captured in …` (compile + trace capture); it is not part of TTFT.

**What the test asserts.** On the multi-device path: the requested number of tokens was generated,
batched rows (which all get the same prompt) decode identically, and generation is not degenerate.
It does **not** assert TTFT or tok/s locally — in CI the case writes a benchmark JSON that
`.github/scripts/utils/validate_perf_targets.py` checks against `models/model_targets.yaml` (see
[CI](#ci)).

> Long-context cases (64k+) download a public-domain corpus (Frankenstein, War and Peace) on first
> run and cache it under `demo/sample_prompts/.context_cache`. `vision_demo.py` is Blackhole-only.

| Env var | Default | Effect |
| ------- | ------- | ------ |
| `QWEN35_NO_THINK` | unset | `1` sends an empty thinking block instead of seeding `<think>` |
| `QWEN35_REF_PROMPT` | unset | `1` uses the reference 64k extractive-summary prompt for ISL >= 4k |
| `QWEN35_TEMP` / `QWEN35_TOP_K` / `QWEN35_TOP_P` | `0` / `0` / `1.0` | sampling (default: greedy) |
| `QWEN35_REP_PENALTY` / `QWEN35_NO_REPEAT_NGRAM` | `1.0` / `0` | repetition controls |
| `QWEN35_TP_PREFILL_EAGER` / `QWEN35_TP_DECODE_EAGER` | unset | `1` runs prefill / decode eagerly instead of traced (debugging) |
| `QWEN36_BATCHED_DECODE_MODE` | `shard` | batched logits readback: `shard` (per-shard on device), `sample`, or `host` |
| `QWEN36_DEBUG_DECODE_TIMING` | unset | `1` logs per-phase decode-step timing (update / execute / readback) |

### Measuring performance

**End-to-end (TTFT, tok/s).** Run the demo case and read its result line. TTFT is host wall time
around the traced prefill; decode tok/s is the mean over decode steps after the first, each step
timed end to end (input update + device + readback + sampling). Measure with nothing else running
on the box.

**Device profile (per-op time).** Tracy is built by default (`ENABLE_TRACY=ON`). Profile a
component test, not the full demo: the 40-layer demo issues more ops than the per-device profiler
buffer holds ("Profiler DRAM buffers were full, markers were dropped"), so its report comes out
incomplete. Always pass `--timeout`, because profiling makes a run several times slower than
pytest.ini's 300 s default allows for. For example, the MoE block at prefill seq 512:

```bash
python -m tracy -r -p -v -o generated/profiler/qwen36 \
    -m "pytest models/demos/blackhole/qwen36/tests/test_moe_tp.py -svq -k prefill512 --timeout 1200"
# per-op CSV: generated/profiler/qwen36/reports/<timestamp>/ops_perf_results_<timestamp>.csv
```

The CSV has one row per op per device, with `DEVICE KERNEL DURATION [ns]`, core count, shapes,
dtypes and memory configs. To profile a multi-layer prefill, build `Qwen36Model` with `n_layers=4`, run `prefill_traced_chunked` twice, wrap the second
call in `signpost("start")` / `signpost("stop")` (from `tracy`), and call
`ttnn.ReadDeviceProfiler(mesh_device)` between the calls so the device buffer is flushed.

**Perf targets.** CI targets per model / SKU / ISL live in `models/model_targets.yaml`
(`qwen3.6-35b-a3b`: `bh_quietbox_2` and `wh_llmbox_perf`), with a +-20% tolerance.

## Tests

There are two tiers of tests under `tests/`.

### Single-device unit / component tests — **9B**

These run on a single P150 against the 9B checkpoint (the conftests
`setdefault HF_MODEL=Qwen/Qwen3.5-9B`). They validate each component's forward
pass against a torch reference (PCC) — see `tests/pcc_thresholds.json` for the
per-test thresholds.

`tests/unit/` (component PCC vs torch):

| Test                       | Validates                                            |
| -------------------------- | ---------------------------------------------------- |
| `test_embedding.py`        | token embedding                                      |
| `test_rms_norm.py`         | zero-centered RMSNorm (the "+1" fold)                |
| `test_rope.py`             | partial-rotary RoPE (host freqs + on-device lookup)  |
| `test_mlp.py`              | single-device SwiGLU MLP (layer 0)                   |
| `test_moe.py`              | single-device sparse MoE MLP (MoE checkpoint only; decode + prefill) |
| `test_attention.py`        | single-device gated full attention (layer 3)         |
| `test_gdn.py`              | single-device Gated DeltaNet (layer 0)               |
| `test_lm_head.py`          | LM head logits (bf8 vs bf16)                          |
| `test_layer.py`            | full decoder-block sanity (no NaN/Inf, non-constant) |
| `test_model.py`            | Generator decode contract: traced vs paged decode    |
| `test_substate.py`         | weight `substate` helper (pure CPU, no device)       |

`tests/` (single-device, also 9B):

| Test                     | Validates                                                  |
| ------------------------ | ---------------------------------------------------------- |
| `test_prefill.py`        | masked fixed-bucket + chunk-outer prefill vs `prefill_paged` |
| `test_weight_mapping.py` | HF → internal weight key remapping (pure CPU)              |

Run the 9B unit suite (with `HF_MODEL=Qwen/Qwen3.5-9B`, `MESH_DEVICE=P150`):

```bash
pytest models/demos/blackhole/qwen36/tests/unit/ -v -s
pytest models/demos/blackhole/qwen36/tests/test_prefill.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_weight_mapping.py -v -s
```

> `test_prefill.py` auto-skips cases longer than `--max-prefill` (default 8192).
> Raise it to exercise long-context prefill, e.g. `--max-prefill 131072`.

### Tensor-parallel tests — **27B (P150x4)**

The `*_tp` tests exercise the multi-device TP path and default to the 27B
checkpoint. They must run on the `(1,4)` mesh with `FABRIC_1D` (the
`parametrize_mesh_tp` helper wires this from `MESH_DEVICE`; it also accepts
`N150x4`, the Wormhole LoudBox mesh — see
[Validated results](#validated-results)). PCC thresholds
are in `tests/pcc_thresholds.json`.

| Test                  | Validates                                                            |
| --------------------- | ------------------------------------------------------------------- |
| `test_mlp_tp.py`      | TP SwiGLU MLP (column/row-parallel + reduce-scatter)                |
| `test_moe_tp.py`      | TP sparse MoE MLP (router + experts + shared; MoE checkpoint only; decode + prefill) |
| `test_attention_tp.py`| TP gated full attention: decode / prefill / paged-KV contract       |
| `test_gdn_tp.py`      | TP Gated DeltaNet: decode + chunk-prefill                           |
| `test_model_tp.py`    | full-model TP contract: paged+traced path matches the bespoke oracle |
| `test_generate_tp.py` | full-model bespoke `generate_tp` on a real prompt (answer oracle)   |

Run the 27B TP suite (with `HF_MODEL=Qwen/Qwen3.6-27B` or `Qwen/Qwen3.5-27B`,
`MESH_DEVICE=P150x4`):

```bash
pytest models/demos/blackhole/qwen36/tests/test_mlp_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py -v -s
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py -svq
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
```

> The MoE-specific tests (`test_moe_tp.py`, and the MoE path in `test_model_tp.py` /
> `test_generate_tp.py`) require the sparse checkpoint — run them with
> `HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=P150x4`. On the dense 27B they are inert
> (`num_experts == 0`), and the dense/MoE checkpoints must not be mixed in one run
> (each test file `setdefault`s or expects a single `HF_MODEL`).

> `test_substate.py` and `test_weight_mapping.py` are pure-CPU and need no device.
> `test_weight_mapping.py`'s shape constants assume the 9B checkpoint.

### Running the tests on Wormhole — **35B-A3B (N150x4)**

Wormhole runs the sparse-MoE `Qwen3.6-35B-A3B` only, and only on the `(1,4)` mesh. The TP tests
above are the applicable set; everything single-device is Blackhole-only in practice:

* `tests/unit/*` skip themselves (`run_for_blackhole`, "only runs for Blackhole"). `test_substate.py`
  is pure CPU and passes anywhere.
* `test_prefill.py`, `test_weight_mapping.py` and the vision tests (`test_patch_merger.py`,
  `test_vision_attention.py`, `test_vision_block.py`, `test_wrapped_model.py`, `test_mlp.py`,
  `test_model.py`) target the 9B/27B/vision checkpoints and do not apply to 35B-A3B.

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_MODEL=Qwen/Qwen3.6-35B-A3B MESH_DEVICE=N150x4

pytest models/demos/blackhole/qwen36/tests/test_gdn_tp.py         -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_attention_tp.py   -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_moe_tp.py         -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_rope_tp.py        -q --timeout=600
pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py    -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_sampling.py       -q --timeout=600
pytest models/demos/blackhole/qwen36/tests/test_decode_bucketing.py -q --timeout=900
pytest models/demos/blackhole/qwen36/tests/test_model_tp.py       -q --timeout=2400
```

> Run these **module by module**, not as one pytest session: a single long session can die with a
> `Fatal Python error: Bus error`. `tt-smi -r` clears a wedged card.

> `test_decode_bucketing.py` needs vLLM for one case; without it that case skips.

> **Watcher on Wormhole:** set `TT_METAL_WATCHER_DISABLE_ETH=1` alongside `TT_METAL_WATCHER`;
> without it every test errors at mesh open and the cards need `tt-smi -r`.

## CI

The Qwen3.6 CI legs live in `tests/pipeline_reorg/` (unit: `models_unit_tests.yaml`, e2e:
`models_e2e_tests.yaml`, vLLM: `vllm_model_tests.yaml`); perf targets for the e2e legs are in
`models/model_targets.yaml`. The 35B-A3B unit and e2e entries each serve both arches from one
command: `MESH_DEVICE` (and, for the unit leg, the watcher setup) come from per-SKU fields.

| Model | Leg | Arch / SKU | Tier | Runs |
| ----- | --- | ---------- | ---- | ---- |
| Qwen3.6-35B-A3B | unit | BH QuietBox 2 — `bh_quietbox_2`, `MESH_DEVICE=P150x4` | 1 | `test_moe_tp`, `unit/test_moe`, `test_gdn_tp`, `test_attention_tp` (watcher on) |
| Qwen3.6-35B-A3B | unit | WH LoudBox — `wh_llmbox`, `MESH_DEVICE=N150x4` | 2 | same files (`unit/test_moe` skips; watcher on, ethernet cores excluded) |
| Qwen3.6-35B-A3B | e2e | BH QuietBox 2 — `bh_quietbox_2` | 1 | `text_demo.py` `traced_128`, `traced_4k`, `determinism_128` |
| Qwen3.6-35B-A3B | e2e | WH LoudBox — `wh_llmbox_perf` | 2 | same cases |
| Qwen3.6-27B | unit, e2e, vLLM | BH QuietBox 2 — `bh_quietbox_2` | 1 | see the yamls |

Scheduled runs pick every leg up automatically: the Tier 1 workflows run the Blackhole legs and the
Tier 2 workflows run the Wormhole ones. To run one leg by hand (Actions tab, or `gh`), select the
model `qwen3.6-35b-a3b` and the SKU:

```bash
# Blackhole (Tier 1)
gh workflow run "(Tier 1) Models Unit Tests"       --ref <branch> -f model=qwen3.6-35b-a3b -f sku="bh_quietbox_2 (BH QB2)"
gh workflow run "(Tier 1) Models End-To-End Tests" --ref <branch> -f model=qwen3.6-35b-a3b -f sku="bh_quietbox_2 (BH QB2)"
# Wormhole LoudBox (Tier 2)
gh workflow run "(Tier 2) Models Unit Tests"       --ref <branch> -f model=qwen3.6-35b-a3b -f sku="wh_llmbox (T3000)"
gh workflow run "(Tier 2) Models End-To-End Tests" --ref <branch> -f model=qwen3.6-35b-a3b -f sku="wh_llmbox_perf (T3000 perf)"
```

The Wormhole runners read the checkpoint from `/mnt/MLPerf/huggingface` with the HF hub offline, and
the shared tensor cache is mounted read-only by default. A first Wormhole run therefore needs the
checkpoint on that share and `-f mlperf-write-access=true`, so it can write the Wormhole tensor cache
(kept separate from Blackhole's: the cache path includes the device name).
