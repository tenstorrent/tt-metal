<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 on Tenstorrent P150

## Coding agent quickstart

Tell your coding agent:

> run laguna-2.1-xs and then give me a command to point the `pool` coding agent from poolside.ai at the running model.

TTNN bring-up + vLLM serving of [`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1),
a ~31B GLM/Qwen3-style MoE (256 experts, top-8, shared expert; 40 layers, 10 full-attention + 30
sliding-window(512); router `sigmoid(logits)+e_bias`, `norm_topk_prob`, no router bias).

In this runbook, **one P150 means one Blackhole ASIC/device as enumerated by `tt-smi`**. It does not
mean a board name that may contain more than one ASIC. `p150x2` and `p150x4` therefore mean two and
four independently enumerated devices in a 1×D mesh. On the bring-up host, `p150x2` is two ASICs on
one physical P300 card.

- **Recommended one-card target:** `p150x2` (D=2), ring/two-link, at 131,072 tokens.
- **One-chip result:** `p150` (D=1) is ruled out by weight capacity before KV allocation or warmup.
- **Regression profile:** `p150x4` (D=4), TP=4 / EP=4, `FABRIC_1D_RING`.
- **Precision (selected policy):** BF16 activations/norms/router, BFP8 attn/dense/shared + KV + LM-head,
  BFP4 routed experts, fp32/HiFi4 SDPA. See `doc/datatype_sweep/`.
- **Serving:** stock upstream vLLM 0.24.0 + the public `tenstorrent/vllm-tt-plugin` + this model's
  `vllm_ext` package — **no vLLM fork**. Adapter: `tt/generator_vllm.py`.

The detailed qualification order, commands, gates, and result template are in
[`doc/bringup_profiles.md`](doc/bringup_profiles.md).

## Context / capability

| Profile | Devices / mesh | Configured context | Max sequences | Status on 2026-08-21 |
|---|---:|---:|---:|---|
| `p150` | 1 / 1×1 | **65,536 configured** | 1 | **Rejected:** full 40-layer model OOM at the LM head even with a 4,096-token cap |
| `p150x2` | 2 / 1×2 | **131,072** | 1 | **Qualified and default:** correctness, memory, trace, exact-cap, decode, and canonical prefix caching pass |
| `p150x4` | 4 / 1×4 | **131,072** | 8 | Existing measured regression profile |

The checkpoint declares 262,144 tokens, but no serving profile advertises that value. D1 passed the
bounded topology and corrected representative-layer checks, then failed decisively while loading the
full 40-layer model: at the final LM-head allocation it had about 2,692 MiB allocated per bank,
80.1 MiB free per bank, and only 26.27 MiB contiguous, while the allocation needed 27.30 MiB per bank.
This happened with `max_seq_len=4096`, before KV allocation or warmup, so lowering context cannot make
the current one-ASIC weight layout viable.

D2 ring/two-link on ASICs `0,1` passed all 12 representative packed-path checks, the full 40-layer
131,072-token uniform-KV smoke, trace replay, and device-versus-host greedy comparison (8/8). A direct,
nondegenerate traced decode-SDPA boundary test also passed. The authoritative post-fix cold server
retained a 0.2193 post-trace free fraction and 525.8 MiB largest contiguous free per bank, above the
0.10 / 128 MiB gates. Engine initialization took 527.76 s (499.24 s compilation), `/health` returned
200, and `/v1/models` advertised 131,072 with prefix caching disabled. Raw deterministic repeats,
chat, tool calling, two 128-token decode replays, and a real `pool` coding-agent shell-tool round trip
passed; the server remained healthy after the expected once-only allocator warning on the first eager
prefill. Artifact:
`/home/ttuser/laguna-qualification/p150x2-qualified-noalloc-20260821T123447`.

The exact-cap request also passed: ISL 130,048 + OSL 1,024 used the full 131,072-token contract,
reached 100% KV occupancy, completed without an error, released KV back to 0%, and left `/health` at
200 with no new fault marker. `p150x2` is therefore promoted as the launcher default.
`doc/context_contract.json` is the machine-readable source of truth.

A standalone warm prompt-128/generate-128 diagnostic measured 190.7 ms TTFT, 25.51 tokens/s for the
logits-only loop, 20.02 tokens/s with device token output, and 19.75 tokens/s including host readback.
The official cold concurrency-1 runs completed 3/3 without errors: ISL 1,024 / OSL 128 measured
1,278.96 ms TTFT, 50.46 ms TPOT (19.82 decode tokens/s), and 16.650 aggregate output tokens/s; ISL
32,768 measured 53,546.66 ms TTFT, 51.53 ms TPOT (19.41 decode tokens/s), and 2.130 aggregate output
tokens/s. Decode remains decent at 32K, but cold long-context end-to-end throughput is prefill-bound;
the 2.130 aggregate rate is not the decode gate, but the 53.55 s TTFT is an important serving caveat.
The qualified canonical-prefix path below reduces that cost when a reusable prefix exists; a genuinely
cold request still pays the full prefill cost.

At the exact cap, ISL 130,048 / OSL 1,024 completed in 437.862 s with 381,378.79 ms TTFT,
55.21 ms TPOT (**18.11 decode tokens/s**), and 2.339 aggregate output tokens/s. This proves capacity
and stable decode at the limit; it also makes the roughly 6.36-minute cold prefill cost explicit.

After that full-cap run, `pool` invoked `pwd` in
`/tmp/laguna-p150x2-pool-smoke-20260821T130000`, received exit 0 and the exact path, returned
`LAGUNA_POOL_OK`, and exited 0 in 21.78 s. Final health was 200; the fault scan found only the already
documented allocator advisory.

### Cold chat latency sweep

The latest complete latency curve used the OpenAI chat endpoint, OSL 512, concurrency 1, prefix cache
off, temperature 0, ignore EOS, and seed 1234. The model entered measurement with the power-of-two
prefill ladder compiled during boot; each row is one measured request with no explicit per-length
warmup. Requested random ISL and actual server-counted prompt tokens differ because the random token
IDs were decoded to text, wrapped in the chat template, and tokenized again.

| Requested ISL | Actual prompt | TTFT | TPOT | E2EL | Decode tok/s/user | Aggregate output tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 82 | 0.206 s | 50.09 ms | 25.801 s | 19.97 | 19.844 |
| 1,024 | 1,066 | 2.215 s | 50.46 ms | 28.002 s | 19.82 | 18.284 |
| 2,048 | 1,939 | 2.569 s | 50.50 ms | 28.372 s | 19.80 | 18.046 |
| 4,096 | 4,138 | 8.992 s | 50.57 ms | 34.835 s | 19.77 | 14.698 |
| 8,192 | 8,234 | 19.720 s | 50.72 ms | 45.638 s | 19.72 | 11.219 |
| 16,384 | 16,426 | 46.708 s | 51.02 ms | 72.780 s | 19.60 | 7.035 |
| 32,768 | 32,810 | 122.162 s | 51.61 ms | 148.532 s | 19.38 | 3.447 |
| 65,536 | 65,578 | 359.013 s | 52.78 ms | 385.982 s | 18.95 | 1.326 |
| 130,048 | 130,090 | 381.719 s | 55.11 ms | 409.878 s | 18.15 | 1.249 |

All nine requests completed with exactly 512 output tokens, final health 200, and no serving-time
compile, retrace, or critical fault. The final row used 130,090 actual prompt tokens plus 512 output
tokens, within the 131,072-token serving cap. Full-precision values are committed in
[`doc/vllm_integration/p150x2_latency_sweep_20260821.tsv`](doc/vllm_integration/p150x2_latency_sweep_20260821.tsv),
with method and provenance in
[`doc/vllm_integration/p150x2_latency_sweep_20260821.md`](doc/vllm_integration/p150x2_latency_sweep_20260821.md).
This is a cold latency curve with one sample per point, not a variance study; the repeated-prefix
results below separately measure qualified cache hits.

## Determinism and prefix-cache safety

> **`p150x2` enables qualified prefix caching by default.** Set `TT_LAGUNA_PREFIX_CACHE=0` for an
> immediate fail-closed rollback. `p150` and `p150x4` remain cache-off and are not qualified for it.

Safety comes from canonical admission, not from accepting every 64-token vLLM cache match. The physical
KV block size remains 64, but only complete **8,192-token prompt checkpoints** are inserted and admitted.
The final 64-token vLLM block is already recomputed; Laguna floors the remaining candidate hit to 8K,
so the fresh tail always runs with the same outer-chunk geometry as cold prefill. Generated decode KV,
partial prompt chunks, and sub-8K matches cannot become reusable checkpoints. The qualified envelope is
exactly one sequence, one uniform `FullAttentionSpec` KV group, scheduler chunked prefill off, internal
8K pipelined prefill on, and no speculative decoding, lookahead, encoder request, or external KV connector.
Launcher and engine-side validation reject deviations.

The model keeps absolute resume positions as runtime data: fill tables are host-rebased, RoPE uses
indexed embedding into persistent outputs, and chunked SDPA receives `chunk_start_idx_tensor`. Resumed
suffixes, including small ones, use the canonical pipeline kernel family. Every permitted shape is
warmed before decode trace capture, then TTNN program-cache misses are forbidden. An unseen
specialization therefore fails closed instead of compiling under the resident trace.

The final raw-token qualification used three repetitions per performance point. Cold and hit generations
were token-for-token identical to the cache-off oracle; all reported cached-token counts were exact.

| Prompt | Cache-off TTFT | Cache-on cold TTFT | Admitted prefix | Hit TTFT | TTFT speedup | TPOT off / cold / hit |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 52,681.834 ms | 52,654.973 ms | 24,576 | 15,889.832 ms | **3.314×** | 51.510 / 51.548 / 51.522 ms |
| 65,536 | 134,040.189 ms | 133,891.981 ms | 57,344 | 23,011.514 ms | **5.818×** | 52.587 / 52.605 / 52.605 ms |

Cache-on cold TTFT was 0.9995× and 0.9989× the cache-off oracle; cold and hit TPOT stayed within
0.08% of it. Partial-hit cases admitted exactly 0 / 32,768 / 65,536 / 122,880 tokens at 2K / 32K /
65K / near-cap prefixes and matched the oracle. A 2K-then-32K oldest-hash poisoning sequence admitted
0 / 0 / 24,576 tokens as intended, and a prompt extended across 8K by generated decode tokens admitted
zero. Prometheus recorded exactly 491,520 hit tokens over 1,472,572 queried tokens.

On device, all five flexible-SDPA, indexed-RoPE, and full/sliding resumed-layer tests passed while
program-cache misses were forbidden. The full server froze 889 entries after trace capture; post-trace
DRAM retained a 0.2144 free fraction and 510.8 MiB largest contiguous free per bank. Final health passed,
with the same known once-per-thread active-trace allocator advisory and no critical fault, cache miss,
corruption, or device death. Evidence:

- cache-off oracle: `/home/ttuser/laguna-qualification/p150x2-prefix-cache-20260821/cache-off-oracle.json`
- cache-on result: `/home/ttuser/laguna-qualification/p150x2-prefix-cache-20260821/cache-on-canonical-8k/results/full_qualification.json`
- hardware JUnit: `/home/ttuser/laguna-qualification/p150x2-prefix-cache-20260821/hardware-kernel-layer-canonical-admission.xml`

## Serving profiles

The serving env is **self-contained**: a dedicated venv holding `ttnn` built from this checkout, stock
`vllm==0.24.0`, the public `tenstorrent/vllm-tt-plugin`, and this model's `vllm_ext` package. Everything
is built from this repo — no hand-prepared environment.

### Quick start

Inspect and launch the recommended two-device profile explicitly:

```bash
REPO_ROOT=/path/to/tt-metal
MODEL_DIR="$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1"
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh" config
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh"
tail -f ~/laguna-logs/latest.log
```

The clean p150x2 configuration reports `prefix_cache=1`, policy `qualified`, canonical quantum 8,192,
and `--enable-prefix-caching --enable-prompt-tokens-details --no-enable-chunked-prefill`. To roll back
without an experimental acknowledgement, add `TT_LAGUNA_PREFIX_CACHE=0`; the launcher then passes
`--no-enable-prefix-caching` and reports `operator_rollback_disabled`.

The one-device profile remains available for diagnostics, but cannot load the complete model in the
current weight layout:

```bash
LAGUNA_PROFILE=p150 "$MODEL_DIR/serve_vllm.sh" config
```

Running `serve_vllm.sh` without `LAGUNA_PROFILE` selects the qualified `p150x2` default. Select
`LAGUNA_PROFILE=p150x4` explicitly for the four-ASIC regression profile. The
launcher validates that `TT_VISIBLE_DEVICES` contains exactly 1, 2, or 4 distinct identifiers for the
selected profile. Device identifiers are host-specific; inspect `tt-smi` instead of assuming that a
particular pair is physically connected.

First run builds tt-metal (**~1–3 h**) then vLLM from sdist (**~30–45 min**); after that it is just the
**~10 min** server boot. The build streams to the same log, so `tail -f` shows real progress
throughout. Ready when the current run's log says `Application startup complete`; then check
`/health` and `/v1/models`. Do not use a KV-size line or an older log as the readiness signal for a
different profile.

Each run writes its own `~/laguna-logs/laguna_serve_<timestamp>.log` and repoints `latest.log` at it,
so a boot's log is never clobbered by the next one and `tail -f latest.log` always follows the run in
progress. Override the directory with `LAGUNA_LOG_DIR=/path`.

**Prerequisites:** Linux x86-64, Python 3.12, a tt-metal build toolchain (repo `INSTALLING.md` →
`./install_dependencies.sh`), submodules (`git submodule update --init --recursive`), a Tenstorrent
Blackhole device count matching the selected profile with `tt-smi` on PATH, and the HF model cached
(gated — `hf auth login`, ~63 GB;
`huggingface-cli` is removed in `huggingface_hub >= 1.0`).

### The env (`setup_vllm.sh`)

`serve_vllm.sh` calls this for you; run it directly only to rebuild (`--force`) or build elsewhere
(`VLLM_ENV=/path ./setup_vllm.sh`). Default location `.venv/` in the model dir (gitignored). Pins live in
`requirements.txt`, dependency overrides in `overrides.txt`. What it does and why:

| Step | Why |
|---|---|
| `uv venv --python 3.12` | Matches the Python the C++ extension is built against. |
| `./build_metal.sh` (if `_ttnn.so` absent) then `uv pip install -e <repo root>` | **`ttnn` must come from this checkout, not PyPI** (see below). The editable install only wires the built tree in; the build step produces `_ttnn.so`. |
| `VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm --extra-index-url …/whl/cpu --index-strategy unsafe-best-match --override overrides.txt vllm==0.24.0` | PyPI vLLM is CUDA, so build from sdist against the `empty` target; the `tt` platform comes from the plugin at runtime. The CPU torch index is required or `torch==2.11.0` pulls ~4 GB of `nvidia-*-cu13` wheels. **Slow step.** |
| `uv pip uninstall torchaudio` | `transformers>=5.12` imports it if present, and the wheel pulled alongside CPU torch is unloadable. |
| `uv pip install vllm-tt-plugin @ git+…@c127c17…` | The `tt` platform + `EXTRA_MODELS_DIR` registration, pinned to the exact tested plugin commit. Setup verifies its PEP 610 commit metadata. |
| `uv pip install -e vllm_ext` | Laguna-scoped canonical prefix admission and capability handling plus the newline-tolerant `poolside_v1` parser required for `auto` tool-calling. |

**Why `ttnn` is built here, not installed from PyPI:** the 30 sliding-window layers pass
`sliding_window_size` through the internal chunked prefill SDPA path — and **no released `ttnn` accepts
`sliding_window_size` on the chunked op** (absent from 0.74.0, 0.75.0, `origin/main`). This branch
removes that restriction, so the serving env must be built from this checkout. The qualified p150x2
resume path also requires this checkout's flexible runtime-offset SDPA and indexed-RoPE support.
**Why `uv` and not `pip`:** `ttnn` pins
`numpy<2` while vLLM 0.24.0 wants
`opencv-python-headless>=4.13.0` (needs `numpy>=2`) — only `uv pip --override` resolves it (`overrides.txt`).

### Profile controls

Use `serve_vllm.sh config` as the source of truth for the final vLLM arguments. It validates the
overrides without opening a device:

```bash
LAGUNA_PROFILE=p150 models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh config

# A D2 topology/link combination may be used only after tests/qualify_topology.py passes for it.
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 \
TT_LAGUNA_CCL_TOPOLOGY=linear TT_LAGUNA_CCL_NUM_LINKS=2 \
  models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh config
```

Safe overrides are `LAGUNA_MAX_MODEL_LEN` (may only lower the profile cap), `LAGUNA_MAX_NUM_SEQS`
(may only lower the profile limit), `LAGUNA_TRACE_REGION_SIZE`, `LAGUNA_LOG_DIR`, and
`TT_VISIBLE_DEVICES`. For D2/D4, `TT_LAGUNA_CCL_TOPOLOGY=linear|ring` and
`TT_LAGUNA_CCL_NUM_LINKS=1|2` select a previously qualified combination; the launcher derives and
checks the matching `LAGUNA_FABRIC_CONFIG`. The initial trace reservation is 1.5 GB. Reduce it only
when the measured memory gate requires it, and repeat trace capture/replay after every change.
The qualified post-trace margins are `TT_LAGUNA_MIN_DRAM_FREE_FRACTION=0.10` and
`TT_LAGUNA_MIN_CONTIGUOUS_MIB=128`; changing either requires
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1` and makes the run diagnostic-only.

On this P300_X2 host, selecting one ASIC is a custom cluster, so `p150` automatically exports the
checked-in 1×1 `TT_MESH_GRAPH_DESC_PATH`. D2/D4 remove that singleton descriptor; an inherited custom
descriptor is rejected unless `LAGUNA_ALLOW_CUSTOM_MESH_GRAPH_DESC=1` is set deliberately. The D1
profile also selects `TT_LAGUNA_DECODE_SDPA_PC=0`, the TTNN default decode configuration that passes
the long-position diagnostics. Forcing `=1` on D1 restores the custom k64 configuration and is a
warning-labelled, known-inaccurate debugging override, not a qualification setting.

> **Do not set `TT_METAL_HOME`.** `ttnn` self-locates its runtime root from the tree it was installed
> from; pointing `TT_METAL_HOME` at a different tt-metal tree mixes in another version's kernels.

### Qualify topology before loading weights

Run every mesh and every D2 topology/link candidate in a **fresh Python process after a device reset**.
Do not open D1 and then D2, change a fabric mode, or try another physical pair in the same process.

```bash
REPO_ROOT=/path/to/tt-metal
MODEL_DIR="$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1"
tt-smi -r all
cd /tmp
env -u TT_METAL_HOME TT_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT" \
  "$MODEL_DIR/.venv/bin/python" -m \
  models.autoports.poolside_laguna_xs_2_1.tests.qualify_topology --profile p150

# Reset again, then test one D2 candidate. Repeat in a new process for every physical pair and
# linear/ring + one/two-link combination; record the emitted QUALIFY_TOPOLOGY JSON.
tt-smi -r all
env -u TT_METAL_HOME TT_VISIBLE_DEVICES=0,1 PYTHONPATH="$REPO_ROOT" \
  "$MODEL_DIR/.venv/bin/python" -m \
  models.autoports.poolside_laguna_xs_2_1.tests.qualify_topology \
  --profile p150x2 --topology linear --num-links 1
```

Use only a combination that completes all open/close cycles plus eager and traced all-reduce checks.
Reset once more before starting the full model. `serve_vllm.sh stop` performs the server teardown and
`tt-smi -r all` reset; a missing `tt-smi` is reported rather than silently treated as success.

### Verify

```bash
curl -s localhost:8000/health && echo OK
curl -s localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'  # poolside/Laguna-XS-2.1

# Tool-calling (auto): expect finish_reason=tool_calls + get_weather({"city":"Paris","metric":true})
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"poolside/Laguna-XS-2.1",
  "messages":[{"role":"user","content":"What is the weather in Paris right now? Use the get_weather tool with metric units."}],
  "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"metric":{"type":"boolean"}},"required":["city"]}}}],
  "tool_choice":"auto","temperature":0,"max_tokens":1024,
  "chat_template_kwargs":{"enable_thinking":true}}' | python3 -m json.tool
```

### Teardown + mesh reset (ALWAYS between runs)

Hard-killing a fabric server can dirty Ethernet cores. Stop and reset before opening any profile,
changing a D2 topology, or selecting another physical pair.

```bash
models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh stop     # TERM/KILL the group + tt-smi -r all
# by hand:  pkill -TERM -f "vllm serve poolside"; sleep 10; pkill -KILL -f "vllm serve poolside"; tt-smi -r all
```

### Notes / gotchas

- **`TT_LAGUNA_HYBRID_KV=0` is mandatory.** Hybrid KV is known unsafe in the current vLLM/plugin path:
  its multi-group allocator does not preserve Laguna's 10-full/30-sliding logical-to-physical group
  mapping. The launcher rejects any nonzero value. All profile qualification uses uniform BFP8 KV.
- `p150` and `p150x2` are single-user profiles (`--max-num-seqs 1`). The established `p150x4`
  regression profile retains `--max-num-seqs 8`; acceptance performance is still measured at
  concurrency 1.
- **`enable_thinking:true`** is the tool-calling default for this model (both true/false verified).
- `TT_LAGUNA_PREFILL_FAST=1` ≈ 1.43× prefill. All `TT_LAGUNA_*` env vars are passed through to the worker
  by the plugin — set them in the serve env.
- The D2 standalone warm loop measured 20.02 tokens/s through device token output and 19.75 tokens/s
  including host readback (25.51 tokens/s logits-only). This is a useful implementation diagnostic;
  base decode acceptance uses `.venv/bin/vllm bench serve`, while prefix-cache acceptance uses the
  raw-token two-phase harness documented above.
- D1's two custom-k64 long-position failures pass with the profile's
  `TT_LAGUNA_DECODE_SDPA_PC=0` setting, but the full model cannot load its LM head on one ASIC. D1 is
  rejected on capacity regardless of its component-correctness result.
- **Warmup:** every prefill program shape is compiled before the decode trace is captured
  (`warmup_model_prefill` warms the power-of-two bucket ladder up to the servable context). A private
  completion latch suppresses the plugin's redundant second full ladder pass. Do not set
  `TT_LAGUNA_PREFILL_WARM_CAP` below `--max-model-len` for serving.
- **Qualified D2 prefix resume:** vLLM supplies absolute prompt end and canonical cached-prefix length.
  The adapter uploads only `tokens[start:end]`, buckets by `end-start`, and retains `start` as the
  absolute KV/RoPE offset. Only complete 8K prompt checkpoints are admitted over the 64-token physical
  blocks; smaller or decode-produced matches are recomputed. This path requires `p150x2`, max sequences
  1, uniform KV, 8K internal prefill chunks, scheduler chunked prefill off, and speculative decoding off.
- **Resume terminal row:** a host one-hot is copied into a persistent per-bucket selector and the
  selector matmul writes into a persistent one-row output. The selected index is relative to the
  scheduled suffix, so resumed prefills return the true last-real-token logits without creating a
  per-request selector/output buffer or reading back the full hidden state.
- **Resume bucket-padding isolation:** each layer allocates one adapter-private physical KV scratch block.
  Separate persistent prefill-only page tables map whole attention-padding blocks to that zeroed
  scratch block while mapping paged-fill padding to the kernel's `-1` write-skip sentinel. This avoids
  both block-0 corruption and concurrent writer races on scratch; the scheduler/decode table is never
  modified. The private prefill tables extend through
  `max_model_len + largest_bucket`, and shared RoPE tables extend to the same horizon (262,144 for the
  131,072-token D2 profile), so a near-cap resumed suffix stays addressable without exposing a larger
  serving context.
- **Fill stability:** the adapter host-rebases every request's persistent fill row so column zero maps
  to its canonical scheduled start. Cold and resumed serving therefore avoid an absolute-start device
  slice; fixed per-chunk relative slices remain part of the warmed internal pipeline. This is not a
  global claim that eager prefill performs no device allocation.

## How it runs fork-free (`vllm_ext`)

The tt-metal-side compatibility layer for **stock** vLLM 0.24.0 plus the pinned public plugin is the
checked-in `vllm_ext/` package — no site-packages, vLLM, or external plugin source edit is required:

- The engine-core/launcher hooks that used to require a vLLM fork are **upstreamed in vLLM 0.24.0**, so
  the public plugin runs on stock vLLM.
- Laguna registers via the plugin's **`EXTRA_MODELS_DIR`** mechanism — a bundle folder with
  `vllm_ext/extra_models/laguna/vllm_metadata.json` (`arch=LagunaForCausalLM` →
  `…tt.generator_vllm:LagunaForCausalLM`; the plugin prefixes `TT`). No plugin source change.
- The pinned plugin disables prefix caching for sliding-window models. `vllm_ext` can preserve it only
  when a model explicitly advertises both normal and sliding-window prefix support. Qualified p150x2
  advertises both when its default cache policy is enabled; the wrapper then restores the requested
  engine setting and validates the canonical admission envelope. Other Laguna profiles remain off.
- The `poolside_v1` tool + reasoning parsers ship **in stock vLLM 0.24.0**. One catch fixed here: the
  stock tool parser's regex requires a newline after the function name (`<tool_call>NAME\n…`), but this
  checkpoint emits the arg tags immediately after the name (`<tool_call>NAME<arg_key>…`, no newline), so
  stock `auto` tool-calling silently returns `finish_reason=stop`. `vllm_ext/laguna_vllm_ext` is a
  `vllm.general_plugins` entry point that eagerly re-registers `poolside_v1` with a **newline-tolerant**
  regex (parses both grammars). `setup_vllm.sh` installs it; existing `--tool-call-parser poolside_v1`
  flags are unchanged.

Refresh just this package in an existing env:
`.venv/bin/pip install -e vllm_ext`.

## Coding agent (`pool`)

`pool` (Poolside's terminal agent, github.com/poolsideai/pool) talks to this server in **standalone mode**.

```bash
# 1. Serve Laguna (above), confirm /v1/models returns poolside/Laguna-XS-2.1.
# 2. Install pool:
curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh   # accept EULA; adds ~/.local/bin
# 3. Point pool at the local endpoint — POOLSIDE_STANDALONE_BASE_URL is the mode switch:
export POOLSIDE_STANDALONE_BASE_URL=http://localhost:8000/v1    # note the /v1 (without it: 404 default-agent)
export POOLSIDE_API_KEY=EMPTY                                   # any value; local server ignores it
export POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1"
export POOLSIDE_STANDALONE_CONTEXT_LENGTH=131072                # recommended p150x2/p150x4
# 4a. Interactive:      cd <project> && pool
# 4b. One-shot (CI):    pool exec --unsafe-auto-allow --sandbox disabled -p "…task…"
```
Tool-calling round-trips automatically through `poolside_v1`. Production requests re-prefill the
transcript because APC is disabled; prefer focused prompts and one active `pool` session because the
recommended D2 profile is intentionally single-user.

## Tests

```bash
REPO_ROOT=/path/to/tt-metal
MODEL_DIR="$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1"
cd /tmp && env -u TT_METAL_HOME PYTHONPATH="$REPO_ROOT" LAGUNA_PROFILE=p150 TT_VISIBLE_DEVICES=0 \
  TT_LAGUNA_DECODE_SDPA_PC=0 \
  "$MODEL_DIR/.venv/bin/python" -m pytest "$MODEL_DIR/tests/test_multichip_decoder.py" -q
```

- `tests/test_laguna_test_utils.py` — device-free profile, topology, and memory-gate contract.
- `tests/qualify_topology.py` — bounded open/close plus eager/traced CCL qualification.
- `tests/test_multichip_decoder.py` — production packed path on D=1/2/4; representative layers 0, 1,
  and 4 must reach PCC ≥ 0.995. Its D2 direct traced-SDPA test uses nonzero V data and probes both
  sides of k64/block64 boundaries through position 131,071.
- `tests/test_generator_vllm_prefix_resume.py`, `tests/test_prefill_page_table.py`, and
  `tests/test_prefill_runtime.py` — device-free suffix slicing, host-rebased fill, runtime SDPA/RoPE
  inputs, private scratch/split tables, terminal selection, validation, and one-pass warmup contracts.
- `vllm_ext/tests/test_prefix_cache_quantum.py` — ownership-safe canonical admission, prompt-only hash
  insertion, poisoning protection, metric correction, and fail-closed scheduler/KV geometry.
- `tests/test_prefix_cache_hardware.py` — p150x2 flexible-SDPA, indexed-RoPE, and representative
  full/sliding layer accuracy with program-cache misses forbidden across changed offsets.
- `tests/prefix_cache_qualification.py` — cache-off oracle and cache-on raw-token exactness, admission,
  poison-order, decode-boundary, Prometheus, health, and 32K/65K latency gates.
- `tests/smoke_full_model.py --profile <profile> --enforce-memory-margin` — all-40-layer build,
  uniform KV, prefill, trace replay, and measured DRAM margin.
- `tests/full_model_checks.py` — full-model top-1/5/100 and autoregressive checks against the AIME24
  reference. Acceptance bars are top-1 ≥ 0.90, top-5 ≥ 0.98, and top-100 = 1.00.

See `doc/bringup_profiles.md` for the cold-serving, prefix-cache, performance, and promotion gates. Passing
unit or layer tests alone does not qualify a profile.
