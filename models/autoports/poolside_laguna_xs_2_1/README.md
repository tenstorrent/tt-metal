<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Laguna-XS-2.1 on Tenstorrent p150x2

## Coding agent quickstart

Tell your coding agent:

> run laguna-2.1-xs and then give me a command to point the `pool` coding agent from poolside.ai at the running model.

TTNN bring-up + vLLM serving of [`poolside/Laguna-XS-2.1`](https://huggingface.co/poolside/Laguna-XS-2.1),
a ~31B GLM/Qwen3-style MoE (256 experts, top-8, shared expert; 40 layers, 10 full-attention + 30
sliding-window(512); router `sigmoid(logits)+e_bias`, `norm_topk_prob`, no router bias).

In this runbook, **one P150 means one Blackhole ASIC/device as enumerated by `tt-smi`**. `p150x2` and
`p150x4` therefore mean two and four independently enumerated devices in a 1×D mesh. On the
qualification host, `p150x2` is one physical dual-P150 card containing two ASICs.

- **Recommended one-card target:** `p150x2` (D=2), ring/two-link, at 131,072 tokens.
- **One-chip result:** `p150` (D=1) is ruled out by weight capacity before KV allocation or warmup.
- **Regression profile:** `p150x4` (D=4), TP=4 / EP=4, `FABRIC_1D_RING`.
- **Precision (selected policy):** BF16 activations/norms/router, BFP8 attn/dense/shared + KV + LM-head,
  BFP4 routed experts, fp32/HiFi4 SDPA. See `doc/datatype_sweep/`.
- **Serving:** stock upstream vLLM 0.24.0 + the public `tenstorrent/vllm-tt-plugin` + this model's
  `vllm_ext` package — **no vLLM fork**. Adapter: `tt/generator_vllm.py`.

The detailed qualification order, commands, gates, and result template are in
[doc/bringup_profiles.md](doc/bringup_profiles.md).

## Context / capability

| Profile | Devices / mesh | Configured context | Max sequences | Status on 2026-08-22 |
|---|---:|---:|---:|---|
| `p150` | 1 / 1×1 | **65,536 configured** | 1 | **Rejected:** full 40-layer model OOM at the LM head even with a 4,096-token cap |
| `p150x2` | 2 / 1×2 | **131,072** | 1 | **Qualified and default:** correctness, memory, trace, exact-cap, decode, and canonical prefix caching pass |
| `p150x4` | 4 / 1×4 | **131,072** | 8 | Existing measured regression profile |

The checkpoint declares 262,144 tokens, but no production serving profile advertises that value. The
explicit fail-closed `p150x2` probe passed two deterministic 131,136+16 boundary repeats with exact
output-token agreement. Its 262,112+32 exact-cap request reached 97.0934% KV occupancy cleanly, then
exceeded the deliberate 1,200-second client latency budget and was aborted. The 262K envelope is
therefore rejected on performance and remains unqualified; production stays at 131,072 tokens. D1
passed the bounded topology and corrected representative-layer checks, then failed decisively while
loading the full 40-layer model: at the final LM-head allocation it had about 2,692 MiB allocated per
bank, 80.1 MiB free per bank, and only 26.27 MiB contiguous, while the allocation needed 27.30 MiB per
bank. This happened with `max_seq_len=4096`, before KV allocation or warmup, so lowering context cannot
make the current one-ASIC weight layout viable.

The original 2026-08-21 cache-off D2 promotion run used ring/two-link on ASICs `0,1` and passed all 12
representative packed-path checks, the full 40-layer 131,072-token uniform-KV smoke, trace replay, and
device-versus-host greedy comparison (8/8). A direct, nondegenerate traced decode-SDPA boundary test
also passed. That cold server retained a 0.2193 post-trace free fraction and 525.8 MiB largest
contiguous free per bank, above the 0.10 / 128 MiB gates. Engine initialization took 527.76 s
(499.24 s compilation), `/health` returned 200, and `/v1/models` advertised 131,072 with prefix caching
disabled. Raw deterministic repeats, chat, tool calling, two 128-token decode replays, and a real
`pool` coding-agent shell-tool round trip passed; the server remained healthy after the expected
once-only allocator warning on the first eager prefill. Artifact:
`/home/ttuser/laguna-qualification/p150x2-qualified-noalloc-20260821T123447`.

The exact-cap request also passed: ISL 130,048 + OSL 1,024 used the full 131,072-token contract,
reached 100% KV occupancy, completed without an error, released KV back to 0%, and left `/health` at
200 with no new fault marker. `p150x2` is therefore promoted as the launcher default.
`doc/context_contract.json` is the machine-readable source of truth.

The retained promotion run also exercised a real coding-agent round trip: `pool` invoked `pwd` in
`/tmp/laguna-p150x2-pool-smoke-20260821T130000`, received exit 0 and the exact path, returned
`LAGUNA_POOL_OK`, and exited 0 in 21.78 s. Final health was 200; the fault scan found only the already
documented allocator advisory.

### Production cold-salted latency sweep (2026-08-22)

The latest complete curve uses the production `p150x2` defaults: streaming prefill and qualified prefix
caching are both enabled. Every row supplied a unique cache salt, and the final hit and cached-token
counters were zero, so these are cold requests rather than cache hits. The OpenAI chat benchmark used
OSL 512, concurrency 1, temperature 0, ignore EOS, seed 1234, no benchmark warmup, and no ready-check
probe. Requested random ISL and actual server-counted prompt tokens differ because text and chat-template
round trips retokenize the input.

| Requested ISL | Actual prompt | TTFT | TPOT | E2EL | Decode tok/s/user | Aggregate output tok/s | TTFT speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 82 | 0.215 s | 50.07 ms | 25.801 s | 19.97 | 19.844 | 0.959x |
| 1,024 | 1,066 | 2.213 s | 50.45 ms | 27.995 s | 19.82 | 18.289 | 1.001x |
| 2,048 | 1,939 | 2.563 s | 50.49 ms | 28.366 s | 19.80 | 18.050 | 1.002x |
| 4,096 | 4,138 | 8.984 s | 50.56 ms | 34.822 s | 19.78 | 14.703 | 1.001x |
| 8,192 | 8,234 | 19.680 s | 50.71 ms | 45.592 s | 19.72 | 11.230 | 1.002x |
| 16,384 | 16,426 | 33.931 s | 51.00 ms | 59.993 s | 19.61 | 8.534 | **1.377x** |
| 32,768 | 32,810 | 67.812 s | 51.60 ms | 94.178 s | 19.38 | 5.437 | **1.801x** |
| 65,536 | 65,578 | 156.630 s | 52.77 ms | 183.595 s | 18.95 | 2.789 | **2.292x** |
| 130,048 | 130,090 | 380.812 s | 55.10 ms | 408.967 s | 18.15 | 1.252 | 1.002x |

Speedup is the 2026-08-21 TTFT divided by current TTFT. Streaming cuts cold TTFT by 1.377x at 16K,
1.801x at 32K, and 2.292x at 65K while keeping TPOT within 0.035% across the sweep. At 130K, both paths
compute the same aligned 131,072 rows, so latency is unchanged. All 9/9 requests completed with exactly
512 output tokens; final health was 200, with no prefix resume, serving-time compile/cache miss, request
error, or critical fault. This is one sample per point, not a variance study. Full precision, gates,
and provenance are in the [2026-08-22 sweep notes](doc/vllm_integration/p150x2_latency_sweep_20260822.md)
and [TSV](doc/vllm_integration/p150x2_latency_sweep_20260822.tsv). The
[2026-08-21 pre-streaming cache-off baseline](doc/vllm_integration/p150x2_latency_sweep_20260821.md)
is retained as historical evidence.

### Default D2 streaming prefill

`p150x2` now defaults to `TT_LAGUNA_STREAMING_PREFILL=1`. The adapter warms a finite compute ladder
of 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, and 8,192 rows, plus one canonical long-stream case at
a nonzero absolute position, before decode trace capture. A short cold request up to 8,192 tokens uses
the smallest fitting shape. A longer request runs chunk-major through the full decoder stack in
complete 8,192-token chunks; its final partial chunk and every later scheduler continuation use the
canonical 8,192-query geometry. D1 and D4 retain the monolithic ladder. The explicit p150x2
`TT_LAGUNA_STREAMING_PREFILL=0` rollback also retains it, is unqualified, and therefore requires
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`; the resolved config and startup log identify that state.

The full-stack 16,400-token hardware gate computed 24,576 rows instead of the former 32,768-row
bucket, matched hidden and logits at PCC 1.0 with the same argmax and top-10, and improved median warm
prefill from 46.804378 s to 33.622952 s (**1.392037×**) without a program-cache miss. Method, memory,
and artifacts are recorded in
[p150x2 hybrid-KV and streaming-prefill qualification](doc/vllm_integration/p150x2_hybrid_kv_qualification_20260822.md).

Two decode-side candidates were also closed out explicitly. The fused single-device QKV split was
numerically exact but 0.83% slower and is not wired into the p150x2 multichip path, so
`TT_LAGUNA_FUSE_QKV_DECODE=0` remains the default; see the
[fused decode-QKV disposition](doc/vllm_integration/fused_qkv_decode_qualification.md). Compact
cross-ASIC argmax was exact, but measured 25.334 ms versus 10.755 ms for the existing B=1 greedy path,
so it also remains default-off. Its executable contract and retained measurements are in
[`test_compact_argmax_hardware.py`](tests/test_compact_argmax_hardware.py).

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

### Experimental cache-off hybrid KV

The production default remains **uniform KV with prefix caching on**. A separate fail-closed
`p150x2` alternative is available only with `TT_LAGUNA_HYBRID_KV=1`,
`TT_LAGUNA_PREFIX_CACHE=0`, `LAGUNA_MAX_NUM_SEQS=1`, and
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`. It validates the exact 40-layer attention pattern and maps
the 10 full-attention plus 30 sliding-window layers into four ten-layer block-table groups. Equal slots
across the groups alias exactly ten physical K/V tensor pairs while retaining forty logical cache
views; prefix caching and hybrid KV cannot be combined.

On the matched 131,072-token hardware runs, hybrid KV residency was 1,634.3 MiB versus 5,445.3 MiB
uniform: a 3,811.0 MiB (**70.0%**) saving. Deterministic output and usage were byte-identical in the
retained comparison, and matched 1,066+128 and 8,234+32 requests showed no latency regression. This
qualifies the path as an experimental cache-off alternative, not as the production default. Exact
layout, pool sizing, measurements, commands, and limitations are in the
[p150x2 hybrid-KV and streaming-prefill qualification](doc/vllm_integration/p150x2_hybrid_kv_qualification_20260822.md).

### Experimental sparse-MoE token dispatch

The production decoder includes a fail-closed, default-off token-dispatch path selected by
`TT_LAGUNA_MOE_TOKEN_DISPATCH=1` with `TT_LAGUNA_PREFIX_CACHE=0` and
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`. The launcher requires p150x2, one sequence, streaming
prefill on, uniform KV, and every DFlash/context/multi-sequence/tile-sparse experiment off. It routes
only the selected token rows to the existing stacked `exp_gate_up` / `exp_down` tensors, preserves
the qualified 256-row router and shared-expert numerics, applies routing weights in the fused
post-combine reduction, and falls back outside its exact D2/bucket/precision envelope.

The production-stacked layer-1 / 8,192-token hardware gate matched the established path at PCC
0.9997239543 and improved warm MoE latency from 255.861025 ms to 53.281487 ms (**4.802062x**), with
stable program-cache cardinality and lower peak memory than the gate. Its first cold compile cost
53.246 s, and cumulative 39-layer boot/residency, cross-bucket/layer reuse, and full-model
logits/tokens have not yet been qualified. The launcher therefore reports `qualified=0` and keeps the
feature off by default. Results and the literal remaining promotion protocol are in the
[p150x2 token-dispatch qualification](doc/vllm_integration/p150x2_token_dispatch_qualification_20260822.md).

### Experimental DFlash controller

The five-layer DFlash draft core and served greedy controller are implemented behind the default-off
`TT_LAGUNA_DFLASH=1` envelope. The launcher rejects untested stacking with hybrid KV, sparse MoE,
context probes, multi-sequence serving, prefix caching, or speculative decode. Target verification
and committed output tokens were exact. Draft-token
checking requires literal equality for unique official BF16 maxima; for an exact official BF16 tie,
the TT token must belong to the exact maximum set. The separate deterministic non-tied hardware vector
remains exact 15/15.

The real-context serving gate accepted one draft and committed two tokens. Its 188.880 ms target-verify
median is already at least 94.440 ms per committed token before draft cost, versus about 50.5 ms for
the qualified baseline. DFlash therefore fails the no-performance-regression gate and remains
correctness-scoped, experimental, and default-off. The accuracy contract, block-tail fallback, hardware
evidence, and disposition are in the [DFlash reference contract](doc/dflash_reference.md).

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

### Fail-closed experimental envelopes

These modes do not change the production defaults and require an explicit experimental
acknowledgement. The two-sequence, 65,536-token-per-request probe is available as:

```bash
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 \
  TT_LAGUNA_MULTI_SEQ_POOL=1 TT_LAGUNA_PREFIX_CACHE=0 \
  LAGUNA_MAX_MODEL_LEN=65536 LAGUNA_MAX_NUM_SEQS=2 \
  LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1 \
  "$MODEL_DIR/serve_vllm.sh" config
```

Its hardware probe passed exact output tokens for two 8,192+64 requests issued first sequentially and
then concurrently. The requests overlapped, concurrent wall time was 24.086 s versus 27.410 s
sequential (**1.138×**), and the worst concurrent/sequential TPOT ratio was 0.9991. This remains an
opt-in probe pending the final promotion audit; it is not the production max-sequences setting.

The separate 262,144-token probe requires the experimental cache-off hybrid layout:

```bash
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 \
  TT_LAGUNA_CONTEXT_PROBE=1 TT_LAGUNA_HYBRID_KV=1 TT_LAGUNA_PREFIX_CACHE=0 \
  LAGUNA_MAX_MODEL_LEN=262144 LAGUNA_MAX_NUM_SEQS=1 \
  LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1 \
  "$MODEL_DIR/serve_vllm.sh" config
```

This probe is rejected and unqualified. Two deterministic 131,136+16 boundary repeats produced exact
matching output tokens, but the 262,112+32 exact-cap request exceeded the 1,200-second latency budget
after reaching 97.0934% KV occupancy and was aborted. The launcher reports
`context_status=experimental_262144_probe` and rejects every other over-cap combination; this command
is retained for diagnosis only and does not expand the 131,072-token production contract.

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
| `uv pip install -e vllm_ext` | Laguna-scoped canonical prefix admission, exact hybrid-KV scheduler/pool hooks, capability handling, and the newline-tolerant `poolside_v1` parser required for `auto` tool-calling. |

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
`TT_VISIBLE_DEVICES`. The only over-cap exceptions are the two exact experimental envelopes above;
each has its own boolean guard, exact geometry checks, and required
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`. For D2/D4, `TT_LAGUNA_CCL_TOPOLOGY=linear|ring` and
`TT_LAGUNA_CCL_NUM_LINKS=1|2` select a previously qualified combination; the launcher derives and
checks the matching `LAGUNA_FABRIC_CONFIG`. The initial trace reservation is 1.5 GB. Reduce it only
when the measured memory gate requires it, and repeat trace capture/replay after every change.
The qualified post-trace margins are `TT_LAGUNA_MIN_DRAM_FREE_FRACTION=0.10` and
`TT_LAGUNA_MIN_CONTIGUOUS_MIB=128`; changing either requires
`LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1` and makes the run diagnostic-only.

On the dual-P150 qualification host, selecting one ASIC is a custom cluster, so `p150` automatically
exports the checked-in 1×1 `TT_MESH_GRAPH_DESC_PATH`. D2/D4 remove that singleton descriptor; an inherited custom
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

- **Production keeps `TT_LAGUNA_HYBRID_KV=0`.** The explicit experimental `=1` path preserves the
  validated four-group/ten-pair alias layout, requires cache-off single-sequence serving plus the
  experimental acknowledgement, and fails closed on any model, scheduler, block-pool, or plugin
  mismatch. It is hardware-qualified as the 70%-smaller cache-off alternative but is not promoted.
- Production `p150` and `p150x2` remain single-user profiles (`--max-num-seqs 1`). The explicit
  65,536-token `TT_LAGUNA_MULTI_SEQ_POOL=1` probe permits exactly two `p150x2` sequences with uniform
  KV and both caches off; it is not yet a production profile. The established `p150x4` regression
  profile retains `--max-num-seqs 8`; acceptance performance is still measured at concurrency 1.
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
- **Warmup:** every admitted D2 compute shape is compiled before the decode trace is captured.
  `warmup_model_prefill` warms the finite 32-through-8,192 ladder and one canonical long-stream case;
  later chunks reuse those programs with absolute positions supplied as runtime data. A private
  completion latch suppresses the plugin's redundant second pass. Do not set
  `TT_LAGUNA_PREFILL_WARM_CAP` for serving; a cap below the required 8,192 outer chunk makes long D2
  requests fail before device execution.
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
  modified. Streaming pads only the current tail chunk, so private prefill tables and shared RoPE cover
  `max_model_len` rounded once to the next 8,192-token boundary. The 131,072-token profile is already
  aligned and therefore uses a 131,072-token horizon, not the former doubled horizon; a near-cap suffix
  stays addressable without exposing a larger serving context.
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
- When `TT_LAGUNA_HYBRID_KV=1`, the same general plugin admits only Laguna's exact cache-off D2
  scheduler envelope, applies the exact block-pool floor, and preserves four distinct page-table
  groups while aliasing their equal slots onto ten K/V tensor pairs. When `VLLM_PLUGINS` is unset,
  vLLM auto-loads the installed plugins. Any explicit allowlist must contain the exact `tt`,
  `tt_model_registry`, and `laguna_tt_ext` entries for every Laguna launch; the local extension owns
  the qualified tool parser in every profile as well as the cache hooks. An incomplete or lookalike
  entry is rejected.
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

`pool` is Poolside's separate terminal agent; it is not bundled with this repository. Follow the
[official `pool` installation instructions](https://github.com/poolsideai/pool), verify the install with
`pool --version`, and then use it with this server in **standalone mode**.

```bash
# 1. Serve Laguna (above), confirm /v1/models returns poolside/Laguna-XS-2.1.
# 2. Install pool using its official instructions, then confirm it is available:
pool --version
# 3. Point pool at the local endpoint — POOLSIDE_STANDALONE_BASE_URL is the mode switch:
export POOLSIDE_STANDALONE_BASE_URL=http://localhost:8000/v1    # note the /v1 (without it: 404 default-agent)
export POOLSIDE_API_KEY=EMPTY                                   # any value; local server ignores it
export POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1"
export POOLSIDE_STANDALONE_CONTEXT_LENGTH=131072                # recommended p150x2/p150x4
# 4a. Interactive:      cd <project> && pool
# 4b. One-shot (CI):    pool exec --unsafe-auto-allow --sandbox disabled -p "…task…"
```
Tool-calling round-trips automatically through `poolside_v1`. The production default can reuse only
complete 8,192-token prompt checkpoints; it recomputes shorter/partial or decode-produced transcript
segments. Prefer one active `pool` session because the recommended D2 profile is intentionally
single-user.

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
- `tests/test_prefill_buckets.py`, `tests/test_prefill_runtime.py`,
  `tests/test_generator_vllm_prefix_resume.py`, and `tests/test_prefill_page_table.py` — device-free
  finite streaming ladder, canonical long-tail planning, single-rounded capacity, suffix slicing,
  host-rebased fill, runtime SDPA/RoPE inputs, private scratch/split tables, terminal selection, and
  one-pass warmup contracts.
- `tests/test_streaming_prefill_hardware.py` — opt-in D2 full-stack 16,400-token streamed-versus-legacy
  PCC, logits, top-k, program-cache, memory, and latency gate.
- `vllm_ext/tests/test_prefix_cache_quantum.py` — ownership-safe canonical admission, prompt-only hash
  insertion, poisoning protection, metric correction, and fail-closed scheduler/KV geometry.
- `tt/kv_grouping.py`, `tests/test_hybrid_kv_grouping.py`, and `vllm_ext/tests/test_hybrid_kv.py` —
  exact four-group/ten-pair layer aliases, per-group page tables, pool sizing, plugin admission, and
  fail-closed hybrid feature combinations.
- `tests/test_prefix_cache_hardware.py` — p150x2 flexible-SDPA, indexed-RoPE, and representative
  full/sliding layer accuracy with program-cache misses forbidden across changed offsets.
- `tests/prefix_cache_qualification.py` — cache-off oracle and cache-on raw-token exactness, admission,
  poison-order, decode-boundary, Prometheus, health, and 32K/65K latency gates.
- `tests/serving_envelope_qualification.py` and `tests/test_serving_envelope_qualification.py` — live
  two-sequence and 262K exact-token/performance/fault-scan harness plus its device-free contracts.
- `tests/smoke_full_model.py --profile <profile> --enforce-memory-margin` — all-40-layer build,
  uniform KV, prefill, trace replay, and measured DRAM margin.
- `tests/full_model_checks.py` — full-model top-1/5/100 and autoregressive checks against the AIME24
  reference. Acceptance bars are top-1 ≥ 0.90, top-5 ≥ 0.98, and top-100 = 1.00.

See `doc/bringup_profiles.md` for the cold-serving, prefix-cache, performance, and promotion gates. Passing
unit or layer tests alone does not qualify a profile.
