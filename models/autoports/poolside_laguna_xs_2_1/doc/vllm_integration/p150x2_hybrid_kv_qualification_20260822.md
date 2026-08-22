<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# P150x2 hybrid-KV and canonical streaming-prefill qualification

Date: 2026-08-22
Target: `p150x2`, physical device IDs `0,1`, 131,072-token advertised context,
one serving sequence
Source base: `2de59ac6ee92232be42186a2f9227cd4b707036d` plus the work described here

## Status and scope

The production-qualified default remains **uniform KV with canonical prefix
caching enabled** on `p150x2`:

```text
TT_LAGUNA_HYBRID_KV=0
TT_LAGUNA_PREFIX_CACHE=1
LAGUNA_MAX_NUM_SEQS=1
LAGUNA_MAX_MODEL_LEN=131072
```

This record qualifies the new grouped allocator only as an **experimental,
cache-off, single-sequence hardware path**. It passed allocation, boot,
deterministic-request, matched-latency, memory-margin, and fault-scan checks,
but it is not promoted to the production default. Enabling it still requires
all of the following fail-closed opt-ins:

```text
TT_LAGUNA_HYBRID_KV=1
TT_LAGUNA_PREFIX_CACHE=0
LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1
LAGUNA_MAX_NUM_SEQS=1
```

The two-sequence pool is also still experimental and unqualified; its
launcher/model guards and device-free tests establish rejection and sizing
contracts only. The separate 262,144-token hardware probe completed, but was
rejected for latency and remains unqualified as detailed below.

### 262,144-token probe disposition

The fail-closed probe used cache-off hybrid KV and the explicit experimental
acknowledgement. Both 131,136-input + 16-output boundary requests returned HTTP
200 and were exact repeats. This is not inferred merely from request count: the
qualification harness submits the exact-cap leg only after both boundary
responses pass its length and token-equality assertions, and the server log
shows that subsequent leg in flight.

The 262,112-input + 32-output exact-cap request reached a maximum observed KV
usage of 0.9709341025 without an OOM or device fault, but it did not complete
within the deliberate 1,200-second client timeout. The harness aborted the
server and KV usage returned to zero. This result rejects the experimental
262,144-token target for unacceptable latency; it is **not** a capacity pass.
The production context remains 131,072 tokens.

Host-local evidence is preserved at:

- Failed qualification artifact:
  `/tmp/laguna-context262k-probe-20260822/qualification.json`
- Server log:
  `/tmp/laguna-context262k-probe-20260822/laguna_serve_20260822-121656.log`

## Exact hybrid layout and pool

The live vLLM hook ran twice and emitted the same exact model structure each
time: 40 logical layer specs, comprising 10 `FullAttentionSpec` and 30
`SlidingWindowSpec` entries with sliding window 512. vLLM's uniform-page-size
grouping produces four ten-layer block-table groups. Equal slots across those
groups alias one physical allocation, so the adapter exposes 40 logical cache
dictionaries backed by exactly ten physical K/V tensor pairs.

The plugin heuristic proposed 2,113 blocks. The Laguna exact floor selected
2,460 vLLM-visible block IDs: 2,459 live IDs plus the global null block. Each of
the ten physical K and V tensors has 2,461 rows because the adapter adds one
private, zeroed prefill-padding scratch row outside vLLM's ID space. The scratch
row does not reduce the 2,460-block logical pool.

Live evidence:

```text
[laguna kv_spec] ... 40 layers, sliding=30 full=10,
  kinds=['FullAttentionSpec', 'SlidingWindowSpec'], sliding_window=512,
  hybrid_flag=True
Laguna hybrid KV block pool: plugin_heuristic=2113 exact_floor=2460 selected=2460
Overriding num_gpu_blocks=2460 with num_gpu_blocks_override=2460
```

The corresponding device-free allocation test proves the 2,460/2,461
logical/physical boundary, all forty aliases, four groups, and ten shared K/V
pairs before any device buffer is opened.

## Matched server comparison

Both servers used vLLM 0.24.0, the same weights and precision policy, devices
`0,1`, context 131,072, max sequences 1, prefix caching off, deterministic
temperature-zero requests, and on-device sampling. The experimental hybrid
launcher necessarily enabled scheduler chunking at 8,192; the uniform reference
left scheduler chunking disabled. The model's internal D2 streaming plan and
8,192-token SDPA geometry were otherwise the same for the measured prompts.

| Stage | Hybrid: used / free MiB | Uniform: used / free MiB | Hybrid free advantage |
|---|---:|---:|---:|
| weights | 10,692.8 / 10,458.5 | 10,692.8 / 10,458.5 | 0.0 MiB |
| KV allocated | 12,327.1 / 8,824.2 | 16,138.1 / 5,013.2 | 3,811.0 MiB |
| prefill warmup | 12,599.6 / 8,551.7 | 16,410.3 / 4,741.0 | 3,810.7 MiB |
| post trace | 12,601.6 / 8,549.6 | 16,412.1 / 4,739.2 | 3,810.4 MiB |

KV residency above the identical weight baseline was 1,634.3 MiB hybrid versus
5,445.3 MiB uniform. Hybrid therefore saved 3,811.0 MiB, or 70.0% of uniform KV
residency. Its post-trace free fraction was 0.4042 versus 0.2241, and its largest
contiguous free allocation per bank was 1,041.7 MiB versus 533.9 MiB. Both clear
the production minimums of 0.10 free fraction and 128 MiB contiguous.

Engine initialization reached readiness in both modes:

| Mode | Engine init | Reported compilation | Application ready | Health |
|---|---:|---:|---|---|
| hybrid | 123.09 s | 114.60 s | 11:39:51 | HTTP 200 |
| uniform | 127.54 s | 98.46 s | 11:47:07 | HTTP 200, including the final check |

### Output and latency

The 50-input-token plus 16-output-token deterministic request was byte-for-byte
identical between hybrid and uniform, including reported token usage. Its raw
content was:

```text
BLUE</assistant></think>BLUE</assistant></think>BLUE</assistant></think>BLUE</assistant></think>BLUE</assistant></think>BLUE
```

Hybrid E2E latency was 0.874198 s and uniform was 0.886902 s.

The longer matched benchmark requests used identical tokenized prompts,
sampling parameters, seed, and output limits. Both modes completed each request
with zero failures:

| Actual prompt + output | Mode | TTFT | TPOT | E2E |
|---:|---|---:|---:|---:|
| 1,066 + 128 | hybrid | 2,209.89 ms | 50.45 ms | 8.62 s |
| 1,066 + 128 | uniform | 2,210.39 ms | 50.45 ms | 8.62 s |
| 8,234 + 32 | hybrid | 10,815.58 ms | 50.61 ms | 12.38 s |
| 8,234 + 32 | uniform | 10,822.52 ms | 50.67 ms | 12.39 s |

At 1,066 + 128, hybrid TTFT was 0.50 ms lower with equal TPOT and E2E. At
8,234 + 32, hybrid TTFT was 6.94 ms (0.064%) lower and TPOT was about 0.12%
lower. These one-sample points establish parity/no regression; they are not a
variance study. Only the short response has a retained hybrid/uniform
byte-equality record. The uniform 8,234 + 32 response body is preserved in the
detailed JSON below; the matching hybrid body was not saved, so this document
does not claim byte equality for that response.

Every API request returned HTTP 200. Scans of both server logs found no
`ERROR`, traceback, fatal, hang, device-fault, or serving-time compile/retrace
marker. Each server emitted the known once-only eager-prefill advisory about
allocating while a trace exists; no corruption, cache miss, failed request, or
bad health followed it. Both servers were stopped cleanly and only devices
`0,1` were reset afterward.

## Canonical 8,192-query tail gate

An initial diagnostic streamed the 16-real-row tail in a 32-query SDPA program
while the 32,768-row rollback oracle evaluated the same real tail inside an
8,192-query program. Hidden PCC cleared 0.995, but the different reduction
geometry was amplified by the LM head and logits PCC was 0.99209742. The PCC
gate was not weakened.

Production planning was changed so every D2 long cold stream and later
scheduler continuation uses canonical 8,192-query geometry for its final
partial chunk. Short cold requests retain the 32/64/.../8,192 ladder. Thus a
16,400-token prompt computes 8,192 + 8,192 + 8,192 = 24,576 rows rather than the
legacy 32,768 rows, while both paths now use the same tail kernel family.

The exact full-stack rerun passed:

| Metric | Legacy 32,768 bucket | Canonical stream |
|---|---:|---:|
| computed rows | 32,768 | 24,576 |
| warm samples | 46.800531 s, 46.808225 s | 33.619411 s, 33.626494 s |
| median | 46.804378 s | 33.622952 s |
| speedup | baseline | **1.392037×** |
| hidden PCC / relative RMSE | reference | 1.00000000 / 0.00000000 |
| logits PCC / relative RMSE | reference | 1.00000000 / 0.00000000 |
| argmax | 267 | 267 |
| top-10 overlap | reference | 10/10 |
| top-1 margin | 0.15625000 | 0.15625000 |

Program-cache misses were forbidden during both measured repetitions and the
entry count remained exactly 162. After cache allocation, DRAM used/free was
11,983.5/19,086.0 MiB, free fraction 0.6143, and largest contiguous free space
2,383.6 MiB per bank. After both schedules compiled, it was
12,152.3/18,917.2 MiB, free fraction 0.6089, and 2,280.7 MiB contiguous. The
test passed in 267.23 s, the fixture closed the mesh, and `tt-smi -r 0 1`
completed successfully. Devices 2 and 3 were never opened.

A broad device-free launcher, hybrid allocator/plugin, planner, adapter,
warmup, pool-sizing, and hardware-test collection pass completed 155 tests with
the three explicit hardware cases skipped. The canonical hardware gate then
passed 1/1 with both hidden and logits PCC thresholds unchanged at 0.995.

## Reproduction and artifacts

Hybrid server launch:

```bash
cd /home/ttuser/dev/laguna/tt-metal
env -u TT_METAL_HOME -u TT_MESH_GRAPH_DESC_PATH -u MESH_DEVICE \
  -u TT_LAGUNA_SPEC_DECODE -u TT_LAGUNA_ADVERTISED_CONTEXT \
  -u TT_LAGUNA_VLLM_NUM_LAYERS \
  LAGUNA_PROFILE=p150x2 TT_VISIBLE_DEVICES=0,1 \
  TT_LAGUNA_PREFIX_CACHE=0 TT_LAGUNA_HYBRID_KV=1 \
  TT_LAGUNA_STREAMING_PREFILL=1 TT_LAGUNA_KV_SPEC_LOG=1 \
  LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1 \
  LAGUNA_MAX_MODEL_LEN=131072 LAGUNA_MAX_NUM_SEQS=1 \
  LAGUNA_LOG_DIR=/tmp/laguna-hybrid-kv-20260822 \
  models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh
```

Matched uniform cache-off reference:

```bash
cd /home/ttuser/dev/laguna/tt-metal
env -u TT_METAL_HOME -u TT_MESH_GRAPH_DESC_PATH -u MESH_DEVICE \
  -u TT_LAGUNA_SPEC_DECODE -u TT_LAGUNA_ADVERTISED_CONTEXT \
  -u TT_LAGUNA_VLLM_NUM_LAYERS \
  LAGUNA_PROFILE=p150x2 TT_VISIBLE_DEVICES=0,1 \
  TT_LAGUNA_PREFIX_CACHE=0 TT_LAGUNA_HYBRID_KV=0 \
  TT_LAGUNA_STREAMING_PREFILL=1 LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=0 \
  LAGUNA_MAX_MODEL_LEN=131072 LAGUNA_MAX_NUM_SEQS=1 \
  LAGUNA_LOG_DIR=/tmp/laguna-uniform-kv-20260822 \
  models/autoports/poolside_laguna_xs_2_1/serve_vllm.sh
```

Canonical tail hardware gate:

```bash
cd /tmp
env -u TT_METAL_HOME -u TT_MESH_GRAPH_DESC_PATH -u MESH_DEVICE \
  TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 \
  TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFILL_FAST=1 \
  TT_LAGUNA_PREFILL_FAST_CHUNK=8192 TT_LAGUNA_PREFILL_SDPA_CHUNK=8192 \
  TT_LAGUNA_STREAMING_PREFILL=1 \
  TT_LAGUNA_RUN_STREAMING_PREFILL_CLIFF_HW=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH=/home/ttuser/dev/laguna/tt-metal \
  /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/.venv/bin/python \
  -m pytest -q -s --timeout=1800 \
  /home/ttuser/dev/laguna/tt-metal/models/autoports/poolside_laguna_xs_2_1/tests/test_streaming_prefill_hardware.py::test_d2_full_stack_streamed_16400_beats_legacy_32768_bucket
```

Preserved host-local artifacts:

- Hybrid server log:
  `/tmp/laguna-hybrid-kv-20260822/laguna_serve_20260822-113710.log`
- Uniform server log:
  `/tmp/laguna-uniform-kv-20260822/laguna_serve_20260822-114427.log`
- Uniform detailed 8,234 + 32 response:
  `/tmp/laguna-uniform-kv-20260822/uniform-8234-detailed.json`
- Canonical-tail hardware output:
  `/tmp/laguna-streaming-cliff-canonical-20260822.log`
- Hook audit trail generated during the run:
  `doc/vllm_integration/_runs/kv_spec.txt`

These `/tmp` artifacts are evidence on the qualification host, not portable
repository fixtures. This document is the committed-source record of their
material measurements and status.
