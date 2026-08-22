<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# P150 profile qualification

This is the acceptance runbook for Laguna-XS-2.1. It separates configuration from evidence: a
profile existing in code does not make it qualified. As of 2026-08-21, D1 is rejected because the
full model cannot allocate its LM head even with a 4,096-token cap. D2 ring/two-link is the recommended
configuration after topology, representative correctness, full 131K uniform-KV smoke, and trace replay;
cold live API, decode-performance, exact-cap, and canonical prefix-cache gates pass. D2 is the selected
default.

## Profiles and decision

One `P150` is one Blackhole ASIC/device enumerated by `tt-smi`. A D2 pair must be two such devices
that pass the topology test together; a board label is not a substitute for enumerated device IDs.
On this host, `p150x2` is one physical P300 card containing two enumerated Blackhole ASICs.

| Profile | Mesh | Uniform-KV context | Serving concurrency | Fabric / CCL | Role |
|---|---:|---:|---:|---|---|
| `p150` | 1×1 (D1) | 65,536 configured | 1 | disabled / no collectives | Rejected: weights OOM at LM head |
| `p150x2` | 1×2 (D2) | 131,072 | 1 | ring, two links | Qualified and selected default |
| `p150x4` | 1×4 (D4) | 131,072 | 8 configured; test at C1 | ring, two links | Regression profile |

D1 missed the capacity gate before KV allocation, so no D1 performance run is useful for selection.
Preserve that failure rather than lowering its configured context: the failure occurs in weights and
is context-independent. Qualify D2 at 131,072 against the same 15 tokens/s at ISL 1K and 10 tokens/s
at ISL 32K gates. D2 passes those decode gates plus the exact 131,072-token request, so it replaces
`p150x4` as the launcher default. D4 remains the explicit regression profile.

The HF configuration declares 262,144 tokens. That value is not a serving target in this bring-up.
Do not raise a profile cap based on component-level addressability.

Status shorthand in this document: G0 is mesh/open-close plus eager/traced collective qualification;
G1 is representative packed decoder correctness. Neither gate implies that full-model serving is
qualified.

## Fixed invariants

- Use uniform BFP8 KV: `TT_LAGUNA_HYBRID_KV=0`. Hybrid KV is unsafe in the current vLLM/plugin
  integration because its grouped allocator does not preserve Laguna's 10-full/30-sliding mapping.
  The launcher intentionally rejects a nonzero value.
- Production p150x2 prefix caching defaults on with `TT_LAGUNA_PREFIX_CACHE=1`. An explicit `=0` is the
  fail-closed operator rollback and never requires `LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES=1`. `p150` and
  `p150x4` remain cache-off and unqualified for prefix caching.
- The qualified scheduler/KV envelope is max sequences 1, scheduler chunked prefill disabled, model
  internal 8,192-token pipelined prefill enabled, physical KV blocks of 64, one uniform
  `FullAttentionSpec` cache group, and no speculative decoding, lookahead, encoder requests, hybrid KV,
  or external KV connector. Launcher and engine-side validators fail closed on a mismatch.
- Cache insertion and admission use only complete 8,192-token **prompt** checkpoints. Candidate hits are
  floored to that quantum before block references are incremented, so rejected tail blocks remain fresh
  and writable. Partial prompt chunks and generated decode KV are never allowed to poison a canonical
  checkpoint.
- A resumed prefill consumes vLLM's absolute range `[start_pos, prompt_lens)`: upload only that token
  suffix, choose the compute bucket from `prompt_lens-start_pos`, and retain `start_pos` as the absolute
  KV/RoPE offset. The admitted start is 8K-aligned; the adapter pads the fresh tail to canonical pipeline
  geometry.
- Each uniform-KV layer reserves one physical block outside vLLM's logical block-id range as private
  prefill scratch. Separate persistent prefill-only tables map whole attention-padding blocks to the
  zeroed scratch block and paged-fill padding to the kernel's `-1` write-skip sentinel. This prevents
  both physical-block-0 corruption and writer races on scratch. Never rewrite the scheduler/decode
  table or count scratch as logical capacity.
- The private prefill page table and shared RoPE tables cover `max_model_len + largest_bucket`
  (262,144 positions for the 131,072-token D2 profile). This is internal padded-compute addressability,
  not an advertised context extension.
- The complete prefill bucket ladder is warmed in the plugin's first compile phase. Laguna's private
  `_prefill_programs_warmed` latch makes the plugin's second call a no-op except for restoring its
  public flag; the runtime-offset slots required by canonical suffixes are preallocated. A qualification
  log must show only one full-model prefill warmup pass.
- Last-real-token selection uses a persistent per-bucket one-hot input and persistent one-row output;
  its matmul is compiled during that one ladder pass. A resumed prefill selects row
  `prompt_lens-start_pos-1`, not the absolute prompt-end row, without creating a per-request
  selector/output buffer.
- The adapter host-rebases each persistent fill row so column zero represents that request's scheduled
  start. Cold and resumed serving avoid an absolute-start device slice; the internal pipeline uses only
  fixed relative chunk slices. RoPE gathers absolute positions into persistent outputs, and SDPA consumes
  a runtime start tensor. After all shapes and the decode trace are built, program-cache misses are
  forbidden. Cold token and activation outputs can still allocate; this is not a claim that eager
  prefill is globally allocation-free under a resident trace.
- Start with a 1,500,000,000-byte trace region. A smaller value is allowed only when measured DRAM
  margin requires it, and the full trace capture/replay and performance suite must then be repeated.
- Keep the launcher memory gates at `TT_LAGUNA_MIN_DRAM_FREE_FRACTION=0.10` and
  `TT_LAGUNA_MIN_CONTIGUOUS_MIB=128`. Changing either requires the diagnostic-only experimental
  override acknowledgement and cannot qualify a profile.
- Use vLLM 0.24.0 and plugin commit `c127c17d80d66ee83d23064d3a62ac844a1170de` from
  `requirements.txt`. Do not patch site-packages or follow a moving plugin branch.
- Leave `TT_METAL_HOME` unset when using the model's self-contained environment.
- For D1, use the checked-in `tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto`
  and `TT_LAGUNA_DECODE_SDPA_PC=0`. The launcher sets both. `TT_LAGUNA_DECODE_SDPA_PC=1` is the
  known-inaccurate custom k64 long-position diagnostic on D1, not an acceptance setting.
- D2/D4 use normal mesh discovery and must not inherit the singleton descriptor. Custom graph
  descriptors require the explicit diagnostic opt-in `LAGUNA_ALLOW_CUSTOM_MESH_GRAPH_DESC=1`.

## 1. Record the environment

Create a timestamped artifact directory outside the source tree or under an intentionally managed
qualification directory. Save the tt-metal commit and dirty diff, `tt-smi` inventory/topology,
firmware and UMD versions, host kernel, vLLM version, plugin commit, all environment overrides, and
the output of the launcher's read-only configuration check.

```bash
REPO_ROOT=/path/to/tt-metal
MODEL_DIR="$REPO_ROOT/models/autoports/poolside_laguna_xs_2_1"
ARTIFACT_DIR=/path/to/laguna-qualification
LAGUNA_PROFILE=p150 "$MODEL_DIR/serve_vllm.sh" config
TT_VISIBLE_DEVICES=0,1 LAGUNA_PROFILE=p150x2 "$MODEL_DIR/serve_vllm.sh" config
```

`TT_VISIBLE_DEVICES` defaults to `0`, `0,1`, or `0,1,2,3`, but identifiers are host-specific. Resolve
the intended ASICs with `tt-smi`; never infer a physical D2 pair solely from adjacent numbers.

## 2. Qualify mesh and CCL in fresh processes

Reset before the first attempt and between every device selection, mesh size, topology, or link-count
attempt. Each invocation below is a new Python process. Never open D1 and D2 sequentially in one
interpreter: a previous fabric mesh can leave Ethernet state that turns the next result into a false
failure or a hang.

```bash
tt-smi -r all
cd /tmp
env -u TT_METAL_HOME TT_VISIBLE_DEVICES=0 PYTHONPATH="$REPO_ROOT" \
  "$MODEL_DIR/.venv/bin/python" -m \
  models.autoports.poolside_laguna_xs_2_1.tests.qualify_topology \
  --profile p150 --open-cycles 3 --trace-replays 3
```

For every plausible physical D2 pair, run all four supported CCL candidates separately, resetting
before each invocation:

```bash
tt-smi -r all
cd /tmp
env -u TT_METAL_HOME TT_VISIBLE_DEVICES=0,1 PYTHONPATH="$REPO_ROOT" \
  "$MODEL_DIR/.venv/bin/python" -m \
  models.autoports.poolside_laguna_xs_2_1.tests.qualify_topology \
  --profile p150x2 --topology linear --num-links 1 --open-cycles 3 --trace-replays 3
```

Repeat with `linear/2`, `ring/1`, and `ring/2`. A candidate passes only if all open/close cycles and
both eager and traced value checks complete without watcher, timeout, CCL, or reset errors. Select the
fastest passing combination only after later full-model measurements; topology success alone is not a
performance result. Save each emitted `QUALIFY_TOPOLOGY` JSON line, including failures.

Measured G0 status on this host (2026-08-21):

- D1 `TT_VISIBLE_DEVICES=0` plus the checked-in P150 1×1 descriptor: three open/close cycles with
  eager and traced checks passed.
- D2 `TT_VISIBLE_DEVICES=0,1`, ring/two-link: three cycles passed.
- D2 `TT_VISIBLE_DEVICES=2,3`, ring/two-link: three cycles passed.
- D2 `TT_VISIBLE_DEVICES=0,1`, linear/two-link: three cycles passed.
- D2 linear/one-link and ring/one-link have no recorded result. Ring/two-link on `0,1` is the selected
  bring-up configuration because it also passed the later correctness, capacity, trace, and standalone
  performance work; the selection remains host-specific.

These identifiers describe this host only. Preserve the raw output with the final qualification
artifact; the summary above is not portable to different wiring.

Reset again before loading model weights. Use `serve_vllm.sh stop` after a server run; it terminates
the process group and invokes `tt-smi -r all` when available.

## 3. Correctness and capacity

Run from `/tmp` with the intended profile selected before pytest collection. The packed gate/up path
is the production path and must remain enabled.

```bash
cd /tmp
env -u TT_METAL_HOME PYTHONPATH="$REPO_ROOT" LAGUNA_PROFILE=p150 TT_VISIBLE_DEVICES=0 \
  TT_LAGUNA_DECODE_SDPA_PC=0 \
  "$MODEL_DIR/.venv/bin/python" -m pytest -q \
  "$MODEL_DIR/tests/test_multichip_decoder.py"

env -u TT_METAL_HOME PYTHONPATH="$REPO_ROOT" TT_VISIBLE_DEVICES=0 \
  TT_LAGUNA_DECODE_SDPA_PC=0 \
  "$MODEL_DIR/.venv/bin/python" \
  "$MODEL_DIR/tests/smoke_full_model.py" \
  --profile p150 --max-seq-len 65536 --enforce-memory-margin

env -u TT_METAL_HOME PYTHONPATH="$REPO_ROOT" TT_VISIBLE_DEVICES=0 \
  TT_LAGUNA_DECODE_SDPA_PC=0 \
  "$MODEL_DIR/.venv/bin/python" "$MODEL_DIR/tests/full_model_checks.py" \
  prefill_autoreg --profile p150 --max-seq-len 65536 --enforce-memory-margin \
  --outdir "$ARTIFACT_DIR/full-model"

env -u TT_METAL_HOME PYTHONPATH="$REPO_ROOT" TT_VISIBLE_DEVICES=0 \
  TT_LAGUNA_DECODE_SDPA_PC=0 \
  "$MODEL_DIR/.venv/bin/python" "$MODEL_DIR/tests/full_model_checks.py" \
  teacher --profile p150 --enforce-memory-margin
```

Use `p150x2`, its qualified `TT_VISIBLE_DEVICES`, fabric/topology/link variables, and
`--max-seq-len 131072` for D2. `full_model_checks.py` requires the AIME24 reference at
`tests/reference_outputs/readiness_aime24_chat.refpt`; if it is not already present, produce it on a
suitable HF reference host with `tests/gen_aime_reference.py` and record its hash. Acceptance requires:

- Representative decoder layers 0 (full+dense), 1 (sliding+MoE), and 4 (full+MoE), prefill and
  traced decode, PCC at least 0.995 versus HF.
- Full-model prefill/teacher-forcing top-1 at least 0.90, top-5 at least 0.98, and top-100 exactly
  1.00, plus a nondegenerate autoregressive smoke. Use `tests/full_model_checks.py`; archive its JSON
  and completion artifacts.
- Valid non-aligned short prompts and the profile's maximum address range. The end-to-end serving
  length gate below, rather than a synthetic zero-value cache test, establishes servability.
- No fallback, watcher, timeout, allocator, or trace-replay error.

Measured component/full-model status on this host:

- D1 originally passed 10/12 representative packed-path cases with custom k64 decode SDPA; both
  long-position failures pass with the profile's `TT_LAGUNA_DECODE_SDPA_PC=0` fallback. D1 is still
  rejected: a full 40-layer load with `max_seq_len=4096` OOMed at the final LM head, before KV or
  warmup. Per bank it had about 2,692 MiB allocated, 80.1 MiB free, and 26.27 MiB largest contiguous
  free; the requested allocation needed 27.30 MiB per bank. Context reduction cannot recover weight
  capacity.
- D2 ring/two-link on `0,1` passed all 12 production packed-path cases: layers 0, 1, and 4 at decode
  positions 32, 513, and 2,048, with the path-active checks included. The dedicated nondegenerate
  traced decode-SDPA boundary test also passed (1/1); it uses varied nonzero V, permuted block-64 pages,
  and probes both sides of k64 boundaries through positions 131,070 and 131,071.
- The D2 all-40-layer smoke allocated uniform BFP8 KV for 131,072 tokens, completed prefill and trace
  replay, and matched device-versus-host greedy sampling for all 8 checked tokens. This is a valid
  nondegenerate capacity/trace smoke; it does not replace the separate AIME24 top-k artifact.

Capture measured DRAM snapshots after all four lifecycle points: weights loaded, full uniform KV pool
allocated, maximum-context warmup completed, and decode trace captured/replayed. Every snapshot must
have at least 10% total DRAM free and at least 128 MiB largest contiguous free **per bank**. A missing
snapshot fails the gate. Static estimates in `context_contract.json` are not substitutes.

The authoritative post-fix D2 cache-off baseline used one full prefill warmup pass and recorded weights
0.4899 / 1,273.4 MiB, uniform KV 0.2325 / 609.7 MiB, prefill warmup 0.2194 / 526.1 MiB, and trace
0.2193 / 525.8 MiB (free fraction / largest contiguous free per bank). Engine initialization took
527.76 s, of which compilation took 499.24 s. Prefix caching was false; `/health` remained 200 and
`/v1/models` advertised 131,072. Two raw 129+1 requests were exact, chat returned the requested exact
answer, the tool call parsed successfully, and two post-prefill 128-token decode replays were exact at
about 6.42 s each (~19.9 tokens/s). The expected thread-local active-trace warning appeared once on
the first eager prefill; there was no allocator error, corruption, device death, or replay failure.
After the exact-cap run, a real `pool` smoke invoked `pwd`, observed exit 0 and the exact working path,
returned `LAGUNA_POOL_OK`, and exited 0 in 21.78 s. Final health remained 200 and the fault scan found
only the already recorded advisory.
Artifacts are under
`/home/ttuser/laguna-qualification/p150x2-qualified-noalloc-20260821T123447`.

The canonical-8K cache-on qualification retained 0.2144 post-trace free fraction and 510.8 MiB largest
contiguous free per bank. Engine initialization took 524.97 s (497.13 s compilation), and TTNN froze
889 program-cache entries after the trace. All five runtime-offset hardware cases passed with misses
forbidden. The full raw-token oracle/candidate suite passed exactness, admission, poison-order,
decode-boundary, metrics, health, and performance gates. Its cache-off oracle, final cache-on result,
hardware JUnit, and server log are under
`/home/ttuser/laguna-qualification/p150x2-prefix-cache-20260821`.

## 4. End-to-end serving and prefix-cache qualification

Launch the candidate, wait for `Application startup complete` in that run's timestamped log, then
save results for all of the following:

- `GET /health` succeeds and `GET /v1/models` reports `poolside/Laguna-XS-2.1` with the profile cap.
- A deterministic chat completion succeeds.
- Auto tool choice returns a valid `poolside_v1` tool call.
- One focused `pool exec` request completes against the local `/v1` endpoint.
- The maximum-length request completes: D1 uses ISL 64,512 + OSL 1,024; D2 uses ISL 130,048 + OSL
  1,024. The sum equals the profile's advertised context.

The max-length request can use the same official client as performance qualification:

```bash
"$MODEL_DIR/.venv/bin/vllm" bench serve \
  --backend openai --model poolside/Laguna-XS-2.1 --dataset-name random \
  --random-input-len 64512 --random-output-len 1024 --random-range-ratio 0 \
  --num-prompts 1 --max-concurrency 1 --request-rate inf --ignore-eos --temperature 0 \
  --save-result --save-detailed --result-dir "$ARTIFACT_DIR" \
  --result-filename p150-max-context.json
```

For D2, change the two random lengths and filename to 130048, 1024, and `p150x2-max-context.json`.
Keep the historical cache-off exact-cap result as the capacity oracle. Production p150x2 now launches
cache-on; verify the launch and engine both report enabled, the CLI includes
`--enable-prefix-caching --enable-prompt-tokens-details --no-enable-chunked-prefill`, and the log freezes
a nonzero program-cache count after trace capture. Also verify the explicit rollback separately:
`TT_LAGUNA_PREFIX_CACHE=0` must report `operator_rollback_disabled` and
`--no-enable-prefix-caching` without an experimental acknowledgement.

Qualify caching against fresh, otherwise-identical cache-off and cache-on servers. Use raw token IDs,
streaming timings, unique cache salts, and three repetitions:

```bash
RUN_ID=laguna-prefix-$(date +%Y%m%dT%H%M%S)
"$MODEL_DIR/.venv/bin/python" -m \
  models.autoports.poolside_laguna_xs_2_1.tests.prefix_cache_qualification \
  off --run-id "$RUN_ID" --repetitions 3 --output "$ARTIFACT_DIR/cache-off-oracle.json"

# Stop/reset, launch the clean p150x2 default-on profile, then:
"$MODEL_DIR/.venv/bin/python" -m \
  models.autoports.poolside_laguna_xs_2_1.tests.prefix_cache_qualification \
  on --run-id "$RUN_ID" --repetitions 3 \
  --oracle "$ARTIFACT_DIR/cache-off-oracle.json" \
  --output "$ARTIFACT_DIR/cache-on-candidate.json"
```

The cache-on artifact passes only when all of these hold:

- cold and hit output token IDs exactly match the cache-off oracle for 2K, 32K, 65K, near-cap, and
  both full performance prompts;
- cached-token counts exactly follow 8,192-token canonical admission: partial cases
  0 / 32,768 / 65,536 / 122,880 and full cases 24,576 / 57,344;
- a 2K-before-32K oldest-hash sequence yields 0 / 0 / 24,576 admitted tokens, and a prompt extended
  across the 8K boundary by generated decode tokens yields zero;
- the Prometheus hit delta is at least the exact expected 491,520 tokens and cache-off records zero;
- median hit TTFT improves over candidate cold and is at least 3× faster at 32K and 2× at 65K;
- cache-on cold TTFT is at most 1.05× the oracle, and cold/hit TPOT are each at most 1.02× it; and
- health remains good, program-cache misses remain forbidden, and no allocator error, corruption,
  device death, watchdog, or other critical fault appears. The known once-per-thread active-trace
  allocation advisory is recorded but is not alone a failure.

## 5. Performance and default selection

Base decode acceptance comes from `.venv/bin/vllm bench serve`, at concurrency 1. The `t/s/u` gate is
**decode throughput `1000 / mean_tpot_ms`**, not aggregate `output_throughput`, which includes cold
prefill time. Prefix-cache acceptance uses the raw-token two-phase client in section 4 so it can prove
exact output IDs, cache admission, and streaming TTFT. Do not substitute an internal layer timer.
Save detailed per-request data and report TTFT separately so a slow cold prefill is never hidden.

The D2 standalone prompt-128/generate-128 warm diagnostic measured 190.7 ms TTFT, 39.203 ms/token
logits-only (25.51 tokens/s), 49.949 ms/token with device token output (20.02 tokens/s), and
50.634 ms/token including host readback (19.75 tokens/s). It demonstrates that D2 clears the floor in
that harness, but it is not substituted into the acceptance fields
below.

The authoritative cold online run used no benchmark warmups and three successful requests per
workload. ISL 1,024 / OSL 128 measured TTFT 1,278.96 ms, TPOT 50.46 ms, **19.82 decode tokens/s**, and
16.650 aggregate output tokens/s. ISL 32,768 / OSL 128 measured TTFT 53,546.66 ms, TPOT 51.53 ms,
**19.41 decode tokens/s**, and 2.130 aggregate output tokens/s. Both decode gates pass (15 and 10
tokens/s respectively), with no request errors; per-request decode rates varied by less than 0.1%
from their medians. The 53.55 s cold 32K TTFT is a material limitation:
a genuinely cold request still pays it, while the qualified canonical-prefix path accelerates reuse.

The final canonical-prefix run measured these medians over three requests per state:

| Prompt | Cache-off TTFT | Cache-on cold TTFT | Admitted prefix | Hit TTFT | Speedup | TPOT off / cold / hit |
|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 52,681.834 ms | 52,654.973 ms | 24,576 | 15,889.832 ms | **3.314×** | 51.510 / 51.548 / 51.522 ms |
| 65,536 | 134,040.189 ms | 133,891.981 ms | 57,344 | 23,011.514 ms | **5.818×** | 52.587 / 52.605 / 52.605 ms |

Both cold TTFT ratios were below 1.0, every cold/hit TPOT ratio was below 1.001, and all output token
IDs matched the cache-off oracle. These are prefix-cache gates, separate from the decode-rate floors.

The exact-cap ISL 130,048 / OSL 1,024 request completed 1/1 with no error: TTFT 381,378.79 ms,
TPOT 55.21 ms (**18.11 decode tokens/s**), aggregate output throughput 2.339 tokens/s, and total
duration 437.862 s. KV occupancy reached 100.0%, returned to 0% after completion, and `/health`
remained 200. The warning count stayed at the expected one and no fault marker appeared. This passes
the 131,072-token capacity gate while documenting the roughly 6.36-minute cold prefill.

```bash
# Repeat the command with ISL=32768 and retain detailed per-request timings.
ISL=1024
"$MODEL_DIR/.venv/bin/vllm" bench serve \
  --backend openai --model poolside/Laguna-XS-2.1 --dataset-name random \
  --random-input-len "$ISL" --random-output-len 128 --random-range-ratio 0 \
  --num-warmups 0 --num-prompts 3 --max-concurrency 1 --request-rate inf \
  --ignore-eos --temperature 0 --save-result --save-detailed \
  --result-dir "$ARTIFACT_DIR" --result-filename "isl${ISL}-cold-c1.json"
```

For each ISL, compute decode tokens/s from TPOT. The profile passes when:

- median is at least 15 tokens/s at ISL 1,024 and at least 10 tokens/s at ISL 32,768;
- no individual run is more than 10% below the applicable threshold; and
- `max(abs(run - median)) / median` is at most 0.05.

`p150x4` remains an explicit regression profile, but D2 selection is based on the absolute decode
floors above rather than a same-throughput comparison against twice as many ASICs. The old context
record reports D4 observations, but its cited raw sweep is not present in this checkout; remeasure D4
before claiming a new D4 regression number or changing its path.

`p150` is rejected on full-model weight capacity. `p150x2` ring/two-link on this host passes the
selected capacity, correctness, cold API, memory, and decode-performance gates and `/v1/models`
advertises 131,072. Its canonical prefix cache passes exactness, safety, and no-regression gates and is
enabled by default; `p150x4` remains an explicit regression profile. A clean-environment
`serve_vllm.sh config` resolves devices `0,1`,
context 131,072, max sequences 1, ring/two-link, prefix caching on, 8,192-token admission, one uniform
KV group, and scheduler chunked prefill off. Explicit `TT_LAGUNA_PREFIX_CACHE=0` is the rollback.

## Qualification artifact template

Store raw logs/result JSON beside a summary with this minimum shape. Use `null` for an unrun field and
`status: incomplete`; never encode an estimate as a measurement.

```json
{
  "schema_version": 2,
  "status": "incomplete",
  "profile": "p150x2",
  "timestamp_utc": null,
  "source": {
    "tt_metal_commit": null,
    "dirty_diff_sha256": null,
    "vllm_version": "0.24.0",
    "vllm_tt_plugin_commit": "c127c17d80d66ee83d23064d3a62ac844a1170de"
  },
  "hardware": {
    "tt_smi_device_ids": [],
    "bdfs": [],
    "firmware": null,
    "umd": null,
    "topology_inventory_artifact": null
  },
  "configuration": {
    "mesh_shape": [1, 2],
    "advertised_context": 131072,
    "max_num_seqs": 1,
    "uniform_kv": true,
    "prefix_cache": true,
    "prefix_cache_quantum_tokens": 8192,
    "kv_block_size_tokens": 64,
    "cache_admission": "complete_canonical_prompt_chunks",
    "kv_group_policy": "single_uniform_full_attention",
    "scheduler_chunked_prefill": false,
    "speculative_decode": false,
    "external_kv_connectors": false,
    "trace_region_size_bytes": 1500000000,
    "mesh_graph_descriptor": null,
    "tt_laguna_decode_sdpa_pc": 1,
    "fabric": "FABRIC_1D_RING",
    "ccl_topology": "ring",
    "ccl_num_links": 2,
    "prefill_private_scratch_blocks_per_layer": 1,
    "prefill_internal_rope_horizon": 262144,
    "prefill_full_ladder_passes": 1
  },
  "topology_runs": [],
  "memory": {
    "gate": {"min_free_fraction": 0.1, "min_contiguous_bytes_per_bank": 134217728},
    "weights": null,
    "uniform_kv": null,
    "max_context_warmup": null,
    "trace_replay": null,
    "passed": false
  },
  "accuracy": {
    "layer_pcc": {"0": null, "1": null, "4": null},
    "full_model": {"top1": null, "top5": null, "top100": null},
    "autoreg_nondegenerate": null,
    "passed": false
  },
  "serving": {
    "health": null,
    "models_context": null,
    "chat": null,
    "tool_call": null,
    "pool_smoke": null,
    "max_context": {"input_tokens": 130048, "output_tokens": 1024, "passed": false}
  },
  "prefix_cache": {
    "production_enabled": true,
    "engine_config_reports_enabled": null,
    "operator_rollback_verified": null,
    "cache_off_oracle_artifact": null,
    "cache_on_candidate_artifact": null,
    "hardware_junit_artifact": null,
    "program_cache": {
      "frozen_after_trace": null,
      "entries": null,
      "varied_offset_no_miss": null
    },
    "correctness": {
      "exact_output_ids_vs_oracle": null,
      "partial_expected_cached_tokens": [0, 32768, 65536, 122880],
      "poison_order_expected_cached_tokens": [0, 0, 24576],
      "decode_boundary_expected_cached_tokens": 0,
      "passed": false
    },
    "metrics": {
      "minimum_expected_hit_tokens": 491520,
      "observed_hit_tokens": null,
      "observed_query_tokens": null,
      "passed": false
    },
    "latency": {
      "full_32768": {"runs": 3, "minimum_speedup": 3.0, "oracle": null, "cold": null, "hit": null},
      "full_65536": {"runs": 3, "minimum_speedup": 2.0, "oracle": null, "cold": null, "hit": null},
      "cold_ttft_max_ratio": 1.05,
      "cold_and_hit_tpot_max_ratio": 1.02,
      "passed": false
    },
    "promotion_gate": false
  },
  "performance": {
    "tool": "vllm bench serve",
    "max_concurrency": 1,
    "isl_1024": {"runs_tps": [], "median_tps": null, "max_deviation_fraction": null},
    "isl_32768": {"runs_tps": [], "median_tps": null, "max_deviation_fraction": null},
    "passed": false
  },
  "d4_regression": {"baseline_artifact": null, "candidate_artifact": null, "passed": null},
  "decision": {"selected_default": null, "reason": null}
}
```
