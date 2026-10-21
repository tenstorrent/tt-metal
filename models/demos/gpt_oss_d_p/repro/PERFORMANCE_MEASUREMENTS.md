# GPT-OSS-120B performance measurements without reference data

This supplements [the original reproduction guide](../GPT_OSS_120B_4X2_PD_PERF_REPRO.md).
It measures standalone prefill and disaggregated prefill → native KV migration →
4×2 reload decode on two Blackhole Galaxies. It deliberately does not generate or
load golden KV files, run a reference model, or run PCC checks. Successful cases
validate execution, stream completion and migration range accounting, not model
numerical accuracy.

## Tested source and configuration

| Repository | Original pin | Reproduction branch |
| --- | --- | --- |
| tt-metal | `fbbda775e3dd1a4dcf41ca2d3f9e649ff7be6436` | `divanovic/mg-gptoss-pd-repro-20261021-metal` |
| tt-blaze | `d6d0b09f22f60a161d66fe4db90b9abed81ef82d` | Unchanged; use the commit pin |
| tt-d-gen | `f5a46649a2d1e9a4ccae8b1e2504c63925b16473` | `divanovic/mg-gptoss-pd-repro-20261021-dgen` |
| Blaze's nested tt-metal | `44d66500520fda9f2c7060c0f6b41ec48f7ab37e` | Pinned submodule |

D-gen requires the minimal lockstep prerequisite from upstream `f57f0ccb`,
backported as `7f6caf7d`, and the diagnostic correction `4411a95b`. The tested
D-gen head for the final 64K/1K retry is `5ca79e3076f193c092962d34a3b69731e39aff91`,
which fixes an admission deadlock for already-prepared local migration children.
The original guide comparisons and two earliest P/D sweep cases used `7f6caf7d`;
the next 26 P/D cases used `4411a95b`. All standalone cases do not use D-gen.
These source cohorts are preserved in the report and per-run CSV rather than
claimed as a homogeneous run. The unmodified guide's
D-gen pin does not accept its required lockstep setting. Blaze model source is
unchanged. Metal adds measurement harnesses; model math and precision are unchanged.

- Full 36 layers; original quantized GPT-OSS-120B checkpoint and its tokenizer.
- Prefill: 32 chips, mesh 4×8, SP=4/TP=8, linear topology, `FABRIC_1D`, BF4 expert
  weights, BF8 KV, unbounded physical sliding-cache storage, trace disabled.
- Decode: four 4×2 stages, reload pipeline, sampling enabled, lockstep enabled,
  decode chunk size 8, one KV copy per head.
- P/D and both sweep modes: capacity 131,072, two allocated slots, one active request.
- Original standalone comparison only: one slot, 65,536 capacity, 32K chunks.
  Do not treat its throughput difference from P/D as pure migration overhead.
- RDMA with native DMK device I/O; staging slab size 16 MiB, four slabs.
- `ai-dynamo==1.5.0`, Open MPI 5.0.7 ULFM.

## Runtime prerequisites

Use the original guide's build and owner setup, with the corrections above and
below. Reserve both hosts, verify current ownership and ensure that no existing
process owns their TT devices before starting a new owner. The performance client
connects to existing owners; it must not open those devices itself.

The executed October 21 deployment used:

```text
ROOT=/data/divanovic/gptoss-mg-repro-20261021
PREFILL_HOST=bh-glx-120-c02u02   PREFILL_IP=10.33.24.65
DECODE_HOST=bh-glx-120-c06u14    DECODE_IP=10.33.26.33
MODEL=/mnt/weka/model-weights/llm/openai/gpt-oss-120b
BASE_RUN=$ROOT/runtime/repro-full36
```

`ROOT/runtime` contains the exact build scripts, environments, launch commands,
configs and process identity records from that run. They contain hostnames,
absolute paths and October 21 deadlines: adapt them to a fresh allocation before
using them again. Do not rerun them against the recorded processes or stale
owner manifests. SSH from the laptop used ProxyJump through
`slurm-login-01.exabox.tenstorrent.com`; login-node-to-compute SSH was unavailable.

```bash
ssh -J slurm-login-01.exabox.tenstorrent.com divanovic@bh-glx-120-c02u02
```

Relevant deployed files:

| File under ROOT | Purpose |
| --- | --- |
| `runtime/build-role.sh` | Separate prefill Metal and Blaze-pinned Metal builds/environments |
| `runtime/build-native-engine.sh` | Native engine and Python bindings |
| `runtime/build-native-transport.sh` | KV manager and DMK transport |
| `runtime/start-standalone-baseline.sh` | Standalone reference performance configuration, without golden data |
| `runtime/start-decode-owner.sh` | One retained 36-layer decode owner for all chunk groups |
| `runtime/start-prefill-owner-v2.sh` | Instrumented prefill owner, initially 32K chunk |
| `runtime/prepare-chunk-group.py` | Fresh prefill config/socket/table per chunk, retaining decode artifacts |
| `runtime/start-service-v3.py` | etcd, frontend, native managers and workers with identity/deadline records |
| `runtime/prepare-prefill-sweep.py` | One cached standalone prefill model per chunk group |
| `runtime/stop-task-process.py` | Stop a recorded task-owned process tree; it refuses the decoder-owner record |

Use the native manager executable directly at
`$ROOT/tt-gen/build-kv-manager/kv_manager`. Do not append another `/kv_manager` to
that executable path. This build's importable bindings are under
`$ROOT/tt-gen/bindings/python`. Native shared libraries were staged under
`$ROOT/runtime/native-metal-lib`; the recorded launch also uses dependency
libraries from `/data/divanovic/gptoss120b-pd/runtime/native-sysroot/usr/lib/x86_64-linux-gnu`.
Provide those dependencies or rebuild an equivalent local environment on a new
installation. Node-local build symlinks are not automatically valid on both hosts.

The prefill runner uses `PREFILL_TTNN_CACHE=$ROOT/cache/prefill-runner-weights`.
Its `gpt_oss_d_p_bh_32dev/4x8` entry points to the standalone runtime's populated
`$ROOT/cache/prefill-weights/tensor_cache_bfp8_MeshShape([4, 8])` directory.
This layout correction prevents an unnecessary or failed cache rebuild.

Worker metric ports are **8089 on prefill and 18089 on decode**. Port 8089 on the
decode host conflicts with the loaded model's telemetry. Manager metrics use
19100 on each host; frontend HTTP uses 8000 on prefill. The measurement harness
expects these ports and the served model name `gpt-oss-120b-4x2-chat`.

## Create exact-length input without goldens

Run in an environment with the checkpoint's `transformers` tokenizer. `REPRO`
is this directory; output paths must be new.

```bash
export REPRO="$TT_METAL_HOME/models/demos/gpt_oss_d_p/repro"
python "$REPRO/make-perf-prompt.py" \
  --model "$MODEL" --tokens 65536 --output "$ROOT/evidence/prompt-64k.json"
```

The prompt is a fresh nonce, repeated `hello`, and a short summarization request.
The chat-template token count must match the requested ISL exactly. Prompt JSON
saves the text, token IDs, template/token hashes and HTTP payload. The payload
requests streaming, usage accounting and **at most 32 output tokens**. EOS remains
enabled. Actual completion counts and finish reasons are saved, not assumed.

## Standalone prefill baseline and sweep

The original guide's PCC/golden command is replaced by:

```bash
# In the built prefill environment, with all 32 devices idle.
# Set the prefill environment/topology/cache variables from the recorded launcher.
python "$REPRO/prefill-perf-only.py" \
  --prompt "$ROOT/evidence/prompt-64k.json" --chunk 32768 \
  --report "$ROOT/evidence/prefill-baseline.json"
```

This loads all 36 layers, compiles, performs one request warmup and three measured
requests, then closes the mesh. The first uncached launch may use
`GPT_OSS_WEIGHTS_FROM_CACHE=0`; cached runs use `1`. Setup events separate imports,
mesh opening, checkpoint loading, model/cache construction and compilation.

For the sweep, generate prompt files named `<token_count>.json` for all nine ISLs.
For example, create all nine inputs without loading model weights:

```bash
for tokens in 1024 2048 4096 5120 8192 16384 32768 56320 65536; do
  python "$REPRO/make-perf-prompt.py" --model "$MODEL" --tokens "$tokens" \
    --output "$ROOT/evidence/sweep/prompts/$tokens.json"
done
```

Then run one group for a selected chunk, using a fresh evidence directory:

```bash
export GPT_OSS_WEIGHTS_FROM_CACHE=1
mkdir -p "$ROOT/runtime/sweep-prefill-c8192"
# DEADLINE_EPOCH must be earlier than the current reservation's end.
python "$REPRO/prefill-sweep-group.py" \
  --prompts "$ROOT/evidence/sweep/prompts" --chunk 8192 \
  --output "$ROOT/evidence/sweep/prefill" \
  --events "$ROOT/runtime/sweep-prefill-c8192/events.jsonl" \
  --deadline "$DEADLINE_EPOCH"
```

The group uses one loaded model for all valid ISLs for that chunk, with fixed
131,072 capacity and two slots. Each ISL gets one warmup and three measurements.
Every real input token is recomputed; the last chunk is padded. Both real-token
and padded-position rates are recorded. The measured interval includes host input
conversion, chunk calls and final device synchronization.

## P/D startup and measurement

1. Start the retained decoder once. Wait for all 41 reload images and fresh
   `decode/external.ready.json`, `decode.pb` and device map.
2. Start the instrumented prefill owner with matching chunk/config paths. Replace
   its ordinary runner invocation with `python "$REPRO/run-prefill-timed.py"` and
   set `PREFILL_PERF_EVENTS=$RUN/prefill/chunk-times.jsonl`. Wait for its KV table,
   device map and readiness. The wrapper adds host timestamps without extra device
   synchronization or model changes.
3. Start etcd/frontend, then native managers on both hosts, then both workers.
   Require the native lockstep setting and fresh owner artifact paths. Check
   readiness from actual logs and service state before sending traffic.
4. Run each measurement on the **prefill host**, with no competing chat clients:

```bash
# This Python is the built tt-d-gen/Dynamo adapter environment.
"$ROOT/tt-gen/adapters/dynamo/.venv/bin/python" "$REPRO/run_pd_case.py" \
  --run "$RUN" --output "$ROOT/evidence/sweep/pd/32k-c8k" \
  --model "$MODEL" --isl 32768 --chunk 8192 \
  --prefill-host "$PREFILL_HOST" --prefill-ip "$PREFILL_IP" \
  --decode-ip "$DECODE_IP" --deadline "$DEADLINE_EPOCH"
```

`RUN` must describe the active 8K prefill owner in this example. This command makes
one warmup and three measured requests, with fresh nonce input each time. It saves
commands, prompts, SSE events, outputs, counter snapshots, log windows and range
coverage. Existing output paths are refused. A failed run is retained and fails
the case; do not silently retry into the same directory.

For another ISL with the same chunk, keep all services/model owners loaded and
use a fresh case directory. When changing chunk, drain requests and stop only
that group's workers/managers/prefill owner, create fresh prefill artifacts and
reattach to the existing decoder. Never reset the retained decoder as part of a
chunk change. No standalone prefill test can share the prefill device owner.

The sweep contains 29 valid combinations per mode:

```text
ISLs:   1024 2048 4096 5120 8192 16384 32768 56320 65536
chunks: 1024 4096 8192 16384 32768 65536
condition: chunk <= ISL; K means 1024
```

## Measurement and acceptance definitions

- **Warmup** is one request after internal model compilation. Cold startup is
  separately reported; it is not part of the three measured-request means.
- **TTFT** is HTTP submission to first nonempty content or reasoning delta.
- **Prefill** in P/D is the span of host-observed chunk compute calls. Active time
  is their interval union. No extra device synchronization is introduced.
- **Actual prefilled tokens** exclude observed prefix reuse. The common chat
  template can reuse 64 tokens; the actual counter delta is recorded for each run.
  Report this separately from ISL.
- **Computed positions including padding** are chunk-call count × fixed width.
  The engine's raw `prefill_tokens_processed` counter is logical, not padded.
- **KV bytes** are the source manager's wire payload counter delta, not link
  headers or receiver drain bytes.
- **Transfer sum** sums per-migration intervals; **active time** is their union;
  **span** runs from first command to last completion. Retries/local dispatch
  repeats sharing one migration ID are counted once, from first admission.
- **Overlap** is intersection of prefill-active and transfer-active interval
  unions. Non-overlap is each union minus that intersection. Initial prefill and
  final transfer tail describe exposed boundary segments, not extra costs to add.
- **Decode tokens/s/user** is native inter-token interval count divided by native
  interval seconds. For 32 outputs there are normally 31 intervals. These are host
  timestamps after the decode committer publishes, not isolated device cycles.
- **Total** is HTTP request elapsed time. Never derive it by adding overlapping
  stage intervals. Client, prefill and source manager use one host clock.
- Each accepted P/D run requires complete SSE/usage, exact input count, bounded
  output count and one successful completion for every expected layer/range.
  The final prompt block is computed on decode and excluded from migrated ranges.
- A partial tail can legitimately miss an SP shard. The corrected native manager
  logs that per-device empty intersection at DEBUG. Aggregate empty transfers,
  failed migrations and actual stale-shard errors still fail validation.
- Means/sample variances use only the three post-warmup runs; sample variance
  uses N−1. Three samples do not establish long-term stability.

Host analysis checks, without devices:

```bash
python3 -m unittest discover -s "$REPRO" -p test_perf_analysis.py
```

The deployed `runtime/audit-results.py` additionally audits all saved case files,
input/chunk identities, four-run structure, interval conservation, token counters,
decode-rate calculations, aggregate means and sample variances. It issues no
inference or device reads.

Retain failed attempts and all runtime/source hashes alongside the final report.
After the sweep, shut down only verified task-owned services/model owners and
release this task's unused allocation. Do not reuse recorded physical addresses
or process IDs in a new deployment.
