# GPT-OSS-120B P/D measurement configuration and commands

This document records the October 22 experiment. The saved launch records and stopped-source archive are the exact execution record. Historical continuation scripts are single-use and must not be restarted after partial progress. The fixed deadlines and allocation IDs below describe this run, not a reusable reservation.

## Measured revisions

| Component | Revision |
|---|---|
| tt-metal | `b3b5e13bba59dedaf5134e51faa9a1ef0c5fd4ed` |
| tt-blaze | `d6d0b09f22f60a161d66fe4db90b9abed81ef82d` |
| tt-blaze/tt-metal | `44d66500520fda9f2c7060c0f6b41ec48f7ab37e` |
| tt-gen | `5ca79e3076f193c092962d34a3b69731e39aff91` |
| P/D harness source | `48a7e30f487ff34cf841536d78d576dfc2181ad5` |

The measured Metal HEAD stayed at `b3b5e13bba59dedaf5134e51faa9a1ef0c5fd4ed` throughout the sweep. Later harness/report publication commits are not measured revisions. The only additional Metal files were the six unchanged P/D harness files below. Their bytes matched the published harness revision in a read-only check at 06:58 UTC.

| Harness file | SHA256 |
|---|---|
| `models/demos/gpt_oss_d_p/repro/make-perf-prompt.py` | `ca9045b402579e496dcc6727980f2b4eef76ad2d2fc17f6bad10f56da76bd386` |
| `models/demos/gpt_oss_d_p/repro/measure_pd_request.py` | `8fc1ea4f2bd0a82d7fc225ed88e4105dfd32a99259f46553583c37d0c9f90e99` |
| `models/demos/gpt_oss_d_p/repro/perf_analysis.py` | `8b73291a4f9996f141b5aabe471011d35722007a041f51f1d8c2eabc60885db5` |
| `models/demos/gpt_oss_d_p/repro/run-prefill-timed.py` | `e5dd290f04947daed2187739324f513ff8bbb3d76ea5ed69f2eef69e4649425c` |
| `models/demos/gpt_oss_d_p/repro/run_pd_case.py` | `b8fc39f2fcb8ebf0ce48126f926bed1f6cdaf237df20ea282071df0d21f91f97` |
| `models/demos/gpt_oss_d_p/repro/test_perf_analysis.py` | `5fcac111aa479e0dba1823f8f6f3b6604aa0b853f28204c0f6c94da24e105b8c` |

`tt_metal/impl/program/program.cpp` SHA256 was `0fa05e32a12c1d9f6de6ab6519324afd28689c73078aa2cd8ac76cc2f03eae6a`. Each prefill wrapper records the exact Metal HEAD, this hash, chunk size, source default 2880 and effective payload in `runtime/p<payload>-c<chunk>/prefill/fabric-config.json`.

## Matrix and request contract

- Payload bytes: 2880, 4352, 8704.
- ISL in K tokens: 1, 2, 4, 5, 8, 16, 32, 55, 64. K=1024.
- Chunk in K tokens: 1, 4, 8, 16, 32, 64; include exactly the 29 pairs where chunk<=ISL for each payload, 87 cases total.
- Group order: chunks 64K, 32K, 16K, 8K, 4K, 1K; within each chunk, payloads 2880, 4352, 8704. ISLs increase within each group.
- One warmup, then three measured requests per case; all four retained. Output cap 32 with EOS enabled. No selective reruns or excluded slower means.
- Existing synthetic fresh-nonce prompt generation is used. Actual prompt text, outputs, usage, prompt/reused/newly prefilled token accounting and finish reason are retained with each request. Final exports add explicitly derived padded input capacity and padding rows from the saved chunk-call counts, as described below.
- Full 36-layer model. Prefill uses SP4/TP8, linear topology / FABRIC_1D, capacity 131072 and two allocated slots, with one active request. Decoder uses four 4x2 stages, 41 reload images, full36 and 16 KV configurations.
- No prefill-only or standalone run, golden generation, PCC or numerical-accuracy evaluation was performed as part of this comparison.

## Model and environment

Shared task root: `/data/divanovic/gptoss-b3b5e13-pd-20261022`.
Local coordinator root: `/Users/divanovic/gptoss-b3b5e13-pd-20261022`.
Model: `/mnt/weka/model-weights/llm/openai/gpt-oss-120b`.

The prefill entrypoint is `runtime/run-prefill-verified.py`. It imports the model configuration, asserts source default 2880, applies `EXPERIMENT_PAYLOAD` to `GptOss120BConfig.FABRIC_PAYLOAD_SIZE` before runner/model import, checks exact HEAD and core SHA, then enters the native prefill runner. Model source is not edited between payload groups.

The final group's actual environment included:

```bash
EXPERIMENT_PAYLOAD=8704
ARCH_NAME=blackhole
PREFILL_MODEL=gpt_oss_d_p
GPT_OSS_WEIGHTS_FROM_CACHE=1
GPT_OSS_BOUNDED_SLIDING_KV=0
PREFILL_NUM_LAYERS=36
PREFILL_NUM_USERS=2
PREFILL_SP=4
PREFILL_TP=8
PREFILL_CHUNK_SIZE=1024
PREFILL_MAX_SEQ_LEN=131072
PREFILL_ASYNC_LAYER_COMPLETION=1
PREFILL_LAYER_ACK_D2H=0
PREFILL_ENABLE_MIGRATION=1
PREFILL_MOCK_MIGRATION=0
PREFILL_MIGRATION_EXPORT_TO_FILE=1
PREFILL_USE_TRACE=0
PREFILL_TOPOLOGY=linear
PREFILL_FABRIC_MODE=1d
PREFILL_GATE_FALLBACK_MODE=DEVICE_FP32
OMP_NUM_THREADS=32
MKL_NUM_THREADS=32
TOKENIZERS_PARALLELISM=false
```

Prefill caches are `cache/prefill-weights`, `cache/prefill-runner-weights` and `/home/divanovic/.cache/tt-metal-cache/b3b5-20261022-prefill`. The wrapper records the group-specific H2D service ID, completion ring, migration table and device map. The 36-layer manifest is `models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_120b_chat_36l.json`. The exact wrapper is retained after its process exits.

Decoder runtime uses `runtime/decode-env`, the pinned nested Metal tree, `GPTOSS_KV_DEPTH=131072`, `GPTOSS_SAMPLE=1`, socket `gptoss_b3b5_20261022_b08_decode`, `TT_MPI_IFACE=ens5f0np0`, OpenMPI `/opt/openmpi-v5.0.7-ulfm`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `BLAZE_TENSORBIN_CACHE=cache/decode-tensorbin` and `/home/divanovic/.cache/tt-metal-cache/mg-repro-20261021-decode`. No speculative cache mutation or healthy-device reset was used.

## Actual execution commands

The sole laptop controller performed ownership and exact process-identity guards before dispatch. The following are recorded operations, not instructions to repeat them in the existing experiment.

B08 native build used `runtime/build-decode-b08.sh`, launch receipt `runtime/build-decode-b08-launch.json`; it passed native compilation, TTNN import and exit0 at 03:09:13.644438 UTC, 304.063 seconds after launch. Build logs are `evidence/build-decode-b08.log` and `evidence/build-decode-b08-exit.txt`. It was not rebuilt after readiness.

The retained B08 owner was launched once with:

```bash
bash /data/divanovic/gptoss-b3b5e13-pd-20261022/runtime/start-decode-owner-b08.sh
```

Inside its existing deadline wrapper, the actual launcher is:

```bash
bash /data/divanovic/gptoss-b3b5e13-pd-20261022/tt-blaze/scripts/gptoss_reload_4x2/launch_chat_showcase.sh /data/divanovic/gptoss-b3b5e13-pd-20261022/runtime/repro-b08-full36/decode 36
```

The four native owners run `scripts.gptoss_reload_4x2.showcase` with `--layers 36 --kv-depth 131072 --sample`, the model path above, output `runtime/repro-b08-full36/decode` and the recorded socket prefix. Driver launch was 03:10:37.040328 UTC; readiness was 03:48:34.688 UTC, 2277.647672 seconds later. Earlier failed/preempted attempts remain separate.

For each new payload/chunk group, the controller used `runtime/prepare-group-b08.py --payload <bytes> --chunk <tokens>`, then launched that group's `start-prefill-owner.sh` once. For the final group the recorded command was:

```bash
bash /data/divanovic/gptoss-b3b5e13-pd-20261022/runtime/p8704-c1024/start-prefill-owner.sh
```

Its deadline wrapper runs `python -u /data/divanovic/gptoss-b3b5e13-pd-20261022/runtime/run-prefill-verified.py` in the verified environment. It launched at 06:54:41.462370 and entered its request loop at 06:56:50.532 UTC, 129.069630 seconds later.

Each group attaches manager-prefill, manager-decode, worker-prefill and worker-decode through `runtime/start-service-b08.py <service> --run <group>`. Decoder services use `runtime/slurm-service-supervisor-b08.py` in separate bounded persistent Slurm steps. Manager binary is `tt-gen/build-kv-manager/kv_manager`; workers use the D-gen environment and `python -m tt_dynamo.main`, group-specific JSON, model path, served name `gpt-oss-120b-4x2-chat`, component `gptoss_prefill` or `gptoss_decode`, reasoning parser `gpt_oss` and tool parser `harmony`.

For example, the actual final payload4352/64K-input/1K-chunk case command was:

```bash
/data/divanovic/gptoss-b3b5e13-pd-20261022/tt-gen/adapters/dynamo/.venv/bin/python /data/divanovic/gptoss-b3b5e13-pd-20261022/tt-metal/models/demos/gpt_oss_d_p/repro/run_pd_case.py --run /data/divanovic/gptoss-b3b5e13-pd-20261022/runtime/p4352-c1024 --output /data/divanovic/gptoss-b3b5e13-pd-20261022/evidence/sweep/p4352/pd/64k-c1k --model /mnt/weka/model-weights/llm/openai/gpt-oss-120b --isl 65536 --chunk 1024 --prefill-host bh-glx-120-b09u02 --prefill-ip 10.32.17.193 --decode-ip 10.32.17.129 --deadline 1792655400
```

`run_pd_case.py` executes the warmup plus three measured requests through the retained frontend; the underlying request JSON and stream are preserved. Every actual case command is retained in its launch record and coordinator log. The source archive preserves the exact prompt, timing and aggregation code.

Read-only saved-evidence auditing uses:

```bash
python3 /data/divanovic/gptoss-b3b5e13-pd-20261022/runtime/audit.py /data/divanovic/gptoss-b3b5e13-pd-20261022 --output /data/divanovic/gptoss-b3b5e13-pd-20261022/evidence/report/sweep-audit.json
```

## Transport, ownership and limits

Current measurement allocations are b09u02/142547 for prefill and b08u02/142679 for decode. Every compute dispatch checks both exact divanovic RUNNING records. Prefill/startup use laptop ProxyJump through `slurm-login-01.exabox.tenstorrent.com`, the explicit key `/Users/divanovic/.ssh/id_rsa`, BatchMode, IdentitiesOnly, IdentityAgent=none, ConnectTimeout30 and ServerAliveInterval15/ServerAliveCountMax3. Decoder service/identity control uses standard `srun --overlap --jobid 142679 -w bh-glx-120-b08u02 -N 1 -n 1 --cpu-bind=none`, with host/job checks inside the step. The existing decoder transport additionally sets UseKeychain=no. It permits only fixed service, identity and cleanup operations.

Only control-step soft NPROC is raised to at least4096 within its hard limit; model/global limits and CPU policy are unchanged. Final B08 manager/worker steps142679.591/.592 have affinity0–63. The first accepted B04 case used60CPU decoder services versus64CPU model; B08 services use64CPU. This difference is recorded as a cohort confound.

Fixed deadlines: benchmark07:50, services07:55, owners08:00, work08:05 UTC; new benchmark work stops at07:35. Longer reservations do not extend running wrappers. Mutations, launches and model requests are never retried. Only read-only allocation/identity transport failures have at most three attempts separated by10seconds. Ownership, identity, permission and model errors remain fatal.

## Timing definitions and interpretation

TTFT is request submission to the first nonempty content or reasoning delta. Prefill rate is actual newly prefilled tokens divided by the host-observed prefill span; logical input, reused tokens and padded processing remain distinct. Native decode tokens/s/user is generated-token intervals divided by the first-to-last native host-commit interval; the first token is excluded from that interval count. Total is client-observed request duration. These serving-path measurements are not device-only kernel timings.

KV bytes denote migrated payload. Migration intervals run from the source-manager command to successful completion, including control-plane time. Prefill intervals use the host compute-call boundary with the existing async layer drain and no extra synchronization. Transfer sum, union-active duration and first-to-last span are separate. Observed overlap is the intersection of the host-prefill and manager-migration intervals, not direct hardware overlap; non-overlap is each active interval minus that intersection. Initial prefill and final transfer tail expose start/end serialization. Never add overlapping durations or concurrent stage totals as wall time. Warmups, three measured samples, arithmetic means and sample variance/SD are retained separately.

The first accepted case used b09/b04u02; the remaining86 use b09/b08. Previous baselines used different hosts and historical D-gen cohorts. Host replacements, independent startup/cache state, fixed group order and the60/64CPU service difference prevent packet-only or commit-only causal claims. Three samples per case support descriptive comparisons, not a significance claim. No numerical-accuracy claim is made.

Hardware incompatibility, failed recovery, preemptions, login transport failures and the process-disappearance cleanup defect are preserved in DECODE-DRAM-INVESTIGATION.md, PREEMPTION-RECOVERY.md, SSH-CONTROL-INVESTIGATION.md, LOGIN-TIMEOUT-INVESTIGATION.md and STOP-PROCESS-RACE-INVESTIGATION.md. Their build/startup/reload attempts and failure logs remain separate from accepted measurement timings.

## Frozen previous-result provenance

The matched previous arrays are frozen in `baseline.json`; SHA256 `93973bc249c60e441b7a8abf90a500e2fd94f280982468fe792f4528737d467d`. They are not replaced by this run. The three historical reports below are preserved alongside this report.

| Payload | Historical Metal source | Prefill / decoder hosts | D-gen cohort | Record |
|---|---|---|---|---|
|2880|Base fbbda775e3dd1a4dcf41ca2d3f9e649ff7be6436; published measured harness 3441c663200ca87f7deafe7c48d39ba61c40d5c0|c02u02 / c06u14|64K/c64K and32K/c32K:7f6caf7d; next26:4411a95b; final64K/c1K retry:5ca79e3076f193c092962d34a3b69731e39aff91|[2880 historical report](baseline-2880-report.md)|
|4352|Base aa93f1b795694a2cea1dd8ceb952147dce620463; tested configuration2898a1b47b7411611e9ab41b2873e16cf5aa9ac7|c05u02 / b03u02|5ca79e3076f193c092962d34a3b69731e39aff91 throughout|[4352 historical report](baseline-4352-report.md)|
|8704|Base9f7df82859f1b5e2e7d7f91e6fb3b1e69e9d628f; tested configuration076d79443138613e9e1a96b948d02db69858274a|b05u02 / b03u02 for first8 cases (64K,32K,16K chunks); b05u02 / b08u02 for remaining21|5ca79e3076f193c092962d34a3b69731e39aff91 throughout|[8704 historical report](baseline-8704-report.md)|

Historical Blaze and nested Metal pins match the table above. The historical2880 repair/retry cohorts are inherited evidence, not extra requests in the present87-case sweep. These differing source and host cohorts prevent a commit-only or packet-only causal interpretation.

## Derived padding accounting

Derived fixed input capacity: number of saved host prefill_chunk calls times configured chunk tokens; padding is capacity minus sum(to_token-from_token). Counts tensor rows across SP, not multiplied by layers or TP. This is derived accounting, not an additional performance measurement. The existing logical `prefill_tokens_processed` counter is unchanged. Added fields are `prefill_chunk_calls`, `prefill_padded_capacity_tokens_derived` and `prefill_padding_tokens_derived`; the original summary before enrichment is retained. Sources at the measured revision: `models/demos/common/prefill/runners/prefill_runner.py:684`, `models/demos/common/prefill/runners/prefill_runner.py:714`, `models/demos/common/prefill/runners/runner_utils.py:64`, `models/demos/gpt_oss_d_p/tt/tt_prefill_runtime.py:345`. The 348 raw request prompt token counts, successful streams and logical chunk ranges were checked; `raw-request-inventory.json` hashes 8439 files.

Full exports are stored losslessly as gzip files to satisfy the repository file-size limit. `export-packaging.json` records compressed and original SHA256 values; decompress with `gzip -dk <file.gz>`. The interactive view uses compact table JSON; the full export retains every request and field.
