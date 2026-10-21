# GPT-OSS-120B 8,704-byte P/D experiment

## Scope and completed measurements

This experiment repeats the completed 4,352-byte sweep with `GptOss120BConfig.FABRIC_PAYLOAD_SIZE = 8704`. It measures full 36-layer P/D performance, retaining the decoder within each host cohort; preemption required one replacement between the 16K and 8K chunk groups. It does not generate/use goldens, run PCC, rerun standalone, or establish numerical accuracy.

All 29 cases completed on October 21 at 17:57 UTC. The final saved-evidence audit at 17:57:29.333471 UTC passed with zero errors: 29 warmups and 87 measured requests. The dashboard retains immutable 2,880-byte and 4,352-byte baselines. Publication and cleanup receipts accompany the finalized report.

## Source and machines

| Component | Pin / identity |
|---|---|
| Metal base | `9f7df82859f1b5e2e7d7f91e6fb3b1e69e9d628f` |
| Metal configuration commit | `076d79443138613e9e1a96b948d02db69858274a` |
| Metal experiment branch | `divanovic/mg-gptoss-pd-payload8704-20261021-metal` |
| Model delta | `models/demos/deepseek_v3_d_p/reference/gpt_oss_120b_config.py`: 4352 → 8704 |
| Blaze | `d6d0b09f22f60a161d66fe4db90b9abed81ef82d` |
| Blaze nested Metal | `44d66500520fda9f2c7060c0f6b41ec48f7ab37e` |
| D-gen | `5ca79e3076f193c092962d34a3b69731e39aff91` |
| Prefill | `bh-glx-120-b05u02`, IP `10.32.16.193`, Slurm 141909 |
| Current decode | `bh-glx-120-b08u02`, IP `10.32.17.129`, Slurm 142028 |
| Decode for first eight cases | `bh-glx-120-b03u02`, IP `10.32.16.65`, Slurm 141893; preempted at 16:02:23 UTC |
| Model checkpoint | `/mnt/weka/model-weights/llm/openai/gpt-oss-120b` |
| Remote root | `/data/divanovic/gptoss-payload8704-20261021` |
| Local controller | `/Users/divanovic/gptoss-payload8704-20261021` |

The final decoder allocation142028 had a scheduled expiry of **2026-10-21 22:13:34 UTC**; prefill141909 expires20:25:51UTC. Slurm denied the requested in-place extensions; prefill was not extended. The fresh six-hour decoder reservation incorporates the requested extra three hours. Existing process deadlines remain unchanged: benchmarks18:39:46, services18:42:46, model owners18:43:46, work18:53:46UTC. A longer allocation does not extend already-running timeout wrappers. These job IDs and deadlines describe this run only; a later reproduction must obtain new allocations and set its guards/deadlines before launching anything.

2,880 B used c02u02 / c06u14; 4,352 B used c05u02 / b03u02. The first eight 8,704 B cases used b05u02 / b03u02. After decoder preemption, the remaining21 cases used b05u02 / b08u02. Prefill and decoder host identity, startup/cache state and run order can affect timing; these comparisons do not isolate packet size alone. Topology and harness remain unchanged. Historical D-gen cohorts are recorded. No goldens or PCC checks.

## Setup and dependencies

The new root has independent runtime, IPC service IDs and evidence directories. Source/environment/weight-cache symlinks point to the completed project; the Metal checkout has a new branch and its previous commits remain intact. Use exact pinned source and the recorded environment, not arbitrary main.

- Prefill native build: task `tt-metal`, release build in `/home/divanovic/.cache/tt-metal-builds/mg-repro-20261021-prefill/Release`.
- Decode native build: `tt-blaze/tt-metal`, release build in the corresponding `...-decode/Release`.
- Python environments: `runtime/prefill-env` and `runtime/decode-env`.
- Dynamo workers: `tt-gen/adapters/dynamo/.venv/bin/python`.
- KV manager: `tt-gen/build-kv-manager/kv_manager` with the pinned `runtime/native-metal-lib` and shared native sysroot libraries.
- MPI: `/opt/openmpi-v5.0.7-ulfm`.
- Shared CPM cache: `/data/divanovic/gptoss120b-pd/runtime/cpm-decode`.
- Builds and imports are checked on each target host. Reused build directories can still recompile targets; actual compilation/import duration is recorded separately from inference.
- Read-only `tt-smi -s` before startup validates 32 discovered devices and 32 ready DRAM statuses, with no task device holders.

The prior decoder's owner-stop path leaves parked firmware and exits through`os._exit`. Before this new run, one bounded site `recover-hosts` operation on idleb03u02 reset and validated the devices/links. It passed on the first attempt. It retained firmware/version/device/link validation and skipped optional sys-triage plus the multi-host MPI stress test for this single-host recovery. Never reset a live healthy owner.

## Access and execution

Slurm commands run through the login host; compute SSH originates on the laptop using ProxyJump:

```bash
ssh divanovic@slurm-login-01.exabox.tenstorrent.com
ssh -J divanovic@slurm-login-01.exabox.tenstorrent.com divanovic@bh-glx-120-b05u02
ssh -J divanovic@slurm-login-01.exabox.tenstorrent.com divanovic@bh-glx-120-b08u02
```

The final controller, now exited, was `resume-after-preemption-b08.py`, using `ops_b08.py`, `ops_b08_graceful.py` and `case_ops_b08.py`. It retains the ready8K prefill and existing etcd/frontend, launches one fresh b08 decoder after its build passes, then runs only the21 remaining cases. Inspect its launch JSON, actual process identity, log and remote status before any action. **Do not invoke it again.** Its single `run-sweep.lock` prevents a second coordinator. It stops on failures and preserves owners/evidence for diagnosis. The former `run-sweep.py`, `resume-on-b05.py` and `resume-after-decode-cache-retry.py` have stopped; do not restart them. No further access to released b03u02, b03u08, c05u02 or c08u02 is authorized.

### Historical initial startup sequence

The following records the initial b05/b03 startup that produced the first eight cases. These historical host/script names must not be used to restart that run. The historical b08 continuation commands are in the final section.

1. Verify the exact two job IDs, user `divanovic`, `RUNNING` state and node names.
2. Check no task model/process/device holders; perform required decoder recovery and save its launch/log/exit.
3. Run bounded `runtime/build-prefill-b05.sh prefill` on prefill and `runtime/build-role-b03.sh decode` on decode. Wait for exit0 and TTNN import checks.
4. Create64K group with `runtime/prepare-chunk-b05.py --chunk 65536 --name sweep-b05-pd-c65536` and launch its recorded`start-prefill-owner.sh` onprefill.
5. Launch `runtime/start-decode-owner-b03.sh` once ondecode. Wait for all41reload images and`external.ready.json`, `decode.pb`, `decode-device-map.txt`.
6. Startetcd/frontend onprefill with`runtime/start-service-b05.py`. Attach both native managers and workers for the current group. Existing-record/port checks reject duplicates.
7. Run the cases below serially. Each case driver produces one warmup and three measurements.
8. Audit saved evidence after each group. Stop that group's workers/managers, use`runtime/shutdown-prefill-b05.py` to send the supported H2D shutdown sentinel, require explicit sentinel/mesh-close logs, exit0 and idle devices. Then start the next chunk group. The guard includes64K; the decoder stays loaded.
9. Final29-case audit, reporting, source hooks/review/publication, raw archive, verified task-only cleanup, release both allocations and verify terminal Slurm states.

No direct login-to-compute coordinator is used because that SSH path is not functional in this environment. The detached local coordinator requires the laptop/network to remain available; model drivers themselves are remote and deadline-bounded.

## Runtime settings

- Full36 layers; original quantized weights.
- PrefillSP4/TP8, two allocated user slots, capacity131072.
- Linear prefill topology: `PREFILL_TOPOLOGY=linear`, `PREFILL_FABRIC_MODE=1d`; single-Galaxy XY-torus physical descriptor unchanged.
- No prefill trace, asynchronous layer completion, native migration enabled.
- Each group imports the config through`run-prefill-verified.py`, asserts8704, and writes`prefill/fabric-config.json`.
- Decode uses four4x2stages, capacity131072 and unchanged sampling/lockstep configuration. Initial decoder socket IDs used`gptoss_payload8704_20261021`; the b08 replacement uses`gptoss_payload8704_20261021_b08`.
- One active request; both model owners are retained within a group. Each decoder is retained across its assigned chunk groups; preemption required replacement between the16K and8K groups.

## Sweep contract and exact case driver

K=1024. ISLs:1K,2K,4K,5K,8K,16K,32K,55K,64K. Chunks:1K,4K,8K,16K,32K,64K where chunk<=ISL, yielding29cases. Order:64K,32K,16K,8K,4K,1K chunk groups, ascending ISL within each.

For the current b08 continuation, the controller executes this harness onprefill, with values supplied from the case:

```bash
ROOT=/data/divanovic/gptoss-payload8704-20261021
"$ROOT/tt-gen/adapters/dynamo/.venv/bin/python" \
  "$ROOT/tt-metal/models/demos/gpt_oss_d_p/repro/run_pd_case.py" \
  --run "$ROOT/runtime/sweep-b05-pd-c8192" \
  --output "$ROOT/evidence/sweep/pd/8k-c8k" \
  --model /mnt/weka/model-weights/llm/openai/gpt-oss-120b \
  --isl 8192 --chunk 8192 \
  --prefill-host bh-glx-120-b05u02 \
  --prefill-ip 10.32.16.193 --decode-ip 10.32.17.129 \
  --deadline 1792607986
```

This is an exact example of the existing coordinator's request, not an instruction to duplicate it. Outer request timeout is2100s, additionally constrained by the deadline. Outputcap32, EOSenabled, warmup+3 follow the unchanged pinned harness. Prompts, full outputs, usage, raw timings and failures are preserved.

## Accounting and reporting

- Newly prefilled + reused tokens equals logical ISL. Fixed-width padding is reported separately.
- Prefill throughput = newly prefilled tokens / host-observed prefill service span.
- TTFT = first streamed token arrival minus request start; total duration includes the streamed request through completion.
- KV wire payload is distinct from chunk-table metadata. Transfer sum, union-active duration and span have separate columns.
- Overlap = intersection of prefill and transfer active intervals. Non-overlapped prefill/transfer are the respective set differences. Initial prefill and final transfer tail expose sequential ends.
- Decode t/s/user = native host-commit generated-token intervals / first-to-last host-commit duration. It is not an HTTP wall-time proxy.
- Warmup remains separate; means and sample variances use the three measured requests.
- Concurrent reload phase totals and overlapping transfer durations are never added into wall time or TTFT.

Read-only final audit command:

```bash
python3 /data/divanovic/gptoss-payload8704-20261021/runtime/audit-pd-b05.py \
  /data/divanovic/gptoss-payload8704-20261021 \
  --output /data/divanovic/gptoss-payload8704-20261021/evidence/report/sweep-audit.json
```

The read-only collector `collect_b08.py` with`setup_summary_b08.py` reads saved files only. During measurement, local`publish-b08.py` added immutable2880/4352baselines and updates three identical-style graphs with synchronized chunk checkboxes, improvement columns against both references, full timing columns and per-runJSON. Missing values stay blank. `make-report.py` includes every slower mean with sampleSD; `export-results.py` exports all requests and independently recomputes statistics. Source/evidence packaging happens after the coordinator exits. No active script is modified or archived mid-run.

- Livepage: http://127.0.0.1:8768/payload4352-comparison.html
- Newreport: http://127.0.0.1:8768/PAYLOAD8704_REPORT.md
- Historicalcomplete4352report/package links remain on the page.

## Replacement event (October21,14:28UTC)

The user assigned already-reserved b05u02 under141909 after requesting clean shutdown/release of c05u02/141892. No measurements ran on the first8704prefill. Initial b05 health read0xffffffff on PCIe device26; recovery passed at14:34:21UTC, and post-recovery health reported32devices/32DRAMready. Decoder4143783 and its four ranks were retained. New group names use `sweep-b05-pd-c<chunk>` and fresh IPC/table artifacts. The old stopped group remains preserved. Prefill allocation ends20:25:51UTC; decoder ends19:03:46UTC, so common earlier experiment deadlines remain unchanged.

## Decoder cache-read failure and unchanged retry

The first decoder rank0 failed at14:38:06UTC with `OSError: [Errno 9] Bad file descriptor` while reading cached weight metadata. Other ranks continued independent captures until the collective failure check; overall exit1 occurred14:53:03 after35/41 image timing records. All ranks closed devices and exited. The failed attempt is preserved under `evidence/failed/decode-attempt1-cache-read` and excluded from successful startup duration, while its elapsed time remains in the setup breakdown.

A fresh host-only process successfully read and parsed both exact metadata sidecars100times with stable hashes. The5,309,568byte blob checksum matched its recorded content hash. Idle health passed32devices/32DRAMready. Cache resides onNFSv4; the evidence does not establish whether the transient failure originated in the filesystem or process-local descriptor handling. No cache deletion, weight conversion, model-code change or device reset was used.

One unchanged retry started14:59:48UTC with `runtime/start-decode-owner-b03-retry1.sh`, using fresh process-memory captures and the same source/weights/deadlines. The first prefill remained loaded and became ready14:59:24UTC after991.537s. The then-current coordinator validated the exact retry launch record and checked rankerrors as well as image counts/readiness. Successful repetition does not prove the original root cause. Final status and durations come from the saved report, not these historical PIDs.

The unchanged decoder retry reached full36 readiness at15:31:31UTC, with41/41images andfourliveowners,1902.425s (31m42s) afterlaunch. Its successful capture is recorded separately from the2133.658s failedfirstattempt. The originalEBADFcause remainsunresolved.


## Decoder preemption and continuation on b08u02

At2026-10-21 16:02:23UTC, Slurm preempted decoder job141893 onb03u02. The first eight cases were complete and audited. No8K request had started. The controller exited at16:03:13 because PAM correctly denied compute access without a reservation. Decoder process survival was not checked after ownership was lost. Do not infer liveness from old ready files.

The first replacement b03u08/job142022 failed its idle PCIe health probe and bounded site recovery; PCIe24–31 remained unreachable. Diagnostics remain under `evidence/recovery-decode-b03u08`. No model ran on this host. Its task descendants and device holders were verified absent, and its allocation was released.

The next replacement, **bh-glx-120-b08u02**, has IP`10.32.17.129` and job**142028** (six hours, expires22:13:34UTC). Idle health passed32devices and32DRAMready. Prefill remains **bh-glx-120-b05u02**, IP`10.32.16.193`, job**141909**, expires20:25:51UTC. The user requested three extra hours for both. Slurm denied both in-place extension attempts; the fresh six-hour decoder reservation includes the requested extra three hours. Prefill was not extended. Existing runner deadlines remain unchanged.

The continuation uses `resume-after-preemption-b08.py` on the laptop, with `ops_b08.py`, `ops_b08_graceful.py`, and `case_ops_b08.py`. It retains the ready8Kprefill and existing etcd/frontend, verifies the eight completed cases, waits for the recorded decoder build, launches one fresh decoder, checks early rank errors and readiness, then executes only8K/4K/1Kgroups. Never restart an old controller or launch this controller twice.

New commands and evidence:

- Build: `bash runtime/build-role-b08.sh decode`, bounded by the recorded2400s wrapper; `runtime/build-decode-b08-launch.json` and `evidence/build-decode-b08.log`/`build-decode-b08-exit.txt`.
- Owner: `bash runtime/start-decode-owner-b08.sh`; `runtime/decode-b08-owner-launch.json`, `evidence/decode-b08-owner.log`/exit. It captures the same36layers/16KVconfigs with131072capacity and two slots.
- Fresh output: `runtime/repro-b08-full36/decode`; socket prefix`gptoss_payload8704_20261021_b08`.
- Services: `python3 runtime/start-service-b08.py <component> --run <group>`. Existing frontend/etcd stay in`runtime/repro-full36/services`; they are not duplicated.
- Future groups: `python3 runtime/prepare-chunk-b08.py --chunk <4096-or-1024> --name sweep-b05-pd-c<chunk>`.
- Audit remains `python3 runtime/audit-pd-b05.py /data/divanovic/gptoss-payload8704-20261021 --output /data/divanovic/gptoss-payload8704-20261021/evidence/report/sweep-audit.json`, reading saved evidence only.

The partial8Kprefill manager was stopped and archived under`runtime/sweep-b05-pd-c8192/before-decoder-preemption-replacement`; prefill owner1817998 and its H2D/ACK identity were retained. Updated configs bind the new decoder host/IP/socket and table path. Device-free tests verified the remaining21pairs, no repeated completed groups, one decoder launch, stale-job rejection, early-error precedence over ready files, and new configuration paths. A bounded host TCP round trip passed.

Report the two8704decoder cohorts separately:64K/32K/16Kchunks onb03u02 and8K/4K/1Kchunks onb08u02. The host change prevents attributing all differences solely to packet size. Retain original setup, failed cache capture, successful preempted capture, failed replacement recovery and new startup as separate phases.

The b08 decoder reached readiness at17:06:42.218UTC after2613.718s (43m34s) from launch, with41captured images and all36layers/16KVconfigs. The existing controller attached native services and resumed the8K group at17:08:15UTC, retaining the prefill owner. This startup is a separate phase from the earlier b03 attempts. Exact readiness and owner verification are saved in `b08-ready-summary.json`.

## Final audit, publication and teardown

All 29 cases finished at 17:57 UTC. The final audit at 17:57:29.333471 UTC verified 116 requests (29 warmups and 87 measurements) with zero errors, failures or pending requests. Accepted request span was 8,581.476 seconds, including group changes and the preemption/replacement gap; the sum of individual request durations was 484.153 seconds. Neither measure includes initial startup before the first accepted request.

The source commit is `076d79443138613e9e1a96b948d02db69858274a`; its remote hash was verified. The existing configuration checks passed in 28.134 seconds, and the exact source hash remained unchanged. Final documentation hook results accompany the portable package.

The final prefill closed through the supported H2D sentinel with shutdown-complete, exit 0 and idle devices. Decoder owner-stop ended with MPI launcher exit 1 at 18:07:52.569903 UTC, reporting improper termination of rank 3. This warning is retained; it is not described as a clean decoder exit. Independent process and device-holder checks passed on both b05u02 and b08u02. The initial final-check SSH timeout was resolved by using the same accepted key directly through ProxyJump; shutdown was not repeated and no device reset was used. Both allocations 141909 and 142028 were cancelled at 18:13:53 UTC; terminal CANCELLED states and an empty queue were verified at 18:15:45.885228 UTC. Slurm accounting storage was disabled, so live controller records were used. See `allocation-release.json`. No compute access occurred after release.

The raw source/evidence archive is described by `source-evidence-archive.json`; it includes failed attempts, actual commands, prompts/outputs, stopped controllers, accounting and cleanup records. Every archive member has a verified SHA256. It predates the result-publication commit and excludes weight caches, environments and build trees.
