# GPT-OSS-120B: three payloads on Metal b3b5e13

## Cleanup outcome: unverified after allocation expiry

All87 cases are published and audited. Final process/device-holder absence was not verified before ownership expired. The decoder wrapper records timeout exit124; no supported owner-stop or manual base-service stop was sent. Slurm now has no active task allocations; terminal records are unavailable and accounting is disabled. No compute host was accessed after expiry. Full task completion is not claimed. See [DEADLINE-OUTCOME.md](DEADLINE-OUTCOME.md) and the [verified9928-file evidence archive receipt](source-evidence-archive.json).


## Measurement result

**87/87 cases complete and audited: 348 requests (87 warmups +261 measured), zero errors, failed cases or pending measurements.** The final request completed at07:09:23UTC; final prefill closure was verified at07:11:07UTC, and the controller exited. The publisher exited after a final independent saved-evidence audit. Publication and resource cleanup are recorded separately in the completion receipts.

Each table entry is mean ± sample SD of three requests after one warmup. These are descriptive serving-path observations; no packet-only, commit-only or numerical-accuracy claim is made. All slower results remain included.

[Interactive report](b3b5-packet-comparison.html) · [All metrics and previous comparisons](b3b5-comparison.csv) · [Every request](b3b5-runs.csv.gz) · [Full summary](full-b3b5-status.json.gz) · [Exact commands, pins, timing boundaries and baseline provenance](MEASUREMENTS.md)

## Throughput by input and chunk

Prefill tokens/s uses actual newly prefilled tokens, not padded capacity. In this run18 requests reused0 tokens and330 reused64 tokens. The first accepted case uses b09/b04u02; the other86 use b09/b08. The first B04 services had60 allowed CPUs versus64 for the model; B08 services use64. Hosts, cache state, fixed run order and historical source cohorts prevent a controlled packet-only comparison.

### Chunk 64K

| ISL | Payload2880 tokens/s | Payload4352 tokens/s | Payload8704 tokens/s |
|---:|---:|---:|---:|
|64K|21125.841 ± 9.707|22533.226 ± 32.760|23904.166 ± 8.524|

### Chunk 32K

| ISL | Payload2880 tokens/s | Payload4352 tokens/s | Payload8704 tokens/s |
|---:|---:|---:|---:|
|32K|23284.435 ± 33.641|24958.000 ± 31.826|26636.783 ± 67.271|
|55K|19438.292 ± 16.994|20795.908 ± 34.568|22078.274 ± 5.687|
|64K|21957.390 ± 8.114|23409.968 ± 25.756|24938.709 ± 32.434|

### Chunk 16K

| ISL | Payload2880 tokens/s | Payload4352 tokens/s | Payload8704 tokens/s |
|---:|---:|---:|---:|
|16K|23858.572 ± 190.839|25632.376 ± 28.391|27370.006 ± 164.530|
|32K|22938.175 ± 28.135|24586.367 ± 101.160|26236.873 ± 260.529|
|55K|19001.642 ± 6.946|20244.952 ± 31.547|21523.688 ± 14.912|
|64K|21689.485 ± 31.131|23095.991 ± 29.854|24587.356 ± 44.508|

### Chunk 8K

| ISL | Payload2880 tokens/s | Payload4352 tokens/s | Payload8704 tokens/s |
|---:|---:|---:|---:|
|8K|22902.987 ± 278.691|24341.063 ± 233.216|24504.970 ± 549.730|
|16K|23070.518 ± 88.245|24224.293 ± 270.161|24416.538 ± 189.397|
|32K|22075.865 ± 265.812|23466.633 ± 235.769|24082.293 ± 101.131|
|55K|21030.815 ± 24.259|22248.551 ± 73.424|23316.349 ± 169.350|
|64K|20975.344 ± 127.748|22241.313 ± 77.687|23466.526 ± 28.855|

### Chunk 4K

| ISL | Payload2880 tokens/s | Payload4352 tokens/s | Payload8704 tokens/s |
|---:|---:|---:|---:|
|4K|12544.029 ± 255.859|12692.815 ± 54.944|12028.354 ± 431.211|
|5K|7984.344 ± 141.321|7836.855 ± 107.312|7654.082 ± 134.814|
|8K|12728.351 ± 143.706|12550.568 ± 266.631|12536.253 ± 212.434|
|16K|12478.089 ± 253.331|12821.730 ± 141.098|12651.710 ± 137.490|
|32K|12883.105 ± 124.713|12827.124 ± 39.911|12848.649 ± 126.140|
|55K|12740.199 ± 139.504|12655.856 ± 101.316|12540.317 ± 230.406|
|64K|12837.480 ± 159.877|12749.382 ± 190.516|12574.608 ± 114.335|

### Chunk 1K

| ISL | Payload2880 tokens/s | Payload4352 tokens/s | Payload8704 tokens/s |
|---:|---:|---:|---:|
|1K|2840.810 ± 78.991|2983.694 ± 45.618|2955.999 ± 17.965|
|2K|3099.691 ± 35.555|3112.243 ± 70.827|3034.622 ± 77.474|
|4K|3173.992 ± 33.514|3109.295 ± 23.835|3096.116 ± 48.464|
|5K|3175.666 ± 29.862|3188.692 ± 44.943|3203.495 ± 19.481|
|8K|3208.481 ± 13.631|3218.068 ± 26.954|3159.636 ± 16.269|
|16K|3145.528 ± 27.469|3213.321 ± 3.200|3238.862 ± 25.819|
|32K|3185.282 ± 26.890|3206.274 ± 12.561|3234.721 ± 26.207|
|55K|3219.106 ± 30.755|3231.502 ± 24.936|3217.729 ± 20.992|
|64K|3182.400 ± 36.504|3249.594 ± 23.069|3221.320 ± 11.685|

## Timing interpretation

TTFT is request submission to the first nonempty content/reasoning delta. Migration intervals include source-manager command and successful completion, including control-plane time. Overlap is the intersection of observed host-prefill and manager-migration intervals. Transfer sum, union-active time and first-to-last span are distinct. Never sum overlapping intervals as wall time. Native decode uses host-commit token intervals; total request duration is client observed. All14 comparison metrics, sample SDs, previous means, differences and percentages are in the CSV; the full summary retains additional counters, warmups, prompt/output paths and response text.

## Setup and attempt separation

Accepted requests exclude startup/build time; startup is reported separately. Successful retained decoder B08: build/import304.063s, launch03:10:37.040328→READY03:48:34.688,2277.647672s; all41 images/full36/16KV. B04u02: READY after2406.083s, one accepted case, then preempted02:11:43. B04u08: launched02:32:13.849339,28/41images before preemption02:59:08, no accepted request. Their survival after ownership loss is unknown. B05 recovery failed before any model/build; initial B10 seven-bank hardware was incompatible with unchanged eight-bank layout. All raw attempt logs and rank capture/install timing remain in the verified evidence archive. They are not combined into one startup duration.

| Prefill group | Launch UTC | Ready log | Exit |
|---|---|---|---|
|p2880-c1024|2026-10-22T06:21:49.552666+00:00|2026-10-22 06:24:39.137|0|
|p2880-c16384|2026-10-22T04:37:02.425058+00:00|2026-10-22 04:40:18.489|0|
|p2880-c32768|2026-10-22T04:10:21.747891+00:00|2026-10-22 04:13:54.436|0|
|p2880-c4096|2026-10-22T05:45:59.250592+00:00|2026-10-22 05:48:38.407|0|
|p2880-c65536|2026-10-22T00:59:18.952752+00:00|2026-10-22 01:12:07.660|0|
|p2880-c8192|2026-10-22T05:06:35.354378+00:00|2026-10-22 05:09:26.184|0|
|p4352-c1024|2026-10-22T06:38:49.724921+00:00|2026-10-22 06:40:54.907|0|
|p4352-c16384|2026-10-22T04:48:07.510320+00:00|2026-10-22 04:50:27.003|0|
|p4352-c32768|2026-10-22T04:20:19.819777+00:00|2026-10-22 04:22:21.935|0|
|p4352-c4096|2026-10-22T05:58:28.834157+00:00|2026-10-22 06:00:25.548|0|
|p4352-c65536|2026-10-22T02:10:01.247006+00:00|2026-10-22 02:12:46.678|0|
|p4352-c8192|2026-10-22T05:26:32.284055+00:00|2026-10-22 05:28:28.418|0|
|p8704-c1024|2026-10-22T06:54:41.462370+00:00|2026-10-22 06:56:50.532|0|
|p8704-c16384|2026-10-22T04:57:10.632037+00:00|2026-10-22 04:59:46.087|0|
|p8704-c32768|2026-10-22T04:28:55.569302+00:00|2026-10-22 04:30:55.407|0|
|p8704-c4096|2026-10-22T06:10:05.450383+00:00|2026-10-22 06:12:15.207|0|
|p8704-c65536|2026-10-22T03:53:24.079553+00:00|2026-10-22 03:56:01.493|0|
|p8704-c8192|2026-10-22T05:36:09.990150+00:00|2026-10-22 05:38:07.524|0|

Detailed actual launch/readiness/closure records: [prefill-startup-closure.json](prefill-startup-closure.json). Per-image reload/capture and all build attempts are retained as separate raw logs, not summed as wall time.

## Incidents and limitations

The log-in timeout and later read-only connection reset have unresolved network causes. A separate disappearing-process cleanup bug was reproduced and repaired in a narrow versioned helper;25 host-only recovery guards passed. The final cleanup helper has10 host-only guard tests. No model requests or mutations were retried; all87 accepted cases use their original four requests.

- [DECODE-DRAM-INVESTIGATION.md](DECODE-DRAM-INVESTIGATION.md)
- [PREEMPTION-RECOVERY.md](PREEMPTION-RECOVERY.md)
- [SSH-CONTROL-INVESTIGATION.md](SSH-CONTROL-INVESTIGATION.md)
- [LOGIN-TIMEOUT-INVESTIGATION.md](LOGIN-TIMEOUT-INVESTIGATION.md)
- [STOP-PROCESS-RACE-INVESTIGATION.md](STOP-PROCESS-RACE-INVESTIGATION.md)

## Review and preservation

Independent review recomputed1479 mean/variance fields, checked348 requests and87 comparison rows, and verified the seven metric selectors/six tables. Final raw inventory covers8439 accepted-case files including prompts, outputs, request streams and counters. Padding capacity is an explicitly derived export, never relabeled as a measured counter. Baseline bytes remain frozen at SHA256 `93973bc249c60e441b7a8abf90a500e2fd94f280982468fe792f4528737d467d`.

Historical reports and their cohorts are included as baseline-2880-report.md, baseline-4352-report.md and baseline-8704-report.md. Failed attempts remain separate from accepted results. The completion receipt identifies publication commits, archive hashes, teardown outcomes and released allocation IDs.

Full exports are stored losslessly as gzip files to satisfy the repository file-size limit. `export-packaging.json` records compressed and original SHA256 values; decompress with `gzip -dk <file.gz>`. The interactive view uses compact table JSON; the full export retains every request and field.
