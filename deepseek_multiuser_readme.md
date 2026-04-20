# DeepSeek V3 multi-user benchmark runbook

End-to-end workflow for running the DeepSeek V3 multi-user perf benchmark on a Blackhole Galaxy cluster, collecting ethernet link metrics before and after every run, and post-processing the logs into per-host charts and a combined CSV.

Contents:

- [One-time setup](#one-time-setup)
- [Per-experiment workflow](#per-experiment-workflow)
- [Post-processing](#post-processing)
- [Sample benchmark output](#sample-benchmark-output)

---

## One-time setup

Do these once per development machine. Repeat only when you want to update to newer branches.

### 1. Check out the required branches

```
# tt-metal
cd /path/to/tt-metal
git fetch origin
git checkout asaigal/multi_user_perf_debug
git pull

# tt-blaze (hosts the benchmark client)
cd /path/to/tt-blaze
git fetch origin
git checkout asaigal/pm_perf
git pull
```

### 2. Build tt-metal

```
cd /path/to/tt-metal
./build_metal.sh --build-tests
```

`--build-tests` is required because the post-mortem ethernet-metrics collector (`run_cluster_validation`) is built as part of the test suite.

### 3. Obtain the pipeline config file

The pipeline config file describes the model parallelism topology for your specific cluster (number of pods, hosts per pod, device assignments, etc). **Ask Aditya for the pipeline config file that matches the cluster you're running on.**

### 4. Generate the rank binding and rank file

Pipeline configs are the input; the rank-binding YAML and the MPI rank file are derived artifacts that the tt-run launcher consumes. Regenerate them whenever the pipeline config changes.

```
cd /path/to/tt-metal
python3 models/demos/deepseek_v3_b1/scaleout_configs/generate_blitz_decode_pipeline_configs.py \
    <PIPELINE_CONFIG_FILE>
```

This produces two artifacts in the current directory:

- `blitz_decode_pipeline_rank_binding_single_pod.yaml` — consumed by `tt-run --rank-binding`
- `blitz_decode_pipeline_rank_file_single_pod` — consumed by `mpirun --map-by rankfile:file=...`

### 5. Build the tt-blaze benchmark client

```
cd /path/to/tt-blaze/pipeline_manager
SKIP_TOKENIZER=1 ./setup.sh
```

The benchmark uses synthetic inputs, so the tokenizer dependency is not needed. `SKIP_TOKENIZER=1` keeps the build lean and faster.

---

## Per-experiment workflow

One "experiment" = recover the cluster → capture pre-workload link metrics → launch the model pipeline → run the benchmark → capture post-workload link metrics → kill the pipeline. You can repeat this loop as many times as you want; each iteration should use a distinct pair of log files so you can compare them later.

### 1. Recover the cluster

**Must be run before every experiment.** Resets Blackhole state on the given hosts so the next workload launch starts from a clean baseline.

```
cd /path/to/tt-metal
./tools/scaleout/exabox/recover.sh --hosts <COMMA_SEPARATED_HOST_LIST>
```

Example `--hosts`: `bh-glx-120-a06u02,bh-glx-120-a06u08,bh-glx-120-a07u02,bh-glx-120-a07u08`.

### 2. Collect pre-workload ethernet metrics

Dump every link's counters right after recovery, before any application traffic hits the fabric. This is the "clean baseline" snapshot for this run.

```
mpirun --host <HOST_LIST> \
    --mca btl_tcp_if_exclude docker0,lo,tailscale0 \
    ./build/tools/scaleout/run_cluster_validation --log-ethernet-metrics \
    2>&1 | tee pre_workload_run<N>.log
```

`<HOST_LIST>` is the comma-separated list of hosts (same as step 1). `pre_workload_run<N>.log` is the output — use a distinct filename for each experiment so post-processing can attribute counters to runs.

### 3. Launch the model pipeline

This brings up the DeepSeek V3 pipeline across all ranks. The command does not exit on its own — it prints a readiness line and then keeps sockets alive until interrupted.

```
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run \
    --tcp-interface ens5f0np0 \
    --mpi-args "--map-by rankfile:file=blitz_decode_pipeline_rank_file_single_pod \
                --bind-to hwt:overload-allowed \
                --host <host1>:4,<host2>:4,<host3>:4,<host4>:4 \
                --tag-output" \
    --rank-binding blitz_decode_pipeline_rank_binding_single_pod.yaml \
    python -m models.demos.deepseek_v3_b1.demo.cli \
        --max-new-tokens 8192 \
        --weights real \
        --cache-path /mnt/models/deepseek-ai/cache-2026-03-22/ \
        --launch-only
```

The `--host` value is a comma-separated list of `hostname:slots` entries, one per host in the pod. For a 4-host pod each host gets 4 slots (one per MMIO-attached Blackhole device), so the format is literally `host1:4,host2:4,host3:4,host4:4` with no spaces — substitute the actual hostnames used in steps 1–2.

**The pipeline is ready to receive benchmark traffic when you see:**

```
Pipeline launched; keeping sockets alive until interrupted.
```

Leave this terminal running. All subsequent steps happen in a separate shell.

Notes:

- `--launch-only` tells the CLI to bring up the pipeline and wait for external traffic rather than running its own inference loop.
- `TT_METAL_SLOW_DISPATCH_MODE=1` is required by this branch; do not omit it.

### 4. Run the multi-user benchmark

Open a second shell on the node where rank 0 is pinned.

First, set `TT_VISIBLE_DEVICES` to the device vector for rank 0. This vector is in `blitz_decode_pipeline_rank_binding_single_pod.yaml` in the tt-metal directory (generated in step 4 of [One-time setup](#one-time-setup)) — find the entry for rank 0 and export its `TT_VISIBLE_DEVICES` value verbatim:

```
export TT_VISIBLE_DEVICES=<TT_VISIBLE_DEVICES_VECTOR_FOR_RANK_0>
```

Then run the multi-user sweep:

```
cd /path/to/tt-blaze/pipeline_manager
./build-full/dummy_pipeline_connector \
    --h2d-socket-id deepseek_h2d \
    --d2h-socket-id deepseek_d2h \
    --sweep \
    --isl 128 \
    --osl 128 256 512 1024 2048 4096 8192 16384 \
    --connect-to-deepseek \
    --users 1 4 16
```

**What the benchmark does:** `dummy_pipeline_connector` connects to the already-launched DeepSeek pipeline via the host-to-device (`h2d`) and device-to-host (`d2h`) sockets, issues prefill + decode requests with synthetic token inputs, and measures two latency/throughput metrics end-to-end:

- **TTFT (Time To First Token)** — prefill latency, reported as Avg / P50 / P99 across the concurrent user population.
- **TPS (Tokens Per Second)** — decode throughput, reported per-user (Avg / P50 / Min) and as an aggregate across all users.

`--sweep` causes the benchmark to run the full cartesian product of the values passed to `--isl`, `--osl`, and `--users`, producing one perf table per concurrency level with rows for each ISL and columns for each OSL. In the command above, the sweep is:

- 1 ISL × 8 OSLs × 3 concurrency levels = **24 cells per experiment**

When the benchmark finishes it prints a table of perf numbers to stdout (see [Sample benchmark output](#sample-benchmark-output) below for the layout). Copy this output to a file so it can be paired with the pre- and post-workload ethernet metrics during post-processing.

### 5. Collect post-workload ethernet metrics

With the pipeline still running, capture the counters again. The counters accumulated during steps 3–4, so this snapshot records everything the workload did to the fabric.

```
mpirun --host <HOST_LIST> \
    --mca btl_tcp_if_exclude docker0,lo,tailscale0 \
    ./build/tools/scaleout/run_cluster_validation --log-ethernet-metrics \
    2>&1 | tee post_workload_run<N>.log
```

**Use a different filename from the pre-workload log** — the whole point is being able to diff the two.

### 6. Kill the workload

Tear down the pipeline. Run on the node where `tt-run` was launched (or any node; `pkill` runs locally):

```
pkill -f python
```

Be aware that `pkill -f python` will terminate *any* Python process on the host, not only the pipeline. If you have other Python work on this machine, narrow the pattern — e.g. `pkill -f deepseek_v3_b1.demo.cli`.

### Repeating experiments

Loop steps 1–6 as many times as you want. Each experiment should produce a distinct pair of ethernet-metrics logs, for example:

```
pre_workload_run1.log   post_workload_run1.log
pre_workload_run2.log   post_workload_run2.log
pre_workload_run3.log   post_workload_run3.log
...
```

Name them however you like — the post-processing scripts accept arbitrary file paths and let you assign run labels on the command line.

---

## Post-processing

Three scripts in the tt-metal repo consume the logs collected during the experiments:

- `plot_resends.py` — generates per-host bar charts for TXQ0 resends and Uncorrected_CW from the ethernet-metrics logs (total and max, across all supplied logs).
- `ethernet_metrics_to_csv.py` — emits a single CSV file with one row per (run, link) pair from the ethernet-metrics logs for offline analysis.
- `plot_benchmark_perf.py` — parses the benchmark perf tables saved in step 4, generates one bar chart per (metric × concurrency) comparing across runs, and emits summary + detail CSVs.

All three scripts take the same `LABEL=PATH` positional syntax: each positional arg is one log, optionally prefixed with a display name.

### Generate per-host bar charts

For a single cluster's worth of experiments, pick either the pre- or the post-workload logs (typically post-workload, since that's the state produced by the run):

```
cd /path/to/tt-metal
python3 plot_resends.py \
    --cluster-name "My Rev C Cluster" \
    --out-prefix revC \
    "Run 1=post_workload_run1.log" \
    "Run 2=post_workload_run2.log" \
    "Run 3=post_workload_run3.log" \
    "Run 4=post_workload_run4.log"
```

Output (in the current directory unless `--out-dir` is passed):

- `revC_resends_total.png` — per-host total TXQ0 resends across runs
- `revC_resends_max.png` — per-host max TXQ0 resends on any single link, with per-bar labels identifying the offending link
- `revC_ucw_total.png` — per-host total Uncorrected_CW across runs
- `revC_ucw_max.png` — per-host max Uncorrected_CW on any single link, with per-bar labels

Hosts are auto-discovered from the logs; no hardcoded host lists are needed. Pass `--help` for the full CLI.

### Generate the combined CSV

```
cd /path/to/tt-metal
python3 ethernet_metrics_to_csv.py \
    --output ethernet_metrics.csv \
    "Run 1=post_workload_run1.log" \
    "Run 2=post_workload_run2.log" \
    "Run 3=post_workload_run3.log" \
    "Run 4=post_workload_run4.log"
```

One row per (run, link) pair. Columns:

| Column | Meaning |
|---|---|
| `run` | Run label from the CLI |
| `host`, `tray`, `asic`, `channel`, `port_id`, `port_type` | Link identifier |
| `txq0_resends`, `txq1_resends`, `txq2_resends` | Per-queue TX retry counts |
| `total_txq_resends` | Sum of TXQ0 + TXQ1 + TXQ2 |
| `corrected_cw` | FEC-corrected codewords |
| `uncorrected_cw` | Codewords that escaped FEC (TXQ0 retransmits are triggered by these on the peer endpoint) |
| `crc_err` | Packet-level CRC errors (count of packets that reached the framer with uncorrectable errors) |
| `retrains` | Number of link renegotiations during the run |
| `unique_id` | 64-bit board/chip identifier — all links on the same ASIC share this value |

Rows are grouped by run (in CLI order), then sorted within each run by `(host, tray, asic, channel, port_id)`.

### Generate benchmark-perf charts and summary tables

For the benchmark output you saved from step 4 of each experiment, run:

```
cd /path/to/tt-metal
python3 plot_benchmark_perf.py \
    --out-prefix benchmark \
    --out-dir ./out \
    "Run 1=benchmark_run1.log" \
    "Run 2=benchmark_run2.log" \
    "Run 3=benchmark_run3.log" \
    "Run 4=benchmark_run4.log"
```

Outputs (in `./out/`):

- One grouped bar chart per (metric × concurrency) — OSL on the X axis, one bar per run at each OSL. Metrics charted: Avg TPS (per user), P99 TTFT, Avg TTFT, Aggregate Output tok/s. For a run that sweeps 1u/4u/16u/64u, that's 16 PNGs total.
- `benchmark_summary.csv` — one row per (run × concurrency) with summary columns: mean / min / max per-user TPS, peak and mean aggregate tok/s, mean Avg TTFT, P50/P99 TTFT, total errors. A Markdown version of the same table is also printed to stdout.
- `benchmark_detail.csv` — tall format, one row per (run × concurrency × ISL × OSL × metric × value) — friendly for pivoting in Excel/pandas if you need to drill into specific cells.

The script accepts any file that *contains* the benchmark output (you can tee the full terminal dump from step 4); it scans for the `## Concurrency: N users` / `====== METRIC ======` section headers and ignores everything else.

### Comparing pre- vs. post-workload logs

You can also CSV both the pre- and post-workload logs and diff them to see exactly what the workload contributed per link:

```
python3 ethernet_metrics_to_csv.py \
    --output ethernet_metrics_pre.csv \
    "Run 1=pre_workload_run1.log" "Run 2=pre_workload_run2.log" ...

python3 ethernet_metrics_to_csv.py \
    --output ethernet_metrics_post.csv \
    "Run 1=post_workload_run1.log" "Run 2=post_workload_run2.log" ...
```

Then join the two CSVs on `(run, host, tray, asic, channel, port_id)` and subtract to get workload-only deltas.

---

## Sample benchmark output

The table below is produced by a successful run of the multi-user benchmark on a healthy 4-host Rev C configuration. The layout is one block per concurrency level (1, 4, 16 users), each block containing rows of perf metrics over the ISL × OSL grid.

```
######################################################################
## Concurrency: 1 users
######################################################################

======================================================================
Avg TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        16.8        16.8        16.8        16.8        16.8        16.8        16.8        16.8

======================================================================
P50 TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        16.8        16.8        16.8        16.8        16.8        16.8        16.8        16.8

======================================================================
P99 TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        16.8        16.8        16.8        16.8        16.8        16.8        16.8        16.8

======================================================================
Avg TPS (per user)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       612.4       612.3       610.8       607.5       608.6       607.8       600.8       582.2

======================================================================
P50 TPS (per user)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       612.4       612.3       610.8       607.5       608.6       607.8       600.8       582.2

======================================================================
Min TPS
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       612.4       612.3       610.8       607.5       608.6       607.8       600.8       582.2

======================================================================
Aggregate Output tok/s
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       570.9       590.9       599.9       602.1       605.8       606.4       600.1       581.9

======================================================================
Elapsed (s)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        0.22        0.43        0.85        1.70        3.38        6.75       13.65       28.15

======================================================================
Errors
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128           0           0           0           0           0           0           0           0


######################################################################
## Concurrency: 4 users
######################################################################

======================================================================
Avg TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        61.6        61.6        61.6        61.6        61.6        61.6        61.6        61.6

======================================================================
P50 TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        62.1        62.1        62.1        62.1        62.1        62.1        62.1        62.1

======================================================================
P99 TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        63.1        63.1        63.1        63.1        63.1        63.1        63.1        63.1

======================================================================
Avg TPS (per user)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       608.5       609.2       607.9       605.7       606.7       606.2       600.2       581.9

======================================================================
P50 TPS (per user)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       609.9       609.9       608.2       605.8       606.8       606.3       600.2       581.9

======================================================================
Min TPS
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       605.9       607.9       607.2       605.3       606.5       606.2       600.1       581.8

======================================================================
Aggregate Output tok/s
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128      1887.2      2128.0      2267.5      2338.4      2383.8      2403.3      2390.1      2322.5

======================================================================
Elapsed (s)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        0.27        0.48        0.90        1.75        3.44        6.82       13.71       28.22

======================================================================
Errors
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128           0           0           0           0           0           0           0           0


######################################################################
## Concurrency: 16 users
######################################################################

======================================================================
Avg TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       241.6       241.6       241.6       241.6       241.6       241.6       241.6       241.6

======================================================================
P50 TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       241.8       241.9       241.8       241.8       241.9       241.9       241.9       241.9

======================================================================
P99 TTFT (ms)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       249.8       249.8       249.8       249.8       249.9       249.8       249.9       249.8

======================================================================
Avg TPS (per user)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       502.1       507.8       509.2       508.9       510.9       512.0       507.8       493.9

======================================================================
P50 TPS (per user)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       502.3       507.9       509.3       509.0       510.9       512.0       507.8       493.9

======================================================================
Min TPS
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128       491.1       502.1       506.4       507.5       510.2       511.6       507.6       493.8

======================================================================
Aggregate Output tok/s
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128      4125.1      5492.8      6569.5      7270.1      7709.4      7951.8      8004.4      7845.3

======================================================================
Elapsed (s)
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128        0.50        0.75        1.25        2.25        4.25        8.24       16.38       33.41

======================================================================
Errors
======================================================================
   ISL \ OSL         128         256         512        1024        2048        4096        8192       16384
------------------------------------------------------------------------------------------------------------
         128           0           0           0           0           0           0           0           0
```

Save this output verbatim alongside the pre- and post-workload ethernet-metrics logs for the experiment — having all three artifacts makes it trivial to correlate "which cell regressed" with "which link drove the regression".
