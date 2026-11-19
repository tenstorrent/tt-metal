# Llama-3.3-70B TP32 serving on WH Galaxy: how to reproduce the reference run

This folder runs the model the way the reference numbers below were measured: same branches, same
server flags, same config, same benchmark, same scoring and gates. Nothing else is needed. In
particular, **no `LLAMA_GALAXY_*` or `TT_PLUGIN_*` environment variables**: every kept lever is now
the default code path, and the scheduler settings are config keys.

## 1. Checkouts

| repo | branch | commit |
|---|---|---|
| tt-metal | `gtobar/prefill-defaults` | the commit that contains this README |
| vllm-tt-plugin (fork `gtobarTT/vllm-tt-plugin`) | `gtobar/serving-config` | `6d24eff` |

```bash
cd ~/teja/tt-metal && git fetch origin gtobar/prefill-defaults && git checkout gtobar/prefill-defaults
cd ~/teja/vllm-tt-plugin && git remote add gtobar https://github.com/gtobarTT/vllm-tt-plugin.git
cd ~/teja/vllm-tt-plugin && git fetch gtobar gtobar/serving-config && git checkout gtobar/serving-config
```

- Only Python and YAML changed since `gtobar/prefill-levers` (`3dd62e0b`). If that branch is
  already built in `~/teja/tt-metal`, no rebuild is needed.
- The plugin must be installed from this checkout (editable install), the same as before.
- Both trees must be clean under `models/` and `src/`. The script logs the commit and a dirty count of
  each tree.

## 2. Clear old switches

```bash
unset $(env | cut -d= -f1 | grep -E '^(LLAMA_GALAXY_|TT_PLUGIN_EXP_|TRACE_REGION_SIZE$)')
```

The script refuses to start (exit 7) while any of them is set. The env-var names from the previous
branches are no longer read, but a mix of old and new setups is the main way runs stop being comparable.

## 3. Run

```bash
cd ~/teja/tt-metal/models/demos/llama3_70b_galaxy/vllm_tp32
bash run_serving.sh <tag>
```

`MMLU=limit` runs the 20-per-subject MMLU subset instead of the full set; `MMLU=0` skips MMLU.
Expect about 1 h 45 min with full MMLU. The first start after switching branches also builds the bfp4
weight cache, which adds time.

What the script does, in order:

1. Resets the Galaxy, starts the server in the background and waits until it answers.
2. Checks the server log for the five scheduler lines (keep_decode_chain, admission cap,
   output flush, interleave disabled, arrival wait), then sends one smoke request.
3. Runs `~/teja/bench_tis.sh` (all SLA cases, fresh seeds) and scores it with `slide5.py`.
4. Runs `needle_probe.py` (long prompts) and `prefix_needle_probe.py` (prefix-cache hits).
5. Runs MMLU with `~/teja/run_mmlu.sh`: `NC=8`, `TMO=86400` for the full set. lm-eval queues every
   request at once and counts the queue wait against the client timeout, so a short timeout fails the
   full set.
6. Stops the server, resets the Galaxy, copies the logs back to `~/teja/runs/`.

Server log and bench output go to tmpfs (`LOGDIR`, default `/dev/shm/llama70b_tp32`) while running.
Logging to disk caused latency tails in earlier runs.

Exit code 0 means every SLA row and every gate passed. Otherwise:

| bit / code | meaning |
|---|---|
| 1 | an SLA row failed |
| 8 | MMLU below 80% or a garbage answer |
| 16 | long-prompt needle failed |
| 32 | prefix-hit needle failed |
| 3, 4, 5, 6, 7 | reset failed, server died at start, not ready in 2 h, smoke failed, old env var set |

## 4. The setup, in full

Server flags (in `run_serving.sh`):

```
--data_parallel_size 4 --max_num_seqs 8 --async-scheduling --host 127.0.0.1 --port 8000
--max-model-len 36864 --enable-prompt-tokens-details --override-generation-config '{"top_k": 32}'
--additional-config "$(cat serving_tt_config.json)"
```

`serving_tt_config.json`:

```json
{"tt": {"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D_RING",
        "worker_l1_size": 1344544, "trace_region_size": 800000000,
        "max_prefill_requests_per_step": 1, "flush_prefill_output": true, "decode_interleave_enabled": false,
        "keep_decode_chain": true, "arrival_wait_ms": 8}}
```

- `trace_region_size` 800000000 is also the registry value in `models/model_trace_region_sizes.yaml`.
  The 512-row cached-prefix shapes and the replica-miss traces do not fit in less.
- `arrival_wait_ms` requires `flush_prefill_output`. `keep_decode_chain` requires async scheduling.
  The plugin refuses to start otherwise.
- bfp4 weights (FF2, attention) and HiFi2 prefill SDPA are on through `LlamaOptimizations.performance()`,
  which is what vLLM uses. `accuracy()` keeps them off.

## 5. Differences from the earlier "config C" run

| | config C | this setup |
|---|---|---|
| precision | bfp8 FF2/attention weights, HiFi4 prefill SDPA | bfp4 FF2/attention weights, HiFi2 prefill SDPA |
| scheduler | async only | async + admission cap 1 + output flush + keep_decode_chain + arrival wait 8 ms |
| trace region | 640 MB | 800 MB |
| cached-prefix path | ring + replica | also 512-row padding at 2560/3584, replica miss with fill, ring group 1, replica up to 5120 |
| prefill ops | WO all-gather up to 2048, minimal matmul `qkv,mlp2048,wo2048,mlp3072`, 16 CCL buffers | WO all-gather off, minimal matmul `qkv,mlp2048,mlp3072,mlp512`, 32 CCL buffers |
| logs | disk | tmpfs |

## 6. Reference results

Same host, five runs of this configuration (pm12, pm14-16 used the old switches with the same values;
cfg1 is this folder's setup with no env vars). `score_<tag>.txt` has the same rows.

| SLA line | target | pm12-16 (4 runs) | cfg1 |
|---|---|---|---|
| Decode, 1 user at 4K (tok/s) | ≥ 25 | 58.0 | 58.0 |
| 60 RPM: TTFT mean / P99 | ≤ 1 / 5 s | 0.60-0.61 / 1.35-1.36 s | 0.60 / 1.35 s |
| 60 RPM: decode mean / P99 (tok/s/user) | ≥ 25 / 15 | 28.6 / 20.9 | 28.6 / 20.9 |
| 60 RPM: end-to-end mean / P99 | ≤ 20 / 140 s | 18.5 / 33.0 s | 18.5 / 33.0 s |
| 60 RPM + prefix: TTFT mean / P99 | ≤ 0.5 / 1.5 s | 0.4952-0.4989 / 1.13-1.14 s | 0.4959 / 1.14 s |
| 240 RPM: TTFT P99 / decode P99 (tok/s/user) | ≤ 5 s / ≥ 15 | 4.2 s / 16.3-16.4 | 4.2 s / 16.4 |
| Cold 16K / 32K TTFT | ≤ 5 / 10 s | 1.75-1.76 / 3.42 s | 1.76 / 3.42 s |
| Prefix hit rate | 40% | 39.0% | 39.0% |
| Needle / prefix needle | all | 21/21, 30/30 | 21/21, 30/30 |
| MMLU (mmlu_llama, 5-shot, chat) | ≥ 80% | 82.37% (20 per subject) | 82.27% ± 0.31 (full, 14042) |

Notes for comparing:

- Decode P99 is a floor: `1000 / p99_tpot_ms`, the tok/s/user of the slowest 1% of tokens. Higher is better.
- The prefix hit rate scores PASS at ≥ 38%. The workload is designed for 40%, and the target slide
  counts 38.7% as a pass.
- The 60 RPM + prefix TTFT mean sits at the target (0.4952-0.4989 over five runs). The script prints it
  with four decimals. It is the row most sensitive to host noise: logs on disk, low free disk space and
  other jobs on the host move it past 0.5.
- 128 concurrent users is out of scope at TP32 (4 lanes x 8 = 32 users).
