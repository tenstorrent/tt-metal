# GRPO Rollout Benchmark — ttml-local vs ttt-remote generation

Measures the **generation/rollout cost** of GRPO-on-BoolQ (Llama-3.2-1B-Instruct)
under two interchangeable backends. Everything except the generation path is held
identical, so any difference in `gen_time_s` / `tok_per_s` is attributable to
generation.

| Backend | Generation | Process model | Dispatch |
|---------|-----------|---------------|----------|
| `ttml`  | Local on the ttml mesh, `ttml.models.KvCache`, eager per-step decode (no trace) | 1 process | WORKER, 0 trace region |
| `ttt`   | Remote `TttGenerationWorker` (tt-transformers, data-parallel submeshes, captured decode trace) | 2 `tt-run` ranks (rank 0 = ttml training, rank 1 = ttt generation) | ETH, trace region |

**Why one backend per run (never both):** ttml opens Wormhole with WORKER dispatch
and a 0-byte trace region (hardcoded in `tt-train/sources/ttml/core/mesh_device.cpp`),
while tt-transformers needs ETH dispatch + a trace region. The two device bring-ups
are incompatible in one process, so the `ttt` backend is split across two
device-disjoint ranks and you select exactly one backend per invocation.

## Running

`TT_METAL_HOME` must be set. From anywhere:

```bash
tt-train/sources/examples/grpo_rollout_benchmark/runner.sh \
    --backend ttml --ttml-devices 2 --steps 20 --repeats 3

tt-train/sources/examples/grpo_rollout_benchmark/runner.sh \
    --backend ttt  --ttml-devices 2 --ttt-devices 2 --steps 20 --repeats 3
```

- `--ttml-devices` — ttml chips. `ttml` backend: `{1,2,4,8}`. `ttt` backend: the
  training rank's chips.
- `--ttt-devices` — (ttt backend only) generation rank's chips.
- `--steps K` — GRPO steps per run (default 20).
- `--repeats R` — run the whole bench R times (default 1). Each repeat is a fresh
  process (clean cold start) and appends to the same CSV, tagged by run index.

Supported `ttt` splits (train+gen chips): **2+2** (`configurations/local4`) and
**4+4** (`configurations/local8`).

## What is held constant (fair comparison)

`benchmark_common.py` pins `COMPLETIONS_PER_STEP` (32), `NUM_GENERATIONS` (4),
`MAX_COMPLETION_LENGTH` (256), `TEMPERATURE`, dataset seed, prompt, and reward. The
per-device train batch is derived from the device count so every configuration
generates the same batch each step — device count changes *speed*, not workload.
`--steps` is enforced by deriving `prompts_to_train`; no trainer changes.

## Output

One CSV per (backend, device split), all repeats appended, at
`generated/tt-train/grpo_bench/grpo_bench_{backend}_{ttml}x{ttt}.csv`:

```
run, step, backend, ttml_devices, ttt_devices, reward, avg_len,
gen_time_s, gen_tokens, tok_per_s, step_time_s
```

Each run also prints a median `gen_time_s` / `tok_per_s` over its steps (excluding
step 0, which absorbs device/trace warmup). Compare backends by reading those
medians; use the leading `run` column for run-to-run variance.

## Layout

```
benchmark_common.py   shared dataset/prompt/reward, balanced config, CSV monitor, WeightSyncCallback
bench_ttml.py         ttml backend entry (1 process)
bench_ttt.py          ttt backend entry (2 ranks; reads GRPO_BENCH_* env from runner.sh)
runner.sh             backend/device/steps/repeats dispatch + arg validation
configs/              ttml_{1,2,4,8}dev.yaml  (device_config + balanced batch)
configurations/       local4 (2+2) / local8 (4+4) tt-run rank bindings + mgd.textproto
utils/ttml_local/     LlamaGRPOCompleter + deps (local generation)
utils/ttt_remote/     LlamaCompleterRemoteRollout, TttGenerationWorker, MPI transport
```
`utils/` is split into two subpackages because the two backends' `llama_overrides.py`
differ; each subpackage is a self-contained copy of the files its completer imports.

## Caveats

- **ttml multi-chip runs carve the visible system via `TT_VISIBLE_DEVICES`.** Enabling
  fabric requires the visible cluster to equal the opened mesh — otherwise the extra
  chips are inactive and `open_device` fatals ("Fabric is being used but Device N is not
  active"). `runner.sh` exports `TT_VISIBLE_DEVICES` (echoed before each run) covering
  exactly `ceil(devices/2)` N300 boards (board = 2 chips): 2→`0`, 4→`0,1`, 8→`0,1,2,3`.
  These are **PCIe board indices, hardware-specific** — override by pre-setting
  `TT_VISIBLE_DEVICES`. After masking, chips renumber to UMD ids `0..N-1` (MMIO chips
  first, then their remote halves), which is why the YAML `device_ids` start at 0; if the
  logical mesh ordering matters for DDP correctness, verify the physical mapping.
- **`ttt` 4+4 (`local8`) is known to hang** in the cross-rank handshake on this host
  (inherited from `grpo_remote_rollout`). 2+2 is the reliable ttt path today.
- **`ttml_8dev.yaml` is new**: validate on-device that `enable_fabric(8)` resolves to a
  valid mesh-graph descriptor on the 4×N300 host before trusting its numbers. Its
  `device_ids` are UMD chip ids `0..7` (not the PCIe indices `TT_VISIBLE_DEVICES` uses).
- The two-rank rank bindings (`configurations/*/rank_bindings.yaml`) pin whole N300
  boards via `TT_VISIBLE_DEVICES` and are **hardware-specific** — adjust for your host.
- Fabric is pinned `FABRIC_2D` once at import in `bench_ttt.py` and must never be re-set.
