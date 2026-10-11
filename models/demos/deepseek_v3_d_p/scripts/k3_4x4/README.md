# Kimi K3 prefill on one Blackhole galaxy: stage and chunk-size experiments

Scripts used for `tt/kimi_k3/K3_PREFILL_4X4_SINGLE_GALAXY_PLAN.md`: two 4x4 pipeline stages per galaxy,
per-op profiles, and chunk-size sweeps. They take the repo root from their own location; host-specific
paths have defaults that can be overridden with environment variables (see the header of each script).

| Script | What it does |
|---|---|
| `link_check.sh` | Checks the galaxy's ethernet links against the BH galaxy cabling spec. Exit 0 only when no ASIC pair is degraded. Run it after every reset, before any multi-stage run. `--retrain` lets `run_cluster_validation` retrain missing links first. |
| `run_stage_test.sh` | Runs a pytest through `scripts/run_safe_pytest.sh` on the top or bottom 4x4 half (or the full 8x4), with the matching device list and mesh graph descriptor. |
| `launch_runner.sh` | Runs the real prefill runner and producer: two 4x4 stages, one 4x4 stage, or one 8x4 stage, traced, with KV-cache PCC against the 56k golden. |
| `chunk_sweep.sh` | One traced 4x4 stage at several chunk sizes (default 5120 and 2560). |
| `binding_1rank_top_2d.yaml` | Rank binding for one 4x4 stage on the top half, FABRIC_2D. |
| `cadence.py` | Throughput from the runner's timing CSVs: steady-state tokens/s from the last rank's chunk cadence, plus per-rank compute. |
| `ops_cmp.py` | Per-op device time per chip across several `ops_perf_results` CSVs, with ratios. |
| `raw_ops.py` | Rebuilds a per-op CSV from the raw profiler logs when Tracy's report step fails. |

## Examples

```bash
S=models/demos/deepseek_v3_d_p/scripts/k3_4x4

# Gate: links must be clean before a two-stage run.
$S/link_check.sh || { tt-smi -r && $S/link_check.sh; }

# Two 4x4 stages, FABRIC_2D, 5120-token chunks; then the throughput.
$S/launch_runner.sh 2d 5120 ~/k3_runs/pp2_2d
python3 $S/cadence.py ~/k3_runs/pp2_2d 5120

# One 4x4 stage at 5120 and 2560-token chunks.
$S/chunk_sweep.sh ~/k3_runs/sweep 5120 2560

# Profile one 12-layer stage at 640 tokens per chip, then compare with another profile.
KIMI_K3_TEST_SEQ_LEN=2560 $S/run_stage_test.sh top x /tmp/p.log --profile \
  "models/demos/deepseek_v3_d_p/tests/kimi_k3/test_transformer_depth.py::test_depth_ladder_matches_golden[blackhole-L12-fabric2d-4x4]" -rs
python3 $S/ops_cmp.py a=<first ops_perf csv> b=<second ops_perf csv>
```

## Things that bite

- **A reset can leave link channels untrained.** A pair that trains with one of its two channels makes
  multi-stage runs hang intermittently. `tt-smi -r` does not always fix it; `tt-smi -glx_reset` (needs
  IPMI access, so usually run on the host) usually does. `link_check.sh` catches it.
- **The reference cabling spec does not match every galaxy.** On bh-glx-120-c02u02 the tray 1<->3 and
  2<->4 ASIC 3/4 links run on channels 8/9, not 0/1. `run_cluster_validation` reports those as missing
  plus extra; `link_check.sh` nets them out. Links that the spec does not list at all are not checked.
- **`run_safe_pytest.sh` verdicts:** it can print PASS on a pytest collection error and FAIL when only the
  profiler's report step failed. Read pytest's own summary line, which `run_stage_test.sh` prints.
- **Profile from the weight cache.** Without `TT_KIMI_K3_PREFILL_TTNN_CACHE` the depth test rebuilds every
  expert weight (~40 min), and under `--profile` Tracy then drops op records and the report fails.
- **ccache:** `TT_METAL_CCACHE_KERNEL_SUPPORT=1` breaks kernel builds with ccache older than 4.8, because
  `tt_metal/jit_build/build.cpp` passes ccache settings inline. The scripts unset it.
- **Configs are keyed to tokens per chip.** K3 tables have tuned entries at 640 and 1280 tokens per chip.
  Any other size falls back to default configs without a warning in most cases; check the profile.
