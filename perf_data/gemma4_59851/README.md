# Issue #59851 data: Gemma4 prefill matmul ns/token vs chunk size

Work-in-progress data, not for merge. Drop this directory before opening a PR.

Branch `mmanzoor/gemma4-matmul-chunk-sweep` = main 6ba9d172b63 + the scripts in `models/demos/gemma4_d_p/scripts/`.
Single Blackhole chip for the sweeps (one CP8/TP4 device's shapes, production code path); 8x4 galaxy for the layer
profile.

Results page: https://claude.ai/artifact/HcxvCfFyHP5fhGPPV1hNpt (private; source in `artifact/`).
Issue comment draft: `issue_comment_draft.md`.

## Reproduce on any Blackhole galaxy

From the tt-metal root, python env active (`HF_HOME`, `HF_MODEL=google/gemma-4-31B-it`, `TT_CACHE_PATH` set for the
`real` and `layers` steps; the HF snapshot must already be local):

```bash
S=models/demos/gemma4_d_p/scripts/run_matmul_chunk_sweep.sh
$S <out> sweep 3                        # 1 chip, ~8 min per run: every projection x C=256..32768, prod/LoFi/HiFi2
$S <out> real                           # 1 chip, ~2 min: real TP-rank-0 weights at C=1K..16K (extracts them first)
$S <out> fallback                       # 1 chip, ~8 min: LoFi + fp32 acc on ttnn's default config at C >= 10K
$S <out> layers 4096 8192 16384 2048    # all 32 chips, ~50 min: per-layer profile for the shares of layer time
$S <out> report                         # no device: rebuild every summary and figure from <out>
python perf_data/gemma4_59851/artifact/build.py <out> [<other machine summary.csv>]   # rebuild the results page
```

The scripts each step uses:
- `matmul_chunk_sweep.py`: the measurement.
- `matmul_chunk_sweep_report.py`: summary.csv, raw.csv, ns_per_token.png.
- `matmul_chunk_sweep_plot.py`: the compact figure.
- `matmul_chunk_sweep_weights.py`: real weights.
- `layer_perf_shares.py`: layer shares and the whole-prefill estimate. It needs no `tt_perf_report`, unlike
  `layer_perf_report.py`.

Checked on c04u02 against the earlier outputs: `report` reproduces summary.csv, fallback/summary.csv and
layer_shares.csv cell for cell, and `matmul_chunk_sweep_weights.py` reproduces the earlier weight file bit for bit.

## Files

- `c04u02/`: final results, bh-glx-120-c04u02 (2026-10-14; GDDR 14 Gbps, TDP limit 130 W)
  - `summary.csv/.txt`: 1 run, 375 points (7 ops x 25 C; attention at prod/LoFi/HiFi2).
  - `fallback_summary.csv/.txt`: `--fallback-lofi` experiment, 2 runs, C=10240..32768.
  - `layer_shares.csv/.txt`: galaxy layer profile at 2K/4K/8K/16K, plus the 256K prefill estimate.
  - `ns_per_token.png`: every op and metric.
  - `ns_per_token_compact.png`: the issue figure.
- `b09u08_*`: first machine, bh-glx-120-b09u08 (2026-10-10): 3 sweep runs and the real-weight check. It agrees with
  c04u02 within 5% for C > 2048. At C <= 2048 it is 5-18% faster; that region is bound by weight reads from DRAM, and
  its GDDR speed is unknown.
- `artifact/`: results page source.
  - `template.html`.
  - `data.json`: embedded data.
  - `build.py`.
- Raw data on the NFS share 10.32.13.1:/data (not in git; the repo's pre-commit hook caps files at 500 KB):
  - `/data/mmanzoor/matmul_chunk_sweep_c04u02/`: profiler dirs, raw.csv, meta.jsonl.
  - `/data/mmanzoor/matmul_chunk_sweep/`: b09u08 data, `bundle_59851.tar.gz`, `tp0_weights.pt`.

## Configuration regimes on main (the curves have steps here)

| C | program config | attention compute | MLP compute |
|---|---|---|---|
| <= 2048 | 1D in0-mcast, in0 width-sharded in L1 (64-84 cores; down 42) | HiFi2, fp32 acc, L1 acc | LoFi, fp32 acc, L1 acc |
| 2304-3840 | explicit 2D mcast (72-120 cores) | HiFi2, fp32 acc | LoFi, fp32 acc |
| 4096-10240 | explicit 2D mcast (8192: 96 cores) | LoFi, fp32 acc | LoFi, fp32 acc |
| >= 12288 | ttnn default (per_core_M > 4 does not fit L1) | HiFi2, bf16 acc | LoFi, bf16 acc |

## Findings (c04u02)

- **Recommended chunk: C = 8192.** Matmuls over a 256K prefill take 3.52 / 2.33 / 1.56 / 1.60 s at C = 2K / 4K / 8K /
  16K, which is 32 / 32 / 25 / 28% of layer time.
- **Up to 2K, weight reads set the pace.** The projections read weights at 300-380 GB/s, so ns/token falls as 1/C.
  Attention fidelity matters little there (0-10%).
- **The 1D -> 2D switch at C = 2304 is a regression.** For example, up goes from 94 to 123 us.
- **C = 8192 uses 96 of 120 cores** and costs about the same per token as 7168. C = 10240, the last explicit config,
  is about 15% cheaper per token for every op.
- **From C = 12288, ttnn's default config is the limit.** Attention runs at HiFi2 with bf16 accumulation and is
  1.3-1.6x slower per token than at 8K. Forcing LoFi + fp32 acc helps only global QK; o_proj gets 1.4-1.6x worse.
- **MLP is about 68% of matmul time at 8K**, and it still improves at larger C.
- **SDPA is 50% of layer time at 8K.**
- **PCC is >= 0.9987 everywhere** (explicit configs >= 0.9997). Real weights time within 1.3% of random weights.
