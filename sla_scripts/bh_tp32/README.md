# Llama-3.3-70B as one TP32 model on a Blackhole Galaxy

Scripts for the vLLM SLA runs and prefill profiling of `models/demos/llama3_70b_galaxy` on one BH Galaxy (8x4 mesh).
No paths are baked in: set these for your host first.

```bash
export TT_METAL_DIR=/path/to/tt-metal          # this repo, built, python_env with vLLM installed
export HF_MODEL=/path/to/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/path/to/weight-cache       # created on first run if empty
export PLUGIN_DIR=/path/to/vllm-tt-plugin
export OUT=/path/to/results/run1
source sla_scripts/bh_tp32/env.sh                 # best known model switches
```

`env.sh` sets `TT_CFG_EXTRA='"max_new_prefills_per_step": 1'`, a vllm-tt-plugin setting that only exists with
`vllm_tt_plugin_max_new_prefills.patch` applied (made against plugin 042a948):
`git -C $PLUGIN_DIR apply $TT_METAL_DIR/sla_scripts/bh_tp32/vllm_tt_plugin_max_new_prefills.patch`.
Without the patch, `export TT_CFG_EXTRA=` after sourcing `env.sh` (that costs the 32-at-once TTFT mean, about 4.0 -> 5.1 s).

| Script | What it does |
|---|---|
| `run_sla.sh` | Starts the server, runs every SLA line, prints the table, stops the server with SIGINT. |
| `run_server.sh` / `bench.sh` / `summary.py $OUT` | The same steps, one at a time. |
| `sanity_check.py [url]` | Known-answer check (short questions one at a time and batched, a ~9K-token continuation); `bench.sh` runs it first and stops if it fails. |
| `prefill_profile.sh "tag:ENV=1" ...` | Per-op device time of one prefill chunk of a 1-layer model, per variant. |

Tests in `models/demos/llama3_70b_galaxy/tests/` (run with `MESH_DEVICE=TG pytest <file> -s`):
- `test_prefill_switch_pcc.py`: full 80-layer model; last-token logits with the prefill switches vs all off.
- `test_rs_tune.py` / `test_mm_tune.py`: isolated reduce-scatter and minimal-matmul sweeps (`MM_M=2048|4096`).
- `test_prefill_layer_prof.py`: the 1-layer prefill that `prefill_profile.sh` profiles. `test_mesh_health.py`: quick all-gather check.
- `test_agmm_tune.py`: fused all-gather + minimal_matmul (`all_gather_minimal_matmul_async`) vs the unfused pair at the
  FF2 (and FF1/FF3+SwiGLU) prefill shapes on the 8x4 mesh.
- `test_cpf_port_pcc.py`: full 80-layer model, real text; the ported Wormhole prefill levers (pad 1024, seq-split,
  QKV seq-split, CPF ring + replica) and the BH switches (norm bf8, fused WO) cold and warm, vs levers off.
- `test_mmrs_prefill_tune.py`: fused `matmul_reduce_scatter_prefill` vs matmul + `reduce_scatter` at the FF1/FF3, WO and
  FF2 shapes (`MMRS_FABRIC`, `MMRS_PAYLOAD`, `MMRS_LINKS`). `test_mmrs_tune.py`: the strided fused variant at FF1/FF3.
- `test_overlap_prefill.py`: matmul beside a reduce-scatter on two sub-devices (the deepseek_v3_d_p pattern). The overlap
  variant does not run yet: `ttnn.reduce_scatter` on a sub-device that does not start at (0, 0) applies the offset twice
  and finds 0 cores.
- `test_pp_stage_bench.py` (single device): one full Llama-70B layer on one chip at 512-4096 tokens, the per-stage cost of
  a 32-stage pipeline-parallel prefill.
- `test_sdpa_decode_cores.py` (single device): paged decode SDPA at 48 / 64 / 88 cores.

`ttft_tail.py <result.json>`: where a TTFT tail comes from (slowest requests, burst size, early vs steady state); needs
`bench.sh`'s `--save-detailed` output. `LLAMA_SWITCH_TIMING=1`: logs the prefill<->decode switch cost every 100 switches.
`run_server.sh` takes `MAX_BATCHED_TOKENS` and `TT_CFG_EXTRA` (more `additional_config` "tt" keys).

Reset the Galaxy (`tt-smi -glx_reset_auto`) before an accuracy or SLA run whenever the previous process hung, was
hard-killed, failed to start, or ran an experimental op: the chips can keep producing garbage in the next process otherwise.

Decode timing: `LLAMA_DECODE_TIMING=1` logs host time per decode step every 200 steps (`DECODE_TIMING` lines in the server log).

The env switches are described in `env.sh` and at their use sites. All of them are off by default.

## Status (Nov 20 2026, tt-galaxy-43, `env.sh` as committed; runs/sla_cap1)

| SLA line | Target | Levers off | Now |
|---|---|---|---|
| 1 user 4K: TTFT / decode | 1 s / 25 | 0.32 s / 51.4 | 0.27 s / 51.4 |
| Cold 16K / 32K single | 5 / 10 s | 1.97 / 5.66 s | 1.58 / 5.03 s |
| 60 RPM: TTFT mean / P99 | 1 / 5 s | 0.62 / 1.14 s | 0.43 / 0.76 s |
| 60 RPM: decode mean / P99 | 25 / 15 | 29.4 / 22.9 | 35.3 / 29.8 |
| 60 RPM: e2e mean | 20 s | 17.1 s | 14.2 s |
| 60 RPM + 40% prefix: TTFT mean / P99 | 0.5 / 1.5 s | 0.77 / 2.93 s (fail) | 0.46 / 1.04 s |
| 60 RPM + 40% prefix: decode / e2e mean | 25 / 20 s | 26.7 / 20.0 s | 33.9 / 15.6 s |
| 240 RPM: TTFT mean / P99 | 1 / 5 s | 1.27 / 9.00 s (fail) | 0.58 / 3.00 s |
| 240 RPM: decode mean / P99 | 25 / 15 | 21.5 / 16.4 (fail) | 25.7 / 21.6 |
| 32 x 4K at once: TTFT mean / P99 | 1 / 5 s | 4.81 / 9.16 s | 3.99 / 7.60 s (fail) |
| 32 x cold 16K at once: TTFT | 5 s | not run | not run (about 10x away on one Galaxy) |
| Concurrent users | 128 | 32 | 32 (fail) |
| Quick MMLU / needle | 80% | 81.4% / 0 misses | 81.9% / 0 misses |

Next:
- 32 at once needs prefill throughput, not tuning: TP32 spends ~1.5 of every 3.3 ms per 4K layer in CCLs. One full layer
  on one chip (`test_pp_stage_bench.py`) gives a 32-stage pipeline 58-70K tok/s vs ~17.6K today (3.3-4.0x), before
  stage hand-off, KV re-shard into the decode layout and pipeline fill. Costs ~2.2 GB of extra weights per chip.
- 128 users: decode batch 128 (Wormhole TP8/DP4 ran 128 users at ~15.5 tok/s/user at 4K).
- Ruled out: 3-4 prefill links (the fabric has 2 eth channels per neighbor), decode interleave off, the prefill<->decode
  switch (~3% at 240 RPM), fused MM+RS for FF1/FF3/FF2 in-model, SDPA grid 11.
