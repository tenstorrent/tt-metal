# Early-Q fused all_to_all_matmul vs TTNN chain: run scripts and results

- `scripts/mla_early_q_ab_glx.sh`: Galaxy 8x4 command file for tt-sbatch (`-p bh_110a_sc16`); runs
  `test_early_q_fused_ab.py` (full layer, early-Q block, per-op host times) with persistent scratch on and off.
- `scripts/mla-ab-loudbox.sh`: LoudBox bh-lb-a10u24, chips 0-3 as a 1x4 ring (`MLA_PERF_RING4=1`), via
  `scripts/exabox/node-run`; `AB_TEST` picks the test.
- `scripts/lb-guard.sh`: keeps the LoudBox reservation from idling (runs the op's unit tests after 45 idle minutes).
- `results/results.<job>.jsonl`: Galaxy runs. 239295: native op, one program per chip, per-call scratch, plus the
  generic op. 240094: persistent scratch on/off. 240679: one merged program for the mesh (4539a4d5d91).
- `results/loudbox.results.jsonl`: LoudBox 1x4 runs.
