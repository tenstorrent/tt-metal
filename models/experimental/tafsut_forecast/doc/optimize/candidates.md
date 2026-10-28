# Tafsut candidates (PLAN G1 item 6)

## Contract summary (CONTRACT v1.3)
- **Target:** `forward` on `paper`, 1024 series x context 2048 -> 64 (S = 97, pad 128), precision `accuracy`; device
  time from the traced bench (`tests/perf/test_forward_perf.py`).
- **Guards** (none may regress by more than 3 %): `long_context` (32 x 32768 -> 64, accuracy), `long_horizon`
  (512 x 2048 -> 2048, two forwards, accuracy), `paper` at `performance` (ships only at >= 5 % faster than accuracy
  and every floor passing, PLAN v1.12).
- **Floor:** `doc/optimize/gates.json` (the G1 floors, drift against the fixed round-0 records `iter_00/bench.json` +
  `iter_00/suite.json`, forward 1 and the `f2_*` teacher-forced forward 2), checked by `tests/perf/check_gates.py`
  on the full bench and suite sets.
- **Gate:** accept a change at >= 2 % target device time (`compare_bench.py ... --target paper:accuracy`), no guard
  worse than 3 %, `check_gates.py` 0 failures. G2 per round is automatic (A entries); G1 / G3 go to the orchestrator.
- **Exit:** after a round whose accepted changes give < 5 % in total; or matmuls > 65 % of FLOPs with non-matmul
  < 15 % of device time; or paper <= 200 ms accuracy / <= 150 ms performance; or 4 rounds done.

One line per candidate, in the order tried. Status: proposed / screening / kept / dropped / blocked. Rulings from D2.

| id | round | change | ruling (D2) | status | evidence |
|---|---|---|---|---|---|
| R0 | 0 | round-0 baseline: Chronos-2 structure (fused QKV, masked SDPA, traced forward, read-out cut) | fixed record (D2) | kept | `iter_00/bench.json`: paper 1697.47 ms accuracy, 1653.57 ms performance (job 1009) |
| R1-1 | 1 | RoPE: batch into M for `rot`, then fused `qkv_heads_rope` (model-local generic_op, absolute kernel paths, micro-bench, fresh-input trace test); alt: fold the signed permutation R into Wq/Wk (order-only if bitwise-equal weights, else v1.19) | pre-approved | proposed | |
| R1-2 | 1 | 2D program configs per role (one sweep job) | pre-approved | proposed | |
| R1-3 | 1 | relu fused in ff-up program config | pre-approved | proposed | |
| R1-4 | 1 | residual add in wo / ff-down epilogue, or add_rms_norm | pre-approved | proposed | |
| R1-5 | 1 | SDPA output_concat_heads | pre-approved | proposed | |
| R1-6 | 1 | long_context chunk micro-bench 1088/32 vs 1152/64 | pre-approved | proposed | |
| R1-7a | 1 | fp32_dest_acc_en off, HiFi2 linears | performance preset only; at accuracy needs an X gate | proposed | |
| R1-7b | 1+ | RMSNorm-gamma fold; exp_approx_mode | v1.19: variance table, full check_gates, device check that can fail | proposed | |
| R1-8 | 1 | QKV HiFi4 -> HiFi2 at the performance preset | allowed under drift; accuracy keeps HiFi4 | proposed | |

Tracking (PLAN v1.12): the performance preset ships only at >= 5 % faster than accuracy; round 0 it is 2.6 %.
