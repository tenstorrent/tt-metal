# Stage 8: datatype sweep (short)

**Selected: C0, the all-BFP8 policy `act_bf16__w_bfp8_all__hifi2` (person decision).** BFP8 weights for
every projection, HiFi2 matmuls with fp32 accumulation, BF16 activations, embedding and norms, fp32
DeltaNet recurrent state, fp32 decision head. This is the stage-6 policy. Stage 6's latency and accuracy
numbers stay the headline. The policy now lives in
[`selected_precision_config.json`](selected_precision_config.json), and the default construction path reads it.

Gate (person-approved, unchanged), on the 25-row stage-6 HF bf16 golden (`tests/e2e/test_model.py`):
TT top-1 == HF top-1 on >= 24/25 rows, with a miss allowed only where the HF top-2 gap is < 0.05. The
readout-logit PCC must be >= 0.99 on every row.

C0 result (measured, this sweep): **25/25 agree**, logit PCC min 0.99801 / median 0.99995,
final-hidden PCC min 0.99898, max |prob diff| 0.0394. 27.30 GiB DRAM after load. Warmed request latency,
burst median (ms): 148.6 / 366.1 / 665.6 / 1487.0 / 3521.7 at buckets 128 / 1024 / 2048 / 4096 / 8192.
This matches stage 6 within 1 % (stage 6: 148.4 / 366.0 / 665.4 / 1481.5 / 3512.9).

## Candidates

All candidates use BF16 activations, a BF16 embedding, fp32 DeltaNet state and an fp32 head. Latency is the
full request (tokenize + upload + forward + readback), batch 1, eager, from `tests/perf/test_model_perf.py`,
unchanged. Burst is the median of 5 passes, each after 10 s idle. Sustained is 2 warm-ups, then the median
of 7 back-to-back passes.

| id | MLP gate_up / down | attention, GDN, readout | agree | logit PCC min / median | hidden PCC min | max dprob | DRAM GiB | burst ms 128 / 1024 / 2048 / 4096 / 8192 | sustained ms at 2048 | gate |
|---|---|---|---:|---|---:|---:|---:|---|---:|---|
| **C0** | bfp8 HiFi2 | bfp8 HiFi2 | 25/25 | 0.99801 / 0.99995 | 0.99898 | 0.0394 | 27.30 | 148.6 / 366.1 / 665.6 / 1487.0 / 3521.7 | 924.7 | **pass (selected)** |
| C1 | bfp4 HiFi2 | bfp8 HiFi2 | 25/25 | **0.98895** / 0.99967 | 0.98935 | 0.0636 | 19.33 | 138.8 / 365.5 / 663.4 / 1456.7 / 3371.0 | 853.4 | fail: `l01_faq_match_230opt` logit PCC 0.98895 < 0.99 |
| C2 | bfp4 LoFi | bfp8 HiFi2 | 25/25 | **0.98895** / 0.99967 | 0.98935 | 0.0636 | 19.33 | 138.7 / 298.5 / 552.4 / 1197.8 / 2791.5 | 700.2 | fail: same row, same value |
| C3 | bfp4 LoFi (+ bfp4 LoFi attention qkv+gate, GDN in_qkv+in_zba) | o_proj / out_proj / readout bfp8 HiFi2 | - | - | - | - | - | - | - | not run (see below) |
| C4 | bfp8 LoFi | bfp8 HiFi2 | - | - | - | - | - | - | - | not run (see below) |

C3 and C4: not run. The person selected C0 after C1 failed the logit-PCC gate (l01 0.98895 < 0.99). C2 had
already been measured when that decision arrived; its numbers are kept here for information.

Findings (measured unless labelled):

- The BFP4 MLP passes the decision part of the gate (25/25) but fails the per-row logit-PCC part on one row:
  `l01_faq_match_230opt` (230 options, 3303 tokens). Its PCC is 0.98895 against a 0.99 bar. C0 gets 0.99801
  on the same row.
- C1 and C2 give bit-identical outputs on all 25 rows: the same logits, PCCs and per-layer trace. Inferred:
  LoFi loses nothing over HiFi2 when the weight is BFP4 and the activation BF16. C2 is 17 % faster than C0 at 2048
  burst (552.4 vs 665.6 ms) and 21 % faster at 8192 (2791.5 vs 3521.7 ms).
- BFP4 with HiFi2 (C1) is barely faster than BFP8 (C0) above the 128 bucket (663.4 vs 665.6 ms at 2048). At
  these prefill lengths the MLP matmuls are compute bound (inferred), so fidelity matters more than weight bytes.
- DRAM: a BFP4 MLP frees 7.97 GiB (27.30 -> 19.33 GiB). This matters for stage 12, where the vision weights
  must also fit. With C0 selected, 4.29 GiB stays free after load (measured from the e2e load log).
- Not measured: C4 (BFP8 MLP + LoFi) and a mixed policy that restores `mlp_down` to BFP8 while keeping
  `mlp_gate_up` BFP4 LoFi. These two are the most likely to pass the gate faster than C0.

## Default path and proof of consumption

- `tt/optimizations.py`: `PrecisionPolicy.default()` reads `doc/datatype_sweep/selected_precision_config.json`.
  `PplxDeciderModel.from_snapshot()`, `Optimizations.build()` and the stage-1 test helpers call it when they
  get no explicit policy. The demo, the e2e test and the perf test therefore take the selected policy without
  extra flags. `$PPLX_DECIDER_PRECISION_CONFIG=<json>` swaps in another policy; the candidate files are in
  [`candidates/`](candidates/).
- `tests/e2e/test_precision_config.py` builds the model through the default path (layers 0 and 3, embedding,
  head). It reads `tensor.dtype` of every loaded projection weight, records weight dtype, `math_fidelity` and
  output dtype of every projection matmul in one forward, and asserts both match the JSON. Measured on the
  default path: all 7 roles show `BFLOAT8_B` device tensors and `MathFidelity.HiFi2` matmul calls with BF16
  output. With `$PPLX_DECIDER_PRECISION_CONFIG=candidates/C2_bfp4_mlp_lofi.json`, the MLP roles show
  `BFLOAT4_B` and `LoFi`.
- `pytest tests/e2e/test_precision_config.py tests/e2e/test_model.py -q -s` on the default path: **7 passed**
  (1 + the 6 stage-6 tests), 25/25, logit PCC min 0.99801. Log:
  `artifacts/pplx_decider/stage8/default_path/pytest.log`.
- The weight cache moved to one directory for all policies, `weight_cache/<revision>/weights`. A cache file
  name carries the dtype, so a candidate adds only the files for its new dtypes. The BFP4 files from C1/C2
  (9.0 GB) were deleted after the sweep. The 27 GB BFP8 cache (819 files) remains.

## Not done (person decision: wrap up with C0)

- Pareto charts, post-selection re-measure (the C0 numbers above and stage 6 stand) and the stage-1 per-layer PCC
  re-run. C0 is the stage-1 policy, so its stage-1 PCC numbers apply unchanged.

## Files

| file | content |
|---|---|
| `selected_precision_config.json` | The selected policy (C0), read by `PrecisionPolicy.default()`. |
| `candidates/*.json` | One policy per candidate. |
| `sweep_results.json`, `sweep_results.csv` | Per candidate: group dtype/fidelity, gate result, PCCs, DRAM, latency per bucket, command, hardware. |
| `sweep_report.py` | Builds the two results files from `artifacts/pplx_decider/stage8/<candidate>/`. |
| `work_log.md` | Commands, timings and decisions. |
