# AutoDebug Report: optimized-decoder evidence recovery

Date: 2026-07-31

## Scope and runner limitation

This is an inspection-only diagnosis for the AutoFix loop. No TT device command
was run and no implementation file was modified.

The required fresh AutoDebug runner was attempted from this directory:

```text
/home/mvasiljevic/tt-metal/.agents/scripts/autodebug.sh --agent codex ...
```

It exited immediately with `autodebug.sh: codex executable not found`.
Consequently this report is the documented serial fallback, not a successful
fresh CLI subprocess.

Problem under investigation: current optimized-decoder commit `c55a8c067c8`
references absent `candidates/*.log`; its checked-out profiler directories omit
CSV and console artifacts needed to verify candidate selection and the OPT-013
runtime dtype/fidelity policy.

## Headline findings

1. **The exact missing candidate logs are recoverable from reachable git
   commit `72e1a09218e`.** That commit contains 32 files under
   `doc/optimized_decoder/candidates/`, including both-batch default, split,
   packed-interleaved, HiFi2, BFP4-attention, BFP4-down, BF16-cache and geometry
   trials, plus real-weight correctness logs.
2. **The detailed final profiler artifacts are also recoverable from
   `72e1a09218e`.** For each of `final_full_b1`, `final_linear_b1`,
   `final_prefill_full_b1`, and `final_prefill_linear_b1`, it contains:
   `console.log`, detailed signpost-filtered `perf_report.csv`, aggregated
   `perf_summary.csv`, `perf_report.txt`, and `perf_summary.png`.
3. **These artifacts authenticate to the exact current implementation.** The
   blob ID of
   `models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py` is
   `102021a4c86d1bb894aede52d618f20d092b6d41` in all of:
   `ba942aefbd9`, `72e1a09218e`, `9a76c81ffab`, and current
   `c55a8c067c8`. A direct diff of `72e1a09218e` versus `c55a8c067c8`
   over the model `tt/` and `tests/` paths is empty. Commit `72e1a09218e` is a
   direct child of `ba942aefbd9`; it is the evidence follow-up created seconds
   after the implementation checkpoint.
4. **The recovered detailed CSV directly supplies OPT-013 evidence.** For
   example, `tracy/final_full_b1/perf_report.csv` records operation ID 189 as
   `MatmulDeviceOperation 32 x 5120 x 14336`, `LoFi BF16 x BFP8 => BF16`,
   `Input 1 Datatype=BLOAT8_B`, `DRAM Sharded=True`,
   `Input 0 Memory=DEV_0_L1_WIDTH_SHARDED`, and
   `Inner Dim Block Size=4`. The same CSV contains the other dominant rows
   needed to verify BFP4/LoFi packed gate/up and BFP8/LoFi output/down.
5. **The original raw Tracy operation CSV is not present in the reachable
   evidence commit.** The committed `perf_report.csv` is the detailed
   signpost-filtered op table, not the original large
   `ops_perf_results_*.csv`. Searches of the current model tree, `/tmp`, and
   surviving generated profiler directories did not find the four original raw
   files from the 2026-07-29 13:58/14:12 captures. The committed console logs
   show their provenance and the generated trace timestamps, but the raw files
   themselves appear to have been cleaned.
6. **Codex logging corroborates provenance but should not replace git
   artifacts.** `/home/mvasiljevic/.codex/logs_2.sqlite` contains the
   2026-07-29 session records that issued/inspected the Qwen candidate and
   profiler commands. Git blobs are the stronger recovery source because they
   preserve exact content and identity.

## Recovery inventory

Primary source commit:

```text
72e1a09218e4110dae8fef25a015fa4bc2f7fa21
parent: ba942aefbd9225f7b6265218737cf8839c188df5
subject: Record Qwen optimized decoder evidence
timestamp: 2026-07-29T14:17:01+00:00
```

Candidate evidence includes:

```text
candidates/default_{b1,b32}.log
candidates/default_linear_{b1,b32}.log
candidates/default_{full,linear}_prefill_{b1,b32}.log
candidates/split_default_{b1,b32}.log
candidates/packed_interleaved_{b1,b32}.log
candidates/default_hifi2_{b1,b32}.log
candidates/bfp4_attention_{b1,b32}.log
candidates/bfp4_attention_real_full.log
candidates/bfp4_down_{b1,b32}.log
candidates/bf16_cache_{b1,b32}.log
candidates/ds_gateup_iw{2,5,10}_*.log
candidates/geometry_{down_iw17_n2,o_iw24_n2,qkv_iw10_n7}_*.log
candidates/default_real_{full,linear}.log
```

Other exact evidence in the same commit:

```text
final_cache.log
final_static_checks.log
watcher/{full,linear}_{b1,b32}.log
tracy/final_{full,linear}_b1/{console.log,perf_report.csv,perf_summary.csv}
tracy/final_prefill_{full,linear}_b1/{console.log,perf_report.csv,perf_summary.csv}
```

The candidate log for final full batch-1, for example, ends with:

```text
FALLBACK_AUDIT throw_exception_on_fallback=True
FULL_TRACED_SYNTHETIC_PCC batch=1 step=1 0.9990026170563894
FULL_TRACED_SYNTHETIC_PCC batch=1 step=2 0.9995803962132329
FULL_TRACED_SYNTHETIC_LATENCY batch=1 median_ms=1.218025 min_ms=1.218025
```

This exactly supports the rounded README values.

## Hypothesis experiments

### H1: The README references evidence that was never preserved

Experiment:

```bash
git log --all -- models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder
git ls-tree -r --name-only 72e1a09218e -- \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder
git show 72e1a09218e:models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/default_b1.log
```

Result: the referenced logs are complete reachable git blobs.

Verdict: **refuted**.

### H2: The recovered evidence belongs to different implementation code

Experiment:

```bash
for c in ba942aefbd9 72e1a09218e 9a76c81ffab c55a8c067c8; do
  git rev-parse \
    "$c:models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py"
done
git diff --stat 72e1a09218e c55a8c067c8 -- \
  models/autoports/qwen_qwen3_6_27b/tt \
  models/autoports/qwen_qwen3_6_27b/tests
```

Result: every revision reports implementation blob
`102021a4c86d1bb894aede52d618f20d092b6d41`; the code/test diff is empty.

Verdict: **refuted**.

### H3: OPT-013 requires a new device profile because row dtype/fidelity was lost

Experiment:

```bash
git show \
  72e1a09218e:models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_full_b1/perf_report.csv
git show \
  72e1a09218e:models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_linear_b1/perf_report.csv
```

Result: both detailed tables include per-op input/output datatypes, math
fidelity, DRAM-sharded status, input memory configuration, inner block width,
device time and advice. They are sufficient to verify that the selected policy
reached measured dominant operations.

Verdict: **refuted for OPT-013**. A rerun is required only if the reviewer
specifically requires the original unfiltered `ops_perf_results_*.csv`, not for
runtime dtype/fidelity verification.

### H4: The original raw profiler CSV can be recovered locally

Experiment: searched the current checkout, generated profiler report roots and
`/tmp` for the four capture timestamps and for Qwen raw profiler filenames.
Inspected the reachable git trees and original console logs.

Result: the detailed filtered and summary CSVs survive in git; the original raw
operation CSVs for these four captures do not.

Verdict: **not recoverable from inspected local state**.

## Recommended smallest repair

Restore only the omitted evidence from the authenticated evidence commit; do not
rerun hardware merely to reproduce already preserved results:

```bash
git restore --source=72e1a09218e -- \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/final_cache.log \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/final_static_checks.log \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/watcher \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_full_b1/console.log \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_full_b1/perf_report.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_full_b1/perf_summary.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_linear_b1/console.log \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_linear_b1/perf_report.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_linear_b1/perf_summary.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_full_b1/console.log \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_full_b1/perf_report.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_full_b1/perf_summary.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_linear_b1/console.log \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_linear_b1/perf_report.csv \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_linear_b1/perf_summary.csv
```

After restoration, verify source identity and claims:

```bash
test "$(git rev-parse c55a8c067c8:models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py)" = \
     "$(git rev-parse 72e1a09218e:models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py)"
grep -R "FULL_TRACED_SYNTHETIC_LATENCY" \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates
grep -R "LoFi BF16 x BFP" \
  models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/*/perf_report.csv
```

## Minimal focused rerun if raw CSV is made a hard requirement

Candidate selection does **not** need rerunning: exact both-batch candidate logs
are recoverable. The smallest new profiler matrix is the final default path at
batch 1 for the two layer kinds and two phases. Run profiler and watcher
separately per `$tt-device-usage`; this report does not authorize combining
them.

```bash
python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_full_b1 \
  models/autoports/qwen_qwen3_6_27b/tests/optimized_traced_synthetic_pcc.py \
  --kind full --batch 1

python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_linear_b1 \
  models/autoports/qwen_qwen3_6_27b/tests/optimized_traced_synthetic_pcc.py \
  --kind linear --batch 1

python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_full_b1 \
  models/autoports/qwen_qwen3_6_27b/tests/optimized_full_attention_synthetic_pcc.py \
  --mode prefill --sequence 33 --batch 1

python -m tracy -r -p -v \
  -o models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/tracy/final_prefill_linear_b1 \
  models/autoports/qwen_qwen3_6_27b/tests/optimized_linear_attention_synthetic_pcc.py \
  --mode prefill --sequence 5 --batch 1
```

For each run, preserve the generated
`reports/*/ops_perf_results_*.csv` as the raw operation CSV, then derive
signpost-filtered detailed and summary CSVs using the installed CLI syntax
(`tt-perf-report --help` first because the historical version used
`--summary-file`):

```bash
tt-perf-report <raw-ops.csv> \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
  --csv <artifact-dir>/perf_report.csv \
  --summary-file <artifact-dir>/perf_summary.csv \
  > <artifact-dir>/perf_report.console.log
```

Use `PERF_PREFILL` / `PERF_PREFILL_END` for prefill. Record:

- current `git rev-parse HEAD`;
- optimized-decoder blob ID;
- exact profiler and report commands;
- raw CSV SHA-256;
- filtered CSV SHA-256;
- dominant matmul rows showing input/weight/output dtype, fidelity, DRAM
  sharding, memory config, block geometry and time.

This four-run matrix is sufficient to regenerate raw profiler provenance and
OPT-013 evidence. Batch 32 does not need a profiler rerun because the goal asks
for candidate performance at both batches, which the recovered exact candidate
logs already provide; runtime dtype materialization is phase/config based and
is directly verified at batch 1.

## Final status

**Diagnosis: evidence-copy omission, not missing experimental work.**

The smallest proven fix is to restore the exact authenticated evidence from
`72e1a09218e` and rerun independent stage review. Only if the reviewer rejects
the committed detailed filtered CSVs for lack of original raw operation CSVs
should the four serialized final-default profiler runs above be performed.
