# Gemma4 TSU optimization checkpoint

Closure requested on 2026-09-29. This is an optimization checkpoint, **not** a
full-matrix qualification or model release pass.

## Retained result

- Primary4096/128/C1 remote:23.23199->19.73241ms meanTPOT,
  43.04->50.67805TSU (**+17.735%**), TTFT2089.999ms. CI36564611976 passes
  all four requests and exact benchmark text comparisons.
- Additional shared batching, matched local two-repeat measurements:
  4KC8 **+3.035%TSU**,4KC16 **+4.441%**, shortC8 **+2.969%**,
  shortC16 **+4.580%**. C1 is unchanged within noise. All104 paired responses
  match text and token lengths; E2EL and steady ITL also improve.
- Eighteen C16 shared-suite qualitative responses match pinned controls.
  Inherited256-token caps and the controlled thermodynamics wording caveat
  are not presented as ideal-quality evidence.
- Context262144,32 cache slots and1GB trace budget are unchanged.

Retained production changes are in TT-Metal `eb1d2af61c7630c7424e9593b646093ef7be45af`
and `c9ec3469f1b875e7e5e505660c4421e5126e8dad`: decode-trace retention and
safe input reuse, asynchronous replay, the already-proven K1 terminal path,
and dense-batch shared MLP/paired collective/fused-tail work. Attention,
routing and indexed experts remain per-row. Rejected expert-union hooks are
removed; tiny fabric tuning is not selected.

## Provenance and qualification limits

Selected runtimec9ec3469 was built once by run36576438888/job109433282221.
Reuse the published image, without rebuilding:

`ghcr.io/tenstorrent/tt-agentic-bringup-qb2/vllm-tt-metal-src-dev-ubuntu-22.04-amd64@sha256:ad58effd178b9d8c7689a392159532d30aea0c1fe24bec82c06dbc0d3ebf4bbd`

`image_reuse.json` records the tag, digest, every CI reuse and local image-only
import hashes. The local exact-image focused benchmark completes104requests
with zero failures and all output/harness checks exact. It is reported
separately in `perf_summary.json`; earlier paired shared-batching
measurements used the development image with identical selected mounted source.
Those regimes must not be conflated.

Published-image pooled TPOT/TSU:4KC1 19.62790ms/50.94788;4KC8
184.29683ms/5.42603;4KC16 351.77281ms/2.84274;shortC8
175.43495ms/5.70012;shortC16 341.98513ms/2.92410. The first4KC8 cohort
has a10.677s single-request inter-token gap; the second returns to180.14918ms
TPOT and both steady medianITLs match177.55ms. Startup/capture is plausible,
but the cause is not proven. Both repeats are included: pooled4KC8 TSU is
2.042% below the mounted-source selected run, not a universal exact-image
performance pass. No matched exact-image shared-disabled run was performed.

- Full29-row C1/C8/C16 sweep: **NOT RUN**, stopped before dispatch by closure
  direction. The configured TTI revision is
  `6d88032ed5f8259333233c53db671cd29aad377d`.
- Selected remote five-row qualification36584709251: **BLOCKED before image
  execution**, not a model failure. All three attempts land on
  `120-qb2-p04t07` and fail checkout with EACCES `.git/FETCH_HEAD`.
  Jobs109461946997,109463261707,109464955726; all build jobs skipped.
- Remote control36581849489 attempt2 succeeds on `120-qb2-p03t02`:
  five rows,52 requests,zero failures. It reuses the earlier eb1d2af image.
- No further CI, runner repair, new optimization, or C32 qualification is
  authorized in this closure. SWE36530661132 remains untouched.

The first selected failure log records:

```text
2026-09-29T14:43:10.7047780Z Runner name: '120-qb2-p04t07'
2026-09-29T14:44:02.4708224Z ##[error]File was unable to be removed Error: EACCES: permission denied, unlink '/home/ubuntu/actions-runner/_work/tt-agentic-bringup-qb2/tt-agentic-bringup-qb2/.git/FETCH_HEAD'
```

## Evidence

- `work_log.md`: experiment chronology, commands, rejected candidates and
  user interventions.
- `perf_summary.json`: scoped metrics and limitations.
- `../../readiness_vllm/tsu_optimization/c8c16_default_comparison.json`:
  exact normalized harness/output checks and paired metrics.
- `../../readiness_vllm/tsu_optimization/ci_c1_comparison.json`:
  primary remote improvement.
- `../../readiness_vllm/tsu_optimization/exact_image_import.json`:
  image-only source/native import provenance.
- `review_checkpoint_closure.md`: fresh independent clean-pass restricted to
  the completed paired-source/runtime checkpoint; it does not pass the
  then-running exact-image benchmark or unrun full sweep.

Only reduced non-serving paths were device-profiled. There is no measured
full-model serving device-only duration; do not subtract reduced profile
durations from end-to-end serving TPOT or call aggregate output throughput TSU.
