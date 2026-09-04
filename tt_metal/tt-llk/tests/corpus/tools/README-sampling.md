# laneMO — stratified silicon differential SAMPLING for the cross-lane ops

The 31 cross-lane / multi-input ops (coverage class NOT-EXHAUSTIBLE) cannot be
exhausted (input space 2^(32*lanes)) and single-lane SMT cannot model them.
This is the SAMPLING analog of the laneMK 2^32 streamer: same object-identity-
preserving persistent-session pattern, but each dispatch's operand-A input is a
**stratified sample tile** instead of an ascending counter, so a defensible
high-coverage sample of the input space is fed to the certified kernel and the
sem/hand legs' output bytes are compared.

**Honesty (absolute).** A verdict from this tool is `SAMPLED-CONSISTENT`
(N samples, 0 diffs) or `SAMPLED-DIVERGENT(+witness)` — a DISTINCT, **WEAKER**
class than the proven-equal ops. Sampling is NOT a proof. Never call a sampled
op "verified" or "proven", and never conflate `SAMPLED-CONSISTENT` with
machine-certified-equal.

## Pieces (all additive; retire nothing)

| file | role |
|---|---|
| `lanemo_sample_gen.py` | device-independent stratified operand-A sampler: specials + biased-exponent grid (0..255) + cross-lane structure shapes + seeded uniform random fill. Pure function of `(op, seed)` so sem and hand replay the identical stream. |
| `selftest_lanemo_sample.py` | determinism / strata / known-equal / seeded-divergent (+ checkpoint localization). No device. |
| `lanemo_sample_sweep.py` | orchestrator: object-identity gate → run sem + hand legs over the same seeded sample stream → compare per-leg output SHA → verdict. |
| `lanemo_run_op.sh` | per-op runner (one op, run-to-completion, exit; `--requeue`-safe). |
| `lanemo_array.sh` | Slurm array task: `$SLURM_ARRAY_TASK_ID` → op; submission is ONE `sbatch --array` line. |
| `LANEMO_SAMPLE` hook in `python_tests/test_sfpu_unary.py` | the persistent-session device leg (mirrors the laneMK `LANEMK_STREAM` hook). |

## Device hook contract

`LANEMO_SAMPLE="n_tiles,seed,ckpt,outfile"` + `LANEMO_OP=<op>` streams `n_tiles`
stratified operand-A sample tiles through the certified kernel in one open
session and emits the per-leg `output_sha256` + checkpoint SHAs (every `ckpt`
samples, for divergence localization). Only operand A is sampled — the tt-llk
stimuli harness exposes raw injection for operand A only (`lanejn_raw_a`); an
op's operand B keeps its fixed generated stimulus, so for a genuinely binary op
this is PARTIAL sampling (A varies, B fixed), recorded as such.

## Submit (one line)

```
sbatch --array=1-<N> --requeue --export=ALL -J lanemo_op \
       -p <glx-partitions> --exclude=<poisoned> --time=720 lanemo_array.sh
# env: LANEMO_OPS_LIST LANEMO_RUN_OP OPS_TSV BUILD VENV LLK_HOME PYDIR OUT
#      [N_SAMPLES] [SEED] [CKPT]
```

## Integration status (honest)

The sampler core + orchestrator + array scaffolding are complete and selftested,
and the `LANEMO_SAMPLE` device hook is wired into the unary harness
(`eltwise_unary_sfpu`, where the laneMK single-input ops live). The 31 target
ops are NOT routed through that harness — they live in `test_sfpu_blaze.py` /
`test_sfpu_coverage.py`. Producing live silicon verdicts requires, per op:
(1) a genuine second device implementation (a sem/hand pair) — most of the 31
have only ONE device implementation; and (2) wiring the `LANEMO_SAMPLE` hook
into that op's harness node + building the pin-59 dual-leg ELFs. See
`laneMO-evidence-20260904/SAMPLE-LEDGER.tsv` for the per-op feasibility.
