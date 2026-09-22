# Reproduce / re-measure

Written for the **post-rebase re-measure**. Asif's branch moves underneath this work, so a
carried number is not a measured one — re-run the baseline before comparing anything.

## Setup

```bash
source /data/kmabee/gemma4_runs/env.sh        # TT_METAL_HOME, HF_HOME, TT_CACHE_PATH
cd $TT_METAL_HOME
export PATH=$TT_METAL_HOME/python_env/bin:$PATH   # tt-perf-report lives here
S=models/demos/gemma4_d_p/docs/prefill_perf/scripts
```

Preflight — all four should pass before any measurement:

```bash
hostname; git rev-parse --short HEAD
find ttnn tt_metal -newer ttnn/ttnn/_ttnn.so \( -name '*.cpp' -o -name '*.hpp' \) | wc -l  # expect 0
for d in /dev/tenstorrent/*; do fuser $d 2>/dev/null && echo "$d HELD"; done               # expect silent
./python_env/bin/python3 -c "import ttnn; print('ttnn ok')"
```

**On a box you have not used before**, `python_env/bin/python` may dangle — uv writes the
interpreter as an absolute `$HOME` path and `$HOME` is local disk. Symptom is
`python_env/bin/python: No such file or directory`. Fix without rebuilding:

```bash
cd python_env && ln -sfn /usr/bin/python3.10 bin/python
sed -i 's|^home = .*|home = /usr/bin|' pyvenv.cfg && cd ..
```

## 1. The whole scoreboard — 9 runs, ~37 min

> ⚠️ **`GEMMA4_PREFILL_L1_ACT` is dead.** The rebase onto mmanzoor/svuckovic inverted
> this knob: `gemma4_d_p` now puts short-lived prefill activations in L1 **by default**
> and the opt-out is `GEMMA4_ACTIVATIONS_DRAM_ONLY=1`. The old name survives only in
> `models/demos/gemma4/` (a different model) and as a stale comment in
> `gemma4_d_p/tt/attention/__init__.py`. Passing it was a silent no-op that produced two
> "different" configs with bit-identical numbers, so **`run_e2e.sh` now exits 2** if you
> do. Old `L1 off` → `dram_only`; old `mmanzoor_only` → `base`.

```bash
OUT=/data/kmabee/remeasure_$(date +%m%d)
DRAM="GEMMA4_ACTIVATIONS_DRAM_ONLY=1"
FIX="GEMMA4_NORM_SHARD=1 GEMMA4_MLP_MM_CFG=1 GEMMA4_MLP_MM_GRID=12x10 GEMMA4_ATTN_MM_PC=1"

for C in 2048 4096 8192; do                      # grouped by width: each trio adjacent
  $S/run_e2e.sh --chunk $C --out $OUT --label dram_only_c$C $DRAM
  $S/run_e2e.sh --chunk $C --out $OUT --label base_c$C
  $S/run_e2e.sh --chunk $C --out $OUT --label all_fixes_c$C  $FIX
done
```

Then one command per config for the verdict:

```bash
for cfg in dram_only base all_fixes; do
  $S/check_baseline.py --config $cfg $OUT/${cfg}_c*.log
done
```

It prints expected vs measured vs drift for `a`, `slope` and the 256k total at each width,
and exits non-zero if anything moved more than 1%.

**Interpreting drift.** Within-session σ is **0.03%** on `a` (3 repeats, measured);
cross-session and cross-box drift is **0.2–0.4%**. 1% is comfortably outside both, so a
failure is a real change: a rebase that moved the base, a different board, or a fix that
stopped engaging. **Check the witness lines before blaming the model** — `run_e2e.sh`
enforces them and writes `witness_fail=` into the log footer.

## 2. Per-op attribution — ~20 min per A/B pair

```bash
$S/capture.sh --chunk 2048 --idx 0 --out $OUT --label off  $BASE
$S/capture.sh --chunk 2048 --idx 0 --out $OUT --label on   $FIX
$S/diff_ops.py $OUT/off_local.txt  $OUT/on_local.txt  --layers 50 --iters 2
$S/diff_ops.py $OUT/off_global.txt $OUT/on_global.txt --layers 10 --iters 2
```

Add the two whole-model numbers and **reconcile against the e2e delta in `a`**. Landed fixes
here close at 100% (`MLP_MM_CFG`) and 95% (`ATTN_MM_PC`). An attribution that does not close
is a finding, not a rounding error.

## Rules that cost real time here

- **A capture is not done when pytest prints `passed`** — tracy post-processes for minutes.
  `capture.sh` polls for the ops CSV at `-size +1M`.
- **Never capture the full traced run.** 1.19e9 zones, OOM-killed while saving, and it took
  the box down with it. Isolated-layer captures only.
- **Render both sides of a comparison with the same tool in the same session.** Device
  spread on a single op is ~17%, and a number lifted from an older render already produced
  one false regression in this work.
- **A knob inert in BOTH directions is probably not wired.** A window sweep once returned a
  clean null across a 4× range because the "knob" was a dataclass default that
  `from_hf_config` overrides. Every flag now logs a witness; trust the witness, not the edit.
- **Anchor a witness to program output**, never to something the runner echoed. A pattern
  that matched the driver's own header once certified a run that never started.
- **`slope` is the control for any floor fix.** >2% movement means the patch did more than
  one thing.
- **Compound cumulatively, never multiply.**

## Not covered: accuracy

Everything here is `traced_perf` device timing. **No PCC or accuracy check has been run** —
not on Asif's changes and not on ours. The three fixes should be numerics-neutral (same
dtypes, same math fidelity, and the fused GELU is deliberately carried inside the new matmul
program config rather than dropped), but that is reasoning, not measurement. If the rebase
is motivated by suspected PCC regressions, that needs its own harness.
