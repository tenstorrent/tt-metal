# Reproduction commands

Run from the repository root in the source-built environment. Execute hardware
commands serially and wait for device close between commands. The following
shell variables only abbreviate paths:

```bash
MODEL_DIR=models/autoports/google_gemma_4_26b_a4b_it
EVIDENCE_DIR=$MODEL_DIR/doc/functional_decoder
MODULE=models.autoports.google_gemma_4_26b_a4b_it.tests
```

The layer mapping is `0 -> sliding`, `5 -> full`. Substitute both pairs below.
For example, set `LAYER=0`, `KIND=sliding` and
`LAYER_TYPE=sliding_attention`; for layer5 use `KIND=full` and
`LAYER_TYPE=full_attention`. Real weights are loaded at the pinned HF
revision in `tests/run_decoder.py`; host references and setup are outside passes.

```bash
python_env/bin/python -m "$MODULE.run_decoder" --layer "$LAYER" \
  --length 4096 --real --decode --steps 128 --verify-program-cache \
  --output "$EVIDENCE_DIR/headline_${KIND}.json"

python_env/bin/python -m "$MODULE.request_reuse" --layer "$LAYER" \
  --output "$EVIDENCE_DIR/reuse_${KIND}.json"
python_env/bin/python -m "$MODULE.batched" --batch 32 --layer "$LAYER" \
  --output "$EVIDENCE_DIR/batch32_${KIND}.json"
python_env/bin/python -m "$MODULE.prefix_continuation" --layer "$LAYER" \
  --output "$EVIDENCE_DIR/continuation_${KIND}.json"
```

Maximum and near-maximum tests use `LENGTH=262144` and `LENGTH=262143`.
A cached reference file is optional; omit `--reference-file` to compute it.

```bash
python_env/bin/python -m "$MODULE.long_context" --layer "$LAYER" \
  --length "$LENGTH" \
  --reference-file "$EVIDENCE_DIR/reference_${KIND}_${LENGTH}.pt" \
  --output "$EVIDENCE_DIR/long_${KIND}_${LENGTH}_fixed.json"
```

Final profiler commands use the exact headline and reject program-cache misses
in warmed prefill/capture. Watcher variables must be unset. The optional bulk
zone dumps are disabled, but complete C++ device analysis is retained.

```bash
python_env/bin/python -m tracy -r -p -v --op-support-count 100000 \
  --no-op-info-cache --disable-device-data-dump-to-files \
  --disable-device-data-push-to-tracy \
  -o "$EVIDENCE_DIR/tracy/$KIND/raw_final" -n headline \
  -m "$MODULE.run_decoder" --layer "$LAYER" --length 4096 \
  --real --decode --steps 128 --profile --verify-program-cache \
  --output "$EVIDENCE_DIR/profile_${KIND}_final.json"
```

`tracy/<kind>/provenance.json` identifies the generated raw CSV copied to
`tracy/<kind>/ops.csv`. Calculate the complete-layer summary with:

```bash
python_env/bin/python -m "$MODULE.summarize_perf" \
  "$EVIDENCE_DIR/tracy/$KIND/ops.csv" \
  --layer-type "$LAYER_TYPE" \
  --output "$EVIDENCE_DIR/tracy/$KIND/whole_layer.json"
```

The exact
five tt-perf-report commands per kind, including full128-replay CSV/text and
representative-replay text, are recorded in `tracy/<kind>/report_commands.json`.
They are CPU-only analysis of existing captures.

Final watcher commands are separate from profiling:

```bash
TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH="$EVIDENCE_DIR/watcher/$KIND" \
  python_env/bin/python -m "$MODULE.run_decoder" --layer "$LAYER" \
  --length 4096 --real --decode --steps 128 --verify-program-cache \
  --output "$EVIDENCE_DIR/watcher_${KIND}_final.json"

python_env/bin/python -m pytest -q "$MODEL_DIR/tests/test_functional_decoder.py" \
  --basetemp "$EVIDENCE_DIR/pytest_final_tmp"
```

Console logs use the output JSON stem with `.log`. Watcher-generated logs are
under `watcher/<kind>/generated/watcher/`; the compact summary records their
inspection. Repository pre-commit hooks run on all stage-owned files. This is
Python/docs-only work and requires no C++ build.
