# Testing

## Tests

Run from the repository root:

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

`tests/` holds the no-device tests (protocol, chat format, sampling, prefill driver, MTP session and MTP rows /
verify / draft, batched and MTP lanes, lane scheduler, vision inputs, vLLM / ASGI adapters, launcher), the
component tests against the `tt/` torch reference, and the device tests (`*_device.py`, `*_tp.py`). The
checkpoint-reading tests skip unless `QWEN38_CHECKPOINT` (or `MODEL_WEIGHTS_DIR`) names the checkpoint.

## The acceptance gate

Start the server with `--acceptance --require-json-96`. The CPU greedy records in
`tools/acceptance/greedy-prompts.jsonl` replay before the server listens (with `--prefill-slab` also the slab
record, `document`); the `json` record must match 96/96 or the server refuses to serve, and every record's first
divergence index is written to `acceptance.json` in the run directory. A replay that leaves the CPU stream earlier
than its pinned index in `tools/ci/baselines/` (for example `A3-chunked-32k-divergence_index.json` or
`A3-mtp4-32k-divergence_index.json`) is a regression. See [NUMERICS.md](NUMERICS.md).

## The reference corpus

`tools/reference/` freezes `Q38-REF-v1`: 36 teacher-forced items (the acceptance prompts as their records and as
the server render them, requests with tools, the opening of two books, evaluation items and two long prompts scored
in windows) with sha256s. `tools/qwen38_reference_corpus.py` has the modes:

- `hf`: the Transformers `qwen4_exp` model on the CPU (bf16 weights, fp32 LM head; needs a transformers checkout
  with the `qwen4_exp` class and several hundred GB of RAM);
- `oracle`: the same through `tt/` (bf16 or BF4-emulated experts);
- `device`: a served chain's agreement records, written by the server with
  `--agreement-reference FILE [--agreement-full-logits DIR]`;
- `score`: compares two columns (top-1 / top-5 agreement, clear-margin top-1, truncated KL over the shared top-32
  support, first divergence);
- `freeze` and `verify`: build and check the corpus files.

`hf` and `oracle` keep each scored position's full logits with `--full-logits DIR`, so two columns can be compared
over the whole vocabulary.

## The regression harness

`tools/ci/q38_ci.py` (standard library only) turns the tools' outputs into gated verdicts. `tools/ci/pins.json`
names every gated value as `<job>/<configuration>/<metric>` with a rule and a status:

- rules: `band` (two-sided relative band: a result better than the band is a stale target), `floor`, `ceiling`,
  `not_earlier` (divergence indices; null is the largest), `exact`, `at_most` and `flips` (per-item eval answers the
  baseline passes and the run fails, at most N per task);
- status `todo` warns, `active` gates; a pin with `baseline: true` compares a per-key map against
  `tools/ci/baselines/<pin id with - for />.json`;
- jobs: `A1` corpus agreement against the HF reference, `A2` the CPU oracle against HF, `A3` the start-up acceptance
  replay and HTTP probes, `D1` performance, `C1` / `C2` lm-eval answers and accuracies. Device-timed pins are
  `not_gated` above `idle_loadavg_1min` unless they set a `loaded_tolerance`.

A job list (`qwen38-ci-jobs/v1`) names the commands to run: a `command` job runs to completion; a `server` job
starts a launcher, waits for `READY`, probes the server and stops it. Each job's artifacts are validated into one
result file (`qwen38-ci-result/v1`) under `<out>/<date>/<stamp>-<lane>-<head12>/<job>/`, beside `summary.json`,
`summary.txt` and `progress.log`.

    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci run --jobs JOBS.json --out RESULTS
    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci report --run RESULTS/<date>/<run>
    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci seed --runs RESULTS/<date>/<run>... --write

`validate` re-checks a job result against the pins. `seed` promotes `todo` pins whose last three idle runs agree
within the pin's tolerance and writes the baseline files of baseline-backed pins.
