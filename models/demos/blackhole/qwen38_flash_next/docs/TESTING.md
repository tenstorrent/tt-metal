# Testing

> Historical source notes from Samuel Jett (sjettTT, sjett@tenstorrent.com), frozen at
> `cd9a11771107ea2c27da3303a0556ff7343e4af5`. Measurements and implementation details below
> describe that source snapshot. Use the [port README](../README.md) for current commands and
> [PROVENANCE.md](../PROVENANCE.md) for dependencies and the separation of source and port evidence.

The no-device tests, the acceptance gate, the reference corpus and the regression harness.

## The no-device tests

Run from the repository root:

    QWEN38_CHECKPOINT=/data/Qwen3.8-Flash-Next python_env/bin/python -m pytest models/demos/blackhole/qwen38_flash_next/tests

`tests/` holds the no-device tests: component tests against the `tt/` torch reference (the CPU oracle every test
compares against), static contract tests over the `ttnn/` device model, the protocol, sampling, prefill-driver,
reference-corpus and harness tests.  Set `QWEN38_CHECKPOINT` for the checkpoint-reading ones (the config, chat
format and component tests).

## The acceptance gate

Start the server with `--acceptance --require-json-96` (the README's section 4).  The shipped CPU greedy records
under `tools/acceptance/greedy-prompts/` replay against the CPU before the server listens (the twelve study records;
with `--prefill-slab` also the slab record, `prompt-document-greedy.json`, 2,228 prompt tokens); the `json` record
must match 96/96 or the server refuses to serve, and every record's first divergence index lands in `acceptance.json`
in the run directory.  The pinned indices a start is compared against are `tools/ci/baselines/A3-chunked-32k-divergence_index.json`
(plain decode with the chunked prefill) and `tools/ci/baselines/A3-mtp4-32k-divergence_index.json` (`--mtp 4`); a
replay that leaves the CPU stream earlier than its pin is a regression.  `NUMERICS.md` has the tables and what they
measure.

## The reference corpus and the agreement tool

`tools/reference/` freezes `Q38-REF-v1`: 36 teacher-forced items (the twelve acceptance
prompts as their records render them and as the server renders a request today, four requests with tools, the first
1024 tokens of two books, four evaluation items, two long prompts scored in 32-position windows), about 13k positions,
with sha256s.  `tools/qwen38_reference_corpus.py hf` runs the Transformers `qwen4_exp` model on the CPU over it (bf16
weights, fp32 LM head; a transformers checkout with the `qwen4_exp` model class, several hundred GB of RAM, hours) and
keeps per position the top-32 ids and log-probs and the teacher's log-prob; `oracle` does the same through `tt/` (bf16
or BF4-emulated experts); `device` converts a served chain's agreement records; `score` compares two columns (top-1 /
top-5 agreement, clear-margin top-1, truncated KL over the shared top-32 support, first divergence).  The HF column is
the acceptance reference the other columns are read against.  `hf` and `oracle` keep every scored position's full
logits with `--full-logits DIR` (fp16, one `.pt` per item); the server keeps the device's full-vocabulary logits at
the same positions with `--agreement-reference FILE --agreement-full-logits DIR` (one eager gather of the LM head's
bf16 row per recorded position, beside the agreement records), so two columns can be compared over the whole
vocabulary and not only over the candidate row.

## The regression harness (development)

`tools/ci/q38_ci.py` (standard library only) turns the numbers the tools already produce into gated verdicts.
`tools/ci/pins.json` names every gated value as `<job>/<configuration>/<metric>` with a rule and a status:

- rules: `band` (a symmetric relative band, two-sided as in tt-metal's model targets: a result better than the band
  is a stale target), `floor` (target minus slack), `ceiling` (target times 1 + tolerance, with a warning level),
  `not_earlier` (divergence indices; null is the largest), `exact`, `at_most`, `flips` (per-item eval answers: items
  the baseline passes and the run fails, at most N per task);
- status `todo` warns only, `active` gates; a pin with `baseline: true` compares a per-key map (per prompt, item or
  task) against `tools/ci/baselines/<pin id with - for />.json`;
- jobs: `A1` corpus agreement against the HF reference (the scorer's `score.json`), `A2` the CPU oracle against HF,
  `A3` the startup acceptance replay plus the runner's probes (the verbatim echo of a sentence at temperature 0,
  N short completions that must all finish, one-token replies at several prompt lengths for TTFT), `D1` perf (the
  timing runner's period, the ledger's prefill and decode rates, TTFT, startup, captures, program cache, DRAM
  headroom), `C1`/`C2` lm-eval per-item answers and accuracies. Device-timed pins are `not_gated` when the 1-minute
  load average is above `idle_loadavg_1min`, or use their `loaded_tolerance`.

A job list (`qwen38-ci-jobs/v1`) names the commands to run on one lane: a `command` job runs to completion, a
`server` job starts a launcher, waits for its `READY` marker, probes the server over HTTP and sends the server pid
SIGTERM; each job then collects its artifacts (globs over the launchers' evidence directories, `$Q38_CI_RUN_DIR`
for an earlier job's files) and validates them. One result file per job (`qwen38-ci-result/v1`: host, lane, head,
runtime identity, load average, item ids, every observed value, one verdict per pin with observed and expected
side by side) lands under `<out>/<date>/<stamp>-<lane>-<head12>/<job>/`, with `summary.json`, `summary.txt` and
`progress.log` beside them.

    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci run --jobs JOBS.json --out RESULTS
    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci report --run RESULTS/<date>/<run>
    python -m models.demos.blackhole.qwen38_flash_next.tools.ci.q38_ci seed --runs RESULTS/<date>/<run>... --write

`seed` promotes `todo` pins whose last three idle runs agree within the pin's tolerance: the target becomes the
median (`band`, `ceiling`), the minimum or the preset proposal (`floor`) or the common value, the run ids are
recorded on the pin, and baseline-backed pins get their baseline file written. Until then the committed targets
are proposals from the runs named in each pin's `note` and the baselines' `source`; only the rules that already
gate today are `active` (the json record's 96/96 replay, the echo, request completion, a program-cache delta of 0).
