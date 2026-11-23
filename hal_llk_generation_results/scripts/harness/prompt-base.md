Write the LLK functions used by the tests listed below from scratch, {{API_TASK}}.
Target architecture: Blackhole.

Build ONE shared library: implement each LLK function once, in headers under
tt_metal/tt-llk/tests/hal_llk/, and use that same implementation from every test that needs
it. Do not write per-test variants of a function.

## Tests
{{TEST_TABLE}}

Scope: every variant the python test selects with the command shown in the table.

## What you can see
- The tests themselves: the original test sources in tt_metal/tt-llk/tests/sources/, the
  python tests in tests/python_tests/ and the helpers in tests/helpers/. These tell you what
  each LLK function must achieve and how it is called.
{{API_SEE}}
- A spec of the LLK library: tt_metal/tt-llk/llk_spec/tt_llk_blackhole/ mirrors
  llk_lib/ and common/inc/ with every function SIGNATURE and doc comment, but no
  function bodies and no raw instruction encodings. Use it to learn the exact names,
  template parameters, signatures and the types/enums you must use.

## What you cannot see
- The original LLK implementations and everything else under tt_llk_blackhole/{{API_CANNOT_SEE}}.
  Reading them is blocked. Include headers by name as the spec {{API_INCLUDE_GUIDE}} show,
  but do not try to read the originals in any other way.
- You therefore cannot compare against the original implementations. Correctness
  is verified only by the tests passing.

## What to produce
- {{API_PRODUCE}} implementations of every LLK function the tests use, in new headers under
  tt_metal/tt-llk/tests/hal_llk/, with exactly the names and signatures from the spec,
  so the test bodies stay the same.
- For every test in the table, the HAL test source named in the table: a copy of the
  original test source that is identical except that it includes your headers
  (e.g. #include "../hal_llk/...") instead of the original llk_*.h headers.
- The HAL python tests in the table already exist (the harness generated them); they run
  your test sources. Do not change them.

{{API_HOW}}

## Rules
- Do NOT modify anything outside tests/hal_llk/ and the HAL test sources named in the table.
- If the device hangs, reset it with `tt-smi -r` and continue.
- Do not run commands in the background or schedule wake-ups. Run the final
  verification in the foreground, wait for it, and end with the final report.

## Verification
- Run each HAL test from tt_metal/tt-llk/tests with the command in the table.
- The results must match the original: every test must pass with the same tolerance.
- Do not finish until every test passes.
- Before finishing, run every HAL test once more, each one from a clean device: reset the
  device with `tt-smi -r` right before each test and run nothing else in between. All of
  them must pass from that clean state.

## Sources for every finding
For every gap, contradiction or problem in your final report, name exactly where it comes from:
the document or header file (and section) that says what, {{API_COMPILER}}, and what you observed
in a test. Then classify it as one of:
documentation gap (the docs are missing or wrong), implementation gap ({{API_NAME}} lacks or
mishandles something), contradiction between sources (two documents, or a document and the code,
disagree), or other. Do not report a finding without its source.

## Final report
At the end, list:
1. every LLK function you implemented, which {{API_PARTS}} it uses and which tests use it,
2. {{API_REPORT_2}},
3. every place where you had to write individual configuration fields instead of a higher-level construct,
4. anything in the reference material or the spec that was unclear, missing or wrong,
5. how you worked out what each function must do without the original implementation,
6. the final result of every test in the table (passed / failed / skipped counts).
