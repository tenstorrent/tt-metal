# AutoFix Report: bounded profiler evidence

## Starting evidence

- Original command: `python -m tracy -r -p -v -m pytest models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k profile_warmed_prefill_and_traced_decode -s`
- The combined process captured all four windows, but later reported that the raw
  on-device event buffer filled and markers were dropped.

## Hypothesis experiment

- Hypothesis: collecting four layer/mode windows in one process exceeds the
  profiler event-buffer budget; one bounded window per fresh process will not.
- Prediction: four individually selected pytest parameter cases each emit their
  start/end signposts, finish without a dropped-event/buffer-full warning, and
  produce a complete filtered `tt-perf-report` table and CSV.
- Fix: parameterize only the profiler harness and return after the selected
  full-prefill, full-decode, DeltaNet-prefill, or DeltaNet-decode window.
- Experiment: run four serialized Tracy commands, each selecting one parameter
  case, retain its console and ops CSV, grep consoles for buffer-full/dropped
  event signatures, then filter by the matching signpost pair.
- Result: all four tests passed, emitted their requested signpost pair, closed
  the device, and had no buffer-full/dropped-event warning. Reports contain
  50/58/224/90 device ops and 2.584/1.528/5.086/1.563 ms respectively for
  full-prefill/full-decode/GDN-prefill/GDN-decode.
- Verdict: verified.
- Evidence: `../perf/captures/{full_prefill,full_decode,gdn_prefill,gdn_decode}/`.
- The superseded combined-capture source, filtered CSVs, and tables were removed
  after replacement so the stage has one unambiguous profiler evidence set.

## Final status

Fixed. The bounded capture harness preserves the same warmed invocation and
trace-replay boundaries while keeping every capture within the event budget.
