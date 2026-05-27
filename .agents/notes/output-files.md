# Output Files

Unless your current bringup skill specifies otherwise, the final result of your work should be organized like this.

## Files to Commit

```text
models/autoports/<model>/tt/<skill_name>.py
models/autoports/<model>/doc/<skill_name>/
  work_log.md
  README.md
  tracy/<layer_kind>/prefill_perf_report.csv
  tracy/<layer_kind>/prefill_perf_report.txt
  tracy/<layer_kind>/decode_perf_report.csv
  tracy/<layer_kind>/decode_perf_report.txt
```

Raw pytest logs, watcher logs, Tracy captures, and failed exploratory attempts may be linked from the work log but generally we don't need to keep a lot of these in git.

## Work Log

`doc/<skill_name>/work_log.md` is for the trail and for learning from experience. Keep updating is as you work, it will be a useful reference for you and for others. Consider it a scratchpad or logbook.

- what you did and in which order you did it
- problems you encountered and how you overcame them
- important judgement calls you had to make and how you made them
- things you wish you had known when you started
- surprises good and bad
- insights

This is your working log and the place we will look to learn how to improve next time. Use it with this in mind.

## Final Report

`doc/<skill_name>/README.md` is your report on the final status achieved.

- model id, HF revision/path, hardware, branch/commit, and environment;
- decoder layer kinds and representative layer indices;
- Decoder-class constructor/forward contract;
- state-dict key mapping and real/synthetic weight behavior;
- prefill/decode PCC against HF, including thresholds and tested shapes;
- paged KV-cache, page-table, and current-position evidence;
- trace replay evidence for decode;
- sequence lengths tested and any measured capacity limits;
- determinism, watcher, and runtime fallback status;
- warmed prefill/decode latency and `tt-perf-report` conclusions;
- other criteria as determined by your skill

In some ways this is also a forcing function; if when writing it you realize there are conditions unfulfilled or simplifications/short-cuts that undermine the mission of full, real model bring-up take that as an impulse to revisit those areas and try to resolve them so the report is one you can be proud of.

## Keep Out

Do not store full model weights, binary tensors, program-cache directories, giant profiler captures, or every failed run under `doc/<skill_name>/`.
