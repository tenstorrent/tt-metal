# B1 new-source benchmark confirmation

Candidate performance remains unselected pending C8 scored acceptance. Exact4096input/128output, B1/C1, eight requests all completed: mean TTFT 753.407429ms, TPOT 28.317514ms, decode 35.313834tokens/s/user (1000/meanTPOT). Client workload 34.800406s; wrapper 42.356729s. Separate warmup excluded. All eight generated texts match the prior selected phase-archive benchmark; compact digests are in JSON, raw text remains private.

Unchanged stage11 matrix128/128 n8 C1 completed8/8. Mean TTFT 111.194925ms and TPOT 25.775680ms. Actual benchmark duration 27.080678s versus TTI wrapper 34.508291s. Reported128input is nominal: adapter observes seven141-token and one142-token chat-templated prefills. Output128 each. Report has no performance targets; no eval/spec-test scores produced.

Execution3a7704d6, publication4b7d9a3b, actual TTI client af1ef17. All14 runtime Python hashes plus selected precision JSON match publication; all frozen launch files unchanged after run. Full60/context262144, B1, optionalpool unset, response debugfalse and request logging disabled; unbuffered clients.73 valid phase frames contain2185 matched unique submit/completion pairs, final pending/buffer0, errors empty and no loss. No profiler used; host-wall phases are not device timings.

Device occupancy 0.088526220556h from queue start to clean close20:44:51.568246UTC with owners empty, versus forecast 0.076054659132h and box0.15h. Wrapper duration 0.089458747429h includes later exit and is not device cost.

Relative to prior headline: TTFT delta -0.353448ms, TPOT delta +0.005749ms. These sequential observations and exact text equality support regression confirmation; they do not establish causal speedup, quality or scored C8 acceptance. Benchmark order changed; prior matrix preceded headline whereas this run followed it. No central results or telemetry packet updated.
