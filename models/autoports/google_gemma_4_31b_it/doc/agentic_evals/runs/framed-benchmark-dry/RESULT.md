# Frozen local benchmark arm dry passed

Required headline workload completed8/8 sequential requests at exactly4096 input/128 output tokens, B1/C1: mean TTFT755.099ms, mean TPOT28.3253ms,35.3041 decode tokens/s/user through vLLM. Request workload duration34.8219s excludes CLI startup. All eight generated texts exactly match the prior eef534 benchmark. TTFT differs by+0.9853ms and decode rate by−0.0110 tokens/s/user; these are descriptive differences, not causal speed claims.

The unchanged stock TTI stage11 matrix completed8/8 at nominal128/128, concurrency1. Actual adapter prompts are142,143,141,141,141,141,142,141 tokens after chat formatting. Mean TTFT112.137ms, TPOT25.7978ms,38.7630 decode tokens/s/user; request duration27.1107s. Random native matrix prompts are not proven identical to the earlier run, so its differing generated texts do not establish an inference regression. No spec tests or accuracy scores were run.

Frozen TTf57978407eaacf09587f5793b8ae46ca7af839ea, detached TTIaf1ef17abfecb059da13d0a1287c00eace0f210e, stock pluginca7a18246e4111629ed6cb4a7d1192002749fccf and HF842da identity are recorded in freeze.json/source-launch.json. All26 frozen file hashes and resolved executed client commands verify. Runtime is full60/B1/maxseq1/context262144, response-debugfalse, request loggingoff and staticloggingnull. All clients exited0.

Phase export has72 frames,2185 contiguous matched steps,26 prefill and2159 decode completions, no duplicates/gaps/loss/errors. Sampling modes are2176 device steps and9 accepted host-compatibility mixed-prefill steps. Supported final drain is the actual final export with pending0, followed by fresh running/waiting0 and graceful owned shutdown. Device owners are empty.

Device hold from02:00:55.622968 to02:05:26.668952UTC is0.0752905511h, below forecast0.07722610789h and box0.15h. Wrapper ended02:05:29.645226 and its0.0761173h is not substituted for device hold. No profiler was used; device execution/roofline fields remain unknown rather than receiving host-wall values.

This passes only the frozen LOCAL benchmark arm. It is not combined stage dry or release. GPQA finalcohort still requires its separate frozen dry. Only the explicit fresh ARTIFACT_ROOT parameter may change for release; source/config/semantic-command changes invalidate this arm dry. Dry artifacts remain preserved at bringup/artifacts/framed-benchmark-dry-run. Compact measurements and evidence are in RESULT.json.
