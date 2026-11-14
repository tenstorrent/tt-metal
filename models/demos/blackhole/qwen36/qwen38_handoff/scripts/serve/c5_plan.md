# C5 plan: serve Qwen3.6-27B with vLLM async scheduling + device-resident decode

## Key finding
vLLM 0.26 already ENABLES async scheduling by default (async_scheduling=None -> True when executor supports it).
C4 already requested it: server.log:72 "Asynchronous scheduling is enabled." and then the TT plugin turned it off:
server.log:82 WARNING ... tt/platform.py:2169] Async scheduling was requested, but TT model Qwen36ForCausalLM (...qwen36_vllm) does not declare support (`model_capabilities['supports_async_decode']`). Disabling async scheduling.
server.log:1928 (EngineCore) "Asynchronous scheduling is disabled."   => C4 ran SYNC.
Cause (INFERRED): qwen36_vllm.py was edited at 20:42:33, after C4 (20:30) and C4off (20:37) launched; the
capability dict is built at class-definition time by _build_model_capabilities() (qwen36_vllm.py:58-77), so C4 saw the pre-change dict.
With the current worktree file and QWEN36_SERVE_DEVICE_DECODE unset (default "1", model.py:25-28) caps = supports_async_decode True,
max_device_top_k, supports_device_penalties False. So NO extra flag is needed; just relaunch. Do NOT set QWEN36_SERVE_DEVICE_DECODE=0.

## (a) Launch command (C5)
cd /home/ttuser/atupe/tt-inference-server
WT=/home/ttuser/atupe/tt-metal/.claude/worktrees/qwen38-optimizations
HF_TOKEN=dummy TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 \
python3 run.py --model Qwen/Qwen3.6-27B --tt-device p300x2 --workflow server --local-server --dev-mode \
  --tt-metal-home $WT --tt-metal-python-venv-dir /home/ttuser/atupe/python_env_vllm \
  --host-weights-dir /home/runara/models/Qwen3.6-27B --skip-system-sw-validation --no-auth
(Env passthrough: build_local_server_env does env = os.environ.copy(), run_local_server.py:150, so inline env vars reach the server.
Optional explicit belt-and-braces: add --vllm-override-args '{"async-scheduling": true}'; _append_vllm_arg emits bare `--async-scheduling` for True,
run_vllm_api_server.py:1074-1084; run_vllm_api_server.py:1128-1180 merges. Only needed if you want the flag visible in "non-default args". To force off: {"async-scheduling": null} does NOT work (None drops the key = default on); use "--no-async-scheduling" which is a bare CLI token, not expressible via bool False (False emits nothing). INFERRED.)
Mandatory keep: spec's additional_config sample_on_device_mode=decode_only (release_model_spec.json ~13860), trace_mode all (model_runner log "trace_mode=all").

## (b) Log greps (server log, e.g. $S/C5/server.log)
PROOF of async + device-resident:
 1. grep -n "Asynchronous scheduling is" log     -> APIServer line "enabled" AND EngineCore line "enabled" (C4 had enabled then disabled).
 2. grep -c "Async scheduling was requested, but TT model" log  -> must be 0.
 3. grep -n "does not advertise decode_input_update_contract\|uses the legacy decode input reload contract" log -> must be empty (model has decode_input_update_contract = 1, qwen36_vllm.py:101).
 4. grep -n "TTModelRunner: trace_mode=all, sample_on_device_mode=decode_only" log  (model_runner.py:298 line) -> need trace on + device sampling.
 5. grep -n "TT submissions:" log -> "N ordinary decode, 0 verify, M overlapped an outstanding step" every 128 submissions (async_decode.py:76,689-696). M > 0 and growing = overlap really happening (split submit/readback). C4 had M == 0 all the way (sync).
 6. grep -n "TT async decode: .* overlapped an outstanding step, .* not overlap-safe" log (async_decode.py:710) -> must end with "0 of them a step that was not overlap-safe". Printed at powers of two of overlapped submissions.
 7. Using custom scheduler class ...TTScheduler warning (scheduler.py:192) is printed in sync too; not proof.
 Device-resident path itself has no dedicated log line (INFERRED); evidence = items 1-6 plus tput/ITL vs C4 (TPOT ~34.6 ms at conc 1) and correct.sh equal to C4 outputs ($S/C4/correct.txt).
REJECTED / FALLBACK:
 - "Async scheduling was requested, but TT model ... does not declare support" (platform.py:2168-2176) => sync fallback.
 - "Asynchronous scheduling is disabled" in EngineCore after "enabled" in APIServer.
 - "TT cannot verify vLLM's asynchronous-scheduling gate" / "serves without asynchronous scheduling" (platform.py:809-812) (spec-decode patch; not expected).
 - ValueError "sample_on_device_mode=... does not support on-device sampling" (platform.py:2135).
 - "does not advertise decode_input_update_contract >= 1" / "legacy decode input reload contract ... async overlap is disabled" (async_decode.py:1252,1259).
 - Per-request host fallback (silent, INFERRED): model_runner.py:3009-3040 returns False (host sampling, no resident chain) for temperature!=0 with top_k<1 or >32 (max_device_top_k), any penalties (supports_device_penalties False), logit_bias/min_p/bad_words/allowed_token_ids, structured output, logprobs. Bench uses temperature 0 so device sampling holds.
 - "TT submissions: ... 0 overlapped" for the whole run => async on but nothing overlapped (steady fast path rejected: async_decode.py:598-623).

## (c) Bench / correctness invocation
bench5.sh: runs ONE case, R5 = random ISL 300, OSL 128, concurrency 1, 8 prompts, 2 warmups, temp 0, seed 0, ignore-eos, port 8000;
 uses D=$(pwd)/$1, so run it FROM $S (the serve dir), arg = output dir name; the dir must exist (mkdir -p).
 cd $S && mkdir -p C5 && ./bench5.sh C5      # writes C5/R5.json, C5/R5.out; prints "R5 rc=0"
 python3 sum5.py C5 R5 > C5/sum.txt          # sum5.py <dir> <comma list>; (C4 used ./bench.sh C4 for R1-R5, sum.py)
 (bench.sh R1..R5 = same wrapper with more cases; C4 sum.txt shows R5: TPOT 35.85 ms, 27.28 out tok/s; C4off R5: TPOT 34.64.)
correct.sh: takes ABSOLUTE dir (uses $D/a_req.json, ...); 1 sequential + 4 concurrent chat requests, temp 0, max_tokens 96, thinking off.
 cd $S && ./correct.sh $S/C5 > $S/C5/correct.txt    # compare: diff $S/C4/correct.txt $S/C5/correct.txt
The exact C4 command lines were NOT recorded (only launch_ts, bench_rc.txt: "R1..R5 rc=0", run.out, correct.txt); the above is reconstructed from the scripts. (INFERRED)
Also save: cp the server log (path printed in run.out: workflow_logs/local_server/vllm_local_*.log) to $S/C5/server.log; C4 server.log is that file.
