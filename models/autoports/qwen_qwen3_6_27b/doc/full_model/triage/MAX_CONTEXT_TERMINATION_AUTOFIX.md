# Maximum-context termination AutoFix

## Verdict

The S262,144 run was terminated by its external execution/session boundary,
not by demonstrated host memory, TT DRAM, device-health, or model failure.
There is no basis for reducing the supported context from this run.

The successful all-layer S192,511 log spans 00:12:35 to 02:00:10, or about
107 minutes 34 seconds.  Prefill work is linear in the number of 32,768-token
full-stack chunks; scaling that observation predicts roughly 146 minutes 29
seconds at S262,144.  The failed process was externally terminated after about
56 minutes, only 38% of that prediction, and emitted no Python or TT exception.

Independent system evidence at termination records about 4.19 GiB process RSS,
109 GiB system memory available, zero `oom`, `oom_kill`, and `max` events in the
process cgroup, no kernel kill record, and healthy devices after termination.
The existing B1 C262,144 capacity artifact separately proves weight, KV-cache,
state, and RoPE residency, but does not replace a completed public-wrapper run.

## Diagnostic change

`tests/full_model_long_prefill.py` now prints a flushed start marker and a
host-only heartbeat every five minutes by default, including elapsed seconds
and process peak RSS.  It prints the same values on normal return or exception.
The heartbeat neither calls TTNN nor changes model execution.

## Required rerun

Run detached from the bounded unified execution session and allow at least
three hours:

```bash
nohup env PYTHONUNBUFFERED=1 python \
  models/autoports/qwen_qwen3_6_27b/tests/full_model_long_prefill.py \
  --sequence 262144 --max-context 262144 --skip-decode \
  --heartbeat-seconds 300 \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/full_model_long_prefill_s262144_final.json \
  > models/autoports/qwen_qwen3_6_27b/doc/full_model/logs/full_model_long_prefill_s262144_final.log \
  2>&1 < /dev/null &
```

Until that run completes, S192,511 is the largest proven public full-wrapper
prompt.  It is not proven to be the largest physically feasible prompt, so it
must not be presented as a hard maximum.
