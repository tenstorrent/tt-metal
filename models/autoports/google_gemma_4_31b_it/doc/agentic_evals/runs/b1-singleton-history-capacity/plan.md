# B1 singleton real-history capacity preparation

Goal: capacity-only validation of smallest supported B1 on one saved long native SWE conversation, approximately107585 observed-token stress target, followed by two short appended return turns. Preserve original content/template, advertised262144context, APC and chunked prefill. C1/B1. This cannot rehabilitate quarantined scores or accept a scored singleton route.

Forecast0.10deviceh (B1 startup about3min plus long prefill about48s and short returns); box0.18h. Coordinator owns queue; no hardware launch or runtime/publication edit by preparer. Locate existing real-history replay helpers and authoritative native conversation, retain raw content privately, prepare exact bounded commands/client. Confirmation requires actual native prompt counts, zero preemptions/diagnostic loss, returning prefix reuse>=0.9, and source identity. Final evidence only after completed data.

## Prepared commands and clarified scope

Coordinator approved three unchanged real Django prefixes with original native prompt counts 103509, 105535, 107474, source `bringup/artifacts/stage12-real-histories/swe-peak3-http/django__django-11265__ohWF75G.jsonl`. Original tools, messages and template kwargs remain exact. The two subsequent prefixes include actual recorded returns, not the newly generated diagnostic outputs. Each call requests 128 tokens with ignore_eos=true; last prompt plus output reaches107602, covering prior107585 prompt-plus-completion stress envelope. No tools executed or trajectory/accuracy equality claimed. Model transport name is normalized; max_tokens128/temperature0/streamfalse/nativeIDs/ignore_eos are explicit diagnostic overrides, not release changes.

Coordinator launch commands from execution root:
```sh
bash bringup/artifacts/b1-singleton-history-capacity/launch.sh
python_env/bin/python bringup/artifacts/b1-singleton-history-capacity/client.py
```
Run client alongside server after device admission; server launcher bounded648s. Client waits health<=360s; requests bounded180s each; queue owner enforces total .18h box. Forecast remains .10h, adding~11s for384 output tokens. C1/B1 full60, context262144, APC, async/chunked defaults, pinned stock plugin and sampler override unchanged. Retention/phase export enabled. Source/runtime hashes, original source digest, native usage and exact prompt-ID counts, engine idle, preemption delta and zero retention loss checked after live data. Attribute cache reuse by sequential prefill starts/lengths under C1; report derived skipped-prefix fraction, not API cache counters if missing. Require>=.9 for both returning prefixes. No final result before live.
