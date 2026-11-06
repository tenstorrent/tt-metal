# Django capped-loop diagnosis

The root cause remains open. The original 32,768-token capped response remains a failure, even though its native finish reason was `tool_calls`. The bounded investigation used 0.540240841111 device hours against a final 1.0-hour box; no speed credit or model-quality pass is claimed.

Full60 HF and TT consumed the same reconstructed 42,823-token prefix. Pinned HF chunked CPU execution completed in 5007.956 s (99,583.125 MiB peak RSS), within the extended 7100 s allowance; the initial pilot stopped on its forecast. The one selected next-token row has KL(HF||TT) 0.1351848168, JS 0.033789358, equal top1 and centered maximum absolute logit difference 14.1400853. Release top20/top_p0.95 support mismatch makes that truncated KL infinite; JSON records null and the explicit zero-support flag. These observations neither eliminate precision effects nor establish that the model inherently loops.

## Native replay findings

Both fresh native-chat requests verified every current stock prompt ID, used T1/top_p0.95/top_k20, max32768 and request timeout3600, planned clients1 on B8 for route reproduction. Seed0 returned200tokens (33reasoning) in30.905714s and a finite edit-script toolcall. Seed1 returned1043tokens (1000reasoning) in69.130624s and a finite source-inspection toolcall. Neither reached the cap or showed the detected periodic/repeated-a loop. Tool commands were never executed; no task-correctness judgment follows. Stochastic probing stopped after seed1.

Private raw outputs and token IDs remain in `bringup/artifacts/django-cap-loop/native-replay-seed{0,1}/response.json` and `generated-token-ids.json`; compact summaries are beside them. Seed1 has its exact `request.json`; seed0 uses source `native-replay-request.json` plus the recorded request hash because the older client did not persist a per-run request file.

## Limits and next action

The original sampling seed and prompt token IDs were not recorded. Reconstruction matches its token count and the exact current stock server tokenization, not a provable historical token sequence. The HF result controls only the first conditional row, not either complete stochastic output or the original loop onset. Existing timing evidence separately identifies11 missing native FormatError responses; their token counts and cutoffs remain unknown.

No runtime or precision fix was accepted by this effort. Preserve the failure and obtain raw response/usage on any future normal-rung capped or malformed event; a captured loop onset can justify a targeted same-prefix HF/TT comparison. Additional random probes are not justified now and must not delay scheduled rungs. See `result.json` for predictions, costs and exact evidence paths, and the existing tokenization/timing evidence for their independently recorded findings.
