# AutoFix: sharded prefill norm and collective workspace capacity

Failing contract: optimize full-width real TP4 prefill without lowering precision, logical sequence support or context; retain a valid traced residual layout.

AutoDebug (AUTODEBUG.md) rederived the sharded norm scratch allocation from current generic normalization sources. The7-column,128×768 FP32 configuration needs925696 static bytes/core, conflicting with live layer tensors. In-place output does not eliminate that scratch. The8-column128×672 configuration needs825344 bytes/core; subblock3 divides21 tiles and fits FP32 destination limits. The original height planner also admitted empty trailing shard rows for awkward tails.

Accepted fix:8-column block sharding, block_w21, subblock3; height=32*ceil(total_rows/256), grid_y=ceil(total_rows/height); inplace only on a newly converted private sharded tensor. Original residual ownership is retained. Norm8 and norm8pool complete the exact initial-prefill/decode/read/warmed-prefill sequence and traced replay with real weights. Sliding prefillPCC=.99991582, decode passes. Final boundary, batch, stress, watcher, maximum-context and two-layer integration checks pass; see acceptance_summary.json and stack.json.

Separately, allocating distinct L1 all-gather outputs for all four batch signatures caused full-attention circular-buffer overlap (1126144 live floor versus1130176 static end). Pooling the physically identical32-row outputs by projection role fixes the focused run. Views preserve logical B1/8/16/32. Full-stack ownership must share the two output backings across decoder layers: each AG output is consumed immediately by a non-inplace norm, before any later same-role AG overwrites it. Residual outputs must not alias the workspace. A focused alternating-layer/batch replay test was implemented by an isolated AutoFix subagent; parent owns all hardware validation.

Launcher preflight initially failed because bubblewrap is absent. Retried the inspection-only launcher with AUTODEBUG_SKIP_CHILD_SANDBOX=1 under existing danger-full-access authorization, personal CODEX_HOME and no ambient API keys. No credentials or remote state changed.

Status: focused repairs and all54 final correctness cases pass. All eight final profiles pass. Independent review is recorded separately in stage_review.md. No kernel repair or final performance claim is inferred from static analysis alone.
