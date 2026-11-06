# Response-logging benchmark disposition

All clients completed: headline8/8 exact4096/128 atC1/B1, matrix8/8 nominal128/128 atC1, and one warmup. Full60 runtime matches13 publicationab273 source files; launch hashes unchanged. This logging-enabled streaming configuration is **unselected** for performance measurement.

Headline wall35.136170s, delivered TTFT4386.854759ms and TPOT0.037546873ms reflect buffered SSE delivery, not token-out decode cadence. Matrix wall27.124569s likewise completed, but its streaming timing is buffered. Artificial reciprocal-TPOT decode rates are omitted. Native matrix prefill lengths are retained in RESULT.json phase accounting; nominal128 is not a claim about template length.

Complete archive:72frames,4370events,2185 contiguous paired steps; finalpending0/errorsnone/no retentionloss. Idle shutdown has no remaining owners. Device charge0.083072753h versus0.076054659h forecast and0.15h box. This is no model speedup claim.

Freeze logging off for streaming B1 benchmark: existing ab273 baseline TTFT753.7608765ms, decode35.3210051tokens/s/user, matrix27.0979348s remains selected. Freeze logging on only for nonstreaming GPQA B8 where complete response visibility is required. Per-arm logger environment and flags are part of frozen configuration. No runtime patch, additional device test or central/telemetry edit performed. See AUTOFIX.md for source-proven cause and repair.
