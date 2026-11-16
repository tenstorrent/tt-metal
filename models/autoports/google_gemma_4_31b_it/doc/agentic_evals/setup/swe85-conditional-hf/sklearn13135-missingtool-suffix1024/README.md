# Sklearn 13135 call29 suffix-1024 matched served/full candidate

Prepared host-side only; no inference was launched. Each mode uses the exact original final1024 prompt token IDs (indices 16371–17394), with no decode/re-encode, seeds42/43/44, the same chosen64 continuation, and one free-run cap1024. The served manifest uses the selected `stage12_head_bfp8_lofi` policy from `head_bfp8_build_kwargs.json`; full mode uses full precision.

The predecessor full-precision suffix512 run did not reproduce the original full-prompt seed42 `missing_tool` observation. A separate selected-policy head-BFP8 suffix512 correction is also pinned, because the earlier suffix512 candidate accidentally used head-BFP4 for served mode. That BFP4 run is retained as historical evidence and is excluded from the selected-policy forecast and semantic comparison.

The original full-prompt `missing_tool` observation began at generated token665. This candidate extends both context and cap to1024, allowing the run to reach that onset if generation continues; it does not establish behavior under the original17395-token context.

The forecast uses the measured selected-policy BFP8 suffix512 served controls and exact full-precision suffix512 chosen64 controls. It includes broad allowance for the doubled context/cap and seed variation. The run is serialized served then full, each under a 600-second timeout. Root must bind and accept the P2 and suffix512 evidence, forecast, manifests, commands, and source hashes, then remove `STOP_BEFORE_LAUNCH` only after admission. Full requires a root forecast reconciliation for served. Active HF/model processes and device owners are checked before each actual launch. HF follow-up is deferred until matched TT evidence and is not part of this candidate.

No task-success, model-cause, or blanket-waiver conclusion follows from this diagnostic.
