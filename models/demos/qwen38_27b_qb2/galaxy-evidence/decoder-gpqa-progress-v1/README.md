# BFP8 serving checks and partial GPQA snapshot

All six real API checks passed, including streaming/greedy equivalence,
multi-turn chat and concurrent greedy repeatability at client concurrency 128.
The original accuracy service remains active with invocation
`ecb6408f79414ed0974c6352f006bce5`. Its deployment records all eight worker
bindings and the effective all-BFP8 precision policy.

At 10:43:14 UTC the progress receipt reported **162 correct out of 180
completed**, zero truncations, with 18 of the fixed 198 questions outstanding.
This is a partial completion-order snapshot, not a final score or qualification.
The unchanged gate is 177 naturally completed correct responses out of all 198.
The independent audit remains queued behind the full evaluator.

Retained evidence includes the API receipt, deployment/source/precision record,
exact evaluation protocol, progress state, metadata-only response JSONL snapshot
and live service observation. Files were read sequentially during live execution;
their sample counts may advance between reads. The snapshot includes response
hashes but no private benchmark prompts or response text. The complete final
run and independent completion audit must supersede this snapshot.
