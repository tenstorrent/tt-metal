# Physical prefix transfer and full-model continuation

Collected October 11, 2026 from 10.228.203.98. This is a correctness milestone
on the isolated prefix/offload branch; no serving or AgentX launch occurred.

- [Opaque-byte transfer](receipts/prefix-transfer-v2/transfer.json): all TP ranks,
  shuffled pages, FP32 recurrent state, BF16 history, unchanged neighbours and
  stable tensor addresses passed. The adapter preserves physical BFP8 bytes.
- [Four-layer continuation](receipts/prefix-continuation-v1/layers-4/continuation.json)
  passed before the full-model attempt.
- [Full-64-layer continuation](receipts/prefix-continuation-v2/layers-64/continuation.json)
  passed at October 10, 23:32:37 UTC with clean teardown. Physical TP4 IDs were
  `[0,4,12,8]`, canonical batch 16, active slots 1 and 3, packed decode bucket 8.

The full test captures a 4096-token prefix, restores it into independent pages
and a different slot, appends 32 suffix tokens, then performs 32 teacher-forced
decode steps. All compared vocabulary logits are bit exact. A second checkpoint
after decode restores at 4160 tokens while preserving the original trace and
resident buffer addresses; the next output is exact and the neighbour remains
unchanged. This is an exclusive single-threaded fixture, not live scheduler
concurrency, independent model-quality qualification or 128K/256K validation.

Checkpoint payload is 296,550,400 bytes (282.8125 MiB). Capture took 2.207 s,
restore 3.542 s and original prefix prefill 1.185 s. Each transfer used 16,768
windows with randomized physical pages; the largest retained host window was
786,432 bytes. Physical transfer counters include convolution read/modify/write.
Restore is currently slower than recomputing this short prefix. File I/O may
be served from the OS page cache; no SSD line-rate or TTFT improvement is claimed.

Failures are retained alongside success:

1. First physical transfer used the wrong mesh coordinate when obtaining a
   host shard. Child views preserve parent coordinates; rank 1 does not become
   coordinate `(0,0)`. Corrected to allocate from the active view and read
   `(0,rank)`. The failed attempt stopped before device writes.
2. First full-model continuation hit pytest's default 300-second timeout during
   model setup, before numerical validation. The bounded controller now passes
   an explicit stage timeout. It reused the successful four-layer receipt only
   after checking full source identity except the supervisor; the 64-layer
   retry passed. This was a harness timeout, not an observed numerical failure.

Launch records and frozen source manifests live under the control directories
in `receipts/`. They retain exact opt-in environment, checkpoint identity,
commands, hardware lock, memory bounds and persistent systemd invocations.
The jobs survived disconnects, not reboot. Large checkpoint payloads and model
weights are deliberately absent from Git. Logs are gzip-compressed without
altering content; `collection.json` records exact retained hashes.

Published adapter and test bytes match the full64 frozen manifest. The
supervisor has formatting-only differences: its AST matches exactly. The
frozen supervisor, diff and `published-source-check.json` preserve that
distinction; no active or historical source was rewritten.

Source: `tt/prefix_transfer.py`; physical tests:
`tests/test_prefix_transfer.py`, `tests/test_prefix_continuation.py`; bounded
controller: `demo/run_prefix_validation.py`. CPU planning/codec/storage suite:

```sh
python3 -m unittest discover -s models/demos/qwen38_27b_qb2/tests/unit -p 'test_prefix*.py'
```

Next work is serving leases/lifecycle, admission/eviction/replica affinity and
faster transfer. Capability flags remain off until integration validates them.
