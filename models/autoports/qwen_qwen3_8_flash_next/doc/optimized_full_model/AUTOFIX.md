# AutoFix — real-fixture scorer upload

Status: discriminating hardware validation pending. Production scorer is unchanged.

The failing contract is that a real 16MiB BF16 key fixture can be uploaded to the
four-chip mesh before either reference or native scorer executes. Two bounded
attempts stalled; the second marker locates the wait inside key upload. Small
anonymous synthetic inputs pass the native kernel, including Q1/Q3/Q32, exact
per-head dots and ordered selection. The source-only AutoDebug report found no
CB violation and identified a weak exactness gate, now fixed.

Priority1 hypothesis: long-term DMA pinning of borrowed mmap-backed fixture pages.
CPU pointer/mapping checks and TTNN's64MiB replicated-size pin threshold support
this route; see upload_pinning_diagnosis.md. The minimum change is .clone() of
BF16 fixture keys before upload, preserving all values. Raw probe fixture queries
are cloned as well. No dtype, fidelity, cache, score arithmetic or model policy
changes are part of this repair.

Discriminating check: rerun the same full-capacity score comparison after healthy
reset, using cloned anonymous keys. INPUT_UPLOAD_BEGIN must be followed by
KEY_UPLOAD_DONE, then exact score/ordered-selection checks must pass. A successful
run supports the storage/pinning explanation without proving the internal Linux
kernel defect. If upload still stalls, this hypothesis is refuted and the next
experiment must isolate pin threshold and row-major upload from tilization.

Changes: tests/probe_causal_indexer_scores.py and tests/probe_causal_indexer_raw.py.
Both now require bitwise equality, with error magnitudes diagnostic only. The raw
probe additionally checks per-head dots and invalid zeros. Uploaded real-fixture
acceptance, tracking, watcher and integrated full-model gates remain pending.

## Hardware validation result

After recovery2 list/reset/list and mesh arithmetic passed, causal_all_v3.json/log completed in18s (exit0) with cloned keys and unchanged native kernels. Upload markers completed; all27 real-fixture Q1/Q3/Q32 cases across positions0,3,127,2047,2048,32767,205504,262112,3 had bitwise-equal live scores and exact ordered IDs on all4ranks. This verifies the clone workaround and supports file-backed pinning as the trigger; the internal kernel defect is not proven. Small raw per-head dot/zero checks also passed. No model math repair was required.
