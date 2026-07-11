# AutoFix: exact decode evidence and sliding cache ownership

## Starting evidence

`stage_review_resume.md` rejected the old advertised-context decode claim because
it decoded the same token at an already-filled position and compared TTNN against
TTNN. Fresh `AUTODEBUG.md` inspection independently confirmed that evidence bug
and found a runtime defect: padded lanes from non-aligned sliding prefill could
wrap modulo 1024 and overwrite live cache slots.

## Hypothesis experiments

Two independent focused investigations traced the padding writes and the HF
oracle semantics. The padding hypothesis predicted ownership corruption after
lengths 1025 and 1057. The oracle hypothesis reproduced the official HF layer's
single-query math and matched stock HF at near-1.0 PCC for both target layer
kinds. Hardware tests then confirmed the cache defect and established that
changed stable trace buffers are consumed at random positions and at the
1023->1024 wrap.

The retained runtime fix bulk-fills complete valid tiles, then uses device-side
sequential paged updates for only the valid tail. The retained evidence fix uses
262143 history tokens, a distinct final token, stable captured allocations,
direct HF-vs-TTNN comparison, nonidentity page mapping, and a wrong-position
sensitivity control.

## Final status

- Standard suite: 25 passed, 8 explicitly gated skips.
- Sliding ownership: distinct decode PCC >= 0.997702; K/V PCC >= 0.999885.
- Changed-input traces: all direct PCC >= 0.998948; deterministic repeated replay.
- Exact total context 262144: sliding PCC 0.999406, full PCC 0.998875.
- Watcher: 4 changed-input trace cases passed; no error/assert/hang indicators.
- Final-source performance: warmed prefill 3.521/4.254 ms and traced decode
  2.577/2.911 ms for sliding/full; canonical text artifacts contain the full
  human-readable operation tables.

No proposed hypothesis was left unresolved, and no fallback or capability waiver
was introduced. `stage_review_autofix.md` records the final `clean-pass`.
