# Opt-in native B1: above 40 tokens/sec/user

With **`QWEN38_COMPACT_B1_RECURRENCE=1`**, warmed full64 B1 throughput is
**36.0909 → 40.6731 tokens/sec/user (+12.70%)**. All three candidate repeats
exceed 40. The unflagged runtime retains the baseline. Median generator TTFT is
**436.933 → 436.612 ms**; this establishes no meaningful prefill improvement.
Independent stage review returned **clean-pass**; see [the review](STAGE_REVIEW.md).

| Same native workload | Control | Opt-in candidate |
|---|---:|---:|
| Token-out TSU, median of 3 | 36.090930 | 40.673091 |
| All TSU repeats | 36.090930 / 36.093913 / 36.088429 | 40.671143 / 40.674132 / 40.673091 |
| TTFT ms, median of 3 | 436.932691 | 436.612106 |
| E2E decode ms/token | 27.707793 | 24.586280 |
| Corrected representative device-work estimate ms | 26.790020 | 23.694282 |
| Fresh AIME prefill top1/top5/top100 | Selected policy retained | 99/100/100% |
| Fresh AIME teacher forcing top1/top5/top100 | Selected policy retained | 98/100/100% |

All native timing rows use 64 layers, batch 1, actual context allocation 262144,
4096 input and 128 output tokens, physical chunk 4096 and three warmed repeats.
TTFT starts before reset/input preparation and includes prefill, terminal/head,
sampling and first-token readback; tokenization/load/JIT are excluded. Decode
measures the remaining 127 transitions and final buffered-output boundary. Each
loop has 127 model/sampler replays, zero host refresh/copy/capture or inner
readback, and one final boundary readback. All candidate tokens, mixed repeats,
five complete 512-entry cache-state maps and 12 continuation logits are exact.
Teacher-forcing throughput remains a separate diagnostic.

The default-off candidate changes B1 unmasked T1 decode only. Compact readers
remove head expansion/permutation, four workers handle independent value columns
per head, and the original HiFi4 FP32 recurrence compute source is retained.
A NoC kernel expands BF16 head rows to exact FP32 bits before the original FP32
normalizations. Gate fusion retains explicit FP32 conversion. Actual decode
matmuls remain BF4/LoFi and the head BF8/HiFi2. Masked decode, larger batches,
prefill, selected precision, residual/state layouts and native context are preserved.

Fresh shared 512-token native-chat controls pass: explanation (133 tokens),
structured JSON (114) and code (158) reach EOS with correct answers and match
reviewed controls. Integrated watcher10/NOINLINE with all features enabled passes
exact reduced-fixture states/logits/tokens; full64 tracked validation separately
prepares every supported prefill bucket. Real before/after profiler tables and
source provenance show 21 fewer structural operations per linear-attention layer.
The 23.694 ms estimate is a source-corrected representative extrapolation, not a
direct full64 device measurement. Its gap to measured E2E is 3.76%, below the
contract's 10–15% investigation trigger. The traffic roofline estimate is 7.701 ms.

A coherent direct-L1 prefill trial passes full64 token checks but establishes no
TTFT gain (median 437.451 ms), so its placement is not selected. Legal larger-grid,
head and sampler alternatives are recorded in the rejection ledger. Memory
accounting separates normal DRAM from the fixed trace arena: prepared normal
DRAM costs an additional 98,304 bytes/device; reduced trace subuse does not return
ordinary DRAM capacity. No serving, SWE-task or concurrent-capacity gain is claimed.

See [performance accounting](perf_summary.json), [memory accounting](memory_summary.json),
[commands and evidence](work_log.md), and [the experiment ledger](../local_b1_optimization/work_log.md).
Raw CSV/Tracy/logs/arrays and weights remain local; the patch contains compact code/docs only.
