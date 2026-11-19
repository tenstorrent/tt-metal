# Short-capacity prefill temporal alignment adaptation

CPU-only investigation; no new hardware claim. Parent owns device execution.

The measured full-attention `prefill5_cap512_chunk128_l3_b32_s513`
691.9028 ms is not a useful measurement of four 128-token chunks. The inherited
scheduler chooses single-token updates whenever the absolute start is not a
multiple of `_attention_chunk(state)`. A large-enough page table selects 512.
Thus length 513 schedules one 128-token chunk, 384 single-token updates, and
one final token: 386 chunks. The 512-temporal control takes two chunks.
`flat_prefill.py` lines 192–202 and frozen `optimized_decoder.py` line 1238
establish the cause. Native SDPA requires both q- and k-chunk alignment
(`sdpa_device_operation.cpp` lines 392–403); removing the guard is invalid.

The isolated `tt/prefill_alignment_candidate.py` class selects K=128 only at
cache capacities <=2048, and preserves the inherited decode K rule explicitly.
It does not mutate policy or methods during forward. The regular scheduler
then handles length 513 as 128+128+128+128+1, and continues to own unaligned
public starts through single-token updates. Long-context K=512 is unchanged.
Smaller SDPA blocks can accumulate more rounding, so unchanged oracle gates
are required before considering this adaptation viable. This is not yet a
default. `prefill_alignment8_cpu.json` records 80 nonaligned scheduling checks,
the exact old/new schedules, and the decode-rule negative control.

The cap4096/chunk128 failure is independent: `packed_qkvg` 2D matmul requests
1,608,640 static bytes, beyond physical 1,572,864 L1. At M=4096, grid11x10,
Kblock8, N=3584, per-core M=13 and N=11. Major CB payloads are 1,496,064
bytes; the observed extra 112,576 gives the exact failure. A one-row output
height block preserves M ownership and loops over 13 height blocks instead
of allocating all partial/output rows. Major payloads fall to 291,840 bytes
(excludes other static/live buffers; hardware still must validate). The
4096 retry sets existing `prefill_height_block=true`, applying this legal
bounded-height adaptation to all inherited prefill matmuls.

Run `python -m models.autoports.qwen_qwen3_6_27b.tests.run_multichip_search
models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/prefill_alignment8.json`
through the established serial queue. This is three fresh jobs, caps512,
2048, and4096, on actual TP4 B32/S513, preserving the existing precision,
cache, contribution, and trace gates. The original failure/performance logs
remain intact. If retained, follow with nonaligned/resume and 32K context
gates on integrated defaults; the helper does not justify long-context
precision claims by itself.
