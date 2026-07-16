# Qwen3-32B optimized decoder

This stage adds the single-device optimized dense decoder layer in
`tt/optimized_decoder.py`. It preserves the functional decoder's batch-32
prefill/decode, paged KV-cache, non-aligned logical sequence, deterministic
replay, and representative-layer contracts. It contains no multichip,
full-model, generator, or vLLM work.

## Selected path

- BF16 activations, residuals, and norm weights.
- One packed QKV projection and separate O, gate, up, and down projections, all
  with BFP4 weights/LoFi. BFP8 remains the paged KV-cache dtype.
- BFP8 paged KV caches.
- Explicit 2-D prefill matmul configs with `in0_block_w=10`; public sequence
  lengths remain unrestricted and flattened rows are internally chunked at no
  more than 640.
- A 40-core L1 width-sharded decode residual/norm/gate/up stream, a separately
  swept 32-core down projection, and DRAM width-sharded projection weights with
  DRAM-sharded matmul program configs.
- TTNN QKV head creation/concatenation, rotary embedding, paged cache update,
  prefill SDPA, and decode SDPA composite operations.
- SiLU folded into the gate/up multiply.
- One fixed-address decode-position slot. Host index refresh happens before
  replay, embedding writes into preallocated RoPE outputs, and no allocation or
  per-position dictionary growth occurs while a trace is live.
- The last residual moves directly to the public DRAM output before its view,
  avoiding an extra copy.

`OptimizedDecoder` reuses only construction-time validation/constants from the
functional class and owns both runtime methods. A source-audit test rejects
`super()`, `torch`, `from_torch`, and `to_torch` in measured runtime helpers.

## Final result

Real Qwen3-32B layer 32, Blackhole p300c, one device, batch 32, prompt-derived
HF boundary activations, prefill length 17, decode position 17:

| Path | Prefill PCC | Decode PCC | Warmed prefill | Traced warmed decode |
|---|---:|---:|---:|---:|
| Functional BF16 | 0.999999627 | 0.999869502 | 83.252 ms | 82.101 ms |
| Final optimized | 0.999996978 | 0.998990068 | 5.477 ms | 1.217 ms |

The final path is 15.20x faster for prefill and 67.44x faster for decode. Its
decode is 10.93% faster than the strongest retained earlier correct traced
baseline, 1.366718 ms (`results/candidates/decode_program_configs.json`). All
optimized PCCs exceed the functional stage's 0.99 acceptance
bar. The explained delta from functional BF16 is due to BFP4 projection
weights, BFP8 cache storage, and LoFi compute; real length-31 output and cache
PCCs remain above 0.9988/0.9943.

The authoritative same-run result is `results/final/before_after.json`. It
contains 25 warmed prefill iterations, 200 trace replays, activation provenance,
the complete selected config, and source hashes.

## Operation-topology audit

The audit preceded local knob tuning.

| Region | Current issue/candidate | Action | Evidence |
|---|---|---|---|
| Q/K/V | Three same-input projections | Pack into one QKV matmul and use composite head creation | Final decode has one 100-101 us QKV matmul |
| Attention | Hand-built score/softmax/value graph would add movement | Keep prefill/decode SDPA and TTNN head composites | Decode SDPA is 50-52 us; 8x4 was slower than 8x8 |
| RoPE/index | Per-position tensor allocation cannot coexist safely with a live trace | Use bounded row-major lookup tables and one preallocated position slot | Positions 17-20 replay with stable addresses and output/cache PCC >0.9992 |
| O/residual | Composite output and residual/norm consumers require different legal layouts | Keep the measured boundaries and a persistent 40-core width-sharded residual chain | Boundary ops are 1-3 us; removing compatibility reshards is not legal |
| Gate/up | Two dominant same-input matmuls invite packing | Implement and retry packed output splitting with a legal block-2 config; reject | 1.551 ms versus 1.367 ms contemporary separate path |
| SwiGLU | Separate SiLU then multiply is redundant | Use `mul(..., input_tensor_a_activations=[SILU])` | Correct and retained; elementwise row is ~35 us |
| Decode weights | Interleaved weights underuse DRAM banks | Use DRAM width sharding and DRAM-sharded matmul configs | Final all-BFP4 projections reach 260-297 GB/s |
| Final output | Interleave, reshape, then copy added movement | Materialize final DRAM output directly | No final `CopyDeviceOperation` in Tracy |
| Prefill MLP | `tt-perf-report` suggested L1 input 0 | Retry as a chained L1 candidate; reject | 6.219 ms versus 5.635 ms contemporary DRAM-input path |
| Long prefill | Full intermediates can exceed L1/DRAM and tempt public alignment caps | Slice internal work at logical row boundaries and concatenate | Length 31 and capacity length 8192 pass; no `seq_len % chunk` condition |

There are no collectives in this single-device stage. The final traced replay
has no tilize/untilize, host fallback, or avoidable copy. Remaining reshards are
the measured compatibility boundaries between head/SDPA composites and the
chosen residual/MLP core geometries.

The final-policy MLP sweep uses coherent phase-specific input shards rather
than reusing the 40-core residual shard for every candidate. Gate/up therefore
genuinely exercises 16/20/32/40/80-core K shards and their 10/8/5/4/2-tile
block families. The 16/block-10 and 20/block-8 maxima hit exact L1 limits;
adapted legal variants run and lose. Down 16/20 trials include blocks 25/10/5
and 20/10/8/5 after their block-50/40 maxima fail L1. A 500-replay tie-break
selects down-32 at 1.217996 ms over down-40 (1.219028), down-16/block-25
(1.219652), down-20/block-20 (1.222648), and gate-32 (1.224401).

## Shard-advisor hard gate

The advisor was run this pass after the dense attention+MLP rewrite using the
required bootstrap. Its authoritative artifacts are:

- `shard_advise/report.json` (`total_ops=26`, `final_choices=23`, spill pass ran)
- `shard_advise/final_ir.mlir`

The complete legal recommendation was implemented behind
`decode_matmul_mode="shard_advisor"`, including its 1-D matmuls and advised
residual, Q/K norm, RoPE, and MLP layouts. One necessary adaptation preserves a
sharded input for `nlp_concat_heads_decode`, whose constraint the report marks
unfixable.

| Advisor trial | Prefill PCC | Decode PCC | Prefill | Decode | Decision |
|---|---:|---:|---:|---:|---|
| Full legal layouts | 0.999997490 | 0.985268708 | 5.627 ms | 1.743 ms | Reject: decode fails 0.99 |
| Advisor matmuls/residual, production head layouts | 0.999997490 | 0.999272419 | 5.645 ms | 1.737 ms | Correct but slower; reject |

Applied directionally: lower-precision grouping and a sharded residual/norm/
linear chain. Precision was then selected independently from real-model data.
Rejected with evidence: padded 107/100/80-way L1 layouts and 1-D
interleaved-weight matmuls. The selected DRAM-sharded path is both correct and
30% faster than the correct advisor-matmul trial. Detailed mapping is in
`work_log.md`; raw evidence is `results/candidates/advisor_full_vs_matmuls.json`.

## Correctness, trace state, and capacity

- Layer 32 represents the sole meaningful decoder-layer kind: all 64 layers are
  dense and architecturally identical.
- The final Watcher run passed five tests: the static host-free audit, a
  conservative optimized random stress, final-policy real prefill/decode,
  final-policy logical length 31 plus four decode updates, and advancing traced
  positions 17-20. No Watcher fault signature was found.
- The final policy's real length-17 PCC is 0.999997 prefill and 0.998990 decode.
  Its real length-31 PCC is 0.999995 prefill and 0.998826-0.999354 across
  positions 31-34. The separate conservative random stress obtains 0.996456
  prefill and 0.996529-0.996572 decode and is not used to veto the real winner.
- Advancing trace output PCC is 0.998990-0.999509; key history is
  0.998875-0.998898 and value history is 0.994391-0.994423. Stable addresses
  and changing outputs prove that position/cache state advances without trace
  reallocation. Separate equal-input replays are bitwise deterministic.
- Batch-32 prefill passes at 8192. The isolated 16384 probe reaches a hard
  10,737,418,240-byte allocation failure with a 29,524,352-byte largest free
  block in the final run. `doc/context_contract.json` therefore advertises the
  evidence-backed 8192 floor, not an alignment or convenience cap.

## Limitations

- The public shape is the compiler-derived batch-32 decoder contract; this
  stage does not claim a batch-1 interface.
- 8192 is the largest passing optimized point tested, not the exact adjacent
  maximum. The next tested point, 16384, fails from device DRAM.
- Each prompt-derived activation is an actual HF layer-32 boundary repeated
  across the 32 decoder slots. The length-35 artifact runs layers 0-31 over a
  repeated real prompt-token sequence; this adds non-aligned and advancing
  position coverage while retaining the emitted batch contract. This is not a
  full-model distribution study.
- Position lookup tables are bounded to the supported 8192 context. They are
  model-independent and shareable by a later layer stack; this single-layer
  stage owns one copy.
- Full-model generation, multidevice topology, and serving remain intentionally
  deferred.
