# Remaining decode costs: source-only audit

Audited 2026-11-12. No device execution or production edits. This audit uses
`tracy_movement/{0,3}/decode_perf_report.csv`, their raw `decode_ops.csv`, and
`decode_provenance.json`. Both provenance records identify implementation
`07fd1a68bb3909fff8c0aa668b61b81b9439a8f63e89bfe4cf00174e8a32bcbb`, four devices,
warmed traced decode at position 1024 after prefill 1024, without watcher/tracking.
These are **B1 movement-path profiles**, not measurements of the subsequently
integrated pooled batched path. Reprofile final defaults at B1 and B32 before
accepting rankings or calculating a final stop-rule fraction.

## Measured operation families

Numbers below sum numeric CSV rows after the report's multi-device merge; they
are not the sum of four chip times. Movement includes slices, padding, concat,
reshape, sharding conversion, copies, untilize and typecast. Family times exclude
gaps; the last row reports gaps separately.

| Family | Linear layer 0, us | Full layer 3, us |
|---|---:|---:|
| Five projection matmuls | 131.397 | 125.612 |
| Two asynchronous all-reduces | 28.8575 | 28.56925 |
| Norms | 17.747 | 23.001 |
| Attention kernels including convolution/cache update | 44.548 | 28.089 |
| Movement kernels | 45.345 | 38.049 |
| Other kernels | 20.496 | 20.961 |
| All device kernels | 288.3905 | 264.28125 |
| Op-to-op gaps | 30.866 | 41.776 |

Linear's three `FillPadDeviceOperation` calls cost 25.496 us; g/beta padding
alone costs 20.688 us. Its fused convolution and chunk-GDN together cost 44.548
us. Full attention has 28 movement calls and 21.607 us of their associated gaps;
SDPA plus fused cache update costs only 28.089 us. Both MLP gate/up matmuls read
the same normalized activation, costing 58.888/58.848 us combined. Their
separate calls still require separate weights, so packing does not halve bytes.

The report labels every matmul optimized, but this is not a whole-decoder
roofline pass. The 106–117% reported HiFi2 FLOPs values are derived report
estimates, not physical utilization measurements. B1 contains useful work for
one logical row despite a physical M=32 matmul tile.

## Stored-byte bandwidth floor

Use the installed report's existing Blackhole convention: **512e9 bytes/s per
device**, from
`python_env/lib/python3.12/site-packages/tt_perf_report/perf_report.py:484-495`.
That entry also records eight DRAM-sharded matmul cores. This is the tool's
architectural ceiling, not newly measured sustainable bandwidth or an invented
P300 board aggregate. Earlier stage accounting uses the same ceiling in
`tests/summarize_optimized_final.py:60-71`.

Standard BF8 tiles store 1024 mantissa bytes plus 64 exponent bytes, hence
17/16 bytes/value; see `tt_metal/impl/data_format/tile.cpp:136-145` and
`tt_metal/impl/buffers/prefetcher_matmul_design.md:123`. Production creates
eight-bank width-sharded weights with per-bank N rounded to 32, so stored
projection bytes are `K * ceil(N/256)*256 * 17/16`. Shape and sharding sources:
`tt/multichip_decoder.py:199-266`; the raw CSV confirms BF8 DRAM-width-sharded
decode inputs. Include storage padding for the linear packed projection, whose
logical N=3136 becomes stored N=3328. Do not count the interleaved prefill copy
or unused packed-MLP copy as decode traffic merely because they are resident.

| Active projection storage per chip | Linear bytes | Full bytes |
|---|---:|---:|
| Packed input: K4096, N3136/2560 | 14,483,456 | 11,141,120 |
| Attention output: K1024, N4096 | 4,456,448 | 4,456,448 |
| Gate: K4096, N3072 | 13,369,344 | 13,369,344 |
| Up: K4096, N3072 | 13,369,344 | 13,369,344 |
| Down: K3072, N4096 | 13,369,344 | 13,369,344 |
| Sum | 59,047,936 | 55,705,600 |

Linear recurrent state is FP32 `[B,8,128,128]`: 524,288*B resident bytes per chip
and at least 1,048,576*B read-plus-write bytes/step. Its BF16 three-row history
is 12,288*B resident bytes, with at least 24,576*B read-plus-write bytes if the
whole history is read and replaced. Full KV is BF8 `[pages,1,32,256]` for each
of K/V: 17,408 bytes per allocated 32-token page pair per request per chip.
At context 262144 this is 142,606,336 bytes/request/chip. At decode position
1024, the useful tile-covered prefix contains 1056 tokens: minimum K/V reads
are `2*1056*256*17/16 = 574,464` bytes/request/chip. The 256-token SDPA chunk
may read additional masked tail tiles; do not assert this useful-prefix count
is an exact kernel-traffic measurement. State allocation sources are
`tt/multichip_decoder.py:419-475`.

For consistency with earlier stage accounting, a conservative floor includes
active projection storage plus recurrent read/write or useful-prefix KV reads,
and omits history, other activations, norms, KV writes, metadata and CCL:

| B1 floor at position 1024 | Linear | Full |
|---|---:|---:|
| Bytes/chip | 60,096,512 | 56,280,064 |
| Floor at 512 GB/s, us | 117.376 | 109.922 |
| Floor / profiled kernels+gaps | 36.77% | 35.92% |
| Maximum latency consistent with 70% of this floor, us | 167.680 | 157.031 |

These fractions use profiler latency, not final clean warmed latency. For final
reporting divide by the final default warmed trace time. A stricter useful-only
linear weight floor excludes 835,584 bytes of storage padding, reducing its
floor by 1.632 us; storage padding must not create an artificial stopping pass.
For a truly batched projection path, weights are read once per batch: use
`(59,047,936 + B*1,048,576)/512e9` for linear and
`(55,705,600 + sum_i 2*ceil((position_i+1)/32)*32*256*17/16)/512e9` for full.
**Do not multiply all projection bytes by B** after sharing projections.
Equivalent TP4 accounting multiplies both bytes and bandwidth by four, leaving
the latency floor unchanged. Sum layer floors for serial decoder-stack timing.

## Actionable candidates and constraints

1. **Remove B1 g/beta padding and excess GDN token work.** The current path
   explicitly pads one active token to 32 and invokes chunk-GDN
   (`tt/multichip_decoder.py:618-706`). Compare the existing recurrent kernel
   with native one-token state updates against a setup-owned neutral 32-row
   buffer updated only in its active row. Preserve g=0/beta=0 inactive tails.
   Measure the complete convolution/state/norm group; extra normalization or
   state copying can outweigh the removed padding. The new pooled path already
   changes this family, so first profile its B1/B32 defaults and head-wave tails.
2. **Full-attention partial RoPE fusion.** Two rotary ops plus many
   slice/concat/shard steps are material. The available
   `rotary_embedding_llama_fused_qk` requires BF16 height-sharded Q/K on
   disjoint cores, repeated cos/sin, and a transform tile
   (`.../rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.cpp:25-113`).
   Ornith rotates only the first 64 of 256 dimensions. Adapt to a 64-wide
   rotary subview and preserve the 192-wide tail, or implement identity rotation
   of that tail with verified pairing; applying ordinary full-width RoPE is
   incorrect. Keep SDPA's Q reducer ownership and fused-cache K/V disjointness.
   Count added preparation and tail joins in the before/after result.
3. **Packed gate/up plus fused activation, and larger DRAM reader blocks.**
   Revisit the complete packed projection/SiLU/product/down-consumer family
   using the new actual producer shard layout, at B1 and B32. The earlier
   candidate being slower is evidence for that configuration, not all packed
   layouts. Five matmuls still account for 126–131 us; a bigger pipeline block
   must preserve exact K coverage. Evaluate real-weight PCC and entire layer
   latency rather than only the kernel that improves.
4. **Avoid cache-pool packing on the default batched path.** Setup-owned shared
   K/V pools eliminate concatenating and scattering every request's entire
   allocated context per decode. Those full-pool copies do not occur in these
   B1 profiles and must be counted in private/mixed-pool B32 evidence. Retain
   correctness for subset/permutation scheduling; account for pool memory in
   the context contract.
5. **Fuse residual addition with normalization only with a reusable residual
   result.** `rms_norm` accepts `residual_input_tensor`
   (`ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:22-27,71-75`),
   but the decoder also needs the unnormalized residual after the MLP. Merely
   recomputing the add for that branch does not remove an op. Evaluate a legal
   in-place/dual-result variant or demonstrate that redundant add still wins;
   protect live trace input and workspace ownership.
6. **Collective/residual-layout family remains a whole-layer decision.** Two
   asynchronous ARs cost about 29 us and are intra-layer dependencies. Compare
   persistent hidden-sharded residual + distributed norm + fused gather/matmul
   and row-output matmul/scatter together, including both layer boundaries.
   Immediate restoration to replicated residual does not test this family.
   Keep separate accounting for mandatory model reductions and inter-layer
   conversions. Profile the final shared workspace path before further CCL
   precision or placement changes.

This audit proposes experiments; it is not rejection evidence, a correctness
pass, a final profile, or satisfaction of the optimize Stop Rule.
