# AutoFix record

Fresh-context investigations and isolated repair loops are recorded in
`AUTODEBUG.md`, `AUTODEBUG_LONG_PREFILL.md`, `AUTODEBUG_ROPE.md`,
`AUTODEBUG_PREFILL_CONFIG.md`, and `AUTODEBUG_RECORDED_STATE.md`.

## Packed gate/up split

The packed matmul itself stayed finite and correlated, but direct splitting of
its few-core width-sharded output corrupted values. BFP8/HiFi2, packed-order,
and separate-projection probes refuted precision and weight-order hypotheses.
One interleaved helper boundary before the split restored correctness.

The final fused-RoPE rerun gives direct-sharded-split PCC
`-0.0278/-0.0021` at b1/b32, while the correct packed path runs
`0.358/0.473` ms and the correct separate path `0.392/0.507` ms. The boundary
is therefore both necessary for correctness and faster than separate gate/up.

## Long prefill L1 capacity

Default matmul selection rejects the required DRAM-sharded input B. A 2048-row
chunk still requests 1,585,920 bytes versus 1,572,864 available. A 1024-row
internal chunk passes non-aligned 131071 and exact 131072. Public logical shape
and page/cache semantics remain unchanged.

## Head-width-96 RoPE candidate

Direct `rotary_embedding_hf` rejects width 96 and naive width-128 padding
changes the rotate-half pairing. The proven repair permutes Q/K output
coordinates offline into adjacent real/imaginary pairs, applies the same
permutation to cos/sin tables, then uses `rotary_embedding_llama`. Dot products
and reference PCC are preserved.

Manual RoPE measured `0.459/0.701` ms b1/b32; fused RoPE initially measured
`0.358/0.473` ms on random/zero-cache controls. The later recorded-state loop
below proved that speedup cannot ship with either tested SDPA policy.

## Large-prefill program configuration

The original large-M block width 4 was valid but slow at b32. Block width 2
passed both batches and reduced fused-RoPE b32 prefill from `13.216` to
`10.064` ms. Two further adaptations were executed: inner M/N and an 8x10
grid. Both passed b1 but failed b32 PCC, so neither was rejected on its first
TTNN/API error. The cumulative block-2 default passes final real-weight PCC.

## Multi-user physical-M grid

The expanded batch-2, sequence-33 test exposed a grid undercount: flattening 66
logical rows gives three tiles, while tiled storage pads each user's sequence
independently and needs four. The repair counts
`batch_rows * ceil(sequence/32)`. The focused regression and full 16-test suite
pass after the change.

## Recorded semantic b32 attention state

The fresh AutoDebug report used recorded layer-0 checkpoint activations and
their matching nonzero prefix caches. At BFP8/HiFi2 with BF16 KV, the
controlled RoPE/SDPA matrix was:

| RoPE | SDPA | b32 PCC |
| --- | --- | ---: |
| Fused adjacent-pair | Explicit | `0.988046` |
| Manual canonical | Explicit | `0.986243` |
| Fused adjacent-pair | Default | `0.991249` |
| Manual canonical | Default | `0.999963` |

Two fresh independent hypothesis agents reviewed the evidence. H1 supported a
two-boundary numerical sensitivity without overclaiming a primitive kernel
bug. H2 refuted reduced precision, KV dtype, cache update/fill, and page
routing as primary causes: BF16/BFP8 KV and identity/permuted pages were
effectively identical, and fused/nonfused updates gave the same paired PCC.

The minimal proven policy repair is manual canonical RoPE plus default SDPA;
fused paged K/V update remains enabled. Final BFP4/LoFi+BFP8-cache PCC is
`0.999264/0.998993` at b1/b32, and optimized-prefill-produced prefix caches
pass `0.998985/0.999005`. This repair is retained despite the material latency
rollback because the faster fused/explicit path is not semantically correct.

## Final verification

- Full optimized suite: 16 passed on current source.
- Final recorded-activation performance suite: 4 passed.
- Optimized-prefill-produced semantic-cache decode: 2 passed.
- `TT_METAL_WATCHER=10` representative optimized suite: 5 passed, clean.
- Final four-window Tracy capture completed.
- No repair was retained without an isolated result and cumulative full-path
  correctness/performance evidence.
