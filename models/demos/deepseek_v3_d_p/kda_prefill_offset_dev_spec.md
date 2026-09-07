# KDA prefill offset — dev spec for two production-quality prototypes

Implements: [`kda_prefill_offset_design.md`](kda_prefill_offset_design.md)
Base: PR #52799 head `71c0826018f`. No push; local branches only.

## Notation

| Symbol | Meaning | K3 Galaxy value |
| --- | --- | --- |
| `P` | SP size | 8 |
| `C` | local rows per chip | 640 |
| `G` | `P*C`, global chunk | 5120 |
| `S` | `actual_start` | 960 (worst case) |
| `b` | boundary chip `(S//C) mod P` | 1 |
| `o` | tail length `S mod C` | 320 |
| `h` | head length `C-o` | 320 |

**Existing** — verified by running MLA's own oracle `tt/mla/utils.py:83-111`
(mirrors `writer_update_padded_kv_cache.cpp:100-122`):

| chip | rows | relative token `i` | role |
| --- | --- | --- | --- |
| `b` | `0:h` | `0 .. h-1` | head — chronologically **first** |
| `b` | `h:C` | `G-o .. G-1` | tail — chronologically **last** |
| `c != b` | `0:C` | `m*C-o .. m*C-o+C-1`, `m=(c-b) mod P` | contiguous |

Every chip therefore holds *top-`o` of chunk `k-1`* plus *bottom-`h` of chunk `k`*.
Only chip `b` holds them in the opposite row order — that is the wrap.

## Goal

Two independently reviewable, perf-quality implementations of offset-correct KDA
prefill over MLA's block-cyclic layout, plus a measured comparison against the
`S=0` baseline on PR7.

Out of scope (inherited from the design's non-goals): partial/zero-padded chunks,
non-tile-aligned offsets, producer-path integration, decode.

## Shared foundation (both prototypes)

**Proposed** — one branch `mvasilijevic/kda-offset-base`, both prototypes branch from it.

1. **Topology derivation.** Pure function `offset_topology(S, P, C) -> (b, o, h)`
   plus the chronological chip order `[b, b+1, ..., b-1]`. Host-side, no tensors.
2. **Rotated carry ordering.** Both stateful exchanges currently compose in
   physical rank order and must compose in chronological order instead:
   - `tt/kda/convolution.py:50-73` — `entry_carries` list built as
     `[initial] + [tail(rank-1)]`, concatenated, then `mesh_partition`.
     Rotation = index the list by chronological predecessor. ~5 lines.
   - `tt/kda/recurrence.py:250-286` — `for rank in range(sp_size)` composing the
     gathered affine summaries. Rotation = iterate the rotated order and store
     `entry_states[rank]` by physical rank instead of appending. ~5 lines.
3. **`actual_start` on the forward boundary.** `ttKDA.forward(hidden_states, state,
   actual_start=0)`; validated in `_validate_forward` (`kda.py:189-216`) for
   `S >= 0` and `S % 32 == 0`.

This alone makes every `o == 0` offset correct (design "Alternative C"), and it is
the correctness floor both prototypes build on.

## Prototype A — one-hop ring exchange (`mvasilijevic/kda-offset-ring`)

Realizes the design's Alternative B, with the all-gather replaced by a neighbour shift.

### Mechanism

Each chip ships its *top-`o`* piece one hop **backward** (to rank `c-1`), or its
*bottom-`h`* piece one hop **forward** — whichever is smaller, `min(o, h)`.

| | rows sent | rows kept | resulting chunk on chip `c` |
| --- | --- | --- | --- |
| backward, `c != b` | `0:o` | `o:C` | `A_{(c-b) mod P}` |
| backward, `c == b` | `h:C` | `0:h` | `A_0` |
| forward, `c != b` | `o:C` | `0:o` | `A_{(c-b-1) mod P}` |
| forward, `c == b` | `0:h` | `h:C` | `A_{P-1}` |

After assembly every chip holds `[kept | received]` = one 640-aligned logical chunk,
uniformly. Stock KDA then runs with the shared rotated carry order. The inverse
shift restores MLA placement.

**Proposed exchange point — the layer boundary ("outer"), not post-projection.**
Only the causal convolution and the recurrence are order-sensitive; projection,
gates, RMSNorm and output projection are all token-local. So the shift could
bracket either the whole layer or just the conv+scan window. Measured by width,
the layer boundary is both cheaper and simpler:

| shift point | forward width/device | reverse width/device | total |
| --- | --- | --- | --- |
| **outer** — `hidden_states` in, final output out | 7168 (TP-replicated) | 1792 (`hidden_size/TP`, reduce-scattered) | **8960** |
| inner — `concat(qkv, decay_rank, beta)` in, scan output out | 9216 + ~152 | 3072 (`v_dim/TP`) | ~12440 |

`_convolution_width = q_dim + k_dim + v_dim` (`kda.py:165-166`) is
`3 * 96 * 128 = 36864` for K3 (`kimi_k3_config.py:97-98`) — 5.1x `hidden_size`, so
even sharded 4 ways it is wider than the replicated 7168. `_project_output`
reduce-scatters, making the final output the narrowest tensor in the layer.

Outer also keeps `output_gate` produced and consumed inside the rotated region, so
it needs no special handling, and confines the change to two calls at the forward
boundary instead of threading rotation through the middle of `forward`.

### ⚠️ Primitive gap — the decision that sets Prototype A's value

**Existing:** there is no ring-shift collective. The CCL op set is `all_broadcast,
all_gather, all_reduce, all_to_all_{combine,dispatch}, broadcast, mesh_partition,
reduce_scatter, reduce_to_root`. `ttnn.point_to_point(input, sender_coord,
receiver_coord, topology)`
(`ttnn/cpp/ttnn/operations/point_to_point/point_to_point_nanobind.cpp:20-79`) moves
**one** device shard to **one** device, so a mesh-wide shift needs `P*TP = 32` calls.

**Existing:** the codebase's real 1-hop pattern is `ring_joint_sdpa`, which *fuses*
neighbour transport into the compute op via
`ring_attention_neighbor_halo_exchange_helper`
(`.../ring_attention_all_gather_async/device/..._program_factory.hpp:172-190`,
used at `ring_joint_sdpa_program_factory.cpp:612-667, 2853`). It is a
`ProgramDescriptor` helper, not a standalone ttnn op — taking
`unicast_destination_coord` + fabric semaphores — and it carries a
`RingAttentionRankMapping` abstraction directly analogous to our rotation.
It has Galaxy coverage (`tests/nightly/tg/ccl/test_ring_joint_attention.py`).

**Proposed — an escalation ladder behind one seam `exchange_offset_rows(..., backend)`.**
Each rung answers one question and is deleted once answered:

| rung | mechanism | ops per shift | bytes/device | vs full reshard at `S=960` | cost to build |
| --- | --- | --- | --- | --- | --- |
| **A1** | `all_gather(o-row slice)` + `mesh_partition`, reusing the `exchange_convolution_carry` pattern (`convolution.py:43-73`) | 1 collective | `P*o*W` | **2x** | Python, hours |
| **A2** | `ttnn.point_to_point` ring | 32 unicasts | `o*W` | **16x** | Python, hours |
| **A3** | new neighbour-shift op reusing `ring_attention_neighbor_halo_exchange_helper` | 1 op | `o*W` | **16x** | C++ op + `./build_metal.sh` |

The 14x napkin estimate holds only at A2/A3. A1 recovers only 2x at `o = C/2`
(it improves as `C/o`, so 20x at `o=32`) but is the low-risk correctness vehicle.

**Sequencing decision:** A1 first for correctness bring-up, then A2 to measure
whether the byte win survives 32 launches. Escalate to A3 **only if** A2 shows the
bandwidth win is real but launch overhead eats it — that is the one condition under
which the C++ cost is justified.

## Prototype B — split scan, no data movement (`mvasilijevic/kda-offset-scan`)

Realizes the design's preferred "segment-aware causal execution", in the **uniform**
form: *every* chip splits at row `h`, not just the boundary chip.

**Uniformity is forced, not chosen.** `_scan_grouped_chunks` derives
`group_chunks`/`groups_per_head` from Python ints shared by the whole mesh
(`recurrence.py:326-329`), and `reduce_affine_transforms` /
`affine_exclusive_scan` take `groups_per_head` as a scalar
(`recurrence.py:341-347, 357-364`). A ragged split — 2 fragments on chip `b`, 1
elsewhere — is not expressible. Splitting all chips at `h` keeps every shape uniform.

### Fragment order

`2P` fragments; chronological order is
`b.L, (b+1).L, (b+1).R, ..., (b-1).L, (b-1).R, b.R`.
For `c != b`, `L` and `R` are logically adjacent, so composing them in sequence is
identical to today's single per-chip summary. Only chip `b` has non-adjacent halves.

### Stage-by-stage change

| stage | code | change |
| --- | --- | --- |
| chunk prep | `_prepare_chunk_terms` `recurrence.py:115-143` | **none** — token-local |
| group summaries | `_summarize_chunk_groups` `recurrence.py:172-192` | 2 calls, over `h/32` and `o/32` chunks |
| head summaries | `reduce_affine_transforms` `recurrence.py:341-347` | 2 calls, `groups_per_fragment` each |
| partition exchange | `_distributed_affine_prefix` `recurrence.py:217-296` | see below |
| exclusive prefix | `affine_exclusive_scan` `recurrence.py:357-364` | 2 calls, one per fragment |
| scan | `_scan_chunks` `recurrence.py:202-215` | 2 calls |

**Group granularity is a non-issue for K3.** `summary_group_chunks=20`
(`config.py:73`) is a *ceiling* and `_effective_summary_group_chunks`
(`recurrence.py:194-200`) takes the largest divisor at or below it, so each
fragment (`h/32` and `o/32` chunks, `<= 20`) becomes exactly **one** group for any
32-aligned `o`. The only requirement really is `S % 32 == 0`.
Note `h/32 != o/32` in general (`o=32` gives 19 and 1), so the two fragments need
separate `_reshape_chunks_for_groups` calls, not one batched reshape.

### Partition-summary exchange — two variants

**B1 (first cut):** gather both fragment summaries, `2P` payload, compose `2P-1`
steps in chronological order. Directly yields every entry state and the final state.

**B2 (optimisation, if the prefix shows in the profile):** keep the exchange at `P`
— identical bytes and step count to baseline. Each chip composes `L∘R` locally,
with a per-device constant mask forcing chip `b`'s `R` to the affine identity so it
contributes `L` only. Then `R_entry = select(is_b, full_prefix, L_entry ∘ L_summary)`
using the same per-device mask. Adds two small local ops off the critical path.

### Convolution

`qkv_causal_conv1d_silu` (`kda.py:252-261`) convolves the whole local buffer with one
entry carry, so the split needs 2 calls (`h` rows, `o` rows) — which also delivers
the design's Invariant 3 (head and tail never convolved as adjacent rows) for free.

Carry routing, from `exchange_convolution_carry` (`convolution.py:30-48`) extended to
gather **two** tile slots per rank (end-of-`L`, end-of-`R` — 32 rows each, negligible):

| fragment | entry carry |
| --- | --- |
| `b.L` | caller `initial_carry` |
| `c.L`, `c != b` | slot B of chip `c-1` |
| `c.R`, any `c` | slot A of chip `c` (local) |
| `b.R` | slot B of chip `b-1` |
| replacement state | slot B of chip `b` |

## Cost model (what the measurement should confirm)

| | extra bytes moved | extra op launches |
| --- | --- | --- |
| A | 2 shifts of `min(o,h)` rows | A1: 2 collectives / A2: 64 unicasts / A3: 2 ops |
| B | 0 activation bytes; B1 doubles the summary all-gather, B2 doesn't | conv x2, summarize x2, reduce x2, prefix x2, scan x2 |

A trades a small, bandwidth-bound exchange for an untouched scan. B trades zero
bandwidth for roughly doubled launch count on five stages. The design doc's open
question ("Unknown: whether one segment-aware collective or two phase-specific
collectives gives the best Galaxy latency") is exactly what this resolves.

## Implementation sequence

1. Shared base: topology function + rotated carry order + `actual_start` plumbing.
   Gate: host topology test, and `o == 0` offsets correct on device.
2. Prototype A on top: `exchange_offset_rows` (A1), forward/reverse shift wiring.
   Gate: correctness at `o != 0`.
3. Prototype A2 backend behind the same seam. Gate: parity with A1.
   A3 only if A2's measurement justifies the C++ cost.
4. Prototype B on top of the base: fragment split through the six stages,
   two-slot conv carry, B1 exchange. Gate: correctness at `o != 0`.
5. Prototype B2 exchange variant. Gate: parity with B1.
6. Measurement and report.

## Validation

Inherited from the design's acceptance criteria:

- Host topology test over all 160 tile-aligned starts mod 5120, all 8 boundary chips.
- Device correctness vs. natural-order reference for `o == 0` set
  `{0,640,...,4480}`; `o = 32` (smallest), `320` (worst case), `608` (largest);
  and at least one nonzero `o` on every chip. Output **and both states** compared,
  not just output.
- Production K3 SP8xTP4 `T=5120`, `S=960` real-weight PCC `>= 0.9995`
  (`tests/kda/perf/test_layer_perf.py:44-48`).
- Segmented continuation: replacement state feeds the next call and matches
  one-shot natural execution.
- **Determinism:** the same input run N times produces bit-identical output and
  state, per prototype, per offset.
- Trace replay across two boundary chips and two nonzero splits without stale
  offset capture or program-cache corruption.
- Perf: warm trace wall time + per-op breakdown at `S in {0, 32, 320, 608, 960}`,
  both prototypes, both backend variants, against the `S=0` PR7 baseline.

## Risks and unknowns

- **Unknown:** whether A2's 32 unicasts amortise inside a captured trace. If not,
  Prototype A needs A3 (C++) to realise its bandwidth advantage, and the A-vs-B
  comparison at `o = C/2` is decided by A1's 2x rather than the napkin 14x.
- **Risk:** B's doubled launch count on five stages may exceed A's exchange cost
  at `o = C/2` while winning at small `o` — the crossover is the real result.
- **Risk (Existing, from memory):** `all_to_all_async_generic` hangs on Blackhole;
  not used here, but it rules out that primitive as an A backend.
- **Unknown:** program-cache key growth. Both prototypes make shapes depend on
  `(h, o)`; that is a bounded set of 20 pairs, but trace capture must be verified.
- **Assumed:** BF16 summary transport (`config.py:17`) tolerates B1's extra
  composition step without falling below the 0.9995 PCC gate.
