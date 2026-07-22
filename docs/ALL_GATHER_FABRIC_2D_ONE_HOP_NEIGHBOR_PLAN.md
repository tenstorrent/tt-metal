# Fabric2D one-hop-neighbor all-gather plan

## Objective

Extend the native `ttnn.all_gather` store-and-forward backend so a one-dimensional
line or ring embedded in `FABRIC_2D` sends every Ethernet packet only to the
immediate physical neighbor. A shard reaches a distant rank through repeated
Tensix-controlled receive-and-relay steps, not through one multi-hop Fabric
multicast packet.

The first product target is the eight-rank sparse-MLA ring on the current
Blackhole system. The implementation must remain generic: selection is based on
tensor, topology, resource, and physical-adjacency facts, with no model-name,
exact-shape, environment-variable, or user-facing tuning gate.

The target is to retain the approximately `92 GB/s` effective receive bandwidth
already demonstrated by the native store-and-forward algorithm under
`FABRIC_1D_RING`, while running the same logical ring on `FABRIC_2D`.

## Goals to chase

Treat these as ordered, independently reviewable goals. Do not trade away an
earlier correctness or stability goal to claim a later bandwidth result.

| Goal | Outcome | Measurable exit condition |
| --- | --- | --- |
| G0: Reproducible known-good base | Everyone starts from the same Fabric2D-correct operation and firmware | Release build passes; matched eight-case matrix passes; exact commits and command are recorded |
| G1: Prove one-hop eligibility | The host can identify a direct physical line/ring without a model or shape special case | Every selected logical edge, including ring wrap, resolves to one physical Ethernet hop; non-direct fixtures fall back |
| G2: Functional one-hop Fabric2D backend | The existing native store-and-forward schedule runs through Fabric2D neighbor connections | Exact two-, four-, and eight-rank correctness passes for BF16 and scaled FP8, fresh and persistent outputs |
| G3: Production stability | Cached repeated execution has no protocol race, stale semaphore, corruption, or hang | Ten consecutive full `512K` runs per format pass through `run_safe_pytest.sh`; p90 is no more than 5% above median |
| G4: Useful bandwidth | The new path decisively beats current Fabric2D multicast | Both formats reach at least `70 GB/s` effective receive bandwidth |
| G5: Fabric1D parity | Fabric2D no longer imposes a large transport tax on the same logical ring | Both formats reach at least `85 GB/s` and are within 10% of their matched Fabric1D controls |
| G6: Stretch bandwidth | Recover nearly all measured native-ring capability | Both formats reach at least `90 GB/s`; Fabric1D remains above `90 GB/s` |
| G7: Sparse-MLA product gain | Isolated bandwidth translates into the intended workload | MLA block correctness passes and DSA/GLM warm, cold, and long measurements improve with no scenario regression |
| G8: Generic qualification | The implementation is maintainable beyond this box and workload | No environment knobs, model gates, or all-gather-specific router mode; QuietBox and concurrent Galaxy SP-ring coverage passes |

G0 through G3 are merge blockers. G4 is the minimum performance justification
for selecting the new backend. G5 is the main project target; G6 is the stretch
target. G7 and G8 are required before calling the work product-ready.

## Starting point

The clean starting revision is `6c5573df7a1`. It builds with
`./build_metal.sh --release` and passes all eight matched BF16/scaled-FP8,
Fabric1D/Fabric2D correctness and performance cases through
`scripts/run_safe_pytest.sh`.

The matched `65,536`-row-per-rank baseline measured on 2026-07-22 is:

| Format | Fabric | Median | Effective receive BW | Current algorithm |
| --- | --- | ---: | ---: | --- |
| BF16, `1,152 B` rows | `FABRIC_1D_RING` | 5.717 ms | 92.434 GB/s | one-hop store-and-forward |
| scaled FP8, `704 B` rows | `FABRIC_1D_RING` | 3.516 ms | 91.846 GB/s | one-hop store-and-forward |
| BF16, `1,152 B` rows | `FABRIC_2D` | 26.882 ms | 19.659 GB/s | multi-hop explicit-path multicast |
| scaled FP8, `704 B` rows | `FABRIC_2D` | 16.366 ms | 19.733 GB/s | multi-hop explicit-path multicast |

`effective_receive_bw` is aligned tensor bytes received by one rank from the
other seven ranks divided by the all-gather device-program duration. It is not
aggregate traffic summed over every link or device.

Uncommitted shared-pool and Fabric-router experiments are preserved separately
in the named stash `wip-fabric2d-shared-pool-debug-2026-07-22`. They are not
part of this baseline and must not be reapplied wholesale. A change may be
recovered from that work only as a small, independently tested commit after the
native one-hop path is working.

### Known-good Fabric2D prerequisite stack

The baseline depends on three ordered correctness changes that currently exist
only in the local history of `pjosipovic/sparse-mla-new-all-gather`:

1. `1cedc496e76726fe8faa1e04b9a56bbf8aaf196c` initializes multicast scatter
   payload headers. Without it, uninitialized header state can produce hangs or
   corruption even in an existing Fabric2D shape.
2. `f64216cbc9519f040ad9e5575b5909e6f4f2bd8c` maps logical collective direction
   to the real physical Fabric route. This is required by the existing 4x2
   Fabric2D path.
3. `be9b318dd40e25760b30c845e7bdd4e6306a6097` adds a logical 8x1 ring on
   Fabric2D using explicit per-hop multicast paths, dateline escape VC, corner
   forwarding, and the receiver-L1 initialization barrier.

Therefore, an independent branch that times out in the existing 4x2 test is
not merely missing the 8x1 enablement in `be9b318`; it must first verify that
`1cedc496` and `f64216c` are present. If its base already contains the same
native operation and receiver-L1 structures, port the three fixes in the order
above. Otherwise, use `6c5573df7a1` as the handoff baseline instead of trying
to transplant `be9b318` alone across a structurally different operation. The
large native unicast path must additionally retain
`8d8f45075fb8c5de1084ba28ee0a3793ad7e634e`, which prevents 32-bit worker
partition overflow at the target tensor size.

As of 2026-07-22, the remote tracking branch stops at `f26ee0b1e93`, before
`be9b318`. Another clone therefore cannot fetch the short SHA from `origin`.
Cross-checkout work requires publishing the known-good branch or exporting the
ordered patch series; a locally resolvable short SHA is not a sufficient
handoff.

`be9b318` is a known-good multi-hop Fabric2D baseline, not the proposed final
one-hop design. Its route, dateline, and corner behavior remains the fallback
for non-direct collectives while the new neighbor backend is developed.

## Why this direction is promising

The current Fabric2D path sends one multicast packet over three or four hops.
Some routes turn in the physical topology and some cross the ring dateline. A
packet that crosses the dateline and continues must enter the escape VC and
remain there. The retained implementation is correct, but the complete
Fabric2D forwarding and VC-credit pipeline limits both formats to approximately
`20 GB/s`.

The Fabric1D native unicast backend already implements the desired collective
algorithm and sustains more than `91 GB/s` on the exact pages and tensor depth:

1. A Tensix reader loads its local stripe into a circular buffer.
2. A Tensix writer sends it one Ethernet hop into the neighbor's final output
   address and signals that the stripe is available.
3. The neighbor reader loads that received stripe from its output and sends it
   one hop farther.
4. Both ring directions operate concurrently; on an even ring the antipode
   stripe is split between the two directions.

The proposed Fabric2D flow is:

```text
rank i input/output DRAM
          |
          | Tensix reader
          v
      source CB ---- Tensix writer ---- one-hop Ethernet packet ----+
                                                                  |
                                                                  v
                                                        rank i+1 output DRAM
                                                                  |
                                               data-valid semaphore
                                                                  |
                                                                  v
                                                   next relay iteration
```

Every Ethernet packet terminates after one physical hop. If the logical ring
turns at a rank, the packet is first completed locally and the next Tensix
writer creates a new packet in the new physical direction. Fabric therefore
does not perform a corner transit for this collective.

Likewise, a packet crossing the logical dateline terminates at the adjacent
rank. It does not retain a Fabric receive or egress slot while acquiring a
second channel, so the application-level store-and-forward protocol does not
create the multi-hop VC dependency that requires the current escape transition.
This property must be verified in the route fixture and must not be assumed
from logical rank adjacency alone.

## What must be preserved from the current native operation

This is a new transport backend for the existing operation, not a return to
the composite `all_broadcast + concat` implementation. Preserve:

1. One public `ttnn.all_gather` call and one dispatched mesh workload. Relay
   iterations run inside that workload; they must not become seven host-visible
   collective dispatches.
2. Native row-major input and output, including aligned packed FP8 rows.
3. Direct placement into the final gathered output without a post-collective
   concat.
4. Persistent/preallocated output support and program-cache reuse.
5. Global control semaphores in `L1_SMALL` when that allocator region exists,
   with the existing safe L1 fallback.
6. Current page packetization: coalesce several small pages into one Fabric
   packet and split pages larger than a packet without changing tensor layout.
7. Multiple workers per direction, Fabric mux sharing when needed, two Fabric
   links, independent forward/backward streams, and the even-ring antipode
   split.
8. Generic topology handling, sub-device and sub-core-grid support, and a safe
   fallback for unsupported geometry.
9. The existing correctness fixes:
   - initialized multicast scatter payload state (`1cedc496e76`);
   - physical rather than assumed logical direction mapping (`f64216cbc95`);
   - explicit Fabric2D ring paths and dateline escape handling for the multicast
     fallback (`be9b318dd40`).
10. Release-firmware compatibility. Do not use `--dev`; it enables additional
    watcher/assert code and increases firmware size.

The receiver-L1 bank fan-out and terminal-offload mechanisms are currently
part of the multicast backend. They cannot be claimed as reused by the first
one-hop implementation because store-and-forward initially terminates directly
in output DRAM. Their useful principles—persistent destination ownership,
batched notifications, bank-aware draining, and bounded source lifetime—remain
available for a later measured L1-relay optimization.

## Eligibility and automatic selection

The Fabric2D neighbor backend is eligible only when all of the following are
proven by the operation builder:

- exactly one mesh axis participates in the collective;
- the active topology is a line or ring;
- every forward/backward logical neighbor used by the algorithm is a direct
  physical Fabric neighbor;
- the configured number of links exists for every required adjacency;
- the input/output layout and gather dimension are supported by the existing
  native unicast iterators;
- required worker, mux, CB, semaphore, and output-buffer resources fit;
- no unsupported batch-slice or partial-extent behavior is requested.

For a ring, adjacency validation includes the wrap pair. If any adjacency is
not one physical hop, fall back to the correct multicast backend. Do not
silently send a nominally "one-hop" packet through an intermediate router.

Initial selection should mirror the established Fabric1D size heuristic only
after correctness qualification. Until then, a test-owned internal factory
entry may exercise the backend, but no environment variable or public API knob
is added. Once qualified, use structural and workload-size facts to select it
automatically; keep multicast for small latency-bound messages if it wins.

## Implementation design

### 1. Separate collective scheduling from Fabric transport

Keep the current `AllGatherUnicastFactory` stripe scheduling, iterators,
forward/backward iteration counts, antipode split, CB protocol, and data-valid
semaphores. Isolate only its connection/header construction behind a neighbor
transport abstraction:

- Fabric1D implementation: retain the current linear API and `num_hops = 1`.
- Fabric2D implementation: open a routing-plane connection to the resolved
  immediate `FabricNodeId` and construct a terminal one-hop write header.

The reader should remain Fabric-agnostic. The writer should continue to own all
Fabric traffic and buffer-lifetime completion.

### 2. Resolve physical neighbors on the host

For each rank and direction:

1. Resolve the logical neighbor with the active line/ring topology.
2. Convert source and destination coordinates to `FabricNodeId` values.
3. Require `are_direct_fabric_neighbors(source, destination)`.
4. Record the actual Ethernet direction and link IDs.
5. Append the correct Fabric2D connection-manager runtime arguments.

The path is rank-specific because consecutive logical ring edges on this box
can use different physical directions. Program caching must hash every
attribute that changes generated routing or resource layout.

### 3. Preserve one-hop packet semantics

The Fabric2D writer must encode a terminal operation to the immediate neighbor,
not an explicit multicast range of length one that can accidentally inherit
multi-hop ring state. Data and its data-valid notification must obey the same
ordering as Fabric1D.

Start with the existing conservative completion behavior. Do not defer source
CB reuse, combine data-valid epochs, or alter semaphore granularity while
bringing up routing. Optimize those only after the matched implementation is
correct and stable.

### 4. Keep Fabric firmware changes out of the first prototype

The first implementation should use the existing generic Fabric2D worker
connection and terminal NOC execution APIs. It should not require channel
trimming, shared sender pools, new escape-VC allocation, or all-gather-specific
router firmware.

If the existing connection API cannot express a guaranteed direct-neighbor
terminal packet, add the smallest generic Fabric API needed to express that
property. The API must be usable by other collectives and must not mention
sparse MLA.

## TDD execution plan

### Phase 0: freeze the reproducible baseline

Already complete at `6c5573df7a1`:

- clean release build;
- eight matched correctness/performance cases pass;
- BF16 and scaled-FP8 baselines recorded for both Fabric modes;
- failed shared-pool work preserved outside the active tree.

Repeat the baseline before merging any old experimental change.

### Phase 1: host route-plan tests

Add pure host/unit coverage before changing a device kernel:

1. Two-rank Fabric2D line: one neighbor in each live direction.
2. Four- and eight-rank lines: endpoints have one live direction.
3. Eight-rank direct physical cycle: every ring edge, including wrap, resolves
   to one physical hop.
4. A synthetic/non-direct wrap: neighbor backend is rejected and multicast is
   selected.
5. A physical cycle containing turns: each individual edge is accepted, but no
   generated packet path contains more than one hop.
6. Program-hash inequality when Fabric configuration, topology, axis, link
   count, or neighbor plan changes.

These tests should expose the selected factory and resolved route plan without
requiring an environment variable.

### Phase 2: minimal Fabric2D neighbor transport

Port the existing unicast writer connection/header layer while leaving the
collective protocol unchanged.

Bring-up order:

1. Two ranks, one link, one page, BF16, persistent output.
2. Two ranks, two links, enough pages to use both links.
3. Four-rank line in each physical orientation.
4. Eight-rank ring with small pages and exact output checking.
5. Repeat the eight-rank ring enough times to exercise program-cache reuse and
   semaphore reset.

At every step assert through profiler kernel sources that the native unicast
factory ran and multicast receiver kernels did not run.

### Phase 3: protocol and shape correctness ladder

Expand one dimension at a time:

- fresh output and persistent output;
- BF16 `1,152 B` and scaled-FP8 `704 B` pages;
- page counts below, equal to, and above one packet;
- worker partitions with remainders;
- odd and even page counts;
- line endpoints and ring antipode split;
- one, two, and multiple workers per direction;
- direct worker-to-Fabric and muxed workers;
- repeated invocations exceeding the data-valid sequence/reset window;
- supported gather dimensions and aligned packed-row shapes.

Run all hardware cases with `scripts/run_safe_pytest.sh`. On any timeout, let
the wrapper capture `tt-triage` before reset. Classify the first blocked stage:
reader data-valid wait, writer CB wait, Fabric sender credit, remote NOC
completion, or final aggregate completion. Do not add delays or skip a route to
make a hang disappear.

### Phase 4: exact 512K workload A/B

Run the same `65,536` rows per rank, two links, `14,336 B` payload, persistent
interleaved DRAM output, and release firmware under:

1. Fabric1D one-hop store-and-forward control.
2. Fabric2D multi-hop multicast baseline.
3. Fabric2D one-hop store-and-forward candidate.

Collect seven steady-state samples after compile and warmup. Report median,
minimum, p90, all samples, latency, effective receive bandwidth, page size,
packet payload, worker count, mux mode, and selected backend.

The first Fabric2D target is correctness and stable completion. Performance
gates are:

| Gate | Required effective receive BW, both formats |
| --- | ---: |
| Useful-path gate | at least 70 GB/s |
| Parity gate | at least 85 GB/s and within 10% of matched Fabric1D |
| Stretch gate | at least 90 GB/s |

Each gate also requires exact BF16 output, the existing bounded FP8 comparison,
ten consecutive non-hanging full-size runs, and p90 no more than 5% above the
median.

### Phase 5: tune existing native mechanisms

Tune one mechanism at a time, retaining a change only for at least a 3% median
gain with no correctness or p90 regression.

1. **Worker count:** sweep one, two, and four workers per direction per link.
   Keep the decision generic and based on transaction bytes and bytes per link.
2. **Packet packing:** compare the current maximum-payload packing against
   page-aligned payload sizes. BF16 currently packs twelve rows (`13,824 B`)
   and FP8 packs twenty rows (`14,080 B`) under the `14,336 B` ceiling.
3. **Mux cost:** compare direct worker connections with shared mux connections
   at the worker count where muxing becomes necessary.
4. **CB depth:** increase source buffering only if reader or writer attribution
   shows a producer/consumer bubble.
5. **Completion window:** defer Fabric completion only while all referenced CB
   storage remains immutable, and always flush before buffer reuse, semaphore
   publication, final completion, or connection close.
6. **NoC and DRAM scheduling:** verify balanced bank traffic and actual terminal
   write completion before changing NoC selection. Do not introduce a
   dtype-specific public knob.

Re-run Fabric1D after every retained transport change. The new Fabric2D support
must not regress the existing `>91 GB/s` path.

### Phase 6: optional L1 relay if DRAM reread is proven limiting

The initial algorithm writes a received stripe to final output DRAM and the
next relay iteration reads it back. Fabric1D measurements show this can sustain
the target bandwidth, so do not redesign it preemptively.

If attribution on Fabric2D proves the DRAM reread is the remaining limit,
prototype a bounded L1 relay:

1. Terminate a one-hop packet in a receiver-owned L1 slot.
2. Have a Tensix worker write the slot to final output DRAM and forward the same
   immutable payload to the next physical neighbor.
3. Return slot credit only after both the local DRAM write and forwarded Fabric
   read have completed.
4. Batch data-valid notification and use bank-owned drain scheduling where it
   measurably helps.

This reuses the strongest ideas from the current receiver-L1 path without
reintroducing multi-hop Fabric forwarding. It requires separate proof for slot
ownership, wraparound, antipode halves, partial final packets, and completion
ordering.

### Phase 7: product and topology qualification

After isolated parity:

1. Run sparse-MLA block correctness tests.
2. Re-measure sparse-MLA warm, cold, and long scenarios for DSA and GLM.
3. Verify the full model still uses persistent output buffers and `L1_SMALL`
   control allocation.
4. Exercise a QuietBox-sized topology and an eight-rank Galaxy-compatible
   ring.
5. On Galaxy, run all four concurrent SP=8 rings used by SP=8/TP=4, not only
   one isolated ring.
6. Retain multicast fallback coverage for non-direct rings, two-dimensional
   collectives, batch slices, partial extents, and small-message winners.

## Measurement commands

Build release and install matching host libraries:

```bash
./build_metal.sh --release
```

Run the matched full-size baseline/candidate matrix:

```bash
TT_METAL_DISABLE_PRECOMPILED_FW=1 \
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/ccl/test_all_gather_fabric_2d.py \
  -k 'matched_large_single_axis_correctness or matched_sparse_mla_row_perf' \
  -x -s -v
```

Never add `--dev`. If a run hangs, preserve the safe-wrapper and `tt-triage`
output before changing code or resetting manually.

## Commit strategy

Keep the work bisectable:

1. Host route-plan and eligibility tests.
2. Fabric2D one-hop connection/header support with minimal correctness tests.
3. Full native unicast protocol coverage and automatic factory selection.
4. Exact BF16/FP8 performance fixture and recorded baseline.
5. One standalone commit per retained tuning mechanism.
6. Model and multi-ring qualification updates.

Do not combine generic Fabric firmware experiments with the first operation
port. Do not commit temporary profiling counters, enlarged firmware areas,
environment-variable selectors, delays, or workload-specific gates.

## Completion criteria

The one-hop Fabric2D project is complete when:

- `ttnn.all_gather` automatically selects the neighbor backend for eligible
  one-dimensional direct physical lines/rings under Fabric2D;
- every emitted payload packet is proven to traverse exactly one physical
  Ethernet hop;
- BF16 and scaled-FP8 full-size outputs are correct for fresh and persistent
  buffers and remain correct across repeated cached execution;
- both formats reach at least `85 GB/s` effective receive bandwidth on the
  matched eight-rank case, with a stretch goal above `90 GB/s`;
- Fabric1D stays above `90 GB/s` on the matched control;
- no environment knobs, model gates, or all-gather-specific Fabric router mode
  are required;
- sparse-MLA block and end-to-end performance tests pass; and
- QuietBox and Galaxy-compatible topology coverage, including concurrent
  Galaxy rings, shows no hang, corruption, or material regression.
