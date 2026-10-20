# Fabric Test Streams: design plan

Status: draft, 2026-10-15. Based on main `453ee7bdfa7`. Scope: `tests/tt_metal/tt_fabric/test_infra`.

This plan adds a `stream` field to a traffic pattern. A stream is an ordered list of segments, and each segment has its own command type, payload size and packet count. The stream repeats for a set number of laps. A pattern without a stream counts as a one-segment stream, so this extends the existing spec rather than adding a separate mode.

Line numbers below refer to main. The test infra files cited are unmodified in the experiment tree.

**Revision 2 (after review).** All five review points were checked against the code and adopted.

- **No adapter change.** The benchmark path uses public functions the adapter already has, so the PR stays entirely inside `test_infra`.
- **Hop sentinels in segments.** `resolve_hop_sentinels` now also resolves hops inside stream segments.
- **Summary row.** It takes its bytes from the link with the most bytes.
- **CSV labels.** ntype and ftype report `stream` only when that field actually differs across segments.
- **Mux connections.** They are rejected only on the benchmark path. Validation mode already supports them.
- **Two senders on one router.** This is now rejected only in benchmark mode. In validation mode the infra puts a mux in front of a router that has more than one client.

**Revision 3.** A stream is limited to one router. Streams across several routers were dropped because they add complexity without much benefit; several routers are loaded with one sender each. This removed the two-router example, the re-arm/stateless choice in the kernel, the multi-direction multicast split, the `split_senders_by_direction_for_benchmark` change, gate run W4, and the three-router validation case.

**Revision 4.** `ftype`, `destination` and `mcast_start_hops` apply to the whole stream, so a stream is all unicast or all multicast and segments carry no destinations. Mixing unicast and multicast in one stream is dropped, along with the unicast + multicast mix. Hop resolution needs no change, since hops live only on the pattern. Segments are no longer deduped: each list entry is its own pattern with its own receiver buffer, which keeps W2 and W3 possible without per-segment destinations.

**Revision 5.** v1 keeps the separate-configs design: each segment is its own pattern, and the sender walks an order program over them. One config per flow, where the stream becomes a property of the flow, is the route to all-to-all style patterns. It needs a new segment-list representation across the host, so it moves to [Later](#later-v2), where its costs are spelled out. The YAML is the same under both designs, so v1 tests won't need rewriting.

**Revision 6.** Decisions recorded:

- `size` is inherited.
- `repeat` defaults to one lap, and `num_packets` doesn't set a stream's packet count.
- Stream rows go into the existing golden file; none are needed before mainlining.
- Work happens on a branch from main.
- An explicit `num_packets` on a stream pattern is a build error.

## Contents

1. [Why](#why)
2. [YAML spec](#yaml-spec)
3. [Host build](#host-build)
4. [Sender kernel](#sender-kernel)
5. [Bandwidth accounting](#bandwidth-accounting)
6. [Worker-cost gate](#worker-cost-gate)
7. [Validation](#validation)
8. [First mixes](#first-mixes)
9. [Phases](#phases)
10. [Risks](#risks)
11. [Later (v2)](#later-v2)
12. [Decisions](#decisions)
13. [Files](#files)

## Why

- **Uniform runs reward narrow wins.** Today each benchmark row sends one command type at one size. The Blackhole router study shows the cost:
  - Bit 22 raised 1-hop writes, but the receive ERISC then capped scatter.
  - Bit 23 sped up writes but added 12–20 ns per hop to 16 B latency.

  A mixed run shows that kind of trade-off in a single number.
- **Teams can describe their own traffic.** A collective can write down its chunk shape (page size, how often it signals, scatter or plain writes) and read back the link bandwidth for exactly that shape.
- **It's a candidate for main.** If the mixes turn out to be revealing, this goes upstream regardless of how the router study ends.

## YAML spec

Terms:

- **Segment:** one entry in the stream. It is a traffic shape plus a `count`.
- **Lap:** one pass through the stream, sending each segment's `count` packets in order.
- **`repeat`:** the number of laps the stream runs.

A bare string is shorthand for `{ntype: …}`, so in the simplest case a stream is just a list of command types.

Round-robin of command types (shorthand):

```yaml
senders:
  - device: [0, 0]
    patterns:
      - destination: {device: [0, 1]}
        size: 4352
        repeat: 10000
        stream: [unicast_write, unicast_scatter_write, fused_atomic_inc]
```

A collective chunk: seven page writes, then one write that also signals:

```yaml
        stream:
          - {ntype: unicast_write, count: 7}
          - {ntype: fused_atomic_inc}          # count defaults to 1
```

Mixed payload sizes:

```yaml
        stream:
          - {size: 4352, count: 3}
          - {size: 16}
```

A stream in `defaults`, applied to every flow that a high-level pattern generates. CI YAMLs need this form: per the policy at the top of `test_api.yaml`, CI uses high-level patterns only.

```yaml
defaults:
  ftype: unicast
  size: 4352
  repeat: 5000
  stream:
    - {ntype: unicast_write, count: 7}
    - {ntype: fused_atomic_inc}
patterns:
  - type: full_device_random_pairing
```

### Fields

| Field | Allowed on | Meaning |
|---|---|---|
| `ntype`, `size`, `atomic_inc_val` | segment, pattern, `defaults` | A segment's value overrides the pattern's, which overrides `defaults` and `parametrization_params`. So a `size` sweep fills in every segment that doesn't set its own size. If no level sets `size`, today's 1024 B default applies. |
| `ftype`, `destination`, `mcast_start_hops` | pattern, `defaults` | Shared by every segment, so the whole stream takes one route. A stream is either all unicast or all multicast; mixing the two in one stream isn't supported. |
| `count` | segment | Packets of this segment per lap. Default 1. |
| `repeat` | pattern, `defaults` | Number of laps. Default 1. A stream pattern must not set `num_packets` (see the errors below). It is called `repeat` because high-level patterns already use `iterations` to mean "rerun with a new seed". |
| `vc_id`, `link_id`, `noc_id`, `core` | sender or pattern only | These apply to the whole stream. The sender kernel is compiled for one VC (`EdmSenderType` is picked from `VC_ID`). |

### Build-time errors

- A stream pattern must not have `num_packets`, whether it was set on the pattern, inherited from `defaults`, or set by a parametrization sweep. Packet count is `repeat` × lap length. In a test that mixes stream senders and plain senders, set `num_packets` on the plain patterns instead of in `defaults`.
- A sender with a stream has no other patterns (v1). Running several streams means using several senders.
- A high-level pattern can carry a stream only if each sender it generates has one flow, as with `full_device_random_pairing`. All-to-all style patterns, which give a sender several flows, are rejected in v1. Supporting them needs the one-config-per-flow design (see [Later](#later-v2)).
- A stream uses exactly one router.
  - This mostly follows from `ftype` and `destination` being stream-wide.
  - The remaining check: a multicast stream whose hops name more than one direction is rejected, because it needs a connection per direction.
  - To load several routers, use one sender per router.
- In benchmark mode, no two senders on a chip may use the same router (direction, link).
  - A router has one worker channel, and the benchmark path has no mux.
  - Today this is only caught after the run, in the bandwidth profiler (`tt_fabric_test_bandwidth_profiler.cpp:247-253`). The check moves into `validate_test`.
  - Validation mode allows it: `FabricConnectionManager::process` (`tt_fabric_test_device_setup.cpp:122`) puts a mux in front of any router used by more than one client (sender, credit-returning receiver, or sync core).
- The order program must fit the local-args region, which is 32 KB (`LOCAL_ARGS_BUFFER_SIZE`). It is capped at 256 entries.
- v1 does not support streams with `latency_test_mode`.
- Mux connections:
  - **Validation mode works with them.** Its sends dispatch on `is_mux` (`kernels/tt_fabric_test_kernels_utils.hpp:630-642`).
  - **The benchmark path doesn't.** It casts straight to the router adapter.
  - **What creates a mux.** Flow control does. With sync alone, separate connection managers avoid a mux (`tt_fabric_test_context.cpp:607-613`).
  - **The rule.** `benchmark_mode` plus `enable_flow_control` is rejected for streams. An assert in `FabricConnectionManager::process` is the backstop, since that is where `needs_mux` is decided.

## Host build

> **Key idea.** Each segment becomes an ordinary pattern on the sender, the same thing a multi-pattern sender has today, with `num_packets = repeat × its packets per lap`. The only new object is an **order program**, `[(pattern index, count), …]`, which the sender kernel walks.
>
> Receivers, memory allocation, data validation and connection setup need no stream awareness:
> - Each segment is a normal flow.
> - Its packets still arrive in order, because one flow always takes one route.

**Why separate configs.** A stream here is YAML shorthand over the pattern model that already exists. Every host stage keeps seeing one ntype and one size per pattern:

- `TrafficParameters`;
- the kernel-arg field builders;
- the allocator;
- `FlowDescriptor`;
- the bandwidth profiler.

The cost is that configs multiply (flows × segments), so the design only fits senders with one flow. [Later](#later-v2) describes the one-config-per-flow alternative and when to switch.

Segments are not deduped: `[A×3, B, A×3, C]` becomes four patterns and a four-entry program, and segment i is pattern i.

- The allocator gives each unicast pattern its own receiver payload chunk (`tt_fabric_test_allocator.hpp:1145`). So two segments with identical fields still have different headers.
- That matches a real kernel, which writes different chunks to different addresses.
- It also lets the worker-cost gate build distinct segments without per-segment destinations (W2, W3).

| Stage | Change |
|---|---|
| `ParsedTrafficPatternConfig`, `TrafficPatternConfig` (`tt_fabric_test_common_types.hpp:66, 101`) | Add `stream` (a vector of segments) and `repeat`. Add `stream_program` to `ParsedSenderConfig` and `SenderConfig`. |
| `parse_traffic_pattern_config` (`tt_fabric_test_config.cpp:117`) | Parse `stream` (a sequence where a scalar entry means `ntype`) and `repeat`. The segment parser reuses this function and adds `count`. |
| `merge_patterns` (`tt_fabric_test_config.hpp:244`) | Pass `stream` and `repeat` through, with the specific pattern's values winning. Segments are not merged here, so parametrized defaults still reach them later. |
| `resolve_hop_sentinels` (`:2586`, called at `:1474`) | No change. Hops live only on the pattern, which this pass already resolves; segments inherit the resolved destination in `expand_streams`. |
| `resolve_missing_params` (`:2642`) | Skip the `num_packets` default for stream patterns. Today it fills in 10 (`:2688-2690`), or a random count under the `randomize` policy. Because the default is skipped, any `num_packets` still on a stream pattern was set by the user, through the pattern, `defaults` or parametrization. `expand_streams` rejects it. |
| New pass `expand_streams` | Runs after `resolve_missing_params` (`:2642`, called at `:1483`) and before `split_all_unicast_or_multicast_patterns` (`:2481`, called at `:1486`). It:<br>• rejects a stream pattern that has `num_packets`;<br>• merges each segment over its pattern;<br>• rejects multicast hops that name more than one direction;<br>• turns each segment into its own pattern;<br>• sets each pattern's `num_packets`;<br>• writes the program onto the sender. |
| `split_senders_by_direction_for_benchmark` (`:2528`) | No change. A stream sender has one direction group, so the pass keeps the sender whole (`:2561-2563`) and the program stays with it. |
| `expand_link_duplicates`, `resolve_sender_config` (`:2613`, `:1258`) | Copy the program along with the sender. |
| `validate_test` (`:1829`) | Add the build-time errors listed above. |
| Sender kernel args (`tt_fabric_test_device_setup.cpp`) | Append `repeat` and the program to the sender's local args. Add a `STREAM_MODE` compile-time arg. |
| `--dump-built-tests` serializer (`:2818`, `:2854`) | Emit the expanded patterns and the program, so the expansion can be checked without hardware. |
| Command-line filters (`check_ntype_filter :638`, `check_size_filter :740`) | A stream matches if any of its segments matches. |

## Sender kernel

```cpp
// tt_fabric_test_sender.cpp, STREAM_MODE branch
for (uint32_t lap = 0; lap < repeat; lap++) {
    for (uint32_t e = 0; e < num_entries; e++) {
        auto* cfg = sender_config->traffic_config_ptrs[entry_cfg[e]];
        for (uint32_t c = 0; c < entry_count[e]; c++) {
            cfg->template send_stream_packet<BENCHMARK_MODE>(slot_state);  // blocks until sent
        }
    }
}
```

**Order is strict.** The worker sends segments in program order and blocks until each send succeeds. In validation mode with flow control, that means waiting on a segment's receiver credits even when a later segment's receiver has room. That is how a real collective kernel behaves.

The existing multi-pattern loop works differently: it skips a blocked pattern (`kernels/tt_fabric_test_sender.cpp:80-96`). That behaviour stays for senders without a stream.

### Benchmark mode

As today, the worker writes no payload data. The whole path uses public functions the adapter already has, so `edm_fabric_worker_adapters.hpp` is not changed.

**Setup.** `conn->setup_stateful_send_cmd_bufs()` (`edm_fabric_worker_adapters.hpp:371`) arms two NoC command buffers for that router:

- `data_noc_cmd_buf` for the header copies;
- `sync_noc_cmd_buf` for the credit bumps.

**For each packet:**

1. Wait for a free slot on the router.
2. Read the slot index: `conn->buffer_slot_index.get()`. The field is public (`:650`).
3. Then do one of these:
   - **The slot holds a different segment's header.** Call `conn->send_current_slot_stateful_non_blocking_from_address(header, sizeof(PACKET_HEADER_TYPE))` (`:404`). It copies the header on `data_noc_cmd_buf`, bumps the credit on `sync_noc_cmd_buf` and advances the slot index, all using the stateful command buffers. Then record the segment in the slot table.
   - **The slot already holds this segment's header.** Call `fabric_detail::update_credits_and_slots<true>(conn)` (`:797`). It bumps the credit using the stateful buffer and advances the slot index.

**Why not `send_payload_flush_non_blocking_from_address`.**

- Its credit bump is the stateless `noc_inline_dw_write`, which uses `write_at_cmd_buf` (`dataflow_api.h:2401-2403`).
- By default, `sync_noc_cmd_buf` is that same command buffer (adapter `:199-200`).
- So calling it in the middle of the stream would clobber the stateful credit setup. That is also why today's fast path arms its credit state only after warmup (`kernels/tt_fabric_test_kernels_utils.hpp:1083-1089`).

Details:

- **Prebuilt headers.** Headers are built once per segment at setup. The sender already gets one header buffer per pattern (`SenderKernelMemoryMap::get_packet_header_address`, `kernels/tt_fabric_test_kernels_utils.hpp:1501`), so each segment maps onto its own.
- **Slot table.** One byte per slot, starting empty. The first lap fills the slots naturally, so there is no separate warmup phase.
- **When copies happen.**
  - Let N be the router's worker slot count (`num_buffers_per_channel`). Packets k, k+N, k+2N… all land in the same slot.
  - Let L be the lap length in packets. If L divides N, each slot always sees the same segment, and copies stop after the first lap. `[write×7, fused_atomic_inc]` on 8 or 32 slots is an example.
  - Otherwise some packets need a copy. The host logs the expected copy fraction, so results can be read with that in mind.
- **Ordering.** The header copy and the credit bump go from the worker to the same Ethernet core on the same NoC, with no barrier between them. `send_current_slot_stateful_non_blocking_from_address` is the production function that pairs exactly these two writes.
- **No unrelated NoC writes while armed.** The adapter warns that under `DM_DYNAMIC_NOC`, write and write_reg traffic on a worker RISC share one physical command buffer (`:372-374`). The stream loop therefore issues no other NoC writes while the stateful state is armed. Benchmark mode already skips progress writes.
- **One arm for the whole run.** Since a stream has one router, the stateful setup is armed once and never re-armed.

### Validation mode

The existing per-packet send is used, called in program order and blocking until it succeeds. That send:

- takes flow-control credits when flow control is enabled;
- fills the payload;
- writes the payload, then the header;
- advances the buffers.

The receiver side is unchanged.

## Bandwidth accounting

- **Today's bug.**
  - The profiler stores one pattern's size and packet count per (device, direction, link) (`tt_fabric_test_bandwidth_profiler.cpp:282-302`).
  - It multiplies that by the number of flows crossing the link. That number comes from `outgoing_traffic_`, which is built by tracing each flow's path (`:115-122`).
  - So it assumes every flow on a link has the same size and count, which is wrong for any mix.
- **Fix.** During the same path trace, add each pattern's payload bytes and packet count to every (device, direction, link) it crosses, so bytes = Σ size × packets. Every uniform test gets the same number as today.
- **Cycles are unchanged.** A sender's elapsed cycles already apply to every (direction, link) it uses (`:236-256`).
- **Per-link output:** payload GB/s, packets/s and cycles/packet. atomic_inc carries no payload, so any mix that includes it needs cycles/packet shown next to GB/s.
- **Summary row.**
  - **Today.** The summary (`:395-431`) computes bytes as `packet_size × num_packets × max_traffic_count`. Size and count come from whichever link was visited last.
  - **The problem with streams.** Links can carry different loads. Take the forwarded case from [First mixes](#first-mixes): chip 0 streams to chip 2 through chip 1, while chip 1's own worker streams to chip 2.
    - Link 0 → 1 carries only chip 0's stream.
    - Link 1 → 2 carries both streams.
    - If the two streams differ in sizes or counts, today's code reports whichever link it happened to visit last.
  - **The fix.**
    - Take bytes and packets from the link with the most bytes. Break ties by packets, then cycles. That is link 1 → 2 in the example.
    - Keep dividing by the maximum cycles over all links, as today, so every uniform test reports the same number as now.
- **CSV and goldens.** v1 adds no new columns.
  - A one-segment stream reports exactly like the plain pattern, so existing goldens stay valid.
  - ntype reports the segments' ntype when they all share one, and `stream` only when they differ. So a size mix of `unicast_write` segments reports `unicast_write`, and a command-type mix reports `stream`. ftype is stream-wide, so it reports as today.
  - packet_size reports the mean payload per packet, rounded to a whole byte. The golden loader parses that column as an integer (`tt_fabric_test_bandwidth_results.cpp:217`).
  - The test name identifies the stream.
  - The summary row currently takes its ftype, ntype and size from the first pattern (`:404-409`). It switches to the same rules.
- **Goldens.** Stream rows go into the existing per-platform golden file. No separate file is needed.
  - **Matching.** A golden row is matched on test name, ftype, ntype, topology, num_devices, num_links and packet_size (`tt_fabric_test_bandwidth_results.cpp:27-32`). ntype is compared as a string, so `stream` works. A stream test's unique name keeps its row distinct.
  - **Why not a separate file.** The loader picks exactly one file per platform (`golden_<metric>_summary_<arch>_<cluster>.csv`, `tt_fabric_test_results.cpp:245-254`). A separate stream golden would need new code to choose between files.
  - **Before mainlining.** No golden rows are needed while the work is on a branch. A result with no matching row reports `NO_GOLDEN`, which doesn't fail the run (`tt_fabric_test_bandwidth_results.cpp:480`). Stream rows are added to the golden file only if P3 leads to a PR.
- **Mix penalty.** This is computed after the run, not in the harness, for v1:
  - Penalty = measured cycles/packet ÷ Σ fᵢ·cᵢ, where fᵢ is each segment's share of packets and cᵢ is its cycles/packet from a pure run in the same session.
  - Above 1 means mixing costs something. Below 1 means the types overlap.
  - The script starts in the experiment tools, and moves into the harness if it proves useful.

## Worker-cost gate

Losing today's single-pattern fast path could make the worker the bottleneck instead of the router. These runs settle that before anything else builds on the stream path. All runs use:

- benchmark mode, 1 hop, 4352 B;
- two router builds: stock, and the fastest experimental build (13129088, W32/fwd8, about 120 cycles/packet).

| Run | Stream | What it isolates | Expected |
|---|---|---|---|
| W1 | One segment, identical to today's single pattern | Cost of the program loop and slot check | Within run-to-run noise of today's number |
| W2 | `[A, A′]`: two write segments with identical fields. Each gets its own receiver buffer, so their headers differ (lap of 2, so no copies) | Two configs and no copies; the router sees pure write traffic | Same as W1 |
| W3 | `[A×3, A′×3]`: a lap of 6, which never divides a power-of-two slot count | Header-copy cost. The router still sees pure write traffic, so any change from W1 is worker cost. | Measure the change: it is the copy cost per packet |

## Validation

A new `test_yamls/test_fabric_stream_sanity.yaml` runs with data validation on (not in benchmark mode). It covers:

- all four command types in one stream;
- sizes from 16 B to the maximum payload in one stream;
- a multicast stream (`E: max` line multicast) mixing the command types multicast supports;
- flow control on and off. Flow control can put a mux in front of a router, which validation mode supports;
- a stream carried by a high-level pattern.

Passing means 0 failures in all of these:

- the stock router;
- the study's validated flag set: 546176 and its extensions, never the benchmark-only bits 0/4/17/20/21;
- channel trimming (`enable_channel_trimming`), because multi-hop and multicast segments use forwarding channels that the capture pass has to see;
- the existing `test_fabric_mesh_sanity_common.yaml`, unchanged (96 executed, 0 failed).

## First mixes

| Mix | Stream | What it shows |
|---|---|---|
| Command types | `[unicast_write, unicast_scatter_write, fused_atomic_inc, atomic_inc]` | Whether a change that speeds up writes costs scatter or atomics on the receive side |
| Sizes | `unicast_write` at `[4352, 16]`, then `[4352, 1024, 256, 16]` | Whether the core's per-packet work for small packets overlaps the wire time of large ones |
| Collective chunk | `[unicast_write×7, fused_atomic_inc]` | A realistic write-then-signal shape |

Where they run:

- 1-hop east-west and 1-hop north-south.
- One forwarded case: chip 0 streams to chip 2 through chip 1, while chip 1's own worker streams east. This is the only way to get two flows onto one router.

Each mix runs alongside pure runs of its segments, for the mix penalty. Router builds: stock, 546176, 13129088, and 13129088 W32/fwd8.

## Phases

**P1.** Spec, parser, `expand_streams`, the benchmark-mode stream path in the sender, and the bandwidth-accounting and summary-row fixes.
*Exit:* W1–W3 are measured, W1 is within noise, and existing benchmark YAMLs report the same numbers as before the accounting fix.

**P2.** Validation-mode stream path, the stream sanity YAML, and the build-time checks.
*Exit:* 0 failures on stock, under the study's sanity flags, and with channel trimming.

**P3.** The three mixes, the mix-penalty script, and runs across the router builds.
*Exit:* a decision on whether the mixes reveal something the uniform rows don't. If they do, open a PR to main with docs in `test_yamls/test_api.yaml`.

**P4.** The v2 items below, only if P3 shows the tool is worth extending.

### P1 status (2026-10-15)

Done on branch `aagarwal/fabric-test-streams` (worktree `/home/aagarwal/tt-metal-streams`, uncommitted), and ported into the experiment tree for the fast-build runs (`fabric_exp/streams/port/`).

- **Worker-cost gate,** 1-hop east, 4352 B writes, benchmark mode, no BW telemetry:

  | Run | Stock GB/s | 13129088 W32/fwd8 GB/s | 13129088 cycles/packet |
  |---|---|---|---|
  | W0, today's single-pattern path | 26.48 | 44.82 | 131.1 |
  | W1, one segment | 26.48 | 44.86 | 131.0 |
  | W2, two segments, no copies | 26.49 | 44.92 | 130.8 |
  | W3, lap of 6, 2 of 3 packets copy a header | 26.48 | 41.50 | 141.6 |

  The stock router hides all worker cost. On the fast build W1 and W2 match W0 within noise, and W3's header copies add 10.5 cycles/packet. That is only the part showing above the router's 131 cycles; P3 found that a stream where every packet copies is capped by the worker at about 170–177 cycles/packet (correction, see P3 status).
- **Accounting fix:** existing YAMLs report the same bytes as before on every summary and per-link row. Checked on 10 2D Mesh tests (`fabric_exp/exp_2d_4352.yaml`) and 42 Linear/Ring runs (trimmed from `test_fabric_ubench_at_least_2x2_mesh.yaml`). The only summary differences are on 3-iteration ring rows, where averaging bandwidth and cycles separately skews their product.
- **Validation smoke** (`fabric_exp/streams/stream_smoke.yaml`): command-type mix, size mix, `E: max` multicast stream, collective chunk with flow control, a stream on `full_device_random_pairing`, and a short run without flow control. 6 executed, 0 failed on stock. Long validation runs need flow control, as in the existing sanity YAMLs; without it a sender can lap its receiver's buffer.

Where the code departs from the text above:

- The order program is just the per-segment counts, since segment i is pattern i.
- Streams are capped at 64 segments (`kMaxStreamSegments`).
- The mixed-ntype label is `STREAM`, matching the uppercase enum names in the ntype column.
- atomic_inc segments carry size 0, so they count as no payload. The allocator's credit calculation now treats a 0-byte packet as taking no buffer space; before, it divided by the size.
- The benchmark-mode "one sender per router" check runs only for tests that contain a stream, so existing tests can't start failing at build time.
- The validation-mode stream send path is already in (planned for P2).
- The branch carries a measurement-only `TT_FABRIC_NO_BW_TELEMETRY` switch in `test_tt_fabric.cpp`. Drop it before any PR.
- Not done in P1: the copy-fraction log and negative tests for the build-time errors. Both landed in P2.

### P2 status (2026-10-15)

- **Stream sanity YAML** (`test_yamls/test_fabric_stream_sanity.yaml`, data-validated). It has 10 tests and uses only high-level patterns that give each sender one flow: `full_device_random_pairing` and `unidirectional_linear`. It covers:
  - all four command types, at 1 and 2 links and 1024 and 4096 B;
  - sizes from 16 B to 4 KB;
  - a collective chunk;
  - line multicast with every command type;
  - 1D unicast and multicast;
  - one short run without flow control.
- **Results, all "10 executed, 0 failed":** stock (branch), and the experiment tree with no flags, 546176, 4740480, and 13129088 W32/fwd8. The existing `test_fabric_mesh_sanity_common.yaml` still gives 96 executed, 0 failed on both trees.
- **Build-time errors:** 13 negative cases in `fabric_exp/streams/negative/` (`run_negative.sh`), one per error, each checked for its message: 13 passed. They run on hardware, with a chip reset after each, because no parser unit tests exist.
- **Channel trimming.** On main, the trimming replay hangs for multi-hop traffic on this system, with or without a stream:
  - plain `unidirectional_linear` multicast with flow control hangs (Mesh);
  - plain `unidirectional_linear` unicast hangs (Linear);
  - main's own 1-hop `ChannelTrimming` shape from `test_features.yaml` passes, with and without a stream.

  Those hanging configurations are ruled out, not probed further. So the CI YAML has no trimming test: CI YAMLs need high-level patterns, and those give multi-hop routes. In the experiment tree, stream trimming passes at 1 hop and 3 hops (`fabric_exp/streams/trim_stream_exp.yaml`, benchmark mode, capture and replay).
- **Header-copy log.** The benchmark stream loop counts its header copies into result word 8, and the host logs "N of M packets copied a header" per stream sender. It showed that the default trimmed 2D build gives the worker channel 17 slots (study §27). So a 4-segment lap copies on every packet, and `[write×7, fused_atomic_inc]` copies on 25% of packets. The W2 "no copies" case holds only for an even slot count such as W32. Copies are hidden on the stock router but cap the worker on fast builds (see P3 status), so a mix penalty has to be read next to the logged copy fraction.

### P3 status (2026-10-15)

**Runs.** Four mixes at three locations: 1-hop east-west, 1-hop north-south, and the forwarded case (chips 0 and 1 both stream to chip 2, so chip 1's east router carries two flows). Each mix runs next to single-segment runs of every segment it uses.
- **Files:** `fabric_exp/streams/p3_mixes.yaml` (33 tests), `p3_runs.sh`, and `mix_penalty.py`, which writes `results/p3_penalty.md`.
- **Builds:** stock (branch), 546176, 13129088, and 13129088 W32/fwd8, plus 4740480 for the east-west subset.
- **Results:** 33 executed, 0 failed on every build. Benchmark mode, no BW telemetry.
- **Metric:** cycles/packet is 1.35 GHz ÷ packets/s, as in the study's `cyc.py`. For the forwarded case it is the shared link.
- **Mix penalty:** measured cycles/packet ÷ Σ share × pure cycles/packet.

Pure runs, 1-hop east-west, cycles/packet:

| | stock | 546176 | 4740480 | 13129088 | 13129088 W32/fwd8 |
|---|---:|---:|---:|---:|---:|
| write 4352 B | 221.9 | 153.1 | 133.5 | 132.6 | 131.1 |
| scatter 4352 B | 220.8 | 191.3 | 172.6 | 172.6 | 171.5 |
| fused_atomic_inc 4352 B | 222.6 | 179.6 | 156.7 | 156.6 | 158.7 |
| atomic_inc | 374.2 | 374.8 | **509.4** | **509.6** | **507.1** |
| write 1024 / 256 / 16 B | 223.4 / 223.4 / 221.6 | 151.2 / 152.1 / 153.3 | 144.7 / 144.8 / 144.7 | 137.2 / 136.8 / 137.0 | 138.6 / 138.3 / 138.4 |

Mix penalties (measured ÷ predicted), with the share of packets that copied a header:

| Location | Mix | stock | 546176 | 13129088 | 13129088 W32/fwd8 |
|---|---|---:|---:|---:|---:|
| E/W | commands `[write, scatter, fused, atomic]` | 0.899 (100%) | 1.029 (100%) | 1.061 (100%) | 1.066 (0%) |
| E/W | sizes `[4352, 16]` | 0.967 (100%) | 1.158 (100%) | 1.316 (100%) | 1.034 (0%) |
| E/W | sizes `[4352, 1024, 256, 16]` | 0.979 (100%) | 1.113 (100%) | 1.247 (100%) | 0.957 (0%) |
| E/W | chunk `[write×7, fused]` | 0.999 (25%) | 0.973 (25%) | 0.997 (25%) | 0.974 (0%) |
| N/S | commands | 0.942 | 1.045 | 1.072 | 1.073 |
| N/S | sizes `[4352, 16]` / 4 sizes | 0.969 / 0.978 | 1.150 / 1.102 | 1.337 / 1.291 | 1.000 / 0.999 |
| N/S | chunk | 1.000 | 0.998 | 0.995 | 0.999 |
| forwarded | commands / sizes 2 / sizes 4 / chunk | 0.980 / 0.951 / 0.958 / 1.000 | 1.032 / 1.000 / 1.000 / 1.007 | 0.987 / 0.959 / 0.979 / 0.970 | 0.994 / 0.975 / 1.022 / 0.979 |

N/S and forwarded copy shares match E/W. W32/fwd8 also changes the forwarding channels, so forwarded-case differences between the last two columns aren't due to copies alone. Full tables are in `fabric_exp/streams/results/p3_penalty.md`.

**What the mixes show that the study's uniform rows didn't:**
1. **Bit 22 slows atomic_inc by 36%:** 374.8 → 509.4 cycles/packet, the same on E/W, N/S and the forwarded link. Bit 23 adds nothing to it. The study's rows were write and scatter only. The command mix exposed it: 231 → 257 cycles/packet from 546176 to 4740480. A uniform atomic row shows it too, but none had been run.
2. **Mixing command types costs the fast routers 3–7%,** on top of the average of the pure runs: 1.03–1.07 on E/W and N/S. On stock, mixing helps (0.90–0.94).
3. **Bit 22 makes small writes slower than 4352 B writes:** 144.7 against 133.5 cycles/packet. Bit 23 brings them back to about 137.
4. **Size mixes and the collective chunk show no router-level penalty** where the worker isn't the limit (W32 column: 0.96–1.03). Every build is bound by per-packet cost at 1 hop, so mixing sizes neither costs nor overlaps.
5. **Forwarded traffic adds no extra penalty** (0.95–1.03).

**What has to be fixed before streams go to main: the test's own worker caps the stream when headers are copied.**
- With every packet copying a header, the sizes mixes measure 177.3 (lap 2) and 169.4–169.6 (lap 4) cycles/packet on 546176, 4740480 and 13129088 alike. Identical results on three different routers mean the worker is the limit, not the router.
- That puts a header copy at roughly 100 worker cycles, not the 16 estimated in P1.
- On the default build (17 worker slots) most multi-segment streams copy on every packet. So on any router faster than about 170 cycles/packet, a stream reports a false 10–34% "mix penalty". The stock router (about 222) hides it.
- Options:
  - **(a)** Find out why a 96 B stateful header copy costs about 100 worker cycles, and make it cheaper.
  - **(b)** Have the kernel report whether the worker ever had to wait for a free slot. A run where it never waited is worker-bound, and the harness would flag that result.
  - Both are inside test_infra.

**Decision needed.** The mixes do reveal router behaviour the uniform rows missed (items 1–3), which meets the P3 bar for a PR to main. The worker cap should be fixed or flagged first, and opening a PR is the user's call.

### Worker-cap follow-up (2026-10-15)

**(b) Worker-bound flag: done, on the branch and ported.**
- **What it does:** the benchmark stream loop counts packets that found every router slot full (result word 10). The host logs that share next to the header-copy share. When fewer than 1% of a sender's packets had to wait, the host warns that the run measured the worker, not the router.
- **Separation:** router-bound runs wait on 15–25% of packets on the fast builds and about 80% on stock. Those shares match the router returning credits in groups (25% when they come in fours), so the share is never near 100%. Worker-bound runs wait on 0.0%.
- **Where it fires:** on 13129088 (17 worker slots), W2 and both size mixes; on 13129088 W32, W3. It fires on nothing else in the gate, the east-west mixes, or the stock gate. Script: `fabric_exp/streams/stream_flags.py`.
- **Side effect:** the extra free-slot check changed the worker-bound speed of copy-every-packet streams from 177 to 161 cycles/packet. Router-bound results didn't move (pure write 132.6, command mix 257.7, chunk 134.8). The worker loop's pace depends on small code details, which is one more reason to flag such runs rather than trust them.

**(a) Cheaper header copies: tried, not adopted.** Superseded: the real cost was elsewhere in the loop, see "Faster benchmark send loop" below.
- **Setup:** variants of the copy path in the experiment tree's kernel (temporary, then restored), each run on 13129088 with 17 slots.
- **Measure:** W2, the two-segment lap where every packet copies a header and the worker is the limit.

| Variant | W2 cycles/packet |
|---|---:|
| Current: adapter's stateful copy, then the stateful credit | 161.2 |
| V1: posted header writes | 164.6 |
| V2: header length preset once in the command buffer (one fewer register write per copy) | 157.9 |
| V3: V2 plus posted writes | 155.3 |
| V4: copy every packet, no slot table | 148.8 (one segment: 142.2) |

- **What it shows:** the slot-table bookkeeping costs about 12 cycles/packet, and the copy path costs about 142–149 cycles/packet even with no bookkeeping. The register-level trims save at most 6.
- **Not enough to matter:** none of this gets a copy-every-packet stream below the fast routers (131–138 cycles/packet). The gains are small and depend on code layout.
- **Conclusion:** keep the flag as the protection.
- **What would really fix it:** not copying at all. That means a stream order that is periodic in the worker slot count, which the default 17-slot channel rules out unless packets are reordered within a lap. That reordering is a semantic change (the plan's v2 "shuffle within a lap" item).

### Faster benchmark send loop (2026-10-16)

**Why the worker was slow.**
- **Measurement:** the loop now times its own waits for a free slot, and the run log reports the worker's busy cycles per packet with that time left out. That showed even a credit-only send cost the worker about 113 cycles. With a header copy on every packet it cost 161 (13129088, 17 slots).
- **What the disassembly showed:**
  - **The connection object lives in L1.** The test infra builds it in the kernel-config region.
  - **The adapter's free-slot check runs `invalidate_l1_cache()` (a `fence`) on every packet.** That empties Blackhole's 4-line L1 cache, so each packet reloads the connection's fields from L1 at about 10 cycles per miss.
  - **Each operation is cheap on its own,** timed in the kernel: L1 read about 10.6, NoC register read about 10.4, NoC register write about 3.4, fence after a register write about 21.
  - **A two-segment loop spills its counters to the stack.** Each segment switch then spends about 19 cycles reloading them.
- **What the 53 cycles of a credit-only send are,** measured part by part with a local connection and batched checks:
  - the credit send: ready poll, send write, slot bookkeeping and NoC write counters, about 36;
  - the free-slot check: about 75 per check, roughly one check per 4 packets, so about 18 per packet;
  - loop control.
- **No time is spent waiting on the NoC interface:** the command buffer was ready on the first poll for every packet.

**The new loop** (branch and experiment tree; test kernel only, no adapter change). The router sees the same traffic as before: one credit per packet, every packet into a free slot.
- The loop works on a local-memory copy of the connection and writes it back for `close_connections`.
- It reads the free-slot count once and uses that many slots before checking again.
- Laps of up to 64 packets are flattened into a local array of (segment, header address), so the loop is one counter. Longer laps keep the nested loops.

Worker busy cycles per packet:

| | 13129088, 17 slots: one segment | 13129088, 17 slots: two segments, every packet copies | W32: two segments, no copies |
|---|---:|---:|---:|
| old loop | 113.1 | 161.1 | 118.6 |
| local connection + batched check (E3b) | 53.3 | 94.6 | 71.7 |
| **new loop (E3b + flat lap)** | **57.9** | **80.1** | **58.0** |

- **The header copy itself** now costs about 22 worker cycles.
- **Busy time grows with waits.** Every wait adds a free-slot check and timer reads, so busy time is higher where most packets wait: stock, at 88% waits, reads 93–117. Those runs are router-bound anyway.
- **Code-layout sensitivity.** One variant used the flat loop only for multi-segment streams and kept the nested loop for one segment. Through the shared per-packet lambda, the nested loop then compiled to 75.8. So the flat loop is used for every lap up to 64 packets.
- **The worker's pace depends on how the compiler lays out the loop,** which is why the worker-bound flag stays.

**P3 rerun with the new loop** (`p3_runs.sh p3f`; 33 executed, 0 failed on every build; no worker-bound flags anywhere; `results/p3f_penalty.md`). Single-type runs are unchanged (within 0.9 cycles/packet). Mix penalties:

| Location | Mix | stock | 546176 | 13129088 | 13129088 W32/fwd8 |
|---|---|---:|---:|---:|---:|
| E/W | commands | 0.899 | 1.042 | 1.062 | 1.055 |
| E/W | sizes `[4352, 16]` / 4 sizes | 0.967 / 0.983 | 0.989 / 0.988 | 1.018 / 0.957 | 1.031 / 0.964 |
| E/W | chunk | 1.000 | 0.979 | 0.992 | 0.974 |
| N/S | commands | 0.932 | 1.057 | 1.072 | 1.073 |
| N/S | sizes 2 / 4 | 0.970 / 0.979 | 0.998 / 0.998 | 1.021 / 1.014 | 1.000 / 0.999 |
| N/S | chunk | 1.000 | 0.998 | 0.994 | 0.999 |
| forwarded | commands / sizes 2 / sizes 4 / chunk | 0.987 / 0.951 / 0.958 / 1.000 | 1.039 / 1.000 / 1.000 / 1.008 | 0.998 / 0.957 / 0.979 / 0.970 | 0.999 / 0.976 / 1.023 / 0.979 |

The size mixes that showed 1.10–1.34 on 546176 and 13129088 (worker-capped, every packet copying) are now 0.96–1.02, the same as the W32 runs that copy nothing. The P3 findings stand:
- the bit-22 atomic_inc slowdown;
- the 3–7% command-type penalty on the fast builds;
- the small-write slowdown under bit 22;
- no penalty from mixing sizes or from forwarded traffic.

The worker cap is fixed for these mixes. The flag still guards against cases this loop doesn't cover.

### Existing benchmark send paths moved onto the fast loops (2026-10-16)

- **Single-pattern benchmark senders run the stream loop as a one-segment stream** (`repeat` = the pattern's packet
  count). The router-visible traffic is the same: headers into the first slots, then one credit per packet. The old
  `send_packets_stateful` helper is gone.
- **Multi-pattern benchmark round robin keeps its behaviour:** each pattern writes headers for its first slot-count
  packets, then stateless credits. It now works on local copies of each connection, with a batched free-slot check
  per connection.
- **Validation-mode sends are unchanged.**
- **Every benchmark sender writes the copy, wait and busy counters.** The profiler warns about any worker-bound
  sender. Plain senders log their counters at debug level.
- **Before/after on 13129088:**
  - 1-hop writes +1.0 to 1.6% (E/W 134.6 -> 132.5 cycles/packet, N/S 136.2 -> 134.3, bidirectional 140.0 -> 138.7);
  - half-ring multicast -5.9% cycles (251.6 -> 236.8, at 1 and 2 links);
  - every other row within ±0.7%.
- **Stock:** unchanged within 0.2%. Full-ring multicast varies ±1-2% from run to run.
- **Regressions:** mesh sanity 96/0 and stream sanity 10/0.
- **The study's 4352 B set** (§29) was rerun on the new loop: `fabric_exp/RESULTS.md` §31.

**Branching.** Build P1–P3 on a branch from main. The experiment tree carries uncommitted harness changes (phase profiler, trimming hooks), and keeping those out keeps the PR clean. Port the work to the experiment tree only for router-build measurements.

## Risks

- **The worker becomes the limit** (W1–W3). The stream path would still work for validation and workload modelling, but its benchmark numbers would need that caveat.
- **Header-copy cost** when a lap's length doesn't divide the slot count (W3).
- **Capacity.** Many segments can exhaust the local-args region or the receivers' `max_configs_per_core`. Both fail at build time with a clear message.
- **New router behaviour.** Changing the packet type on every packet is new for the router and may expose bugs in experimental builds.
  - Run on stock first.
  - Under the standing rule, a hang rules that build out; it does not start a search for the boundary.
- **CSV consumers.** v1 avoids breaking them by keeping the schema unchanged.

## Later (v2)

- **Idle segments.** `{idle: <cycles>}` makes the worker wait before its next credit bump. This models compute gaps and short bursts.
- **Lap completion time.** The last packet of a lap triggers a reply, and the sender times the round trip for each lap. This approximates "time until a chunk's semaphore lands", which is the latency number a collective developer cares about.
- **Seeded shuffle within a lap**, to avoid aliasing with periodic router behaviour such as bit 22's batches of 4.
- **Streams on all-to-all style patterns, via one config per flow.** See below.

### One config per flow

**When to do it.** Only if mainlining needs streams on patterns that give a sender several flows. CI YAMLs use high-level patterns only, and many of those are all-to-all style.

**What it is.** The stream becomes a property of a flow, the way `size` is today. Each flow has one traffic config whose packet shape follows its segment list.

**Why v1's design doesn't stretch to all-to-all.**

- **Configs multiply.** Each sender and each receiver gets one config per flow per segment. A 4-segment stream on 2x4 all-to-all needs 28 configs per sender instead of 7, and a 32-chip Galaxy would need 124.
- **Receiver buffers shrink.** The allocator splits a receiver core's usable L1 evenly across its configs (`tt_fabric_test_allocator.hpp:881-882`), up to 64 configs per core (`:906`).
- **No order across destinations.** One program per sender has no defined order across destinations.

**What one config per flow gives.**

- **Order is defined.** It is strict within a flow. Across flows it is today's round robin, which skips blocked flows. That settles the open question of whole stream per destination versus packet-by-packet interleaving.
- **Config count matches today.** It equals the number of flows.
- **One router per stream comes for free**, because a flow has one route.
- **The order program goes away**, and so does the "a sender with a stream has no other patterns" rule.
- **It is closer to a real kernel.** The writes and the signal land in one receiver buffer and one counter.

**What it costs.**

- **Representation (the main cost).** Resolved patterns carry a segment list instead of one (ntype, size, atomic_inc_val). Every consumer of that shape has to change:
  - `TrafficParameters` (`tt_fabric_test_traffic.hpp:28-30`);
  - the per-ntype field builders for kernel args;
  - the fused-ntype setup in `tt_fabric_test_device_setup.cpp:1490, 1591`;
  - `FlowDescriptor` (`tt_fabric_test_context.cpp`);
  - the bandwidth profiler and results.

  The safe form normalises every pattern to a segment list, with a plain pattern as one segment. Keeping the scalar fields as a fallback would let a consumer silently compute bytes as size × packets for a mixed flow.
- **Allocator (small).** All of its shape-dependent logic is in the receiver pass (`tt_fabric_test_allocator.hpp:946-1173`):
  - ntype decides which addresses are needed (`:953-955`, `:1014-1019`, `:1124-1130`);
  - size is checked against the chunk (`:971`, `:1135`);
  - size sets the credits (`:1162-1165`).

  These become three questions per flow: does any segment need a payload chunk, does any need an atomic counter, and what is the largest packet? The multicast search already finds a core with both kinds of address, since fused writes need both. Credits count capacity in largest-packet units, and total packets are repeat × lap length.
- **Receiver validator.** A new stream validation config walks the same segment list and calls the existing write, atomic and scatter poll/validate code for each packet. It uses one payload region sized for the largest segment plus one atomic counter. Today each validation class assumes a fixed shape per config (`kernels/tt_fabric_test_kernels_utils.hpp:1905-2060`). Benchmark-mode receivers don't validate (`kernels/tt_fabric_test_receiver.cpp:65-100`), but they still parse their config args.
- **Sender config.** One prebuilt header per segment inside the config. The payload offset advances by each segment's size.

**What carries over from v1.**

- **Unchanged:** the YAML, the bandwidth-accounting fix, the slot-tag logic in the benchmark path, and the stream sanity YAML.
- **Results:** for senders with one flow, both designs send the same packets in the same order, so P1–P3 numbers stay valid.
- **Thrown away:** only `expand_streams` and the order program.

## Decisions

All decided.

1. **Default segment size.** Inherited from the pattern, falling back to 1024 B as today.
2. **`repeat` when absent.** One lap. `num_packets` plays no part in a stream's packet count.
3. **Goldens.** Stream rows fit the existing golden schema and file, so there is no separate golden. None are needed until mainlining. A one-segment stream reports exactly like the plain pattern.
4. **Branch.** Develop on a branch from main, and port to the experiment tree only for measurements.
5. **An explicit `num_packets` on a stream pattern is rejected at build time.** This applies whether it comes from the pattern itself, from `defaults`, or from a parametrization sweep.
   - It stops a perf YAML that sets `num_packets: 100000` in `defaults` from quietly running a single lap.
   - It stops a `num_packets` sweep from producing identical tests.
   - The cost: a test that mixes stream senders and plain senders must set `num_packets` on the plain patterns rather than in `defaults`.

## Files

- `test_infra/tt_fabric_test_common_types.hpp`
- `test_infra/tt_fabric_test_config.hpp`, `.cpp`
- `test_infra/tt_fabric_test_device_setup.cpp`, `.hpp`
- `test_infra/kernels/tt_fabric_test_sender.cpp`
- `test_infra/kernels/tt_fabric_test_kernels_utils.hpp`
- `test_infra/tt_fabric_test_bandwidth_profiler.cpp`
- `test_infra/tt_fabric_test_bandwidth_results.cpp`
- `test_infra/tt_fabric_test_context.cpp` (the backstop assert for mux plus benchmark mode, if `process` doesn't already have the test mode)
- `test_infra/test_yamls/test_api.yaml` (docs)
- `test_infra/test_yamls/test_fabric_stream_sanity.yaml` (new)
