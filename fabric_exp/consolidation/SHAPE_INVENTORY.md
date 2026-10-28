# Router shape inventory (v3q), 2026-10-26

Facts for the shape note: what a router must cover, what its main loops carry and where each value comes from, and
what keeping state in memory costs today. Measured on v3q (the lead) and, for contrast, main's router (v1 at flags 0).
Everything here is offline except the PC samples, which are stage 3's (ROUTER_MAP_V3.md).

## 1. What a router must cover

From the host builder, the coverage mocks (`coverage.py`, now with Galaxy cases) and the runtime-arg dumps (section 2).

| Dimension | Values |
| --- | --- |
| Topology | 1D Linear and Ring; 2D Mesh and Torus; express (a torus plus dim-0 chords as Z edges); multi-mesh (VC1, intermesh, VC0→VC1 crossover); VC2 |
| VC0 sender channels | 2 (1D), 4 (2D), **5 on every router of an express fabric** (worker + 3 mesh + Z; the builder sizes all routers to the widest family) |
| VC1 / VC2 sender channels | up to 4 (3 mesh + Z) / 1 |
| Downstreams per VC0 receiver | 1 (1D), up to 3 (2D), **up to 4 (express)** |
| Router directions | E, W, N, S, plus Z (16 Z-facing routers in the 8x4 express Galaxy) |
| ERISCs per router | 2 on Blackhole (E0 sender, E1 receiver); 1 on Wormhole and with `TT_METAL_DISABLE_FABRIC_TWO_ERISC` |
| Features | deadlock avoidance (bubble flow control on rings and tori), first-level ack, the mux (tensix extension), UDM relay, BW telemetry, dynamic telemetry, perf telemetry, code profiling, channel trimming (capture and replay), data cache, TXQ spin-wait flags |

**Which program v3q runs** (offline builds of every coverage config at -O3 / -O2 / -Os, 1,299–1,304 of 1,435 ELFs
built; the failures are the inherited `nofuse` and Wormhole VC1 IRAM ones):

| Config | E0 | E1 |
| --- | --- | --- |
| BH 2D and 1D, VC0 and multi-mesh VC1, with or without telemetry, trimming, code profiling | sender program | receiver program |
| BH ring, torus (P150 and Galaxy) | sender program | generic loop |
| **Galaxy express (all 28 configs)** | **generic loop** | **generic loop** |
| VC2 | generic | generic |
| data cache, TXQ spin-wait | generic | receiver program |
| mux, UDM, one-ERISC BH, every Wormhole config | generic (one ERISC) | — |

The specialized programs cover exactly what the study benchmarked. Express, rings and tori (E1), VC2, the mux, UDM and
Wormhole get none of v3's loop work. Express falls back because the sender program visits at most four VC0 channels
(`vc0_visit_limit`, `shape_supported`).

## 2. Runtime arguments: which differ between routers

`argdump.py`: a mock-cluster run per topology with `TT_FABRIC_EXP_DUMP_ARGS` (a study hook in
`compute_mesh_router_builder.cpp`), one line per router ERISC with its compile-time args, defines and runtime args.
"Binaries" = distinct (ERISC, compile-time args, named args, defines).

| Case | Router ERISCs | Binaries now | Runtime fields that vary within a binary | Binaries if all runtime args were compile-time |
| --- | ---: | ---: | --- | ---: |
| Galaxy 2D mesh | 416 | 168 | packed downstream NoC x | 192 (+14%) |
| Galaxy 1D linear | 416 | 120 | none | 120 |
| Galaxy ring | 512 | 120 | none | 120 |
| Galaxy torus | 512 | 120 | packed downstream NoC x | 160 (+33%) |
| Galaxy express torus | 544 | 216 | packed downstream NoC x | 352 (+63%) |
| P150x8 2D mesh | 80 | 80 | none (every router is its own binary already) | 80 |

- **Identical on every router of a config:** the 10 sender-channel connection semaphores, the downstream buffer-index
  semaphore, the 10 downstream teardown semaphores, the relay flag. Making them compile-time costs nothing.
- **Constant within each binary:** the has-downstream mask, downstream buffer bases, registration, location-info and
  buffer-index semaphore addresses, packed NoC y. Also free to make compile-time.
- **The only per-router value:** the packed downstream NoC x coordinates (the Ethernet cores this router forwards to on
  its own chip). Compile-time would cost +14% to +63% binaries on 2D Galaxy configs.
- Not runtime args, but runtime by nature: the connected worker's semaphore NoC address and each forwarding channel's
  upstream credit address (written by the connection protocol), and the chip's mesh coordinates (routing table).

## 3. What each main loop carries, and where it lives

Where it lives is measured: PC samples on the busiest ERISCs of stage 3's rows (2D and 1D, -O3), counting time on
loads and stores through `sp` or through the loop's state pointers, plus the instruction that waits on such a load
(`pcs_stack.py`).

**E0 (`sender_erisc::run_main_loop`)**

| State | Origin | Changes | Where (cycles/packet in memory) |
| --- | --- | --- | --- |
| VC0 free-slot values (grouped read) | stream registers | per pass | registers |
| worker slot, sent, completed, acked, expected | loop | per pass | registers |
| worker connection semaphore pointer | runtime arg (fixed L1 address) | never | register (ledger E0-16: compile-time) |
| worker semaphore NoC address (64-bit) | worker's connection | per connection | stack, 0.3–0.5 |
| far ring free slots and next slot | loop | per send | the send's far-slot value at `16(sp)`, ~5 on every row |
| `ForwardingState::in_flight` | loop | per visit | stack (`184(sp)`), read every pass: 4–6 |
| forwarding channel `slot`, `sent`, `completed` (`channels[CH]`, an array in a struct) | loop | per visit | stack (`112`–`120(sp)`), ~7.5 on forwarding routers |
| upstream credit NoC address (64-bit) per forwarding channel | connection protocol | never after connect | stack, ~1.4 on forwarding routers |
| slot ranges, counter addresses, stream ids | compile time | — | immediates |

E0's memory traffic: **8–12 cycles/packet on 1-hop rows** (6.5–8.3% of its time), **16–21 on forwarding rows** (3-hop
middle, multicast, forwarded; 12.5–14.6%).

**E1 (`receiver_erisc::run_main_loop`)**

| State | Origin | Changes | Where (cycles/packet in memory) |
| --- | --- | --- | --- |
| channel state: available count, slot address, uncompleted, idle polls, pending atomic, index | loop | per packet | registers (0.0–0.2 through `sp` on every row) |
| slot records (one byte per slot) | loop | per packet | registers or immediate words |
| downstream cursor: slot address, free slots | loop | per forward | memory through the cursor pointer: ~5.5 (load + store) |
| downstream cursor: first and last slot address | runtime args (constant per binary) | never | memory, ~1.3 |
| downstream cursor: NoC coordinate, credit addresses | runtime args (NoC x per router) | never | not measurable (pre-programmed command buffer on the stateful forward) |
| mesh id and coordinates | routing table (per chip) | never | registers |
| trid tracker, credit sender | loop / compile time | per packet | registers |

E1's memory traffic: **0.0–0.2 cycles/packet on local-delivery rows; 5.6–7.4 on forwarding rows**, all in the
downstream cursors.

**Register use.** Every main loop (sender, receiver, generic, main's) saves all 13 callee-saved registers (`ra`,
`s0`–`s11`): every loop uses the whole register file, so state added to a loop competes for registers.

## 4. Static picture across configs and levels

`stack_scan.py v3q`: per ELF, the loop function's frame and its stack loads and stores (static counts, prologue and
epilogue excluded; for the generic loop the function is `kernel_main`, which also holds init code, so its counts
overstate the loop). Medians:

| Config | E0 program: frame, stack ld / st (-O3; -Os) | E1 program: frame, stack ld / st (-O3; -Os) |
| --- | --- | --- |
| BH / Galaxy 2D | sender: 256 B, 44 / 43; 41 / 35 | receiver: 144 B, 3–4 / 16; 0 / 13–14 |
| BH / Galaxy 1D | sender: 192 B, 17 / 36; 16 / 15 | receiver: 144 B, 5 / 16; 3 / 8 |
| multi-mesh VC1 | sender: 352 B, 79 / 82; 75 / 63 | receiver: 240 B, 19 / 46; 21 / 24 |
| Galaxy express | generic: 1,232 B, 127 / 142 | generic: 672 B, 37 / 71 |
| ring / torus | sender: 192–256 B, 25–58 / 39–47 | generic: 528–752 B, 30–54 / 50–73 |
| Wormhole 2D | generic: 1,056 B, 98 / 123 | — |

-O2 is within a few instructions of -O3 everywhere. **Main's router** (the Galaxy job's own builds, 204 per ERISC):
E0 `kernel_main` frame 800 B median (1,344 max), 70 stack loads; E1 528 B, 33 loads.

## 5. What this says for the shape

Observations, not decisions:
1. **Runtime args are not the problem they looked like.** Every value except the downstream NoC x coordinates is
   already the same within each binary, so it can be a compile-time constant for free. That is a host change (pass
   them as named compile-time args) plus the kernel reading constants.
2. **Coordinates can stay runtime.** They cost nothing measurable today where the forward pre-programs a command
   buffer; compile-time coordinates would add 14–63% binaries on 2D Galaxy configs.
3. **E0's cost is a data-layout choice.** The forwarding channels' state is an array of records in a struct
   (`ForwardingState::channels[]`, `in_flight`), and the far-ring cursor ends up in a stack slot: 8–12 cycles/packet on
   every row, 16–21 on forwarders. Per-channel state as separate per-channel objects (compile-time channel list) and
   the per-pass values (in-flight mask, far slot) as loop locals would let the compiler keep the per-pass part in
   registers. The register file is full, so the per-visit records may stay in memory; that part only costs on
   forwarding routers.
4. **E1 already shows the target:** with its per-channel state as loop locals it has no stack traffic at all. Its one
   memory cost is the downstream cursors (5.6–7.4 on forwarders), of which ~1.3 is the constant slot range.
5. **Coverage is the bigger gap than cycles.** Express falls back on both ERISCs, rings and tori on E1, Wormhole,
   VC2, the mux and UDM entirely. One program per role must cover all of them (any channel count, Z included,
   single-ERISC), or v3 keeps three loops.

## Files

- `consolidation/argdump.py` (`run`, `report`): runtime-arg dumps over mock clusters; dumps in
  `~/.tt-metal-v3r-builds/argdump/<case>/args.jsonl`. The host hook (`TT_FABRIC_EXP_DUMP_ARGS`) is in
  `tt_metal/fabric/compute_mesh_router_builder.cpp` (study-only, inert unless set).
- `consolidation/coverage.py`: Galaxy cases `glx_{2d,1d,ring,torus,express}` added; builds `cov/builds/v3q_{O3,O2,Os}`.
- `consolidation/stack_scan.py`: program mix and loop stack counts per build (`stack_scan.json` in each build dir).
- `consolidation/pcs_stack.py`: sampled time on stack traffic for the busiest ERISCs of stage-3 rows.
