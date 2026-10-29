# Router variant snapshots

Each directory holds the files that differ between router versions under test, copied before the version's runs so it
can be swapped back in for A/B runs (the session scratchpad is not durable). Restore by copying each file back to its
path in the tree:
- `fabric_erisc_router_v2.cpp` -> `tt_metal/fabric/impl/kernels/edm_fabric/`
- `fabric_edm_packet_transmission.hpp` -> `tt_metal/fabric/hw/inc/edm_fabric/`

| Dir | Version | RESULTS.md |
| --- | --- | --- |
| `s66` | The user's §66: §65 plus the immediate-send fast path, idle steps completing all slots but the newest, no `[[unlikely]]`, and no `!any_sender_channel_serviced` guard | §66 |
| `s63r` | §63 rebuilt from the session transcript (same line numbers as §63; reproduces §63's results) | §66 |
| `s63g` | `s63r` + the `!any_sender_channel_serviced` guard (byte-identical ERISC `.text` here); the recommended version | §66 |
| `s66u` | `s66` + `[[unlikely]]` on the admission drain (diagnostic) | §66 |
| `s66d` | `s66` with the drain back at the end of the inner loop (diagnostic) | §66 |
| `s62x` | `s63g` with `can_defer_local_atomic_flush()` compiled to false: §62 behavior (baseline) | §67 |
| `s67` | The user's §67: idle drain in the completion `else` arm, fused-only deferral in the header, E/W drain after `forward_2d_dispatch`, guard restored | §67 |
| `s67a` | `s67` with one drain before the forward on every router | §67 |
| `s67b` | `s67` with the idle drain back at §63's place | §67 |
| `s67c` | `s63g` router + `s67` transmission header | §67 |

`<variant>/elf/<config hash>/` holds the ERISC ELFs of the 4352 / mix / flushed-signal configs from that variant's run (the
hash names the compile-time config, so the same hash pairs across variants). The transmission header is §63's in all
variants except `s67*`. §62, §65 and the pre-atomic "before" were lost with the scratchpad; their results stay in `results/`.
| `s69` | The user's FLUSH_RELAXED opt-in: `NocFlushType` header field (`fabric_edm_packet_header.hpp`), packed `uint32_t` pending state with trid-window bypass for plain writes, completion guard, and the test kernel's `kActiveFlushMode` (`tt_fabric_test_kernels_utils.hpp`, -> `tests/tt_metal/tt_fabric/test_infra/kernels/`). `s63g` router files compile against this header unchanged and treat RELAXED as STRICT | §70 |
| `s69a` | `s69` without the completion guard; instead a plain write bypasses the pending relaxed atomic only while its slot's trid is outside the atomic's 4-trid window (at most 12 bypasses). Diagnostic | §70 |
| `s71` | The user's exact-C1 relaxed flush: global check until a write bypasses, then an acked-trid pointer walked over all 16 trids; no completion guard, bypass bounded by the trid window; `can_defer` requires 16 receiver slots and 16 trids; RELAXED on scatter + atomic via bit 6 of `chunk_encoding` (`fabric_edm_packet_header.hpp`, `api_common.h` -> `tt_metal/fabric/hw/inc/`) | §71 |
| `s71b` | `s71` with the trid walk replaced: at the first bypass, global outstanding (SW acked − `NIU_MST_WR_ACK_RECEIVED`) is compared with trid_s's outstanding count (global read first); once equal only trid_s is watched (pending = `0x100 \| s<<4 \| s`, so the drain waits on trid_s alone). Diagnostic | §71 |
| `e0seg` | The tree's router (s71) with E0 loop-segment timers on phase IDs E0 never closes (loop top 16, worker can-send 32, pre-send 2, completion step 64, connection check 128, forwarding channels 256, iteration body 512, worker visit 2048) plus a profiler copy (`fabric_phase_profiler.hpp`, -> `tt_metal/fabric/hw/inc/edm_fabric/telemetry/`) that counts only the timed phase (the unmodified profiler is kept in `s71/` for the restore). Level 0 builds to s71's instructions. Diagnostic | §73 |
| `s74` | The user's out-of-order-safe two-slot check: at the first bypass, global outstanding, trid_s and trid_p read in one group until g == s + p; pending keeps the non-zero of s / p (`0x100 \| b<<4 \| a`), narrowing to b if a acks first; standalone atomic's last write trid = `transaction_id − 1`; bypass bound > 1; `can_defer` with >= 16 receiver slots. Header, `api_common.h`, test kernel = s71's | §74 |
| `s74c` | The committed tree (5ef745cb2a0): s74 after the pre-commit formatting; router + `fabric_erisc_datamover_channels.hpp` (-> `tt_metal/fabric/hw/inc/edm_fabric/`). Control for §77 | §77 |
| `s77` | The user's §77: no `SlotAddresses` table / buffer index in `StaticSizedSenderEthChannel` (stride advance everywhere), compile-time free-slots stream ids in `sender_channel_try_send` and both sender steps, no `[[unlikely]]` on the pending-atomic and TXQ-status branches | §77 |
| `s77u` | `s77` with the five `[[unlikely]]` restored (diagnostic) | §77 |
| `s77t` | Slot-table removal only: `s74c` router + `s77` channels (diagnostic) | §77 |
| `s77s` | Compile-time stream ids only: `s77u` router with the by-stride advance call kept + `s74c` channels. The recommended version | §77 |
| `s77sn` | `s77s` without the five `[[unlikely]]` (diagnostic) | §77 |
| `s78` | The user's §78: `s77s` + the pre-send payload size read as the 4 B word at offset 40 and masked after the TX-queue test + 4 B `LeanStrideSenderEthChannel` with the compile-time slot advance (router + `fabric_erisc_router_speedy_path.hpp`). Every variant from here on also carries the speedy path (`s74c`–`s77sn` got the committed one) | §78 |
| `s78w` | Word load only: `s77s` + the word load (diagnostic) | §78 |
| `s78l` | Lean channels only: `s77s` + the lean channel type and compile-time advance, `s78` speedy path (diagnostic) | §78 |
| `s78n` / `s77si` | `s78` / `s77s` with `run_fabric_edm_main_loop` `noinline` on both ERISCs (diagnostic) | §78 |
| `s78n0` / `s77si0` | `s78` / `s77s` with the main loop out of line on E0 only (`if constexpr (MY_ERISC_ID == 0)` picks a `noinline` wrapper) (diagnostic) | §78 |
| `s79` | the user's: `s77s` + worker connection transitions in `noinline` cold helpers + grouped free-slots mask 0x7FF | §79 |
| `s79c` | `s77s` + the cold connection helpers only (diagnostic) | §79 |
| `s79m` | `s77s` + the 0x7FF grouped free-slots mask only; best E/W small writes so far (recommended baseline) | §79 |
| `s80` | the user's: `s79m` + forwarding sender channels de-templated (`ForwardingSenderChannelState` per channel, `noinline` init, one runtime step in a `#pragma GCC unroll 1` loop); small writes +1.8 to +2.9 | §81 |
| `s79mc` | `s79m` with the two unused cold-helper templates removed (the user's cleanup); same code as `s79m` on every config. The consolidation baseline | §82 |
| `s79mc_reorder` / `s79mc_field` | `s79mc` + one benign edit from `consolidation/perturb.py` (two sender-channel members swapped / one member added, set at init); within noise | §82 |

## Strict E0 Codegen & Register-Allocation Rules (from §§77–81 Disassembly Analysis)

> **Status (2026-10-24, §82):** these are observations about GCC 15.1 at -O3 on s77–s80, not rules for writing router
> source. The consolidation work (`consolidation/PLAN.md`) does not let source rely on allocator, scalarization or
> peeling behavior. Rule 1's mechanism (a cold call inside the loop) differs from RESULTS §79, which places the calls
> outside the loop and the +17 instructions in duplicated code. Phase 0 found s79m flat under allocator, scheduling,
> GCSE and loop-invariant-motion changes, but +20% at -Os.

On RV32IM, E0's inner 32-iteration loop has **27 allocatable registers** (`32` minus `zero, ra, sp, gp, tp`: `12` callee-saved `s0..s11` and `15` caller-saved `a0..a7, t0..t6`). Disassembling `s77s`, `s77t`, `s78`, `s78n0`, `s77si0`, `s79m`, `s79c`, `s79`, and `s80` establishes four mechanical rules that govern E0 performance:

1. **Rule 1 — Keep the 32-Iteration Inner Loop 100% Call-Free (Leaf-Loop Rule):**
   - Never place any `noinline` function call (`jal` / `jalr`) inside the `for (size_t i = 0; i < iterations_between_ctx_switch_and_teardown_checks; i++)` loop, even inside an `[[unlikely]]` cold branch.
   - When the inner loop has zero calls (`s77s`, `s79m`), GCC's `IRA` uses all 15 caller-saved registers (`a0..a7, t0..t6`) alongside `s0..s11` to hold loop-invariant hardware base addresses and constants (`0xffb90000` `ETH_TXQ0`, `0xffb20000` `NOC0`, `0x16000` L1 completion counters base, `0xffb00000` NoC SW counters base, `0xffb01000` `did_something`, `64`, `2`, `1`) across all 32 iterations.
   - Adding even a cold `jal` inside the loop (`s79c`, `s79`) causes `IRA` to charge caller-save spill/restore costs to every value live across the loop back-edge, evicting those hardware bases from caller-saved registers and forcing +17 `lui`/`li` re-materializations per pass inside the loop (`8ef0`, `8f50`, `8f58`, `8f7c`, `9008`, etc.).
   - Any `noinline` helper must sit **outside** the 32-iteration loop (either at `kernel_main` startup or in the outer loop between 32-iteration bursts).

2. **Rule 2 — Fit Loop Masks and Constants in 12-Bit Signed Immediates (`-2048 .. +2047`):**
   - Constants used inside the loop that exceed RISC-V's 12-bit signed immediate (such as `0x1FFFF` for `(1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1` in `s77s`, or the four 32-bit compile-time `last_slot_addr` constants in `s78`) get hoisted by `-ftree-loop-im` into pseudo-registers before the loop.
   - When the 27-register budget is full, `IRA` spills those hoisted constants to the stack and reloads them on every pass (`lw t6, 56(sp)` in `s77s`; `lw t1, 80(sp)` in `s78`), clobbering scratch registers and displacing hot state.
   - Narrowing the free-slots mask to `0x7FF` (`s79m`, valid since `num_buffers <= 0x7FF` and bits 11–16 are always zero) turns `lw t6, 56(sp); and` into a single `andi t1, t1, 2047`, freeing a stack slot and register so `channel_connection_established[0]` stays in `s10` without the per-pass `li a6, 1; sw a6, 12(sp)` store (`123.8 -> 121.7` cyc/pkt on E/W 16 B writes).

3. **Rule 3 — Steering `-ftree-sra` by Moving Channel State into Memory Does Not Promote `ch0`'s Counts into Registers:**
   - Around 18 registers in the inner loop are needed for shared hardware bases, constants, VC0 outbound receiver state, loop counters, and `free_slots_raw[0..3]`, leaving ~9 registers for channel-specific state and temporaries.
   - Because the outer loop (`while (continue_running_main_run_loop)`) contains calls (`execute_pause_command` and the base-firmware `jalr` inside `run_coordinated_context_switch_to_base_firmware`) and inlines 7 context-switch spin loops at the same loop depth, any persistent channel counter can only live in the **12 callee-saved registers (`s0..s11`)**.
   - Three separate attempts to steer `-ftree-sra` failed to help `ch0`:
     - **Shrinking `StaticSizedSenderEthChannel` (`s77t`, `s78l`, `s78`):** Caused `-ftree-sra` to promote all 4 channels' `cached_next_buffer_slot_addr` into registers (`a1, s8, s10, s11`), evicting `0xffb90000` (TXQ0 base), `0x16000` (L1 counter base), and `64` (`+11.5` cyc/pkt on small writes).
     - **Moving `ch1..3` into a `noinline`-initialized struct array stepped by `#pragma GCC unroll 1` (`s80`, §81):** GCC's early GIMPLE `cunroll` pass (`-fpeel-loops`) still peeled the 3-iteration loop into 3 copies at fixed stack offsets (`352/356`, `400/404`, `448/452`). Worse, the 5 `s`-registers freed from `ch1..3`'s counts did **not** go to `ch0`'s `packets_sent` (`16(sp)`) or `completions_received_and_processed` (`8(sp)`): because `ch0`'s counts sit behind conditional branches (`< 50%` static weight per pass), `IRA` gave the freed `s`-registers to values shared across all 4 peeled channels (`64` in `s1`, `0x2000` in `s3`, `remote_address_base` in `s7`, `slot_size_bytes` in `s8`) and outer-loop context-switch variables (`s0, s2, s5, s10`), while `ch1..3` now paid 2 stack loads each per idle pass (`+1.8 to +2.9` cyc/pkt on small writes).

4. **Rule 4 — Keep `run_fabric_edm_main_loop` Inlined:**
   - Marking `run_fabric_edm_main_loop` `noinline` (`s78n`, `s78n0`, `s77si`, `s77si0`) consumes 4–6 registers in the loop frame just to hold base pointers to `kernel_main`'s stack objects and consistently regresses 1D neighbor exchange (`super_speedy_mode`) by `+6.6 to +10.2` cyc/pkt.

| `v3qm` | Measurement only: `v3q` with the post-merge packet-header include path (`hostdev/`), nothing else; builds all 336 post-merge production configs; the census's v3 column | §98 census |
| `mainv2cap` | Capture only: main's router (`fabric_erisc_router.cpp` as merged, 6d8a103) in the v2 slot, so post-merge mock runs (`coverage.py mock`, `TT_COV_DIR=cov_m`) log every config type; never measured | §98 |
