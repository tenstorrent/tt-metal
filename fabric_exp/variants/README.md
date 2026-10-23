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
