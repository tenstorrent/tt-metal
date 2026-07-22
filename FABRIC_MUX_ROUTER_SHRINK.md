# nkapre/tt-ccl-codegen-v2 — remaining work: shrink the MUX EDM router for ACTIVE_ETH

This branch adds the fabric-MUX Python binding (`ttnn.FabricMuxConfig`, commit "nanobind: expose
fabric MUX …") so the tt-ccl-codegen RS/AG line-MUX path can run. **The pybind + build are complete
and correct.** One blocker remains before the CCL codegen's `reduce_scatter num_workers=2` (fabric-MUX)
runs on Blackhole — and it lives in *this* repo's fabric code, not the codegen.

## Symptom (bh-lb-20, 8× p150b, FABRIC_2D, first on-silicon probe 2026-07-22)
```
TT_FATAL: Program size (30816) too large for kernel config buffer (25600) on ACTIVE_ETH
  → RuntimeError @ tt_metal/impl/program/program.cpp:2544  (state.offset <= max_size)
```
The RS-line-MUX brings fabric up with the tensix-MUX router variant; the resulting **EDM router
kernel** on the ACTIVE_ETH core needs 30816 B of kernel config vs Blackhole's **25600 B** budget
(~5.2 KB over). The single-worker (non-mux) path fits under the plain router. craq-sim validated the
mux path bit-identical because it does not enforce the 25600 B ACTIVE_ETH limit (sim≠silicon).

## Root cause (verified by code read)
The router's ACTIVE_ETH footprint is dominated by its `.text`, which is compiled for the **global
compile-time max sender-channel count**:
- `tt_metal/fabric/builder/fabric_builder_config.hpp:83` — `static constexpr num_max_sender_channels`
- `tt_metal/fabric/erisc_datamover_builder.cpp:235` — `FabricEriscDatamoverConfig` uses
  `num_max_sender_channels` for 2D routing "so layout is stable across all configs" (global, not per-op).

## Why NOT a blind edit
`num_max_sender_channels` is a **global `static constexpr`** shared by *every* FABRIC_2D fabric user on
a build. Lowering it changes the router for all CCL ops — including the ones that already beat ttnn on
this box (all_gather ~1.7×, reduce_scatter ~1.2×) — and risks fabric deadlock from under-provisioned
channels. It must not be reduced blind; it needs fabric-team validation across all fabric users.

## Safe fix options (fabric-team, ranked)
1. **Mux-scoped router variant / channel count** — give the tensix-MUX bring-up its own (smaller)
   sender-channel provisioning instead of the global 2D max, so only the mux path's router shrinks.
   Lowest blast radius; the right fix.
2. **Move router CT config → an L1 config tensor** — shrink the router `.text` without changing channel
   counts (a few KB off the binary; needs a router-build change).
3. **Bidirectional mux on distinct ACTIVE_ETH cores** — the CODEGEN side is architecturally blocked
   here (`reduce_scatter_builder.py:1720`: mux requires `num_links==1`, "owns the single link"), so both
   directions' routers co-land; splitting would require the mux to span two links, which contradicts its
   design. Not a codegen fix.

## What's done vs remaining
- DONE: fabric-MUX pybind + build (this branch); codegen RS-line-MUX candidate
  (`tt-ccl-codegen/docs/pending_upstream/rs_line_mux_candidate.patch`) applies + builds; correctness
  clean on the non-mux path (all_gather 10/10, RS+AR 20/20 PASS).
- REMAINING: see below — the mux path has TWO independent silicon bugs.

## Update (2026-07-22): per-axis probing found TWO blockers, and a concrete lever for #1

Probing num_workers=2 per cluster-axis on this build:
- **ring_size 4 (axis 1): the 30816 ACTIVE_ETH overflow** (config, this doc).
- **ring_size 2 (axis 0): the Line-MUX BUILDS then HANGS at exec** — an eth-sync deadlock in the
  codegen num_workers=2 mux kernel (single-worker RS passes the identical config). Separate bug, lives
  in the codegen (`reduce_scatter_line_writer.cpp` / mux handshake), not tt-metal.

**Concrete tt-metal lever for blocker #1 (the design-swarm's recommendation):** the router overflows
because it is built *single-erisc*. `tt_metal/fabric/erisc_datamover_builder.cpp` `is_fabric_two_erisc_enabled()`
(~lines 100-110) already splits senders→erisc0 / receivers→erisc1 for the DISABLED path the winning ops
use (each half fits under 25600). If the mux bring-up lands single-erisc, gate the split back on for it:

```cpp
// after computing tensix_extensions_enabled / single_erisc_dispatch:
bool mux_mode = mc.get_fabric_tensix_config() == tt::tt_fabric::FabricTensixConfig::MUX;
return arch_bh && (!tensix_extensions_enabled || mux_mode) && !single_erisc_dispatch;
```
This is byte-identical for every DISABLED (non-mux) bring-up — the all_gather/reduce_scatter wins are
unaffected. **VERIFY FIRST on-device** whether the mux run's `get_fabric_tensix_config()` is actually
`MUX` or `DISABLED`, and whether the split is already active — that decides between "no tt-metal change
needed" and this gated edit. Then fix blocker #2 (Watcher-isolate the mux writer/reader eth-sync
deadlock) before the path runs end-to-end. Full analysis:
`tt-ccl-codegen/docs/pending_upstream/rs_line_mux_SILICON_BLOCKER.md`.

## FINAL verdict (2026-07-22) — the real blocker is a fabric credit-return deadlock

On-device diagnosis with strict reset hygiene settled it:
- **Blocker #1 (30816 overflow) is a device-hygiene artifact, not a config limit.** `tt-smi -r` +
  `flush.sh` before every run -> 0 overflows in 8 runs; the `[NKAPRE-2ERISC]` log (added at
  `erisc_datamover_builder.cpp` on this branch) shows `tensix_cfg=DISABLED two_erisc=true` — the split
  is active and fits. The gated `is_fabric_two_erisc_enabled()` edit above is therefore a no-op for
  this path (kept as instrumentation only). **num_workers=1 is 5/5 PASS.**
- **Blocker #2 (num_workers=2 hang) is a genuine tt-metal FABRIC credit-return DEADLOCK** — the real
  wall. Live mid-hang Watcher dump: both mux relay cores wedge in `wait_for_empty_write_slot()`
  (`edm_fabric_worker_adapters.hpp:305`); `tt_fabric_mux.cpp:115` `forward_data()` never gets an
  outbound send slot — the EDM/router credit never returns. Reproduces on BOTH ring_size 2 AND 4.

**The fix belongs to fabric transport, not the pybind or the codegen:** mux/EDM credit servicing under
2-erisc-split + MUX mode (`tt_fabric_mux.cpp`, `erisc_datamover_builder.cpp`
`should_risc_service_{sender,receiver}_channels`). This branch's pybind is correct and complete; the
credit-deadlock fix is the remaining upstream work. Full evidence:
`tt-ccl-codegen/docs/pending_upstream/rs_line_mux_SILICON_BLOCKER.md` (Update 3).
