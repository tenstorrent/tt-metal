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
- REMAINING: option 1 or 2 above (fabric-team), then re-probe `reduce_scatter --fleet-tuple bh:8
  --num-links 1 --num-workers 2` — expect it to fit under 25600 B and PCC-pass, then measure the mux
  perf and extend to all_gather line-MUX to beat all_reduce.
