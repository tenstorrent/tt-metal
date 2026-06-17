# Handoff — land LTX 1x4 audio submesh (fabric-connection isolation fix), for a Loudbox agent

## Why you (loudbox) instead of galaxy
The blocker for LTX 1x4 is a **ttnn fabric-core bug**, not LTX code. Iterating it on the galaxy is
painful: multi-tenant (foreign `sulphur`/bare-`smarton` jobs), the board keeps wedging on
ethernet-FW-init, and the owner has forbidden device resets there ("don't reset; you broke the
hardware last time"). A loudbox you own is the right place to **reproduce + iterate + validate the
fabric fix fast** (your own build, free to reset). The **final LTX E2E land stays on the galaxy**
(the 22B AV pipeline needs the 4x8); the loudbox proves the fabric fix on a cheap repro first.

CONFIRM FIRST (I don't know your LB config): arch (WH/BH), chip count, mesh shape, whether you own
it (free to `tt-smi` reset + full rebuild). You need a **multi-chip** LB (≥2 chips as a mesh) to
reproduce — a single-chip box has no inter-chip fabric/CCL and cannot repro.

## The goal
Overlapping submeshes that both run CCL on shared physical chips currently DEADLOCK. Fix that so a
child submesh (LTX audio, 1x4) can run its CCL while sharing chips with a parent (LTX video, 4x8)
that already ran CCL. Then LTX `LTX_AUDIO_SUBMESH=1x4` lands.

## Root cause — VERIFIED IN SOURCE (do not re-litigate)
The fabric connection state is **per physical chip at a fixed L1 address, shared across all
submeshes and command queues** — there is no per-submesh/per-cq isolation and no clean teardown:
- `fabric_connection_sync_t {uint32 lock; uint32 initialized;}` + a 128B `WorkerToFabricEdmSender`
  object live at fixed `MEM_FABRIC_CONNECTION_LOCK_BASE` (size 144B):
  - struct: `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h:638-651`
  - addr (BH): `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:137-138` (WH: wormhole/dev_mem_map.h:119-120)
  - HAL type: `tt_metal/api/tt-metalium/hal_types.hpp:52` (`FABRIC_CONNECTION_LOCK`)
- `get_or_open_fabric_connection()` (`tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp:406-427`):
  **if `sync->initialized==1`, it REUSES the existing connection object** — no cq/submesh awareness.
  First submesh to do CCL on a chip opens+initializes it; the next submesh on that shared chip
  reuses the **stale** connection (wrong EDM endpoint/routing for ITS mesh) → deadlock.
- `release_fabric_connection()` only frees the spinlock, NOT the EDM connection → no teardown today.
- `MeshDevice::quiesce_devices()` (`tt_metal/distributed/mesh_device.cpp` ~1555-1572) was tried as a
  cross-submesh drain; it HANGS (its `wait_for_completion`/`finish_nolock` blocks on the stale
  connection that never drains).
=> **cq routing alone cannot fix this** (it's per-chip L1, not cq-keyed). Confirmed on galaxy: a child
all_gather on cq1 succeeds once, but the next CCL on a shared chip hangs; a full LTX E2E got through
ALL of video then hung on the first audio decode on cq1.

## Current state (galaxy, branch `smarton/audio-submesh-e2e`, pushed to origin)
- `7af5fbf` — lazy `_ensure_audio_submesh` (defer submesh build to first decode_audio; video runs clean).
- `079d567` — route audio submesh onto cq1 (`ttnn.command_queue(1)` wrapper; test opens with
  `num_command_queues=2`). Makes child CCL *work* but does NOT fix the cross-submesh deadlock.
- A galaxy agent is attempting the contained fix below; coordinate / don't duplicate-push.

## The fix to validate (contained first, then proper)
**Contained (try first — maybe no rebuild):** between video and the first audio CCL, when the device
is IDLE (parent synchronized, nothing in flight), **host-reset the `FABRIC_CONNECTION_LOCK` L1 region
(144B zeros: lock=0, initialized=0, + the connection object) on every worker core of the shared
chips**, forcing the child to `open()` a FRESH connection. Get the addr via HAL
(`FABRIC_CONNECTION_LOCK`); write zeros per-core (pure-Python L1 write if ttnn exposes one → no
rebuild; else a small C++ helper + build). Hook it in `pipeline_ltx.py` right after
`_ensure_audio_submesh` / before the first audio CCL, gated on `LTX_AUDIO_SUBMESH`.
**If that's insufficient** (child CCL still hangs ⇒ the EDM *router* side, not just L1, holds the
parent's connection): implement the proper fix — either (a) per-submesh fabric-connection slots
(index the sync region by submesh/cq instead of one fixed addr — touches the L1 map + the device
adapter `get_fabric_connection_sync`), or (b) a real fabric EDM teardown/re-handshake API callable
between stages. Both are core-fabric changes; the LB is where you iterate them safely.

## TDD loop (prove on the cheap repro BEFORE any LTX E2E)
1. Repro: `models/tt_dit/tests/models/ltx/test_submesh_repro.py` (`test_submesh_cq1_repro` — currently
   HANGS). Re-size the parent/child shapes to YOUR LB mesh (e.g. parent 1x4, overlapping child 1x2),
   keep the structure: parent CCL → create child + apply the fabric-sync reset on shared chips →
   child CCL → a follow-up CCL must not deadlock + clean close. The repro must go **HANG → PASS**.
   This is the fast (~minutes) iteration loop — do NOT iterate on the 20-min LTX E2E.
2. Once the repro PASSES on the LB, hand the validated fix back; the **galaxy** runs the real LTX E2E
   (`test_pipeline_distilled -k bh_4x8sp1tp0_ring`, `LTX_AUDIO_SUBMESH=1x4`, `num_command_queues=2`,
   `LTX_TRACED=0`, warm cache, `--timeout=0`, `OUTPUT_PATH`) → must reach "Saved video".

## Gates before landing (all)
- Repro PASS on LB (no hang, clean close).
- Galaxy E2E reaches "Saved video"; mp4 valid (pyav: h264 1920x1088 ~145f@24fps + aac).
- Audio PCC-vs-torch > 0.95 (`test_audio_decode_girl` with `LTX_AUDIO_SUBMESH=1x4`).
- E2E audio-decode stage faster than full-mesh ~3.2s (the point of 1x4); 1x4 audio ≈ full-mesh audio.
Land on `ltx-perf` ONLY if all pass. A core fabric C++ change should also go as a ttnn PR for
metal-team review, not silently onto a perf branch.

## Safety
- **Loudbox:** if it's YOUR dedicated box, build + reset freely there to iterate. If shared, use its
  broker/queue rules; don't disrupt others.
- **Galaxy (final E2E only):** broker MCP only; **NO device resets**; never SIGKILL a running CCL job
  (wedges); SIGTERM-only your own orphans; queue behind foreign tenants.
- Before editing source Read tt-buddy `code-comments.md`; before commits Read `commit-messages.md`
  (footer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`). Timestamps UTC+PT. Commit
  frequently; never push a hanging default.

## Key files
- `tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h:638-651` (sync struct)
- `tt_metal/hw/inc/internal/tt-1xx/{blackhole,wormhole}/dev_mem_map.h` (`MEM_FABRIC_CONNECTION_LOCK_BASE`)
- `tt_metal/fabric/hw/inc/udm/tt_fabric_udm_impl.hpp:406-441` (get_or_open / release — the reuse + the missing teardown)
- `tt_metal/distributed/mesh_device.cpp` (`create_submesh` ~609, `quiesce_devices`/`quiesce_internal` ~1555)
- `tt_metal/api/tt-metalium/hal_types.hpp:52` (HAL `FABRIC_CONNECTION_LOCK` addr)
- LTX: `models/tt_dit/pipelines/ltx/pipeline_ltx.py` (`_ensure_audio_submesh`, `_decode_audio_impl`, `_audio_cq`)
- Branch: `smarton/audio-submesh-e2e` (origin). Mission log: `/home/smarton/LAND_1X4_MISSION.md`.

## Honest caveats
- The LB mesh ≠ galaxy 4x8; the repro proves the FABRIC FIX, not LTX perf. The 22B LTX E2E likely
  won't fit/run on a small LB — keep the final E2E land on the galaxy.
- The contained L1-reset may not be enough if the EDM router holds the parent's connection
  independently of the L1 object; if so it's a true core-fabric change (per-submesh isolation or a
  teardown API) — that's the real scope, iterate it on the LB and PR it to the metal team.
