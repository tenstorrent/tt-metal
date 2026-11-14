# FSD ↔ PSD placement comparison (SC36 aisle-C)

Compares the **FSD route** and the **PSD route** for the same two blitz MGDs on
`SC36_32x4_revAB_subtorus_aisleC` (20 hosts, `bh-glx-120-c*`), after the SC36
`cluster_id` 110→120 alignment ([tt-cluster-descriptors #26](https://github.com/tenstorrent/tt-cluster-descriptors/pull/26)).

**Method (fully offline, no FM, both inputs through the one PSD solver):**

1. `generate_cluster_descriptor --fsd <SC36 aisle-C FSD>` → 20 FSD-derived per-host
   cluster descriptors (the **FSD** side).
2. `generate_rank_bindings` (the SAT placement solver) on the FSD-derived descriptors,
   and on the committed per-host descriptors (the **PSD** side).
3. Both fed in matching rank order (rank *i* = the same physical host).

> tt-metal has no FSD-*driven* solver yet (issue #52859); `generate_rank_bindings`
> always runs the PSD solver. So "FSD vs PSD" here is *FSD-derived descriptors vs
> committed descriptors, both through that one solver* — the correct offline comparison.

## What is aligned, what is not

| dimension | superpod (64-stage) | quad10 (10-stage) | verdict |
|---|---|---|---|
| **host / ring placement** — `rank → mesh_id → mesh_host_rank` | **64/64 identical** | **11/11 identical** | ✅ **ALIGNED** |
| ethernet connectivity — `(asic_location, channel)` edge set | identical (140 links) | identical | ✅ ALIGNED |
| `asic_locations` map | identical | identical | ✅ ALIGNED |
| `TT_VISIBLE_DEVICES` — exact per-stage match | 0/64 | 5/11 | ❌ not aligned |
| &nbsp;&nbsp;↳ of which land on the same sub-torus *shape* | 30/64 | 8/11 | — |

**One-line verdict:** the FSD and PSD routes place **every ring stage on the same
physical host**; they disagree only on the **local `TT_VISIBLE_DEVICES` chip labels
inside each galaxy**, and that disagreement is *not* a connectivity difference (see below).

## rank → mesh_id (ALIGNED — byte-identical)

`rank → mesh_id → mesh_host_rank` is the placement that matters: it says which ring
stage (rank) runs as which mesh, on which host slot. It is **byte-identical** between
the FSD and PSD routes for both MGDs.

- **superpod:** a closed 64-stage ring; rank *i* → `mesh_id` *i* for all 64, both routes.
- **quad10:** 10-stage heterogeneous ring; the mapping below is identical on both routes,
  including the split stage (`mesh_id 7` across `mesh_host_rank` 0 and 1).

| rank | mesh_id | mesh_host_rank | chips | mesh role |
|---|---|---|---|---|
| 0 | 0 | 0 | 32 | 4x8 full galaxy (forced) |
| 1 | 1 | 0 | 8 | 4x2 sub-galaxy (free) |
| 2 | 2 | 0 | 16 | 4x4 full-width even (forced) |
| 3 | 3 | 0 | 8 | 4x2 sub-galaxy (free) |
| 4 | 4 | 0 | 8 | 4x2 sub-galaxy (free) |
| 5 | 5 | 0 | 8 | 4x2 sub-galaxy (free) |
| 6 | 6 | 0 | 8 | 4x2 sub-galaxy (free) |
| 7 | 7 | 0 | 8 | 4x4 split across 2 hosts (forced) |
| 8 | 7 | 1 | 8 | 4x4 split across 2 hosts (forced) |
| 9 | 8 | 0 | 16 | 4x4 full-width odd (forced) |
| 10 | 9 | 0 | 8 | 4x2 sub-galaxy (free) |

## TT_VISIBLE_DEVICES (NOT aligned — local chip labels)

`TT_VISIBLE_DEVICES` is the list of **local 0–31 chip indices** the stage pins inside
its galaxy. It differs because the two descriptors *number the chips differently* —
**not** because the wiring differs. Proven: the ethernet `(asic_location, channel)` edge
multiset is byte-identical (140 links); `asic_locations` is identical. What differs is
bookkeeping the FSD→descriptor converter emits:

| | committed (PSD) descriptor | FSD-derived descriptor |
|---|---|---|
| chip-index 0–31 numbering | live-discovery order (scrambled) | sequential by slot |
| `chip_unique_ids` | real ASIC serials | synthetic |
| board grouping | 1 UBB board of 32 | 4 boards of 8 |

A 32-chip galaxy is **highly symmetric** (all 32 chips share the same 1-hop connectivity),
so each sub-galaxy stage has several equivalent seatings — only **two** distinct
half-galaxy shapes exist: asic-locs `{1,2,5,6}` vs `{3,4,7,8}`. The SAT solver enumerates
candidate tiles **in chip-index order** and has **no canonical geometric tie-break**, so a
different numbering flips which equivalent half it seats. The *forced* stages (full
galaxy, full-width slices) are identical on both routes; only the *free* sub-galaxy stages differ.

### quad10 — full per-stage TT_VISIBLE_DEVICES

| rank | mesh_id | chips | FSD `TT_VISIBLE_DEVICES` | committed `TT_VISIBLE_DEVICES` | result |
|---|---|---|---|---|---|
| 0 | 0 | 32 | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31` | `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31` | ✅ identical |
| 1 | 1 | 8 | `0,1,4,5,16,17,20,21` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 2 | 2 | 16 | `2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31` | `2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31` | ✅ identical |
| 3 | 3 | 8 | `10,11,14,15,26,27,30,31` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 4 | 4 | 8 | `8,9,12,13,24,25,28,29` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 5 | 5 | 8 | `0,1,4,5,16,17,20,21` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 6 | 6 | 8 | `2,3,6,7,18,19,22,23` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 7 | 7 | 8 | `18,19,22,23,26,27,30,31` | `18,19,22,23,26,27,30,31` | ✅ identical |
| 8 | 7 | 8 | `2,3,6,7,10,11,14,15` | `2,3,6,7,10,11,14,15` | ✅ identical |
| 9 | 8 | 16 | `2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31` | `2,3,6,7,10,11,14,15,18,19,22,23,26,27,30,31` | ✅ identical |
| 10 | 9 | 8 | `8,9,12,13,24,25,28,29` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |

**quad10:** 5/11 exact, 8/11 same shape. The 5 exact matches are precisely the
forced stages (32-chip galaxy, 16-chip full-width slices, the split 4x4); the 6 differing
stages are the free 4x2 sub-galaxy tiles.

### superpod — full per-stage TT_VISIBLE_DEVICES (64 stages)

Every stage is an 8-chip `4x2` sub-galaxy tile, so none is forced — all 64 are free choices
among the two equivalent half-galaxy shapes.

| rank=mesh_id | FSD `TT_VISIBLE_DEVICES` | committed `TT_VISIBLE_DEVICES` | result |
|---|---|---|---|
| 0 | `10,11,14,15,26,27,30,31` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 1 | `10,11,14,15,26,27,30,31` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 2 | `8,9,12,13,24,25,28,29` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 3 | `0,1,4,5,16,17,20,21` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 4 | `8,9,12,13,24,25,28,29` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 5 | `10,11,14,15,26,27,30,31` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 6 | `2,3,6,7,18,19,22,23` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 7 | `2,3,6,7,18,19,22,23` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 8 | `10,11,14,15,26,27,30,31` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 9 | `8,9,12,13,24,25,28,29` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 10 | `0,1,4,5,16,17,20,21` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 11 | `8,9,12,13,24,25,28,29` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 12 | `10,11,14,15,26,27,30,31` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 13 | `10,11,14,15,26,27,30,31` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 14 | `2,3,6,7,18,19,22,23` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 15 | `0,1,4,5,16,17,20,21` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 16 | `8,9,12,13,24,25,28,29` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 17 | `0,1,4,5,16,17,20,21` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 18 | `2,3,6,7,18,19,22,23` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 19 | `10,11,14,15,26,27,30,31` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 20 | `2,3,6,7,18,19,22,23` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 21 | `0,1,4,5,16,17,20,21` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 22 | `8,9,12,13,24,25,28,29` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 23 | `0,1,4,5,16,17,20,21` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 24 | `2,3,6,7,18,19,22,23` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 25 | `10,11,14,15,26,27,30,31` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 26 | `8,9,12,13,24,25,28,29` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 27 | `0,1,4,5,16,17,20,21` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 28 | `2,3,6,7,18,19,22,23` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 29 | `10,11,14,15,26,27,30,31` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 30 | `2,3,6,7,18,19,22,23` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 31 | `0,1,4,5,16,17,20,21` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 32 | `8,9,12,13,24,25,28,29` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 33 | `0,1,4,5,16,17,20,21` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 34 | `2,3,6,7,18,19,22,23` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 35 | `10,11,14,15,26,27,30,31` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 36 | `8,9,12,13,24,25,28,29` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 37 | `0,1,4,5,16,17,20,21` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 38 | `8,9,12,13,24,25,28,29` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 39 | `0,1,4,5,16,17,20,21` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 40 | `8,9,12,13,24,25,28,29` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 41 | `10,11,14,15,26,27,30,31` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 42 | `2,3,6,7,18,19,22,23` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 43 | `10,11,14,15,26,27,30,31` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 44 | `2,3,6,7,18,19,22,23` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 45 | `0,1,4,5,16,17,20,21` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 46 | `8,9,12,13,24,25,28,29` | `0,1,4,5,24,25,28,29` | ⚠️ same shape, diff chips |
| 47 | `0,1,4,5,16,17,20,21` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 48 | `2,3,6,7,18,19,22,23` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 49 | `10,11,14,15,26,27,30,31` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 50 | `8,9,12,13,24,25,28,29` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 51 | `0,1,4,5,16,17,20,21` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 52 | `2,3,6,7,18,19,22,23` | `10,11,14,15,18,19,22,23` | ⚠️ same shape, diff chips |
| 53 | `2,3,6,7,18,19,22,23` | `2,3,6,7,26,27,30,31` | ⚠️ same shape, diff chips |
| 54 | `10,11,14,15,26,27,30,31` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 55 | `2,3,6,7,18,19,22,23` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 56 | `10,11,14,15,26,27,30,31` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 57 | `8,9,12,13,24,25,28,29` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |
| 58 | `0,1,4,5,16,17,20,21` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 59 | `8,9,12,13,24,25,28,29` | `8,9,12,13,16,17,20,21` | ⚠️ same shape, diff chips |
| 60 | `10,11,14,15,26,27,30,31` | `0,1,4,5,24,25,28,29` | ⚠️ diff shape |
| 61 | `2,3,6,7,18,19,22,23` | `8,9,12,13,16,17,20,21` | ⚠️ diff shape |
| 62 | `0,1,4,5,16,17,20,21` | `10,11,14,15,18,19,22,23` | ⚠️ diff shape |
| 63 | `8,9,12,13,24,25,28,29` | `2,3,6,7,26,27,30,31` | ⚠️ diff shape |

**superpod:** 0/64 exact, 30/64 land on the same half-galaxy shape, 34/64 on the
complementary half. All 64 remain on the **same host** (`rank → mesh_id` identical).

## Conclusion

- **Host/ring placement is fully aligned** between FSD and PSD — same stage, same host, everywhere.
- The only divergence is **local `TT_VISIBLE_DEVICES` labels**, caused by the descriptor
  chip-numbering/board-grouping convention over a symmetric galaxy with a tie-break-free solver —
  **not** a connectivity difference.
- On real hardware this washes out: FM resolves `TT_VISIBLE_DEVICES` from live UMD, not the mock descriptor.
- To make the labels identical too: either (a) have `generate_cluster_descriptor_from_fsd` preserve
  the committed chip-numbering + 1-board-of-32 grouping, or (b) add a canonical geometric tie-break
  to the SAT solver so symmetric choices resolve deterministically. (b) is the more robust fix.
