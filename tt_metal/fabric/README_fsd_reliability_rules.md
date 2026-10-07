# FSD, PSD, and reliability mode

How a factory system descriptor (FSD) and the live physical system descriptor (PSD) interact with
`FabricReliabilityMode`, and which missing cables land in `get_downed_links()` versus
`get_unused_downed_links()`.

The mesh graph descriptor (MGD) decides which of those cables fabric actually routes. It does not
decide which cables the FSD expected.

A **discrepancy** here is a cable the FSD lists that the PSD does not have. A cable that exists only
on the PSD is not one of these records. The FSD stays the expected graph.

Without an FSD, STRICT is unchanged: a missing MGD connection is fatal, and there is no link-health
set to file it in.

A STRICT mesh graph may run with relaxed system health. A RELAXED mesh graph may not run with STRICT
system health. That combination errors before placement.

## 1. STRICT

`STRICT_SYSTEM_HEALTH_SETUP_MODE` does not allow a discrepancy.

If any FSD cable is missing from the PSD, initialization errors on every rank. That includes an
intramesh cable and an intermesh cable, and it includes a cable the MGD does not use. The error names
both scopes (`intramesh=` and `intermesh=`). A descriptor that matches the live cables is allowed
through.

The sets below are still built, so the error can see every hole, and then init stops. Callers do not
keep going on a STRICT mismatch.

## 2. Cables the mesh graph does not use

An FSD cable missing from the PSD that the MGD does not route is not a downed link. It is filed in
`get_unused_downed_links()`.

Intramesh, that is a cable whose direction on the mesh graph is `NONE` (the subtorus connection on
the 4-host pod: chip 0 chan 0 to chip 8 chan 0). Intermesh, that is a cable whose two meshes are not
a requested pair. A requested pair can still file a missing cable as unused when the live cables
already cover its count; that split is rule 4.

This split happens before routing-plane classification. It applies in relaxed mode, which is the mode
that finishes init and leaves the sets for the caller. STRICT still errors on these cables under
rule 1; it does not skip them because they are unused.

## 3. RELAXED intramesh

A routing plane is one parallel ethernet path in a direction (north, south, east, or west). The mesh
uses the smallest count in a row or column, so every hop has the same number of planes.

`RELAXED_SYSTEM_HEALTH_SETUP_MODE` keeps that count at the MGD count when the FSD has at least that
many cables. A missing FSD cable does not lower it. When the MGD count is higher than the FSD, the
count drops to the FSD count. The gap is a downgraded routing plane, not a downed link.

Counts below are per direction, on one chip. A cable is stored once in each direction.

| MGD count | FSD cables | PSD cables | Routing planes | `get_downed_links()` | `get_unused_downed_links()` |
|-----------|------------|------------|----------------|----------------------|-----------------------------|
| 2 | 2 | 1 | 2 | 1 | 0 |
| 2 | 4 | 3 | 2 | 0 | 1 |
| 4 | 2 | 1 | 2 | 1 | 0 |

Row 1: the MGD still needs the missing cable, so that one channel is downed. Planes stay at 2.

Row 2: three live cables already cover the MGD's 2 planes, so nothing is downed. The one missing
factory cable is unused. The extra live cable is not a downed link.

Row 3: the MGD is 4 and the FSD is 2, so routing planes drop from 4 to 2. Those two channels are
downgraded routing planes. They are not downed links. The one FSD cable missing from the PSD stays
downed. Nothing moves to the unused set.

The 4-host pod with intra-mesh count 2 is the first two rows. Chip 0 chan 6 to chip 4 chan 0 is the
mesh connection (row 1). Chip 15 chan 4 to chip 23 chan 4 is the torus connection (row 2). The same
pod with intra-mesh count 4 is row 3 on that mesh connection: planes are 2, and one channel per
direction stays downed.

## 4. RELAXED intermesh

Intermesh has no routing planes. Registered means a live cable pairing actually binds: `min(MGD, PSD)`.

STRICT does not use this table. After classification, `get_downed_links()` is the cables the mesh
graph still uses. An intramesh cable is checked against that mesh's policy. Intermesh is one policy
for the whole graph. An unused hole is not in that set.

A missing factory cable the MGD still needs stays in `get_downed_links()`. That is
`max(0, min(MGD, FSD) − PSD)` real cables. The rest of the factory-to-live mismatch is unused. A
live spare past the MGD count is in neither set. Channels the MGD asks for beyond the FSD,
`max(0, MGD − FSD)`, are not registered and are not link records.

Counts below are per direction. A cable is stored once in each direction.

| MGD count | FSD cables | PSD cables | Registered | `get_downed_links()` | `get_unused_downed_links()` | Not registered |
|-----------|------------|------------|------------|----------------------|-----------------------------|----------------|
| 2 | 2 | 1 | 1 | 1 | 0 | 0 |
| 2 | 4 | 3 | 2 | 0 | 1 | 0 |
| 4 | 2 | 1 | 1 | 1 | 0 | 2 |
| 8 | 4 | 1 | 1 | 3 | 0 | 4 |

On the 4-host pod each boundary has 16 factory cables. Three boundaries are fully live. Mesh 2 to
mesh 3 is missing chip 5 chan 9 to chip 29 chan 9, so that boundary is 16 factory and 15 live.

The intra-count-2 graph asks for 2, 8, 4, and 16 on the four boundaries, all RELAXED. Count 4 on the
broken boundary is already covered by the 15 live cables, so that cable is unused and 4 channels
are registered.

The intra-count-4 graph asks for 2, 6, 16, and 12, all RELAXED. Count 16 on the broken boundary
still needs the missing cable, so it stays downed and 15 live cables are registered.

A third graph keeps intra-mesh relaxed and sets every intermesh connection to STRICT, with counts
2, 8, 4, and 16. Count 4 on the broken boundary is covered by both the 16 factory cables and the 15
live cables, so that missing cable is unused and initialization succeeds.
