# Task plan — capture layer (v1)

Scope: given a fabric debug **manifest** already on disk, attach with **ttexalens**, peek only the
ethernet cores that the manifest names as fabric routers, and write a **snapshot** JSON that the
viewer can join onto the manifest.

Parent plan: `visualizer_plan.md` (this is task 3). Topology emit (task 0+1) is already done; this
task is the first consumer of that file.

**Explicitly deferred** (later tasks, same snapshot schema where possible):

- per-router ABI from the EDM builder (named stream IDs, VC counts, credit transport, CT flags)
- typed L1 decode (`routing_l1_info_t`, circular buffers, packet headers)
- stall-score heatmap as a first-class derived field (v1 stores raw numbers; polarity is documented)
- multi-sample polling / recording CLI (schema allows `samples[]`; v1 writes one sample)
- frontend (task 2)
- SSH/SCP (or hostname-list) gatherer that runs capture on every box and copies files back.
  The *end result* — one visualizer showing every host's peeks — is in scope; automating the copy is not.

Those are additive: the id that names **one fabric router** stays `(mesh_id, chip_id, eth_chan)`,
and that router's object later grows `l1` / named `abi` values without renaming the live dump.

That triple is **not** “one capture file” and **not** a VC buffer. It names one ethernet **tile**
on one chip — one ERISC, one physical cable end, one hop in the picture:

- A Wormhole chip has 16 ethernet tiles (`eth_chan` 0–15). T3K only uses some of them as fabric
  routers (40 across 8 chips).
- Each of those tiles runs **one** fabric router kernel. Sender channel 1, receiver VC0, stream 22,
  etc. all live **inside** that one router. They are fields under `streams` (and later `l1`), not
  extra keys.
- So a snapshot **file** is “everything this host peeked at this moment.” A snapshot **id** is
  “this one ethernet core.” Clicking a west hop in the UI looks up that hop's `src` triple and
  shows that router's streams.

---

## 1. What v1 must answer

A hang-debug capture, with **no live tt-metal process**, must:

1. **Find the peek targets from the manifest**, not from ttexalens's eth-block list or idle/active
   classification. Iterate chips with `is_local == true` and that chip's `routers[]`. Skip chips
   this process does not own (`is_local == false`: other host's ASICs). This is not UMD's
   "remote chip" on a T3K (those are still on this machine and stay `is_local: true`).
2. **Read live numbers** from those ERISCs: core health (reset, wall clock) and NOC overlay
   `BUF_SPACE_AVAILABLE` for fabric streams 0–29.
3. **Write a snapshot** keyed the same way as the manifest's `endpoint` (`mesh_id`, `chip_id`,
   `eth_chan`) so a later viewer can join without guessing coordinates.
4. **Survive partial failure**: one unreadable core is recorded as an error on that router, not a
   process abort. Hang debug is exactly when some cores are dead.

Non-goal for v1: interpreting SRAM, naming streams from CT flags, or coloring links. The snapshot is
raw peeked state plus enough identity to match a manifest.

---

## 2. How this uses the manifest we already emit

Default input, written at fabric init:

`generated/fabric/fabric_debug_manifest_rank_<rank+1>_of_<world_size>.json`

(same directory as the other fabric artifacts; `get_logs_dir()` / `generated` / `fabric`).

| Manifest field | Capture use |
|---|---|
| `kind` / `manifest_version` | Refuse files that are not v1 manifests |
| `run.arch` | Pick WH vs BH overlay register indices (`BUF_SPACE_AVAILABLE` is 64 vs 297) |
| `run.host_rank` / `run.mpi_rank` / `world_size` | Copy into the snapshot. `host_rank` is the mesh-graph slice this host owns; `mpi_rank` is the MPI process that wrote the file. They are not the same on a split cluster. |
| `run.fabric_config` | Copied for identity only; capture does not change what it peeks |
| `chip.is_local` | Skip if false |
| `chip.physical_chip_id` | ttexalens `context.devices[id]` |
| `router.eth_chan` | Index into `device.get_block_locations("eth")` |
| `router.logical_core` | Assert against `loc.to("logical")[0]` before peeking; mismatch is a capture error, not a silent wrong tile |
| `router.virtual_core` | Documentation / future addressing; ttexalens peeks with the `OnChipCoordinate` from the eth-block list, which is already the right object. Metal "virtual" == ttexalens `"translated"`. |
| `links[]` | Not required to *perform* a peek. Copied or referenced only so the snapshot can stand next to the same edge ids. v1 does not need to duplicate the full link list. |

**Do not** use:

- `device.idle_eth_blocks` / `active_eth_blocks` to choose targets. On single-host T3K those
  happened to match the 40 fabric routers; on multi-host they do not (cross-host hops look idle).
- `get_block_locations("eth")` as the iteration set (128 tiles on T3K). Use it only as the
  **lookup table** indexed by `eth_chan`.

Proven on T3K: 8 chips × 16 eth tiles = 128 locations; manifest names 40 routers; `eth_chan` indexes
those locations and logical/translated coords match.

---

## 3. ttexalens surface (reuse, do not grow the dumper)

Reuse from `fabric_erisc_dumper.py` / `fabric_erisc_utils.py` / `fabric_erisc_constants.py`:

- `init_ttexalens()`
- `read_from_device(coord, address, device.id, nbytes, context)` → 4 bytes, little-endian u32
- `get_stream_reg_address(stream_id, "BUF_SPACE_AVAILABLE", arch)`
- `normalize_architecture` / `detect_device_architecture` (cross-check against `run.arch`)
- `ERISC_REGISTERS` (`ETH_RISC_RESET`, `ETH_RISC_WALL_CLOCK_0/1`)
- `STREAM_REGISTER_MASK` (17-bit mask on overlay values, same as the dumper)

Do **not** subclass `ERISCDumper` or add flags to the CLI dumper. That tool enumerates cores,
filters idle, and prints human matrices. Capture is a library + small CLI that is driven by JSON.

Coordinate trap (already verified, must not regress):

- Peek with the `OnChipCoordinate` from `get_block_locations("eth")[eth_chan]`.
- Never construct a coordinate from `noc0` using the manifest's `virtual_core` without going through
  ttexalens's `translated` space.

---

## 4. Snapshot schema (v1, the new contract)

New file: `tt_metal/fabric/debug/visualizer/schema/fabric_debug_snapshot_schema.json`

This is the second half of the stability boundary. Capture writes it; the viewer will read it;
ControlPlane never sees it.

Draft shape (keys, not byte-golden):

```json
{
  "snapshot_version": 1,
  "kind": "fabric_debug_snapshot",
  "captured_at": "ISO-8601",
  "manifest": {
    "path": "...",
    "manifest_version": 1,
    "run": {
      "arch": "...",
      "fabric_config": "...",
      "host_rank": 0,
      "mpi_rank": 0,
      "world_size": 1
    }
  },
  "samples": [
    {
      "capture_time": "ISO-8601",
      "routers": [
        {
          "id": { "mesh_id": 0, "chip_id": 1, "eth_chan": 0 },
          "physical_chip_id": 0,
          "ok": true,
          "error": null,
          "health": {
            "reset": 1,
            "wall_clock": 123456789
          },
          "streams": {
            "0": { "buf_space_available": 0 },
            "22": { "buf_space_available": 8 }
          }
        }
      ]
    }
  ]
}
```

Rules:

- `id` is the same `endpoint` object as the manifest: one ethernet tile / fabric router. Join key
  for the viewer. Not a VC or sender/receiver channel.
- `streams` is keyed by **numeric stream id as a string** (JSON object keys are strings). Do not
  store dumper labels (`sender_ch0`, …). Those labels are ABI and will rot; ABI emit (task 4)
  attaches names later.
- v1 peeks streams **0–29** `BUF_SPACE_AVAILABLE` plus the three health registers. That matches
  what the dumper already knows how to read and is enough for a later heatmap.
- `ok: false` + `error` string for cores we intended to peek but could not (missing device, chan
  out of range, logical_core mismatch, read exception, in-reset if we choose to still record it).
- One element in `samples[]` for v1. The array exists so polling does not break the schema.
- Polarity is **not** stored per stream in v1. Document in the schema description:
  - streams 14–29: high free-slots = healthy
  - streams 0–13: low/zero remote space = healthy (acks/completions/pkts_sent)
  A later decoder or the UI applies that, or ABI replaces it.

`run.arch` from the manifest is authoritative for overlay indices. Cross-check it against
`detect_device_architecture()` and warn on disagreement, but keep the manifest value: it comes
from `tt::ARCH` at fabric init, while `detect_device_architecture()` has its own fallback chain
that lands on wormhole when it cannot identify the device. An arch that `normalize_architecture`
does not recognize aborts the capture rather than defaulting, because the wrong arch reads a real
but unrelated stream register and produces a snapshot that looks plausible and is wrong.

Do not treat `manifest.path` as identity. After someone copies files off the cluster, that path is
wrong. Identity is the `run` fields inside the JSON.

---

## 4a. Several hosts, one visualizer

Capture v1 still writes **one snapshot on the machine it runs on**. The visualizer later opens a
**folder** of those files (plus the matching manifests) and draws one cluster. There is no extra
combine-into-one-JSON step on the servers.

Workflow (same idea as tmux/xpanes today):

1. On each host: run capture against that host's manifest.
2. Copy every manifest + snapshot into one directory (by hand or scp).
3. Open the visualizer on that directory.

How numbers land on the right box in the picture: every router is already named
`(mesh_id, chip_id, eth_chan)` in both files. That name is the same on every host. The viewer does
not use hostnames or original file paths.

Drawing the map from several manifests:

- Take chips/links from all files together.
- When two files describe the same chip, use the one marked `is_local: true` (that is the copy that
  actually lists ethernet cores). The other file's stub is empty on purpose.
- Missing peeks stay grey. Capturing only 1 of 2 hosts is allowed.

Refuse to mix files from different fabric inits (`fabric_config` / arch disagree). If two snapshots
both contain numbers for the same `(mesh_id, chip_id, eth_chan)`, that is a bug, not a merge.

**MPI process number vs mesh host rank.** The filename already uses MPI rank
(`rank_1_of_2.json`). JSON `run.host_rank` is the mesh-graph host id. They can differ on a split
cluster. `run.mpi_rank` is required on every manifest and copied into every snapshot so identity
survives renaming. Do not key the viewer off filenames.

**Time.** Do not try to align clocks across hosts. Hang debug assumes the fabric is mostly stuck, so
snapshots taken a bit apart still describe the same jam. If something still moves, it is likely
bouncing between a few values, not a live trace. Note this in the tool README when that exists;
the snapshot schema does not need synchronized timestamps.

SSH/hostname automation is future work in `visualizer_plan.md`. Same files, just collected for you.

---

## 5. Code layout

All new Python lives under the visualizer tool root, not next to the dumper:

```
tt_metal/fabric/debug/visualizer/
  schema/
    fabric_debug_manifest_schema.json      # already exists
    fabric_debug_snapshot_schema.json      # this task
  capture/
    __init__.py
    peek.py          # ttexalens attach + per-router reads
    snapshot.py      # assemble JSON, write file
    cli.py           # argparse entry — this is what you run
```

How to run it: invoke `cli.py` (or a later `python -m` equivalent). That is the capture tool.
One process, one host, one manifest in, one snapshot out. Same as running `fabric_erisc_dumper.py`
today, except the core list comes from the manifest.

```bash
python3 tt_metal/fabric/debug/visualizer/capture/cli.py \
  --manifest generated/fabric/fabric_debug_manifest_rank_1_of_1.json \
  -o generated/fabric/fabric_debug_snapshot_rank_1_of_1.json
```

`--manifest` is required. Capture never invents topology or guesses which rank file to use.

Imports from the sibling dumper package: add `tt_metal/fabric/debug` to `sys.path` the same way
other debug scripts do, and import `get_stream_reg_address`, constants, architecture helpers.
Do not copy overlay address math.

---

## 6. Validation

Offline (no devices):

- Snapshot schema self-validates (Draft 2020-12).
- A tiny fixture manifest (one local chip, one router) plus a fake peek function produces JSON that
  validates and preserves the endpoint id.

On this T3K (after any fabric init that writes the manifest):

1. Capture reports **40** attempted routers, matching the manifest.
2. Every successful peek has `logical_core` equal to `loc.to("logical")[0]`.
3. `physical_chip_id` in the snapshot equals the manifest and exists in `context.devices`.
4. Snapshot validates against the schema.
5. A second capture of the same hang-idle fabric is not required to be byte-identical (wall clocks
   move); router **ids** and count must be identical.

Do **not** require `TT_METAL_SLOW_DISPATCH_MODE`. Capture is out-of-process. Devices must be
powered and not exclusively locked in a way that blocks ttexalens; fabric firmware should have been
initialized at least once so the manifest exists.

---

## 7. Risks / open questions

- **ABI not present yet.** Stream 22 is "probably sender free slots" only under the current EDM
  assignment. v1 must treat ids as opaque hardware streams so CT-flag configs do not silently
  mislabel. The dumper groups are a *read list*, not a *semantic map* in the snapshot.
- **Cores in reset.** Record them with `ok: false` (or `ok: true` plus `health.reset == 0`) rather
  than dropping them; dropping would look like "this hop does not exist."
- **Multi-host peek.** Capture on host 0 cannot read host 1's ASICs. That is expected; host 1 runs
  its own capture. Merge is the viewer's job. Cannot fully test on this single-rank T3K.
- **Blackhole.** Overlay index path is already in `fabric_erisc_constants.py`; untested here.
- **Import path.** `fabric_erisc_utils.py` uses a same-directory import. Capture must run with
  `debug/` on `sys.path` or we later turn both into a package. Prefer the same hack the dumper uses
  rather than a repo-wide package refactor in this task.
- **Manifest vs live topology drift.** If someone captures after a different fabric config without
  regenerating the manifest, coords may still match (same cables) but stream meanings differ. Copy
  `run.fabric_config` into the snapshot so the mismatch is visible.

---

## 8. Step order

1. Draft `fabric_debug_snapshot_schema.json`.
2. Implement manifest load + router enumeration (no ttexalens yet): unit-testable.
3. Implement peek (`health` + streams 0–29) using ttexalens + existing address helpers.
4. CLI: manifest in, snapshot out.
5. Run against a live T3K manifest; check 40 routers and coordinate match.
6. Schema-validate the written snapshot.
