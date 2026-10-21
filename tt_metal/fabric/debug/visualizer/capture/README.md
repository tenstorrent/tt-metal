# Fabric capture

Capture reads, from a run's devices, the L1 and registers the run's [fabric manifest](../manifest/README.md) points at,
and writes them to a snapshot. It doesn't interpret them: [decode](../decode/README.md) does, reading the snapshot
through the same manifest. Capture works out every read from the manifest, so it knows no fabric struct or address.

## What it reads

Every read is one of two kinds, and comes from one of two places in the manifest:

| | Same on every router core, by name | Per router, found by walking the router |
| --- | --- | --- |
| **L1 bytes**: `size` bytes at `address` | `archs.<arch>.areas`, such as `heartbeat` | regions: any node under the router with an `address` and a `size` |
| **A register**: one 32-bit word | `archs.<arch>.core_registers`, such as `soft_reset` | stream registers: any node under the router with a `stream_id` and a `register` |

Only local chips' routers are read. A stream register's address is
`base + stream_id * stride + register_offsets[register]`, from `archs.<arch>.stream_registers`, and each (stream id,
register) pair is read once, however many nodes name it.

**The path rule.** A region's path is its keys from the router down, joined by `/`: object keys as they are, array
indices as numbers. `channels/senders/vc0/ch0/ring_buffer` is the router's
`["channels"]["senders"]["vc0"]["ch0"]["ring_buffer"]`. A router's path is `M<mesh>/C<chip>/<router key>`, such as
`M0/C0/E1`, the string `link.peer` uses. Decode finds a region's manifest node, and so its layout, by following its
path.

## The snapshot

There is one snapshot per manifest: `fabric_snapshot_rank_<r>_of_<n>_<UTC>.json` and a `.bin` of the same name, named
after `fabric_manifest_rank_<r>_of_<n>.json`. The `.bin` holds the L1 bytes, end to end; the `.json` holds everything
else. Capture writes the `.bin` first and the `.json` second, each to a temporary file that is then renamed into place,
and won't overwrite a snapshot.

```json
{
  "kind": "fabric_snapshot",
  "snapshot_version": 1,
  "manifest": { "file": "/.../fabric_manifest_rank_1_of_1.json", "sha256": "...",
                "run": { "arch": "wormhole_b0", ... } },
  "capture": { "started_at": "2026-10-19T15:30:00Z", "finished_at": "...", "host": "wh-lb-85", "argv": [...],
               "ttexalens": "0.4.1", "liveness_rounds": 3, "liveness_interval_s": 1.0 },
  "raw": { "file": "fabric_snapshot_rank_1_of_1_20261019T153000Z.bin", "size": 6291296, "sha256": "..." },
  "routers": {
    "M0/C0/E1": {
      "device": { "unique_id": "0x000000036190302b", "ttexalens_id": 4 },
      "core": { "translated": [25, 16], "logical": [0, 0] },
      "liveness": [
        { "t_ns": 0, "heartbeat": 3703177291,
          "core_registers": { "soft_reset": 290816, "wall_clock_low": 0, "wall_clock_high": 0 } },
        ...
      ],
      "started_ns": 0, "finished_ns": 0,
      "core_registers": { "soft_reset": 0, "wall_clock_low": 0, "wall_clock_high": 0 },
      "areas": { "heartbeat": { "address": 8064, "size": 4, "offset": 0 }, ... },
      "regions": { "channels/senders/vc0/ch0/ring_buffer": { "address": 100256, "size": 62272, "offset": 4028 }, ... },
      "streams": { "0": { "buf_space_available": { "before": 14, "after": 14 } },
                   "31": { "remote_src": { "before": 0, "after": 0 } }, ... },
      "error": null
    }
  }
}
```

- **Keys.** A router is keyed by its path, a region by its path inside the router, an area and a core register by its
  name, and a stream register by stream id and then register. Ids 30 and 31 can each have two registers, so an id on its
  own isn't a key.
- **Areas and regions** copy `address` and `size` from the manifest and add `offset`, where their bytes start in the
  `.bin`. The copy lets decode check, region by region, that it has the right manifest.
- **Words.** Core and stream register values and liveness samples are raw 32-bit words, read little-endian, with no
  masking or sign conversion: that is decode's job.
- **Liveness rounds.** Before the main pass, capture makes `liveness_rounds` rounds, `liveness_interval_s` seconds
  apart. Each round visits every router in manifest order and appends a sample to its `liveness`: the host time `t_ns`,
  the `heartbeat` area's word, and the core registers. Rounds go across all routers, so samples from different routers
  are close in time, and decode can tell whether each router's heartbeat and clock moved. `heartbeat` is null when the
  manifest's `heartbeat` area is.
- **Order.** Capture first finds every router's chip and core, then makes the liveness rounds. Then, router by router in
  manifest order, it reads the core registers, the stream registers (`before`), the areas, the regions, and the stream
  registers again (`after`). Each area and region is one read.
- **Errors.** `error` is null, or `{stage, path, message}` for the read that failed. `stage` is `open`, `liveness`,
  `core_registers`, `streams_before`, `areas`, `regions` or `streams_after`. `path` is the area's, region's or core
  register's key (`heartbeat` or a core register's for `liveness`), `<stream id>/<register>` for a stream register, and
  null for `open`. The first error stops that router for the rest of the capture, rounds included; what it read before
  stays in the snapshot, and the other routers carry on.
- **Times.** `t_ns`, `started_ns` and `finished_ns` come from `time.monotonic_ns()`, so decode can tell how far apart
  two reads were and turn a wall clock's change into a rate.
- **The manifest.** Capture reads the manifest file once, hashes those bytes and parses the same bytes, so
  `manifest.sha256` is the hash of exactly what it used. `run` is a copy, for people reading the file.

[`fabric_snapshot_schema.json`](fabric_snapshot_schema.json) is the `.json`'s JSON Schema.

## Taking a snapshot

Capture reads through ttexalens from its own process, so it works while the process that owns the fabric is hung, or
after it is gone. It needs the run's manifests, which the run writes when `TT_METAL_FABRIC_GENERATE_MANIFEST=1` (see the
[manifest README](../manifest/README.md)). From the repo root:

```bash
python3 -m tt_metal.fabric.debug.visualizer.capture generated/fabric
```

Pass manifest files or directories; a directory stands for the `fabric_manifest_rank_*.json` files in it. Each snapshot
is written next to its manifest, or into `-o DIR`. Before each main pass, capture makes `--liveness-rounds` rounds
(default 3; 0 turns them off), `--liveness-interval` seconds apart (default 1.0), so by default a snapshot takes at
least 2 s. Capture prints one line per snapshot and one per router error. It exits non-zero when ttexalens doesn't start
or a manifest can't be read or its snapshot written; a router error is only recorded in the snapshot.

**Finding a router.** A router's chip is the one ttexalens has under the chip's `asic_id`, and it must be of `run.arch`.
The router's core is `identity.virtual_core` as a translated coordinate, if ttexalens puts it at
`identity.logical_core`; if not, capture tries ttexalens's eth core number `identity.eth_chan` and says which it read.
When neither is at the logical core, that is an `open` error for the router.

## Components

| File | What it does |
| --- | --- |
| [`reads.py`](reads.py) | `get_read_plan`: what to read on each router core, from the manifest |
| [`snapshot.py`](snapshot.py) | `take_snapshot`: makes the liveness rounds and the main pass through a device and writes the snapshot |
| [`exalens.py`](exalens.py) | `open_exalens`: the device that reads through ttexalens. The only module that imports ttexalens |
| [`__main__.py`](__main__.py) | The command line: takes a snapshot of each manifest given |
| [`fabric_snapshot_schema.json`](fabric_snapshot_schema.json) | The snapshot `.json`'s JSON Schema |
| [`tests/test_reads.py`](tests/test_reads.py) | Hand-written cases for the read plan |
| [`tests/test_snapshot.py`](tests/test_snapshot.py) | Cases for the snapshot, with a fake device and clock |
| [`tests/test_exalens.py`](tests/test_exalens.py) | Cases for finding a router's chip and core and reading it, with fake ttexalens objects |
| [`tests/check_capture.py`](tests/check_capture.py) | Takes a snapshot of each manifest CI writes, with a fake device, and checks it |
| [`../common/manifest_lookup.py`](../common/manifest_lookup.py) | `lookup`, which reads a manifest value and checks its JSON type, and `ManifestError`. Shared with decode |

## Running the tests

From the repo root:

```bash
python3 -m unittest discover -s tt_metal/fabric/debug/visualizer/capture/tests -t .
```

As with decode's tests, `-t .` makes the repo root the top level the tests import `tt_metal.fabric...` from.

`tests/scripts/run_fabric_manifest_tests.sh` runs the same command, then runs `tests/check_capture.py` on every manifest
it writes. The check takes a snapshot with a fake device whose bytes say what was read, so it can check that the
snapshot has everything the manifest names and nothing else, that each area's and region's bytes are at its offset, that
each register was read at the address the manifest gives it, that each router has two liveness samples taken before its
main pass, and that the snapshot matches the schema. It needs `jsonschema`, and deletes each snapshot once checked. To
run it on other manifests, pass their files or directories:

```bash
python3 -m tt_metal.fabric.debug.visualizer.capture.tests.check_capture generated/fabric
```

## Future work

**Tests on hardware.** No test checks capture on real chips yet. The plan is pytest tests that bring a fabric up the way
fabric jobs do, through conftest's `mesh_device` fixture with a `fabric_config`, then run capture against it while its
owner holds it, after the owner is killed and after it exits, and check that the owner's next op still gives the right
result. Each snapshot would be checked using only its manifest: every router read without an error and, while the
routers run, each routing table naming its router's mesh and chip, `edm_status` at `ready_for_traffic`, the ERISCs out
of reset and the wall clock rising. They would run in CI on N300, T3K and Blackhole machines.
