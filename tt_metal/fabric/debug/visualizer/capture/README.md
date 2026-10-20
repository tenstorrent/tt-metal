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
               "ttexalens": "0.4.1" },
  "raw": { "file": "fabric_snapshot_rank_1_of_1_20261019T153000Z.bin", "size": 6291296, "sha256": "..." },
  "routers": {
    "M0/C0/E1": {
      "device": { "unique_id": "0x000000036190302b", "ttexalens_id": 4 },
      "core": { "translated": [25, 16], "logical": [0, 0] },
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
- **Words.** Core and stream register values are raw 32-bit words, read little-endian, with no masking or sign
  conversion: that is decode's job.
- **Order.** Capture first finds every router's chip and core. Then, router by router in manifest order, it reads the
  core registers, the stream registers (`before`), the areas, the regions, and the stream registers again (`after`).
  Each area and region is one read.
- **Errors.** `error` is null, or `{stage, path, message}` for the read that failed. `stage` is `open`,
  `core_registers`, `streams_before`, `areas`, `regions` or `streams_after`. `path` is the area's, region's or core
  register's key, `<stream id>/<register>` for a stream register, and null for `open`. The first error stops that
  router; what it read before stays in the snapshot, and the other routers carry on.
- **Times.** `started_ns` and `finished_ns` come from `time.monotonic_ns()`, so decode can tell how far apart two reads
  were.
- **The manifest.** Capture reads the manifest file once, hashes those bytes and parses the same bytes, so
  `manifest.sha256` is the hash of exactly what it used. `run` is a copy, for people reading the file.

[`fabric_snapshot_schema.json`](fabric_snapshot_schema.json) is the `.json`'s JSON Schema.

## Components

| File | What it does |
| --- | --- |
| [`reads.py`](reads.py) | `get_read_plan`: what to read on each router core, from the manifest |
| [`snapshot.py`](snapshot.py) | `take_snapshot`: reads the plan through a device and writes the snapshot |
| [`fabric_snapshot_schema.json`](fabric_snapshot_schema.json) | The snapshot `.json`'s JSON Schema |
| [`tests/test_reads.py`](tests/test_reads.py) | Hand-written cases for the read plan |
| [`tests/test_snapshot.py`](tests/test_snapshot.py) | Cases for the snapshot, with a fake device and clock |
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
each register was read at the address the manifest gives it, and that the snapshot matches the schema. It needs
`jsonschema`, and deletes each snapshot once checked. To run it on other manifests, pass their files or directories:

```bash
python3 -m tt_metal.fabric.debug.visualizer.capture.tests.check_capture generated/fabric
```
