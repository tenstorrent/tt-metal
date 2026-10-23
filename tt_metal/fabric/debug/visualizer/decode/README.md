# Fabric decode

Decode turns raw bytes read from a fabric run into what they mean: which packet sits in a channel, where it is going,
and what a router's routing tables say. It reads every layout and value it needs from the run's [fabric manifest](../manifest/README.md), so it doesn't copy the fabric's structs or encoding values. It is plain Python, standard library only, and needs no device and no build.

## Where it fits

The visualizer is made of components that hand each other files:

```
fabric init (C++)           a hang                      offline                     view
-----------------           ------                      -------                     ----
manifest:             --->  capture:              --->  decode:               --->  viewer
what every router is,       the raw L1 and              what the bytes mean,
where its state is,         registers the manifest      read through the
how that state is           points at, read from        manifest
laid out                    the devices
```

- The **manifest** describes one run: every router, the L1 regions and registers it keeps, the layouts of the structs
  in them, and the values routes are encoded with.
- **Capture** reads the bytes and registers the manifest points at. It doesn't interpret them.
- **Decode** interprets them.
- The **viewer** shows what decode makes of them.

## What decode reads, and what it knows

The manifest describes every value decode reads, but describing a value isn't the same as decode knowing what to do
with it. A fabric change falls into one of three cases:

1. **The value's meaning is data in the manifest.** Decode reads the meaning along with the value, so the change needs
   no change to decode. This covers struct layouts, the names of enum values decode only displays, which member of a
   tagged union a tag picks, and the values routes are encoded with: bit positions, field widths, and the action each
   2D route table code stands for.
2. **The manifest names the value, but its meaning is in decode's code.** The manifest says a 1D hop called
   `write_only` has value 1, but not what writing or forwarding does: decode holds that. The same goes for each 2D
   direction's step, what each scatter chunk type does, and which Z port role is a step within the mesh. Decode
   accepts only the names it knows, so a renamed, added or removed value fails, naming what the manifest holds and
   what decode knows, instead of being decoded with a guessed meaning. Decode has to be taught the new value.
3. **The fabric's logic changes, and no value does.** For example, a router starts reading a different byte of the
   route map, or a new header field starts changing where a packet goes. The manifest doesn't change, so nothing
   fails, and only review catches it. Decode holds this logic as rules, written in Python to match the fabric's C++.
   [ROUTES.md](docs/ROUTES.md) lists each rule with the code it mirrors, so a fabric change to that code can update the
   rule with it.

## The decoded file

There is one decoded file per snapshot, `fabric_decoded_rank_<r>_of_<n>_<UTC>.json`, named after its snapshot.
[`fabric_decoded_schema.json`](fabric_decoded_schema.json) is its JSON Schema.

```json
{
  "kind": "fabric_decoded",
  "decoded_version": 1,
  "decoded_at": "2026-10-21T12:00:00Z",
  "decoder": { "git_sha": "...", "git_dirty": false },
  "sources": [
    {
      "snapshot": { "file": "/.../fabric_snapshot_rank_1_of_1_20261021T120000Z.json", "sha256": "..." },
      "manifest": { "file": "/.../fabric_manifest_rank_1_of_1.json", "sha256": "...", "run": { ... } },
      "capture": { "host": "...", "started_at": "...", ... },
      "warnings": ["the run was built from aaaaaaaaaa; decode is at bbbbbbbbbb"]
    }
  ],
  "routing_tables": {
    "M0/C0/E1": {
      "chip": "M0/C0",
      "routers": ["M0/C0/E1", "M0/C0/E0", "M0/C0/S1", "M0/C0/S0"],
      "raw": { "bytes": { "offset": 164, "size": 2704 } },
      "fields": { "my_mesh_id": 0, "my_device_id": 0, "intra_mesh_direction_table": [ ... ], ... },
      "route_table_2d": {
        "M0/C5": { "delivers": ["M0/C5"], "steps": [{ "src": "M0/C0", "direction": "south", "dst": "M0/C4" }, ...], "exits": [], "odd": [] },
        ...
      },
      "mcast_trees": { "y": [{ "child": 1, "parent": 0, "direction": "south" }], "x": [ ... ] },
      "path_table_1d": null
    }
  },
  "routers": {
    "M0/C0/E1": {
      "source": 0,
      "error": null,
      "areas": { "heartbeat": 2882396417, ... },
      "regions": {
        "fields/edm_status": { "raw": 2746467283, "name": "ready_for_traffic" },
        "channels/senders/vc0/ch0/ring_buffer": [{ "bytes": { "offset": 4028, "size": 4448 } }, ...]
      },
      "streams": {
        "channels/senders/vc0/ch0/fields/free_slots": {
          "stream_id": 0, "register": "buf_space_available", "bits": 17,
          "before": 14, "after": 14, "before_word": 14, "after_word": 14
        }
      },
      "elements": {
        "channels/senders/vc0/ch0/credits/completed": { "array": "credit_counters/to_sender_completion", "index": 0, "value": 1234 }
      },
      "health": { "t_ns": [...], "eriscs": { ... }, "wall_clock": { ... }, "heartbeat": { ... } },
      "routing": {
        "table_key": "M0/C0/E1",
        "state_manager": { "state": { "raw": 1, "name": "running" }, "command": { "raw": 0, "name": "run" } },
        "raw": { "bytes": { "offset": 164, "size": 2704 } }
      },
      "channels": {
        "channels/senders/vc0/ch0": {
          "kind": "sender", "depth": 14, "producer": "worker",
          "connection": { "raw": 0, "name": "unused_connection_value" },
          "producer_core": { "x": 0, "y": 0, "whose": "last" },
          "unsent": { "before": 0, "after": 0, "not_observable": null },
          "uncredited": { "before": 0, "after": 0, "not_observable": null },
          "credits": { "completed": { "frees": "receiver_slot", "backing": "stream_register", "pending": { ... }, ... } },
          "speedy_batch": null, "within_bounds": true, "moved": false,
          "slots": [
            { "empty": true, "header": null, "route": null, "scatter": null, "problems": [] },
            { "empty": false, "header": { ... }, "route": { "start": "M0/C1", "entered_from": "west", "walk": { ... },
              "landing": false, "rebuilt": null }, "scatter": null, "problems": [] },
            ...
          ]
        },
        "channels/receivers/vc0/ch0": {
          "kind": "receiver", "depth": 8,
          "waiting": { "before": 0, "after": 0, "not_observable": null },
          "unacked": { "before": null, "after": null, "not_observable": "first-level ack is off on this VC" },
          "moved": false, "slots": [ ... ]
        }
      }
    }
  }
}
```

- **Sources.** Each source is a snapshot with the manifest it was taken with:
  - their files and sha256s, and the manifest's `run`;
  - the snapshot's `capture` block (host, times, liveness settings);
  - the warnings [INPUTS.md](docs/INPUTS.md) describes.

  A router's `source` is its source's index. A decode of one snapshot has one source; the list is there so that every
  rank of a fabric can share one file.
- **Keys.** A router is keyed by its path (`M<mesh>/C<chip>/<router>`) and an area by its name. Regions, streams and
  elements are keyed by their path in the router, by the path rule in the [capture README](../capture/README.md).
- **Values.** An area's or region's value takes the shape of its manifest type ([TYPE_PARSER.md](docs/TYPE_PARSER.md)):
  - an int;
  - `{raw, name}` for an enum value;
  - an object for a struct, and a list for an array;
  - `{bytes: {offset, size}}` for bytes, which points into the snapshot's `.bin` rather than copying it. A ring buffer
    is a list of its slots.
- **Streams.** There is one entry per stream reference, keyed by the reference's own path, with its stream id and its
  register. Streams 30 and 31 each serve two references, and each reference gets its own entry.
  - **Before and after.** Capture reads each stream register twice: before it reads the router's L1, and again after.
    The router keeps running while capture reads it, and these registers are its live credit and free slot counts, so
    the two reads show whether they moved while the L1 was read. When they match, the counts and the L1 describe the
    same moment. When they differ, traffic moved during the read: the channel is alive, but the L1 isn't one instant
    for that register, so anything that combines the counts with the L1 is approximate.
  - **Values.** `before` and `after` are the word's low `bits` bits, as the manifest's `stream_registers` gives them,
    decoded by the reference's schema. `before_word` and `after_word` are the whole words, since the bits above the
    value can hold other state: on Blackhole, `buf_space_available` keeps more state above its count.
- **Elements.** There is one entry per `{array, index}` reference: the element of the decoded array it names. Each one
  the manifest writes is a credit on a VC whose credits use L1 counters.
- **Health.** The router's liveness facts, described in [HEALTH.md](docs/HEALTH.md).
- **Routing tables.** Each router holds its chip's routing table in its `routing_table` area. The file stores each
  unique table once, in `routing_tables`, keyed by the first router holding it, with its fields and the routes, trees
  or 1D path entries it gives. A router's `routing` names its copy (`table_key`), and holds its own `state_manager` and
  bytes. So `areas` doesn't hold `routing_table`. [ROUTING_TABLES.md](docs/ROUTING_TABLES.md) has the rules.
- **Channels.** Each active channel of a router, keyed by its path in the router, has its counts and credits
  ([FLOW_CONTROL.md](docs/FLOW_CONTROL.md)) and its ring buffer slots: each slot's header and where its packet goes
  ([RING_BUFFER_SLOTS.md](docs/RING_BUFFER_SLOTS.md)). `slots` is null when capture didn't read the ring. A channel no
  ERISC runs isn't listed. A used slot may be history: its packet may be waiting, sent or done. The counts say how many
  packets are waiting, but not which slot has a packet that is waiting.
- **A router capture stopped part way through.** Its `error` is capture's `{stage, path, message}`. Everything read
  before the error is decoded. A value never read is null: a stream word, an element of an array that wasn't read, or a
  sample's value in `health`. An area or region never read is absent.

## Decoding a snapshot

Decode needs only the snapshot, its `.bin` and its manifest, so it runs anywhere, including on a laptop. From the repo
root:

```bash
python3 -m tt_metal.fabric.debug.visualizer.decode generated/fabric
```

- **Inputs.** Pass snapshot files or directories; a directory stands for the `fabric_snapshot_*.json` files in it.
- **Output.** Each decoded file is written next to its snapshot, or into `-o DIR`, replacing one of the same name.
- **The manifest.** A snapshot is decoded with the manifest it names, or with a file of that name beside it. For a
  snapshot whose manifest has moved, pass `--manifest` with manifests or directories of them; decode takes the one with
  the snapshot's sha256 ([INPUTS.md](docs/INPUTS.md)).
- **What it prints.** One line per decoded file, then its warnings and router errors. The line says how many routers
  capture couldn't fully read, how many ring buffer slots are used, and how many packets the counts say are unsent in
  senders and waiting in receivers.
- **Exit status.** Non-zero when a snapshot is refused or its decoded file can't be written. A router that capture
  couldn't fully read isn't a failure: it is recorded in the file.

## Components

| File | What it does |
| --- | --- |
| [`routes.py`](routes.py) | The route rules. Reads 1D route words, 2D route maps, the routing tables, multicast trees and scatter writes from raw bytes, the way the fabric's routers and route setters do. Follows a 1D route along the manifest's wiring, and builds the map an intermesh landing gives a packet. In depth: [ROUTES.md](docs/ROUTES.md) |
| [`type_parser.py`](type_parser.py) | Reads the struct, union and enum layouts the manifest describes, checks them, and decodes bytes into values with them. It holds no fabric names. In depth: [TYPE_PARSER.md](docs/TYPE_PARSER.md) |
| [`inputs.py`](inputs.py) | Loads a snapshot, its `.bin` and the manifest it was taken with, refuses them when they don't belong together, and warns when the run's build and decode's checkout may differ. In depth: [INPUTS.md](docs/INPUTS.md) |
| [`routers.py`](routers.py) | `decode_router`: decodes one router's part of a snapshot, its areas and regions and each stream and element reference, keyed by its path in the router. It puts the path rule, the type parser and `inputs.py` together, and has no rules of its own |
| [`route_context.py`](route_context.py) | `get_route_context`: what the route rules need of a manifest, read once per snapshot, and each chip's key and coordinate. `name_walk_2d` names a 2D walk's chips by key. Shared by the routing tables and the ring buffer slots |
| [`routing_tables.py`](routing_tables.py) | `get_routing_table_store`: decodes the routing tables in each router's L1 `routing_table` area into their fields and the routes, trees or 1D path entries they give, storing each unique table once, with each router's own state. In depth: [ROUTING_TABLES.md](docs/ROUTING_TABLES.md) |
| [`ring_buffer_slots.py`](ring_buffer_slots.py) | `get_ring_buffer_slots`: decodes the packet header in each ring buffer slot of a router's active channels, from the slot's L1 bytes, and where the packet goes, starting where the slot's channel says. A rule that refuses a slot's bytes is recorded as a problem, never raised. In depth: [RING_BUFFER_SLOTS.md](docs/RING_BUFFER_SLOTS.md) |
| [`flow_control.py`](flow_control.py) | `get_channel_flows`: what each active channel's counts and credits say: a sender's unsent and uncredited packets and the credits coming back from its peer, and a receiver's waiting or unacked packets, each with why when the snapshot can't give it. In depth: [FLOW_CONTROL.md](docs/FLOW_CONTROL.md) |
| [`health.py`](health.py) | `get_health`: a router's liveness samples and main pass as facts: whether each ERISC is held in reset, how the wall clock moves, and whether the heartbeat moves. In depth: [HEALTH.md](docs/HEALTH.md) |
| [`decoded.py`](decoded.py) | `decode_snapshot`: puts a snapshot's sources, its routing tables, and its decoded routers with their health, routing and channels into the decoded file, and `write_decoded` writes it |
| [`__main__.py`](__main__.py) | The command line: decodes each snapshot given |
| [`fabric_decoded_schema.json`](fabric_decoded_schema.json) | The decoded file's JSON Schema |
| [`tests/test_routes.py`](tests/test_routes.py) | Hand-written cases for each rule |
| [`tests/test_type_parser.py`](tests/test_type_parser.py) | Hand-written cases for every schema form, and for each check of the layouts |
| [`tests/test_inputs.py`](tests/test_inputs.py) | Cases for loading, pairing and refusing snapshots, and for the build identity warnings |
| [`tests/test_routers.py`](tests/test_routers.py) | Cases for decoding one router, including one capture stopped part way through, and snapshots that disagree with the manifest |
| [`tests/test_routing_tables.py`](tests/test_routing_tables.py) | Cases for storing tables once (agreeing routers, `state_manager`, a corrupted byte, two chips), 2D routes and trees, a 1D path table with an extension word, and renamed members |
| [`tests/test_ring_buffer_slots.py`](tests/test_ring_buffer_slots.py) | Cases for empty ring buffer slots, each place a route starts (2D sender, receiver, a link leaving the mesh, a landing before and after its rebuild; 1D sender and receiver), extension words, scatter writes, problems, and renamed header members |
| [`tests/test_flow_control.py`](tests/test_flow_control.py) | Cases for each count by producer (worker connected and not, sibling-fed, unfed), counters that wrap, both credit backings with a peer in and out of the snapshot, first-level ack on and off, speedy batches, and manifest values the rules don't know |
| [`tests/test_health.py`](tests/test_health.py) | Cases for each health fact: reset bits by physical ERISC, the wall clock's join and wrap, and heartbeat words from other writers |
| [`tests/test_decoded.py`](tests/test_decoded.py) | Cases for the decoded file and the command line, from a one-router 2D snapshot written to a temporary directory: its routing table stored once, its channels' counts and slots, and the schema, when `jsonschema` is installed |
| [`tests/check_manifest_routes.py`](tests/check_manifest_routes.py) | Checks that `routes.py` can read the manifests CI writes |
| [`tests/test_check_manifest_routes.py`](tests/test_check_manifest_routes.py) | Cases for the check's `--expect-z-peers` |
| [`tests/check_decode.py`](tests/check_decode.py) | Decodes each manifest's types, and a snapshot of each manifest CI writes taken with a fake device, and checks the decoded file |
| [`../common/manifest_lookup.py`](../common/manifest_lookup.py) | `lookup`, which reads a manifest value and checks its JSON type, and `ManifestError`. Shared with the visualizer's other Python packages |
| [`../common/router_paths.py`](../common/router_paths.py) | The path rule: every region, stream reference and element reference under a router, and the node at a path. Shared with capture |

## Running the tests

From the repo root:

```bash
python3 -m unittest discover -s tt_metal/fabric/debug/visualizer/decode/tests -t .
```

The tests import `tt_metal.fabric...`, so `-t .` makes the repo root the top level they're imported from. They use `unittest` rather than pytest because the repo's root `conftest.py` imports torch.

`tests/scripts/run_fabric_manifest_tests.sh` runs the same command, then runs `tests/check_manifest_routes.py` on every manifest it writes.
The check runs `get_encodings_from_manifest`, then `get_mesh_view` on each mesh, then `get_line_view_1d` on a 1D run, and
fails, naming the manifest and the key, when a fabric change renames or removes a key or name `routes.py` reads, changes a value's type or a path's format, or adds a value it doesn't know. The script then checks the express mock's manifests again with `--expect-z-peers`, which fails a manifest in which `get_mesh_view` finds no Z peer. To run it on other manifests, pass their files or directories:

```bash
python3 -m tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes generated/fabric
```

Last, the script runs `tests/check_decode.py` on every manifest it writes. The check decodes every struct in the
manifest's types from zeros and from random bytes, and every tagged union under each value of its tag and under one that
names no member, checking that the union picks the member it says. It then takes a snapshot through capture with a fake
device that returns random bytes, except in a ring buffer, whose first slot is random and the rest zero, and decodes it.
It checks that the decoded file:

- has a value for each area other than the routing table, which is in `routing_tables`, and for each region, stream
  reference and element reference of every router;
- has health with a sample per liveness round and one for the main pass;
- has each active channel's counts and a slot per ring slot, the first used and the rest empty, and no other channel;
- names, for every router, a routing table copy of its chip holding it, one copy per router since random tables never
  match, each 2D copy with a route toward every chip of its mesh;
- matches the schema.

Filling only each ring's first slot keeps the file small: with every slot random, an express rank decodes to over
400 MB. Like capture's check, it needs `jsonschema`, and it deletes each snapshot and decoded file once checked:

```bash
python3 -m tt_metal.fabric.debug.visualizer.decode.tests.check_decode generated/fabric
```
