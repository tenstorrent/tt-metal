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
   [ROUTES.md](ROUTES.md) lists each rule with the code it mirrors, so a fabric change to that code can update the
   rule with it.

## Components

| File | What it does |
| --- | --- |
| [`routes.py`](routes.py) | The route rules. Reads 1D route words, 2D route maps, the routing tables, multicast trees and scatter writes from raw bytes, the way the fabric's routers and route setters do. In depth: [ROUTES.md](ROUTES.md) |
| [`tests/test_routes.py`](tests/test_routes.py) | Hand-written cases for each rule |
| [`tests/check_manifest_routes.py`](tests/check_manifest_routes.py) | Checks that `routes.py` can read the manifests CI writes |
| [`tests/test_check_manifest_routes.py`](tests/test_check_manifest_routes.py) | Cases for the check's `--expect-z-peers` |
| [`../common/manifest_lookup.py`](../common/manifest_lookup.py) | `lookup`, which reads a manifest value and checks its JSON type, and `ManifestError`. Shared with the visualizer's other Python packages |

## Running the tests

From the repo root:

```bash
python3 -m unittest discover -s tt_metal/fabric/debug/visualizer/decode/tests -t .
```

The tests import `tt_metal.fabric...`, so `-t .` makes the repo root the top level they're imported from. They use `unittest` rather than pytest because the repo's root `conftest.py` imports torch.

`tests/scripts/run_fabric_manifest_tests.sh` runs the same command, then runs `tests/check_manifest_routes.py` on every manifest it writes.
The check runs `get_encodings_from_manifest`, then `get_mesh_view` on each mesh, and fails, naming the manifest and the
key, when a fabric change renames or removes a key or name `routes.py` reads, changes a value's type, or adds a value it doesn't know. The script then checks the express mock's manifests again with `--expect-z-peers`, which fails a manifest in which `get_mesh_view` finds no Z peer. To run it on other manifests, pass their files or directories:

```bash
python3 -m tt_metal.fabric.debug.visualizer.decode.tests.check_manifest_routes generated/fabric
```
