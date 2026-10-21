# Inputs

[`inputs.py`](../inputs.py) loads what decode reads for one snapshot: the snapshot's `.json`, its `.bin`, and the
manifest it was taken with. It checks that the three belong together, and refuses them when they don't. It also
compares the build that wrote the manifest with the checkout decode runs from, and warns when they may not be the same
code.

## What it takes and returns

The caller passes the path of a snapshot's `.json` as capture wrote it, and decode's own checkout. It may also pass
manifests to choose from. It gets back an `Input`: the parsed snapshot and the sha256 of its bytes, the `.bin`'s
bytes, the manifest and the path it was found at, and the warnings.

```python
checkout = get_checkout(pathlib.Path(__file__).parent)  # once per decode
loaded = load_input(snapshot_path, checkout)  # or load_input(snapshot_path, checkout, manifest_paths)
loaded.manifest, loaded.raw, loaded.warnings
```

Anything that keeps the snapshot from being decoded raises `InputError`, naming the file and the reason.

## The files

| File | Written by | Where |
| --- | --- | --- |
| `fabric_manifest_rank_<r>_of_<n>.json` | the run, at fabric init, when `TT_METAL_FABRIC_GENERATE_MANIFEST=1` | `<TT_METAL_LOGS_PATH>/generated/fabric/`, on the rank's host |
| `fabric_snapshot_rank_<r>_of_<n>_<UTC>.json` | capture | beside the manifest, or in capture's `-o DIR` |
| `fabric_snapshot_rank_<r>_of_<n>_<UTC>.bin` | capture | beside its `.json`, which names it in `raw.file` |

The snapshot records the manifest's absolute path on the capture host and its sha256 (`manifest.file`,
`manifest.sha256`), and the `.bin`'s name, size and sha256 (`raw`). "The snapshot" in the
[capture README](../../capture/README.md) describes the rest of the format.

## Finding the manifest

1. **With manifests given** (`manifest_paths`), it takes the one whose sha256 is the snapshot's. This is for a snapshot
   whose manifest has moved, or a directory of several runs' manifests.
2. **Otherwise, the path the snapshot names,** which is where the manifest was on the capture host.
3. **Then a file of that name beside the snapshot.** This covers a snapshot copied off its host together with its
   manifest, such as a whole capture directory.

Whichever file it finds, its sha256 must be the snapshot's.

## What it refuses, and why

- **Another manifest.** Capture worked out every address it read from that manifest's bytes. Fabric init deletes the
  previous manifest and writes a new one every launch (`remove_stale_fabric_manifest`, then `write_fabric_manifest`,
  in `FabricFirmwareInitializer::init`). So a different sha256 means another launch, whose addresses and layouts can
  differ, and decoding against it would read the wrong bytes as the right fields with nothing to show for it. There is
  no override.
- **Another `.bin`.** Its size and sha256 must be the snapshot's `raw`. A short or swapped `.bin` would put every offset
  on the wrong bytes.
- **Something that isn't a version 1 fabric snapshot.** That means `kind` and `snapshot_version`, and the keys decode
  reads, each with its JSON type. Those include each router's entry: `liveness`, `core_registers`, `areas`, `regions`
  and `streams`, and `started_ns` and `error`, which can be null. A router missing one is refused at load, naming the
  router and the key, rather than failing part way through a decode.

It checks only that each router's entry has those keys. `routers.py` looks inside them, region by region, including the
check that each region's copied `address` and `size` are its manifest node's.

## Build identity

The manifest records the build that wrote it, as `run.git_sha` and `run.git_dirty`, from the build's version header.
`get_checkout` finds the commit decode runs from, with the two commands the build uses (`git rev-parse HEAD`, and
`git diff-index --quiet HEAD --` for whether tracked files changed).

**Why a warning, not a refusal.** The manifest carries the layouts and encoding values, so a different commit doesn't
make decode read bytes wrong. But decode's rules, such as `routes.py`'s, are code. A fabric change that no manifest
value shows can only show up as a difference in commits. That is case 3 of "What decode reads, and what it knows" in
the [README](../README.md).

| Case | Warning |
| --- | --- |
| Both at one commit, neither with local changes | none |
| The manifest has no `run.git_sha` (written before it recorded one) | decode can't tell whether its rules match the run's |
| `run.git_sha` is empty (built outside git) | the same |
| Decode isn't in a git checkout | the run's commit, and that decode can't tell |
| Different commits, or local changes on either side | both commits, each with "with local changes" where it applies |

`get_checkout` is called once per decode and handed to `load_input`, so the tests pass a `Checkout` and never mock git.

## What it doesn't cover yet

- **One snapshot at a time.** Decoding the snapshots of all of a launch's ranks as a set isn't here yet: checking that
  they are one launch, with every rank once, and merging them. Nor is collecting snapshots from several hosts.
- **Snapshot version 1 only.**
- **No schema check.** Decode is standard library only, so it checks just the keys it reads, rather than validating the
  snapshot against `fabric_snapshot_schema.json`. Capture's CI check validates every snapshot it takes against that
  schema. The same goes for the manifest: [`type_parser.py`](../type_parser.py) and [`routes.py`](../routes.py) check
  what they read from it.

## Files cited

| File | Symbols |
| --- | --- |
| `tt_metal/impl/device/firmware/fabric_firmware_initializer.cpp` | `FabricFirmwareInitializer::init` |
| `tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.cpp` | `remove_stale_fabric_manifest`, `write_fabric_manifest` |
| `cmake/version.cmake` | the `git rev-parse HEAD` and `git diff-index --quiet HEAD --` it runs |
| [`capture/README.md`](../../capture/README.md) | "The snapshot": the snapshot's format |
| [`manifest/README.md`](../../manifest/README.md) | the manifest, and when it is written |
