# Reader

`parse_manifest_json` (`fabric_manifest.hpp`) reads a manifest's JSON back into the `ManifestFacts` the writer made
it from ([WRITER.md](WRITER.md)). It sits beside `make_manifest_json` in `fabric_manifest.cpp`, so a change to the
format changes both in one file, and every consumer that reads the manifest in C++ gets the same model the builders
fill. Capture reads it this way, and builds what it reads from the model (`collect_capture_plan`).

The two are inverses: for any manifest the writer writes, in either fields layout,

```cpp
make_manifest_json(parse_manifest_json(json), layout) == json
```

## Strict

The reader is built on the visualizer's strict JSON reader ([common/docs/JSON_READER.md](../../common/docs/JSON_READER.md)),
so it fails, naming the key by its JSON pointer:

- on a key it doesn't read, whatever its value: `Fabric manifest: /run has keys this reader doesn't read: /run/pid`;
- on a key it reads that's missing: `Fabric manifest: /meshes/M0/chips/C0/routers/E0/shape/num_vcs is missing`.

Every key is required, except the few the writer leaves out where they don't apply, which it takes as optional: a
mesh's `torus` and `credit_transport`, the run's `subcontext`, the fabric's `routing_2d_route_buffer_size` and
`routing_1d_extension_words`, and a sender credit's `acked`. Where the writer has nothing to write it writes `null`,
and the reader reads that key as nullable rather than optional.

It also fails on:

| What | Why |
| --- | --- |
| Another `file_kind` or `manifest_version` | The file isn't a manifest, or not one this reader's format describes |
| A `run.build` other than this build's | The type names are the writing build's: the reader fills `RunInfo::build` from its own build |
| A type name this build doesn't register (`is_registered_type_name`) | Decode couldn't read bytes as it |
| An enum spelling that names no enumerator (`parse_lower_enum_name`) | The writer writes only names from the build's enums |
| A key, path or `erisc<n>`, `ch<n>` or `vc<n>` the writer wouldn't write | Keys and paths parse with the parsers in `fabric_manifest_names.hpp` and `fabric_manifest_paths.hpp`, which accept only what the writer writes |
| A `fields` id `field_sets` doesn't have | The manifest refers to a block it doesn't hold |
| A live direction that isn't a direction, or out of enum order | The writer writes each live direction once, in enum order |
| An arch area that isn't `active` | Every router core writes an arch area |
| A region that's `multi_sampled` but not `active` | Sampling a region nothing writes says nothing |

## What it checks rather than reads

Some keys the writer derives rather than copies from the model. The reader takes each with `check`, comparing it with
what the writer would write, rather than reading it into the model:

| Key | Checked against |
| --- | --- |
| `file_kind`, `manifest_version` | `k_manifest_file_kind`, `k_manifest_version` |
| `run.build` | This build's identity |
| `vocabulary` | The vocabulary this build writes |
| A chip's `is_local` | Whether the chip has a `physical_chip_id` |
| A credit counter array's `indexed_by` | The array's entry in the model's `k_credit_counter_arrays`, which the writer writes from too |
| A core register's `name` and `type` | The register at that place in the array, and the word type |

A remote chip has only its place in the mesh, so when `is_local` is false the reader reads nothing past it.

## Fields layouts

The reader takes either layout ([WRITER.md](WRITER.md) describes them). A manifest with `field_sets` is `SHARED`: the
reader reads each set once, before `meshes`, and each `fields` must be the id of one. A manifest without it is
`PER_ROUTER`, and each `fields` must be the block itself. Either way, each router, channel and edge gets its own copy
of its block in `fields_by_erisc`.

## What doesn't come back

The reader gives back everything the writer writes. A few things the model holds aren't written, so a model read back
differs from the one written, though writing it again gives the same JSON:

- A field with no content is written as `null`, so its category comes back as the default.
- Each router's per-VC channel and edge lists come back with one entry per VC up to `MAX_NUM_VCS`, as the router pass
  fills them, whatever size a hand-built model gave them.

## Tests

- `WriterParserRoundTrip`, in every manifest fixture: reads the manifest the run wrote, writes it again in the
  `SHARED` layout and compares the two, then does the same in the `PER_ROUTER` layout. On a mismatch it prints the
  first operations of `json::diff`, each naming the pointer that differs.
- `ManifestReader.*`, with no device: the round trip on hand-built facts with every optional part, both layouts
  reading into the same facts, and each kind of failure above.
