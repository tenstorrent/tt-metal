# Writer

The writer (`fabric_manifest.cpp`) turns `ManifestFacts` into the manifest's JSON and writes the file. It reads nothing
but the model: no ControlPlane, cluster, HAL or builder. Every key name and the shape of every block are decided here,
with the spellings it shares with the rest of the manifest code in `fabric_manifest_names.hpp`. A change to either
changes the manifest format, so it goes with a change to `fabric_manifest_schema.json` and [FORMAT.md](FORMAT.md).

## Writing a manifest

`write_fabric_manifest` (`fabric_manifest.hpp`) is what fabric init calls ([BUILDER_HOOKS.md](BUILDER_HOOKS.md)). It:

1. **Builds the facts.** `ManifestFacts` gets `run`, `fabric` and `arch` from the describe file

   ([DESCRIBE.md](DESCRIBE.md)), and `meshes` from the topology pass ([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)). The fabric
    routers must already be compiled, so the builder context exists.
2. **Makes the JSON** with `make_manifest_json`.
3. **Writes the file** to this rank's path (`fabric_manifest_path`) under a temporary name, `<path>.tmp.<pid>`, then

   renames it into place, so a reader never sees a partial file. On a failure it removes the temporary file.

Any failure is rethrown as `Failed to write fabric manifest <path> (unset TT_METAL_FABRIC_GENERATE_MANIFEST to skip)`, followed by the message starting with `Fabric manifest:` that says what went wrong.

`remove_stale_fabric_manifest` deletes this rank's file from an earlier run. Fabric init calls it whether or not
generation is on, so a manifest on disk always belongs to the run that wrote it.

## The JSON

`make_manifest_json` writes the blocks in this order:

| Key                | Written by                | From                                                                                |
| ------------------ | ------------------------- | ----------------------------------------------------------------------------------- |
| `manifest_version` | N/A                       | `FABRIC_MANIFEST_VERSION` (`fabric_manifest.hpp`)                                   |
| `file_kind`        | N/A                       | Always `"fabric_manifest"`, so a tool can tell the manifest from other fabric files |
| `run`              | `make_run_json`           | `ManifestFacts::run`                                                                |
| `fabric`           | `make_fabric_json`        | `ManifestFacts::fabric`                                                             |
| `arch`             | `make_arch_json`          | `ManifestFacts::arch`                                                               |
| `types`            | `make_types_json`         | The type collector (below)                                                          |
| `vocabulary`       | `make_vocabulary_json`    | `k_vocabulary_lists` (below)                                                        |
| `meshes`           | `make_mesh_json` per mesh | `ManifestFacts::meshes`                                                             |

The JSON keeps keys in the order the writer adds them. `types` comes before `meshes` in the file, but it's written last:
it's built from the schemas the other blocks write, so `arch` and `meshes` are made first.

Each model struct has a `make_<name>_json` that writes it, such as `make_router_json` and `make_sender_channel_json`.
They follow a few conventions:

- **Keys** come from `fabric_manifest_names.hpp`: meshes are keyed `M<mesh_id>`, chips `C<chip_id>`, and routers by
direction letter and routing plane, such as `E0`. Channels, edges and ERISCs are keyed `vc<v>`, `ch<c>`, `edge<n>` and
`erisc<n>`.
- **Enum values** are written as their enumerator's name in lower case (`lower_enum_name`), such as `"fabric_2d"`. A
value with no enumerator name, such as a bitmask combination, is written as its number.
- **A router another part refers to** is written as its path, such as `M0/C7/E0` ([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)).
- **Something that isn't there** is `null` rather than left out, such as a link with no peer or a field the builder
didn't allocate. Keys are left out only where they don't apply, such as `torus` on a 1D mesh or a field the builder
doesn't emit on this run.
- **Two entries with one key** fail the writer, such as two fields keyed alike.

## Schemas and the type collector

Every L1 region, stream register and enum field is written with a **schema**, `schema_name` in
`fabric_manifest_names.hpp`, which names how to read the value:

| Element | Schema |
| --- | --- |
| Unsigned or signed integer | `u8` to `u32`, `i8` to `i32`, as wide as one element |
| Enum | `enum:<Name>` |
| Struct | `struct:<Name>` |
| Tagged union | `union:<Name>` |
| Raw bytes | `bytes` |
| Padding | `pad` |

A schema names a type, and `types` describes every type a schema names. The writer keeps those two in step with a
`TypeCollector` (`fabric_manifest_type_collector.hpp`):

- **Every schema outside** `types` **goes through** `make_schema_json`, which writes the schema and hands its type to the
collector. Nothing else writes a schema there, so none can be missed.
- **The collector follows each type to what it names.** A struct's description lists its members, and a tagged union's
its tag enum and cases ([STRUCT_LAYOUTS.md](STRUCT_LAYOUTS.md)). The collector adds each struct, enum and tagged union
once, by name, and then the types its members and cases name. It fails when two different types have one name, which
would make their schema ambiguous.
- `types` **is written from the collector**, each kind sorted by name: a struct's size and its members in order, each
with its name, offset, size and schema; an enum's values by their lower-case names; and a tagged union's size, the
enum its tag holds, and its cases, each with the tag value that picks it. A member holding a tagged union also names
the struct member that holds its tag (`tag_member`). A member's schema needs no collecting, since the collector already
followed it.

So `types` holds exactly what the manifest's schemas reach, and every schema resolves. A type no schema reaches isn't
written: a struct only named by a field the builder didn't allocate, for example.

## Vocabulary

Some keys hold one of a fixed set of names, such as a channel's `status` or a field's `category`. `vocabulary` lists
every name each can hold, so a reader knows them all without the C++ enums.

The lists are `k_vocabulary_lists` in `fabric_manifest_model.hpp`. Each entry names the list and the type whose names it holds. For an enum, the list is every enumerator in lower case. For the field kinds, it's every alternative of the
`Content` variant, spelled as the `kind` key spells it (`kind_name`, such as `l1` or `stream`).

These are names the manifest writes as strings, so nothing appears both here and in `types.enums`, which describes enums
whose integer values are read from the device.

## Changing the format

1. Change the writer, and the names in `fabric_manifest_names.hpp` if a key's spelling changes.
2. Update `fabric_manifest_schema.json` and [FORMAT.md](FORMAT.md).
3. If readers can't take the new shape as it is, bump `FABRIC_MANIFEST_VERSION`.
4. For a new label key, add a list to `k_vocabulary_lists` and the same name to the schema's `vocabulary`.
