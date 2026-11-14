# Writer

The writer (`fabric_manifest.cpp`) turns `ManifestFacts` into the manifest's JSON and writes the file. It reads nothing
but the model: no ControlPlane, cluster, HAL or builder. Every key name and the shape of every block are decided here,
with the spellings it shares with the rest of the manifest code in `fabric_manifest_names.hpp`. A change to either
changes the manifest format, so it goes with a change to `fabric_manifest_schema.json` and [FORMAT.md](FORMAT.md).

## Writing a manifest

`write_fabric_manifest` (`fabric_manifest.hpp`) is what fabric init calls ([BUILDER_HOOKS.md](BUILDER_HOOKS.md)). It:

1. **Agrees on the run id.** Rank 0 makes it and broadcasts it over the current world, which is this fabric's ranks
   (a job tt-run splits into sub-contexts has a fabric, and so a run, per sub-context). Every rank of the fabric must
   call `write_fabric_manifest`.
2. **Builds the facts.** `ManifestFacts` gets `run`, `fabric` and `arch` from the describe file
   ([DESCRIBE.md](DESCRIBE.md)), and `meshes` from the topology pass ([TOPOLOGY_PASS.md](TOPOLOGY_PASS.md)). The fabric
   routers must already be compiled, so the builder context exists.
3. **Makes the JSON** with `make_manifest_json`.
4. **Writes the file** to this rank's path in the run's folder, `<logs_dir>/generated/fabric/runs/<run_id>/` (the path
   helpers are in `common/fabric_visualizer_files.hpp`), under a temporary name, `<path>.tmp.<pid>`, then renames it
   into place, so a reader never sees a partial file. On a failure it removes the temporary file.

Any failure is rethrown as `Failed to write fabric manifest <path> (unset TT_METAL_FABRIC_GENERATE_MANIFEST to skip)`,
followed by the message starting with `Fabric manifest:` that says what went wrong.

Each run writes into a folder of its own, so a manifest on disk always belongs to the run named by its folder and its
`run.run_id`.

## The JSON

`make_manifest_json` writes the blocks in this order:

| Key                | Written by                | From                                                                                |
| ------------------ | ------------------------- | ----------------------------------------------------------------------------------- |
| `manifest_version` | N/A                       | `k_manifest_version` (`common/fabric_visualizer_files.hpp`), which readers check    |
| `file_kind`        | N/A                       | `k_manifest_file_kind`, `"fabric_manifest"`, so a tool can tell the manifest apart  |
| `run`              | `make_run_json`           | `ManifestFacts::run`                                                                |
| `fabric`           | `make_fabric_json`        | `ManifestFacts::fabric`                                                             |
| `arch`             | `make_arch_json`          | `ManifestFacts::arch`                                                               |
| `vocabulary`       | `make_vocabulary_json`    | `k_vocabulary_lists` (below)                                                        |
| `meshes`           | `make_mesh_json` per mesh | `ManifestFacts::meshes`                                                             |

The JSON keeps keys in the order the writer adds them.

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
- **Values the builder gives each ERISC** are keyed `erisc<r>` (`make_by_erisc_json`): every `fields` object, a sender
channel's `credits`, and a receiver channel's `forwards_on`.
- **Two entries with one key** fail the writer, such as two fields keyed alike.
- **Regions and registers are written as they are.** The writer copies what the passes worked out and computes nothing
of its own (below).

## Types

The manifest names the type each region or register holds; it doesn't describe the type's members. Decode, built from
the same commit, gets the type back from its name ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)).

Each name is already on the model when the writer gets it: a region can only be made for a registered type, so the
writer has nothing to look up or check. It writes the name as a string, next to where the bytes are:

| What | Written by | Keys |
| --- | --- | --- |
| An L1 region | `make_l1_json` | `address`, `num_elements`, `element_size`, `type` |
| A stream register | `make_stream_reg_json` | `stream_id`, `address`, `value_bits`, `type` |
| A core register | `make_core_registers_json` | `name`, `address`, `type` |
| The run's packet header | `make_fabric_json` | `fabric.packet_header`, the header type's name |

A configuration enum field is different: no captured bytes hold it, so its type isn't registered. The writer writes its
`value` and its enumerator's name in lower case (`value_name`), and leaves out the enum's own name.

## Vocabulary

Some keys hold one of a fixed set of names, such as a channel's `status` or a field's `category`. `vocabulary` lists
every name each can hold, so a reader knows them all without the C++ enums.

The lists are `k_vocabulary_lists` in `fabric_manifest_model.hpp`. Each entry names the list and the type whose names it
holds. For an enum, the list is every enumerator in lower case. For the field kinds, it's every alternative of the
`Content` variant, spelled as the `kind` key spells it (`kind_name`, such as `l1` or `stream_reg`).

An enum field isn't a label key: it writes its enumerator's name next to its value (`value_name`), so its enum has no
list here.

## Changing the format

1. Change the writer, and the names in `fabric_manifest_names.hpp` if a key's spelling changes.
2. Update `fabric_manifest_schema.json` and [FORMAT.md](FORMAT.md).
3. If readers can't take the new shape as it is, bump `k_manifest_version` in `common/fabric_visualizer_files.hpp`,
   and the schema's `manifest_version` with it.
4. For a new label key, add a list to `k_vocabulary_lists` and the same name to the schema's `vocabulary`.
