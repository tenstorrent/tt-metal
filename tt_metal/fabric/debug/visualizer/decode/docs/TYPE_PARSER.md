# Type parser

[`type_parser.py`](../type_parser.py) reads the types the fabric manifest describes, its structs, unions and enums,
checks them, and decodes bytes into values with them. Every type it knows comes from the manifest, so it holds no
fabric names: a struct, union or enum the fabric adds, or a layout that changes, decodes with no change here. That is
case 1 of "What decode reads, and what it knows" in the [README](../README.md).

## What it takes and returns

- **`get_types_from_manifest(manifest)`** reads the layouts for the manifest's `run.arch` (`archs.<arch>.types`) and its
  `enums`, once per manifest. It checks all of them before any bytes are read, and gives a `Types`.
- **`decode_value(node, raw, offset, types, where)`** decodes the bytes at `raw[offset:]` that a manifest node
  describes.
  - The node is anything with a schema and the keys that size it: an L1 region, an area, or the packet header's
    `{size, schema}`.
  - `raw` is usually a snapshot's whole `.bin`, and `offset` is where the snapshot says the node's bytes start in it.
  - `where` is the node's place in the manifest, for messages.
- **`decode_struct(name, raw, offset, types)`** decodes one struct by its name in `types`.
- **`get_shape(node, types, where)`** gives a node's checked layout without decoding it.

```python
types = get_types_from_manifest(manifest)
area = manifest["archs"][arch]["areas"]["routing_table"]
table = decode_value(area, raw, offset, types, f"archs.{arch}.areas.routing_table")
table["my_mesh_id"], table["state_manager"]["state"]  # 0, {"raw": 1, "name": "running"}
```

What comes back is plain data (ints, dicts, lists, strings and `None`), so it writes to JSON as it is. A description
decode can't use raises `ManifestError`, naming its key. Bytes past the end of `raw` raise `ValueError`.

## The schema forms, and what each decodes to

The same keys describe a value wherever it sits (manifest README, "Regions, stream registers and schemas"):

| Schema, and the keys beside it | Decoded as |
| --- | --- |
| `u8` to `u64`, `i8` to `i64` | an int, little-endian, signed for `i` |
| `enum:E` | `{raw, name}`, with `name` null when `E` has no name for the value |
| `struct:S` | `{member: value}` in member order, with `pad` members left out |
| `union:U`, on a struct member with `tag` | `{member, value}` for the member the tag picks, or `{member: null, bytes}` (see [Tagged unions](#tagged-unions)) |
| `bytes` | `{bytes: {offset, size}}`, where the bytes are in `raw`. They aren't copied |
| `pad` | left out of its struct |
| any of these with `num_elements` and `size_per_element` | a list of the element's values |
| `enum:E` with `num_elements` and `bits_per_entry` | a list of `{raw, name}` (see [Packed tables](#packed-tables)) |

- **Integers.** A C++ `bool` is written as `u8`, so it decodes to 0 or 1.
- **Enums.** The manifest gives an enum no width, so a value is as wide as the node or member that holds it.
  - A value with no name keeps its raw value, with `name` null, so the caller can report it. That happens for a bitmask
    such as `DynamicStatistics`, whose combined values have no name, and for a corrupted word.
- **Bytes.** Only an offset into `raw` and a size come back, so nothing is copied: ring buffers are most of a snapshot's
  bytes. An array of `bytes`, such as a ring buffer of slots, gives each element's own offset.
- **Member names.** They are unique within a struct, except `pad`: the HAL's layouts call every gap `pad`, so
  `kernel_config_msg_t` has three of them.

## Tagged unions

A union is a struct member whose bytes hold one of several types. Another member of the same struct, its `tag`, says
which. The packet header's `command_fields` is one, tagged by `noc_send_type`.

- **The tag** names a sibling member that holds one value of the union's `tag_schema` enum.
- **Order.** The plain members are decoded first, then the unions, since a tag can come after its union. In the 2D
  header, `noc_send_type` is at byte 42, and `command_fields` at byte 0. The keys keep member order either way.
- **Picking a member.**
  1. The member whose `tag_value` is the tag's name.
  2. If none has it, the `default_member`. That holds for any other tag value, including one with no name.
  3. If there is no default either, `{member: null, bytes}`: the union's bytes, as raw. The manifest README gives the
     same rule.
- **Size.** A member reads only its own size, which can be smaller than the union's.
- **Refused at load:**
  - a tagged union nested in a union;
  - an array of unions;
  - a union with no tag.

  `decode_value` also refuses a tagged union given to it directly, since there is no sibling to read the tag from.

## Packed tables

A packed table is `num_elements` entries of `bits_per_entry` bits in `size` bytes, each entry a value of the table's
enum. The routing table's two direction tables are packed `compressed_routing_values`, and a scatter write's
`chunk_encoding` is packed `NocScatterWriteChunkEncoding`.

- **Reading.** Entries are read lowest bits first, from one run of bits across the bytes, so an entry can straddle two
  bytes. They are read with [`routes.read_packed`](../routes.py), the route decoder's rule under "Packed values" in
  [ROUTES.md](ROUTES.md), so the rule lives in one place.
- **Decoding.** Each entry decodes like any enum value, as `{raw, name}`. A raw value no name covers keeps `name` null,
  such as a corrupted direction table entry.

## What it checks, before any bytes

`get_types_from_manifest` checks every type, and `get_shape` every node it's given. Each failure raises
`ManifestError` naming the key:

| What | Checked |
| --- | --- |
| **Schemas** | Each is one of the forms above, and resolves: `struct:` and `union:` to a type of that kind in `types`, `enum:` to an enum in `enums` |
| **Sizes** | An integer element is its width, and a struct or union element is the type's size. An array's count times its element size is its size, and a packed table fits in its bytes. Sizes and counts are at least 1 |
| **Keys** | `size_per_element` and `bits_per_entry` aren't both given, and neither comes without `num_elements`. A packed table holds an `enum:`. `tag` is on `union:` members only, and every `union:` member has one |
| **Structs** | Members lie inside the struct, and names other than `pad` are unique |
| **Unions** | `tag_schema` is an enum. Members start at 0, fit in the union, and have unique names. Each `tag_value` names a value of the enum, once. `default_member` names a member |
| **Tags** | A tag names a sibling holding one value of its union's tag enum |
| **Enums** | No value has two names |
| **Types** | A type's `kind` is absent, meaning a struct, or `union` |

The manifest writer checks most of this too: `check_schemas_resolve` when it writes, and `validate_members` at compile
time. Decode checks again because it may read a manifest that another checkout's writer wrote. A check that fails here
names the key, instead of decoding bytes wrong.

## Design decisions

- **Every type's size first, then the layouts.** A member can name a type that comes later in the manifest, so
  `get_types_from_manifest` reads every type's size before any layout. That table exists only while the types are read.
  Afterwards `Types.size_of` gets a size from the type itself.
- **Members are parsed once.** Each member's layout is read into a `Shape` when the types are read. A region's or area's
  node is parsed when it is decoded, and a router has few of them.
- **The four kinds of type are `StructType`, `UnionType`, `EnumType`,** and the integer, `bytes` and `pad` forms. They
  are frozen dataclasses. The decoders take a `Types` from `get_types_from_manifest`, which has checked everything in
  it.

## What it doesn't cover

- **Bitfields.** None of today's structs has any, and the schema grammar has no form for one.
- **Flags.** The manifest doesn't mark an enum as a bitmask, so combined values come out with no name.
- **Meaning.** It doesn't know which struct is a routing table, or what a value means. That is for decode's other
  parts, such as [`routes.py`](../routes.py).

## When the fabric changes

| Change | What happens here |
| --- | --- |
| A struct, union or enum is added, or a layout changes | Nothing: it decodes from the manifest |
| An enum gains a value, or a union a member | Nothing |
| A new schema form, such as a float or a bitfield | The writer and this parser both change. Until then, loading fails, naming the schema |
| A tagged union nested in a union, or an array of unions | Loading fails, naming the member. This parser would need the case added |

## Files cited

| File | Symbols |
| --- | --- |
| `tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp` | `schema_name` |
| `tt_metal/fabric/debug/visualizer/manifest/fabric_manifest.cpp` | `types_json`, `make_enums_json`, `check_schemas_resolve` |
| `tt_metal/fabric/debug/visualizer/manifest/struct_layout.hpp` | `validate_members`, `LAYOUT_PAD`, `LAYOUT_PACKED`, `LAYOUT_TAGGED_UNION`, `LAYOUT_UNION_DEFAULT` |
| [`../../manifest/README.md`](../../manifest/README.md) | "Regions, stream registers and schemas" |
| [`ROUTES.md`](ROUTES.md) | "Packed values" |
