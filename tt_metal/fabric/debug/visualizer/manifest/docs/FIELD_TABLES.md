# Field tables

Most of what the manifest records about a router is a value its kernel was fed: an L1 address, a stream register id, a
flag, a count. The manifest reads each of these through a **field table** in `fabric_manifest_fields.hpp`: one entry per
value, saying which kernel argument feeds it and how to read it. The router pass reads the tables
([ROUTER_PASS.md](ROUTER_PASS.md)). Facts that need logic beyond reading one argument (computed facts) are code in the
router pass instead.

[FIELDS.md](FIELDS.md) lists every entry, generated from the tables. This doc explains what an entry says, how it's
read, and how the tables make sure nothing the kernel is fed goes unaccounted for.

## A field in the manifest

A field is keyed by name, and says what kind of value it holds. Each ERISC a router runs on has its own fields, since
the builder gives each ERISC its own arguments, so a `fields` object is keyed by ERISC first. One field of each kind,
from ERISC0 of router `E0` on a T3K's chip `C0`:

```json
"edm_status":                      { "category": "lifecycle", "kind": "l1", "address": 98624, "num_elements": 1,
                                     "element_size": 4, "type": "EDMStatus" },
"erisc_sync":                      { "category": "lifecycle", "kind": "stream_reg", "stream_id": 31,
                                     "address": 4290113536, "value_bits": 24, "type": "unsigned int" },
"handshake_sender":                { "category": "lifecycle", "kind": "flag", "value": true },
"producer_credit_return_noc":      { "category": "flow_control", "kind": "number", "value": 0 },
"producer_credit_return_cmd_buf":  { "category": "flow_control", "kind": "enum", "value": 3,
                                     "value_name": "at_cmd_buf" }
```

An `l1` or `stream_reg` field says where some router state is and the registered type it holds, for decode to read
([TYPE_REGISTRY.md](TYPE_REGISTRY.md)). A `flag`, `number` or `enum` field is the value itself: what the kernel was
configured with.

A field is `null` when the builder didn't allocate what it describes, such as an L1 region whose address is 0 or a
stream register whose id is the unused id. A field the builder doesn't emit at all on this run, such as a 2D-only
argument on a 1D run, is left out.

## The tables

There's one table per place a field sits. Each is read once per router, channel or edge, and within that once per
ERISC, from the arguments the builder gave that ERISC ([BUILDER_HOOKS.md](BUILDER_HOOKS.md) covers how those are
recorded). Each ERISC's values are recorded as given: two ERISCs can differ, such as on which channels each runs, and
nothing checks that they agree.

| Table | Entries | Sits at |
| --- | --- | --- |
| `k_router_fields` | 37 | `routers.<key>.fields.erisc<r>` |
| `k_sender_channel_fields` | 8 | `routers.<key>.channels.senders.vc<v>.ch<c>.fields.erisc<r>` |
| `k_receiver_channel_fields` | 8 | `routers.<key>.channels.receivers.vc<v>.ch<c>.fields.erisc<r>` |
| `k_downstream_edge_fields` | 2 | `routers.<key>.intra_chip_downstream_edges.vc<v>.edge<e>.fields.erisc<r>` |

The router table includes what differs between a router's ERISCs by design, such as each one's role in the startup
sync and which channels it services.

## An entry

Each entry is a `RouterField`. Every member except `source` is wrapped in `Required`, which has no default, so an entry
that leaves one out doesn't compile.

```cpp
RouterField{
    .key = "conn_info",
    .source = NamedArg{"LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", ArgIndex::COMPACT},
    .category = CONTROL_INFO,
    .content = register_l1_region<EDMChannelWorkerLocationInfo>(),
    .description = "Where the producer writes where it is, and the router its read counter.",
},
```

| Member | Says |
| --- | --- |
| `key` | The field's key in the manifest |
| `source` | Which value the kernel is fed holds the field (below) |
| `category` | What the field is for (below) |
| `content` | The field's kind and, for an L1 region or stream register, the registered type it holds (below) |
| `description` | One sentence for [FIELDS.md](FIELDS.md) and for readers of the manifest |

### Source

A field is read from one of two places:

- `NamedArg`: a named compile-time argument, such as `"EDM_STATUS_PTR_ADDR"`. Almost every field is one.
- `BuilderMember`: a function that reads a member of the router's `FabricEriscDatamoverBuilder`. It's for the few
values the kernel gets as runtime arguments: those are positional, with no name to look up, so the field reads the
builder member that `get_runtime_args` pushes instead. Only per-channel and per-edge tables use it.

A per-channel or per-edge argument's name has a `{}` for each index it takes, filled in for the channel or edge being
read. Its `ArgIndex` says which index:

| `ArgIndex` | Fills `{}` with | Example |
| --- | --- | --- |
| `NONE` | Nothing: the name has no `{}` | `EDM_STATUS_PTR_ADDR` |
| `COMPACT` | The channel's index over this router's own channel counts, as most per-channel arguments are indexed | `SENDER_CH_{}_IS_INJECTION` |
| `FABRIC_POSITION` | The channel's index over the fabric's maximum counts, as the free-slots stream table is indexed | `SENDER_CHANNEL_{}_FREE_SLOTS_STREAM_ID` |
| `VC_AND_EDGE` | The edge's VC, then its edge number | `VC{}_FREE_SLOTS_FROM_DOWNSTREAM_EDGE_{}_STREAM_ID` |

A `NamedArg` also says when the builder emits it (`Emitted`): `ALWAYS`, `FABRIC_2D` (only with 2D routing on), or
`CHANNEL_TRIMMING_CAPTURE` (only with the channel trimming capture on). A field whose argument isn't emitted on this
run is left out.

### Category

What the field is for, so a reader can group a router's fields:

| Category | For |
| --- | --- |
| `LIFECYCLE` | Startup, handshake, sync and teardown, and which channels each ERISC runs |
| `KERNEL_PARAMS` | Settings the kernel runs with, such as context switching, NOC and command buffer choices, and credit amortization |
| `FLOW_CONTROL` | Moving packets: free slots, packet counts, and where credits return |
| `CONTROL_INFO` | How a producer connects to a channel: the connection's state, where the producer writes its details, and the semaphores the connection uses |
| `DIAGNOSTICS` | Telemetry, profiling and channel trimming capture |

### Content

The field's kind, in the JSON's `kind`:

| Content | `kind` | The argument is | Notes |
| --- | --- | --- | --- |
| `register_l1_region<T>()` | `l1` | An L1 address holding a `T` | `null` when the address is 0. The router pass adds the address |
| `register_stream_region<T>(register)` | `stream_reg` | A stream id. The entry says which of the stream's registers holds a `T` | `null` for the unused stream id. The router pass adds the register's address and how many of its bits hold the value, from the HAL |
| `content::Number{}` | `number` | A plain value | |
| `content::Flag{}` | `flag` | 0 or 1 | Any other value fails |
| `enum_of<E>()` | `enum` | A value of enum `E` | Written with its enumerator's name. A value that isn't one of `E`'s enumerators fails |

`T` must be registered ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)), and the field names it by its registered name. The
entry registers the region with no location; the router pass fills it in when it reads the argument. TYPE_REGISTRY.md's
"Where regions are registered" walks an `l1` entry and a `stream_reg` entry through with real values.

`E` is a configuration value the kernel is given, so it isn't registered: the manifest writes the value and its
enumerator's name.

## Reading a table

The router pass reads each table with `collect_fields_by_erisc`, once per router, channel or edge, and within that once
per ERISC. For each entry:

1. **Skip it if it isn't emitted.** A `NamedArg` the builder doesn't emit on this run, such as a 2D-only argument on
   1D, is left out.
2. **Read the value.** The argument's name gets the channel's or edge's indices in its `{}`, and is read from this
   ERISC's arguments.
3. **Make the field** (`make_field`), by the entry's content:

| Content | What the value becomes |
| --- | --- |
| L1 region | The region's address. An address of 0 makes the field `null` |
| Stream register | The stream id. `describe_stream_register` works out the register's address and value width from the HAL. The unused stream id makes the field `null` |
| Number | The value as it is |
| Flag | `true` or `false`. Anything other than 0 or 1 fails |
| Enum | The value and its enumerator's name. A value that isn't one of the enum's enumerators fails |

## Every kernel input is accounted for

The router pass starts each router by checking its kernel inputs (`check_kernel_inputs_accounted`). Every named
argument any ERISC is fed, and every define, must be one of:

- **Read by a field table.**
- **Read by the router pass's own code**, listed in `k_router_pass_args` or `k_router_pass_defines`. These are the
arguments the router pass needs for computed facts, such as `NUM_ACTIVE_ERISCS` or `IS_SENDER_CHANNEL_{}_SERVICED`.
- **Left out on purpose**, listed in `k_unrecorded_args` or `k_unrecorded_defines` with a reason: recorded elsewhere
(`NUM_LOCAL_EDMS` is the chip's `local_sync`), the same on every router (`MAX_NUM_VCS`), or not yet described.

Otherwise fabric init fails, naming the input. A new argument therefore can't slip past the manifest: whoever adds it
decides whether the manifest records it.

Arguments match by family: a run of digits or a `{}` matches any other, so `SENDER_CH_{}_IS_INJECTION` covers every
channel's. Defines match exactly.

The other checks:

- **The router pass can't read an unlisted argument.** It reads every name through `RouterArgs`, which fails for a
name no table reads and `k_router_pass_args` doesn't list, so the lists stay true to the code.
- **A read argument must exist.** `RouterArgs` fails for one the builder didn't feed: "... reads ..., but ERISC<n>
wasn't given it".
- **At compile time** (`static_assert` in `fabric_manifest_fields.hpp`): an argument's `{}` count matches its
`ArgIndex`, only per-channel and per-edge tables use indices or builder members, and nothing is both read and listed
as unrecorded.

## Adding a field

1. Add an entry to the table for where it sits, with its key, source, category, content and description.
2. If its content is an L1 region or stream register of a type that isn't registered yet, add the type to
   `RegisteredTypes` ([TYPE_REGISTRY.md](TYPE_REGISTRY.md)). The build fails until you do. A configuration enum needs
   nothing unless enchantum can't find its enumerators.
3. If the argument was listed in `k_unrecorded_args`, remove it from there. The compile-time check fails until you do.
4. Regenerate [FIELDS.md](FIELDS.md) (below). `ManifestDocs.UpToDate` fails until you do.

## Regenerating FIELDS.md

`FIELDS.md` is generated from the field tables and the unrecorded lists. To update it, run the docs test with
`TT_METAL_FABRIC_MANIFEST_WRITE_DOCS` set to the file to write:

```bash
TT_METAL_FABRIC_MANIFEST_WRITE_DOCS=tt_metal/fabric/debug/visualizer/manifest/docs/FIELDS.md \
    ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ManifestDocs.*
```
