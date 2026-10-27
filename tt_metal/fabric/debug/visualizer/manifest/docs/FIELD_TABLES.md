# Field tables

Most of what the manifest records about a router is a value its kernel was fed: an L1 address, a stream register id,  
a flag, a count. The manifest reads each of these through a **field table** in `fabric_manifest_fields.hpp`: one entry  
per value, saying which kernel argument feeds it and how to read it. The router pass reads the tables ([ROUTER_PASS.md](ROUTER_PASS.md)). Facts that need logic beyond reading one argument (computed facts) are code in the router pass  
instead.

[FIELDS.md](FIELDS.md) lists every entry, generated from the tables. This doc explains what an entry says, how it's
read, and how the tables make sure nothing the kernel is fed goes unaccounted for.

## A field in the manifest

A field is keyed by name, and says what kind of value it holds and how to read it. One of each kind, from a router:

```json
"edm_status":                      { "category": "lifecycle", "kind": "l1", "address": 98496, "size": 4,
                                     "schema": "enum:EDMStatus", "cleared_by_host": false },
"erisc_sync":                      { "category": "lifecycle", "kind": "stream", "stream_id": 31,
                                     "register": "remote_src", "schema": "u32" },
"handshake_sender":                { "category": "lifecycle", "kind": "flag", "value": true },
"producer_credit_return_noc":      { "category": "flow_control", "kind": "number", "value": 0 },
"producer_credit_return_cmd_buf":  { "category": "flow_control", "kind": "enum", "value": 3,
                                     "schema": "enum:NocCmdBuf" }
```

A field is `null` when the builder didn't allocate what it describes, such as an L1 region whose address is 0 or a
stream register whose id is the unused id. A field the builder doesn't emit at all on this run, such as a 2D-only
argument on a 1D run, is left out.

## The tables

There's one table per place a field sits. Each is read once per router, ERISC, channel or edge, from the argument
values the builder fed to the kernel ([BUILDER_HOOKS.md](BUILDER_HOOKS.md) covers how those are recorded):

| Table | Entries | Sits at | Read from |
| --- | --- | --- | --- |
| `k_router_fields` | 26 | `routers.<key>.fields` | Every ERISC's arguments, which must agree |
| `k_erisc_fields` | 10 | `routers.<key>.eriscs.erisc<n>.fields` | That ERISC's arguments |
| `k_sender_channel_fields` | 8 | `routers.<key>.channels.senders.vc<v>.ch<c>.fields` | The arguments of the ERISCs that run the channel. An ERISC that doesn't run it may have the same value or 0 |
| `k_receiver_channel_fields` | 8 | `routers.<key>.channels.receivers.vc<v>.ch<c>.fields` | Every ERISC's arguments, which must agree |
| `k_downstream_edge_fields` | 2 | `routers.<key>.intra_chip_downstream_edges.vc<v>.edge<e>.fields` | Every ERISC's arguments, which must agree |

## An entry

Each entry is a `RouterField`. Every member except `source` is wrapped in `Required`, which has no default, so an entry
that leaves one out doesn't compile.

```cpp
RouterField{
    .key = "conn_info",
    .source = NamedArg{"LOCAL_SENDER_CH_{}_CONN_INFO_ADDR", ArgIndex::COMPACT},
    .category = CONTROL_INFO,
    .content = l1<EDMChannelWorkerLocationInfo>(),
    .description = "Where the producer writes where it is, and the router its read counter.",
},
```

| Member | Says |
| --- | --- |
| `key` | The field's key in the manifest |
| `source` | Which value the kernel is fed holds the field (below) |
| `category` | What the field is for (below) |
| `content` | The field's kind, and the type at its address or in its register (below) |
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
| `l1<T>()` | `l1` | An L1 address holding a `T` | `null` when the address is 0. Also says whether fabric init clears the region (`cleared_by_host`) |
| `stream<T>(register)` | `stream` | A stream register id, and which of its registers holds a `T` | `null` for the unused stream id |
| `content::Number{}` | `number` | A plain value | |
| `content::Flag{}` | `flag` | 0 or 1 | Any other value fails |
| `enum_of<E>()` | `enum` | A value of enum `E` | A value that isn't one of `E`'s enumerators fails |

`T` and `E` must be types the struct layouts can describe ([STRUCT_LAYOUTS.md](STRUCT_LAYOUTS.md)). The field's schema
names the type, and `types` describes it.

## Reading a table

The router pass reads each table with `collect_fields`. For each entry, it skips a `NamedArg` the builder didn't emit,
reads the value with the channel's or edge's indices filled in, and makes the field: an L1 address or stream id fills
in the content, and a flag or enum value is checked. A value read from more than one ERISC must agree, as the tables
section says, or the router pass fails naming the ERISC that differs.

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

- **The router pass can't read an unlisted argument.** Its argument reader (`get_named_arg`) fails for a name no table
reads and `k_router_pass_args` doesn't list, so the lists stay true to the code.
- **A read argument must exist.** One the builder didn't feed fails: "missing fabric router named compile-time
argument".
- **At compile time** (`static_assert` in `fabric_manifest_fields.hpp`): an argument's `{}` count matches its
`ArgIndex`, only per-channel and per-edge tables use indices or builder members, and nothing is both read and listed
as unrecorded.

## Adding a field

1. Add an entry to the table for where it sits, with its key, source, category, content and description.
2. If its content names a struct, the struct needs a layout ([STRUCT_LAYOUTS.md](STRUCT_LAYOUTS.md)). An enum needs

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
