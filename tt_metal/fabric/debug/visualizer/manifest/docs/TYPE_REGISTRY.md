# Type registry

`fabric_manifest_type_registry.hpp` lists every C++ type the manifest may name, and gives each one the name the
manifest writes and decode reads. Every L1 region and stream register in the manifest says which of these types its
bytes hold.

## The problem it solves

The manifest says where a router's state lives. Something also has to say what that state is, so that decode can turn
captured bytes into values. Fabric already defines what each piece of state is: the structs and enums the router kernel
and the host share, such as `EDMChannelWorkerLocationInfo` or `EDMStatus`.

So the manifest names the C++ type, and decode, built from the same commit as the run, reads the bytes straight into
that type. Nothing describes a struct's members a second time, and a decoder that uses a member compiles against it.

For that to work, a name has to meet three conditions:

1. **Decode has the type.** The writer must only write names of types decode can map back to.
2. **A name means one type.** Two types with one name would leave decode unable to tell which bytes it has.
3. **The type can be rebuilt from bytes.** Decode copies bytes into the type, so the type must be trivially copyable.

The type registry is how the build guarantees all three. It's one compile-time list of types. The manifest can only name
a type through the registry, and now decode has a list of actual types that it can map the manifest-stated types to.

## What gets registered

A type is registered when decode reads it out of captured bytes: the elements of an L1 region, or a stream register's
word.

| Group | Types |
| --- | --- |
| Words | `uint32_t`: single words, the credit counter arrays, the ETH firmware mailbox, and the counts in stream registers |
| Raw bytes | `std::byte`: what the manifest doesn't type, such as leftover L1 and the perf telemetry buffer |
| Enums in L1 or a stream register | `EDMStatus`, `TerminationSignal`, `ConnectionValue`, `CoordinatedEriscContextSwitchState` |
| Structs in L1 | `handshake_info_t`, `SenderChannelProducerCursor`, `EDMChannelWorkerLocationInfo`, `ChannelTrimmingOverrides`, `CodeProfilingTimerResult`, `FabricTelemetry`, `routing_l1_info_t` |
| Packet headers | Every packet header type fabric can pick for a run (`PacketHeaderTypes`) |

Some types are deliberately left out:

- **A type that only appears inside another one**, such as `WorkerXY` inside `EDMChannelWorkerLocationInfo`. Decode
reaches it through the outer type.
- **Configuration enums** the kernel is given as a value, such as its direction (`eth_chan_directions`) or a NoC
command buffer (`NocCmdBuf`). No captured bytes hold them: the manifest writes the value and its enumerator's name
directly, as an `enum` field.
- **Array types.** An array is a region with more than one element, so its element type is what's registered.

## Registering a type

Add the type to `RegisteredTypes`, in the group it belongs to. That's the only edit that makes a type nameable:

```cpp
using RegisteredTypes = typename detail::concat_type_lists<
    TypeList<
        // ...
        // Structs held in L1.
        erisc::datamover::handshake::handshake_info_t,
        SenderChannelProducerCursor,
        MyNewStruct,
        // ...
    PacketHeaderTypes>::type;
```

### What the build checks

| Check | Fails when | Message |
| --- | --- | --- |
| `get_manifest_type_name<T>()` | `T` isn't registered | "The manifest names only registered types: add T to RegisteredTypes" |
| `are_names_unique` | Two registered types have the same name once namespaces are dropped | "Two registered types have the same name without their namespaces" |
| `are_template_arguments_named` | A registered type's name lost a template argument (see "Quirks" below) | "A registered type's name lost a template argument that equals its default" |
| `are_trivially_copyable` | A registered type can't be copied from bytes | "Decode rebuilds a registered type from its bytes, so every registered type must be trivially copyable" |
| `registers_every_packet_header` | A packet header type fabric can pick isn't in `PacketHeaderTypes` | "A packet header type fabric can pick isn't in PacketHeaderTypes" |
| `register_stream_region<T>` | `T` doesn't fit in a stream register's word | "A stream register holds one word, so its value fits in a uint32_t" |

All of these fail at compile time, so a type the manifest can't name never reaches a run.

## Registering a region

A region is a piece of router state the manifest describes: an L1 region or a stream register. Two functions in
`fabric_manifest_model.hpp` make one:

```cpp
// An L1 region of num_elements Ts, element_size bytes apart from address.
template <typename T>
constexpr content::L1 register_l1_region(uint32_t address = 0, uint32_t num_elements = 1, uint32_t element_size = sizeof(T));

// A stream register holding a T. Every stream has the same set of registers, and `reg` picks one of them (such as
// BUF_SPACE_AVAILABLE). The stream id isn't known yet, so neither is the address: the router pass reads the stream id
// from the kernel's arguments and works out the address with describe_stream_register.
template <typename T>
constexpr StreamRegField register_stream_region(StreamRegister reg);
```

Registering a region does two things, and nothing else:

1. **It returns the region**, with `T`'s registered name as its `type`.
2. **It checks `T`** at compile time: `T` must be registered, and a stream register's `T` must fit in a word.

Nothing is recorded anywhere. "Registering" means the region holds a registered type, so decode can read it.

What a region still needs is its location in this run: an L1 address, or the stream a register belongs to. Where that
comes from depends on where the region is registered.

## Where regions are registered

There are three places, each with its own source for the location:

| Where | Location comes from | Filled in by | Regions |
| --- | --- | --- | --- |
| A field table entry (`fabric_manifest_fields.hpp`) | The kernel argument the entry's `source` names: an L1 address, or a stream id | The router pass's `make_field` | Every `l1` and `stream_reg` field ([FIELDS.md](FIELDS.md)) |
| The router pass (`fabric_manifest_router_pass.cpp`) | The builder's state, or arguments the router pass reads itself | The router pass's `describe_*` function, when it registers the region | Ring buffers, credit counter arrays, leftover L1, and stream-backed credits |
| Describe (`fabric_manifest_describe.cpp`) | The HAL and fabric's architecture constants | Describe's `describe_*` function, when it registers the region | Fabric telemetry, the routing table, the ETH firmware mailbox, base firmware words, the heartbeat word |

### 1. A field table entry

A table entry is `constexpr`, built before any router exists, so it can't know the address. It pairs the argument
that holds the location (`source`) with the region at that location (`content`), and registers the region with no
location:

```cpp
RouterField{
    .key = "edm_status",
    .source = "EDM_STATUS_PTR_ADDR",              // the argument that holds the address
    .category = LIFECYCLE,
    .content = register_l1_region<EDMStatus>(),   // one EDMStatus; the address is filled in later
    .description = "How far the router has come through startup and teardown.",
},
```

For each router, the router pass reads the argument `source` names, and `make_field` copies the entry's region with
the argument as its address. On a T3K, router `E0` on chip `C0` is fed `EDM_STATUS_PTR_ADDR = 98624`:

```cpp
[&](manifest::content::L1 l1) -> manifest::Content {
    l1.address = arg;                                      // 0 -> 98624
    return l1;
},
```

```json
"edm_status": { "category": "lifecycle", "kind": "l1", "address": 98624,
                "num_elements": 1, "element_size": 4, "type": "EDMStatus" }
```

A per-channel or per-edge entry names its argument with a `{}`, filled with the channel's index. For VC0 sender
channel 1, `conn_info`'s source `LOCAL_SENDER_CH_{}_CONN_INFO_ADDR` becomes `LOCAL_SENDER_CH_1_CONN_INFO_ADDR`.

A stream register's argument is a stream id, not an address, so `describe_stream_register` works out the address from
the HAL: the stream's block, plus the register's offset in it, and how many of its low bits hold the value.

```cpp
RouterField{
    .key = "erisc_sync",
    .source = "MULTI_RISC_TEARDOWN_SYNC_STREAM_ID",
    .category = LIFECYCLE,
    .content = register_stream_region<uint32_t>(StreamRegister::REMOTE_SRC),
    ...
},
```

```json
"erisc_sync": { "category": "lifecycle", "kind": "stream_reg", "stream_id": 31, "address": 4290113536,
                "value_bits": 24, "type": "unsigned int" }
```

An argument of 0, or the unused stream id, means the builder didn't allocate the buffer or register, and the field is
written as `null`.

### 2. The router pass, from the builder

Some regions aren't one argument the kernel is fed. The router pass works out their location from the builder's state,
or from several arguments, and registers them with it:

| Function | Region | Location |
| --- | --- | --- |
| `describe_ring_buffer` | A channel's ring buffer, as the run's packet header type with one element per slot | The channel allocator's base address and slot count, and `CHANNEL_BUFFER_SIZE` as the slot size |
| `describe_l1_array<T>` | The four credit counter arrays, as `uint32_t` | Each array's base address argument, with the size taken from the spacing between them |
| `collect_leftover_l1` | Leftover L1, as `std::byte` | From the builder's `get_allocated_l1_end_address` to `max_l1_loading_size` |
| `read_fed_stream` | A credit on stream registers, as `uint32_t` | The `TO_SENDER_<c>_PKTS_COMPLETED_ID` or `TO_SENDER_<c>_PKTS_ACKED_ID` argument, through `describe_stream_register` |

```cpp
// describe_ring_buffer: the allocator's address and slot count, and the channel's slot size, typed as the header
// visit_packet_header_type picks for the run.
register_l1_region<Header>(address, num_slots, slot_size);
```

### 3. Describe, from the architecture

Regions the architecture fixes on every router core are registered once per manifest, in `arch.l1_areas`:

| Function | Region | Location |
| --- | --- | --- |
| `describe_hal_l1_area<T>` | Fabric telemetry (`FabricTelemetry`), the routing table (`routing_l1_info_t`), the ETH firmware mailbox (`uint32_t`) | The HAL's address and size for the area. The size must hold a whole number of `T`s, which is `num_elements` |
| `describe_base_fw_word` | A base firmware word, as `uint32_t` | The HAL's Ethernet firmware mailbox values. `null` when the HAL gives none, as on Wormhole |
| `describe_heartbeat` | The router kernel's heartbeat word, as `uint32_t` | Fabric's heartbeat address for the architecture |

```cpp
// describe_hal_l1_area<FabricTelemetry>: as many FabricTelemetry as the HAL's area holds.
register_l1_region<T>(get_hal_l1_address(hal, area_id), area_size / sizeof(T));
```

### Choosing where to register a new region

- **The kernel is fed its address or stream id as one named argument:** add a field table entry
  ([FIELD_TABLES.md](FIELD_TABLES.md)). The router pass fills in the location, and the field shows up in FIELDS.md.
- **The builder knows where it is, but no single argument says so:** register it in a router pass `describe_*`
  function, with the location from the builder ([ROUTER_PASS.md](ROUTER_PASS.md)).
- **The architecture fixes it on every router core:** register it in describe, with the location from the HAL
  ([DESCRIBE.md](DESCRIBE.md)).

In every case, add the region's type to `RegisteredTypes` first if it isn't there; the build fails until you do.

## Names

A registered type's name is the compiler's spelling of it without its namespaces (`ttsl::short_type_name<T>`, from
`std::source_location`):

| Type | Name |
| --- | --- |
| `EDMStatus` | `EDMStatus` |
| `connection_interface::ConnectionValue` | `ConnectionValue` |
| `erisc::datamover::handshake::handshake_info_t` | `handshake_info_t` |
| `HybridMeshPacketHeaderT<68>` | `HybridMeshPacketHeaderT<68>` |
| `uint32_t` | `unsigned int` |
| `std::byte` | `byte` |
| `ChannelTrimmingOverrides` | `FabricDatapathUsageL1Results<true, 3, 10>` |

## How decode uses it

Decode isn't built yet. This is how it reads a region with the registry; the code is a sketch.

Decode is built from the same commit as the run, so it has the same `RegisteredTypes`. It turns a name back into a type
by walking the list at compile time and comparing each type's name:

```cpp
// Go through RegisteredTypes and ensure the type corresponding to `name` is registered.
template <typename F>
void visit_registered_type(std::string_view name, F&& f) {
    const bool found = []<typename... Ts>(TypeList<Ts...>, std::string_view name, F& f) {
        return ((manifest_type_name<Ts> == name && (f.template operator()<Ts>(), true)) || ...);
    }(RegisteredTypes{}, name, f);
    TT_FATAL(found, "{} names no registered type: was the manifest written by another build?", name);
}
```

An L1 region is `num_elements` elements, `element_size` bytes apart from `address`. Element `i` is the `sizeof(T)`
bytes at `address + i * element_size`, copied into a `T`:

```cpp
// Element `i` of `region`, from the capture's copy of the core's L1.
template <typename T>
T read_element(std::span<const std::byte> l1, const Region& region, uint32_t i) {
    std::array<std::byte, sizeof(T)> bytes;
    std::memcpy(bytes.data(), l1.data() + region.address + (i * region.element_size), sizeof(T));
    return std::bit_cast<T>(bytes);
}

visit_registered_type(region.type, [&]<typename T>() {
    for (uint32_t i = 0; i < region.num_elements; ++i) {
        show(read_element<T>(l1, region, i));
    }
});
```

A stream register is one word at `address`. Decode keeps the word's low `value_bits` bits, since the bits above can hold
other state, and reads that value as the register's type.

How decode then shows a type's members to a person is decode's own concern. The registry only gets it the type.

## Quirks

**A default template argument drops out of the name.** The compiler leaves out a template argument that equals its
default: `LowLatencyPacketHeaderT<0>` comes out as `LowLatencyPacketHeaderT<>`, and `HybridMeshPacketHeaderT<36>` as
`HybridMeshPacketHeaderT<>`. Those two names are written by hand, next to `manifest_type_name`. If a default changes, or
a new registered template uses its default, `are_template_arguments_named` fails on the `<>` and names what to add.

**An alias is named as what it aliases.** `uint32_t` is `unsigned int`, and `ChannelTrimmingOverrides` is
`FabricDatapathUsageL1Results<true, 3, 10>`. A reader looking for the alias in the manifest won't find it.

**Names drop namespaces.** Two registered types with the same name in different namespaces fail
`are_names_unique`. Give one of them a hand-written name, next to the two above.

**An element can be bigger than its type.** When `element_size` is larger than `sizeof(T)`, only the first `sizeof(T)`
bytes of each element are typed, and the rest are raw. A ring buffer slot is typed as the run's packet header, so the
header is the typed part and the payload after it stays raw.

**Host and device must lay a type out the same way.** Decode copies bytes the device wrote into the host's version of
the type. That holds for the registered types because they're the structs host and device already share, with
fixed-width members. A registered type with a pointer, a `size_t` or a `long` member would differ between the 64-bit
host and the 32-bit RISC-V cores.

**A new header size registers itself; a new header template doesn't.** `PacketHeaderTypes` expands each value in
`PKT_HDR_1D_EXTENSION_WORDS` into a `LowLatencyPacketHeaderT<N>`, and each value in `PKT_HDR_2D_ROUTE_BUFFER_SIZES` into
a `HybridMeshPacketHeaderT<N>`.

- **A new size for one of those templates** is registered with no edit.
- **A new header template**, such as a routing mode with its own header struct, has to be added to
`detail::packet_header_type_list` by hand, since that names the two templates. `registers_every_packet_header` fails the
build until it is: it walks fabric's own `for_each_packet_header_type` and checks each header it picks is registered.

**Configuration enums carry a name the manifest doesn't write.** An `enum` field keeps its enum's name
(`ttsl::short_type_name<E>`) in the model, for FIELDS.md and error messages. The manifest writes only the value and its
enumerator's name, so the unregistered enum's name never looks like a registered type to a reader.
