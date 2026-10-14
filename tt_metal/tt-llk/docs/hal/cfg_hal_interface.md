<!-- Ported verbatim from Confluence: "Configuration Registers (hal::cfg)" (https://tenstorrent.atlassian.net/wiki/spaces/LLK/pages/2708799504/Configuration+Registers+hal+cfg), last modified 2026-10-01. Links to other Confluence pages point to their local copies in this folder. -->

# Configuration Registers (hal::cfg)

> **Info**
>
> **CFG field catalog:** [Browse all 340 generated CFG fields](cfg_field_catalog.html) with their descriptions and physical locations.

To get started include the public header, select a generated field and its section, choose the hardware access path, and pass a field value.

```cpp
#include "hal/cfg.h"
```

## Quick start

> **Note**
>
> **Template argument order**
>
> Operations that select an access path consistently use `<Access, Field, Section, ...>`. Additional compile-time arguments, such as a value, word offset, word count, GPR transfer size, WRCFG completion policy, or thread target, follow the section. Builders that do not select an access path use `<Field, Section, ...>`, for example `set<F, S>()` and `from_gpr<F, S>()`.

A compile-time field write specifies the access path, field, repeated section, and value:

```cpp
void configure()
{
    cfg::write<
        cfg::Access::TensixCfgUnit,
        cfg::AluAccCtrl::Fp32_enabled,
        cfg::Sec::S0,
        1 /* value */>();
}
```

The field descriptor supplies the physical address, width, mask, and shift. Because the value is also a template argument, the operation is encoded during compilation and emits an immediate-operand `RMWCIB3` instruction.

Use the compile-time `set<Field, Section, Value>()` form to describe several field updates in one call:

```cpp
cfg::write<cfg::Access::TensixCfgUnit>(
    cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, 0>(),
    cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(),
    cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, 1>(),
    cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0, 0>());
```

The assignments are grouped by physical CFG word. Fields in the same word are combined, while fields at different addresses produce separate hardware updates. In this example, 2 instructions will be generated, one for `AluAccCtrl` fields (`RMWCIB3`) and one for `StaccRelu` fields (`RMWCIB0`)

## Functional model

Every generated Blackhole CFG field belongs to one of the two architectural register scopes represented by `RegisterScope`:

| `RegisterScope` value | Purpose and behavior | Supported access |
| --- | --- | --- |
| `RegisterScope::Thread` | Private to each Tensix thread and organized as 16-bit words. A `SETC16` write replaces the complete word; bits not included in the supplied value are written as zero. | Written through the `Tensix Configuration Unit`. Read through the `RISC MMIO` |
| `RegisterScope::State` | Shared 32-bit configuration used by `ALU`, `PACK`, `UNPACK`, `THCON`, and global hardware. Banked blocks use the `CFG` state selected by the issuing thread. Partial field writes preserve unrelated bits. | Read or written through RISC `MMIO` and the `Tensix Configuration Unit`. `THCON` words may also be written through the `Tensix Scalar Unit`. |

### Register families

Generated field descriptors follow the functional divisions of the Blackhole CFG address space:

| Family | Contains | Example field |
| --- | --- | --- |
| Thread | Per-thread execution controls, CFG-state selection, address modifiers, synchronization, and instruction controls. | `CfgStateId::StateID` |
| ALU | Math formats, accumulation, fidelity, rounding, ReLU, and related execution controls. | `AluAccCtrl::Fp32_enabled` |
| PACK | Packer formats, address generation, masks, counters, tile layout, and destination configuration. | `PackCounters::pack_per_xy_plane` |
| UNPACK | Unpacker source formats, dimensions, strides, address generation, contexts, and control. | `Unpacker[0].Cntx[0].Base` |
| THCON | Thread-controller contexts for pack and unpack, including base addresses, context selection, and tile descriptors. | `Thcon[Reg0].TileDescriptor.InDataFormat` |
| Global | PRNG, RISC, cache, source/destination access, and other shared controls. | `PrngSeed::Seed_Val` |

### Fields, repeated sections, and CFG state

A generated field expression is a typed description of one hardware field. For example, `Thcon[Reg0].TileDescriptor.InDataFormat` identifies its `RegisterScope`, register family, physical word, width, mask, and shift. The descriptor is selected at compile time; kernel code does not repeat the raw address or bit layout.

`Sec::S0` through `Sec::S7` select repeated copies of a descriptor. A section is always explicit in the public operations:

```cpp
cfg::write<
    cfg::Access::TensixCfgUnit,
    cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat,
    cfg::Sec::S1>(format);
```

> **Warning**
>
> `RegisterScope`**,** `Sec`**, and the active CFG-state bank are different concepts.**
>
> `RegisterScope` classifies a field as `Thread` or `State`. `Sec` selects a repeated copy in the descriptor layout. The active bank of `RegisterScope::State` is selected independently by `CfgStateId::StateID`. Tensix instructions use the bank selected for the issuing thread; `Access::MMIO` resolves the same bank on each call, so kernel code never passes a CFG pointer.

### Field groups

Some descriptor families group the fields of one multi-word register. `Thcon[Reg0].TileDescriptor` is such a group: it contains `InDataFormat`, `XDim`, `YDim`, and the other descriptor fields, plus a `Raw` field that starts at the register's first word and spans all four of its words.

A group names a register rather than one field. Whole-word operations, which use their descriptor only as an address anchor, accept a group directly and anchor on its `Raw` field:

- `read_word()`
- GPR transfers, through `write(hal::gpr(...))` and `from_gpr()`
- Prepacked word writes

Writing `Thcon[Reg0].TileDescriptor` and `Thcon[Reg0].TileDescriptor.Raw` produces identical code. A multi-word anchor also bounds the access: a word offset, word count, or GPR transfer must stay within the group. Field operations such as `set()`, single-field `write()`, and `read()` require an individual field, because a group has no mask or shift.

### Hardware access paths

| Access path | Use it for | Hardware operations |
| --- | --- | --- |
| `Access::MMIO` | RISC-side reads and writes for `RegisterScope::State`, and reads for `RegisterScope::Thread`. MMIO does not write the `Thread` scope. | Memory-mapped read, store, or read-modify-write. |
| `Access::TensixCfgUnit` | Normal Tensix-side programming of both register scopes, and `RegisterScope::State` reads into a GPR. | SETC16, RMWCIB, WRCFG, or RDCFG, selected from the operand and the field's `RegisterScope`. |
| `Access::TensixScalarUnit` | Explicit GPR-to-THCON transfers. | REG2FLOP. |

### Compile-time and runtime lowering

The field, section, access path, transfer width, and completion policy are compile-time choices. For Tensix access paths, compile-time values and GPR indices are encoded as instruction immediates; runtime values and GPR indices use runtime operand encoding. Both forms select the same hardware instruction. MMIO remains a memory-mapped C++ access and does not emit a Tensix instruction.

A GPR operand, created with `hal::gpr<Index>()` or `hal::gpr(index)`, names only the register. The transfer width and WRCFG completion policy are template arguments of the operation that consumes it.

| Operand form | Example | Encoding | Hardware instruction |
| --- | --- | --- | --- |
| Compile-time field value | `write<TensixCfgUnit, F, S, Value>()` or `set<F, S, Value>()` | Immediate | `SETC16` or `RMWCIB` |
| Runtime field value | `write<TensixCfgUnit, F, S>(value)` or `set<F, S>(value)` | Runtime | `SETC16` or `RMWCIB` |
| Compile-time GPR index | `hal::gpr<Index>()` | Immediate GPR index | `WRCFG`, `RDCFG`, or `REG2FLOP` |
| Runtime GPR index | `hal::gpr(index)` | Runtime GPR index | `WRCFG` or `REG2FLOP` |

## Operations

Choose an operation based on the form of the data and the hardware path that must perform the access.

| Operation | Use | Lowers to | Behavior |
| --- | --- | --- | --- |
| [Write one field](#write-one-field) | `write<A, F, S>(value)` or `write<A, F, S, Value>()` | `SETC16`, `RMWCIB`, or MMIO store/RMW | Writes one unshifted runtime or compile-time field value. |
| [Write several fields](#write-several-fields) | `write<A>(set(...), set(...))` | `SETC16`, `RMWCIB`, or MMIO store/RMW | Groups assignments by physical word and combines compatible field masks. |
| [Read CFG](#read-cfg) | `read()` or `read_word()` | `RDCFG` or MMIO load | Extracts a field through MMIO, returns a complete MMIO word, or moves a state word to a GPR through RDCFG. |
| [Transfer CFG words through GPRs](#transfer-cfg-words-through-gprs) | `write<A, F, S, Size, Completion>(hal::gpr(...))` or `from_gpr<F, S, Size, Completion>(hal::gpr(...))` | `WRCFG` or `REG2FLOP` | Replaces one or four complete `RegisterScope::State` words from GPRs. |
| [Write prepacked words](#write-prepacked-words) | `write<MMIO, F, S, Count>(std::array)` | MMIO store | Stores `Count` already-packed complete `RegisterScope::State` words through MMIO. |
| [Iterate descriptor collections](#iterate-descriptor-collections) | `forEach()` | No instruction by itself | Expands fixed descriptor collections at compile time. |

In the whole-word operations, `F` may be a field or a [field group](#field-groups).

## Write one field

Use a single-field write when the caller has one unshifted field value. The descriptor provides the address, mask, width, and shift.

### Compile-time value

Place the value after the section when it is known at compile time:

```cpp
cfg::write<
    cfg::Access::TensixCfgUnit,
    cfg::AluAccCtrl::Fp32_enabled,
    cfg::Sec::S0,
    1 /* value */>();
```

This form is available for `Access::TensixCfgUnit`. It encodes the address and value as instruction immediates and lowers to `SETC16` for `RegisterScope::Thread`, or one or more `RMWCIB` instructions for the affected byte lanes of `RegisterScope::State`.

### Runtime value

Pass the value as a function argument when it is computed at runtime:

```cpp
cfg::write<
    cfg::Access::TensixCfgUnit,
    cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat,
    cfg::Sec::S1>(format);

cfg::write<
    cfg::Access::MMIO,
    cfg::PrngSeed::Seed_Val,
    cfg::Sec::S0>(seed);
```

The Tensix path lowers to the runtime encoding of `SETC16` or `RMWCIB`. The MMIO path performs a store for a complete 32-bit field or a read-modify-write for a partial field.

> **Warning**
>
> `SETC16` **replaces a complete 16-bit thread-CFG word.** If several fields share that word, compose them in one grouped write. Bits not supplied by the composed value are written as zero.

## Write several fields

Use `set()` assignments when several independent field values form one configuration step. Both compile-time and runtime assignments use the same grouping rules.

```cpp
cfg::write<cfg::Access::TensixCfgUnit>(
    cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0, 0>(),
    cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0, 1>(),
    cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0, 1>(),
    cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0, 0>());
```

Assignments are grouped by `RegisterScope` and resolved physical word address. Fields in the same word are composed even when they are not adjacent in the argument list. Distinct words are emitted in the order in which their addresses first appear.

- A group containing only `set<Field, Section, Value>()` assignments uses compile-time immediate encoding.
- A group containing any `set<Field, Section>(value)` assignment uses runtime encoding.
- Overlapping assignments are diagnosed during compilation.
- With `Access::MMIO`, every assignment must target `RegisterScope::State`.
- If the order between physical-word updates matters, use separate `write()` calls.

For `RegisterScope::State`, grouping combines compatible masks and preserves unrelated bits. For `RegisterScope::Thread`, grouping composes one complete 16-bit value for each `SETC16` destination word.

## Read CFG

### Read and extract one field through MMIO

```cpp
std::uint32_t format = cfg::read<
    cfg::Access::MMIO,
    cfg::Thcon[cfg::Reg0].TileDescriptor.InDataFormat,
    cfg::Sec::S1>();
```

`read()` masks the selected field and shifts it down to bit zero.

### Read one complete word through MMIO

```cpp
std::uint32_t descriptor_word = cfg::read_word<
    cfg::Access::MMIO,
    cfg::Thcon[cfg::Reg0].TileDescriptor,
    cfg::Sec::S1>();
```

`read_word()` uses the field or group only as an address anchor. It does not apply a field mask or shift. An optional `WordOffset` selects a later physical word; it must stay within the CFG bank, and within the anchor when the anchor spans several words.

For a `RegisterScope::Thread` read, leave `ThreadTarget::Current` selected on the owning TRISC, or explicitly select `T0`, `T1`, or `T2` when another RISC must name the thread-CFG bank:

```cpp
std::uint32_t state_id = cfg::read<
    cfg::Access::MMIO,
    cfg::CfgStateId::StateID,
    cfg::Sec::S0,
    cfg::ThreadTarget::T1>();
```

### Read one state-CFG word into a GPR

```cpp
cfg::read<
    cfg::Access::TensixCfgUnit,
    cfg::PrngSeed::Seed_Val,
    cfg::Sec::S0>(hal::gpr<5>());
```

This lowers to `RDCFG`. It reads the complete containing 32-bit state-CFG word, not only the named field. The destination GPR index must be known at compile time, and `RDCFG` supports only 32-bit transfers.

## Transfer CFG words through GPRs

Use a GPR-backed write when a complete state-CFG word, or four consecutive words, is already present in the Tensix GPR file. The destination anchor must start at bit zero of its word, and the complete transfer must stay within the state-CFG bank and within the anchor when the anchor spans several words.

### Write through the Tensix Configuration Unit

```cpp
cfg::write<
    cfg::Access::TensixCfgUnit,
    cfg::Thcon[cfg::Reg3].Base_address,
    cfg::Sec::S0>(hal::gpr<4>());

cfg::write<
    cfg::Access::TensixCfgUnit,
    cfg::Thcon[cfg::Reg0].TileDescriptor,
    cfg::Sec::S1,
    cfg::GprTransferSize::Bits128>(hal::gpr<16>());
```

These calls lower to `WRCFG`. The transfer size and completion policy follow the section: `Bits32` is the default transfer size; `Bits128` transfers four consecutive words and requires a four-word-aligned destination. `WrcfgCompletion::Wait` is the default completion policy and emits one `NOP` after the `WRCFG`.

> **Warning**
>
> Use `WrcfgCompletion::Deferred` when the next instruction does not read what the `WRCFG` wrote; one instruction of separation is all the hardware needs.

### Write THCON through the Tensix Scalar Unit

```cpp
cfg::write<
    cfg::Access::TensixScalarUnit,
    cfg::Thcon[cfg::Reg0].TileDescriptor,
    cfg::Sec::S1,
    cfg::GprTransferSize::Bits128>(hal::gpr<16>());
```

This lowers to `REG2FLOP`. The scalar-unit path accepts only THCON destinations, and a `Bits128` transfer must remain inside the THCON range. The source GPR index must be in `[0, 63]`, and a `Bits128` source must be four-register aligned. `REG2FLOP` emits no completion `NOP`, so the completion policy has no effect on this path.

### Combine field assignments and GPR transfers

```cpp
cfg::write<cfg::Access::TensixCfgUnit>(
    cfg::set<cfg::AluFormatSpecReg::SrcA_val, cfg::Sec::S0>(src_a_format),
    cfg::from_gpr<
        cfg::Thcon[cfg::Reg3].Base_address,
        cfg::Sec::S0,
        cfg::GprTransferSize::Bits32,
        cfg::WrcfgCompletion::Deferred>(hal::gpr<4>()),
    cfg::set<cfg::DestOffset::Enable, cfg::Sec::S0, 1>());
```

A combined call requires `Access::TensixCfgUnit`. `from_gpr()` operations take part in the same plan as `set()` assignments:

- Field assignments are grouped by physical word across the entire call, including assignments on opposite sides of a `from_gpr()`.
- Each field group is emitted at the position of its first assignment; each GPR transfer is emitted at its own position.
- A GPR transfer owns every bit of its destination words. A `set()` into a word written by a `from_gpr()` in the same call, or two overlapping GPR spans, is diagnosed during compilation.

In this example, the `SrcA_val` update is emitted first, followed by the `WRCFG`, then the `DestOffset` update. When a field write must be ordered after a GPR transfer to the same or a related word, use separate `write()` calls.

## Write prepacked words

### Why use prepacked words?

Use a prepacked write when an existing LLK type already owns the complete physical CFG layout. Types such as `pack_counters_u` contain one packed word, while types such as `unpack_tile_descriptor_u` expose a fixed array of consecutive words through `.val`.

Writing the packed representation directly preserves that whole-word layout and avoids decomposing it into independent field writes. Use this path only when the caller intentionally owns every bit in the destination word or word range.

### Write consecutive words

```cpp
const std::array<std::uint32_t, TILE_DESC_SIZE> descriptor_words {
    tile_descriptor.val[0],
    tile_descriptor.val[1]};

cfg::write<
    cfg::Access::MMIO,
    cfg::Thcon[cfg::Reg0].TileDescriptor,
    cfg::Sec::S0,
    TILE_DESC_SIZE /* word count */>(descriptor_words);
```

This performs `TILE_DESC_SIZE` consecutive MMIO stores beginning at the first word of the descriptor. The anchor supplies only the address; no field mask or shift is applied. The source is a `std::array` with at least the requested number of words, and the complete range must stay within the state-CFG bank. Because the anchor is a field group, the word count must also stay within the descriptor's four words.

### Replace one complete word

A word count of one replaces a single complete word, even when the anchor field is narrower than 32 bits:

```cpp
cfg::write<
    cfg::Access::MMIO,
    cfg::PackCounters::pack_per_xy_plane,
    cfg::Sec::S0,
    1 /* word count */>(std::array<std::uint32_t, 1> {pack_counters.val});
```

In this example, `pack_per_xy_plane` is an 8-bit field, but `pack_counters.val` replaces the complete containing CFG word. A single-field `write<MMIO, F, S>(value)` on the same anchor would instead read-modify-write only the field's bits.

### Update a packed object between writes

When one packed object programs several sections, write it, change the object, and write it again. Each MMIO call resolves the active CFG-state bank itself:

```cpp
cfg::write<
    cfg::Access::MMIO,
    cfg::Thcon[cfg::Reg0].TileDescriptor,
    cfg::Sec::S0,
    TILE_DESC_SIZE>(std::array<std::uint32_t, TILE_DESC_SIZE> {tile_descriptor.val[0], tile_descriptor.val[1]});

tile_descriptor.f.in_data_format = src_b_format;
tile_descriptor.f.x_dim          = src_b_face_dim;
tile_descriptor.f.z_dim          = src_b_num_faces;

cfg::write<
    cfg::Access::MMIO,
    cfg::Thcon[cfg::Reg0].TileDescriptor,
    cfg::Sec::S1,
    TILE_DESC_SIZE>(std::array<std::uint32_t, TILE_DESC_SIZE> {tile_descriptor.val[0], tile_descriptor.val[1]});
```

The first call programs the source-A descriptor in `Sec::S0`. After the packed object is changed, the second call programs the source-B descriptor in `Sec::S1`.

> **Warning**
>
> Prepacked writes are MMIO-only whole-word operations. Do not use them for independent fields or partially packed values. Use a single-field `write()` for one unshifted field value, or grouped `set()` assignments when several fields must be composed.

## Iterate descriptor collections

Some generated descriptor families expose compile-time collections. Use `forEach()` when the same operation must be instantiated for every unpacker, context, register set, or mapping entry.

```cpp
cfg::Unpacker.forEach([&](auto U) {
    cfg::Unpacker[U].Cntx.forEach([&](auto C) {
        cfg::write<
            cfg::Access::MMIO,
            cfg::Unpacker[U].Cntx[C].Base,
            cfg::Sec::S0>(base[U][C]);
    });
});
```

`U` and `C` are compile-time indices, so every selected descriptor remains valid as a template argument. `forEach()` emits no hardware operation by itself; operations in the callable determine the generated accesses.
