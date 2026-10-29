# Debug event ids: a worked example

This is a framework for giving ids to debug events in device-side code. Each event's id is hard-coded into a
load-immediate instruction. When the instruction is executed, it puts the id into a CPU register without reading
any memory. That lets the rest of the event-reporting code write the id from the register to the event's place in
L1. The event's metadata never reaches the device. The linker script, `tt_metal/hw/toolchain/main.ld`, tags the
tables that hold it, and the strings they point to, with `(INFO)`, so they are kept in the ELF but never allocated on
the device. The host reads them, and the device only ever sees the id. Generating and placing that hard-coded
instruction takes several steps, listed below.

| step | stage | action | the id is |
|---|---|---|---|
| 1 | source | the debug event site is written, with its metadata struct | — |
| 2 | preprocess | the macro names the site's handle `__tt_observability_event_0_6_9bb67bec457229db` (a name, not the id) | not yet a number |
| 3 | compile | the metadata struct becomes assembler directives, plus `lui`/`addi` | not yet a number |
| 4 | link | the linker places the handle | `0x06800005` |
| 5 | link | the linker fills in the instructions | `0x06800005` |
| 6 | link | the linker fills in the record | `0x06800005` |
| 7 | link | the relocations are kept | `0x06800005` |
| 8 | cache | the ELF is stored; nothing of this is loadable | `0x06800005` |
| 9 | load | the host reserves a block for the image | `51` |
| 10 | load | the host rewrites the instructions | `51` |
| 11 | load | the host rewrites the record | `51` |
| 12 | load | the host registers the site and decodes its metadata | `51` |
| 13 | run | the device packs the id into a marker | `51` |
| 14 | decode | the host names the marker | `51` → `T1_Zone5`, colour `0x42D4F4`, unit microseconds |

This page follows one debug event through every stage, from the line of device-side source that declares it, through
compile, link and load, to the device sending its id and the host turning that id back into the event's name and
metadata.

**The event.** For this example we picked a scoped event that carries three fields of metadata, a name, a colour and a
time unit: `DeviceZoneScopedNCU("T1_Zone5", 0x42D4F4, TimeUnit::Microseconds)`. The unit is how a consumer should
present the zone's duration; the device and the host still count in cycles. It marks the start and end of a region of code. It sits
in the demo compute kernel next to nine others (`T1_Zone0` … `T1_Zone9`), which carry the two-field `ZoneColorMeta`. We
chose it because it is in the middle of the kernel, so its numbers are not zeros. Every other kind of debug event gets
its id the same way.

**The data.** Every value below is real. It was read from the JIT-built ELF and the loader's output on a Blackhole
p100a, running the demo program:

```
test_streaming_profiler_zones --gx 1 --gy 1 --iters 2 --markers 1
```

Why it works this way is in [the design document](OBSERVABILITY_EVENT_IDS.md); the metadata rules and the record
layout are in `tt_metal/hw/inc/hostdev/observability_event_meta.h`. This page only shows each step and where each number
comes from.

Code identifiers on this page (`DeviceZoneScopedN`, the `.tt_observability_event_*` sections, `TT_PROFILER_TU_ID`, file names and
log text) are quoted exactly as they are in the source today; they will be renamed separately.

## Before you start: the fixed numbers

These are constants, chosen once in the source. Every other number on this page is derived from them or from
the kernel.

| value | name | defined in | what it is |
|---|---|---|---|
| `0x06600000` | `TT_OBSERVABILITY_EVENT_STR_VMA` | `tt_metal/hw/toolchain/main.ld` | address the linker gives `.tt_observability_event_str` (the strings) |
| `0x06700000` | `TT_OBSERVABILITY_EVENT_META_VMA` | `tt_metal/hw/toolchain/main.ld` | address the linker gives `.tt_observability_event_meta` (the records) |
| `0x06800000` | `TT_OBSERVABILITY_EVENT_IDS_VMA` | `tt_metal/hw/toolchain/main.ld` | address the linker gives `.tt_observability_event_ids` (the handles) |
| `16` | `kRecordHeaderBytes` | `tt_metal/hw/inc/hostdev/observability_event_meta.h` | bytes of a record's fixed header; the site's metadata fields follow it |
| `0xFFFF` | `TT_ZONE_STALL_ID` | `tt_metal/hw/inc/hostdev/observability_event_id.h` | reserved id; the space handed out is `0 … 0xFFFE` |

The three addresses are **made up**. None of these sections is ever loaded onto the device, so their addresses
do not refer to real memory; they only have to exist so the linker can compute pointers between the sections.
They continue a pattern already in `main.ld` (DPRINT's string sections sit at `0x06400000` and `0x06500000`): 1 MB
apart, well away from the device's real memory map. `0x06800000` has one extra requirement, explained in step 10.

**Reading hex dumps.** RISC-V stores a 32-bit number lowest byte first ("little-endian"), and `readelf -x` prints
bytes in file order. So the number `0x06800005` is stored as the bytes `05 00 80 06` and shows up in a dump as
`05008006`. To read a dumped word as a number, reverse its byte pairs.

---

## 1. Source: the debug event site

The TRISC1 (math) kernel opens the event:

```cpp
{
    DeviceZoneScopedNCU("T1_Zone5", 0x42D4F4, kernel_profiler::TimeUnit::Microseconds);
    ...   // ~12 us of work
}
```

It sits at **line 126** of `test_streaming_profiler_zones/kernels/zones_compute.cpp` (through the kernel's `ZONE_U`
macro); the host reports that line at the end. It is the sixth of the kernel's ten scoped events, `T1_Zone0` …
`T1_Zone9`.

**The site's metadata is a struct.** Each zone macro takes exactly the fields of one fixed type and passes the site
a constant value of it; the host gets that value back, field by field. `DeviceZoneScopedN(name)` passes
`ZoneMeta{name}`, `DeviceZoneScopedNC(name, color)` passes `ZoneColorMeta{name, color}`, which the other nine zones
use, and `DeviceZoneScopedNCU(name, color, unit)` passes this one:

```cpp
// tt_metal/hw/inc/hostdev/observability_event_meta.h
enum class TimeUnit : uint8_t { Cycles, Nanoseconds, Microseconds, Milliseconds };

struct ZoneUnitMeta {
    const char* name;
    uint32_t color;  // 0xRRGGBB
    TimeUnit unit;
};
```

Underneath, every macro hands its value to `TT_DEBUG_SITE`, the metadata API in
`tt_metal/hw/inc/hostdev/observability_event_id.h`. That takes any **simple aggregate**, a struct with public data members
only, so a new macro with a new type needs no change below it:

- each field is a string literal (`const char*`), an integer, enum or `bool` of 1, 2, 4 or 8 bytes, or a nested
  struct that follows the same rules;
- 1 to 8 fields per struct;
- the first field is the site's name;
- every value is a compile-time constant.

Breaking a rule (an array, a `float`, a base class, …) fails the build with one readable error. The device never sees
any of this: it only ever has the site's id.

## 2. Preprocess: the site gets a label name

> **None of the numbers below is the event id, and none is part of it.** They only spell the *name* of an
> assembler label, so the next steps can refer to the handle. The id is the label's *address*, which the
> linker (step 4) and the loader (step 9) decide. The name is gone once the file is assembled.

`DeviceZoneScopedNCU(name, color, unit)` expands to `TT_DEBUG_SITE(hash, ZoneUnitMeta{name, color, unit})`,
which spells the label name from three parts:

| part | value | from | why it is in the name |
|---|---|---|---|
| `TT_PROFILER_TU_ID` | `0` | the JIT compile command: `-DTT_PROFILER_TU_ID=0`, the source's index in its link | two sources of one link can be merged into one assembly by LTO; this keeps their names apart |
| `__COUNTER__` | `6` | the seventh site in this source: `0` is `STACK-OVERFLOW` (in `kernel_profiler_streaming.hpp`), `1`–`5` are `T1_Zone0`–`T1_Zone4` | keeps the names of different sites in one source apart |
| hash of the value | `9bb67bec457229db` | an FNV-1a hash of the metadata value: its type and every field, here `"T1_Zone5"`, `0x42D4F4` and `TimeUnit::Microseconds` | a site inside a template can be instantiated with different values (a colour from a template parameter); they share the counter, so the hash keeps them apart |

Label name: **`__tt_observability_event_0_6_9bb67bec457229db`**. Two different sites, or one site with two different values, must
not share a name; nothing else depends on it.

> **Naming the label is an optimization, not a requirement.** A site could give its handle an anonymous local label,
> and every copy would still get a correct, unique id with its own record. What a fixed name buys is deduplication:
> when the compiler expands one site many times (an event in an `inline` function called from ten places), every copy
> carries the same label, and the `.ifndef` from step 3 emits the handle and the record only for the first. The site
> gets one id and one record however many copies of its code exist, which saves id space and keeps the host's table to
> one entry per site. Sharing a name is also what makes the value hash necessary: copies may share one record only if
> they carry the same value.
>
> The deduplication works within one source file. An event in a header that two sources include still gets two ids,
> one per source: `TT_PROFILER_TU_ID` is part of the name precisely to keep those two apart.

**Why the TU id and the counter are not enough.** They give a number per *place in the source*, and one place can
carry several values:

```cpp
template <uint32_t Color>
void colored_zone() {
    // __COUNTER__ is expanded here, once
    DeviceZoneScopedNCU("Zone", Color, kernel_profiler::TimeUnit::Microseconds);
}

void kernel_main() {
    colored_zone<0xFF0000>();   // red
    colored_zone<0x00FF00>();   // green
}
```

The preprocessor runs before templates are instantiated, so `__COUNTER__` becomes one number when the template is
read, and both instantiations emit the same label. Without the hash, the `.ifndef` from step 3 would skip the second
copy's handle and record, and its `lui`/`addi` would load the first copy's id: one id, one record holding red, and
green zones silently reported as red. With the hash, red and green get two labels, so two ids and two records, while
copies with the same value still share one label and are deduplicated:

| label made of | same value | different values |
|---|---|---|
| TU id + counter | one site (right) | one site, wrong values |
| TU id + counter + value hash | one site (right) | one site per value (right) |

**Collision odds.** Only values of the *same* site compete, since the TU id and counter already separate different
sites. For a 64-bit hash, k distinct values of one site collide with probability about k² / 2⁶⁵: for **1,000**
distinct values that is about **3 × 10⁻¹⁴** (real template sites have a handful of values). A collision would
behave like the case without the hash, so it is not checked for.

## 3. Compile: the struct becomes directives, plus two instructions

**How a struct becomes assembler text.** The site is one `asm` statement whose text is not written out by hand but
computed by the compiler:

```cpp
static constexpr auto tt_site_meta = ::tt::debug_event::ZoneUnitMeta{"T1_Zone5", 0x42D4F4, TimeUnit::Microseconds};
asm((::tt::debug_event::detail::emit_site(tt_site_meta, "__tt_observability_event_0_6", __FILE__, __LINE__)) : "=r"(v));
```

`emit_site` is a `constexpr` function, so GCC runs it **during compilation** and uses the text it returns as the asm
template. It walks the struct's fields with a C++17 structured binding, here
`auto& [name, color, unit] = meta;`, finding how many
fields there are by testing how many initializers `T{...}` accepts, and writes one directive per field. A binding
needs a fixed number of names, so `emit_site` checks the field count and uses the binding written for it; those are
written out for 1 to 8 fields, which is where the limit of 8 fields per struct comes from. Nothing of it runs on the
device. The text it returns for this site:

```asm
.ifndef __tt_observability_event_0_6_9bb67bec457229db
.pushsection .tt_observability_event_ids,"",@progbits
__tt_observability_event_0_6_9bb67bec457229db:	.byte 0                        # the handle: exactly one byte
.popsection
.pushsection .tt_observability_event_meta,"",@progbits
.balign 4
.long __tt_observability_event_0_6_9bb67bec457229db                            # header: id slot
.pushsection .tt_observability_event_str,"MS",@progbits,1
8880:	.asciz "tt::debug_event::ZoneUnitMeta:swb"
.popsection
.long 8880b                                                     # header: signature
.pushsection .tt_observability_event_str,"MS",@progbits,1
8880:	.asciz "/localdev/…/kernels/zones_compute.cpp"
.popsection
.long 8880b                                                     # header: file
.long 126                                                       # header: line
.pushsection .tt_observability_event_str,"MS",@progbits,1
8880:	.asciz "T1_Zone5"
.popsection
.long 8880b                                                     # field 1, name: a pointer to the string
.balign 4
.long 4379892                                                   # field 2, color: 0x42D4F4
.balign 1
.byte 2                                                         # field 3, unit: Microseconds
.balign 4                                                       # pad the record to a multiple of 4
.popsection
.endif
	lui %0, %%hi(__tt_observability_event_0_6_9bb67bec457229db)
	addi %0, %0, %%lo(__tt_observability_event_0_6_9bb67bec457229db)
```

The compiler replaces `%0` with a register (`s4` in this kernel).

- `.byte 0` puts one byte in `.tt_observability_event_ids`. Its value is irrelevant; its **address** will be the id.
- **The record** is a 16-byte header (id slot, signature, file, line) followed by the struct's fields in declaration
  order: a string becomes a 4-byte pointer, an integer or enum is stored at its own size, and each field is aligned to its
  size (at most 4). The record is then padded to a multiple of 4. `ZoneUnitMeta` gives
  **16 + 4 + 4 + 1 = 25 bytes, padded to 28**; `ZoneColorMeta` gives 24 and `ZoneMeta` 20. Records differ in size by
  type.
- **The signature** makes every record self-describing: `"<type name>:<field codes>"`, one code per field (`s` string,
  `b`/`h`/`w`/`q` for 1/2/4/8 bytes, braces around a nested struct's codes). `tt::debug_event::ZoneUnitMeta:swb` is
  the name (a string), the colour (a 4-byte integer) and the unit (a 1-byte enum); the header's own four words
  are not in it. The host sizes and reads any record from its signature, without knowing the type; the type name lets a
  reader ask for the struct back by name.
- `126` is `__LINE__` from step 1.

**A richer struct works the same way.** A new macro can give its sites any number of attributes, grouped however
suits them, by passing its own type to the metadata API. A type with the style in a nested struct and a further
field:

```cpp
struct Style { uint32_t color; uint8_t width; };
struct MyZoneMeta { const char* name; Style style; uint16_t flags; };

TT_DEBUG_SITE(site, MyZoneMeta{"T1_Zone5", {0x42D4F4, 2}, 0x0001});
```

`emit_site` flattens the nested struct in place and marks its bounds with braces in the signature. The record body
it writes after the same 16-byte header (the text `emit_site` returns for this value):

| bytes | directive | field |
|---|---|---|
| 16–19 | `.long 8880b` → `"T1_Zone5"` | `name` |
| 20–23 | `.long 4379892` (`0x42D4F4`) | `style.color` |
| 24 | `.byte 2` | `style.width` |
| 26–27 | `.short 1` (after 1 byte of padding, aligned to 2) | `flags` |

Signature `"MyZoneMeta:s{wb}h"`, 28 bytes in all. The host reads it with no code of its own for this type, and
`site.meta->as<MyZoneMeta>()` gives the struct back.

Nothing is a number yet: the id is "wherever `__tt_observability_event_0_6_9bb67bec457229db` ends up".

## 4. Link: the handle is placed

The linker gathers every `.tt_observability_event_ids` byte of this image into one section and puts it at `TT_OBSERVABILITY_EVENT_IDS_VMA`,
**`0x06800000`**. The bytes are laid end to end, one per site, in link order (each label also ends in its value's
hash, shortened to `_…` here):

```
offset  address   label             site
  0     06800000  __tt_observability_event_0_1_…   T1_Zone0
  1     06800001  __tt_observability_event_0_2_…   T1_Zone1
  2     06800002  __tt_observability_event_0_3_…   T1_Zone2
  3     06800003  __tt_observability_event_0_4_…   T1_Zone3
  4     06800004  __tt_observability_event_0_5_…   T1_Zone4
  5     06800005  __tt_observability_event_0_6_…   T1_Zone5     ← this event
  6     06800006  __tt_observability_event_0_7_…   T1_Zone6
   …
  9     06800009  __tt_observability_event_0_10_…  T1_Zone9
 10     0680000a  __tt_observability_event_0_11_…  TRISC-KERNEL (the whole-kernel event opened by the wrapper `trisck.cc`)
 11     0680000b  __tt_observability_event_0_0_…   STACK-OVERFLOW
```

`T1_Zone5` is the byte at **offset 5**, so its address, and its link-time id, is
`0x06800000 + 5 =` **`0x06800005`**. The section is 12 bytes long because the image has 12 sites (one byte each).

## 5. Link: the instructions are filled in

A 32-bit value cannot fit in one RISC-V instruction, so the id is split across two:

| instruction | carries | for `0x06800005` |
|---|---|---|
| `lui` | the upper 20 bits | `0x06800` |
| `addi` | the lower 12 bits | `0x005` = 5 |

`lui` puts its 20 bits at the top of the register and `addi` adds the low 12, giving back `0x06800005`.

The linker writes them into the kernel's code:

```
address  word      instruction
7cd0:    06800a37  lui  s4, 0x6800
7cd4:    005a0a13  addi s4, s4, 5
```

- **`7cd0` / `7cd4`**: where the linker placed these two instructions. A kernel's code is linked right after its
  RISC's firmware code (`.text __fw_export_text_end` in `main.ld`), which for TRISC1 ends at `0x76f0`; this site
  falls `0x5e0` bytes into the kernel.
- **Reading the words**: in a `lui` word the immediate is the first 5 hex digits (`06800|a37`); in an `addi`
  word it is the first 3 (`005|a0a13`). The rest encodes the register (`s4`) and the opcode, and never changes.

## 6. Link: the record is filled in

`.tt_observability_event_meta` sits at `TT_OBSERVABILITY_EVENT_META_VMA`, **`0x06700000`**. Records are in the same order as the handles, and the
five before ours are all 24-byte `ZoneColorMeta` records, so `T1_Zone5`'s record starts at
`0x06700000 + 5 × 24 = 0x06700000 + 0x78 =` **`0x06700078`**. Its 28 bytes, as `readelf -x` prints them:

```
0x06700078:  05008006  cb006006  22006006  7e000000  ed006006  f4d44200  02000000
```

The seven groups are the record's seven 4-byte words from step 3, in file byte order. Reversing each group gives the
number:

| word | bytes in file | as a number | meaning |
|---|---|---|---|
| id slot | `05 00 80 06` | `0x06800005` | the handle's address from step 4; it becomes the id at load (step 11) |
| signature | `cb 00 60 06` | `0x066000cb` | where `"tt::debug_event::ZoneUnitMeta:swb"` starts in `.tt_observability_event_str` |
| file | `22 00 60 06` | `0x06600022` | where the file path starts in `.tt_observability_event_str` |
| line | `7e 00 00 00` | `0x7e` = 126 | the line from step 1 |
| field 1, `name` | `ed 00 60 06` | `0x066000ed` | where `"T1_Zone5"` starts in `.tt_observability_event_str` |
| field 2, `color` | `f4 d4 42 00` | `0x0042d4f4` | the colour, stored as the value itself |
| field 3, `unit` | `02` `00 00 00` | `2` | `TimeUnit::Microseconds`, one byte, then 3 bytes of padding |

**Where `0x066000ed` comes from.** `.tt_observability_event_str` sits at `TT_OBSERVABILITY_EVENT_STR_VMA`, `0x06600000`, and is just the
strings of all sites packed back to back, each ending in a `\0` byte, in the order the sites were emitted:

| starts at | offset | string | its size (with `\0`) |
|---|---|---|---|
| `0x06600000` | 0 | `tt::debug_event::ZoneColorMeta:sw` (the other zones' signature) | 34 |
| `0x06600022` | 34 | `/localdev/…/zones_compute.cpp` | 124 |
| `0x0660009e` | 158 | `T1_Zone0` | 9 |
| `0x066000a7` | 167 | `T1_Zone1` | 9 |
| … | | | |
| `0x066000cb` | 203 | `tt::debug_event::ZoneUnitMeta:swb` (this site's signature) | 34 |
| **`0x066000ed`** | **237** | **`T1_Zone5`** | 9 |

`237` is how many bytes of strings came **before** it: `34 + 124 + 5 × 9 + 34 = 237`. Strings that several sites
share appear only once: the section is marked mergeable (`"MS"` in step 3), so the linker keeps one copy of the file
path and of the `ZoneColorMeta:sw` signature, and every `T1_ZoneN` record points at it.

The host turns a pointer back into a string by subtracting the section's address: `0x066000ed − 0x06600000 = 237`,
then reads from byte 237 up to the next `\0`.

## 7. Link: the relocations are kept

A relocation is the linker's note to itself: "the word at address X holds symbol S, encoded as type T". The link
command has `-Wl,--emit-relocs`, which tells the linker to leave these notes in the ELF instead of discarding them.
The three that mention this handle:

```
at address  type            symbol
00007cd0    R_RISCV_HI20    __tt_observability_event_0_6_9bb67bec457229db   ← the lui from step 5   (upper 20 bits)
00007cd4    R_RISCV_LO12_I  __tt_observability_event_0_6_9bb67bec457229db   ← the addi from step 5  (lower 12 bits)
06700078    R_RISCV_32      __tt_observability_event_0_6_9bb67bec457229db   ← the record's id slot from step 6 (all 32 bits)
```

These three lines are exactly the list of places the host will change in steps 10 and 11. They sit among 20,923
relocations in this ELF (1,072 of them for `.text`).

## 8. Cache: nothing of it is loadable

The ELF goes into the JIT cache as `kernels/zones_compute/12217311898039523400/trisc1/trisc1.elf`. Only two of its
segments are copied to the device:

```
01  .text
02  .data .bss
```

No `.tt_observability_event_*` section is in either, so the device never receives a byte of them. The file on disk is never
modified again; every process rebases its own in-memory copy.

## 9. Load: the image gets a block

When the program is launched, the host reads each kernel ELF (`llrt::get_risc_binary`) and gives it the next free block of
the process's id space, starting from 0.

**How many ids an image needs.** Every site emitted exactly one byte into `.tt_observability_event_ids` (step 3), so the
section's size in bytes is the number of sites. The loader reads that size from the section header. This image's
section is 12 bytes, so it needs 12 ids.

The images this run loaded, in load order. The streaming profiler's own kernels load first: its clock-sync kernel
(`tile_sync`) has 2 sites. Next come the two data-movement kernels, with
15 sites each because `--markers 1` compiles three point markers into them (`_Event`, `_Data`, `_Iter`). The compute
kernels have 12:

| load order | image | `.tt_observability_event_ids` size = sites | block |
|---|---|---|---|
| 1st–2nd | the profiler's kernels: `tile_sync` on BRISC and on a DRAM core | 2, 2 | `[0, 4)` |
| 3rd | `zones_dm/…/ncrisc.elf` | 15 | `[4, 19)` |
| 4th | `zones_dm/…/brisc.elf` | 15 | `[19, 34)` |
| 5th | `zones_compute/…/trisc0.elf` | 12 | `[34, 46)` |
| **6th** | **`zones_compute/…/trisc1.elf`** | **12** | **`[46, 58)`** |
| 7th | `zones_compute/…/trisc2.elf` | 12 | `[58, 70)` |

`base = 4 + 15 + 15 + 12 = 46`. Our event was at offset 5 in step 4, so its id becomes **`46 + 5 = 51`**. In all,
70 of the 65,535 ids are assigned. Which profiler kernels load, and so where the demo's blocks start, depends on the
card; on a p300a, the Ethernet clock and relay kernels load too, and the profiler's kernels take 13 ids.

## 10. Load: the instructions are rewritten

The id is now 51 (step 9). The loader takes the two `.text` relocations from step 7 and writes 51 into them,
split the same way the linker split `0x06800005` in step 5:

| instruction | carries | for `51` (`0x00000033`) |
|---|---|---|
| `lui` | upper 20 bits, rounded: `(51 + 0x800) >> 12` | `0` |
| `addi` | lower 12 bits: `51 & 0xfff` | `0x033` = 51 |

(The `+ 0x800` is there because `addi` treats its 12 bits as signed; it only matters for values whose bit 11 is
set, and 51 is not one of them.)

```
         on disk                           in memory
7cd0:  06800a37  lui  s4, 0x6800     →   00000a37  lui  s4, 0x0
7cd4:  005a0a13  addi s4, s4, 5      →   033a0a13  addi s4, s4, 51
```

Only the immediate digits change (`06800 → 00000`, `005 → 033`). The `lui` now loads 0, because 51 fits in the
`addi` alone; it stays anyway, since the loader rewrites numbers in place and never removes an instruction.

**Why the linker script uses a high address.** At link time the `lui` is only kept if it has something to load.
Had the section been linked at a small address (below 4096), the `lui` would have loaded 0 and the linker would
have deleted it, leaving just the `addi`, which can hold at most 2047. The loader could then never give that event
an id above 2047. Linking at the high address `0x06800000` guarantees every event keeps its `lui`, so the loader
can write any id up to `0xFFFE`.

## 11. Load: the record is rewritten

The third relocation from step 7 points at `0x06700078`, the id slot of `T1_Zone5`'s record in `.tt_observability_event_meta`. That
address is *where* the value sits and does not change; the loader replaces the value inside it with the same 51, as
the whole 32-bit word (`R_RISCV_32`, stored as the bytes `33 00 00 00`). The rest of the record is untouched:

```
0x06700078:  05008006 cb006006 …   →   33000000 cb006006 …
```

**Getting from `0x06700078` to 51:**

```
id slot at 0x06700078 holds        0x06800005      (step 6)
minus where .tt_observability_event_ids was linked − 0x06800000   = 5, the event's offset
plus this image's block start      + 46            (step 9)
                                   = 51
```

**After steps 10 and 11, the same 51 is in three places:**

| section | what holds 51 | who uses it |
|---|---|---|
| `.tt_observability_event_ids` | the handle's address: the section now starts at 46, and this is byte 5 | the loader (it is where the 51 came from) |
| `.text` | the `lui`/`addi` pair, which together produce 51 | the device, which puts it in the marker (step 13) |
| `.tt_observability_event_meta` | the record's id slot | the host, which maps 51 to the site (step 12) |

The device and the host never consult each other; they agree because both were filled in from the same symbol.

## 12. Load: the site and its metadata are registered

The registry walks `.tt_observability_event_meta` from the start, one record at a time. For each record it reads the 16-byte
header, follows the signature pointer, and lets the field codes after the `:` say how many bytes follow and what
they are. For `T1_Zone5`, `swb` means a 4-byte string pointer, a 4-byte integer, then a 1-byte enum; padded to a
multiple of 4, the record is 28 bytes, and the next one starts right after it. It turns every string pointer back
into a string (subtract `0x06600000`, read up to `\0`, as in step 6) and publishes:

```
sites[51] = { name "T1_Zone5", file ".../kernels/zones_compute.cpp", line 126,
              meta = "tt::debug_event::ZoneUnitMeta:swb" { "T1_Zone5", 0x42d4f4, Microseconds } }
```

A reader gets the struct back by asking for its type. `as<T>()` checks the signature against `T`'s and walks `T`'s
fields with the same structured binding the compiler used in step 3, filling each one in. The host and the device
include the same `hostdev/observability_event_meta.h`, so both name the type alike:

```cpp
if (auto v = site.meta->as<tt::debug_event::ZoneUnitMeta>()) {
    // v->name == "T1_Zone5", v->color == 0x42D4F4, v->unit == TimeUnit::Microseconds
}
```

This happens before the image's bytes are copied to the device, so the site is known before the event can fire.

## 13. Run: the device packs the marker

**The id is already decided.** It was fixed at load time (step 10): 51 is written into the bytes of the kernel's
code before the kernel is copied to the device. Nothing the device does at run time chooses, computes or looks up
the id.

What the device does need is the value *in a register*, to store it into the marker. So when the event closes, the
two instructions from step 10 copy that constant into register `s4`, the same way a kernel loads any literal
number such as `x = 51`:

```
7cd0:  lui  s4, 0x0         s4 = 0
7cd4:  addi s4, s4, 51      s4 = 0 + 51 = 51
```

They read no memory and depend on nothing at run time; they would produce 51 on any core, every time.

**Building the marker word.** The first word of an event packet holds two things side by side: a packet type in the
top 5 bits and the event id in the bottom 27 bits. The source line that builds it is `ppfmt::w0` in
`kernel_profiler_streaming.hpp`:

```cpp
word0 = (type << 27) | (id & 0x7FFFFFF);      // 0x7FFFFFF = the low 27 bits set
```

For this event, `type` is 3 (`ZONE_S`, the short 2-word packet for a scoped event) and `id` is 51:

```
type << 27        = 3 << 27   = 0x18000000
id & 0x7FFFFFF    = 51        = 0x00000033     (51 is far below 2^27, so the mask changes nothing)
word0 = OR of the two         = 0x18000033
```

The compiler turns that line into these instructions:

```
7d20:  slli s4, s4, 5          ┐ id & 0x7FFFFFF: shifting left by 5 and back right by 5 pushes the top 5 bits
7d28:  srli s4, s4, 5          ┘ out and brings zeros in, leaving the low 27 bits. s4 = 51.
7d24:  lui  a4, 0x18000        a4 = 0x18000 << 12 = 0x18000000 = 3 << 27   (the type, already in position)
7d2c:  or   s4, s4, a4         s4 = 0x18000000 | 51 = 0x18000033
7d40:  sw   s4, -2048(a1)      store word0 into this RISC's ring buffer
```

> The `5` in `slli`/`srli` is the **width of the type field** (32 − 27 = 5 bits). It has nothing to do with our
> event being at offset 5; that is a coincidence of this example.

word0 = **`0x18000033`**: type 3 in the top 5 bits, id 51 (`0x33`) in the bottom 27. An event too long for the
2-word packet ships type 2 (`ZONE_ATOMIC`) instead, giving `0x10000033`; the id part is the same.

The metadata adds nothing here: the packet is the same two words whatever struct the site carries.

## 14. Decode: the host names it

The relay carries the ring buffer to the host. The decoder masks off the type to recover the id,
`0x18000033 & 0x7FFFFFF = 51` (`0x7FFFFFF` is 27 one-bits), and calls `observability_event_of(51)`, which is one array load:

```
sites[51]  →  "T1_Zone5"  zones_compute.cpp:126  ZoneUnitMeta{ color 0x42d4f4, unit Microseconds }
```

A consumer reads the metadata from there with `as<ZoneUnitMeta>()`: it can draw `T1_Zone5` in its own colour and
show its duration in microseconds. A run of this program that completes its capture receives every event: 105 scoped events
and 12 point events, with 0 stalls.

---

**Next run.** Another process may load the images in a different order and give `trisc1.elf` a different block.
The ELF on disk still says `0x06800005`; only the in-memory copy, and so the id on the wire, changes. Names and
metadata never change.

## Reproduce

```sh
R=runtime/sfpi/compiler/bin/riscv-tt-elf-readelf
O=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
E=<cache>/kernels/zones_compute/<hash>/trisc1/trisc1.elf   # as linked
X=$E.xip.elf                                               # as loaded, written by the loader

$R -s -W $E | grep __tt_observability_event_                              # step 4: handles and their addresses
$O -d --start-address=0x7cd0 --stop-address=0x7cd8 $E      # step 5
$R -x .tt_observability_event_meta $E                                     # step 6: records (T1_Zone5's at 0x06700078)
$R -p .tt_observability_event_str $E                                      # step 6: strings and their offsets
$R -r -W $E | grep __tt_observability_event_0_6_                          # step 7
$R -l -W $E | grep -A3 "Segment Sections"                  # step 8
$R -S -W $X | grep tt_zone_ids                             # step 9: section address = base, size = sites
$O -d --start-address=0x7cd0 --stop-address=0x7cd8 $X      # step 10
$R -x .tt_observability_event_meta $X                                     # step 11
```

The `.xip.elf` is rewritten on every load, so it shows the block the *last* process gave the image.
