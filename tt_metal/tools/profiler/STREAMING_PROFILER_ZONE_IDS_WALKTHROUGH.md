# Zone ids, one zone end to end

This follows a single zone, `T1_Zone5` in the streaming-profiler demo's compute kernel, from its source line to the
name the host prints. Every value below is real: it was read off the JIT-built ELF and the loader's output on a
Blackhole p100a (bh-17), running `test_streaming_profiler_zones --gx 1 --gy 1 --iters 2 --markers 1`. The mechanism
is described in [`STREAMING_PROFILER_ZONE_IDS.md`](STREAMING_PROFILER_ZONE_IDS.md); this page only shows it
happening, and says where every number comes from.

| step | stage | action | the id is |
|---|---|---|---|
| 1 | source | the zone site is written | — |
| 2 | preprocess | the macro names the site's handle `__tt_zone_0_6` (a name, not the id) | not yet a number |
| 3 | compile | the site becomes assembler directives + `lui`/`addi` | not yet a number |
| 4 | link | the linker places the handle | `0x06800005` |
| 5 | link | the linker fills in the instructions | `0x06800005` |
| 6 | link | the linker fills in the record | `0x06800005` |
| 7 | link | the relocations are kept | `0x06800005` |
| 8 | cache | the ELF is stored; nothing of this is loadable | `0x06800005` |
| 9 | load | the host reserves a block for the image | `47` |
| 10 | load | the host rewrites the instructions | `47` |
| 11 | load | the host rewrites the record | `47` |
| 12 | load | the host registers the name | `47` |
| 13 | run | the device packs the id into a marker | `47` |
| 14 | decode | the host names the marker | `47` → `T1_Zone5` |

## Before you start: the fixed numbers

These are constants, chosen once in the source. Every other number on this page is derived from them or from
the kernel.

| value | name | defined in | what it is |
|---|---|---|---|
| `0x06600000` | `TT_ZONE_STR_VMA` | `tt_metal/hw/toolchain/main.ld` | address the linker gives `.tt_zone_str` (the strings) |
| `0x06700000` | `TT_ZONE_META_VMA` | `tt_metal/hw/toolchain/main.ld` | address the linker gives `.tt_zone_meta` (the records) |
| `0x06800000` | `TT_ZONE_IDS_VMA` | `tt_metal/hw/toolchain/main.ld` | address the linker gives `.tt_zone_ids` (the handles) |
| `16` | `TT_ZONE_META_RECORD_BYTES` | `tt_metal/hw/inc/hostdev/profiler_zone_id.h` | bytes per record: four 4-byte fields |
| `0xFFFF` | `TT_ZONE_STALL_ID` | `tt_metal/hw/inc/hostdev/profiler_zone_id.h` | reserved id; the space handed out is `0 … 0xFFFE` |

The three addresses are **made up**. None of these sections is ever loaded onto the device, so their addresses
do not refer to real memory; they only have to exist so the linker can compute pointers between the sections.
They continue a pattern already in `main.ld` (DPRINT's string sections sit at `0x06400000` and `0x06500000`): 1 MB
apart, well away from the device's real memory map. `0x06800000` has one extra requirement, explained in step 10.

**Reading hex dumps.** RISC-V stores a 32-bit number lowest byte first ("little-endian"), and `readelf -x` prints
bytes in file order. So the number `0x06800005` is stored as the bytes `05 00 80 06` and shows up in a dump as
`05008006`. To read a dumped word as a number, reverse its byte pairs.

---

## 1. Source: the zone site

The TRISC1 (math) kernel opens a zone:

```cpp
{
    DeviceZoneScopedN("T1_Zone5");
    ...   // ~12 us of work
}
```

It sits at **line 118** of `test_streaming_profiler_zones/kernels/zones_compute.cpp`; the host reports that line at
the end. It is the sixth of the kernel's ten zones, `T1_Zone0` … `T1_Zone9`.

## 2. Preprocess: the site gets a label name

> **Neither number below is the zone id, and neither is part of it.** They only spell the *name* of an
> assembler label, so the next steps can refer to the handle. The id is the label's *address*, which the
> linker (step 4) and the loader (step 9) decide. The name is gone once the file is assembled.

`DeviceZoneScopedN(name)` expands to `TT_ZONE_DEFINE_ID(hash, name)`, which spells the label name from two numbers:

| part | value | from | why it is in the name |
|---|---|---|---|
| `TT_PROFILER_TU_ID` | `0` | the JIT compile command: `-DTT_PROFILER_TU_ID=0`, the source's index in its link | two sources of one link can be merged into one assembly by LTO; this keeps their names apart |
| `__COUNTER__` | `6` | the seventh site in this source: `0` is `STACK-OVERFLOW` (in `kernel_profiler_streaming.hpp`), `1`–`5` are `T1_Zone0`–`T1_Zone4` | keeps the names of different sites in one source apart |

Label name: **`__tt_zone_0_6`**. Two different sites must not share a name; nothing else depends on it.

## 3. Compile: directives plus two instructions

The compiler copies the site's asm into the kernel's assembler text verbatim (shown with `-fno-lto`; under LTO it is
the same text, only the register differs):

```asm
	.ifndef __tt_zone_0_6
.pushsection .tt_zone_ids,"",@progbits
__tt_zone_0_6:	.byte 0                         # the handle: exactly one byte
.popsection
.pushsection .tt_zone_str,"MS",@progbits,1
8880:	.asciz "T1_Zone5"                        # name string
8881:	.asciz ".../kernels/zones_compute.cpp"   # file string
.popsection
.pushsection .tt_zone_meta,"M",@progbits,16      # 16 = bytes per record
.balign 4
.long __tt_zone_0_6                              # record field 1: id   (4 bytes)
.long 8880b                                      # record field 2: name (4 bytes)
.long 8881b                                      # record field 3: file (4 bytes)
.long 118                                        # record field 4: line (4 bytes)
.popsection
.endif
	lui s3, %hi(__tt_zone_0_6)
	addi s3, s3, %lo(__tt_zone_0_6)
```

- `.byte 0` puts one byte in `.tt_zone_ids`. Its value is irrelevant; its **address** will be the id.
- Each `.long` is 4 bytes, so one record is **4 × 4 = 16 bytes** (`TT_ZONE_META_RECORD_BYTES`).
- `118` is `__LINE__` from step 1.

Nothing is a number yet: the id is "wherever `__tt_zone_0_6` ends up".

## 4. Link: the handle is placed

The linker gathers every `.tt_zone_ids` byte of this image into one section and puts it at `TT_ZONE_IDS_VMA`,
**`0x06800000`**. The bytes are laid end to end, one per site, in link order:

```
offset  address   label            site
  0     06800000  __tt_zone_0_1    T1_Zone0
  1     06800001  __tt_zone_0_2    T1_Zone1
  2     06800002  __tt_zone_0_3    T1_Zone2
  3     06800003  __tt_zone_0_4    T1_Zone3
  4     06800004  __tt_zone_0_5    T1_Zone4
  5     06800005  __tt_zone_0_6    T1_Zone5     ← this zone
  6     06800006  __tt_zone_0_7    T1_Zone6
   …
  9     06800009  __tt_zone_0_10   T1_Zone9
 10     0680000a  __tt_zone_0_11   TRISC-KERNEL (the whole-kernel zone opened by the wrapper `trisck.cc`)
 11     0680000b  __tt_zone_0_0    STACK-OVERFLOW
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
7c34:    06800a37  lui  s4, 0x6800
7c38:    005a0a13  addi s4, s4, 5
```

- **`7c34` / `7c38`**: where the linker placed these two instructions. A kernel's code is linked right after its
  RISC's firmware code (`.text __fw_export_text_end` in `main.ld`), which for TRISC1 ends at `0x7660`; this site
  falls `0x5d4` bytes into the kernel.
- **Reading the words**: in a `lui` word the immediate is the first 5 hex digits (`06800|a37`); in an `addi`
  word it is the first 3 (`005|a0a13`). The rest encodes the register (`s4`) and the opcode, and never changes.

## 6. Link: the record is filled in

`.tt_zone_meta` sits at `TT_ZONE_META_VMA`, **`0x06700000`**. Records are 16 bytes each and in the same order as
the handles, so `T1_Zone5`'s record, number 5, starts at `0x06700000 + 5 × 16 = 0x06700000 + 0x50 =`
**`0x06700050`**. `readelf -x` prints it as:

```
0x06700050:  05008006  a9006006  09006006  76000000
```

The address on the left is where the row starts. The four groups are the four 4-byte fields from step 3, in file
byte order. Reversing each group gives the number:

| field | bytes in file | as a number | meaning |
|---|---|---|---|
| id | `05 00 80 06` | `0x06800005` | the handle's address from step 4 |
| name | `a9 00 60 06` | `0x066000a9` | where `"T1_Zone5"` starts in `.tt_zone_str` |
| file | `09 00 60 06` | `0x06600009` | where the file path starts in `.tt_zone_str` |
| line | `76 00 00 00` | `0x76` = 118 | the line from step 1 |

**Where `0x066000a9` comes from.** `.tt_zone_str` sits at `TT_ZONE_STR_VMA`, `0x06600000`, and is just the
strings of all sites packed back to back, each ending in a `\0` byte, in the order the sites were emitted:

| starts at | offset | string | its size (with `\0`) |
|---|---|---|---|
| `0x06600000` | 0 | `T1_Zone0` | 9 |
| `0x06600009` | 9 | `/localdev/…/zones_compute.cpp` | 124 |
| `0x06600085` | 133 | `T1_Zone1` | 9 |
| `0x0660008e` | 142 | `T1_Zone2` | 9 |
| `0x06600097` | 151 | `T1_Zone3` | 9 |
| `0x066000a0` | 160 | `T1_Zone4` | 9 |
| **`0x066000a9`** | **169** | **`T1_Zone5`** | 9 |

`169` is not a size of anything in this zone. It is how many bytes of strings came **before** it:
`9 + 124 + 4 × 9 = 169`. The long file path appears only once: the section is marked mergeable (`"MS"` in step 3),
so the linker keeps one copy and every `T1_ZoneN` record's file field points at it, `0x06600009`.

The host turns a pointer back into a string by subtracting the section's address: `0x066000a9 − 0x06600000 = 169`,
then reads from byte 169 up to the next `\0`.

## 7. Link: the relocations are kept

A relocation is the linker's note to itself: "the word at address X holds symbol S, encoded as type T". The link
command has `-Wl,--emit-relocs`, which tells the linker to leave these notes in the ELF instead of discarding them.
The three that mention this handle:

```
at address  type            symbol
00007c34    R_RISCV_HI20    __tt_zone_0_6      ← the lui from step 5   (upper 20 bits)
00007c38    R_RISCV_LO12_I  __tt_zone_0_6      ← the addi from step 5  (lower 12 bits)
06700050    R_RISCV_32      __tt_zone_0_6      ← the record's id field from step 6 (all 32 bits)
```

These three lines are exactly the list of places the host will change in steps 10 and 11. They sit among 20,751
relocations in this ELF (811 of them for `.text`).

## 8. Cache: nothing of it is loadable

The ELF goes into the JIT cache as `kernels/zones_compute/13881033146769804318/trisc1/trisc1.elf`. Only two of its
segments are copied to the device:

```
01  .text
02  .data .bss
```

No `.tt_zone_*` section is in either, so the device never receives a byte of them. The file on disk is never
modified again; every process rebases its own in-memory copy.

## 9. Load: the image gets a block

When the program is launched, the host reads each kernel ELF (`ll_api::memory`) and gives it the next free block of
the process's id space, starting from 0.

**How many ids an image needs.** Every site emitted exactly one byte into `.tt_zone_ids` (step 3), so the
section's size in bytes is the number of sites. The loader reads that size from the section header. This image's
section is 12 bytes, so it needs 12 ids.

The images this run loaded, in load order. The two data-movement kernels have 15 sites, because `--markers 1`
compiles three point markers into them (`_Event`, `_Data`, `_Iter`); the compute kernels have 12:

| load order | image | `.tt_zone_ids` size = sites | block |
|---|---|---|---|
| 1st | `zones_dm/…/ncrisc.elf` | 15 | `[0, 15)` |
| 2nd | `zones_dm/…/brisc.elf` | 15 | `[15, 30)` |
| 3rd | `zones_compute/…/trisc0.elf` | 12 | `[30, 42)` |
| **4th** | **`zones_compute/…/trisc1.elf`** | **12** | **`[42, 54)`** |
| 5th | `zones_compute/…/trisc2.elf` | 12 | `[54, 66)` |

`base = 15 + 15 + 12 = 42`. Our zone was at offset 5 in step 4, so its id becomes **`42 + 5 = 47`**. The total,
66, is what the receiver reports at teardown: `zone ids: 66 of 65535 assigned`.

## 10. Load: the instructions are rewritten

The id is now 47 (step 9). The loader takes the two `.text` relocations from step 7 and writes 47 into them,
split the same way the linker split `0x06800005` in step 5:

| instruction | carries | for `47` (`0x0000002f`) |
|---|---|---|
| `lui` | upper 20 bits, rounded: `(47 + 0x800) >> 12` | `0` |
| `addi` | lower 12 bits: `47 & 0xfff` | `0x02f` = 47 |

(The `+ 0x800` is there because `addi` treats its 12 bits as signed; it only matters for values whose bit 11 is
set, and 47 is not one of them.)

```
         on disk                           in memory
7c34:  06800a37  lui  s4, 0x6800     →   00000a37  lui  s4, 0x0
7c38:  005a0a13  addi s4, s4, 5      →   02fa0a13  addi s4, s4, 47
```

Only the immediate digits change (`06800 → 00000`, `005 → 02f`). The `lui` now loads 0, because 47 fits in the
`addi` alone; it stays anyway, since the loader rewrites numbers in place and never removes an instruction.

**Why the linker script uses a high address.** At link time the `lui` is only kept if it has something to load.
Had the section been linked at a small address (below 4096), the `lui` would have loaded 0 and the linker would
have deleted it, leaving just the `addi`, which can hold at most 2047. The loader could then never give that zone
an id above 2047. Linking at the high address `0x06800000` guarantees every zone keeps its `lui`, so the loader
can write any id up to `0xFFFE`.

## 11. Load: the record is rewritten

The third relocation from step 7 points at `0x06700050`, the id field of record 5 in `.tt_zone_meta`. That address
is *where* the value sits and does not change; the loader replaces the value inside it with the same 47, as the
whole 32-bit word (`R_RISCV_32`, stored as the bytes `2f 00 00 00`):

```
0x06700050:  05008006 …   →   2f000000 …
```

**Getting from `0x06700050` to 47:**

```
field at 0x06700050 holds          0x06800005      (step 6)
minus where .tt_zone_ids was linked − 0x06800000   = 5, the zone's offset
plus this image's block start      + 42            (step 9)
                                   = 47
```

The record's position gives the same offset: `(0x06700050 − 0x06700000) / 16 = 0x50 / 16 = 5`, record 5. That
match is only because records and handles were emitted in the same order; the loader uses the field's value, not
the position.

Nothing moves by `0x06700000`: the only section that moves is `.tt_zone_ids`. The name and file fields point into
`.tt_zone_str`, which stays where it is, so `0x066000a9` and `0x06600009` are still right; the line is a plain
number. One word of the record's four changes.

**After steps 10 and 11, the same 47 is in three places:**

| section | what holds 47 | who uses it |
|---|---|---|
| `.tt_zone_ids` | the handle's address: the section now starts at 42, and this is byte 5 | the loader (it is where the 47 came from) |
| `.text` | the `lui`/`addi` pair, which together produce 47 | the device, which puts it in the marker (step 13) |
| `.tt_zone_meta` | the record's id field | the host, which maps 47 to `"T1_Zone5"` (step 12) |

The device and the host never consult each other; they agree because both were filled in from the same symbol.

## 12. Load: the name is registered

The registry reads record 5, turns its two pointers back into strings (subtract `0x06600000`, read up to `\0`, as in
step 6), and publishes:

```
sites[47] = { name "T1_Zone5", file ".../kernels/zones_compute.cpp", line 118 }
```

This happens before the image's bytes are copied to the device, so the name exists before the zone can fire.

## 13. Run: the device packs the marker

**The id is already decided.** It was fixed at load time (step 10): 47 is written into the bytes of the kernel's
code before the kernel is copied to the device. Nothing the device does at run time chooses, computes or looks up
the id.

What the device does need is the value *in a register*, to store it into the marker. So when the zone closes, the
two instructions from step 10 copy that constant into register `s4`, the same way a kernel loads any literal
number such as `x = 47`:

```
7c34:  lui  s4, 0x0         s4 = 0
7c38:  addi s4, s4, 47      s4 = 0 + 47 = 47
```

They read no memory and depend on nothing at run time; they would produce 47 on any core, every time.

**Building the marker word.** The first word of a zone packet holds two things side by side: a packet type in the
top 5 bits and the zone id in the bottom 27 bits. The source line that builds it is `ppfmt::w0` in
`kernel_profiler_streaming.hpp`:

```cpp
word0 = (type << 27) | (id & 0x7FFFFFF);      // 0x7FFFFFF = the low 27 bits set
```

For this zone, `type` is 3 (`ZONE_S`, the 2-word zone packet) and `id` is 47:

```
type << 27        = 3 << 27   = 0x18000000
id & 0x7FFFFFF    = 47        = 0x0000002F     (47 is far below 2^27, so the mask changes nothing)
word0 = OR of the two         = 0x1800002F
```

The compiler turns that line into these instructions:

```
7c84:  slli s4, s4, 5          ┐ id & 0x7FFFFFF: shifting left by 5 and back right by 5 pushes the top 5 bits
7c8c:  srli s4, s4, 5          ┘ out and brings zeros in, leaving the low 27 bits. s4 = 47.
7c88:  lui  a4, 0x18000        a4 = 0x18000 << 12 = 0x18000000 = 3 << 27   (the type, already in position)
7c90:  or   s4, s4, a4         s4 = 0x18000000 | 47 = 0x1800002F
7ca4:  sw   s4, -2048(a1)      store word0 into this RISC's ring buffer
```

> The `5` in `slli`/`srli` is the **width of the type field** (32 − 27 = 5 bits). It has nothing to do with our
> zone being at offset 5; that is a coincidence of this example.

word0 = **`0x1800002F`**: type 3 in the top 5 bits, id 47 (`0x2f`) in the bottom 27. A zone too long for the
2-word packet ships type 2 (`ZONE_ATOMIC`) instead, giving `0x1000002F`; the id part is the same.

## 14. Decode: the host names it

The relay carries the ring buffer to the host. The decoder masks off the type to recover the id,
`0x1800002F & 0x7FFFFFF = 47` (`0x7FFFFFF` is 27 one-bits), and calls `site_of(47)`, which is one array load:

```
sites[47]  →  "T1_Zone5"  zones_compute.cpp:118
```

The run's subscriber saw all of them: `subscriber saw 105 zones, 12 points, 0 stalls`.

---

**Next run.** Another process may load the images in a different order and give `trisc1.elf` a different block.
The ELF on disk still says `0x06800005`; only the in-memory copy, and so the id on the wire, changes. Names never
change.

## Reproduce

```sh
R=runtime/sfpi/compiler/bin/riscv-tt-elf-readelf
O=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
E=<cache>/kernels/zones_compute/<hash>/trisc1/trisc1.elf   # as linked
X=$E.xip.elf                                               # as loaded, written by the loader

$R -s -W $E | grep __tt_zone_                              # step 4: handles and their addresses
$O -d --start-address=0x7c34 --stop-address=0x7c3c $E      # step 5
$R -x .tt_zone_meta $E                                     # step 6: records (record 5 at 0x06700050)
$R -p .tt_zone_str $E                                      # step 6: strings and their offsets
$R -r -W $E | grep __tt_zone_0_6                           # step 7
$R -l -W $E | grep -A3 "Segment Sections"                  # step 8
$R -S -W $X | grep tt_zone_ids                             # step 9: section address = base, size = sites
$O -d --start-address=0x7c34 --stop-address=0x7c3c $X      # step 10
$R -x .tt_zone_meta $X                                     # step 11
```

The `.xip.elf` is rewritten on every load, so it shows the block the *last* process gave the image.
