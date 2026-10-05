# Zone ids, one zone end to end

This follows a single zone, `T1_Zone5` in the streaming-profiler demo's compute kernel, from its source line to the
name the host prints. Every value below is real: it was read off the JIT-built ELF and the loader's output on a
Blackhole p100a (bh-17), running `test_streaming_profiler_zones --gx 1 --gy 1 --iters 2 --markers 1`. The mechanism
is described in [`STREAMING_PROFILER_ZONE_IDS.md`](STREAMING_PROFILER_ZONE_IDS.md); this page only shows it
happening.

Byte dumps are little-endian: the word `0x06800005` appears in a hex dump as `05008006`.

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

---

## 1. Source: the zone site

The TRISC1 (math) kernel opens a zone:

```cpp
{
    DeviceZoneScopedN("T1_Zone5");
    ...   // ~12 us of work
}
```

It sits at line 118 of `test_streaming_profiler_zones/kernels/zones_compute.cpp`; that line number is what the host
reports at the end. It is the sixth of the kernel's ten zones, `T1_Zone0` … `T1_Zone9`.

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
8880:	.asciz "T1_Zone5"
8881:	.asciz ".../test_streaming_profiler_zones/kernels/zones_compute.cpp"
.popsection
.pushsection .tt_zone_meta,"M",@progbits,16
.balign 4
.long __tt_zone_0_6                              # record: id
.long 8880b                                      #         name
.long 8881b                                      #         file
.long 118                                        #         line
.popsection
.endif
	lui s3, %hi(__tt_zone_0_6)
	addi s3, s3, %lo(__tt_zone_0_6)
```

Nothing is a number yet: the id is "wherever `__tt_zone_0_6` ends up".

## 4. Link: the handle is placed

`main.ld` puts `.tt_zone_ids` at `0x6800000` and lays the image's handles out back to back, one byte each, in link
order:

```
[ 6] .tt_zone_ids  PROGBITS  06800000  size 00000c   (no A flag)

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
 10     0680000a  __tt_zone_0_11   TRISC-KERNEL
 11     0680000b  __tt_zone_0_0    STACK-OVERFLOW
```

`T1_Zone5` is at offset **k = 5**, so its link-time id is **`0x06800005`**.

## 5. Link: the instructions are filled in

The two relocations from step 3 are resolved to `0x06800005`: `hi20 = 0x06800`, `lo12 = 5`.

```
7c34:  06800a37   lui  s4, 0x6800
7c38:  005a0a13   addi s4, s4, 5
```

## 6. Link: the record is filled in

Record 5 of `.tt_zone_meta` (16 bytes, at `0x06700000 + 5 × 16`):

```
0x06700050:  05008006  a9006006  09006006  76000000
               id         name       file      line
             06800005   066000a9   06600009   118
```

and the two string pointers land in `.tt_zone_str`:

```
066000a9  "T1_Zone5"
06600009  ".../kernels/zones_compute.cpp"
```

## 7. Link: the relocations are kept

`-Wl,--emit-relocs` (on the link command) leaves the linker's work list in the ELF. The three entries that mention
this handle:

```
.rela.text
  00007c34  R_RISCV_HI20    __tt_zone_0_6 + 0      ← the lui
  00007c38  R_RISCV_LO12_I  __tt_zone_0_6 + 0      ← the addi
.rela.tt_zone_meta
  06700050  R_RISCV_32      __tt_zone_0_6 + 0      ← the record's id word
```

They sit among 20,751 relocations in this ELF (811 in `.rela.text`).

## 8. Cache: nothing of it is loadable

The ELF goes into the JIT cache as `kernels/zones_compute/13881033146769804318/trisc1/trisc1.elf`. Its loadable
segments:

```
01  .text
02  .data .bss
```

No `.tt_zone_*` section is in a segment, so the device never receives a byte of them. The file on disk is never
modified again; every process rebases its own in-memory copy.

## 9. Load: the image gets a block

When the program is launched, the host reads each kernel ELF (`ll_api::memory`) and gives it the next free block of
the process's id space.

**How many ids an image needs.** Every site emitted exactly one byte into `.tt_zone_ids` (step 3,
`.byte 0`), so the section's size in bytes *is* the number of sites. The loader reads that size from the section
header and needs nothing else. This image's section is 12 bytes, so it needs 12 ids.

The images this run loaded, in load order. The two data-movement kernels have 15 sites, because `--markers 1`
compiles three point markers into them (`_Event`, `_Data`, `_Iter`); the compute kernels have 12:

| load order | image | `.tt_zone_ids` size = sites | block |
|---|---|---|---|
| 1st | `zones_dm/…/ncrisc.elf` | 15 | `[0, 15)` |
| 2nd | `zones_dm/…/brisc.elf` | 15 | `[15, 30)` |
| 3rd | `zones_compute/…/trisc0.elf` | 12 | `[30, 42)` |
| **4th** | **`zones_compute/…/trisc1.elf`** | **12** | **`[42, 54)`** |
| 5th | `zones_compute/…/trisc2.elf` | 12 | `[54, 66)` |

`base = 15 + 15 + 12 = 42`. The id of `T1_Zone5` becomes **`base + k = 42 + 5 = 47`**. (Total: 66, the number the
receiver reports at teardown: `zone ids: 66 of 65535 assigned`.)

## 10. Load: the instructions are rewritten

`RebaseZoneIds(42)` takes the two `.rela.text` entries from step 7 and re-encodes `v = 47`:
`hi20 = (47 + 0x800) >> 12 = 0`, `lo12 = 47`.

```
            on disk                          in memory
7c34:  06800a37  lui  s4, 0x6800     →   00000a37  lui  s4, 0x0
7c38:  005a0a13  addi s4, s4, 5      →   02fa0a13  addi s4, s4, 47
```

Only these two words change. The `lui` now loads 0, because every id below 2048 fits in the `addi`'s 12 bits; it
stays anyway, since the loader rewrites immediates in place and never removes an instruction.

## 11. Load: the record is rewritten

The `R_RISCV_32` entry from step 7 gets the same value:

```
0x06700050:  05008006 …   →   2f000000 …        (word0: 0x06800005 → 47)
```

The name, file and line words are untouched.

## 12. Load: the name is registered

The registry reads record 5, follows its two pointers into `.tt_zone_str`, and publishes:

```
sites[47] = { name "T1_Zone5", file ".../kernels/zones_compute.cpp", line 118 }
```

This happens before the image's bytes are packed for the device, so the name exists before the zone can fire.

## 13. Run: the device packs the marker

When the zone closes, the kernel builds word0 of the packet from the id in `s4` (TRISC1's text, unchanged since
step 10):

```
7c84:  slli s4, s4, 5          s4 = 47 << 5      ┐ together: keep the low 27 bits,
7c88:  lui  a4, 0x18000        a4 = 3 << 27      │ packet type ZONE_S
7c8c:  srli s4, s4, 5          s4 = 47           ┘
7c90:  or   s4, s4, a4         s4 = 0x1800002F
7ca4:  sw   s4, -2048(a1)      word0 into the RISC's ring
```

word0 = **`0x1800002F`**: type 3 in bits 31:27, id 47 in bits 26:0. (A zone that needs the 3-word form ships type 2,
`0x1000002F`; the id field is the same.)

## 14. Decode: the host names it

The relay carries the ring to the host. The decoder takes `word0 & 0x7FFFFFF = 47` and calls `site_of(47)`, one
array load:

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

$R -s -W $E | grep __tt_zone_                              # step 4: handles
$O -d --start-address=0x7c34 --stop-address=0x7c3c $E      # step 5
$R -x .tt_zone_meta $E                                     # step 6 (record 5 at 0x06700050)
$R -r -W $E | grep __tt_zone_0_6                           # step 7
$R -l -W $E | grep -A3 "Segment Sections"                  # step 8
$R -S -W $X | grep tt_zone_ids                             # step 9: section address = base, size = sites
$O -d --start-address=0x7c34 --stop-address=0x7c3c $X      # step 10
$R -x .tt_zone_meta $X                                     # step 11
```

The `.xip.elf` is rewritten on every load, so it shows the block the *last* process gave the image.
