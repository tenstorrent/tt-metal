# Zone ids, one zone end to end

This follows a single zone, `BR_Zone0` in the streaming-profiler demo kernel, from its source line to the name the
host prints. Every value below is real: it was read off the JIT-built ELF and the loader's output on a Blackhole
p100a (bh-17), running `test_streaming_profiler_zones --gx 1 --gy 1 --iters 2`. The mechanism is described in
[`STREAMING_PROFILER_ZONE_IDS.md`](STREAMING_PROFILER_ZONE_IDS.md); this page only shows it happening.

Byte dumps are little-endian: the word `0x06800000` appears in a hex dump as `00008006`.

| step | stage | action | the id is |
|---|---|---|---|
| 1 | source | the zone site is written | — |
| 2 | preprocess | the macro names the site's handle | `__tt_zone_0_1` |
| 3 | compile | the site becomes assembler directives + `lui`/`addi` | a label |
| 4 | link | the linker places the handle | `0x06800000` |
| 5 | link | the linker fills in the instructions | `0x06800000` |
| 6 | link | the linker fills in the record | `0x06800000` |
| 7 | link | the relocations are kept | `0x06800000` |
| 8 | cache | the ELF is stored; nothing of this is loadable | `0x06800000` |
| 9 | load | the host reserves a block for the image | `12` |
| 10 | load | the host rewrites the instructions | `12` |
| 11 | load | the host rewrites the record | `12` |
| 12 | load | the host registers the name | `12` |
| 13 | run | the device packs the id into a marker | `12` |
| 14 | decode | the host names the marker | `12` → `BR_Zone0` |

---

## 1. Source: the zone site

The BRISC kernel opens a zone:

```cpp
{
    DeviceZoneScopedN("BR_Zone0");
    ...   // ~1 us of work
}
```

It sits at line 119 of `test_streaming_profiler_zones/kernels/zones_dm.cpp`; that line number is what the host
reports at the end.

## 2. Preprocess: the site gets a label

`DeviceZoneScopedN(name)` expands to `TT_ZONE_DEFINE_ID(hash, name)`, which builds the label from two numbers:

| part | value | from |
|---|---|---|
| `TT_ZONE_TU_TAG` | `0` | the JIT compile command: `-DTT_ZONE_TU_TAG=0` (this is the only TU in the link) |
| `__COUNTER__` | `1` | the second zone site in the TU. Counter `0` went to `STACK-OVERFLOW`, declared at namespace scope in `kernel_profiler_streaming.hpp` |

Label: **`__tt_zone_0_1`**.

## 3. Compile: directives plus two instructions

The compiler copies the site's asm into the kernel's assembler text verbatim (shown with `-fno-lto`; under LTO it is
the same text, only the register differs):

```asm
	.ifndef __tt_zone_0_1
.pushsection .tt_zone_ids,"",@progbits
__tt_zone_0_1:	.byte 0                         # the handle
.popsection
.pushsection .tt_zone_str,"MS",@progbits,1
8880:	.asciz "BR_Zone0"
8881:	.asciz ".../test_streaming_profiler_zones/kernels/zones_dm.cpp"
.popsection
.pushsection .tt_zone_meta,"M",@progbits,16
.balign 4
.long __tt_zone_0_1                              # record: id
.long 8880b                                      #         name
.long 8881b                                      #         file
.long 119                                        #         line
.popsection
.endif
	lui s2, %hi(__tt_zone_0_1)
	addi s2, s2, %lo(__tt_zone_0_1)
```

Nothing is a number yet: the id is "wherever `__tt_zone_0_1` ends up".

## 4. Link: the handle is placed

`main.ld` puts `.tt_zone_ids` at `0x6800000` and lays the image's 12 handles out back to back:

```
[ 6] .tt_zone_ids  PROGBITS  06800000  size 00000c   (no A flag)

06800000  __tt_zone_0_1     ← BR_Zone0
06800001  __tt_zone_0_2     ← BR_Zone1
   …
0680000a  __tt_zone_0_11    ← BRISC-KERNEL
0680000b  __tt_zone_0_0     ← STACK-OVERFLOW
```

`BR_Zone0` is at offset **k = 0**, so its link-time id is **`0x06800000`**.

## 5. Link: the instructions are filled in

The two relocations from step 3 are resolved to `0x06800000` (`hi20 = 0x06800`, `lo12 = 0`):

```
5370:  06800c37   lui  s8, 0x6800
5374:  000c0c13   addi s8, s8, 0        (objdump prints it as "mv s8,s8")
```

## 6. Link: the record is filled in

Record 0 of `.tt_zone_meta` (16 bytes):

```
0x06700000:  00008006  00006006  09006006  77000000
               id         name       file      line
             06800000   06600000   06600009   119
```

and the two string pointers land in `.tt_zone_str`:

```
06600000  "BR_Zone0"
06600009  ".../kernels/zones_dm.cpp"
```

## 7. Link: the relocations are kept

`-Wl,--emit-relocs` (on the link command) leaves the linker's work list in the ELF. The three entries that mention
this handle:

```
.rela.text
  00005370  R_RISCV_HI20    __tt_zone_0_1 + 0      ← the lui
  00005374  R_RISCV_LO12_I  __tt_zone_0_1 + 0      ← the addi
.rela.tt_zone_meta
  06700000  R_RISCV_32      __tt_zone_0_1 + 0      ← the record's id word
```

They sit among 21,274 relocations in this ELF (998 in `.rela.text`).

## 8. Cache: nothing of it is loadable

The ELF goes into the JIT cache as `kernels/zones_dm/11584978620218114031/brisc/brisc.elf`. Its loadable segments:

```
01  .text
02  .data .bss
```

No `.tt_zone_*` section is in a segment, so the device never receives a byte of them. The file on disk is never
modified again; every process rebases its own in-memory copy.

## 9. Load: the image gets a block

When the program is launched, the host reads the ELF (`ll_api::memory`). The registry measures `.tt_zone_ids`
(12 bytes = 12 sites) and hands out the next free block of the process's id space. The image loaded before it was
the NCRISC kernel, so:

| image | sites | block |
|---|---|---|
| `zones_dm/…/ncrisc.elf` | 12 | `[0, 12)` |
| **`zones_dm/…/brisc.elf`** | **12** | **`[12, 24)`** |
| `zones_compute/…/trisc0.elf` | 12 | `[24, 36)` |
| `zones_compute/…/trisc1.elf` | 12 | `[36, 48)` |
| `zones_compute/…/trisc2.elf` | 12 | `[48, 60)` |

`base = 12`. The id of `BR_Zone0` becomes `base + k = 12`. (Total: 60, the number the receiver reports at teardown:
`zone ids: 60 of 65535 assigned`.)

## 10. Load: the instructions are rewritten

`RebaseZoneIds(12)` takes the two `.rela.text` entries from step 7 and re-encodes `v = 12`:
`hi20 = (12 + 0x800) >> 12 = 0`, `lo12 = 12`.

```
            on disk                          in memory
5370:  06800c37  lui  s8, 0x6800     →   00000c37  lui  s8, 0x0
5374:  000c0c13  addi s8, s8, 0      →   00cc0c13  addi s8, s8, 12
```

Only these two words change. The `lui` stays: the loader rewrites immediates in place and never removes an
instruction.

## 11. Load: the record is rewritten

The `R_RISCV_32` entry from step 7 gets the same value:

```
0x06700000:  00008006 …   →   0c000000 …        (word0: 0x06800000 → 12)
```

The name, file and line words are untouched.

## 12. Load: the name is registered

The registry reads record 0, follows its two pointers into `.tt_zone_str`, and publishes:

```
sites[12] = { name "BR_Zone0", file ".../kernels/zones_dm.cpp", line 119 }
```

This happens before the image's bytes are packed for the device, so the name exists before the zone can fire.

## 13. Run: the device packs the marker

When the zone closes, the kernel builds word0 of the packet from the id in `s8` (BRISC's text, unchanged since
step 10):

```
53cc:  lui  s5, 0x18000        s5 = 3 << 27     packet type ZONE_S
53d4:  and  s8, s8, t2         s8 = 12 & 0x7FFFFFF
53dc:  or   s8, s8, s5         s8 = 0x1800000C
53e4:  sw   s8, -800(a2)       word0 into the RISC's ring
```

word0 = **`0x1800000C`**: type 3 in bits 31:27, id 12 in bits 26:0. (A zone that needs the 3-word form ships type 2,
`0x1000000C`; the id field is the same.)

## 14. Decode: the host names it

The relay carries the ring to the host. The decoder takes `word0 & 0x7FFFFFF = 12` and calls `site_of(12)`, one
array load:

```
sites[12]  →  "BR_Zone0"  zones_dm.cpp:119
```

The run's subscriber saw all of them: `subscriber saw 105 zones, 0 points, 0 stalls`.

---

**Next run.** Another process may load the images in a different order and give `brisc.elf` a different block. The
ELF on disk still says `0x06800000`; only the in-memory copy, and so the id on the wire, changes. Names never change.

## Reproduce

```sh
R=runtime/sfpi/compiler/bin/riscv-tt-elf-readelf
O=runtime/sfpi/compiler/bin/riscv-tt-elf-objdump
E=<cache>/kernels/zones_dm/<hash>/brisc/brisc.elf        # as linked
X=$E.xip.elf                                             # as loaded, written by the loader

$R -s -W $E | grep __tt_zone_                            # step 4: handles
$O -d --start-address=0x5370 --stop-address=0x5378 $E    # step 5
$R -x .tt_zone_meta $E | head -4                         # step 6
$R -r -W $E | grep __tt_zone_0_1                         # step 7
$R -l -W $E | grep -A3 "Segment Sections"                # step 8
$R -S -W $X | grep tt_zone_ids                           # step 9: section address = base
$O -d --start-address=0x5370 --stop-address=0x5378 $X    # step 10
$R -x .tt_zone_meta $X | head -4                         # step 11
```

The `.xip.elf` is rewritten on every load, so it shows the block the *last* process gave the image.
