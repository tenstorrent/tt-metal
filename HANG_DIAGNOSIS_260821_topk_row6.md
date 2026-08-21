# ROOT CAUSE FOUND — device 21 is running a kernel with one flipped bit; `TopkLargeIndices` (op 53111) hangs the whole 8×4 mesh

**Verdict**: **Device 21's copy of the `compute/trisc1` kernel for `TopkLargeIndicesDeviceOperation` has a single flipped bit.** It turns
`lui a5,0x28000` into `0x280007b3`, an encoding this target cannot decode. trisc1 dies ~0x348 bytes into the kernel, inside the top-k math
template setup, on all 120 cores. The op never retires, and the mesh piles up behind it.

The bit did not flip in DRAM and it did not flip in the workers' L1. It flipped on the **DRAM → prefetcher-L1 relay**, and the prefetcher's
kernel-binary cache then **latched it permanently**: every relaunch replays the corrupt copy from L1 instead of re-reading DRAM.

**This is not the fault described in `HANG_HYPOTHESIS_260820_combine_col3.md`.** That was a GDDR byte-lane-3 read fault on device 10 of a
different host, with stale-byte substitution at offsets ≡3 (mod 4) and saturated EDC counters. Every one of those signatures is **absent**
here (§4). Same blast radius, different defect.

| | |
|---|---|
| **Source** | `new_bh-glx-120-d08u02_hang_v00` (triage capture, 10:02) + live read-only probes against the still-hung process |
| **Host** | `bh-glx-120-d08u02`, 32× Blackhole, logical mesh 8×4, flash bundle 19.12.0.0, KMD 2.10.0, UMD 0.9.9 |
| **Test** | `test_prefill_transformer_chunked.py::test_glm_prefill_transformer_chunked_no_pcc -k "glm52 and torus-xy-8x4 and L78 and preload0 and chunks_eleven and ten_iters"` |
| **State** | pytest `pid 108664` started 09:28, still hung when this was written. All probes read-only, no writes, no core halts. |

---

## 1. The fault, measured

Kernel `compute/13219877858508998789/trisc1` (`.text` = 23,536 B) is loaded at L1 `0xb220` on device 21.
One byte differs from the ELF:

```
 .text offset  +0x0368        L1 address 0x0000b588
 expected      b7 07 00 28  =  0x280007b7   lui a5,0x28000
 observed      b3 07 00 28  =  0x280007b3   xor 0x04  -> bit 2 cleared, 1 -> 0
```

`0x280007b3` is opcode `0110011` (OP, R-type) with `funct3=0`, `funct7=0x14`, `rs1=x0`, `rs2=x0`, `rd=a5`. `funct7=0x14` is not a defined
encoding for `funct3=0` (valid: `0x00` ADD, `0x20` SUB, `0x01` MUL). binutils for this target agrees — it refuses to name it:

```
$ riscv-tt-elf-objdump -D -b binary -m riscv
   0:   280007b7        lui     x15,0x28000        <- expected
   0:   280007b3        .insn   4, 0x280007b3      <- observed: undecodable
```

### 1a. Where it is in the source

```
$ riscv-tt-elf-addr2line -f -C -i -e .../compute/13219877858508998789/trisc1/trisc1.elf.xip.elf 0x7618
ckernel::ckernel_template::program()          tt_llk_blackhole/common/inc/ckernel_template.h:365
ckernel::_llk_math_topk_xl_copy_init_()       tt_llk_blackhole/llk_lib/experimental/llk_math_eltwise_unary_datacopy_topk_xl_copy.h:42
llk_math_topk_xl_copy_init()                  hw/ckernels/blackhole/metal/llk_api/experimental/llk_math_topk_xl_copy_api.h:17
ckernel::topk_xl_copy_tile_init()             hw/inc/api/compute/experimental/topk_xl.h:172
process_chunk<2048>                           ttnn/.../topk_large_indices/device/kernels/compute.cpp:90
kernel_main()                                 ttnn/.../topk_large_indices/device/kernels/compute.cpp:134
```

The instruction builds a Tensix instruction word (`0x28000000`) that is then written into the math template's instruction slots
(`sw a5,20(s11)` / `28(s11)` / `32(s11)`). It is **top-k math-template setup, 0x348 bytes into a 23 KB kernel** — trisc1 reaches it almost
immediately after GO, every launch. The stuck op reported by triage is `53111 TopkLargeIndicesDeviceOperation` on device 21, 120 cores. Same
op, same kernel, same 120 cores.

### 1b. Three sibling devices hold the identical binary, byte-perfect

Devices 17, 25 and 29 have the same kernel cached at the same L1 base `0xb220` (they ran this op earlier and moved on), which makes them a
free control. Full 23,536-byte compare:

```
 dev  core    diffs vs ELF (offset >= 0x20)     direct L1-vs-L1
  17  12-8         0                             dev17 vs dev25:  0 diffs
  25  11-8         0                             dev17 vs dev29:  0 diffs
  29  11-8         0                             dev25 vs dev29:  0 diffs
  21   1-2         1  -> +0x368                  dev17 vs dev21:  1 diff  (+0x368: b7/b3)
                                                 dev25 vs dev21:  1 diff
                                                 dev29 vs dev21:  1 diff
```

(The first 32 bytes of the `.text` section are an XIP header that is not loaded into L1. It shows up as a 30-byte diff on **every** device
including the healthy ones, and is the reason triage reported all 720 findings — see §5.1.)

### 1c. It is real device state, not a readback artifact

The same word read eight different ways and 20 times over 10 s — `l1_mem_access.read` at 4/16/64/4096-byte widths, offset and sliced,
`read_words_from_device`, `read_from_device` at 4 and 256 bytes:

```
  every path, every repeat:  b3 07 00 28     (1 distinct value in 20 reads)
```

And it is not one core. Of device 21's tensix cores, **all 70 readable via the 4-byte probe and all 120 via the block scan return `b3`**; on
devices 17/25/29/20/16 the same address reads `b7` on every core that holds this kernel.

### 1d. The DRAM source copy is CORRECT

Kernel binaries live in a DRAM buffer, allocated **top-down** (`tt_metal/impl/program/program.cpp:2385`), page size 2048, interleaved
`bank = page % 8`. Scanning the top of device 21's DRAM for a 40-byte signature preceding the corrupt site found it in 10 s:

```
  bank 3 @ 0xfefe3000 + 0x368  ->  b7   CORRECT
```

Reconstructing the whole 23,536-byte blob across banks and comparing against L1:

```
 dev  core    DRAM blob vs L1 blob
  21   1-2      1 diff   +0x0368: DRAM b7 -> L1 b3
  21   2-2      1 diff   +0x0368: DRAM b7 -> L1 b3
  21   5-9      1 diff   +0x0368: DRAM b7 -> L1 b3
  17  12-8      0 diffs
  25  11-8      0 diffs
  29  11-8      0 diffs

 cross-device DRAM comparison:  dev21 vs dev17 = 0 diffs,  dev21 vs dev25 = 0 diffs
```

The DRAM read path at that exact address is also clean and stable — 3 aliased NOC ports × 20 reads of 2 KB, `0` repeat diffs and `0`
cross-port diffs (this is the oracle from `HANG_HYPOTHESIS…§1a`, applied at the address that matters).

**So DRAM holds the right byte, on device 21, stably, through every port. The corruption is downstream of DRAM.**

### 1e. The corruption is latched in the prefetcher's L1 cache

Searching the dispatch cores' L1 for the same signature:

```
  dev21 PREFETCH_HD 16-2   sig @ L1 0x0fc280  ->  byte 0x0fc2a8 = 0xb3   CORRUPT
  dev21 DISPATCH_HD 16-3   sig @ L1 0x04e610  ->  byte 0x04e638 = 0xb3   CORRUPT
  dev17 PREFETCH_HD 16-2   sig @ L1 0x0fc280  ->  byte 0x0fc2a8 = 0xb7   correct
  dev25 PREFETCH_HD 16-2   sig @ L1 0x0fc280  ->  byte 0x0fc2a8 = 0xb7   correct
  dev20 PREFETCH_HD 16-2   sig @ L1 0x0fc280  ->  byte 0x0fc2a8 = 0xb7   correct
```

Same L1 address on all four devices; device 21 alone is wrong. `0xfc2a8` reads `b3` on 20 reads over 10 s.

**And it is exactly one byte, not a broadly corrupt cache.** Diffing the full 1.5 MiB prefetcher L1 across the four row-6 devices, restricted
to the 64 KiB region holding this blob:

```
  dev17 vs dev21:  2 diffs   0x0f1ce2: 70/60      0x0fc2a8: b7/b3
  dev17 vs dev25:  1 diff    0x0f1ce2: 70/40
  dev17 vs dev29:  1 diff    0x0f1ce2: 70/50
  dev25 vs dev29:  1 diff    0x0f1ce2: 40/50
```

`0x0f1ce2` differs on all four (0x70/0x60/0x40/0x50, 0x10 apart) — a per-device metadata field, benign. `0x0fc2a8` is device 21's, and only
device 21's. **One flipped bit in a 64 KiB region of cached program binaries.**

### 1f. Why it is permanent

Two dispatch paths exist for kernel binaries (`tt_metal/impl/program/dispatch.cpp:1740-1775`):

* **cache miss** → `CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER` with `base_addr = kernel_bins_base_addr`: the prefetcher reads the binary **from
  DRAM** into its own L1 ring buffer (`dispatch.cpp:2890`).
* **cache hit** → `add_prefetch_set_ringbuffer_offset` + `add_prefetch_relay_ringbuffer`: the prefetcher relays **from its L1 copy**. DRAM is
  never re-read.

So a single bit error during the one-time fill is written into the cache and replayed on every subsequent launch, forever, silently. That is
exactly what the measurements show: DRAM right, prefetcher L1 wrong, dispatcher L1 wrong, all 120 workers wrong and identical.

---

## 2. How one bad bit hangs 32 chips

```
DRAM kernels_buffer on dev21 (bank 3 @ 0xfefe3000) -- CORRECT
        │
        ▼  CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER  (one-time cache fill)   <== BIT FLIPS HERE
prefetcher 16-2 L1 0xfc2a8 = b3   -- latched, replayed on every launch
        │
        ▼  relay_ringbuffer -> dispatcher 16-3 L1 0x4e638 = b3
        ▼  NOC multicast, one transaction, 120 destinations
120 × worker L1 0xb588 = b3   (identical on every core -- single upstream source)
        │
        ▼  trisc1 executes .text+0x348 = 0x280007b3  -- undecodable instruction
        ▼  inside ckernel_template::program(), top-k math template setup
trisc1 never programs the math template, never advances
        │
        ▼  op 53111 TopkLargeIndices never retires on dev21 (120 cores)
        ▼  row 6's HighBwAllGather (53113) blocks on dev 17, 25, 29
        ▼  remaining 28 devices pile up at op 53132 HighBwAllGather
whole 8x4 mesh hung
```

Corroborating state from the capture:

* `dump_op_window`: `53111 RUNNING` on device **21** only; `53113 RUNNING` on **17, 25, 29**; `53132 RUNNING` on the other **28**.
* Row 6 of the mesh is exactly `{25, 29, 21, 17}` — and exactly the four devices `check_binary_integrity` flagged.
* `dump_fast_dispatch` `last_wait_count`: dev **21 = 5760**, dev 17/25/29 = **6000**, all others = **8280**. Device 21 is the furthest
  behind, i.e. the primary blocker; 17/25/29 are one step behind the rest, i.e. second-order.
* Live `riscv_pcs` (MMIO `0xFFB13138`, non-invasive) on device 21: brisc is oscillating between two addresses inside the writer kernel — a
  wait loop — while the trisc PC words are **pinned to a single value on all 120 cores**. Nothing is advancing.

---

## 3. Live probe method (all read-only, reproducible)

1. **Failure-set extraction** — parse the 720 `check_binary_integrity` lines into `(dev, core, risc, l1_addr, elf_path)`.
2. **L1 vs ELF** — `create_l1_memory_access(loc).read(kernel_offset, len(.text))`, twice per site, compare, histogram diffs mod 4.
3. **Cross-device L1 vs L1** — same kernel, same base, on the flagged devices and healthy controls.
4. **DRAM source location** — scan `[0xFC000000, 0xFF000000)` per bank, top-down, for a 40-byte signature preceding the corrupt site.
5. **DRAM blob reconstruction** — `bank = (start_bank + page) % 8`, `row = (start_bank + page) // 8`, address `0xfefe3000 + row*2048`.
6. **Cross-port + repeat oracle** at that DRAM address (3 subchannel ports × 20 reads).
7. **Dispatch-core L1 search** — signature scan over the full 1.5 MiB of `16-2` (prefetch) and `16-3` (dispatch).
8. **Prefetcher L1 pairwise diff** across the four row-6 devices, bucketed by 64 KiB region.
9. **ARC telemetry sweep** — raw tags 0-127 on all 32 devices via `_umd_device.read_arc_telemetry_entry(NocId.NOC0, tag)`, bypassing
   ttexalens' stale `telemetry_tags_map`.
10. **Non-invasive PC** — 5 words at `0xFFB13138`, sampled over time.

Scripts: `e1_read_l1_text.py`, `e6_dram_scan.py`, `e8_edc.py`, plus the inline probes above.

---

## 4. The `HANG_HYPOTHESIS_260820_combine_col3.md` fault is NOT present here

Every discriminator from that document was run. All negative:

| test (doc §) | that fault would show | measured here |
|---|---|---|
| GDDR EDC counters, tags 46-50 (§1c) | `255 / 255 / 1 / 1` on the bad instance | **0 on all 8 instances of all 32 devices** |
| cross-port oracle, 64 KB+ (§1a) | thousands of diffs, all at offset ≡3 (mod 4) | **768/768 endpoints clean**, 7 addresses (`0xb220`, `0xdc90`, `0xe110`, `0xe780`, `0xedc0`, `0xf240`, `0x10000000`), 256 KB × 10 reps, all 32 devices |
| repeat-read at the kernel-bin address | non-determinism | 3 ports × 20 reads, **0** repeat and **0** cross-port diffs |
| byte-lane structure (§1e) | diffs confined to one lane, mod-4 histogram | **one bit in one byte**; nothing lane-structured |
| corruption shape (§1e) | stale bytes from a wrong pipeline position, 32-byte chunks | **single bit cleared**; no shift model applies |
| `DDR_STATUS` / `DDR_SPEED` (§1d) | (did not flag it there either) | `0x55555555` / 16000 MT/s on all 32 |
| blast radius | 1/8 of pages of **every** DRAM tensor corrupt | DRAM provably clean; one byte in one cached binary |

Two consequences worth carrying forward:

* **The EDC counters are confirmed useless as a gate a second time.** They were the cheapest detector for the device-10 fault; here they are
  silent while a chip runs an illegal instruction. Consistent with §8 of the earlier document.
* **The cross-port oracle is a DRAM-array/read-path detector only.** It cannot see this fault class at all, because the DRAM is fine. Adding
  it to the loop harness, as §6 of the earlier doc recommends, would not have caught this hang.

### New, unexplained: telemetry tag 71

Sweeping raw ARC telemetry tags 0-127 on all 32 devices, one tag isolates device 21 and nothing else:

```
  tag 71:   device 21 = 0x00000101      all 31 other devices = 0x00000000
```

Tag 71 is beyond this UMD's enum (`telemetry.hpp` ends at `NUMBER_OF_TAGS = 69`), so its meaning is unknown here. It is the only telemetry
observable on the host that fingers the culprit chip. Worth decoding against the 19.12 firmware — if it is an error or retry counter, it is a
candidate detector for exactly this class.

---

## 5. Three triage defects that buried the finding

`check_binary_integrity` **did catch this** — device 21's 120 `trisc1` entries are the real fault. They were buried under 600 false positives
and an inverted signal (a real 1-byte error looks *less* alarming than a benign 1130-byte one).

### 5.1 The XIP header is compared but never loaded (affects all 720 findings)

The first 32 bytes of a kernel's `.text` are an XIP header (`01 00 00 00 … 30 96 da 2a 1f 5f 00 00 f0 5b 00 00`, where `0x5bf0` = 23536 = the
section length). They are not what sits in L1 at `kernel_offset`. Result: a 30-byte diff on **every** core of **every** device, healthy ones
included. Fix: compare `section.data[0x20:]` against `kernel_offset + 0x20`, or record the real load base.

### 5.2 Stale launch message vs an overwritten kernel ring buffer (600 of the 720)

Devices 17/25/29 were flagged on `writer_interleaved_scalar` / `reader_interleaved_no_bcast` / `eltwise_binary_sfpu_scalar` at
`0xdc90`/`0xe110`/`0xe780`/`0xedc0`/`0xf240`. Those cores' L1 does not hold those kernels. Proven by content:

```
L1 [0xb220, 0x10e10) on dev17/25/29  ==  compute/13219877858508998789/trisc1/.text, byte-perfect
```

— a single later kernel spanning all five flagged addresses. The launch message still names the older program while the ring buffer has moved
on, so the comparison is against the wrong binary. Every one of those 600 findings is benign. Fix: skip cores whose launch message is not the
currently-dispatched program, or validate the ring-buffer generation before comparing.

### 5.3 `dump_callstacks` reports `PC = 0x0` for all 600 worker RISCs on device 21

The capture shows `PC 0x0` and `Waypoint X` for all 5 RISCs on all 120 cores of device 21, while reading real PCs for that same device's
dispatch and ethernet cores. The live `riscv_pcs` register at `0xFFB13138` returns real, plausible PCs on those same cores. So the `0x0` is a
read failure, not device state — and it is the single most misleading line in the capture, since it reads as "cores are dead/in reset" and
sends triage toward a reset or NOC hypothesis.

---

## 6. Actions

### 6.1 Decide transient-vs-sticky before releasing the hang

Two mechanisms fit every measurement, and one non-destructive test cannot separate them:

* **(a) a transient bit error on the DRAM→prefetcher-L1 relay**, latched permanently by the prefetcher cache; or
* **(b) a stuck cell in the prefetcher core's L1 at `0xfc2a8`, bit 2**.

The discriminator is a write test, so it must come after the hang is no longer needed:

```
write 0xb7 to device 21, core 16-2 (PREFETCH_HD), L1 0xfc2a8, then read back
   reads b7  ->  cell is fine; (a): a one-off relay error, latched by the cache
   reads b3  ->  (b): stuck L1 bit in the prefetcher core -- this chip corrupts every program it caches
```

Then walk the whole prefetcher L1 with a pattern (write/read-back per 4 KB) to see whether `0xfc2a8` is unique or the first of many. If (b),
quarantine the chip; if (a), the chip is probably fine and the software defect in §6.2 is the whole story.

### 6.2 The prefetcher cache has no integrity check (the real software defect)

A single bit error during the one-time DRAM→L1 fill becomes a permanent, silent, whole-mesh hang, and nothing in the stack ever re-reads DRAM
to notice. Three options, cheapest first:

1. **A debug/CI mode that re-verifies the cached copy against DRAM** after fill (or on every N-th launch). Cost is one DRAM re-read per
   program; it would have named this in the first iteration instead of hour two.
2. **A checksum stored alongside each cache entry**, verified by the prefetcher on relay. Cheap on device, catches the fill error at the
   moment it happens.
3. **An env flag to disable the prefetcher cache** (`relay_paged` every launch). Not for production, but it is the one-line experiment that
   proves the diagnosis: with the cache off, DRAM is re-read and this hang should not reproduce.

### 6.3 Make an undecodable instruction attributable

trisc1 hitting an illegal encoding produced no assert, no watcher entry, no waypoint — just 120 pinned PCs and a two-hour mesh hang.
`dump_lightweight_asserts` and `dump_watcher_ringbuffer` both report `pass`. A trap handler on the TRISCs that records `mcause`/`mepc` to a
known L1 slot would turn this from "cores are stuck somewhere" into "illegal instruction at kernel+0x348 on device 21" in one triage run.

### 6.4 Fix the three triage defects in §5

In priority order: 5.2 (600 false positives, actively harmful — it hides the real finding), 5.3 (actively misleading), 5.1 (noise on every
device). With 5.1 and 5.2 fixed, this entire investigation collapses to one triage line:

```
Device 21: functional_workers [...]: trisc1: Data mismatch in section .text at address 0x0000b220  (1 byte)
```

### 6.5 Add a kernel-text provenance check to `tools/triage`

None of the existing checks compares the three copies of a program binary that must agree. The check is cheap and needs no ground truth
beyond the on-disk ELF:

```
for each device, for the currently-dispatched program:
    A = on-disk .text[0x20:]
    B = DRAM kernels_buffer (reconstruct across banks: bank=(k+p)%8, row=(k+p)//8)
    C = prefetcher L1 ring-buffer copy
    D = worker L1 at the real load base
    A==B==C==D  or  report device, stage of first divergence, offset, expected/observed
```

Divergence between any adjacent pair names the leg that corrupted it — DRAM write, relay fill, or multicast — which is precisely the
question this investigation had to answer by hand.

### 6.6 Treat this run's results as suspect

`test_glm_prefill_transformer_chunked_no_pcc` runs **no PCC check**, so a corrupted kernel produces wrong numbers silently until it happens
to hang. Device 21 has been executing a broken top-k math template for however long that cache entry has been warm. Any iteration on this
host that "passed" before the hang should be treated as unverified.
