# Prefetcher L1 kernel-binary cache — reproducer and diagnostics

Tooling for a fault class where a **kernel binary is correct in DRAM but corrupt in the
prefetcher's L1 cache**, which then multicasts the corrupt copy to every worker core.

First diagnosed on `bh-glx-120-d08u02`, device 21: **one bit** (`b7` → `b3`, bit 2) in a
23,536-byte kernel, which hung a 32-chip mesh for hours with no assert, no watcher entry and no
telemetry. Full analysis in [`HANG_DIAGNOSIS_260821_topk_row6.md`](../../HANG_DIAGNOSIS_260821_topk_row6.md).

---

## 1. Why this fault is permanent, and why nothing catches it

Kernel binaries live in a DRAM buffer allocated top-down
([`program.cpp:2385`](../../tt_metal/impl/program/program.cpp)). Then:

* **cache miss** → `CQ_PREFETCH_CMD_PAGED_TO_RINGBUFFER` reads the binary from DRAM into the
  prefetch core's L1 ring buffer ([`dispatch.cpp:2890`](../../tt_metal/impl/program/dispatch.cpp))
* **cache hit** → dispatch relays from that L1 copy (`add_prefetch_relay_ringbuffer`,
  [`dispatch.cpp:1766`](../../tt_metal/impl/program/dispatch.cpp)) — **DRAM is never re-read, and
  nothing verifies the cached copy**

Three consequences, and all the tooling here follows from them:

1. A single bit error during the one-time fill is **latched and replayed on every subsequent
   launch**, forever, silently. It does not clear on process restart of the *cache entry's*
   lifetime and it produces no diagnostic.
2. **DRAM stays valid ground truth** for whatever is currently cached. That is what makes the
   read-only diagnostics here possible against a live hang.
3. The corrupt byte reaches every worker via **one** multicast, so all destinations are wrong
   *identically* — which is how you tell an upstream source error from per-core corruption.

The ring buffer is **1024 KB** (864 KB with more than one HW CQ), rooted at `scratch_db_base_`
([`dispatch_settings.cpp:69`](../../tt_metal/impl/dispatch/util/dispatch_settings.cpp)).

---

## 2. Prerequisites

```bash
cd $TT_METAL_HOME
source python_env/bin/activate     # provides ttexalens; ttnn only needed for stage 2
```

* All read-only tools need **ttexalens** only. They are safe against a live hang: no writes, no
  core halts, no reset.
* `l1_pattern_test.py` **writes** to L1 and needs the device **free**.
* Stage 2 of the reproducer needs **ttnn** (one device, no mesh, no model).
* `l1_confirm_state.py` uses `riscv-tt-elf-objdump` / `addr2line` from
  `runtime/sfpi/compiler/bin` or `/opt/tenstorrent/sfpi/compiler/bin`. It degrades gracefully if
  neither is present.

Get the real dispatch-core coordinates for your host from triage `dump_fast_dispatch` — the
`PREFETCH_HD` and `DISPATCH_HD` rows. On `bh-glx-120-d08u02` they are `16-2` and `16-3` on every
device **except device 2, which is `15-2`/`15-3`**. Don't assume.

---

## 3. Quick start

```bash
# Is anything corrupt right now?  Read-only, safe on a live hang, ~2 s per core.
python3 tools/l1_prefetch_diag/l1_cache_audit.py --json audit.json

# Something found?  Establish the whole chain end to end.
python3 tools/l1_prefetch_diag/l1_confirm_state.py --audit-json audit.json

# Reproduce it after a reset.  Stage 1 needs no Metal and takes under a second.
python3 tools/l1_prefetch_diag/l1_repro.py --devices 21
```

### Decision tree

```
                    l1_cache_audit.py
                            |
              +-------------+--------------+
         findings                      clean
              |                            |
   l1_confirm_state.py            device free?
   (proves the chain,                  |
    names LETHAL/SILENT)      +--------+---------+
                             yes                no
                              |                  |
                   l1_pattern_test.py    nothing more to do read-only;
                   (is a cell bad?)      re-audit after the next workload
                              |
                    +---------+---------+
                  FAIL                PASS
                    |                   |
        cell is bad:            cells sound: the error
        sub-second              enters on the DRAM->L1
        permanent repro,        relay -> l1_repro.py --stage 2
        hardware defect
```

---

## 4. The tools

### 4.1 `l1_cache_audit.py` — audit every cached binary *(read-only)*

Compares every kernel binary currently cached in the dispatch cores' L1 against the on-disk ELF
corpus. **Self-discovering: nothing is hardcoded**, so it keeps working after a reset, a rebuild,
or a different workload, when every DRAM address, L1 offset and bank assignment has changed.

```bash
python3 l1_cache_audit.py [--devices all|21,17] [--cores auto|16-2,16-3]
                          [--cache-root DIR]... [--halo 256] [--json OUT] [--quiet]
```

| option | meaning |
|---|---|
| `--devices` | `all` (default) or a comma list |
| `--cores` | `auto` probes columns 15/16/17 × rows 2/3; or give an explicit noc0 list |
| `--cache-root` | kernel cache dir, repeatable. Default `~/.cache/tt-metal-cache/*/kernels` |
| `--halo` | matching bytes required either side of a reported mismatch (default 256) |
| `--json` | write machine-readable findings — **required input for `l1_confirm_state.py`** |

Exit status: `0` clean, `1` corruption found, `2` could not run.

**How to read it.** A finding prints the ELF, the resident window and its agreement percentage,
then each bad byte with its L1 address, `.text` offset, expected/observed values, and the exact
bit numbers:

```
device  21 16-2: *** 1 corrupt blob(s) *** (184 found, 503924 B compared, 0.8s)
    1 byte(s) / 1 bit(s) wrong in .../compute/13219877858508998789/trisc1/trisc1.elf.xip.elf
    resident window 23504 B, 23503 agreeing (100.00%)
      L1 0x0fc2a8  .text+0x00368  b7 -> b3  xor 04  bit(s) [2]
device  17 16-2: clean  (187 blobs, 502432 B, 0.8s)
```

Three implementation details make it trustworthy, each learned by getting it wrong first:

* **Anchors are 64-byte windows proven unique across the whole corpus.** A short anchor collides
  between kernels that share a function prologue and reports *megabytes* of phantom corruption on
  perfectly healthy devices.
* **Every occurrence of an anchor is scored and the best-agreeing alignment wins.** The ring
  buffer can hold more than one copy of a blob; taking the first `find()` hit mis-aligns against
  a stale copy and *silently hides* the real finding.
* **A mismatch is reported only when `--halo` bytes on both sides match.** The relayed length is
  `kg_transfer_info.lengths[]`, not the whole `.text` body, so a blob's tail runs past its cache
  entry into the next program. The halo rule rejects those boundaries, truncation tails and
  misalignment, and keeps isolated bit errors.

If you ever need to loosen `--halo`, expect boundary noise and read the agreement percentage
before believing a finding.

### 4.2 `l1_confirm_state.py` — confirm the troublesome state occurred *(read-only)*

Takes the audit JSON and establishes the full chain for each corrupt byte, so there is no doubt
about what happened or where.

```bash
python3 l1_confirm_state.py --audit-json audit.json
                            [--worker-scan 160] [--oracle-reps 20]
                            [--dram-top 0xFF000000] [--dram-low 0xFC000000]
                            [--dram-chunk 0x200000] [--skip-dram]
```

| step | what it establishes |
|---|---|
| `[1] ON-DISK ELF` | the reference word and its disassembly |
| `[2] DRAM` | locates the `kernels_buffer` copy **by content** and reads the same byte, then runs the cross-port oracle there (3 aliased NOC ports × `--oracle-reps` reads). DRAM correct + cache wrong ⇒ the error entered on the relay, and the DRAM read path is exonerated |
| `[3] PREFETCH L1` | re-reads the corrupt byte through 4 independent paths and widths — rules out a readback artifact |
| `[4] FAN-OUT` | how many cores received the corruption via the multicast |
| `[5] DECODE` | disassembles the word before and after, resolves it to a source location, and states whether the byte is **LETHAL** (undecodable → the core traps) or **SILENT-OR-WRONG** (still decodes → executes with wrong effect) |

The DRAM scan walks down from `--dram-top` because the `kernels_buffer` is allocated top-down.
Widen `--dram-low` if it reports "not found in the scanned range". `--skip-dram` skips step 2,
which is the only slow part (~10 s).

Reference output for the device-21 fault:

```
device 21  core 16-2  compute/13219877858508998789/trisc1/trisc1.elf.xip.elf
  .text+0x00368   expected b7   observed b3   xor 04   bit(s) [2]

  [1] ON-DISK ELF   word 0x280007b7   lui a5,0x28000
  [2] DRAM          bank 3 @ 0xfefe3368  byte = b7  CORRECT
                    port 0-5 / 0-7 / 0-6: 1 distinct value over 20 reads, byte b7
                    cross-port oracle: all 3 ports AGREE -> DRAM read path is not the fault
  [3] 16-2 L1 0x0fc2a8   read(4)=b3  read(64)@-32=b3  read(4096)@-2048=b3  read_words=b3
                    1 distinct value across 4 independent read paths -> NOT a readback artifact
  [4] fan-out       122 core(s) hold the CORRUPT byte, 0 hold the correct byte
  [5] EXECUTED      word 0x280007b3   UNDECODABLE -- illegal instruction
      ckernel::ckernel_template::program()        ckernel_template.h:365
      ckernel::_llk_math_topk_xl_copy_init_()     ...topk_xl_copy.h:42
      ckernel::topk_xl_copy_tile_init()           topk_xl.h:172

  VERDICT: LETHAL -- the core traps on an undecodable instruction
```

`[4]` counts every core holding that blob, which includes the audited dispatch core and its peer —
so 120 workers reads as 122.

### 4.3 `l1_pattern_test.py` — L1 cell test *(WRITES; device must be free)*

Pattern-writes the dispatch cores' prefetch ring buffer and reads it back. This is the test that
distinguishes a **bad L1 cell** from **corruption in transit**, and if a cell is bad it is also
your permanent, sub-second reproducer.

```bash
python3 l1_pattern_test.py [--devices all] [--cores 16-2,16-3]
                           [--start 0x0F0000] [--len 0x020000]
                           [--seed 1] [--dry-run] [--force]
```

Patterns: `0x00`, `0xff`, `0x55`, `0xaa`, walking ones, walking zeros, address-dependent, random.
Contents are saved and restored per 4 KB block; a block that fails to restore is reported, which
is expected at a genuinely faulty location. Failures are reported per **(address, bit)** with the
count of patterns that caught them.

```bash
# targeted: one known 4 KB block, effectively instant
python3 l1_pattern_test.py --devices 21 --start 0xfc000 --len 0x1000

# fleet screen: whole ring buffer, both dispatch cores, every chip
python3 l1_pattern_test.py --start 0x020000 --len 0x100000

# validate plumbing without writing anything (safe on a live hang)
python3 l1_pattern_test.py --dry-run
```

**Safety.** It refuses to write while any process holds `/dev/tenstorrent`, and prints the pids.
`--force` overrides that; only use it if you accept corrupting the running process.

### 4.4 `l1_repro.py` — the reproducer

Orchestrates both stages, cheapest first.

```bash
python3 l1_repro.py [--devices 21] [--cores 16-2,16-3]
                    [--ring-start 0x020000] [--ring-len 0x100000]
                    [--stage 1|2|both] [--rounds 20] [--iters 40]
                    [--workload-device 0] [--outdir l1_repro_out]
```

| stage | needs | proves | runtime |
|---|---|---|---|
| 1 — L1 cell test | ttexalens, device free | a stuck/weak L1 cell in the ring buffer | **< 1 s** |
| 2 — relay test | ttnn, one device | the DRAM→L1 relay corrupts on transfer | ~10 s/round |

Stage 2 runs `_workload.py` in its own process, which **exits before each audit**, so nothing
contends for the device — L1 is SRAM and keeps its contents across process exit.

Exit status: `1` reproduced, `0` not reproduced, `2` could not run.

### 4.5 `_workload.py` — stage-2 workload

Not run directly in normal use. One device, no mesh, no model. Its job is not the arithmetic —
it is to compile and dispatch many **distinct** programs (varying op, shape, dtype and layout) so
the prefetcher cache fills and wraps, walking the whole ring buffer. Every fill is a
`PAGED_TO_RINGBUFFER` read from DRAM into the prefetch core's L1: the leg where the error enters.

---

## 5. Runbook: reproducing the hang

### 5.1 After a machine reset — start here

```bash
source python_env/bin/activate
python3 tools/l1_prefetch_diag/l1_repro.py --devices 21
```

**Stage 1 FAIL is the good outcome for reproducibility.** A bad cell is reset-proof, so you get a
sub-second deterministic reproducer, on one device, with no model — and you never run the
2-hour test again. Record the address and bit, then treat the chip as a hardware defect.

**Stage 1 PASS** means the cells are sound and the error enters on the transfer. Stage 2 takes
over automatically, dispatching small programs and auditing between rounds until it catches one.

Either way, when something is reported:

```bash
python3 tools/l1_prefetch_diag/l1_confirm_state.py \
        --audit-json l1_repro_out/audit_round001.json
```

### 5.2 If stage 2 does not reproduce

The workload's programs have to actually land on the faulty address, and a single bad bit is
**mostly silent** — it only shows when the byte that lands there has that bit set (roughly half
the time), *and* lands in code rather than data or padding, *and* in a path that executes.

* Raise `--iters` so more distinct programs cycle through, and `--rounds` to keep going.
* Confirm the sweep covers the real ring buffer. The audit prints the L1 base of every blob it
  finds, which brackets `scratch_db_base_`; pass that as `--ring-start`.
* Run stage 1 across the **whole** ring buffer on **all** devices and both dispatch cores. A
  targeted 4 KB window will miss a bad cell at a different offset.

### 5.3 Last resort — reproduce with the real workload

If neither stage catches it, fall back to the full model run and audit *between* iterations:

```bash
# in one shell: the workload that originally hung
python3 -m pytest models/demos/deepseek_v3_d_p/tests/test_prefill_transformer_chunked.py \
        -k "..." -xvs

# in another: audit every iteration. Read-only, safe alongside the running workload.
while true; do
  python3 tools/l1_prefetch_diag/l1_cache_audit.py --quiet --json /tmp/a.json && sleep 30 \
    || { python3 tools/l1_prefetch_diag/l1_confirm_state.py --audit-json /tmp/a.json; break; }
done
```

That loop is also the recommended **CI gate**: it names the device, address and bit in seconds
instead of surfacing hours later as an unattributable mesh-wide hang.

### 5.4 Confirming the software amplifier

Independently of the chip, confirm that the missing cache-integrity check is what turns one bad
bit into a mesh hang: force `relay_paged` on every launch (cache disabled) so DRAM is re-read
each time. The hang should disappear while the chip is still defective. That both proves the
diagnosis and gives you a stopgap if the host has to stay in the fleet before the RMA.

---

## 6. Reading triage output for this fault class

`check_binary_integrity` **does** catch this, but its output inverts the signal. Three things to
know, all measured on the original capture:

* **The line count is not a byte count.** The check logs one boolean per (core, risc, section), so
  a 1-byte error and a 1,131-byte error produce identical-looking lines.
* **A big mismatch is the benign one.** 96–99% of bytes differing means the tool followed a stale
  launch message and compared against the *wrong binary* — the L1 content is a correct, later
  kernel. Single-digit byte counts are the real defects.
* **Which riscs fire tells you which it is.** All 5 riscs failing on a core ⇒ wrong-content
  artifact (a byte error cannot hit five independent kernels at once). **One** risc failing on
  **every** core ⇒ one upstream source, faithfully broadcast — a real defect.

Also: the first 0x20 bytes of a kernel's `.text` are an XIP header that is never loaded into L1,
so it shows as a ~30-byte diff on every core of every device including healthy ones. The tools
here skip it (`HDR = 0x20`). And `dump_callstacks` can report `PC = 0x0` for worker RISCs when its
read fails — cross-check with the non-invasive `riscv_pcs` MMIO block (5 words at `0xFFB13138`)
before concluding cores are dead.

---

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no kernel cache found` | pass `--cache-root <dir>/kernels`; the build may use a non-default cache |
| audit finds 0 blobs on a core | wrong coordinates — get `PREFETCH_HD`/`DISPATCH_HD` from triage `dump_fast_dispatch`, or use `--cores auto` |
| audit reports huge diff counts | you lowered `--halo`, or the corpus does not match the running build. Check the agreement percentage; a real finding sits at ~100% |
| `REFUSING: pid(s) ... hold /dev/tenstorrent` | expected. Kill the Metal process, or use `--dry-run` for a read-only check |
| `kernels_buffer not found in the scanned range` | widen `--dram-low` (default `0xFC000000`). Never scan at or above `0xFF000000` — the top 16 MiB is register space and yields phantom corruption |
| `Could not find L1 memory block` | that coordinate is not a tensix/dispatch core on this device (harvesting differs per chip) |
| stage 2 workload skips ops | harmless — it prints which and continues; op availability varies by build |

---

## 8. Caveats

* A **clean audit** proves that nothing *currently cached* is corrupt. It cannot certify a chip
  healthy — only the pattern test can speak to the cells.
* The cross-port oracle in `l1_confirm_state.py` step 2 compares read against read, so it measures
  DRAM read **self-consistency, not correctness**. Two reads can be identically wrong. Use it to
  exonerate the DRAM read path, never to quote an error rate.
* `_workload.py` (stage 2) **has not been executed against hardware** — the device was held by the
  hung run throughout development. Stage 1 and both diagnostics are validated end to end against
  the live fault.
* These tools read and write dispatch-core L1 directly. They are diagnostics, not a supported API;
  L1 layout and the dispatch command set can change between Metal versions.
