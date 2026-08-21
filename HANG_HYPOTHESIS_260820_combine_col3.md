# ROOT CAUSE FOUND — device 10 has a single-DRAM-bank byte-lane fault; `Combine` (op 4375) hangs the whole 8×4 mesh

**Verdict**: **Device 10 (Tray 2 / N3, `0000:43:00.0`, unique id `0xd83c92c09148d83e`) has an unreliable DRAM read datapath on one GDDR channel — byte lane 3 (bits 31:24) only.**
Reads from that channel intermittently return a wrong most-significant byte in each 32-bit word; bytes 0-2 are always correct. Confirmed by direct DRAM reads (§1a). The **read path is provably unreliable**; whether the stored MSB is also wrong (i.e. the write path is broken on the same lane) is untestable without writing a known pattern — see §5.1. That poisons the `dst_chip` field of the dispatched MoE routing metadata; `writer_combine` then uses it as an *unchecked array index*, ends up on a wild fabric connection, and spins forever. Everything else in this log is downstream of that.

This is **not** a TT-NN/TT-Metal logic bug and **not** load imbalance. It is silent data corruption on one chip. Two contributing software defects (§5) turn it from a diagnosable assert into a 2-hour unattributable mesh-wide hang.

| | |
|---|---|
| **Source** | `ognjen/llm_triage_cut.txt` (+ the fuller `ognjen/llm_triage.txt`), plus live read-only device probes (§3) |
| **Run** | `MODEL=KIMI_K2_7 ITERS_ID=ten_iters CHUNKS_ID=chunks20 TRACE_ID=notrace .../launch.sh 100`, iteration 16 (`.../log_16`) |
| **Test** | `test_kimi_prefill_transformer_chunked_perf[blackhole-k27-mesh-8x4-L61-preload0-chunks20-ten_iters-margin_auto-notrace]` |
| **Host** | `bh-glx-110-c10u08`, 32× Blackhole (4 trays), logical mesh 8×4, FABRIC_2D, `HybridMeshPacketHeaderT<35>` |
| **State** | pytest `pid 3889574` still hung when this was written (~1 h 50 m). All probes below were run against the live hang, read-only, no core halts. |

---

## 1. The fault, measured

`dispatched_metadata` is `[1,1,41312,3]` INT32 in DRAM — per dispatched row: `{dst_chip, dst_token_idx, dst_topk_indice}`, 12 valid bytes in a 64-byte DRAM page. `reader_untilize` DMAs those pages into `c_9` (`cb_metadata_batch`) on each untilizer core. Blackhole interleaves DRAM buffers over **8 banks** (`LOG_BASE_2_OF_NUM_DRAM_BANKS = 3` — visible in `writer_combine`'s own generated code as `srli t2,s4,0x3` / `andi t0,s4,7`), so `bank = page_id % 8`.

Counting pages in `c_9` whose byte 3 / 7 / 11 (the MSB of each `int32`) is non-zero, grouped by `page_id % 8`:

```
                bank: 0      1      2      3      4      5      6      7
  dev10 core 2-2:   0/8    0/8    0/8   *4/8*   0/8    0/8    0/8    0/8
  dev10 core 3-2:   0/8    0/8    0/8   *7/8*   0/8    0/8    0/8    0/8
  dev10 core 4-2:   0/8    0/8    0/8   *2/8*   0/8    0/8    0/8    0/8
  dev 8 core 2-2:   0/8    0/8    0/8    0/8    0/8    0/8    0/8    0/8
  dev 8 core 3-2:   0/8    0/8    0/8    0/8    0/8    0/8    0/8    0/8
  dev 9  (2-2,3-2,4-2): all zero
  dev11  (2-2,3-2,4-2): all zero
  dev16  (2-2,3-2,4-2): all zero
  dev17  (2-2,3-2,4-2): all zero
  dev18  (2-2,3-2,4-2): all zero
  dev19  (2-2,3-2,4-2): all zero
```

**One device. One bank. One byte lane out of four.** (The "clean" pages in bank 3 are pages where the injected byte happened to be `0x00` — undetectable by this test, so the real hit rate in bank 3 is likely 8/8.)

A representative pair from `dev10 core 3-2`, `c_9` pages 2 and 3 (adjacent rows of the same expert region):

```
page 2 (bank 2, clean):  13 00 00 00 | 20 01 00 00 | 03 00 00 00   -> dst_chip=19  tok=288  topk=3
page 3 (bank 3, FAULTY): 13 00 00 0f | 83 01 00 99 | 02 00 00 9a   -> dst_chip=0x0f000013  tok=0x99000183  topk=0x9a000002
                                  ^^          ^^          ^^   byte 3 of each word
```

The **low three bytes are provably correct**: the `dst_token_idx` sequence across that expert's rows is `266, 271, 288, →387←, 450, 460, 487, 494 …` — strictly monotone, so `0x…0183 = 387` is the right value with only its MSB clobbered. The injected bytes (`0x0f, 0x99, 0x9a, 0x2a, 0xce, 0xbf, 0x94, 0x56, 0x0e, 0x98, …) look like bf16/bf8 activation bytes, i.e. other tenant data on the same bank.

### 1a. Direct DRAM reads — the fault is in the READ PATH, not the stored data

Blackhole exposes each GDDR channel through **three aliased NOC endpoints**. Reading the *same address*
through two of them is a free, non-destructive read-integrity oracle: any disagreement is a proven read
error, with no need for ground truth, no writes and no core halts.

`dispatched_metadata_addr = 0x9E6C0440` (from `reader_untilize` RT arg [4], via the launch message:
`kernel_config_base = 0x9f80`, `rta_offset[1] = 0x2c`). Metal's bank index maps to channels as
`bank = page_id % 8` (`LOG_BASE_2_OF_NUM_DRAM_BANKS = 3`); grouping the 24 endpoints by identical
content identifies the 8 channels, and the faulty one is metal **bank index 3 = NOC ports `0-5`, `0-6`, `0-7`**.

64 KB read per channel, twice on one port and once on a second port of the same channel:

```
device 10 @ 0x9e6c0440 (metadata buffer)      repeat-read diffs   cross-port diffs
  bank 0 (0-0)                                        0                  0
  bank 1 (0-2)                                        0                  0
  bank 2 (0-9)                                        0                  0
  bank 3 (0-5)                                       48               9150   <== FAULTY
  bank 4..7                                           0                  0
device 10 @ 0x10000000 (unrelated offset, 2.5 GB away)
  bank 3 (0-5)                                       48              16178   <== FAULTY
  all other banks                                     0                  0
device  8 @ 0x9e6c0440 (control, same ports)
  all banks                                           0                  0
```

**100 % of differing bytes are at offset ≡ 3 (mod 4).** Same address, same instant, two ports into the
same channel, three different byte-3 values:

```
port 0-5:  03 00 00 ff | 4d 01 00 ff | 04 00 00 ff
port 0-7:  03 00 00 3e | 4d 01 00 be | 04 00 00 3e
port 0-6:  03 00 00 be | 4d 01 00 3f | 04 00 00 3e
           ^^^^^^^^ low 3 bytes identical and correct on all three
```

Stored bits cannot differ per port, so **the read datapath is definitely broken.**

Note the scope of that claim. Over the same 64 KB, ports `0-5` and `0-7` agreed on **49152 / 49152**
bytes in lanes 0-2 and disagreed on 9150 / 16384 bytes in lane 3 — which is simultaneously the proof
that the three ports front the same physical memory, and the proof that lane 3 is unreliable. It is
**not** proof that the array holds the correct MSB: if lane 3 is broken on *writes* as well, the stored
byte was garbage from the moment it was written and the read-side variability would look identical from
outside. Defensible statement: *bytes 0-2 are provably correct and stable; byte 3 is unreliable and its
true stored value is unknowable from outside.* Only a write-then-read-back of a known pattern separates
the two — tier 1 of `tools/dram_stress` (§5.1), which was deliberately not run against the live hang.

Also note the three ports are **not** alternative routes: they all disagree with each other, so none is
authoritative, and the other seven channels do not hold this data at all (`bank = page_id % 8`). There is
no working path to those bytes and therefore no software workaround.

*Read-only caveat, and why §1b supersedes the rates below*: comparing read-vs-read measures
**self-consistency, not correctness** — two reads can be identically wrong. Read-only probing therefore
*understated* the fault badly (it suggested ~0.3 % for back-to-back reads). The real rate, measured
against a written reference in §1b, is **99.6 %** on realistic data. What the read-only tests do
establish rigorously is the *localisation*: whole channel, address-independent (same behaviour 2.5 GB
apart), uniform across all 16 byte-3 positions of a 64-byte burst, byte lanes 0-2 perfect.

Other facts from the read-only phase:

* **Still failing at idle**, ~2 h into the hang: VCORE 816 mV, TDP 57 W, TDC 71 A, 47 °C — indistinguishable
  from its neighbours. So the bad state persists without load once it has appeared.
* `DDR_STATUS = 0x55555555` and `DDR_SPEED = 16000 MT/s` on device 10, identical to all 31 other chips.
  **The DDR status word does not reflect this fault.**

Reproduce in seconds, read-only, on a live hang:

```python
r1 = lib.read_from_device(port_0_5, 0x10000000, device_id=10, num_bytes=65536, context=ctx)
r2 = lib.read_from_device(port_0_7, 0x10000000, device_id=10, num_bytes=65536, context=ctx)
bad = [i for i in range(len(r1)) if r1[i] != r2[i]]      # thousands, all i % 4 == 3
```
See `ognjen/probes/06_dram_find_channels.py`, `07_dram_crossport_oracle.py`, `08_dram_lane_characterize.py`.

### 1b. Write-then-read with a known reference — the authoritative characterisation

Destructive but tiny: a 4 KB window at `0x9E6C0440 + 0x40000`, inside the `dispatched_metadata`
allocation (per-bank size `0x50B00`, so real mapped DRAM) but 256 KB past the ~12 KB this chunk uses, so
nothing in the hung program reads it. No reset, no process interference.

**Per-channel, per-byte-lane, random data, 4 KB written then read back 20×:**

```
device 10                                    device 8 (control)
 bank  port     l0    l1    l2      l3        bank  port    l0 l1 l2 l3
  0    0-0       0     0     0       0          0..7  all     0  0  0  0
  1    0-2       0     0     0       0
  2    0-9       0     0     0       0        (every lane of every channel clean)
  3    0-5       0     0     0   20376  <==
  3    0-7       0     0     0   20408  <==
  3    0-6       0     0     0   20392  <==
  4..7 all       0     0     0       0
                                 /20480 samples
```

**99.5 % of lane-3 bytes wrong. Lanes 0-2: zero errors, ever, anywhere.**

**Pattern dependence (400 reads of 4 KB, port `0-5`)** — this is the key discovery:

| data written into lane 3 | lane-3 wrong | fully-clean reads |
|---|---|---|
| random | **99.614 %** | **0 / 400** |
| uniform `0xA5` | 0.059 % | 390 / 400 |
| word-alternating `0x00`/`0xFF` | 0.037 % | 390 / 400 |
| control channel, random | 0.000 % | 400 / 400 |

Uniform and alternating patterns pass; random fails almost totally. Because:

**The wrong bytes are stale/misplaced reference bytes, not random bit errors.**
* All 1024 observed lane-3 values appear somewhere in the reference (1024/1024).
* In one 4 KB read the *entire* lane-3 stream was the first 64-byte burst's 16 lane-3 bytes **tiled**:
  ```
  expected [0:16]  9a 94 4d 8c c9 29 9d d8 52 0a ac 9b e4 b5 e6 b8
  observed [0:16]  9a 94 4d 8c c9 29 9d d8 52 0a ac 9b e4 b5 e6 b8   <- correct
  expected[16:32]  6c 9b b5 69 43 e3 50 fb 40 ef d6 ca ac 11 33 64
  observed[16:32]  9a 94 4d 8c c9 29 9d d8 52 0a ac 9b e4 b5 e6 b8   <- the first 16 again
  ```
* Only **26 distinct whole-buffer results across 400 reads** (control: 1) — a small family of stale
  variants, not fresh randomness. 842/1024 positions were *never* correct in 400 reads; 0/1024 always were.

Substituting a stale byte is **invisible** when neighbouring lane-3 bytes are identical, which is exactly
why uniform data passes and why the metadata mostly survived: `dst_chip` / `dst_token_idx` /
`dst_topk_indice` all have `0x00` MSBs, so a stale `0x00` reads as correct. Only where the stale source
byte happened to be non-zero did a row break — the handful of rows that hung the mesh.

**Read-size dependence** (single transaction, faulty channel):

```
   4 B: 18.85 %      32 B: 98.83 %      256 B: 99.41 %
  16 B: 74.80 %      64 B: 99.41 %     4096 B: 99.41 %
```

**The write path and the array are fine.** Uniform and alternating patterns round-trip at >99.9 %, which
they could not do if lane-3 writes never landed. So the earlier "stored data may also be garbage" caveat
is now resolved: **writes land, cells hold, the read return path is what fails.**

**Consequence for the run**: one byte in four, in 1/8 of the pages of *every* DRAM tensor on device 10, is
wrong essentially always for high-entropy data. bf16/bf8 activations and weights in that channel were
not occasionally bit-flipped — they were **reliably corrupted**. This chip has been producing garbage for
the whole run, not just at the moment of the hang.

Scripts: `ognjen/probes/09_dram_400reads_known_ref.py`, `10_dram_readsize_sweep.py`,
`11_dram_per_lane_table.py`.

### 1c. Independent hardware confirmation — GDDR EDC error counters

The GDDR6 controllers keep their own EDC (error-detection-code) counters, per GDDR instance, split by
direction. That is exactly the shape of this fault, so it is a fully independent check on the pattern
test. Read via UMD's `FirmwareInfoProvider.get_aggregated_dram_telemetry()` (which walks the MRISC
telemetry table at `GDDR_TELEMETRY_TABLE_ADDR`), one ARC read per chip, no DRAM traffic, no halts:

```
dev inst | corr_rd corr_wr | uncorr_rd uncorr_wr | temp_top temp_bot | verdict
 10    0 |      0      0   |     0       0       |    50      46     | ok   training=SUCCESS
 10    1 |      0      0   |     0       0       |    50      48     | ok   training=SUCCESS
 10    2 |      0      0   |     0       0       |    50      48     | ok   training=SUCCESS
 10    3 |    255    255   |     1       1       |    50      46     | *** EDC ERRORS ***  training=SUCCESS
 10    4 |      0      0   |     0       0       |    50      48     | ok   training=SUCCESS
 10    5 |      0      0   |     0       0       |    48      50     | ok   training=SUCCESS
 10    6 |      0      0   |     0       0       |    48      50     | ok   training=SUCCESS
 10    7 |      0      0   |     0       0       |    50      48     | ok   training=SUCCESS
  8..17 all instances: 0 0 0 0                                       (devices 8, 9, 11, 0, 12, 16, 17)
```

**GDDR instance 3 of device 10, and nothing else on the host.** The same instance the pattern test
localised as metal bank index 3. (Two independent methods both landing on index 3 is consistent with an
identity mapping between metal's bank index and the GDDR instance number; a coincidence would be 1-in-8.)

MRISC FW is 2.16, and per the 19.8 release notes MRISC ≥2.14 clears the EDC counters *after* training — so
these errors accumulated at runtime, not during bring-up. Both corrected counters are pinned at **255,
which is a saturation floor, not a measurement**, so no rate can be derived from them.

**New information the pattern test could not provide: the WRITE direction errors too.**
`corr_edc_wr = 255`, `uncorr_edc_wr = 1`. §1b concluded "writes land, cells hold" because uniform patterns
round-trip at >99.9 % — that remains true, and is consistent: EDC covers both directions, and write errors
that get detected and retried still land correctly while incrementing the counter. So the physical fault is
plausibly **bidirectional on that byte lane**, with the write direction largely recovered by EDC/retry and
the read direction delivering uncorrected wrong data. Note the EDC counters have **no lane resolution** —
they give instance + direction only. Lane resolution comes from the pattern test (§1b); the two are
complementary and agree on the instance.

### 1d. Everything else the hardware exposes says this channel is HEALTHY

This is the important negative result — of every observable enumerated in `ognjen/dram_telem.md`, exactly
one flags this chip:

| observable | faulty instance reads | expected if healthy | caught it? |
|---|---|---|---|
| `corr_edc_rd/wr`, `uncorr_edc_rd/wr` | **255 / 255 / 1 / 1** | 0 | **YES** |
| `DDR_STATUS` (tag 22) | `0x55555555` | `0x55555555` | no |
| ⤷ training_complete bit | 1 | 1 | no |
| ⤷ `gddr_error` bit | 0 | 0 | no |
| ⤷ **BIST complete / BIST failed** | **1 / 0 (passed)** | 1 / 0 | **no** |
| `DramTrainingStatus` (all 8) | SUCCESS | SUCCESS | no |
| `dram_speed` | 16000 MT/s, all 8 equal | equal | no |
| `dram_temperature_top/bottom` | 50 / 46 °C (peers 48-50) | no outlier | no |
| `MAX_GDDR_TEMP` | 50 °C (dev 8: 50, dev 9: 52) | no outlier | no |
| `MRISC_INIT_STATUS` (`0xFFB14010`) | `0xdeadbeef` | `0xdeadbeef` | no |
| `MRISC_POST_CODE` (`0xFFB14014`) | `0x2e`, same as all 7 healthy | — | no |
| `MRISC_MSG_REGISTER` (`0xFFB14018`) | `0` (idle) | 0 | no |
| DDR reset unit (`0x80030010`) | `0xFFFFFFFF` | `0xFFFFFFFF` | no |
| `ARC_STATUS` (`0x80030060`) | `0xc0de0041` | alive | no |
| `ENABLED_GDDR` (tag 36) | `0xFF` — nothing harvested | `0xFF` | n/a |

**BIST passes on a channel with saturated EDC counters and uncorrected errors in both directions.** So
neither training status nor BIST can be trusted as a DRAM health gate.

Two practical notes for anyone repeating this:

* **ttexalens' `telemetry_tags_map` is stale** (`hardware/arc_block.py`) — it stops at tag 63 and omits
  39-51, which includes `GDDR_*_CORR_ERRS` (46-49) and `GDDR_UNCORR_ERRS` (50). Because
  `tt_exalens_lib.read_arc_telemetry_entry` gates integer tags against that map, a sweep of tags 0-127
  reports the GDDR error tags as "unavailable" even though the firmware publishes them. Go through
  `device._umd_device.read_arc_telemetry_entry(NocId.NOC0, tag)` or `FirmwareInfoProvider` instead. This
  cost a wrong "not available on 19.12" conclusion mid-investigation and is worth fixing upstream.
* **Tier 3 identifies the MRISC node empirically**: of the three NOC nodes per GDDR instance, exactly one
  returns `0xdeadbeef` at `0xFFB14010`. On device 10 those are bank 0→`0-11`, 1→`0-2`, 2→`0-9`,
  **3→`0-5`**, 4→`9-11`, 5→`9-3`, 6→`9-8`, 7→`9-6`. Cheaper and safer than assuming the
  documented port-0/port-2 convention.

Script: `ognjen/probes/12_gddr_edc_telemetry.py`.

### 1e. Mechanism pinned: byte lane 3 is assembled from the wrong pipeline position (100 % deterministic)

**Superseded first draft**: this section originally concluded "exactly one 64-byte burst late", inferred
from the six-word sample below. A burst-identity pattern then showed the magnitude was wrong. The exact
model, verified at 100.00 %, is in *Decisive experiment* further down. Both the six-word sample and the
random-data test in §1b are consistent with it.

`ognjen/tt_dram_lane_probe.py` (the DIIM-287 probe) independently reproduces the localisation and, from
its own pattern vector, pins the mechanism.

Verdict: **45 PASS / 3 FLAKY** — device 10 channel 3, all three subchannels, byte lanes `[3]`, bits 24-31.
Device 8: all 24 locations PASS. Restore succeeded everywhere except the faulty channel (expected — lane 3
cannot be read back faithfully to restore it). It also adds a physical hint I did not have:
`byte lane 3: clamshell hint -> top die (advisory)`.

The mechanism falls out of its reported diffs. The probe writes a 77-word pattern vector; the wrong lane-3
bytes are **the lane-3 bytes of the word 16 positions earlier** — 16 words = **64 bytes = one GDDR burst**:

```
 off   word  wrote       read        lane3   expected[word-16]        lane3
 0x044   17  0x00000010  0xff000010  0xff <- pats[1] =0xffffffff      0xff   MATCH
 0x048   18  0x00000020  0xaa000020  0xaa <- pats[2] =0xaaaaaaaa      0xaa   MATCH
 0x04c   19  0x00000040  0x55000040  0x55 <- pats[3] =0x55555555      0x55   MATCH
 0x050   20  0x00000080  0xcc000080  0xcc <- pats[4] =0xcccccccc      0xcc   MATCH
 0x054   21  0x00000100  0x33000100  0x33 <- pats[5] =0x33333333      0x33   MATCH
 0x05c   23  0x00000400  0xff000400  0xff <- pats[7] =0xffff0000      0xff   MATCH
```

**6 / 6 exact**, from a pattern vector unrelated to the one in §1b — where the same 64-byte periodicity
showed up as `observed[16:32] == expected[0:16]`. Two independent tools, two independent patterns, same
answer:

#### Decisive experiment — burst-identity pattern

Write a pattern where 64-byte burst *k* carries its own lane-3 signature (`lane3 = 0x40+k` on all 16 of its
words), read back, and read off which source burst each returned burst's lane 3 came from:

```
FAULTY ch3:   0<-0   1<-0   2<-1x8,0x8   3<-2x8,1x8   4<-3x8,2x8  ...  31<-30x8,29x8
CONTROL ch0:  0<-0   1<-1   2<-2         3<-3         4<-4        ...  31<-31
              lanes 0-2 wrong = 0 in every read on both channels
```

Every destination burst takes **8 words from burst k-1 and 8 from burst k-2**. 8 words = 32 bytes, so the
granularity is the 32-byte chunk and the shift alternates. Fitting that against random data:

```
even-chunk shift 2x32B, odd-chunk shift 4x32B  ->  100.00% match (2024/2024)
the opposite parity assignment                 ->    1.09% match
```

> **Byte lane 3's data is assembled from the wrong pipeline position: in 32-byte chunks, alternating chunks
> are 64 B and 128 B stale. Lanes 0-2 are always on time. The fault is 100 % DETERMINISTIC.**

The *phase* of the alternation flips between reads (one rep matched at 100 % with the parities swapped),
and that is the entire source of the apparent intermittency — the structure never varies.

**This makes the fault deterministic, not intermittent.** Earlier estimates in this document of
"~24 % of reads" came from read-vs-read self-consistency comparisons and are wrong; the `FLAKY` verdict from
`tt_dram_lane_probe.py` is a classifier gap, not a property of the fault.

**Revised physical read**: 32 bytes on a 32-bit channel is 8 beats, and a 2-then-4-entry alternating offset
looks like a **read-pointer / FIFO-depth fault in that lane's read datapath** — not an analog eye or
timing-margin problem. A marginal eye produces scattered, voltage- and temperature-dependent *bit* errors;
this produces byte-exact data from a deterministic wrong offset. Consequence: **a retrain may well not fix
it**, contrary to the eye-closure hypothesis in §6. Test that first after the reset.

Scripts: `ognjen/probes/13_dram_burst_identity.py`, `14_dram_shift_model.py`.

| observation | why the wrong-pipeline-position model explains it |
|---|---|
| uniform data reads back correct (§1b, 0.06 % wrong) | previous burst's lane-3 byte == current one → lag invisible |
| word-alternating `0x00`/`0xFF` also correct | lag is 16 words, an even multiple of the period-2 pattern → invisible |
| random data 99.6 % wrong | previous burst differs → visible almost always |
| the first chunks of a read are correct | nothing precedes them in the transaction (shifts land before the start) |
| 4 B reads only 18.9 % wrong, ≥32-64 B saturate at ~99 % | a single-word read is its own first burst; multi-burst reads always lag |
| metadata mostly survived, a few rows broke | metadata lane-3 bytes are almost all `0x00`, so a lagged `0x00` reads correct |
| EDC errors in *both* directions (§1c) | timing/capture fault on the lane is not inherently one-directional |

Not a stuck bit, not an undriven wire, not cell corruption, and not a marginal analog eye — see the
revised physical read above.

Note this is a *narrower and different* flavour from DIIM-287's documented signature (`0xdeadbeef ->
0xffffbeef`, top **16** bits stuck high — two byte lanes, deterministically stuck). Ours is one byte lane,
lagged rather than stuck. And the hang path differs too: DIIM-287 hangs UMD's membar readback loop, whereas
this one hung a TT-NN combine kernel through an unchecked `dir_to_slot[]` index (§2). Same fault class, two
distinct blast radii — both worth adding to the ticket.

### Feedback for `tt_dram_lane_probe.py`

1. **It silently fails to read the EDC counters.** Its telemetry block reports
   `corr_rd / corr_wr / uncorr_rd / uncorr_wr = null` for every channel, because it goes through
   `ttexalens.tt_exalens_lib.read_arc_telemetry_entry`, which gates integer tags against the stale
   `telemetry_tags_map` in `ttexalens/hardware/arc_block.py` (stops at 63, omits 46-50). Use
   `device._umd_device.read_arc_telemetry_entry(NocId.NOC0, tag)` or
   `tt_umd.FirmwareInfoProvider.get_aggregated_dram_telemetry()`. Consequence: the docstring's claim that
   *"neither TAG_GDDR_STATUS nor the EDC counters flag this"* is, at least for this instance, **wrong** —
   EDC reads `255 / 255 / 1 / 1` on the faulty instance and `0` on all 47 healthy ones (§1c), making it the
   cheapest detector available. `TAG_GDDR_STATUS` genuinely does not flag it (BIST passes).
2. **Add a "lane is shifted by N chunks" classifier.** The tool's model is stuck-high/stuck-low, so this
   lands in `FLAKY` / "intermittent" (exit 3, inconclusive) even though it is 100 % deterministic. Search
   `observed_lane[i] == expected_lane[i - s]` per byte lane over **32-byte** granularity (not whole bursts),
   allowing a per-chunk-parity shift pair — the fault here needs shifts (2, 4) with either phase. That is a
   few lines and turns "inconclusive" into an exact diagnosis.
3. **Document that uniform patterns are blind to this.** A `0x00`/`0xFF`-only test reads this channel back at
   99.94 % correct. The walking-ones/zeros vector is what exposes it; that is the tool's real value and
   worth stating in `WHAT THIS DETECTS`.
4. **`noc2axi_port` is `null`** (tag 72 is not published on FW 19.12), so it falls back to the assumed
   port-0/port-2 convention. A cheaper and more robust method: of the three NOC nodes per channel, exactly
   one returns `0xdeadbeef` at `MRISC_INIT_STATUS`. Measured on device 10: ch0→`0-11`, ch1→`0-2`, ch2→`0-9`,
   **ch3→`0-5`**, ch4→`9-11`, ch5→`9-3`, ch6→`9-8`, ch7→`9-6`.

### It is real device state, not a readback artifact

The same 12 bytes read seven different ways — exact window, oversized read + slice, 16-byte chunks, 4-byte chunks, start −0x10 sliced, start −0x04 sliced, repeat — plus a completely separate API path (`read_words_from_device` vs `create_l1_memory_access`), **all agree byte-for-byte**. (This is the check `tools/triage/probe_l1_read.py` exists for; it comes out negative.)

### Nothing is moving

A full-L1 double read (`[0, 0x180000)` in 4 KB blocks, 20 s apart, no halts) on device 10 `1-2 (0,0)`, device 10 `2-2 (1,0)` and device 8 `1-2 (0,0)`:

```
dev 10 core 1-2: 0 changed blocks, 0 changed words
dev 10 core 2-2: 0 changed blocks, 0 changed words
dev  8 core 1-2: 0 changed blocks, 0 changed words
```

**Byte-for-byte identical across 20 seconds** — no CB pointer, no semaphore, no payload byte moved. Device 10 is frozen, not slow.

---

## 2. How one bad byte hangs 32 chips

```
DRAM bank 3 on dev10 corrupts byte 3 of every uint32
        │
        ▼  reader_untilize DMAs metadata pages -> c_9
c_9 row: dst_chip = 0x0F000013 instead of 0x00000013
        │
        ▼  writer_untilize:270  `dst_chip == linearized_mesh_coord` -> false -> "non-local"
        ▼  writer_untilize:306  12-byte metadata write into the sender's c_19 ring
c_19 ring slot on the sender core (0,0): poisoned metadata (measured: slots 1, 20, 28)
        │
        ▼  reader_combine:334  `dst_chip = meta0`  -- NO BOUND CHECK (only output_page_idx is bounded, :344)
        ▼  reader_combine:357-367  route_info[0..3] pushed into c_3 (cb_route_info)
c_3 page 0 on dev10 (0,0), measured:
        route=3           <- get_route(): clean (only compares row/col)
        distance=0x02800005  <- manhattan_distance(coord, 0x0A00001F): GARBAGE
        out_page_idx=3935 <- meta1*8+meta2: clean for this row (its injected bytes were 0x00)
        dst_chip=0x0A00001F  <- POISONED (0x1F = 31 is the true value)
        │
        ▼  writer_combine:307  dest_chip_ids[167772191]  -- wild load off a 32-byte stack array
        ▼  writer_combine:312  get_next_hop_router_direction(garbage_mesh, garbage_chip)
        ▼  routing table miss -> decompress_value() returns INVALID_ROUTING_TABLE_ENTRY = 0xFF
        ▼  writer_combine:320  ASSERT(dir_to_slot[0xFF] != EMPTY)   <- no-op in release
        ▼  writer_combine:321  fabric_connections.get(<garbage slot>).sender
        ▼  wait_for_empty_write_slot() on a bogus WorkerToFabricEdmSender  ==>  SPINS FOREVER
```

Compare device 8's `c_3` page 0 (healthy, same layout, same kernel): `route=3 distance=3 out_page_idx=1387 dst_chip=15` — all sane — and its page 1 holds `route=0xFFFFFFFF`, the `ROUTE_INFO_SENTINEL`, i.e. it ran to completion.

The generated code for the wild index, straight out of device 10's own cached ELF
(`.../kernels/writer_combine/11900733317722444754/brisc/brisc.elf`):

```
5c6c: li   a3,5
5c70: li   a4,255            # INVALID_ROUTING_TABLE_ENTRY (default arm of decompress_value)
5c74: bltu a3,a5,5c84        # compressed 3-bit entry > 5  -> take the 255 path
5c78: addi a4,gp,1216        # else CSWTCH.231[compressed]
5c80: lbu  a4,0(a5)
5c84: add  a4,a4,sp
5c88: lbu  a5,72(a4)         # dir_to_slot[a4]   <-- a4 = 221 (0xDD) or 255 => OOB stack read
5c8c: li   a4,76
5c90: mul  s7,a5,a4          # slot * sizeof(connection)  => wild fabric connection
```

`dir_to_slot[]` is declared with `eth_chan_directions::COUNT` = **5** entries
([writer_combine.cpp:185](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L185)), while
[`decompress_value`](tt_metal/fabric/hw/inc/fabric_direction_table_interface.h#L34-L46) can return **0xDD** (`INVALID_DIRECTION`) or **0xFF** (`INVALID_ROUTING_TABLE_ENTRY`).

### Mesh-wide fan-out of the single stuck core

1. Device 10's `writer_combine` never drains `c_3` → `reader_combine` blocked at `:361` `cb_reserve_back` → no ring credits returned → all three `writer_untilize` blocked at `:293` `credits_sem.wait_min(1)` → `reader_untilize` `:212` and `untilize_combine` `:118/:120` backed up. Exactly the state in `dump_callstacks`.
2. The other seven devices of the combine group finish and park at [writer_combine.cpp:381](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L381), `noc_semaphore_wait(exit_sem_ptr, combine_devices - 1)` — the exit all-to-all. Their `reader_combine` NCRISCs are already back in firmware (`wait_for_brisc_notification`) and their untilizer cores have retired; `dump_running_operations` shows op 4375 with `Device Cnt 8, Core Cnt 11` = 4 cores on device 10 + 1 on each of the other seven.
3. A combine group is **one mesh column** (`combine_devices = mesh_rows = 8`), and the stuck group is exactly column C3 = {8,9,10,11,19,18,17,16}. Columns C0/C1/C2 completed and ran on through 4376 `ReshapeView` and 4377 `PostCombineReduce` into 4378 `ReduceScatter`, which is row-wise across the 4 columns — so all 24 of them are blocked waiting for C3. `cq_dispatch.last_wait_count` = 2040 on C3 vs 2160 elsewhere: a clean 3-op skew.

```
R\C   C0            C1            C2            C3  (= the stuck combine group)
0     0  (T1:N1)    4  (T1:N5)    12 (T2:N5)    8   (T2:N1)
1     1             5             13            9
2     2             6             14            10  <-- FAULTY CHIP
3     3             7             15            11
4     27 (T3:N4)    31 (T3:N8)    23 (T4:N8)    19  (T4:N4)
5     26            30            22            18
6     25            29            21            17
7     24            28            20            16
```

---

## 3. Live probe method (all read-only, reproducible)

Scripts are in the session scratchpad; each is ~30 lines of `ttexalens`:

1. **Progress test** — `create_l1_memory_access(loc).read()` over `[0, 0x180000)` in 4 KB blocks, twice, 20 s apart, on dev10 `1-2`/`2-2` and dev8 `1-2`. Result: zero changed words → frozen.
2. **CB discovery** — read `[0x9f80, 0xbf80)` (kernel-config region; `kernel_config_base = 0x9f80` from `dump_callstacks -vv`'s `Base` column) and locate the 32×4-word local-CB config array by matching `c_3.page_size == 0x3810` (= `l1_alignment + aligned_output_page_size` = 16 + 14336). Found at `cb_l1_base = 0xa130` on every core. Layout on the sender `1-2 (0,0)`:

   | CB | addr | size | pages | page_size |
   |---|---|---|---|---|
   | `c_1` dispatched_metadata scratch | `0x01b380` | `0x800` | 32 | `0x40` |
   | `c_2` experts_tok_counter | `0x01bb80` | `0xc00` | 2 | `0x600` |
   | `c_3` **cb_route_info** (merged hdr+payload) | `0x0fdd80` | `0x7020` | 2 | `0x3810` |
   | `c_5` packet header | `0x104dc0` | `0xc0` | 2 | `0x60` |
   | `c_8` expert_region_offsets | `0x01c780` | `0x600` | 1 | `0x600` |
   | `c_18` receive_buf ring | `0x01cd80` | `0xe0000` | 64 | `0x3800` |
   | `c_19` **metadata ring** | `0x0fcd80` | `0x1000` | 64 | `0x40` |

   On the untilizer cores `2-2/3-2/4-2`: `c_1` counter copy `0x01b380`, `c_0` tiles `0x01bf80`, `c_2` `cb_untilize` `0x020380` (`0xe0000`), `c_9` **cb_metadata_batch** `0x100380` (64 × `0x40`).
3. **Poison map** — walk `c_9`'s 64 pages on every untilizer core of all 8 column-3 devices, flag `dst_chip >= 32 || tok >= 41312 || topk >= 8`, and bucket by `page % 8`. Produces the table in §1.
4. **Readback cross-check** — 7 read variants + `read_words_from_device` on one faulty and one clean page.
5. **Load-balance check** — read `c_2` (global `[384]` counts) and slice by `counter_offset`.

---

## 4. Hypotheses that the probes killed

The layout, measured rather than assumed: `num_routed_experts = 384`, `dispatch_group_size = 8` (= `mesh_rows`, TT_FATAL-enforced), **`experts_per_chip = 12`**, **`num_dispatch_groups = 4`** (one per mesh column), so `counter_offset = dispatch_group_idx * 96 + mesh_row * 12`, and for column C3 that is `288 + row*12`. The measured global counts array is non-zero exactly on `[288, 384)` — group 3 — which confirms the mapping end to end.

Per-device row counts for the stuck group (this is the *whole* combine workload of each chip):

```
row 0  dev  8:  973 rows      row 4  dev 19: 1182 rows
row 1  dev  9:  930 rows      row 5  dev 18: 1231 rows
row 2  dev 10: 1282 rows      row 6  dev 17: 1448 rows
row 3  dev 11:  995 rows      row 7  dev 16:  947 rows
```

* **Load imbalance — dead.** Device 10 has 1282 rows, *below* device 17's 1448. 1282 × 14336 B ≈ **18 MB** out of one sender core: single-digit milliseconds. It has been stuck for ~2 hours.
* **Unbounded work from a `uint32` overflow in the `start_token + expert_tokens` clamp — dead.** Every entry of the counts array is in `[0, 388]`; nothing is anywhere near `2^32`. (The overflow is still a genuine latent defect — see §5.4.)
* **Fabric circular wait between C3's Combine and C0-C2's ReduceScatter — dead.** The freeze is upstream of the fabric: the poisoned `dst_chip` is sitting in `c_3` on the stuck core, and the RS ops are parked at their pre-transfer `noc_semaphore_wait_min`.
* **`data_ready` ring desync from a stale semaphore increment — dead.** The ring pointers and slot contents are self-consistent; the sender is blocked on `cb_reserve_back`, not polling.
* **The earlier PC sample was misleading.** `dump_callstacks` put device 10's writer at [writer_combine.cpp:312](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L312) (`PC 0xb14c` → `.text 0x5c3c`, the `beq` for `dst_mesh_id == my_mesh_id`) — straight-line code no branch targets, so it reads as "running". `check_broken_components` flags that exact core as *"Was halted by triage but is no longer halted"*; the full-L1 freeze test overrides it. The PC is nevertheless **12 bytes before the wild `dir_to_slot[]` index**, which is where it actually died.
* **`dump_expert_counts.py`'s `UnsafeAccessException` — red herring.** It defaults to `--metal-device-id=0 --core=1-2 --counts-cb=10` and is written for the `unified_routed_expert_ffn` CB layout; device 0 is running ReduceScatter, so it read a foreign CB config.
* **Standard health checks miss this entirely.** `check_eth_status`, `check_noc_status`, `check_l1_status`, `check_core_magic`, `check_binary_integrity`, `compare_kernel_text` (every kernel IDENTICAL), `dump_lightweight_asserts`, `dump_watcher_ringbuffer` — all pass. `device_telemetry` reports `DDR Status 0x55555555` and `16000 MT/s` **identically on all 32 chips, device 10 included**. Nothing in the existing check suite looks at DRAM *payload* integrity.

---

## 5. Actions

### 5.1 Confirm and quarantine the chip (do this first)

Run the tool that already exists for exactly this, against device 10:

```bash
source python_env/bin/activate
./build_metal.sh -c --build-tests
tools/dram_stress/run_dram_stress.sh --mode quick     # then --mode soak
```

Tier 1 (`tests/tt_metal/tt_metal/deployment/dram/` → `build/test/tt_metal/unit_tests_deployment`) classifies failures into **write vs read** errors per PCI BDF / device id / **bank id**. Expect device `10` / `0000:43:00.0` to fail on **bank 3** with a byte-lane pattern (the `bytewise-SSN` and `marching one/zero bits` patterns should isolate the lane). That pins it to a DRAM cell/PHY/training fault versus a NoC-to-DRAM read-path fault.

Then: reset/retrain, and if it persists, take the chip out of the fleet. Every result produced on device 10 since this appeared is suspect.

### 5.2 Assume silent numerical corruption, not just this hang

The routing metadata is not special — it is simply the tensor where a corrupted MSB is *lethal* rather than merely wrong. **Every DRAM tensor on device 10 has 1/8 of its pages in bank 3**, including the bf8 dispatched activations and the FFN weights. Byte 3 of each word of those is being corrupted too, which for bf16/bf8 payloads is a silently wrong exponent/mantissa byte in every fourth byte.

This run is iteration 16 of a 100-run loop. **Any earlier iteration on this host that "passed", or that failed PCC / `margin_auto` for unclear reasons, should be treated as invalid** until device 10 is cleared. Worth diffing against the `260818_binary_integrity*` and `log_perf_*` result sets already in the tree.

### 5.3 Make combine fail loudly instead of hanging (contributing defect #1)

The same kernel already guards its handshake path properly and its payload path not at all. Compare:

*handshake*, [writer_combine.cpp:209-219](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L209-L219):
```cpp
ASSERT(fabric_route < eth_chan_directions::COUNT);
if (fabric_route >= eth_chan_directions::COUNT) { return; }
const uint8_t connection_slot = dir_to_slot[fabric_route];
ASSERT(connection_slot != DIR_TO_SLOT_EMPTY);
if (connection_slot == DIR_TO_SLOT_EMPTY) { return; }
```

*payload*, [writer_combine.cpp:307-321](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_combine.cpp#L307-L321) — `ASSERT` only, i.e. nothing in release:
```cpp
const uint32_t dst_chip_device_id = route_info[3];            // unbounded, straight from DRAM
pkt_route_info.dst_chip_id = dest_chip_ids[dst_chip_device_id];   // 32-byte stack array, unchecked
fabric_route = get_next_hop_router_direction(...);            // can return 0xDD / 0xFF
ASSERT(dir_to_slot[fabric_route] != DIR_TO_SLOT_EMPTY);       // no-op in release
auto& payload_sender = fabric_connections.get(dir_to_slot[fabric_route]).sender;
```

Three small changes:

1. Bound `dst_chip` at its source, next to the `output_page_idx` guard that is already there —
   [reader_combine.cpp:334-348](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp#L334-L348):
   ```cpp
   uint32_t dst_chip = meta0;
   ASSERT(dst_chip < total_mesh_devices);
   if (dst_chip >= total_mesh_devices) { noc_semaphore_inc(untilizer_credits_noc_addrs[c], 1); continue; }
   ```
   `writer_untilize:270` needs the same test before it classifies a row as non-local.
2. Give the payload path the handshake path's runtime guards, and either size `dir_to_slot[]` at 256 entries or reject `fabric_route >= eth_chan_directions::COUNT` **before** indexing it.
3. Make the drop visible — `WATCHER_RING_BUFFER_PUSH` / a lightweight assert / a dropped-row counter in L1 — so triage names the device and the value instead of showing a spinning core. Dropping a row loses a token (wrong output) but that is strictly better than a silent multi-hour mesh hang, and it makes the *hardware* fault attributable in one triage run.

### 5.4 Fix the latent `uint32` overflow while you are in there (contributing defect #2)

Not the cause here (counts max out at 388), but the clamp in all three untilizer kernels is unsafe —
[reader_untilize.cpp:176-184](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_untilize.cpp#L176-L184),
[writer_untilize.cpp:234-242](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/writer_untilize.cpp#L234-L242),
[untilize_combine.cpp:104-112](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/compute/untilize_combine.cpp#L104-L112):

```cpp
} else if (start_token + expert_tokens > max_dispatch_buffer_token_size) {   // wraps
```
A count near `2^32` wraps, defeats the clamp, blows `actual_batches` up to ~10^8, and — because all three kernels use the identical formula — the pipeline stays in lockstep and runs forever with no assert. Same class of failure the counts array itself could produce if this bank fault ever hits it:
```cpp
if (start_token >= max_dispatch_buffer_token_size) {
    expert_tokens = 0;
} else {
    const uint32_t room = max_dispatch_buffer_token_size - start_token;
    if (expert_tokens > room) { expert_tokens = room; }
}
```
Also clamp each `local_expert_counts[e]` at load time so one bad word cannot corrupt the running `start_page_tiled` for every later expert.

### 5.5 Also worth hardening

* `reader_combine`'s poll uses `if (*data_ready_sem_ptrs[c] == consumed[c]) continue;` ([reader_combine.cpp:308-312](ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/reader_combine.cpp#L308-L312)). If that semaphore word ever ends up *below* `consumed[c]`, the equality can never be satisfied again and the sender consumes phantom ring entries forever. Use a wrap-safe `if ((int32_t)(ready - consumed[c]) <= 0) continue;`.
* Host-side validation of `expert_token_counts` before launching combine: every entry `<= max_dispatch_buffer_token_size`, and the sum consistent with the chunk's token count. That is a cheap tripwire for exactly this bank fault.
* `tools/triage`: add a DRAM payload-integrity check (write/read-back a pattern per bank on every device) — none of the existing checks would have caught this. And `dump_expert_counts.py` should take the combine CB layout and default to the device `dump_op_mesh` marks `[!]`, rather than device 0.

---

## 6. Is it di/dt? And what to do around the reset

### What the signature supports

**Consistent with di/dt as a *trigger*.** An intermittent, single-byte-lane **read** failure is exactly what a
marginal read eye looks like. Training passes at boot in a quiet electrical environment (`DDR_STATUS` =
`0x55555555`), then load-induced supply droop and thermal drift close that one lane's DQ/DQS timing
margin, and nothing ever re-checks. That matches the observed "works fine for a while after a reset,
then starts corrupting" behaviour exactly.

**But the localisation argues against a board- or rail-level power-integrity problem.** A genuine di/dt
event on VDDQ / the PHY rail would not confine itself to *one byte lane of one channel on one chip* out
of 32 sharing the tray and running the identical workload. That points to a **per-lane marginality** —
DQ/DQS read timing on that byte lane, or a physical defect in those 8 DQ nets (ball / via / trace /
solder joint), or the GDDR device itself. di/dt would then be the aggravator, not the cause.

**One data point pushes against live droop being *required*:** it is still failing right now, with the
machine idle-hung for ~2 h at VCORE 816 mV / TDP 57 W / 47 °C, indistinguishable from its neighbours. So
once the fault appears it **persists at idle** — either training drifted and stays bad until retrain, or
the lane is marginal enough to fail even unloaded. Either way, a reset plausibly clears it for a while,
as you predicted.

### The discriminating experiments (all need the reset, so they come after)

1. **Does the failing lane come back in the same place?** Reset, reproduce, then run the cross-port
   oracle. *Same chip + same channel + same byte lane* ⇒ physical / per-lane marginality (board-level
   or RMA). *Moves around* ⇒ global power / training instability.
2. **Does it need load?** After reset, run the oracle at idle, then under `dram_stress --mode soak`, and
   record time-to-first-error. Idle-clean but load-dirty ⇒ droop / thermal. Dirty at idle right after
   training ⇒ hard marginality.
3. **Capture ARC telemetry continuously** through the failing workload (VCORE, TDP, TDC, GDDR temp,
   throttle counters). A droop or thermal excursion on device 10 correlated with the first error is the
   di/dt smoking gun. Note the current telemetry version on this host did not expose
   `THM_LIMIT_THROTTLE` / `VDD_LIMIT_THROTTLE` / `GDDR_TEMP` keys — worth checking whether a newer ARC FW
   does, since their absence is why nothing flagged this.
4. **Per-byte-lane read-eye margins after retrain.** If the ARC exposes training margins, a lane sitting
   at near-zero read margin names the culprit directly and distinguishes "marginal by design point"
   from "damaged".
5. **Tier-1 `tools/dram_stress`** for the authoritative per-bank write-vs-read classification, and to
   confirm whether the **write** path on that lane is affected too. The cross-port oracle only proves the
   read path is broken; writes are untestable without writing, which I did not do.

### Add the cross-port oracle as a routine check

It needs no ground truth, no writes, no core halts, and takes seconds per chip:

```
for each device, for each of the 8 GDDR channels:
    a = read(port[0], addr, 64 KB)
    b = read(port[1], addr, 64 KB)          # different NOC port, same channel
    c = read(port[0], addr, 64 KB)          # repeat on port[0]
    any a != b  or  a != c   ==>  read-integrity fault; report device, channel,
                                  and histogram the differing offsets mod 4 to name the byte lane
```

Run it between iterations of the 100-run loop and this class of fault is caught in seconds, attributed to
a chip, a channel and a byte lane — instead of surfacing two hours later as an unattributable mesh-wide
hang, or worse, as silently wrong numerics.

**Detector, not quantifier.** The oracle compares read against read, which measures *self-consistency,
not correctness* — two reads can be identically wrong. Here it localised the fault perfectly (chip,
channel, byte lane, address-independence) but **understated the error rate by two orders of magnitude**
(~0.3 % vs the 99.6 % measured in §1b against a written reference), and suggested a wrong mechanism
("intermittent undriven residue" rather than "stale bytes returned, near-total on high-entropy data").
Use the oracle to decide *whether and where*; use write-then-read-back to decide *how bad, which
direction, and why*.

---

## 8. Post-reset verification (clean reset, 2026-08-20, no process holding the devices)

ARC uptime 41 s, all 32 devices visible, `DDR_STATUS = 0x55555555`, `dram_speed = 16000`,
`ENABLED_GDDR = 0xff` on every chip. **Device 10 is completely clean. The fault does not survive a reset.**

| test | before reset (device 10, ch 3) | after reset |
|---|---|---|
| `tt_dram_lane_probe.py`, all 3 subchannels | **FLAKY**, byte lanes [3], bits 24-31 | **PASS** |
| per-lane table, 3 ports × 4 lanes | **20376 / 20408 / 20392** lane-3 errors | **0** on every lane |
| burst-identity mapping | `2<-1x8,0x8  3<-2x8,1x8 …` | `0<-0  1<-1 … 31<-31` — perfect identity |
| shift-model fit | **100.00 %** (2024/2024) | 0.4 % — chance level, no shift |
| cross-port oracle, 64 KB × 2 addresses | 9150 / 16178 diffs, all at offset ≡3 (mod 4) | **0** diffs, every bank, both addresses |
| EDC counters, instance 3 | 255 / 255 / 1 / 1 | 0 / 0 / 0 / 0 |

**Full-host sweep: 768 endpoints (32 devices × 8 channels × 3 subchannels), 709 632 word comparisons,
all PASS.** No other chip on the host is currently affected.

### What this settles

* The fault is **runtime-induced state, not a permanent physical defect**. A retrain clears it. This
  retires the "damaged lane / RMA" reading and the §1e caveat that a retrain might not fix it.
* **EDC counters are confirmed useless as a pre-flight gate.** All 256 instances read 0 both before *and*
  after the probe hammered DRAM with traffic. They are a post-hoc detector ("did this run corrupt
  anything?"), never a bring-up check.
* Reproduction now requires running the workload with a per-iteration gate. Given the fault appeared
  ~3.5 min into iteration 16 after 15 prior bring-ups, expect roughly one *hang* per few hours of looping —
  and the hang rate is a **lower bound** on the corruption rate, because
  `test_prefill_transformer_chunked` runs **no-PCC** (`margin_auto` is a perf margin, not a numerical check).

### Recommended loop harness

Bracket every iteration:

* **before** — `tt_dram_lane_probe.py` (or probes 11/13/14): catches a chip that came up bad. Seconds.
* **after** — probe 12, GDDR EDC counters: catches corruption that occurred *during* that iteration.

That pair would have named this fault the first time it happened instead of two hours into a mesh-wide hang.

### Additional bug in `tt_dram_lane_probe.py`: exit 4 on every healthy host

Its exit-code logic (line ~744) returns **4** ("firmware reports a channel bad") when
`mrisc_init_status not in (None, 0xDEADBEEF)`. Measured across all 32 devices, through the *assumed*
noc2axi port:

```
ch 0..3 -> 0xdeadbeef ("finished")     ch 4..7 -> 0x00000000 ("started")
```

Perfectly deterministic and device-independent — 128 of 256 telemetry entries. The `0x0` is not a firmware
failure, it is the script reading `MRISC_INIT_STATUS` through a node that is not that channel's MRISC port.
Confirmed by scoping the run: `--devices 10 --channels 3` exits **0**; including any of channels 4-7 exits
**4**. So **the exit status is unusable as a CI gate today** — it fails a fully healthy 32-chip host.

Fix: locate the MRISC node empirically (of the three nodes per channel, exactly one returns `0xdeadbeef`),
or treat a `0x0` read through an unconfirmed port as *unknown* rather than *failed*. Empirical map for
device 10: ch0→`0-11`, ch1→`0-2`, ch2→`0-9`, ch3→`0-5`, ch4→`9-11`, ch5→`9-3`, ch6→`9-8`, ch7→`9-6`.

(Note: `tag 72 / TAG_GDDR_MRISC_NOC2AXI_PORT`, which would give the port authoritatively, is not published
by FW 19.12 — hence the assumption in the first place.)
