# Engram host path: problem statement, measurements, plan

DeepSeek-V4.1 prefill, 2026-10-28. Written on the Galaxy 8x4 (bh-glx-110-c10u02) for a session on the 2x2 QuietBox
that will continue the work. Branch `dnijemcevic/model_bringup`. Read this whole file before changing anything.

## 1. Context in five lines

- The model runs end to end on the Galaxy 8x4: tt-d-gen prefill + stand-in decode answer "Paris" (K.1 passed).
- Owner decision 2026-10-28: the one configuration is the **block MoE with fused head and fused tail on the
  FABRIC_2D_TORUS_XY torus** (setting `MOE_IMPL=block`, task FIX.6). Flat MoE and plain FABRIC_2D are no longer
  optimized on the Galaxy.
- One warm 5120-token chunk (40 layers): **219.5 ms on the device, 624 ms of host wall**. The model is host-bound.
- The host time is the target of the perf work now. A perf agent's ranked list is in
  `bringup/ledgers/galaxy_8x4-full/perf_proposals.md` (~280 ms per chunk in total). The largest single item is the
  engram host path, this file.
- Owner rules that bind any change: no padding or duplicated work, shipped precision only (no precision raise), the
  same path must stay deterministic, push only on the owner's word.

## 2. What the engram host path does

`tt/engram.py` (`TtEngramRows`), called from `tt/model.py` `begin_chunk(tokens, start)` **before layer 0 of every
chunk**, for each engram layer (layers **1 and 14**; config `engram_layer_ids`):

1. Hash the chunk's token ids with the 3-id tail carried from the previous chunk (`reference/ngram.py`
   `EngramHasher.hashes`): `[S, 24]` table row ids (`(max_ngram_size - 1) * n_heads = 24` per token).
2. Gather those rows from the memory-mapped checkpoint table (`reference/weights.py` `EngramTable.rows`): 256 fp8
   bytes + 8 e8m0 scale bytes per row; ids sorted first. 5120 tokens -> **122,880 random row reads per layer**.
3. Dequant fp8 x e8m0 group scale -> fp32 -> bf16 (exact: bf16 holds fp8 x 2^k), assembled into `[5120, 6144]` bf16
   (63 MB per layer).
4. Upload with `ttnn.from_torch` (tilize, host copy, PCIe), rows split over mesh axis 0, **replicated over the TP
   axis** (4x on the 8x4, 2x on the 2x2).

The tables: per engram layer `layers.<L>.engram.embed.weight` `[384,006,168, 256]` uint8 (91.6 GiB) + `.scale`
`[.., 8]` (2.9 GiB), in `model-00047` (layer 1) and `model-00048` (layer 14) of the HF checkpoint (rev 5254448).

The rows depend only on token ids (+ the 3-id tail), so they can be built before they are needed. The engine
(tt-d-gen) does not need to change for anything below.

## 3. Measurements on the Galaxy host

Host: 566 GB RAM, `/scratch` = one Samsung PM9A3-class 7.68 TB NVMe (nvme3n1) under LVM, **read_ahead_kb = 256**, no
sudo. 64 CPUs.

### 3a. In the model (cProfile of 5 warm runs, block on torus, context 0; posprof/hostprof_torus.pstats)

`begin_chunk` = **145 ms per chunk** on the critical path (both layers), of a 624 ms wall:
- `EngramTable.rows` 62 ms (gather 43.5 self, fp8->float 15.3, argsort 3)
- `host_rows` assembly ~34 ms self (zeros 5.4, `.to(bf16)` 5.1, slice copy), hashing 1.5 ms
- upload (`from_torch`, both layers) ~37 ms; `begin_chunk` own 9 ms
These were **page-cache hot**: the timed runs repeat the same chunk.

### 3b. Standalone bench (`tools/engram_bench.py`, layer 1, 5120 tokens, nice 19 on 16-32 cores beside a device gate)

All runs: rows **byte-equal** to `EngramTable.rows` (`--check`).

| mode | threads | gather ms | total ms |
|---|---|---|---|
| cold, mmap (today's code) | 1 | 4566 | 4646 |
| cold, mmap | 8 | 4518 | 4561 |
| cold, mmap + `MADV_RANDOM` | 1 | 792 | 931 |
| cold, mmap + `MADV_RANDOM` | 8 | **484** | **587** |
| cold, mmap + `MADV_RANDOM` | 32 | 545 | 649 |
| cold, `pread` per row (page cache, read-ahead on) | 8 / 32 | 5192 / 5200 | 5287 / 5303 |
| cold, `pread` + `FADV_RANDOM` | 8 / 32 | 483 / 547 | 580 / 662 |
| same ids repeated (hot) | 1 / 8 | 191 / 172 | 316 / 263 |

Cold = the layer's table byte ranges dropped with `posix_fadvise(DONTNEED)` before each run, each run a fresh
process (a mapped page cannot be dropped).

### 3c. What the numbers say

1. **Cold, today's code costs ~4.6 s per engram layer per chunk** (~9 s per chunk for both layers). A real new
   prompt's rows are cold unless the tables are resident. The 145 ms in the model profile is the hot best case.
2. The cause is **read-ahead amplification**: each random 256-byte row faults in up to 256 KB (read_ahead_kb).
   122,880 x 256 KB ~ 30 GB to use 32 MB; 30 GB / 4.5 s ~ 6.9 GB/s = the drive's sequential limit. Bandwidth-bound,
   so threads do nothing (4566 vs 4518 ms).
3. **`madvise(MADV_RANDOM)` on the mapping** (or `posix_fadvise(FADV_RANDOM)` for `pread`) removes it: 4.6 s ->
   0.79 s single-threaded, **0.48 s with 8 threads**. No root needed. Rows unchanged.
4. Beyond 8 threads nothing improves (page-fault path / Python loop). 122,880 random 4 KB reads at the ~1M IOPS such a
   drive does at high queue depth would take ~0.12 s: io_uring / `O_DIRECT` batched reads, or keeping the tables in
   RAM, are the next steps.
5. The hot numbers in 3b (172-191 ms gather) disagree with the in-model 3a (~22 ms per layer). The bench ran at nice
   19 on a subset of cores beside a device gate, and `--same-ids` has no warm-up run before the median. **Re-measure
   hot on an idle host** before trusting either.
6. `hash_ms` of 45-75 ms in fresh-process runs is first-call cost (in the model: 1.5 ms for both layers); ignore it in
   cold rows.

### 3d. Update 2026-10-28 10:30: real disk traffic, and the table resident in RAM

Disk counters per gather (`/proc/self/io` read_bytes, major faults; bench fields `gather_disk_mb`,
`gather_major_faults`), layer 1, cold, 5120 tokens:

| mode | gather ms | read from disk | major faults |
|---|---|---|---|
| default read-ahead, 1 / 8 threads | 4480 / 4493 | **23 GB** | 104k |
| `MADV_RANDOM`, 1 / 8 threads | 612 / 384 | **527 MB** (= 135k x 4 KB) | 135k |

- Confirms 3c.2: 23 GB read to use ~32 MB (~700x). With `MADV_RANDOM` exactly one 4 KB page a fault.
- The "1 thread" rows are not single-threaded: torch spreads the fancy-indexing gather over its intra-op pool
  (here 32 cores), so the cold reads already ran in parallel. That is why the Python threads only add ~1.6x.

**Table resident in RAM** (one layer, 94.4 GiB copied once into anonymous memory with `MADV_HUGEPAGE`,
`engram_bench.load_resident`; 93 s at 1 GiB/s here, at nice 19 beside a device gate; the NVMe reads ~6.9 GB/s, so
~30 s for both tables on an idle host), gather per layer per 5120-token chunk, rows byte-equal:

| torch intra-op threads | Python threads | gather ms (median of 5) |
|---|---|---|
| 1 | 1 | 92.1 |
| 1 | 8 | 25.4 |
| 8 | 1 | 13.8 |
| 8 | 8 | **10.0** |
| 32 | 1 | 174.1 (oversubscribed: set thread counts explicitly) |

RAM: both engram layers need **~189 GiB** (fp8 rows 2 x 91.6 + scales 2 x 2.9). The format is already the
checkpoint's 1-byte fp8 + e8m0 scale per 32; smaller (fp4) would change the numbers (owner rule: shipped precision
only); dequantized bf16 would be ~378 GiB. The Galaxy host has 566 GB (446 GB available with the model running): it
fits. **Check the 2x2 host's RAM before choosing this there.** The tables would live in the tt-d-gen runner process,
next to its KV transfer buffers.

Consequence for the plan: where the RAM exists, a resident table (load once at model load, multi-threaded gather,
huge pages) replaces stages 1/1b; the remaining engram host cost is then dequant + assembly + upload (~80 ms a
chunk), which stages 2-4 address.

### 3e. The 2x2 QuietBox host (2x2 session, 2026-10-28)

Host: 503 GB RAM (~339 GB available), one Samsung 990 EVO Plus 4 TB NVMe (consumer, no LVM), **read_ahead_kb = 128**,
AMD EPYC 8124P 16 cores / 32 threads, sudo, THP `madvise`, idle (load 0.35). Full engram tables of rev 5254448
(`[384006168, 256]`, `[384016682, 256]`; 4 - 1 n-grams x 8 heads = 24 ids a token). Layer 1, 5120 tokens, rows
byte-equal in every run. Logs: `/localdev/dnijemcevic/bringup/engram_2026-10-28/`.

| mode | threads | gather ms | read from disk | Galaxy |
|---|---|---|---|---|
| cold, mmap (today's code) | 1 / 8 / 32 | 7279 / 7174 / 7403 | 16.9 GB | 4566 / 4518 |
| cold, mmap + `MADV_RANDOM` | 1 / 8 / 16 / 32 | 1265 / **424** / 435 / 484 | 0.95 GB | 792 / 484 / - / 545 |
| cold, `pread` + `FADV_RANDOM` | 8 / 16 / 32 | 425 / 433 / 475 | | 483 / - / 547 |
| hot, same ids (idle host, 7 runs) | 1 / 8 / 16 | **9.9** / 19.4 / 24.8 | 0 | 191 / 172 (busy host) |

- 3c.5 settled: hot, the gather is ~10 ms a layer (the Galaxy's 172-191 ms was its busy host); extra threads only
  cost when the pages are hot. Hot, what remains per layer is the fp32 dequant (~28 ms at torch's default threads)
  and the bf16 assembly (~12 ms).
- Resident (one layer, `load_resident`, 16 readers): load **109 s** (0.93 GB/s: the copy, not the drive, limits it;
  the Galaxy also got ~1 GiB/s). Gather median of 6 by torch intra-op x Python threads: 1x1 184 ms, 1x8 21.6, 4x8
  12.9, 8x1 15.3, **8x8 8.9**, 16x1 11.2, 16x8 8.8. The fp32 dequant depends on torch's threads: 1 / 4 / 8 / 16 ->
  322 / 85 / 53 / 42 ms (stage 4's lookup-table dequant removes it).

In the model (`tt/engram.py`, settings `ENGRAM_TABLE=mmap|random|resident`, `ENGRAM_GATHER_THREADS`; defaults today's
mmap / 1 until the owner picks): `tests/test_engram_table_modes.py` (CPU) checks the rows byte for byte in every mode
(resident with `DS41_TEST_ENGRAM_RESIDENT=1`); the engram_rows component tests of layers 1 and 14 pass on the 2x2 with
`random` / 8 threads (rel 0, bit-identical repeats).

**bfp8 probe A (API) on the QB: yes** (`tests/ttnn/unit_tests/operations/ds41_engram_bfp8/probes/probe_001.py`, one
chip, 2560 x 6144 real table rows = one mesh row's share of a 5120 chunk). `ttnn.experimental.disaggregation.
tensor_from_bfp8_bytes(raw, shape)` wraps pre-packed bfp8_b TILE bytes as a host tensor (BFLOAT8_B, TILE) with no
float conversion; uploaded, its device values equal `from_torch(dtype=bfloat8_b)`'s bit for bit, and a `ttnn.linear`
on it gives the same output bit for bit. Host time: wrap 11.7 ms + upload 1.1 ms vs `from_torch(bfloat8_b)` 69.3 +
6.6 ms (16.7 MB = 1.0625 B a value vs bf16's 2). The tile's shared exponent covers 16 consecutive values of one
token row, aligned to 16, so a 256-value table row is 16 whole groups (the byte-shuffle assembly of D holds). Not yet
tested: per-device shards of a mesh tensor from such host tensors. Preview of B on these rows: bfp8 changes 10.0 % of
the values, rel L2 3.5e-3 vs the exact rows; a format change, so C (accuracy) and the owner decide.

## 4. Plan (owner-agreed staging; each stage = its own setting in `tt/settings.py`, its own commit, its own A/B)

Invariant for every stage: **the device rows stay bit-identical to today's** (check by byte comparison, not a
tolerance), so the model's accuracy cannot move.

| stage | setting (proposed) | what | where it can be done |
|---|---|---|---|
| 0 | - | bench + byte-equality harness (done: `tools/engram_bench.py`) | anywhere (CPU/NVMe) |
| 1 | `ENGRAM_GATHER_THREADS` + `MADV_RANDOM` | no read-around on the table mapping; gather split over N threads | **2x2 host** (CPU/NVMe) |
| 1b | (same) | beyond 0.48 s cold: io_uring or `O_DIRECT` batched reads into a pinned buffer; or tables resident in RAM (`MADV_WILLNEED` at load / mlock), if RAM allows | **2x2 host** |
| 2 | `ENGRAM_UPLOAD_SPLIT` | upload once per mesh row (not replicated over TP), all-gather over TP on the device | develop on 2x2 (TP 2), measure on Galaxy (TP 4) |
| 3 | `ENGRAM_DEVICE_DEQUANT` | upload fp8 bytes + e8m0 scales (half the bytes), one device op fp8 x scale -> bf16; removes the host float conversion and the 63 MB assembly | develop on 2x2 (op correctness), measure on Galaxy |
| 4 | `ENGRAM_ROWS_ASYNC` | `begin_chunk` starts a worker and returns; between layer dispatches poll it and enqueue a **non-blocking** host->device write into a buffer preallocated at load (same command queue: ordering guarantees the rows land before the layer reads them; the device has idle gaps since the model is host-bound, so the PCIe transfer hides there); sync fallback if not ready at the layer | logic + contract on 2x2 (K.1 contract tests), timing on Galaxy |

Order: stages 1-3 **shrink** the work, stage 4 **hides** what remains. Layer 1 needs its rows ~10 ms after the
chunk starts, so only a small workload can hide there; layer 14's can hide behind layers 0-13. Doing 4 first would
hide a bloated workload and mask the effect of 1-3.

Stage 1 is mostly a change to `reference/weights.py` `EngramTable` (shared with the CPU reference). Keep the
reference's numerics; a performance path may live in `tt/engram.py` instead if the reference must stay as it is.

## 4a. Proposed final design (2026-10-28; owner to close the open items) and work split

Owner decision: the engram work is done on the 2x2 and then ported to the Galaxy; the Galaxy meanwhile implements the
other host-time picks (P.1-P.5: prepared reduce-scatters, prepared fused tail, per-layer prepared fused head,
persistent CCL outputs, one-program RoPE).

| decision | proposal | alternative |
|---|---|---|
| table storage | resident in RAM: anonymous memory, `MADV_HUGEPAGE`, loaded once at model load (~189 GiB, ~30 s); setting `ENGRAM_TABLE=resident\|mmap\|auto` (auto = resident when RAM allows) | mmap + `MADV_RANDOM` + threads (small hosts) |
| lookup | multi-threaded gather, explicit thread counts (torch 8 x Python 8 measured best, ~10 ms per layer per chunk) | - |
| dequant | on the device: upload fp8 bytes + e8m0 bytes (half of today's bf16), then two 256-entry lookup tables with `ttnn.embedding` (fp8 byte -> bf16 value, e8m0 byte -> bf16 power of two) and one multiply; exact, no fork needed | host dequant as today |
| upload | split over TP, all-gather over TP on the device | replicated as today |
| timing | worker started at `begin_chunk`; poll between layer dispatches; non-blocking write on the same command queue into buffers preallocated at load; synchronous fallback at the layer's deadline; within the chunk only (no tt-d-gen change) | also prefetch the next chunk (needs the runner to see its ids) |
| code location | model-only path in `tt/engram.py`; `reference/weights.py` stays the reference | change `EngramTable` itself |
| acceptance | rows bit-identical at every stage; A/B warm and cold; K.1 for the async stage | - |

Open for the owner: (1) ~189 GiB of RAM + ~30 s of load time per runner process on the Galaxy acceptable? (2) device
dequant via the lookup-table trick, or host dequant first? (3) next-chunk prefetch in or out of scope?

### Coordination rules between the 2x2 and the Galaxy sessions
- Number ranges: Galaxy uses DS41-130..149 and F207..F219; the 2x2 uses DS41-150..169 and F220..F229 (two F200 / F201
  collided last time).
- Settings: the engram settings go in their own block in `tt/settings.py` (both sessions append settings there).
- Ownership: the 2x2 owns `tt/engram.py`, `tt/engram_kv.py`, `reference/weights.py`; the Galaxy owns `tt/ccl.py`,
  `models/demos/moe_block`, the RoPE and reduce-scatter forks. A `tt/model.py` change is named in its commit message.
- Git: `git fetch && git rebase origin/dnijemcevic/model_bringup` before every push; never force; small commits.
- After pulling the Galaxy's picks, the 2x2 re-runs its own gates: they change shared paths (CCL, MoE block) it runs.
- The real combined saving exists only after the port, measured on the Galaxy: an engram worker competes for CPU and
  the GIL with the main dispatch thread the picks speed up.

## 4b. Formats today, the RAM path, and the revised order (2026-10-28, supersedes the stage order of 4 / 4a)

### Where 8 -> 16 bit happens today (per chunk, per engram layer, all before layer 0 is dispatched)
1. Host, `EngramHasher.hashes`: token ids -> 24 row ids per token (122,880).
2. Host, `EngramTable.rows`: gather 256 fp8 (e4m3) bytes + 8 e8m0 scale bytes per row (~32 MB, still 8-bit).
3. Host, `EngramTable.rows` + `host_rows`: fp8 -> **fp32** (`.float()`), x 2^scale in fp32, -> **bf16** into a
   fresh `[5120, 6144]` buffer (63 MB). Exact (bf16 holds e4m3 x 2^k).
4. Host, `to_device` (`ttnn.from_torch`): tilize + PCIe, split over mesh rows, replicated over the TP columns.
5. Device, `tt/engram_kv.py`: one `ttnn.linear`, bf16 rows x `wkv` (already **bfp8_b** on the device: setting
   `DENSE_WEIGHTS=bfp8`, converted once at load), fp32 accumulation, fp32 out (4 keys + 1 value).
6. Device, `tt/engram_add.py`: gate + residual add into the 4 streams, all fp32.
The only 8 -> 16 conversion is step 3, on the CPU, every chunk.

### Why the rows are bf16 and not 8-bit
DeepSeek's `fp8_linear` feeds e4m3 into FP8 tensor cores (H800). Tenstorrent's matrix engine has no e4m3 input; its
8-bit format is `bfloat8_b` (one exponent per 16 values, 7 magnitude bits): exact only for values within ~3 binades
of their group's largest. bf16 reproduces the table values exactly, which is why the agents chose it. `bfp8_b` rows
would be fp8-class precision (DeepSeek itself re-quantizes this activation to 8-bit), so they may be within the
reference's own margin; only the ladder decides. The `engram_kv` matmul is 0.58 ms of device time per chunk, so the
format matters for host time and PCIe bytes, not for device time.

### The RAM path: what is in host RAM
- Persistent (loaded once at model load): the table in its **8-bit checkpoint format**, fp8 bytes 91.6 GiB + e8m0
  2.9 GiB per engram layer, ~189 GiB for layers 1 and 14. Never 16-bit. (`engram_bench.load_resident`: byte copy into
  anonymous memory with `MADV_HUGEPAGE`.)
- Transient per chunk and layer, freed after the upload: gathered rows ~32 MB (8-bit); today's fp32 intermediate
  ~126 MB; the bf16 rows 63 MB.
- Per-chunk conversion variants, increasingly lean: (a) today: 8-bit -> fp32 -> bf16 on the host; (b) host
  fp8 -> bf16 directly through a 256-entry lookup table, no fp32 intermediate (63 MB transient); (c) no host
  conversion: upload the ~32 MB of 8-bit bytes and dequant on the device (two `ttnn.embedding` lookup tables + one
  multiply, exact), or pre-packed `bfp8_b` if the API probe below passes.

### Revised order on the QB
| order | step | why | size |
|---|---|---|---|
| 1 | baseline the QB host with the bench: RAM, disk model, `read_ahead_kb`, cold and hot | everything is judged against it | 30 min |
| 2 | fix the I/O: `MADV_RANDOM` + threaded gather, then the resident table if RAM allows | the real serving cost: ~4.5 s per layer per chunk cold today | seconds -> ~10 ms per layer |
| 3 | async: worker at `begin_chunk`, poll between layers, non-blocking write, deadline fallback | takes what remains off the critical path (layer 14 fully, layer 1 partly) | most of ~145 ms per chunk |
| 4 | shrink the conversion and upload: decide by the bfp8 probe between pre-packed `bfp8` and fp8 bytes + device dequant | what async cannot hide (layer 1 is needed ~10 ms in), half the PCIe bytes, less host CPU competing with the dispatch thread for the GIL | ~45-80 ms of host work per chunk |
| 5 | upload split over TP, all-gather on the device | 1/4 of the bytes on the Galaxy (1/2 on the QB) | small; the effect shows on the Galaxy |

### The bfp8 probe (step 4's decision), stop at the first "no"
Storage idea: `bfp8_b` groups are 16 consecutive values of a row and a table row is 256 values, so a row can be stored
pre-packed as 256 mantissa bytes + 16 exponent bytes (272 B, ~ today's 264 B) and a device tile built from gathered
rows by a byte shuffle only (no float math).
| # | experiment | QB? | what the Galaxy adds |
|---|---|---|---|
| A | **API probe** (first): build a `bfp8_b` TILE host tensor from raw pre-packed bytes and upload it, no float conversion. If ttnn cannot, stop here and use fp8 bytes + device dequant | yes (ttnn + one device) | nothing |
| B | offline error stats: pack ~1M random table rows into 16-value groups; fraction of values changed, relative L2 error, vs DeepSeek's own fp8 activation quantization of the same rows | yes (CPU) | nothing |
| C | accuracy: env-switched `typecast(rows -> bfp8_b)` on the device before `engram_kv`; the ladder + smoke vs the baseline; then once with host packing (`from_torch(dtype=bfloat8_b)`) to rule out a rounding-mode difference | yes, indicative: the 2x2 subset (0-3, 14, 20, 21, 24) holds both engram layers | the full 40-layer ladder at the 56k target (error through 40 layers, unscaled model): a short confirmation |
| D | convert the table offline to row-packed bfp8 (~97 GiB per layer), byte-shuffle assembly, timing vs today's ~80 ms of dequant + assembly + upload | yes (disk for ~2 x 97 GiB, ideally the RAM) | final timing: Galaxy host CPU, TP 4, chunk 5120 in the model |
QB notes: benchmark at 5120 tokens (the Galaxy chunk); the full engram tables of rev 5254448; after pulling, the
block-MoE default may not build on the QB's FABRIC_2D (env fallback in section 5).

## 5. 2x2 vs Galaxy: what can differ, and where each piece belongs

### Do on the 2x2 (QuietBox)
- Stages 0, 1, 1b: host I/O work. No device needed. Use the **full** engram tables (the other session says it has
  them) and `--tokens 5120` (the Galaxy chunk), not the 2x2's 2048, so the numbers compare.
- Functional development of stages 2-4 (the device op for stage 3, the async/poll logic of stage 4) and their
  correctness checks: the byte-equality tests, the component/contract tests that exist on the 2x2.

### Leave for the Galaxy
- The A/B measurements that decide the picks (orchestrator perf tasks: frozen tests, ladder, warm wall + device
  profile), stage 2's real effect (TP 4), stage 4's timing, K.1 on the torus.

### What may go wrong between the two
1. **The host is different.** RAM (the two tables are ~189 GiB: whether they fit in page cache changes "hot" vs
   "cold" completely), the disk (model, NVMe or not, RAID/LVM), `read_ahead_kb` (`cat /sys/block/<dev>/queue/
   read_ahead_kb`; it decides the amplification in 3c.2), CPU count, sudo. Record all of them next to every number.
2. **Cold must really be cold.** `posix_fadvise(DONTNEED)` does not drop pages that any process still maps; the bench
   uses a fresh subprocess per cold run for that reason. If the box has spare RAM and the table was just read, a
   "cold" run can still be warm: check a cold run is slow without `MADV_RANDOM` first (it should be seconds).
3. **Checkpoint identity.** The 2x2 bring-up runs the **scaled** model (fewer experts and heads, chunk 2048), but the
   engram layers (1, 14) and their tables must be the full ones of rev 5254448 for the numbers to transfer. Check
   the table shapes `[384006168, 256]` / `[384016682, 256]` and `max_ngram_size`, `n_heads` (24 ids a token).
4. **Mesh and fabric.** 2x2 = TP 2 (upload replication 2x, not 4x), FABRIC_2D (always 2D on the QB, owner). The
   Galaxy now runs the torus. Stage 2's saving is about half on the 2x2.
5. **The MoE default changed.** After FIX.6, `MOE_IMPL=block` is the default everywhere. Block was A/B'd on the 2x2
   by the owner, but its fused head did **not** build on the 8x4's FABRIC_2D lines (DS41-129, a 16-slot forwarding
   limit in `routing_stream.py:612`). If a 2x2 device run fails at MoE build after pulling, that is why:
   `DS41_MOE_IMPL=flat` (or `DS41_MOE_HEAD=ops`) for engram-only work, and report it; do not change the default.
6. **Chunk size.** The 2x2 runs 2048-token chunks: in-model engram numbers there are for 2048 tokens (~40 %).
   Compare per token, or use the bench at 5120.
7. **Page cache sharing with model loads.** A device run maps hundreds of GB of weights; the engram pages compete
   with them for the page cache. Measure the engram path both alone and inside a model run.
8. **Seed-box re-verify debt** (unrelated to engram, but the 2x2 is where it gets paid): FIX.1, FIX.2, FIX.5, the W.1
   fabric change, `max_y_writers`, `TP_SUM_BLOCKS=cols` (DS41-128), the contract geometry changes. List in
   `bringup/ledgers/galaxy_8x4-full/supervision.md` ("SEED-BOX RE-VERIFY").

## 6. How to run the bench

```
export PYTHONPATH=$PWD; source python_env/bin/activate
H=<HF checkpoint dir with model-00047/48>
# cold, today's code vs no read-around, 1/8/32 threads, rows checked
python -m models.demos.deepseek_v41_flash_d_p.tools.engram_bench --ckpt $H --layers 1 --threads 1,8,32 --repeats 2 --cold --check
python -m models.demos.deepseek_v41_flash_d_p.tools.engram_bench --ckpt $H --layers 1 --threads 1,8,32 --repeats 2 --cold --madv-random --check
# pread variant
python -m models.demos.deepseek_v41_flash_d_p.tools.engram_bench --ckpt $H --layers 1 --threads 8,32 --repeats 2 --cold --io pread --madv-random --check
# hot (repeat the same ids; re-measure on an idle host, see 3c.5)
python -m models.demos.deepseek_v41_flash_d_p.tools.engram_bench --ckpt $H --layers 1 --threads 1,8 --repeats 5 --same-ids --check
```

Prints one JSON line per run (hash / gather / dequant / assemble ms, lookups, bytes) and a median table.

## 7. Where things are

- This file: `models/demos/deepseek_v41_flash_d_p/bringup/ENGRAM_HOST_PATH.md`
- Bench: `models/demos/deepseek_v41_flash_d_p/tools/engram_bench.py`
- Engram code: `tt/engram.py`, `reference/weights.py` (`EngramTable`), `reference/ngram.py` (`EngramHasher`),
  `tt/model.py` (`begin_chunk`)
- Perf proposals (whole host path, block on torus): `bringup/ledgers/galaxy_8x4-full/perf_proposals.md`
- The run's record: `bringup/ledgers/galaxy_8x4-full/supervision.md` (2026-10-27/28 entries)
