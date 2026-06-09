# Theory of NOC Alignment in TT-Metal

This document is the *general theory* behind NOC alignment in TT-Metal, written for agents and engineers who need to reason about **any** data-movement kernel, not a specific op. The companion doc `NOC_ALIGNMENT_IN_TT_METAL.md` is a more concrete walkthrough (and includes an extended conv-centric case study). This one tries to extract the rules you can apply anywhere.

Scope: software behavior. Hardware micro-architecture (DRAM controller internals, flit-level bandwidth accounting, NIU pipeline stalls, posted-write reordering) is deliberately out of scope. Where software behavior *depends* on a hardware rule, the rule is stated as an axiom with a citation to the header that declares it.

Architecture-specific constants are parameterized. The two values that matter are:

- `A_L1`  — NOC alignment for traffic terminating at an L1 endpoint (Tensix or Eth core)
- `A_DRAM` — NOC alignment for traffic *sourced from or targeting* DRAM

You should think in symbols, then substitute. On current silicon the values are:

| Arch | `A_L1` | `A_DRAM` |
|------|-------:|---------:|
| Wormhole | 16 B | 32 B |
| Blackhole | 16 B | 64 B |

Sources:
- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h:290`
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374`
- `tt_metal/impl/allocator/l1_banking_allocator.cpp:216`

> Mental hygiene: never write "16" or "32" in kernel or host code when you mean an alignment. Use `hal::get_l1_alignment()` / `hal::get_dram_alignment()` or the per-buffer `buffer.alignment()`. Hard-coded alignment constants are the single biggest source of cross-arch bugs in this codebase.

---

## 1. Axioms

These are the rules the rest of the theory derives from. Each one is a *correctness* constraint — the NOC rejects non-conforming transactions; they are not perf knobs.

### A1. Address alignment is a NOC legality rule. Size alignment is weaker and silent.

A NOC transaction must have:

- source address aligned to the source-endpoint alignment,
- destination address aligned to the destination-endpoint alignment.

Size alignment is a softer story (see §4.4 for the empirical picture). The hardware does not reject every non-aligned size the way it rejects non-aligned addresses — short or odd sizes often run and just produce wrong data. Treat size alignment as a **correctness invariant your kernel must maintain**, not a guarantee the hardware will enforce.

In practice, TT-Metal collapses "source alignment" and "destination alignment" into a single *pair alignment* that you look up by `(src_type, dst_type)` (§2). If you violate **address** alignment, the transaction is illegal — not slow, not padded, not forgiven.

### A2. Alignment depends on the endpoints, not on the initiator.

The NIU issuing the request is irrelevant. What matters is where the bytes *live* and where they are *going*. A Tensix writing to DRAM pays DRAM alignment on the DRAM side; a Tensix writing to another Tensix pays only L1 alignment. You cannot get cheaper alignment by picking a different initiator.

### A3. L1 alignment is symmetric; DRAM alignment is asymmetric, and the asymmetry is observable.

For current arches `A_L1` is identical for read and write. `A_DRAM` is not: the *read* side is stricter than the *write* side at the hardware level. On Wormhole, DRAM read requires 32 B alignment; DRAM write accepts 16 B.

TT-Metal's allocator collapses this asymmetry by taking `max(read_align, write_align)` and exposing only one alignment per memory type (see `buffer.cpp:560` and `l1_banking_allocator.cpp:216`). That collapse is deliberate — software treats DRAM alignment as a single value, and any buffer's `alignment()` is `A_DRAM_read`.

But the asymmetry is still **observable** once you bypass the allocator. A raw `noc_async_write` to a 16-aligned DRAM destination is legal and passes the sanitizer; a raw `noc_async_read` from a 16-aligned DRAM source fails the sanitizer with `"invalid address alignment in NOC transaction"`. This is empirically verified (see §11). For any code you don't fully own the alignment of, treat the *read* side as load-bearing.

### A4. The allocator already picked the alignment for you.

`buffer.alignment()` is the authoritative number for any allocated buffer. It is chosen at buffer construction based on the buffer's type and where it lives. If you have a `Buffer*`, stop reasoning about arch constants and use `buffer->alignment()`.

- `tt_metal/impl/buffers/buffer.cpp:552`
- `tt_metal/impl/buffers/buffer.cpp:560`

### A5. Pages are the NOC's unit of abstraction.

Page-based APIs (`noc_async_read_page`, `TensorAccessor::noc_async_read_page`, etc.) guarantee alignment *per page*. If your kernel moves data one page at a time, and the accessor was constructed correctly on the host, NOC alignment is the allocator's problem, not yours. Violations almost always come from kernels that stop using page APIs and do raw `noc_async_read` / `noc_async_write` with hand-computed addresses.

---

## 2. The Endpoint Pair Rule

NOC alignment is a function of **where the bytes are at each end of the wire**. The allocator uses this table when choosing a buffer's alignment (`l1_banking_allocator.cpp:216`):

```
            dst
            L1         DRAM
src   L1    A_L1       A_L1         (Tensix/Eth → Tensix/Eth or → DRAM: L1 alignment)
      DRAM  A_DRAM     A_DRAM       (DRAM → Tensix/Eth: DRAM alignment rules)
```

Two non-obvious consequences:

1. **Tensix-to-DRAM writes pay only `A_L1` on the source side.** The DRAM-side write alignment is also relaxed in hardware, which is why the collapsed `A_DRAM` is dominated by the read side. Software-visible effect: a DRAM buffer's `alignment()` is `A_DRAM`, because the allocator must make the *read* case legal.

2. **A buffer's alignment is a property of the buffer, not of a specific transaction.** A DRAM buffer is always aligned to `A_DRAM`, even if you only ever write to it. This is safe pessimism; don't try to "save space" by undercutting it.

---

## 3. The Buffer Contract: `page_size` vs `aligned_page_size`

This is the core TT-Metal abstraction.

```
buffer.page_size()          // logical page size — what the tensor schema says
buffer.alignment()          // correctness floor for this buffer type
buffer.aligned_page_size()  // = align_up(page_size(), alignment())
```

Adjacent pages in a device buffer are spaced by `aligned_page_size`, **not** `page_size`. So if `page_size = 14` and `alignment = 16`, page N starts at offset `16*N`, and there are 2 "wasted" bytes of padding between each pair of logical pages.

```
┌────────────┬──┬────────────┬──┬────────────┬──┐
│  page 0    │pd│  page 1    │pd│  page 2    │pd│  ...
│  14 bytes  │2 │  14 bytes  │2 │  14 bytes  │2 │
└────────────┴──┴────────────┴──┴────────────┴──┘
 └──── aligned_page_size = 16 ────┘
```

Key invariants:

- Page-address arithmetic in the NOC address generators uses `aligned_page_size` as the stride (`dataflow_api.h:1052`, `tensor_accessor.h:71,96`).
- `page_size == aligned_page_size` is the "nothing to think about" case. Most tiled tensors land here (a tile's page size is already a multiple of `A_L1` and `A_DRAM`).
- `page_size < aligned_page_size` is the hard case. Row-major tensors live here whenever the logical row width in bytes is not a multiple of the buffer's alignment.
- `page_size > aligned_page_size` cannot happen — `aligned_page_size` rounds up.

### When the pad bytes contain garbage

The pad region between logical pages holds *whatever was there before*. It is not zeroed. Kernels that compute across the pad — e.g. a row-major reduction that is handed `aligned_page_size` bytes instead of `page_size` bytes — will silently fold garbage into the result. This is the single most common row-major correctness bug in TT-Metal. Always reason about which stride your compute uses, and whether the extra lanes are meaningful payload.

---

## 4. The Transaction Model

Every NOC transaction reduces to four numbers:

```
  src_noc_addr   size   dst_local_addr   noc_cmd
```

Legality requires:

- `src_noc_addr % A_src == 0`
- `dst_local_addr % A_dst == 0`
- `size % max(A_src, A_dst) == 0`  (conservative; see A3)

### 4.1 Three flavors of NOC API

At the kernel level, every NOC call lives in one of three tiers, distinguished by how much alignment they handle for you:

**Tier 1 — Page APIs.** `noc_async_read_page`, `noc_async_write_page`, `TensorAccessor::noc_async_read_page` (`dataflow_api.h:1052`, `tensor_accessor.h:71,96`). These consume a page ID plus an optional offset. The address generator materializes the NOC address using `aligned_page_size`, so *page-to-page* alignment is correct by construction. You are still responsible for alignment *within* a page if you use the `offset` parameter.

**Tier 2 — Raw NOC APIs with explicit address and size.** `noc_async_read`, `noc_async_write`, `noc_async_read_one_packet`, etc. These do no alignment for you. They are the right tool when you know the geometry and need to move data that doesn't fit the page abstraction. You own the alignment problem.

**Tier 3 — Semaphore and inline APIs.** `noc_semaphore_*`, `noc_inline_dw_write`. These move scalar control state (typically 4 bytes). They have their own alignment rules (usually 4-byte word-aligned), and they are not bandwidth-relevant, so we don't cover them here beyond "use them only for control, not for bulk data."

### 4.2 Fragmentation

Any transaction larger than `NOC_MAX_BURST_SIZE` (arch-specific: 8 KB on Wormhole, 16 KB on Blackhole) is split into multiple packets by the low-level firmware. This happens *inside* `noc_async_read` / `noc_async_write` and is invisible to the caller — but every fragment must independently satisfy the alignment rules. If you pass a legal top-level `(addr, size)`, each fragment is legal by construction, because the firmware only splits at burst-size boundaries and the burst size is itself a multiple of every alignment.

### 4.3 What "illegal" means

The NOC refuses address-misaligned transactions. At the kernel level that surfaces through the watcher / NOC sanitizer (`TT_METAL_WATCHER=1`, which is what `scripts/run_safe_pytest.sh --dev` enables). When a transaction violates address alignment, the sanitizer stops the device and prints, for example:

```
Device 0 worker core(...): BRISC using noc1 tried to unicast write 128 bytes from
local L1[0x019520] to DRAM core w/ virtual coords (x=9,y=0) DRAM[addr=0x001d7721]
(invalid address alignment in NOC transaction).
```

Without the watcher, the same transaction is undefined behavior: typically a hang or silently corrupt data, never a "slow but correct" fallback. The only engineering response to "my addresses might be unaligned" is to ensure they aren't.

### 4.4 What is NOT caught automatically

The sanitizer is not a complete alignment checker. A kernel can produce wrong results without triggering any watcher warning. Empirically (see §11):

- **Short-but-legally-aligned size**: e.g. writing 96 bytes instead of 128 bytes from L1 to DRAM (96 is a multiple of `A_L1` and `A_DRAM_write`). Sanitizer silent, output tensor is garbage in the un-written tail of each page.
- **Aligned-but-shifted address**: e.g. writing to a DRAM destination at `base + 16` instead of `base` (still 16-aligned, which is legal for DRAM writes). Sanitizer silent, data lands at the wrong place.
- **Reader size not matching DRAM-read alignment**: e.g. reading 16 bytes from DRAM into L1 (16 is `A_L1` but not `A_DRAM_read=32`). Sanitizer silent on the size mismatch, correctness fails. The read size is **not** validated against `A_DRAM_read` the way an *address* is.

What **is** caught by the watcher on top of NOC-address alignment:

- **CB overflow**: if an L1 source or destination address plus the transaction size steps past the end of a circular-buffer page. Error string: `"NOC transaction overflows a circular buffer"`. This is a separate bounds check, not an alignment check.

Rule: the sanitizer catches *illegal-to-the-NOC* and *out-of-bounds-in-L1-CB* cases. It does not catch *wrong-but-legal* transfers. Kernel correctness is still your responsibility.

---

## 5. Row-Major as the Canonical Problem Case

Tile layouts make alignment trivial: a BF16 tile is 32×32 BF16 = 2048 bytes, a multiple of every alignment on every arch. Sharding, interleaving, and per-page accounting all just work.

Row-major is where the work is. A single row-major "page" (sometimes called a *stick*) has size:

```
page_size = row_width_elements * element_size_bytes
```

which can be arbitrarily small. The hardware doesn't care how small your logical row is; it cares about the bytes that physically cross the NOC. Three regimes:

```
regime A:  page_size is a multiple of A_dst
           → page_size == aligned_page_size
           → no wasted padding, raw NOC ops are safe if addresses agree
           → this is the target state

regime B:  page_size is not a multiple of A_dst
           → page_size < aligned_page_size
           → the buffer has per-page pad bytes
           → kernels must step by aligned_page_size, not page_size
           → raw NOC ops on page_size bytes are ILLEGAL (A1)
           → page APIs are SAFE because they step at aligned stride

regime C:  page_size < A_dst
           → even a single page is shorter than a legal NOC transaction
           → pure reject territory for raw NOC ops
           → page APIs still work (they transfer aligned_page_size ≥ A_dst bytes)
```

The width thresholds that matter in practice for BF16 (2 bytes/elem):

- 8 channels  = 16 bytes  → minimally L1-safe (meets `A_L1 = 16`)
- 16 channels = 32 bytes  → meets Wormhole DRAM read (`A_DRAM = 32`)
- 32 channels = 64 bytes  → meets Blackhole DRAM read (`A_DRAM = 64`)

For FP32 (4 bytes/elem), halve the thresholds in channels. For BFP8 / INT8, quadruple them.

### 5.1 Three strategies for row-major

Every row-major op in the codebase does one of these three things — there is no fourth:

**S1: Align the data.** Pad the logical shape on the host so the row width in bytes is a multiple of the target alignment. The kernel sees a "fat" tensor and moves it normally. Bytes of padding are meaningful only to the layout; the compute ignores them (or treats them as zero if the math tolerates it).

**S2: Align the stride only.** Keep the logical shape the same but allocate with `aligned_page_size` spacing (the default buffer contract already does this). The kernel must use `aligned_page_size` as its stride for every NOC transfer. Compute kernels that read/write by page ID through `TensorAccessor` get this for free; kernels using raw NOC ops must carry a separate variable for logical vs physical row width.

**S3: Reject.** The op validates at host-launch time that `page_size == aligned_page_size` (or `page_size % alignment == 0`) and errors out otherwise. The caller is responsible for padding upstream.

Rule of thumb for choosing: S1 when the pad bytes are harmless (e.g. channel padding for image-shaped tensors where downstream ops already slice the real channels), S2 when the pad would contaminate compute (so it stays outside the logical region), S3 when the kernel is too hot or too hairy to make either work.

---

## 6. Sharded Buffers and Alignment

Sharded buffers add one more layer: the *shard* is a contiguous region of a single core's L1 (or a single DRAM bank), and each shard contains an integer number of pages.

The invariants the allocator enforces:

- `shard_size_bytes % page_size == 0` (`buffer.cpp:77-80`) — shards are whole-page.
- `page_size` in a sharded buffer is already the "aligned" page size for alignment-aware shards. A sharded buffer has no per-page pad gap inside a shard; the pad moves to the end of the shard if it exists at all.
- The shard's start address is aligned to the buffer's alignment, because the allocator picks it that way.

Practical consequence: in a sharded row-major buffer, the *stick width* is effectively `page_size`, and the kernel can loop at that stride without a separate `aligned_stick_bytes`. This is why many sharded ops gate on `page_size == aligned_page_size` up front (S3 from §5.1) — it lets the kernel assume no padding at all.

When it doesn't hold, the op either pads the shard to be alignment-friendly (S1) or uses an auxiliary aligned-scratch buffer in L1 and copies through it (S2-variant; see `fold` and `pad_multi_core` for clean examples in the existing codebase).

### 6.1 Why cross-shard boundaries are not a new rule

A NOC read that spans two shards on two different cores is actually two independent NOC transactions, issued one page at a time. The per-page alignment rules apply. There is no "inter-shard alignment" concept — the NOC sees only (source_addr, dest_addr, size) per transaction, and the allocator already made each page's start address aligned.

---

## 7. Performance Theory at the Software Level

Hardware-level NOC performance (flit efficiency, wire bandwidth, NIU scheduling) is out of scope. What *is* in scope: the software-visible patterns that make kernels fast or slow given the alignment rules above.

### 7.1 Transaction count matters more than alignment overhead

Each NOC transaction has fixed per-transaction cost (header, NIU command buffer setup, issue latency). Within legal alignment, **fewer larger transactions beat many smaller ones.** If you have a choice between:

- N reads of `aligned_page_size` bytes each,
- 1 read of `N * aligned_page_size` bytes,

the single read is strictly faster, assuming the source region is contiguous and the destination is contiguous. This is why tile layouts dominate row-major layouts at high throughput: a tile packs 1 KB–2 KB of data into one transaction, while row-major may need one transaction per row.

### 7.2 Contiguity is a software choice

Two pages that are logically adjacent in the tensor are contiguous in memory only if:

- the buffer is interleaved **and** they live on the same bank (rare — interleaving deliberately spreads them), or
- the buffer is sharded **and** they are in the same shard, or
- the buffer is a single contiguous allocation and the layout doesn't stripe them.

Kernels that assume adjacency without checking end up either issuing one NOC call per page (slow) or issuing one large call that crosses a bank boundary (illegal, §Q6). The safe pattern is: use `TensorAccessor` and page APIs; let the accessor decide which pages can be coalesced.

### 7.3 NOC0 vs NOC1 for DRAM (the split-reader footgun)

Every Tensix core exposes two independent NIUs: NOC0 and NOC1. They are nominally symmetric, so kernels that want to double their issue rate use both in parallel (the "split reader" pattern).

For **L1 ↔ L1** traffic, this is straightforward: both NOCs route through the mesh and the two halves add up. For **DRAM traffic, NOC0 and NOC1 are not symmetric.** The physical path from any given Tensix to the DRAM controllers is better along one NOC for reads and along the other for writes; which one is which depends on the core's position in the grid and is not something a kernel author should solve themselves.

Software rule:

- For DRAM-bound work, do **not** use split-reader to parallelize across NOCs. Use the default reader/writer kernel configuration, which already picks the right NOC for each direction.
- Split-reader is a good optimization only when both halves of the work are L1-to-L1 (or more generally, when DRAM is not on the path for either half).
- If you are DRAM bandwidth-bound, more NIU issuers won't help. Increase the *size* of each DRAM transaction (§7.1) or increase parallelism *across cores* instead.

### 7.4 NIU saturation

Each NOC has a finite number of outstanding transactions it can track (bounded by the transaction-ID space and the NIU's command buffers). Pushing past that limit stalls the issuing kernel until some transactions drain. Two software implications:

- **Batched issue is not free.** You cannot "fire a million `noc_async_read`s in a loop and then one barrier." Somewhere in that loop you'll wall into NIU saturation and effectively serialize on the slowest in-flight transaction. If you need thousands of transfers, stage them: issue a bounded window, barrier partially, issue more.
- **Semaphore- and barrier-driven back-pressure is the normal mechanism.** The `cb_reserve_back` / `cb_push_back` pattern in circular buffers already does this for you; custom data movement that bypasses CBs must reinvent it.

### 7.5 Small transactions waste more than just bandwidth

Even though the hardware-level flit efficiency is out of scope, one software consequence matters: when a kernel issues many small transactions, it spends more time in *issue code* (filling the NIU command buffer) and less time in *wait code*. On a reader kernel, that's directly visible as "Tensix time in reader" dominating over "Tensix time waiting for NOC." If you're profiling a reader and the wait is short, the fix is almost never a faster NOC — it's a larger per-transaction size.

### 7.6 Alignment padding is not a perf problem

The padding between pages in a row-major buffer with `page_size != aligned_page_size` is at most `alignment - 1` bytes per page, which on L1 is ≤15 bytes. The "wasted" bandwidth is real but almost always irrelevant. Don't contort your layout to eliminate it; contort your layout only when the logical row is *shorter than `A_dst`*, because that crosses from regime B into regime C and becomes a correctness problem (§5).

---

## 8. Decision Procedure for Kernel Authors

When writing a new kernel that touches NOC, answer these in order:

1. **What are the endpoint types?** L1↔L1, L1↔DRAM, DRAM↔L1, DRAM↔DRAM? This gives you the alignment pair from §2.

2. **Does the buffer(s) go through a `Buffer*` allocation?** If yes, use `buffer->alignment()` and `buffer->aligned_page_size()`. Do not hard-code numbers.

3. **Can the work be expressed page-by-page through a `TensorAccessor`?** If yes, use page APIs (Tier 1). Alignment is handled for you. Skip to step 6.

4. **Can it, but the page APIs are too coarse-grained?** Check whether you need raw NOC APIs (Tier 2) at all. Often the right fix is "issue larger pages," not "bypass page APIs."

5. **If you must use raw NOC APIs:**
   - carry `logical_size_bytes` and `physical_stride_bytes` as separate variables,
   - every call must pass a size that is a multiple of the destination's alignment,
   - every address must be pre-aligned to the destination's alignment,
   - if your input can be sub-alignment, either reject it at host-side validation or stage through an aligned L1 scratch region.

6. **Are you DRAM-bound and tempted to use split-reader?** Don't. Fix DRAM bandwidth with larger per-transaction size or more cores, not more NIUs (§7.3).

7. **Are you processing row-major data?** Check which of the three regimes in §5 you're in and pick S1, S2, or S3 accordingly. Document the choice at the top of the kernel — future readers will need it.

---

## 9. Invariants and Red Flags

Things to check before believing a kernel is correct:

- **`page_size` used as a stride in NOC address arithmetic** (instead of `aligned_page_size`). Red flag: silent row-major corruption in regime B.
- **Raw `noc_async_read` / `noc_async_write` with a hard-coded size constant** (e.g. `16`, `32`). Red flag: works on one arch, breaks on another.
- **Kernel dereferences `(char*)page_ptr + page_size * i`** to compute the start of page `i`. Red flag: same as above.
- **Hard-coded alignment in host code.** Red flag: any numeric literal from the set {16, 32, 64} near a buffer allocation, shard spec, or tensor layout calculation.
- **`noc_async_read` of a size *that is not a multiple of the source endpoint's read alignment*.** Red flag: silently wrong data — the sanitizer does **not** catch size mismatches, only address misalignment. Reader correctness is your responsibility.
- **Assumption that "adjacent pages are contiguous in memory"** in an interleaved buffer. Red flag: they are not; interleaving stripes them across banks.
- **Split-reader over a DRAM-bound path** (§7.3). Red flag: no perf gain, possibly worse than single-reader due to asymmetric NOC paths.
- **"It works on Wormhole"** as the correctness argument. Red flag: `A_DRAM` is 32 on WH and 64 on BH. A path that is 32-byte aligned is WH-legal but BH-illegal.

---

## 10. What This Theory Does Not Tell You

Scoped out, intentionally:

- Flit-level bandwidth accounting and the cost of sub-word transactions. Software-irrelevant.
- Read/write DRAM-alignment asymmetry at the DRAM-controller level. Collapsed by the allocator; treat DRAM as a single alignment.
- Posted vs. non-posted write semantics and their ordering guarantees. Control-flow detail; kernels use the defaults.
- Inline-write hardware corner cases on Blackhole. Covered by the `noc_inline_dw_write` wrappers; do not reinvent.
- NOC virtual channels and VC-aware routing. Internal to the NIU firmware.
- Per-core grid topology and NOC distance cost. Performance-tuning detail, not a correctness concern.

If a kernel ever needs to reason about these, the theory in this doc is not sufficient — at that point you need hardware documentation or a performance-specialist review.

---

## 11. Empirical Grounding

The rules in this document are checked by perturbation experiments on `test_upsample.py::test_upsample_nearest_interleaved` (a row-major interleaved path with `noc_async_read` + `noc_async_write` calls whose addresses and sizes can be poisoned by hand). Runner: `scripts/run_safe_pytest.sh --dev` (watcher on, NOC sanitizer on, ebreak asserts on). Device: Wormhole (`A_L1 = 16`, `A_DRAM_read = 32`, `A_DRAM_write = 16`). Test shape `[1, 64, 32, 32]`, scale 2×2, BF16 → `page_size = 128 B`.

Each row perturbs **one** argument at the `noc_async_read` / `noc_async_write` call site; everything else is unchanged.

| # | Site | Perturbation | Outcome | Watcher message |
|---|---|---|---|---|
| 0 | — | baseline | PASS | — |
| 1 | writer (L1→DRAM) | `size=96` (mult of 16 & 32, short) | correctness fail | *silent* |
| 2 | writer | `size=127` (odd) | PASS | *silent* (size evidently rounded or tolerated) |
| 3 | writer | `size=129` (overflows 128 B CB page) | caught | `NOC transaction overflows a circular buffer` |
| 4 | writer | `dst_noc_addr + 1` (misaligned DRAM dst) | caught | `invalid address alignment in NOC transaction` |
| 5 | writer | `read_addr + 1` (misaligned L1 src) | caught | `invalid address alignment in NOC transaction` |
| 7 | writer | `read_addr + 16` (16-aligned, but overflows CB page) | caught | `NOC transaction overflows a circular buffer` |
| 8 | reader (DRAM→L1) | `size=16` (mult of `A_L1`, not of `A_DRAM_read`) | correctness fail | *silent* |
| 9 | reader | `src_noc_addr + 1` (misaligned DRAM src) | caught | `invalid address alignment in NOC transaction` |
| 10 | reader | `src_noc_addr + 16` (16-aligned but not 32-aligned) | caught | `invalid address alignment in NOC transaction` |
| 11 | writer | `dst_noc_addr + 16` (16-aligned DRAM dst, WRITE) | correctness fail | *silent* (legal — DRAM-write align is 16) |

Take-aways for the theory:

- **Address alignment** is enforced by hardware and observable through the sanitizer (E4, E5, E9, E10).
- **DRAM read vs write asymmetry is real and observable**: E10 (read-side, 16-aligned DRAM) is caught; E11 (write-side, 16-aligned DRAM) is silent.
- **Size alignment is not directly enforced** by the NOC address-alignment check. Short sizes (E1, E8) produce silently wrong data. Sizes that step *past* the bounds of an L1 CB page (E3, E7) are caught by a separate CB-overflow check, but that is incidental to alignment.
- **Odd sizes that stay in bounds can pass**: E2 with `size=127` passed. The hardware or firmware appears to either pad/round or tolerate it; the theory does not promise this and you should not rely on it.
- **Don't conflate sanitizer errors**: `invalid address alignment in NOC transaction` and `NOC transaction overflows a circular buffer` are different checks. The first proves the NOC rejected the address; the second proves the L1 buffer bound was exceeded.

What this does not disprove:

- Running *without* `--dev` will not surface any of these warnings. In production runs the first two silent-fail modes produce wrong results and the alignment-violation modes hang or corrupt memory with no warning. Do not treat the absence of a watcher error as evidence that a kernel is right.
- The sample is one op on one arch. The theory (especially §5's three row-major regimes) should be re-checked on Blackhole with `A_DRAM=64`, and with kernels that issue multiple transactions per page.

---

## Appendix A: Source Pointers

Authoritative constants:

- `tt_metal/hw/inc/internal/tt-1xx/wormhole/noc/noc_parameters.h` — WH alignment, burst, word sizes.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h` — BH alignment, burst, word sizes.
- `tt_metal/hw/inc/internal/tt-1xx/wormhole/core_config.h` — compile-time `LOG_BASE_2_OF_L1_ALIGNMENT`, `LOG_BASE_2_OF_DRAM_ALIGNMENT`.
- `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h` — same, BH values.

Host-side contract:

- `tt_metal/impl/allocator/l1_banking_allocator.cpp:216` — endpoint-pair alignment rules.
- `tt_metal/impl/buffers/buffer.cpp:552,560` — `alignment()`, `aligned_page_size()`.
- `tt_metal/api/tt-metalium/hal.hpp` — runtime alignment query surface.
- `tt_metal/hal.cpp` — `get_l1_alignment`, `get_dram_alignment`.

Kernel-side API surface:

- `tt_metal/hw/inc/api/dataflow/dataflow_api.h` — NOC read/write/page APIs.
- `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h` — `InterleavedAddrGen`, `InterleavedPow2AddrGen`, address generators.
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h` — page-aware accessor with `aligned_page_size`.

---

## Appendix B: One-Page Summary

- Address alignment is hardware-enforced and sanitizer-observable; size alignment is software-enforced and silent when violated. An unaligned address is **illegal**, not slow. A misaligned size is **wrong**, not illegal.
- `A_L1 = 16 B`; `A_DRAM` is 32 B on Wormhole, 64 B on Blackhole.
- `buffer.alignment()` is the allocator's promise. `buffer.aligned_page_size()` is the physical stride. `buffer.page_size()` is the logical content size.
- Page APIs handle alignment for you. Raw NOC APIs do not.
- Row-major is the hard case because logical row width can be smaller than `A_dst` or non-multiple of it. Three strategies: pad the data (S1), pad the stride only (S2), reject (S3).
- Fewer, larger NOC transactions beat many small ones.
- For DRAM traffic, don't use split-reader — NOC0 and NOC1 are not symmetric to DRAM. Stick with the default reader/writer config.
- When debugging, the first thing to check is whether `page_size == aligned_page_size`. Most row-major correctness bugs live in the gap.
