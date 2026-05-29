# Quasar DM Core Cache Management

> **Note:** This document is based on implementation work and hardware documentation.
> Items marked with **(?)** need verification. Please fact-check before relying on details.

## Cache Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Quasar DM Cores (x8)                             │
│                                                                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            ┌────────────┐   │
│  │ DM Core 0  │  │ DM Core 1  │  │ DM Core 2  │    ...     │ DM Core 7  │   │
│  │            │  │            │  │            │            │            │   │
│  │ L1D$ 4K 2W │  │ L1D$ 4K 2W │  │ L1D$ 4K 2W │            │ L1D$ 4K 2W │   │
│  │ L1I$ 4K 2W │  │ L1I$ 4K 2W │  │ L1I$ 4K 2W │            │ L1I$ 4K 2W │   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            └─────┬──────┘   │
│        │               │               │                         │          │
│        └───────────────┴───────────────┴────────── ... ──────────┘          │
│                                   │                                         │
│                                   ▼                                         │
│                       ┌───────────────────┐                                 │
│                       │   L2$ (shared)    │  128 KB, 4-way associative      │
│                       │                   │  64B cache line                 │
│                       └─────────┬─────────┘                                 │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
                       ┌───────────────────┐
                       │       TL1         │  (Tensix L1 / Shared Node SRAM)
                       │   Node Memory     │
                       └───────────────────┘
```

**Key points:**
- L1 D$: 4KB, 2-way, write-back, private per core
- L1 I$: 4KB, 2-way, private per core
- L2: 128KB, 4-way, write-back, shared between all 8 DM cores
- **L1 and L2 are coherent** — L2 flush probes L1 D$ for dirty data before writing to TL1
- TL1 is the backing memory visible to NoC/DMA
- Address 0-4MB is cacheable, 4-8MB aliases to same range but uncached ("write around")

---

## Operations Summary

| Function | Scope | What it does | Verified |
|----------|-------|--------------|----------|
| `flush_l1_dcache(addr)` | Single line or full | L1 D$ → L2 (writeback + invalidate) | Tested |
| `invalidate_l1_dcache(addr)` | Single line or full | Discard L1 D$ line (no writeback) | Tested |
| `invalidate_l1_icache()` | Full only | Discard L1 I$ (for code changes) | Not tested |
| `flush_l2_cache_line(addr)` | Single 64B line | L2 → TL1 (probes L1 D$ first) | Tested |
| `flush_l2_cache_range(addr, size)` | Address range | L2 → TL1 (loops over lines) | Tested |
| `flush_l2_cache_full()` | Entire 128KB | L2 → TL1 (probes L1 D$ first) | Tested |
| `invalidate_l2_cache_line(addr)` | Single 64B line | Discard L2 line (no writeback) | Tested |
| `invalidate_l2_cache(hartid)` | Entire 128KB | Discard L2 (requires all DM cores?) | Not tested |

---

## Flush vs Invalidate

```
┌─────────────────────────────────────────────────────────────┐
│                         FLUSH                               │
│   Writes dirty data to next level, then invalidates line(?) │
│                                                             │
│   Cache ──(write dirty)──► Next Level                       │
│          then discard(?)                                    │
│                                                             │
│   Use: Before someone else needs to read your data          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       INVALIDATE                            │
│   Discards line immediately, dirty data is LOST             │
│                                                             │
│   Cache ──(discard)──► /dev/null                            │
│                                                             │
│   Use: When cached data is stale and you need fresh read    │
│   DANGER: If line is dirty, modifications are lost!         │
└─────────────────────────────────────────────────────────────┘
```

**(?)** Whether flush also invalidates the line, or just writes back and keeps it cached, is unclear from documentation. SiFive docs suggest CFLUSH.D.L1 does both writeback + invalidate.

---

## API Details

### L1 Data Cache

```cpp
// Flush single line (addr != 0) or entire cache (addr == 0)
void flush_l1_dcache(uintptr_t addr);

// Invalidate single line (addr != 0) or entire cache (addr == 0)
void invalidate_l1_dcache(uintptr_t addr);
```

**Implementation:** Custom RISC-V instructions (SiFive/rocket-chip heritage — per code comments)
- `CFLUSH.D.L1` (simm12 = -0x40): flush + invalidate
- `CDISCARD.D.L1` (simm12 = -0x3E): invalidate only

**L1 D$ size:** 4 KB — sourced from SiFive X280 documentation, not directly verified on Quasar

### L1 Instruction Cache

```cpp
void invalidate_l1_icache();  // Uses FENCE.I instruction
```

No flush exists — I$ is read-only from CPU perspective.

**L1 I$ size:** 4KB

### L2 Cache

```cpp
// Single line (64 bytes)
void flush_l2_cache_line(uintptr_t addr);

// Range (automatically aligns to 64B boundaries)
void flush_l2_cache_range(uintptr_t start_addr, size_t size);

// Entire 128KB cache
void flush_l2_cache_full();

// Invalidate single line (no writeback)
void invalidate_l2_cache_line(uintptr_t addr);

// Invalidate entire L2 - requires coordination across all DM cores
void invalidate_l2_cache(uint32_t hartid);
```

**Implementation:** Memory-mapped registers in cache controller
- `L2_FLUSH_ADDR`: Write address to flush that line
- `L2_INVALIDATE_ADDR`: Write address to invalidate that line
- `L2_FULL_INVALIDATE_ADDR`: Bitmask register for full cache invalidation

---

## Code Examples

### Correct: Flush to TL1 (L2 probes L1 automatically)

```cpp
// Write data
volatile uint32_t* ptr = (volatile uint32_t*)0x20000;
for (int i = 0; i < 16; i++) {
    ptr[i] = data[i];
}

// Flush to TL1 - L2 flush automatically probes L1 D$ for dirty data
flush_l2_cache_range(0x20000, 16 * sizeof(uint32_t));
// Data now visible in TL1
```

### Correct: Single line flush

```cpp
uintptr_t addr = 0x20000;
volatile uint32_t* ptr = (volatile uint32_t*)addr;
*ptr = 0xDEADBEEF;

flush_l2_cache_line(addr);   // Probes L1 D$, then flushes L2 line to TL1
```

### Correct: Invalidate I$ after loading new code

```cpp
// New code was written to instruction memory (by overlay load, etc.)
invalidate_l1_icache();  // Clear stale instructions
// Safe to jump to new code now
```

---

## Bad Usage Examples

### BAD: Invalidate when you meant flush

```cpp
volatile uint32_t* ptr = (volatile uint32_t*)0x20000;
*ptr = 0xDEADBEEF;

invalidate_l1_dcache(0x20000);  // WRONG: Data is LOST!
flush_l2_cache_line(0x20000);
// 0xDEADBEEF never reaches TL1
```

**Fix:** Use `flush_l1_dcache` to preserve data.

### BAD: Unaligned range assumption

```cpp
// Data at 0x20010 (not 64B aligned)
flush_l2_cache_line(0x20010);  // Flushes line containing 0x20010
// This works, but be aware: likely flushes 0x20000-0x2003F (entire 64B line)
```

**Note:** `flush_l2_cache_range` handles alignment automatically.

### BAD: Single-core L2 invalidate (?)

```cpp
// Only running on core 0
invalidate_l2_cache(0);  // May deadlock or cause undefined behavior (?)
// Per code comments, hardware waits for all DM cores to signal
```

**Possible workaround (unverified):**
```cpp
volatile uint64_t* inv_reg = (volatile uint64_t*)L2_FULL_INVALIDATE_ADDR;
*inv_reg = 0xFF;  // Signal all DM cores at once (?)
while (*inv_reg != 0);
```

---

## Constants

From `overlay_addresses.h` (verified):

```cpp
#define L2_CACHE_LINE_SIZE  64          // bytes
#define L2_CACHE_SIZE       (128 * 1024) // 128 KB, 4-way associative
```

L1 D$ size: 4 KB — per SiFive X280 documentation (?)

---

## Test Coverage

| Operation | Tested | Test Name |
|-----------|--------|-----------|
| `flush_l1_dcache(addr)` | Yes | `QuasarL1DCacheOps.FlushLine` |
| `flush_l1_dcache(0)` | Yes | `QuasarL1DCacheOps.FlushFull` |
| `invalidate_l1_dcache(addr)` | Yes | `QuasarL1DCacheOps.InvalidateLine` |
| `invalidate_l1_dcache(0)` | Yes | `QuasarL1DCacheOps.InvalidateFull` |
| `invalidate_l1_icache()` | No | — |
| `flush_l2_cache_line(addr)` | Yes | `QuasarL2CacheFlush.FlushLine` |
| `flush_l2_cache_range(...)` | Yes | `QuasarL2CacheFlush.FlushRange` |
| `flush_l2_cache_full()` | Yes | `QuasarL2CacheFlush.FlushFull` |
| `invalidate_l2_cache_line(addr)` | Yes | `QuasarL2CacheFlush.InvalidateLine` |
| `invalidate_l2_cache(hartid)` | No | — (requires multi-core coordination) |

---

## References

- SiFive X280 Core Manual, sections 3.4.2, 6.1.1, 6.1.2 — referenced in code comments
- Chipyard rocket-chip — referenced as basis for Quasar DM cores in code comments
- `tt_metal/hw/inc/internal/tt-2xx/risc_common.h` — implementation
- `tt_metal/hw/inc/internal/tt-2xx/quasar/overlay/overlay_addresses.h` — L2 constants
