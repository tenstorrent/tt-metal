# Fabric Back Pressure Analysis: Default vs Large Packet (15232)

**Date:** 2026-02-27
**Profiles:**
- Default packet size: `generated/profiler/reports/2026_02_27_14_52_29/profile_log_device.csv`
- Large packet size (15232): `generated/profiler/reports/2026_02_27_14_53_10/profile_log_device.csv`

**Setup:** 4 devices (D0–D3), 50 iterations, trace_id=1, Blackhole architecture @ 800 MHz

---

## FWD-FABRIC-WAIT — The Headline Number

| Metric | Default | Large (15232) | Change |
|--------|---------|---------------|--------|
| **Mean** | 139 ns | 331 ns | **+138% (2.4×)** |
| **Median** | 60 ns | 342 ns | **+470% (5.7×)** |
| **Min** | 55 ns | 55 ns | same |
| **Max** | 952 ns | 1,575 ns | +65% |
| **Std** | 127 ns | 217 ns | +71% |
| **Count** | 16,000 | 16,000 | same |

The median tells the real story: with default packets, **most waits are very short (60 ns)** — the fabric slot is almost always ready. With large packets, **the median jumps to 342 ns** — the forwarder is routinely blocked waiting for an empty slot.

## Back Pressure as % of Forward Loop

| Metric | Default | Large (15232) | Change |
|--------|---------|---------------|--------|
| **Mean** | 29.6% | **57.6%** | +28 pp |
| **Median** | 34.0% | **57.5%** | +24 pp |
| **Max** | 44.8% | **61.0%** | +16 pp |
| **Verdict** | ⚡ MODERATE | **⚠ SIGNIFICANT** | |

With large packets, **the forwarder spends >57% of its loop time blocked** waiting for fabric write slots. The fabric is clearly saturated.

## Distribution Shape Change

The default has a **bimodal distribution**: median (60 ns) ≪ mean (139 ns), meaning most waits are near-zero with occasional long tails. The large packet has a **uniform/symmetric distribution**: median (342) ≈ mean (331), meaning the forwarder is *consistently* blocked on every slot.

## End-to-End Impact

| Metric | Default | Large (15232) | Change |
|--------|---------|---------------|--------|
| FWD-FORWARD-LOOP | 9,598 ns | 11,487 ns | **+20%** |
| R1-WAIT-NEIGHBOR | 5,810 ns | 6,314 ns | +8.7% |
| R2-WAIT-NEIGHBOR | 5,205 ns | 5,249 ns | +0.8% |
| R1-COMPUTE | 5,943 ns | 6,415 ns | +7.9% |
| R2-COMPUTE | 5,246 ns | 5,298 ns | +1.0% |
| R1-SEND | 1,817 ns | 1,814 ns | ~0% |
| R2-SEND-STREAMING | 4,836 ns | 5,345 ns | +10.5% |
| **Critical path (Reader)** | **11,177 ns** | **11,726 ns** | **+4.9%** |

The forward loop grows +20% but the critical path only grows +4.9% because the reader wait is dominated by the entire round-trip (compute + fabric latency), and the extra fabric stall gets partially absorbed in the pipeline.

## Per-Zone Summary

### Default Packet Size

| Zone Name | Count | Mean (ns) | Median | Min | Max | Std | Max/Mean |
|-----------|-------|-----------|--------|-----|-----|-----|----------|
| FWD-FABRIC-WAIT | 16,000 | 139 | 60 | 55 | 952 | 127 | 6.84× ⚠️ |
| FWD-FORWARD-LOOP | 800 | 9,598 | 9,422 | 8,431 | 12,321 | 698 | 1.28× |
| R1-COMPUTE | 4,800 | 5,943 | 5,722 | 3,071 | 10,852 | 1,355 | 1.83× |
| R1-LOCAL-INPUT | 1,600 | 163 | 150 | 140 | 230 | 26 | 1.41× |
| R1-SEND | 1,600 | 1,817 | 1,816 | 1,710 | 2,008 | 41 | 1.10× |
| R1-WAIT-NEIGHBOR | 1,600 | 5,810 | 5,571 | 3,410 | 10,326 | 1,342 | 1.78× |
| R2-COMPUTE | 4,800 | 5,246 | 5,340 | 2,234 | 7,164 | 648 | 1.37× |
| R2-SEND-STREAMING | 1,600 | 4,836 | 4,617 | 2,464 | 9,309 | 1,275 | 1.92× |
| R2-WAIT-NEIGHBOR | 1,600 | 5,205 | 5,314 | 2,228 | 7,159 | 686 | 1.38× |

### Large Packet Size (15232)

| Zone Name | Count | Mean (ns) | Median | Min | Max | Std | Max/Mean |
|-----------|-------|-----------|--------|-----|-----|-----|----------|
| FWD-FABRIC-WAIT | 16,000 | 331 | 342 | 55 | 1,575 | 217 | 4.76× ⚠️ |
| FWD-FORWARD-LOOP | 800 | 11,487 | 11,471 | 10,411 | 12,975 | 425 | 1.13× |
| R1-COMPUTE | 4,800 | 6,415 | 6,115 | 3,260 | 11,508 | 1,596 | 1.79× |
| R1-LOCAL-INPUT | 1,600 | 163 | 151 | 140 | 244 | 26 | 1.49× |
| R1-SEND | 1,600 | 1,814 | 1,811 | 1,701 | 2,051 | 42 | 1.13× |
| R1-WAIT-NEIGHBOR | 1,600 | 6,314 | 5,981 | 3,569 | 11,056 | 1,571 | 1.75× |
| R2-COMPUTE | 4,800 | 5,298 | 5,411 | 1,626 | 7,520 | 937 | 1.42× |
| R2-SEND-STREAMING | 1,600 | 5,345 | 4,991 | 2,634 | 9,980 | 1,491 | 1.87× |
| R2-WAIT-NEIGHBOR | 1,600 | 5,249 | 5,372 | 1,440 | 7,460 | 960 | 1.42× |

## Per-TRISC Analysis

### Default Packet Size

| Zone | TRISC | Role | Mean (ns) | Med | Min | Max |
|------|-------|------|-----------|-----|-----|-----|
| R1-COMPUTE | TRISC_0 | Unpack | 5,969 | 5,736 | 3,584 | 10,424 |
| R1-COMPUTE | TRISC_1 | Math | 5,533 | 5,316 | 3,071 | 9,535 |
| R1-COMPUTE | TRISC_2 | Pack | 6,327 | 6,099 | 3,966 | 10,852 |
| R2-COMPUTE | TRISC_0 | Unpack | 5,200 | 5,308 | 2,234 | 7,149 |
| R2-COMPUTE | TRISC_1 | Math | 5,326 | 5,385 | 3,046 | 7,128 |
| R2-COMPUTE | TRISC_2 | Pack | 5,213 | 5,324 | 2,240 | 7,164 |

### Large Packet Size (15232)

| Zone | TRISC | Role | Mean (ns) | Med | Min | Max |
|------|-------|------|-----------|-----|-----|-----|
| R1-COMPUTE | TRISC_0 | Unpack | 6,471 | 6,137 | 3,774 | 11,145 |
| R1-COMPUTE | TRISC_1 | Math | 5,943 | 5,529 | 3,260 | 10,836 |
| R1-COMPUTE | TRISC_2 | Pack | 6,831 | 6,496 | 4,141 | 11,508 |
| R2-COMPUTE | TRISC_0 | Unpack | 5,248 | 5,371 | 1,626 | 7,458 |
| R2-COMPUTE | TRISC_1 | Math | 5,388 | 5,480 | 2,078 | 7,520 |
| R2-COMPUTE | TRISC_2 | Pack | 5,259 | 5,386 | 1,928 | 7,470 |

## Waterfall Timeline Comparison

### Default Packet Size

```
Critical path: 11,177 ns (READER)

    Time │ Reader (NCRISC)        │ Writer (BRISC)         │ Compute (TRISC_1)      │ Sync
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
       0 │ ▒ R1-LOCAL-INPUT       │ ▓ R1-SEND              │ █ R1-COMPUTE           │
     163 │ ░ R1-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    1817 │             ↓          │ ▓ R2-SEND-STREAMING    │             ↓          │ W←C
    5533 │             ↓          │             ↓          │ █ R2-COMPUTE           │ R1✓
    5973 │ ░ R2-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    6653 │             ↓          │         (done)         │             ↓          │
   10860 │             ↓          │         (done)         │         (done)         │
   11177 │         (done)         │         (done)         │         (done)         │
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
   TOTAL │               11177 ns │                6653 ns │               10860 ns │
```

### Large Packet Size (15232)

```
Critical path: 11,726 ns (READER)

    Time │ Reader (NCRISC)        │ Writer (BRISC)         │ Compute (TRISC_1)      │ Sync
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
       0 │ ▒ R1-LOCAL-INPUT       │ ▓ R1-SEND              │ █ R1-COMPUTE           │
     163 │ ░ R1-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    1814 │             ↓          │ ▓ R2-SEND-STREAMING    │             ↓          │ W←C
    5943 │             ↓          │             ↓          │ █ R2-COMPUTE           │ R1✓
    6477 │ ░ R2-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    7159 │             ↓          │         (done)         │             ↓          │
   11331 │             ↓          │         (done)         │         (done)         │
   11726 │         (done)         │         (done)         │         (done)         │
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
   TOTAL │               11726 ns │                7159 ns │               11331 ns │
```

## Behavioral Change: Alternating Pattern Vanishes

| Metric | Default | Large (15232) |
|--------|---------|---------------|
| Lag-1 autocorrelation | **−0.82 (STRONG ALTERNATING)** | −0.29 (NO PATTERN) |
| Cross-pair correlation | −0.90 (STRONG ANTI-CORR) | −0.35 (WEAK ANTI-CORR) |
| Per-device even/odd Δ | 20–25% STRONG | 4–8% NONE |
| FWD-FABRIC-WAIT spread | 106 ns mean | 22 ns mean |
| R1-COMPUTE spread | 1,611 ns mean | 955 ns mean |

With default packets, the system had a strong **alternating pattern** where device pairs traded off who was fast/slow each iteration (likely due to contention on shared fabric links). With large packets, this pattern **largely disappears** — the fabric is so uniformly saturated that all devices experience similar back pressure, and the alternation signal is washed out.

## Device Imbalance

### Default Packet Size

| Zone | D0 Mean | D1 Mean | D2 Mean | D3 Mean | Imbalance |
|------|---------|---------|---------|---------|-----------|
| FWD-FABRIC-WAIT | 142 | 142 | 136 | 137 | 4.6% |
| FWD-FORWARD-LOOP | 9,603 | 9,612 | 9,601 | 9,577 | 0.4% |
| R1-COMPUTE | 5,974 | 5,932 | 5,918 | 5,947 | 1.0% |
| R1-WAIT-NEIGHBOR | 5,837 | 5,799 | 5,787 | 5,816 | 0.9% |
| R2-COMPUTE | 5,224 | 5,246 | 5,273 | 5,242 | 0.9% |
| R2-WAIT-NEIGHBOR | 5,185 | 5,207 | 5,227 | 5,200 | 0.8% |

### Large Packet Size (15232)

| Zone | D0 Mean | D1 Mean | D2 Mean | D3 Mean | Imbalance |
|------|---------|---------|---------|---------|-----------|
| FWD-FABRIC-WAIT | 332 | 331 | 329 | 330 | 1.0% |
| FWD-FORWARD-LOOP | 11,499 | 11,509 | 11,458 | 11,484 | 0.4% |
| R1-COMPUTE | 6,444 | 6,422 | 6,367 | 6,428 | 1.2% |
| R1-WAIT-NEIGHBOR | 6,345 | 6,309 | 6,276 | 6,325 | 1.1% |
| R2-COMPUTE | 5,286 | 5,281 | 5,355 | 5,270 | 1.6% |
| R2-WAIT-NEIGHBOR | 5,240 | 5,238 | 5,296 | 5,223 | 1.4% |

## Key Takeaways

1. **Large packets cause 2.4× more fabric wait per slot**, confirming back pressure increases significantly
2. **57% of the forwarder's time is spent blocked** vs 30% with default — fabric bandwidth is the limit
3. **Despite the forwarder taking +20% longer**, the critical path only grows +4.9% — the pipeline absorbs some of the extra latency
4. **R1 is more affected than R2**: R1-WAIT grew +8.7% vs R2-WAIT +0.8%, because R2 streams from compute and the extra fabric latency overlaps with compute time
5. **The alternating pattern vanishes** with larger packets — uniform saturation replaces the contention-driven oscillation seen with smaller packets
