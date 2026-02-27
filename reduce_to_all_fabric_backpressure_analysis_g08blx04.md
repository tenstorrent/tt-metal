# Fabric Back Pressure Analysis: Default vs Large Packet (15232)

**Date:** 2026-02-27
**Profiles:**
- Default packet size: `generated/profiler/reports/2026_02_27_17_04_17/profile_log_device.csv`
- Large packet size (15232): `generated/profiler/reports/2026_02_27_17_05_36/profile_log_device.csv`

**Setup:** 4 devices (D0–D3), 50 iterations, trace_id=1, Blackhole architecture @ 1350 MHz

---

## FWD-FABRIC-WAIT — The Headline Number

| Metric | Default | Large (15232) | Change |
|--------|---------|---------------|--------|
| **Mean** | 48 ns | 240 ns | **+400% (5.0×)** |
| **Median** | 35 ns | 150 ns | **+329% (4.3×)** |
| **Min** | 33 ns | 33 ns | same |
| **Max** | 321 ns | 1,016 ns | **+216% (3.2×)** |
| **Std** | 30 ns | 221 ns | **+637% (7.4×)** |
| **Count** | 16,000 | 16,000 | same |

With default packets, **most waits are very short (35 ns median)** — the fabric slot is almost always ready. With large packets, **the median jumps to 150 ns** and the standard deviation explodes from 30 → 221 ns, indicating highly variable and often blocked fabric waits.

## Back Pressure as % of Forward Loop

| Metric | Default | Large (15232) | Change |
|--------|---------|---------------|--------|
| **Mean** | 21.6% | **61.3%** | +39.7 pp |
| **Median** | 21.4% | **61.6%** | +40.2 pp |
| **Max** | 33.3% | **65.5%** | +32.2 pp |
| **Verdict** | ⚡ MODERATE | **⚠ SIGNIFICANT** | |

With large packets, **the forwarder spends >61% of its loop time blocked** waiting for fabric write slots. The fabric is clearly saturated.

## Distribution Shape Change

The default has a **right-skewed distribution**: median (35 ns) ≪ mean (48 ns), meaning most waits are near-zero with occasional long tails. The large packet has a **much wider spread**: median (150 ns) vs mean (240 ns) with std (221 ns) nearly matching the mean — the forwarder is *consistently* blocked on every slot, with frequent spikes well above the mean.

## End-to-End Impact

| Metric | Default | Large (15232) | Change |
|--------|---------|---------------|--------|
| FWD-FORWARD-LOOP | 4,470 ns | 7,835 ns | **+75%** |
| R1-WAIT-NEIGHBOR | 2,746 ns | 4,425 ns | **+61%** |
| R2-WAIT-NEIGHBOR | 2,177 ns | 3,497 ns | **+61%** |
| R1-COMPUTE | 2,916 ns | 4,417 ns | **+51%** |
| R2-COMPUTE | 2,168 ns | 3,595 ns | **+66%** |
| R1-SEND | 1,082 ns | 1,082 ns | ~0% |
| R2-SEND-STREAMING | 2,235 ns | 3,855 ns | **+72%** |
| **Critical path (Reader)** | **5,020 ns** | **8,019 ns** | **+60%** |

The forward loop grows +75% but the critical path grows +60% because all fabric-dependent zones scale proportionally. R1-SEND is unaffected (~0%), confirming the regression is purely in fabric transit, not local data packing.

## Per-Zone Summary

### Default Packet Size

| Zone Name | Count | Mean (ns) | Median | Min | Max | Std | Max/Mean |
|-----------|-------|-----------|--------|-----|-----|-----|----------|
| FWD-FABRIC-WAIT | 16,000 | 48 | 35 | 33 | 321 | 30 | 6.67× ⚠️ |
| FWD-FORWARD-LOOP | 800 | 4,470 | 4,439 | 3,967 | 6,388 | 201 | 1.43× |
| R1-COMPUTE | 4,800 | 2,916 | 2,887 | 1,221 | 5,253 | 374 | 1.80× |
| R1-LOCAL-INPUT | 1,600 | 96 | 89 | 83 | 143 | 15 | 1.49× |
| R1-SEND | 1,600 | 1,082 | 1,080 | 1,024 | 1,231 | 24 | 1.14× |
| R1-WAIT-NEIGHBOR | 1,600 | 2,746 | 2,673 | 1,179 | 4,998 | 371 | 1.82× |
| R2-COMPUTE | 4,800 | 2,168 | 2,166 | 961 | 3,916 | 296 | 1.81× |
| R2-SEND-STREAMING | 1,600 | 2,235 | 2,188 | 1,083 | 4,381 | 313 | 1.96× |
| R2-WAIT-NEIGHBOR | 1,600 | 2,177 | 2,180 | 297 | 3,924 | 320 | 1.80× |

### Large Packet Size (15232)

| Zone Name | Count | Mean (ns) | Median | Min | Max | Std | Max/Mean |
|-----------|-------|-----------|--------|-----|-----|-----|----------|
| FWD-FABRIC-WAIT | 16,000 | 240 | 150 | 33 | 1,016 | 221 | 4.23× ⚠️ |
| FWD-FORWARD-LOOP | 800 | 7,835 | 7,783 | 7,229 | 9,372 | 316 | 1.20× |
| R1-COMPUTE | 4,800 | 4,417 | 4,167 | 1,916 | 9,088 | 1,317 | 2.06× ⚠️ |
| R1-LOCAL-INPUT | 1,600 | 97 | 90 | 83 | 143 | 15 | 1.48× |
| R1-SEND | 1,600 | 1,082 | 1,077 | 1,024 | 1,254 | 27 | 1.16× |
| R1-WAIT-NEIGHBOR | 1,600 | 4,425 | 4,217 | 1,986 | 8,816 | 1,346 | 1.99× |
| R2-COMPUTE | 4,800 | 3,595 | 3,600 | 949 | 6,774 | 1,074 | 1.88× |
| R2-SEND-STREAMING | 1,600 | 3,855 | 3,634 | 1,399 | 8,186 | 1,317 | 2.12× ⚠️ |
| R2-WAIT-NEIGHBOR | 1,600 | 3,497 | 3,511 | 296 | 6,767 | 1,166 | 1.93× |

## Per-TRISC Analysis

### Default Packet Size

| Zone | TRISC | Role | Mean (ns) | Med | Min | Max |
|------|-------|------|-----------|-----|-----|-----|
| R1-COMPUTE | TRISC_0 | Unpack | 2,835 | 2,763 | 1,261 | 5,063 |
| R1-COMPUTE | TRISC_1 | Math | 2,782 | 2,738 | 1,221 | 4,779 |
| R1-COMPUTE | TRISC_2 | Pack | 3,129 | 3,079 | 1,656 | 5,253 |
| R2-COMPUTE | TRISC_0 | Unpack | 2,180 | 2,182 | 961 | 3,916 |
| R2-COMPUTE | TRISC_1 | Math | 2,177 | 2,173 | 1,139 | 3,507 |
| R2-COMPUTE | TRISC_2 | Pack | 2,145 | 2,144 | 1,140 | 3,778 |

### Large Packet Size (15232)

| Zone | TRISC | Role | Mean (ns) | Med | Min | Max |
|------|-------|------|-----------|-----|-----|-----|
| R1-COMPUTE | TRISC_0 | Unpack | 4,513 | 4,296 | 2,081 | 8,909 |
| R1-COMPUTE | TRISC_1 | Math | 3,996 | 3,698 | 1,916 | 7,822 |
| R1-COMPUTE | TRISC_2 | Pack | 4,740 | 4,537 | 2,324 | 9,088 |
| R2-COMPUTE | TRISC_0 | Unpack | 3,516 | 3,516 | 949 | 6,774 |
| R2-COMPUTE | TRISC_1 | Math | 3,742 | 3,731 | 1,139 | 6,207 |
| R2-COMPUTE | TRISC_2 | Pack | 3,528 | 3,522 | 1,140 | 6,747 |

## Waterfall Timeline Comparison

### Default Packet Size

```
Critical path: 5,020 ns (READER)

    Time │ Reader (NCRISC)        │ Writer (BRISC)         │ Compute (TRISC_1)      │ Sync
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
       0 │ ▒ R1-LOCAL-INPUT       │ ▓ R1-SEND              │ █ R1-COMPUTE           │
      96 │ ░ R1-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    1082 │             ↓          │ ▓ R2-SEND-STREAMING    │             ↓          │ W←C
    2782 │             ↓          │             ↓          │ █ R2-COMPUTE           │ R1✓
    2842 │ ░ R2-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    3316 │             ↓          │         (done)         │             ↓          │
    4960 │             ↓          │         (done)         │         (done)         │
    5020 │         (done)         │         (done)         │         (done)         │
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
   TOTAL │                5020 ns │                3316 ns │                4960 ns │
```

### Large Packet Size (15232)

```
Critical path: 8,019 ns (READER)

    Time │ Reader (NCRISC)        │ Writer (BRISC)         │ Compute (TRISC_1)      │ Sync
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
       0 │ ▒ R1-LOCAL-INPUT       │ ▓ R1-SEND              │ █ R1-COMPUTE           │
      97 │ ░ R1-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    1082 │             ↓          │ ▓ R2-SEND-STREAMING    │             ↓          │ W←C
    3996 │             ↓          │             ↓          │ █ R2-COMPUTE           │ R1✓
    4522 │ ░ R2-WAIT-NEIGHBOR     │             ↓          │             ↓          │
    4936 │             ↓          │         (done)         │             ↓          │
    7738 │             ↓          │         (done)         │         (done)         │
    8019 │         (done)         │         (done)         │         (done)         │
─────────┼────────────────────────┼────────────────────────┼────────────────────────┼───────
   TOTAL │                8019 ns │                4936 ns │                7738 ns │
```

## Behavioral Change: Synchronization Degrades

| Metric | Default | Large (15232) |
|--------|---------|---------------|
| Cross-device spread | 17.3% (SYNCHRONIZED) | 20.7% (NOT SYNCHRONIZED) |
| D0 lag-1 autocorrelation | +0.194 (NO PATTERN) | −0.011 (NO PATTERN) |
| D28 lag-1 autocorrelation | **−0.341 (ALTERNATING)** | +0.103 (NO PATTERN) |
| R1-R2 Pearson correlation | 0.528 (WEAK) | 0.112 (INDEPENDENT) |
| FWD-FABRIC-WAIT spread | 15 ns mean | 19 ns mean |
| R1-COMPUTE spread | 347 ns mean | 707 ns mean |

With default packets, devices are **well-synchronized** (all spike together → global fabric bottleneck), and D28 shows a weak alternating pattern. With large packets, devices become **NOT synchronized** (spike independently → local contention), and the alternating behavior vanishes. R1 and R2 also decouple (correlation 0.528 → 0.112), suggesting fabric congestion makes queuing dynamics dominate timing.

## Device Imbalance

### Default Packet Size

| Zone | D0 Mean | D4 Mean | D24 Mean | D28 Mean | Imbalance |
|------|---------|---------|----------|----------|-----------|
| FWD-FABRIC-WAIT | 50 | 47 | 48 | 48 | 6.7% |
| FWD-FORWARD-LOOP | 4,476 | 4,467 | 4,489 | 4,450 | 0.9% |
| R1-COMPUTE | 2,923 | 2,916 | 2,934 | 2,890 | 1.5% |
| R1-SEND | 1,082 | 1,081 | 1,081 | 1,081 | 0.1% |
| R1-WAIT-NEIGHBOR | 2,754 | 2,747 | 2,763 | 2,720 | 1.6% |
| R2-COMPUTE | 2,157 | — | — | — | 1.4% |
| R2-SEND-STREAMING | 2,244 | — | — | — | 1.8% |
| R2-WAIT-NEIGHBOR | 2,166 | — | — | — | 1.7% |

### Large Packet Size (15232)

| Zone | D0 Mean | D4 Mean | D24 Mean | D28 Mean | Imbalance |
|------|---------|---------|----------|----------|-----------|
| FWD-FABRIC-WAIT | 247 | 233 | 246 | 235 | 5.8% |
| FWD-FORWARD-LOOP | 7,935 | 7,721 | 7,948 | 7,736 | 2.9% |
| R1-COMPUTE | 4,378 | 4,443 | 4,484 | 4,362 | 2.8% |
| R1-SEND | 1,083 | 1,080 | 1,085 | 1,079 | 0.5% |
| R1-WAIT-NEIGHBOR | 4,385 | 4,438 | 4,544 | 4,334 | 4.9% |
| R2-COMPUTE | 3,592 | — | — | — | 7.8% |
| R2-SEND-STREAMING | 3,819 | — | — | — | 5.7% |
| R2-WAIT-NEIGHBOR | 3,469 | — | — | — | 13.1% |

## Key Takeaways

1. **Large packets cause 5× more fabric wait per slot**, confirming back pressure increases significantly
2. **61% of the forwarder's time is spent blocked** vs 22% with default — fabric bandwidth is the limit
3. **The critical path grows +60%** (5,020 ns → 8,019 ns), driven almost entirely by fabric wait increases
4. **R1 is more affected than R2**: R1-WAIT grew +61% vs R2-WAIT +61%, but R1 was already higher, so the absolute gap widens
5. **Device synchronization degrades** with larger packets — globally correlated spikes become independent per-device contention
6. **R1-SEND is unaffected** (1,082 ns in both) — confirming the regression is purely fabric-transit, not data packing
7. **Recommendation:** The default packet size provides significantly better fabric utilization. The 5× increase in back pressure with large packets indicates the fabric bandwidth is being saturated, and the increased variability suggests unfair queuing under load.
