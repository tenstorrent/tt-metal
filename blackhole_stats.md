# Blackhole Stats

Hardware numbers for Blackhole, with the file in `tt-metal` that each one comes from.
Everything below was read from source at commit `dbb91c2dfe0` (branch `main`) — no numbers
were measured on hardware for this document.

- [DRAM](#dram)
- [DRAM bandwidth](#dram-bandwidth)
- [NoC bandwidth](#noc-bandwidth)
- [Grid layout](#grid-layout)
- [Where to find diagrams](#where-to-find-diagrams)
- [Sources](#sources)

---

## DRAM

| Property | Value | Source |
| --- | --- | --- |
| DRAM banks (channels / `dram_views`), full part | **8** | `tt_metal/soc_descriptors/blackhole_140_arch.yaml` |
| DRAM banks, harvested part (P100 / p100a) | **7** | `tools/scaleout/factory_system_descriptor/utils.cpp:866` |
| NoC endpoints (subchannels) per bank | 3, of which **2 are usable by kernels** | `blackhole_140_arch.yaml`; `tests/tt_metal/tt_metal/api/test_dram_kernels.cpp:453` |
| Bank size (`dram_view_size` / `dram_bank_size`) | 4278190080 B = 0xFF000000 ≈ **3.98 GiB** | `blackhole_140_arch.yaml` |
| Total DRAM | ~32 GB (8 banks) / ~28 GB (7 banks) | derived |
| Memory type | GDDR6 @ 16 GT/s | `ttnn/core/operation.cpp:36` |

### Harvesting

DRAM harvesting on Blackhole is **independent of tensix (column) harvesting**, so any product can
appear with 0 or 1 DRAM channels harvested. Bank count is a compile-time input to the kernel/firmware
build key, so both variants are precompiled:

```cpp
// tt_metal/jit_build/jit_device_config.cpp:84
if (arch == tt::ARCH::BLACKHOLE) {
    return {0, 1};   // DRAM-harvest counts to precompile
}
```

Per-board harvesting masks (`tools/scaleout/factory_system_descriptor/utils.cpp:866`):

| Board | `dram_harvesting_mask` | Banks |
| --- | --- | --- |
| P100 / p100a | `8` (0b1000 → physical channel 3 harvested) | 7 |
| P150, P300, UBB_BLACKHOLE | `0` | 8 |

Harvested channels are removed and the remaining ones compacted — a physical channel maps to its
logical index via `physical - harvested_before(mask, physical)`
(`tt_metal/llrt/metal_soc_descriptor.cpp:143-152`).

On Blackhole (and only Blackhole), the **NOC0 endpoint of every DRAM view is reserved for syseng
firmware** and excluded from Metal's DRAM core list (`metal_soc_descriptor.cpp:53-60`), which is why
each bank exposes 2 usable endpoints rather than 3.

### Querying the real count at runtime

Never hardcode 8 — harvested parts expose 7, and any striping (KV-cache ND-shards, disaggregation
address tables) must derive from the same device count to stay consistent.

```python
device.dram_grid_size().x        # Python — see models/demos/common/prefill/runners/migration.py:435
```
```cpp
device->num_dram_channels();     // C++ — tt_metal/api/tt-metalium/device.hpp:83
soc_desc.get_num_dram_views();   // tt_metal/llrt/metal_soc_descriptor.cpp:154
```

The nominal constant exists but is documented as a fallback only
(`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:22-25`):

```python
# Nominal DRAM bank count for a full (unharvested) Blackhole part. Prefer get_num_dram_banks(device)
# at runtime: harvested parts expose fewer banks (e.g. 7) ...
BH_NUM_DRAM_BANKS = 8
```

---

## DRAM bandwidth

| Metric | Blackhole | Wormhole | Source |
| --- | --- | --- | --- |
| Aggregate peak | **512 GB/s** | 384 GB/s | `.../8_dram_adjacent_core_read/test_dram_read.cpp:282` |
| Aggregate peak (op perf model) | **512 GB/s** | 258 GB/s (6 ch × 2 banks × 21.5) | `ttnn/core/operation.cpp:36-42` |
| **Per bank (peak)** | **64 GB/s** (512 / 8; = 32-bit GDDR6 channel @ 16 Gbps/pin) | 21.5 GB/s | derived |
| Per reader core (measured, large txns) | **~50 GB/s** (~79% of a bank) | ~28 GB/s | `ttnn/.../data_movement/common/common.cpp:89` |

On a 7-bank p100a the aggregate peak is ~448 GB/s; per-bank is unchanged at 64 GB/s.

### Measured NoC↔DRAM bandwidth vs transaction size

`noc_dram_bw` from `ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:89-103` — GB/s for a
single reader core, used by the CCL and TM roofline models:

| Transaction size | Wormhole | Blackhole |
| --- | --- | --- |
| 16 B | 0.436 | 0.387 |
| 32 B | 0.868 | 0.772 |
| 64 B | 1.736 | 1.545 |
| 128 B | 3.489 | 3.088 |
| 256 B | 6.975 | 6.176 |
| 512 B | 13.889 | 12.361 |
| 1024 B | 27.891 | 24.710 |
| 2048 B | 28.411 | **49.164** |
| 4096 B | 28.227 | 50.238 |
| 8192 B | 28.537 | 50.393 |
| 16384 B | 27.831 | 50.636 |
| 32768 B | 27.758 | 50.695 |
| 65536 B | 28.694 | 50.626 |

**Page size matters more than bank count.** Below ~2 KB throughput falls off a cliff — 512 B gets
12.4 GB/s, only ~24% of peak. Blackhole needs ≥2 KB transactions to beat Wormhole at all.

This is also why DRAM-sharded matmul places **one reader core adjacent to each bank** rather than
fewer readers with bigger stripes: a single core tops out near 50 GB/s, below the bank's 64 GB/s.
See `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md`.

---

## NoC bandwidth

| Property | Blackhole | Wormhole | Source |
| --- | --- | --- | --- |
| **Link width** | **64 B/cycle** | 32 B/cycle | `tests/tt_metal/tt_metal/data_movement/python/constants.py:9` |
| NoC clock (fallback constant) | 1.35 GHz | 1.0 GHz | `constants.py:16` |
| **Per link, per direction** | **~86 GB/s** (64 × 1.35) | ~32 GB/s | derived |
| Independent NoCs | 2 (NOC0 / NOC1), each a 2D torus | 2 | `blackhole_140_arch.yaml` |
| Per-core aggregate (both NoCs driven) | ~173 GB/s | ~64 GB/s | derived |

```python
# tests/tt_metal/tt_metal/data_movement/python/constants.py:9
NOC_WIDTHS   = {"wormhole_b0": 32, "blackhole": 64}      # bytes/cycle
NOC_FREQ_GHZ = {"wormhole_b0": 1.0, "blackhole": 1.35}   # fallback only
```

`NOC_WIDTHS` is the theoretical ceiling in the tooling: the heatmap warns when a measured core
exceeds it (`heatmap.py:72`) and it sets the plot y-max (`plotter.py:24`).

> **Use B/cycle, not GB/s.** 1.35 GHz is only a *fallback*. Tests query the device's real AICLK and
> convert with that (`stats_reporter.py:233-244`, `tech_reports/PCIe_bandwidth/PCIe_bandwidth.md:70`).
> Under thermal/power limits the actual clock is lower, so bytes/cycle is the stable unit.

### CI bandwidth bounds — bytes/cycle

From `tests/tt_metal/tt_metal/data_movement/python/test_mappings/test_bounds.yaml`.
`riscv_0` = sender/writer, `riscv_1` = receiver/reader. Percentages are against BH's 64 B/cycle link.

**Core-to-core (near link peak)**

| Pattern | BH | WH | BH % of link |
| --- | --- | --- | --- |
| One to All Unicast Directed Ideal | 62 | 31 | 97% |
| One Packet Read / Write Directed Ideal | 61 / 61 | 28 / 30 | 95% |
| L1 Interleaved Page Directed Ideal | 61 (r1) / 62 (r0) | 31 / 31 | 95–97% |
| One from All Directed Ideal | 60 | 30 | 94% |
| Loopback Directed Ideal | 60 | 28 | 94% |
| One to One / One from One Directed Ideal | 59 / 59 | 30 / 28 | 92% |

**Multicast (well below link peak)**

| Pattern | BH | WH |
| --- | --- | --- |
| One to All Multicast Linked Loopback Directed Ideal 2.0 | 41 | 24.7 |
| One to All Multicast Linked Directed Ideal | 39 | 22 |
| One to All Multicast Linked Semaphore Directed Ideal | 37 | 22 |
| One to All Multicast Directed Ideal (unlinked) | **24** | 14 |

Unlinked multicast gets only ~38% of link peak — that gap is the reason linked multicast exists.

**DRAM paths**

| Pattern | BH | WH |
| --- | --- | --- |
| DRAM Interleaved Page Directed Ideal | 60 (r1) / 58 (r0) | 30 / 26 |
| DRAM Sharded Read Directed Ideal | 60 | 30 |
| DRAM Sharded Read Transaction ID Directed Ideal | 60 | 30 |
| DRAM Directed Ideal | 33 (r1) / 34 (r0) | 22 / 21 |

**Multi-core interleaved (contention-bound)**

| Pattern | BH | WH |
| --- | --- | --- |
| Multi Interleaved 2x2 Read / Write | 26 / 22 | 13.8 / 16 |
| Multi Interleaved 2x2 (both) | 13 / 13 | 7.9 / 8.1 |
| Multi Interleaved 6x6 Read / Write | 4.6 / 3.5 | 3.2 / 2.9 |
| Multi Interleaved Read / Write | 1.5 / 1.2 | 1.9 / 1.7 |
| Multi Interleaved (both) | 0.6 / 0.6 | 1.0 / 1.0 |

Note the inversions: Blackhole is **slower than Wormhole** on the heavily-contended interleaved
cases (Multi Interleaved, and the write side of 2x2). The wider link does not help when the
bottleneck is contention.

**Op-level and host**

| Pattern | BH | WH |
| --- | --- | --- |
| PCIe Read Bandwidth | 42.5 | 20.1 |
| Reshard Hardcoded Medium | 30 | 15 |
| Reshard Hardcoded Many Cores | 25 | 10 |
| Conv Halo Gather | 20 | 10 |
| Reshard Hardcoded Small / 2 Cores to Many | 7 / 7 | 3 / 3 |
| Conv Act with halo 3x3 | 6 | 3 |
| Deinterleave Multi Core / Single Core | 2 / 1.7 | 2 / 1.7 |
| Conv Act with halo 3x3 Small | 0.6 | 0.3 |

---

## Grid layout

17 × 12 tiles in **NOC0 coordinates**, generated from `tt_metal/soc_descriptors/blackhole_140_arch.yaml`.
`Dc.s` = DRAM channel *c*, subchannel *s*; `TX` = Tensix; `rtr` = router-only.

```
y\x   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
  0 D0.0  rtr PCIe  rtr  rtr  rtr  rtr  rtr  ARC D4.0  rtr PCIe  rtr  rtr  rtr  rtr  rtr
  1 D0.1  ETH  ETH  ETH  ETH  ETH  ETH  ETH  rtr D4.1  ETH  ETH  ETH  ETH  ETH  ETH  ETH
  2 D1.0  TX   TX   TX   TX   TX   TX   TX   rtr D5.0  TX   TX   TX   TX   TX   TX   TX
  3 D1.2  TX   TX   TX   TX   TX   TX   TX   rtr D5.2  TX   TX   TX   TX   TX   TX   TX
  4 D2.1  TX   TX   TX   TX   TX   TX   TX   rtr D6.1  TX   TX   TX   TX   TX   TX   TX
  5 D3.0  TX   TX   TX   TX   TX   TX   TX   rtr D7.0  TX   TX   TX   TX   TX   TX   TX
  6 D3.2  TX   TX   TX   TX   TX   TX   TX   rtr D7.2  TX   TX   TX   TX   TX   TX   TX
  7 D3.1  TX   TX   TX   TX   TX   TX   TX   rtr D7.1  TX   TX   TX   TX   TX   TX   TX
  8 D2.2  TX   TX   TX   TX   TX   TX   TX   rtr D6.2  TX   TX   TX   TX   TX   TX   TX
  9 D2.0  TX   TX   TX   TX   TX   TX   TX   rtr D6.0  TX   TX   TX   TX   TX   TX   TX
 10 D1.1  TX   TX   TX   TX   TX   TX   TX   rtr D5.1  TX   TX   TX   TX   TX   TX   TX
 11 D0.2  TX   TX   TX   TX   TX   TX   TX   rtr D4.2  TX   TX   TX   TX   TX   TX   TX
```

| Tile type | Count | Location |
| --- | --- | --- |
| Tensix (`functional_workers`) | **140** | x = 1–7 and 10–16, y = 2–11 (14 × 10) |
| DRAM | 8 channels × 3 endpoints | column x=0 (channels 0–3), column x=9 (channels 4–7) |
| Ethernet | 14 | row y=1, x = 1–7 and 10–16 |
| PCIe | 2 | `2-0`, `11-0` |
| ARC | 1 | `8-0` |
| Router-only | 23 | row y=0 (minus PCIe/ARC/DRAM) + column x=8 |

| Memory | Size | Source |
| --- | --- | --- |
| `worker_l1_size` | 1572864 B = **1.5 MB** | `blackhole_140_arch.yaml` |
| `eth_l1_size` | 524288 B = **512 KB** | `blackhole_140_arch.yaml` |
| `dram_bank_size` | 4278190080 B ≈ 3.98 GiB | `blackhole_140_arch.yaml` |

### Notes on reading the grid

- **The Tensix block is split** by the router-only column at x=8. There is no compute at x=8 or x=0/x=9.
- **DRAM y-ordering is deliberately non-monotonic** — channel 2's endpoints land at y=4, 8, 9. This
  is the anti-congestion placement from `tech_reports/Saturating_DRAM_bandwidth/`: readers sit
  adjacent to their bank so return traffic is one hop, and same-row reader pairs use different NoC
  virtual channels.
- **`harvested_workers` is empty in the descriptor.** Harvesting is applied at runtime from the
  device's mask, so a real p150 reports fewer than 140 Tensix.
- **These are NOC0 coordinates.** Blackhole has `translation_id_enabled: True`, so kernel-visible
  translated coordinates differ. `metal_SocDescriptor` holds TRANSLATED coords in
  `dram_view_worker_cores` (see the header note on `is_noc0_dram_endpoint`).
- Per-DRAM-view worker/eth endpoint assignment is in the `dram_views` block. The NOC0 assignment
  must match what CMFW uses to read DRAM telemetry, to avoid SYS-1419.

### Other Blackhole features from the descriptor

```yaml
features:
  noc:      { translation_id_enabled: True }
  unpacker: { version: 2, inline_srca_trans_without_srca_trans_instr: True }
  math:     { dst_size_alignment: 32768 }
  packer:   { version: 2 }
  overlay:  { version: 2 }
```

---

## Where to find diagrams

No Blackhole grid image ships in this repo — the only checked-in NoC-grid images are Wormhole's
(`tech_reports/EthernetMultichip/images/wormhole_80_noc_view.png`,
`docs/source/images/tenstorrent-wormhole-logical-noc-diagram.webp`).

- **Canonical BH NoC diagrams:** <https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/README.md>
  — tile layout, NOC0/NOC1 torus wiring, coordinate translation. This is what
  `docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst:1140` points at.
- **Rest of the BH ISA tree:** <https://github.com/tenstorrent/tt-isa-documentation/tree/main/BlackholeA0>
  for Tensix tile internals.
- **Generate a PNG from the descriptor:** `tests/tt_metal/tt_metal/data_movement/python/heatmap.py`
  renders the grid with ARC/PCIe/DRAM/ETH labeled, reading `blackhole_140_arch.yaml`
  (`heatmap.py:51-80`). It overlays measured per-core bandwidth, so it needs a profiler CSV.
- **NoC traffic visualization:** [tt-npe](https://github.com/tenstorrent/tt-npe) via
  `--collect-noc-traces`, viewed in the NPE tab of
  [TT-NN Visualizer](https://github.com/tenstorrent/ttnn-visualizer). See
  `docs/source/ttnn/ttnn/profiling_ttnn_operations.rst:290`.

---

## Sources

| File | What it provides |
| --- | --- |
| `tt_metal/soc_descriptors/blackhole_140_arch.yaml` | grid size, DRAM channels/views/size, eth, workers, router-only, L1 sizes, features |
| `tt_metal/llrt/metal_soc_descriptor.cpp` | harvesting → logical channel mapping; NOC0 DRAM endpoint exclusion |
| `tt_metal/jit_build/jit_device_config.cpp:84` | which DRAM-harvest variants get precompiled |
| `tools/scaleout/factory_system_descriptor/utils.cpp:855-875` | per-board tensix/DRAM/eth harvesting masks |
| `ttnn/core/operation.cpp:33-45` | peak DRAM BW in the op performance model |
| `ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:89-103` | measured NoC BW tables (DRAM, L1 read/write/local) |
| `tests/tt_metal/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/test_dram_read.cpp:280-292` | per-arch DRAM BW constants; optimal bank→reader assignment |
| `tests/tt_metal/tt_metal/data_movement/python/constants.py` | NoC width (B/cycle) and clock per arch |
| `tests/tt_metal/tt_metal/data_movement/python/test_mappings/test_bounds.yaml` | CI bandwidth bounds per pattern per arch |
| `tests/tt_metal/tt_metal/api/test_dram_kernels.cpp:453` | usable DRAM endpoints per bank |
| `models/demos/common/prefill/runners/migration.py:435` | runtime bank-count query helper |
| `tech_reports/Saturating_DRAM_bandwidth/Saturating_DRAM_bandwidth.md` | why readers sit adjacent to banks; VC assignment |
