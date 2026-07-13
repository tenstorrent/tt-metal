# SP1 — DRAM read ceiling & amortization (REVERIFIED, corrects sibling branch)

**Board:** BH p150b, 8 DRAM banks (`dram_grid_size 8-1`), 11×10 compute grid, 1.3498 GHz.
**Tool:** upstream `tests/.../8_dram_adjacent_core_read/test_dram_read` (trusted, in main), bf16 (df=1),
8 cores = 1 dedicated reader per bank, each reading its bank contiguously via bank_id addressing with
triple-buffered TRID reads. **Independent of the sibling `bw-optimal-matmul-bh` branch.**

## Method note — the measurement flaw that poisoned prior numbers
The benchmark *reports* wall-clock BW: `input_size / (steady_clock around Enqueue+Finish+
ReadMeshDeviceProfilerResults)`. That window includes dispatch latency **and** (when profiler is on) a
multi-ms profiler dump. So wall-clock BW **understates** the DRAM read, badly at small sizes. The
truth is the **kernel-time** BW = bytes / (BRISC-KERNEL end−start cycles / 1.35 GHz), parsed from the
device profiler CSV (`parse_kernel_bw.py`).

## Result 1 — the ceiling is ~500–511 GB/s, NOT ~445
| total | wall-clock BW | **kernel-time BW** |
|---|---:|---:|
| 8 MB | 137 | **501.6** |
| 16 MB | 240 | **506.7** |
| 32 MB | 250 | **509.4** |
| 64 MB | 372 | **510.6** |
| 128 MB | 383 | **510.2** |
| 256 MB | 448 | **511.4** |

Kernel-time BW is ~500–511 GB/s (98–100 % of the 512 spec) at **every** size ≥ 8 MB. The sibling
branch's headline **"~449 ceiling"** and **"BW climbs 171@8MB → 414@128MB (amortization)"** are
**wall-clock/dispatch artifacts** — there is no DRAM-level amortization knee down to 1 MB/bank.

## Result 2 — the real lever is contiguous BURST size, not total size
Kernel-time BW vs per-bank data at fixed burst (n=2048 ⇒ 16 KB = 8-tile bursts):
`64KB→386, 128KB→440, 256KB→472, 512KB→492, 1MB→501`. Mild secondary knee below ~256 KB/bank.

Kernel-time BW when bursts are forced tiny (n=256 ⇒ block_w=1 ⇒ ~2 KB single-tile bursts):
`~98–105 GB/s regardless of total size (32KB→512KB/bank all ~100)`.

**⇒ DRAM read BW is governed by the length of each contiguous `noc_async_read` burst.** ~16 KB
bursts (8 contiguous tiles) → ~500. ~2 KB bursts (1 tile, strided) → ~100. Need ≳8 KB bursts for
≳450. Total per-bank volume is a weak secondary factor above ~256 KB.

## Implications for the matmul design
- **Ceiling to target is ~500, and it is reachable at realistic sizes** — the M<<N in1 tensors
  (1.5–113 MB) are NOT too small for peak, *if read as large contiguous bursts*.
- **This is exactly why width-sharding in1 wins**: a width-sharded [K,N] gives each core a K-deep
  contiguous column band → naturally large bursts. Interleaved in1 (consecutive tiles → different
  banks) forces ~1-tile strided bursts → the ~100 GB/s floor. The layout *is* the burst size.
- **Reader placement matters** (adjacent-core reads here). Real op must keep readers near their banks.
- Open (→ SP2/SP5): can >8 reader cores (needed to add compute cores) each issue large-burst reads
  and still aggregate to ~500? With width-sharded in1 each core reads K-deep columns (huge bursts),
  so this looks feasible; must be measured, not assumed.

## Reusable artifacts
- `parse_kernel_bw.py` (scratchpad copy): parses profiler CSV → kernel-time BW.
- Run: `TT_METAL_DEVICE_PROFILER=1 test_dram_read --data-type 1 --num-banks 8 --k K --n N
  --num-blocks NB --num-tests 6 --use-device-profiler --bypass-check`, then parse
  `generated/profiler/.logs/profile_log_device.csv` with total_bytes = k*n*2.
