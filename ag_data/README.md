# AllGather Blackhole tuning + comparison data

Machine: 8x Blackhole p150b (physical 2x4 mesh, used as a (1,8) line/ring), num_links = 2
(hardware-detected), fabric packet payload 15232 B, DRAM-interleaved, 8 devices on the
gather axis. All timings are DEVICE KERNEL DURATION.

## tuning_raw_measurements.csv
Every measurement taken while tuning the two factories (1049 rows, 75 sweeps). Columns:

| column | meaning |
|---|---|
| `experiment` | sweep tag (one pytest run) |
| `round` | which question the sweep belonged to (r1 unicast, r2 multicast, r3 crossover) |
| `shape` | output tensor shape |
| `lay` | `tile` or `rm` (row-major) |
| `topo` | `ring` or `line` |
| `links` | fabric links per connection (always 2 on this machine) |
| `knobs` | the 8-slot override tuple, see below. `(factory default)` = scaffold already removed |
| `us` | **median** over iterations of (max over the 8 devices) |
| `mean_us` | same, mean instead of median |
| `spread_%` | (max-min)/median across iterations -- a noise indicator, not an error bar |
| `eff_%` | roofline (PM IDEAL) / measured |
| `iters` | profiled iterations after dropping compile + trace-capture |

Knob tuple index: `0` workers_per_dir, `1` packets_per_cb_entry, `2` run_cap_bytes **+1**
(so an explicit "no cap" is expressible), `3` mux_slots_per_channel, `4` cb_depth,
`5` unused (was TRID read_ahead, since dropped), `6` signals_per_stripe,
`7` factory pin (1 = unicast, 2 = multicast, 0 = let the heuristic choose).
A `0` in slots 0-6 means "keep whatever the factory chose".

Caveats worth carrying to the 4-device machine:
- Compare knob values **within one `experiment`**; cross-run drift has flipped 5% verdicts.
- Shapes below ~2 MB/link (13-35 us) are noise-dominated; `spread_%` of 25-50% occurs even
  at 50 iterations, so sub-5% differences there mean nothing.
- The 4-device machine has **4 links**, so at the same output shape its `per_link_bytes` is
  half this machine's. Every threshold in the tuned heuristics is expressed in
  `per_link_bytes`, so they should be compared at matched per-link bytes, not matched shape.

## impl_comparison.csv
Three-way comparison of `ttnn.experimental.all_gather_async`, `ttnn.all_gather` on this
branch, and `ttnn.all_gather` on `main`. Columns: `source` (branch/main), `test`, `impl`,
`topology`, `shape`, `config` (full pytest id), `measured_us`, `roofline_us`,
`efficiency_pct`, `op_code`.

Tests:
- `tensor_size` -- 16 tile shapes, 8 KB .. 96 MB per link, gather on the last dim.
- `page_size` -- 8 row-major shapes, 64 B .. 8 KB pages, fixed ~8.4 MB per link.
  `experimental.all_gather_async` has **no native row-major path** (it falls back to a
  composite AllBroadcast + Concat), so it is excluded here rather than compared unfairly.
- `page_size_tile` -- the page axis every implementation can run natively: a tile page is
  32x32 elements, so the dtype sets its size (bfloat8_b 1088 B, bfloat16 2048 B,
  float32 4096 B). Shapes are paired with the dtype to hold the output near 16.8 MB.
  Only three points, but all three implementations run them natively.
