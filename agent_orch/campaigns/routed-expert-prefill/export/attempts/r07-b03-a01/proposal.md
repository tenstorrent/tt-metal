# r07-b03-a01: NoC-1 weight reads (writer up rows + down tail tiles) go through each DRAM bank's FREE tile (neither the NoC-0 nor the NoC-1 soc-descriptor endpoint) when it is fewer north hops from the reading core than the default NoC-1 endpoint, and the NoC-0/NoC-1 weight split moves from ~0.60 to ~0.56 to use the freed NoC-1 links

## Motivation
- r05-b02-a01 (+2.4%) showed the weight stream is NoC-link-bound: the hot links are NoC 1's
  DRAM-column links, because every NoC-1 read of a bank targets the soc descriptor's fixed
  NoC-1 endpoint (UMD port 1, physical rows 1/10/4/7) and the -y response wraps round the column.
  The 60/40 split balanced NoC 0 and NoC 1 at ~52-56 GB/s on the hottest link.
- r06-b03-a01 remapped NoC-1 reads to each bank's NEAREST tile (any of 3) and moved the split to
  ~50/50. Result: DN -7..-8% on dsv4-pro / kimi-k2.7 / minimax-m3 (down remap works and helps),
  but GU +10..+17% on every DRAM-bound shape (-5.7% overall). Its reflection names two
  confounded causes for GU: (a) the split change to 0.53 and (b) **port concentration**: the
  nearest tile was often the bank's NoC-0 endpoint tile, so that one AXI port carried the bank's
  NoC-0 gate stream AND most of its NoC-1 up stream (up to ~55 GB/s on one port, vs ~33 today),
  while the other ports idled. Its next #2: "never use the NoC-0 endpoint tile for NoC-1 reads;
  restrict the choice to {default port 1, free port}".
- r06-b03-a01's link model used the column-0 NoC-0 endpoints for banks 5-7 too. The yaml
  (`blackhole_140_arch.yaml` dram_views) gives NoC-0 ports [2,0,0,0,2,2,2,2], so on x = 9 the
  NoC-0 tiles are rows 11/3/8/6 and the free tiles rows 0/2/9/5. Recomputed with r06-b04-a01's
  corner sender placement and down multicast NoC choice (/tmp/r07b03/model.py, 440 GB/s
  aggregate), hottest link NoC0/NoC1 and hottest DRAM port:

  | layout | today s0 0.60 | free-port remap s0 0.60 | free-port remap s0 0.55 |
  |---|---|---|---|
  | 5 x 2 | 54 / 52, port 33 | 54 / 40 | 49 / 45, port 30 |
  | 4 x 2 (y2-9) | 54 / 63 | 54 / 50 | 50 / 56 |
  | 5 + 5 | 54 / 52 | 54 / 42 | 50 / 47 |
  | 3+3+2+2 | 54 / 52 | 54 / 44 | 50 / 49 |
  | 4+3+3 | 54 / 61 | 54 / 44 | 49 / 50 |

  (nearest-of-3 as in r06-b03-a01 gives similar link loads but a 55 GB/s hottest port.)
  The free-port remap alone takes ~20% off NoC 1's hottest link but leaves NoC 0 as the max at
  0.60; it only pays when the split moves to ~0.55-0.56, where both NoCs sit at ~49-50
  (-7..-10% on the hottest link) with no port above today's.

## Mechanism
1. Host (program factory): r06-b03-a01's guarded per-core remap (BH, 8 DRAM channels, translated
   coords match `(17 + b/4, 12 + 3(b%4) + port)`, no tensor prefetcher / DRISC, no streaming
   profiler; `DS_NO_DRAM_EP_REMAP=1` turns it off), but the candidate tiles per bank are only
   {port 1 (default NoC-1 endpoint), the free port} where free = the port that is neither port 1
   nor the bank's NoC-0 endpoint, taken from `get_preferred_worker_core_for_dram_view`-equivalent
   yaml data [2,0,0,0,2,2,2,2] (verified on the host against the soc descriptor's NoC-0 DRAM
   endpoint through `virtual_core_from_logical_core` / the allocator's dram channel coords; on a
   mismatch the remap is off). Choice per core row: fewer (row_tile - row_core) mod 12 hops.
   8 writer runtime args after the phase-A sems, writer define `RE_DRAM_EP_REMAP`.
2. Writer: zero per-tile cost. The host passes 4 words = (remapped - default) mod 2^32 of the
   writer's firmware NoC-1 DRAM bank table row (8 x uint16 xy, BRISC-local memory); a noinline
   `dram_port_remap_toggle<arg>()` (weight_runs.hpp) adds them to the table and negates them in place.
   It is called right before and right after each NoC-1 weight read loop (UP_SPLIT up rows,
   DOWN_SPLIT tail tiles), so the accessor reads in between resolve to the chosen tiles and every
   other NoC-1 DRAM access (output drain etc.) sees the default table. (The first eval used
   r06-b03-a01's per-run 8-way coordinate compare instead and overflowed the kernel config buffer on
   kimi-k2.7 by 128 B; see reflection.)
3. Split, only when the remap is on: `up_head_rows = (w + 4) / 8` (w=16: 3 -> 2 of 16, s0 0.594 ->
   0.5625; w=8: 2 -> 1) and `d_head_tiles = (11 N + 10) / 20` (N=21: 13 -> 12, N=18: 11 -> 10,
   N=11: 7 -> 6; N=12 and N=9 unchanged).
4. Size enabler: the profiler-only MATH perf counters (`compute/re_hwprof.hpp`, r03-b02-a03's
   measurement node) become opt-in (`RE_HWPROF`), as r04-b02-a02 did on its branch: -1.0 KB text
   on TRISC1 in profiled builds. No effect on non-profiled builds; the measured MATH thread loses
   a few counter reads per window and one L1 load per K-block.
Files: `unified_routed_expert_ffn_program_factory.cpp`, `kernels/weight_runs.hpp`,
`kernels/dataflow/unified_routed_expert_ffn_writer.cpp`, `kernels/compute/re_hwprof.hpp`.

## Why this is not a repeat
- Repair of r06-b03-a01 along its own next #2 and #3: the NoC-0 endpoint tile is never used for
  NoC-1 reads (removes the port-concentration cause (b)), and the split moves only to ~0.56, not
  0.53/0.50 (smaller step on cause (a), chosen from the corrected model's balance point).
- Built on r06-b04-a01, whose bottom-row senders now multicast down on NoC 1, so NoC 1 carries
  more down multicast than when r06-b03-a01 ran; relieving NoC-1 read responses matters more.
- Not the gate/up wrap fix (r06-b04-a01 next #1) and not a planner change.

## Expected effect and risk
- If the corrected model holds: GU and DN -3..-7% on the DRAM-bound shapes (dsv3, glm, k2.7, k3,
  dsv4-pro, dsv4-flash, minimax-m3), mistral / gpt-oss ~0. +2..+4% geomean.
- If r06-b03-a01's GU loss was mostly the split (cause a), GU loses a few %: visible as GU up on
  the top rows (rows.py / zones per row) while DN improves.
- Bit-identical (same GDDR addresses). A wrong tile would give garbage (accuracy_fail) or a hang.
- Writer size +~170 B (as in r06-b03-a01); h7168 headroom is a few hundred bytes.
