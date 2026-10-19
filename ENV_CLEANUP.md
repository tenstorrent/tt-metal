# Flat routed expert / combine: env cleanup

Inventory of every env read in the flat routed expert op (C++ plan, program factory, DRISC, the with-program /
combine-overlap ops) and the combine (sparse_combine_rf.py), with the class each one gets:
- **a** - a production choice: a typed option (C++ `FlatLayoutOptions` in `FlatRoutedExpertConfig`, part of the plan
  key and the program hash; nanobind kwargs; Python presets), default = today's resolved value, logged once.
- **b** - a perf / debug probe: kept, but read in one place into one typed struct (`FlatProbes` / `FlatDebug`;
  `CombineProbes` on the Python side), clearly marked as probes - never set by model code.
- **c** - a dead experiment whose result is recorded in a worklog: deleted (host read + kernel branch).

## C++ (121 env vars)
| env | read at | what (comment at the read) | default | class | home |
|---|---|---|---|---|---|
| `FLAT_CMB_LOG` | flat_combine_overlap_device_operation.cpp:103; flat_routed_expert.cpp:205 |  | off | b | FlatDebug (debug output) |
| `FLAT_WP_KEYLOG` | flat_with_program_device_operation.cpp:19 | (debug: the cache key of every miss) | off | b | FlatDebug (debug output) |
| `FLAT_WP_LOG` | flat_with_program_device_operation.cpp:78 |  | off | b | FlatDebug (debug output) |
| `MIMO_FL_CMB_DRISC` | fre_plan.cpp:538 | DRISC inside the combine overlaps: the program factory plans with cmb_rt set (the host weight-layout plan does not) | false | a | FlatLayoutOptions.drisc_in_overlap |
| `MIMO_FL_CMB_EARLY` | fre_program_factory.cpp:457 | (perf probe: combine released at launch, garbage) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_CMB_NO_GO` | fre_program_factory.cpp:451; flat_routed_expert.cpp:411 | (flat_with_program: cumulative counts, no go) | false | a | FlatRoutedExpertConfig.cmb_no_go (set by flat_with_program) |
| `MIMO_FL_CMB_SLOT_ORDER` | flat_combine_overlap_device_operation.cpp:43; fre_program_factory.cpp:462; flat_routed_expert.cpp:410 | Row-major y: combine walks every chip's schedule entries (an expert, or one chunk of a pinned one) in the order the flat expert computes them; its readers rebui | false | a | FlatRoutedExpertConfig.cmb_slot_order (set by flat_with_program) |
| `MIMO_FL_COLS` | fre_plan.cpp:804; fre_plan.cpp:804 | MIMO_FL_COLS=n (probe): down cores only in columns [0, n), e.g. 11 on a 12-column grid to reproduce the 11 x 10 | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DCH` | fre_plan.cpp:969 | (perf probe: down chains) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DN_NOC0` | fre_program_factory.cpp:1330 | perf probe (read at program build): MIMO_FL_DN_NOC0=1 - the down cores' h chain / y writes on NOC0, their weight reads on NOC1 (default: the other way round) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DN_NOREAD` | fre_program_factory.cpp:376 | traffic probes (read at program build; garbage outputs): MIMO_FL_GU_NOREAD / MIMO_FL_DN_NOREAD - no gate/up / down weight reads; MIMO_FL_H_NODATA - no h data fr | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRING` | fre_plan.cpp:591 | (perf probe: the down weight ring in experts) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC` | flat_drisc.cpp:56 | default on where the device has programmable DRAM cores (Blackhole DRISC firmware); MIMO_FL_DRISC=0: off | auto (on with DRISC firmware) | a | FlatLayoutOptions.drisc |
| `MIMO_FL_DRISC_ALT_BURST` | flat_drisc.cpp:288 | (probes: the alternate tiles - each bank's NOC1 endpoint, whose DMA shares the port with the NOC1 DRAM reads (the x relays) - with their own DMA depth / burst s | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_ALT_DMA_MAX` | flat_drisc.cpp:285 | (probes: the alternate tiles - each bank's NOC1 endpoint, whose DMA shares the port with the NOC1 DRAM reads (the x relays) - with their own DMA depth / burst s | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_ALT_MAXCNT` | fre_program_factory.cpp:547 | switch mode: the alternate tiles serve calls with counts up to this (dual or not: without it a non-dual pure plan ran every bank's two slots on one tile, ~21-28 | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_ASSIGN` | fre_plan.cpp:1082 | the forwarder: the reader (NOC1), or its DRISC (DRAM tile, writes on NOC0) (perf probe MIMO_FL_DRISC_ASSIGN=0: DRISC mode with the readers' assignment) (pure wi | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_BURST` | flat_drisc.cpp:236 | (probe: DMA AXI burst beats; 0 = untouched) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_DBG` | fre_plan.cpp:1019; fre_plan.cpp:1100 |  | off | b | FlatDebug (debug output) |
| `MIMO_FL_DRISC_DMA_MAX` | flat_drisc.cpp:239 | (probe: DMA reads in flight per stream) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_DRAIN` | flat_drisc.cpp:244 | (probes, V4.1 E 24 one tile per bank, delivery isolated, M 80: 59.3 us / expert; DRAIN 60.5, FREE_SENT 62.6: one tile's DMA + writes are bounded by the tile its | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_DUAL` | fre_plan.cpp:685 |  | auto (dual iff reader down columns) | a | FlatLayoutOptions.drisc_dual |
| `MIMO_FL_DRISC_FREE_SENT` | flat_drisc.cpp:247 | (probe: a block slot is refilled once its writes left L1) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_JOB1_NOC` | flat_drisc.cpp:182 | job 1, the free tile's second slot (MIMO_FL_DRISC_JOB1_NOC): NOC0, but NOC1 in pure mode with reader down columns (K2.7 E12 one tile, M 512 / 2048: 128.9 / 449. | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_MAXCNT` | fre_program_factory.cpp:537 | dual DRISC slots (plan.dual): the per-call switch, the largest per-expert count served by the DRAM cores (MIMO_FL_DRISC_MAXCNT; default 256 tokens, 128 with rea | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_MPFE` | flat_drisc.cpp:152 | per call, the GDDR arbitration weights (se_drisc.hpp SD_MPFE). Default: untouched, except plans with reader down columns, whose extra Tensix DRAM reads share th | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_NOC1` | flat_drisc.cpp:183; flat_drisc.cpp:185 | job 1, the free tile's second slot (MIMO_FL_DRISC_JOB1_NOC): NOC0, but NOC1 in pure mode with reader down columns (K2.7 E12 one tile, M 512 / 2048: 128.9 / 449. | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_PROF` | flat_drisc.cpp:253 | (debug: per-section cycles of the last call, in the dump) | off | b | FlatDebug (debug output) |
| `MIMO_FL_DRISC_PURE` | flat_drisc.cpp:122 | 63.4 / 101.2 -> 41.5 / 44.3 / 49.5 / 60.0 / 101.2, GLM-Flash 32.9 / 35.3 / 38.7 / 47.9 / 68.3 -> 33.3 / 33.8 / 38.2 / 46.4 / 68.3; K2.7 E12, reader down columns | -1 = auto (pure without reader down columns) | a | FlatLayoutOptions.drisc_pure (preset DRISC_ONLY) |
| `MIMO_FL_DRISC_RD_DOWN` | fre_plan.cpp:700 | the freed cells as down cores (MIMO_FL_DRISC_RD_DOWN, default: only with every slot on DRISC; next to the remaining Tensix readers they slow large M: hybrid 204 | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_RD_SWAP` | fre_plan.cpp:693; fre_plan.cpp:693 | (probe MIMO_FL_DRISC_RD_SWAP=1: the reader cells' halves swapped, so the Tensix slots 8.. run on the cells next to the DRAM columns) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_SLOTS` | fre_plan.cpp:546 | default: one DRISC slot per bank (the hybrid: GLM-Flash us / expert vs Tensix readers M 64 34.0 / 35.4, 256 49.0 / 49.7, 512 71.3 / 70.6, 2048 273.9 / 273.4); a | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_STREAM_SPLIT` | flat_drisc.cpp:277 | a bank's two tiles DMA on different streams: both on stream 0 with ~7 reads in flight each, next to heavy NOC1 DRAM reads through the bank's NOC1 endpoint (MIMO | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_TILES` | flat_drisc.cpp:126 |  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_UB` | flat_drisc.cpp:250 | (probe: blocks per DRISC unit) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DRISC_XNOC` | fre_program_factory.cpp:224; fre_program_factory.cpp:224; fre_program_factory.cpp:708 |  | - | c | **deleted** (result: TS_HANG_WORKLOG (full separation fix; superseded by unicast x)) |
| `MIMO_FL_DUMP_RT` | fre_program_factory.cpp:1531 |  | off | b | FlatDebug (debug output) |
| `MIMO_FL_DW_BATCH` | fre_program_factory.cpp:1373; fre_program_factory.cpp:1373 | (MIMO_FL_DW_BATCH: down blocks per read barrier, default DW_BATCH; read at program build) | 0 = DW_BATCH | a | FlatLayoutOptions.dw_batch |
| `MIMO_FL_DW_ROT` | fre_plan.cpp:1214 | The down cores' weight reads rotate over the DRAM banks instead of one bank per core (bank-pinned streams ran at 55% of DRAM peak alone, rotating 65%; worklog 8 | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DYN_PULL` | fre_program_factory.cpp:345 | The schedule's inputs (counts, regions, expert ids: one DRAM page each, all in bank 0) read once per call: reader 0's BRISC (se_reader) reads them, compacts thi | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_DYN_ZONE` | fre_program_factory.cpp:360 | (perf probe: a profiler zone around se_dyn_load) | off | b | FlatDebug (debug output) |
| `MIMO_FL_DZ` | fre_program_factory.cpp:411 | detail zones in the down compute (a sample of sub-blocks) | off | b | FlatDebug (debug output) |
| `MIMO_FL_FIDELITY` | fre_program_factory.cpp:69 | the expert matmuls' fidelity: the config's (default LoFi); MIMO_FL_FIDELITY=LoFi / HiFi2 / HiFi3 / HiFi4 overrides it for every flat expert (probes) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_FWD_VC` | flat_drisc.cpp:265; fre_program_factory.cpp:417 | (perf probe: the weight writes' unicast VC) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_GU_BF16` | fre_plan.cpp:351 | (perf probe: plain bf16 gate/up DEST, half-sync 8 tiles; inaccurate) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_GU_FULL_SYNC` | fre_program_factory.cpp:1235 | perf probe (read at program build): MIMO_FL_GU_FULL_SYNC=1 - fp32 gate/up in a full-sync DST (8 tiles: the whole MT x 2 NP sub-block in one K loop, no row passe | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_GU_NOREAD` | flat_drisc.cpp:262; fre_program_factory.cpp:375 | (perf probe: no gate/up weight DMA, garbage) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_GU_NREG` | fre_plan.cpp:959 | gate/up weight ring: 2 expert slices (with pinning: the pinned expert's and one cycling); MIMO_FL_GU_NREG (probe): more regions, so more small experts stream in | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_GU_SPLIT` | fre_plan.cpp:84; fre_plan.cpp:110 | MIMO_FL_GU_SPLIT with np 3 / 4 (probe), or an uneven split (np34): the gate/up sub-block in a full-sync fp32 DST (8 tiles: one row tile of 2 np <= 8 tiles per M | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_HBUF` | fre_plan.cpp:1160 | (perf probe: cap the h unit buffers) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_HG` | fre_plan.cpp:288; fre_plan.cpp:288 |  | auto | a | FlatLayoutOptions.half_width |
| `MIMO_FL_HG_AUTO` | fre_plan.cpp:386 | no even split: 16-column units (half-width gate / up tiles, MIMO_FL_HG_AUTO=0: off) with NU units a core, the most gate/up cores that are a multiple of the 16 r | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_HG_DRISC` | fre_plan.cpp:574 | 16-column units: DRISC as the other plans without reader down columns (pure slots, Tensix readers above 96 tokens / expert; MIMO_FL_HG_DRISC=0: Tensix readers o | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_HG_NU` | fre_plan.cpp:387; fre_plan.cpp:434 | no even split: 16-column units (half-width gate / up tiles, MIMO_FL_HG_AUTO=0: off) with NU units a core, the most gate/up cores that are a multiple of the 16 r | 0 = auto | a | FlatLayoutOptions.hg_units |
| `MIMO_FL_HPIECES` | fre_program_factory.cpp:400; fre_program_factory.cpp:400 | perf probe (read at program build, not hashed): MIMO_FL_X_RESIDENT=1 - x never moves. The relay kernels (read / tilize / multicast / helpers) return at once and | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_HU` | fre_plan.cpp:552 | h units (MIMO_FL_HU = row tiles per unit): one M-group, one subgrid, no reader down columns | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_H_NODATA` | fre_program_factory.cpp:377 | traffic probes (read at program build; garbage outputs): MIMO_FL_GU_NOREAD / MIMO_FL_DN_NOREAD - no gate/up / down weight reads; MIMO_FL_H_NODATA - no h data fr | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_H_VC` | fre_program_factory.cpp:414 | (perf probe: the h exchange's unicast VC) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_LINK_DEPTH` | fre_program_factory.cpp:402 | (perf probe: h pieces in flight per chain link) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_MT` | fre_plan.cpp:563 | (perf probe: cap the sub-block's row tiles) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_ND` | fre_plan.cpp:911; fre_plan.cpp:978 | (perf probe: cap the down cores; the rest stay idle) | 0 = plan | a | FlatLayoutOptions.max_down |
| `MIMO_FL_NH` | fre_plan.cpp:581 | (perf probe: x helpers per rectangle, 1-4; 2 on GLM-Flash: M 512 / 2048 83.5 / 326.3 vs 70.6 / 273.4, the helpers take down cells) | 0 = plan (1) | a | FlatLayoutOptions.x_helpers |
| `MIMO_FL_NO_ACT` | fre_program_factory.cpp:1260 | probe: no gate/up activation (wrong h), the bare matmul + pack | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_NO_MATH` | fre_program_factory.cpp:378; fre_program_factory.cpp:383 | traffic probes (read at program build; garbage outputs): MIMO_FL_GU_NOREAD / MIMO_FL_DN_NOREAD - no gate/up / down weight reads; MIMO_FL_H_NODATA - no h data fr | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_NO_SHAPE` | fre_plan.cpp:465 | Full grid, one subgrid, a split of fewer gate/up cores than the fixed rectangles' 64 (K3: 48, 16 cells idle): shape the rectangles to the split as with the rows | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_PCD_R` | fre_plan.cpp:974 | (perf probe: the readers' down columns) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_PIN_RATIO` | fre_program_factory.cpp:99 | pin only a dominant expert (se_dyn_pin); MIMO_FL_PIN_RATIO (read at program build; 0: always pin) | 8 | a | FlatLayoutOptions.pin_ratio |
| `MIMO_FL_RDN_HI` | fre_plan.cpp:1244 | (perf probe MIMO_FL_RDN_HI=1: tails among readers 8.. only, as DRISC mode's default hybrid) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_RDOWN` | fre_plan.cpp:525; fre_plan.cpp:528 | (perf probe: force the readers' down columns on / off) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_RD_SAMECOL` | fre_plan.cpp:331 | MIMO_FL_RD_SAMECOL (with rows capped): a bank's second reader in its first reader's column (nearest free row) instead of the column east of it, so columns 1 and | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_RELAY_HELP` | fre_plan.cpp:780 | MIMO_FL_RELAY_HELP="x,y;x,y" (west, east; one helper each, MIMO_FL_NH=1): the helpers on these free cells - next to moved primaries (MIMO_FL_RELAY_PRIM), so the | - | c | **deleted** (result: TS_HANG_WORKLOG waves 6-8 (as RELAY_PRIM)) |
| `MIMO_FL_RELAY_PRIM` | fre_plan.cpp:761 | MIMO_FL_RELAY_PRIM="x,y;x,y" (west, east): the two primary relays on these free cells instead - e.g. next to each rectangle's far corner, where a NOC1 multicast | - | c | **deleted** (result: TS_HANG_WORKLOG waves 6-8 (full separation candidate, superseded by unicast x)) |
| `MIMO_FL_RM_CHUNKS` | fre_plan.cpp:1190 | x staging depth probes: MIMO_FL_RM_CHUNKS (relay row-major chunks), MIMO_FL_SB_SLOTS (relay tilized super-blocks), MIMO_FL_X_SLOTS_MAX (the gate/up x ring, if i | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_ROWS` | fre_plan.cpp:299; fre_plan.cpp:300; fre_plan.cpp:302 | MIMO_FL_ROWS=y0,y1 (probe): plan on grid rows [y0, y1] only, leaving the rest of the chip to another op (e.g. combine_fabric2d's senders / untilizers on rows 0- | all rows | a | FlatLayoutOptions.rows |
| `MIMO_FL_SB_SLOTS` | fre_plan.cpp:1193 | x staging depth probes: MIMO_FL_RM_CHUNKS (relay row-major chunks), MIMO_FL_SB_SLOTS (relay tilized super-blocks), MIMO_FL_X_SLOTS_MAX (the gate/up x ring, if i | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_SE_DBG` | fre_program_factory.cpp:1160 | debug probe MIMO_FL_SE_DBG=1: the receivers / x forwarders DPRINT their counters every 4M loop passes | off | b | FlatDebug (debug output) |
| `MIMO_FL_SKIP_COLS` | fre_plan.cpp:809 | MIMO_FL_SKIP_COLS="x,x,.." (probe): no down cores in these columns, e.g. 6 on a Galaxy chip (12 x 10, east readers in column 7): the column a p150's 11 x 10 doe | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_TZ_SKIP` | fre_program_factory.cpp:976 | MIMO_FL_TZ_SKIP: the relays' tilizers only do the CB handshakes; MIMO_FL_XMC_NODATA: the multicasters send only the XARR counts (garbage x on the gate/up cores) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_UNEVEN` | fre_plan.cpp:418; fre_plan.cpp:418 | no even split (e.g. I 2304: 72 columns, no It / np a multiple of the 16 readers): an uneven one, the pair-sets a multiple of the banks (MIMO_FL_UNEVEN=0: off) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_WAITZ` | fre_program_factory.cpp:408 | gate/up x / weight wait zones on sub-block WAITZ | off | b | FlatDebug (debug output) |
| `MIMO_FL_W_NOFWD` | flat_drisc.cpp:256; fre_program_factory.cpp:363 | (perf probe: weight blocks signalled, not written to the gate/up cores) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_W_NOREAD` | flat_drisc.cpp:259; fre_program_factory.cpp:368 | (perf probe: no weight DMA, garbage) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_X1` | fre_program_factory.cpp:144; fre_program_factory.cpp:144 | MIMO_FL_X1=1 (prototype, read at program build): x is read from DRAM once. Rectangle 0's relays (primary + helpers) read and tilize every super-block and the pr | false | a | FlatLayoutOptions.x_once |
| `MIMO_FL_X2` | fre_program_factory.cpp:152; fre_program_factory.cpp:152 | MIMO_FL_X2=1: x read once with both multicasters: rectangle k's relays read and tilize the super-blocks g with g % 2 == k (helpers sharing them as before) and i | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XDOWN` | fre_plan.cpp:832 | MIMO_FL_XDOWN="x,y;x,y;.." (probe, with the rows capped): extra down cores outside the rows, e.g. the grid row 0 cells combine leaves free when it runs on row-m | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XFWD` | fre_plan.cpp:1198; fre_plan.cpp:1198 |  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XHELP_ALL` | fre_program_factory.cpp:945 | (probe MIMO_FL_XHELP_ALL=1: the helpers read and tilize all of x, the primaries only multicast - and read it all on small-M calls, the role split's) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XHELP_NOC0` | fre_program_factory.cpp:942 | (probe MIMO_FL_XHELP_NOC0=1: the helper relays read x on NOC0 - through each bank's NOC0 endpoint, not the NOC1 endpoint a DRISC tile DMAs through - in dynamic- | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XIDX_ALL` | fre_plan.cpp:1203; fre_plan.cpp:1203; fre_program_factory.cpp:897 | (MIMO_FL_XIDX_ALL=1: room for the call's token index; no gain measured - K3 E1 / E12 / E56 within 0.1%) | - | c | **deleted** (result: "no gain measured - K3 E1 / E12 / E56 within 0.1%" (comment at the read)) |
| `MIMO_FL_XMC_NOC1` | fre_program_factory.cpp:713; fre_program_factory.cpp:713 | MIMO_FL_XMC_NOC1=1: only the x relays move (multicast + helper sends on NOC1, x reads on NOC0); the gate/up receivers stay on NOC1 - the multicast off the NoC t | - | c | **deleted** (result: TS_HANG_WORKLOG (split config wedged NOC1; superseded)) |
| `MIMO_FL_XMC_NODATA` | fre_program_factory.cpp:992 |  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XMC_POSTED` | fre_program_factory.cpp:995 | (probe: posted x multicasts - no per-destination acks) | - | c | **deleted** (result: TS_HANG_WORKLOG (posted multicast: no fix)) |
| `MIMO_FL_XMC_UNLINKED` | fre_program_factory.cpp:998 | (probe: the x multicasts without linked chains) | - | c | **deleted** (result: TS_HANG_WORKLOG (unlinked multicast: no fix)) |
| `MIMO_FL_XPRIM_ONLY` | fre_program_factory.cpp:424 | x-path probes: MIMO_FL_XSOLO=N - x resident for every consumer, N relays only read their share of one copy of x (no tilize / multicast); MIMO_FL_XPRIM_ONLY=1 (w | - | b | FlatProbes - kept (x-path probes entangled with X_RESIDENT in 4 relay / receiver kernels; listed for Marko) |
| `MIMO_FL_XRD_BATCH` | fre_program_factory.cpp:928 |  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XRD_BIG` | fre_program_factory.cpp:921 | (probe MIMO_FL_XRD_NOC=0/1: the x readers' NoC) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XRD_DYN` | fre_program_factory.cpp:717 | (probe MIMO_FL_XRD_DYN=1: the relays' kernels in dynamic-NoC mode - per-RISC command buffers and counters - so their x reads and multicasts may share a NoC, e.g | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XRD_NOC` | fre_program_factory.cpp:918; fre_program_factory.cpp:919 | (probe MIMO_FL_XRD_NOC=0/1: the x readers' NoC) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XRD_SKIP` | fre_program_factory.cpp:924 | (probe MIMO_FL_XRD_NOC=0/1: the x readers' NoC) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_XSOLO` | fre_program_factory.cpp:423; fre_program_factory.cpp:423 | x-path probes: MIMO_FL_XSOLO=N - x resident for every consumer, N relays only read their share of one copy of x (no tilize / multicast); MIMO_FL_XPRIM_ONLY=1 (w | - | b | FlatProbes - kept (x-path probes entangled with X_RESIDENT in 4 relay / receiver kernels; listed for Marko) |
| `MIMO_FL_XSOLO_IDS` | fre_program_factory.cpp:1055 | MIMO_FL_XSOLO_IDS=a,b,..: which relays (indices into p.relays) take the N slots (default 0..N-1) | - | b | FlatProbes - kept (x-path probes entangled with X_RESIDENT in 4 relay / receiver kernels; listed for Marko) |
| `MIMO_FL_XSOLO_K` | fre_program_factory.cpp:907 | perf probes (read at program build; not in the program hash, so a process uses one setting): MIMO_FL_XRD_SKIP=n: the relays read no x rows from DRAM (garbage ti | - | b | FlatProbes - kept (x-path probes entangled with X_RESIDENT in 4 relay / receiver kernels; listed for Marko) |
| `MIMO_FL_XSOLO_NOIDX` | fre_program_factory.cpp:913 |  | - | b | FlatProbes - kept (x-path probes entangled with X_RESIDENT in 4 relay / receiver kernels; listed for Marko) |
| `MIMO_FL_XSOLO_SEQ` | fre_program_factory.cpp:910 | row mask (4095: 4096 rows; 7: 8 rows) | - | b | FlatProbes - kept (x-path probes entangled with X_RESIDENT in 4 relay / receiver kernels; listed for Marko) |
| `MIMO_FL_X_CHAIN` | fre_program_factory.cpp:184 |  | auto: snakes where the plan allows, else multicast | a | FlatLayoutOptions.x_path |
| `MIMO_FL_X_CHAIN_DEPTH` | fre_program_factory.cpp:1007 | (relay: head-write batches in flight, 1..8) | 0 = kernel default 4 | a | FlatLayoutOptions.x_relay_depth |
| `MIMO_FL_X_CHAIN_NOC` | fre_program_factory.cpp:191; fre_program_factory.cpp:225; fre_program_factory.cpp:1205 |  | 0 (NOC0) | a | FlatLayoutOptions.x_fwd_noc |
| `MIMO_FL_X_CHAIN_SNAKES` | fre_program_factory.cpp:188 |  | 1 | a | FlatLayoutOptions.x_snakes |
| `MIMO_FL_X_FWD_DEPTH` | fre_program_factory.cpp:192 | (0: se5_recv's default, 6) | 0 = auto (3 with the NCRISC forwarder) | a | FlatLayoutOptions.x_fwd_depth |
| `MIMO_FL_X_FWD_NC` | fre_program_factory.cpp:190 |  | true (modes 2/3) | a | FlatLayoutOptions.x_fwd_ncrisc |
| `MIMO_FL_X_PIECES` | fre_program_factory.cpp:189 |  | 0 = auto (4 / 2 / 1) | a | FlatLayoutOptions.x_pieces |
| `MIMO_FL_X_RESIDENT` | fre_program_factory.cpp:425 | x-path probes: MIMO_FL_XSOLO=N - x resident for every consumer, N relays only read their share of one copy of x (no tilize / multicast); MIMO_FL_XPRIM_ONLY=1 (w | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_X_SLOTS_MAX` | fre_plan.cpp:1125 |  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_YRM_NOWRITE` | fre_program_factory.cpp:596 | perf probes (read at program build): MIMO_FL_YRM_NOWRITE - the writers drop the y writes; MIMO_FL_YRM_PACKTILE - the down compute packs tiles instead of untiliz | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_YRM_PACKTILE` | fre_program_factory.cpp:605 |  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_YRM_SELF` | fre_program_factory.cpp:599 | (perf probe: y written onto itself in L1, no DRAM) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_YRM_VC` | fre_program_factory.cpp:602 | MIMO_FL_YRM_PACKTILE - the down compute packs tiles instead of untilizing (y is garbage either way) | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_Y_L1` | fre_program_factory.cpp:445; fre_program_factory.cpp:454; flat_routed_expert.cpp:432 |  | false | a | FlatRoutedExpertConfig.y_l1 (set by flat_with_program / overlap) |
| `MIMO_FL_Y_NC` | fre_program_factory.cpp:1336 | y writes from the NCRISC on its NOC (se6_dw.cpp) instead of the BRISC, whose NOC carries the h chain (one subgrid): bfp8 x + bf16 h at M 512 / 2048 / 5120 -9 /  | unset | b | FlatProbes (perf / traffic probe; default = today) |
| `MIMO_FL_Y_NOC1` | fre_program_factory.cpp:1341; fre_program_factory.cpp:1341 | MIMO_FL_Y_NOC1=1 (with the NCRISC's y writes): those writes on the other NoC than its down-weight reads - the y traffic off the x relays' multicast NoC (NOC0),  | - | c | **deleted** (result: TS_HANG_WORKLOG (y writes on NOC1: no fix; superseded)) |
| `MIMO_FL_ZONES` | fre_program_factory.cpp:405 | device-profiler zones in the kernels that have them | off | b | FlatDebug (debug output) |

## Python (combine)
| env | read at | what | default | class | home |
|---|---|---|---|---|---|
| `SPC_PIECE` | sparse_combine_rf.py:38 | row bytes per fabric packet (wider rows split) | 7168 | a | `piece_max=` (exists; harness passes it) |
| `SPC_QP` | sparse_combine_rf.py:997,1007 | a port's L1 hand-off injection ring slots | 64 | a | `inj_slots=` |
| `SPC_D` | sparse_combine_rf.py:1005 | ring depth override (after the arena fit) | fit | a | `depth=` (exists; the override becomes `depth_exact=`) |
| `SPC_PROF`, `SPC_TPROF` | 1099, 1115 | profile words / per-call timeline | off | b | `CombineProbes` |
| `SPC_PACE`, `SPC_INJB`, `SPC_ABL`, `SPC_CREDTHR`, `SPC_NOYREAD`, `SPC_NODRAIN` | 1100-1128 | pacing / injection batch / ablation / credit threshold / no y read / no drain | as today | b | `CombineProbes` |
| `FC_PLAN_V` | combine_plan.py:1203 | planner v1 vs v2 | 2 | c | delete v1 (v2 is the only one used since 10-17) |

## Callers that set env (move to options)
- `models/demos/glm53_flash_d_p/tt/experts_ts.py` FLAT_ENV (CMB_DRISC 1, DRISC_DUAL 0, ND 32, NH 1) -> preset `GLM_FLASH`.
- `models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`: K3_GU64 setdefaults (CMB_DRISC 1, DRISC_DUAL 0, HG 1,
  HG_NU 3, ND 28, NH 2 / 1) -> preset `K3_GU64`; scoped `MIMO_FL_X1` / `MIMO_FL_DW_BATCH` / `K3_EXP_ENV` -> per-call
  options; `MIMO_FL_Y_L1` set before every call -> the with-program op sets `cfg.y_l1`; `K3_D` -> constructor arg.
- `flat_routed_expert.cpp` `setenv(MIMO_FL_CMB_SLOT_ORDER / CMB_NO_GO / Y_L1)` -> config fields.
- `models/demos/deepseek_v3_d_p/tests/perf/scratch_flat_combine_rf.py`: FC_* stay env (test harness) but become typed
  options passed into the ops; MIMO_FL_* given to a harness run are parsed into options in one place.
- Tests: test_flat_combine_overlap.py (MIMO_FL_ROWS), test_flat_bench.py (ROWS / XDOWN / RD_SAMECOL), test_flat_uneven_split.py
  (HG), hang/test_noc_wedge_probe.py, glm53 probes -> options / probes.

## Not converted (listed for Marko)
- `flat_expert.py`'s `FlatExpert` (the Python research builder, py_kernels): ~76 env reads; used only by
  test_flat_stage_probe.py and the 11 x 10 unit test's reference. Left as is.

## Outcome (branch mstaletovic/flat-env-cleanup)

- **a (typed)**: `FlatLayoutOptions` in `FlatRoutedExpertConfig` (plan cache key, program hashes; nanobind
  `ttnn._ttnn.operations.bringup.FlatLayoutOptions`, the ops' `layout=`), presets in
  `ttnn/ttnn/bringup/flat_routed_expert_ttnn/flat_layouts.py`: `DRISC_ONLY` (pure DRISC-only: drisc_in_overlap,
  drisc_dual False, drisc_pure 1), `GLM_FLASH`, `GLM_FLASH_AG`, `k3_gu64(acc_on_ports)`; `flat_layout(...)` builds an
  options object, `layout_from_env(...)` parses the old MIMO_FL_* names for test harnesses only. The op-internal flags
  (Y_L1, CMB_NO_GO, CMB_SLOT_ORDER) are config fields (`flat_routed_expert_with_program(y_l1=...)`). The plan log line
  prints the resolved layout once per plan.
- **b (probes)**: `FlatProbes::from_env()` (C++, one place, raw values parsed where used: `cstr / on / u32`),
  `CombineProbes.from_env()` (Python).
- **c (deleted)**: MIMO_FL_RELAY_PRIM / RELAY_HELP, DRISC_XNOC, XMC_NOC1, Y_NOC1 (+ SE_YRM_NOC), XMC_POSTED /
  XMC_UNLINKED, XIDX_ALL (+ XRD_IDX_ALL): the hang hunt's superseded fixes / probes (TS_HANG_WORKLOG.md waves 4-9:
  the unicast x path replaced them) and a no-gain probe (comment at the read: K3 E1 / E12 / E56 within 0.1%). The relay
  keeps the (unused) token-index arena arg so the runtime-arg order and the programs stay identical.
- Combine: `piece_max` (was SPC_PIECE; module default 7168), `inj_slots` (SPC_QP), `ring_depth` (SPC_D).
- Callers: experts_ts (GLM_FLASH), experts_ag (GLM_FLASH_AG), the K3 block (flat_layout / gu64 / combine_depth /
  x_once / dw_batch / layout_overrides instead of flat_env, K3_GU64, K3_D, K3_EXP_X1 / DW_BATCH / ENV), the fused
  harness and the tests.

## Behaviour notes

- The plan cache (`flat_routed_expert_plan`) ignored the environment: a process that changed a MIMO_FL_* layout knob
  after its first plan reused the old plan (e.g. zz_scratch_flat_cache_key's second MIMO_FL_ND). The layout is now in
  the key.
- Process-wide env coupling is gone: the K3 block's / GLM's `setdefault` leaked into every later FlatRoutedExpert
  (a second K3 block's shared expert got the routed experts' layout; the first did not, being built before the
  setdefault). Now each op has its own layout; the defaults reproduce the first-block / fresh-process behaviour.
- flat_with_program's setenv(MIMO_FL_Y_L1 / CMB_NO_GO / CMB_SLOT_ORDER) was sticky for the process (a later
  flat_combine_overlap call would have inherited them); now per call.
- `flat_drisc_pure()` / `flat_drisc_tiles()` still read a process-wide value set by the last plan built (the DRISC
  senders are per device): two plans of different pure-ness in one process share it - unchanged, listed for Marko.
- GLM-5.3 (H 6144: reader down columns) cannot run pure DRISC-only (drisc_dual False + drisc_pure 1): the plan
  refuses it ("reader down columns need 7 non-DRISC slots (or dual slots)") - both before and after.
