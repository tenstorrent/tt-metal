# SP fused matmul+collectives and the scaling campaign (reconstructed 2026-09-18 21:55 after the session scratchpad was
# wiped; numbers from memory and the conversation record; the original logs are gone)

Repo /home/imichalak/tenstorrent/tt-metal, branch `imichalak/llama-sp/6-sp-fused-matmul-ccl` (local only, never push;
coordinator commits). Harness: `generated/spfuse/` (git-ignored, durable): env.sh, build.sh (build lock + install),
devrun.sh (device lock + watchdog + reset), bench_sp_train.sh (BATCH= PROFILE= MEMEFF=, impl composed|fused|nocomm),
matrix.sh, kit/ (tracy ops-CSV analyzer), mgd/ (1x4 ring + line). Rules: build only via build.sh; device runs only via
devrun.sh (dangerouslyDisableSandbox); never leave the tree non-compiling; no git state changes by agents; report logs.

## Commits (all local)
c8b6f6938ed tt-train `sp_column_parallel_linear`/`sp_row_parallel_linear`, Composed impl, switch, 1x4 MGDs
27b45bb60c0 matmul slice-schedule mechanism (SP_SLICE_SCHEDULE / SP_AG_WAIT), sp_matmul_fusion_common, transpose fix
0a1d4feae70 matmul_reduce_scatter_sp_async  |  3e6660a66f5 all_gather_matmul_sp_async (+ fused-AG worker-order deadlock fix)
06cf95342bf tt-train Fused wiring  |  9b91b672493 Fused default  |  af7b7477e4d + 9a0f9ad5749 two-queue plumbing (cherry-picks)
Uncommitted in tree: agent S (sp_overlap.{hpp,cpp}, test_sp_overlap.py, tt-train edits) and agent F (ttnn fused-op edits).

## Design in one paragraph
View [B,1,S,X] as [B*T,1,S/T,X]: one matmul sub-batch per (batch, slice). The 2D-mcast matmul loops over batches with
fuse_batch=false; the SP schedule permutes that order (packed rt-arg words), the in0 sender waits on the all-gather's
per-direction slice semaphores (dir0 count k, dir1 count k+1 because the writer signals the local slice on dir1) and
reads local slices from the sharded input; the RS readers wait per slice on an ordinal table. Collective workers take the
bottom `ccl_core_rows` (=2) rows; matmul the rest. Precision: fused == same-config unfused bitwise; vs ttnn.linear's auto
block config PCC 0.99987 (its own bf16 K-block rounding); fp32-acc: 1 bf16 ulp.

## Op-level numbers (1x4 galaxy, 2 links, Llama-8B tp4 S=2048 B=1, HiFi2/bf16acc, us fused vs unfused)
RS: out_proj 376 vs 448 ring (1.19x) / 408 vs 548 line (1.34x); w2 667 vs 789 / 717 vs 884; dgrad_col 390 vs 410 / 415 vs 508.
AG: qkv 467 vs 494 ring (1.06x) / 452 vs 619 line (1.37x); gate_up 1105 vs 1192 / 1115 vs 1301; dgrad_row 395 vs 395 / 380 vs 517.
Standalone 2 links: AG dim2 185 ring / 309 line; RS 234 / 337; all_reduce 414 / 643; matmuls qkv 304, out_proj 216,
gate_up 1002, w2 542. bfp8 payloads: AG 110 ring / 181 line; RS 123 / 309.
Decomposition: sub-batched matmul re-reads the weight per slice (+2-6% AG side, +4-47% RS side, DRAM-bound dgrad worst);
2 rows let the collective run its default workers (1 row: 2 workers/link, AG 235/430); last slice cannot overlap.

## Step-level matrix (s/step, mean of steps 3-5; nocomm = collectives replaced by uninitialised outputs)
| batch | topo | nocomm | composed | fused | collectives | fused recovers |
| 1 | ring | 0.533 | 0.603 | 0.599 | 70 ms (11.6%) | 4 |
| 1 | line | 0.534 | 0.629 | 0.597 | 95 ms (15.1%) | 32 |
| 2 | ring | 0.914 | 1.053 | 1.019 | 139 ms (13.2%) | 34 |
| 2 | line | 0.913 | 1.107 | 1.029 | 194 ms (17.5%) | 78 |
| 5 memeff | ring | 2.397 | 2.907 | 2.882 | 510 ms (17.5%) | 25 |
| 5 memeff | line | 2.397 | 3.074 | 2.907 | 677 ms (22.0%) | 167 |
B=3 composed ring 1.451 (largest batch without recompute; B=5 OOMs, DRAM 4.24/4.27 GB per bank).
Phase split ring (ms): B=1 forward 128 nocomm / 158 composed / 157 fused; backward 297 / 340 / 336. B=5 memeff forward
555 / 674 / 678; backward 1743 / 2128 / 2097. Backward carries 60-77% of the collective cost; recompute adds a 3rd AG.
KEY FINDING: on ring the fused FORWARD recovers ~0 at step level (loses 4 ms at B=5) though HiFi2/bf16acc microbenchmarks
gave +6-8%: training uses HiFi4 + fp32acc + packer_l1_acc -> perf must be timed at that config (agent F).

## Campaign 2 levers
S: two-stream scheduling (CQ1 + CCL sub-device): backward RS overlaps its own wgrad, AG overlaps the previous linear's
deferred wgrad; then forward micro-batch pipelining for B>1. Early datapoint: split grid rows=1 costs nocomm only 3 ms.
F: backward shapes (gate_up dgrad K=7168 RS, w2 dgrad N=3584 AG) + HiFi4 timing; bfp8 payloads (RS output bfp8 = half
bytes, numerics to report); in1 reuse across same-slice sub-batches; wide-N subblock; scope L1-resident RS partial.
Per-site fused/composed policy will be needed (fuse only where net positive at HiFi4).
Lessons: tracy captures the child's stdout (idle watchdog must equal the hard budget); pgrep patterns match your own
shell; the session scratchpad is not durable -> everything now lives in generated/spfuse.

## Agent F STATUS 22:05 (after the wipe; every number below is quoted from the lost logs via the conversation record)
### Repo edits (uncommitted, all compile, installed libs 16:53 = this tree)
1. AG early signal: `all_gather_async/device/kernels/minimal_default_reader.cpp` (define `AG_FUSED_SIGNAL_ON_RECEIVE`,
   fuse_op only): a FORWARDED slice is signalled to the fused consumer as soon as all its chunk semaphores have arrived,
   not after it has been forwarded (the forward loop interleaves the sem waits with CB pushes that block behind the
   writer's local send, so the old signal fired one slice-time late: ring T=4 dir0#1 at 2S = AG end -> the matmul was
   effectively serialised after its local slice; line rank 0 got slices at 2S,3S,3S). Builder flag
   `fused_op_signal_on_receive` (trailing default false) in `all_gather_async_default_program_factory.{cpp,hpp}`; the SP
   op passes true (param `ag_signal_on_receive`, env `TT_SP_AG_SIGNAL_LATE=1` restores the old timing). Standalone AG
   and all_gather_matmul_async unchanged.
2. in1 L1 residency (`SP_IN1_RESIDENT`): `MatmulFusedOpSignaler::sp_in1_resident` (ccl_op_fusion.hpp); the legacy 2D-
   mcast factory sizes the in1 CB to the whole per-core slab (num_blocks x in1 block) when interleaved in1 + bcast_batch
   + one N block + slab + other CBs + 32 KiB <= L1, and defines it for the in1 sender/receiver kernels
   (`reader_bmm_tile_layout_in1_{sender,receiver}_writer_padding.cpp`: after the first pass over K they only
   reserve/push the resident pages, no DRAM read / multicast / handshake; compute kernel untouched). Knob `in1_resident`
   (env `TT_SP_IN1_STREAM=1` disables) on both SP ops and sp_matmul_schedule_test. Resident on this grid: qkv (1 MiB
   slab), out_proj (704 KiB), dgrad qkv/col (1056 KiB), dgrad out_proj/row (768 KiB). NOT resident: gate_up (4.9 MiB),
   w2 (2.5 MiB), gate_up dgrad (4.9 MiB), w2 dgrad (2.5 MiB).
3. Wide-N: `sp_matmul_program_config` picks per_core_N in [ceil(Nt/12), 2*ceil] minimising pcN*(sh+sw)/(sh*sw), but
   only when the slab is resident (first version widened gate_up 19->20 and lost 6%: streamed = DRAM-bound). Effect:
   N=4096 RS shapes 11 (2x1) -> 12 (2x4 bf16acc / 2x2 fp32acc); qkv/dgrad_row unchanged.
4. Tests: B=5 twins of all Llama shapes, the two backward shapes (`llama8b_dgrad_col_gate_up[_b5]`,
   `llama8b_dgrad_row_w2[_b5]`), `perf2` mode (ckc bf16acc|fp32acc x payload bf16|bfp8: fused | unfused | linear |
   sub-batched mm alone | CCL alone | numerics vs fp32 ref and bfp8-vs-bf16), CHECK_SKIP_CASES for the huge B=5 refs.
### Correctness on this tree (2 links): mm_smoke 9 passed; test_sp_matmul_schedule 100 passed; AG ring check 21 passed
(bitwise vs same-config matmul incl. B=5 qkv/dgrad_row, program cache, trace). NOT yet re-run: AG line check, RS ring/
line checks, tt-train test_sequence_parallel / test_sp_linear_ops_1x4 (queued in f_v3.sh).
### Baseline (shipped tree, HiFi2/bf16acc) fused / unfused us, 2 links  [B=1 | B=5]
AG ring: qkv 467/494 | 2327/2353; gate_up 1103/1191 | 5559/5991; dgrad_row 395/396 | 1943/1899.
AG line: qkv 452/618 | 2249/2951; gate_up 1115/1299 | 5613/6409; dgrad_row 380/518 | 1850/2494.
RS ring: out_proj 375/447 | 1737/2055; w2 672/790 | 3332/3579; dgrad_col 390/411 | 1872/1965.
RS line: out_proj 408/548 | 1891/2500; w2 718/884 | 3461/4000; dgrad_col 416/510 | 1950/2404.
1 link: AG ring qkv 679/663, gate_up 1276/1340, dgrad_row 614/562 (B=5 3400/3203, 6394/6656, 3051/2743); AG line
667/893, 1239/1569, 639/795 (3309/4371, 6227/7735, 3144/3919); RS ring out_proj 469/618, w2 719/967, dgrad_col 479/581
(2046/2945, 3372/4448, 2069/2855); RS line 648/834, 780/1171, 653/797 (2999/3937, 3519/5437, 3003/3838).
Sub-batched matmul alone vs linear (B=1 -> B=5): out_proj 245/215 -> 1203/988; w2 573/550 -> 2918/2542; dgrad_col
268/182 -> 1340/960 (the re-read share GROWS with B because ttnn.linear gets a taller per_core_M).
### This tree, HiFi2/bf16acc, B=5 decomposition (2 links, rows=2 w=4)
ring: qkv_b5 fused 1416 vs unfused 2340 (1.65x), mm_sp_alone 1072 (resident) vs linear 1480, AG 859, serial 2062;
gate_up_b5 5842 vs 5864 (1.00x; mm 5657 with the since-reverted per_core_N=20, linear 5002); dgrad_row_b5 1387 vs 1898
(1.37x; mm 945 vs linear 1039). line: qkv_b5 1728 vs 2931 (1.70x); gate_up_b5 5946 vs 6456 (1.09x); dgrad_row_b5 1706 vs
2490 (1.46x). Weight residency makes the resident sub-batched matmuls FASTER than ttnn.linear (60 GMAC/ms).
### Backward shapes, HiFi2/bf16acc, 2 links (cur tree | shipped-equivalent via env knobs)
gate_up dgrad RS: ring B=1 1071/966 (0.90x | 0.90x), B=5 5381/4940 (0.92x | 0.92x); line 1091/1001 (0.92x), 5449/5195
(0.95x); matmul alone 1017 (B=1) / 5199 (B=5) already > linear+RS: DRAM-bound weight re-read (231 MB/op at B=1).
w2 dgrad AG: ring B=1 564/581 (1.03x | 0.90x), B=5 2944/2928 (0.99x | 0.91x); line 579/693 (1.20x | 1.11x), 2954/3400
(1.15x | 1.07x).
### HiFi4/fp32acc table (tt-train's config), RS op, ring, 2 links -- see the perf2 table block below (already in this
file above if the append survived; otherwise: out_proj 356/507 (1.42x) B=1, 1522/2421 (1.59x) B=5; w2 804/1053 (1.31x),
4112/5269 (1.28x); dgrad_col 421/516 (1.22x), 1900/2549 (1.34x); gate_up dgrad 1415/1907 (1.35x), 7448/9539 (1.28x) --
at HiFi4 every RS site wins because the matmul is compute-bound and ttnn.linear's HiFi4 configs are slow (w2 846 vs
sub-batched 743; gate_up dgrad 1712 vs 1437)). bfp8 partial (fp32acc): out_proj 254/364 (1.43x), 1134/1816; w2 740/926,
3842/4736; dgrad_col 330/386, 1617/2002; gate_up dgrad 1373/1775, 7265/9119; RS alone 124 (vs 236 bf16); bfp8-vs-bf16
fused PCC 0.99992, max|d| 0.25-0.88 = 16-56 bf16 ulp at the output scale (rms ~3); vs fp32 ref PCC 0.99992 (bf16
payload 0.99999). The RS reduction re-quantises the running sum to the input dtype at EVERY hop (add_tiles in DST,
pack_tile to a CB in the input format), so a bfp8 partial is bfp8-rounded T-1 times, not once.
### In flight at the wipe: f_v2 runner (ag_perf2_ring queued behind the profiler run); restarted as f_v3.sh:
ag_perf2_ring, rs_perf2_line, ag_perf2_line, rs_perf2_ring (re-run for the log), AG line / RS ring / RS line checks, AG
ring check, tt-train sp suites. Logs: generated/spfuse/logs/f_v3_*.log; table: `python3 f_perf2_table.py logs/f_v3_*perf2*.log`.

## Agent F: task 5 scoping -- L1-resident partial for matmul_reduce_scatter_sp_async (NOT implemented)
Today the fused RS op materialises the full [B,1,S,N] partial in DRAM (matmul writes it, RS readers read it back: 2 x
16 MB per Llama shape at B=1, 80 MB at B=5, plus the RS intermediate traffic). The strided RS design
(`StridedReduceScatterFusedOpSignaler`: `mm_progress_counters_addr` per-MM-core progress counters on the RS cores,
`rs_credit_counters_addr` / `num_rs_readers` credits back to the MM cores, `mm_window_blocks` = how many M blocks of a
core's output stay resident; matmul side `MM_WINDOW_BLOCKS` / `MM_WINDOW_TOTAL_M_TILES` in minimal_matmul_program_factory
.cpp:453-458, 777-790) keeps the MM output in a rolling window of W M-blocks per core and lets the RS readers pull each
block straight from the producing core's L1 (or a W-deep DRAM ring), releasing the slot with a credit.
Porting that to the SP op means: (a) the 2D-mcast writer kernels write each sub-batch's out block into slot (j mod W) of
a per-core L1 ring instead of DRAM and wait for the RS credit before recycling a slot (per-slot semaphore, W x
out_block_tiles x 2 KiB per core: 44 KiB per slot for N=4096 -> W=4 fits easily beside the resident weight for out_proj,
not beside gate_up-dgrad's streamed CBs (in1 double buffer 176 KiB is fine, so yes it fits there too)); (b) the RS
readers (ring + line) address the partial by (core, slot) NoC reads instead of TensorAccessor page ids -- the RS reader
walks a slice tile-by-tile in row-major tile order, while the matmul's slice is scattered over 96 cores as 2x11/12 tile
blocks, so the reader needs the per-core block map (a 96-entry rt table) and issues 12 NoC reads per tile row instead
of one DRAM read per tile; (c) the ring RS's second pass over its own intermediate (penult staging) and the final
reduce still go through DRAM/L1 as today. Cost estimate: ~2 days (writer kernel slot logic + credit semaphores, two RS
reader kernels' addressing, factory plumbing for both topologies, ordinal waits -> per-slot credits, tests). Gain
bound: it removes the partial's DRAM write+read (32 MB at B=1 out_proj = ~100 us of DRAM traffic, mostly already
hidden under compute) and lets the RS start a slice a few blocks earlier; it does NOT shorten the RS's own link time
(225 us ring / 336 line at 2 links), which is what remains exposed (exposed_rs 60-130 us at B=1 on the ring after the
tail). Verdict: only worth it after the two-stream backward and bfp8 decisions; at HiFi4 the RS op is already 1.22-1.59x.
Profiling decision (22:35): four tracy device-profile attempts of the Llama-8B step stalled (0 steps after 30+ min,
9% CPU, no profiler log writes); the FSDP-era profiles were TinyLlama. Dropped for now: the measurement basis is the
nocomm ideal + composed/fused step times + the naive-profiler PHASES split (forward/backward/optimizer). If per-op data
is needed, profile a 2-4 layer Llama-8B variant instead of 32 layers.

## Agent F 22:36 -- perf2 table, AG op, ring, 2 links (this tree; log f_v3_ag_perf2_ring.log)
Columns: fused | unfused = all_gather_async + ttnn.linear (same ckc; bfp8 payload: both paths gather the bfp8-typecast activation, matmul output bf16) | speedup | linear alone | sub-batched matmul alone (same kernels, identity schedule, resident weight where it fits) | AG alone | exposed = fused - mm_alone. Numerics on B=1: fused vs fp32 ref (PCC, max|d|, bf16 ulp at the output rms scale: max / p99.9), bfp8-payload fused vs bf16-payload fused, and the bfp8-typecast gathered activation vs the bf16 input (4 ulp = bfp8 quantisation).
```
op  topo ckc     payload shape                      cfg       fused unfused     x linear  mm_sp   ccl   expo  PCC ref  max|d| ulp max/p99.9 PCC vs bf16 max|d| ulp max/p99.9 gathered ulp
ag  ring bf16acc bf16    llama8b_dgrad_row          3/2x3       294     395  1.34    210    184   185    110  0.99994  0.3825 24.5/10.22
ag  ring bf16acc bf16    llama8b_dgrad_row_w2       10/1x5      565     580  1.03    410    552   190     13  0.99994  0.4679 29.9/10.27
ag  ring bf16acc bf16    llama8b_gate_up            19/2x1     1092    1189  1.09   1003   1054   189     37  0.99994  0.4132 26.4/10.29
ag  ring bf16acc bf16    llama8b_qkv                4/2x4       307     494  1.61    304    213   185     93  0.99994  0.3582 22.9/10.26
ag  ring bf16acc bf16    llama8b_dgrad_row_b5       3/2x3      1385    1898  1.37   1036    879   859    507
ag  ring bf16acc bf16    llama8b_dgrad_row_w2_b5    10/1x5     2946    2944  1.00   2243   2858   860     89
ag  ring bf16acc bf16    llama8b_gate_up_b5         19/2x1     5548    5941  1.07   5005   5314   861    234
ag  ring bf16acc bf16    llama8b_qkv_b5             4/2x4      1414    2354  1.66   1482    981   862    433
ag  ring bf16acc bfp8    llama8b_dgrad_row          3/2x3       208     300  1.44    174    160   128     48  0.99992  0.442  28.3/13.22    0.99990     0.4375 28.0/12.00    4.0/4.00
ag  ring bf16acc bfp8    llama8b_dgrad_row_w2       10/1x5      561     502  0.89    393    502   128     59  0.99992  0.4679 29.9/13.19    0.99990     0.5    32.0/12.00    4.0/4.00
ag  ring bf16acc bfp8    llama8b_gate_up            19/2x1     1050    1035  0.99    927    945   131    105  0.99992  0.4701 30.1/13.15    0.99990     0.5    32.0/12.00    4.0/4.00
ag  ring bf16acc bfp8    llama8b_qkv                4/2x4       226     397  1.75    267    189   128     38  0.99992  0.4675 29.9/13.28    0.99990     0.4375 28.0/12.00    4.0/4.00
ag  ring bf16acc bfp8    llama8b_dgrad_row_b5       3/2x3      1002    1438  1.44    873    764   574    238
ag  ring bf16acc bfp8    llama8b_dgrad_row_w2_b5    10/1x5     2856    2582  0.90   2149   2612   575    244
ag  ring bf16acc bfp8    llama8b_gate_up_b5         19/2x1     5332    5156  0.97   4603   4827   577    506
ag  ring bf16acc bfp8    llama8b_qkv_b5             4/2x4      1075    1836  1.71   1268    876   577    199
ag  ring fp32acc bf16    llama8b_dgrad_row          3/1x3       307     428  1.39    250    218   188     89  0.99999  0.0392 2.5/1.82
ag  ring fp32acc bf16    llama8b_dgrad_row_w2       10/2x2      718     913  1.27    786    778   205    -60  0.99999  0.0564 3.6/1.82
ag  ring fp32acc bf16    llama8b_gate_up            19/2x1     1408    1824  1.30   1715   1444   203    -36  0.99999  0.0647 4.1/1.82
ag  ring fp32acc bf16    llama8b_qkv                4/2x2       349     519  1.49    341    300   192     50  0.99999  0.0604 3.9/1.82
ag  ring fp32acc bf16    llama8b_dgrad_row_b5       3/1x3      1486    2068  1.39   1293   1127   866    359
ag  ring fp32acc bf16    llama8b_dgrad_row_w2_b5    10/2x2     3889    4708  1.21   4134   3951   885    -62
ag  ring fp32acc bf16    llama8b_gate_up_b5         19/2x1     7347    9319  1.27   8774   7332   878     15
ag  ring fp32acc bf16    llama8b_qkv_b5             4/2x2      1729    2578  1.49   1775   1540   872    189
ag  ring fp32acc bfp8    llama8b_dgrad_row          3/1x3       252     337  1.34    214    208   128     44  0.99997  0.1367 8.7/5.30      0.99997     0.125  8.0/6.00      4.0/4.00
ag  ring fp32acc bfp8    llama8b_dgrad_row_w2       10/2x2      714     833  1.17    751    668   136     46  0.99997  0.1479 9.5/5.30      0.99997     0.1562 10.0/6.00     4.0/4.00
ag  ring fp32acc bfp8    llama8b_gate_up            19/2x1     1361    1726  1.27   1683   1287   138     74  0.99997  0.1526 9.8/5.33      0.99997     0.125  8.0/6.00      4.0/4.00
ag  ring fp32acc bfp8    llama8b_qkv                4/2x2       308     422  1.37    302    277   130     31  0.99997  0.1471 9.4/5.32      0.99997     0.125  8.0/6.00      4.0/4.00
ag  ring fp32acc bfp8    llama8b_dgrad_row_b5       3/1x3      1234    1658  1.34   1151   1069   577    165
ag  ring fp32acc bfp8    llama8b_dgrad_row_w2_b5    10/2x2     3748    4294  1.15   3962   3576   587    172
ag  ring fp32acc bfp8    llama8b_gate_up_b5         19/2x1     6976    8799  1.26   8335   6646   585    330
ag  ring fp32acc bfp8    llama8b_qkv_b5             4/2x2      1528    2091  1.37   1592   1425   580    103
```
Reading it: at HiFi4/fp32acc (training config) every AG site wins on the ring: qkv 1.49x (B=1) / 1.49x (B=5), out_proj
dgrad 1.39x / 1.39x, gate_up 1.30x / 1.27x, w2 dgrad 1.27x / 1.21x -- the HiFi4 ttnn.linear is slow enough that even the
streamed-weight sub-batched matmul is at par or faster (gate_up 1444 vs 1715; w2 dgrad 778 vs 786) and the early AG
signal hides the collective (exposed -60..+90 us at B=1; the negative values are the fused op's matmul finishing
before the standalone one would). At HiFi2 the resident sites (qkv 1.61x, out_proj dgrad 1.34x) gain from the weight
residency (mm alone 213 vs linear 304) while the streamed ones stay neutral (gate_up 1.09x, w2 dgrad 1.03x/1.00x).
bfp8 payload: AG alone 185 -> 128 us; fused qkv 226 us (HiFi2) / 308 (HiFi4) vs 307 / 349 with bf16 payload, i.e. -26%
/ -12%; numerics: the gathered activation is quantised to 4 bf16 ulp; the fused output moves by 28-32 ulp-at-scale max
(HiFi2, PCC 0.99990) / 8-10 ulp (HiFi4, PCC 0.99997) vs the bf16 payload, and 0.99992 / 0.99997 vs the fp32 reference
(bf16 payload: 0.99994 / 0.99999). Note the AG-side bfp8 also means the wgrad's gathered activation is bfp8.

## Agent F 22:43 -- perf2 table, RS op, line, 2 links (this tree; log f_v3_rs_perf2_line.log; same columns as the RS ring table)
```
op  topo ckc     payload shape                      cfg       fused unfused     x linear  mm_sp   ccl   expo  PCC ref  max|d| ulp max/p99.9 PCC vs bf16 max|d| ulp max/p99.9 gathered ulp
rs  line bf16acc bf16    llama8b_dgrad_col          12/2x4      370     508  1.37    182    227   337    143  0.99995  0.3013 19.3/10.73
rs  line bf16acc bf16    llama8b_dgrad_col_gate_up  11/2x1     1076    1003  0.93    742   1074   338      2  0.99992  0.9893 15.8/7.36
rs  line bf16acc bf16    llama8b_out_proj           12/2x4      360     548  1.52    215    182   336    177  0.99995  0.2647 16.9/8.57
rs  line bf16acc bf16    llama8b_w2                 11/2x1      715     885  1.24    552    572   337    143  0.99994  0.5484 17.5/8.89
rs  line bf16acc bf16    llama8b_dgrad_col_b5       12/2x4     1572    2403  1.53    951   1116  1516    457
rs  line bf16acc bf16    llama8b_dgrad_col_gate_up_b5 11/2x1     5409    5222  0.97   4195   5210  1517    198
rs  line bf16acc bf16    llama8b_out_proj_b5        12/2x4     1558    2500  1.61    987    822  1517    735
rs  line bf16acc bf16    llama8b_w2_b5              11/2x1     3445    4000  1.16   2542   2914  1517    531
rs  line bf16acc bfp8    llama8b_dgrad_col          12/2x4      345     450  1.30    148    234   309    111  0.99986  0.5919 37.9/20.16    0.99990     0.5    32.0/16.00
rs  line bf16acc bfp8    llama8b_dgrad_col_gate_up  11/2x1     1025     996  0.97    746   1051   315    -26  0.99983  1.287  20.6/11.64    0.99990     1      16.0/8.00
rs  line bf16acc bfp8    llama8b_out_proj           12/2x4      334     488  1.46    182    181   309    152  0.99986  0.4531 29.0/16.53    0.99990     0.375  24.0/12.00
rs  line bf16acc bfp8    llama8b_w2                 11/2x1      626     825  1.32    520    576   309     51  0.99985  0.9587 30.7/15.81    0.99990     0.75   24.0/12.00
rs  line bf16acc bfp8    llama8b_dgrad_col_b5       12/2x4     1476    2126  1.44    784   1130  1423    346
rs  line bf16acc bfp8    llama8b_dgrad_col_gate_up_b5 11/2x1     5116    4997  0.98   4083   5206  1424    -91
rs  line bf16acc bfp8    llama8b_out_proj_b5        12/2x4     1458    2230  1.53    818    828  1425    630
rs  line bf16acc bfp8    llama8b_w2_b5              11/2x1     3051    3732  1.22   2397   2927  1424    124
rs  line fp32acc bf16    llama8b_dgrad_col          12/2x2      430     614  1.43    297    346   340     83  0.99999  0.1492 9.6/4.21
rs  line fp32acc bf16    llama8b_dgrad_col_gate_up  11/2x1     1419    1993  1.40   1717   1424   350     -5  0.99999  0.3714 5.9/2.26
rs  line fp32acc bf16    llama8b_out_proj           12/2x2      382     606  1.59    278    247   337    134  0.99999  0.1156 7.4/3.34
rs  line fp32acc bf16    llama8b_w2                 11/2x1      848    1152  1.36    841    741   348    107  0.99999  0.2175 7.0/3.11
rs  line fp32acc bf16    llama8b_dgrad_col_b5       12/2x2     1931    3000  1.55   1623   1758  1517    173
rs  line fp32acc bf16    llama8b_dgrad_col_gate_up_b5 11/2x1     7342   10000  1.36   8741   7330  1527     12
rs  line fp32acc bf16    llama8b_out_proj_b5        12/2x2     1593    2865  1.80   1424   1251  1517    342
rs  line fp32acc bf16    llama8b_w2_b5              11/2x1     4164    5638  1.35   4424   3794  1521    371
rs  line fp32acc bfp8    llama8b_dgrad_col          12/2x2      379     550  1.45    260    348   312     31  0.99990  0.5251 33.6/15.06    0.99990     0.4375 28.0/14.50
rs  line fp32acc bfp8    llama8b_dgrad_col_gate_up  11/2x1     1367    1925  1.41   1679   1433   332    -66  0.99990  0.8669 13.9/8.04     0.99990     0.875  14.0/8.00
rs  line fp32acc bfp8    llama8b_out_proj           12/2x2      345     547  1.59    247    251   310     94  0.99990  0.3674 23.5/12.49    0.99990     0.375  24.0/12.00
rs  line fp32acc bfp8    llama8b_w2                 11/2x1      766    1092  1.43    810    742   322     23  0.99990  0.6902 22.1/11.46    0.99990     0.75   24.0/12.00
rs  line fp32acc bfp8    llama8b_dgrad_col_b5       12/2x2     1654    2691  1.63   1471   1782  1424   -129
rs  line fp32acc bfp8    llama8b_dgrad_col_gate_up_b5 11/2x1     7188   10001  1.39   8739   7309  1437   -121
rs  line fp32acc bfp8    llama8b_out_proj_b5        12/2x2     1514    2607  1.72   1252   1245  1425    269
rs  line fp32acc bfp8    llama8b_w2_b5              11/2x1     3832    5410  1.41   4314   3810  1432     23
```
Reading: line RS at HiFi4/fp32acc: out_proj 1.59x (B=1) / 1.80x (B=5), w2 1.36x / 1.35x, qkv dgrad 1.43x / 1.55x, gate_up dgrad 1.40x / 1.36x (bf16 payload). bfp8 partial cuts the line RS from 337 to 309 us only (the line RS is latency/hop-bound, not byte-bound: standalone bf16 337 vs bfp8 309), so the line gains 3-10% from bfp8 vs 25-35% on the ring.

## Agent F 22:50 -- perf2 table, AG op, line, 2 links (this tree; log f_v3_ag_perf2_line.log; same columns as the AG ring table)
```
op  topo ckc     payload shape                      cfg       fused unfused     x linear  mm_sp   ccl   expo  PCC ref  max|d| ulp max/p99.9 PCC vs bf16 max|d| ulp max/p99.9 gathered ulp
ag  line bf16acc bf16    llama8b_dgrad_row          3/2x3       357     517  1.45    210    184   309    173  0.99994  0.3825 24.5/10.22
ag  line bf16acc bf16    llama8b_dgrad_row_w2       10/1x5      576     697  1.21    403    548   312     28  0.99994  0.4679 29.9/10.27
ag  line bf16acc bf16    llama8b_gate_up            19/2x1     1111    1296  1.17   1004   1046   311     65  0.99994  0.4132 26.4/10.29
ag  line bf16acc bf16    llama8b_qkv                4/2x4       369     618  1.68    303    214   309    156  0.99994  0.3582 22.9/10.26
ag  line bf16acc bf16    llama8b_dgrad_row_b5       3/2x3      1706    2493  1.46   1036    886  1451    820
ag  line bf16acc bf16    llama8b_dgrad_row_w2_b5    10/1x5     2954    3402  1.15   2199   2866  1448     88
ag  line bf16acc bf16    llama8b_gate_up_b5         19/2x1     5595    6438  1.15   5003   5303  1452    292
ag  line bf16acc bf16    llama8b_qkv_b5             4/2x4      1733    2950  1.70   1481    980  1452    754
ag  line bf16acc bfp8    llama8b_dgrad_row          3/2x3       242     371  1.54    174    160   198     82  0.99992  0.442  28.3/13.22    0.99990     0.4375 28.0/12.00    4.0/4.00
ag  line bf16acc bfp8    llama8b_dgrad_row_w2       10/1x5      562     576  1.02    393    501   199     61  0.99992  0.4679 29.9/13.19    0.99990     0.5    32.0/12.00    4.0/4.00
ag  line bf16acc bfp8    llama8b_gate_up            19/2x1     1058    1102  1.04    926    945   201    113  0.99992  0.4701 30.1/13.15    0.99990     0.5    32.0/12.00    4.0/4.00
ag  line bf16acc bfp8    llama8b_qkv                4/2x4       255     469  1.84    266    189   198     66  0.99992  0.4675 29.9/13.28    0.99990     0.4375 28.0/12.00    4.0/4.00
ag  line bf16acc bfp8    llama8b_dgrad_row_b5       3/2x3      1144    1804  1.58    873    768   936    376
ag  line bf16acc bfp8    llama8b_dgrad_row_w2_b5    10/1x5     2851    2904  1.02   2136   2611   937    240
ag  line bf16acc bfp8    llama8b_gate_up_b5         19/2x1     5363    5463  1.02   4613   4840   938    523
ag  line bf16acc bfp8    llama8b_qkv_b5             4/2x4      1190    2203  1.85   1266    877   938    312
ag  line fp32acc bf16    llama8b_dgrad_row          3/1x3       366     552  1.51    248    218   311    148  0.99999  0.0392 2.5/1.82
ag  line fp32acc bf16    llama8b_dgrad_row_w2       10/2x2      718    1018  1.42    756    770   322    -52  0.99999  0.0564 3.6/1.82
ag  line fp32acc bf16    llama8b_gate_up            19/2x1     1398    1956  1.40   1683   1428   322    -30  0.99999  0.0647 4.1/1.82
ag  line fp32acc bf16    llama8b_qkv                4/2x2       402     643  1.60    339    298   314    105  0.99999  0.0604 3.9/1.82
ag  line fp32acc bf16    llama8b_dgrad_row_b5       3/1x3      1759    2668  1.52   1288   1128  1450    631
ag  line fp32acc bf16    llama8b_dgrad_row_w2_b5    10/2x2     3879    5116  1.32   4140   3948  1502    -69
ag  line fp32acc bf16    llama8b_gate_up_b5         19/2x1     7312    9767  1.34   8740   7323  1452    -11
ag  line fp32acc bf16    llama8b_qkv_b5             4/2x2      1973    3115  1.58   1762   1536  1454    438
ag  line fp32acc bfp8    llama8b_dgrad_row          3/1x3       263     405  1.54    213    208   199     55  0.99997  0.1367 8.7/5.30      0.99997     0.125  8.0/6.00      4.0/4.00
ag  line fp32acc bfp8    llama8b_dgrad_row_w2       10/2x2      711     901  1.27    752    668   207     43  0.99997  0.1479 9.5/5.30      0.99997     0.1562 10.0/6.00     4.0/4.00
ag  line fp32acc bfp8    llama8b_gate_up            19/2x1     1338    1787  1.34   1667   1285   208     54  0.99997  0.1526 9.8/5.33      0.99997     0.125  8.0/6.00      4.0/4.00
ag  line fp32acc bfp8    llama8b_qkv                4/2x2       307     502  1.64    301    278   200     29  0.99997  0.1471 9.4/5.32      0.99997     0.125  8.0/6.00      4.0/4.00
ag  line fp32acc bfp8    llama8b_dgrad_row_b5       3/1x3      1274    1971  1.55   1132   1068   938    205
ag  line fp32acc bfp8    llama8b_dgrad_row_w2_b5    10/2x2     3729    4580  1.23   3928   3574   947    155
ag  line fp32acc bfp8    llama8b_gate_up_b5         19/2x1     6964    9072  1.30   8335   6648   944    315
ag  line fp32acc bfp8    llama8b_qkv_b5             4/2x2      1519    2417  1.59   1576   1427   941     92
```
Reading: line AG at HiFi4/fp32acc (bf16 payload): qkv 1.60x (B=1) / 1.58x (B=5), out_proj dgrad 1.51x / 1.52x, gate_up 1.40x / 1.34x, w2 dgrad 1.42x / 1.32x. bfp8 activation payload: line AG 309 -> 198 us; fused qkv 402 -> 307 (HiFi4), numerics identical to the ring run (same kernels: 4 ulp on the gathered activation, 8-10 ulp-at-scale on the output at HiFi4, 28-32 at HiFi2).

## Agent F 22:57 -- perf2 table, RS op, ring, 2 links (durable re-run; log f_v3_rs_perf2_ring.log; reproduces the wiped run within 1-2%)
```
op  topo ckc     payload shape                      cfg       fused unfused     x linear  mm_sp   ccl   expo  PCC ref  max|d| ulp max/p99.9 PCC vs bf16 max|d| ulp max/p99.9 gathered ulp
rs  ring bf16acc bf16    llama8b_dgrad_col          12/2x4      343     410  1.19    181    228   234    116  0.99995  0.3013 19.3/10.20
rs  ring bf16acc bf16    llama8b_dgrad_col_gate_up  11/2x1     1050     981  0.93    781   1057   236     -7  0.99992  0.9893 15.8/7.12
rs  ring bf16acc bf16    llama8b_out_proj           12/2x4      313     448  1.43    216    183   234    130  0.99995  0.2638 16.9/8.14
rs  ring bf16acc bf16    llama8b_w2                 11/2x1      667     790  1.18    539    573   234     94  0.99994  0.5186 16.6/8.57
rs  ring bf16acc bf16    llama8b_dgrad_col_b5       12/2x4     1420    1965  1.38    956   1120  1070    300
rs  ring bf16acc bf16    llama8b_dgrad_col_gate_up_b5 11/2x1     5341    4960  0.93   4174   5224  1073    117
rs  ring bf16acc bf16    llama8b_out_proj_b5        12/2x4     1216    2056  1.69    987    825  1072    391
rs  ring bf16acc bf16    llama8b_w2_b5              11/2x1     3308    3579  1.08   2562   2909  1072    398
rs  ring bf16acc bfp8    llama8b_dgrad_col          12/2x4      241     266  1.10    148    229   123     12  0.99988  0.4826 30.9/18.17    0.99992     0.375  24.0/13.50
rs  ring bf16acc bfp8    llama8b_dgrad_col_gate_up  11/2x1      992     834  0.84    777   1078   126    -86  0.99985  1.239  19.8/10.67    0.99992     0.875  14.0/7.00
rs  ring bf16acc bfp8    llama8b_out_proj           12/2x4      206     303  1.48    181    181   123     24  0.99988  0.3973 25.4/14.99    0.99993     0.3125 20.0/12.00
rs  ring bf16acc bfp8    llama8b_w2                 11/2x1      597     651  1.09    516    576   124     21  0.99987  0.9587 30.7/14.53    0.99993     0.625  20.0/10.00
rs  ring bf16acc bfp8    llama8b_dgrad_col_b5       12/2x4     1028    1314  1.28    799   1129   565   -101
rs  ring bf16acc bfp8    llama8b_dgrad_col_gate_up_b5 11/2x1     5005    4419  0.88   4131   5214   566   -208
rs  ring bf16acc bfp8    llama8b_out_proj_b5        12/2x4      824    1370  1.66    817    826   564     -2
rs  ring bf16acc bfp8    llama8b_w2_b5              11/2x1     2989    2986  1.00   2410   2927   565     62
rs  ring fp32acc bf16    llama8b_dgrad_col          12/2x2      420     515  1.23    299    345   237     75  0.99999  0.1151 7.4/3.26
rs  ring fp32acc bf16    llama8b_dgrad_col_gate_up  11/2x1     1424    1913  1.34   1719   1446   251    -22  0.99999  0.2406 3.8/1.75
rs  ring fp32acc bf16    llama8b_out_proj           12/2x2      355     506  1.42    279    247   234    109  0.99999  0.0878 5.6/2.64
rs  ring fp32acc bf16    llama8b_w2                 11/2x1      803    1058  1.32    842    742   242     60  0.99999  0.1718 5.5/2.47
rs  ring fp32acc bf16    llama8b_dgrad_col_b5       12/2x2     1901    2550  1.34   1619   1753  1074    148
rs  ring fp32acc bf16    llama8b_dgrad_col_gate_up_b5 11/2x1     7444    9514  1.28   8747   7338  1086    107
rs  ring fp32acc bf16    llama8b_out_proj_b5        12/2x2     1517    2422  1.60   1423   1246  1071    272
rs  ring fp32acc bf16    llama8b_w2_b5              11/2x1     4109    5259  1.28   4401   3822  1081    287
rs  ring fp32acc bfp8    llama8b_dgrad_col          12/2x2      330     389  1.18    279    356   127    -26  0.99992  0.3302 21.1/12.46    0.99992     0.375  24.0/12.00
rs  ring fp32acc bfp8    llama8b_dgrad_col_gate_up  11/2x1     1373    1791  1.30   1700   1434   135    -61  0.99992  0.7637 12.2/6.64     0.99992     0.75   12.0/6.50
rs  ring fp32acc bfp8    llama8b_out_proj           12/2x2      254     365  1.44    250    251   124      3  0.99992  0.2539 16.2/10.53    0.99992     0.25   16.0/10.00
rs  ring fp32acc bfp8    llama8b_w2                 11/2x1      739     929  1.26    830    754   133    -15  0.99993  0.5217 16.7/9.81     0.99992     0.5    16.0/10.00
rs  ring fp32acc bfp8    llama8b_dgrad_col_b5       12/2x2     1619    2001  1.24   1536   1787   566   -168
rs  ring fp32acc bfp8    llama8b_dgrad_col_gate_up_b5 11/2x1     7267    9111  1.25   8746   7316   578    -49
rs  ring fp32acc bfp8    llama8b_out_proj_b5        12/2x2     1133    1812  1.60   1280   1252   566   -120
rs  ring fp32acc bfp8    llama8b_w2_b5              11/2x1     3849    4732  1.23   4337   3818   573     32
```

## Agent F 23:08 -- answers to the coordinator's three questions
(1) What changed vs the committed ops (mechanisms, all default-on, each with an env kill-switch):
  a. AG signal timing (AG op only, biggest AG effect): the all-gather reader used to signal a forwarded slice only after
     forwarding it (its chunk-semaphore waits are interleaved with CB pushes that block behind the writer's local send),
     i.e. one slice-time late; on a T=4 ring dir0's first slice was signalled at the END of the collective, so the
     matmul serialised after its local slice (467 = 77 local + 223 AG + 3x77 + ... matched exactly). Now a forwarded slice
     is signalled when its last chunk has landed. Kill-switch TT_SP_AG_SIGNAL_LATE=1.
  b. Weight (in1) L1 residency (both ops): when Kt x per_core_N tiles fit next to the other CBs (qkv 1 MiB, out_proj
     704 KiB, qkv-dgrad 1056 KiB, out_proj-dgrad 768 KiB; NOT gate_up / w2 / their dgrads) the in1 CB is sized to the
     whole per-core slab; the first sub-batch reads+multicasts it, the other B*T-1 sub-batches (and h-blocks) only
     re-publish the pages. This is why the sub-batched matmul ALONE is now faster than ttnn.linear (bf16acc qkv 213 vs
     304, out_proj 183 vs 216): 1 weight read instead of T (B*T) and no repeated column multicast. It is also what makes
     the RS op faster (out_proj 376 -> 313 = matmul 245 -> 183; the RS overlap itself is unchanged, exposed 130 both
     times). Kill-switch TT_SP_IN1_STREAM=1.
  c. per_core_N 11 -> 12 with a 2x4 (bf16acc) / 2x2 (fp32acc) subblock for the N=4096 RS shapes, only when resident
     (out_proj, qkv-dgrad); no isolated measurement of c alone (b and c land together in the 245 -> 183).
  Unchanged: schedules, the 2-row collective region and worker counts, the RS-side waits, the K-block order (so bitwise
  equality with the same-config plain matmul is preserved -- residency only changes WHEN in1 tiles are read, not what
  the compute kernel sees).
(2) Yes, the same check-verified path. Since the last kernel edit (16:35 in1 residency) / last host edit (16:52 wide-N
  gating) / install (16:53), check mode on THIS tree: AG ring 21 passed (19:5x, log wiped; re-queued as
  f_v3_ag_check_ring), AG line 21 passed (f_v3_ag_check_line.log, durable): bitwise_equal=True vs the plain matmul with
  the same program config on every case incl. B=5 qkv/out_proj-dgrad and the w2-dgrad shape, PCC 0.99994 vs fp32
  (0.99999 at fp32acc), program-cache +1/+0, trace == eager. RS ring / line checks are running now (f_v3_rs_check_ring,
  f_v3_rs_check_line; each 19 cases incl. out_proj B=5). test_sp_matmul_schedule 100 passed and mm_smoke 9 passed on
  this tree (19:4x, logs wiped).
(3) Yes: perf2 "unfused" = all_gather_async(x) + ttnn.linear(gathered, w, transpose_b, dtype=bf16,
  compute_kernel_config=ckc) for the AG op and reduce_scatter_minimal_async(ttnn.linear(x, w, ..., dtype=out_dtype,
  ckc)) for the RS op, with the same ckc (bf16acc = HiFi2/bf16 dest/L1 acc, fp32acc = HiFi4/fp32 dest/L1 acc =
  ComputeKernelConfig::matmul()) as the fused op; the standalone collectives run their default workers on the full grid
  (as in the committed round); "linear" in the table is that same ttnn.linear alone.
Tree state: consistent and buildable; build.sh ttnncpp ttnn _ttml rc=0 at 16:53 installed exactly this host tree (no
ttnn/ source edit since 16:52; only the two Python test files changed later). `ninja -n ttnncpp ttnn` reports nothing to
rebuild. tt-train has agent S's uncommitted edits (built by S; I never touched tt-train).
