# PDL x genify_matmul integration (#65150) — running log

Checkout `/localdev/avelickov/tt-metal`, branch `avelickov/65150-matmul-integration` (main 8e28a65 + the op).
Switch: `TT_GENIFY_MATMUL=1` (`models/experimental/op_genify/genify_matmul_adapter.py`). Off = stock, unchanged.

## Changes

- `models/experimental/panoptic_deeplab/tt/genify_conv.py` (new): `create_conv_layer()` returns a
  `GenifyConv1x1` for every 1x1 conv when the switch is on, else the stock `TtConv2d`. `GenifyConv1x1`
  has TtConv2d's `__call__(x, return_output_dim=False)` contract; weight `[Cin, Cout]` = `w[:, :, 0, 0].T`
  and bias `[1, Cout]` are built once at model build from the same host tensors, in the conv's weights
  dtype (bfp8), DRAM interleaved TILE. RELU -> `activation="relu"` (bias epilogue), output dtype bfp8, the
  stock compute config (LoFi, fp32 DEST, approx) via `genify.compute_config`. Sharded A is read in place
  (output keeps A's sharding); interleaved A (DRAM or L1) -> L1 interleaved output. A rejected sharded
  layout would be resharded (warning, `self.reshards`) — none was rejected. `deallocate_activation`
  honoured like conv2d (never frees a DRAM input). Logs one `[genify] <conv path>: ...` line per conv.
- Stride-2 1x1 (res3.0 / res4.0 shortcut): subsample in TILE layout = tile-aligned slice of a
  `[N, Ho, 2W, C]` view (even rows) + a selection matmul on genify (`S' [1, groups, 32, 64] @ pixels
  [1, groups, 64, C]`, batched B, HiFi2, bit-exact, bfp8 A), then the conv's genify matmul. Generic fallback
  (row-major input, odd shapes): untilize + strided RM slice + tilize.
- `tt_bottleneck.py`, `tt_aspp.py`, `tt_semseg.py` (`tt_insemb.py` inherits): `_create_conv_layer` calls
  `create_conv_layer` instead of `TtConv2d`.
- `tt_bottleneck.py` `forward_110_cores`: identity -> `ttnn.to_memory_config` (I2S) when it is
  interleaved (only a genify shortcut reading an interleaved block input produces that; the stock identity is
  always sharded -> `ttnn.reshard`, unchanged).
- `tt_upsample.py`: both bilinear passes on genify when the switch is on (W pass: A L1 interleaved,
  shared Mw [1,1,256,1024]; H pass: MhT @ batched B = Y_width, output DRAM), same dtype / compute config.
- Tools (this folder): `pdl_profile_table.py` (stock vs genify per-matmul table from two ops CSVs),
  `pdl_test_subsample.py` (subsample variants, exactness + device time), `pdl_test_upsample_hang.py`
  (repro of the genify hang below). `pdl_logs/` = breadcrumbs.

## Results (final code)

- PCC, `test_tt_model.py -k panoptic_deeplab_110_cores` (thresholds 0.985 / 0.948 / 0.992):
  stock 0.98500 / 0.95974 / 0.99277; genify 0.98538 / 0.95384 / 0.99254 (identical in all 4 genify runs).
- Unit tests with the switch on: test_resnet (stem, res2-5 layers with RM bf16 inputs, full), test_aspp,
  test_semseg, test_insemb, test_tt_upsample bfp8 (18) all pass. ASPP / semseg / insemb / res2 PCCs are
  bit-identical to stock (genify's HS/interleaved 1x1 convs + upsample reproduce the stock results exactly);
  res3/res4/res5 PCC vs torch 0.99917/0.99923/0.99552 (stock 0.99907/0.99917/0.99518). test_tt_upsample's
  bf16 PDL-size cases HANG (op bug below).
- Profiles (one inference; `DEVICE KERNEL DURATION`):
  stock `generated/profiler/reports/2026_11_17_20_32_48/ops_perf_results_2026_11_17_20_32_48.csv` (313 ops, 5759.0 µs);
  genify `generated/profiler/reports/2026_11_17_21_10_11/ops_perf_results_2026_11_17_21_10_11.csv` (292 ops, 5634.6 µs;
  repeat `2026_11_17_20_53_23`: 5623.1 µs). Table: `pdl_profile_table.md` (+ gitignored `.csv`).
  | | stock | genify |
  |---|---:|---:|
  | 55 matmuls (kernel) | 989.8 | 967.2 |
  | + glue charged to them | 1170.7 | 1042.2 |
  | whole inference | 5759.0 | 5634.6 (-2.2%) |
  By class (matmul kernel µs, stock -> genify; glue in brackets):
  HS in place x17 113.9 -> 131.8 (1.16x); BS in place x16 233.3 -> 256.3 (1.10x: expand 0.96-0.99x,
  reduce 1.27-1.43x); interleaved A x14 200.4 [+121.7 I2S] -> 303.4 [+24.8] (322.2 vs 328.2 incl. glue);
  stride-2 x2 29.3 [+57.2] -> 36.5 [+50.2]; upsample x6 412.8 -> 239.2 (0.58x; H pass 122 -> 67 µs).

## Op problems for the code-gen run

1. HANG: `genify_matmul(A [1,32,512,128] bf16, B [1,32,128,1024] bf16 (batched), dtype=bf16, compute
   HiFi2 / approx / fp32_dest_acc_en=False)`, A/out DRAM interleaved, B DRAM or L1 interleaved. All 64 cores:
   writer at `tail_split::wait_at_least` (genify_matmul_writer.cpp:257, waiting for the reader's tail share),
   reader at `cb_reserve_back` (genify_matmul_reader.cpp:720, still in its K-walk), packer waiting for free
   output tiles -> circular wait between the tail write split and the K-walk. bfp8/LoFi same shapes: OK.
   Repro: `scripts/run_safe_pytest.sh --dev models/experimental/op_genify/integration/pdl_test_upsample_hang.py -k h_bf16_dram`
   (also `h_bf16_l1b`); in-model trigger: `test_tt_upsample.py` bf16 1x128x256 scale-4 cases with the switch on.
2. Interleaved-A path (R1/R1a) under-fills the grid on PDL's tall-skinny shapes (bfp8, LoFi, bias+relu,
   out L1 interleaved): 32768x256x128 A DRAM -> 27 cores, 41 µs (stock HS matmul 9.4 + I2S 24.7; genify's
   own HS path on [384,256]x86: 10.3); 32768x256x32 A L1 -> 49 cores, 23-26 µs (genify HS 4.4, stock 3.6);
   8192x512x64 A L1 -> 37 cores, 12.2 µs (HS 5.9); 2048x2048x256 A L1 -> 88 cores, 28.5 µs (BS 18.4, stock 13.8);
   8192x512x256 / 2048x1024x512 A DRAM: 27.3 / 17.3 µs (stock 16.6 / 10.6 + I2S). The split cost model seems
   to over-weight B re-reads vs per-core A read time. Exception: 2048x1024x2048 A DRAM 33.9 µs vs stock 72.3.
3. Sharded paths: HS ~1.2x stock on small shapes (fixed handshake rounds), BS reduce convs (K 1024/2048)
   1.27-1.43x; BS expand convs at or below stock.
4. No rejected layouts, no wrong results.

## Log
- 2026-11-17 20:26 — genify run 1 (`TT_GENIFY_MATMUL=1 scripts/run_safe_pytest.sh .../test_tt_model.py -k panoptic_deeplab_110_cores -s`):
  PASS. semantic 0.98538 / center 0.95384 / offset 0.99254 (stock on this branch: 0.9850 / 0.9597 / 0.9928;
  thresholds 0.985 / 0.948 / 0.992). All 49 1x1 convs logged a `[genify]` line, 3 upsample calls (6 matmuls)
  on genify. No layout was rejected (no reshard): HS res2 [320,*] x110 cores (maxpool's grid), res3 [96,*] x86,
  heads [320,*] x103; BS res4 [256,*] 8x8, res5 [224,*] 8x10 all read in place. Interleaved A: res3.0/res4.0/res5.0
  conv1 + res5.0 shortcut (DRAM), ASPP convs.0/convs.4/project and decoder project convs (L1).
- 20:32 — stock profile: PASS 0.98500 / 0.95974 / 0.99277, 313 ops, 5759.0 µs
  (`generated/profiler/reports/2026_11_17_20_32_48/ops_perf_results_2026_11_17_20_32_48.csv`).
- 20:38 — genify profile v1: PASS (same PCC as run 1), 296 ops, 5775.3 µs (+16.3)
  (`generated/profiler/reports/2026_11_17_20_38_00/...`). 55 matmuls 960.6 vs 989.8 µs (-29.2), but the
  stride-2 subsample cost 133 µs (res3.0; strided RM slice alone 91.7 µs) + 52 µs (res4.0) vs stock's
  I2S + halo + move 36 / 19 µs. Interleaved-A genify calls pick few cores (res3.0.conv1 DRAM A
  32768x256x128: 27 cores, 40.5 µs; decoder res2 project 32768x256x32 L1 A: 49 cores, 21-23 µs).
- 20:45-20:51 — subsample microbench (`pdl_test_subsample.py`, `--profile`), all variants exact except LoFi:
  res3.0 (res4.0) after the 17.6 (9.2) µs rows slice: v1 strided RM slice path 115 (42); v2 RM reshape
  path 62 (36); v4 TILE reshape 206 (105); v3 selection matmul K=W HiFi4 22.7 (7.5), HiFi2 16.5 (6.3);
  v3g grouped K=64 HiFi2 9.7 (4.8) -> adopted (bfp8 A, exact). LoFi selection: maxdiff 0.19 (rejected).
- 20:53 — genify profile v2 (selection subsample): PASS, same PCC, 292 ops, 5623.1 µs (-135.9 vs stock).
- 20:55 — genify unit tests (`--run-all`, not 20_cores): resnet/aspp/semseg/insemb + small upsample pass;
  `test_bilinear_upsample_ttnn_matmul_vs_ttnn_upsample[DataType.BFLOAT16-1-128-256-32-4-False-False-memory_config11-110_cores]`
  HANG (genify H pass), later cases failed on the dead device (run stopped; runner reset the device).
- 21:00-21:04 — `pdl_test_upsample_hang.py`: W pass bf16 DRAM runs (PCC 0.9961 vs fp32 torch, bf16 DEST);
  H pass bf16 DRAM HANG (--dev triage above); H pass bfp8 DRAM OK; H pass bf16 with B in L1 HANG.
- 21:05 — genify test_tt_upsample bfp8 cases (18) PASS. 21:07 — stock unit tests (resnet/aspp/semseg/insemb) for PCC compare.
- 21:10 — final genify profile (final code): PASS, same PCC, 292 ops, 5634.6 µs (-124.4 vs stock; run-to-run
  ~0.2%, mostly decoder.res2.project_conv 23 -> 25-26 µs).
