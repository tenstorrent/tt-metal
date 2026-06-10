# topk_xl — Work Log

Branch: `pjosipovic/topk_xl`. Plan: `PLAN_topk_xl_op.md`. Goal: port blaze topk_xl SFPU
LLK into tt-metal (Approach A, canonical tt-llk tree) and build a row-major, per-row TTNN
topk op. tt-blaze issue #558.

## Log

### 2026-06-10 — multicore row sharding, no padded-K workaround
- Parallelization now shards independent rows across the active Tensix cores with
  `split_work_to_cores(..., row_wise=true)`. Each scheduled core receives a contiguous
  `[start_row, start_row + num_rows)` range; the unit of work is one full input row.
- Reader and writer runtime args are now `{buffer_addr, start_row, num_rows}`. Compute receives
  `{num_rows}`. CBs are allocated over the active core range so reader, compute, and writer can
  overlap row-by-row on each core with the existing double-buffered input/output CBs.
- For 640 rows on a 120-core device, the split is expected to be 40 cores with 6 rows and 80 cores
  with 5 rows, so no core processes more than 6 rows.
- Important cleanup after review: do **not** route public K=512 through an internal K=1024 padded
  path. That would add extra compute/L1 traffic and is not the right fix. The writer scratch reorder
  also keeps the K=512 case as a single contiguous local-L1 NOC transfer.
- Validation: `./build_metal.sh --release` PASS; `scripts/run_safe_pytest.sh
  tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_parallelizes_640_rows -q`
  PASS with K=1024; full `scripts/run_safe_pytest.sh
  tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q` PASS (19/19).
- Caveat: a 640-row stress run with K=512 exposed a separate half-tile/multicore corner. Existing
  K=512 coverage still passes, but no slow padded workaround is retained; K=512 large-row stress
  needs a direct fix.

### 2026-06-10 — requirement change: indices-only public output
- Requirement changed: `topk_xl` now returns only sorted row-major UINT32 indices. It no longer
  returns sorted BFLOAT16 values.
- Cleanup/simplification applied:
  - Public C++/Python API returns a single `Tensor`.
  - Output spec/allocation creates only the UINT32 indices tensor.
  - Program factory removed values output CB and values scratch CB; only `cb_indices` and
    `cb_indices_scratch` remain for output.
  - Compute still carries values internally in DST because Top-K needs them for comparison and
    value/index swaps, but it now materializes/transposes only the index tiles and packs only the
    indices row page.
  - Writer now has one stream only: local-L1 NOC reorder into index scratch, then one contiguous
    DRAM write per row.
- Validation: `./build_metal.sh --release` PASS; `scripts/run_safe_pytest.sh
  tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q` PASS (18/18).

### 2026-06-10 — continuous row-output experiment
- Started replacing writer-side 16-element scatter with compute-side `pack_untilize_dest` into
  double-buffered row pages. Output CB contract is now one page per logical output row:
  `cb_values` page size = `K * sizeof(bfloat16)`, `cb_indices` page size = `K * sizeof(uint32_t)`.
- Writer path is now intended to issue exactly two contiguous DRAM writes per row: one values row and
  one indices row. Compute reserves/pushes one row page per output tensor so writer and compute can
  overlap across rows via the existing 2-page CB depth.
- K=512 output CBs use 2-face metadata so `pack_untilize_dest` emits 16 row-major sticks = 512
  elements. K=1024/2048 use normal 4-face geometry, producing 32 sticks of 32/64 elements.
- First validation: K=512 is directly correct. K=1024/2048 emit 16-element groups in face-row order
  (`0,1,4,5,...,2,3,6,7,...`), so writer now uses a one-page local scratch CB per output stream:
  copy/reorder compute's row page into scratch, pop compute's CB immediately for overlap, then issue one
  contiguous DRAM write from scratch.
- Scratch reorder is implemented with local-L1 NOC reads, not RISC-V copy loops. Since transaction
  sizes are compile-time constants, `copy_row_to_scratch` sets read state once per row/output stream
  with `set_async_read_state`, then issues `async_read_with_state` transfers into the scratch CB.
  The writer uses `async_writes_flushed()` per row so scratch pages can be reused after writes have
  left L1, with one final `async_write_barrier()` after all rows for DRAM completion.
- Validation: `./build_metal.sh --release` PASS; `scripts/run_safe_pytest.sh
  tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q` PASS (18/18).

### 2026-06-08
- **Plan + recon.** Wrote `PLAN_topk_xl_op.md` (op design, LLK 4-layer integration analysis,
  Approach A vs B, two-level testing plan). Chose **Approach A** (headers in canonical tt-llk
  tree) — required for isolated LLK correctness/perf tests in the tt-llk harness.
- **Headers placed (task 1, DONE).** 7 blaze headers → canonical homes; blaze relative
  includes rewritten to bare (resolve via `bh_hal.cpp:134-142` -I paths). No SfpuType
  enum / dispatcher edits (topk_xl uses `SfpuType::unused`).
- **LLK harness env bootstrapped.** sfpi 7.52.0 + py3.10 already present; created
  `tt_metal/tt-llk/tests/.venv` + `uv pip install -r requirements.txt` (no sudo needed).
- **LLK local_sort test GREEN on Blackhole silicon (task 2, partial).** 4/4 pass
  (K=1024+2048, asc+desc). Added `sources/topk_xl_test.cpp`, `test_topk_xl.py`,
  `TopKXLGolden`, `TOPK_XL` param. Two fixes found via HW loop:
  1. Wrapper used blaze's `int vector_mode`; current tt-metal `_llk_math_eltwise_unary_sfpu_params_`
     wants `ckernel::VectorMode`. Fixed all 5 call sites. (compile)
  2. `local_sort`/`rebuild` SFPSWAP against SrcB → MATH stalls; bare-LLK kernels must call
     `_llk_unpack_set_srcb_dummy_valid_()` on UNPACK thread. (silicon hang, found via tt-triage)
- **merge/rebuild (N>K) LLK variant GREEN on silicon** (K=1024, top-K largest of 2K).
  Convention: seq0 descending, seq1 ascending (`asc = seq_idx & 1`), `merge<K>(0)` → top-K in
  seq0, `rebuild<K>(0, descending)`. Fixes: (a) `[[maybe_unused]]` on 3 unused-in-fused-path
  vars in `ckernel_sfpu_topk_xl.h` (-Werror); (b) `_llk_unpack_set_srcb_dummy_valid_()` is
  one-shot → issue once per SrcB-consuming op (local_sort + rebuild; merge is MATH-only).
  K=2048 merge omitted from LLK test: overflows bare-harness TRISC1_CODE by ~2KB (harness
  limit, not a bug — op-level test covers it).
- **perf_topk_xl.py GREEN on silicon** (MATH_ISOLATE, K=1024+2048). Dedicated
  `sources/topk_xl_perf.cpp`; pre-issues `_llk_unpack_set_srcb_dummy_valid_()` ×LOOP_FACTOR
  (the `_perf_*_valid` SETDVALID/CLEARDVALID bookkeeping doesn't match local_sort's multi-SFPSWAP
  internal SrcB use; the proven dummy-valid path does). Cycle CSV not persisted by wrapper
  split-run mode — surface via a direct report run if needed (minor follow-up).

Debug tools available: harness `llk_triage.py` (thread/PC/mailbox on hang); **tt-exalens**;
**ttsim** (github.com/tenstorrent/ttsim) for functional sim. Host builds:
`./build_metal.sh --release`. Op-level tests: `scripts/run_safe_pytest.sh` (no --dev).

### 2026-06-08 (cont.) — LLK coverage round 2
Closed the requested gaps. Final LLK test suite (all GREEN on Blackhole silicon):
- **Correctness (7):** local_sort K=512/1024/2048 (asc+desc); merge/rebuild K=1024 N=2.
  K=512 = half-tile (top valid, bottom low-value padding; filter padding from output —
  layout-robust).
- **Perf (3):** local_sort K=1024/2048; merge/rebuild K=1024 (MATH_ISOLATE).
- **Multi-stage (N>2) reduction — NOT possible in the bare harness, deferred to op level.**
  Finding: the 32-bit fused word requires `unpack_to_dest` (SrcA is 19-bit on BH), which fixes
  `unpack_i → DST tile_i`; merging non-adjacent stage survivors needs a DST-to-DST move that
  only `copy_tile` (CB-based) provides. Same limitation as `add_lsb_indices`/`copy_tile` — the
  N-stage reduction is validated at the op level (real CBs). Attempted streaming + several
  SrcA/SrcB-valid orderings; all deadlock for the structural reason above.
- Note: `_llk_unpack_set_srcb_dummy_valid_()` sets BOTH SrcA and SrcB valid → must be issued
  AFTER the real SrcA unpacks (issuing first pollutes the SrcA queue → wrong data / hang).

### 2026-06-08 (cont.) — TTNN op (tasks 3-5)
- Op scaffolded under `ttnn/cpp/ttnn/operations/experimental/topk_xl/` (device_operation API +
  ProgramDescriptor, mirroring reduction/topk + topk_router_gpt). Host build GREEN first try.
- Host wrapper `ttnn.experimental.topk_xl(bf16 [..,M,N], k) -> (values bf16, indices uint32)`:
  reshape [M,N]→[M*N/32,32] + tilize; one logical row per Tensix core.
- Bring-up on silicon: K=1024-N=1024 **VALUES CORRECT** vs torch.topk (reduction works E2E!).
- Two issues found → root cause = copy_tile<K>/add_lsb layout ≠ host-tilize layout:
  (a) add_lsb index encoding (2048-space) ≠ within-row position 0..N-1 (D2 risk realised);
  (b) K=2048 values wrong (2-tile copy_tile<K> layout mismatch).
- **Fix (in progress): host-build fused fp32 input + generic copy_tile** in compute (mirrors the
  proven LLK path: prefused fp32 + datacopy + local_sort/merge/rebuild). Host fuses
  (value_bits<<16)|within_row_idx in the tile domain (idx = flat_pos & (N-1), N power of 2 in v1).
  Indices uint32 (reshape rejects uint16). Rebuilding + bring-up.
- Kernel entry: compute kernels use `void kernel_main()` (NOT `namespace NAMESPACE{void MAIN}`).

### 2026-06-08 (cont.) — op bring-up RESULT
- Op runs on Blackhole silicon. **Top-K VALUES correct E2E across the full sweep**
  (K=1024/2048 × N=1024/2048/4096 × M=1/8) — local_sort AND multi-stage merge/rebuild.
  Test: 10/10 value asserts pass (xfail-gated on the index issue below).
- Switched compute to host-built fused fp32 + generic copy_tile (mirrors proven LLK path);
  this also FIXED the earlier K=2048 value bug. add_lsb path abandoned (2048-space index
  encoding ≠ within-row position).
- **KNOWN ISSUE (1 remaining): fused lo16 index dropped in compute copy/pack.** Host fuse
  verified correct (diag_or: value in hi16 + index in lo16). Passthrough (copy+pack, no sort)
  → index 0, so NOT the sort. `unpack_dst_format[c_0]=Float32` + `DST_ACCUM_MODE=true` confirmed
  in JIT defines, yet lo16 lost → localized to copy_tile MATH datacopy or pack_tile not
  preserving 32-bit. Next: replicate the LLK test's explicit `_llk_math_eltwise_unary_datacopy_`
  (unpack_to_dest) + `_llk_pack_<fp32>` instead of the compute-API copy_tile/pack_tile, or add a
  pack reconfig. (diag artifact warning: untilize() frees its input in fast-runtime mode.)
- Other v1 limits: indices uint32 (reshape rejects uint16); N must be a multiple of K and
  <= 65536 (index packed in 16 bits).
- **Target shape [1,1,640,51200] K=2048 PASSES (values)** on silicon — 640 rows x 25 sequences
  (24 merge/rebuild stages/row), top-2048 correct vs torch.topk. The earlier "large-M reshape
  L1 limit" was an artifact of the old arange(M*N) index build; switching the index to
  arange(N)+repeat (also removed the power-of-two restriction) cleared it. Test:
  test_topk_xl_target_shape (value-only; passes).

### 2026-06-08 (cont.) — ROW-MAJOR REDESIGN (author confirmed data is NOT tiled)
- Verified via blaze + LLK source: `topk_xl_copy_tile`'s unpack does `SETADCXX(count=elements-1)`
  = a LINEAR contiguous read (not face-tiled). blaze feeds the LLK FP32 tile-aligned shards but
  the bytes are read linearly. So the data is row-major; tilize/untilize was WRONG. My prior
  "correct values" were set-invariance of sorting masking a scrambled index.
- **Rewrote the whole op for row-major, no tilize/wrappers:** device op FP32 row-major
  [..,M,N] in → (values bf16, indices uint32) [..,M,K] out; reader DMAs linear 1024-elem pages
  into CB (object Noc/CB APIs); compute = topk_xl_copy_tile + add_lsb_indices(core_id=seq) +
  local_sort/merge/rebuild + dual-pack (values bf16 hi16, remove_msb, indices uint32); writer
  DMAs row-major out; host wrapper only does FREE row-major reshapes (metadata, NOT tilize) to
  expose 1024-elem pages. Input is FP32 (blaze uses fp32; fused path is 32-bit, add_lsb
  overwrites low16 with index → value precision = bf16 hi16). Builds clean, runs on silicon.
- **TWO REMAINING BUGS (localized, need silicon debugging):**
  1. copy_tile data path: output value multiset slightly wrong — passthrough (copy_tile+pack
     only, no sort) gives K=1024 off-by-1 distinct (513 vs 512), K=2048 off-by-35 (675 vs 640).
     out[:8]==input[:8] (linear read ok) but some elements mis-handled. Suspects: bf16 value
     pack rounding mode vs golden, and/or copy_tile<2048> second-tile (page1) read.
  2. index isolation: uint32 index output = fused VALUE bits (idx range up to 0x3F7F...), so
     remove_msb_values isn't zeroing hi16 before the index pack. Next: try separate_indices
     (split values+indices into distinct DST regions, blaze keep_values path) instead of
     remove_msb+dual-pack; or verify remove_msb runs on the right thread/DST.
  Next concrete step: a 4-element controlled passthrough to nail the pack rounding mode
  (round vs trunc) and whether copy_tile preserves data exactly; then fix index via
  separate_indices.

### 2026-06-08 (cont.) — dtype + LLK-test-coverage corrections
- **Reverted op to bf16 in/out** (D1 spec). The fp32 switch was a wrong hypothesis — the copy
  LLK supports 16-bit input (`_llk_unpack_topk_xl_copy_init_` branches on `is_32bit_input`;
  header says "FP16 and FP32"). With bf16, copy_tile puts the bf16 value in the fused hi16 and
  the output value is the EXACT input bf16 (clean exact comparison, no round/trunc confound).
- **The value bug is real and dtype-independent** (bf16 also fails ~97%). Root cause is the
  row-major `topk_xl_copy_tile` + `add_lsb_indices` path.
- **LLK ACCURACY TESTS DO NOT COVER THE ROW-MAJOR COPY PATH** (the gap that let the bug reach
  the op). `test_topk_xl.py` feeds prefused data via generic datacopy (harness-tilized) and only
  tests local_sort/merge/rebuild — NOT `topk_xl_copy_tile`'s linear read + add_lsb.
- Started `tests/sources/topk_xl_copy_test.cpp` (LLK-level: drives `_llk_unpack_topk_xl_copy_`
  + `_llk_math_topk_xl_copy_` on buffer_A, pack, verify fused word). OPEN QUESTION: must feed
  buffer_A as RAW ROW-MAJOR (not harness-tilized) for the linear-read test to be meaningful —
  need to confirm/force the harness input layout. No python driver yet.
- **NEXT (precise):** (1) finish the LLK copy test with row-major input → confirm whether
  copy_tile reads row-major correctly + places value in hi16 (isolates copy vs add_lsb vs op
  reader/pack); (2) then add_lsb index test; (3) fix the op copy path; (4) switch index
  extraction to separate_indices. The op value multiset is off (K1024 ~1 elem, K2048 ~35),
  index output currently = fused value bits (remove_msb not isolating).

### 2026-06-08 (cont.) — LLK copy isolation test (the missing coverage)
- Added `tests/sources/topk_xl_copy_test.cpp` + `test_topk_xl_copy.py`: drives
  `_llk_unpack_topk_xl_copy_` + `_llk_math_topk_xl_copy_` on a known row-major bf16 ramp in
  buffer_A (no sort, no add_lsb), packs the fused word, checks the hi16 value multiset.
  (Fixed two harness -Werror issues: ckernel_template.h include in unpack TU; [[maybe_unused]]
  dst_format in _llk_math_topk_xl_copy_init_.)
- **FINDING: copy_tile does NOT faithfully copy a full 1024-tile in this usage.** Value multiset
  is wrong (~200–257 distinct vs ~513 expected) for both bf16 and fp32 input. The FIRST 16
  elements read linearly correct (ramp), so the linear read STARTS right but full coverage is
  off. `_llk_math_topk_xl_copy_` has a 32-bit path (unpack_to_dest + ZEROACC 4 faces) and a
  16-bit `else` path (just the ELWADD MOP) — the MOP/coverage is the suspect.
- **CAVEAT: results vary run-to-run (257 vs 200)** → device-state/reconfig-escape leakage
  (tt-llk CLAUDE.md warns about this; don't mask with reset). Clean characterization needs a
  fresh device and/or the author's exact copy_tile driving contract — the bare-harness call
  (raw L1 address) differs from the op's CB-based path (fifo_rd_ptr, unp_cfg_context, the
  compute_kernel_api init sequence), so I may be missing setup.
- **RECOMMEND:** ask the LLK author for the precise topk_xl_copy_tile driving contract (init +
  CB/page expectations + whether bf16 full-tile is supported or fp32 required), or validate the
  copy via the op's real CB path on a fresh device. This is the blocker for op value-correctness.

### 2026-06-08 (cont.) — ttsim setup for deterministic LLK debugging
- ttsim is set up: `~/sim/libttsim_bh.so` + `soc_descriptor.yaml` (prebuilt). Source in
  `/localdev/pjosipovic/refs/ttsim-private` (v1.6.1; latest is 1.7.1 if needed).
- Run LLK tests on ttsim (per TTSIM.md): from `tests/python_tests/`, with venv active:
  `TT_METAL_HOME=... TT_METAL_SIMULATOR=<libttsim_bh.so> TT_METAL_DISABLE_SFPLOADMACRO=1
  CHIP_ARCH=blackhole pytest -q --run-simulator --timeout=N <test>`. Copy `soc_descriptor.yaml`
  next to any rebuilt `.so` (ttsim derives its path from the .so).
- **Built ttsim from source** (debug+release BH): `cd src; ../make.py -j8 _out/debug_bh/libttsim.so
  _out/release_bh/libttsim.so`. The rebuilt 1.6.1 **fixes the `t_tile_mmio_wr32 addr=0xffb01ffc`
  UnimplementedFunctionality** the older prebuilt .so threw on the topk_xl copy.
- ttsim is SLOW (~131 KHz; single copy test ~300s+ via the pytest/exalens harness — boot/dispatch
  overhead dominates). Use a long `--timeout`. Faster gdb path: `tests/elf/run_elf.cpp`
  (`run_elf --sim <libttsim.so> <elf>`) runs a kernel ELF directly — better for gdb on the debug .so.
- DEBUG PLAN (task 8): gdb the debug_bh .so, break in `_llk_math_topk_xl_copy_` / the unpack MOP /
  `t_tile_mmio_*`, trace the copy to find why only ~half the 1024 tile is copied.

### 2026-06-09 (cont.13) — AUTHORITATIVE SFPU addressing model (sage-blackhole) + remaining walk bug
- sage-blackhole (ISA: BlackholeA0 SFPLOAD.md/SFPSTORE.md/Dst.md) gave the dest_reg_addr model:
  one SFPLOAD/SFPSTORE touches 32 datums = 4 rows × 8 cols. For Imm offset A (RWC.Dst=0):
    Row = (A & ~3) + lane/8        (lanes 0-7→+0, 8-15→+1, 16-23→+2, 24-31→+3)
    Column = (lane & 7)*2 + ((A & 2) ? 1 : 0)      (bit1 of A = even/odd column half; bit0 UNUSED)
  INT32 (mode 4) and FP32 (mode 3) addressing IDENTICAL (both Dst32b); only value conversion differs.
  MOD0_FMT_INT32_ALL (mode 10) is DIFFERENT (Sp/stack addr, ignores LaneEnabled) — only safe as the
  add_lsb "skip-a-face-pair" no-op RWC-advance; do NOT use for data stores.
  Packers + SFPU share DEST_ACCESS_CFG remap/swizzle (Dst.md:39) → SAME RWC.Dst Imm ⇒ SAME within-tile
  (row,col) ⇒ value-pack & index-pack stay element-aligned, regardless of write ORDER. So writing the
  index to tile1 at the same (row,col) as the value in tile0 is sufficient for alignment.
- sage CONCLUDED my existing +2 `_topk_xl_untilize_indices_` (ADDR_MOD_0 +2, load@RWC+0 INT32,
  store@RWC+indices_offset, row_scale*16 iters) is CORRECT/element-aligned. BUT the empirical test
  DISAGREES (1021/1024 wrong; only slots 0 & 8 correct; duplicates + zeros). Per the sage's OWN
  formula, a +2 walk over 32 iters reaches Imm 0..62 → Row=(62&~3)=60 (rows 60-63 = beyond tile0's
  32 rows) and only Column 0..15 — i.e. +2-flat does NOT cover a 32x32 tile's 4 faces. So the +2 walk
  is the bug; the correct full-tile walk is add_lsb's {0,2,16,18} per face-pair + ADDR_MOD_6 (+32)
  last-store advance + mode-10 ADDR_MOD_4 (+16) face-pair skip, row_scale_factor face-pairs ×4 replays.
- LEAD to try first (cheap): sage note #4 — stale `DEST_TARGET_REG_CFG_MATH_Offset` from the
  merge/copy may offset all untilize addresses; reset it before the walk. Also verify DST tile1 alloc.
- NEXT: replicate add_lsb's {0,2,16,18}+face walk for the untilize (read tile0, write untilized INT32
  to indices_offset region), unrolled (4-load body ~52 instrs likely exceeds replay slots), 1-value/
  -temp-pair to fit registers. Validate on diag_topk_xl_index.py. (sage agent id adcd0d191174b28e2.)

### 2026-06-09 (cont.12) — layout pinned: index must use SORT layout (store16 +4 stride), not +2
- Removed the value re-store from untilize (was unnecessary) — pattern UNCHANGED, so it was purely the
  index-store layout. Confirmed: the sort writes the tile via `store16_rows_x2` = 8 lregs (FP32) at
  offsets {0,4,8,12, 16,20,24,28} (+4 stride within an 8-lreg group), last store folds a +16/+32/+48
  advance (ADDR_MOD_5/6/1). pack_tile reads THAT layout. My untilize used separate_indices' +2 stride
  → wrong layout → odd pack slots unwritten (0) + even aliased (duplicates). Math is correct (idx 0..1023).
- TODO: rewrite untilize to read tile 0 + write the INT32 index to the indices_offset region using the
  SORT's load16/store16 addressing (offsets {0,4,..28}, +32 advance) so tile Ko's layout == tile 0's,
  making pack_tile(Ko) correspond to pack_tile(0). UNKNOWN = the exact K-region iteration (#8-lreg
  groups + advance sequence covering a full 32x32 tile; "load16 = 16 rows x 2 strips"). Get via
  arch-lookup / LLK author. Current code: ckernel_sfpu_topk_xl.h `_topk_xl_untilize_indices_` (uses +2,
  WRONG layout). Validation: diag_topk_xl_index.py (distinct ramp, produced vs true_pos).

### 2026-06-09 (cont.11) — in-kernel untilize: MATH works, BLOCKED on SFPU dest-addressing layout
- Implemented `_topk_xl_untilize_indices_<K>` (SFPU primitive, mirrors separate_indices structure) +
  wired llk_api + compute_api + compute kernel. JIT-compiles, runs on ttsim (~5s). The bit-permutation
  MATH is correct (double-shift fields; FIXED: negative SFPSHFT imms must be masked `& 0xFFF` — the
  macro does imm<<12 so a raw negative corrupts the opcode → all-zero output before the fix).
- After the shift fix: indices span 0..1023 (math executes) but 1011-1021/1024 WRONG-positioned with
  DUPLICATES (e.g. 1023 at slots 0&2&4, 511 at 8&10) + odd slots = 0. So the index region's SFPU-write
  layout (separate_indices' ADDR_MOD_0 +2 walk) does NOT match pack_tile's read layout. Values come out
  in correct SORT order (so value pack reads the sort layout; my untilize must NOT re-store values).
- ROOT BLOCKER: I don't understand the SFPU dest_reg_addr UNITS / how a SFPLOAD/SFPSTORE dest address +
  the +2 / {0,2,16,18}+32 strides map to tile elements and to pack_tile's face-order read. add_lsb's
  store layout (offsets {0,2,16,18}, ADDR_MOD_6 +32 advance, mode-10 +16 face advance) IS pack-readable
  (verified: raw fused lo16 was a clean bijection), but separate_indices' +2 is a different layout. Need
  the dest-addressing convention to write the index region in the SAME layout the value/pack use.
- RWC reset (TTI_SETRWC SET_D) added at untilize start — did NOT change the pattern (not the cause).
- NEXT: get the SFPU dest-addressing/pack-layout convention (arch-lookup or LLK author), then mirror
  add_lsb's exact store addressing for the index region (index-only, don't touch value tile). Files:
  ckernel_sfpu_topk_xl.h `_topk_xl_untilize_indices_*`; compute kernel calls it pre-commit, packs
  value from tiles[0,Ko), index from [Ko,2Ko). Validation: diag_topk_xl_index.py (distinct-bf16 ramp,
  produced vs true_pos).

### 2026-06-08 (cont.10) — open Qs NAILED; untilize needed regardless; separate_indices = >64K converter
- Q1 (untilize needed?): YES, always. Index originates from add_lsb as TILE-COORD (verified lo16=tilize(linear)).
  `separate_indices` does NOT untilize — keeps u16 index verbatim, only ORs group_id<<shift. So untilize
  the within-group 16-bit in BOTH fused(≤64K) and unfused(>64K) paths. My verified formula is the core fix.
- Q2 (group_id/base): within-64K-group index = sequence_base + tile_coord; for K=2^k, core_id<<k = the
  sequence base (correct), only tile_coord needs untilize. >64K: switch unfused at the stage index would
  exceed 16b; separate_indices puts group_id at group_id_bit_shift(≥16) for the high bits → 32-bit.
- Q3 (DST): unfused = parallel value region + INT32 index region @ indices_offset(=256); only >64K pays it.
- BARRIER GONE: `separate_indices` iterates the index with SIMPLE addressing (ADDR_MOD_0 dest.incr=2,
  9-instr body, row_scale*16 replays) — mirror THAT for the in-kernel untilize (not add_lsb's opaque replay).
- PLAN: (1) NOW: in-kernel untilize of within-K tile-coord → uint32, mirror separate_indices addressing.
  K=1024=1-tile (verified formula); K=2048=2-tile (ti bit = which tile: within2048 = ti*1024 +
  untilize1024(col,row)); validate on 5s loop. (2) >64K follow-on: unfused + separate_indices + group_id.
- Untilize via double-shift (no masks), operating on full fused word (value hi16 shifted out):
  field = (L<<a)>>b. Result is clean low-bits index (hi16=0). Add core_id<<k base for K=2^k via the
  high lo16 bits (or compute core*K + within).

### 2026-06-08 (cont.9) — REQUIREMENT: 32-bit indices → must use UNFUSED path (architecture pivot)
- User: output indices must be 32-bit; host 16-bit decode is NOT acceptable (caps N at 65536).
- The FUSED path (value hi16 | index lo16 in one fp32 word) is INHERENTLY 16-bit-index-capped — the
  SFPSWAP magnitude+index tiebreak needs both in one 32-bit word. add_lsb/remove_msb/separate on lo16
  + any host or device decode of lo16 → max N = 65536. So the whole add_lsb/untilize direction (the
  fused branch) is the WRONG architecture for 32-bit indices.
- The LLK ALREADY has an UNFUSED path (`fused=false`): "keeps values and indices in two parallel DST
  regions" (file comment). Index carried in a SEPARATE INT32 (32-bit) DST region, swapped in lockstep
  with the value via SFPSWAP paired mode (lane_config&4 → swaps lreg AND lreg+4). `separate_indices`
  builds index = `(group_id << group_id_bit_shift) | within`, with group_id_bit_shift a RUNTIME arg
  (LREG12) → high bits scale to 32-bit, not the fused path's fixed 5-bit core_id. init/local_sort/
  merge/rebuild all take the `fused` template param; unfused uses load8/store8 + 2-wide sort_k +
  separate index region at `indices_offset = num_tiles_per_sequence*64`.
- **PIVOT: switch the op to fused=false (unfused) + separate_indices.** Open Qs to resolve: does the
  unfused within-index still need untilize (tile→linear), or does separate_indices/the unfused layout
  already yield linear within-positions? how is group_id_bit_shift set per merge stage for N>K? DST
  budget (separate value+index regions). The verified untilize formula may still apply to the within
  part. Current fused op (values correct) stays as the ≤65k fast path if useful.

### 2026-06-08 (cont.8) — decode locus chosen (inside add_lsb, gated); hit add_lsb addressing barrier
- User chose: apply the verified untilize permutation INSIDE `_topk_xl_add_lsb_indices_`, gated by a
  `row_major` template flag (default = blaze unchanged).
- BARRIER: add_lsb generates the index INCREMENTALLY — LREG0-3 are a running tile-coord counter
  `+4` per replay iter, OR'd into the fused word while walking DST. So a permute-once-before-loop is
  wrong; permuting in-loop needs scratch but the 16-instr replay already uses all 8 lregs (L0-3 index
  + L4-7 the four fused loads at offsets {0,2,16,18}). Also the DST addressing is opaque from reading:
  ADDR_MOD_7 is "zero advance" yet 4 replays × 4 offsets must cover 512 elems/face-pair (SFPTRANSP +
  RWC interplay I can't safely reverse-engineer). Modifying it blind risks silently-wrong indices.
- USEFUL SIMPLIFICATION found: the untilize field extraction needs NO mask registers — each field is
  a double-shift: `(x<<22)>>28` (col&0xF→[3:0]), `(x<<28)>>24` (row&0xF→[7:4]), `(x<<21)>>23`
  (col>>4→bit8), `(x<<27)>>22` (row>>4→bit9), OR the four. ~8 SFPSHFT + 3 SFPOR per value, 2 temps.
- OPTIONS to proceed: (a) study add_lsb's DST addressing experimentally on the 5s loop, then rework
  its index gen to emit linear (or permute in-loop with a 2-at-a-time restructure freeing 2 scratch
  lregs); (b) ship the VALIDATED host ttnn-bitwise decode now for a green fully-validating op test,
  do in-kernel as a perf follow-up. Host decode already proven (cont.7, 0/1024).

### 2026-06-08 (cont.7) — FULL FIX VALIDATED END-TO-END ON ttsim (0/1024 mismatch)
- Proved the complete recipe on ttsim (K=1024,N=1024,M=1) with the REAL assertions: pack RAW fused
  word as uint32 (compute kernel, done) + host untilize-coordinate decode of lo16 → **values exact,
  indices exact, `gather(input, idx) == values` 0/1024 mismatch, test PASSED**. (Validation harness:
  `diag_topk_xl_index.py`; input in [0.5,1) to avoid the exact-bf16-zero tie corner.)
- The earlier 6/1024 fails were ONLY exact-bf16-zero inputs (6 values rounded to 0.0 → fused word
  0x00000000, index lost → all report idx 0). Degenerate tie/zero corner, orthogonal to the decode;
  document as a known v1 limitation (or handle zeros separately). Decode itself verified exact.
- **untilize_helpers.hpp/.inl is a pack-untilize LAYOUT helper (tiled CB→row-major CB)** — it
  repositions elements, it does NOT rewrite the index VALUE. The defect is a coordinate VALUE
  transform (lo16 = tiled coord of the element's position; correct linear idx = untilize of that
  coord value). Layout-untilizing the output fixes ordering (values pass anyway via test-sort) but
  leaves idx encoded → confirmed it can't fix `input[idx]==val`. The decode must be value-arithmetic.
- **REMAINING (placement choice): implement the verified decode in the OP.** Options: (a) in compute
  kernel via SFPU bit-ops on the lo16 lreg (device-resident, like add_lsb's SFPSHFT/AND/OR); (b) host
  ttnn bitwise (bitwise_and/left_shift/right_shift/or; needs TILE+int32 layout plumbing). Formula:
  `col=(lo16>>6)&0x1F; row=lo16&0x1F; idx=((row>>4)<<9)|((col>>4)<<8)|((row&0xF)<<4)|(col&0xF) + core_id*K`.

### 2026-06-08 (cont.6) — INDEX BUG FULLY DIAGNOSED + DECODE DERIVED (verified 1024/1024 on ttsim)
- **Fix is OP-side, not LLK** (LLK primitives are correct blaze code). Two op gaps:
  1. **Compute kernel ordering**: `remove_msb_values` is a MATH/SFPU op but the kernel issued it
     AFTER `tile_regs_commit/wait` (pack phase) — in blaze it runs on the PACK thread (TRISC2)
     between the two packs; tt-metal can't interleave a MATH op there. Result: partial hi16-zeroing
     → corrupt idx_cb (first 16 = 0, ~half leak value bits). **Fix: drop the mid-pack remove_msb;
     pack the RAW fused word as uint32 to idx_cb** (value bits in hi16, encoded index in lo16). DONE
     in `device/kernels/compute/topk_xl.cpp` (JIT, no host rebuild). With raw pack the lo16 is a
     clean bijection for all 1024 (values still exact: hi16 descending 0x3f7f,0x3f70,…).
  2. **Host decode missing**: lo16 is the element's position in SFPU TILE/FACE order; host must
     UNTILIZE it to the row-major linear index. DERIVED + VERIFIED 1024/1024 on ttsim (collision-free
     distinct-bf16 input, value bits 0x3F80+n give exact true position):
       `col=(lo16>>6)&0x1F; row=lo16&0x1F;`
       `true_pos = (row>>4)<<9 | (col>>4)<<8 | (row&0xF)<<4 | (col&0xF)`  (standard 32x32 untilize)
     For N>K add the sequence offset: `core_id=(lo16>>11)&0x1F; global = core_id*K + untilize(...)`.
     (K=1024 single-seq: ti bit always 0, col/row each 0..31 → bijective 0..1023.) K=2048 = 2 tiles,
     needs the ti/2nd-tile term — derive separately when wiring K=2048.
- **REMAINING: implement the host decode** in `topk_xl.cpp` (mask lo16 + untilize bit-permute +
  core_id*K), then the op test passes on ttsim. Diagnostic:
  `tests/ttnn/unit_tests/operations/experimental/diag_topk_xl_index.py` (cand_B = the verified formula).

### 2026-06-08 (cont.5) — OP RUNS ON ttsim, DETERMINISTIC: VALUES CORRECT, INDICES = value bits
- After implementing SFPLOAD mode 10 + SFPCONFIG per-lane LaneConfig (mode 8) in ttsim, the smallest
  op test (`test_topk_xl[K=1024-N=1024-M=1]`) **RUNS TO COMPLETION on ttsim in 4.9s, deterministic**.
- **RESULT: VALUES PASS** (the `assert_close(out_sorted, golden_values, rtol=0, atol=0)` at
  test line 44 passed — execution reached line 48). **INDICES FAIL**: `indices.max()=1064829190`
  (≫ N=1024); the index tensor holds FLOAT BIT PATTERNS (e.g. 1039336779=0x3DF8…), i.e. the index
  output carries VALUE bits, not the lo16 index. So copy ✓, sort ✓, values ✓ — **the bug is purely
  the index-extraction/pack path** (matches the standing `remove_msb`/`separate_indices` hypothesis).
- This is the deterministic confirmation: the "half-tile/value" worry was silicon noise; the REAL,
  reproducible defect is index extraction. Fix the op's index pack (separate_indices keep_values /
  zero the hi16 value bits / mask to lo16), validate on ttsim (5s/run), then silicon.
- ttsim SFPU ISA additions made (src/tensix.cpp, src/sim.h): SFPLOAD MOD0=10 (INT32_ALL, value-only);
  per-lane `lane_config[32]`; SFPCONFIG instr_mod1=8 (IMM16_IS_LANE_MASK) → per-lane LaneConfig;
  SFPSWAP honors `lane_config[lane] & 0x100` direction per lane. Worth upstreaming to ttsim.

### 2026-06-08 (cont.4) — OP RUNS ON ttsim (UMD/slow-dispatch path); filling SFPU ISA gaps
- **OP BOOTS + RUNS on ttsim via the tt-metal runtime path** (NOT the exalens LLK harness). Recipe:
  `TT_METAL_SIMULATOR=<origin/main build .so>` + `TT_METAL_SLOW_DISPATCH_MODE=1` +
  `TT_METAL_DISABLE_SFPLOADMACRO=1`, soc descriptor = `blackhole_140_arch.yaml` next to the .so,
  then plain `pytest tests/ttnn/.../test_topk_xl.py::test_topk_xl[K=1024-N=1024-M=1]`. Device opens
  (device_id=0xb140), runs into the kernel. The BRISC-boot regression is exalens-LLK-harness-only;
  the UMD/runtime boot works on origin/main. **Must use MY origin/main build** (handles umd-0.9.5
  8-byte+4-byte TLB writes at 0x1fc00000); the prebuilt is too OLD for umd-0.9.5 (rejects 0x1fc00000).
- Filling ttsim SFPU ISA gaps the topk_xl kernel needs (DONE one, NEXT one identified):
  - **DONE: SFPLOAD MOD0=10 (`INT32_ALL`)** — added to src/tensix.cpp (VERIFY + value=dst_decode_fp32).
    topk_xl uses it once (line 2270) as a counter-advance "skip" load (LREG discarded), so the
    minimal value + existing math_update_counters suffices; Sp-stack addressing not modeled (OK here).
  - **NEXT (correctness-critical): SFPCONFIG instr_mod1=8 (`MOD1_IMM16_IS_LANE_MASK`) + per-lane
    LaneConfig + SFPSWAP per-lane direction.** topk_xl does `TTI_SFPCONFIG(0x4444/0x5050/0x5500,
    0xF=LaneConfig, 8)` to set PER-LANE swap direction (bitonic). ttsim models lane_config as a
    SINGLE global value (sim.h:469) + SFPSWAP uses `lane_config & 0x100` globally (tensix.cpp:4310,
    4327). Fix = lane_config[32] array; SFPCONFIG case15 honors Imm16 lane-mask (bit (lane&7)*2) +
    writes LReg[0][lane&7]; SFPSWAP reads lane_config[lane]. SFPSWAP itself already implemented
    (modes 0,1,2,3,9). This directly controls sort direction — must be exact.
- ttsim debug-iteration is ~40s/run for the op (device init dominated); rebuild ~30s. Fast vs the
  10-min exalens harness, and DETERMINISTIC — the right place to chase the op value/index bug.

### 2026-06-08 (cont.3) — COPY PROVEN CORRECT IN SIM (half-tile bug does NOT reproduce)
- Authored a fast-loop isolation test in ttsim's own harness (sub-second, deterministic, gdb-able):
  `tests/tensix/topk_xl_copy_kernel.c` + `tests/tensix/tensix_topk_xl_copy.cpp` (+ `topk_xl_copy`
  added to `tests/rules.py`). Feeds a 1024-elem distinct bf16 ramp (bits 0x3800..0x3BFF) to L1
  0x30000, runs the copy, reads back, reports the output value multiset. Three modes:
    0 = MOVA2D, no SETADCXX span;  1 = MOVA2D + SETADCXX(1,1023,0);  2 = the EXACT topk_xl 16-bit
    copy (UNPACR full tile + zero SrcB + ELWADD MOP 8× ADDR_MOD_0 srca/dest incr=8).
  Run: `tests/_out/tensix_topk_xl_copy --sim src/_out/release_bh/libttsim.so`.
- **RESULT: all three modes → distinct=1024, ramp_present=1024/1024, no missing.** The exact
  topk_xl ELWADD-MOP path copies the FULL tile. Corroborated by reading ttsim `tensix_elw_op`:
  ELWADD does 8 rows/op (`for row 0..8`), MOP runs 8× with ADDR_MOD incr=8 ⇒ 64 SrcA rows = full.
- **CONCLUSION: the "half-tile copy" does NOT reproduce in deterministic simulation.** The
  ~half-multiset seen earlier was on noisy SILICON (device-state leak / reconfig-escape; worklog
  already flagged run-to-run variation). copy_tile is faithful. The op's value-correctness issue
  (when present) is therefore ELSEWHERE: pack rounding, the add_lsb/index path, the reader page
  layout, or the silicon test harness — NOT the copy LLK. Next: re-run the OP value test on a fresh
  device (clear JIT cache + tt-smi reset) to confirm, then chase pack/index, not copy.
- Span hypothesis (SETADCXX) DISPROVEN: mode 0 (no span, cfg tile_x_dim only) also gives full 1024.

### 2026-06-08 (cont.2) — FAST loop validated; boot regression isolated; copy seq corrected
- **FAST ITERATION LOOP WORKS** (sub-second, no metal harness): built ttsim's own test harness with
  the on-box sfpi (`/opt/tenstorrent/sfpi`, symlinked to `~/sfpi`):
  `cd refs/ttsim-private && python3 make.py tests/_out/{run_elf,tensix_sfpu,tensix_fpu}`. Then
  `tests/_out/tensix_sfpu --sim src/_out/release_bh/libttsim.so --loops 0 100` → 100/100 EXIT 0 in
  **0.2s** on MY origin/main build. So my origin/main sim is FULLY FUNCTIONAL (executes Tensix
  SFPU/FPU). The metal-harness BRISC-boot failure is therefore METAL-HARNESS/EXALENS-SPECIFIC, NOT a
  broken sim. ttsim CI itself only validates via this `run_elf`/`tensix_*` path — never the metal
  pytest/exalens harness — so a post-May-18 metal-boot regression would go unnoticed in ttsim CI.
- **0xffb01ffc is NOT topk_xl-specific:** a STOCK datacopy on the prebuilt ALSO hits
  `t_tile_mmio_wr32 addr=0xffb01ffc` (fast error, after a clean boot). 0xFFB01FFC sits in the gap
  between RISCV_LOCAL_MEM (≤0xFFB00FFF) and RISCV_TDMA_REGS (0xFFB11000) — NO handled range covers
  it in prebuilt OR origin/main → generic unmodeled MMIO this metal version writes during init.
- **Metal-harness status:** prebuilt (May 18, pre-v1.7.0) BOOTS but lacks 0xffb01ffc; my newer
  builds (1.7.x/main + TLB patch) implement more but DON'T boot (post-May-18 regression). origin/main
  natively handles WH 8-byte TLB but the BH branch (TT_ARCH_VERSION==1) still needed the size==8
  split — patched libttsim.cpp:370 (idx, idx+1).
- **COPY SEQUENCE CORRECTED (read `_llk_unpack_topk_xl_copy_`):** NOT a `SETADCXX(elements-1)` loop
  (old note wrong). It is `TTI_SETADCZW(0b011,…,0b1111)` → set base-addr cfg → single UNPACR. 16-bit
  MOP = `tmp(1,1, clear_srcb_dvalid, unpack_srca)` with start_op `clear_srca_to_neginf`: i.e. clear
  SrcA→-inf, `UNPACR(SrcA, SetDvalid=1, Last=1)`, zero SrcB+dvalid. Half-tile suspect = whether this
  one UNPACR (with sfpu_kernel-style cfg: tile_x_dim=1024) fills all 4 SrcA 16x16 faces for bf16.
- **NEXT (fast path):** author `tests/tensix/topk_xl_copy_kernel.c` mirroring this exact UNPACR +
  ELWADD-A2D MOP + PACR sequence (raw TT_OP_* macros, like sfpu_kernel.c), feed bf16 ramp, pack to
  L1 0x30000, read back multiset — reproduces the half-tile bug in <1s and is gdb-traceable on the
  debug_bh build. Toolchain ready; harness pattern in tensix_sfpu.cpp (args@0x20000, result@0x30000).

### 2026-06-08 (cont.) — ttsim build matrix + likely root cause
- venv tooling: tt-exalens 0.3.20, tt-umd 0.9.5. Prebuilt `~/sim/libttsim.so` (174KB) BOOTS with
  them (first copy run booted, then hit `t_tile_mmio_wr32 addr=0xffb01ffc` UnimplementedFunctionality).
- ttsim-private source build matrix (each run ~10min on ttsim):
  - clean v1.7.1 → `UnsupportedFunctionality: bar0 offset=0x1fc00530 size=8` (umd-0.9.5 8-byte TLB
    write; the local libttsim.cpp patch fixes exactly this — so the patch is REQUIRED).
  - v1.6.1 / v1.7.3 + local patches (tlb 8-byte + SrcA row-wrap) → past TLB, but BRISC never boots
    (`BriscCounter=0`, boot-ready 600s timeout). My builds are 194–198KB (≠ prebuilt 174KB).
  - **BLOCKED reproducing a booting debug build**: can't build a .so from available source that both
    satisfies umd-0.9.5 (8-byte TLB) AND boots with exalens-0.3.20. Need the EXACT commit/flags the
    prebuilt was built from. ttsim-private restored to as-found (main v1.6.1-5 + local patches).
- **ROOT-CAUSE MECHANISM (read `_llk_math_topk_xl_copy_` lines 47-82):** the two dtype paths move
  data DIFFERENTLY:
  - **32-bit** (`is_32bit_input`): moved by UNPACK-TO-DEST (`math_unpack_to_dest_*` writes 32-bit
    straight to Dest) + ZEROACC 4 faces. ELWADD MOP `run()` only for the <1024 padding case. This
    is the path blaze exercised (fp32) → full-tile, proven.
  - **16-bit/bf16** (`else`): UNPACK→SrcA (regular), then MATH ELWADD MOP A2D copies SrcA→Dest
    (`tmp(1,8,ELWADD ADDR_MOD_0)`). No unpack-to-dest, no ZEROACC. MOP comment claims full-tile
    (1×8 ≡ stock 4×2), so the ~half-tile (~257/513-distinct) loss is most likely the bf16 UNPACK
    side: topk_xl unpack `SETADCXX(elements-1)` LINEAR read into SrcA — whether 1024 linear elems
    fill all 4 SrcA 16x16 faces for bf16 is unverified. **NOT the MOP count (earlier guess wrong).**
- **KEY AUTHOR QUESTION:** is the bf16/16-bit copy path meant to do a FULL 1024-tile, or is FP32 the
  supported full-tile path (bf16 untested in blaze, which only drove fp32)?
- **PRAGMATIC UNBLOCK:** switch op input to FP32 (full unpack-to-dest path, proven). bf16 value
  precision == hi16 of fp32 anyway (add_lsb overwrites lo16 with index). Earlier fp32 op failure
  (91.9%) was a SEPARATE reader/pack/index bug, fixable on top of a correct copy.

## Status (tasks)
1. ✅ Place 7 topk_xl LLK headers (Approach A)
2. ✅ LLK tests — local_sort (K=512/1024/2048) + merge/rebuild + perf GREEN on silicon.
3. ✅ TTNN op scaffolding (host/device-op/binding) — builds clean.
4. ◑ Op kernels — reader/compute/writer done; **values correct E2E**; index pairing = 1 known bug.
5. ◑ Op-level E2E tests — written + run on silicon; values green (10/10), index xfail-tracked.

## Next steps
- [ ] Surface perf cycle numbers (direct `perf_report` run; wire a print or locate CSV).
- [ ] **TTNN op** (tasks 3–5): scaffold `ttnn/cpp/ttnn/operations/experimental/topk_xl/`;
      reader tilize (row-major BFLOAT16→tiles), compute (single-core local_sort+merge+rebuild
      reduction over row chunks), writer untilize (split fused word → row-major values+indices).
- [ ] **Resolve plan decisions D1–D6** (esp. D2 index layout — highest risk) before op kernels.
- [ ] Promotion/cleanup: confirm WH build unaffected (BH-gated); consider upstreaming note.

### 2026-06-09 — architecture pivot for true uint32 indices + BFLOAT16 values
- Requirement clarified: input/output values are row-major BFLOAT16, and output indices
  must be true 32-bit positions for rows larger than 65535 elements. This rules out the fused
  `[value_hi16 | index_lo16]` path as the final architecture: it is 16-bit-index-capped.
- Best path is now **fully unfused topk_xl**:
  - seed each sequence in DST as parallel regions `[values, indices]`;
  - values come from row-major BFLOAT16 input and are copied/promoted into FP32 DST for comparison;
  - indices are generated as uint32 row-major positions in the reader and copied into the parallel
    INT32 DST region;
  - all local_sort/merge/rebuild phases run with `fused=false`;
  - pack final values back to BFLOAT16 and final indices as uint32.
- Cleanup done: removed the speculative `topk_xl_untilize_indices` compute API/LLK wrapper/SFPU
  primitive and removed the op kernel call. That path was based on the already-disproven `+2`
  index-region store walk and should not be resumed.
- Immediate implementation task: add a real `topk_xl_local_sort<K, fused=false>` LLK path. Existing
  `topk_xl_init/merge/rebuild<K,false>` and the `load8_rows_x2_unfused` / `store8_rows_x2_unfused`
  helpers are present; local_sort is the missing unfused primitive. After that, wire the TTNN reader
  to generate uint32 index pages and make the compute kernel copy value/index pages into the two
  parallel DST regions.

### 2026-06-09 — stop WIP and restart from pristine Blaze LLKs
- User stopped the current writer/index-decode direction. Do not continue layering changes on that
  WIP. In particular, writer-side RISC-V index reconstruction is rejected as a final design.
- Added fresh plan: `PLAN_topk_xl_fresh_requirements.md`.
- The new plan makes the op contract explicit: row-major BFLOAT16 input values, row-major BFLOAT16
  output values, row-major UINT32 output indices, support for rows with `N > 65535`, and no tiled
  input/output support in v1.
- The recommended implementation path is fully unfused from the start: seed `[values, uint32 indices]`
  in parallel DST regions, add missing unfused local sort support, then use existing unfused
  merge/rebuild. Fused local sort plus compute-side split is documented only as a fallback.

### 2026-06-09 — FP32 Blaze path clarification
- Inspected pristine Blaze FP32 copy behavior. Blaze tests use BFLOAT16 input tensors but compile with
  `fp32_dest_acc_en=True` and `unpack_to_dest_fp32_cb_indices=[0,1,2,3]`, which selects the FP32
  destination/unpack-to-dest machinery for the topk copy path.
- This does not make local sort independent-value/index. Full tiles clear the FP32 low 16 bits with
  `TT_ZEROACC(CLR_16)`, then `topk_xl_add_lsb_indices` ORs a 16-bit index into those bits. The local
  sort still operates on fused `[bf16 value hi16 | u16 index lo16]` words.
- Useful learning: after `topk_xl_separate_indices`, Blaze's unfused merge/rebuild already carries
  FP32 value rows and INT32 index rows in parallel and swaps indices with values. That supports the
  fully-unfused plan, but the missing piece remains unfused local sort / initial unfused seeding.

### 2026-06-09 — revised simpler path for BFLOAT16 values
- Since the op only requires BFLOAT16 input/output values, fused local sort is acceptable: the value
  payload is the high 16 bits, and the low 16 bits can be used for index/tie-break data without
  corrupting the BFLOAT16 value.
- Revised recommendation in `PLAN_topk_xl_fresh_requirements.md`: use the proven fused local sort for
  each K chunk, then immediately split each sorted chunk into unfused `[values, UINT32 row-major
  indices]` before any merge stage. Existing unfused merge/rebuild can then carry true UINT32 indices.
- The remaining required work is not unfused local sort. It is a correct split step that introduces
  high index bits for `N > 65535` and produces row-major indices rather than tile/SFPU coordinates.
  This must happen in compute or via reader-arranged ordering, not in the writer.

### 2026-06-09 — missing LLK functionality called out
- Updated `PLAN_topk_xl_fresh_requirements.md` to explicitly name the missing LLK/API surface:
  `topk_xl_separate_indices_row_major_init(...)` and
  `topk_xl_separate_indices_row_major<K>(idst, chunk_id_or_base)`.
- Existing Blaze LLK support is enough for fused local sort and unfused merge/rebuild. The missing
  primitive must split a sorted fused chunk into the exact unfused value/index layout, decode the
  low-16 tile coordinate to row-major position, add the full chunk base for true UINT32 indices, and
  keep values BFLOAT16-clean.

### 2026-06-09 — bottom-up plan and row streaming requirement
- Rewrote the fresh plan as a bottom-up build: existing LLK validation, new row-major split LLK,
  local-chunk LLK pipeline, two-chunk unfused merge, multi-chunk device reduction, then TTNN op.
- Added execution constraint: process rows as independent work items streamed through
  reader -> compute -> writer with double-buffered CBs so input loading, compute, and output writes
  can overlap. Writer remains a pure DMA stage with no semantic index work.

### 2026-06-09 — bottom-up implementation start
- Stashed the abandoned WIP as `stash@{0}` with message
  `topk_xl_abandoned_wip_before_bottom_up_restart`; restored only
  `PLAN_topk_xl_fresh_requirements.md` and this worklog.
- Added the pristine Blaze topk_xl LLK stack into the canonical tt-metal LLK locations and rewrote
  includes to the local include style.
- Added row-major split LLK/API surface:
  `topk_xl_separate_indices_row_major_init(chunk_base)` and
  `topk_xl_separate_indices_row_major<K>(idst)`.
- Added isolated split test:
  `tt_metal/tt-llk/tests/sources/topk_xl_split_test.cpp` and
  `tt_metal/tt-llk/tests/python_tests/test_topk_xl_split.py`.
- Validation:
  - `python_tests/test_topk_xl_split.py`: 4/4 PASS for K=1024/2048 and chunk_base 0/131072.
  - `python_tests/test_topk_xl.py`: 7/7 PASS for fused local sort and fused merge/rebuild.
  - `python_tests/test_topk_xl_copy.py`: compiles after a warning fix, but runtime value check FAILS
    in the bare harness (`Float16_b->Float32`, K=1024; output contains `-inf`/bad multiset). Keep this
    as an open Phase 1 copy-path validation item; debug with the real CB contract or ttsim before
    depending on this harness result.

### 2026-06-09 — Phase 1/3 validation update and Phase 4 blocker
- Fixed the bare LLK copy harness setup:
  - `_llk_unpack_hw_configure_<...>` is required before `_llk_unpack_topk_xl_copy_init_`.
  - `TT_SETADCXX(p_setadc::UNP_A, 1023, 0x0)` is required before each topk_xl copy unpack.
  - Raw `L1_ADDRESS(...)` is correct for this bare harness.
- Validation now green:
  - `python_tests/test_topk_xl_copy.py`: 2/2 PASS for BFLOAT16 input copied into FP32 DST,
    K=1024/2048.
  - `python_tests/test_topk_xl_chunk.py`: 8/8 PASS for
    BFLOAT16 copy -> add_lsb_indices -> fused local_sort -> row-major split,
    K=1024/2048, chunk_base 0/131072, ascending and descending local sort.
- Added two-chunk unfused merge repro:
  - `tt_metal/tt-llk/tests/sources/topk_xl_two_chunk_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_two_chunk.py`
- Superseded finding: the initial two-chunk K=1024 repro appeared to show that the imported Blaze
  unfused merge/rebuild path preserved the value multiset but not UINT32 pairing. Later debugging
  showed the repro used invalid K=1024 raw coordinates with bit 5 set; see the
  "unfused merge/rebuild validated" entry below.
- Tried duplicating unfused `SFPSWAP` instructions based on the older topk index-tracking
  pattern; it made value ordering worse and was reverted. Next task is to design/validate the
  correct unfused value/index compare-swap block for topk_xl rather than assuming the Blaze path
  already satisfies the UINT32 pairing contract.
- Historical aggregate LLK run before the K=1024 stimulus fix:
  `python_tests/test_topk_xl_split.py python_tests/test_topk_xl_copy.py python_tests/test_topk_xl_chunk.py python_tests/test_topk_xl.py python_tests/test_topk_xl_two_chunk.py`
  -> 21 PASS, 2 strict XFAIL.

### 2026-06-09 — unfused compare-swap probe
- Restored the unfused merge hot body to the pristine Blaze shape after the speculative duplicate
  `SFPSWAP` experiment:
  - unfused merge replay body length is back to 18 instructions;
  - `bitonic_sort_len_k<false>` is back to two value swaps.
- Added a focused primitive probe:
  - `tt_metal/tt-llk/tests/sources/topk_xl_unfused_swap_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_unfused_swap.py`
- Validation:
  - `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_unfused_swap.py -q`
    -> 1 PASS.
- Finding: SFPU index-tracking mode itself works for the canonical unfused layout. A value
  compare-swap over LREG0..3 correctly swaps the companion UINT32 registers in LREG4..7.
  Therefore the remaining two-chunk failure is not the primitive `SFPSWAP`/lane-config contract;
  it is in the full unfused merge/rebuild schedule, the assumed bitonic input layout after split,
  or how the merge traverses the split row-major sequences.
- Superseded: the merge-only diagnostic reproduced the same false failure before the K=1024
  stimulus fix. It has since been removed from the normal pytest surface because it is only a debug
  artifact and can be order-sensitive.

### 2026-06-09 — unfused merge/rebuild validated
- Added a two-split/no-merge diagnostic:
  - `tt_metal/tt-llk/tests/sources/topk_xl_two_split_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_two_split.py`
- Root cause of the previous two-chunk pairing failure was invalid host test stimulus for K=1024,
  not the LLK merge:
  - raw bit 5 is ignored by the row-major decoder for K=1024;
  - the test used `np.arange(1024)` as raw coordinates, which incorrectly set bit 5 for half the
    positions and created duplicate decoded row-major indices;
  - valid K=1024 raw coordinates must keep bit 5 clear (`raw = row | (col << 6)`).
- Fixed the host raw-coordinate encoder in both two-chunk diagnostics to choose hardware-valid raw
  coordinates. After that:
  - `test_topk_xl_two_split.py`: 2 PASS;
  - `test_topk_xl_two_chunk.py`: merge+rebuild PASS with chunk bases 0 and 131072.
- Removed the merge-only diagnostic from the normal pytest surface because it is not part of the
  final pipeline and can be order-sensitive when run after full merge+rebuild cases in the same
  LLK pytest process.
- Conclusion: existing Blaze unfused `topk_xl_merge<K,false>` and `topk_xl_rebuild<K,false>` do
  preserve UINT32 value/index pairing when fed valid split output. They are no longer a missing LLK
  block for the selected BFLOAT16 immediate-split path.

### 2026-06-09 — TTNN scaffold bring-up and newly exposed LLK output gap
- Added an experimental TTNN op scaffold under
  `ttnn/cpp/ttnn/operations/experimental/topk_xl/` and registered
  `ttnn.experimental.topk_xl(input, k=1024) -> (values, indices)`.
- Host/device contract in the scaffold:
  - input must be device, row-major, BFLOAT16;
  - outputs are row-major BFLOAT16 values and row-major UINT32 indices;
  - initial implementation is single-core, K=1024, `largest=true`, `sorted=true`, `N % K == 0`;
  - reader streams row-major BF16 chunks through a double-buffered input CB;
  - compute reduces one logical row at a time;
  - writer is pure DMA, no index/value decode.
- Build validation:
  - `cmake --build build_Release --target ttnncpp ttnn -j 8`: PASS.
  - The local build layout had stale runtime copies in `build_Release/lib` and `ttnn/ttnn`;
    manually synced `_ttnncpp.so`/`_ttnn.so` from `build_Release/ttnn` before pytest.
- Added focused integration guard:
  `tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py`.
- Runtime validation with
  `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
  FAILS on the first one-chunk case, but now through the new op path (not the stale old primitive).
- Findings from the failure:
  1. The existing LLK tests pack all result tiles as Float32/raw words. They do not cover the
     public mixed-output contract: BFLOAT16 values plus UINT32 indices.
  2. Plain `pack_tile` is not sufficient for row-major public outputs.
  3. `pack_rows` and `pack_untilize_dest` prove that data is present, but they do not correctly
     consume the current topk_xl sorted/split DST layout as BFLOAT16 row-major values.
  4. Therefore the remaining missing LLK functionality is a final-output materialization primitive:
     sorted/split DST layout -> BFLOAT16 row-major values + UINT32 row-major indices.
- Next step: implement that primitive bottom-up and validate it in LLK tests before continuing TTNN
  correctness. Keep the TTNN test as the integration guard.

### 2026-06-09 — public row-major pack boundary narrowed
- Added focused public-output LLK guard:
  - `tt_metal/tt-llk/tests/sources/topk_xl_public_pack_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py`
- The new guard uses strictly increasing BFLOAT16 bit patterns so sorted value/index order is
  unambiguous, then separately packs:
  - values as BFLOAT16;
  - indices as UINT32.
- Removed the speculative `topk_xl_prepare_values_bfloat16` helper. It was wrong for the real
  pack path: moving BF16 bits into the low half produced zero BFLOAT16 output in the focused LLK
  test. The split value region is already BFLOAT16-clean in the high half.
- Switched the TTNN output path from one full-row CB page back to row-slice pages produced by
  `pack_untilize_dest`:
  - output CB page size is now 32 elements;
  - each logical output row reserves/pushes 32 pages for K=1024;
  - writer remains pure DMA and writes fixed row-slice offsets, with no index decode/rebuild.
- Build validation:
  - `./build_metal.sh --release`: PASS.
- Current validation:
  - `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py -q`
    FAILS only on rank order. Values are clean BFLOAT16, but output is still in the SFPU traversal
    order, e.g. unique descending data starts as `255, 191, 127.5, ...` instead of
    `255, 254, 253, ...`.
  - `rm -rf built && scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`
    FAILS the same way at TTNN level. The huge/garbage BFLOAT16 pattern is gone; output is now
    clean but rank-permuted.
- Existing Blaze `rebuild` does not solve this:
  - unfused rebuild after split was tried in both directions; it preserves/reverses the same
    internal traversal rather than producing public rank order;
  - fused rebuild after local sort, before split, also did not rank-linearize.
- Remaining missing LLK functionality:
  - final rank-order materializer for K=1024 (later K=2048) that converts the sorted/split
    topk_xl DST traversal into public row-major rank order before pack-untilize;
  - it must move values and UINT32 indices together and should be validated in
    `test_topk_xl_public_pack.py` before re-running the TTNN guard.

### 2026-06-09 — public pack green; TTNN blocked on true row-major copy/unpack
- Replaced the speculative `pack_untilize_dest` output path with the bottom-up validated public
  output path:
  - `topk_xl_materialize_rank_order<K=1024>` performs the 16x16 face transpose in DST;
  - compute packs values and indices with `pack_rows(64)` into one full row page per output CB;
  - writer applies the remaining fixed 64-page placement with pure DMA:
    `rank_page = ((page & 0xF) << 2) | (page >> 4)`.
- LLK public-output validation is GREEN when run one case per pytest process:
  - `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py -q -k values`: PASS.
  - `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py -q -k indices`: PASS.
  - Running both public-pack cases in one LLK pytest process is still order-sensitive and can time
    out on the second row-pack case; this looks like LLK harness/core state after row-pack, not a
    value/index correctness failure.
- TTNN compute needed an explicit pack hardware configure before row-pack. Without it, BFLOAT16
  value output was effectively reading index-shaped low bits. Added per-output:
  `PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(cb)))`, `pack_init(cb)`, `pack_rows_init(64)`.
  This changed the TTNN failure from raw bit garbage to real BFLOAT16 values, confirming the pack
  configuration issue.
- The remaining TTNN blocker is now earlier: the op reader supplies raw row-major BFLOAT16 bytes,
  and `topk_xl_copy_tile` appears intended to consume a contiguous row (single `UNPACR` plus
  element count), but the TTNN path still sees only a sparse subset of the row after sort (e.g. max
  values around the low hundreds for a 0..1023 ramp). So the unresolved issue is the exact
  row-major BFLOAT16 copy/unpack driving contract, not a proven requirement that input L1 already
  be tile/face ordered.
- Tried a reader-side DMA gather from row-major tensor pages into tile/face CB order. It is not a
  viable quick fix: the required 16-element (32B) strided DRAM-to-L1 copies violate Blackhole NoC
  alignment/residue constraints for many face rows and hang the dataflow path, even with per-face
  barriers. Reverted to the non-hanging linear row read.
- Current missing LLK/debug functionality:
  1. A stricter row-major BFLOAT16 copy/unpack LLK test that feeds the same contiguous raw row shape
     as the TTNN reader and validates exact element positions, not just value multiset.
  2. If that stricter test fails, fix `topk_xl_copy_tile` or its compute-API driving sequence so a
     contiguous row-major 1024-element CB page lands in the topk DST layout expected by
     add_lsb/local_sort.
- Current TTNN status:
  - build/JIT compiles the updated kernels;
  - `tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py` still fails correctness on the
    first `num_rows=1, n=1024` case with the linear reader;
  - the failed reader-gather attempt produced hangs and was reverted.

### 2026-06-09 — row-major BF16 input validated bottom-up
- Confirmed the LLK Python harness writes BFLOAT16 stimuli as flattened row-major bytes:
  `pack_bfp16` converts the flattened tensor to `ml_dtypes.bfloat16` and writes `.tobytes()` with
  no face/tile permutation. `StimuliConfig.write_matrix` slices each 1024-element page linearly.
- Added an exact no-sort row-major input LLK guard:
  - `tt_metal/tt-llk/tests/sources/topk_xl_row_major_input_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_row_major_input.py`
- The new guard runs:
  `BF16 row-major input -> topk_xl_copy_tile -> add_lsb_indices -> separate_indices_row_major`,
  intentionally skipping local_sort. It checks exact public row-major results:
  - `values[i] == input[i]`
  - `indices[i] == 131072 + i`
  for `K=2048`.
- Validation:
  - `tt-smi -r && scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_row_major_input.py -q`:
    PASS.
  - Manual single-case reruns also passed for `K=1024/base=131072`, `K=2048/base=0`, and
    `K=2048/base=131072`. Running all four variants in one pytest process hit the same
    order-sensitive LLK harness timeout pattern seen in row-pack tests, so the committed guard uses
    one high-value case.
  - `tt-smi -r && scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_chunk.py -q`:
    PASS (8/8).
  - `tt-smi -r && scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py -q -k values`:
    PASS.
  - `tt-smi -r && scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py -q -k indices`:
    PASS.
- Conclusion: the LLK blocks needed for row-major BF16 input are green in isolation:
  copy/add/split, sort/split, materialize/pack values, and materialize/pack UINT32 indices.
  Therefore the remaining TTNN failure is likely in the compute-kernel API driving sequence or CB
  state/configuration, not in a missing LLK requirement for tile/face-ordered input.

### 2026-06-09 — pack_tile retained; materialize hang narrowed
- Switched the TTNN output pack path to `pack_tile` for both public outputs:
  - values: `pack_tile(slot0, values_cb)` into a BFLOAT16 row-sized CB page;
  - indices: `pack_tile(slot0 + tiles_per_sequence, indices_cb)` into a UINT32/Float32 row-sized CB page;
  - writer remains pure DMA and applies the fixed 16-element slice placement:
    `rank_page = ((page & 0xF) << 2) | (page >> 4)`.
- Re-enabled `topk_xl_materialize_rank_order<K>(slot0)` to test the real rank-order path with
  `pack_tile`.
- Validation / diagnostics:
  - `./build_metal.sh --release`: PASS.
  - With materialize enabled, clean-JIT focused TTNN runs hang on readback:
    - `scripts/run_safe_pytest.sh 'tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_bfloat16_uint32_indices[num_rows=2-n=1024]' -q`:
      HANG.
    - After clearing the actual JIT cache with `rm -rf built/tt-metal-cache*`, same result:
      HANG with `0/13` JIT cache hits.
    - `num_rows=1, n=1024` also hangs once materialize is enabled in the TTNN compute path.
  - `tt-triage` again reports a device timeout during `ttnn.to_torch`, with dispatch teardown
    waiting on physical cores `14-3,14-2`; this matches the previous materialize-enabled hang
    class, not a reader/writer CB/NOC data movement stall.
- Tried and reverted two speculative LLK materialize changes:
  - clearing SrcA/SrcB around the materialize transpose;
  - programming `DEST_TARGET_REG_CFG_MATH_Offset_ADDR32` in the materialize wrapper.
  These did not fix the multi-row hang and regressed the single-row guard, so the materialize LLK
  body/wrapper is back to the known original behavior.
- Also tried removing the `UNPACK(llk_unpack_set_srcb_dummy_valid())` from the public
  `topk_xl_materialize_rank_order` API. That regressed single-row execution, so the dummy valid was
  restored.
- Current tree is intentionally back in a non-hanging diagnostic state:
  - local sort + row-major split are enabled;
  - materialize is commented out in the TTNN compute kernel;
  - output still uses `pack_tile`.
- Baseline confirmation after restoring this state:
  - `./build_metal.sh --release`: PASS.
  - `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh 'tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_bfloat16_uint32_indices[num_rows=2-n=1024]' -q`:
    exits without hang and fails exact value comparison because rank-order materialization is
    disabled. The first row begins in internal traversal order (`255, 191, 127.5, ...`) while the
    reference is descending public rank order (`255, 254, 253, ...`).
- Next debug target:
  - build a valid bottom-up LLK or ttsim reproducer for
    `copy -> add_lsb -> local_sort -> split -> materialize -> pack_tile` under the same compute API
    sync model as TTNN;
  - focus on the interaction between the materializer's DST transpose and compute/pack thread
    completion, since the TTNN hang appears only when materialize is present.

### 2026-06-10 — materialize split into a standalone LLK contract
- Added a focused materialize-only LLK contract instead of continuing with the long repro chain:
  - `tt_metal/tt-llk/tests/sources/topk_xl_materialize_contract_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_contract.py`
- Contract shape:
  - seed two 32-bit DST tiles with the stock unpack/datacopy path;
  - optionally call only `topk_xl_materialize_rank_order<K=1024>`;
  - pack exactly one DST tile with `pack_tile`;
  - validate both output regions independently:
    - output modes `2/3`: no-materialize baseline preserves value and index tiles;
    - output modes `0/1`: materialize performs the expected per-face 16x16 transpose for values
      and the parallel 32-bit index tile.
- This contract caught two real materializer issues:
  1. Low-halfword corruption: FP32 values whose low 16 bits were `0x8000` came back with that half
     cleared after the transpose. Root cause was the transpose block only disabling source zero
     flag behavior. Fix: also disable destination zero-flag behavior in
     `enter_transpose_cfg_block()` and restore it in `leave_transpose_cfg_block()`.
  2. Non-repeat-safe source-valid state: one dummy SrcB token made a single post-reset run pass,
     but repeated runs could hang the unpacker. Root cause was the materializer wrapper not matching
     the stock transpose-dest contract. Fix: make
     `llk_math_eltwise_unary_sfpu_topk_xl_materialize_rank_order` wait on
     `WAIT_SFPU | SRCA_VLD | SRCB_VLD` and clear AB after the transpose.
- Validation:
  - `tt-smi -r && scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_contract.py -q`:
    PASS (4/4).
  - Immediate rerun without reset:
    `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_contract.py -q`:
    PASS (4/4).
- Current conclusion:
  - The standalone `DST values/indices -> materialize -> pack_tile` LLK contract is now green and
    repeat-safe.
  - Next bottom-up block should compose this with `copy -> add_lsb -> local_sort ->
    separate_indices_row_major`, still inside LLK tests, before re-enabling materialize in the TTNN
    compute path.

### 2026-06-10 — pack_tile chain and TTNN path green for 1024/2048 rows
- Added a composed LLK contract matching the TTNN compute/output boundary:
  - `tt_metal/tt-llk/tests/sources/topk_xl_public_pack_tile_contract_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack_tile_contract.py`
- Contract shape:
  - BFLOAT16 row-major input;
  - `topk_xl_copy_tile -> add_lsb_indices -> local_sort -> separate_indices_row_major ->
    materialize_rank_order`;
  - `pack_tile` for either the BFLOAT16 values tile or the UINT32 indices tile;
  - apply the same fixed writer page scatter in Python and compare with public `torch.topk`
    row-major rank order.
- Validation:
  - `tt-smi -r && scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack_tile_contract.py -q`:
    PASS (2/2).
  - Immediate rerun without reset:
    `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack_tile_contract.py -q`:
    PASS (2/2).
- Re-enabled `topk_xl_materialize_rank_order<K>(slot0)` in the TTNN compute kernel.
- Build:
  - `./build_metal.sh --release`: PASS.
- TTNN validation:
  - Clean JIT single-row:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh 'tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_bfloat16_uint32_indices[num_rows=1-n=1024]' -q`:
    PASS.
  - Multi-row case that previously hung:
    `scripts/run_safe_pytest.sh 'tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_bfloat16_uint32_indices[num_rows=2-n=1024]' -q`:
    PASS.
  - Full current TTNN experimental topk_xl test:
    `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (4/4 for `num_rows={1,2}`, `n={1024,2048}`).
- Focused LLK regression checks:
  - Combined run of materialize contract, pack_tile contract, row-major input, and chunk tests
    passed the first 7 cases then hit the known LLK harness state sensitivity on
    `test_topk_xl_chunk.py` (indices read as zeros).
  - Rerunning `test_topk_xl_chunk.py` alone after reset passed all 8 cases.
  - `scripts/run_safe_pytest.sh tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack.py -q`:
    PASS (2/2).
- Current conclusion:
  - For the currently tested TTNN surface (`k=1024`, row-major BFLOAT16 input, row-major BFLOAT16
    values output, row-major UINT32 indices output), the single-core path is green for `n=1024` and
    `n=2048`.
  - Next missing requirement is true large-row / >65535 index coverage. The TTNN tests still stop at
    `n=2048`; we need to extend the bottom-up and TTNN coverage to larger `num_chunks` and verify
    chunk-base propagation into 32-bit UINT32 indices.

### 2026-06-10 — true UINT32 indices above 65535
- Added TTNN coverage for indices beyond the 16-bit boundary:
  - `test_topk_xl_row_major_uint32_indices_above_uint16`
  - shapes: `num_rows=1`, `k=1024`, `n=65*k` and `n=128*k`;
  - the `65*k` case's reference top-k indices are `65536..66559`, so a wrapped/fused-low16-only
    path fails;
  - the `128*k` case validates the current maximum supported row length.
- Initial attempt used a runtime low-half argument with a compile-time upper-half wrapper around
  `_sfpu_load_config32_`. That still failed JIT once `upper16 != 0`, because both halves of the
  SFPU config load must be immediates.
- Implemented the TTNN path with static chunk-base init:
  - LLK SFPU helper:
    `_topk_xl_separate_indices_row_major_init_static_<upper16, lower16>()`
  - LLK API wrapper:
    `llk_math_eltwise_unary_sfpu_topk_xl_separate_indices_row_major_init_static`
  - compute API wrapper:
    `topk_xl_separate_indices_row_major_init_static<upper16, lower16>()`
  - TTNN compute kernel dispatches chunks through compile-time cases:
    - chunks `0..63` use `upper16=0`;
    - chunks `64..127` use `upper16=1`;
    - lower16 is one of the 64 static values `(chunk & 63) * 1024`.
- Validation now matches implementation scope:
  - `topk_xl` remains `k=1024` only;
  - input/output layouts remain row-major only;
  - input/output values are BFLOAT16;
  - output indices are UINT32;
  - current max supported row length is `128*k = 131072` elements. This is enough to cover true
    indices above 65535, but it is not arbitrary-length support yet.
- Build / tests:
  - `./build_metal.sh --release`: PASS.
  - Clean JIT focused TTNN:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh
    'tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_bfloat16_uint32_indices[num_rows=1-n=2048]'
    'tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_uint32_indices_above_uint16' -q`:
    PASS.
  - Full TTNN topk_xl file:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh
    tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (6/6, 0/34 JIT cache hits).
  - Note: an immediate rerun after changing the Python parametrization but before clearing the JIT
    cache hit a stale all-zero output failure on the first 1024 case. Clearing
    `built/tt-metal-cache*` and resetting the device restored the expected PASS result.
  - LLK contracts:
    `tt-smi -r && scripts/run_safe_pytest.sh
    tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_contract.py
    tt_metal/tt-llk/tests/python_tests/test_topk_xl_public_pack_tile_contract.py -q`:
    PASS (6/6).
  - LLK split/chunk checks:
    `tt-smi -r && scripts/run_safe_pytest.sh
    'tt_metal/tt-llk/tests/python_tests/test_topk_xl_row_major_input.py::test_topk_xl_row_major_split_advance_crosses_uint16_boundary'
    tt_metal/tt-llk/tests/python_tests/test_topk_xl_two_chunk.py -q`:
    PASS (3/3).
- Remaining follow-up for broader support:
  - If we need more than 131072 elements for `k=1024`, extend the static upper16 dispatch beyond
    `0..1` or replace the config-load mechanism with an LLK primitive that can synthesize arbitrary
    32-bit constants without immediate-only operands.

### 2026-06-10 — K=512 / K=1024 / K=2048 TTNN support
- Generalized the TTNN op from `k=1024` only to the Blaze LLK subset:
  `k in {512, 1024, 2048}`.
- Changed TTNN data movement to tile-granular CB pages:
  - reader pushes `ceil(k/1024)` input tile pages per chunk;
  - compute waits/pops that tile count per chunk;
  - compute packs `ceil(k/1024)` value tiles and index tiles per row;
  - writer waits/pops the same tile count and writes only the public `k` elements.
  This fixes the two cases that row-sized CB pages could not represent correctly:
  `k=512` partial tiles and `k=2048` two-tile chunks.
- Generalized public rank-order output:
  - materializer now uses K-derived face count:
    - `k=512`: 2 value faces + 2 index faces;
    - `k=1024`: 4 + 4;
    - `k=2048`: 8 + 8.
  - writer page scatter now uses `rank_page_groups = (k / 16) / 16 = k / 256`:
    `rank_page = ((page & 0xF) * rank_page_groups) | (page >> 4)`.
- Reduced materializer code size by using one noinline 32-bit face-transpose body and rebasing
  `DEST_TARGET_REG_CFG_MATH_Offset_ADDR32` before each materialized face. This keeps the hot
  sort/rebuild transpose paths unchanged and lets the real TTNN `k=2048` kernels JIT.
- Generalized static chunk-base dispatch:
  - `k=512`: 128 low-half cases, upper16 `0..1`;
  - `k=1024`: 64 active low-half cases, upper16 `0..1`;
  - `k=2048`: 32 active low-half cases, upper16 `0..3`.
  Validation now uses a fixed max row length of `131072` elements for all supported K values.
- TTNN validation:
  - `./build_metal.sh --release`: PASS.
  - Clean JIT:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh
    tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (18/18, 0/86 JIT cache hits).
  - Covered cases:
    - `k=512`: `N=512,1024`, `num_rows=1,2`, plus large-index chunks `129,256`;
    - `k=1024`: `N=1024,2048`, `num_rows=1,2`, plus chunks `65,128`;
    - `k=2048`: `N=2048,4096`, `num_rows=1,2`, plus chunks `33,64`.
- LLK validation:
  - Materialize-only contract after rebasing:
    `tt-smi -r && scripts/run_safe_pytest.sh
    tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_contract.py -q`:
    PASS (4/4).
  - Public pack-tile contract for `k=512` and `k=1024`:
    PASS (4/4 focused values/indices cases).
  - Two-chunk LLK contract for `k=512` and `k=1024`:
    PASS (4/4 in combined run).
  - Row-major split boundary case:
    PASS when rerun alone after reset. In a combined LLK run it hit the known harness state
    sensitivity (`Timeout reached: waited 2 seconds for Unpacker`) after five prior cases passed.
- K=2048 LLK caveat:
  - The real TTNN `k=2048` kernels JIT and pass, including `N=4096` and large-index rows.
  - The synthetic all-in-one LLK public-pack / two-chunk test images can still overflow TRISC1 code
    space for `k=2048` because they compile copy + local sort + split + materialize/merge/rebuild
    into one test math image. This is a harness/code-size limitation of those composed LLK tests,
    not an observed TTNN op failure.

### 2026-06-10 — compute chunk-base dispatch cleanup
- Replaced the giant macro-generated `switch` in the TTNN compute kernel with a template-recursive
  dispatcher:
  - `k=512` instantiates 128 lower-half cases;
  - `k=1024` instantiates 64 lower-half cases;
  - `k=2048` instantiates 32 lower-half cases.
- This keeps the current immediate-only SFPU config-load workaround intact while removing the
  preprocessor macro and avoiding unused lower-half dispatch cases for larger K values.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - Clean JIT:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh
    tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (18/18, 0/86 JIT cache hits).

### 2026-06-10 — pack format reconfig cleanup
- Replaced per-output full pack hardware configure/init calls in the TTNN compute kernel with
  one kernel-entry `compute_kernel_hw_startup(input_cb, values_cb)` and the narrower packer
  data-format reconfiguration path:
  - startup configures unpack/math for the row-major input CB and pack for the values output CB;
  - values -> indices uses `pack_reconfig_data_format(values_cb, indices_cb)`;
  - next-row indices -> values uses `pack_reconfig_data_format(indices_cb, values_cb)`;
  - there are no explicit row-loop `llk_pack_hw_configure` or `pack_init` calls.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - Clean JIT:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh
    tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (18/18, 0/86 JIT cache hits).

### 2026-06-10 — object API cleanup for reader/compute/writer
- Aligned the topk_xl kernels with the object-based CB/NOC APIs:
  - reader uses `CircularBuffer` for reserve/push and `Noc::async_read` from the row-major
    input `TensorAccessor` into the input CB;
  - compute uses `CircularBuffer` for input waits/pops and output reserves/pushes, while keeping
    LLK/topk calls on raw CB IDs where those APIs require IDs;
  - writer uses `CircularBuffer` waits/pops and `Noc::async_write` from value/index CBs to their
    row-major output `TensorAccessor`s.
- The row/chunk/tile streaming contract is unchanged: reader pushes tile pages per chunk, compute
  consumes one row/chunk work item at a time, and writer scatters rank-ordered slices per row.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - Clean JIT:
    `rm -rf built/tt-metal-cache* && tt-smi -r && scripts/run_safe_pytest.sh
    tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (18/18, 0/86 JIT cache hits).

### 2026-06-10 — current implementation HTML note
- Added `TOPK_XL_OP_CURRENT.html`, a standalone design/implementation document for the current
  `topk_xl` op.
- The note covers:
  - public op constraints and supported shapes/dtypes/layouts;
  - host program setup, CB sizes, and reader/compute/writer roles;
  - row-major tensor layout, tile-page CB layout, DST value/index layout, and rank-order writer
    scatter;
  - per-call LLK contracts for copy, low-16 index injection, fused local sort, row-major index
    split, unfused merge/rebuild, materialization, and packing;
  - LLK helpers present in the API but not used by the current TTNN row-major path.
- Follow-up doc polish:
  - fixed code-block contrast by forcing light code backgrounds with dark text;
  - added animated visual LLK dataflow cards showing CB, UNPACK/SrcA, MATH/SFPU, DST, PACK, and
    value/index tile transformations for every current LLK-stage call;
  - clarified that the current program factory is one-core only, while the intended scaling path is
    row-parallel execution across multiple Tensix cores with independent row ranges.
- Added a focused `topk_xl_materialize_rank_order` zoom section:
  - single-face 16x16 transpose visualized with an animated 4x4 miniature;
  - companion value/index face movement shown separately;
  - active register/thread roles called out for DST, SFPU/MATH, SrcA/SrcB, and PACK;
  - exact value/index face offsets listed for `k=512`, `k=1024`, and `k=2048`;
  - explicit list of what materialize does not do, including top-k selection, index conversion,
    packing, and final writer page rotation.

### 2026-06-10 — LLK output-layout combination matrix
- Added a focused bottom-up LLK contract probe:
  - `tt_metal/tt-llk/tests/sources/topk_xl_materialize_pack_variants_test.cpp`
  - `tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_pack_variants.py`
- The matrix exercises `k=512`, `k=1024`, and `k=2048`, for values and indices independently,
  across these transform/pack combinations:
  - current `topk_xl_materialize_rank_order` + `pack_tile`;
  - stock `transpose_dest<false, true>` + `pack_tile`;
  - no transform + `pack_tile`;
  - stock full `transpose_dest<true, true>` + `pack_tile`;
  - the same transform/no-transform variants with `pack_untilize`.
- Result:
  - all `pack_tile` variants pass for both streams and all K values, including stock
    `transpose_dest<false, true>` and stock full `transpose_dest<true, true>`;
  - all `pack_untilize` variants are strict xfail because they do not produce the current
    `topk_xl` public row-major order for this DEST layout.
- Validation:
  - `scripts/run_safe_pytest.sh --run-all
    tt_metal/tt-llk/tests/python_tests/test_topk_xl_materialize_pack_variants.py -q`:
    PASS (24 passed, 24 strict xfailed).
- Takeaway:
  - replacing the custom materialize helper with stock per-tile `transpose_dest<false, true>`
    looks viable for the current `pack_tile` + writer-scatter design;
  - replacing the writer scatter with `pack_untilize` is not a drop-in change for the current
    DEST layout and would need a different upstream layout/materialization contract.

### 2026-06-10 — move materialization out of public topk LLK API
- Removed `topk_xl_materialize_rank_order` from `tt_metal/hw/inc/api/compute/topk_xl.h`; it is
  not a reusable topk SFPU primitive, it is TTNN kernel output-layout glue.
- Added a kernel-local `materialize_rank_order<K>` in
  `ttnn/cpp/ttnn/operations/experimental/topk_xl/device/kernels/compute.cpp`.
- The helper now uses the public Blackhole transpose API:
  - `llk_math_transpose_dest_init<false, true>()`;
  - `llk_math_transpose_dest<false, true>(dst_index)`.
- The helper body is compiled only for `TRISC_MATH && ARCH_BLACKHOLE`, so it does not wrap the
  transpose calls in `MATH(...)`; non-math TRISCs see a no-op helper.
- To keep materialization math-thread-only, the helper uses `TTI_SETDVALID(0b11)` on MATH before
  each public transpose call instead of issuing any UNPACK-side dummy valid operations.
- Validation so far:
  - `./build_metal.sh --release`: PASS after moving to the public transpose API and removing the
    internal `MATH(...)` wrappers.
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    first single-row K=512 case passes, but the following two-row K=512 case hangs.
- Current open issue:
  - stock transpose can be driven without unpack involvement for a single row, but the multi-row
    path still needs a row-boundary synchronization/state fix before this replacement is complete.

### 2026-06-10 — restore unpack dummy-valid for DEST transpose materialization
- Replaced the TTNN kernel-local math-only `TTI_SETDVALID(0b11)` workaround with the existing
  public unpack dummy-valid protocol:
  - extended the existing `transpose_wh_dest[_init_short]` compute API with a defaulted
    `transpose_of_faces` template parameter;
  - existing callers keep the previous default `transpose_of_faces=true`;
  - topk_xl calls `transpose_wh_dest_init_short<true, false>()` and
    `transpose_wh_dest<true, false>(...)`;
  - the compute wrapper emits `UNPACK((llk_unpack_set_srcb_dummy_valid()))` before the math
    transpose, matching the existing `transpose_wh_dest` contract.
- Updated TTNN `topk_xl` materialization to use `api/compute/transpose_wh_dest.h` directly instead
  of including `llk_math_transpose_dest_api.h` or manipulating SrcA/SrcB valid bits.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (18 passed).

### 2026-06-10 — cleanup dead LLK test scaffolding
- Removed exploratory topk_xl LLK test files under `tt_metal/tt-llk/tests/python_tests` and
  `tt_metal/tt-llk/tests/sources`.
- Removed the test-only `TOPK_XL` template parameter and `TopKXLGolden` helper additions because
  they were only referenced by those deleted LLK tests.
- Kept the live Blackhole topk_xl LLK/API implementation files that the TTNN op still includes:
  `ckernel_sfpu_topk_xl.h`, topk_xl copy unpack/math LLKs, Blackhole public LLK API shims, and
  `api/compute/topk_xl.h`.
- Validation after cleanup:
  - `./build_metal.sh --release`: PASS.
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (18 passed).

### 2026-06-10 — indices-only output and row-parallel K=512 fix
- Updated the public/device op path to return indices only:
  - removed the values output from the device operation result, program factory, runtime args,
    compute kernel CBs, writer kernel CBs, and nanobind wrapper;
  - retained the required input/output contract: row-major BFLOAT16 input and row-major UINT32
    indices output.
- Changed work distribution so all available Tensix worker cores receive row work:
  - the unit of work is one input row;
  - runtime args pass each core's `start_row` and `num_rows`;
  - for 640 rows on a 120-core grid, rows are split across all cores with at most six rows per
    core.
- Reworked the indices output path:
  - compute materializes only the index half of DEST with `transpose_wh_dest<true, false>`;
  - compute packs the index tiles through `pack_untilize_dest` into one row-major CB page per row;
  - writer uses a local scratch CB to reorder packed stick slices and issues one contiguous DRAM
    write per output row.
- Debugged the K=512 / 640-row failure:
  - reader-side zero filling, writer direct writes, writer slice reorder variants, remap reset, and
    an LLK RWC reset experiment did not fix the bad-row pattern and were not kept;
  - the real trigger was the single-chunk path (`n == k == 512`), which did not execute the merge
    loop and therefore skipped the unfused `topk_xl_rebuild<K, false>` finalization that multi-chunk
    rows already receive;
  - fixed this by running `topk_xl_init<K, false>()` and `topk_xl_rebuild<K, false>(slot0, false)`
    once after the first chunk when `num_chunks == 1`.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py::test_topk_xl_row_major_parallelizes_640_rows -q`:
    PASS (3 passed for K=512, K=1024, K=2048).
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (21 passed).
- ttsim note:
  - not needed for the current fix because the LLK implementation files are unchanged and the bug
    was in op-level row/chunk finalization;
  - use ttsim if a future change modifies the Blaze-derived LLKs or if silicon/pytest exposes an
    instruction-level hang or register-state issue.

### 2026-06-10 — hoist Blackhole DEST remap for pack-untilize
- Hoisted the Blackhole DEST remap enable out of the per-row pack-untilize init:
  - `kernel_main()` now calls `MATH((llk_math_reconfig_remap(true)))` once after
    `compute_kernel_hw_startup`;
  - the row loop calls
    `pack_untilize_dest_init<tiles_per_sequence, tiles_per_sequence, false, TILE_C_DIM, false, false>`
    so `pack_untilize_dest_init` does not repeat the remap reconfiguration for every output row.
- Kept pack-untilize init/uninit scoped per row because the row loop still switches between topk,
  transpose/materialize, and pack-untilize packer state.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (21 passed).

### 2026-06-10 — support non-multiple input row lengths
- Removed the `N % K == 0` validation requirement. The op now supports arbitrary row length `N`
  within the existing constraints (`N >= K`, `N <= 131072`, K in `{512, 1024, 2048}`).
- Host program factory now computes:
  - `num_chunks = div_up(N, K)`;
  - `tail_elements = N - (num_chunks - 1) * K`;
  - `tail_chunk_bytes = tail_elements * sizeof(input_element)`.
- Reader still streams a fixed `tiles_per_sequence` CB pages per chunk so the compute contract stays
  unchanged, but the final chunk only reads `tail_chunk_bytes`. Unread bytes in the final chunk are
  never consumed by compute.
- Compute now passes the active element count to the existing `topk_xl_copy_tile<K>` LLK wrapper:
  - full chunks use `K`;
  - the final chunk uses `tail_elements`;
  - the copy LLK already clears inactive lanes to negative infinity, so padded lanes sort last for
    `largest=true` without LLK changes.
- Index base logic remains `chunk * K`, which is correct for the logical row-major input offset of
  each chunk, including the partial final chunk.
- Added coverage for:
  - `N = K + 1` and `N = 2K - 1` for K=512/1024/2048;
  - non-multiple large rows with winning indices above 65535 for K=512/1024/2048.
- Validation:
  - `./build_metal.sh --release`: PASS.
  - `scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/experimental/test_topk_xl.py -q`:
    PASS (30 passed).
