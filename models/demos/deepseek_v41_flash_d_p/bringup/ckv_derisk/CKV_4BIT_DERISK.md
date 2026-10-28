# Compressed KV in 4 bits: de-risk (L0, CPU)

Location: `models/demos/deepseek_v41_flash_d_p/bringup/ckv_derisk/`. The codec is `tt/ckv_codec.py` (tests
`tests/test_ckv_codec.py`, CPU, 8 pass); the studies are `tools/ckv_study.py` / `tools/ckv_bopt.py` (moved 2026-10-28).
DESIGN CHANGE (owner 2026-10-28, supersedes sections 3 and 5 where they differ): only the attention table goes 4-bit;
the served records stay bf16 rows holding the bfp4-rounded values (contract and tt-d-gen unchanged; load_served repacks
exactly). The window goes to the reference's width later on the same two-tensor fork (fp8 per 32: section 7).
Owner rule F208: the cache is stored at the reference's width (fp4) in the device's own format. Overseer 2026-10-28.

## 1. Where the cache lives today (bf16, 1024 B per row)

| what | where |
|---|---|
| the producer: ckv = rope(latent), bf16 TILE `[S/SP/r, 512]` | `tt/compressor.py:456` (`TtCompressor.__call__`), stock rope `:503-517` |
| the per-chunk write | `tt/model.py:422-423` -> `TtCompressor.write` `tt/compressor.py:548` -> `_gathered` (:519, to ROW_MAJOR) -> `TtServedRows.write` `:281` |
| served cache (contract record, TP-dedup stripes) | `TtServedRows._put` `:264` -> `ttnn.bringup.update_padded_kv_cache` (:270 / :277); `[1,1,rows/r,512]` bf16 ROW_MAJOR, one 32-token record per ND shard |
| attention table (STATIC_TABLES=on, default) | `AttnTable` `tt/compressor.py:153`; `append` :184 -> `ttnn.experimental.slice_write` :182 into `[1,1,max_seq/r + 128 + S, 512]` bf16 ROW_MAJOR, **replicated on every chip**, rows `[0, hist)` = history, then the window region (`put_window` :189) |
| host boundary | `TtServedRows.load_host` :321 / `to_host` :312, `AttnTable.load_host` :198; `tt/model.py:214` / `:244` |
| the reader | `tt/sparse_attention.py:255` `_static` -> `tt/attention.py:589` `_sdpa` = stock `ttnn.transformer.sparse_sdpa(kv_format=BF16)`; index list `[window 128 \| top-k 512]` (`_index_static` :238), k_chunk 128 |
| the kernel | `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/sparse_sdpa_reader.cpp:182-246` (BF16 dual-NoC row gather) and `compute/sparse_sdpa_compute.cpp:246-249` (tilize of the gathered rows) |

Access pattern: only `sparse_sdpa` reads the compressed KV, as 512 single rows per query (the top-k). The 2048 x 8-row
candidate blocks belong to the index-key cache (`tt/indexer.py:335`), not to this cache.
38 of 40 layers read one of 4 sources: L2 (L2-7), L8 (L8-13), L14 (L14-19), L20 (ratio 1, L20-39).
The attention table holds two kinds of row: the compressed history, and the bf16 window rows (fp8 in the reference;
a separate fix).

Prior art I read: the GLM SCALED_FP8 path. Its row is `[fp8 latent 512 B | fp32 scales | bf16 rope]`, 656 B, in an
fp8 ROW_MAJOR tensor (`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:52-120`). The kernels are
`sparse_sdpa_reader.cpp:143-197` (gather plus `scatter_packed_scales`) and `sparse_sdpa_compute.cpp:173-245`
(`tilize_packed_field` + `mul_tiles_bcast_cols`, one scale per 128 columns). Row pages that are not multiples of
64 B (656 B) already gather correctly, so a 288-B row page is in the same class.
The K3 cast ops (2125d74fd90, `experimental/deepseek_prefill/per_token_cast_to_fp8`) do an SFPU block amax plus a
power-of-two scale, with the scales packed into the row.
Blackhole has no e2m1 data format: `MxFp4` exists only in the Quasar LLK (`tt_metal/hw/ckernels/quasar`).

## 2. L0 numbers

Input: the golden's native fp32 rows, `golden_full/galaxy_8x4/s10240_c5120/kv_cache/layer_{2,8,14,20}`: 5120 / 5120 /
5120 / 10240 rows. Formats:
- **A** = the reference's `fp4_quant_dequant(block 16, e4m3 scale)`; the codec matches it bit for bit.
- **B** = bfloat4_b, one shared exponent per 16, tt-metal's host packer rule (round to nearest even on the
  exponent-shifted mantissa, clamp at 7).
- **B_trunc** = the same with truncation, in case the device packer truncates.
- **bf16** = what the device stores today.

All errors are relative L2 against native.

### Rows (rel L2 vs native)
| source | bf16 | A | B | B_trunc | A vs B | A rope / B rope dims | min row cos A / B | norm ratio A / B |
|---|---|---|---|---|---|---|---|---|
| L2  | 0.0017 | 0.0958 | 0.1114 | 0.2095 | 0.1470 | 0.099 / 0.106 | 0.9931 / 0.9908 | 1.000 / 1.001 |
| L8  | 0.0017 | 0.0948 | 0.1195 | 0.2249 | 0.1535 | 0.093 / 0.138 | 0.9922 / 0.9859 | 1.000 / 1.003 |
| L14 | 0.0017 | 0.0953 | 0.1115 | 0.2103 | 0.1471 | 0.096 / 0.111 | 0.9911 / 0.9907 | 1.000 / 1.002 |
| L20 | 0.0017 | 0.0955 | 0.1128 | 0.2119 | 0.1480 | 0.097 / 0.108 | 0.9913 / 0.9907 | 1.000 / 1.001 |

- B is 1.17-1.26x A's error. A and B round onto different grids, so they differ by 0.15 from each other.
- Quantizing the device's bf16 rows instead of fp32 changes B by less than 0.0003.
- A per-group best exponent for B (E_max or E_max + 1) gains nothing: 0.11141 -> 0.11141. E+1 is picked in 0.03 %
  of groups (`bopt.py`).

### Attention impact
Setup:
- Real queries: `rope(q_resid wq_b^T)` from chunk 1's dumps, 256 random positions in [5247, 10240).
- 64 heads, keys = [128 native window rows | top-512 compressed rows of the source], the sink included.
- `K.attend` is the reference kernel; native fp32 everywhere except the compressed rows.

Columns:
- **o** = attention output rel L2.
- **logit** = rel L2 of the compressed-key logits.
- **top32** = mean overlap of each (query, head)'s top 32 compressed keys with native.
- **mass** = the share of softmax mass on the compressed keys.

| layer (src) | mass | bf16 o / logit / top32 | A o / logit / top32 | B o / logit / top32 | B_trunc o | B / A (o) |
|---|---|---|---|---|---|---|
| L2 (2)   | 0.65 | 0.0006 / 0.0006 / 0.998 | 0.0402 / 0.0375 / 0.925 | 0.0444 / 0.0427 / 0.915 | 0.217 | 1.10 |
| L3 (2)   | 0.46 | 0.0007 / 0.0005 / 0.998 | 0.0474 / 0.0275 / 0.925 | 0.0528 / 0.0326 / 0.911 | 0.245 | 1.11 |
| L14 (14) | 0.56 | 0.0005 / 0.0008 / 0.998 | 0.0288 / 0.0428 / 0.934 | 0.0312 / 0.0508 / 0.923 | 0.187 | 1.09 |
| L20 (20) | 0.51 | 0.0003 / 0.0005 / 1.000 | 0.0253 / 0.0268 / 0.986 | 0.0262 / 0.0310 / 0.982 | 0.179 | 1.04 |
| L21 (20) | 0.48 | 0.0005 / 0.0005 / 1.000 | 0.0319 / 0.0259 / 0.985 | 0.0321 / 0.0304 / 0.981 | 0.178 | 1.01 |
| L24 (20) | 0.48 | 0.0004 / 0.0004 / 1.000 | 0.0263 / 0.0228 / 0.990 | 0.0287 / 0.0267 / 0.988 | 0.204 | 1.09 |

Notes:
- The indexer's top-512 does not depend on this cache. It scores the separate index-key cache, and the 128-wide
  index query cannot be dotted with the 512-wide ckv rows. That is why I report the attention-logit top-32 overlap
  instead.
- L8's attention could not be measured, because the golden dumps no `q_resid` for L8. Its rows are the worst for B
  (rope dims 0.138).
- Reading: A is the reference's own rounding of the cache. It moves the attention output by 0.025-0.047 from native;
  B moves it by 0.026-0.053, which is 1-11 % more. That is far below the factor 4-5 that truncation would cost.
  Whether B fits depends on the per-step limit: calib x the reference's own drift on the same comparison (rule 14).

## 3. Recommendation: B (row-packed bfloat4_b), on one condition

- **Accuracy.** B is close to A where it matters: attention output +1..+11 %, top-32 overlap -0.2..-1.4 points.
- **Size.** Both are 288 B per row, 3.56x less DRAM, gather traffic and memory than bf16. The replicated attention
  table at 1M tokens and r = 2 drops from 512 MiB to 144 MiB per source per chip.
- **Read decode.** For B the unpacker reads bfloat4_b natively, so decoding is a byte shuffle on the data-movement
  RISCs only. The compute engine does no decode; today's K tilize disappears, and the QK / PV matmuls read a 4-bit
  operand. For A, Blackhole has no e2m1 format, so every gathered row of every query must be decoded in software
  (cost below).
- **Spec.** It matches the spec's own wording ("fp4 -> bfp4", owner 2026-10-28) and the experts' precedent
  (DS41-10 / DS41-74).
- **Condition.** The write must round like the host packer. Truncation doubles the row error (0.21) and makes the
  attention output 4-5x worse (0.18-0.25). The first L1 test measures this. If the device packer truncates, the
  write kernel rounds to the 3-bit grid of the group's max exponent on the SFPU (a few ops) before packing.

### Kernel work, B
- **Write: a new fork op** (proposed `ttnn.bringup.ckv_pack_bfp4`), or an option of `update_padded_kv_cache`.
  - Input: ckv bf16 TILE.
  - Packer (or SFPU round + packer) to bfloat4_b tiles in L1.
  - The writer RISC turns each 32-row band of 16 tiles into 32 row records of 288 B (`codec.tiles_to_rows`).
  - Rows go into the served slab, with the records in the server's order, and into the attention table.
  - Both tensors become uint8 ROW_MAJOR `[.., rows, 288]`, with no padding of the shape.
  - Per chunk this is S/r rows, which is small.
- **Read: a sparse_sdpa fork option `kv_format=ROW_BFP4`.** Fork the stock op first (INDEX.md: the `sdpa` fork
  holds precision options only).
  - Reader and writer each gather half of a 32-row slab, as the SCALED path does: 288-B pages, 16-B aligned.
  - Each RISC shuffles its own rows into 16 bfloat4_b tiles (`codec.rows_to_tiles`): per row, 32 exponent bytes and
    32 copies of 8 bytes. Rows are disjoint, so the two RISCs' writes do not collide.
  - The tiles are pushed to `cb_k_in` as Bfp4_b, with no tilize and no scale multiply.
  - The sentinel tail stays zero rows (exponent 0, mantissa 0).
- **Window rows.** The window must stay in its own format (bf16 now, the reference's fp8 later), so the fork reads
  two KV tensors. The index list is already `[window 128 | top-k 512]` with k_chunk 128, so chunk 0 is the window
  and chunks 1-4 are compressed.
  - The compaction for starts < 128 (`SparseIndexTables` :109) currently moves top-k ids into chunk 0. It must keep the
    two parts apart: for example, the window chunk carries its own valid count and the top-k always starts at
    chunk 1.
- **Contract and host.**
  - `serving_contract.md`: 1_compress_kv_r2 and 3_compress_kv_r1 become 288 B/row records. The server ships bytes
    by position, so it is unaffected apart from the width; this is still an owner / server sign-off.
  - `load_host` / `to_host` encode and decode with the codec.
  - `state.formats()` reports the new format.
- **Throughput risk.** Each query re-shuffles its 512 rows (no reuse across queries). My estimate is 200-400 RISC
  cycles per row, about 40-80 us per query per core on two RISCs. That is 1-2 queries per core per layer at
  chunk 5120 on 8x4. The shuffle has to overlap the gather and the matmuls; L1 measures it.

### Kernel work, A (if B fails the accuracy limits)
- **Write.** A new op, extending the K3 per_token_cast_to_fp8 idea.
  - SFPU amax per 16 columns: half a tile row, so a face-wise reduction.
  - Scale = round_e4m3(amax/6), done with integer bit operations.
  - Thresholds |x| against mid_k x s: exact products, avoiding x/s. A division-free compare can differ from the
    reference's x/s at exact ties, so it needs a bit-exact test.
  - Codes and nibble packing on the RISC, plus the row-major write.
- **Read.** No hardware decode is available. Two options:
  - (i) RISC lookup expansion (2 nibbles -> 2 bf16 per load/store), then the existing tilize, then a group-16 scale.
    That is 2 scales per tile row, so `mul_tiles_bcast_cols` needs a face-wise variant or a full scale tile built
    by the RISC. About 3x the shuffle cost of B.
  - (ii) Tilize the bytes as raw 16-bit, then an SFPU decode of 4 nibbles per element into 4 tiles. The resulting
    column permutation is folded into wq_b at load time, and the output is un-permuted before the inverse RoPE.
    Then a scale multiply, about 400 SFPU cycles per decoded tile, which roughly doubles the compute per query.
  - A's decode itself is exact: e2m1 x e4m3 needs at most 5 significant bits, so it fits bf16, and bfp8 when a face
    row is one scale group.

## 4. CPU codec (the oracle for L1)
`codec.py` provides:
- `encode_a` / `decode_a`
- `encode_b` / `decode_b(truncate=)`
- `rows_to_tiles` / `tiles_to_rows`, which use tt-metal's bfp tile byte layout: 64 exponent bytes (face 0..3, rows
  0..15), then the faces' mantissas, 8 B per face row, with the low nibble first.

`test_codec.py`: 8 tests pass (`python -m pytest -q -p no:cacheprovider test_codec.py`, 1.5 s). They check:
- A is bit-exact with `reference/kernels.py:fp4_quant_dequant` on random, tiny, zero, tie, large and golden rows.
- A's decode is exactly representable in bf16.
- Both formats round-trip their bytes.
- B matches a scalar transcription of `convert_u32_to_bfp<Bfp4_b>` element by element.
- The tile shuffle is invertible, and decoding tiles by the tt-metal layout equals the row decode.

Not yet checked: the codec's bytes against ttnn's own host packer (`ttnn.from_torch(bfloat4_b)`). Importing ttnn on
this host opens the cluster, so this is L1 test 0.

## 5. L1 test plan (one card, `scripts/run_safe_pytest.sh` with a single-card selector, after the overseer frees the mesh)
0. **Host packer.** Bytes of `ttnn.from_torch(rows, bfloat4_b, TILE)` == `rows_to_tiles(encode_b(rows))`.
1. **Device packer rule.** `ttnn.typecast` / a pack of golden + random rows to bfloat4_b on the device, read back:
   bytes vs `encode_b` round-to-nearest-even and vs truncation, and the same from an fp32 DEST. This decides
   whether the write needs an SFPU pre-round.
2. **Write kernel.** ckv bf16 TILE `[n, 512]`, with n = 2560 / 5120 (a chunk's groups) and an incomplete last group:
   - the uint8 ROW_MAJOR records equal `encode_b` exactly;
   - the served-slab path (update_padded_kv_cache with 288-B rows, metadata path for r = 2) and the attention table
     hold the same bytes, read back by `to_host`.
3. **Gather + decode.** 512 random row ids and 2048 8-row blocks of a 64k-row table, plus sentinel tails of
   0 / 5 / 31 rows. Dump the L1 tiles (a debug output of the fork): bytes == `rows_to_tiles`, values == `decode_b`.
4. **sparse_sdpa ROW_BFP4** at DS41 shapes (H per chip, 640 keys, the bf16 window + 512 compressed): vs CPU `attend`
   on `decode_b` rows, with the rule-14 limit and the devfmt diagnosis; also the old bf16 path on the same inputs.
5. **Micro-benchmark** (device profiler): sparse_sdpa bf16 vs ROW_BFP4 kernel time at chunk 5120 / 8x4 per-chip
   query counts, the shuffle cycles per row, and the gather bandwidth at 288-B pages.

L2 (later): `/scratch/dnijemcevic/diag/test_repro_tail.py` with REPRO_LAYERS=3, behind a setting
`CKV_FORMAT=bf16|bfp4_rows` (default unchanged until the gates say so).

## 6. Proposed known-issues entry (to add when the tree is free; next id DS41-133)
- **[DS41-133] The 4-bit compressed KV: bfloat4_b rows land within 1-11 % of the reference's fp4 on attention output.**
  - Symptom: none (the device stored bf16).
  - Cause: F208 needs fp4 storage. Against native rows, A (e2m1 + e4m3 per 16) is rel 0.095 and B (bfp4_b, round
    to nearest even) is 0.112; truncation is 0.21. Attention output: A 0.025-0.047, B 0.026-0.053, truncation
    0.18-0.25.
  - Fix: B as 288-B rows; the write must round to nearest even. Detail: CKV_4BIT_DERISK.md.
  - Found: deepseek_v41_flash_d_p de-risk 2026-10-28.

## 7. Prior art cited (added by the overseer 2026-10-28, both re-read)
- **K3 MoE FP8 Wire** (claude.ai artifact 6fGHNmeV9wazFvBeJM4fff; branch `pjosipovic/k3-moe-fp8-wire`, tip babcaf396f6,
  ported to deepseek_prefill as 2125d74fd90 on `origin/claude/glm53-prefill-best*`). Fp8 e4m3 with one power-of-two
  scale per 32 values, the 112 fp32 scales in the row tail (4032 B rows; segmented 1024 B with `packed_row_align=64`).
  Dequantize = one token per tile row, `mul_tiles_bcast_cols` with one scale per row (FPU, no SFPU on the data); 1/scale
  exact as 2^-e. The packer truncates to e4m3; their RNE pass was removed with no change in final PCC (0.99404 fp8 vs
  0.99424 bf16 wire). Fp8 buffers force 32-bit DEST (~20 us). Bearing here: option A's write (section 3) and the window
  KV (reference fp8 per 32, power-of-two) map onto these ops nearly as-is (`scale_block_w=32`, `pack_scales_in_row`), an
  alternative to bfp8 tiles for the window. Their "truncation is fine" holds for fp8 on a MoE wire; for 4 bits this
  report measured truncation at 2x the row error and 4-5x the attention error (section 2), so it does not carry over.
- **PR #49811, "Power-of-two FP8 scaling"** (deck https://pavlejosipovic.github.io/fp8-scaling-presentation/, branch
  `pjosipovic/power-of-two-fp8-scaling`). `per_token_cast_to_fp8(round_scale_to_power_of_two=)`, block 128, fp32 scales;
  host E4M3FN fix (finite to 448, overflow saturates to 0x7E). Blackhole hardware fact: the DEVICE PACKER TRUNCATES the
  e4m3 mantissa ([448, 480) -> 0x7E) while the host encoder rounds to nearest even, so device and host can differ by one
  ULP; device scales can differ from the host's by one power of two at a reduction threshold. The 656 B GLM row
  (512 e4m3 + 64 bf16 RoPE + 4 fp32 scales) is PR #49812. Neither source covers bfp4 / fp4 / e2m1, bfp8, or the
  bfp4 packer: L1 test 1 (section 5) is still the only evidence for the bfp4 rounding rule, and it should cover bfp8
  too (FIX.7's GEMM inputs are packed to bfp8 on the device).

## 8. L1 test 1 result: the device packer does not round bfp4 to nearest (2x2 QB, 2026-10-28)
`tests/test_bfp_packer_rounding.py` (one card, ~1 min): crafted tiles packed by the device (a copy kernel from 16- and
32-bit DEST, ttnn.typecast, a matmul with a bfp output), the packed tiles read back as raw bytes and compared byte for
byte with CPU rules; each case below matches its rule exactly.
- **The packer rounds to bfloat8_b (7 bits), half away from zero.** With `bfp8_pack_precise` (what ttnn.typecast to
  bfloat8_b sets) it rounds once at the group's shared exponent. Without it (the default: a matmul's bfp output, a
  typecast to bfloat4_b) the packer's source format is bfloat8_b and each value is first rounded to 7 significant bits
  at its own exponent (a value may carry into the next exponent and bump the group's), then as above.
- **bfloat4_b = that bfloat8_b with the low 4 bits dropped: truncation in effect.** Row error on the golden compressed
  KV (layers 2 / 8 / 14 / 20, rel L2 vs native): host RNE 0.112-0.120, device 0.199-0.214, truncation 0.208-0.224.
  So option B needs a write that rounds to nearest itself (section 3's condition).
- **bfloat8_b (FIX.7's GEMM inputs) rounds, it does not truncate.** Golden rows: RNE 0.0075-0.0080; typecast (precise)
  the same; the default path (matmul output dtype bfp8) 0.0090-0.0096; truncation 0.0121-0.0131. It differs from the
  host packer only at exact ties (and the double rounding of the default path).
- **tt-metal bug found:** `ComputeKernel::config_hash()` (tt_metal/impl/kernels/kernel.cpp, Wormhole / Blackhole)
  leaves `bfp8_pack_precise` out of the kernel's cache key (the Quasar version has it): two kernels that differ only in
  it share one compiled binary (12 copy configurations gave 6 cache entries, all with the non-precise pack format).
  Stock ops are not hit as far as seen (typecast always sets it for bfp8 outputs, never for bfp4); a fork that toggles
  it would be. Fix proposed (append the flag to the hash only when set, so existing hashes do not change); not made.
