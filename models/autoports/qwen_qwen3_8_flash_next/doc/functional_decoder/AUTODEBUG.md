# AutoDebug: Qwen3.8-Flash-Next functional-decoder feasibility

Date: 2026-11-12. Inspection only; no device access, device runs, launcher, recursive agents, or implementation changes. This report is the only file written.

## Findings and decision

**The current probes do not establish a passing decoder, but they also do not establish that the requested decoder is impossible on one Blackhole.** The concrete obstacles are the lookup implementations, physical storage layout, exact hash arithmetic, and missing decoder/trace acceptance evidence.

1. **Native tiled gather admits block-float inputs that its element-copy kernels cannot interpret.** The probe's BFP8/BFP4 gather results are consistent with this direct source-contract violation. BFP4 additionally produces a calculated zero-byte element width that falls through to 16-bit accesses. This is a correctness and buffer-bounds problem, not evidence that all compressed lookups fail.
2. **`ttnn.embedding` is a BF16 row-major lookup behind a whole-table conversion.** BFP8 success on 4,096 rows does not demonstrate resident compressed-table lookup. BFP4 fails during construction of an unsupported row-major tensor, before the embedding reader. Merely admitting BFP4 or changing the error string would not solve capacity.
3. **A naïve `[2,500,012,20]` UINT32 packed-row table followed by ordinary `gather(..., dim=0)` is not a viable repair.** On Blackhole, its 80-byte rows consume 128-byte DRAM pages; gather also transposes the table and requests approximately 10 MB of L1 staging per participating core. Packing records into larger pages and introducing direct selected-page/row access removes these particular blockers.
4. **Dense 4-bit storage with a shared codebook and one BF16 scale per row is a concrete capacity candidate.** A grouped-page design below consumes **26,241,007,616 bytes**, including per-shard/per-bank rounding. Keeping the other layer-1 weights in their checkpoint BF16 form leaves **2,684,085,544 bytes** from the reported free capacity, before their physical padding, runtime state, activations, trace, and workspaces. Accuracy at decoder PCC ≥ 0.995 remains unproved.
5. **Exact hashing can be represented using two UINT32 limbs; no native TTNN I64 tensor is required.** Products and XOR must retain their full precision until remainder. A static integer check passed 16,432 comparisons for the proposed limb/remainder representation. This is not device execution evidence.
6. **A CPU-precomputed PLE embedding is a valid diagnostic submodule boundary, but it changes the requested complete decoder boundary if used to claim acceptance.** The installed HF decoder performs PLE lookup inside the decoder. Its ordinary `inputs_embeds` argument does not bypass PLE.

The smallest credible continuation is an exact device hash plus a bounded-memory packed lookup/dequantization primitive, followed by the missing decoder implementation and real-weight prefill/decode/trace validation. The present blocker is **no implemented and validated compressed lookup/decoder path meeting the contract**, not the BF16 byte count alone. There is no configuration-only fix supported by this audit.

## Scope and evidence ledger

The inspected repository HEAD is `d7e788cd88199501a8546ddb7bc381a601af47a9`. The target directory is untracked in this checkout. Its complete inventory at inspection contained two Python probes and evidence/log files; no `functional_decoder.py`, TT decoder, hash implementation, or packed lookup implementation was present. Therefore module-to-module TT/HF equivalence cannot yet be established. This report audits the available operators and derives the missing implementation contract.

Pinned reference:

- Snapshot: `/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa`.
- Installed distribution: `transformers==5.22.2`, verified through package metadata. The snapshot config names **`qwen4_exp` / `qwen4_exp_text`**, despite the published model name. Use the installed Qwen4Exp reference, not an assumed Qwen3 decoder.
- Reference file: `python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py`; SHA-256 `e57b3617e71e81ecb7e5ae3b420c45b5aee80d76a07aa8c88b797c2642f1f5df`.
- `config.json` SHA-256: `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.
- `model.safetensors.index.json` SHA-256: `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.

Direct observations and their limits:

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| [Storage probe JSON][probe-json] and [probe source][probe] | BF16 round-trip/embedding/gather PCC 1; BFP8 round-trip 0.9999703169 and embedding 0.9999709129; both compressed gathers report NaN PCC; BFP4 embedding raises a layout error; BFP4 storage PCC 0.9933435917 | Decoder accuracy, all shards, full-table capacity, trace replay, exact hash correctness |
| Probe loads `shard_0[:4096,:]`, uses 16 handpicked row IDs twice | A small, real-weight lookup experiment | Distribution of token-hashed IDs, all 128 shards, all head ranges, or stateful decode |
| Probe memory view: 8 banks × 4,272,341,376 bytes | Reported free capacity 34,178,731,008 bytes at that probe's initialization | Free space after decoder setup, fragmentation, or trace reservation; it opens with `trace_region_size=0` |
| Safetensors headers read directly in this audit | All 128 PLE shards are BF16 `[2500012,160]`; exact weight totals below | Physical TT allocation sizes or quantized accuracy |
| Tiny I64 checkpoint buffers read directly | Actual multipliers, primes, offsets | Device arithmetic execution |
| Source inspection of wrapper → validation → factory → kernels | Concrete layout, dtype, staging, and arithmetic contracts | Exact first corrupted runtime tensor, device timing, or successful custom-kernel compilation |

`gather_pcc: NaN` is a NaN **correlation**, not a saved demonstration that every output element is NaN. A constant/invalid result can also make correlation undefined. The artifact does not save tensors or finite/variance statistics; the diagnosis below relies on the kernel contract rather than inventing those missing observations. The log does not record a source revision or compiled-binary hash, so attribution to the current tree is source-supported and sequence-compatible, not binary provenance proof.

The capacity analysis assumes one representative decoder layer loaded at a time, covering all main text-layer kinds. It does not assume all 48 layers resident simultaneously or claim full-model feasibility.

## 1. Why the existing compressed lookup probes behave differently

### Embedding: conversion precedes the selected-row read

The call is `embedding(uint32 RM IDs [1,32], TILE weights [4096,160], layout=TILE)` with default `runtime_rows=False` and no output dtype override ([probe][probe], lines 29–63).

In [embedding.cpp][embedding-wrapper], lines 30–42, **the entire TILE weight argument is converted to ROW_MAJOR before lookup**. The primitive then requires ROW_MAJOR BF16 weights, interleaved IDs/weights, and UINT32 or BF16 IDs ([embedding validation][embedding-validation], lines 53–67). The 32 IDs and width 160 select the fused tilized-output factory. This is an output-tilization choice, not compressed table support.

The shared [untilize output-dtype helper][untilize-dtype] converts BFP8 to BF16, but returns BFP4 unchanged. [Untilize output-spec construction][untilize-spec] requests ROW_MAJOR with that dtype. [Tensor-spec validation][tensor-spec] forbids ROW_MAJOR for both BFP8 and BFP4; its shared error text misleadingly names only BFLOAT8_B. This accounts for BFP4's error without alleging that the checkpoint or probe selected the wrong dtype.

For BFP8 the chain is:

`TILE BFP8 table → full RM BF16 table → selected BF16 rows → TILE output`.

Thus the observed embedding PCC close to the storage round-trip is expected. At probe scale the expanded table is only 1,310,720 bytes. For one real shard, expansion is at least 800,003,840 bytes; for a single concatenated table it exceeds 102.4 GB. Keeping all BFP8 shards resident is itself too large, even before conversion. Processing shards one by one from host storage during decode would change the device-only contract.

The existing [embedding RM reader][embedding-reader] does have the right access pattern conceptually: read an index, fetch only its row using a tensor accessor, wait for the NoC read, then publish the row. Its implementation and validator do not currently provide the proposed packed decode operation. Changing `runtime_rows`, requesting a low-precision output, or adding a BFP4→BF16 whole-table cast does not implement that operation.

### Gather: the lowered block-float path violates its data format

The probe calls `gather(w, 0, index)` with TILE `w=[4096,160]` and UINT32 TILE `index=[32,160]`. [Gather routing][gather-routing] chooses codegen only when its support predicate permits it. [The codegen predicate][gather-codegen] admits BF16 input only and rejects both BFP dtypes. Therefore compressed weights take native gather.

[Preprocessing][gather-transform] transposes dimension 0 to the last dimension, producing input `[1,1,160,4096]` and index `[1,1,160,32]`. The relevant WH transpose does not promote BFP8 to BF16 ([transpose wrapper][transpose-wrapper], lines 307–343). The intervening [padding helper][fill-padding] returns immediately for these tile-aligned dimensions, so its conditional BF16 staging cannot rescue this probe. Nothing in native gather's input validation rejects BFP4/BFP8 ([native gather validation][gather-validation], lines 46–159).

Concrete factory values are:

| Quantity | Compressed probe path |
|---|---:|
| Tile geometry | 32 × 32, four 16 × 16 faces |
| `Ht` | 160 / 32 = 5 |
| `Wt_input` | 4096 / 32 = 128 |
| `Wt_index` | 32 / 32 = 1 |
| Selected native factory | `SingleRowMultiCore`, because 128 > threshold 60 |
| Work split | One active core for the one index tile; one loop per each of 5 tile rows |
| Input CB capacity | Two input tiles: BFP8 2,176 B; BFP4 1,152 B |
| Output CB capacity | One output tile: BFP8 1,088 B; BFP4 576 B |

Sources: [factory selector][gather-validation], lines 17–43, and [native factory][gather-factory], lines 201–355. Factory names do not imply that this case uses many active cores.

The [native reader][gather-reader], lines 80–86, computes element widths using integer `tile_bytes / 1024`. The resulting BFP8 width is **1**, and BFP4 width is **0**. The [read/write helpers][gather-helpers] treat these widths as ordinary integer array elements; the default case for zero uses `uint16_t`. But [the actual BFP layout][bfp-packing], lines 482–500, puts shared exponents before packed mantissas. The [tile-size definitions][tile-sizes] give 1,088/576 bytes, not 1,024 independent 1-byte/half-byte scalar records.

- BFP8 copies bytes starting in the exponent section as if they were element payloads and writes only 1,024 bytes of a 1,088-byte packed output. It does not construct valid output exponents for arbitrarily gathered values.
- BFP4 uses 16-bit array indexing for a 576-byte tile: the 1,024 output positions imply 2,048 bytes of stores. That exceeds the one-tile output CB. Corresponding input indexing also exceeds a packed tile's bounds. Increasing the CB would not repair the incorrect format interpretation.

This direct contradiction remains even if the preceding transpose were numerically perfect. The earlier transpose can independently change block-float quantization, so the probe alone does not locate the first numerical error in the composite. That uncertainty does not make this gather implementation valid.

The producer/consumer ledger is otherwise count-matched for this case. For each of five tile rows, the [writer][gather-writer] reserves, reads, waits, and pushes 128 input tiles one at a time. The reader waits/pops exactly those 128 tiles, including bitmap-skipped tiles, reserves/pushes one output tile, and pops one index tile. The writer waits/writes/barriers/pops one output tile. The bitmap skips inner scans, not producer tile reads. There is no demonstrated off-by-one CB-count mismatch needed to explain the compressed result; the format/bounds violation is sufficient.

The smallest defensive library repair is to reject unsupported block-float input before transforms/dispatch. Actual support needs packed-format-aware selection and reconstruction or selected-tile decode to BF16. Replacing the width calculation with a fractional width is not enough. Neither the defensive rejection nor generic whole-table decode solves PLE capacity. Existing [gather unit tests][gather-tests] principally exercise ordinary BF16/FP32 values; they do not demonstrate this packed lookup.

Probe comparisons use `ttnn.to_torch` before calculating PCC, providing a host readback completion boundary. These are not observations based only on an enqueue returning. They still provide no trace-capture/replay evidence.

## 2. Capacity: reconcile logical weights, pages, and future allocations

Reading only safetensors headers gave:

| Representative layer | PLE table | MoE expert matrices | All remaining weights/buffers | Total checkpoint bytes |
|---|---:|---:|---:|---:|
| 0: linear attention | 0 | 5,033,164,800 | 154,793,408 | 5,187,958,208 |
| 1: linear attention + PLE | 102,400,491,520 | 5,033,164,800 | 220,473,048 | 107,654,129,368 |
| 3: full/indexed attention | 0 | 5,033,164,800 | 141,769,216 | 5,174,934,016 |

Layer 1 therefore has **5,253,637,848 non-table bytes**, not just the rounded 5 GB expert estimate. There are `R=128×2,500,012=320,001,536` stored rows of 160 values. The HF conversion mapping [concatenates shards along dimension 0][hf-conversion], so the 128 files/parameters are row partitions, not 128 independent n-gram heads.

With non-table weights held at checkpoint precision:

| PLE representation | Table bytes | Free minus table and non-table checkpoint bytes | Qualification |
|---|---:|---:|---|
| BF16 logical payload | 102,400,491,520 | −73,475,398,360 | Cannot be fully resident |
| Native BFP8 tiles | 54,400,696,320 | −25,475,603,160 | Already too large; conversion worsens peak |
| Native BFP4 tiles | 28,800,368,640 | 124,724,520 | Only about 119 MiB margin before runtime costs |
| Dense 4-bit codes, no metadata | 25,600,122,880 | 3,324,970,280 | Incomplete representation unless scales/codebook elsewhere |
| Dense 4-bit + BF16 scale/row | 26,240,125,952 | 2,684,967,208 | Payload bound; feasible grouped layout below |
| Dense 4-bit + FP32 scale/row | 26,880,129,024 | 2,044,964,136 | Payload bound; physical packing still required |
| Packed payload as 80-byte RM row pages | 40,960,196,608 | −12,035,103,448 | 128-byte physical stride on Blackhole, even before scales |
| Dense 5-bit + BF16 scale/row | 32,640,156,672 | −3,715,063,512 | Does not fit with BF16 non-table weights |

Native BFP tile calculations use per-shard row padding to 2,500,032, five tiles across width 160, and 1,088/576 bytes per tile. They include tile/exponent overhead but the table above excludes final bank-tail padding. BFP4 with bank-tail rounding is **28,800,516,096 bytes**, leaving **124,577,064 bytes**. These small distinctions matter near the capacity boundary. All checkpoint totals still need TT physical padding and any duplicated/transposed weight copies budgeted.

Blackhole [DRAM alignment is 64 bytes][bh-align]. An interleaved RM page is a row ([RM page construction][rm-pages], lines 113–147), and [allocator accounting rounds each page][physical-sizing], lines 207–228. Consequently `[R,20]` UINT32 is not a 25.6 GB resident table: `20×4=80` rounds to 128 per row. A separately allocated `[R,1]` scale tensor would also waste a page per scale. Shape-based byte counts that omit these rules are invalid.

One physically concrete compact layout is:

- Preserve 128 tensors, one per checkpoint row partition; avoid one enormous flat tensor and its possible 32-bit element/offset limits.
- Each RM UINT32 page contains **32 records**: 32×20 packed-code words followed by 16 words holding pairs of BF16 scale bit patterns. Width is `640+16=656` UINT32 words; page size **2,624 bytes**, already divisible by 64.
- Each shard has `ceil(2,500,012/32)=78,126` pages. Across eight banks the accounted allocation is `8×ceil(78,126/8)×2624` bytes per shard.
- Across 128 shards: **26,241,007,616 bytes**. Remaining free capacity after the 5,253,637,848 non-table bytes is **2,684,085,544 bytes**.
- A global or per-head shared 16-entry codebook is small. A distinct 16-entry BF16 codebook for every row adds 10,240,049,152 bytes and defeats this budget. Per-row scales with a shared codebook are the intended candidate.

The compact pages need a reader that fetches a selected page/record. Fetching one complete 2,624-byte page per selected row is a simple safe starting point; it avoids unaligned 82-byte record reads and stages only about 42 KB across sixteen independent lookups per token. Do not reinterpret an arbitrary record's byte offset as an aligned NoC address. Sub-page reads are a later optimization requiring the alignment/offset contract to be preserved.

Time-ordered resource ledger for a future implementation:

1. Start from the measured free capacity, remeasured with the chosen trace reservation and device setup. Do not subtract a trace reservation twice if the measurement already includes it.
2. Stream checkpoint shards through bounded **setup-time** host quantization and upload only their final packed representation. Retaining the entire BF16 device copy, a full-table transpose, or an unpacked table invalidates the budget.
3. Add all non-table weights in their actual TT layouts, shared codebook, shard descriptor/address tables, optional hash-product table, recurrent/conv/KV/indexer states, and persistent input/output buffers.
4. Add the peak simultaneously live prefill/decode intermediates, temporary weight conversion buffers, and retained trace allocations. Check per-bank free space and allocator placement, not only summed capacity.

The remaining 2.68 GB is a capacity hypothesis, not an allocation success. Batch size, context length, activation strategy, and trace reservation were not fixed by the probe. The checkpoint's maximum context 262,144 is not evidence that dense quadratic prefill intermediates fit; long-context attention needs its own bounded-memory plan.

## 3. Packed lookup alternatives and their intervention boundaries

### Custom UINT32 4-bit records with scales/codebook

The mathematical reconstruction is `row[j] ≈ scale[row] × codebook[nibble(row,j)]`. Eight 4-bit codes fit in one UINT32; 160 values need 20 words. Uniform signed quantization, a calibrated nonuniform codebook, or a small number of group scales are possible **weight representations**, not HF architecture changes, provided the final decoder passes the requested accuracy gate. Zero rows, scale rounding, clipping, and BF16 output rounding must be specified.

Use raw integer bit extraction for `(word >> (4*k)) & 15`. Converting an arbitrary 32-bit packed word to BF16/FP32 before extraction loses payload bits. BF16 scales stored in UINT32 pages must be decoded from their bit patterns, not numerically converted from the packed pair. A small table lookup for the 16 code values is easy after extraction; the hard part is obtaining the correct large-table row without expanding/scanning the table.

Ordinary RM gather cannot do that directly in this tree:

- A proposed `[2,500,012,20]` UINT32 shard with `[N,20]` UINT32 row IDs and `dim=0` is transposed by [gather preprocessing][gather-transform] to `[1,1,20,2,500,012]` / `[1,1,20,N]`.
- Codegen rejects RM. The native selector uses `RmSingleRowSingleCore` for decode-sized `N≤1920`; increasing `N` selects the multicore variant, which still mirrors the entire input row.
- [RM factory sizing][rm-gather-factory], lines 374–414, allocates an input stick of `2,500,012×4=10,000,048` bytes on each participating core. Blackhole has only [1,572,864 bytes total L1 per Tensix][bh-l1], before other reservations. Allocation/validation can fail even earlier in the transpose; the gather factory alone is already an independent contradiction.
- [The RM writer][rm-gather-writer], lines 37–60, loads the complete stick. This is not a random DRAM row lookup. Storing the transpose offline avoids one runtime transpose but leaves the same staging requirement.

**Minimal useful repair boundary:** add a direct row/page-selection factory that consumes device row IDs and reads only the selected records from interleaved DRAM, then unpacks the small result. It can be a new packed-embedding op, or a row-axis gather extension that bypasses the existing last-axis transpose/staging path. Simply using UINT32 fixes the packed scalar representation issue but does not fix lookup geometry, padding, or stateful hashing.

For the grouped layout, compute `shard = global_id // 2,500,012`, `local_row = global_id % 2,500,012`, `page = local_row // 32`, and `slot = local_row % 32` **on device**. Shard bases and accessor metadata can be fixed, uploaded setup data. The device reader chooses the shard and interleaved bank/page from the current ID; Python must not read IDs back to choose which table to dispatch or load. Check the final partially filled page explicitly.

Existing RM gather is reasonable for a **small already-selected buffer**, bounded codebook, or small diagnostic table. Fixed small chunks plus scanning/masking every chunk could avoid the L1 contradiction and remain device-only in principle, but introduce full-table work and a large trace schedule. No such implementation or measured performance is present. It is not evidence that direct large-shard RM gather works.

### Native BFP4 table with a selected-tile reader

A second credible route retains native BFP4 tile encoding, fetches the five tiles containing a chosen 160-wide row, decodes those tiles in small L1 buffers, and emits only the chosen row in BF16. This avoids both whole-table untilization and scalar-copy gather. It needs exponent-aware row/face mapping and target-compatible unpack/pack configuration; existing generic gather provides neither.

With BF16 MoE weights its capacity margin is only 124.6 MB, so it is a fragile starting point. Quantizing the two expert matrices to native BFP8 would reduce their payload from 5,033,164,800 to 2,673,868,800 bytes; with bank-rounded BFP4 PLE and other weights unchanged, total weight storage would be approximately **31,694,857,944 bytes**, leaving **2,483,873,064 bytes** before further runtime costs. This is another candidate, subject to combined decoder accuracy and physical allocation checks.

The observed 0.9933436 is BFP4 **storage** PCC over one sampled slice. It is neither proof of decoder PCC failure nor evidence of a pass. Projection, nonlinear gating, residual injection, expert routing, and recurrent accumulation can alter the effect of the error. A custom 4-bit codebook also has different error characteristics from block floating point. Neither may be rejected or accepted solely by that storage result. If additional scales/outlier corrections are needed, charge their metadata and runtime buffers to the same budget.

## 4. Exact I64 n-gram semantics without host fallback

[The HF hash implementation][hf-hash] and actual checkpoint buffers agree on these invariants:

- `ngram_size=3`, `heads_per_ngram=8`: eight 2-gram heads followed by eight 3-gram heads. Each head contributes 160 values, concatenated to 2,560.
- PLE belongs to zero-based decoder layer **1**: `ple_layer_ids=[2]` is checked against `layer_idx+1` ([HF decoder][hf-decoder], line 1262). Its PLE ordinal is 0; use that ordinal if rebuilding hash constants.
- Multipliers are **23,703,573,157,769; 20,109,073,645,365; 8,052,911,324,071**.
- Head vocabularies are **20,000,003; 20,000,023; 20,000,033; 20,000,047; 20,000,059; 20,000,063; 20,000,069; 20,000,077; 20,000,081; 20,000,093; 20,000,107; 20,000,147; 20,000,153; 20,000,159; 20,000,161; 20,000,171**.
- Offsets are prefix sums of those vocabularies. Their total is **320,001,446**, padded to **320,001,536** stored rows. The padding rows are not part of the hash vocabulary. Preserve numeric shard order when loading; HF's [natural-key sorting][hf-natural-sort] handles trailing numeric components such as `shard_10`, rather than placing them lexically before `shard_2`.
- For head `h`, form `mixed = XOR_j(token[t-j] × multiplier[j])` over 2 or 3 positions, then `global_id = mixed % prime[h] + offset[h]`. Do not reduce products modulo the prime before XOR; modular reduction does not commute with XOR.

Tokens are valid IDs in `[0,248319]`; EOS is 248044. The largest possible product with the actual multipliers is **5,886,047,582,964,040,311**, below `2^63−1`. All products and their XORs therefore have sign bit zero. For this exact valid-token domain, unsigned 64-bit remainder matches the HF signed-I64 `torch.remainder`. This is a proof for the pinned checkpoint/domain, not permission to change signed-remainder semantics for arbitrary future constants or invalid negative tokens.

Concrete falsification of a 32-bit shortcut: current/previous tokens `(1,2)` produce mixed value **53,991,755,950,563**. Modulo head-0 prime 20,000,003 gives **7,851,802**. Truncating that mixed value to 32 bits first gives remainder **17,039,243**. Even tiny token IDs need the high product bits; floating-point multiplication/remainder is also unsuitable for exact indexing at this magnitude.

TTNN device tensor validation [does not admit I64/U64][tensor-spec]. Feasible alternatives are:

1. A custom data-movement kernel using integer 64-bit operations or explicit high/low UINT32 limbs, with compiler support and runtime behavior verified later. Native tensor-dtype support is not required to hold two words.
2. An exact setup-time product table for all `248320×3` token/multiplier pairs, stored as high/low UINT32 words: **5,959,680 payload bytes**. This is static weight-derived data; device lookup plus XOR/remainder stays device-only. It avoids relying on a wide tensor multiply, but still requires a suitable small-row reader and proper physical packing.

A simple exact remainder algorithm shifts the two limbs through a 32-bit remainder, one bit at a time. With `r<p`, each step forms `2*r+bit < 2*p < 2^26`, subtracting `p` if necessary. It never needs floating point or a wide division. Host-only arithmetic verification in this audit checked **16,432 head remainders**: three explicit histories plus 1,024 histories generated with Python `Random(1234)`, two n-gram lengths, eight applicable head primes each. Limb XOR and bit-serial remainder matched Python integer multiplication/XOR/modulo in every comparison. This checks the proposed arithmetic identity, not a kernel or the full history/mask logic.

History is equally part of the hash:

- HF keeps two previous token IDs in cache `state_idx=2`; missing initial history is EOS-filled, not zero-filled. Clone/read the prior context before updating it.
- `_shift_right_ignore_eos` resets lookback after a **previous** EOS. An EOS at the current position does not retroactively erase that position's preceding history. Test first token, consecutive EOS, and chunk boundaries.
- At model entry, `conv_mask` replaces masked IDs with EOS before passing them to PLE ([HF model boundary][hf-input-boundary], lines 1465–1483).
- Keep the token-history cache distinct from DeltaNet convolution state (`state_idx=0`) and PLE dilated-convolution state (`state_idx=1`).

Global row IDs fit UINT32 after exact remainder and offset. The 128 physical row partitions are independent of the 16 semantic head ranges; do not identify a head with a shard.

## 5. Complete decoder boundary and all main layer kinds

The requested acceptance matrix must cover at least these representative main text layers, with real weights and prefill plus stateful decode:

| Layer | Required behavior | Important trap |
|---|---|---|
| 0 | Hyper-connection input mixing → GatedDeltaNet → injection; hyper-connection mixing → routed/shared MoE → injection | Standard Qwen residual/RMSNorm code is not equivalent |
| 1 | Full PLE injection first, then the same linear-attention/MoE structure | Supplying ordinary token embeddings or omitting PLE does not exercise this kind |
| 3 | Hyper-connections, QSA-indexed full attention with gated output, routed/shared MoE | Short attention tests can miss QSA pruning entirely |

Relevant HF contract ([decoder][hf-decoder], [hyper-connections][hf-hc], [linear attention][hf-linear], [full attention][hf-attn], [MoE][hf-moe]):

- The residual width is **4×2560=10240**. Each hyper-connection normalizes in 2,560-wide groups, projects through rank 320, applies SiLU after division by 4, then sigmoid mixing and a mean over four streams. Injection is `2*sigmoid(projected_normed_input/4)` times a shared 2,560-wide block output, added back to the original four streams.
- The ordinary Qwen4Exp RMSNorm uses **`1 + weight`**, whereas the gated DeltaNet RMSNorm uses its weight directly. Conflating them changes real-weight computation ([norm implementations][hf-norm]).
- DeltaNet uses 16 key heads and 48 value heads, each dimension 128; Q/K repeat three times, with Q/K L2 normalization. Its recurrent reference accumulates in FP32, decays state, subtracts the state prediction from V, applies beta, updates state, then projects with Q. The configured output gate is **sigmoid**. Prefill/chunk and single-token recurrent paths must agree at their state boundary.
- The MoE routes top 10 of 512 experts using FP32 softmax and normalized selected probabilities (`norm_topk_prob=True` default), retains ID/weight pairing, and adds a sigmoid-gated shared expert. A host loop driven by downloaded router IDs would violate the trace contract. Unsorted top-k is not intrinsically wrong if IDs/weights stay paired and the contributions are summed.
- Full attention uses 24 query heads, 2 KV heads, dimension 256, Q/K norms, RoPE, an output sigmoid gate, and a separate indexer. The indexer compresses visible tokens in groups of four and selects up to 512 groups, retaining the incomplete tail. **At 2,052 visible tokens** there are 513 complete groups and actual pruning starts; up to 2,051 visible tokens all complete groups plus tail can be retained. A short-sequence dense-attention substitute can pass while missing the long-context semantics ([QSA selection][hf-qsa]).
- PLE projects 2,560 gathered features into a 10,240-wide key and 2,560-wide value, group-normalizes key/query, computes the signed square-root-transformed dot-product gate, broadcasts gated values to four streams, then adds a normalized depthwise convolution with kernel 4 and dilation 3. Its convolution needs nine prior positions. The HF cache returns prior plus current values before retaining the tail ([PLE][hf-ple], [cache update][hf-cache]); caching nine positions is not itself an off-by-one bug.

No TT implementation was available to accuse of violating these rules. They are necessary module contracts and test coverage gaps, not claimed observed TT decoder bugs. No acceptance is claimed for vision or MTP execution.

### Precomputed embeddings: legitimate and changed contracts

An ordinary functional decoder legitimately receives externally prepared hidden states, RoPE tensors, masks, and cache inputs at its declared boundary. Preparing a host HF reference or lookup oracle, or transforming static checkpoint weights before device upload, also does not constitute a runtime host fallback.

However, the installed [HF layer][hf-decoder], lines 1277–1280, calls PLE internally with `ple_input_ids`; [PLE forward][hf-ple], line 1236, calls the n-gram embedding. At the model boundary, `inputs_embeds` replaces the main unigram embedding, while `ple_input_ids` still supplies or reconstructs token identities ([HF input handling][hf-input-handling], lines 1394–1419). There is no ordinary `inputs_embeds` promise to supply the concatenated PLE n-gram features.

Useful and honest partial work includes loading exact BF16 rows from the correct safetensor partitions on the host, feeding the resulting `[B,S,2560]` PLE features to an explicitly factored **post-lookup PLE/decoder** comparison, and validating projection, gating, convolution, attention, and MoE independently. Such a report must label lookup/hash and their token-history state as excluded. Feeding the **whole PLE output** instead excludes still more computation and is weaker evidence.

To meet the current user contract, an upstream producer of those features must itself be device-resident and included in the demonstrated device-only inference chain. Precomputing the exact rows for a known prompt/decode sequence, narrowing the table to observed test IDs, CPU-dispatching missing rows, or preloading future outputs proves a restricted fixture only. Newly generated IDs/history must be usable without those host interventions. Moving the lookup outside the timed region does not turn it into static parameter preparation.

Accepting a host-precomputed PLE feature tensor as the permanent decoder input is possible only under an explicitly revised task boundary. This audit does not make that revision.

## 6. Minimal continuation and decisive acceptance checks

No implementation is included. A repair owner can make the next decision concrete in this order:

1. **Guard the known unsupported operator route.** Reject direct BFP4/BFP8 scalar gather, or implement correct packed handling. Do not treat a successful API return as support. This fixes a library hazard independently of decoder feasibility.
2. **Evaluate a bounded packed representation on real weights.** Compare row-scale/codebook and selected-tile BFP4 candidates using deterministic samples spanning heads, shards, actual hash-distributed IDs, and outliers. Measure finite outputs, reconstruction error, PLE intermediate errors, and full decoder PCC; do not substitute storage PCC for the ≥0.995 decoder gate. Calibration must not memorize evaluation rows.
3. **Implement the exact device hash and direct packed row/page read.** Preserve the full table, all head offsets, EOS/mask semantics, numeric shard order, integer payload bits, and physical page layout. Keep intermediate buffers proportional to selected rows/pages. Compile and verify the custom device path before making hardware claims.
4. **Implement and validate each missing decoder kind.** Compare real-weight layer outputs for prefill, first decode, and multi-step decode against the pinned HF reference. Include independent cache state and prompt/chunk boundaries. For QSA, include cases around the first real pruning boundary and the configured context requirements.
5. **Demonstrate trace correctness with changing inputs and state.** Load weights and allocate stable state/buffer addresses before capture; capture hash, lookup, dequantization, PLE, attention/MoE, and state updates as applicable. Replay with different token histories, selected shards, masks, and cache positions. Compare after explicit completion/readback, and verify no `.to_torch`, host arithmetic, per-token disk/PCIe weight service, or Python branching on downloaded IDs is needed inside the claimed inference path. Host-provided decoder test inputs are distinct from host-computed model intermediates.
6. **Measure actual peak physical capacity at the same configuration.** Account for retained traces, all output/workspace allocations, per-bank rounding, and the preparation-time peak. Do not reuse the zero-trace probe's free-space number as admission after other allocations.

If a tested compact representation cannot satisfy decoder PCC while retaining enough physical capacity for the required shapes and state, that combined accuracy/capacity result would be a concrete blocker. No such result exists in the supplied evidence. Conversely, arithmetic feasibility and nominal free space alone are insufficient to mark the stage complete.

## Review of headline claims and other potential issues

After drafting, the headline chains were checked again against dispatch, factories, formats, and reference boundaries:

| Candidate | Direct contradiction / supporting contrast | Smallest intervention | Verdict |
|---|---|---|---|
| Native BFP scalar gather | Codegen rejects BFP; native accepts it; 1088/1024→1 and 576/1024→0 conflict with packed exponent/mantissa layout; BF16 control passes | Reject unsupported input or implement packed-aware selected reads | Confirmed source defect; consistent with NaN PCC, exact runtime first-bad tensor unrecorded |
| Embedding does not perform compressed lookup | Wrapper converts the full table; primitive requires RM BF16; BFP8 conversion works, BFP4 RM spec fails | Direct compressed reader, not only dtype admission | Confirmed current limitation and explanation of embedding contrast |
| Naïve UINT32 RM row gather solves storage | 80-byte pages align to 128; lowered column stick is 10,000,048 bytes vs 1,572,864 total L1 | Group records into pages and add direct row/page access | Refuted for current direct large-shard gather proposal; no such decoder implementation currently exists |
| All 4-bit methods fail accuracy | Only one BFP4 storage PCC was measured | Full decoder comparisons for each candidate | Unproved; not a blocker claim |
| No I64 dtype makes exact hashing impossible | Actual products fit signed 64 bits; limbs/remainder retain exact information | Device integer/limb kernel, optionally static product table | Refuted as an impossibility argument; implementation remains missing |
| Precomputed PLE can pass the unchanged boundary | HF invokes lookup inside the layer; main `inputs_embeds` leaves PLE active | Include device producer or explicitly revise boundary | Not a valid unchanged-contract acceptance shortcut |

Other issues and evidence gaps are deliberately not promoted to root causes:

- The probe does not save output finiteness, variance, selected lowered program identifiers, or tensors. Those omissions limit symptom attribution but do not invalidate the direct gather format violation.
- The BFP4 exception message names BFLOAT8_B because of a shared validator string. It is a diagnostic discrepancy, not evidence of an unintended BFP8 cast.
- Earlier transpose behavior for BFP could introduce additional error. It is not necessary to make native scalar gather invalid; no unsupported claim that the transpose is correct is made. Padding fill is a no-op for the aligned probe and is not a competing explanation for that case.
- The native gather producer still reads all input tiles even when the reader bitmap skips processing. This is a performance limitation, not the cause of the dtype contrast.
- Probe exception paths can leave temporary objects to Python/device cleanup. No observed leak is established, and this is not the explanation for the initial reported free capacity or the packed-data corruption.
- The HF loader's `force_cpu=True` n-gram concatenation and `_no_placement_params` facilitate the reference's large table handling. They do not grant a CPU fallback exception to the requested TT inference contract.
- An unsorted-but-paired MoE top-k or the nine-position PLE conv cache is not intrinsically incorrect; the reduction and returned prior-plus-current state explain those apparent differences.

## Source links

All repository paths below resolve inside `/home/ttuser/qwen38-flash/tt-metal`. HF references are the installed local source, not an assumed upstream revision.

[probe-json]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/doc/functional_decoder/ple_storage_probe.json
[probe]: /home/ttuser/qwen38-flash/tt-metal/models/autoports/qwen_qwen3_8_flash_next/tests/probe_ple_storage.py:21
[embedding-wrapper]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/embedding/embedding.cpp:30
[embedding-validation]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/embedding/device/embedding_device_operation.cpp:34
[embedding-reader]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp:65
[untilize-dtype]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/common/common.hpp:294
[untilize-spec]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize/device/untilize_device_operation.cpp:314
[tensor-spec]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/tensor/spec/tensor_spec.cpp:114
[gather-routing]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp:536
[gather-transform]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp:117
[gather-codegen]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/codegen/gather_codegen_supported.cpp:62
[gather-validation]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp:17
[gather-factory]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp:201
[gather-reader]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_single_row_multi_core.cpp:80
[gather-writer]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_single_row_multi_core.cpp:56
[gather-helpers]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_common.hpp:17
[gather-tests]: /home/ttuser/qwen38-flash/tt-metal/tests/ttnn/unit_tests/operations/data_movement/test_gather.py:45
[transpose-wrapper]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/transpose/transpose.cpp:295
[fill-padding]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/fill_pad/fill_pad.cpp:21
[bfp-packing]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/data_format/blockfloat_common.cpp:482
[tile-sizes]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/api/tt-metalium/tt_backend_api_types.hpp:122
[bh-align]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374
[bh-l1]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:33
[rm-pages]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/tensor/spec/layout/page_config.cpp:113
[physical-sizing]: /home/ttuser/qwen38-flash/tt-metal/tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:207
[rm-gather-factory]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_program_factory.cpp:361
[rm-gather-writer]: /home/ttuser/qwen38-flash/tt-metal/ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp:37
[hf-conversion]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/conversion_mapping.py:2131
[hf-natural-sort]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/core_model_loading.py:1319
[hf-hash]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1026
[hf-ple]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1177
[hf-decoder]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1252
[hf-hc]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:995
[hf-linear]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:456
[hf-attn]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:811
[hf-moe]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:913
[hf-norm]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:143
[hf-qsa]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:715
[hf-cache]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/cache_utils.py:1192
[hf-input-boundary]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1465
[hf-input-handling]: /home/ttuser/qwen38-flash/tt-metal/python_env/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1394
