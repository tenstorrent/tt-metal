# AutoDebug: TP4 B32 long-prefix decoder accuracy regression

## Finding

**The leading diagnosis is long-context numerical error in the packed B32 SDPA chunk policy. K128 alone at HiFi2 restores the originally failing query, while HiFi4 with K64 still fails early. A joint HiFi4/K128 control passes the first eight steps; these controls retain production prefill chunk sizes and BF8 collectives.** The default K64 policy's long chain of BF16 online-softmax updates is a concrete precision exposure. Its exact responsibility remains unproved, and K128 alone still fails later at step 1013. No concrete page-address, capacity, request-order, state-restore, or reference-input bug was found that explains the reported failure.

The decisive next experiment is to replay **identical captured Q, KV cache, page table and positions** through separately varied SDPA fidelity, K chunk size and batch partitioning, while measuring the layer-0 result before layer 3. A blanket tolerance relaxation is not supported by this investigation.

Inspection only: no device execution, reset, launcher invocation, or source edits were performed by this investigation. Only this report was written. Three isolated xhigh inspections covered the harness, attention lowering, and prefill/recurrent state. Existing experiment artifacts were read, including four separately produced controls that completed during inspection. Additional unfinished jobs are not treated as results.

## Evidence and revision scope

Model-local paths below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`; paths beginning `ttnn/` are relative to the repository root.

- Checkout HEAD: `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`, with substantial pre-existing uncommitted work. HEAD alone does not identify the tested implementation.
- Production `tt/multichip_decoder.py` SHA256 remains `80ad61a028434db6d7ebaa46f0e051ef375b09f3246aca0126b50fbdcfb523e9`, matching `doc/optimized_multichip_decoder/final_v1_context128k_answer1024_32_command.json`.
- Frozen single-chip implementation `tt/optimized_decoder.py`: `b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e`, matching the reference result's `implementation_files`.
- Original failing harness: `0be25c4aadeeff3b9592a8b01f1eca4a08e89c826100599ced4fb97735343b7f`. Line references to the original harness refer to the immutable `doc/optimized_multichip_decoder/source_snapshots/multichip_packed_stack_0be25c4aadee.py.txt`. The live harness subsequently changed to `7226b522cd38…`, adding PCC-failure JSON/tensor capture. That later capture must not be attributed to the original run.
- The first two controls execute archived probe `source_snapshots/multichip_context_accuracy_probe_c2af4ea9ff3c.py.txt`; the later HiFi2/K128 and HiFi4/K64 controls execute `multichip_context_accuracy_probe_3b7b4a169834.py.txt`. These are under `doc/optimized_multichip_decoder/`. Control selections come from the corresponding `_command.json` sidecars; the result's stripped command omits `--accuracy-control`.

### Direct observations

| Evidence | What it establishes | Limit |
|---|---|---|
| `final_v1_context128k_answer1024_32.log:21–31` | Original run fails `decode_6_11`, PCC `0.9914271235466003` | No original actual tensor or preceding PCC history retained |
| `context128k_answer1024_reference_32.json:25–27,41–88` | Single-chip reference generation completed B32, capacity 132128, 1024 replays | `--baseline` saves outputs; it performs no independent HF comparison |
| `final_v1_answer1024_32_setup_clean.json` | Same production SHA passes prefix 4095, 1024 steps; minimum recorded PCC `0.9960771799` | Different capacity, page permutations and random activations |
| `final_v1_extended_context64k_32_setup_clean.json` | Same SHA passes prefix 65535, 128 steps; minimum PCC `0.9966393113` | Different random workload; not the same query truncated to 64K |
| `final_v1_max_context_{0,3}.json` | Separate single-layer B1 tests pass prefill through 262144 and baseline-checked terminal decode at 262143; terminal PCCs `0.99982953`, `0.99989152` | Private-state B1 path, not packed B32 stacked decode |
| `final_v1_recorded_32_3_long.json` | Recorded-HF B32 layer-3 gate passes 128 steps, minimum PCC `0.9997988939` | Its `cases` have prefixes 4095/4096/4097, not 128K; layers are checked independently |
| `context128k_accuracy_chunk1024.log` and `.json.pcc_failure.json` | Later same-source 128K control with both prefill chunks set to 1024 fails `decode_1_16`, PCC `0.9948351383` | Stops before original failing query |
| `context128k_accuracy_sdpa_hifi4_k128.json` | Later same-source 128K control passes all 256 decode checks through step 7; `decode_6_11` PCC `0.9991433620` | Changes fidelity and chunk size together, including prefill decode-tail calls; does not establish the full 1024-step gate |
| `context128k_accuracy_sdpa_hifi2_k128.json.pcc_failure.json` | K128 at unchanged HiFi2 passes original `decode_6_11` at `0.9991970062`, but fails `decode_1013_12` at `0.9946861863` | Not a complete fix; terminal state comparisons are still unreached |
| `context128k_accuracy_sdpa_hifi4_k64.json.pcc_failure.json` | HiFi4 with unchanged K64 fails `decode_1_16`, PCC `0.9945239425` | Fidelity alone does not clear even the early window |

Reaching the original failure means 128 sampled/full prefill checks and 203 earlier per-row decode checks passed. The packed rank-agreement check also passed through replay 6, but it correlates the whole packed tensor and is not an individual row-11 rank check. Terminal conv/recurrent/KV comparisons had **not** run: original harness lines 253–272 only warm inspection programs; actual state comparisons occur after all decode steps, lines 307–345.

## Exact failing workload

The harness uses a two-layer stack, layer 0 linear attention followed by layer 3 full attention. Decode hidden inputs are fresh random, teacher-forced activations, not previous stack outputs. Only cache/recurrent state feeds back between steps.

From original harness lines 198–225 and 290–304:

| Quantity | Row 11 |
|---|---:|
| Initial logical prefix | `131071 + 11 % 3 = 131073` |
| Continuation | `17 + 11 % 2 = 18` tokens |
| Decode start | `131091` |
| Failing step / position | `6` / `131097` |
| Valid attention length | `131098` |
| Logical cache page / token offset | `4096` / `25` |
| K64 chunk index / start | `2048` / `131072` |
| Valid / masked entries in final K64 chunk | `26` / `38` |
| Capacity / pages per request | `132128` / `4129` |

**Step 6 crosses neither a page nor a K64 boundary.** Row 11 step 0 already uses page 4096, offset 19, and chunk 2048. Thus the label alone is not evidence of a newly triggered 128K addressing boundary.

The initial full-attention prefill pads row 11 through position 131103. Continuation overwrites positions 131073–131090; decode 0–6 overwrite 131091–131097. Future padded entries 131098–131103 and all of logical page 4097 must remain masked. This makes a future-cache poisoning probe useful, but no incorrect mask was found in source.

`allocate_batch_state` gives row 11 pages `[45419,49548)` (`11 * 4129` offset); page-table entries are shuffled local indices, then offset on device (`tt/multichip_decoder.py:506–528,1172–1192`). A cheap CPU replay of the seed-119 permutations maps logical page 4096 to local page 1099, pooled page 46518. Capacity and table padding are sufficient: native decode pads 4129 table entries to 4132, while this final K64 chunk reads logical pages 4096 and 4097. Maximum planned position is 132114, below capacity 132128. The lower reader uses 32-bit page indices; there is no demonstrated 16-bit wrap.

## Why SDPA is the first numerical suspect

### Effective schedules differ much more than the policy names suggest

Production packed attention calls `paged_scaled_dot_product_attention_decode` with BF16 Q, BF8 KV, scale `1/16`, grid `(11,10)`, K64, and HiFi2/`fp32_dest_acc_en=True` (`tt/multichip_decoder.py:183–185,1122–1148,1245–1255`). The frozen single-chip reference executes each request separately with four KV heads, grid `(8,8)`, K256, and HiFi4 (`tt/optimized_decoder.py:449–459`).

The factory's core allocation (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:202–216`), default 16-core cap (`ttnn/cpp/ttnn/operations/transformer/sdpa_config.hpp:21`), and workload partition (`…/device/kernels/rt_args_common.hpp:60–91`) instantiate as follows:

| Path | Batch per SDPA call | KV heads/chip | Cores per KV head/request | K chunk | Global chunks | Sequential chunks per worker |
|---|---:|---:|---:|---:|---:|---:|
| Final TP4, position 131097 | 32 | 1 | 3 | 64 | 2049 | 683 |
| Frozen single-chip, same position | 1 | 4 | 16 | 256 | 513 | 32–33 |
| Final TP4 near passing 64K start | 32 | 1 | 3 | 64 | 1025 | 341–342 |
| Final TP4 B1 terminal position 262143 | 1 | 1 | 16 | 64 | 4096 | 256 |

The B32 allocation uses 96 of 110 cores, leaving 14 idle. At the failing position the three local ranges are `[1366,2049)`, `[683,1366)`, and `[0,683)`. The global chunk count exceeds 2048, but **no worker executes a 2048-iteration loop**; its count is 683. Reader, compute and mask calculations agree on these bounds by inspection.

The local CB ledger is consistent even for the odd count of 683: after each chunk, `c28` holds one max half tile, `c30` one sum half tile, and `c26` eight output half tiles; the score, correction and current-output temporaries are empty. The first chunk initializes `c26`; the next 682 update it. Buffer IDs do not swap despite the ping-pong comment. The three-core reduction is `core2 -> core1 -> core0`; each round stages 10240 bytes of sum/max/output and the two-round 20480-byte staging allocation covers the root's final block exactly. No parity-dependent final-buffer or transfer-size mismatch was found (`sdpa_flash_decode.cpp:529–574,607–646`; `writer_decode_all.cpp:305` and the host tree helper).

### FP32 destination arithmetic still spills the running state to BF16

The SDPA factory hardcodes `im_df` and `stats_df` to `Float16_b`, independently of `fp32_dest_acc_en` (`sdpa_decode_program_factory.cpp:507–508`). Circular buffers for QK, intermediate output, running output, maxima, sums and correction factors use those formats (`:694–712`).

In `…/device/kernels/compute/sdpa_flash_decode.cpp:543–574`, each additional chunk:

1. Computes and packs the max correction factor.
2. Rescales and packs the previous sum and output accumulator.
3. Adds and packs the new sum and output contribution.
4. Copies current max/sum into BF16 previous-state buffers.

FP32 destination registers improve each calculation; they do not preserve FP32 values across these pack/unpack boundaries. The failing B32 schedule therefore has about **21 times** the sequential chunk-update depth of the single-chip reference, twice the 64K B32 depth, and greater depth than the passing B1 maximum-context decode.

This is a concrete mechanism that predicts context-length and batch sensitivity. It can produce query/head-dependent errors, so an early isolated bad query after several passing ones is compatible with it. It does **not** prove that these operations caused this particular stack PCC failure: the original artifact has no per-layer outputs or captured SDPA operands. HiFi2 QK/PV arithmetic, preceding cache differences, masking, and later projection/MLP amplification still need separation.

The smallest first intervention is a test-only change to the SDPA policy at fixed operands: change K64 to K128/K256 at fixed HiFi2, or HiFi2 to HiFi4 at fixed K64. The external K128-only result already supports chunk-policy sensitivity at the original query, but not full correctness. A true FP32-running-state kernel variant would be a deeper intervention, with CB sizing and all producer/consumer formats updated together.

## Competing explanations

| Candidate | Direct code evidence and prerequisites | Match to observations | Falsifier / smallest useful intervention |
|---|---|---|---|
| B32 SDPA accumulation depth / arithmetic | Proven BF16 intermediates and 683 updates/worker; must show attention is first inaccurate stage | Best match: K128 at unchanged HiFi2 restores the failing query, HiFi4/K64 still fails early, and joint HiFi4/K128 passes the early window | Identical-QKV replay with independently varied K chunk, fidelity and batch waves; inspect the late K128 failure too |
| Long-prefix prefill state or KV drift | Production prefill chunks are linear 32768/full 16384 versus reference 1024; production also quantizes prefill norm activations and large reductions to BF8 | Alone is weaker: chunk1024 still fails, while SDPA-policy change passes with original large chunks; state interactions remain possible | Compare states and layer-0 decode output before full attention; restore identical states across decode-policy controls |
| BF8 decode all-reduce / BF16 accumulation | Four rank partials are quantized to BF8; default AR destination accumulator is not FP32 | Joint SDPA control retains BF8 AR and passes; AR may amplify changed operands but a standalone AR fault is less explanatory | Replay identical rank partials through correctly allocated BF16-payload and FP32-accumulator controls |
| Reference/harness mismatch | Different archived harness hashes, no reference checksum or saved input hashes | Hash difference is explained by deletion of unused zeros; input draw schedules and commands match | Hash reference and first eight input streams; regenerate only if these disagree |
| Masking / paging | Final chunk contains future padded data; table offsets and mask bounds must be correct | Worth testing, but neither step 6 nor row 11 introduces a page/chunk transition; source calculations are coherent | Poison only future cache entries and replay identical Q/valid KV; output must be invariant |

### Prefill and recurrent-state qualifications

`tt/multichip_decoder.py:141–145` selects large chunks only for replicated, non-fused prefill. `:665–683` quantizes normalized prefill activations, and `:421–436` uses reduced-precision communication for segments of at least 512 rows. These differ from the single-chip path and can alter layer-0 recurrence and the layer-3 prefix KV cache. Chunking is mathematically equivalent only in exact arithmetic; it is not proof of a logic bug.

Linear history/state handling is coherent by inspection: pooled requests are read/written independently (`:531–554,765–774`); history retains the last three **logical** raw projections (`:803–815`); inactive recurrent tail rows use `g=0,beta=0` (`:817–820`). The 18-token continuation replaces all three direct prefix convolution-history rows. A prefix-history defect could survive to decode 6 indirectly through recurrent state, but not as an unreplaced raw history row. The pooled decode kernel has no prefix-length index.

Both implementations still use **internal GDN chunks of 32**, irrespective of outer prefill chunk size (`tt/multichip_decoder.py:868`, `tt/optimized_decoder.py:390`). Row 11 has 4097 internal prefix chunks in either path. Copying FP32 state between outer chunks does not itself introduce BF16 rounding. Thus a claim that the 32768 default directly makes the recurrent scan's reduction chunks 32 times larger would be false.

An additional numerical difference exists in the packed gate weights: TP stores each rank's eight `b/a` columns followed by padding (`:243–245`), whereas the reference stores all 32 heads contiguously (`tt/optimized_decoder.py:84–95`). BFP8 shares exponents over groups of 16 values (`tt_metal/impl/data_format/blockfloat_common.cpp:437–458`), so splitting these groups can change gate-weight quantization. This can seed recurrent differences but has not been tied to the failing row or length. It is not a demonstrated long-index defect.

### BF8 all-reduce lowering and completion

`decode_reduce` explicitly casts each partial to BF8, uses matching BF8 persistent receive buffers, and requests BF16 output (`tt/multichip_decoder.py:82–98,438–454`). The selected minimal overload passes that input/buffer pair directly to the primitive (`ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:541–572`). Its factory derives input/receive format from the input and output format separately (`…/device/all_reduce_async_program_factory.cpp:275–276,401–432`). Thus BF8 input plus BF16 output is intentional and is not itself a format contradiction.

For B32 there are four tiles per output shard, 16 received tiles per core over four ranks, and four output tiles. With two links, each handles 16 output cores / 64 tiles. The receive kernel waits for rank arrivals before publishing the 16-tile CB; compute waits for all 16, performs the two rank-pair additions, and publishes four output tiles (`…/kernels/dataflow/reduction_receiver.cpp:31–36`, `…/kernels/compute/reduction.cpp:20–67`). No prefix-dependent count or partial final reduction block appears here. The default binding has `fp32_dest_acc=False`; requesting BF16 output does not change that (`all_reduce_async_nanobind.cpp:129–136`, factory `:469–476`).

Each layer issues two reductions and alternates its two workspace slots. Shared workspace reuse is serial on the same queue. In the harness, unique state backing tensors are cloned, two warmups and capture each precede restore, positions and input are refreshed before replay, and replay is blocking on CQ0 (original harness `:228–233,274–304`). There is no source-level completion/lifetime mismatch established by these artifacts. The numerical assertion occurs after replay and download, not merely after host enqueue.

## Additional completed controls, separate from the original failure

During inspection, `context128k_accuracy_chunk1024.log` and `.json.pcc_failure.json` completed from an external run. They use unchanged production SHA `80ad61a…`, the later diagnostic harness `7226b522cd38…`, fixed capacity 132128, eight decode steps, and test-only `chunk_size=1024` for both layers (`tests/multichip_context_accuracy_probe.py:16–24`). This investigation did not launch that run.

The control fails at `decode_1_16`, PCC `0.9948351383`, after 128 prefill checks and 48 decode checks pass. Minimum sampled prefill PCC is `0.9987183213`. Cheap CPU analysis of its captured actual/expected row confirms PCC, RMSE `0.0585646`, actual/expected standard deviations `0.498137/0.466001`; this is a numerical mismatch across the vector, not an all-NaN result.

Fixed context keeps capacity, page permutations, prefill inputs and the first eight decode inputs identical to the 1024-step fixture. This result **weakens “large prefill chunks alone cause the regression”**. It does not prove prefill state is correct, does not establish the original row-11 error is unchanged, and does not validate the large-chunk path. Chunk size changes other numerical/kernel choices, and the control fails before reaching original step 6.

The second external control, `context128k_accuracy_sdpa_hifi4_k128.json`, completed successfully at 06:02:48 UTC. It retains production prefill chunk sizes, weights, pooled kernels and BF8 AR while using HiFi4/K128 SDPA. All 128 prefill checks and 256 decode checks pass. Minimum decode PCC is `0.9990581274` (`decode_3_4`), and original `decode_6_11` reaches `0.9991433620`. K128 at position 131097 requires 1025 chunks, or 341–342 per worker, instead of 683; it also changes matrix blocking and the final masked chunk geometry (26 valid / 102 masked entries).

The third external control, `context128k_accuracy_sdpa_hifi2_k128.json.pcc_failure.json`, requests the full 1024 steps and retains HiFi2. It passes 32428 decode checks before failing `decode_1013_12`, PCC `0.9946861863`; original `decode_6_11` passes at `0.9991970062`. This expanded wrapper adds optional page-table padding, but that code is inert for K128 here: both the single-request 4136-entry and packed 4132-entry attention tables are already divisible by four. The actual intervention for this control is the K chunk, not a switch to HiFi4. The late failure is at row 12 position 132101, page 4128 offset 5, in a K128 chunk with six valid and 122 masked entries. Its mechanism has not been localized and need not be identical to the original failure.

The fourth external control, `context128k_accuracy_sdpa_hifi4_k64.json.pcc_failure.json`, changes fidelity while keeping K64. It fails at `decode_1_16`, PCC `0.9945239425`, after 48 decode checks pass. The expanded wrapper's padding is also inert at K64. Taken with the K128-only result, this weakens a HiFi2-only explanation and strengthens a chunk-geometry/online-accumulation explanation. The changed first-failure label does not prove that every K64 discrepancy has the same low-level cause.

Together these are the strongest available contrasts and rank the SDPA chunk policy above large-prefill-chunk-only or BF8-AR-only explanations. The controls are **not strictly decode-only**: the wrapper installs the policy before prefill, and unaligned full-attention continuation executes `decode_forward` (`tt/multichip_decoder.py:1001–1012`). Source dataflow limits that confound: layer 3 is last, its KV update precedes SDPA, and its outputs do not feed future hidden inputs or persistent state. Thus the changed continuation SDPA results should not seed different state in this stack. Actual Q/KV equality across separately executed runs remains uncaptured; trace/allocation effects are not independently excluded. Neither control establishes the full 1024-step fix.

## Discriminating probes, in priority order

These are follow-ups requiring authorized device experiments, not actions performed here. Preserve the frozen reference and production source.

1. **Localize the first divergent stage with the existing eight-step workload.** Keep `--batch 32 --prefix-length 131071 --context 132128 --layout replicated`, seed/draw order, and all 32 requests; capture per-row/per-rank output after layer 0 and after layer 3 through step 7. Compare both full output and residual contribution, plus RMS/max error. Capture initial and post-continuation conv/recurrent state per head, and sample layer-3 KV across the prefix. Do not compare eight-step terminal states with the 1024-step terminal-state fixture. Generate a separate short-horizon state fixture if needed.
2. **Replay SDPA on identical operands.** Freeze the failing Q/K/V/table/position after cache update. Compare a CPU FP32 attention calculation on the same quantized operands with HiFi2/K64, HiFi4/K64, HiFi2/K128 or K256, and HiFi4/K256. Separately replay smaller batch waves with more cores per head, preserving request mappings. If longer chunks/more cores resolve the attention error while fidelity alone does not, that supports the BF16 update-chain diagnosis. Keep operand dtype, layout and scale consistent and verify effective kernel configs.
3. **Separate prefill state from decode arithmetic.** Restore the same post-continuation snapshot for each decode policy. Compare each pooled recurrent row with the existing private-state `chunk_gdn_fused` decode path on identical raw projections/state. Isolate layer-0 and layer-3 prefill chunk changes separately. If layer 0 is already inaccurate, investigate gate quantization, recurrence and state initialization before attributing the stack output error to SDPA.
4. **Test BF8 AR using identical partials.** Compare host FP32 sum, BF8-payload/BF16-output AR, the same BF8 payload with FP32 destination accumulation, and BF16 payload with matching BF16 receive scratch. Measure before AR, after AR and after residual/MLP. Changing payload and accumulator together does not separate their effects. Check each row's rank agreement independently.
5. **Check final-chunk masking.** With Q and valid KV fixed, overwrite cache entries strictly beyond position 131097, including padded positions 131098–131103 and logical page 4097, with large finite values. Output should be unchanged. This probes the actual mask contract without relying on the apparent 128K label. Use a fresh snapshot between trials.
6. **Verify trace and provenance once.** Compare uncaptured and captured execution from identical snapshots/inputs at the first bad stage. Record reference SHA256, all relevant source/config hashes, page tables and input hashes. Avoid changing the total batch, capacity or random draw order as a purported single-variable control.

The passing `sdpa_hifi4_k128` control changes both fidelity and chunk size (executed probe snapshot `c2af4ea9ff3c`, lines 22–31). Later single-factor controls establish that K128 helps the original query while HiFi4 alone does not clear the early window. Fixed-operand comparison is still needed to identify the numerical mechanism. Any policy selected for production must pass the full 1024-step, state, and other required gates at the recorded source revision.

## Other potential issues and evidence gaps

- **The “exact FP32” policy label is stronger than its execution contract.** Besides BF16 running-state buffers, SDPA's separate `exp_approx_mode` defaults to `true` unless explicitly set (`sdpa_decode_program_factory.cpp:221–223`). Moreover, the selected BF16 causal half-tile path uses `VectorMode::R`, which forces approximate score exponentiation even with that flag false (`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp:369–407`). `math_approx_mode=False` does not make this an exact-softmax path. Baseline shares the half-tile behavior; this is a labeling/coverage discrepancy, not a distinguishing cause. Merely toggling the exp flag is not a clean exact-exp control here.
- **Do not use current `--precise-ccl` as a valid BF16 control.** `tests/multichip_precise_ccl_candidate.py:9–25` omits the default BF8 input cast but inherits BF8 receive scratch. A BF16 input makes the factory bind a 16-tile, 32768-byte/core CB to scratch providing only `16 * 1088 = 17408` bytes/core (`all_reduce_async_program_factory.cpp:275,402–415`). Generic CB construction should reject this oversized binding (`tt_metal/impl/buffers/circular_buffer_config.cpp:72–83,223–229`); this is not evidence of default-path memory corruption. Rebuild both shared scratch buffers as BF16 for a BF16-payload probe. The default failing path has matching BF8 input/scratch and does not select this candidate.
- **Baseline top-level source metadata is misleading.** The harness always stamps `tt/multichip_decoder.py` at original line 423, even with `--baseline`. Thus the reference result's top-level `source_sha256=5cdc9e59…` does not identify its executing decoder. Use `implementation_class` and `implementation_files`, which correctly name/hash `OptimizedDecoder`. This does not show a reference mismatch.
- **The original failure has insufficient localization.** No original actual row, per-layer output or predecode state comparison survives. A successful sampled prefill PCC or final-context B1 test cannot certify all B32 pooled states or the whole prefix KV cache.
- **Older candidate passes are not current-source controls.** SDPA geometry and BF8 payload experiments using source `5cdc9e59…` support numerical sensitivity, but they do not isolate the current `80ad61a…` failure. Similarly, matching source and command schedules cannot independently prove historical tensor identity when the fixture omits inputs and checksums.

## Claim review

The draft was checked against the executable paths and passing observations, including the completed single-factor and joint-policy controls; two independent inspectors also reviewed the draft's claims. No page-wrap, missing restore, unmasked padded-token bug, or deterministic recurrent-state corruption is promoted without a contradiction in the consumed contracts. The B1 maximum-context decode really does compare against baseline, not only against itself. Large prefill chunks, gate packing, approximate exponentials, BF8 collectives and metadata discrepancies remain qualified alternatives or test gaps. The headline combines the proved SDPA precision/schedule exposure with the strongest observed attention-policy contrasts; **the specific arithmetic cause and the full 1024-step fix remain unconfirmed**.
