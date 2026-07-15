# AutoDebug: Gemma 4 Stage 07 full-prefill LM head

## Report location and scope

The AutoDebug skill normally writes `./AUTODEBUG.md`. The repository-root
`AUTODEBUG.md` is a tracked, unrelated Qwen report, so this investigation is
intentionally recorded at the stage-local path
`models/autoports/google_gemma_4_31b/doc/optimized_full_model/AUTODEBUG_PREFILL_LM_HEAD.md`
instead. This was an inspection-only investigation: no implementation code was
edited and no TT hardware reproduction was run.

## Diagnosis

The Stage 07 DRAM-sharded LM-head optimization is valid only for a terminal
activation whose physical height is one tile (32 rows), but `_terminal` applies
that fixed one-tile contract unconditionally. The readiness prefill path asks
for every logit, passes a logical 249-row activation (149 prompt tokens plus 100
generated tokens), and TTNN pads it to a physical height of 256. Conversion to
the fixed width-sharded input spec `(32, 672)` therefore fails before the LM-head
matmul is enqueued.

Changing only the input shard height and `per_core_M` is not a valid fix. For
the selected `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, current
TTNN validation explicitly requires `M == 1` tile. The program factory also
rejects `per_core_M > 1` when an input K shard contains more than one block; the
selected Gemma geometry has seven blocks per shard. At the requested 8-tile M,
the selected 8192-column/BF16 configuration would additionally require about
2.08 MiB of input, weight, output, and reshard circular-buffer/backing capacity
per participating core, above Blackhole's 1.5 MiB L1.

The smallest intervention is to keep the proven one-tile LM-head program and
tile the **normalized** prefill activation along sequence into logical slices
of at most 32 rows, project each slice with the existing sharded LM-head helper,
and concatenate the resulting logits along sequence. This preserves the plain
prefill RMSNorm result, the selected BF16/HiFi2 LM-head policy, arbitrary
non-aligned logical lengths, and the optimized decode/serving-TTFT path.

## Evidence-ranked findings

### Finding 1 — High confidence: fixed one-tile input sharding is applied to a physical 256-row tensor

The exact producer-to-failure chain is:

1. `run_prefill_check` concatenates the reference's prompt and generated token
   sequences and calls `prefill_forward(..., prompt_lens=[full_len],
   return_all_logits=True)` (`models/common/readiness_check/run_prefill_check.py:91-100,131-137`).
   The checked reference contains 149 prompt tokens and 100 generated tokens,
   so `full_len` is 249.
2. `Gemma4FullModel.prefill_forward` sends the whole hidden activation to
   `_terminal` when `return_all_logits` is true
   (`models/autoports/google_gemma_4_31b/tt/model.py:567-569`). It does not take
   the last-tile slice used by ordinary prefill.
3. The Stage 07 input memory config hard-codes shard height 32 and width
   `5376 / 8 = 672` (`tt/model.py:323-335`). `_terminal` applies it
   unconditionally (`tt/model.py:449-454`).
4. TTNN's tensor-spec validation explicitly uses the physical tensor and shard
   shapes. Width sharding requires the physical shard height to equal the full
   physical tensor height (`tt_metal/impl/tensor/spec/tensor_spec.cpp:14-54`).
   Logical M=249 has tile-padded physical M=256, producing the observed
   `Shard height 32 must match physical height 256` fatal.
   An independent, no-device TensorSpec probe with the exact `(32,672)` shard
   confirmed the boundary: logical M=1, 21, and 32 construct successfully;
   M=33, 149, and 256 are rejected against physical heights 64, 160, and 256.
5. The recorded stack ends in `InterleavedToShardedDeviceOperation` tensor-spec
   creation, before a matmul or device synchronization
   (`doc/optimized_full_model/run_prefill_check.log`). This is a synchronous
   planner/spec failure, not an asynchronously surfaced decoder error.

The failing observation and passing contrasts match exactly. Decode has logical
M=1/physical M=32. Ordinary host prefill and device-logit prefill slice the last
logical tile before `_terminal` (`tt/model.py:548-568,603-618`). Reduced mixed
prompt tests exercise lengths 33 and 17 but request only sampler-ready last-row
logits (`tests/test_full_model.py:288-351`); they never pass a multi-tile tensor
to `_terminal`.

This regression was introduced at the optimization boundary. The Stage 06
terminal used an interleaved, auto-selected `ttnn.linear` for any M; Stage 07
replaced it with a fixed one-tile width-sharded input and program configuration
without adding an M-shape branch.

### Finding 2 — High confidence: dynamic padded-M sharding plus dynamic `per_core_M` is rejected by the selected operation

For physical M=256, a dynamic width-sharded input would need shard height 256
and `per_core_M=8`. That gets past the first tensor-spec check, but not the
selected matmul contract:

- DRAM-sharded matmul validation requires M tiles to equal `per_core_M`, then
  separately requires `M == 1` (`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1364-1376`).
  M=8 therefore fails even with a matching shard and program config.
- The specialized factory requires `num_blocks_per_shard == 1` whenever
  `per_core_M > 1`
  (`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:288-299`).
  Gemma's hidden width is 168 tiles. With 8 input shards and `in0_block_w=3`,
  the program has 56 K blocks, or seven blocks per input shard. M=8 is therefore
  rejected by a second independent constraint.
- The factory sizes its input, weight, output, and reshard buffers linearly in
  `per_core_M` and/or `per_core_N`
  (`.../matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:181-226`).
  For M=8, split N=8192, eight banks/cores, BF16 tiles, and block W=3, the
  relevant capacity is approximately:

  | Buffer role | Bytes |
  |---|---:|
  | multicast input (`8 * 3`, double-buffered) | 98,304 |
  | DRAM-weight block (`32 * 3`, triple-buffered) | 589,824 |
  | sharded input backing (`8 * 21`) | 344,064 |
  | output/intermediate (`8 * 32`) | 524,288 |
  | resharded output backing (`8 * 32`) | 524,288 |
  | **Total** | **2,080,768** |

  This is already above 1,572,864 bytes, before incidental allocator pressure.
  Raising `in0_block_w` to 21 would make one K block per input shard, but the
  already-recorded M=1/block-W=21 candidate reaches 2,294,528 bytes and fails
  L1 capacity
  (`doc/optimized_full_model/candidates/lm_head_dram8_split4096_block21.xml`).

Consequently, dynamic M is not a Python configuration correction. It requires
changing and validating TTNN C++/kernel support and retuning the selected split,
or selecting a different prefill matmul. It is materially larger and riskier
than the model-local shape repair.

### Finding 3 — High confidence: normalized-row tiling matches the existing passing contract and preserves non-aligned lengths

The current working unit is already the right unit for tiling:

- Every logical slice of 1 through 32 rows in TILE layout has physical height
  32 and therefore matches the fixed input shard.
- Existing non-aligned paths construct exactly such a final slice. For example,
  prompt length 33 selects logical `[32:33]`, while length 149 selects
  `[128:149]` (`tt/model.py:548-561,603-617`). These paths passed the reduced
  Stage 07 tests.
- For the readiness shape, `[0:32], ..., [192:224], [224:249]` yields seven
  full logical tiles and one 25-row logical tail; every slice has physical
  height 32. Concatenation along `-2` restores logical M=249 in order.
- The TTNN slice no-op occurs only for a complete full-range slice
  (`ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:180-183`). Every
  M>32 chunk here is partial. TILE slices require a tile-aligned height begin
  (`.../slice/device/slice_device_operation.cpp:155-166`), which starts
  0,32,64,... satisfy; the end is padded and then viewed back to the requested
  logical extent (`.../slice/slice.cpp:325-379`).
- Final RMSNorm is row-local and logit softcapping is elementwise, so splitting
  only the projection over M does not alter cross-row mathematics. Normalizing
  the complete hidden tensor first also preserves the existing prefill RMSNorm
  selection: the Gemma RMSNorm deliberately uses its sharded fast path only for
  M<=32 and its plain path for larger prefill tensors
  (`models/demos/gemma4/tt/rms_norm.py:146-176`).

The normal serving path remains fast. Both sampler-ready device prefill and the
ordinary last-logit prefill already project only the final <=32-row tile.
Decode also remains a single physical tile and keeps the selected traced
program. Only `return_all_logits=True`—the readiness/accuracy path that truly
needs all sequence logits—executes multiple M tiles.

### Finding 4 — Medium confidence: the regression escaped because no test crosses the terminal's M=32 boundary with all logits

The static contract test verifies that DRAM sharding is enabled and that the
selected values are 8 cores, block W=3, and split 8192
(`tests/test_full_model_contract.py:39-59`). Hardware tests cover M=32 terminal
inputs, non-aligned last-tile extraction, mixed prompts, and traced decode, but
none call `return_all_logits=True` with logical M>32. The full readiness command
is the first test that does so.

This is a focused test gap, not evidence against the existing reduced tests:
those tests correctly cover the serving and token-out contracts they name.

## Intervention adjudication

| Candidate | First tensor-spec check | Matmul/factory support | L1/resource fit | Non-aligned M | Serving impact | Verdict |
|---|---|---|---|---|---|---|
| Dynamic physical-height shard and dynamic `per_core_M=ceil(M/32)` | Can be made to pass | Fails explicit `M==1`; also fails seven-blocks-per-shard factory condition for M>1 | Selected M=8/N=8192 geometry is about 2.08 MiB versus 1.5 MiB | Planner could represent it, but kernel is unsupported | Would introduce new programs and retuning | Reject as a focused Stage 07 fix |
| Normalize once, tile normalized rows to <=32, reuse fixed `per_core_M=1`, concatenate sequence logits | Passes for every slice | Reuses the already-proven operation unchanged | Keeps one-tile L1 working set | Tail slice retains logical row count and physical 32-row padding | Ordinary TTFT/decode already use one tile and are unchanged | Recommended |
| Slice hidden first and call the current `_terminal` once per slice | Passes | Reuses proven operation | Fits | Handles tails | Only affects all-logit path | Viable, but it changes large-prefill RMSNorm from the existing plain kernel to repeated sharded-RMSNorm calls; normalized-first tiling is the lower-correctness-risk form |
| Restore/duplicate an interleaved BF16 LM-head weight for prefill | Passes through old generic path | Supported | Consumes another roughly 704,643,072 bytes/device for tied BF16 LM-head payload, before allocator overhead | Supported | Adds resident memory and a runtime fallback | Reject; avoid duplicate weight and fallback contract |

## Smallest intervention boundary

Refactor the DRAM-sharded portion of `_terminal` into a helper that accepts one
already-normalized logical slice of at most 32 rows and performs the existing:

1. `to_memory_config` to the fixed `(32,672)` width shard;
2. eight split BF16/HiFi2 linears with the current `per_core_M=1` program;
3. sharded-to-interleaved conversion and vocab concatenation.

Then have `_terminal`:

1. run `final_norm.forward(hidden)` once and release `hidden`;
2. use the helper directly for M<=32;
3. for M>32, iterate logical sequence ranges `[start:min(start+32,M)]`,
   slice the normalized tensor, call the helper, and concatenate tile logits
   along `-2` in DRAM;
4. release every owned slice/intermediate exactly once; and
5. apply the existing softcap once to the assembled logits.

The loop must use the logical `normed.shape[-2]`, not its padded height. The
last 1–31 logical rows should remain visible as that exact logical extent while
TTNN supplies the required physical 32-row tile. Keep the current full-range
alias rule in mind: the M<=32 branch should hand the original normalized tensor
directly to the helper; the M>32 branch uses partial slices. Keep the parent
normalized tensor alive while its partial slices are projected, and release a
slice only after its projection consumer has been submitted.

## Focused experiments

These are the smallest experiments that adjudicate the repair. The no-device
TensorSpec boundary probe described above was run; no TT-hardware experiments
were run because this investigation is inspection-only and TT-hardware-free.

1. **Pure/static tile planner:** for logical M values
   `1, 31, 32, 33, 63, 64, 65, 149, 249`, assert contiguous ranges, each width
   in `[1,32]`, and exact reconstruction of M. Assert 249 produces eight ranges
   ending in `[224,249]`.
2. **Reduced component shape/ownership probe:** with layers `(0,5)`, call
   `prefill_forward(..., return_all_logits=True)` at M=32, 33, 149, and 249.
   Assert output shapes `[1,M,262144]`, no allocation/alias error, and a second
   call after reset succeeds. M=33 is the minimal regression boundary; M=249 is
   the exact readiness shape.
3. **Projection equivalence probe:** for an already-normalized synthetic or
   reduced-model M=33/249 tensor, compare concatenated tiled projection to the
   Stage 06 interleaved BF16/HiFi2 projection (or HF logits) with the established
   PCC/top-k criterion. Separately compare the last row from all-logit prefill
   to the sampler-ready last-tile path, allowing for the already-existing plain
   versus sharded RMSNorm kernel distinction.
4. **Serving-path guard:** rerun mixed lengths 33/17 device-logit prefill and
   traced token-out decode. Verify `_terminal` still receives one physical tile,
   trace counters remain unchanged, and the headline decode result does not
   regress.
5. **Full gate:** rerun the exact AIME24 `run_prefill_check`. It must return 249
   logits and meet Stage 07 top-5 >=98% and top-100=100%. Then rerun teacher
   forcing so the repair is checked alongside, but not confused with, decode.
6. **Profiler check:** confirm serving TTFT still has eight vocab-split linears
   for one M tile, whereas the readiness all-logit path has eight M tiles times
   eight vocab splits. This distinguishes intentional all-logit work from an
   accidental serving regression.

## Claim review and remaining uncertainty

- The headline fixed-shard finding directly produces the exact fatal and
  explains every named pass/fail contrast.
- The dynamic-M rejection is not speculative: it is independently enforced by
  tensor/matmul validation and the selected program factory. The L1 estimate is
  supporting evidence, not the sole reason for rejection.
- Normalized-first tiling is mathematically row-preserving and reuses the exact
  proven narrow program, but actual PCC/top-k results and slice ownership still
  require the focused hardware component tests above.
- No evidence implicates decoder layers, KV cache contents, collectives, RoPE,
  sampling, or trace replay. The complete decoder has already returned its
  hidden tensor, and the fatal occurs synchronously while constructing the
  terminal reshard output spec.
