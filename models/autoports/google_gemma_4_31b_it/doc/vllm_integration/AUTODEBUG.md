# AutoDebug: shared sampling cannot yet implement top-k 64

## Verdict

**Do not fix Stage09 by only raising `TTSampling.max_top_k` to 64.** There are three independent boundaries: the autoport rejects 64, the common parameter formatter clamps it to 32, and the shared device operation only computes/consumes 32 finalists. Removing the first two barriers without repairing the operation exposes incorrect mask generation and reads of other users' probabilities (and, for the last user, unproduced buffer contents).

An exact top-64 candidate-selection repair is feasible inside the existing shared operation, using the generic top-k helpers. Keep the canonical split/gather → global selection → temperature/softmax → nucleus/draw pipeline. No model-specific sampling algorithm is necessary. “Exact” here means retaining the correct global top-64 set and rank order; it does not claim bit-exact agreement with HF's floating-point probabilities or RNG.

Investigation is inspection-only against HEAD `9df04b7df6909d3c5e5e8e9b286b539855df7af5`. Read the startup run contract and eval inputs. No device commands, source changes, removed implementations, or Gemma-specific generic-framework branches were used. Only this report was written. The requested HF revision/defaults are report inputs, not independently reproduced runtime observations.

## Observations and causal chain

1. `models/autoports/google_gemma_4_31b_it/tt/generator.py:74–83` constructs the shared sampler with `max_top_k=32`. `set_sampling`, lines 261–267, rejects stochastic k outside `[1,32]`. This is the direct explanation of the reported rejection; no kernel execution is needed to produce it.
2. That generator imports `format_sampling_params` from `models/common/sampling/generator.py` (line 16), and invokes it at line 272. The formatter's lines 841–846 hard-code a cap of 32. Changing the constructor and admission check alone silently converts requested k64 to k32. Its second positional argument, 32, is **batch capacity**, not top-k capacity. Temperature 1 remains 1 after inversion (line 839).
3. `TTSampling._normalize_device_params` clamps against the instance capacity (`tt_sampling.py:221–246`), but cannot recover information already lost in the formatter. `max_top_k` is read from args at line 304; changing the fallback default there does not override the autoport's explicit 32.
4. Candidate preparation is already capacity-shaped: per-split/per-device top-k, concatenation or all-gather (`tt_sampling.py:1169–1319`), and global-index offsets (`628–645`). Offset tensors scale with `max_top_k`. Values reach the final operation as BF16 TILE/interleaved; global token IDs are UINT32 ROW_MAJOR/interleaved after offset addition and untilize (`1336–1393`). The Python API and `sampling.cpp` directly delegate to the device op; there is no hidden second implementation that supports 64.
5. C++ validates shape, dtype/layout, 1–32 users, tile-aligned width, and power-of-two width tile count (`sampling_device_operation.cpp:20–180`). It does not validate the **contents** of device k. Therefore no C++ guard protects a direct k64 call from the kernel contract below.

For S equal vocabulary partitions, retaining each partition's top64 contains every global top64 candidate: any omitted candidate has at least 64 candidates ahead of it within its own partition. Keeping only 32 per partition is insufficient when all global top64 lie in one partition. The canonical candidate width is `W=S*max_top_k`, not `W=k`. For example, S=2 gives W64/Wt2 at capacity32 and W128/Wt4 at capacity64; S=4 gives W128/Wt4 and W256/Wt8. These are instantiated examples, not an assertion about an unobserved Stage09 mesh. Tied candidates need the existing tie policy preserved.

## Device contract violations at k64

All paths below are under `ttnn/cpp/ttnn/operations/reduction/sampling/device/`.

| Layer | Current code | Consequence |
|---|---|---|
| Compute | `kernels/compute/sampling.cpp:459–478`: `nearest32_K=32`, `logk=5`, Kt=1 | Only global top32 is retained, regardless of runtime k. |
| Merge | Same file, 309–350: compare single tiles, rebuild with final flag `true`, pack only left value/index tiles | Changing K to 64 is insufficient: the merge must preserve two-tile sequences. |
| Mask | Factory line 493 passes `ids_per_batch=32`; writer lines 108–109 generates one tile using k−1 | `generate_mask` reduces the position modulo tile-span (`sdpa_decode/.../dataflow_common.hpp:251–255`). k33 leaves only rank0 unmasked; k64 unmasks all 32 existing finalists. |
| Writer | `writer_interleaved.cpp:131–155,165–221`: waits one values/index tile, then extends the second face by k−16 entries | Ranks beyond 31 read adjacent rows, not the same user's second tile. |
| Completion | Writer lines 229–237 pops one values/index tile and barriers the output write | A two-tile producer requires corresponding two-tile waits/pops; a successful enqueue is not correctness evidence. |

The correct tiled **element** offset for user u and sorted rank r is:

```text
offset(u,r) = (r/32)*1024 + (u/16)*512
            + ((r%32)/16)*256 + (u%16)*16 + (r%16)
```

All divisions are integer divisions. Apply this separately through the typed BF16 values and local-index pointers: indices are UINT16 on WH/BH and 32-bit on the other-architecture branch. Do not use a fixed byte stride for both.

Concrete contrast, user0:

- k32: current ranges `[0,16)` and `[256,272)` are correct.
- k64: current ranges `[0,16)` and `[256,304)` include 32 elements from rows1/2. Correct ranges are `[0,16)`, `[256,272)`, `[1024,1040)`, `[1280,1296)`.
- User31 currently scans through element1055, past the one produced tile (1024 elements), while correct rank63 is element2047 of the second tile.

Each core computes the batch tile using its own k/temperature, so adjacent-row data is not necessarily even prepared with that other user's parameters. The resulting local index is then applied to the current user's row of global token IDs. This can select an incorrect token or distribution without producing an obviously invalid vocabulary ID. A peaked distribution whose nucleus closes before rank32 can mask the writer bug; it does not make k64 supported.

A cheap standalone Python calculation confirmed the offset mismatch for users0/15/16/31: zero mismatches at k32, one at k33, and 32 at k64. This establishes the address arithmetic, not hardware behavior.

## Smallest coherent repair boundary

This is an implementation prescription, not an applied or hardware-validated patch.

1. **Keep capability coherent at the host boundary.** Configure this autoport for capacity64 and validate against that capacity. Add a capability argument to the common formatter (preserving default32 for other callers) and pass the sampler's capacity; replace its hard-coded clamps with that argument. Preserve batch padding and reciprocal-temperature behavior. Update the shared normalization documentation that currently assumes a two-face row. Do not silently clamp the requested default64 to32.
2. **Replace the embedded one-tile top-k merge inside the same sampling compute kernel with the existing generic K64-capable helpers.** Evidence: `reduction/topk/device/kernels/compute/topk_local.cpp:128–185`, and `topk_common_funcs.hpp:29–112,194–370`. Use the non-fused stable network (current sampling factory sets stable=true); preserve paired local indices. For K64: logk6, Kt2, tiles_per_seq2, alternating local sort direction (`switch_dir=true`), initial sequence count Wt/2, seq_per_2tiles2. Follow the generic merge/rebuild schedule, which handles both target tiles, then transpose both result tiles back (`transpose_and_pack`, helper lines374–400). Do not copy the local top-k kernel's transposed transmission output directly into sampling's row-oriented softmax.
3. **Give compute and writer the same static finalist capacity.** One minimal no-new-public-API choice is K64 for accepted input widths >=64, with runtime per-user k masked down within it; explicitly handle/reject W32 as discussed below. Alternatively an explicit capacity attribute can preserve old K32 specialization, but must participate in program-cache identity and factory/kernel args. Never make cached compile-time capacity depend on mutable device-k contents without a cache contract. Neither approach adds a separate sampling path.
4. **Propagate Kt2 through mask, softmax, and writer.** Factory `ids_per_batch` must reflect 64; writer must generate two mask tiles and wait/pop two values/index tiles. Replace the two-face loops with rank loops over the above coordinate mapping for both nucleus accumulation and draw. Retain current float32 cumulative sums and RNG construction. Use the same retained-rank count and accumulated mass in both loops.
5. **Validate the supported contract.** Clearly bound k to `[1,64]` at host entry points; do not expand silently to arbitrary K. A shape-only validator cannot inspect device k cheaply. Keep W power-of-two and candidate-index width constraints; handle narrow W deliberately. Check compilation and runtime across the actual target before declaring support.

Existing factory capacities are sufficient for Ht1/Kt2: input streams4 tiles; transposed buffers Wt; values/local-values/output-indices each2; mask4; max/sum statistics each1 (`sampling_program_factory.cpp:188–243`). No blanket buffer expansion is necessary for 64. Larger W does increase transposed and final-index storage, so this is not a no-cost capacity increase.

### Concrete buffer ledger: W128, 32 users, Ht1, Wt4

| Phase | Existing K32 | Repaired K64 |
|---|---|---|
| Reader → initial sort | Produce4 value/index tiles; consume2+2 | Same |
| Transposed merge workspace | Hold4, rewrite, eventually consume4 | Same occupancy; two-tile sequence semantics |
| Finalist extraction | Produce1 value and1 index tile | Produce2 each |
| Writer → mask | Produce1 | Produce2 |
| Mask + temperature | Rotate1 value tile | Rotate2; temp remains one scalar tile |
| Max/exp/sum | Reduce1, max1/sum1; rotate1 | Reduce2, max1/sum1; rotate2 |
| Normalization | Consume1 value +1 sum; produce1 local-value | Consume2 values +1 sum; produce2 local-values |
| Writer drain | Wait/pop1 local-value +1 index | Wait/pop2 each |
| Other drains | Final-index sticks32, random1; compute mask1/max1/temp1/scalers1 each | Same except mask2 |

If only compute K is raised, `topk_mask.wait_front(2)` waits forever because writer still produces one mask tile. If only that mask count is also fixed, the writer remains incorrect. These are inseparable parts of the repair. NOC reads finish before reader pushes; writer barriers its output write. Follow-up tests must read outputs/synchronize and cover repeated cached and traced calls, not merely reach the end of an enqueueing function.

## Discriminating follow-up tests (not run)

1. **Host chain:** k64/temp1/p.95 survives autoport admission, common formatting, and sampler reset unchanged in every active lane; unused lanes remain greedy. k65 is rejected by the capacity64 adapter. Test batches1/8/16/32 and mixed k1/32/64. Explicitly verify the formatter's batch-size argument was not mistaken for top-k capacity.
2. **Shared low-level op:** W64/128/256, k1/16/17/31/32/33/48/64, p1 and .95, users1/16/17/32. Use BF16-distinct logits with substantial mass at ranks33–64; independent row permutations and non-identity global token IDs expose value/index mismatch and cross-row reads. Include user15/16/31 boundaries. Check forbidden IDs **and empirical tail mass**, since membership alone passes a sampler that only draws top32. Use p1 to isolate top-k correctness; p0 deliberately selects the first token.
3. **Canonical split integration:** one case places all global top64 in one partition, another distributes them across partitions. Compare against the global top64 distribution. Keep the existing split/gather sampler and record effective widths and mesh axis. Validate scalar temperature1 and a non-unit temperature, mixed requests, and greedy regression.
4. **Completion/cache/trace:** repeated calls changing per-user k within static capacity, same signatures with new buffers for cache-hit rebinding, and separate model/sampler trace replay. Read tokens and synchronize after each relevant sequence. Preserve seeded replay behavior; no claim that device RNG matches HF RNG.
5. **Oracle quality:** extend `models/common/tests/test_tt_sampling.py` (MAX_TOP_K32 at line38) and `tests/ttnn/unit_tests/operations/reduce/test_sampling_distribution.py` (K20 at line24). Basic `test_sampling.py:80–96` checks only support membership, and its p branch ignores top-k in normalization. `ttnn/ttnn/operations/reduction.py:264–271` explicitly skips golden comparison. These existing checks cannot certify k64 distribution correctness. Use quantization-aware references and statistically justified tolerances, avoiding nucleus-boundary ties in the main regression.

## Other potential issues and scope limits

- **W32 is an independent existing hang risk:** validation accepts Wt1, but the embedded sampling sort unconditionally waits for two input tiles (`compute/sampling.cpp:250–253`) while the reader produces only one (`reader_values_indices_tensor.cpp:38–52`). This does not explain the Stage09 rejection or the multi-partition k64 boundary. A shared repair must either reject this width or correctly implement it; replacing helpers alone should not be assumed to validate every narrow-tile detail.
- **Nucleus boundary semantics:** writer stops at cumulative probability strictly `> p`, while the distribution-test oracle drops ascending mass `<= 1-p`. For exactly representable probabilities [.5,.25,.25] and p.5, those retain two versus one token. Also .95 is BF16 .94921875 on device. These are separate semantic/oracle questions, not evidence for the reported rejection. Pin the intended policy before changing it as part of a broader correctness patch.
- **Model defaults and serving import:** this investigation proves the sampling call path from the supplied autoport generator, not that a running serving process imports it or supplies the pinned HF defaults. No live serving evidence was collected.

## Report review

Rechecked headline claims against actual imports, formatter lines, factory arguments, mask helper modulo arithmetic, compute production counts, writer addresses, and the generic top-k helper's two-tile rebuild. The first failure is host rejection; silent clamp is the next counterfactual blocker; kernel findings apply once k64 reaches the operation. The isolated W32 problem and top-p policy discrepancy were kept secondary. Source evidence establishes that the current path cannot correctly support k64; the proposed coordinated repair still requires compilation and device distribution/trace tests before it can be called validated.
