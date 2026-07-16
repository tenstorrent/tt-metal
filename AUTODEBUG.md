# AutoDebug: compact ragged denoise MoE correctness regression

Date: 2026-07-16

Scope: inspection-only comparison of the current uncommitted
DiffusionGemma-local compact-ragged denoise MoE against the production
zero-drop `capacity=256` sparse path. No TT hardware reproduction was attempted
and no implementation file was changed.

## Executive finding

The compact metadata/router path is not the leading suspect. For the production
shape `(S,E,K,R)=(256,128,8,32)`, its primary/overflow map is a zero-drop
bijection, its expert-ID sort is the order required by the later 8-way reduce,
and the recorded device check says slot tokens, inverse slots, scaled route
weights, and segment sparsity were elementwise exact.

The earliest likely numerical divergence is the **first expert gate/up
projection**:

- the `capacity=256` baseline has no tuned config and TTNN auto-selects a
  batched-matmul K block of one tile;
- compact primary segments force the `capacity=32` tuned configs
  (`gate/up in0_block_w=22`, `down=2`);
- compact overflow segments use sparse-matmul configs
  (`gate/up in0_block_w=44`, `down=3`).

Thus the compact path changes projection program, K blocking, BF16
spill/packing cadence, and for overflow rows even the matmul factory before
combine. This is the first operation with a code-proven different
floating-point reduction after the mathematically exact gather, and it is
consistent with the measured one-layer `PCC≈0.999241`, `max_abs≈0.0127`.

The second strong suspect is combine. Baseline performs one
`comb @ down_flat` matmul over `E*C=32768` columns. Compact first materializes
each weighted contribution in BF16 and then performs an 8-way
`fast_reduce_nc`. Expert-ID ordering is correct, but product rounding and the
reduction tree are not the same as the baseline matmul.

Nothing inspected establishes that either delta alone causes the full
`seed0 0.99609375 -> 0.89453125` trajectory change. Diffusion feedback can
amplify either small local delta. The focused A/Bs below separate them without
rerunning broad precision sweeps.

## Direct observations

1. The compared production path is zero-drop:
   `sparse_experts_forward(..., capacity=256)` creates `E*C=32768` expert rows.
   Since each token can route to an expert at most once, no expert can receive
   more than `S=256` assignments.
2. The compact path creates 192 fixed 32-row segments: 128 primary
   one-per-expert segments and 64 overflow/dummy segments.
3. The supplied device evidence reports:
   - compact metadata elementwise exact for slot token, inverse token slot,
     scaled route weight, and sparsity;
   - a two-step reduced full trajectory watcher-clean;
   - three 48-step trace replays healthy;
   - one-layer compact-vs-capacity output `PCC≈0.999241`,
     `max_abs≈0.0127`.
4. The strict committed comparisons are trajectory metrics, not local
   equivalence metrics:
   - capacity baseline: seed 0 `0.99609375`, seed 1 `0.9140625`;
   - compact: seed 0 `0.89453125`, seed 1 `0.90625`.
   Seed 1 was already below the unchanged `>0.95` gate, but compact also causes
   a large seed-0 regression and therefore cannot be defaulted on the present
   evidence.
5. Existing ragged-prefill bit-identity does not clear this denoise candidate.
   Ragged prefill compares against the shared dense prefill path, where both
   sides use explicit BF16 weighting plus `fast_reduce_nc`. Its expert calls use
   a different HiFi4/`packer_l1_acc=False` policy from this denoise comparison.
   The denoise comparator here is the HiFi2/`packer_l1_acc=True` capacity path's
   one-hot gather, batched expert matmuls, and matmul combine.

## Operation-by-operation comparison

### 1. Router through normalized top-k

The two denoise router functions execute the same operations:

`chunked RMSNorm -> scale -> hidden_size**-0.5 -> linear -> softmax -> topk ->
sum-normalize`.

The compact path returns the normalized top-k values and indices before the
dense scatter. The baseline scatters the same values to `[S,E]`, multiplies by
the BF16 per-expert scale, and later runs a second top-k in
`build_capacity_dispatch`.

Verdict: **equivalent for the observed checkpoint/run, low suspicion**.

- Compact metadata sorts IDs and applies
  `BF16(normalized_value * per_expert_scale)` in the custom kernel.
- Baseline applies the same BF16 scale after scatter.
- The second baseline top-k can reorder the eight routes by scaled weight, but
  it does not define combine order: it scatters each route into the fixed
  expert-major column `expert*C + slot`. The baseline matmul therefore reduces
  in expert-column order.
- Compact explicitly sorts each token's eight `(expert,value)` pairs by expert
  ID before writing both `token_slot` and `route_weight`, so the pairs remain
  aligned and `fast_reduce_nc` receives expert-major order.
- The recorded elementwise route-weight/metadata check is direct evidence that
  scale conversion and ordering matched on the real device run.

Remaining condition: the baseline second top-k preserves the original route set
only when all eight scaled route weights remain above the zero-filled dense
entries. The current code does not assert positive/nonzero per-expert scales.
The elementwise real-checkpoint check rules this out for the reported run, but a
generic regression should assert the invariant.

### 2. Primary and overflow packing

For each expert, the custom kernel assigns rank in token order:

- ranks `0..31` -> primary segment `expert`, same row as rank;
- rank `>=32` -> that expert's contiguous overflow segment and row
  `rank % 32`;
- `token_slot[k,token]` points back to that exact packed row.

This matches the capacity path's exclusive token-order slot, modulo changing
the physical row stride from 256 to compact 32-row segments. Top-k IDs are
unique, so each routed pair has one rank and one packed row.

Verdict: **correct for `(256,128,8,32)`, low suspicion**.

A device-free simulation checked 100 random routing matrices and proved every
inverse slot selected the expected `(token,expert)` pair. An adversarial
feasible distribution over nine experts has counts
`[248,225,225,225,225,225,225,225,225]`, requiring the maximum 63 overflow
segments, or 191 total; the fixed allocation of 192 is sufficient.

The kernel comment's unrounded expression
`E + floor((S*K-E)/32) = 188` is not itself a mathematical upper bound because
each partially filled overflow segment incurs a ceiling. Rounding that value to
192 happens to cover the true production maximum of 191. This comment/formula
should not be generalized without a proof or explicit target-shape guard.

### 3. Gather

Baseline gathers with `disp.T @ hidden`; compact gathers with
`ttnn.embedding(slot_token, hidden_flat)`.

Verdict: **different implementation but unlikely to be the observed source**.
For every referenced row, the baseline dot product contains exactly one
`1 * hidden[token]` and only zero products, while embedding copies the same BF16
row. Padding rows differ operationally—compact copies token 0 while capacity
dispatch has zero rows—but expert FFNs are row-local and no compact
`token_slot` references padding. A direct stage dump has not yet proved the
active gathered rows bit-identical, so this belongs in the first hardware
probe.

### 4. Expert gate/up/GeGLU/down

Baseline `capacity=256` does not enter the `C == DEFAULT_CAPACITY` tuned branch.
Its batched expert calls pass `program_config=None`. TTNN's auto config for a
batched second operand chooses `k_tiles_per_core=1`.

Compact does three different things:

- primary rows always call `build_tuned_configs` when `segment_rows == 32`,
  irrespective of `DG_SPARSE_MOE_TUNED`;
- for `(H,I)=(2816,192)`, those configs select gate/up K block 22 and down K
  block 2;
- overflow rows use `sparse_matmul` with K blocks 44 (gate/up) and 3 (down).

Verdict: **highest-confidence local source and earliest likely divergence**.
The mathematical FFN is the same, but the K accumulation/spill/pack sequence is
not. The primary/overflow boundary also means two tokens routed to the same
expert can use different numerical kernels solely because one is rank 31 and
the other rank 32.

The lower-level factories make this more than a geometry-only concern. With
the active default `packer_l1_acc=True`, the regular batched-matmul factory
enables BF16 L1 accumulation when it has more than two K blocks. Baseline
gate/up therefore uses 88 one-tile blocks and baseline down uses six; compact
primary uses four gate/up blocks and three down blocks. The sparse overflow
factory enables the same BF16 spill mode for more than one block, and its
gate/up and down each use two blocks. The paths therefore round partial sums at
different boundaries.

Compute-config handling has a separate discrepancy. With default
`DG_SPARSE_EXPERT_FP32_FULL_SYNC=0`, both paths use the same HiFi2/BF16 policy.
If the Blackhole full-DST diagnostic is enabled, the baseline wraps its expert
config with `expert_compute_kernel_config`, while compact bypasses that helper
for both primary and overflow. That option mismatch is definite but does not
explain the reported default run.

### 5. Combine and TP all-reduce

Baseline:

`out = matmul(comb[B,S,E*C], down_flat[B,E*C,H], HiFi2)`

Compact:

`selected = embedding(token_slot, packed_down)`

`weighted = BF16(selected * route_weight)`

`out = fast_reduce_nc(weighted, expert-ID-sorted K dimension)`

Both then call the same TP `ccl_allreduce`.

Verdict: **mathematically equivalent, numerically non-equivalent, second
strongest suspect**.

Sorting by expert ID fixes the known route-order hazard, but it does not make
the two reductions bit-identical:

- compact inserts a BF16 materialization after each multiply;
- `fast_reduce_nc` adds eight contiguous BF16 contributions in its own
  destination-register loop;
- baseline matmul traverses 1024 K tiles, mostly zeros, in four-tile blocks and
  uses its own BF16 L1-accumulation spill cadence. It does not materialize the
  same eight weighted BF16 tensors or use the same reduction tree.

The TP all-reduce is downstream-identical, so it cannot be the first
compact-vs-capacity divergence when fed the same pre-reduce tensor.

### 6. Dummy overflow segments, aliases, and lifetimes

Unused overflow segments map to expert 0 and gather token 0, so they compute
nonzero dummy expert outputs. This is intentional only if no inverse slot can
reference them. The packing proof and recorded metadata check establish that
condition for the production shape.

Verdict: **not a credible numerical cause of the reported one-layer delta**.
Changing dummy values cannot affect output without a bad `token_slot`, and that
would produce assignment-level corruption rather than the observed
high-correlation continuous error.

There are still lifetime/test gaps:

- `grouped_input` is a reshape alias of `gathered` but is not explicitly
  deallocated;
- `primary_down`/`overflow_down` remain live aliases while their reshaped
  `*_flat` tensors are deallocated;
- optional output handles (`gate_output`, `up_output`, `down_output`) are not
  explicitly reconciled with the returned tensor handles;
- the scale cache is keyed by Python `id(scale)` and can return stale data if a
  model is destroyed/rebuilt on the same mesh and an object ID is reused.

These are ownership/hygiene risks. Healthy repeated trace replay and watcher
results argue against them as the current correctness cause, but an allocator
stress test should clear them before default-on.

## Ranked focused hardware A/B experiments

### E1 — stage-wise expert-kernel parity (run first)

Use one captured layer input and one captured routing result. Build both packed
representations once, then compare active rows before combine:

1. baseline `capacity=256` gathered rows;
2. compact embedding-gathered rows;
3. gate outputs;
4. up outputs;
5. GeGLU outputs;
6. down outputs.

For each compact active row, compare with baseline row
`expert*256 + token_rank`. Split compact rows into primary and overflow.

Then run the same 32 active rows and expert weights through:

- baseline auto batched matmul (`in0_block_w=1`);
- compact primary configs (`22/2`);
- compact overflow sparse configs (`44/3`).

Promotion/refutation:

- If gathered active rows differ, stop at gather.
- If gather is exact and gate/up first differ, H1 is proven.
- If forcing compact active rows through the baseline expert programs restores
  the baseline down rows and materially restores one-layer output, expert
  geometry is causal.
- Do not use `DG_SPARSE_MOE_TUNED=0` as the discriminator: compact currently
  ignores that selector for its primary config.

### E2 — combine-only discriminator

Freeze one set of expert `down` rows; do not rerun routing or experts.

1. Place those identical rows into a capacity-256 `down_flat` and apply the
   baseline `comb @ down_flat`.
2. Select the same rows through compact `token_slot`, multiply by the same
   route weights, and run `fast_reduce_nc`.

Compare bit match, PCC, max absolute error, and the final one-layer output after
the same all-reduce. This independently measures combine without expert noise.

If needed, split it again:

- compare baseline matmul to a host FP32 sum;
- compare explicit BF16 weighted products followed by expert-ID left fold to
  `fast_reduce_nc`;
- compare sorted and original top-k order to verify that current sorting is the
  correct branch.

### E3 — router/scale/second-top-k equivalence

Capture the router's normalized top-k tensors once and feed the same tensors to
both metadata builders. Download only compact metadata and the baseline
second-top-k result.

For every token assert:

- identical set of eight expert IDs;
- exact BF16 scaled weight per expert;
- compact `(token_slot,route_weight)` pairs remain aligned after ID sort;
- all scaled active weights are nonzero and greater than dense zero entries.

This should refute router suspicion quickly. If it fails, the first failed
token supplies a concrete scale/tie reproducer.

### E4 — adversarial packing and dummy-segment poison

Run metadata-only cases for:

- all tokens selecting the same eight experts (maximum concentration);
- the feasible 191-segment nine-expert distribution above;
- random real-shape routes.

Assert a zero-drop inverse bijection and correct expert for every referenced
row. Poison all unreferenced rows/dummy segment outputs with large finite
sentinels (or NaNs if the kernels safely propagate them) and verify the combine
output is unchanged. This independently refutes both packing and dummy-read
hypotheses.

### E5 — alias/lifetime and repeated-trace stress

After E1–E4:

- run many eager calls and capture/replay cycles while recording allocator
  high-water marks and output digests;
- compare current alias handling with an instrumented branch that explicitly
  releases reshape sources in the same style as `sparse_experts_forward`;
- enable watcher and vary allocator pressure between calls.

A stable digest with bounded memory demotes the ownership concerns to cleanup.
Any output dependence on allocator pressure promotes a lifetime bug.

### E6 — only after local isolation, rerun trajectories

Apply one isolated experimental change at a time:

1. baseline expert programs with compact metadata/combine;
2. baseline combine with compact metadata/experts;
3. both baseline expert programs and baseline combine.

Run a one-layer exact-input check first, then one denoise step, then the two
eight-step seeds. Do not infer causality from committed agreement alone; require
the local stage expected by E1/E2 to move toward baseline.

## Other potential issues and test gaps

1. `compact_ragged_max_segments` can return fewer than `num_experts` for
   generic shapes where `S*K < E` (for example `S=32,E=128,K=1`). The kernel
   unconditionally reserves one primary segment per expert, so the function is
   only safe under a stronger shape contract than its API states.
2. The unit tests added with this change cover path selection and one segment
   count. They do not exercise the custom kernel, metadata bijection, route
   scale/order, primary/overflow boundary, combine parity, or tensor lifetime.
3. Compact primary configuration ignores `DG_SPARSE_MOE_TUNED`; compact expert
   calls also ignore `DG_SPARSE_EXPERT_FP32_FULL_SYNC`. Selector behavior should
   be explicit even if the final optimized path intentionally fixes a config.
4. Early `return` from the custom kernel on segment overflow leaves output
   buffers partially/uninitialized rather than surfacing an error. The target
   shape is covered, but unsupported shapes can fail silently.

## Post-draft claim review

The headline was rechecked against the lowered code before finalizing:

- expert weight shapes are `[1,E,H,I]` / `[1,E,I,H]`, so TTNN does take the
  batched-second-operand auto branch that selects `k_tiles_per_core=1`;
- regular and sparse matmul factories confirm the different
  `packer_l1_acc` thresholds and BF16 intermediate format used above;
- the binary op defaults `weighted` to the BF16 left operand's dtype;
- `fast_reduce_nc` iterates the eight inputs into one destination accumulator
  in input order, making the compact expert-ID sort relevant and sufficient for
  pair order, but not equivalent to the capacity matmul;
- the 100-random-case packing simulation and feasible 191-segment construction
  were rerun successfully;
- Python syntax compilation passed for both inspected Python implementation
  files, and `git diff --check` passed.

The first draft incorrectly suggested ragged prefill used matched expert
matmul geometry; source inspection showed that its K blocks also differ. The
report now relies on the material distinction that ragged prefill uses the
HiFi4/`packer_l1_acc=False` policy and shares the explicit
multiply-plus-`fast_reduce_nc` combine with its dense comparator. The local
pytest checks could not be executed because `pytest` is not installed in the
available Python environment; no dependency was installed for this
inspection-only run.

## Bottom line

Do not default the compact path yet. The code supports a narrower diagnosis
than “compact ragged is numerically different”: routing and packing are
substantially cleared, while the first expert projection and the final combine
both intentionally change floating-point execution.

Run E1 and E2 on the same captured layer. They will identify whether the
one-layer `0.999241/0.0127` delta begins in expert K-blocking or appears only in
the BF16 weighted `fast_reduce_nc` combine. Only then is an eight-step replay
informative.
