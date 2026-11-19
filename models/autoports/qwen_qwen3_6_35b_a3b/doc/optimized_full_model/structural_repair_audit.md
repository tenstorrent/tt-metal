# Structural recurrence and all-reduce repair audit

CPU-only AutoFix hypothesis inspection, 2026-11-16. No production edits or
hardware execution. The following changes are source-supported candidates;
device equality, allocation tracking and timing still decide acceptance.

## 1. Remove only the terminal recurrent inverse permutation pair

In `tt/multichip_decoder.py`, `LinearTP4Mixin._recurrent_sum` takes FP32
`products` of logical/padded shape `[1,8,128,128]`. It permutes to
`[1,128,8,128]` (padded head-row dimension32), reshapes to `[8,16,8,128]`,
and executes the existing `fast_reduce_nc(dims=[0,1])`. This ordered reduction
and all FP32 arithmetic must remain byte-for-byte unchanged.

After reduction, `total[:,:,:8,:]` has logical `[1,1,8,128]`, physical
`[1,1,32,128]`. The helper permutes it to logical `[1,8,1,128]`, physical
`[1,8,32,128]`. The second call in `_recurrent_step` immediately applies the
same `(0,2,1,3)` permutation, returning to `[1,1,8,128]`. This is the exact
inverse pair to remove. The first `_recurrent_sum` computes the memory term
and genuinely needs `[1,8,1,128]` to subtract from `v`; leave it alone.

Minimal change sketch:

```python
def _recurrent_sum(self, products, *, head_rows=False):
    # Existing input permute, reshape and fast_reduce_nc unchanged.
    ...
    rows = total[:, :, :8, :]
    return rows if head_rows else ttnn.permute(rows, (0, 2, 1, 3))

# Only the second call in _recurrent_step:
output = self._recurrent_sum(existing_products, head_rows=True)
return output, new
```

The resulting logical shape, padding extent, dtype and memory placement match
the existing caller. The next consumer converts valid heads to BF16, reshapes
to `[1,8,128]`, and reduces only the last128 channels. Padded head rows cannot
affect valid rows. The selected `TP4WeightedNormTail` subsequently explicitly
zero-pads to32 heads before its native reader. Nevertheless inspect padding
and valid-head equality in a component probe: tensor metadata equality alone
does not establish that native permutation lowering copied the valid elements
as expected.

Do not remove the initial products permutation, change the `[8,16,8,128]`
factorization, replace two-stage reduction with `sum`/matmul, use native FDR,
or change the state rounding and output-state alias. Those alter the selected
numerical contract rather than deleting an inverse pair.

## 2. Keep logical rows through the direct all-reduce

`MultichipDecoder._reduce` currently changes `[1,1,R,2048]` into logical
`[1,1,32,2048]` with zero padding, moves it to `direct_ar_memory`, performs
the direct all-reduce, then slices it back to R. For every accepted direct
path `1 <= R <= 32`; tiled physical shape is already `[1,1,32,2048]`.

Minimal candidate: omit the `ttnn.pad` and return
`reshape(total, (1, rows, 2048))` directly, omitting `total[:,:,:rows,:]`.
Keep the initial DRAM conversion/reshape and L1 reshard for this experiment;
removing those simultaneously would obscure causality. All collective kwargs,
scratch selection, axis, topology, semaphore and output dtype remain unchanged.

Native evidence:

- `all_reduce_async.cpp`'s overload taking an explicit buffer calls the
  minimal primitive directly. It does not select a different collective
  family from the logical height.
- `device/all_reduce_async_device_operation.cpp` validates L1 width sharding,
  page alignment, even mesh width and scratch capacity, but has no
  logical-height32 restriction. Its output spec inherits the input logical
  shape. Blackhole DRAM input is explicitly forbidden; keep the L1 reshard.
- `device/all_reduce_async_program_factory.cpp` uses padded shape and shard
  tile counts. The unchanged `[32,64]` shards on32 cores hold two tiles/core;
  the unchanged `[32,256]` scratch shards hold the four-rank intermediate.
- `kernels/compute/reduction.cpp` adds corresponding tiles across the four
  rank blocks; rows never reduce into each other. Preserve its existing pair
  order, current native library, FP32-destination option and BF16 output pack.

| Boundary | Logical candidate | Physical/storage invariant |
|---|---|---|
| Before direct AR | `[1,1,R,2048]` | `[1,1,32,2048]`, same 32 L1 width shards |
| Scratch | `[1,1,32,8192]` | shared dtype-specific allocation, same semaphore |
| Output | `[1,1,R,2048]` | BF16, same L1 shard shape and backing tile count |
| Residual consumer | `[1,R,2048]` | replicated across1x4, no inter-layer layout change |

For R<32 the unused physical rows need no zero values: elementwise collective
arithmetic cannot contaminate valid rows, and the returned logical shape keeps
those rows invisible. Test a poison-padding control to catch any unexpected
consumer dependence. R32 is an unchanged-shape control. Include both BF16 MoE
partials and FP32 attention partials; no BFP8-policy change is proposed.

This is not a change from all-reduce to reduce-scatter/gather, or a repair of
the earlier native FP32 collective arithmetic. Keep the selected direct path
and its exact rank-reduction order, including any existing rounding behavior.
Do not return a logical32-row residual and rely on consumers to ignore it:
that changes routing and batch semantics.

## Prior controls do not reject these candidates

`doc/optimized_multichip_decoder/AUTOFIX_functional.md` records native recurrence
and native norm/projection variants with sensitive layer34 minima below0.99.
They changed arithmetic and remain rejected. The exact fused recurrence plus
modified FP32 collective passed its component oracle but took0.869184612 ms
versus the selected linear0.719504474 ms, so it also remains rejected. Neither
control removed only the inverse permutation pair or only the logical pad/slice
pair while retaining the current primitive reduction kernels. Do not resurrect
those hypotheses under the label of structural cleanup.

## Focused validation plan for the parent

First compare each change independently against a saved current-policy
control in one reduced probe. Preserve all weights and runtime policy. Capture
valid logical tensors per rank, convolution/recurrent state, exact token/logit
outputs and eager versus traced results. Require bit equality for these pure
data-movement changes, not merely PCC. Then test the combined candidate.

1. Recurrence: changed q/k/v/decay/beta across at least3 steps; compare both
   recurrent memory sums, final head outputs and next-state buffers. Cover
   `round_state=True/False` and `output_state` in-place/out-of-place because
   singleton prefill and masked inactive-slot decode share this method.
2. AR: R1/3/31/32; BF16 and FP32 inputs; distinct per-rank cancellation-sensitive
   values; valid-row equality and poisoned padded rows; repeated changing-input
   replay over shared scratch. Compare to existing production AR, not a new
   Torch FP32 oracle (the retained native rounding is part of this contract).
3. Run current selected layer0 and sensitive layer34 with real activations,
   then a batch3 mixed active/inactive generator trace probe. Run full-attention
   layer3 as the shared `_reduce` sibling consumer. No blanket tracker exemption.

Existing runner commands below apply after the parent installs its isolated
candidate or production edit; no candidate flags are invented here. All are
serial hardware jobs, with assigned/idle chips and the normal lane environment:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder --layer 0 --length 129 --context 512 --batch 1 --decode-steps 128 --seed 42 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/structural_l0.json
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder --layer 34 --length 128 --context 512 --batch 1 --decode-steps 192 --seed 154 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/structural_l34.json
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_decoder --layer 3 --length 129 --context 512 --batch 3 --decode-steps 16 --seed 42 --output models/autoports/qwen_qwen3_6_35b_a3b/doc/optimized_full_model/structural_l3_b3.json
```

For timing, rerun the selected small reduced/full-model performance harness
without tracking/watcher/profiler, then refresh affected final accuracy and
trace lifetime gates. Existing `probe_mesh_fp32_all_reduce.py` supplies useful
fixtures but tests a different arithmetic oracle and fixed32-row inputs; it
must be adapted to old-versus-new logical-row equivalence before it establishes
this optimization. `run_multichip_exact_recurrence_probe.py` installs the prior
fused recurrence and is not the appropriate candidate runner here.
