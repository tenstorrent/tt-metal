# Prefill-specific packer accumulation candidate

`candidates/precisionphase65.py` is a new frozen source-only candidate copied
from `prefill64.py`. Existing candidates and the main decoder were not modified.
The parent reported B32/S33 prefill at 39.08 ms for bank60 with packer accumulation
disabled versus 34.87 ms for norm50 with it enabled. Those are motivating
observations, not an isolated before/after measurement: the candidates also
differ in other details. This candidate permits a controlled prefill comparison
without undoing the validated decode setting.

## Policy and selection

The original `self.matmul_configs` are unchanged. Under the validated policy,
their `packer_l1_acc` remains false for decode. A second immutable dictionary,
`self.prefill_matmul_configs`, is created during weight setup with identical
fidelity, approximate-mode, and FP32-destination settings. Its packer accumulation
defaults to true. The global override is `prefill_packer_l1_acc`; per-group
overrides are `attention_prefill_packer_l1_acc` and
`mlp_prefill_packer_l1_acc`. For a disabled control, set the global override false.
No mutable current-phase field, runtime config mutation, or public API change
is introduced.

| Path | Selected compute configuration |
| --- | --- |
| Compact corrected attention projection in `_linear` | Original dictionary; entire method unchanged |
| `_linear_raw`, tuned projection program, at most 32 rows | Original dictionary |
| `_linear_raw`, standard/minimal nondecode branch, more than 32 rows | Prefill dictionary |
| `_linear_raw`, unprepared small matrix, at most 32 rows | Original dictionary conservatively |
| Explicit `_finish(..., prefill=True)` fused minimal SwiGLU | Prefill dictionary |
| Explicit `_finish(..., prefill=False)` fused minimal SwiGLU | Original dictionary |
| `minimal_mlp_flat_prefill` fused SwiGLU | Prefill dictionary |

Selection uses the actual per-call logical row count after existing chunking.
For B32/S33, 1056 rows split into 512/512/32; the first two matmuls use the prefill
dictionary and the 32-row tail retains the tuned decode program and its precision
policy. B1/S33 uses the prefill dictionary. B32/S1 decode uses the original
dictionary. Small prefill matrices deliberately remain eligible for the tuned
decode program; no alignment constraint is added to logical sequence lengths.
The corrected flat-prefill helper calls `_linear_raw` and inherits this routing.

## Why large prefill may benefit

The standard two-dimensional multicast matmul factory enables packer L1
accumulation when the flag is true and there is more than one K block. See
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`
around its `packer_l1_acc_en` calculation. With FP32 destination enabled, the
intermediate CB remains FP32 under either setting, even when final output is
BF16. Thus this is not a claim that the false setting spills BF16 partials in
the current policy.

The compute kernel
`ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp`
uses packer accumulation to combine intermediate K blocks in L1 and postpones
the partial reload until the final iteration in the non-bias path. Without
`PACKER_L1_ACC`, spilled partials enable reload at each iteration. Large prefill
has more M/N output subblocks for this repeated work, so its opportunity can be
larger than decode. BF16 output by itself does not explain the speed difference:
the repeated intermediate accumulation/reload schedule is the relevant source
mechanism. Different summation paths can still change numerical results, so
prefill state and the subsequent long decode sequence must be checked.

One limitation is explicit: the current `experimental.minimal_matmul` factories
read `packer_l1_acc` from the supplied config but do not use it to gate their
kernel accumulation. The standard kernel `device/kernels/compute.cpp` enables
L1 accumulation after the first K block and disables it after the K loop;
`compute_metal2.cpp` has the corresponding unconditional transitions. Supplying
the phase-specific config to fused SwiGLU therefore expresses the intended
policy but is not expected to change its current implementation. No shared
source was changed, and no performance benefit is attributed to that call
without device evidence.

## Source-only verification and required device checks

`candidates/check_precisionphase65_cpu.py` parses the original and new source,
asserts the compact corrected `_linear` method is AST-identical, and executes the
actual new `_linear_raw` method against a fake TTNN interface to inspect routing.
B32/S1, B1/S32, B1/S33, B32/S33, B1/S513, and B3/S129 pass, including the chunked
small tails. Output shapes are preserved. Results are in
`precisionphase65_routing_cpu.json`.

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_precisionphase65_cpu.py
```

No device run was made by this helper. The parent should compare prefill64 and
precisionphase65 under identical policies and weights, run short correctness,
and rerun the unchanged 512-step sequence because prefill state has changed.
Report warmed prefill and traced warmed decode independently, then collect the
relevant operation rows if prefill improves or regresses. Decode source routing
is preserved, but a passing long run must be established rather than inferred.

## Subsequent attribution update

The parent reports that integrated57 with attention packer accumulation false
and MLP accumulation true passes 512 steps at the same minimum PCC, 0.995822, as
both false, with compact decode around 2.34886 ms. Captured decode attention
partials in projection58 were identical under both packer settings. Therefore
the evidence does not establish that disabling decode accumulation repaired the
sequence: the attention prefix/prefill state is a plausible source of the gain.
precisionphase65 may regress that state when it enables attention prefill
accumulation. Its pending long sequence must decide; do not interpret unchanged
decode routing as proof of unchanged sequence correctness. A future explicit
prefix-false/decode-true isolation would test this attribution without changing
the quality gate.
