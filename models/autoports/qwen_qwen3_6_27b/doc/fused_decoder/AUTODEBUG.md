# AutoDebug: MLP shared-LHS packing

## Headline

The reviewer finding is valid as an **exhaustion-gate gap**, not yet as a
correctness bug or performance win. `FusedDecoder._mlp` still issues separate
`linear(hidden_states, mlp_gate)` and `linear(hidden_states, mlp_up)` calls,
while `work_log.md` rejects packing only because there are two peers. Nothing
in the inspected TTNN usage establishes an exact blocker: this stage already
packs unequal-width shared-LHS projections with `ttnn.concat(..., dim=-1)`,
one `ttnn.linear`, and exact output slices.

The candidate must therefore be implemented and measured (or rejected by a
concrete TTNN failure). It may lose: replacing one linear dispatch with two
slice dispatches and doubling the packed output width can cost more than the
saved matmul dispatch, especially for decode b1/b32.

## Smallest correct candidate

At the end of `FusedDecoder.from_state_dict`, for **both layer kinds**, create:

```python
decoder.weights["packed_mlp_gate_up"] = ttnn.concat(
    [decoder.weights["mlp_gate"], decoder.weights["mlp_up"]],
    dim=-1,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

The transferred weights are `[5120, 17408]`, so the packed RHS is
`[5120, 34816]`. In `_mlp`:

```python
projected = ttnn.linear(
    hidden_states,
    self.weights["packed_mlp_gate_up"],
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
gate = projected[..., : self.intermediate_size]
up = projected[..., self.intermediate_size : 2 * self.intermediate_size]
hidden_states = ttnn.multiply(
    gate,
    up,
    input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
```

Packing order is semantically important: gate must be the first slice because
the existing binary kernel applies SiLU only to input A. Swapping slices would
compute `silu(up) * gate`, which is not equivalent.

For compatibility with direct construction, `_mlp` should use the packed path
only when `"packed_mlp_gate_up"` is present and otherwise retain the current
two-linear path. The checked device smoke harnesses directly construct
`FunctionalDecoder`, not `FusedDecoder`, but the constructor remains public
and accepts an arbitrary weights dictionary; an unconditional lookup would
turn any direct `FusedDecoder(...)` harness into a `KeyError`. Do not silently
construct the packed tensor at runtime.

Expected source-level runtime delta for `_mlp` is:

- linears: 3 -> 2 (one packed gate/up plus down);
- explicit slices: 0 -> 2;
- SiLU remains fused into multiply;
- no host transfer or cache/layout contract change.

The meaningful claim is one fewer matmul/linear dispatch, not fewer total TTNN
operations. Profiler rows must confirm the packed matmul and both slices.

## Predicted failure modes

1. **Numerical drift/PCC failure:** one wider matmul may select a different
   program or accumulation path. This is plausible even though concatenation
   is mathematically exact.
2. **Wrong half activated:** reversed concat/slice order silently applies SiLU
   to `up`.
3. **Shape/padding error:** slicing by padded tensor extent instead of logical
   `intermediate_size=17408` can leak padding or make `mlp_down` incompatible.
   Use the exact logical bounds above and cover non-aligned sequence tests.
4. **Decode or prefill regression:** two materialized slices can outweigh the
   saved dispatch; assess full and linear layer kinds, b1 and b32, decode and
   warmed prefill.
5. **Direct-constructor breakage:** unconditional packed-key access raises
   `KeyError` when only `mlp_gate`/`mlp_up` are supplied.
6. **Trace-only failure:** setup-time packing is trace-safe in principle, but
   the wider output/slice buffers can expose trace-region or program-cache
   behavior absent from an untraced PCC run.

## Focused verify/refute plan

First run the static fused tests and the existing fused-aware correctness
matrix (with the candidate enabled through `--decoder fused` where supported):

```text
pytest -q models/autoports/qwen_qwen3_6_27b/tests/test_fused_decoder.py
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --decoder fused --mode decode
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --decoder fused --mode prefill --sequence 33 --batch 1
python models/autoports/qwen_qwen3_6_27b/tests/full_attention_synthetic_pcc.py --decoder fused --mode prefill --sequence 33 --batch 32
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py --decoder fused --mode decode
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py --decoder fused --mode prefill --sequence 65 --batch 1
python models/autoports/qwen_qwen3_6_27b/tests/linear_attention_synthetic_pcc.py --decoder fused --mode prefill --sequence 5 --batch 32
```

Then use the existing trace/profiler harness for the decisive A/B:

```text
python models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --decoder fused --kind {full,linear} --batch {1,32} --perf-iterations 10
python -m tracy -r -p -v -o <artifact-dir> models/autoports/qwen_qwen3_6_27b/tests/traced_synthetic_pcc.py --decoder fused --kind <full|linear> --batch <1|32> --perf-iterations 10
tt-perf-report <artifact-dir>/ops.csv --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-color --csv <artifact-dir>/perf_report.csv --summary-file <artifact-dir>/perf_summary
```

Repeat the repository's existing warmed-prefill profiler invocations for full
and linear, b1/b32. Compare against the retained packed baseline CSVs under
`doc/fused_decoder/tracy/final_*`, not the older functional or SiLU-only rows.
Retain the candidate only if PCC, sequential traced state/cache checks, fallback
hard-failure, and all required profiler gates pass. Otherwise record the exact
PCC/program-config error or measured regression as the rejection evidence.

## Evidence boundary

No TT hardware was used in this diagnosis. The code proves feasibility of the
candidate graph and exposes no exact TTNN blocker; correctness and performance
remain hardware experiments. The requested fresh AutoDebug Codex backend could
not read the checkout because its sandbox launcher lacked `bubblewrap`, and
the supported Claude backend had expired OAuth, so this report was completed
by read-only local inspection after both runner attempts failed.
