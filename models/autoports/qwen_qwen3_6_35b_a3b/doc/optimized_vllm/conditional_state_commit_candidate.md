# Conditional state-slab commit candidate

Status: implementation was delivered as an archived patch; parent subsequently applied and tested it. This subagent did not run device tests. The patch is `conditional_state_commit_candidate.patch`. It is independent of `solo_dead_norm_candidate.patch` and requires no adapter/plugin change.

## Exact change

Enable explicitly with `QwenModel(..., fuse_recurrent_state_commit=True)`. The default is false. The option is copied to each model-owned decoder; it affects only masked linear decode when the existing native state-slab guards and new activity metadata guard pass. Ordinary unmasked decode, prefill, unsupported shapes, and full attention retain the current path.

Today, masked `_linear_attention` calls `_recurrent_step` with `output_state=None`. The selected `StateSlabRecurrence` creates a 524288-byte/rank FP32 L1 state output. The caller moves this to DRAM, then calls the existing `copy_if` to conditionally commit it to persistent DRAM state. The candidate binds the state-slab writer directly to that persistent tensor and consumes the same persistent activity scalar used by the current conditional copy. The return remains `(attention_output, persistent_state)`; attention output is still produced even when activity is zero.

The existing `mesh_state_slab_recurrence.py` implementation is reused, not duplicated. Source validation confirms that its `READER` and `COMPUTE` kernel strings are byte-identical. The writer receives an optional `CONDITIONAL_STATE_COMMIT` define, an activity accessor/address, and writer-private CB27 scratch (128 bytes/worker, including alignment allowance). It reads the scalar on each invocation. Only the four state NoC writes are conditional; `ns.wait_front(4)`, write completion barrier, `ns.pop_front(4)`, output wait/write/barrier/pop remain unconditional. Compute cannot be stranded behind a full state CB when the row is inactive.

The activity contract is deliberately the selected generator scalar: INT32 ROW_MAJOR `(1,)` logical/padded shape in interleaved DRAM. Conditional output must be an existing DRAM state tensor. `supports_native` returns false for unsupported metadata; the decoder then uses its complete original recurrence/staging/`copy_if` path. No host activity value is read or captured. No new tensor refresh is introduced. All persistent slot, token, position, cache and sampling interfaces remain unchanged.

In-place state output already exists in unmasked decode. Each state-slab worker reads and writes a disjoint `(head, column)` slab. The final contraction reads the local newly computed state, not the external persistent destination, so predicating external state writes does not change output arithmetic. Rounded-state consumption, FP32 storage, ordered reductions and all packing formats remain unchanged. Hardware equivalence is still required.

This removes two state-transfer operations and one extra DRAM read/write pair per active linear layer. The 30-layer extra DRAM traffic is 30 MiB/rank per token; this is a source-derived byte count, not a latency prediction. The new writer reads 4 activity bytes on each of32 workers and allocates128 scratch bytes per worker. Trace and runtime savings are unmeasured.

## Focused discriminator

New `tests/run_conditional_state_commit_probe.py` uses the existing fixture loader from `run_stage7_state_slab_probe.py`: real positions128/8157, signed zeros, cancellation, subnormal products and BF16 round midpoints. It calls the actual production state-slab helper for both sides:

- Reference: temporary L1 state → DRAM conversion → existing `copy_if`.
- Candidate: persistent state plus replay-time activity in the same state-slab writer.

For each fixture it compares full FP32 output and state **bits** on all four ranks in eager and traced execution. Inactive steps separately compare state with its previous snapshot. The default eight-step activity sequence is `0,1,0,0,1,1,0,1`; steps3/5 perform no input or activity refresh, while other real-fixture steps change operands. Both rounded and unrounded helper contracts should be run, with rounded1 being the production path.

The two exact signatures, state-copy helper, input uploads and readback paths warm before either capture. Tracker1/skip0 remains enabled. Only the two distinct captured output buffers are acknowledged; persistent state, operands and activity predate capture and are never exempted. Reference intermediates die inside each call. Replays use `blocking=False`; diagnostic host snapshots occur afterwards. Input/state/activity addresses and program-cache entry count must stay stable. Both traces release and the mesh closes in `finally`. There is no component timing mode or profiler.

Run after applying the patch, under the hardware owner's serialized external timeout:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_conditional_state_commit_probe --rounded 1 --steps 8 --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/conditional_state_commit/rounded1.json
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_conditional_state_commit_probe --rounded 0 --steps 8 --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/conditional_state_commit/rounded0.json
```

The reduced bucket and all40 logit harnesses accept `--conditional-state-commit`. The reduced harness passes `fuse_recurrent_state_commit=True` only to the candidate and creates a separate fixed32 reference even when this flag is used alone. That reference retains the original staging path. Its existing ND reference choice remains unchanged (interleaved). The all40 harness applies the flag only when constructing its standalone candidate; HTTP tests still require separately recorded server configuration.

After the microgates, run reduced state/logit coverage with the kept route/solo settings and no ND flag if ND is rejected:

```bash
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_bucket_probe --four-row-bucket --route-mask --route-plan --solo-attention --conditional-state-commit --steps 5 --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/conditional_state_commit/buckets.json
python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_logit_determinism --mode standalone --four-row-bucket --route-mask --route-plan --solo-attention --conditional-state-commit --serial-output-tokens 96 --compare-full-logits /path/to/previously-validated-control.pt --output models/autoports/qwen_qwen3_6_35b_a3b/readiness_vllm/conditional_state_commit/full40.json
```

The full-logit reference must be the recorded, previously validated control artifact; the placeholder above is not an executable artifact path. Require active→dormant→active transitions for low/high slots, exact conv/recurrent/KV state and logits, followed by full40 teacher-forced logits and serial trajectories. Measure native and real-vLLM workloads against matched settings. An isolated conditional-commit timing comparison permits individual attribution. A final all40/performance run may combine independently microgated dead-norm and conditional-commit changes, but must report their gain as combined unless individual timing exists. Repeat mixed-shape trace reuse and final sampling/async gates before adoption.

## Completed source-only checks

- Python compilation of all three patched temporary source copies and the new probe passed.
- Black23.10.1 check of temporary source copies passed.
- Applicable precommit hooks for the new probe passed.
- `git apply --check` passed against the frozen active source.
- AST extraction confirmed identical reader and compute kernel source strings.
- `--mode cpu` loaded/validated all six existing fixtures and wrote `/tmp/qwen-state-commit-qHokaj/cpu_fixture_validation.json` with `hardware_verified=false`; it executed no device comparisons.

Remaining kernel uncertainty: conditional writer compilation, accessor binding, scratch alignment/CB27 legality and exact completion behavior must be established by the device microprobe. Full-model L1/trace pressure and serving benefit remain unverified.

## Parent-reported validation update

The first hardware attempt failed before any comparison or capture: `KernelDescriptor.defines` requires a sequence of `(name, value)` tuples, not a dictionary. The archived patch now uses `[("CONDITIONAL_STATE_COMMIT", "1")]`, matching the parent’s active API fix. The first failure artifact is retained; it has not been overwritten or reclassified.

Parent reports the corrected v2 microprobe passed 96 comparisons with two captures and two releases for each of rounded1 and rounded0. These are component results; integrated reduced/full40 and serving timing evidence remain separate gates.
