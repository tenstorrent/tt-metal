# AutoFix: packed decoder rows

Status: the original B3/layer0 failure is fixed in the inspected device
evidence. B2, B32, and higher-batch full-attention coverage remain pending;
this report does not certify the whole stage. The investigator inspected
source, saved diffs, JSON, and logs, and summarized metrics with host-only
Python. All device experiments and implementation edits were performed by
the main agent.

## Starting evidence and hypothesis

`AUTODEBUG_batch.md` diagnosed a representation error: `[B,1,H]` has physical
height `B*32`, whereas packed `[1,1,B,H]` has height32 for B<=32. The original
`batch3_probe.log` failed in `_finish` when public down-projection output was
resharded using height32. Original source SHA256:
`874baf217892c867e60f0ec2eb7bbb8978d916bcee05f71942aa7c4d7218f5eb`.

Prediction: retaining packed rows through residual addition, post-attention
norm, gate/up/product/down, and the final addition, then interleaving before
the public reshape, removes this failure without changing projection
precision or falling back to another decoder.

## Experiment and verdict

The main agent reran the same real-weight, real-activation B3/layer0/257
continuation case with benchmarking added:

```bash
models/autoports/qwen_qwen3_8_27b/tests/run_optimization_experiment.sh batch3_fixed_l0 --layer 0 --length 257 --batch 3 --continuation --benchmark --activations /home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations --policy-file models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/selected_candidate.json
```

Verdict: **verified at the original failing boundary**. The saved source
`3dd194461ca55f4fb703b8fbdd9153a5c5a237c2284618360c6d547dfbd00805`
contains the coherent packed-row repair. Its only other changes relative to
the failing source are prefill-2D options that are disabled by this test's
policy. Thus this is a focused test of the related representation fixes,
although it does not isolate each individual reshape change experimentally.

| Evidence | B/layer/length | Prefill PCC | Traced decode PCC | Changed-input PCC |
| --- | --- | ---: | ---: | ---: |
| `batch3_fixed_l0.json` | 3/0/257 | 0.999783802 | 0.999845982 | 0.999804378 |
| `movement_batchfix_control_l0.json` | 1/0/128 | 0.999771059 | 0.999833167 | 0.999907851 |
| `movement_batchfix_control_l3.json` | 1/3/128 | 0.999903560 | 0.999903440 | 0.999886692 |

The B3 continuation PCC is `0.999783743`; its four sequential traced steps
have minimum PCC `0.999783576`. All three cases report bitwise repeat replay
and the runtime host-conversion audit passing. The B1 full-attention control
also reports untouched unused cache pages. The logs agree with these JSON
results and contain neither the previous shard-height exception nor reshape
fallback warnings.

The controls use source
`ce2227331317ba043747b645d5950a4a08a8e64b04a8fd71c275ba2acf5f87ed`.
Their additional movement/minimal-prefill switches are disabled in the saved
control policies, leaving the same operative repair. Traced medians are
1.4608 ms for B3/layer0, 0.8320 ms for B1/layer0, and 0.6720 ms for B1/layer3.
These are measured run results, not a claimed speedup from the repair.

## Source review and nearby branches

The repair has the intended invariant: `_linear` extracts logical B by rank;
`keep_sharded=True` returns packed rows; `_norm` preserves packed inputs;
`_finish` retains packed h for both adds; `_public_rows` interleaves B>1 before
the public reshape. `_residual_memory` and norm `block_h` now derive from
ceil(B/32). The selected BFLOAT4_B/LoFi weights/configs and DRAM-sharded matmul
remain active. Public attention inputs and every finished prefill chunk are
rank3; the ordinary multi-token path is unchanged by this repair.

- `packed_mlp`: the default `_linear` return is public rank3, so its existing
  three-index gate/up slices remain valid; down repacks the resulting product.
- `minimal_mlp`: the private rank4 input is supported by the native minimal
  matmul contract, and down handles its packed product. Neither optional MLP
  branch was exercised by the inspected runs.
- `split_attention`: recursive projections still use public default outputs,
  preserving existing concatenation/slicing contracts; untested here.
- **Optional-policy defect:** in inspected live source
  `df63a6b8babbfb5fc551138fcc8bda4f60568494651e0a976bac500d15534adb`,
  `_linear` line228 returns packed output for `carry_output=True` without
  requiring `carry_residual=True`. With `dram=True, carry_residual=False`,
  `_finish` line334 instead adds public `[B,1,H]` to packed `[1,1,B,H]`.
  Those shapes broadcast toward `[1,B,B,H]`, or fail sharded validation;
  either outcome violates the public contract. This is a source-backed
  branch mismatch, not a reproduced hardware result. Restrict the packed
  output to the carried-residual path or unpack before the generic add.
  `carry_output` is absent from all three tested policies.
- `_linear`, `_norm`, and `_finish` still reshape incoming public data before
  interleaving. Current internal B>1 producers return interleaved data, so
  that omission does not undermine the verified path. A caller supplying
  public sharded B>1 input would still rely on native shard recomputation;
  any such supported input-memory contract needs its own boundary test.

## Remaining verification

1. Run the final selected implementation/policy for B2/B3/B32, both layer
   kinds, including singleton prefill, one-token chunk tails, continuation,
   refreshed trace inputs, and full-attention page-table swaps. Existing
   B3 linear-attention evidence does not prove higher-batch full attention.
2. The live runner now asserts public prefill/eager/traced shapes, addressing
   the flattened-PCC weakness. The saved experiment wrapper snapshots model
   source, not test-runner source, so these assertions cannot be dated to the
   earlier run from that evidence alone. Use them in the final suite and add
   per-user checks/distinct rows if checking row isolation; aggregate PCC
   alone can hide a poor row in B32.
3. At inspection, `test_optimized_decoder.py` passes `policy="{}"`, while
   `from_state_dict` still initializes `dict(policy or {})`. Running that
   version tests generic BF8/HiFi2 projections rather than this selected
   sharded policy. Promote the selected defaults or pass the policy file
   before final coverage, and assert the effective dtype/DRAM/carry policy
   in emitted results. Rejecting FunctionalDecoder/FusedDecoder construction
   alone does not prove the optimized matmul path executed.
4. Record final source/policy hashes. The live tree contains later optional
   tuning branches beyond the saved B3 repair, and their inclusion in a final
   candidate requires matching final evidence. Retain the isolated B3 fix
   evidence as causal support, without treating it as unrun B32 coverage.
