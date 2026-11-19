# Norm L1 handoff V3: release after the final projection consumer

`norm149_b32.log` records the first whole-path API failure for the V1 norm
handoff. At MLP down, a static dataflow-buffer region ending at 1,312,448
overlaps a live L1 allocation starting at 1,310,720 on rectangle
`[0,0]-[6,9]`: an overlap of 1,728 bytes. The traceback is
`decode_forward -> _finish -> _linear(mlp.down_proj.weight) -> _linear_raw`.
No whole-decoder PCC or timing result was reached. This is not a numerical
failure or evidence against the handoff family.

Source inspection identifies two unnecessarily live normalized L1 tensors:

1. `decode_forward` retains its input-norm `h` across `_finish`, although all
   attention consumers (including the output gate projection) have completed
   their API calls before `_finish` begins.
2. `_finish` retains post-attention norm `h` after both MLP gate/up projections,
   even after the `projected_input` alias is deleted. MLP down reads `activated`,
   not `h`.

Each `[1,B<=32,5120]` BF16 TILE norm output occupies 40 KiB on each of eight
cores. Retaining both through MLP down can therefore consume 80 KiB/core that
the existing DRAM-output path did not retain. Exact allocator behavior and the
resulting improvement still require device validation.

## V3 change

Frozen `candidates/norm_l1_handoff_v3_on134.py` starts from the V2 clone rounding
candidate. It explicitly releases only these private normalized outputs:

- After the attention call returns, before `_finish`, in both decode branches.
- At the same boundary in prefill, covering small compact chunks and
  continuation paths.
- After the last gate/up projection and deletion of `projected_input`, before
  activation/MLP down, in `_finish`.

Every new release is guarded by full-attention kind, enabled
`norm_projection_l1`, and `h.is_sharded()`. The existing `l1_mlp` postnorm
deallocation remains the single owner when that policy is enabled. No caller
input, residual `x`/`xf`, paged cache, conv/recurrent state, projection output,
or weight is deallocated by this change. All buffers released here are fresh
private `_norm` results and are not returned to the caller.

The release follows the repository's existing explicit `ttnn.deallocate`
pattern: operation API calls have enqueued the last reads in order before
deallocation. No host synchronization or device-to-host transfer is inserted.
Trace allocation/replay correctness must be checked by the existing device
tests, not inferred from source ordering alone.

Arithmetic, norm conversion, residual placement and projection geometry are
unchanged from V2. The generator accepts another immutable base if a separate
V1/full-RNE lifetime-controlled experiment is needed; V1 is not silently
changed or declared correct.

## Artifacts and validation status

- `candidates/make_norm_l1_handoff_candidate_v3.py`
- `candidates/norm_l1_handoff_v3_on134.py`
- `candidates/norm_l1_handoff_v3_source.json`

Generator and py_compile passed. AST comparison confirms that only `_finish`,
`decode_forward` and `prefill_forward` differ from V2. No hardware was used in
source preparation. Parent should first complete the isolated V2 rounding
probe, then run V3 full-attention short correctness, traced replay, repeated
latency and the unchanged long-sequence gate. Any remaining allocation failure
requires inspecting the actual live allocations; it does not justify rejecting
the family after this first lifetime repair.
