# Flat prefill candidate / AutoFix report

Status: implemented as an opt-in module, host checks passed; hardware correctness
and latency remain unmeasured. No device was imported, opened, reset, or used.

The diagnosis in [AUTODEBUG.md](debug_flat_prefill/AUTODEBUG.md) is supported by
the current source: `flat_finish` starts compaction only after the attention
projection, while `flat_prefill_correction` restores its packed projection and
leaves the input norm's hidden expansion in place. The candidate keeps each
scheduled chunk compact from input residual through both norms, projections,
reductions, residual adds, and MLP. Temporal operations retain logical users.
This establishes fewer explicit pack/unpack boundaries in source, not measured
kernel count or latency improvement.

## Files and integration

Added [flat_prefill.py](../../tt/flat_prefill.py). Existing decoder and test files
were not edited. The frozen `optimized_decoder.py` needs no changes.

The parent must make these explicit, backward-compatible changes in TP4's
`multichip_decoder.py` before enabling the candidate:

1. Add an opt-in `prefill_forward` override dispatching
   `flat_prefill_forward(self, x, state=state, start_pos=start_pos, rope=rope)`
   when `self.policy.get("flat_prefill", False)`, otherwise call the inherited
   method. Import the module locally in that opt-in branch to avoid cycles.
2. Add `_collective(..., placement_seq=None)`. Only the initial L1/DRAM test
   changes to `(shape[-2] if placement_seq is None else placement_seq) <= 32`.
   Preserve the physical shape for packing, persistent-buffer admission and
   cache keys, and output restore. `_reduce_flat` passes logical `seq`; norm
   collectives keep their existing compact-row placement.
3. Add `_qkv(..., logical_shape=None)`. Disable its `compact` decode branch
   when `logical_shape` is supplied. The existing projection and packed gate
   extraction operate on the supplied flat hidden. After extracting gate and
   slicing projected QKV to width 2048, if `logical_shape is not None`, restore
   only QKV with `ttnn.reshape(projected, (*logical_shape, 2048))` before the
   head splitter. Keep gate flat. Existing per-head norms and RoPE are unchanged.
4. Add `_gdn(..., packed=None, flat_output=False)`. When `packed is None`, keep
   the existing `b,s` derivation and `_linear(x,"packed_gdn")`. Otherwise reject
   `decode_batch is not None`, and derive `b,s=tuple(packed.shape)[:2]` from the
   supplied **batched BF16** projection. Keep all convolution, padding, beta,
   decay, recurrence, state writes, per-head norm and gating code unchanged.
   Immediately before each return's output projection, if `flat_output`, apply
   `out=ttnn.reshape(out,(1,b*s,1536))`. The candidate invokes this only for s>1.

No attention-output hook is needed: the new module reproduces the small
reference prefill cache/SDPA body with explicit logical batch and sequence, then
crops each user's temporal output and compacts before gate/output projection.
No mutable modes or monkeypatches are used.

## Preserved contracts and conservative fallbacks

- The inherited chunk scheduler, list validation, context validation, absolute
  positions, per-chunk RoPE slices and public time concatenation are reproduced.
  Each completed chunk is restored to `[B,n,1280]` before concatenation.
- Input BF16 recurrent state is widened once, remains live FP32 across every
  chunk including one-token tails, and is published once after the loop. The
  fallback calls the original **chunk operations**, not recursive public prefill,
  avoiding an intervening `_round_recurrent_state` on FP32 live state. Incoming
  FP32 state follows the inherited end-of-call rounding policy.
- Both hidden norms receive original logical batch. Flat execution requires
  sharded residuals, `flat_norm`, `flat_ccl`, and gather norm in
  `{False,"decode","single"}`. B1, n1, replicated residuals, other norm routes,
  `chunk_add_prefill_correction` and `l1_mlp` overrides use the old chunk path.
- GDN's correction uses the original two raw FP32 projections and BF16 combine.
  Calling `_linear` on physical batch1 would newly admit correction-L1 for
  R<=32; the explicit raw calls prevent that change. The combined BF16 tensor
  is restored once before temporal processing. Primary/correction physical
  strides remain owned by existing helpers.
- Full attention keeps per-user page IDs, cache writes, tail zero-padding,
  Q/K norm, causal SDPA, and RoPE. Packed QKVG keeps its gate compact.
- Minimal SwiGLU uses the same 32/512-row switch, configs, BF16 output and
  rounding. Non-minimal policies retain the existing activation arithmetic.
  Row projections reuse TP4 `_linear` and its existing row admission, padding,
  128-row MMRS pieces and trim. Collectives retain original row-reduction
  placement through `placement_seq`; norms retain compact placement.

## Validation and remaining experiment

Executed standard-library AST parse/compile of the candidate without importing
TTNN. Executed the actual candidate and reference scheduler ASTs in independent
fake tensor environments: **728 parity cases passed**, plus five policy override
cases. Covered both layer kinds, B in {1,2,3,32}, lengths
{1,17,31,32,33,63,64,65,257,511,512,513,1025}, BF16/FP32 recurrent input, GDN
starts {0,7}, and attention starts {0,1,31,32,511}. Checked every per-user output
index and chunk sequence, one live state identity, and identical cast/publication
events. These are scheduler and ownership checks, not tensor numerical tests.

Remaining: integrate the three TP4 hooks, run matched candidate/reference device
checks for output and recurrent/conv/KV state, including non-tile R, n<32
collective placement, packed/unpacked attention gates, and 513-token tails.
Verify list semantics and 512-alignment attention on hardware as well. Profile
matched warmed B32/S33 runs with synchronization to prove the movement and
latency hypothesis; compare against selected `flat_finish` plus
`flat_prefill_correction`, not only the older accepted snapshot. Native execution,
precision parity and performance are unverified until those experiments pass.
