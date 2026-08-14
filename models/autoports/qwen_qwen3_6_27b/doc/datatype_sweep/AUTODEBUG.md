# AutoDebug Report: all-projection BFP8 decode L1 clash

## Scope and direct observations

This was an inspection-only pass. No TT device command or implementation edit
was made. The repo-local AutoDebug runner was invoked first, but its fresh Codex
process could not launch local shell reads because its sandbox reported that
`bubblewrap` was unavailable; its fallback explorers encountered the same
launcher failure. The findings below were therefore re-derived in this fresh
subagent context from the requested logs, candidate JSON, work log, git status,
and source.

Both candidate runs fail identically while capturing the first compatibility
decode trace:

- `logs/all_projection_bfp8_hifi2_teacher_forcing.log`
- `logs/all_projection_bfp8_lofi_teacher_forcing.log`

Both report program 696, cores `[0-0 - 7-9]`, an existing L1 allocation at
1,319,104, and a static-CB end at 1,333,760 (14,656 bytes of overlap). Changing
HiFi2 to LoFi does not change any of those values, strongly excluding compute
fidelity as the cause of this allocation failure.

The traceback fixes the failing call more narrowly than the generic
`_tp_linear` frame:

1. `OptimizedDecoder.decode_forward()` reaches `_mlp_decode()` at the first
   decoder layer.
2. `_mlp_decode()` has already completed the gate and up projections, because
   the traceback points to its return call at `multichip_decoder.py:616`.
3. The failing weight is therefore `mlp_down_decode`, not gate/up, attention,
   cache, CCL, or LM head.
4. That call uses `k=4352`, `n=5120`, `decode=True`, `row=True`, and the
   hard-coded `in0_block_w=17`.

The logical per-device matmul is a batch-32 decode activation of shape
`[1, 1, 32, 4352]` times the TP-local down-projection weight
`[1, 1, 4352, 5120]`, producing `[1, 1, 32, 5120]`. All dimensions are already
tile aligned (136 K tiles and 160 N tiles), so the inferred physical tiled
dimensions are the same. The activation L1 shard is `[32, 544]` over an 8x1
width-sharded grid (17 K tiles per shard); the output L1 shard is `[32, 640]`
(20 N tiles per shard). These physical-shard values follow directly from
`_l1_width_memory_config`; the failure logs do not print tensor metadata, so a
focused runtime probe should still assert them before the fix is accepted.

`mlp_down_decode` is created by `_shard_decode_weight(..., dim=-2, k=4352,
n=5120)` and stored in DRAM width-sharded form. In both failed precision
summaries, `mlp_down_dtype` is `BFLOAT8_B`. The completed optimized policy uses
BFP4 for MLP weights, so the all-projection policy increases the down-weight
tile/CB footprint. The decode program was tuned with a full 17-tile K block and
does not adapt that block width to weight dtype.

## Headline finding

### Verified policy/program incompatibility at the MLP down projection

The most likely cause is the combination of a BFP8 `mlp_down_decode` weight and
the maximum legal `in0_block_w=17` retained from the lower-footprint BFP4 MLP
policy. This predicts all observed facts: gate/up complete, failure occurs at
down, HiFi2 and LoFi have identical allocator bounds, and the failure is an L1
static-CB collision rather than a numerical or collective error.

There is also a concrete plumbing discrepancy: `OptimizationPolicy` exposes
`mlp_down_in0_block_w`, and the optimized candidate table contains down-width
variants, but `MultichipDecoder._mlp_decode()` hard-codes 17 in both packed and
unpacked paths. Thus the multichip measured path cannot use that policy field.
For this TP4 geometry, `_decode_program` computes 17 K tiles/core; its
divisibility check means the only legal block widths are 17 and 1. Values such
as 2 or 4 are not valid focused controls for this exact shape/program.

Confidence: high that the failing op and incompatible configuration are
identified; medium-high that changing the block width alone resolves the
collision, pending the hardware A/B below.

## Smallest verify/refute experiment

Run a one-layer, one-step batch-32 decode smoke using the real layer-0 weights
and the failing all-projection BFP8 policy. Instrument or assert immediately
before `ttnn.linear`:

- `weight_name == "mlp_down_decode"`;
- input logical/padded shape and L1 shard spec;
- weight logical/padded shape, `BFLOAT8_B`, DRAM shard spec;
- output memory config;
- program fields (`in0_block_w`, `per_core_M`, `per_core_N`);
- compute fidelity.

Run only two serialized cases after a healthy mesh open: `in0_block_w=17`
(expected exact clash) and `in0_block_w=1` (expected compile and one decode
completion). Test HiFi2 first; if width 1 compiles, repeat LoFi. Compare output
against the same real-weight BF16/BFP4 control with the existing decoder-layer
correctness metric. This is narrower and cheaper than another 64-layer
teacher-forcing run.

## Smallest fix hypothesis

Replace the hard-coded down-projection width in both multichip MLP paths with a
policy-consumed value, and make the all-projection BFP8 policy request the legal
width 1 for `mlp_down_decode`. Preserve width 17 for the passing BFP4 baseline.
After the one-layer A/B passes, rerun the original traced teacher-forcing
command for both HiFi2 and LoFi candidates. This both fixes the likely L1 issue
and closes the existing runtime-policy-consumption gap.

If width 1 still produces the same allocator bounds, the hypothesis is refuted:
capture the lowered matmul program/tensor metadata to determine whether TTNN
ignores the explicit DRAM-sharded program config for BFP8. Do not reject BFP8
from the full-model sweep until that exact op-contract behavior has been taken
through AutoFix.

## Other observations

- KV cache, CCL dtype, and LM-head dtype are present in the candidate policy but
  cannot cause this first-layer MLP-down compile failure; none is used by the
  failing `ttnn.linear` call.
- The 64-layer precision summary proves the requested dtype reaches layer
  construction, while the identical allocator error proves fidelity changes
  do not affect this failure. Neither alone proves the down weight's runtime
  tensor metadata; that is why the shape-exact probe is required.
- Existing optimized-decoder artifacts contain similar L1/CB clashes for wider
  decode blocks, consistent with the mechanism, but they are supporting
  precedent rather than proof for this exact BFP8 TP4 op.
