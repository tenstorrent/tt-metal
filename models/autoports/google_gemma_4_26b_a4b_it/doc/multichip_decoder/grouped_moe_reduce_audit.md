# Grouped shared/routed reduction proposal

2026-09-26, source/CPU only. No runtime change, TTNN import, hardware use or new
measurement. [grouped_moe_reduce.patch](grouped_moe_reduce.patch) is optional and
applies **after** [shared_geometry.patch](shared_geometry.patch), to SHA-256
`17376326fdf3d75c8710cc71cd66fd7cfb6c50a08762eeb80d9d9a29e1f9cb95`.
The combined candidate SHA-256 is
`be27d72163e05ac9ff294624e2475412ebc2bf9ba5f5a1afaa022816cac32426`.
Both artifacts were rebased onto AGMM/topology options and bounded continuation
assembly, beginning with runtime `a102e006caea6b91bf233981a9f734c4a464ebb086a40bab110ff62e8700ac65`.
Exact-base apply checks and full-candidate AST checks pass. Both optional flags
remain disabled by default, so their performance effects can be measured separately
while sharing one source version.
The parent owns candidate selection, CLI wiring and device execution.

The candidate stacks local shared and already-weighted local routed outputs,
performs one existing allreduce, then splits the two outputs before their
independent normalization. It saves one reduce-scatter/all-gather pair without
changing expert selection or normalization order. Whether that saving outweighs
the packing/splitting overhead needs a paired whole-layer measurement.

## Minimal runtime surface

- Add factory `grouped_moe_reduce=False`. Enabling it with
  `sharded_residual=True` fails explicitly; this first candidate covers the
  selected replicated residual layout. AGMM's sharded residual candidate is a
  separate experiment.
- Give `_SharedMLP.__call__` an optional `reduce_output=True` keyword. Its
  default behavior remains unchanged. With false, it returns the existing local
  down projection before `self.reduce`.
- In replicated `_forward`, compute routed experts first, then shared MLP,
  keeping both local. The helper joins `(shared, routed)`, calls `self.allreduce`
  once, and returns slots 0 and 1 respectively. Both ordinary tail norms and
  `_fused_tail` receive separate tensors in their existing argument order.
- No new weights, collectives implementation, topology setting, precision
  policy, cache operation or expert path is introduced. The source patch only
  modifies model-local orchestration and can be reviewed independently of the
  optional shared projection geometry proposal.

## Shape, dtype and collective contracts

`_forward` supplies BF16 shared and expert activations. Optimized shared decode
explicitly produces BF16; the original BF16 shared projection closures infer
the same dtype from BF16 activations/weights. Expert decode's weighted matmul
explicitly outputs BF16 (`tt/optimized_decoder.py:234`); active prefill starts
with BF16 down output and reduces weighted experts in that dtype (lines
243-278). TP, EP, and hybrid selections retain these interfaces. The helper
checks matching BF16 shapes `[1,1,S,H]` before joining; it does not silently cast
a future higher-precision branch.

```text
local shared: [1,1,S,2816] BF16
local routed: [1,1,S,2816] BF16, expert weights already applied
concat tensor dim 1: [1,2,S,2816]
reduce-scatter tensor dim 3 over mesh axis 1: [1,2,S,704]
all-gather tensor dim 3 over mesh axis 1: [1,2,S,2816]
split tensor dim 1: two [1,1,S,2816] tensors
independent shared/routed post norms, then existing sum and final norm
```

The mesh's cluster axis 1 is the four-device axis of the physical 1x4 mesh.
Tensor dimension 1 is only the two-slot stack; sharing the numeric axis index
does not mean the CCL reduces shared and routed slots together.
`models/demos/gemma4/config.py:96-128` always scatters/gathers **tensor dimension
3**, passing the caller's `axis` as `cluster_axis`. The model passes `axis=1`,
one link and the selected topology (Linear by default) through its existing
`CCLManager`. The rebase preserves the caller's Ring/Linear selection. The
memory inventory below specifically describes Linear's intermediate buffer.

For rank-local partials `s_r` and `m_r`, the grouped result is exactly the
elementwise mathematical pair `[sum_r s_r, sum_r m_r]`. It never computes
`sum_r(s_r+m_r)` before the branch norms. Finite-precision reduction scheduling
can still change with shape, so source algebra does not establish bitwise or
PCC equivalence.

`reduce_scatter_minimal_async_op.cpp:16-65` accepts rank-four tensors and requires
the scattered dimension's tile count to divide the ring size. Here 2816/32=88
tiles divides four, giving 22 tiles/704 hidden values per rank. BF16 tile pages
are 2048 bytes and aligned. Non-scattered dimension 1 may be two; it need not
divide the mesh size. Output dtype and page layout inherit the input.

## Decode, prefill and active-expert boundaries

Decode's logical S=1 is tile-padded to 32. Joining on dimension 1 does not join
or compact token rows: that dimension is unpadded 1→2. The concat padding
fallback checks the concatenated dimension, so S padding does not cause the
all-input row-major conversion found in long height-concat. The two split
slices start at aligned H/W coordinates with unit steps and stay tiled.
Both inputs are interleaved; concat permits differing L1/DRAM source placement
while the candidate explicitly selects an interleaved DRAM joined output.

Fresh prefill calls `_forward` on at most 1024 physical token rows. The grouped
tensor therefore remains chunk-bounded even for a 262143-token logical prompt.
Short final chunks retain existing physical padding and logical output trimming.
Prefix continuation and heterogeneous batched decode use existing per-token
decode orchestration and thus reach the same single-row grouped path. The
sharded-residual path is intentionally rejected when this candidate is enabled.

Routing and sparse projection calls are unchanged. TP decode still executes
the eight indexed selected experts; EP decode still allows 0..8 local active
experts, with empty ranks supplying a zero routed slot; prefill still uses its
32-token expert unions. Grouping adds no dense all-expert decode or host-side
route inspection. Page tables, RoPE, K/V preservation and attention's separate
output collective are untouched.

## Memory and possible latency tradeoff

Let `B=ceil(S/32)*32*2816*2` be one BF16 branch's physical payload. The packed
tensor and gathered result each occupy `2B`; scattered output is `B/2`.
For Linear topology, the native RS output-spec code at
`reduce_scatter_minimal_async_op_device_operation.cpp:213-240` allocates an
intermediate with twice the input volume, so grouping gives `4B` there.

| Payload per rank | Decode S1 | Prefill S1024 |
| --- | ---: | ---: |
| One branch B | 180,224 B | 5,767,168 B |
| Joined tensor 2B | 360,448 B | 11,534,336 B |
| RS output B/2 | 90,112 B | 2,883,584 B |
| Linear RS intermediate 4B | 720,896 B | 23,068,672 B |
| Conservative listed-buffer inventory 12.5B | 2,252,800 B | 72,089,600 B |

The inventory adds both local branches, joined input, RS intermediate/output,
gathered output and both split outputs. Some lifetimes do not overlap and some
locals reside in L1; it is a conservative payload inventory, not a measured
DRAM peak or an exact delta over separate reductions. Existing norm/MLP/fabric
workspaces remain outside this small inventory. There are no new persistent
allocations or full-prompt copies. Normal Python reference release preserves
possible slice aliases; no forced deallocation is added.

The candidate removes one RS launch, one AG launch and their corresponding
barrier/semaphore use from the MoE section. It adds one concat and two slices,
and retains local routed output while shared MLP runs. Communicated useful
payload remains the same total two branches; doubling one collective's payload
does not halve network bytes. Decode may benefit from reduced collective setup
cost. Prefill may be more sensitive to the extra DRAM copies and changed
pipeline scheduling. No improvement follows from operation count alone.

Semaphore ping-pong use and the captured op sequence change. Warm the exact
enabled configuration before capture, compare eager and replay outputs, refresh
positions, and inspect complete process closure under the stage's required
Watcher configuration. Compare grouped versus separate reductions with the
same shared precision/geometry, topology, active-expert policy, and complete
layer latency window. Test both layer kinds, a nonaligned multi-chunk prefill,
batched/prefix cache contracts, and EP ownership including an empty rank before
considering acceptance. A successful local collective test cannot replace
these decoder checks.

CPU validation: proposed source AST parses; `git apply --check` passes against
the recorded base; shape/byte arithmetic was checked for S=1,32,33,1023,1024.
No hardware result or changed performance number is asserted.
