# AutoDebug: stacked BF8 attention override and indexed expert merge

## Finding

**No source-proven CCL semaphore, program-cache binding, or missing expert-mix
padding bug was established.** The stack failure remains unlocalized. A later
single-layer sliding capture localizes its separate repeatability failure to
the local routed-expert result, with attention CCL, actual sharded post-attention
normalization, router outputs and expert input stable. Those observations
justify expert-path localization; they do not establish that the stack has the
same cause, or that native BF8 CCL is unsupported.

Two concrete evidence gaps matter:

1. The original native CCL replay probe compares each rank only with that
   rank's first replay. It never checks cross-rank agreement or an oracle.
   Its128-seed/3-repeat passes prove per-rank repeatability, not replica equality.
2. The stack's `read()` calls `torch.equal` before any finite-value check.
   Identical NaNs on every rank also fail that assertion. The failed log gives
   neither layer nor step, so "Stack output replicas differ" alone does not
   prove finite cross-rank divergence.

No implementation fix is proposed or applied.

## Scope and evidence

This is a fresh delegated, source-only AutoFix/AutoDebug investigation. No
TTNN import, device access, target execution, reset, or implementation edit was
performed. The mandated repo runner was attempted from
`/tmp/gemma4-stack-ccl-autodebug` using `.agents/scripts/autodebug.sh`; its fresh
CLI could not execute even `pwd` because its sandbox launcher lacked working
`bwrap`. The run was stopped and its log retained there. This report therefore
comes from the delegated inspection, not from a successful CLI report.

The inspected current files match `stack_selected_bf16.json`:

- `tt/multichip_decoder.py`: SHA256
  `8b59370cda6f4ff88157de123123509036f2e91e8054000c809752e21f933175`.
- `tests/test_multichip_stack.py`: SHA256
  `7e5e1ad3b084214a15b44e22ede77637f1c3f3c96103f5fa2bdc8d1564dbb5cc`.

Current policy: hybrid EP prefill/indexed TP decode, fused tail, shared geometry1,
grouped BF16 shared+routed reduction, LoFi QKV/WO, replicated residuals, BF8 KV
cache. The passing stack uses BF16 attention CCL for both layers. The failed
`stack_selected_full_bfp8.log` changes full-attention CCL to BF8 and ends at
`test_multichip_stack.py:196`, the first read after a blocking replay. It does
not identify which iteration of that read failed. Both prefill output reads
completed before capture. The BF16 stack passes all replica/replay gates with
minimum PCC0.9989911988.

The older `full_ccl_bfp8.json` passes an isolated full layer4096/128 and is a
useful contrary observation to a blanket BF8 prohibition. It uses a different
runtime hash, fixture, prefix and execution graph from the stack. The stack
is a synthetic `(0,5)` handoff at prefix33 and positions33/34, not the same
full-attention input as that isolated pass.

The later `bfp8_boundary_v3_all.json` uses current runtime hash8b59370c and
diagnostic hash3a783db8. Its first changed boundary at step6/position4102 is
`routed_local`, rank1:2697 finite changed elements, maximum difference0.01953125.
WO/cast/RS/AG, sharded post-norm, residual, rounded router scores, selected IDs,
routes and expert input are stable. The BF16 grouped reduction propagates that
local difference to identical changed outputs on every rank. That is a
different observed signature from the still-unlocalized stack assertion.

## CCL and trace-state audit

`MultichipDecoder.from_state_dict()` creates and retains a separate
`CCLManager` for each layer (`multichip_decoder.py:585-594`). The stack retains
both decoders in `contexts`; their managers and semaphores are not temporary.
`models/demos/gpt_oss/tt/ccl.py:39-88` allocates separate RS, AG and barrier
pools. The declared ping-pong buffer cache is unused by this model path.

Each layer calls two allreduces per decode: attention `[1,1,1,2816]` and grouped
MoE `[1,2,1,2816]`. Each allreduce performs RS and AG, consuming one RS set,
one AG set and two barrier indices (`models/demos/gemma4/config.py:96-127`).
Thus each layer consumes set0 then set1 and returns its RS/AG indices to0;
its barrier index returns to0 after each allreduce. The trace records these
addresses; Python counters do not have to advance on replay. No semaphore
pool is shared accidentally between the two decoder instances.

The Linear RS cache-hit callback rewrites reader args0/1/2/3 and writer
args0/1/4/9: input, intermediate, output, ready semaphore and barrier addresses
(`reduce_scatter_minimal_async_program.cpp:1630-1676,2031-2063`). Its local
copy of `RuntimeArgsData` still points to the actual argument storage;
`tt_metal/api/tt-metalium/runtime_args_data.hpp:17-48` proves this is pointer
indirection, not a lost vector-copy mutation. The AG callback likewise patches
input/output/ready/barrier addresses
(`all_gather_async_default_program_factory.cpp:828-873`).

Both operations omit semaphore addresses from their compile-time attribute
hash intentionally and supply them through these callbacks. Tensor specs are
hashed, including dtype. BF16 and BF8 are not deliberately mapped to one
format-blind program key. A hash collision was not observed or inferred.

Trace capture snapshots runtime argument bytes and CB configuration for each
launch (`tt_metal/impl/program/dispatch.cpp:3334-3387`). Final command assembly
copies the saved per-node runtime arguments back (`:3122-3125`). Therefore the
specific story "the second layer overrides the cached program and silently
changes the first layer's captured addresses" is refuted at this interface.

No persistent CCL output is supplied. Temporary RS staging is released by
the wrapper, and the captured command stream can reuse temporary addresses;
that fact alone is not a use-after-free finding. The harness allocates token,
positions, caches, page tables and semaphores before capture, retains both
layer output handles, and creates only host tensors while refreshing inputs
after capture. No new device allocation in its replay loop was identified.

## Cache/interface audit

Each layer gets an independent cache pair, page table and two position tensors.
Cache geometry is taken from the actual local attention configuration:
sliding KV2/D256, full KV1/D512 on TP4; page32 and extent128. Positions33/34 are
inside page1, not page boundaries. Extent128 covers the rounded attention read
window. The CCL dtype override is downstream of the attention/cache producer.

Both layers use the fused update path. Its program-cache callback patches both
cache addresses, both dynamic update-buffer addresses, index-buffer address
and page-table address
(`paged_fused_update_cache_device_operation.cpp:47-128,372-414`). No missing
cross-layer cache binding was found. The stack does not validate logical cache
contents; if its earliest divergence is upstream of WO, compare logical page
readback before treating this inspection as runtime proof.

## Indexed expert merge audit

For current indexed TP decode, `OptimizedExperts._chunk()` selects8 experts,
gathers their routing weights and tilizes the gathered row, then computes
gate/up, GELU×up, sparse down, `permute(down,(0,2,1,3))`, and a final dense mix
(`optimized_decoder.py:193-240`). The sparse projection internals are being
audited separately. This pass checked the weight-selection and merge boundary.

The same selected ID tensor drives the gather and both sparse matmuls; the
compact slot count is8. Gather preserves that ID order. There is no evidence
that sorting or a separate expert permutation mispairs weights and outputs.
Both layers own separate routers and retained index handles.

| Merge operand | Logical shape | Physical tiled shape | Padding producer |
| --- | --- | --- | --- |
| Selected routing weights | `[1,1,1,8]` | `[1,1,32,32]` | `to_layout` with zero pad |
| Permuted expert down output | `[1,1,8,2816]` | `[1,1,32,2816]` | native HC transpose with zero pad |

The model **does supply zero padding through the called APIs**:

- RM gather output uses the index tensor's logical shape `[1,1,1,8]`
  (`gather_device_operation.cpp:127-170`). `to_layout` detects that its physical
  shape is not tile-aligned and calls `tilize_with_val_padding` with default0
  (`core/to_layout/to_layout_op.cpp:25-40,180-220`). The selected default reader
  explicitly fills row0 columns8:32 and rows1:32 before pushing the tile
  (`reader_unary_pad_dims_split_rows_multicore.cpp:52-101`). Its read barrier
  precedes the push. Consequently an extra model-level pad is not justified
  by a missing-call argument.
- Permute `(0,2,1,3)` canonically lowers to HC transpose with default `pad_value=0`
  (`permute.cpp:104-106,143-145`). The interleaved TILE factory sets C8/H1/W2816,
  `needs_padding=true`, and zero padding data
  (`transpose_hc_tiled_interleaved_program_factory.cpp:97-104,155-175`). The
  writer copies only the real H row and explicitly writes K rows8:32
  (`writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp:84-175,183-229`).
  Both payload and padding writes complete before their CB/DFB entries are
  popped. A pure-Python enumeration of these exact destination-byte formulas
  covers all180224 output bytes,88 BF16 tiles, exactly once with no payload/pad
  overlap. This refutes missing or overlapping padding in those formulas;
  it does not prove the runtime buffer contents are correct.

The final mix is HiFi4, BF16 inputs/output, `in0_block_w=1` and a single physical
K tile. Sliding uses FP32 destination accumulation; full does not. That is a
real policy distinction, but no proof of a faulty consumer follows from it.
Padding corruption or an earlier changing projection remains a testable
hypothesis. Zero-times-NaN is not a reason to skip checking the padded B rows.

## Minimal falsifiable controls

1. For the stack, preserve the captured graph and add host-only failure metrics:
   phase, layer, step, rank, nonfinite counts, changed elements and maximum
   finite delta. Do not retain additional device intermediates for this first
   check. This distinguishes true finite replica mismatch from replicated NaNs.
2. For the localized expert path, retain selected RM weights, tiled weights,
   gate/up, hidden, sparse down, permuted down and final mix in the frozen
   component probe. Compare the earliest changing logical **and physical**
   boundary before changing precision, padding or geometry.
3. A source-confirmed metadata-only physical read is available for TILE
   tensors: `p = ttnn.Shape(t.padded_shape); full = ttnn.reshape(t, p, p)`.
   Read each rank of `full` only after blocking replay. The explicit two-Shape
   overload takes the different-logical-volume/same-padded-width view branch
   (`reshape_view/reshape.cpp:626-635`); `tensor_ops.cpp:399-414` constructs a
   view of the same address with shared ownership. It neither launches a
   kernel nor fills padding. Do not substitute `experimental.view(t, tuple)`:
   that vector overload checks the original logical volume and rejects this
   expansion. This recipe was source-verified, not executed here.
4. Native CCL replica coverage remains worth closing independently. The
   unapplied `probe_attention_ccl_replay_stack_candidate.patch` and matching
   `.py.txt` add finite, cross-rank, oracle and exact-repeat checks. Optional
   `--stack-graph --input-memory l1` uses two managers and the four ordered
   collective shapes/dtypes listed above; inputs are independent frozen
   tensors, so this isolates CCL resource reuse rather than reproducing a
   whole model. `--integer-inputs` requires an exact sum oracle using small
   exactly representable values. Normal input mode reports quantization plus
   reduction error and requires PCC≥0.995 and relative L2≤0.05 in addition to
   exact replica/replay equality. These oracle tolerances are component smoke
   checks, not model acceptance criteria.

The native candidate SHA256 is
`2b5811bc7524131b1d29d88d372053288e098d0c9eeda8133e840803313bacf9`.
AST parsing and `git apply --check` passed. It has not been applied or executed
on hardware. No build was required for this report and unapplied Python probe.

## Status

Unresolved, with CCL state/alias hypotheses demoted by source inspection and
the separate sliding failure localized beyond CCL by the hardware owner's
v3 evidence. No BF8 limitation, performance claim, or verified runtime fix is
established by this report. The stack still needs its own first-failure metrics.
