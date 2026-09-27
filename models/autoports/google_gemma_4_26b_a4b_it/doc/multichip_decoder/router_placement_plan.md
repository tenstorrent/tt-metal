# Selected router placement candidate

The current TP4 runtime places the one-core generalized top8 gate at logical
Tensix(10,9), outside the expert and fused-tail normalization compute grids.
This is not a globally reserved core: generic layout operations can use it.
Four persistent buffers move at construction, preserving their dtypes and
HEIGHT_SHARDED L1 [32,32] ROW_MAJOR contract. Active8 execution and runtime
interfaces remain unchanged; no host conversion enters forward or trace.

Full-attention BF8 controls on core(1,0) localized finite replica divergence to
the first tail RMSNorm despite bit-identical physical inputs and weights.
Moving only the gate to(10,9) passed paired output/cache checks, retained I/O
checks and1024 duplicate comparisons. Restoring(1,0) failed, including a
roundtrip allocation control where all four buffer addresses changed. This
rules out reallocation alone as sufficient. Exact native mechanism remains
unproven; source audit did not establish a missing state reset.

Evidence: AUTOFIX_full_router1_bfp8.md, AUTODEBUG_router_norm_state.md,
full_bfp8_router109_stress1024.json,
full_bfp8_router1_roundtrip_control.failure.json.

Integrated runtime SHA256:
f57234ba01a6a6e20e3188ea7690f8508c5a5ca6c075b1fba56bf1b6df28a44b.
Ordinary full-attention4096/128 trace/cache run passes: minimum output PCC
0.9994477563467595, minimum cache PCC0.9999715022296233,
all replicas and duplicate replays exact. Artifact:
full_router109_integrated_bfp8.json. Remaining final gates are pending.

The preceding core(1,0) investigation is retained below as historical evidence.

# Router placement investigation: historical core(1,0)

The TP4 decoder places the generalized top8 gate on logical Tensix(1,0), away
from the indexed expert gate/up projection's input-multicast sender(0,0).
Its input, bias, index table, score output and index output all retain one-core
HEIGHT_SHARDED L1 storage with shard[32,32], row-major shard orientation, and
unchanged tensor dtypes/layouts. Four persistent buffers move once at setup;
forward contains no host conversion or setup copy. Expert weights, active8
selection, projection geometry, precision, collectives and public layouts are
unchanged. OptimizedDecoder remains the unmodified single-chip baseline.

This is an experimentally verified placement workaround, not a claimed native
kernel root-cause fix. Under the original placement, real4096/128 sliding BF8
attention runs fail duplicate replay. Changes localize to the sparse sender's
output columns; increasing its N ownership from32 to64 columns expands the
changed region accordingly. K88,40ms replay delay and L1→DRAM expert input do
not remove it. Native no-inline Watcher passes, showing instrumentation
sensitivity rather than clearing release-mode behavior.

Moving only the gate and its buffers to(1,0) passes original-class128 steps,
boundary/physical-padding checks, and128 changing positions×8 duplicate replay
checks=1024 comparisons. Restoring(0,0) reproduces at step62. Exact artifacts:
`bfp8_output_router_core1.json`, `bfp8_boundary_router_core1_slices.json`,
`bfp8_output_router_core1_stress.json`, `bfp8_output_core0_postcontrol.json`.

Frozen exact-physical experts preceded by the actual native gate pass at both
cores, with identical allocated prefix maps and outputs. Thus no simple
standalone gate-state bug is established; the full surrounding layer matters.
Source review found no missing required startup reset or sender barrier.
`AUTODEBUG_indexed_expert.md`, `AUTODEBUG_expert_buffer_lifetime.md`, and
`AUTOFIX_bfp8_ccl.md` retain the evidence and unresolved low-level mechanism.

The model-local integration has runtime SHA256
`070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`.
It moves buffers immediately after router construction; diagnostic controls
moved them after complete decoder construction. Therefore ordinary paired
PCC/cache/replay, both layer kinds, stack/batch/context and final Watcher checks
remain required for the integrated path. Current acceptance status: pending.
