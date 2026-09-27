# Inter-layer contract

The selected decoder emits and consumes replicated BF16, TILE-layout,
DRAM-interleaved hidden states on the same1x4 mesh. Prefill shape is
`[1,1,S,2816]`; logicalS need not be aligned. Decode uses the existing logical
batch contract, with batch1 represented as`[1,1,1,2816]` and tile-padded storage.
Larger batches retain per-request positions, page-table rows and cache slots.
Padding is an internal implementation detail, never extra active requests.

Pass one decoder's output directly to the next decoder. Do not insert a gather,
reduce-scatter, all-reduce or reshard at this boundary. The local input norm
owns its L1 sharding; the final fused tail owns its BF16 DRAM output. There are
no inter-layer collectives in the selected path. `stack_adjacent_mixed.json` and `stack_adjacent_mixed_33.json` verify the
actual adjacent layer4-to-layer5 handoff with128 advancing traced decodes;
`stack_selected_samekind.json` and its33-token counterpart cover layers0-to-1.
Layer4 fixtures are recomputed through HF layers0–3 for each logical prefix;
nonadjacent0-to-5 diagnostics are not acceptance evidence.

Within a layer, attention is tensor-parallel over local heads, decode experts
execute gate-selected top8 with indexed sparse matmul, and prefill uses active
EP4 expert unions. The residual replicas do not imply replicated layer compute.
Attention output uses local full-H WO followed by Linear RS/AG. Shared/routed
outputs are paired for one RS/AG, then normalized separately. These collectives
belong inside the layer's mathematical reductions.

The selected capacity contract shares decode RS intermediate/output and AG output
storage using one `CollectiveBufferPool(mesh)` passed to the decoder instances.
Keys separate attention from paired MoE roles, logical and padded shape, dtype,
and memory configuration. Semaphores remain private to each decoder. This pool
requires serial execution on the same command queue and mesh; it must never be
shared by concurrent requests, threads or command queues. Keep it alive through
all captured traces. Warm every layer signature before capture. The intervening
other-role collective orders consumption before reuse of a role's buffers, and
the final layer output is a fresh DRAM tensor. Source proof and limitations are
in `AUTODEBUG_ccl_pool.md`; shared/private bitwise reuse controls are in `pool_equivalence.json`.
Each private semaphore set covers the actual 11x10 worker grid; see
`AUTODEBUG_ccl_semaphore_grid.md`.

The union of selected role buffers uses 2,690,688 bytes/device. Private buffers
for every layer are not the full-stack capacity contract: an earlier 37,614,720
byte L1 reservation collided with prefill circular buffers. Capacity validation
must prime the actual shared pool before prefill rather than merely count bytes.
Input/position/page-table updates remain outside capture; traced layer execution
has no host tensor fallback.

Candidate mesh-sharded residuals carry localH704 through distributed norms and
residual updates; their harness-only gather is excluded from measured layer
latency. Candidate local four-core L1 residuals also carry their layout across
the boundary. Compatible fused QKV/WO/RS, reduced payload and persistent-buffer
comparisons are in `candidate_comparisons.md`. These alternatives were measured
slower for the decode target, so full-model bringup should preserve the selected
replicated boundary rather than rediscovering those transitions.
