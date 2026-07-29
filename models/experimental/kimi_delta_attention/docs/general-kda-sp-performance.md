# General KDA sequence-parallel performance design

## 1. Problem framing

Objective: improve Kimi-K3 `SP2×TP4` by implementing three reusable mechanisms:

1. a distributed affine-state algorithm valid for every `SP > 1` and every TP
   mesh width;
2. topology-neutral incremental matmul-to-reduce-scatter readiness, with no
   TP-count branches;
3. one fused local affine composition program valid for any supported group
   count.

### Facts

- The user explicitly requires general changes. Device-count or KDA
  configuration special cases require separate approval.
- `SP2×TP4` wall time is 14.605 ms versus 12.423 ms for `SP4×TP2`. The largest
  attributed penalties are distributed prefix (+1.971 ms), output MMRS
  (+1.510 ms), and local affine composition (+0.666 ms), derived from ten Tracy
  replays in the generated K3 report.
- The public distributed operation returns partition entry states and the
  replicated final state, not prefix transforms
  (`chunk_gated_delta_rule.cpp:890-1023`).
- The BF16/L1 state-relay implementation is mathematically valid for arbitrary
  `sp_size`, but is currently guarded by `sp_size == 4 && tp_size == 2`
  (`chunk_gated_delta_rule.cpp:946-991`).
- The fallback implementation materializes distributed Hillis--Steele affine
  prefixes (`chunk_gated_delta_rule.cpp:993-1021`). For SP2 it launches 24 P2P
  operations and costs 4.337 ms attributed.
- The normal MMRS path supports Ring and Linear topology, but its matmul writer
  globally synchronizes all producers before signaling reduce-scatter
  (`reader_bmm_tile_layout_in1_sender_writer_padding.cpp:682-684` and the
  receiver equivalent at lines 230-232).
- A separate strided MMRS implementation already models block-ready producer
  signaling, but currently validates Ring only
  (`minimal_matmul_strided_reduce_scatter_async_op.cpp:26-35`).
- Standard Ring and Linear reduce-scatter factories already exist and remain
  the source of truth for topology mechanics
  (`matmul_reduce_scatter_async_program_factory.cpp:83-111`).
- `KdaAffinePrefixOperation` is an existing one-program, arbitrary-`G`
  Hillis--Steele implementation with one worker per group
  (`chunk_gdn_phased_program_factory.cpp:699-766`,
  `kda_affine_prefix.cpp:82-142`). It currently emits entry states, not the
  final composed transform.

### Inferred hypotheses

- Replacing the distributed transform-prefix fallback with the already
  validated state relay for every `SP > 1` will reduce launch and transport
  work at SP2. Large-SP latency must be measured because state relay has linear
  dependency depth.
- Releasing MMRS output by logical width stripe, after all M producers for that
  stripe finish, will overlap compute and communication without exposing
  partially written tiles.
- Extending the existing fused affine program to emit the final `(A, B)` will
  replace the SP2 local composition's 24 TTNN launches with one program.

### Constraints and invariants

- No condition on `sp_size`, `tp_size`, mesh shape, model name, or sequence
  length may select a new algorithm or program.
- Existing Ring-versus-Linear program factories may remain distinct because
  topology is a domain property, not a model/device-count special case.
- Public distributed-prefix inputs and FP32 outputs remain unchanged. BF16 is
  an internal compute/transport representation and must satisfy the existing
  PCC contract.
- MMRS output becomes visible to a consumer only after every producer that owns
  the corresponding logical stripe has completed its writes.
- Program-cache reuse, eager repeat, and trace replay must remain correct.
- Retain a change only after focused correctness plus ten-replay real-weight
  SP2×TP4 and SP4×TP2 measurements. Attributed time is diagnostic; acceptance
  uses slowest-device wall time.

### Non-goals

- A TP2/TP4 policy table, model-specific program configuration, or hidden
  environment switch.
- Lower-precision MMRS transport in this change set.
- Changing collective topology or public tensor layout.
- Claiming Galaxy performance from LoudBox.

## 2. Workflow

1. Compose local group summaries into one partition transform.
2. Starting from the replicated initial state, apply each partition transform
   and relay its exit state to the next SP rank on every TP line.
3. Preserve the received state on each rank as that partition's entry state.
4. Multicast the last partition's exit state along every SP line as the final
   state.
5. Apply local group transforms to the partition entry state using the fused
   affine program.
6. During output projection, publish completed logical N stripes from matmul
   producers and let the existing Ring or Linear RS consumer start a stripe as
   soon as its readiness dependency is satisfied.

## 3. Data flow

```text
group summaries (A_g, B_g)
        │
        ▼
fused local compose ──► partition (A_p, B_p)
        │
        ▼
general SP state relay ──► partition entry states + replicated final state
        │
        ▼
fused local apply ──► group entry states ──► final grouped scan

output activation × output weight
        │
        ▼
2-D matmul producers ── completed logical N stripes ──► Ring/Linear RS
```

MMRS readiness is cumulative and monotonic. A stripe-ready value describes
logical output coverage, not a physical core count or a TP rank.

## 4. Domain model

- **Affine transform:** `(A, B)` such that `S_out = A @ S_in + B`.
- **Partition transform:** composition of all local group transforms.
- **Partition entry/exit state:** state immediately before/after a sequence
  partition.
- **Output stripe:** contiguous width tiles of the logical matmul result that
  form the smallest producer/RS dependency unit.
- **Readiness frontier:** cumulative set of output stripes safe for RS to read.
- **Collective topology:** Ring or Linear routing semantics, independent of
  participant count.

## 5. Architecture

### General distributed affine state

Keep the public `kda_distributed_affine_prefix` API. Remove schedule selection:
the function always uses one state-relay workflow for `SP > 1`. TP lines are
enumerated from mesh geometry; the same loop works for any width. Keep the
existing public `identity_a` and `zero_b` arguments for compatibility, validate
them, and mark them as compatibility inputs until a separately approved API
deprecation.

### Reusable affine scan primitive

Keep one `KdaAffinePrefixOperation` implementation and expose two narrow
internal wrappers:

- `kda_affine_compose(A, B, G) -> (final_A, final_B)`
- `kda_affine_apply(A, B, initial, G) -> entry_states`

An internal result mode controls only which already-computed values are written;
the scan algorithm and group mapping are shared. No mode depends on `G`.

### Generic MMRS readiness

Extend the existing CCL fusion contract rather than adding a KDA MMRS:

1. Replace the single whole-matmul completion event with a logical
   output-stripe readiness descriptor.
2. The matmul factory derives producer membership for each stripe from its
   ordinary `M_block`/`N_block` mapping.
3. Producer writers advance a cumulative readiness semaphore only after all M
   blocks for the stripe are complete.
4. Ring and Linear RS factories translate their existing chunk iteration into
   readiness targets and wait immediately before the first read of each stripe.
5. A final full-producer completion remains as a debug/runtime assertion, not
   the start condition.

The readiness descriptor belongs in the CCL fusion layer and is reusable by
both the regular and minimal MMRS program factories. Topology-specific routing
stays inside the existing Ring/Linear RS factories.

Dependency direction remains:

`model helper -> public TTNN operation -> generic fused-op contract -> topology program factory -> kernels`.

## 6. Alternatives

### A. Recommended: promote existing mechanisms into shared primitives

- General state relay for all SP.
- Add compose mode to the existing KDA affine program.
- Add stripe readiness to the existing MMRS fusion contract.

This minimizes duplicate algorithms and contains topology details in existing
factories. MMRS is the largest implementation risk, but the abstraction matches
the data dependency.

### B. Route KDA through minimal strided MMRS

The minimal operation already signals blocks incrementally. It currently
supports only Ring, while LoudBox TP axes normalize to Linear. Adding a second
Linear strided collective would duplicate substantial standard RS machinery
and split future MMRS tuning between two public operations. Rejected.

### C. Keep whole-output MMRS signaling and tune grids/buffers

This is simpler but does not address the proven global producer barrier.
Previous worker/buffer sweeps did not produce a material long-span win.
Rejected as the primary design; ordinary tuning remains available after
streaming is measured.

## 7. Risks and open questions

**Hardest-to-change decision:** define readiness in logical output stripes,
rather than physical producer cores. This determines the fused-op API and must
remain stable as matmul grids and collective worker layouts change.

Risks:

- State relay is linear in SP depth. It is expected to win at SP2/SP4 but may
  lose to transform composition at large SP. The design intentionally avoids a
  hidden threshold; if measurements later justify multiple algorithms, adding
  an explicit caller-selected policy requires user approval.
- BF16 internal state propagation introduces sequence-length/SP-depth-dependent
  numerical error. Test SP2 and SP4 at K3 shape plus generic smaller SP cases.
- Stripe readiness can deadlock or expose incomplete data if matmul block order
  and RS chunk order disagree. Add host-side mapping validation and kernel
  assertions before performance tuning.
- The affine compose mode must handle non-power-of-two `G`; test `G=1,2,3,4,5,8`.

Open questions:

1. Should the generic state relay remain BF16 internally, as the validated SP4
   path does, or should internal dtype become an explicit public operation
   attribute? Recommendation: keep BF16 internal for this KDA-specific API and
   retain FP32 public inputs/outputs.
2. Should MMRS stripe width be fixed by RS `chunks_per_sync`, or derived as the
   least common coverage unit of matmul N blocks and RS chunks?
   Recommendation: derive the least common coverage unit and expose no tuning
   knob until hardware evidence requires one.

## Validation and commit gates

1. **General state relay**
   - Distributed-prefix serial/all-gather oracle, both mesh axes, repeat, trace.
   - SP layer one-shot/chunked equivalence.
   - Real K3 SP2×TP4 and SP4×TP2 ten-replay profiles.
2. **Fused local compose**
   - CPU affine oracle for `G=1,2,3,4,5,8`.
   - Same full-layer correctness and profiles.
3. **Generic MMRS streaming**
   - Existing Ring and Linear MMRS tests across at least 2 and 4 participants.
   - Trace/cache reuse and persistent-buffer ownership tests.
   - K3 real-weight PCC plus SP2×TP4/SP4×TP2 ten-replay profiles.

Each item is a separate validated commit. A change that is correct but does not
improve slowest-device wall time is rejected or retained only as a documented
general infrastructure prerequisite with explicit approval.
