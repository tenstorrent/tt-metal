# Attention gate integration audit 24

2026-11-13. Source-only integration proposal; no runtime source changes, device
opens, TTNN imports, or builds. Parent reports the focused B1/B3 smoke passed
exact native comparison, watcher, allocation tracking, changed inputs and 50
queued replays. Wide logical-head6/32 coverage and uninstrumented timing are in
the parent's queue; this audit makes no claim that those or whole-layer gates
have passed. See `attention_gate_audit.md` and `attention_gate_smoke_v1.json`.

## Minimal hook and admission

Put the opt-in branch in `tt/multichip_decoder.py::_local_compact_attention_decode`,
immediately after `paged_scaled_dot_product_attention_decode` returns (currently
line780), replacing only the head conversion/concat/compact reshape/gate multiply
sequence. Keep `self._linear(out, "self_attn.o_proj.weight")` and both return
branches common and unchanged. In particular, preserve the DRAM conversion
before expanding compact output to public `[B,1,5120]`: the existing comment
records a live-L1/reduce-scatter collision for one-token prefill/resume.

Suggested policy: `attention_gate_mode="off" | "dram" | "l1"`, default `off`;
`attention_gate_cores=8 | 48`; `attention_gate_scope="decode" | "compact"`,
default `decode`. Explicit invalid mode/scope/worker selections should fail at
host setup. Do not silently fall back from an explicitly requested L1 experiment.
The existing off branch must preserve current operations and arithmetic.

Pseudocode (not an applied edit):

```python
use_gate = mode != "off" and (decode or scope == "compact")
if use_gate:
    gate_input = gate if packed_gate else self._linear(xf, "gate")
    destination = ttnn.DRAM_MEMORY_CONFIG
    if mode == "l1":
        destination = self.projection_info["self_attn.o_proj.weight"]["input_mem"]
        assert destination == canonical_output_memory(self.mesh)
        assert attention_activation_dtype == "bfloat16"
        assert o_proj_dispatches_to_the_existing_dram_sharded_projection
    out = attention_gate(out, gate_input, memory_config=destination,
                         cores=selected_workers, output=prepared_output_if_used)
else:
    # Existing height-shard -> concat -> DRAM -> compact -> sigmoid multiply.
    ...
out = self._linear(out, "self_attn.o_proj.weight")
# Existing return/DRAM expansion logic follows unchanged.
```

Use resolved constructor policy/configuration for the assertions, not undefined
symbols from this pseudocode. `attention_gate` already checks Blackhole, same
mesh, BF16 TILE DRAM input, logical heads6 or32, physical `[1,B,32,256]`, gate
logical `[1,B,1536]`/physical `[1,32,1536]`, and B1..32. No public batch alignment
restriction, padding upload or slicing should be added by integration. Native
SDPA output normally has six logical heads; a logical32 conversion is unnecessary.

The actual L1 destination is `projection_info["self_attn.o_proj.weight"]`
`["input_mem"]`, not a guessed grid. `_prepare_matmuls` (lines320–360) resolves
per-projection geometry overrides before creating this config. Default eight
storage cores give width shards `[32,192]` on the first eight row-major workers.
The helper currently supports exactly that config. Its eight/48 compute workers
are independent of those eight storage cores. Noncanonical projection geometry
requires the DRAM family or a separately adapted direct-output helper; it is not
evidence against the current direct-L1 family.

Also verify the actual `_linear` dispatch (lines433–449). Ordinarily B<=32 skips
fused matmul/reduce-scatter and inherits `OptimizedDecoder._linear_raw`
(lines499–538), which casts only if activation dtype differs and then calls
`to_memory_config(x, info["input_mem"])`. Equal configs with no output tensor
return the input unchanged (`to_memory_config_op.cpp:253–265`). Thus exact BF16
config equality removes the restore before `o_proj`. Recompute the real
`fused_rows` condition, including `mmrs_min_rows`/`mmrs_max_rows` and
`fused_mmrs_decode`, rather than checking the last flag alone. A custom range can
select fusion even with that flag false. Fused `_linear` first restores DRAM;
measure it only as a distinct family, not as proof that direct L1 is unhelpful.
If o_proj has a correction projection, inspect both input configs/dispatches;
initial integration can admit the ordinary uncorrected attention projection only.
No precision policy, fidelity, weight, KV dtype or collective is changed here.

## Branch and phase coverage

Both QK implementations converge at exactly this SDPA output. With
`packed_compact_qk_rope=True`, `packed_qk_rope` returns an owning compact DRAM
gate; use it directly. With the split/fused QK implementation, `packed_qkvg=True`
obtains gate from `projected[:,:,2048:]`; preserve that existing result. With
`packed_qkvg=False`, evaluate the existing separate gate projection exactly once
inside the selected boundary. This has no dependency on the QK normalization or
RoPE choice. Do not move projections or alter cache-update ordering to integrate
this candidate. Keep the existing deletion of K/V references after cache update.

Despite its name, `_local_compact_attention_decode` also serves one-token
prefill/resume. `OptimizedDecoder.prefill_forward` (lines1179–1197) uses this path
with `decode=False` for an unaligned absolute position. Actual decode passes
`decode=True`, through both direct compact and public-layout dispatches. Scope
must use that boolean, not tensor shape, B or sequence length. Start decode-only;
explicit compact scope can subsequently include resume with the same validated
BF16 boundary. Aligned prefill chunks remain in `_attention_prefill`. Neither
scope changes the inter-layer hidden-sharded residual contract or adds a CCL.

## Ownership, trace preparation and live L1

The helper is a stateless function, uses program-local CBs, and supports an
explicit output. It reads both inputs, owns every output tile, zeroes padded
users and rejects input/output address aliasing. Do not pass gate, SDPA output,
or a view of another live tensor as its destination. Do not manually deallocate
the split gate/projected backing tensor: ownership/alias behavior differs between
the packed-direct and split branches, and trace may retain the allocation.

Two explicit lifecycle choices exist:

- Minimal first integration: `output=None` uses ordinary TTNN temporary output
  allocation. Warm and capture the whole layer; require replay allocation tracking
  to prove stable captured operation. This does not claim explicit output
  allocation before capture. It avoids introducing persistent per-owner L1.
- Explicit preallocation family: create a fresh owner-local dictionary in the
  constructor, prepare the output during warm/setup before capture, and pass it
  to the helper. Key by logical batch, destination config and mesh/owner identity;
  keep the Tensor object alive through trace release. Shape metadata differs
  even though every batch has the same physical output size. Never allocate a
  missing entry inside a supposedly preallocated capture. One-CQ serial reuse is
  valid only while the subsequent `o_proj` has consumed it before the next write;
  do not claim multi-CQ/concurrent-owner safety from serial replay evidence.

Persistent state is not a one-line safe extension: the 64-owner resource harness
(`tests/check_multichip_stack_resources.py:186–188`) uses `copy.copy(original)`
and explicitly resets only known mutable dictionaries. A new output dictionary
would otherwise be shared between owners. Add its independent reset and per-owner
address assertion beside the existing CCL checks (lines323–341) before claiming
persistent ownership is clean. The model has16 attention owners. Each cached
output is98,304 bytes/device,12,288 bytes per storage core; one batch signature
across16 owners consumes1,572,864 bytes/device or196,608 bytes on each of those
eight cores. Multiple signatures multiply this cost. Do not casually retain all
32 batch signatures, or assume cached outputs do not affect context capacity.

The helper adds16,448 bytes of CB storage per active compute worker while running.
Direct L1 also keeps its98,304-byte output live through the following matmul.
Program-local CBs are not persistent across that sequential operation, but output,
other live activations, persistent communication buffers and matmul CBs must fit
together. Worker48 and worker8 have different CB footprints. Test the actual
whole-layer L1 policy, not just this isolated helper. The existing post-o_proj
DRAM conversion for resume remains essential. Do not remove it as part of this
fusion. If persistent output memory materially changes feasible context, use
physical measurements and update the context contract; a source byte count alone
is not proof that maximum context shrinks.

## Smallest useful whole-layer evidence ladder

First finish the already queued standalone wide/trace/poison and timing runs.
Choose worker count from uninstrumented complete-boundary timing; retain off,
direct DRAM plus existing shard, and direct canonical L1 as coherent families.
Whole-layer timing must include `o_proj` and downstream finish, so a standalone
win cannot hide L1 interference or a later restore.

1. On the current selected full-attention policy, compare all three modes at
   B1/S512 and B32/S33 (six runs), with identical prefill and decode lengths,
   inputs, seeds and frozen baseline. Report warmed prefill and traced warmed
   decode, state/output PCC, trace tracking and dispatch choice. Use unchanged
   accepted gates. Since decode-only scope leaves prefill unchanged, any prefill
   numerical drift needs investigation.
2. Cross the other QK family (split+fused QK/RoPE versus direct packed QKVG) at
   B1/S512 and B32/S33 using off and the winning gate mode (four runs). Keep all
   unrelated flags fixed. If the families interact, expand the three-mode
   comparison there before selection. Add B32/S513 on the selected combination
   to cover a different cache/context boundary and the actual text fixture.
3. Exercise selected integration at nonaligned B3 and B17 and changing positions;
   include existing direct-compact and public-layout decode dispatch. The
   standalone all-B exact tests cover address mapping but do not replace these
   whole-layer gates. If enabling compact/resume scope, compare against off at
   unaligned resume positions31 and33 with actual HF-contract tests as well as
   the accepted frozen-TTNN reference; preserve logical output slicing and state.
4. Extend selected-path long decode and multi-owner/stack stress with independent
   states, changed inputs, queued traces, output buffer/address tracking and
   watcher. If adding persistent outputs, include all16 attention owners and
   assert their mutable buffers are disjoint. The frozen TTNN baseline is not an
   HF oracle; report those provenance/quality gates separately.
5. Profile the final selected path to show head-height conversion, native concat,
   compact materialization and sigmoid-multiply kernels disappeared, and that
   the `o_proj` input conversion is a no-op in direct-L1 mode. Produce before/after
   CSV/provenance and final default-policy timings. Keep watcher/profiler/tracker
   out of performance runs. Existing final stage gates and stop rule still apply.

No main integration, full-model work, default selection, review verdict or stage
completion follows from this source-only proposal.
