# Bounded prefill executor integration

2026-11-14: implementation and CPU/static validation only. Hardware execution belongs to the root device owner. This document is not acceptance evidence.

Files added:

- `tt/prefill.py`: `BoundedPrefill`, layer-major fixed-capacity workspaces, host-only integer planner, bounded embedding/terminal, selected attention/MoE calls and universal component preparation.
- `tt/prefill_transport.py`: replicated BF16 row transport and INT32 fill-page windows, runtime device metadata, eager and device-only entrypoints.
- `tests/run_prefill_transport_probe.py`: exact-bit transport/page checks, zero padding, unchanged program-cache counts under changed offsets, captured runtime metadata and eager work between retained replays.

The root added the required narrow decoder hook: `positions['fill_pages']` selects the persistent fill view and tensor-valued SDPA start. Existing scalar-offset callers remain available for paired validation. The executor does not copy the full-attention algorithm.

## Integration sequence

1. Construct `BoundedPrefill(model)` before any model or sampler trace capture. Its three `[1,round_up(context,32),2048]` ROW_MAJOR BF16 allocations use exactly `workspace_bytes_per_chip` bytes per chip (3GiB at context262144); add a model census after construction. Weights, KV, recurrent, CCL, helpers, sampler and transient peak remain additional capacity costs.
2. In the generator-wide preparation phase, call `executor.prepare(cache=generator.cache, reset_state=generator.reset, progress=callback)`. The callback is optional; reset_state is required and must restore all warmup-touched request/KV/conv/recurrent/RNG state. Bind active slot predicates appropriately for warmup. The method deduplicates layer component configurations, prepares expert output capacities first, then warms attention, MoE, transport, position and terminal variants. Capture decode/sampling only after this and all remaining generator variants finish.
3. Pass the valid host token slice for the current public prefill call to `executor.prefill(ids, cache=cache, slot=slot, start=start)`. `start` is the absolute initial position; `ids` contains only the tokens newly being filled. The method returns a device logits tensor for the final valid token. It preserves one recurrent-state BF16 boundary round per layer/public call, linear chunks64 relative to that call, full-attention256 alignment/singleton boundaries, projection flags, and MoE2048 batches.
4. For explicit full-logits compatibility, provide `on_logits(offset, valid_rows, device_logits)`. Each device tensor has32 rows; only the first valid_rows are public output. The callback must synchronously consume the chunk, ordinarily via the explicit host-logit compatibility boundary, and must not replay decode. The method also returns separate last-row logits. Avoid retaining all device chunks. Normal token-out passes no callback.
5. Copy/consume the returned prefill logits into the generator's already prepared sampler boundary, release temporary prefill logits, and only then replay traces. Generator owns cache allocations, slot activity, public positions, prompt lengths and trace lifetimes; the executor only validates configured cache geometry and the requested slot/context.

## Why the MoE tail inventory is bounded

The accepted `_moe` already pads every nonsingleton logical tail to the next multiple of32 **after** per-row normalization, and routes these dummy rows. The executor zero-pads to that same extent **before** normalization and commits only the valid result rows with transport. It does not pad a small tail to2048 or mask routes away.

For a nonsingleton tail, the selected norm reduces hidden dimensions independently for each row; the linear norm's `NormTerminalScan` also scans each row independently. A zero row normalizes to zero. Router/expert/shared math has no cross-token reduction except routing counts, and the routed dummy rows/counts/dispatch capacity are the same as the already accepted padding. CCL reduction and residual add are rowwise. Thus live-row arithmetic and route load are preserved while scalar tail pad/crop programs disappear. Public/suffix singleton remains the distinct sparse path. There are64 grouped physical token shapes32..2048, one sparse singleton, and seven dispatch-capacity buckets at full context.

This is a source-derived equivalence argument, not measured bitwise evidence. Before acceptance, compare valid outputs of the original exact-tail expression and executor padded expression for n=2,31,33,65,2047 on both attention kinds, using the same input and weights. Do not claim that comparison already passed.

## Validation performed here

- `python -m py_compile` on both implementation files and the transport probe: passed.
- Pure AST extraction compared `attention_chunks` to the inherited `FusedDecoder.prepare_prefill` over both kinds, all256 start residues, and21 boundary lengths:10,752 cases passed, with no TTNN/model runtime import.
- Read the current shared `slot_state._program`: its empty compute kernel includes `api/compute/common.h`; the transport reuses this corrected builder.
- No device access, compilation of device kernels, full-model execution or performance measurement.

First root-owned hardware command (serialized, with the normal environment/device protocol):

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 timeout 1800 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_prefill_transport_probe
```

Then perform the reduced transport/position/MoE paired probes and reduced real-model prefill before integrating acceptance. The tracker probe ends with a replay after eager changed-offset operations. Its output is `doc/full_model/prefill_transport_probe/result.json`; no result exists from this authoring task.
