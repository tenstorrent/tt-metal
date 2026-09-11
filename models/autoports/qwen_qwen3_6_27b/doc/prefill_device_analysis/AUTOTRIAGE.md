# AUTOTRIAGE

Follow-up: [AUTOFIX.md](AUTOFIX.md) records the passing reduced S128 ownership
experiment. The diagnosis below preserves the original evidence and uncertainty.

## Diagnosis

The experimental `cached_recurrence` keeps a trace alive while surrounding eager
work allocates persistent buffers and live activations in memory that the trace
can reuse. Its input/output copies protect tensor ownership at the API boundary,
but do not reserve the captured recurrence's freed intermediate storage.
The allocation tracker confirms this lifetime-contract violation; the specific
buffer overlap that caused the reshape hang remains a source-backed hypothesis.

Scope: real Qwen3.8 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, TP4,
batch 1, 12 local value heads, 32-token recurrence chunks. Inspection only by
this agent; device runs, triage capture, recovery and tracker verification were
performed by the coordinating agent. Production source is unchanged.

## Triage Evidence

- `/tmp/qwen_trace_triage.txt` names operation 2085,
  `ReshapeViewDeviceOperation`, input logical shape `[384,1,128]`, tiled BF16,
  interleaved DRAM. Previous operation 2084 is the BFP8 `[1,12,128,128]`
  recurrent-cache copy. The active reshape spans 48 workers on each of devices
  0–3, 192 workers total. Here 384 is **12 heads × 32 tokens**, not batch 32.
- A host-only CSV parse of `/tmp/qwen_trace_triage.log` finds 192
  `reader_reshape_tiled` NCRISCs, all waiting in `cb_reserve_back` at
  `reader_reshape_tiled.cpp:61`. All 192 corresponding writers have unresolved
  stacks: PCs `0xf7e0` (101), `0xf814` (71), `0xf850` (20). Missing ELF coverage
  does not prove an assert, corrupt binary or a particular writer stop-site.
- Other stacks include dispatch/prefetch and 32 fabric routers. The running
  operation and uniform local-reader waits identify reshape as the first useful
  source boundary; these stacks do not establish a fabric root cause.
- `/tmp/qwen_trace_debug.log`: the first S32 recurrence output and final state
  match eager exactly. On the next prefill their relative L2 differences are
  455.23 and 223.76, followed by the reshape hang. The runtime first warns at
  12:13:14.361 that allocations made with an active trace may be corrupted.
- `/tmp/qwen_trace_tracker.log`: with startup allocation tracking and
  tracebacks enabled, `execute_trace` rejects **three live post-capture buffers
  before the first replay**. IDs 8931 and 8941 have BF16
  `program_cache: CopyDeviceOperation` contexts; 8951 has BFP8. All allocation
  stacks end at `probe.py:175`, `ttnn.copy(source, destination)`. These C++
  program-owned buffers have no Python tensor referrer. The run exits cleanly.
  This proves the post-capture allocation exposure, not exact address overlap:
  the tracker conservatively records allocation lifetimes, not replay ranges.
- `artifacts/traced_s128.json` reports linear-layer median 20.651 ms against
  51.119 ms baseline, but layer hidden values differ by up to `3.78e25`, logits
  top-1 differs and relative L2 overflows. This is a rejected correctness result,
  not an accepted performance improvement. Its recurrent cache is exactly equal
  to baseline, which is compatible with collateral activation corruption.

## Source Evidence

1. `probe.py:159` captures on the first recurrence call. Only its six input
   clones and two final output tensors stay alive in `recurrence_traces`.
   `_sequential_recurrence` in `tt/functional_decoder.py:280` explicitly
   deallocates intermediate state, products and per-token outputs; other
   temporaries lose their last Python reference. Later eager allocations can
   reuse those addresses even though replay still writes there.
2. The input-copy variants are first invoked **after** capture at `probe.py:175`.
   Warming the recurrence alone does not warm those copy programs. The tracker
   directly identifies their persistent allocations as the first exposure.
3. `tt_metal/impl/allocator/trace_allocation_tracker.cpp:79` registers an active
   trace and tracks subsequent allocations. It does not reserve freed capture
   scratch. `allocator.cpp:122` emits the warning seen in the failing run.
   `trace_region_size` reserves trace-command storage; it does not solve
   recurrence-intermediate ownership. No suitable public Python scratch-reserve
   API was found in the inspected trace/allocator interfaces.
4. `tt/multichip_decoder.py:1154` copies the final state to the cache, then its
   tail at line 1222 reshapes `[12,32,1,128]` into `[1,12,32,128]`. The tiled
   reshape lowers that to input `[384,1,128]`, matching triage exactly.
5. `reshape_tiled_program_factory.cpp:300` creates and uploads a DRAM segment
   mapping tensor on a program-cache miss. Lines 309–316 move its ownership into
   the cached program. The first tail execution is after recurrence capture,
   exposing this long-lived map to later replays. Every segment stores an input
   page index, input offset, output offset and length. Corrupt map contents can
   therefore corrupt control/data movement, beyond corrupting numeric values.

The reshape producer/consumer ledger is balanced for valid metadata:

| Resource | Producer | Consumer | Expected work per worker |
| --- | --- | --- | --- |
| Mapping FIFO, capacity 1 | Reader loads and pushes one DRAM map page | Writer waits, processes segments, pops | One output-page map |
| Input-tile FIFO, capacity 1 | Reader loads each new input-page index from the map | Writer releases previous tile on page transition and releases the final tile | 32 input tiles for the inferred `[384,1,128]` → `[12,32,128]` mapping |
| Working scratch, one BF16 tile | Writer assembles segments in 2048-byte L1 scratch | Writer writes one output tile to DRAM | One output tile |

The output has 12 × 1 × 4 = 48 tiles/device, matching the 48 active workers.
Each output tile gathers 32 one-row input tiles. The reader waits for the writer
to free its single input-tile slot. Writer lines 58–61 use mapping offsets and
length directly in `tt_memmove`; bad metadata could produce an invalid local
copy and stop consumption, explaining the reader fanout. This last link needs
metadata/address evidence. The writer's final input-tile pop is **already
present** at lines 71–76; adding another pop is not a supported fix.

## Downstream Effects

The copied trace inputs and captured outputs can remain correct while replay
overwrites another eager tensor, such as Z, an outer residual, or a cached
program buffer. That explains why exact recurrent-cache equality does not clear
this candidate. The reshape mapping is a plausible later victim because its
first allocation occurs after capture and its contents govern local copies.

`TRACE_DEBUG` is not yet an uncontaminated recurrence oracle: it computes eager
`expected = recurrence(*values, **kwargs)` **after** trace replay. On later calls,
the original source tensors themselves were allocated after capture and may
already be overwritten. Copying them to stable trace inputs before replay does
not protect the originals. A before/after source comparison is required before
attributing this discrepancy to the captured arithmetic.

## Proposed Fix

Keep changes within the experimental probe until the following narrow controls
pass:

1. Warm all six input-copy signatures before capture. Warm the entire eager
   prefill once before any cached recurrence trace is created, so tail mapping,
   projection and CCL persistent buffers also predate capture. This removes
   specific lazy-allocation exposure but alone does not protect later live
   eager activations from recurrence scratch reuse.
2. Test ownership of capture intermediates with `trace_keepalive.py` in this
   directory. Wrap only capture in `preserve_trace_tensors()` and retain its
   yielded list alongside the trace until after `release_trace`. The helper
   retains Python-visible op outputs and intercepts explicit Python
   `ttnn.deallocate`; it restores both the function and runtime mode afterwards.
   Fast runtime must be disabled inside that context because its default
   operation wrapper bypasses post-operation hooks. This is an unverified
   experiment, not a production fix.
3. Save host copies of source tensors before replay. Compare their contents
   after replay, and compare trace outputs against an oracle computed before
   replay or from independently preserved inputs. Check an identical replay
   and a replay with changed inputs before returning to S128 layer/cache/logit
   comparisons. Keep timing separate from debug readbacks.

The allocation tracker is deliberately conservative: even successful keepalive
can leave it rejecting unrelated post-capture allocations because it does not
test actual overlap. Do not disable program-cache reporting to establish the
original diagnosis. A keepalive acceptance claim requires observed tensor
preservation and correct repeated outputs; merely suppressing a rejection is
not evidence of safety.

## Uncertainty

- Triage did not capture the reshape map contents, source addresses or writer
  symbols, so the exact overwritten buffer and writer failure mechanism remain
  unverified. Copy-program allocations are confirmed exposure; reshape-map
  corruption is the strongest explanation for the later stop-site.
- The helper cannot retain temporary buffers wholly internal to C++ composite
  ops. Its actual coverage and retained memory need measurement. It must be
  warmed using the same operation configuration used during capture.
- This report makes no precision-instability claim: the selected BFP8 recurrent
  cache is cast to BF16 within recurrence; BF16 activations and CCLs, selected
  BFP4/LoFi projections and cache dtype are unchanged between controls.
- No implementation repair or hardware validation was performed by this agent.
  The helper was parsed with host `python3` without importing TTNN. A Python/docs
  change needs no C++ build. The coordinating agent owns the device verification.
