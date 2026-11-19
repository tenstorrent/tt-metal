# GDN combine destination windows — opt-in candidate

Prepared CPU-only on 2026-11-13. Root applied the patch after native-seed jobs closed at approximately21:10UTC; whole-layer hardware evidence is pending in combine_window27. The preparation account below describes the original CPU-only handoff.

Prepared CPU-only on 2026-11-13. `candidate.patch` changes only
`tt/packed_correction_candidate.py`; it has **not** been applied. No device,
build, TTNN import, model import, or active production file modification was
performed. `cpu_proof.json` records the exact source, proposed source, writer,
patch and repository HEAD hashes. HEAD alone is insufficient because the target
helper is currently untracked. Recheck its content hash before applying.

## Scope and enablement

Setup opt-in `packed_correction_options.prefill_direct_output=true` retains the
current capped input slices, matmul program selection, FP32 joint scratch,
reader, compute arithmetic, CB formats, and BF16 output pack. It allocates one
final `[1,padded_BS,4160]` BF16 interleaved DRAM destination before the chunk
loop. Each existing combine launch writes its local tile range at
`(chunk_start_row / 32) * 130` in that destination. It removes the per-piece
BF16 allocations and final concat. No new collective or residual layout is
introduced. FP32 scratch remains chunk sized; a full-sequence FP32 joint output
is never allocated.

The option is restricted initially to fused combine with FP32 joint output.
It changes only **multi-piece** prefill. Single-piece prefill and decode retain
their original allocation and kernel paths. The default is false. Precision
and program selection remain unchanged. Current best geometry is not encoded
or overridden in this patch: the owner's selected row cap and `prefill_plan`
remain authoritative. Therefore an old candidate geometry cannot silently
replace the current best configuration.

## Addressing and lifetime

The original writer at `optimized_decoder.py:4391` drains CB2 one 2048-byte
BF16 tile at a time, barriers, and pops. The setup-only adapter adds runtime
argument 3 `output_tile_offset` and changes only the destination accessor's
`page_id=tile` to `page_id=tile+output_tile_offset`. The barrier, CB ownership,
worker ranges, original local tile indices, and input addressing are unchanged.
In particular, the reader continues to use each chunk's local row and padded
joint width, including the correction origin at column tile 130. A global
destination offset must never be applied to that reader or compute index.

`_combine(packed, output=..., row_offset=...)` guards flat aligned input/output,
FP32 input/BF16 output, tiled interleaved DRAM, same device, distinct buffer
addresses, offset alignment, output capacity, and uint32 tile addressing.
The output accessor is constructed from the full destination, not a fake
chunk tensor shape. The supplied destination remains live across every chunk.
The runtime argument binds its actual address on every call; no address is
embedded into source or retained as a Python cache key.

“Preallocated once” here means one invocation-owned output before that
invocation's chunk loop, replacing all old piece outputs and concat output.
It does not mean an owner-global mutable buffer that later forwards could
overwrite. No persistent tensor cache, state mutation, or length-keyed growth
is added. As before, trace callers must allocate persistent input/state/output
banks before first capture, warm every relevant program, copy the returned
temporary result into their preallocated output slot inside capture, and
release capture temporaries before replay. This patch does not claim to
establish that trace proof on CPU.

All internal padded rows are written exactly once. The inherited final logical
slice and reshape remain intact, so lengths such as 33 and 513 require no
public alignment. A generic flat 16416-row invocation at cap4096 produces four
4096-row pieces and one 32-row tail, as the CPU proof covers. The actual whole
B32/S513 decoder instead schedules a 512-token temporal chunk (16384 projection
rows, four 4096-row pieces), followed by a separate S1 projection. All these
destination intervals are tile aligned. Input slicing,
short-tail native matmul behavior, and geometry are deliberately retained;
the native row-window reader concerns in `../projection_row_window25_design.md`
are not introduced by this writer-only change.

## CPU evidence and limits

Run from repository root:

```bash
python models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/gdn_combine_window26/prepare_patch.py
git apply --check models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/gdn_combine_window26/candidate.patch
```

The script only generates artifacts; it never applies the patch. It reports
70 complete/disjoint writer interval cases with caps512/4096, B1/3/17/32,
nonaligned logical lengths, cap boundaries, and five-piece tails; 10 exact
shape/address-guard negatives; AST equality of the matmul call/config and
reader runtime arguments; unchanged compact/decode methods; and exact recovery
of the original writer by undoing only its offset addition. Source parsing
and `git apply --check` pass. These are address/ownership proofs, not numerical,
hardware API admission, cache-rebind, or trace-allocation proofs.

For multiple chunks the removed concat reads and writes
`2 * padded_rows * 4160 * 2` bytes per rank. For the actual 16384-row projection
in whole B32/S513 this is 272,629,760 bytes (260 MiB); retained BF16 piece
storage falls by 136,314,880 bytes at the old concat boundary. The generic
16416-row CPU case is 273,162,240 bytes (260.5078125 MiB), but it is not the
whole decoder's projection schedule. The parent reports actual S513 concat
op2656 at 706 us; this is the targeted existing op, not a measured candidate
speedup. The final output already existed in the old path.
Earlier allocation of that output changes allocator placement and the lifetime
overlap with chunk scratch, so actual peak memory still needs measurement.
There is no measured latency claim. S33 measurements do not predict S513;
parent's actual S513 profile decides whether this family is material.

## Required serialized device experiment

After the root's source checkpoint, apply `candidate.patch` and prepare paired
explicit policy JSON files from the **same current default policy**, differing
only in `packed_correction_options.prefill_direct_output` false/true. Preserve
the selected geometry and all other options. No native rebuild is required;
the new writer is a generic-op source kernel compiled on first launch.

Run the existing actual-TP4 real-weight `tests.run_multichip_decoder` harness
with `--layer 0 --batch 32 --length 513`, the paired `--policy-file`, frozen
`doc/multichip_decoder/delivery_perf_ref_l0_b32_s513.pt` as `--reference`, and
distinct `--output` artifacts. First use a bounded focused quality case with
the production cap and at least two pieces (e.g. B3/S1537 => 4611 logical rows)
if a matching frozen reference is available; otherwise B32/S513 is the
existing-reference discriminating case. B1/S513 and B32/S33 are unchanged-path
regressions at cap4096, not evidence that the multi-piece optimization works.

Acceptance keeps all current output/contribution/per-user/cache gates >=.995
against the frozen single-chip TTNN reference, plus exact candidate/control
corrected projection bytes where observable. Cover the nonaligned final
logical slice, decode continuity, changed inputs, fresh output binding,
repeated queued execution, and changed-input traces. A focused adapter check
should write several chunks into a poisoned output and verify the whole
logical result and all internally padded rows, including every chunk boundary;
exercise invalid output capacity/layout/dtype/alias before the first launch.

Trace quality requires `TT_METAL_TRACE_ALLOC_TRACKING=1` and
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, all stable banks allocated before
capture, a real positive control, no exemptions, and replay after eager
interleaving. Watcher evidence must be separate from timing. For performance,
unset watcher/profiler environment variables and compare warmed complete
projection and complete layer boundaries, including the unchanged input
slices and final logical slice. Collect tt-perf-report on S513 separately to
confirm concat removal, unchanged matmul configurations and combine count,
and no new conversion/collective. Do not accept source-only traffic estimates
as performance evidence.
