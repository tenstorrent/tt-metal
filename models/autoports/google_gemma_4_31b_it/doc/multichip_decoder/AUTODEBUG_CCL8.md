# Stage04 BFP8 CCL investigation

## Finding: diagnostic host-read mocking retains L1 tensors across chunks

**High-confidence diagnostic bug; original NaN PCC remains unresolved.** The diagnostic collision is strong evidence of excessive live L1 allocations in that diagnostic run, but is not evidence that the original BFP8 run silently overwrote circular buffers. The diagnostic harness itself introduces a concrete lifetime extension that explains progressive allocation pressure.

Inspection only: no hardware commands, implementation edits, launcher, subagents, or removed Gemma implementations. Only this report was written. Startup contracts were read.

## Direct observations and revision limits

- `sliding_fastnorm8.json`: batch 1, sequence 4096, sliding attention, sharded residual, `gather_before_norm=true`, `ccl8=true`; prefill PCC NaN, decode PCC 0.9998494661621853.
- `sliding_fastnorm16.json`: same reported geometry and gather-before-norm setting, without CCL8; prefill PCC 0.99995164568619, decode PCC 0.9999427222054661.
- `sliding_nan_diag.log`: two complete sequences of finite component checks, then a third sequence reaching the pre-MLP gather output. The gate matmul throws before its finite check: lowest occupied L1 address 630208, static dataflow end 660608, overlapping by 30400 bytes, core range `[0-0 - 10-9]`.
- The log does **not** directly check every norm output. It checks gather inputs/outputs and selected matmul/RS outputs. Finite downstream values provide indirect evidence for earlier operations, not a complete norm audit.
- The saved decoder hashes match the inspected `optimized_decoder.py` and `multichip_decoder.py`: `5267f1ee…b6ff6` and `90a50b06…0c8eb`. The runner hash differs between original and diagnostic manifests. Neither manifest hashes `multichip_candidate.py`.
- At initial inspection the candidate had `diagnose` and `pad_mlp` additions. During this investigation it also acquired `_use_ccl8`, `ccl_group`, and `dram_prefill` changes from another writer. I did not make those edits or use their behavior to explain the old run. References below identify methods where candidate line numbers are unstable. The named original configuration enables neither padding nor DRAM-prefill overrides.

NaN PCC alone is not an elementwise NaN proof: `run_optimized_decoder.py:32-34` uses unguarded `torch.corrcoef` on flattened doubles. Nonfinite data or zero variance can produce NaN. The original artifact does not preserve elementwise finiteness, chunk coordinates, or a first bad tensor.

## Concrete lifetime chain

`tests/run_multichip_decoder.py:61` installs:

```python
patch('ttnn.to_torch', side_effect=None if diagnose else AssertionError(...),
      wraps=ttnn.to_torch)
```

This is a recording `MagicMock`, even with `side_effect=None` and `wraps`. Each `_check` in `MultichipCandidate` passes its device tensor to that mock. Mock call history retains the argument object after the real conversion returns. Those references survive until the surrounding prefill/decode context exits; normal local-variable deletion cannot release them.

The prefill path sets `prefill_l1=True` for B=1 and physical S=1024 (`optimized_decoder.py:573`). Gate/up outputs are BFP8 L1, and attention-output/down outputs are BF16 L1 before the candidate copies them to DRAM (`optimized_decoder.py:254-285`). `_check` runs **before** that copy. Thus the mock retains the original L1 tensors even after Python rebinds `out` to its DRAM successor. Gate/up also remain referenced after `_finish` deletes its locals. The input/output CCL checks mostly retain DRAM tensors, adding lifetime pressure but not directly accounting for this L1 overlap.

| Point | Normal lifetime | Diagnostic added lifetime |
|---|---|---|
| Attention output projection | L1 result becomes unnecessary after DRAM copy | Mock retains checked L1 result |
| Gate/up projection | Needed through elementwise product | Mock retains both beyond product and chunk |
| Down projection | L1 result becomes unnecessary after DRAM copy | Mock retains checked L1 result |
| Next chunk | Completed chunk output is stored in DRAM | Previous checked L1 results remain allocated |

One 1024-by-5376 result contains 5376 tiles: BF16 payload 11010048 bytes per device, BFP8 payload 5849088 bytes per device, distributed over L1 banks. Retaining the two BF16 projection results and two BFP8 gate/up results adds about 33.72 MB of logical tensor payload per chunk per device before bank padding. This is not a claim of an exact allocator address reconstruction: bank placement, aliases and other live tensors are not recorded. It is sufficient to show substantial monotonic extra L1 ownership across checks.

A cheap standalone Python weak-reference experiment reproduced the ownership mechanism without importing TTNN: a tensor stand-in remained alive after a wrapped mock read and deletion of the caller reference, then became collectible after `reset_mock()`. This verifies Python retention, not hardware behavior.

### Lowered gate allocation

For both the BF16-CCL and BFP8-CCL candidates, the gate input is restored to BF16, its local weight is BFP4, and its output is BFP8. Prefill selects `minimal_matmul`, not the decode DRAM-sharded factory. Config is M/K/N blocks 4/8/16, subblock 1/8, LoFi, FP32 destination accumulation disabled. Physical M/K/N are 1024/5376/5376 (32/168/168 tiles).

`experimental/minimal_matmul/device/minimal_matmul_program_factory.cpp:159-177,307-355` sizes four separate CBs from the actual input/weight/output formats:

| CB | Tiles | Bytes/tile | Bytes/core |
|---|---:|---:|---:|
| Input, double buffered | 64 | 2048 | 131072 |
| Weight, double buffered | 256 | 576 | 147456 |
| Output, double buffered | 128 | 1088 | 139264 |
| Intermediate, BF16 | 64 | 2048 | 131072 |
| Total | | | 548864 |

The factory uses the full 11-by-10 grid reported in the error. Its partition formulas give four M tiles/core and sixteen N tiles/core, one M/N block per core, and 21 K blocks for this shape. These values do not change with the CCL toggle. The logged static end minus this CB total is 111744 bytes; the log does not independently identify all contents below that address.

`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3071-3104` compares the lowest occupied L1 address against the allocated static region and throws on overlap. This is a protective validation failure, not a report that the gate already wrote corrupt values. The high-level `B*S <= 1024` admission rule cannot account for diagnostic mock-owned tensors from previous chunks.

## CCL dtype, placement, ownership and completion audit

For the named residual-sharded candidate, per-device prefill chunks are `[1,1,1024,1344]`. Input/pre-MLP norms first gather hidden shards to width 5376, then normalize with replicated BF16 weights. The CCL8 gather path casts DRAM BF16 to DRAM BFP8, gathers into DRAM BFP8, then restores DRAM BF16. Post-attention/post-MLP distributed norms gather width-32 BF16 statistics to width 128; the `width > 128` condition excludes those statistics from BFP8 conversion.

Attention-output and down projections produce local full-width partial sums. Their padded width 5632 is sliced to 5376, copied to DRAM, optionally cast to BFP8, then reduced/scattered along dim 3 to width 1344. The candidate restores BF16 before the post norm. No output buffer is supplied for an in-place cast or persistent collective output.

Supporting lowered contracts:

- `operations/copy/typecast/typecast.cpp:46-59` defaults output memory config to the input's. `device/typecast_device_op.cpp:149-166` creates a fresh output when no preallocated output is given. These casts do not unexpectedly move DRAM CCL tensors into L1.
- `experimental/ccl/composite_common.cpp:33-62`: 5376/4=1344=42 tiles, so this tiled dim-3 RS does not enter the non-tile-aligned composite fallback.
- `reduce_scatter_minimal_async.cpp:93-119` returns the primitive's output (index 1), with intermediate ownership handled by the operation; merely rebinding Python `out` does not demonstrate a caller use-after-free.
- `reduce_scatter_minimal_async_op_device_operation.cpp:38-46,173-245` selects the **Linear** factory, creates output with input dtype and requested DRAM placement, and a separate doubled-leading-dimension intermediate. The ring-only contiguous staging path is irrelevant here.
- `reduce_scatter_minimal_async_program.cpp:1850-1890,2110-2130` derives CB data format/page size from the input and selects `line_reduction.cpp` plus the line reader/writer. BF16 pages are 2048 bytes; BFP8 pages are 1088 bytes. Input/output page counts are 5376/1344 for prefill, versus 168/42 for a padded one-row decode. Shape-dependent scheduling remains a possible lower-level distinction; no full line-protocol correctness proof is claimed.
- `models/common/modules/tt_ccl.py` owns and cycles two sets of AG/RS semaphore handles, with no configuration difference between the two original candidates. No concrete semaphore-ownership violation was found in this bounded inspection.

Ordinary forward calls enqueue device work. Diagnostic `to_torch` calls require the corresponding data to be available on the host and change completion boundaries as well as tensor ownership. The ordinary harness reads results after prefill **and decode**, then synchronizes; later timing/replay sections also synchronize. Reaching a forward return is not a proof of completion, whereas the saved PCC was computed from host-read results. Finite diagnostic CCL samples cannot exonerate the original unsynchronized chain, nor establish where its first divergence occurred.

## One discriminating fix to try

**Fix the diagnostic read guard, preserving the numerical candidate:** when `diagnose=true`, use `contextlib.nullcontext()` for the `to_torch` guard (or patch with `new=original_to_torch`, a plain callable). Keep the recording/raising guard only when diagnostic reads are forbidden. Keep `ccl8=true`, gather-before-norm, unpadded MLP, original prefill placement and original geometry. This is a proposed harness edit only; it was not applied here.

This removes the demonstrated cross-chunk tensor retention without changing CCL precision or gate kernels. A future authorized run should distinguish:

1. Collision disappears and a component becomes nonfinite: the mock obscured the numerical failure; the first failing check bounds further investigation.
2. All diagnostic chunks are finite, ordinary run still has NaN PCC: synchronization sensitivity or unchecked norm/residual/product outputs remain; do not call BFP8 validated.
3. Collision remains: inspect actual live allocations after removing mock ownership; the generic prefill L1 budget then deserves a separate resource audit.

Removing diagnostic reads entirely would also remove retention but lose localization. Switching CCL to BF16 is already known to pass and does not discriminate lifetime from numeric/protocol causes. Padding the MLP or forcing DRAM also changes more variables than this control.

## Adjudication and other potential issues

| Candidate explanation | Direct support | Observations explained | Missing proof / disposition |
|---|---|---|---|
| Diagnostic mock retains L1 outputs | Source references plus host-only weakref experiment | Two chunks succeed, later allocation collides; instrumentation changes allocation | Exact addresses require allocator capture; leading diagnosis for diagnostic failure only |
| Original BFP8 CCL or cast path produces bad data | Controlled CCL8/BF16 result contrast | Prefill-only NaN PCC with passing decode | No original first-bad tensor or failing lower-level contract identified; unresolved |
| Ordinary prefill L1 exhaustion/corruption | Large live working set, diagnostic collision | Could be shape sensitive | Diagnostic has extra owners; validator rejects overlap; original run reaches result generation. Do not headline as original cause |
| BFP8 leakage into BF16 norm/matmul | None in inspected caller path | Would offer a dtype explanation | Explicit restoration and DRAM defaults contradict simple leakage story |
| MLP padding or new candidate selectors caused original failure | None | None | Absent from reported original configuration; excluded |

The retained `q/k/v/a` locals in `prefill_chunk_forward` and the attention argument in `_finish` also contribute to peak lifetime, but are shared by both original CCL choices; there is no evidence here that they explain the controlled contrast. Ordinary numerical quantization alone is not an adequate explanation for NaN PCC. The report's claims were rechecked against the caller, mock behavior, matmul allocation, and CCL dispatch; no claim is made that the proposed diagnostic fix repairs the original numeric failure.
