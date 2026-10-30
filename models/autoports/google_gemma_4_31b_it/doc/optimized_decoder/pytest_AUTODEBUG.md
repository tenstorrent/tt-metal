# Stage03 optimized decoder investigation

## Leading finding: full decode reads two page IDs that the caller never supplied

**A concrete capacity/chunk contract violation is present.** For the delivered seq=33 tests, full-attention decode requests a 128-token SDPA chunk against a page table containing only two 32-token pages per request. The lowered reader consumes four page IDs. The extra IDs come from unspecified alignment padding, which fast-dispatch host writes do not initialize. This is the leading diagnosis for the process-history/refresh-sensitive failure, rather than a precision rejection or an established Python lifetime bug.

Confidence is high in the source violation and the padding mechanism; attribution of the exact observed finite PCC values remains unverified. No tests or hardware commands were run, no implementation files were edited, and no AutoDebug launcher was invoked. Three independent inspection agents assisted with lifetime, refresh, and decode-kernel analysis. The startup run contract and evaluation inputs were read.

Paths below are relative to the checkout. `D` denotes `models/autoports/google_gemma_4_31b_it`; `S` denotes `ttnn/cpp/ttnn/operations/transformer/sdpa_decode`.

## Direct observations versus interpretation

The direct observations in `D/doc/optimized_decoder/final_pytest.log` are:

| Layer / batch | Prefill | Initial eager decode | 32 unchanged trace replays | Refreshed trace |
|---|---:|---:|---|---:|
| Sliding / 1,32,16,8 | Pass | Pass | Pass | Pass |
| Full / 1 | 0.999945 | -0.305115 | Not reached | Not reached |
| Full / 32 | 0.999941 | 0.999952 | Pass | 0.047583 |
| Full / 16 | 0.999943 | 0.999950 | Pass | 0.075941 |
| Full / 8 | 0.999944 | 0.999947 | Pass | -0.231393 |

Evidence: log lines 34–190 and 199–317. All full prefill results pass, so these are decode failures. The B1 failure precedes capture, ruling out a trace-only explanation for the whole matrix.

The exact-current-source watcher controls pass for full B1/B32: all four relevant file hashes (`optimized_decoder.py`, runner, pytest, trace helper) in `final_watch_baseline/manifest.json` match the inspected files. Those controls use `TT_METAL_WATCHER=10`, trace allocation tracking enabled, and program-cache allocations included. The older `final_validation_v2` hashes differ; their semantic equivalence must not be inferred from filenames. The pytest log does not record the trace-allocation environment. It does show fast runtime enabled and logging/comparison mode disabled.

An additional available artifact, `pytest_fixture_cli.json` and its `.log`, uses the actual pytest layer-5 input file and passes B1 eager/replayed/refreshed decode. Its prefill PCC matches the failed pytest case. This weakens a fixture-content explanation without proving all runtime/environment differences controlled.

Interpretation: unspecified page-table padding is a source-backed way for identical logical inputs to depend on prior command-queue contents, and for refreshing valid rows to change otherwise hidden state. The artifacts do not record the offending padding values or fetched physical addresses; their exact numerical effect remains a hypothesis.

## Complete source chain

1. `D/tests/run_optimized_decoder.py:164–167` computes `blocks=(s+32)//32`. At `s=33`, `blocks=2`, constructs INT32 `pages[B,2]`, uploads it row-major, and allocates `2*B+3` cache blocks. Each request has 64 tokens of physical capacity. Blocks 0–2 are zero sentinels; owned IDs are 3 through `2*B+2`.
2. `D/tt/optimized_decoder.py:192–201` chooses `k_chunk_size=64` for sliding and **128 for full**. `_decode_attention_forward`, lines 667–679, passes that config to paged SDPA, using BF16 Q and BFP8 K/V with scale 1.0. No circular-cache modulo is supplied.
3. `S/sdpa_decode.cpp:133–143` preserves a positive explicit chunk size, checking only power-of-two and 32-token alignment. `S/device/sdpa_decode_device_operation.cpp:454–467` checks capacity divisibility only for **non-causal** attention. This causal call passes admission despite capacity 64 < chunk 128.
4. `S/device/kernels/rt_args_common.hpp:45–69` rounds the causal end up to a complete chunk. Initial position 33 gives `ceil(34/128)*128=128`; refreshed positions 26–32 also give 128. There is one chunk with four sequence tiles. For sliding, both phases round to 64, or two tiles. `get_dynamic_Sk_chunk_t` at lines 93–105 returns the fixed positive chunk unchanged; it does not clamp it to cache capacity.
5. `S/device/kernels/dataflow/dataflow_common.hpp:755–781` reads K for every row `0..Sk_chunk_t_dynamic-1`; lines 837–861 do the same for V. The shared mapper in `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp:48–69` divides by `block_size_t=1` and directly dereferences `page_table_ptr[virtual_block]`. `capacity_t=0`, so there is no modulo guard. Full attention reads table indices **0,1,2,3**, although only **0,1** are supplied. The violation is shared by every failing batch, independent of the different QKV splitting paths.
6. The page-table reader in that shared file, lines 72–90, reads the **aligned stick size**, including padding. Row-major default tensor alignment is `{1}`, not an extension to four valid IDs (`tt_metal/impl/tensor/spec/layout/page_config.cpp:49–58`); the host payload is only 8 bytes per row.
7. Fast-dispatch upload sets `page_size_to_write=aligned_page_size` and `data_size_to_copy=page_size` (`tt_metal/impl/buffers/dispatch.cpp:109–110`). It emits an aligned write (lines 570–576), but copies only actual payload bytes (lines 600–615). `DeviceCommand::add_data` (`tt_metal/impl/dispatch/device_command.cpp:907–911`) copies those bytes and advances past the entire aligned slot without zeroing the tail. The hugepage-backed constructor at lines 35–44 does not initialize the supplied queue memory. Therefore IDs 2/3 are not guaranteed zero or valid; they can reflect reused command memory.
8. `D/tests/run_optimized_decoder.py:333–343` rewrites that same page-table allocation with another host upload before replay. `ttnn/core/tensor/tensor_ops.cpp:176–186` routes the host copy to `enqueue_write_tensor`; the pinned-transfer optimization requires page size equal to aligned page size (`dispatch.cpp:1180–1186`) and does not rescue this 8-byte row. Even though the valid page IDs are correctly flipped, its unspecified tail can change. Unchanged replays do not upload the table and thus retain the same tail.

For full attention, each extra physical tile address uses `physical_block * (4 heads * 1 tile/page * 16 width tiles) + head*16`. The invalid IDs can therefore direct reads outside the allocated KV blocks. The caller's logical sentinel check does not inspect the aligned page-table tail and cannot catch these reads. A watcher pass in a fresh process does not establish that padding was safe in the failing process.

## Kernel work and synchronization ledger

All tested batches have page tables `[B,2]`, page size 32, and input decode positions initially all 33. Refreshed output row `j` corresponds to original row `B-1-j` and its position `32-((B-1-j)%7)`. The mapping is applied consistently to x, cosine, sine, current position, page rows, and expected output.

| Mode | Head dim / KV heads | Fixed K chunk | Read sequence tiles | Declared table entries | Missing entries |
|---|---|---:|---:|---:|---|
| Sliding, every tested batch | 256 / 16 | 64 | 2 | 2 | None |
| Full, every tested batch | 512 / 4 | 128 | 4 | 2 | 2 and 3 |

Full uses the 8x8 SDPA grid throughout B1/8/16/32. Core allocation in `S/device/sdpa_decode_program_factory.cpp:202–216` gives respectively 16/2/1/1 cores per KV head and 1/1/1/2 KV heads per core. There is only one live chunk per head: surplus cores get no chunk, while an active reader still reads all four tile rows. This is not a claim of a circular-buffer push/pop mismatch: each full K or V chunk contains `4*16=64` tiles per head, matching the nominal compute chunk. The problem is **where those tiles come from**. Sliding chunks contain `2*8=16` tiles per head, within declared capacity.

The preceding fused cache update itself needs only page 1 for positions 32/33 and page 0 for positions 26–31; its position/page ownership and buffer balance were inspected without finding the same over-read. SDPA binds current-position and page-table buffers explicitly (`S/device/sdpa_decode_program_factory.cpp:1028–1046`), rather than freezing their contents into the trace. This localizes the concrete discrepancy to rounded SDPA reads, not the legitimate cache write at the current token.

`to_torch` consumes prefill and eager output before the PCC checks. The runner synchronizes before capture (line 305), uses blocking replay (lines 311 and 343), and reads the replay output before evaluation. Refresh calls and replay use queue 0; this is not evidence of a missing cross-queue event. Failure before/after these boundaries is actual output divergence, not merely host work reaching the end before completion.

Masking is not a bounds check. `S/device/kernels/compute/sdpa_flash_decode.cpp:408–430` performs QK over the full chunk; lines 490–529 exponentiate/reduce and multiply by the full V chunk. Valid finite extra data may be harmless under a perfect causal mask. Arbitrary fetched bits are not guaranteed valid finite K/V. However, **do not claim NaN poisoning is proven**: the reported PCCs are finite and the harness uses raw `torch.corrcoef`, which would not silently discard NaNs. Static inspection proves the invalid addressing, not the exact arithmetic or collateral effect producing -0.305/0.047/0.076/-0.231.

## Smallest intervention and decisive follow-up

The first intervention should make the full decoder's effective chunk compatible with caller capacity, or provide sufficient explicit valid pages. For this exact test, a full decode chunk of 64 stays within both allocated pages and preserves precision, logical positions, and mathematical attention. A general solution must account for the final rounded chunk, not merely cap a chunk by total capacity: for fixed chunk C and requested last position p, capacity must cover `ceil((p+1)/C)*C`, unless the reader explicitly guards and zero-fills missing rows.

A second clean control is to keep chunk 128 and supply four explicit page-table entries per request with safe initialized backing pages for the masked tail. Do not rely on incidental row alignment, and keep sentinel/ownership checks meaningful. At the generic op boundary, causal admission should reject unsupported capacity/chunk combinations or the reader should implement bounded tail loads. Zeroing host-transfer padding alone would mask this caller/kernel contract mismatch and is not a sufficient general repair.

Suggested focused checks for the parent investigation (not performed here):

1. Repeat the exact eight-case sequence with only the full chunk changed to 64. Keep dtype, fidelity, inputs, replays, and environment unchanged.
2. Independently retain chunk 128 but make four valid page IDs explicit, with zero backing for masked pages. If both controls pass, the capacity explanation gains substantial causal support.
3. If either fails, record page-table logical rows **and aligned tails**, Q, updated K/V, and SDPA output for the first divergent case. Do not infer physical IDs from final PCCs.
4. Verify changed-input eager output as well as replay output using the same valid-capacity configuration. This separates general decode behavior from trace-specific defects.

A chunk-size control alone changes kernel geometry, so the explicit-four-page control is important for distinguishing the capacity violation from an unrelated 128-token kernel bug.

## Other potential issues and rejected headline explanations

### Audit bound-method cycles: real cleanup issue, unsupported root cause

`D/tests/run_optimized_decoder.py:151–163` builds `model -> wrapper -> bound method -> model`. The model retains weights, metadata tensors, and mesh (`optimized_decoder.py:59,112–128,146–148`). Finally closes the mesh while these objects remain live (runner lines 386–389); exception tracebacks can retain the whole run frame. Avoiding permanent instance method replacement would improve ownership clarity.

But `models/conftest.py:13–15` runs `gc.collect()` before each test's run/open. More importantly, owned mesh buffers point to the old MeshDevice (`tt_metal/distributed/mesh_buffer.cpp:271–280`). Close clears its allocator manager and scoped devices and marks it uninitialized (`mesh_device.cpp:1135–1143`). Its `is_initialized()` remains false despite physical-device reopen (lines 236–244), and owned-buffer deallocation checks that flag (`tt_metal/impl/buffers/buffer.cpp:731–758`). The ordinary stale-free/new-allocation story therefore lacks a source mechanism. The optional hybrid L1 mirror cleanup merits separate inspection only if hybrid mode and surviving stale objects are established; neither prerequisite is evidenced here.

### Persistent program cache across meshes: not established

Each MeshDevice constructs a fresh program cache (`tt_metal/distributed/mesh_device.cpp:381`). Retaining an old closed mesh/cache does not make the new mesh reuse that cache. A deeper physical-device/compiled-kernel state bug is possible but presently less supported than the capacity violation. Do not label this a cross-layer program-cache collision without showing the shared cache entry and bad binding.

### Host refresh and trace allocation: no independent contradiction found

The harness preserves device tensor identities and flips all request-indexed values consistently. Refresh creates host tensors, not new device tensors; its padding side effect is the concrete issue identified above. Capture cleanup is balanced for body exceptions (`D/tests/optimized_trace.py:6–18`). Trace allocation accounting does not make out-of-range page IDs valid. A trace-only bug cannot explain full B1's eager failure, and a blanket trace-storage overwrite does not predict 32 good unchanged replays followed by divergence specifically on host refresh.

### Fixtures and precision

The recorder pins config/tokenizer/weights to the requested revision and generates distinct prompts (`record_decoder_activations.py:17–35`). Batch32 versus batch56 CPU reference preparation need not be bitwise identical, but the direct pytest-fixture CLI control substantially weakens this explanation for B1. No evidence justifies relaxing PCC thresholds or rejecting the selected precision policy. Keep the memory contract repair separate from precision selection.

## Post-draft claim review

Rechecked the caller's two-page allocation, explicit chunk propagation, causal validation gap, absence of cache modulo, four-row K/V loops, aligned-stick load, and non-zeroing dispatch copy against source. The leading finding covers all four full batches and avoids the sliding cases. The audit cycle and general lifecycle/cache theories were demoted because the teardown guards and setup GC contradict their simplest causal story. The remaining unproved link is the exact finite numerical corruption caused by the invalid page reads; the report does not claim a hardware-confirmed root cause or a completed fix.
