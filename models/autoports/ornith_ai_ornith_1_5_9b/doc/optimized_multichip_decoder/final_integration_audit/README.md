# Final integration preparation — source audit only

This is a reviewable integration plan against the production decoder with source SHA256 `5cdc9e59dc6c69e3f43284694ad4a225ac340f0c291f42060de94c8a156958a4`. It does not select pending candidates, claim their gates passed, or modify production. Source hashes are recorded in `source_manifest.json`. Root must use the completed serialized evidence before choosing any change. Pending precision/fusion, buffered-state, identity-elimination, and BF16 recurrent-state families are excluded.

## 1. Fast-zero recurrent reader

Port only `tests/kernels/fast_zero/reader.cpp` to `tt/kernels/flat_ordered_post_loop/reader.cpp` after its gates pass. Source comparison confirms compute, writer, and normalization header are byte-identical between these directories. Keep the existing production helper, FP32 state, HiFi4 exact compute, wave104, CB capacities, and output contracts.

The reader adds `api/dataflow/noc.h`, a local `Noc`, NoC-local zero fills, their barriers, and CPU fences before scalar/unit/eye publications. Q/K/V and z fills are8192 bytes each; g4096, beta2048, unit2048, eye4096. It retains the production standalone gate layout: g/beta `[B,1,8]` read tile `batch`, z `[1,1,B,1024]` starts at tile0. **Do not substitute the cumulative conv-wide-gate reader**, which reads one gate tile and packed z at tile64; that belongs to a different, pending family.

Every fill must finish via `write_zeros_l1_barrier()` before scalar writes, DMA overlay, or push. CB producer ownership remains unchanged. Validate bitwise recurrent state, output .995 per head, narrow-wave repeated heads, DRAM/L1 reads, watcher, and the final decoder long gates. This changes no allocation or context capacity.

## 2. Pooled linear B1 and full-attention B1 routing

Port the two narrow behavioral changes from `tests/multichip_pooled_one_full_candidate.py`; do not replace `decode_batch_packed` wholesale and risk losing intervening production fixes.

In `allocate_batch_state` (production:463), change the early singleton return from `if batch == 1` to `if batch == 1 and self.kind != "linear_attention"`. The existing linear pool allocation then handles B1 naturally: tiled BF16 conv `[3,1,1,2048]`, FP32 recurrent `[1,8,128,128]`, and `state_row=0`. Preserve context validation and all existing state uniqueness/reset contracts.

**Keep `allocate_state` unchanged.** Its linear singleton stays privately owned ROW_MAJOR `[1,1,3,2048]`, FP32 recurrent, `state_row=None`. This is a supported distinct API and exercises the original private decode path. A lone selected row from a larger pool also remains a subset, not a new full B1 pool: the pooled fast-path predicate requires backing shape `[3,1,n,2048]` and enumerated state rows, so a B32 backing with one selected request correctly uses per-request extraction/write-back.

In `decode_batch_packed` (production:1259), replace only:

```python
if n > 1 and (self.kind == "full_attention" or pooled_linear or independent_linear):
```

with:

```python
if pooled_linear or self.kind == "full_attention" or (n > 1 and independent_linear):
```

This sends a true pooled linear B1 through tiled convolution and fused recurrence. Private linear B1 still takes the per-request path. All full-attention packed batches, including B1, take `full_attention_decode_batch`. Full-attention B1 allocation remains private: `page_offsets=None`, no additional cache pool or offsets tensor. Its batched method detects the non-ordered-pool case, finds one unique KV pair, aliases that pair, builds the local table, and performs no KV concat/scatter-back. Full B1 can also refer to one row from a larger pool; page_offset is preserved.

Full B1 now uses the existing partial norm/RoPE + batched SDPA geometry (11×10, key chunk128), rather than private `full_attention`'s 8×8/key chunk256 path. Preserve batched attention-table width padding to multiples of4, including allocated contexts96/128/160 and heterogeneous source pools; do not copy the private path's multiple-of8 condition into the batched method. Unused lookup/user rows remain internal padding. `decode_forward` stays unchanged; the nonaligned prefill bridge still invokes the original per-token full-attention path. Do not conflate that bridge with packed B1 routing.

Required integrated checks: private B1, pooled B1, B1 full private cache, singleton row31 from B32 pool, reordered/subset/mixed states, per-request reset/neighbors, full table widths3/4/5 pages, traced refresh/replay, and unchanged long-reference gates for both layer kinds.

## 3. Large-prefill DRAM staging, lifetime, and kind-specific chunks

The intended chunk policy is linear32768/full16384 with minimal matmul K16. Production already uses `MinimalMatmulConfig(M4,K16,N8,subblock1×2,grid11×10)` at lines334–341: **no K precision/geometry change is required**. Keep that same config in ordinary projections and the direct packed-SwiGLU call in `finish`.

The discriminator inherits frozen reference/geometry classes. Do not transplant that inheritance or its frozen decoder methods into production. Port these local behaviors only:

1. Choose a default chunk by layer kind after resolving HF config. Preserve explicit caller chunk overrides. A clean signature is `chunk_size=None`, followed by32768 for linear or16384 for full when omitted; then keep positive/multiple-of32 validation. Do not reinterpret explicit1024 as an implicit default. Audit runners/callers for explicit old defaults so final default measurements actually reproduce the selected policy. Existing long harnesses may set chunk sizes through their reference runner, so record effective runtime values.
2. For replicated residual only, and only `input_layernorm` / `post_attention_layernorm` with more than4096 rows, run RMSNorm directly into DRAM, then BF8 typecast into DRAM. Return to L1 only when rows<=8192; retain DRAM above8192. All rows<=4096 retain current behavior. Thus4096 is the old path,4097–8192 DRAM-first staging with final L1,8193+ DRAM residency. Preserve the current one-token native decode and sharded-residual branches before this condition. No new BF8 policy: norm already uses BF8 at512+ rows; only location/order changes.
3. In the non-fused prefill projection branch of `linear`, use DRAM input when rows>8192, L1 otherwise. Cover normalized packed-input activations **and** BF16 attention/down-projection inputs. Preserve `fused_prefill` branch precedence and one-token `decode_linear`; do not silently claim the DRAM policy has been measured for sharded/fused alternatives. Result dtype remains BF16/DRAM and existing role fidelity remains unchanged. `finish`'s direct packed-SwiGLU consumes the new DRAM h naturally and must keep the same K16 config.
4. In the existing prefill loop: `del chunk` after appending a one-token alignment-bridge result; `del h` after attention has produced `mixed`; and `del mixed, chunk` after `finish` has produced/appended the sliced DRAM result. These are Python reference lifetime reductions, not forced deallocation. Do not call `deallocate` on aliases or persistent state, and preserve queued-operation lifetime management.

### Logical lengths and context arithmetic

Keep the production `n>0`, nonnegative start, and `start+n<=262144` contract. Chunk sizes are internal multiples of32; the public logical n need not be aligned. Linear attention owns packed projection padding, q/k/v convolution, neutral g=0/beta=0 tail rows, logical history extraction, and output slicing. Do not pre-pad the logical linear sequence in the outer loop: its recurrent state/history must advance only for real tokens.

Full prefill with unaligned absolute start first runs one logical decode token at a time until32 alignment (at most31 tokens). Then choose `count=min(remaining,chunk_size)`, pad only the full-attention physical chunk to `ceil(count/32)*32`, fill its physical cache pages, and slice the emitted output to count. Causal attention prevents padded future rows from affecting prior logical outputs. Continuation overwrites those future entries at their true positions. Absolute offsets advance by logical count, not padded count.

At max supported context262144, an aligned-start final chunk ending at logical262139 or262143 pads only up to262144. The context allocator still requires physical page capacity in multiples of32; that existing allocation contract must not become an aligned-public-input requirement. Keep the distinction between logical length and physical cache capacity. Test lengths around4096/8192/16384/32768, continuation starts1/31/32/33, terminal nonaligned262139/262143, and decode at the supported boundary.

## 4. Memory/context consequences and all-layer risks

Fast-zero changes no allocation. Full B1 batched routing changes transient QKV/SDPA workspaces but not persistent KV dtype or capacity. Pooled linear B1 **does** change per-layer history storage:

| Per chip, one linear layer | Private B1 | Pooled B1 |
|---|---:|---:|
| BF16 conv history physical bytes | 12288 | 393216 |
| FP32 recurrent bytes | 524288 | 524288 |

The history increase is380928 bytes/layer, or9142272 bytes (8.71875 MiB) across24 linear layers. Existing optimized B1 analytic budget17570103296 becomes17579245568 bytes for the pooled-B1 API, holding other estimates fixed. Private `allocate_state` retains the old budget. The current context document explicitly says private B1 is12288 bytes and B1 budget unchanged; those statements must be updated if the pooled-B1 default is integrated. Record both allocation APIs and retain supported context262144 unless physical evidence requires otherwise. Existing BF8 KV and FP32 recurrence remain unchanged.

Large chunks shift workspace from L1 to DRAM. A BF16 replicated hidden activation `[1,1,32768,4096]` is256 MiB per chip; at16384 it is128 MiB. BF8 physical tile storage adds exponent overhead (1088 bytes per32×32 tile): corresponding norm buffers are136 MiB and68 MiB. Packed projections, convolution/GDN intermediates, attention QKV, residuals, and packed MLP temporaries can coexist. Deleting outer h/mixed helps but does not prove their internal or downstream temporary peaks are bounded.

At262144 rows, one BF16 hidden tensor is2 GiB per chip. The prefill loop retains output chunks until final concat; input + retained chunks + concat output alone can transiently reach roughly6 GiB, before live layer weights, caches, GDN/SDPA/MLP temporaries, CCL and trace reserve. Do not multiply per-layer transient workspace by32 when execution is serial, but do account for all resident layer weights, all request states, shared CCL buffers, and caller retention of prior layer outputs. Existing8 GiB reserved activation/trace/CCL budget is an analytic envelope, not proof this larger transient peak fits an all-layer stack.

Layer-kind max-context execution and analytic all-layer capacity are distinct. Do not claim all-layer allocation, B32 maximum context, or full-model readiness from one-layer/short-stack measurements. Full-model/vLLM work remains out of scope. Update `context_contract.json` with the actual selected implementation hash, new history accounting, staging/chunk policy, and final evidence only after integration validation; no capability reduction is justified by this source audit.

## 5. Final integration gates

Measure the combined default, not the sum of isolated candidates: warmed prefill and traced warmed decode before/after, every meaningful layer kind, TP4 target mesh, unchanged PCC, runtime fallback audit, pool contracts, max/nonaligned contexts, and watcher-clean coverage. Refresh profiling/provenance after source integration. Preserve the replicated BF16 residual stored in local L1 width shards across decoder layers with no new boundary collective/reshard. Any tests still instantiating historical candidate classes cannot serve as final-default evidence.

This audit wrote only new documentation. It did not select pending fusion/precision variants, edit production, run hardware, or alter gates.

## Follow-up: guarded defaults and unapplied base patch

`base_integration_5cdc.patch` is a unified patch against the exact5cdc decoder and current source snapshots; **it has not been applied**. `git apply --check` passed. It includes only fast-zero reader, pooled linear B1 / full packed-B1 dispatch, the shared-workspace docstring correction to2 MiB/chip, guarded large-prefill DRAM/lifetime changes, and effective chunk reporting in `tests/multichip_probe.py`. Phase kernels, SDPA variants, precision reductions, AR worker alternatives, and other pending fusion options are excluded.

The constructor now proposes `chunk_size=None`. Omission resolves32768/16384 **only when residual_layout is replicated and fused_prefill is false**. Sharded residual or fused-prefill omission resolves1024. Explicit positive/multiple-of32 chunk sizes are never reinterpreted. A pure CPU AST test evaluated24 combinations of layer kind, residual layout, fused flag, and omitted/explicit1024/8192 sizes, all passing. This avoids making an unmeasured large chunk implicit in alternate modes.

The proposed norm staging branch is after the existing singleton-native and sharded-residual returns and requires hidden-layer norm names, replicated residual, not fused-prefill, and rows>4096. Large projection DRAM residency likewise requires replicated/nonfused and rows>8192, after the fused projection and singleton branches. This deliberately preserves existing alternative-mode placement even for explicit large chunk overrides. K16 is unchanged.

The probe defaults its CLI override toNone and omits the constructor keyword in that case for **both** multichip and optimized baseline. Thus the optimized baseline keeps its own1024 default, while the multichip path uses its guarded kind default. An explicit `--chunk-size` applies to either class. Results retain `chunk_size` as the actual `layer.chunk_size` and additionally record `requested_chunk_size`, preventing candidate timings from being mislabeled as the final default.

AST validation and base/proposed SHA256 values are in `base_patch_validation.json`. Only audit artifacts were written. Applying this patch still requires the final combined-default gates and context-contract accounting described above; the patch intentionally does not mark pending context evidence as passed.

Final documentation precision note: the512-row BF8 prefill threshold is evaluated
on the tensor presented to norm/reduction. Full-attention prefill explicitly pads
its last non-aligned chunk before norm, so511logical rows become512 internally
and take BF8; linear511 remains logical511 at this point. Describe this as the
operation's chunk-row threshold, including owned full-attention padding, rather
than a uniform public logical-sequence threshold. No public alignment restriction
is introduced. Prefill residuals cross layers in DRAM interleaved BF16; packed
decode residuals cross layers in local L1 WIDTH_SHARDED BF16.

Pre-final configured Ruff audit (read-only, production5cdc) found four import-order
items, one dict-constructor style item, and two B023 diagnostics for the immediate
weight-column closure's rank. After applying the base patch and before final
measurement, sort imports, use a literal geometry dict, and bind `rank=rank` in
that setup closure. No production files were changed while queues were live.
Ruff is now installed in python_env; Black and clang-format were already present.
