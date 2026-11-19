# Larger prefill chunks: source-only AutoFix discrimination

Status: source diagnosis and concrete retry plan; no hardware run and no production/test source edits. The current failure is an L1 allocation collision, not evidence that the model's context must be reduced.

## Corrected observed boundary

Both `large_chunks_control_0_4096.log` and `large_chunks_control_3_4096.log` contain successful prefill, four decode checks, and state checks for **4097**, after successful 4095 and 4096 cases. The command manifests list the next input as **8191**. Thus the first failing case is 8191 for both layer kinds, not 4097 for the linear layer. The tracebacks do not identify which internal chunk failed; the first-versus-second chunk attribution below is a source-backed hypothesis, not an observed device marker.

Both failures enter the attention output projection's `minimal_matmul` and report the same lower bound: the static dataflow region ends at **915,392**, while a live L1 buffer starts at **844,288**. The overlap is **71,104 bytes per affected core**. The linear stack points to `multichip_sdpa_reference.py:784`; the full-attention stack points to line 874. The validation is performed in `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3068–3095`: it compares static buffer end against the lowest occupied compute-L1 address. It does not identify the owning tensor in this error.

`--roles control` does not invoke the large-geometry implementation. `multichip_large_chunk_geometry.py:12–14`, `multichip_long_prefill_geometry_retry.py:16–17`, and `multichip_long_prefill_geometry.py:15–16` all fall through to the frozen reference. In particular, the command's `--output-k-block 8` does **not** change this control's minimal-matmul K block: it configures only the selected ordinary-matmul override. The actual control remains M4/K16/N8 (`multichip_sdpa_reference.py:313–320`).

## Exact static-buffer cost

For the failing attention output projection, the input is BF16 with local K=1024, the weight is stored BF8, the result is BF16, and accumulation is FP32. Config is M4/K16/N8, subblock1×2. `minimal_matmul_program_factory.cpp:211–223,355–409` allocates:

| Buffer | Tile count | Tile bytes | Bytes/core |
|---|---:|---:|---:|
| Double-buffered activation | 2×4×16 = 128 | 2,048 | 262,144 |
| Double-buffered weight | 2×16×8 = 256 | 1,088 | 278,528 |
| Double-buffered output | 2×4×8 = 64 | 2,048 | 131,072 |
| FP32 intermediate | 4×8 = 32 | 4,096 | 131,072 |
| Total | | | **802,816** |

The observed static end minus these buffers is 112,576 bytes, accounting for the program's starting region/reserved space in this run. This is inferred from the logged address, not a new hardware specification. The buffer sizes themselves do not grow with sequence length when this block config is fixed. Sequence length changes tile work assignments and live tensor sizes. The factory pads K to its block size and bounds M/N tails (`minimal_matmul_program_factory.cpp:313–328`), so a logical 4095 tail is not rejected by a K16 block divisibility rule.

At unchanged resident allocations, these adaptations give substantial headroom:

| Config | CB bytes/core | Change versus M4/K16/N8 | Predicted static end using the same starting region |
|---|---:|---:|---:|
| M4/K8/N8 | 532,480 | −270,336 | 645,056 |
| M4/K4/N8 | 397,312 | −405,504 | 509,888 |
| M2/K16/N8 | 540,672 | −262,144 | 653,248 |
| M4/K16/N4 | 532,480 | −270,336 | 645,056 |

These are source-derived allocation estimates, not measured latency or numerical results. K8 alone would clear the logged collision by 199,232 bytes if resident allocations are otherwise identical. K8 divides local projection K-tile counts 32,96,128, and satisfies the existing minimal config checks with subblock1×2. Changing K blocking can change FP32 intermediate packing order, so PCC must be rechecked. M2 or N4 avoids changing K's reduction grouping and is a useful independent alternative.

## Live tensors and the chunk-boundary hypothesis

The relevant frozen-source code is `prefill_forward:901–931`, `norm:580–590`, `finish:602–625`; current production retains the same pattern at `prefill_forward:938–967`, `norm:615–624`, and `finish:636–659`.

1. The outer `h = self.norm(chunk, "input_layernorm")` is retained for the rest of the chunk and across the loop boundary. Attention needs it for its packed input projection, but the outer frame still owns it during attention output projection and the entire MLP finish.
2. On the next chunk, Python evaluates the RHS of `h = self.norm(...)` before replacing the old `h`. Therefore the previous chunk's BF8 L1 normalization buffer is still live while the new norm performs its allocations.
3. `norm` currently moves its BF16 result into L1, then calls `typecast(..., bfloat8_b)` with an L1 result. The BF16 source and BF8 destination coexist during that conversion. On a second large chunk, they also coexist with the previous outer BF8 `h`.
4. Once the new `h` is assigned, the old one can be released, but the allocator does not compact the surviving new tensor into the released space. Thus a lower new L1 address can persist through attention and collide with the output projection's fixed 802,816-byte CB region.
5. A 4097-token prefill has a 4096-token chunk followed by only one logical token (padded to32 for full attention), whereas 8191 has two nearly full chunks. This is consistent with the observed threshold. Exact tensor addresses at each boundary are still needed to prove which allocation owns address844288 and whether failure occurs in chunk0 or chunk1.

Other lifetimes: `mixed` from the preceding chunk survives until the next attention call returns. The previous `chunk` survives RHS evaluation of the next slice. Neither is required across chunk boundaries; both can be released explicitly. Completed outputs are explicitly moved to DRAM before being retained in `outputs`, so the output list is not intentionally accumulating L1. The attention functions also retain locals (`packed`, `q/k/v`, gate, and recurrent intermediates) until their projection return; their actual memory configurations should be recorded before attributing the collision to one specific local. The current error alone does not prove that all these locals occupy L1.

The probe also retains some preceding-length decode tensors and snapshots after releasing each trace (`multichip_probe.py:218–225,305–360`). A fresh 8191-only process is a useful discriminator against test-history fragmentation; keep the original multi-length test as the realistic reuse check. Do not misclassify a pass in isolation as a complete fix.

## Minimal retries, in useful order

**A. Lifetime-only, no numerical changes.** In prefill_forward, release outer `h` immediately after obtaining `mixed`, before `finish`. After storing the DRAM result, release `mixed` and `chunk`. This prevents the previous normalized L1 tensor surviving next-chunk normalization and removes unnecessary residency during the MLP:

```python
mixed = ...  # existing attention call
del h
outputs.append(ttnn.to_memory_config(self.finish(chunk, mixed), ttnn.DRAM_MEMORY_CONFIG)[:, :, :count, :])
del mixed, chunk
offset += count
```

Use Python reference release, not forced `ttnn.deallocate`, because views and queued operations must retain their normal ownership semantics. The unaligned full-attention decode-at-start branch should likewise avoid carrying an unnecessary prior loop local.

**B. Quantize before staging in L1.** For the existing long-prefill BF8 norm policy, typecast the norm result in its current memory placement before the final L1 move. For the usual DRAM prefill residual this avoids staging the large BF16 copy in L1 and transfers BF8 rather than BF16:

```python
if x.shape[2] >= 512:
    out = ttnn.typecast(out, ttnn.bfloat8_b)
out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
```

Retain the original short-prefill and sharded-residual branches. If norm input/output already resides in L1, explicitly choose DRAM output for this long-prefill norm path before the cast; otherwise the intended peak reduction has not occurred. Validate exact/PCC behavior rather than assuming different placement is bitwise identical.

**C. Adapt the actual minimal config.** Set M4/K8/N8 on the real `self.prefill_program` used by the selected reference/default, or choose a role-specific minimal config for attention output/down projections. Merely passing `--output-k-block 8` with `--roles control` cannot test this. Try K8 and M2/K16 independently before considering a precision change. If a global config is changed, also measure packed-input and fused SwiGLU MLP costs; the latter calls minimal_matmul directly from `finish` and does not pass through `linear` overrides.

**D. Adapt input placement.** The current `linear` wrapper unconditionally stages its activation in L1 (`multichip_sdpa_reference.py:538–545`). Minimal matmul builds TensorAccessor-based readers from the input memory configuration; DRAM input is a supported source path. A role-specific DRAM activation retry can remove the additional full activation staging allocation. It does not, by itself, remove an already-live outer norm tensor. Record which tensor owns the lowest L1 address before claiming this clears the failure. The standard 11×10 matmul family with K8 is a separate useful comparison, with its own per-core-M growth and output buffering.

Collect before/after lowest-L1 address and the address/dtype/memory/shape of outer h at chunk entry, after norm, and before projection. Device memory-view reads must be outside a measured trace. Comparing A, B, C separately identifies whether the dominant issue is previous-chunk residency, conversion peak, or fixed CB footprint; the final kept combination then needs warmed latency measurements.

## Context and logical-length conclusion

Nothing in this evidence forces a context reduction. With dead chunk-local L1 references released, long-context L1 residency can be bounded by one configured chunk, independent of total sequence length; retained outputs and KV/recurrent state keep their existing DRAM ownership. The logged 71,104-byte deficit is smaller than the source-proven savings available from K8, and lifetime/placement changes may preserve K16 entirely. Therefore 4096 remains a viable candidate for all existing contexts, but it is not proven until the adapted implementation passes long and boundary cases.

Retest both layer kinds at 4095/4096/4097, 8191/8192/8193, 16385, 65536 and the advertised maximum including 262139/262143, plus nonzero continuation starts and the existing decode/state gates. Preserve internal padding and logical slicing; never require callers to align public lengths. Test repeated warmed calls and the original mixed-length process, not just one fresh sequence. Watcher and performance runs stay separate. 8192-sized internal chunks deserve the same adaptations before rejection, but their larger live activations and other kernels require separate evidence. Do not extrapolate a fixed output-projection collision into a hard model-context limit.
