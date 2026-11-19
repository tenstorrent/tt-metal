# Prepared maximum-context memory audit

CPU/source audit, updated after completed **PASS** `final.json` and parent-confirmed exit0. The preparation arithmetic below is unchanged; post-prefill and post-decode endpoint censuses match it exactly. The context contract has now been updated from this completed evidence, with no capability reduction. Inference source hashes match the run; the parent subsequently corrected only the runner inventory visitor for future reports. No hardware access or acceptance rerun was performed by this audit.

## Authoritative allocator arithmetic

All values below are bytes **per chip**, batch1, context262144, actual all40 layers. The DRAM memory view has8 banks; multiply each per-bank field by8, never by four mesh ranks.

| Quantity | Arithmetic | Bytes/chip |
| --- | --- | ---: |
| Loaded weights, retained helpers and RoPE | 2363774976 ×8 | 18910199808 |
| Prepared allocation, including persistent state/workspaces | 3171580160 ×8 | 25372641280 |
| Prepared free allocation capacity | 1075759232 ×8 | 8606073856 |
| Allocator capacity in this device configuration | 4247339392 ×8 | 33978715136 |
| Prepared growth beyond loaded constructor | 25372641280−18910199808 | 6462441472 |

Allocated+free equals capacity exactly. Prepared allocated/free are23.63011360/8.01503086 GiB. Report the200000000-byte trace-region configuration separately: do **not** add it to the measured allocation, subtract it from free again, or substitute the old34111622144-byte decoder-stage allocator capacity. Trace reservation/live traces are already part of this device configuration; the memory-view capacity is authoritative. These are prepared-state measurements, also matched by both completed endpoint censuses; they do not measure the transient long-prefill peak.

## Geometry cross-check

Pinned configuration: hidden2048, vocabulary248320,10 full-attention layers,30 linear-attention layers, page32, BF16 paged KV, FP32 recurrent state. TP4 has local one KV head per chip with global head ownership[0,0,1,1]; do not divide local KV by four again.

| Component | Formula | Bytes/chip |
| --- | --- | ---: |
| Full-attention K+V | 10 ×2 ×8192 pages ×32 ×1 head ×256 ×2 | 2684354560 |
| Original page tables, batch1 | 10 ×8192 ×4 | 327680 |
| Linear recurrent backing | 30 ×8 ×128 ×128 ×4 | 15728640 |
| Linear convolution history | 30 ×3 ×2048 ×2 | 368640 |
| Recurrent+conv reset mirrors | 15728640+368640 | 16097280 |
| Recurrent+conv+page+position+activity snapshot logical payload | 15728640+368640+327680+8 | 16424968 |
| Three replicated BF16 residual workspaces | 3 ×262144 ×2048 ×2 | 3221225472 |
| Shared BF16 cosine+sine tables | 2 ×262144 ×64 ×2 | 67108864 |
| BF16 hidden-sharded embedding | 248320 ×512 ×2 | 254279680 |
| BFP8 vocabulary-sharded head, with exponent bytes | (2048/32) ×(62080/32) ×1088 | 135086080 |
| FP32 terminal norm, padded32 rows | 32 ×2048 ×4 | 262144 |
| Each sampler logits/zero-row allocation | 32 ×62080 ×2 | 3973120 |
| Two output-logit and two token snapshots | 2 ×3973120 +2 ×32 ×4 | 7946496 |

These formulas match `final.json.arithmetic` and the inspected cache/slot/generator construction. The FP32 recurrent state is distinct from BF16 paged backing. There is no full-KV reset mirror: `generator.py:116-142` retains KV roots for in-place fill and creates zero mirrors only for recurrent/conv. `serving_state.py:245-257` snapshots recurrent/conv/page tables, not physical K/V pools. Packed sampler/penalty state is permuted in place, not duplicated by `register_packed`. Snapshot8-byte position/activity is a logical payload; each individual row-major scalar allocation has allocator alignment.

Loaded constructor allocations plus the **single** KV pool equal21594554368 bytes/chip. Adding the three prefill workspaces gives24815779840. The remaining prepared growth is556861440 bytes/chip, including recurrent/conv originals and reset/snapshot storage, pages/controls, sampler and output state, prepared CCL/MoE buffers, alignment and other allocations. Do not label that difference a pure sampler or trace size.

Embedding+head+RoPE+terminal norm sum456736768. Subtracting their known payload from the loaded census gives18453463040 bytes/chip for the **retained decoder/helpers/setup residual by census difference**. This is18432000 below the earlier18471895040 planning estimate. It is not a unique learned-weight inventory: the constructor census also includes retained helpers/workspaces and allocation effects. Prefer the measured loaded total18910199808 and the explicitly labeled residual over falsely precise “weights only” claims.

## Inventory caveats and alias semantics

The saved run inventory is incomplete despite its `actual_unique_tensor_allocation_inventory` label. The run-recorded version of `run_full_model_context_acceptance.py:257` returns for `callable(value)` before descending into object fields. Decoder instances inherit `LightweightModule.__call__`; `TTSampling` and `TTPenalties` also inherit that class. Consequently the saved inventory has no decoder-weight category and skips tensors reachable only through those callable objects. Its unique aligned DRAM page total is6672853888 bytes (packed6672853344;544 alignment bytes), much less than the measured25372641280. That gap cannot be attributed entirely to program-owned storage. The allocator census still includes those allocations and remains valid.

The deduplication key `buffer_unique_id()` appropriately counts aliases/views once. Category assignment is **first traversal encounter**, not ownership:

- `slot_snapshots` includes both snapshot buffers and registered live sampler/penalty/control buffers. Its72062528-byte subtotal is not the16.4MB snapshot formula and must not be added as a second sampler budget.
- `model_embeddings_head_rope` includes the shared SiLU LUT in L1 (958464 aggregate tensor bytes) because callable layer traversal was skipped. This is not embedding/terminal storage.
- CCL buffers shared by layers/workspace, KV roots shared across request state aliases, immutable LUT sharing, and output views must never be counted per alias or per layer.
- L1 tensor payload totals (2531328) are aggregate shard payload. The L1 allocator's per-bank allocation591936 is a different measurement; do not compare or sum them as interchangeable quantities.

The parent fixed the visitor after this completed run; the raw inventory remains unchanged. For a complete future unique-tensor breakdown, descend into eligible model objects even when callable; exclude functions/methods/classes explicitly rather than all callable instances, preserve root-buffer deduplication, and label first-owner categories honestly or record all semantic aliases. Include every actual root once, then report the residual to allocator census separately. A CPU audit cannot recover skipped runtime tensor sizes. A later short load/preparation census can correct the detailed inventory without repeating the expensive long acceptance; the saved authoritative allocator totals already establish the prepared capacity arithmetic.

## Completed acceptance and applied contract update

`doc/context_contract.json` now contains `full_model_validation` with the measured loaded/prepared capacity, exact262143-token prefill plus final-position262143 traced decode, completed artifact/log hashes, source provenance, policy scope and partial-inventory limitations. `contract_update_plan.json` is marked `applied_after_completed_acceptance_pass`. Historical decoder-stage evidence, the original failed warmup, and unsupported claims remain distinct. Supported context is262144 and capability reduction remains null.

Completed evidence: one long-prefill call, one acceptance decode, prefill5883.448102414011 seconds, decode0.033730191993527114 seconds. These are maximum-context diagnostics, not standard128/128 headline performance. Final position is262144; all10 full-attention layers on all4 chips pass the last-page checks, preceding cache rows remain unchanged, and240 linear-state/chip checks are finite. The traced decode performs one model and one sampler replay with zero token/position/page/activity refreshes, zero readbacks and zero captures. Device-sampled token82 feeds decode and produces token3. Cleanup releases both traces (release count0→2), with no cleanup errors and normal device close; parent observed process exit0.

Prepared, after-prefill and after-decode DRAM allocation/free/capacity remain25372641280 /8606073856 /33978715136 bytes per chip. L1 endpoint values also match preparation. This does not establish a transient high-water mark. The partial unique-tensor listing is preserved; no skipped counts were fabricated.

Completed `final.json` SHA256: `858e1ad664ca2c04d519e50a30d05119433c79fb7f615e5b8ede86e5e0a0fd4a`. Completed `../logs/context_acceptance_lifetime_fixed.log` SHA256: `43b1a2c843001f9c1525a25199f3c0f314d0a1d5afb2fef2c6f0f86b0a74921a`.

Scope remains batch1/all40 with one request able to occupy the aggregate262144-token pool. It does not imply32 simultaneous262144-token requests. Root `largest_tested_prefill=262144` remains inherited decoder-stage evidence; this full-model test's actual prefill262143 and final decode position262143 are recorded separately. Final-source all40 prefill/teacher accuracy refresh now passes: each reports top1=98/100, top5=100/100 and top100=100/100 in `../prefill_accuracy_lifetime_fixed.json` and `../teacher_accuracy_lifetime_fixed.json`; both provenance files match current production sources. Final representative performance and review remain separate pending work. The installed stage06 context subcheck passes in `../logs/stage06_final_gate.log`; the overall invocation exited2 on serving-contract findings, so no overall gate pass is claimed.
