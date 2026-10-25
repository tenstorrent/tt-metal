# Independent BFP8 geometry assessment

2026-10-25. Source inspection only; no hardware access or model execution. Reviewed `AUTODEBUG.md`, production packing and phase views in `tt/multichip_decoder.py`, `_linear` in `tt/optimized_decoder.py`, `tt/precision.py`, and the DRAM-sharded matmul validation, factory and weight reader.

**Assessment: mix readers=2/block=20 and ordinary down readers=2/block=17 are legal Blackhole candidates and preserve current physical weight packing.** Mix readers=2 is a reasonable first experiment because it preserves eight K-reduction iterations; superiority to readers=1/block=10 is unproven. The latter has 24,576 bytes less static L1 demand, but doubles K-block iterations to sixteen. Reader placement, multicast participation and output writes change with two readers, so preserved K grouping does not establish bitwise identity or better latency.

## Contracts checked

- Linear-attention mix has K=5120 (160 tiles), eight activation storage shards of 20 tiles, and packed physical N=4608 (144 tiles). Its 4160-column logical concatenation includes the two 32-wide A/B groups, each containing only 12 true values. `weight_pad_readers=2` already pads to 512-column multiples; changing readers 1→2 leaves this unchanged. Each bank has 18 tiles, exactly two readers × 9 tiles. Block20 divides K and equals the activation shard width. Block10 also divides both.
- Down has local K=4352 (136 tiles), eight 17-tile activation shards and N=5120 (160 tiles). N is divisible by both 256 and 512, so changing padding readers from one to two does not add columns or change the DRAM shard shape. Each bank has 20 tiles, exactly two readers × 10. Block17 divides K and matches the shard. Block8 is illegal for this ordinary eight-storage-core path, despite dividing136.
- Full-attention mix also retains packing: logical/physical N=3584 is already a multiple of512, giving14 tiles per bank and7 per reader. Its existing block20 and K160 are compatible. This sibling still requires its own real-weight check.
- Python `cores=8` controls activation and output storage. `per_core_N` remains18 for linear mix and20 for down. Sixteen reader workers are independently assigned by the factory. Its writer explicitly reshards each reader's9 or10 columns into the existing eight storage shards. Do not halve Python `per_core_N` or double storage `cores` as part of the reader change.
- Factory compute width differs from reader width: linear mix9 widens to10 with subblock width5 and4 valid columns in the final subblock; down10 stays10. The input buffer uses actual reader width, while output/partial buffering uses widened compute width. The weight reader uses `dram_reader_index * reader_width_tiles` as its bank-column offset, so the second reader consumes the second half of the same packed bank.
- Multi-reader support explicitly requires Blackhole and a NOC0 weight-reader kernel. Ordinary DRAM-sharded validation requires one physical M tile; these conclusions cover packed batches up to32, not arbitrary larger batches.

## Per-core static arithmetic

For BF16 activation/output, BFP8 weight, FP32 destination disabled, no bias, and multiple K blocks: A is double-buffered, B triple-buffered, BF16 output/partials alias. Tile bytes are2048 and1088. The recorded Blackhole static base is111,744 bytes.

| Geometry | A | B | Output/partials | Endpoint including base |
|---|---:|---:|---:|---:|
| Linear mix block20/readers2 | 81,920 | 587,520 | 20,480 | 801,664 |
| Linear mix block10/readers1 | 40,960 | 587,520 | 36,864 | 777,088 |
| Down block17/readers2 | 69,632 | 554,880 | 20,480 | 756,736 |

These reproduce the report's arithmetic independently from the factory formulas. Resident activation/output tensors remain separate live allocations. The initial failure's1,212,032-byte live frontier is evidence for that phase only; neither its reuse for every phase nor synthetic reduced-model allocation success proves full-owner capacity.

## Propagation and remaining obligations

Resolve geometry after final per-layer dtype overrides, before weight packing, `prepare_prefill_rs`, and deep-copy creation of `_batched_decode_impl` and `_prefill_impl`. Preserve the existing nested role dictionaries: top-level `policy.update` replaces a role dictionary wholesale. Later changes to only the base policy do not reach already-created views. Match attention, MLP and down independently; a configuration-name check misses layer exceptions.

The reader change repairs ordinary projections only. Multirow prefill uses a separate 2D config and `prefill_block`; native batched and prefill fused out/down consume `decode_rs_geometry`/`prefill_rs_geometry`. Large linear mix and gate/up prefill capacity still needs the independently proposed smaller K blocks. One-row prefill can use ordinary `_linear` with the prefill activation dtype, so verify its actual dtype instead of assuming this BF16 arithmetic applies unchanged. Promotion to BF16 weights or a changed activation format needs fresh sizing.

## Smallest discriminating real-weight experiment

Load production-packed layer0 BFP8 mix once on all four ranks and reuse the same real normalized activation. At fixed LoFi, compare only `(readers=1, block=10)` against `(readers=2, block=20)`, keeping eight storage cores, padding, output config and all other policy entries fixed. Record actual physical shapes, resolved config, allocation endpoint and live frontier; synchronize/read back each result. Compare every Q/K/V/Z/A/B group against a reference using the same quantized packed weights, excluding only defined padding. Report errors per rank/group, then warmed latency under identical timing conditions. A BFP4 result is not a same-dtype geometry reference.

On that same representative layer, run ordinary BFP8 down with block17/readers2 against the same quantized-weight reference; record5120 output columns and unchanged bank packing. If original readers1 is used as an allocation contrast, reduced-owner success is possible and does not refute the full-owner failure. Before acceptance, repeat on full-attention layer3, verify copied phase policies, and capture/replay with changed inputs. Full64 Stage08 accuracy, full-chunk prefill capacity and warmed trace results remain required after these reduced checks; no serving integration is implicated.

## Follow-up: BF16 decode CCL with BFP4 down

The `decode_ccl_bf16.json` result and log identify an ordinary down failure with BF16 input `[1,1,1,4352]`, BFP4 weight `[4352,5120]`, LoFi and FP32 destination disabled. The preserved `sources/pre_geometry/multichip_decoder.py` SHA256 matches the result's recorded source hash (`e20d5e…e64b8`); its down policy is eight storage cores, block17, readers1. The current workspace source also matches its recorded hash. Thus this assessment uses the failed run's preserved geometry, not the concurrently edited implementation.

The original BFP4 endpoint is exactly `111744 + 2*17*2048 + 3*20*17*576 + 21*2048 = 811904` bytes. The recorded live frontier798336 leaves a13,568-byte overlap. The widened compute width21 must be used for output/partials even though the reader width is20. `calls:2` is an attempted-call count, not proof of two completions; the stack contains a program-cache-hit path. A first-call-only reduced smoke is therefore insufficient.

**Extending ordinary down readers2 adaptation to the BF16 decode-CCL candidate is a minimal legal one-field repair.** It retains block17, eight K iterations, storage grid, weight packing and precision. The resulting BFP4 endpoint is `111744 + 69632 + (3*10*17*576) + 20480 = 495616` bytes, a316,288-byte reduction and302,720 bytes below the recorded frontier. The same divisibility and writer contracts established above apply. Block1 is the only smaller positive divisor of the17-tile storage shard; it changes reduction grouping substantially. No allocator or workspace synchronization change is justified by this evidence.

`DecodeAllReduceWorkspace` keeps two live `[32,20480]` scratch tensors width-sharded over eight cores, each shard holding80 tiles. BF16 promotion raises their combined payload from174,080 to327,680 bytes per storage core (153,600 extra), or1,228,800 extra bytes per device. This establishes increased persistent L1 pressure. It does not by itself attribute the exact798336 address to a particular tensor or explain every difference from another phase's frontier. Preserve both slots: the workspace explicitly requires alternating out/down uses and says one slot is unsafe because arrival is not remote-consumer completion.

Apply the condition to the resolved effective workspace payload before phase-view copies; explicit `ccl_dtype='bfloat16'` covers this precision candidate. The constructor also treats a missing/false CCL dtype as BF16 when constructing a pooled owner, so a general policy helper should reason from the effective payload and active workspace path rather than raw string equality alone. Changing down on a path without that owner is not demanded by this failure. Keep native fused geometry separate.

The smallest useful control is a real layer0 complete ordered out/down decode using the BF16 shared owner and BFP4 weights, repeated with changed inputs through warm/cache-hit execution, comparing readers1 and2 where allocation permits. Add layer3 and readback/trace checks, then rerun the full-owner candidate. The readback should check real down outputs against the same quantized-weight reference; a reduced run with a higher frontier cannot prove the full-owner allocation repair. Static capacity relief is supported; numerical, warmed-performance and full-model acceptance remain for the hardware owner to establish.

## Follow-up: later BFP4 mix and prefill scope

The saved `failures/decode_ccl_bf16_down_repair.json` reports mix calls3 versus calls2 for out/gateup/down and an852608-byte endpoint against833152 live frontier, a19,456-byte overlap. Its recorded decoder hash matches `sources/after_bfp8_geometry/multichip_decoder.py` (`cdfeb0…837f0`). That snapshot changes BF16-CCL down to readers2 but leaves BFP4 mix at block20/readers1. This supports a later ordinary mix allocation failure after two completed layer paths; attempted-call counters alone do not prove device completion.

Mix block20→10 is legal and exactly predicts `111744 + 2*10*2048 + 3*18*10*576 + 18*2048 = 500608` bytes, saving352,000 and leaving332,544 bytes below the recorded frontier. It preserves physical packing and storage/output layout while doubling K-block iterations. A readers2/block20 alternative would predict525184 bytes; neither candidate's numerical/performance preference follows from this allocation evidence. Extending the existing block10 rule to the effective BF16-CCL owner is a narrow repair for the demonstrated failure.

Prefill is a distinct, reachable capacity risk, not an observed consequence of this failed run: `prepare_decode` invokes `_decode()` before its prefill warming loop. The shared owner remains referenced by the layers throughout prefill, so its larger BF16 slots stay live even though prefill collectives themselves use BFP8. For the existing BFP4-weight/BFP8-input 4096-row paths, the 2D factory formula is `2*H*Kblock*1088 + 2*W*Kblock*576 + H*W*2048` (BF16 shared output/partials), plus111744 base:

| Prefill projection | H,W | Original endpoint | Proposed endpoint | Saved |
|---|---|---:|---:|---:|
| Linear mix, Kblock16→8 | 13,14 | 1,195,136 | 839,808 | 355,328 |
| Separate gate/up, Kblock20→10 | 8,13 | 972,416 | 648,576 | 323,840 |

Both smaller K blocks divide K160; the ordinary activation-shard constraint does not apply to this interleaved 2D path. Each saving exceeds the153,600 bytes/core added by BF16 workspace slots. This makes them defensible conservative candidates to restore margin relative to baseline at those phases. It does **not** establish that either original prefill geometry fails: phase-local live allocations differ, and applying the decode frontier833152 to prefill would even falsely reject the proposed839808 mix endpoint.

Recommendation: keep the demonstrated decode repair separate in the evidence. Test original prefill geometry at4096 under the same four-real-layer/full-head owner, inspect the actual phase-local frontier, and shrink only if capacity or a documented margin requirement demands it. If avoiding another expensive full-owner attempt motivates applying both prefill changes now, label them precautionary capacity adaptations and include their accuracy/latency comparison; do not describe them as proven necessary fixes. In particular, unconditional changes to BFP4 prefill outside the BF16-owner configuration would be needlessly broad. Short-prefill success does not settle these4096-row budgets, and full-attention mix/fused row paths retain their separate configs and obligations.

## Final AutoFix follow-up review

The final BF16-CCL implementation uses ordinary mix block10/readers1, ordinary down block17/readers2, and linear-attention prefill mix block8. These satisfy the divisibility, padding, writer and phase-copy contracts checked above. Gate/up prefill remains20 for BFP4, and full-attention prefill mix remains4. The later measured prefill failure (endpoint1195136 versus frontier1140096, recorded in `AUTOFIX.md` and its linked probe) supplies the previously missing justification for adapting linear prefill; no speculative gate/up adaptation was retained.

The new branches are false for baseline attention/MLP/down BFP4 with BFP8 decode CCL, so that baseline's role geometries remain unchanged. Changes occur after final policy overrides and before packing and phase copies. All model-source hashes in the final `results/decode_ccl_bf16.json` match the inspected current files. Its runtime policies confirm the intended geometry and retained gate/up block20. It records full64 preparation, prefill and three traced teacher-forcing runs with top1=.99, top5/top100=1 and trace verification. The separate four-layer/full-head owner artifact passes lengths1,33,4095,4096,4097 with exact repeat tokens, unchanged three trace IDs and capture count3. This provides concrete boundary/tail, cache-hit and full-owner evidence for the observed allocation repairs; it is not a general stage-completion or broader quality verdict.

No concrete fidelity-scope mismatch remains in the active production paths inspected: ordinary `_linear` consumes projection/per-layer fidelity; persistent4096-row fused prefill consumes `fused_prefill`; native batched fused projection consumes `fused_decode`; decode paged SDPA consumes `sdpa_ckc`, while prefill chunked SDPA consumes the fixed shared HiFi4/FP32-destination `ckc`. The amended contract correctly documents the last distinction. The optional `minimal_mlp` path also uses `matmul_ckc`, but it is disabled in these policies and is not an active sweep discrepancy.

Required documentation cleanup only: `AUTOFIX.md` still opens with “Status: experiments pending” and includes unqualified earlier statements such as “No BF4 geometry changes,” despite its final verified closure. Mark that material explicitly historical and put the current repaired/verified status first. `policy_contract.md` still says geometry is inherited and the sweep does not change it; qualify that statement with the validated dtype/workspace-dependent K-block and reader adaptations, while noting unchanged physical packing/topology. No additional source change or expanded hardware sweep is required by this follow-up's findings.
