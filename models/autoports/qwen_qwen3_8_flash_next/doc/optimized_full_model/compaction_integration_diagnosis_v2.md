# Compaction integration diagnosis v2

CPU interpretation only; no new device or performance experiments. Compaction remains **unselected** in production v4. The component's speedup does not override its serving failure. Root cause is not established.

## Observations

- [Native-only serving repro](compaction_serving_repro_v1.json), [log](compaction_serving_repro_v1.log): fails mixed-versus-solo after solo3, solo33, then mixed3/33. **Row0 logits and all12 mutable cache fields across four ranks match; row1 has247197 differing logits, with both outputs finite.** Row1 cache comparison was not reached because the logits assertion fired first. Q2/Q1/Q32 prefill traces remain live together. The JSON body time is66.604s; the caller's approximately80s includes process/setup/teardown outside that field.
- [Eager real-operand probe](compaction_eager_operands_v1.json), [log](compaction_eager_operands_v1.log): passes Q1/2/3/32 on reduced real layers0/1/3. All28 compaction calls match legacy outputs in both logical and full physical storage on every rank:112 logical and112 physical comparisons. Final logits and all12 mutable cache fields match at each Q. The JSON body time is52.504s (approximately57s caller wall time), following the owner's [recovery4](recovery4.log).
- Both runs use Watcher10 and allocation tracking with program-cache allocations included. Their compaction helper and kernel hashes agree. Earlier [integrated serving v1](final_serving_watcher_v1.json) failed row0; this reproduction fails row1. A deterministic claim that a particular slot or Q always fails is therefore unsupported. The [reverted selected control](final_serving_watcher_v2.json) passes all16 serving checks.

## What the eager pass establishes—and its limits

The tested real gathered operands use the expected384-to320 mapping, and native canonicalization produces exactly the same logical values and padding as the legacy compactor for those28 calls. This materially weakens a fixed face-offset, missing-tail, padding-value, or BF16 canonicalization explanation for those inputs. It does not prove equality for every possible model operand/history.

The eager wrapper calls the legacy slice/concat **after native compaction and before the next consumer**, then synchronizes for readback. That legacy path executes untilize/retilize and may condition LLK state. The extra operations, allocations, and reads also change timing and buffer lifetimes. Consequently, even exact final eager logits/cache do not establish correctness of the native-only graph, or rule out a compute-state initialization dependency. The serving wrapper executes no legacy compactor or interior tensor reads for selected calls; its failure is the relevant native-only integration result. Watcher/tracker checks do not prove the absence of every stale-address, wrong-runtime-argument, or initialization dependency.

## Remaining hypotheses and next discriminating check

1. **Trace/allocation/history interaction:** retained cross-Q traces, transient-buffer reuse, runtime address/count updates, or cross-slot workspace state. The standalone probe released each Q trace before moving to the next Q; that scope cannot exclude this family.
2. **Downstream LLK-state dependency:** removing legacy untilize/retilize may expose a later kernel's incomplete configuration initialization. The eager comparator masks this possibility. No offending register or kernel has been identified.

At a later authorized debugging window, first run the **same native-only solo3→solo33→mixed3/33 sequence eagerly**, without the comparator, legacy compaction, or intermediate device reads. Keep only the public call-boundary readbacks already present in the serving repro. Use a separate legacy-only run as the strict reference; record finite logits and every cache field for both rows before asserting. If native-only eager passes while traced execution fails, prioritize retained cross-Q/address/lifetime debugging. If eager also fails, prioritize the first downstream divergence and LLK initialization. This is a hand-off check, not a performed experiment or a request to reopen optimization before the Stop Rule. Production remains on the passing uncompacted path.
