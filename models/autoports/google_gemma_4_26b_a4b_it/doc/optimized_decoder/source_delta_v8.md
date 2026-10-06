# V8 source delta and validation attribution

Current runtime SHA256 is `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`. Its exact v7 ancestor is [the saved source](runtime_v7_before_placement.py.txt), SHA256 `daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b`. [The CPU proof](source_delta_v8.json) records the complete diff,36 unchanged method hashes and exact inverse AST transforms for all four changes. Removing only the listed additions reproduces the original method ASTs exactly. No hardware was run by this auditor.

| Changed method | Delta | Preserved scope |
| --- | --- | --- |
| `OptimizedDecoder.from_state_dict` | Adds setup options `prefill_qkv_minimal_block_h` and `prefill_qkv_input_l1`; defaults resolve full to2/True, sliding4/False. Forwards M block and records producer memory. | All existing default precision/compute/weight/geometry choices and explicit opt-outs remain unchanged. Disabling minimal disables automatic L1 placement. |
| `OptimizedDecoder.normalize` | For enabled multi-row input normalization, calls the original unweighted normalization and emits its final learned-weight multiply directly into L1. | Same arithmetic order, dtype and learned weight; all one-row decode branches and other normalization sites are unchanged. No separate DRAM-to-L1 copy is added. |
| `MinimalPrefillProjection.__init__` | Validates M cap1–4; builds four configs using `min(rows, block_h)`. | K16/N8/subblock1×4/grid11×8, FP32 input/output/accumulation, BFP8 weight and HiFi2 full compute remain unchanged. Full output and sliding QKV retain M4. |
| `MinimalPrefillQKV.__init__` | Forwards the M cap to the generic helper. | Source/decode references, tied K/V, compute-clone flags and weight aliases are unchanged. |

The exact scalar branches pass48 default/override cases. Generic setup passes eight M/K combinations, preserving weight/compute identity and building all tail configurations outside forward. Eighteen normalization cases execute the old/new methods with CPU stand-ins; only the enabled multi-row input's final multiply output memory differs. These checks verify source/control flow, not accelerator numerical equivalence. The changed M block may alter numerical evaluation, so affected full paths were rerun.

## Current and inherited gates

[The v8 manifest](validated_v8_validation_summary.json) has status `current_targeted_and_inherited_gates_passed`: all12 primary plus five boundary commands return0, with no pending or failed correctness item. It verifies current default M4/M2, all four tail configs and the exact producer-memory policy from each report.

Full B32, prefix, BF16-cache compatibility, allocation-tracked nine-request reuse,262144,262143, headline and Watcher pass. Reuse keeps379 program-cache entries with misses forbidden while the trace is live. Both maximum contexts pass all291 sampled rows (minimum0.996029643684647), final tails and boundary decode. Both-kind1025/512 stress passes exact HF and direct fused-preservation gates; full minima are0.996214257441/0.996295795028, sliding0.995555061903/0.995158301497. Four pytest cases pass. Tight1025/cache1152 under Watcher records the actual Q64/K128/read-end1152 boundary; full65/1023/1024/1025 controls pass accuracy, repeated traced decode and warmed timing checks.

Broad sliding B32/prefix/BF16-cache/max evidence remains at v5, and full allocation-tracked sliding reuse at v6. V7 and v8 source proofs retain sliding M4/HiFi4/default normalization placement and unchanged public/cache/lifecycle logic. The manifest rechecks those original report hashes through the v7 ancestor manifest; it does not claim broad sliding contracts reran on v8. Current sliding headline/Watcher/stress/pytest are fresh.

## Selection and storage

The exact M4/K16/N8 L1 attempt collided with live input storage: CB end1307648 exceeded buffer start1114112. M2 reduces per-compute-core CB payload from1,196,032 to737,280bytes and the adapted trial passes. [Matched grid evidence](prospective_qkv_advice_v7.md) rejects110 cores at identical copied-L1 M2:4/8 wins and median paired+2.8355µs. [Producer control](prefill_qkv_producer_v7_layer5.json) compares copied-L1 M2 with direct producer-L1 M2 in32 alternating pairs:22 wins, median paired−128.295µs. These are whole-prefill host controls on the archived v7 plus probe, not v8 native timings.

Resident checkpoint, cache, RoPE and retained shared/QKV weight payload delta is zero. No setup tensor is added. One existing FP32 normalized input tensor changes memory placement; at1024×2816 its payload is11,534,336bytes, distributed by the interleaved L1 allocator. This is not a per-core allocation or peak estimate. The reduced CB counts are per compute core and exclude other kernels, runtime metadata and live activations. [Memory accounting](final_memory_accounting.md) keeps those scopes separate.

Both v8 native profiles and CPU accounting pass at this hash; old native rows and timings remain separately attributed. Independent final stage approval remains pending.

## Native output-memory correction

The current raw full-prefill sequence is FP32 normalized inputL1[1024,2816] → minimal-QKV outputL1[1024,9216] → tiedKV sliceL1[1024,1024] → concatDRAM[1024,10240]. The minimal output request isNone, so `minimal_matmul_device_operation.cpp:297` inherits the input memory. The packed output/slice payloads are37,748,736/4,194,304bytes; concat is41,943,040bytes. Only resident payload delta is zero; more than the normalized input changes transient memory placement. Individual payloads are not a summed live or peak estimate.

The proof inventory and dependent manifest were regenerated to correct this metadata. Exact prior proof bytes are retained in [the old proof snapshot](source_delta_v8_before_output_memory_correction.json), SHA45e6047d…; its generator is also preserved. Runtime and hardware evidence did not change or rerun. The profile guard checked runtime/status, not the old manifest hash. The old manifest SHA651366f9… is recorded in final_policy.json without inventing its original timestamp or an unavailable byte snapshot.
