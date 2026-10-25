# AutoFix — native BF16 KV SDPA

Status: repaired and verified through native/batched layer controls, native-end replay and full64 evaluation.

Fresh source-only diagnosis is AUTODEBUG_kv_bf16.md. Parent independently checked
the factory's double-buffered K/V and reduction staging terms and reproduced the
reported endpoints with standard-library arithmetic: K256/C32=1083648,
K128/C32=813312, K256/C31=1073408, C30=1063168, C16=919808. Recorded live frontier
is1065344. Cap31 still fails, cap30 leaves only2176 bytes, and cap16 retains less
margin than K128/C32. Choose BF16-only K128/C32 as the minimal robust hypothesis;
keep the existing mapped-page divisibility guard and all native capacity.

Source adaptation is after dtype overrides and before phase/batch copies. It
caps full-attention BF16 KV decode chunks at128 only. BFP8/BFP4 KV paths remain
unchanged. No allocator bypass, metadata truncation or context reduction.

First numerical control uses real recorded layer3 inputs and weights, native
262144 storage/map width, prompt127 and four decode steps across the128 boundary:
run_multichip_decoder --layer3 --seq127 --steps4 --capacity-alignment262144
--policy '{"kv_dtype":"bfloat16"}' --baseline-policy '{"kv_dtype":"bfloat16"}'.
The existing single-chip implementation is the same-weight/cache-precision
control. Additional mapped boundary, batched/ragged and native-end generator
checks follow before the full64 retry. No reduced throughput will be ranked.

Evidence correction: original KV smoke capacities were256 and4352 for lengths33
and4097, as the source-only investigator established; they were not8192. All
claims about native-context validation use newly recorded native-sized evidence.

The first intended same-precision control was invalid: OptimizedDecoder treats a
nonempty policy as the complete policy, so {'kv_dtype':'bfloat16'} selected its
fallback BFP8 matmul weights rather than inherited BFP4. Output PCC>.9997 but
cache PCC .99293/.993999 failed the unchanged.995 gate. Preserve this diagnostic
under failures/kv_bf16_native_s127_mismatched_reference.*; do not dismiss it as
SDPA numerical loss. Corrected control explicitly supplies BFP4 attention/MLP/down,
LoFi projection, BF16 KV, and legal K256/C16. Candidate uses BF16 K128/C32. No
threshold relaxation or production arithmetic change is made to fix the control.

Corrected native127 control passes: prefill PCC .999964856, decode PCC
.999954181–.999978655, key PCC .999565131, value PCC .999442046. It preserves
actual native map width, random page ownership and the .995 threshold. The
single-chip reference uses its shared HiFi4 attention config while TP uses the
selected LoFi SDPA; this is a same-weight/cache-precision correctness control,
not an isolated attention-fidelity or latency comparison.

Follow-up command: docker exec qwen38-functional-build bash
models/autoports/qwen_qwen3_8_27b/tests/validate_bf16_kv_geometry.sh. This covers
native-sized maps at lengths1,33,255,4095,4096,4097 (plus completed127), page-remap
replay at255, ragged batches4/8/16/32 with32768 capacity, and a four-real-layer
native262144 generator with262141 nonaligned prefill and explicit final decode
input position262143. The final-position probe checks finite logits and next
position262144; three generated tokens alone would not exercise that final slot.

All eleven numerical/replay cases pass (minimum PCC .999388134): native-map
lengths1,33,127,255,4095,4096,4097 plus ragged batches4,8,16,32 at32768 capacity.
Page-remap replay at255 passes. Summary kv_bf16_geometry_checks.json links each
raw result, capacity, actual chunk, and minimum score. These are reduced-layer
correctness controls, not full-model accuracy or candidate throughput.

Four-layer/full-head native generator probe passes all preparation and boundary
replay, plus262141-token prefill with repeated exact tokens and explicit final
position262143 decode. Logits are finite, resulting next position is262144, and
trace IDs/capture counts are unchanged. Evidence kv_bf16_native_replay.json.
Full64 retry is now running. Native-context capability is preserved by the repair;
these reduced checks do not substitute for the upcoming full-model gate.

Full64 native262144 retry passes all preparation, prefill and three traced
teacher-forcing repeats: top1=0.98,top5=1.0,top100=1.0,
median35.421242427t/s/u, TTFT122.967996ms.
Evidence results/kv_bf16.json includes effective K128 policy and source hashes.
The BF16 policy is slower than the BF8-cache baseline and has no observed
readiness-accuracy advantage; reject on measured comparison, not initial failure.
No DRAM limit or reduced advertised context is inferred from this rejection.
