# Isolated Q-centering streaming integration

This copies the qualified P8 header privately but keeps P in FP32. Private
changes add a separate QK correction pass and one CB20 pop per key chunk.
No frozen/shared header is edited.

Q128/H1/D128, K512 chunks, double-buffered Q/K/V and correction. Q is
device-mean-centered then RNE7/BF16. K/V use RNE5/native-BFP8 or native-group
RNE/BFP4. Correction is HiFi4/FP32-DST mean-Q times ORIGINAL BF16 K, unscaled.
The reader loads16 FP32 correction tiles per K512 chunk and math reuses them
for all four Q-tile rows. CB20 has32 pages, permitting two buffered chunks.

The first full-sync implementation failed even with zero correction and all
copy/add/reinit work bypassed. Capping its logical DST size at4 did not fix it;
the corresponding half-sync skip-add control passed about2.193% L2. Thus full
sync is not accepted for this private algorithm; its underlying failure remains
unresolved. `--full-sync --no-center --skip-add` only retains a diagnostic.

The current implementation defaults to **half-sync**. Width4 QK first packs
its ordinary four FP32 score tiles. An explicit PACK-to-UNPACK semaphore then
fences those stores before a separate correction pass reads two scores through
the existing FP32 CB7 alias and two CB20 corrections into the four DST slots.
PACK-thread SFPU adds slots2/3 to0/1 and repacks only scores, twice per QK block.
The extra score spill is FP32, not BF16. Direct-DST unary loads avoid the SrcB
mantissa limit. CB7 stays at the fixed64-page Q128 arena base; CB6 consumes
and wraps the complete arena every K step. This assumption is deliberately not
generalized to arbitrary Q sizes.

Only `blocked_matmul_and_pack<transpose=true>` runs this pass. It asserts
QCB0/KCB1/scoreCB6, width4/height1 and physical half-sync capacity4. The caller
signals max-reduce readiness only after corrected scores are packed. PV is
unchanged. Both physical SrcA=K/SrcB=Q formats and the complete no-MOP math
program are restored, as is the width4 pack configuration expected by hot
callers. The raw add resets ADDR_MOD7, uses fixed constant1, and does not
overwrite the pack-thread exp grid's programmable constants or replay.

The kernel uses the unbiased cheap exp fit and LoFi denominator reduction;
PV and denominator consume the same truncated SrcB bits of FP32 P. Maxima
remain BF16 and recurrence FP32. This is not the ACCURATE mode, and large
common Q can still expose max/subtraction, probability, and value error.

`q_center_streaming.build(device, original_cpu_inputs, kv_format='b8',
ncores=4, center=True, mean_mode='bf16_fpu')` returns output, attention callable,
preprocessing callable, combined callable, and metadata. Preprocessing
includes device mean, centering, high-precision correction, and K/V packing.
The CLI reports their cost separately and together. `--no-center` keeps the
same attention injection/dataflow but supplies RNE7 original Q and device
zero correction, as an explicitly labeled algorithm control.

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/q_center_streaming.py \
  --label qcenter-integrated-normal-v1 --length 1024 --iters 0
```

The producer currently has a reported ~0.0304% normal correction matmul L2,
which exceeds the original 0.01% gate and is under investigation. The default
gate remains0.01%; to run a deliberately relaxed diagnostic, explicitly add
`--max-correction-l2 0.1` and retain that setting in the result. No claim of
producer qualification is implied. Run matching common_q, structured, and
`--no-center` controls; then BFP4. Also compare `--no-center` against
`--no-center --skip-add`: a zero correction should preserve output, and this
comparison tests the extra FP32 spill/load/add/pack pipeline. The final reference is FP64 attention on
ORIGINAL BF16 Q/K/V over all128 query rows. Source hashes include the private
header and transitive producer/preprocessor sources.

No device/JIT job was run by the author of the half-sync repair. Python/static
checks only; the new postpass requires device qualification by the card owner.
Reject nonfinite output before any timing. Existing full-sync failures are not
valid performance measurements.
