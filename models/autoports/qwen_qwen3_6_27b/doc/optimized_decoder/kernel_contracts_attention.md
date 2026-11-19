# Delivered attention kernel contracts

Source-only audit of [optimized_decoder.py](../../tt/optimized_decoder.py),
SHA256 `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`.
All seven QKV/affine/beta reader, writer and compute source strings are
byte-identical to the exact-probed helpers below. Their three Python wrappers
are AST-equivalent after removing integration name prefixes. No actual bug
was found in these three kernels or their delivered call-site contracts.
This is a focused source audit, **not** an independent stage-review verdict
or a replacement for the final delivery gates.

| Delivered helper | Exact contract | Passed evidence |
|---|---|---|
| `qkv_split_batched_staging` | BF16 TILE DRAM `[1,1,B,8192]`, `1≤B≤32`, Q24/K4/V4 heads of256 elements. Outputs native logical/padded shapes and HEIGHT_SHARDED L1 `[32,256]` per user, first B cores in **eight-column row-major** order. Logical elements are bit-identical; unused head rows are deterministically zero. | [qkv165_probe.json](qkv165_probe.json):36 tensor cases = B1/3/31/32 ×3 changed inputs ×Q/K/V; exact native and CPU values, exact MemoryConfig/shapes, finite zero padding. [Probe](../../tests/run_qkv_batched_staging_probe.py), [frozen helper](candidates/qkv_split_batched_staging.py). |
| `_affinev3_gdn_affine_gate_flat` | BF16 normalized heads `[1,B*48,128]`, real BF16 gamma `[1,1,128]`, packed BF16 `[1,B,N≥16384]`; all TILE DRAM. Outputs BF16 `[1,B,6144]`. Preserves BF16 affine rounding, native FP32 SiLU on packed Z10240:16384, FP32 multiply and final BF16 rounding. Native head RMSNorm stays outside. | [affine170_probe.json](affine170_probe.json):16 cases = B1/3/31/32 ×N16512/17408 ×2 changed draws; all exact native-chain output, repeat equality and supplied-output address reuse. [Probe](../../tests/run_gdn_affine_gate_probe.py), [frozen helper](candidates/gdn_affine_gate_flat_v3.py). |
| `_gate178_gdn_beta_decay` | Packed BF16 `[1,B,N≥16496]`; FP32 dt_bias and uploaded `-exp(A_log)` `[1,1,48]`, all TILE DRAM. B16384:16432 → native BF16 sigmoid; A16448:16496 → native mixed-input **FPU add**, FP32 softplus(beta1,threshold20), FP32 multiply. Outputs beta BF16/g FP32 `[B,1,48]`, padded `[B,32,64]`. | [beta192_probe.json](beta192_probe.json):12 cases = B1/3/31/32 ×3 scales including threshold/saturation values; exact beta, g, add and softplus intermediates, changed-input/program-cache reuse, stable output addresses, repeat equality and finite deterministic padding. [Probe](../../tests/run_beta_decay_probe_v3.py), [frozen V4 helper](candidates/gdn_beta_decay_fused_v4.py). |

## Movement, ownership and lifetime

All three helpers register every input/output tensor in `ttnn.generic_op`,
with buffer addresses in runtime arguments. No measured-path torch conversion,
host numerical fallback, state replacement, or new KV/context restriction is
introduced. Scratch is per-program L1 CB storage; it does not persist across
operations. Output allocation happens through TTNN, including during trace
capture; supplied-output controls are checked by affine/beta isolated probes.
Whole-decoder trace allocation tracking remains a separate delivery gate.

QKV's two RISCs own disjoint horizontal face segments and separate aligned
scratch; each issues its read barrier before local scatter and its completion
barrier afterward. Padding zeroing precedes ordinary transfers. Odd users use
aligned64-byte reads and select the correct32-byte half. Core order matches
the delivered `_compact_cores`/`_compact_height_mem` caller, rather than the
physical device grid width. Native uninitialized padding is deliberately not
used as a comparison oracle; the helper's padding is fully initialized.

Affine's direct32-byte DRAM reads are used only when source/destination modulo64
alignments agree. Other rows stage64bytes and select the parity half. Gamma
reads initialize the row that the compute broadcast consumes. Output tasks
partition192 tiles without overlap; input/output buffer aliasing is rejected.
The caller supplies BF16 native-RMSNorm output and the unchanged BF16 gamma,
then immediately consumes the compact output in its output projection.

Beta/decay owns two output tiles per user on2Bcores, with no writer overlap.
A stays BF16 for native FPU addition; beta input and negative A use direct
FP32 unpack where consumed by SFPU. The final FP32→BF16 pack-format switch is
explicit. V4 processes both8-column groups of each top face (ITERATIONS2,
VectorMode::R), covering all32logical columns; unused padded rows may differ
from native padding but remain finite/repeat-deterministic. The recurrent
reader consumes only row0 and48heads. Diagnostic stages1/2 are probe controls;
the delivered caller always uses default stage0.

## Numerical pitfalls preserved as evidence

[V1 beta177](beta177_probe.json) passed beta but failed decay because precise
SFPU addition changed the native mixed-input FPU addition boundary.
[V2 beta180](beta180_probe.json) restored exact add, softplus and final decay;
[attribution](AUTOFIX_beta_decay_attribution180.md) explains why this should
not be described merely as truncating dt_bias (the checkpoint bias is BF16).
[V3 beta188](beta188_probe.json) failed half-column coverage with one SFPU
iteration/face. [V4 repair](AUTOFIX_beta_decay_row_v4.md) uses two iterations;
only the exact-passing V4 source appears in delivery194.

The kernels are specialized to this decoder's Blackhole geometry and supported
batch range. They do not claim arbitrary head dimensions, device architecture,
or packed-column layouts. Native prefill and public non-aligned sequence
length semantics are outside these decode helpers and remain covered by the
separate final delivery suite.
