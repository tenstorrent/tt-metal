# Endurance gate-rounding hypothesis

Inspection and CPU-only analysis, 2026-11-12. No TTNN imports, device execution, runtime edits, or test edits. Frozen source: `../mlp_norm_shard_dram.py.snapshot`, SHA256 `6c552444ffb96d002fdbdb8be37d07e55c380fba350ca9be33d9ee622c9d50fb`, matching `mlp_norm_shard_8192_0_results.json.provenance.json`.

## Finding

HF A/B projection and beta rounding boundaries remain mismatched after the BF16 recurrent-cache repair. A CPU experiment on the saved 8192-token input stream, changing only those gate boundaries with BF16 cache behavior held fixed, produces about **1.2% state relative L2 difference** late in the run. At position3979 it lowers attention PCC to **0.995616** even though core PCC stays **0.999977**. This is sufficient to justify a new isolated device A/B now that the cache is fixed. It is **not proof that gate rounding alone causes all seven failures**: the CPU gate-only attention PCC at worst device position7703 remains 0.999817, and the saved endurance artifact has no internal state/gate tensors with which to attribute the actual discrepancy.

## Exact source mismatch

Installed HF `modeling_qwen3_5_moe.py:461–465,514–516` computes BF16 `a = in_proj_a(hidden_states)` and `b = in_proj_b(hidden_states)`, followed by:

```text
g = -exp(A_log.float()) * softplus(a.float() + dt_bias)
beta = sigmoid(b)                  # BF16 output
```

The Torch recurrence widens beta after that BF16 sigmoid boundary. Therefore the canonical decode sequence is:

```text
A projection → BF16 → FP32 → add dt_bias → FP32 softplus/g
B projection → BF16 → sigmoid → BF16 → FP32 recurrence
```

Frozen F lines291–292 computes one packed FP32 projection plus a BF16 copy. However lines321–329 take a/b from the **FP32** `projected`, add dt_bias to a only in decode, and retain FP32 sigmoid(b). The already-existing `projected_bf16` is used for convolution/z, not these gates. The decode `linear` path explicitly uses `bias=None` (lines218–231), so a is still the raw projection when extracted: it can be rounded correctly **before** the explicit dt_bias add. The prefill path differs: it fuses dt_bias into `attention_packed` via `attention_bias`, so simply casting that prefill a slice would round the wrong sum. Do not inadvertently double-add bias or bundle a prefill projection rewrite with the decode-only discriminator.

The cache repair correctly rounds the consumed recurrent state at public decode boundaries (lines332–335). It does not repair these independent gate boundaries.

## Prior controls do not refute the corrected-cache hypothesis

- `../stress_hf_ab_0.json`: fails position237, minimum output PCC0.991647. This was before the major recurrence/cache repairs.
- `../stress_gates_exact_1024_0.json`: fails869/876/1005, minimum output PCC0.988365. Its `gates_exact.py.snapshot` has FP32 persistent state with no BF16 consumed-cache boundary; it therefore tests gates while the now-proven cache mismatch remains active.
- `../stress_cache_public_1024_0.json`: passes, minimum output PCC0.998485. This fixes cache behavior but retains the gate mismatch.

No supplied prior control combines canonical decode gate rounding with the repaired cache on the new 8192-token sequence.

## CPU experiment and practical limits

Used `python_env/bin/python` with Torch CPU (4 threads) and safetensors. Loaded saved `prefix` and `tokens` from `mlp_norm_shard_8192_0_results.json.tensors.pt`; loaded only layer0 input-norm, QKV/A/B/Z/conv/gate/norm/output-projection checkpoint weights. The artifact contains prefix, tokens, final reference outputs, and final device outputs only; there are no raw recurrent inputs or states.

Constructed canonical BF16 input norm and Q/K/V convolution, including BF16 HF L2 arithmetic. Ran two CPU recurrent trajectories with identical Q/K/V, identical FP32 recurrence equations, and BF16 cache storage after each decode. Both used identical canonical gates throughout the 128-token prefix; only decode gates differ:

1. Canonical: BF16 A/B projections; BF16 sigmoid beta.
2. Gate-only surrogate for F: FP32 A/B matmul results; FP32 sigmoid beta.

The prefill state was computed by sequential CPU recurrence with only a final BF16 cache rounding, identically for both trajectories. This is an isolated mathematical experiment, **not** a bit-exact replay of HF chunk-prefill or the TT packed-matmul/SFPU implementation. In particular, it excludes TT projection error, TT normalization/convolution differences, matmul memory-read differences, and router discontinuities. Core output was computed from each unrounded new state, then passed through canonical HF BF16 core boundary, gated norm, and output projection to compute the attention PCC below.

| Position | State relative L2 | Core relative L2 | Core PCC | Attention PCC |
|---|---:|---:|---:|---:|
| 128 | 0.000151 | 0.000083 | 0.999999997 | 0.999999504 |
| 1151 | 0.013607 | 0.009899 | 0.999960882 | 0.999832792 |
| 3979 | 0.011865 | 0.006841 | 0.999977463 | 0.995616118 |
| 5011 | 0.011997 | 0.008637 | 0.999963998 | 0.999206064 |
| 6595 | 0.012089 | 0.006509 | 0.999984104 | 0.999935628 |
| 7551 | 0.012136 | 0.008404 | 0.999982574 | 0.999834102 |
| 7668 | 0.012477 | 0.014231 | 0.999931321 | 0.999853813 |
| 7703 | 0.012302 | 0.013451 | 0.999913570 | 0.999816981 |
| 8157 | 0.012842 | 0.007624 | 0.999977830 | 0.999941584 |

At3979, gate-only instantaneous g relative L2 is 3.886e-5 and beta relative L2 is0.001701. At7703 these are0.000132 and0.001957. Small gate differences affect a BF16-quantized recurrent trajectory; the state discrepancy is not monotonically increasing after1024 and the output sensitivity depends strongly on the token. Consequently “late failure” does not by itself establish increasing arithmetic drift or a stale-state defect.

## One discriminating device change

Change **only decode A/B/beta rounding**, leaving the successful BF16 cache boundary and all current recurrence/convolution/norm/routing code intact. Use the existing `projected_bf16` slices (decode packed projection has no bias):

```text
if decode:
    a = FP32(projected_bf16[..., 12288:12320])
    b = FP32(projected_bf16[..., 12320:12352])
    a = a + dt_bias
    g = existing FP32 fused softplus/A_log expression
    beta = FP32(BF16(sigmoid(b)))
else:
    retain existing prefill gate behavior for this isolated test
```

Computing sigmoid from widened rounded b then explicitly rounding the result avoids relying on a BF16 SFPU output path whose rounding may differ from Torch; FP32 sigmoid approximation itself remains a separate small discrepancy. The proposed casts may improve fidelity without changing tensor geometry or requiring another projection.

Replay the **entire fixed 8192 decode stream from the same prefill**, because replaying only the failing token with a newly initialized state cannot test this hypothesis. Compare the seven failure positions, the global output/contribution minima, and failure count against frozen F. A pass is evidence for this gate repair in the tested configuration; partial improvement should be assessed with the parent's state probe rather than immediately bundling unrelated changes. If the cache-correct old42ac control also fails, that further supports revisiting shared gate semantics; if only current F fails, the gate mismatch may instead be a background sensitivity exposed by a newer change.

No device result or guaranteed final fix is claimed here. The strongest supported conclusion is that these canonical rounding boundaries remain absent and are quantitatively large enough to deserve an isolated corrected-cache test.
