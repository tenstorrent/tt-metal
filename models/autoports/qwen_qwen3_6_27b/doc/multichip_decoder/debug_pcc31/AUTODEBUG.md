# AutoDebug: layer 0 B31/S31 PCC failure

## Finding

**The new controls identify an attention-fidelity-dependent accuracy failure on the seeded synthetic inputs. The 96-row fused MMRS tail is not required to reproduce it.** No causal rank/layout error was established. HiFi4 is a single-policy intervention demonstrated to pass this fixture; this does not establish which attention operation contributes most to the failing user's error, or prove the low-level kernels bug-free.

Inspection only: no TTNN imports, device execution, tests, or implementation edits. Findings below use the parent's persisted runs and source inspection, independently of the older synthetic-stack explanation in `../anomaly_ledger.md`.

Revision: both focused files match the requested SHA256 values:

- `tt/multichip_decoder.py`: `082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44`
- `tt/optimized_decoder.py`: `315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22`

## Direct observations

All synthetic controls use the same input/decode/page-table hashes and single-chip reference metadata. Their command manifests record identical model sources and runner SHA `0095c650047bdb2b37b420a23f5a3a957510fd2c6f85471ef017319bb7df9ab1`. Comparing complete selected-policy dictionaries gives exactly the policy delta shown here.

| Artifact under `..` | Policy/input change | Prefill PCC | Decode PCC | Minimum prefill / decode per-user PCC | Result |
|---|---|---:|---:|---|---|
| `pcc31_default_synthetic.json` | Default LoFi | 0.9991919533 | 0.9993844293 | 0.9974781770 / **0.9946967452** | Fail |
| `pcc31_separate_synthetic.json` | Only `fused_mmrs=false` | 0.9991933783 | 0.9993844293 | 0.9974700657 / **0.9946967452** | Fail |
| `pcc31_hifi4_synthetic.json` | Only `attention_fidelity=HiFi4` | 0.9999783361 | 0.9999464966 | 0.9999424413 / 0.9992108423 | Pass |
| `pcc31_default_real_l0.json` | Recorded layer-0 inputs, default LoFi | 0.9999800431 | 0.9999933970 | 0.9999790118 / 0.9999919872 | Pass |

- Only zero-based **user 25, decode** fails the synthetic gate. Its prefill PCC is 0.9995349713. All outputs/state are finite; aggregate output, contribution, and state PCCs pass 0.995.
- Default and separate-MMRS runs have exactly identical **reported** decode per-user lists, aggregate decode PCC, and all four state PCCs. These are metric equalities, not a raw-tensor bit-equality comparison.
- Default prefill conv/recurrent state PCCs are `[0.9999104013, 0.9994195981]`; HiFi4 improves them to `[0.9999952209, 0.9999797068]`. Decode state PCCs improve similarly. These state scores aggregate users and heads; they cannot localize user 25's state error.
- HiFi4 synthetic and default recorded-input runs complete 25 trace replays, with checked replay outputs/state bit-equal to their eager controls. The failing LoFi runs stop before trace capture. Watcher logs contain no assertion/error diagnostic and end in device detach.
- Initial PCCs are computed from output and state reads to host (`run_multichip_decoder.py:154–187`); `ttnn.to_torch` calls `from_device` before conversion (`ttnn/ttnn/operations/core.py:464–467`). The evidence is materialized tensors, not merely the `PREFILL_DONE`/`DECODE_DONE` markers. Checked trace replays are blocking; queued timing replays are followed by device synchronization (`run_multichip_decoder.py:375,403–405`).
- The recorded-input run matches its newly generated baseline metadata, including activation-file SHA `461cc1e48f32cf37254d8e294b4c3fff273393e1e3a7544c63ff9b5448f61214`. `text_l0_b31_s31.json` describes users 0–30, prefill tokens 0–30 and decode token 31 selected from the existing recorded fixture. This is a limited prompt control, not full-model quality evidence.

The original failure is now archived as `pcc31_initial_sharded.log` plus its command manifest: it exits at the host per-user assertion after prefill/decode, with older runner SHA `82329af0...`. The new runner persists failures before raising; model sources match. The `delivery_watcher_*_l0_b31_s31` paths were subsequently reused for a recorded-input pass, so use the archived names for the original failure and `pcc31_synthetic_ref.pt` for further synthetic controls. Precise scores above come from the new artifacts, not an assumed reconstruction of the original assertion.

## Why the MMRS tail does not explain this failed phase

`MultichipDecoder._linear` selects fused MMRS only for registered row-parallel weights with **512 < rows <= 8192** (`multichip_decoder.py:225–230,354–399`). Compact decode is `[1,31,...]`, so its projections use the separate path (`:721–738`).

| Case | Logical rows | Padded rows | MMRS pieces | Final piece valid / zero rows |
|---|---:|---:|---|---|
| B31/S31 | 961 | 992 | 7×128 + 96 | 65 / 31 |
| B17/S33 | 561 | 576 | 4×128 + 64 | 49 / 15 |

Both tails preserve rank 3, tile layout, DRAM storage, mesh axis 1 and last-dimension scatter; the Python partition has no distinct 64-versus-96 branch. Concatenation trims to logical rows before restoring `[B,S,1280]` (`:361–399`). User 25's flattened rows `[775,806)` lie entirely in the full `[768,896)` piece, before the 96-row tail.

More decisively, prefill MMRS handles only the attention output projection and MLP down projection. GDN writes conv history and its working recurrent state **before** attention output projection (`:667,709,719`); `_finish` has no state writes (`:434–470`). Inherited prefill subsequently publishes the BF16 recurrent cache, but from that earlier working state, without consuming MMRS output (`optimized_decoder.py:1170–1171,1203–1206`). The harness uploads decode input independently and supplies saved GDN state, not prefill output (`run_multichip_decoder.py:145–187`). Thus MMRS output error has no intended dataflow into this layer's decode. Turning fusion off reproduces the failure and state metrics, as this dependency graph predicts. Hypothetical memory corruption is not logically excluded, but has no positive evidence here.

Sequence padding is a different boundary: S31→32 versus S33→64. Convolution receives `actual_end=S`; beta/g are zero-padded for chunk GDN (`multichip_decoder.py:639–709`). The same S31 boundary passes with HiFi4 and recorded inputs. B17/S33 is therefore a useful geometry contrast, not a matched-input proof about precision or recurrence padding.

## What the fidelity control actually changes

The explicit default `attention_fidelity=LoFi` overrides the initial GDN HiFi4 configuration (`multichip_decoder.py:61,97–121`). In `self.matmul_configs["attention"]`, the HiFi4 override retains FP32 destination accumulation, `math_approx_mode=False` and GDN `packer_l1_acc=False`. Native MMRS independently receives a descriptor specifying math fidelity (`:394`), with its other defaults unchanged. Weight dtypes, layouts and sharding do not change. The baseline GDN policy specifies HiFi4 (`optimized_decoder.py:39,162–170`).

Affected work includes packed primary/correction projection matmuls, the native prefill convolution, and attention output projection (`optimized_decoder.py:464–605`; `multichip_decoder.py:366–395,637,650–663,689,719`). Packed projection feeds QKV, Z, beta and decay, so changing fidelity can alter both saved state and current decode output. The recurrent update/chunk GDN and normalization configs remain fixed; the B31 decode convolution uses separate multiply/add/SiLU operations without this attention config (`multichip_decoder.py:623–633,675–707`). MLP fidelity is unchanged.

This is not just a passed-but-unused convolution option: in `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/`, the factory lowers `attrs.compute_kernel_config` (`qkv_causal_conv1d_silu_program_factory.cpp:474`), and `kernels/compute/qkv_causal_conv1d_silu.cpp:46–77` builds tap diagonals with broadcast multiplication, performs four-tap matmuls, then SiLU/pack. Conversely, carry is assembled from raw input rows (`kernels/dataflow/writer_qkv_causal_conv1d_silu.cpp:46–105`, `history_exchange_qkv_causal_conv1d_silu.cpp:95–112`). The improved conv-history PCC therefore places at least some precision-dependent discrepancy at the packed projection boundary, without proving it dominates user 25's final error.

**Inference:** the matched-input HiFi4 pass, unchanged failure without MMRS, upstream state improvement, and recorded-input LoFi pass strongly favor an input-sensitive precision-policy limitation over a deterministic MMRS-tail/rank error. The test's global averages conceal one user's gate miss. This conclusion does not depend on the earlier repeated-layer synthetic anomaly.

## Uncertainties and minimal further discrimination

1. **If this synthetic contract must pass, use the already-proven `attention_fidelity=HiFi4` override.** No layout rewrite or PCC-threshold relaxation is supported by these results. Whether production should retain LoFi requires the intended input/accuracy contract; this single recorded fixture is limited evidence.
2. **To localize inherited-state versus current-step error:** cross the two TP4 prefill snapshots (LoFi/HiFi4 conv and recurrent tensors) between fixed-policy decode instances, using the same `d`. Record user-25/head state errors and attention output before the MLP. This separates state propagation from current projection/output sensitivity; global state PCC alone cannot do so.
3. **Only if an operation-level diagnosis is needed:** isolate HiFi4 for packed primary/correction projections, prefill convolution, and output projection while holding the other effective configs fixed. A blanket fidelity pass changes all of them and does not by itself identify a defective primitive. Audit the selected native/kernel path if one isolated operation remains suspect.

## Other potential issues / claim review

No additional implementation bug was established in the inspected causal path. A 96-row native MMRS numerical defect, a general B31 rank-layout defect, trace corruption, or a benign low-level-math explanation should not be headlined: the first is unnecessary for the reproduced failure, the second lacks a source contradiction or matching control, trace capture has not begun, and hardware correctness was not proved. Global PCC can also hide per-user/head state differences; retain the existing per-user gate.
