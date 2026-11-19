# Embedding byte adapter: bitcast narrowing and model-only alternative

Source/CPU investigation only. Device observations below are from the root-owned completed `gate_embedding_probe.json`, which remains unchanged (SHA256 `91fdb14847680df568a93d2aedefe0d9e742c2115ab4daa33f301e854553373b`). No decoder/native edits or device runs were performed here.

## The byte candidate is rejected as currently implemented

Every one of the **144 recorded first-difference examples** equals the expected FP32 bit pattern masked with `0xffffe000`: exactly the bottom 13 bits disappear. For example decay `0x3f7fd17f` becomes `0x3f7fc000`; gate `0xba3a1405` becomes `0xba3a0000`. This is the TF32 input-unpack signature, not a wrong head/index or ordinary BF16 rounding.

The existing UINT32→FP32 `ttnn.bitcast` supplies a direct source explanation:

- `eltwise/unary/unary.cpp:56–60` enables `preserve_fp32_precision` only when the **original input dtype** is FLOAT32. UINT32 input enables 32-bit destination but leaves that flag false.
- `eltwise/unary/device/unary_program_factory.cpp:541–542` changes the input DFB metadata to the **output** format for BITCAST, so the UINT32 word bytes are consumed as Float32.
- The same factory, `:743–748`, selects UnpackToDest only with the preserve flag. Otherwise a Float32 DFB under a 32-bit destination explicitly uses **UnpackToSrc**, losing the low 13 mantissa bits before the identity copy.

Changing output allocation or memory layout cannot set that private preserve flag. Chaining BITCAST with another unary op is explicitly rejected (`unary/device/unary_device_operation.cpp:174–180`). Numerically casting UINT32→FP32 interprets the word as an integer and cannot replace reinterpretation. No exposed model-only compute-config override was found for this bitcast primitive. Shared native edits are outside scope.

No intermediate device capture was available in the failed report, so the actual first divergent intermediate remains to be observed. The probe now supports `--encoding bytes --capture-intermediates --skip-trace`, saving selected bytes, numeric bytes, assembled UINT32 words, immediately post-bitcast FP32, and final reshape output for capture 128. This distinguishes a correct integer assembly followed by bitcast narrowing from an earlier failure. The source and all recorded examples strongly predict the former.

The failed adapter also measured approximately **102.4 us decode**, versus the root-reported exact ROW_MAJOR lookup at **68.7 us**. It is not a production candidate even apart from accuracy. S=128 measured about 291 us per table. The failed report and log were not overwritten.

## Attempted repair without native changes

`tests/gate_embedding_probe.py` now defaults to `--encoding scaled_bf16` and a new output `gate_embedding_probe_repaired.json`. It avoids UINT32→FP32 bitcast entirely. Setup stores four BF16 values per gate:

```
value = high + (middle + low) * inverse_scale
```

`high` truncates the FP32 word to its upper BF16 bits. The residual is scaled by 2^32 only for |value|<2^-100, otherwise by 1. `middle` is the truncated BF16 part of that scaled residual; `low` is the remaining BF16 component. The inverse scale is exactly BF16-representable. This scaling matters: a naive three-BF16 decomposition loses FP32 subnormal residual bits. Infinite gate outputs are stored as high=inf and signed-zero residuals; negative zero signs are retained in all zero addends.

The embedding/gather geometry and 16 MiB/table footprint are unchanged. After selection, runtime widens the four components to FP32, adds middle+low, multiplies by inverse_scale, then adds high. It removes the three shifts, three ORs, and final narrowing bitcast in favor of two adds and one multiply. The original BF16-input→UINT16 index bitcast remains; it has no Float32 input-DFB reinterpretation and is not implicated by this source failure.

CPU validation records **zero FP32 code differences across 65,280 finite BF16 inputs × 32 heads for each gate/decay table**, all nine canonical captures, and S=1/S=128 selection/reassembly geometry (`gate_embedding_probe_scaled_bf16_cpu.json`). This is a CPU representation proof, not proof that device arithmetic preserves every subnormal. The existing byte CPU checks also still pass (`gate_embedding_probe_bytes_recheck_cpu.json`).

Minimal root-owned candidate run:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.gate_embedding_probe --decode-only --trace-kinds decay --capture-intermediates
```

The probe skips trace timing if any canonical capture differs. Keep ROW_MAJOR lookup unless this alternative is device-exact and faster than the measured 68.7 us; no speed claim is made here.

## Frontend diagnostic setup kept current

`linear_frontend_stream_probe.py` now reads the actual ordered-weight assignment AST and replays its `padded.reshape(...).permute(...)` packing. It uploads `[4,512,32,416]` and `[2,1024,32,416]` for the current source, supports the historical permuted packing, and no longer creates `precise_attention_weight` when the current `linear` method does not use it. CPU metadata checks cover both layouts; no decoder import/device setup was used for those checks.

## Repaired-candidate device follow-up: reject exact embedding replacement

The root-owned `gate_embedding_probe_repaired.json` now contains the completed device check, SHA256 `4f7c6b463e432ad478426b513e45c31b91a3d29bce256071a2bd48ca6cd8e93b`. It remains unchanged. All nine **gate** captures are FP32 bit-exact. Each **decay** capture has exactly one error, head 22: a positive subnormal around `1.6925e-40` becomes zero. Timing was correctly skipped. These values remain failures under the exact-code requirement; their small magnitude is not a waiver.

The first captured divergent stage is `selected`, after embedding plus head-column gather, before numeric BF16→FP32 widening or component reconstruction. At position 128, head 22's four stored components should be:

```
[9.183549615799121e-41, 3.3125995043460457e-31,
 1.2638915260041235e-33, 2.3283064365386963e-10]
```

Captured `selected` instead has zero in the first slot, with the other three slots exact. The first BF16 component's raw word is `0x0001`, the smallest BF16 subnormal. The existing capture does not distinguish upload, embedding tilization, and subsequent gather as the first individual operation that flushes it. In particular, the embedding path performs a compute tilize (`embedding/device/kernels/compute/tilize_chunked.cpp:34–62`), so exact BF16 numeric representation on the host does not imply preservation through that device path.

There is a **second loss after selection**: reconstructing from the actual captured components on CPU yields `7.742174015394614e-41`, whereas captured `before_reshape` is already zero. Thus restoring just the missing BF16 high component cannot fix the whole arithmetic path. The normal middle/low values and inverse scale multiply down into a subnormal; the native SFPU multiply is plain `in0 * in1` (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:221–243`), without a software subnormal reconstruction path. The captures establish flushing somewhere in this post-selection widening/add/multiply/add sequence, before final reshape. Final reshape introduces no additional captured error.

I considered scaling the entire value before encoding, rather than only its residual. That would remove the subnormal high component from the table, but still requires a final arithmetic operation producing the same subnormal output and therefore does not address the demonstrated post-selection loss. Returning to raw byte reconstruction meets the separate UINT32→FP32 bitcast narrowing blocker above. A per-subnormal raw FP32 lookup fallback would add the already-working lookup machinery to the embedding adapter, undermining the performance purpose rather than yielding a simple repair.

**Recommendation: stop this embedding candidate and keep the exact ROW_MAJOR FP32 lookup.** There is no inspection-supported simple model-only retry that addresses both observed losses while plausibly improving the existing 68.7 us lookup. No additional device run, decoder edit, acceptance relaxation, or overwrite was performed. Subsequent QZ/state-projection frontend-diagnostic adaptation is intentionally deferred until the root's candidate settles.
