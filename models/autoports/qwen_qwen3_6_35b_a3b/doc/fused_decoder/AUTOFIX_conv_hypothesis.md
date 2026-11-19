# Causal convolution hypothesis — inspection only

Scope: frozen `dram_bias_fixed.py.snapshot`; no device execution, implementation changes, or new numerical measurements. Source references below are relative to the repository root. HF source is `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` (abbreviated HF).

**Finding:** native convolution does omit a BF16 rounding boundary that the HF fallback has before SiLU. This is a plausible contributor to recurrent-state drift, but the supplied evidence does not establish that it causes the position-237 failure. Inspection finds matching carry chronology, channel splitting, and head grouping. The original functional decoder fails the same token despite using a different, unfused convolution; therefore a native-only layout/carry defect is a weak explanation of the shared failure.

The contract is PCC >= 0.995 for both decoder output and residual contribution across the 128 decode tokens following 128 prefill tokens (`tests/stress_fused_decoder.py:79-95`, under the model directory). Existing evidence reports fused output PCC 0.9928344317 and contribution PCC 0.9908002710 at position 237, identically over three repeats (`doc/fused_decoder/stress_dram_fixed_0.json:1-34`). At that position, `routing_probe.json:193-207` reports input-norm PCC 0.9999971791, attention PCC 0.9966506740, and post-norm PCC 0.9975423488. This localizes significant divergence inside attention, but does not isolate convolution from normalization, gates, or the recurrent update. The original functional decoder log ends with `(0,109,0.9939510142,0.9922532627)` (`doc/fused_decoder/logs/stress_baseline_0.log`).

The supported precision difference is:

- Frozen decoder lines 222-237 cast projected QKV to BF16, use BF16 taps (lines 62-71), and request HiFi4 with FP32 destination accumulation (lines 53-58). Native `ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp:64-78` accumulates all four tap matmuls in the destination, applies SiLU there, then packs. Its output is BF16 (`device/qkv_causal_conv1d_silu_device_operation.cpp:190-197`, same operation directory). In shorthand it computes approximately `BF16(SiLU(sum_fp32(x_j*w_j)))`.
- HF lines 230-234 call `F.conv1d` on the weight dtype and then `F.silu` on that returned tensor. For the BF16 checkpoint/reference this introduces the BF16 convolution-output rounding before SiLU: approximately `BF16(SiLU(BF16(sum_fp32(x_j*w_j))))`. The exact convolution accumulation implementation is backend-dependent, but the returned-tensor dtype boundary is explicit. This is a semantic precision mismatch, not evidence that the native kernel computes the mathematical convolution incorrectly.
- The functional baseline (`tt/functional_decoder.py:125-131`) separately multiplies/adds BF16 tensors before SiLU, introducing still more intermediate rounding. Replacing native convolution with that sequence is not a clean HF-rounding experiment.

Carry and layout agree on inspection:

- HF caches four projected tokens, concatenates the next token, computes valid convolution, and keeps only the final output (HF 227-234; prefill cache construction 483-485). That output consumes the newest three old tokens plus the new token. The frozen decoder caches exactly those three old tokens (`218`), concatenates QKV (`224`), and copies `history[:,s:s+3]` back (`239`). Convolution outputs never feed the convolution history. The padded 31 decode rows are excluded from both the logical output and history update (`226,239-242`), so convolution carry itself cannot accumulate 128 steps of arithmetic error.
- The native reader explicitly builds tap `t` from input row `r-3+t` (`device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp:6-10,149-168,223-235`). With `actual_start=0` and the same history supplied as predecessor carry, no chronology mismatch is apparent.
- The writer preserves contiguous Q/K/V channel order (`device/kernels/dataflow/writer_qkv_causal_conv1d_silu.cpp:119-134`). Widths 2048/2048/4096 and reshapes to 16/16/32 heads of dimension 128 match HF 499-512. Native recurrent wrapper `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/fused_recurrent_gated_delta_rule.cpp:85-96` expands each Q/K head with `repeat_interleave(G=2)`, matching HF 517-519.

There is also a distinct normalization precision mismatch that must stay fixed while testing convolution: HF fallback L2 normalization occurs before conversion to FP32 (HF 238-241,328-334); frozen decoder lines 245-249 convert Q/K to FP32 before normalization. Both decoder versions share this choice. HF then maintains recurrent state in FP32 (HF 332-360), as does the frozen decoder (`218,277`). Repeated small input/gate/normalization discrepancies can therefore propagate through recurrent state even though convolution history stores only projected inputs.

**Minimal discriminating experiment, not performed:** freeze the candidate and gate policy, then use the same BF16 projected QKV, three-token carry, and taps for two convolution outputs at selected positions (including 128, 236, 237): (A) native kernel; (B) a host diagnostic HF `F.conv1d` + `F.silu` on those exact inputs, transposed back and uploaded only for this diagnostic. Compare Q/K/V before L2, and check TT history equals the corresponding last three reference projected tokens. Run a second sequential branch from the same post-prefill recurrent state, replacing only decode convolution outputs with B through position 237; keep its projections, L2, gates, recurrence, and routing unchanged. Compare recurrent-state error, attention PCC, router IDs, and decoder PCC with branch A. This isolates convolution from projection and other attention calculations. If B does not materially improve drift, reject convolution as the dominant cause. If it does, distinguish the missing rounding boundary from SFPU SiLU error by comparing host FP32-sum SiLU versus host BF16-rounded-sum SiLU on the same inputs before implementing any kernel change. A fresh-state single-step comparison alone cannot establish the cause of long-sequence drift.

No device result or performance claim is made by this report.

## Follow-up: stock depthwise Conv1D as an HF-rounding diagnostic

The parent reports that an SFPU/addcmul recurrence repair now passes the 128-token stress, while a 1024-token run still fails at 869/876/1005. Those are parent-provided results, not measurements performed in this inspection. This follow-up examines a separate convolution rounding experiment while the parent tests HF BF16 L2 semantics.

**A single stock 8192-channel BF16 depthwise call is a poor diagnostic. A channel-chunked coalesced depthwise call is the feasible source-supported alternative.** There are two reasons:

1. Stock depthwise Conv1D supports only height sharding (`ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp:1275-1279`). Each core retains the complete channel width. At 8192 channels, a minimum 32-row BF16 activation block is 512 KiB; separate raw activation, tilized activation, weights, and output blocks place this well beyond a modest per-core L1 allocation. DRAM slicing along sequence length cannot shrink this channel-width floor. This is a source-derived resource concern, not an observed OOM.
2. Full-width 8192-channel, four-tap reads are 65536 bytes, exceeding Blackhole's 16384-byte coalescing limit (`conv2d_utils.cpp:42-51,581-604`). The resulting non-coalesced BF16 kernel packs each tap's partial in the **output dtype** (`device/kernels/compute_depthwise_conv1d.cpp:51-94`; `conv2d_op_program_factory_common.cpp:214-242`). Thus BF16 output introduces per-tap BF16 rounding, even with FP32 destination accumulation. It does not reproduce the desired single rounding after the convolution sum.

Use channel chunks of 256 (32 calls) for the first diagnostic. This keeps the four-tap read at 2048 bytes, safely inside both Blackhole and Wormhole coalescing limits. The coalesced BF16 kernel accumulates all taps before one output pack (`device/kernels/compute_depthwise_conv1d.cpp:242-275`). With `fp32_dest_acc_en=True`, BF16 inputs/weights, and BF16 output, this establishes the desired final BF16 preactivation boundary, followed by separate `ttnn.silu`. Chunking is mathematically independent across depthwise channels. Larger chunks can be evaluated later; performance is not established here.

Exact proposed setup wiring, executed only in `from_state_dict` (example for one supported logical sequence length `s`; decode has `s=1`):

```python
cc = 256
cfg = ttnn.Conv1dConfig(
    weights_dtype=ttnn.bfloat16,
    shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    act_block_h_override=32,
    output_layout=ttnn.TILE_LAYOUT,
    deallocate_activation=False,
)
slice_cfg = ttnn.Conv2dL1FullSliceConfig
# self.compute: HiFi4, math_approx_mode=False, fp32_dest_acc_en=True.
# Keep one prepared list for each supported s until shape independence is verified.
prepared = []
for lo in range(0, 8192, cc):
    raw = ttnn.from_torch(
        state_dict["linear_attn.conv1d.weight"][lo:lo+cc].reshape(cc, 1, 1, 4),
        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
    )  # host tensor: no device argument
    prepared.append(ttnn.prepare_conv_weights(
        weight_tensor=raw,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        weights_format="OIHW",
        in_channels=cc, out_channels=cc, batch_size=1,
        input_height=1, input_width=s+3,
        kernel_size=(1, 4), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
        has_bias=False, groups=cc, device=mesh_device,
        input_dtype=ttnn.bfloat16, output_dtype=ttnn.bfloat16,
        conv_config=cfg, compute_config=self.compute, slice_config=slice_cfg,
    ))
```

`prepare_conv_weights` actually moves the prepared tensor to the device (`prepare_conv2d_weights.cpp:1669-1672`); no runtime weight conversion/upload is needed. Its exact keyword binding is `conv2d_nanobind.cpp:181-205`. The existing prepared-weight Conv1D test supplies the same 2D preparation metadata (`tests/ttnn/unit_tests/operations/conv/test_conv1d.py:390-435`). Explicit L1-full slicing here avoids the default DRAM auto-slicing path selecting a different layout than preparation; use the same setting for both calls. `act_block_h_override=32` fixes the broadcast-weight block height. Prepare/warm the supported shapes before capture; do not lazily invoke host preparation inside runtime.

Corresponding runtime, only TTNN tensor operations:

```python
history = ttnn.concat([state["conv"], qkv], dim=1)  # [1,s+3,8192], BF16
parts = []
for j, lo in enumerate(range(0, 8192, 256)):
    inp = ttnn.to_memory_config(
        ttnn.to_layout(history[:, :, lo:lo+256], ttnn.ROW_MAJOR_LAYOUT),
        ttnn.DRAM_MEMORY_CONFIG,
    )
    preact = ttnn.conv1d(
        input_tensor=inp, weight_tensor=prepared[j], device=self.mesh_device,
        in_channels=256, out_channels=256, batch_size=1,
        input_length=s+3, kernel_size=4, stride=1, padding=0, dilation=1,
        groups=256, dtype=ttnn.bfloat16,
        conv_config=cfg, compute_config=self.compute, slice_config=slice_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    parts.append(ttnn.reshape(preact, (1, s, 256)))
conv = ttnn.silu(ttnn.concat(parts, dim=-1))
ttnn.copy(history[:, s:s+3, :], state["conv"])
# Existing Q/K/V channel splits and head reshapes follow.
```

API shape semantics are exact: `conv1d.cpp:38-56,90-104` maps input `[1,s+3,C]` to NHWC `[1,1,s+3,C]`, raw weights `[C,1,4]` to `[C,1,1,4]`, and kernel to `(1,4)`. Valid output length is `s`. Already prepared rank-4 weights pass through unchanged. All arguments are keyword-only (`conv1d_nanobind.cpp:63-87`). Direct `conv2d` with these equivalent 4D shapes reaches the same kernel, so it offers no precision/resource advantage.

An alternative precision control is FP32 input, FP32 prepared weights, and **FP32 convolution output**, then `typecast(..., BF16)` before SiLU. All three FP32 formats plus FP32 destination accumulation are required to enter stock convolution's SFPU path (`device/conv2d_op_sharded_program_factory.cpp:1073-1093`). Setting output BF16 immediately disables it. The proposed 256-channel chunk also fits FP32 coalescing (4096-byte read). This is a useful second control if the BF16 coalesced kernel differs unexpectedly, not required for the first rounding comparison.

The existing native `qkv_causal_conv1d_silu` has no activation-disable or preactivation-output option: its compute always applies SiLU before BF16 packing. Its channel-block distribution avoids the stock full-width L1 limitation. A native flag to round the completed sum to BF16 before SiLU would preserve that distribution, but needs an explicit kernel/API change; the stock chunked path above offers an experiment without changing native source. Source inspection establishes feasibility and rounding placement, not runtime correctness, trace compatibility, or performance for this exact shape.

## Follow-up: normalization, projection, and a confirmed cache precision mismatch

CPU-only inspection of `state_probe_1024.json.pos876.pt` found a stronger persistent-error mechanism: **the saved HF recurrent cache is BF16, while the TT cache is FP32**. This corrects the earlier statement about FP32 HF state: the recurrent function computes in FP32, but the caller's cache rounds that state after every forward. In installed `python_env/lib/python3.12/site-packages/transformers/cache_utils.py`, `LinearAttentionLayer.lazy_initialization` takes `self.dtype` from BF16 convolution state (line 926), allocates recurrent state using that dtype (937), and `update_recurrent_state` copies each FP32 result into it (989). `routing_probe.py:111-117` saves that actual cache, not an inferred or reformatted tensor. HF forward invokes this cache update at model source line 548. TT keeps new state in FP32 (`tt/fused_decoder.py:248,312` in the inspected live version).

This is a confirmed mismatch of the installed reference's sequential semantics, although the amount of error it causes still requires the parent's controlled run. The minimal isolated experiment is to round only the persisted TT recurrent state through BF16 after prefill and each single-token decode, then restore FP32 storage if needed: `copy(typecast(typecast(new, BF16), FP32), state)`. Continue computing the current token's output from the unrounded new state, matching HF's order. Do not round at each internal chunk of one HF prefill call: HF cache rounding happens at the forward-call boundary, not within its internal chunk recurrence. Preserve the prior SFPU recurrence fix and hold all other changes fixed.

CPU tensor substitution also isolates the observed attention error from output gated normalization. I loaded only four layer-0 checkpoint tensors with `safetensors.safe_open`: input norm weight, gated norm weight, Z projection weight, and output projection weight. All four are BF16. On CPU, I reconstructed Z with `F.linear(saved_input_norm, Wz)`, applied the exact HF gated norm below to saved core, and projected with `F.linear(..., Wo)`. The reconstructed HF attention equals the saved HF attention **bit for bit**, validating this local replay against the saved execution.

| CPU replay variant at position 876 | PCC versus saved HF attention | Relative L2 versus saved HF attention |
| --- | ---: | ---: |
| Saved HF core, exact HF gated norm, HF-input Z, CPU output projection | 1.0000000000 | 0 |
| Saved TT core, exact HF gated norm, HF-input Z, CPU output projection | 0.9878096235 | 0.1559769 |
| Saved TT core, exact HF gated norm, TT-input Z, CPU output projection | 0.9879128512 | 0.1557570 |
| Saved TT core, TT-style norm rounding, TT-input Z, CPU output projection | 0.9879179145 | 0.1557199 |
| Saved HF core, TT-style norm rounding, HF-input Z, CPU output projection | 0.9999947947 | 0.00326975 |
| Saved HF core, exact HF gated norm, TT-input Z, CPU output projection | 0.9999789988 | 0.00936351 |

“TT-style” here models the visible Python rounding order with CPU FP32 arithmetic; it does not claim to reproduce device RMSNorm or SiLU approximation. Exact HF gated norm is `n = BF16(core)`, `u = n.float()*rsqrt(mean(n.float()**2)+1e-6)`, `v = weight_BF16*u.BF16`, and `out = BF16(v.float()*SiLU(z.float()))`. HF source lines 191-200 explicitly round normalized core before the BF16 weight multiply, and use FP32 SiLU for the gate. TT's generic `norm` applies FP32 weights before BF16 casting (`228-237`) and its recurrence output enters that norm as FP32 (`313-316`); those are real secondary differences. Nevertheless, the CPU replay shows that correcting these output differences cannot remove the dominant error already present in the saved TT core.

The apparently high aggregate state/core PCC hides severe error in individual small heads. Saved core aggregate PCC is 0.9993778526 and relative L2 0.0356640. Per-head core relative error is 0.4430 for head 0, 0.7034 for head 14, and 0.6034 for head 15; their HF core RMS values are 0.00064546, 0.00025275, and 0.00036020 respectively. Headwise RMS normalization gives these small heads much more weight relative to the larger heads that dominate aggregate PCC. Exact HF gated normalization applied to TT core reproduces almost all the bad attention PCC (0.98781 versus the saved ~0.98785). The output normalization therefore **amplifies upstream error**, rather than its arithmetic mismatch being the main source of the discrepancy.

Input norm matches HF's intended formula on inspection: HF normalizes in FP32, multiplies by `(1+weight.float())`, then casts to BF16 (HF 823-831); TT uploads `1+weight.float()` and follows that same order in generic `norm`. CPU recomputation from the saved input token reproduces saved HF input norm bit for bit. Relative error versus saved TT input norm is 0.00293484, PCC 0.9999968604, max absolute error 0.015625. That small device error can feed all projections and recur, but there is no visible Python-level semantic-order error in input normalization. Exact Z projection on saved TT input norm accounts for only a small part of the local attention error (last table row). The saved tensors do not contain pre-convolution QKV projections, so they cannot isolate the persistent impact of projection errors. The BF16 recurrent-cache boundary is the first discriminating experiment to run before broadening projection/norm changes.

Validation performed: CPU-only PyTorch replay using the saved position-876 tensors and the pinned local checkpoint. No device calls, new hardware results, or runtime-source edits were made.

Parent follow-up confirms the cache-boundary hypothesis: `stress_cache_boundary_1024_0.json` passes all 3072 outputs across three bit-identical 1024-token repeats, with minimum output PCC 0.998485278 and minimum contribution PCC 0.998021592. These are parent-reported device results. The tested implementation preserves the FP32 storage buffer and prefill computation, and applies `BF16 -> FP32` rounding to recurrent state **only when reading it as the next single-token decode input**. This is equivalent to HF's persisted BF16 rounding at decode boundaries while avoiding accidental rounding between TT's internal prefill chunks. It also preserves the current token's unrounded FP32 state for computing its output. This result and the CPU substitution evidence establish cache precision as the dominant remaining cause; output gatednorm rounding is secondary. The parent is separately retesting native FDR under the corrected boundary, so this memo does not assert that the earlier SFPU recurrence replacement remains necessary.
