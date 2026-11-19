# Prefill convolution and Q/K boundaries

Source-only artifacts, 2026-11-12:

- `candidates/prefill_hf_helpers.py`: vectorized convolution and external HF-style normalization helpers.
- `candidates/make_prefill_hf_candidate.py`: source transformation with exact match assertions; imports no device modules.
- `candidates/prefill_hf_boundary_v2.py`: recommended frozen boundary32 candidate, including explicit neutral-tail padding.
- `candidates/prefill_hf_boundary.py`: preceding snapshot using native rank4 padding, retained as evidence.

Only docs/candidates were written. Main decoder and shared native sources remain untouched. No device execution or C++ compilation was performed by this helper.

## Hypothesis and exact changes

The stage owner reports that CPU capture attributes the remaining long-decode error primarily to incoming recurrent state. Existing native prefill convolution applies SiLU before the BF16 output boundary, and flat native Q/K use internal FP32 normalization. Installed HF eager code instead evaluates `F.silu(self.conv1d(...))`, with BF16 convolution output, and evaluates `l2norm` on BF16 Q/K before converting recurrence operands to FP32. Sources: `python_env/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:240`, `:259`, and `:497`.

`prefill_hf_conv=True` replaces only multi-token GDN convolution. The helper concatenates existing RM `[B,3,10240]` carry and raw current tokens, forms four shifted length-S views, multiplies BF16 inputs/taps into FP32, adds the four products in order, explicitly casts the sum to BF16, then evaluates BF16 SiLU. Operations are vectorized over B and S; the Python loop has four iterations for the fixed filter taps. It neither loops over users/tokens nor calls torch or a host fallback. Layout conversion is at the boundary between row-major convolution arithmetic and the native chunk's tiled input.

Carry always becomes the last three **raw** rows of history+tokens, using `[S:S+3]` in that concatenation. It is copied into the original allocation before internal output padding. For S1/S2, the remaining old rows are retained correctly. The decoder dispatch still leaves its selected one-token decode path intact; the helper itself supports every positive logical S.

`prefill_hf_qk_norm=True` uses rank4 Q/K `[B,S,16,128]`, with BF16 square, BF16 sum output, BF16 epsilon add, BF16 rsqrt output, and BF16 normalization product. Rank4 V is `[B,S,48,128]`. The native public wrapper automatically disables internal Q/K normalization for rank4 inputs, performs the head split and GQA expansion, and retains its fused/phased kernel dispatch. Legal call arguments are `chunk_size=32`, `use_qk_l2norm=False`, `num_key_heads=16`, `output_head_major=True`, caller-owned constants/state, and the existing compute config. `use_qk_l2norm=True` is explicitly rejected by native source and must not be used as an external-normalization switch.

The v2 helper explicitly pads Q/K/V, beta, and g to the next32-token boundary with `ttnn.pad`. Beta=0 and g=0 make tail steps state-neutral. This avoids native wrapper `pad_time_tile` creation of zeros and its output untilize/slice/tilize branch. Final logical S is exposed through the existing logical/padded shape reshape. There is no public alignment restriction. Both B1 and B32 remain legal because public native dispatch chooses wave-capable fused execution when the head count exceeds the grid.

## Q scale caveat and control

The native rank4 wrapper casts Q/K/V to BF16 and multiplies Q by `scale` before entering the kernel. HF eager converts normalized Q to FP32 before scaling. Default `prefill_scale_output=True` therefore passes native `scale=1.0` and multiplies its FP32 output by `128**-0.5`; recurrent state is independent of Q and is unaffected by moving this scale. This avoids an extra BF16 scaled-Q boundary but changes FP32 rounding order, so output equivalence must be measured rather than claimed bit-exact. Set `prefill_scale_output=False` to measure the legal standard native pre-scaling control. Neither choice changes K or recurrent-state arithmetic.

## Generation and ablations

```bash
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_prefill_hf_candidate.py \
  --source models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/boundary32.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/prefill_hf_boundary_v2.py
```

Enable the two policy flags independently, then together, using the existing frozen-candidate harness. `prefill_scale_output` is an additional output-only control. The transformation prepares RM filter taps during decoder initialization when needed, even if the batched decode convolution policy is disabled, and removes unused native-convolution padding from the replacement branch.

## Validation performed

AST parsing passed for helpers, transformation script, and both frozen outputs. A CPU-only PyTorch test covered B1/B3/B32 and S1/2/3/31/33/129 with eight channels for mapping: FP32 four-tap accumulation followed by BF16 rounding and SiLU was bit-equal to CPU BF16 grouped `F.conv1d` followed by SiLU in all18 cases, with max absolute error0. Segmented continuation using1/2/5/31-token pieces produced identical output and final carry. This validates the proposed boundaries and indexing on CPU; it does not establish TTNN numerical parity or latency.

Required device evidence: prefill output and persisted-state PCC for each ablation, selected long-decode capture against incoming-state attribution, full512-step stress, non-aligned continuation and batch coverage, watcher/trace checks, and warmed prefill/decode latency. No threshold is lowered. Native chunk FPU/reduction differences remain a separate possible source of residual state error; a successful boundary correction must be demonstrated by those gates.
