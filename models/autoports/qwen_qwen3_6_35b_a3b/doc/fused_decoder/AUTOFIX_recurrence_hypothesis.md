# Recurrent-update hypothesis inspection

Scope: inspection only, 2026-11-12. Compared the frozen `dram_bias_fixed.py.snapshot`, the original functional decoder, installed Transformers `qwen3_5_moe/modeling_qwen3_5_moe.py`, and native recurrent/chunk GDN sources. No target execution, hardware access, source edits, or device claims. This is a targeted AutoFix hypothesis check, not a stage review.

## Finding

The native recurrent **equations and head mapping agree with HF**. I found a concrete shared precision mismatch before the recurrence: TT normalizes Q/K entirely in FP32, whereas this installed HF Torch fallback normalizes its BF16 Q/K before widening them to FP32. A second mismatch is that TT feeds FP32 recurrent output directly into gated RMSNorm, while HF rounds that output back to the original query dtype (BF16) first. These are supported semantic differences, not proof of the position-237 failure's cause.

The evidence is consistent with a shared attention discrepancy being amplified by routing: `routing_probe.json` reports input norm PCC 0.999997179, attention 0.996650674, post norm 0.997542349, and final output 0.992834432 at position 237. The native selected expert set differs from HF at that position. Neither these data nor the equations establish that error grows monotonically over the 128 decode steps.

## Source evidence

- HF `l2norm` at lines 238–241 evaluates `(x*x).sum(...) + eps`, `rsqrt`, and multiplication without an FP32 cast. Both Torch GDN paths call it before `.to(torch.float32)` (chunk lines 256–262; recurrent lines 328–334). With the BF16 convolution output used by this reference, intermediate tensor results remain BF16. The reference layer is loaded with checkpoint dtypes and called with BF16 inputs; `tests/reference.py` does not replace normalization. No installed `fla` package directory was found in the inspected Python environment, so the Torch fallback is the expected path; confirm the actual callable before a runtime A/B if needed.
- Frozen candidate lines 244–249 casts Q/K to FP32, then uses RMSNorm with weight `128**-0.5` and epsilon `1e-6/128`. This is algebraically the correct sum-based L2 normalization, but does not reproduce HF's BF16 operation boundaries. Original `tt/functional_decoder.py` lines 136–141 likewise casts before square/sum/rsqrt/multiply. Thus this mismatch can explain a discrepancy shared by baseline and fused candidate; it is not specific to RMSNorm fusion.
- HF recurrence lines 338–366 uses scale `1/sqrt(128)`, decays state before reading it, computes `(v - k @ decayed_state) * beta`, applies the rank-one update, reads output from the updated state, retains FP32 state, and casts output to `initial_dtype`.
- TT `fused_recurrent_gated_delta_rule.cpp` lines 79–103 scales FP32 Q once, expands 16 Q/K heads into 32 value heads with consecutive `repeat_interleave(..., 2, 0)`, applies `exp(g)`, and preserves beta. The compute kernel's loop follows HF's update order. All its circular buffers are FP32, and the candidate passes HiFi4, FP32 destination accumulation, non-approximate compute configuration. State is copied into the persistent FP32 `[1,32,128,128]` buffer. No BF16 state truncation is evident at the Python/C++ storage boundaries.
- The stress run invokes the recurrent kernel with **T=1 per decode step**. Its internal multi-token ping/pong state schedule is therefore not exercised across the 128 calls; persistence is through the returned state and Python `ttnn.copy`. The one-core-per-value-head reader uses head index `h`, consistently with the consecutive Q/K duplication and state layout.
- TT wrapper output stays FP32. Frozen candidate lines 277–283 reshapes it and calls `self.norm` without an intervening BF16 conversion. HF returns BF16 from the recurrence before `Qwen3_5MoeRMSNormGated` (lines 185–200). That gated norm also rounds the normalized activation to BF16 **before** multiplication by the BF16 weight. TT generic `self.norm` applies the FP32 weight inside RMSNorm and only then casts to BF16. This downstream rounding difference is shared with the functional decoder.
- Prefill has another relevant boundary: `chunk_gated_delta_rule.cpp` lines 35–44 and 202–212 forcibly cast Q/K/V to BF16, even when Python normalized Q/K in FP32. Decode does not perform that cast. HF's Torch prefill and decode both perform BF16 L2 normalization before widening. Comparing only decode with a TT-generated initial state cannot distinguish inherited prefill discrepancy from continuing decode discrepancy.

## Minimal discriminating checks

Parent-reported experiment update received during inspection: BF16 A/B projection plus BF16 sigmoid beta gave minimum PCC 0.991647 (`stress_hf_ab_0.json`); casting only the FP32-normalized Q/K to BF16 gave 0.992101 (`stress_qk_bf16_0.json`), versus original 0.992834. All fail only at 237. The exact primitive BF16 L2 test below is already running in the parent. These reported results do not establish the cause and show that a final Q/K cast alone is insufficient.

1. **Q/K normalization boundary A/B, separate from the parent's A/B/beta experiment.** From identical BF16 post-convolution Q/K, compare HF `l2norm` against TT's current FP32 normalization. For an exact fallback-boundary experiment implement the BF16 square, BF16 sum result, BF16 epsilon addition, BF16 rsqrt result, and BF16 final product; casting only the final FP32 normalized tensor is a weaker experiment. Run one recurrent step from identical imported HF state and compare next state and core output, then repeat the fixed 128-step trajectory. Keep the reference unchanged. Improvement in state before any norm/router would directly localize this cause.
2. **Core-output boundary only.** Cast recurrent `o` to BF16 immediately before the gated norm, leaving state untouched. This is a very small isolated change. If attention improves but recurrent-state error is unchanged, it supports an output rounding cause rather than recurrence drift. If required, separately test the HF gated-norm order (FP32 unweighted normalization → BF16 → BF16 weight product → FP32 SiLU gate multiplication → BF16), without bundling it with Q/K changes.
3. **State attribution if those do not decide it.** At prefill completion and immediately before position 237, capture HF and TT recurrent states. For the failing token, use identical Q/K/V/g/beta and identical imported state for both implementations. Compare decayed state, delta, next state, and core output. Separately compare TT with HF initial state versus TT initial state; this separates a kernel discrepancy from inherited state error and current projection/convolution/normalization differences. Inspect per-head decay and state error before claiming 128-step accumulation.

There is no source-supported reason here to change decay order, query scale, beta placement, state orientation, or head duplication. Finite normalized keys and beta in [0,1] make the ideal state update's linear factor `exp(g) * (I - beta*k*k^T)` non-expansive for nonpositive g; slow-decay heads can retain input/rounding differences, but the recurrence does not intrinsically imply explosive drift. Actual device arithmetic remains unmeasured by this inspection.

## Follow-up: FP32 storage does not imply FP32 state arithmetic

New parent-reported state probe evidence: state relative L2 error rises from 0.01137 at 128 to 0.093996 at 236 and 0.095104 at 237 (state PCC 0.998645); core-output relative L2 error is 0.104 at 237, while current-token Q/K/V errors remain about 0.005. This makes repeated state-path precision loss a substantially stronger hypothesis than the original input-boundary hypotheses. I did not execute that probe or independently measure these numbers.

**The native fused kernel definitely has TF32 input boundaries under the inspected JIT configuration.** My earlier statement about FP32 CBs and accumulation described storage/destination types only and must not be read as full FP32 precision throughout the recurrence.

Exact source chain:

1. `fused_recurrent_gated_delta_rule_program_factory.cpp` gives every CB `DataFormat::Float32` and passes `fp32_dest_acc_en`, but does not set `unpack_to_dest_mode`.
2. `tt_metal/jit_build/genfiles.cpp:1065–1088` selects conditional unpack destination `Tf32` for all-FP32 CBs with FP32 accumulation. `tt_metal/jit_build/data_format.cpp:179–220` overrides this to `Float32` only for an explicitly non-default per-CB unpack-to-dest mode. This kernel has none, so its generated unpack destinations are TF32.
3. State decay calls `mul_tiles_bcast_scalar`. `api/compute/bcast.h:403–424` lowers that to `llk_unpack_AB<SCALAR>` and FPU `ELWMUL`; both state and decay enter source registers in the generated TF32 format. Thus even a perfectly computed FP32 `exp(g)` loses low mantissa bits at multiplication. HiFi4 does not restore the omitted input bits.
4. `rank1_update` seeds destination with `copy_init(sd); copy_tile(sd, ...)`. `api/compute/tile_move_copy.h:49–76,126–134` routes through unary unpack/datacopy. The WH LLK direct-to-destination condition is additionally format-dependent (`cunpack_common.h:210–218,245–248`; `llk_unpack_A.h:126` and `llk_math_eltwise_unary_datacopy.h:54`). TF32 destination format does not satisfy the full 32-bit input condition. Therefore this copy also passes through source precision, truncating the FP32 product before adding the rank-one update.
5. Both `k @ sd` and `q @ S_new` likewise use ordinary matmul source unpacking; FP32 destination accumulation does not preserve full-precision state operands there either. The FPU residual subtraction also uses TF32 sources. This is broader than a single faulty copy.

The reduction implementation independently documents precisely this hardware behavior in `reduction/generic/device/reduce_op.cpp:198`: “the FPU truncates fp32 to tf32 on the way into SrcA/SrcB.” The relevant LLK format is e8m10; its FP32 footprint has the lower 13 mantissa bits absent. Repeated truncation of positive decay near one biases decay downward, and truncation of newly computed state magnitudes can accumulate between real updates. The exact per-step bias and its contribution to the observed 9.5% error still require measurement; source alone does not prove a universal 0.1% shrink rate.

### Identity-test interpretation and smallest stronger check

The parent's proposed `q=k=v=0, beta=g=0` test is useful for detecting a first-step FP32-to-TF32 loss. However, **a plateau after the first step is expected for pure idempotent TF32 truncation when exp(0) is exactly 1**: once the state lies on the TF32 grid, multiplication by one and another copy need not lose more bits. A plateau would not refute error accumulation during nontrivial decay/update.

Add the same zero-update test with a fixed small negative g whose `exp(g)` is not TF32-exact, e.g. g = -0.001, and compare against `S0 * exp(n*g)` at 1/32/128 steps. Capture the actual TT `exp(g)` once and form a second host expectation from that captured value, to separate exp approximation from FPU operand truncation. Inspect norm ratio or fitted scalar gain, in addition to PCC and relative L2: scalar shrink can leave PCC near one. A model-local FP32 SFPU recurrence should pass both controls if the hypothesis is right.

Parent-reported identity result received after completing this trace: `recurrent_identity_probe.json` has one-step relative L2 0.0008281423, norm ratio 0.99930918, max absolute error 0.00048825; 32/128 calls retain the same quantized value. This confirms non-identity state arithmetic and the predicted plateau. The nonzero-decay check is running in the parent. The source path above identifies TF32 input narrowing specifically; the identity result alone does not identify which of the several narrowing boundaries dominates.

### Model-local TTNN recurrence is a supported discriminating implementation

For decode, reshape/duplicate normalized Q/K to value-head order and keep all tensors FP32. With state `S=[1,32,128,128]`, `kc=[1,32,128,1]`, `qc=[1,32,128,1]` (query already scaled), and `v,beta=[1,32,1,128]` / `[1,32,1,1]`, express:

```python
sd = ttnn.multiply(S, decay)
vread = ttnn.sum(ttnn.multiply(sd, kc), dim=-2, keepdim=True)
delta = ttnn.multiply(ttnn.subtract(v, vread), beta)
new = ttnn.add(sd, ttnn.multiply(kc, delta))
o = ttnn.sum(ttnn.multiply(new, qc), dim=-2, keepdim=True)
```

This avoids matmul and FPU scalar broadcast for the carried state. Current `binary_ng_device_operation.cpp:51–65` selects SFPU for FP32/FP32 add, subtract, and multiply; `binary_ng_program_factory.cpp:1371–1401` explicitly requests `UnpackToDestFp32` for their input/interim CBs. Generic FP32 sum defaults to `fast_and_approximate_mode=False`; `reduce_op.cpp:198–217` selects FP32 SFPU on WH/BH with FP32 destination accumulation, and `reduce_op_multi_core_w_program_factory.cpp:480–487` explicitly bypasses SrcA TF32 truncation. Use FP32 operands on **both** sides, retain FP32 outputs/state, and use FP32 destination accumulation for the sum. For binary FP32 operations, leave `fast_and_approximate_mode` unset: explicitly passing false is validated as a BF16-output-only feature in this checkout, though FP32/FP32 still selects SFPU by dtype. This is source-supported, not a measured guarantee of the whole proposed graph.

For a native repair, merely setting all CBs to `UnpackToDestFp32` is not sufficient/safe: existing AB matmul/FPU operations still require source-register inputs, and `cb_sd` is consumed both by matmul and by the state-seeding copy. The state-preserving operations need an SFPU/direct-destination path with appropriate per-CB/per-operation configuration, or separated buffers. The model-local decomposition is the cleanest isolated hypothesis test under the current no-native-edit scope.

## Follow-up: existing KDA fused recurrence alternative

Parent-reported primitive-control results: 384 real stress outputs pass with minimum PCC 0.999099 and repeat-bit-equality; 128-step nonzero-decay relative L2 is 1.3e-6. These results were supplied by the parent, not independently executed here.

**There is a feasible existing fused native replacement to test:** `ttnn.experimental.kda.recurrent_chunk_scan` accepts manually prepared one-real-token/32-physical-token terms. Unlike the original GDN kernel, it explicitly preserves the full FP32 base state, updating it in complement form. It still narrows the *increment calculation* through FPU source registers, so it is not a full-FP32 equivalent of the successful primitive control. No native `ttnn.experimental.kda.recurrent_step` binding exists in this checkout: the KDA binder lists only chunk scan and the other chunk/prefix operations, and the searched `recurrent_step` matches are the new model-local helper/tests.

### Exact single-token wiring

Use `BH=32`, `N=1`, `K=V=128`, `groups_per_head=1`. Duplicate normalized Q/K heads consecutively to 32 heads. Below q already includes `1/sqrt(128)` and k is normalized; v is the current value, beta is already sigmoid, g is the scalar log decay per value head, d=exp(g). Allocate all seven terms as interleaved TILE FP32. For each head, construct:

| Argument | Logical shape | Nonzero contents |
|---|---|---|
| `v_beta` | `[32,1,32,128]` | row 0 = beta * v |
| `kd` | `[32,1,32,128]` | row 0 = beta * d * k |
| `q_decay` | `[32,1,32,128]` | row 0 = d * q |
| `intra` | `[32,1,32,32]` | element [0,0] = dot(q,k) |
| `k_dec_t` | `[32,1,128,32]` | column 0 = k |
| `final_decay` | `[32,1,128,1]` | every key row = expm1(g) = d - 1 |
| `t_inv` | `[32,1,32,32]` | identity matrix, constant |

Every unspecified element is zero. Compute `intra[0,0]` with FP32 elementwise multiply plus sum; `ttnn.expm1` is bound in this checkout and avoids explicitly subtracting one from a near-one exp result. Preallocate row/column selectors, identity, and scalar-bound tensors at setup; use device TTNN tensor operations to form runtime terms. Reshape persistent state to `[32,128,128]` and pass it as both `group_entry_states` and mandatory `tail_entry_states` (the latter is ignored for this unsplit one-chip call). Pass stable replicated UINT32 ROW_MAJOR `actual_start=[0]`, `actual_end=[1]`, an explicit FP32-accumulating HiFi4/non-approximate compute config, and DRAM output memory. These bounds describe this isolated one-token operation and remain 0/1 across decoder positions. No absolute decoder-position dependence enters scalar-GDN math. The manually zeroed rows already form identity steps; `actual_end` does not construct the zeros for you.

The call returns BF16 output `[32,1,32,128]` and FP32 state `[32,128,128]`. Extract output row 0 and restore `[1,1,32,128]`; reshape/copy state back to `[1,32,128,128]`. This changes the core-output boundary to BF16, matching the installed HF return dtype; compare against the primitive control with that boundary explicitly accounted for.

Algebraically the kernel computes `U = beta*v - beta*d*k*S`, `Y = d*q*S + dot(q,k)*U`, and `Snew = S + (d-1)*S + k^T*U`, exactly the desired single-token GDN equations before finite-precision effects. The other 31 rows contribute no state updates.

### Why this path preserves the base state, and what it does not preserve

- Factory `recurrent_chunk_scan_program_factory.cpp:289–318` explicitly assigns `UnpackToDest` to input state, tail state, and `state_carry`; `state_ring` and computational terms explicitly use `UnpackToSrc`.
- Kernel `copy_exact` at lines 213–236 seeds both ring and carry through full-FP32 direct-destination copies, and lines 512–515 emit final state through the same exact path.
- `update_state` at lines 248–335 computes only `(expm1(g)*S + k_dec_t@U)` in DST, enables `pack_reconfig_l1_acc(1)`, and accumulates that increment onto the existing full-FP32 state in L1. It avoids repeatedly replacing the large base state with a TF32-truncated copy. This is specifically the mechanism missing from the original native GDN kernel.
- The kernel comments explicitly acknowledge truncation of S when forming the *small decay increment*. Residuals, dot products, and rank-one increments still use FPU source precision. For slow decay this should substantially reduce systematic base-state shrink, but real-model accuracy remains to be measured. Identity should preserve state exactly; the nonzero-decay check may have a small, bounded-per-update error larger than the primitive control.

`prepare_chunk_recurrence` is **not** a transparent shortcut for constructing these terms: it requires BF16 flat Q/K/V/per-key gates, applies its own Q/K normalization and beta sigmoid, has no GQA expansion, and requires physical T divisible by 32. It would introduce several fresh numerical changes. Manual seven-term construction avoids those changes, at the cost of runtime preparation work and 32-row matmul work for a single token. There is no source-only basis to claim this adapter is faster than the current primitive recurrence; benchmark only after identity, nonzero-decay, shared-input step, and real stress accuracy pass.

## Follow-up: flat native chunk prefill and legacy sequential attention

### Flat `chunk_gated_delta_rule` is directly compatible with aligned prefill shapes

The public operation's rank-3 path accepts the raw causal-convolution outputs without model-side head reshape, Q/K RMSNorm, or GQA duplication. For B=1, S=32 or 256, use raw, **unnormalized and unscaled** BF16 TILE tensors `q,k=[1,S,2048]`, `v=[1,S,4096]`, existing log-decay `g=[1,S,32]`, already-sigmoided `beta=[1,S,32]`, and existing FP32 TILE state `[1,32,128,128]`:

```python
o, new = ttnn.transformer.chunk_gated_delta_rule(
    q_raw, k_raw, v_raw, g, beta,
    initial_state=state["recurrent"], output_final_state=True,
    chunk_size=32, num_key_heads=16, use_qk_l2norm=False,
    compute_kernel_config=self.compute, **self.constants,
)
```

The seemingly contradictory `use_qk_l2norm=False` is required: the public wrapper rejects true unconditionally, but **rank 3 itself enables in-kernel normalization**. Do not normalize Q/K before this call or multiply Q by the scale. Default scale is `128**-0.5`; epsilon is fixed at sum-based `1e-6` in both phased and fused factories (`chunk_gdn_phased_program_factory.cpp:330–332`, `chunk_gdn_fused_program_factory.cpp:295–297`). H=16 comes from `num_key_heads`, K=2048/16=128; HV=32 comes from beta's last dimension, V=4096/32=128. Explicit `num_key_heads=16` is clearer even though the equal K/V inference also produces the right geometry.

`chunk_gated_delta_rule.cpp:145–255,333–335` requires S divisible by 32 and chunk size 32 for flat Q/K. Thus S=32 gives NC=1 and S=256 gives NC=8. Fused and phased implementations accept flat input; mono rejects it. Preserve the existing compute config (HiFi4, FP32 destination accumulation, no approximate mode). Pass all four setup-created constants (`eye`, `tril`, `ones`, `masks`) and explicit state so the wrapper performs no fallback host uploads during trace capture. Keep the rank-4/primitive path for genuine nonaligned tails or single-token decode; do not pad a logical tail into a state-advancing flat call without separately proving all padded steps are identity.

GQA mapping is exactly consecutive duplication: `reader_chunk_gdn_prep.cpp:154–171` sets `hk=hv/(HV/Hk)`, so value heads 0/1 read key head 0, 2/3 read key head 1, etc. Its tile row stride is `Hk*Kt=64` for Q/K, and its key-head column offset is `hk*4`, matching the raw concatenated head layout of the convolution output. No physical head duplication is needed.

Default output remains FP32 ROW_MAJOR `[1,S,32,128]`, exactly as for the current rank-4 chunk call, with FP32 final state `[1,32,128,128]` (`chunk_gated_delta_rule.cpp:469–515`). Keep the existing output reshape/tilize/gated-norm path. `output_head_major=True` would return `[32,S,128]` TILE and avoid the native output permutation, but the current downstream expects token-major order; do not enable it as part of this isolated input-fusion test.

### Numerical changes that the prefill A/B must measure

This is algebraically the same L2 normalization but not bit-identical to the current RMSNorm route or HF BF16 primitive normalization:

- Flat preparation computes the diagonal of raw BF16 `x @ x.T` with FP32 accumulation, adds `1e-6`, applies SFPU rsqrt, and folds query scale into that reciprocal norm. It multiplies a diagonal normalization matrix by x (`chunk_gdn_math.hpp:426–494,808–830`). This computes per-row sum-of-squares over the four K tiles, not a mean or a norm over the full 2048-channel width.
- Normalized Q/K scratch (`supd`, `stmp`) is FP32 (`chunk_gdn_phased_program_factory.cpp:232–270`), but the diagonal matrix and subsequent matrix operands pass through FPU source precision. This differs from the current path's FP32 RMSNorm followed by the wrapper's BF16 Q/K cast and BF16 query-scale boundary. In particular, flat preparation does **not** deliberately round the normalized Q/K tensors to BF16 first.
- Gates and beta keep their existing FP32/BF16 input dtype and values; the flat path neither computes sigmoid nor changes the log-decay convention. V remains the same BF16 convolution output.

The minimal useful A/B compares aligned prefill output **and final recurrent state** from identical raw convolution outputs and entry state for S=32 and 256. Then run the existing sequential stress with the proven primitive decode, because a changed prefill state can affect later accuracy even when its immediate output PCC passes. Hold program path/WY inverse selection fixed when isolating normalization from scheduling. No runtime or performance result was obtained by this inspection.

### `gated_delta_attn_seq` has a direct precision blocker

It cannot solve the diagnosed state corruption without native changes. `gated_delta_attn_program_factory.cpp:193–202` explicitly gives **every consumed FP32 buffer** `UnpackMode::UnpackToSrc`, including input/persistent state and the intermediate state. The compute kernel at `gated_delta_attn.cpp:403–447` performs `S_tmp = S * dl_exp` using FPU `mul_tiles_bcast_scalar`, packs that result, then computes `S = S_tmp + s_upd` using FPU `add_tiles`. Consequently both the carried state entering decay and the decayed state entering addition traverse TF32 source-register boundaries. There is no separate direct-destination carry or complement-form update like KDA `recurrent_chunk_scan`.

It also accepts only `C=K=V=128` (`gated_delta_attn_device_operation.cpp:43–51`), with precomputed block-triangular solve inputs, eight FP32 DRAM TILE term tensors, and four diagonal inverse blocks. It rejects C=32 or C=1 outright. Padding a single token to C=128 would not remove its precision blocker and would add large unnecessary work. The existing successful FP32 primitive decode remains the appropriate control; KDA `recurrent_chunk_scan` is the source-supported carry-preserving fused alternative identified above.

## 1024-token follow-up: HF cache is BF16, despite FP32 recurrence math

**New supported diagnosis: the installed HF reference rounds its recurrent cache to BF16 after every model call. The TT repaired path keeps it FP32.** This is a separate semantic difference from the old native kernel's unintended TF32 arithmetic. My earlier statement that HF retains FP32 state described the recurrence function's returned tensor, and was incomplete: the cache object immediately casts that tensor on storage.

Source and saved evidence:

- All four `state_probe_1024.json.pos{128,869,876,1005}.pt` files contain `hf_values['state']` in **torch.bfloat16**, while `tt_values['state']` is **torch.float32**. This is directly inspected CPU tensor metadata, not inferred from source annotations.
- Installed `transformers/cache_utils.py:922–940`: `LinearAttentionLayer.lazy_initialization(conv_states=...)` sets `self.dtype = conv_states.dtype`; later recurrent initialization uses `torch.zeros_like(recurrent_states, dtype=self.dtype)`. The BF16 convolution cache therefore determines BF16 recurrent-cache storage even though the recurrence returns FP32.
- `cache_utils.py:976–990` copies each incoming recurrent tensor into that preallocated BF16 cache. The HF model updates convolution cache before running prefill recurrence (`modeling_qwen3_5_moe.py:479–485`), and always calls `cache_params.update_recurrent_state(last_recurrent_state, ...)` after recurrence (lines 546–548). Decode widens the stored BF16 state inside the Torch recurrence, computes in FP32, then the cache rounds it again.
- Current `FusedDecoder` instead copies FP32 `new` directly into its FP32 persistent buffer. This computes a more precise but different long trajectory than the actual oracle. Exact Q/K/A/B rounding experiments do not address this state-storage boundary.

### CPU discrimination using the saved tensors

I ran only CPU Torch/safetensors code through `python_env/bin/python`; no TTNN/device execution or runtime edits. CPU `q @ saved_TT_post_state / sqrt(128)` matches the saved TT core output with relative L2 between 1e-7 and 3e-7 and PCC 1.0 at all four positions. The core read/reduction is therefore numerically consistent with the stored state.

At position 876, substituting HF state while keeping TT query reduces core relative L2 from 0.035664 to **0.005636**; substituting HF query while keeping TT state leaves **0.033556**. At position 1005 those values are 0.007217 and 0.050756. The dominant error is inherited state, not the current query or the final state-to-output reduction.

At 876, heads 14 and 0 account for 23.0% and 18.6% of state squared error. Their relative state errors are 0.039079 and 0.054083, and fitted TT/HF gains are 0.976164 and 0.965247. Both have small negative g, so prior state persists. Current g differences are only -5.86e-7 and +2.68e-6. Current input differences alone are much smaller than accumulated error: a conditioned inverse/reapply check from the saved post-state yields one-step all-TT-input error about 0.000476 across stable heads, versus accumulated 0.029604. This inverse reconstructs a synthetic prior state because the saved HF post-state is BF16-rounded; it is a local sensitivity check, not an exact recreation of the missing pre-state.

For a stronger cache-only discriminator, I regenerated the exact seed-913 prefix/tokens, loaded only input norm and attention QKV/A/B/conv/gate weights, computed canonical CPU BF16 inputs, seeded both trajectories from the saved HF state at position128, and ran FP32 recurrence with either FP32 cache or BF16-round-after-each-step cache through position1005. The sampled token tensors match the saved ones exactly. At the four saved positions, regenerated g/beta match exactly; Q/K/V mostly match exactly, with the largest listed relative mismatch 1.53e-5 (query at1005), caused by the batched projection/conv reconstruction versus the original per-token calls. The recreated BF16 trajectory is therefore a close numerical control, not a bit-exact oracle recreation.

| Position | FP32-cache vs BF16-cache state relative L2 | CPU FP32-cache vs saved TT state relative L2 | Cosine alignment of cache-only error with observed TT−HF state error | Recreated BF16 vs saved HF state relative L2 |
|---|---:|---:|---:|---:|
| 869 | 0.028580 | 0.007396 | 0.919888 | 0.009303 |
| 876 | 0.028443 | 0.007521 | 0.917891 | 0.009388 |
| 1005 | 0.026891 | 0.007031 | 0.921682 | 0.008430 |

Thus changing only cache storage precision on canonical inputs recreates roughly the observed 2.8–3.0% state-error scale and strongly aligns with its direction. This is substantially stronger evidence than attributing the remaining drift to fused EXP/addcmul. Ternary source inspection also shows FP32 operands use direct-destination unpack (`ternary_program_factory.cpp:1483–1526`); it does not reveal a repeat of the original TF32 carry bug. A shared-input device step remains useful if this cache-boundary fix leaves errors, but it is not the first experiment now.

### Minimal repair experiment

Keep the FP32 SFPU recurrence and its FP32 intermediates. **After the current token output has been computed, round only the state being persisted through BF16, then widen back to the existing FP32 buffer**, e.g. conceptually `copy(typecast(typecast(new, BF16), FP32), persistent_state)`. Apply the same rounding after the logical prefill cache update. This preserves the buffer geometry and avoids reintroducing FPU TF32 corruption. Do not compute the current core output from the rounded cache: HF computes its output from the unrounded FP32 new state and only rounds the persistent state independently. Do not round internal 32-token scan states if HF updates the cache only once for the enclosing prefill call; respect the HF call boundary when prefill is partitioned.

Run the original 128-in/1024-out sequence with this cache boundary change isolated first, then consider combining the remaining input-boundary experiments if needed. Do not modify the canonical reference to retain FP32 state merely to make the comparison pass. No device correctness claim for this proposed repair is made here.

Parent-reported resolution: rounding the consumed cache `BF16 → FP32` at entry to each decode (with the public buffer kept FP32) passes all **3072 outputs**, minimum output PCC **0.998485278**, minimum contribution PCC **0.998021592**, and repeated bit equality in `stress_cache_boundary_1024_0.json`. This is equivalent to HF's preceding cache-write rounding for sequential decode, including the initial decode after prefill, while preserving the current token's unrounded new-state output calculation. Inspecting the unrounded public TT buffer after the step will still show a small dtype-boundary difference from HF's saved cache; compare consumed/rounded states when auditing recurrence equivalence. Continuation prefill must independently apply the same consumed-state rounding at its HF call boundary if it uses this public unrounded buffer.

The parent is separately testing native FDR with that boundary. BF16 state at entry is exactly representable in TF32, so the first narrowing of the stored state is harmless, but FP32 decay factors, decayed state, and rank-one/residual intermediates still traverse the native FPU narrowing points. The successful cache fix does not by itself prove native FDR safe: retain the full original/endurance gates, shared-input step check where necessary, and do not infer native validity solely from BF16 being a subset of TF32. This subsection reports parent-supplied device evidence; this investigator performed CPU/source work only.
