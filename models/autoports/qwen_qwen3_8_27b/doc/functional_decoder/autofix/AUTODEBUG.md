# AutoDebug: Qwen3.8-27B GDN decode L1/CB clash

## Scope and observed facts

- Failing command: `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k real_weights_deltanet -s`.
- Reported behavior: real-weight GDN prefill and its PCC pass; the first decode reaches the post-attention `ffn_norm` and fails with `Statically allocated circular buffers ... clash with L1 buffers`. The reported dynamic L1 allocation begins at 1,447,616 bytes while the program's static CB region ends at 1,520,512 bytes.
- This was a source-only investigation. No TT device was opened and no hardware test was run.
- `git status --short` reports the repo-local `models/autoports/` tree as untracked. No preserved traceback log was present under the functional-decoder evidence directory at inspection time.
- The requested fresh Codex AutoDebug runner could not read the checkout because nested bubblewrap namespaces are disabled on this host. Its supported Claude backend was started, but was stopped before completion at the stage owner's request to prioritize this concise source report. The findings below were checked directly against current source.

## Headline finding: single-device decode forces both norm outputs into interleaved L1

The highest-probability cause is the unconditional single-device decode policy in `models/demos/blackhole/qwen36/tt/layer.py`:

```python
_attn_norm_config = _ff_norm_config = (
    {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
)
```

This policy applies equally to full-attention and DeltaNet layers and specifically makes the failing post-attention `ffn_norm` output L1-resident. `models/common/rmsnorm.py` first runs `ttnn.rms_norm` and then, when `output_mem_config` is set, calls `ttnn.to_memory_config(x, output_mem_config)`. Thus the decode path requests an additional interleaved L1 tensor at precisely the failing boundary.

The address ordering matches a normal L1 admission failure: an already allocated dynamic L1 buffer grows down from the top into the static circular-buffer range required by the next program. The named RMSNorm boundary is therefore likely the first program that cannot coexist with current live L1 allocations; it does not by itself prove an RMSNorm numerical or shape bug.

There is strong local precedent for DRAM placement at the analogous boundary. The multi-device prefill branch deliberately keeps `ff_norm` output in DRAM, with a comment that a full-width L1 norm remains resident across MLP matmuls and clashes with their CBs. The single-device branch does not carry that safeguard into decode.

### Important qualification about DeltaNet state

Source does **not** support the narrower claim that persistent DeltaNet recurrent/conv state itself is intentionally L1-resident:

- `gdn/state.py:init_recurrent_state` does not request `ttnn.L1_MEMORY_CONFIG`; default device allocation is expected to be DRAM.
- `split_fused_conv_state` explicitly clones split convolution-state buffers to `ttnn.DRAM_MEMORY_CONFIG`.

DeltaNet still changes the live-buffer/allocator state before `ffn_norm`, and transient GDN output, residual `h`, or temporary buffers may be the dynamic allocation reported at 1,447,616. A buffer-report experiment is required to identify the exact owner. The proven source discrepancy is the unconditional L1 destination for decode norms, not persistent-state placement.

## Ranked verify/refute experiments for AutoFix

### 1. Force only post-attention decode norm output to DRAM

Hypothesis: the `ffn_norm` L1 destination is the conflicting dynamic allocation.

Smallest experiment: in the single-device decode branch, keep `_attn_norm_config` unchanged but set only `_ff_norm_config` to `None` (default DRAM), or explicitly use `{"output_mem_config": ttnn.DRAM_MEMORY_CONFIG}`. Do not change GDN, precision, shapes, or MLP policy.

Prediction: the original real-weight command advances past `ffn_norm`. If it then passes decode PCC, the hypothesis is verified. If the next failure moves to an MLP matmul with the same clash, the norm placement was one contributor but the single-device decode MLP's L1 intermediates also exceed the available live-buffer budget. If it still fails at the same norm program and addresses, this hypothesis is refuted.

Evidence to capture: complete traceback, failing program/op name, allocation/CB addresses before and after, output memory config, and decode PCC. This is the preferred first fix candidate because it changes placement only and preserves math.

### 2. Isolate attention norm versus FFN norm placement

Hypothesis: L1 retained from the *input* `attention_norm`, rather than the new `ffn_norm` output, is the buffer that overlaps the later static CB region.

Run a 2x2 placement matrix for the two decode norms: `(L1,L1)`, `(DRAM,L1)`, `(L1,DRAM)`, `(DRAM,DRAM)`, with otherwise identical inputs/state. Deallocate behavior in `layer.py` already releases `attn_input` after attention, so `(DRAM,L1)` is expected not to help if deallocation has completed correctly; `(L1,DRAM)` should help under the headline hypothesis.

Verdict rule: only keep the smallest placement change whose row passes. This directly separates stale/lifetime behavior from the post-FFN output request.

### 3. Identify the allocation at 1,447,616

Hypothesis: a transient DeltaNet output or residual tensor, rather than persistent recurrent state, owns the high L1 allocation.

Use allocator/buffer reporting around: after GDN output, after `ttnn.deallocate(attn_input)`, after residual `h`, after `ttnn.deallocate(attn_output)`, and immediately before `ffn_norm`. Record tensor memory configs and device buffer addresses. Also record recurrent, fused-conv, and split-conv state memory configs/addresses.

Prediction: persistent state is DRAM, while one live transient or the requested norm conversion corresponds to the reported L1 range. If persistent state is unexpectedly L1 at runtime despite source defaults, that becomes a separate verified setup bug.

### 4. If DRAM FFN norm only moves the clash, force single-device decode MLP intermediates to DRAM

`models/demos/blackhole/qwen36/tt/mlp.py` selects `ttnn.L1_MEMORY_CONFIG` for all `T <= 512` gate, up, multiply, and down outputs. If experiment 1 advances into MLP and fails there, A/B only `mc = ttnn.DRAM_MEMORY_CONFIG` for `T == 1`.

Prediction: decode completes if cumulative MLP L1 intermediates/CBs are the remaining pressure. Do not batch this change with experiment 1 initially; the two causes need separate proof.

### 5. Compare DeltaNet and full-attention layer-0-equivalent decode with identical placement

Hypothesis: DeltaNet-specific preceding allocation pressure is required to expose the generic single-device L1 policy bug.

Run one-token decode for each meaningful layer kind using the same batch/hidden shape and norm placement. A full-attention pass with `(L1,L1)` and a GDN failure would confirm that the policy is only unsafe under GDN's live-set. Failure in both would point to a generic 27B single-device configuration mismatch inherited from the smaller Qwen36 implementation.

## Lower-ranked possibilities

- The 27B autoport reuses a single-device decoder path documented and tuned for the Qwen3.5-9B implementation. Real 27B hidden/intermediate widths make inherited L1 heuristics more suspect, especially `mlp.py`'s `T <= 512` rule, but that does not explain why the first observed failure is `ffn_norm` as directly as its explicit L1 conversion does.
- A missing `ttnn.deallocate` or asynchronous lifetime delay could leave `attn_output`/`attn_input` live. Source explicitly deallocates both before/at the norm boundary, so this should not be promoted without allocator evidence.
- Changing RMSNorm program config, dtype, fidelity, or GDN math is not a justified first experiment. The failure is capacity/admission, occurs after passing prefill PCC, and the placement A/B is both smaller and more discriminating.

## Recommended first repair candidate

Verify a model/layer-specific single-device decode policy that returns the post-attention norm in DRAM for the 27B GDN path. If it passes the original real-weight decode and PCC, keep that minimal change, then rerun traced decode and watcher separately. Only expand the change to the single-device MLP decode intermediates if the isolated norm fix moves, rather than closes, the L1 clash.
