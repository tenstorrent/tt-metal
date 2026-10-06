# AutoDebug: full-attention BFP8 replica divergence

Source-only investigation. No agents, imports, hardware, experiments, resets, installs,
or runtime/test edits were used for this report.

## Evidence Read

- Runtime under investigation: `tt/multichip_decoder.py`
  `070613ddc32b5cc8a22fd92a64cb541de9a9f27152852f1caad0843d1d06903d`.
- Current ordinary runner hash:
  `c531a9a2f00db1eb9b2135f9b4397cbdf5a388db50776c17b6d4c670899b36ff`.
- Current diagnostic runner hash:
  `a56768b654dc8d2778dbb70cb1df1164cff82672536ca90489230a1a457096cc`.
- `full_router1_bfp8_failure_detail.failure.json`: paired TP1->TP4 ordinary run,
  layer5 full attention, BFP8 attention CCL, trace, check-cache. TP4 fails at
  step41 / position4137 on the first replay read. All values are finite; only
  rank1 differs from rank0, with 6 changed BF16 output values and max diff
  0.0029296875.
- `full_bfp8_ordinary_paired2.failure.json`: same current runtime/runner and
  same paired shape fails again, now at step65 / position4161 on the duplicate
  replay. Only rank3 differs, with 8 finite changed values and max diff
  0.0078125.
- `full_bfp8_ordinary_tp4.json`: current runtime/runner ordinary run with only
  `--tp 4` added passes all 128 traced decode positions for exact replica and
  duplicate-replay checks. It has no TP1 oracle, so `passed`/PCC are null by
  design.
- `full_bfp8_output_only.json`: standalone TP4 diagnostic using the original
  `MultichipDecoder` class, BFP8, passes 128 output-only duplicate checks.
- `full_bfp8_outer_boundaries.json`: standalone TP4 boundary diagnostic, BFP8,
  retaining expected replicated boundaries and padding views, passes 128 checks
  with no physical-only variation.
- `full_router1_ccl_bf16_control.json`: ordinary paired BF16 attention CCL
  control passes 128 decode positions, but its runner hash is older
  (`06f0a057...`). Treat it as strong dtype evidence, not a perfect current-runner
  A/B.

## Strongest Source Distinctions

1. The strongest current distinction is paired TP1->TP4 process state, not the
   layer5 TP4 graph by itself. `tests/run_multichip_decoder.py:108-113` runs TP1
   then TP4 unless `--tp` filters it. The standalone TP4 current-hash ordinary run
   passes; the paired current-hash ordinary run fails twice. The diagnostic
   harness is also standalone TP4 (`tests/diagnose_attention_ccl_boundaries.py:469-492`),
   so its passes do not exercise the TP1 close / TP4 reopen transition.

2. The ordinary paired runner closes the TP1 mesh while many TP1-owned Python
   references are still live in the frame: `decoder`, cache/page/rope tensors,
   input buffers, closures, and possibly the last `y` remain until overwritten or
   collected after `ttnn.close_mesh_device(mesh)` at
   `tests/run_multichip_decoder.py:392-393`. The staged cleanup diagnostic
   `tests/diagnose_paired_cleanup.py:397-402` is therefore a precise existing
   control: it drops TP1 TT ownership and runs `gc.collect()` before closing TP1.

3. The BFP8 attention CCL payload is still the implicated model boundary within
   the paired failure. `_reduce_attention` casts the local attention output to
   `self.attention_ccl_dtype` and immediately calls `self.allreduce`
   (`tt/multichip_decoder.py:931-936`). For layer5, `full_attention_ccl_dtype=None`
   falls back to the explicit `attention_ccl_dtype`
   (`tt/multichip_decoder.py:624-627`). The BF16 control passing while BFP8 fails
   points at this payload format, modulo the runner-hash caveat above.

4. The "Gemma CCL forced deallocation" lead is refuted for this runtime path.
   `tt/multichip_decoder.py:25,29` imports `MeshConfig` from
   `models.demos.gemma4.config` and only `CCLManager` from GPT-OSS. The active
   allreduce is `models/demos/gemma4/config.py:96-133`; in this no-padding path it
   performs reduce-scatter then all-gather and returns `gathered` without
   deallocating input, scattered, or gathered tensors. The forced-dealloc helper in
   `models/demos/gemma4/tt/ccl.py:244-309` is a different, unused helper here.

5. The diagnostic boundary pass is not a clean lifetime match. Its `BoundaryDecoder`
   retains handles in `self.boundaries` (`tests/diagnose_attention_ccl_boundaries.py:179-183`),
   inlines attention RS/AG with retained `attention_cast`, `attention_dram`,
   `attention_rs`, and `attention_ag` (`:185-228`), and clears old retained
   handles only on the next `_forward` entry (`:259-263`). That can mask allocator
   or lifetime bugs. The output-only diagnostic avoids retention but remains
   standalone TP4.

6. Compute-config differences are weak. The ordinary BFP8 failure command already
   selects LoFi QKV/output. The diagnostic also forces LoFi for attention output
   and QKV projection while preserving the original `math_approx_mode`,
   `fp32_dest_acc_en`, `packer_l1_acc`, and `dst_full_sync_en`
   (`tests/diagnose_attention_ccl_boundaries.py:547-562`). This is not the main
   causal split in the current evidence.

## Most Likely Explanation

The current evidence favors a process-state/lifetime interaction exposed by
paired TP1->TP4 execution and made visible only when the full-attention CCL
payload is BFP8. It does not support a blanket "BFP8 RS/AG at H2816 is
intrinsically nondeterministic" claim, because current-hash standalone ordinary
TP4 and standalone diagnostic TP4 both pass. It also does not establish a
low-level CCL root cause; the failure is only observed at final output in the
ordinary paired path.

The single-rank, few-value, finite failures at different steps/ranks
(rank1/first replay and rank3/repeat replay) look more like stale state,
ordering, or late lifetime/allocator interaction than deterministic arithmetic
drift. That is an inference from the source and artifacts, not a proven cause.

## Smallest Verify/Refute Controls

1. Current-runner BF16 paired control: rerun the failing paired command with only
   `--attention-ccl-dtype bfloat16`. This removes the runner-hash caveat on
   `full_router1_ccl_bf16_control.json`.

2. TP1 cleanup control: run `tests/diagnose_paired_cleanup.py` with the failing
   BFP8 paired command. Pass => delayed TP1 TT-object destruction before/after
   close is strongly implicated. Fail => live TP1 Python ownership is not enough;
   continue toward fabric/open-close or prior TP1 execution state.

3. TP1 no-op preamble control: open/close TP1 with fabric disabled, then run the
   unchanged TP4 BFP8 path. This separates device/fabric transition state from
   TP1 model execution and trace capture. This requires only a diagnostic harness,
   not a runtime change.

4. Paired boundary localization: if the paired failure reproduces under a boundary
   diagnostic with a TP1 preamble, compare `attention_ag`,
   post-attention norm/residual, grouped MoE reduced outputs, and final output.
   If `attention_ag` is first bad, freeze real paired `attention_wo_fp32` values
   and replay only cast->RS->AG with cross-rank equality checks. Synthetic CCL
   seeds are not enough for this failure.

5. Full-attention dtype override: in the paired current runner, use
   `--attention-ccl-dtype bfloat8_b --full-attention-ccl-dtype bfloat16` for
   layer5. If it passes, the failing boundary is specifically the full-attention
   BFP8 CCL payload; if it fails, the source assumption about the selected dtype
   path needs rechecking.

## Uncertainty

No hardware was touched and no new traces were captured here. There is no
intermediate tensor snapshot from the ordinary paired failure, so producer
localization remains unresolved. The prior sliding indexed-expert/router-core
workaround should not be treated as the cause of this layer5 full-attention
replica failure; current artifacts show router core1 placement, and the new
symptom is within-replay cross-rank divergence after a paired TP1->TP4 path.

## Coordinator addendum after report

The cleanup control subsequently failed at position4108 on repeat replay: rank2 differed in8 finite values, max0.00390625; `full_bfp8_paired_cleanup.failure.json`. GC collected5662 objects before TP1 close. This refutes cleanup as a sufficient fix. The report's proposed cleanup control is now completed, not pending. Further paired boundary localization continues.
