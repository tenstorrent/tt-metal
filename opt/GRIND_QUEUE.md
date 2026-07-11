# GRIND QUEUE — concrete unmeasured experiments (the loop's never-empty work source)

**The loop NEVER concludes "exhausted."** Each lap: if no experiment is in flight, dispatch the next
`[ ]` item below (in-budget, completes, real data), mark it `[~]` in flight, and on completion record the
number here + PROGRESS LIVE LOG and tick it `[x]`. When ALL are `[x]`, generate the next batch by
re-profiling the current dominant op and enumerating its config space — do not stop.

All device runs via broker, owner `[claude]smarton`, `timeout_sec<=240`, in-budget (block/op harness, NOT
full pipeline — full pipeline cold-compiles >280s and times out even with prewarm; measure denoise deltas
per-block, VAE via prof_vae_ltx.py). Env: TT_METAL_HOME/PYTHONPATH=worktree, caches tt-*-ltxrt, PINNED=0.

## Batch A — SDPA chunk sweep (SDPA = 30% of block = 6.13ms; baseline chunk (192,512)=4.85ms; NEVER swept)
Measure RingJointSDPA FW (largest-FW op in the profile) per chunk. Test:
`test_ring_joint_attention.py::test_ring_joint_sdpa_dit -k "ltx_s2 and <qID> and <kID> and 8rpx4up"` under `tracy -p -r`.
- [~] q256 k512  (job 210815-17)
- [ ] q128 k512
- [ ] q256 k256
- [ ] q128 k256
- [ ] q128 k128
- [ ] q64  k512
- [ ] q256 k128
→ record SDPA FW per config; fastest vs 4.85ms baseline is the win. If any config beats baseline by >5%, wire it into the LTX ring SDPA call (attention_ltx.py) + PCC-gate + block WARM_FWD_MS.

## Batch B — SDPA compute-config sweep (same harness, vary fidelity/accum)
- [ ] LTX_QUANT=all_bf8_lofi_sdpa_lofi_fp32acc on the one-block harness — SDPA LoFi + fp32 dest acc, PCC-gate.
- [ ] exp_ring_joint_sdpa variant (in-tree experimental kernel) — does it lower SDPA FW at ltx_s2 shape?

## Batch C — VAE decode sub-levers (prof_vae_ltx.py, NOT full pipeline; VAE stage = 1.0s)
- [ ] W-mask fold: drop the per-conv `ttnn.mul` width-mask before each W-sharded halo conv (~42 sites; kevinmi in-kernel mask). Measure decode wall delta + DECODE-TRACE-PCC.
- [ ] depth-to-space permute: profile the 4 upsample reshape→permute→reshape; is a fused kernel cheaper? (~84ms device bucket).

## Batch D — adaLN to_out fusion (attention_ltx.py; a2v/v2a/attn2 gated-residual into to_out epilogue)
- [ ] fold the 3 standalone `ttnn.addcmul` into the to_out matmul via the existing fused primitive attn1 uses; PCC-gate + block WARM_FWD_MS.

## Batch E — step-distill characterization (the 6s path; needs the PREWARM path to complete an E2E)
- [ ] Make a full-pipeline run actually COMPLETE: run the prewarm capture+compile, then submit the run TWICE (first warms the trace, second measures) OR raise the run-stage timeout via a 2-reservation split. Then measure 6+2 E2E speed + quality.
  NOTE: raw 4x8 full-pipeline = guaranteed timeout; this is why it's last + needs the multi-pass warm.

## DONE (measured, with the number)
- audio-trace: SHIPPED -0.3s. VAE-trace: 0.19ms DEAD. num_links=4: HW-capped. RMSNorm QK-merge: null (45.08 vs 44.03). tilize: cold artifact. all_bf8 weights: -0.04s null.
