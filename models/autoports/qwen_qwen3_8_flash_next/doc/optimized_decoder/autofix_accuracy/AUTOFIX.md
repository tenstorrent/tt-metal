# Optimized decoder accuracy AutoFix

Status: accuracy repair verified. Final context/capacity checks and profiler collection pass; final performance comparisons and independent stage review are recorded in the stage work log.

## Failing contract and diagnosis

The integrated BFP8/HiFi2 GDN z/a/b projection changed real checkpoint output: batch32 request3 decode PCC0.9927617, below0.995. The97-token independent fresh prefill row5 also fell to0.9895512. Expert fidelity, shared-weight dtype, L1 residual storage and expert L1 placement controls do not fix the decode failure. Independent AutoDebug inspected state ownership, recurrence, projection dispatch and numeric paths and found no demonstrated recurrence or synchronization defect; it recommended splitting GDN precision from fidelity/backend.

## Discriminating controls and repair

`../precision_failure_controls_plan.json` and `../layout_failure_controls_plan.json` record all interventions. BF16 GDN/HiFi2 and BFP8 GDN/interleaved HiFi4 both fail. BF16 GDN/DRAM HiFi4 passes batch32 with the unchanged inherited GR-control rules; request3 decode0.9998763. Removing only the GDN DRAM role while retaining BF16/HiFi4 produces the same decodePCC and passes. The minimum implementation repair preserves BF16/HiFi4 for this sensitive projection while retaining DRAM sharding, BFP8/LoFi experts, BFP8 other ordinary projections, and L1 placement. No global or per-row threshold was changed.

Source changed: `tt/optimized_decoder.py` resolves the attention dtype by layer kind and sets the GDN DRAM role toHiFi4. QSA remains BFP8/HiFi2. The final default was rerun through the gates below; earlier candidate passes are not relabeled final.

## Test/report repairs

The wrapper explicitly identifies optimized versus diagnostic fused construction. Effective precision metadata records actual constructed groups so the inner probe's legacy expert-dtype argument cannot be mistaken for the constructed policy. Plan completion now distinguishes finished jobs from successful gates. The stage adds real checkpoint CI-seed coverage and128in/128changing-decode stress twice, without building a full model or generating text. Token-embedding fixture attempts stopped before device execution because this decoder-only snapshot lacks tokenizer payload and embedding shard130. `probe_optimized_stress.py --activation-source embeddings` retains an explicit optional fixture for a complete snapshot; the delivered acceptance uses the documented Gaussian input distribution. No recorded natural-language activation or text-quality claim is made.

## Validation

The final_v3 plans cover strict synthetic cases, same-seed real-weight CI, original HF checks, continuation, batch32GDN/batch5PLE with the predecessor's narrow independent controls, separate Watcher runs, and a BF16/HiFi4-locked GDN packed/split/reader sweep. The optional token-activation job failed before device open; its replacement stress fixture is explicitly Gaussian. Context, capacity and final profiler evidence are separately listed in the stage README. Each failed or interrupted plan retains its original status; later successful jobs are recorded independently.

## Verified repair results

- `final_v3_synthetic.xml`:14/14 strict cases pass, minimum PCC0.9977573. Previously failing GDN1 step1 now0.9999218; no diagnostic waiver, reseeding or threshold change.
- `final_v3_real_ci0.json`: all7 same-seed real-checkpoint lengths1/2/31/33/34/65/66 pass, minimum0.9998644.
- `final_stress.json`: lengths1/33 plus128in/128out, two exact reset replay passes, minimum decode PCC0.9998453 for128out; runtime Torch/from_torch/to_torch/fallback audit clean.
- `final_v3_batch0_32.json`: request3 decode0.9998763; only inherited fresh request16 contribution sensitivity uses the unchanged matched GR control. New request23/29 failures are absent.
- `final_v3_continuation0.json`: strict pass, former97-row row5 whole/contribution0.9997989/0.9997524.
- `final_v3_batch1_5.json`: literal strict pass; no packed-lookup exception needed. PLE continuation retains only its existing controlled fresh65-row row2/129-row row1 contribution sensitivities.
- `final_v3_watcher_continuation0/1.json`: pass with separate watcher logs, exact replay/reset/isolation. QSA arithmetic is unchanged by this repair; its final watcher/continuation/batch and32768 smoke pass.
- `dram_gdn_hifi4_v2.json`: selected precision locked through legal packed/split and1/2/3-reader geometry comparisons. Packed DRAM minimum0.087944ms versus separate0.097282ms; selected2reader/block2 measures0.088460ms.

No low-level kernel defect is claimed. The demonstrated cause is an unsafe precision/fidelity policy at a sensitive real-model projection. The minimum scoped repair changes that policy. All required functional acceptance bars remain unchanged.
