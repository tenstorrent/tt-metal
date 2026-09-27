> Historical interim review. The candidate gaps below were subsequently exercised by run_candidates5.py and run_candidates6.py, including output-column WO AGMM. See candidate_comparisons.md and candidate_summary.csv. Final-default acceptance remains a separate review.

# Interim Stage05 optimization matrix review

This is a read-only evidence review plus this report, not final acceptance.
No hardware or implementation changes were made by the reviewer. The parent
was running `run_candidates4.py` during inspection; its scheduled rows are
pending until their result JSON exists and passes. Candidate commands and
result JSONs were inspected directly because `candidate_summary.csv` currently
ends before several completed batches.

## Likely selected policy and measured controls

All numbers below are median host microseconds around blocking trace replay,
for real4096/128, batch1. They are not native device durations.

| Candidate | Sliding | Full | Evidence |
| --- | ---: | ---: | --- |
| Stage05 inherited default | 650.322 | 724.813 | `baseline_{sliding,full}.json` |
| QKV BFP4, K22/N2 | 646.615 | 718.872 | `qkv_bfp4_{sliding,full}.json` |
| QKV BFP4, K44/N2 | 645.692 | 717.993 | `qkv4_k44_{sliding,full}.json` |
| QKV BFP4, K88/N2 | 648.121 | 720.242 | `qkv4_k88_{sliding,full}.json` |
| QKV BFP4 K22 plus persistent CCL | 646.253 | 710.330 | `qkv4_persistent_{sliding,full}.json` |
| QKV+WO BFP4 K22 plus persistent CCL | 646.253, PCC fail | 708.362, pass | `combined_{sliding,full}.json` |

Sliding combined minimum PCC0.992892 rejects that exact combined policy;
full minimum PCC0.995366 passes the0.995 bar but leaves little margin for later
changes. Current likely policies are QKV BFP4/sliding WO BFP8 and full QKV+WO
BFP4, with LoFi and persistent CCL. These are candidates, not final defaults.

## Remaining targeted matrix

| Required comparison | Why previous evidence does not close it | Smallest useful next step |
| --- | --- | --- |
| BFP4 QKV core grid/output subblock, both kinds | K22/44/88 all use N2. Stage04 N1/N2/N4 measurements used BFP8 weights. No Stage05 `qkv-n1`/`qkv-n4` result exists. | Hold BFP4/LoFi fixed; compare the legal larger-core N1 and smaller-core N4 families to N2, preferably at the current best K44. Adapt the full-layer N1 grid to the actual11x10 worker shape rather than rejecting an inherited8x12 grid. Count all layout costs. |
| Full WO BFP4 geometry | Stage04 WO N2/N4 and K16/32/64 rejections used BFP8. Current BFP4 WO still uses baseline N1/K8. | With full QKV+WO BFP4, sweep legal K16/32/64 and N2/N4 output families against K8/N1, using a consistent persistent-CCL policy. |
| Final cumulative policy | K44 wins independently, but `qkv4_persistent_*` and `combined_full` retain K22. A `residual_cumulative_*` run changes the residual layout and is not the plain cumulative control. | Measure plain sliding QKV4+best geometry+persistent and full QKV4/WO4+best geometries+persistent. Compare against the strongest passing candidate, then rerun the selected default later. |
| DRAM readers under selected weight policy | New sliding QKV BFP4 r1/r2/r3 controls exist and lose at648.90/652.35/655.33us. Full r2/r3 and shared-role rows are scheduled. **Full BFP4 WO reader1 is missing**: `dram_output_r1_full.json` uses BFP8, while the scheduled BFP4 output rows start at r2. | Finish scheduled role/kind/r1-r3 rows; add BFP4 full WO reader1. If a best DRAM candidate is close enough to compete, cross that winner with the cumulative selected attention/persistent policy rather than combining every losing row. |
| Lower-movement topology with BFP4 attention | `mmrs_v2_*` now supplies real LoFi/final-CCL evidence, but its weights are BFP8; `sharded_bfp8_sliding` crosses MoE payload only, also with BFP8 attention. | At least compare the compatible BFP4 attention family: sliding QKV4 + sharded MoE-BFP8 + Ring/fused AGMM, and full QKV4/WO4 + Ring/fused AGMM/MMRS. Keep the residual sharded through its consumers. Use the existing correct BFP8 paths as controls; do not repeat those exact rows. |
| Persistent/L1 communication | Persistent ordinary CCL is measured in DRAM. `--ccl-l1` exists but no completed result command contains it. | Try the selected replicated persistent family with L1 CCL buffers, including paired MoE packing/consumer costs, or retain exact API/capacity failure evidence after adaptation. |

For the sharded family, passing `--persistent-ccl` alone does not enable ordinary
persistence: construction aliases `allreduce` to `reduce_scatter`, and that
method currently allocates through the ordinary API. Fused AGMM/MMRS do supply
their own persistent buffers. Record the actual buffer contract rather than
claiming persistence from the flag.

The supported SFPU repair makes `residual_sfpu_*` a valid rejected candidate
(685.395us sliding/751.177us full). `residual_cumulative_*` in the running
batch is the appropriate small cross with the proposed reduced-precision
policy; its results can close that comparison without repairing the native
BinaryNg defect in this stage.

## Previously missing work now covered

- Sliding selected-policy packed-versus-separate experts/shared gates:
  `split_expert_final_sliding` and `split_shared_final_sliding` pass but lose
  at686.390/653.573us. Expert/shared dtypes and geometries remain unchanged by
  the proposed attention policy; there is no reason to repeat those exact
  searches merely because QKV weights change.
- Sliding shared-down BFP4, separate attention/shared BFP8 activation trials,
  and shared-prefill K17/L1 placement have actual real-workload results.
  Precision-induced candidate changes in routing still need the final combined
  policy gates, but these are no longer untried advice.
- Real adapted MMRS and sharded-MoE-BFP8 trials close the old synthetic-only
  and omitted-MoE-payload gaps for their measured BFP8 attention policy.
- `AUTOFIX_dram_mesh.md` verifies the multi-reader descriptor repair through
  host controls, compilation/link and the original real reader2 case. It no
  longer justifies omitting the remaining legal reader candidates.
- `AUTOFIX_l1_residual.md` verifies the supported SFPU workaround on both
  kinds. The old PCC collapse is not grounds to discard all L1 residual paths.

Final profile rows, default reproduction, context allocation accounting,
contract/stress/Watcher validation and independent acceptance remain outside
this interim matrix review.

## Parent follow-up: queued coverage and WO helper distinction

`run_candidates5.py` now schedules QKV N1/N4, full BFP4 WO N2/N4 and
K16/32/64, full BFP4 WO reader1, plain cumulative runs, L1 CCL with/without
persistence, BFP4 HiFi2 controls, and compatible BFP4 sharded AGMM/MMRS trials.
Together with the completing reader batch, those rows cover the remaining
existing-target-helper combinations identified above, subject to actual
completion and correctness. This is not a request for a blanket cross product.

The Stage04 `agmm*_headline*` artifacts test **QKV** gather/matmul, via
`_GatherProjection` assigned to `self_attn.source.weights.wqkv`. They are not
WO gather/matmul experiments. The target's current WO fused alternative is
MMRS. `AUTOFIX_fused_ccl.md` explicitly labels the real integration table
"QKV path".

A separate reusable WO-AGMM decomposition exists in
`models/common/modules/attention/attention_1d.py:_fused_all_gather_wo_decode`
(approximately line1288), with weight placement switching from K-sharded to
N-sharded at lines1463–1474. No target Stage04 implementation/result for that
decomposition was found. If assessing that additional material topology
family, use one final-policy candidate per relevant kind: gather local SDPA
width1024/2048 to global4096/8192, multiply the corresponding full-K/704-output
WO column shard, then consume the H704 result directly in distributed
post-attention norm/residual. Remove output RS and avoid immediate replication.
Sliding WO remains BFP8/LoFi; full WO uses BFP4/LoFi. The common auto-config's
eight-device assumptions require adaptation for four devices and22 local
output tiles. The fused API accepts1D/2D matmul programs and has no separate
BFP4 prohibition; normal matmul constraints still apply. This is a distinct
decomposition, not a reason to repeat every already-rejected AGMM setting.
