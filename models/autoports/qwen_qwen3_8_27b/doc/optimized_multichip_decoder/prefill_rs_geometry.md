# Native prefill reduce-scatter geometry

All rows use TP4,4096+128,B1, real weights/activations, BFP4/LoFi weights, BF16 row inputs and BFP8 output. Times are warmed whole-layer host trace microseconds. The table below is the historical8x8/N20 selection (outputK48/downK17,H4,sub2x4). Current v5 selects11x8/N15/H4/sub2x3 with the sameK48/17; see the v5 closure below and prefill_rs_grid_selection.json. Persistent pool and inter-layer contract do not change across rows.

| Sweep | Kind | Candidate | Prefill host us | PCC minimum | Result |
|---|---|---|---:|---:|---|
| geometry | linear | k4_h8 | 6535.98 | 0.99994778 | pass |
| geometry | linear | k8_h8 | 6406.37 | 0.99994797 | pass |
| geometry | linear | out12_down17_h4 | 6301.98 | 0.99994678 | pass |
| geometry | linear | out24_down34_h1 | 7110.53 | 0.99994431 | pass |
| geometry | linear | k4_h8_sub2x4 | 6392.33 | 0.99994778 | pass |
| geometry | linear | k8_h8_sub2x4 | 6340.97 | 0.99994797 | pass |
| geometry | linear | k8_h8_sub4x2 | 6421.52 | 0.99994797 | pass |
| geometry | full | k4_h8 | 6598.08 | 0.99982168 | pass |
| geometry | full | k8_h8 | 6542.70 | 0.99982168 | pass |
| geometry | full | out12_down17_h4 | 6538.54 | 0.99982168 | pass |
| geometry | full | out24_down34_h1 | 7384.15 | 0.99982168 | pass |
| geometry | full | k4_h8_sub2x4 | 6555.30 | 0.99982168 | pass |
| geometry | full | k8_h8_sub2x4 | 6572.76 | 0.99982168 | pass |
| geometry | full | k8_h8_sub4x2 | 6569.96 | 0.99982168 | pass |
| geometry_final | linear | role_k_h4_control | 6376.22 | 0.99994678 | pass |
| geometry_final | linear | role_k_h4_sub2x4 | 6328.59 | 0.99994678 | pass |
| geometry_final | linear | role_k_h4_sub4x2 | 6433.46 | 0.99994678 | pass |
| geometry_final | linear | out12_h8_down17_h4 | 6368.48 | 0.99994678 | pass |
| geometry_final | linear | out24_h4_down34_h2 | — | — | L1 admission; adapted next |
| geometry_final | linear | k8_sub2x4_control | 6358.61 | 0.99994797 | pass |
| geometry_final | full | role_k_h4_control | 6503.17 | 0.99982168 | pass |
| geometry_final | full | role_k_h4_sub2x4 | 6521.76 | 0.99982168 | pass |
| geometry_final | full | role_k_h4_sub4x2 | 6533.76 | 0.99982168 | pass |
| geometry_final | full | out12_h8_down17_h4 | 6572.10 | 0.99982168 | pass |
| geometry_final | full | out24_h4_down34_h2 | — | — | L1 admission; adapted next |
| geometry_final | full | k8_sub2x4_control | 6507.14 | 0.99982168 | pass |
| role_close | linear | out24_down17_h4_sub2x4 | 6309.13 | 0.99994249 | pass |
| role_close | linear | out12_down17_h4_sub2x4 | 6340.19 | 0.99994678 | pass |
| role_close | full | out24_down17_h4_sub2x4 | 6551.52 | 0.99982168 | pass |
| role_close | full | out12_down17_h4_sub2x4 | 6659.44 | 0.99982168 | pass |
| attribution | linear | control24_17_h4 | 6335.43 | 0.99994249 | pass |
| attribution | linear | out16_down17_h4 | 6342.64 | 0.99994769 | pass |
| attribution | linear | out24_h4_down34_h1 | 6993.26 | 0.99994431 | pass |
| attribution | full | control24_17_h4 | 6515.54 | 0.99982168 | pass |
| attribution | full | out16_down17_h4 | 6506.65 | 0.99982168 | pass |
| attribution | full | out24_h4_down34_h1 | 7213.94 | 0.99982168 | pass |
| endpoint | linear | out48_h4_down17_h4 | 6348.84 | 0.99994652 | pass |
| endpoint | linear | control24_h4_down17_h4 | 6363.02 | 0.99994249 | pass |
| endpoint | full | out48_h4_down17_h4 | 6554.32 | 0.99982168 | pass |
| endpoint | full | control24_h4_down17_h4 | 6608.37 | 0.99982168 | pass |

The downK34/H2 failure had CB extent1,299,072 against available1,293,952. Independent downK34/H1 adapts capacity while preserving outputK24/H4/sub2x4 (down sub1x4); it passes but is slower for both layer kinds. OutK16 is legal and measured near-tied. OutK48 fits because one K block removes double buffering; its paired control usesK24. Small K24/K48 differences are near measurement variation, so final default profiles supply the reported outcome.

Source capacity endpoint: ccl_fusion_mcast_2d.cpp input-CB formulas double buffer when B*num_blocks>1. DownK68/H1 requires2*(68*2048+20*68*576)=1,845,248bytes for A+B alone; K136/H1 with a single block requires the same1,845,248bytes. Both exceed1,572,864 physical L1 before output/partials/runtime. This is a source-backed physical bound under the selectedBF16A/BFP4W precision, not an untried API rejection.

Commands, full policies and log paths: matching *_results.json and *_candidates.json; adapter v8/v9 snapshots in prefill_source_provenance. Final source hashes and selection receipt: prefill_selection.json.

Requested compute config has packer_l1_acc=True. The factory disables effective packer accumulation for num_blocks<=2 (ccl_fusion_mcast_2d.cpp:117–127): selected outK48 has one K block, effective packer accumulation off and BFP8 intermediate/output storage sharing; downK17 has eight K blocks, effective packer accumulation on with BF16 partials. This is an inherent measured geometry effect; no claim of identical effective accumulation across K candidates.

V5 grid closure supersedes the8x8/N20 geometry above:11x8/N15/outputW15,sub2x3,H4,outK48/downK17 is selected for both kinds. Full-width native reader repair generalizes to this legal perN.10x8/11x8 grids and role-H/K adaptations plus forward/reverse results are in prefill_rs_grid_selection.json. K34/H2 passes but is slower; H4 capacity fails. Paired means6033.02→5873.44linear and6280.35→6144.76full; final defaults5870.45/6127.69us reproduce selection. These are host prefill timings; final device profiles are reported separately.
