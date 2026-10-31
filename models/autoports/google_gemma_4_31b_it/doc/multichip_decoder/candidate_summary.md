# Candidate comparisons

Host timings, not device rooflines. Exact configurations/source hashes are in adjacent manifests. The corrected `finalpacked8` control supersedes the earlier packed-prefill dtype conversion comparison; see work_log.md.

| Artifact | Batch | Prefill host us | Traced decode host us | Prefill PCC | Decode PCC |
|---|---:|---:|---:|---:|---:|
| candidate_layer0_attn8.json | 1 | 12643.650000 | 458.701445 | 0.999907 | 0.999933 |
| candidate_layer0_down22.json | 1 | 12489.326997 | 459.641984 | 0.999899 | 0.999932 |
| candidate_layer0_finaldown1_cyclic.json | 1 | 11084.476020 | 444.045742 | 0.999920 | 0.999941 |
| candidate_layer0_finalgate1_cyclic.json | 1 | 11235.924962 | 467.353164 | 0.999920 | 0.999941 |
| candidate_layer0_finalo1_cyclic.json | 1 | 11159.829970 | 438.536578 | 0.999920 | 0.999941 |
| candidate_layer0_finalpacked8_batch32_cyclic.json | 32 | 325564.530038 | 637.587594 | 0.999920 | 0.999952 |
| candidate_layer0_finalpacked8_cyclic.json | 1 | 11259.622988 | 430.321782 | 0.999920 | 0.999941 |
| candidate_layer0_finalpacked_cyclic.json | 1 | 11438.139016 | 430.407492 | 0.999921 | 0.999941 |
| candidate_layer0_finalq1_cyclic.json | 1 | 11325.555039 | 447.008875 | 0.999920 | 0.999941 |
| candidate_layer0_finalsplit_cyclic.json | 1 | 11182.872986 | 433.202898 | 0.999920 | 0.999941 |
| candidate_layer0_gate12.json | 1 | 12808.224012 | 502.227937 | 0.999899 | 0.999930 |
| candidate_layer0_gate24.json | 1 | 12664.279959 | 508.159000 | 0.999899 | 0.999927 |
| candidate_layer0_mixed.json | 1 | 12486.040010 | 458.623844 | 0.999899 | 0.999933 |
| candidate_layer0_mlp6144.json | 1 | 13983.694022 | 465.281563 | 0.999887 | 0.999925 |
| candidate_layer0_mlp8.json | 1 | 12552.306987 | 459.396984 | 0.999945 | 0.999933 |
| candidate_layer0_pref14.json | 1 | 12622.299953 | 458.897992 | 0.999888 | 0.999933 |
| candidate_layer0_pref14n8.json | 1 | 12809.626991 | 458.340453 | 0.999888 | 0.999944 |
| candidate_layer0_q21.json | 1 | 12607.390992 | 458.315055 | 0.999899 | 0.999945 |
| candidate_layer0_q24.json | 1 | 12527.428975 | 459.881953 | 0.999899 | 0.999958 |
| candidate_layer0_ring.json | 1 | 11269.342969 | 436.843539 | 0.999920 | 0.999932 |
| candidate_layer0_ring1reader.json | 1 | 11103.117955 | 470.086235 | 0.999920 | 0.999932 |
| candidate_layer0_ringpacked.json | 1 | 11558.355007 | 432.415914 | 0.999921 | 0.999932 |
| candidate_layer0_ringpackedpersistent.json | 1 | 11504.272989 | 426.174820 | 0.999921 | 0.999932 |
| candidate_layer0_ringpackedpersistent_batch32_cyclic.json | 32 | 337698.791991 | 637.385524 | 0.999920 | 0.999952 |
| candidate_layer0_ringpersistent.json | 1 | 11237.083992 | 429.586617 | 0.999920 | 0.999932 |
| candidate_layer0_ringpersistent_batch32_cyclic.json | 32 | 322034.559969 | 640.766789 | 0.999920 | 0.999952 |
| candidate_layer5_finaldown1_cyclic.json | 1 | 14186.452958 | 491.017860 | 0.999990 | 0.999992 |
| candidate_layer5_finalgate1_cyclic.json | 1 | 13899.366022 | 516.036984 | 0.999990 | 0.999992 |
| candidate_layer5_finalo1_cyclic.json | 1 | 13696.669950 | 493.904758 | 0.999990 | 0.999992 |
| candidate_layer5_finalpacked8_batch32_cyclic.json | 32 | 392161.257972 | 933.802094 | 0.999990 | 0.999990 |
| candidate_layer5_finalpacked8_cyclic.json | 1 | 13867.307978 | 477.431101 | 0.999990 | 0.999992 |
| candidate_layer5_finalpacked_batch32_cyclic.json | 32 | 403036.028962 | 933.999477 | 0.999990 | 0.999990 |
| candidate_layer5_finalpacked_cyclic.json | 1 | 14108.925010 | 477.343297 | 0.999990 | 0.999992 |
| candidate_layer5_finalq1_cyclic.json | 1 | 13804.365008 | 487.355484 | 0.999990 | 0.999992 |
| candidate_layer5_finalsplit_batch32_cyclic.json | 32 | 388509.482960 | 937.992836 | 0.999990 | 0.999990 |
| candidate_layer5_finalsplit_cyclic.json | 1 | 13663.236983 | 480.692484 | 0.999990 | 0.999992 |
| candidate_layer5_ring.json | 1 | 13786.558004 | 494.640063 | 0.999990 | 0.999993 |
| candidate_layer5_ring1reader.json | 1 | 13801.747002 | 530.160406 | 0.999990 | 0.999993 |
| candidate_layer5_ring6144.json | 1 | 15352.961025 | 501.794242 | 0.999989 | 0.999993 |
| candidate_layer5_ringpacked.json | 1 | 14089.936041 | 491.149242 | 0.999990 | 0.999993 |
| candidate_layer5_ringpackedpersistent.json | 1 | 14049.217978 | 485.270320 | 0.999990 | 0.999993 |
| candidate_layer5_ringpersistent.json | 1 | 13938.707998 | 487.981648 | 0.999990 | 0.999993 |
| candidate_layer5_ringpref14n8.json | 1 | 13889.164024 | 494.735562 | 0.999989 | 0.999993 |
| candidate_layer5_ringq21.json | 1 | 13669.074047 | 487.262492 | 0.999990 | 0.999992 |
| candidate_layer5_ringq3readers.json | 1 | 13721.965021 | 494.497375 | 0.999990 | 0.999992 |
