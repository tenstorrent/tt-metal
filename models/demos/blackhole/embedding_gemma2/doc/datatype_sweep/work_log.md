# E6a datatype sweep (text tower): work log

Host sjc-snva-tp100, one P150a. Branch `gtobarTT/embedding_gemma2_bringup`, started from
`926a0f2a00`. Environment: `e1/env.sh` + `HF_HUB_OFFLINE=1`, `HF_MODEL=<914f7f89 snapshot>`.
Run artifacts: `tt-ops/runs/embeddinggemma-2/e6/` (`sweep.jsonl` raw records, `logs/`, `scripts/`,
`triage/`). Labels: **[M]** measured, **[I]** inferred.

## 1. Machinery (commit `443082b9a1`)

- `tt/precision.py`: schema, validation (unknown or missing keys raise), loader. Default source is
  `doc/datatype_sweep/selected_precision_config.json`; `EMBEDDING_GEMMA2_PRECISION=<file>` or a
  `precision=` argument (path, dict, PrecisionConfig) overrides it.
- Text modules take the policy from `TextModel.from_state_dict` down: weight dtype per group at
  construction, `MatmulPolicy` (fidelity, fp32 acc, output dtype) per `linear` call, SDPA compute
  config + `exp_approx_mode` in the program config, norm compute configs, residual add dtype,
  mask dtype. Layer exceptions resolve per layer at construction.
- `tt/linear.py` and `tt/rms_norm.py` are shared with the vision / audio towers: the new
  arguments default to the E5a behaviour, so those towers are unchanged.
- The file was written with the E5a policy first. Baseline candidate through the new path
  **[M]**: 4.923 / 7.314 / 22.053 / 29.397 / 109.678 ms (1x128 / 1x512 / 1x2048 / 8x512 / 1x8192
  burst), vs E5a 4.924 / 7.316 / 22.059 / 29.418 / 109.81 — same within noise.
- `tests/pcc/test_precision_config.py`: 3 passed, 1211 checks on the default model **[M]**.

## 2. Harness

`tests/perf/dtype_sweep.py run`: one process, HF fp32 reference computed once, device opened per
candidate. Accuracy on the tests/pcc cases (11 cases of `tests/common.CASES` + the batch-8 padded
case): teacher-forced PCC of all 24 layers, attention / MLP PCC of layers 0 and 5, free-running
last-hidden PCC, per-prompt embedding cosine, demo retrieval argmax TT == HF. Screen = the gate
bars on the gated subset (per-layer / module 0.995, last hidden 0.99, cosine 0.99, retrieval).
Perf: traced, 5 burst windows of 20 replays after 3 s idle (median), then 5 s soak and 5
sustained windows of about 1.5 s. Ranking aggregate: burst 1x512 + burst 8x512. About 195 s per
candidate.

## 3. Phase 1 (single levers; `logs/s2_phase1.log`, commit `443082b9a1`)

- `bf16w_hifi2` hung in trace replay at 1x2048 (first run, `logs/s1_phase1_hang_bf16w.log`).
  Accuracy had passed (cosine 0.99944, last hidden 0.99876); burst 1x128 5.219 ms, 1x512 7.563 ms
  (+6 % / +3.4 % vs bfp8) **[M]**. tt-triage (`triage/bf16w_1x2048.txt`): every worker core of
  one op stuck in `LayerNormDeviceOperation` on [1, 2, 2048, 256] (the k-norm of a sliding layer).
  Killed, `tt-smi -r`, sweep resumed without it. See section 6.
- bfp4, every group, HiFi2 and LoFi: all fail. LoFi and HiFi2 give identical accuracy with bfp4
  weights to 5 digits **[M]**. Closest: `ple_proj` (cosine 0.98982), `final_proj` (last hidden
  0.98964); worst `attn_o` (cosine 0.915). None is faster than bfp8 by more than 0.15 ms on the
  aggregate (weights are about 150 MB in bfp8; the forward is not weight-bandwidth bound **[I]**).
  No layer exception can help a group that is both far from the bar and not faster.
- LoFi, all bfp8 groups: -0.25 ms, last hidden 0.98650 fails. fp32 acc off, all matmuls: -0.14 ms,
  passes (last hidden 0.99066).
- SDPA: fp32_acc off -1.67 ms (pass, last hidden 0.99240); exp_approx -1.44 ms (pass, 0.99608);
  LoFi -0.39 ms (fails: attention PCC 0.99047 at layer 5, 8192 tokens).
- Norms: HiFi2 -0.14 ms (pass); fp32 acc off +0.44 ms slower (pass).
- Activations: every projection output bfp8 -6.56 ms (-18 %), cosine 0.99095 passes but last
  hidden 0.97814 fails. Residual stream bfp8: -0.35 ms, fails (cosine 0.968).
- Masks: bfp8 / bf16 give bit-identical accuracy to bfp4 and no speedup (+0.04 / +0.23 ms).
  bfp4 stays; the context contract is unchanged.

The last-hidden PCC floor (0.99, `test_text_model`) is the binding metric for most levers.

## 4. Schema change (commit `0995c45f9b`)

`act_bfp8` was the largest win, so the activation dtype moved from one `text.activation_dtype` to a
per-group `out_dtype` (records before this commit carry the old field; `act_bfp8` = every group
`out_dtype: bfp8`, same policy).

## 5. Phase 2 (per-group activation dtype, SDPA / LoFi combos; `logs/s3_phase2.log`)

- `act_bfp8_nofinal` = `act_bfp8` (last hidden 0.97814): the final projection output is not what
  breaks the last-hidden PCC.
- Per group **[M]**: attn_qkv -3.16 ms (last hidden 0.99308), mlp_gate_up -2.49 ms (0.99464),
  ple_block -0.41, ple_proj -0.33, mlp_down -0.07, attn_o -0.03; all pass alone.
- `p2_sdpa_fast` (SDPA fp32 acc off + exp_approx) = 35.04 ms, the same as fp32 acc off alone.
  Adding matmul fp32-acc-off or norms HiFi2 to it fails the last-hidden floor (0.97277-0.98865).
- `p2_sdpa_lofi_sliding` (LoFi SDPA on sliding layers, full layers 5/11/17/23 HiFi2 by layer
  exception): last hidden 0.97774, fails.
- bfp8 LoFi per group (mlp / attn / ple): pass, within 0.1 ms of HiFi2; not kept.

## 6. Phase 3 (combinations of the passing winners; `logs/s4_phase3.log`)

| config | ms | last hidden | cosine | |
|---|---|---|---|---|
| p3_qkv_gu | 31.06 | 0.99184 | 0.99467 | pass |
| p3_qkv_exp | 31.62 | 0.99146 | 0.99461 | pass |
| p3_qkv_gu_exp | 29.14 | 0.99002 | 0.99269 | pass |
| p3_qkv_gu_exp_ple | 28.39 | 0.98913 | 0.99240 | fail |
| p3_qkv_sdpafast | 31.17 | 0.98925 | 0.99245 | fail |
| p3_qkv_gu_exp_exc_full | 30.75 | 0.99069 | 0.98988 | fail (exception made it slower and moved the failure to cosine) |

exp_approx is worth -1.9 ms on top of the bfp8 outputs vs -1.44 ms alone **[M]**.

Decision: `p3_qkv_gu_exp`, the fastest config that passes the screen; it is 6.2 % faster than
`p3_qkv_gu` (outside noise: burst windows spread <= 0.01 ms at 1x512). Margin on the last-hidden
floor is 2e-5, so it went straight to the full gate set; `p3_qkv_gu` was the fallback.

## 7. Selection and gates (commit `b6e6e9d30e`)

`selected_precision_config.json` set to `p3_qkv_gu_exp` (checked equal to the sweep candidate's
parsed text policy). `scripts/gates.sh g1`: full_model 30 passed, demo 2 passed, s32768 1 passed,
`tests/pcc -k "not s32768"` 357 passed, vision 91 passed, audio 109 passed (7 deselected),
watcher run 101 passed with 0 error lines **[M]**. The gate metrics equal the sweep screen on
the shared cases (last hidden 0.990018 at `[512, 33]`, cosine 0.99269).

Pass / fail / pass: `logs/pfp_2_planted.log` (bfp5: ValueError, 1 failed),
`pfp_2b_planted_valid.log` (bfp4: model holds BFLOAT4_B w_down), `pfp_3_reverted.log` (1 passed).

Post-selection benchmark through the normal path (`scripts/post_bench.sh`, `post_bench.jsonl`):
selected 4.691 / 6.490 / 17.069 / 22.653 / 79.308 ms burst vs baseline override 4.923 / 7.316 /
22.042 / 29.392 / 105.690 ms (1x128 / 1x512 / 1x2048 / 8x512 / 1x8192).

## 8. bf16 weights hang (not root-caused)

`scripts/bf16w_repro.py bf16w_hifi2 1x2048 traced N [all]`: one captured trace, 200 replays OK;
five captured traces (the perf set), hang after 240-260 replays of the 1x2048 trace
(`logs/bf16w_repro_2.log`, killed by timeout, `tt-smi -r`, device smoke OK). The selected config
with the same five traces: 2000 replays at 1x2048, no hang (`logs/selected_soak_1x2048.log`).
bf16 weights are slower than bfp8 on every measured shape, so the hang does not affect selection;
it is left as an open item (a candidate for E6b / a tt-metal issue if it reproduces with the E5a
vision policy, which also uses bf16 weights).
