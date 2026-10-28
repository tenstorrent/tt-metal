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
(`logs/bf16w_repro_2.log`, killed by timeout, `tt-smi -r`, device smoke OK). With the same five
traces captured, the then-selected `p3_qkv_gu_exp`: 2000 replays at 1x2048, no hang
(`logs/selected_soak_1x2048.log`); the shipped default `p3_qkv_gu`: 5000 replays at 1x2048,
`SOAK_DONE 5000`, no hang (`logs/review_soak_p3_qkv_gu_1x2048.log`, run on `1ea3591cc5`).
bf16 weights are slower than bfp8 on every measured shape, so the hang does not affect selection;
it is left as an open item (a candidate for E6b / a tt-metal issue if it reproduces with the E5a
vision policy, which also uses bf16 weights).

## 9. E6a review fixes (E6a-r)

Review E6a: 1.8e-5 margin on the hard last-hidden gate; bf16w row missing; bf16w hang
unclassified; layer-exception twin of `p3_qkv_gu_exp_ple` untried; chart / doc fixes. Logs and
data under `tt-ops/runs/embeddinggemma-2/e6/` (`logs/r_*`, `probe.jsonl`, `sweep.jsonl`).

Twin (`combos_r.json`, `logs/r_twin.log`): `p3_qkv_gu_exp_ple_exc` = `p3_qkv_gu_exp_ple` with
layers 0 and 5 (worst per-layer PCC of `p3_qkv_gu_exp` in `results_gates_g1.jsonl`: 0.999734,
0.999795) and 23 (last) keeping bf16 qkv / gate_up / ple_block outputs. Screen pass: last hidden
0.99035, cosine 0.99183; 29.48 ms ranking aggregate **[M]**.

Probe (`tests/perf/dtype_probe.py`, `logs/r_probe.log`, `r_probe_mid.log`): 12 gate cases + 54
short / padded / demo inputs + 36 mid-length inputs. The gate values reproduce the sweep bit for
bit (`p3_qkv_gu_exp` 0.99002 at `[512, 33]`). `p3_qkv_gu_exp` fails on a 3-token input (offset
24991): last hidden 0.98236, cosine 0.98724; the twin, `p3_qkv_gu`, `p3_qkv_exp` and the
baseline hold (README table).

First selection: the twin, the fastest that held (commit `cff53689f6`). Its gate run
(`gates.sh r1`, logs `gate_r1_*_twin.log`) failed `test_full_model.py::test_retrieval[audio]` and
`demo.py::test_multimodal_demo`: the caption_speech2s row is an HF near tie (0.00463) and the twin
flips it by 1e-5. Chain stopped, device reset. Override runs (`scripts/mm_check.sh`,
`logs/r_mm_*`): `p3_qkv_gu`, `p3_qkv_exp`, `p3_qkv_gu_exp` pass test_full_model (30) and the
demo (2), row margins 0.00198 / 0.00173 / 0.00214. Final selection: `p3_qkv_gu` (commit
`9adacfe162`), 31.06 ms; `p3_qkv_exp` is 0.56 ms slower on the ranking aggregate but faster at
1x2048 / 1x8192.

bf16w row (`scripts/bf16w_row.sh`, `bf16w_rest.sh`): accuracy reproduces (last hidden 0.99876,
cosine 0.99944); perf one trace per process: 5.215 / 7.560 / 22.198 / 29.678 / 108.554 ms
burst. The first 1x2048 attempt with one trace and the full perf loop hung
(`triage/bf16w_1x2048_single_trace.txt`), so 1x2048 was re-run as burst windows only.

bf16w hang classification **[M]**, every run on bf16w_hifi2 at 1x2048 with `bf16w_repro.py`:

| run | trace | stopped after | running op (tt-triage) |
|---|---|---|---|
| E6a `bf16w_repro_2.log` | 5 traces captured | 240-260 replays | k-norm `rms_norm` [1, 2, 2048, 256] L1 |
| `r_bf16w_trace1_1x2048.log` | 1 trace | 220-240 replays | no op decoded: the launch / watcher mailboxes of worker cores (7..12, 0) hold values like 0x41DC0000 (27.5 as fp32) |
| `r_bf16w_trk_all_1x2048.log`, `TT_METAL_TRACE_ALLOC_TRACKING=1` | 5 traces | 460-480 replays; no "still alive" error | `LayerNormDeviceOperation` [1, 2, 2048, 256] L1 |
| `r_bf16w_eager_1x2048.log` | none (eager `encode`) | 760-780 forwards | `LayerNormDeviceOperation` [1, 2, 2048, 256] L1 (v-norm, no gamma) |
| perf loop, `dtype_sweep.py --shapes 1x2048` | 1 trace | inside the ~600-replay loop (count not logged) | k-norm `rms_norm` [1, 2, 2048, 256] L1 |

Classification: a bf16w-specific device hang in the per-head RMSNorm of the K / V heads at
1x2048 (L1 interleaved input), not a trace issue: it reproduces without any trace, with one trace,
and with five; the allocation tracker finds no unsafe live buffer before any replay. The
compute kernel stalls on every core of the op (unpacker waiting for a context, packer in
`elem_apply_pack`, writer in `cb_wait_front`) **[M]**. In one run the firmware mailboxes of six
worker cores held float-looking words **[M]**, which points at a stray NoC write of tensor data
into worker L1 rather than a kernel deadlock **[I]**. Why bf16 weights trigger it is not known
**[I: bf16 projection weights change L1 / DRAM allocation and matmul program configs upstream;
the norm has the same shape and dtype in both policies]**. With the watcher on
(`TT_METAL_WATCHER=10`, NoC sanitization; `scripts/bf16w_watcher.sh`, `logs/r_bf16w_watcher.log`)
the single-trace repro completes all 1500 replays with 0 error lines in `watcher_bf16w/.../watcher.log`
**[M]**: the hang does not reproduce under the watcher's timing, and the sanitizer flags no
out-of-range NoC access in that run. Timing-dependent, so a race in the op is as likely as a stray
write **[I]**; not pursued further (bf16w is not a selection candidate). The shipped default
`p3_qkv_gu` ran 5000 replays at 1x2048 with the five perf traces captured, `SOAK_DONE 5000`, no
hang (`logs/review_soak_p3_qkv_gu_1x2048.log`, run on `1ea3591cc5`); the earlier default
`p3_qkv_gu_exp` ran 2000 (`logs/selected_soak_1x2048.log`); interleaved five-shape soak below.
Each hang: tt-triage, kill, `tt-smi -r`, device listed OK.

Final runs on `9adacfe162` (`scripts/chain_r.sh`, `logs/chain_r.log`) **[M]**: gates `r1` all pass
(pcc 357, s32768 1, full model 30, demo 2, vision 91, audio 109, watcher 101 with 0 error lines;
`test_precision_config` 1211 checks on `p3_qkv_gu`); allocation tracking over test_trace (6),
test_full_model_trace (11) and the demo (2): no "still alive"; interleaved five-shape soak 2060
replays, 0 mismatches, with and without tracking; post-selection bench `post_bench_r.jsonl`
(selected 4.813 / 6.833 / 20.054 / 24.243 / 93.054 ms burst vs baseline override 4.923 / 7.314 /
22.042 / 29.393 / 105.688). Report-only change after the gates: chart label placement and the
left-panel y range in `dtype_sweep.py` `_chart` (no runtime or test path).

# E6b: vision and audio towers

Same host and card, branch head `1ea3591cc5` at the start. Run artifacts:
`tt-ops/runs/embeddinggemma-2/e6b/` (`sweep_vision.jsonl`, `sweep_audio.jsonl`,
`sweep_audio_bfp4_acc.jsonl`, `probe_{vision,audio}.jsonl`, `audio_sensitivity.json`,
`audio_token_trace.json`, `logs/`, `scripts/`, `refs/` = cached HF fp32 probe references).

## 10. Carry-over (commit `c1efa1a330`)

§8 / §9 and the README "Not done" bullet cited `logs/selected_soak_1x2048.log` as the shipped
default's soak; that run was `p3_qkv_gu_exp`. They now cite `review_soak_p3_qkv_gu_1x2048.log`
(p3_qkv_gu, 5000 replays, `SOAK_DONE 5000`, run on `1ea3591cc5`) and keep the other run labelled.

## 11. Wiring (commit `68f46d1f16`)

- `tt/precision.py`: `VisionPrecision` / `AudioPrecision` with the same strictness as text (every
  key required, unknown keys rejected; a free-text `note` allowed); tower `out_dtype` may be fp32;
  `audio.norms.sharded`; `E5_VISION_POLICY` / `E5_AUDIO_POLICY` are the old hard-coded policies.
  `TextPrecision` parsing is unchanged; `MatmulPolicy.from_dict` / `LayerException.from_dict` take
  the allowed groups / output dtypes as arguments with the text values as defaults.
- `tt/vision/*`, `tt/audio/*`: every weight dtype, compute kernel config, projection output dtype,
  SDPA config (incl. `exp_approx_mode`), norm config and the mask dtype come from the policy;
  `from_state_dict(..., precision=None)` loads the default file at every level (so the unit tests
  of single modules build with the default too). `tt/audio/norm.py`: the E5c c21 block-sharded
  RMSNorm (`e5c/scripts/rejected_norm.py`) behind `audio.norms.sharded`. `audio/linear.py` CB
  budget learned the bfp8 / bfp4 tile sizes. `AudioModel(act_dtype=float32)` still overrides the
  activation-stream output dtypes (the fp32-activation tests).
- `tt/embedding_model.py` (+15/-4 lines, outside the brief's file list; an accepted scope exception,
  needed so `precision=` reaches all three towers through the public API):
  `EmbeddingGemma2.from_pretrained(precision=...)` resolves the policy once and passes it to all
  three towers (it used to reach the text tower only).
- Bit-identity check (`scripts/wire_check.sh`, before any selection): vision + audio suites on the
  wired code with the E5 policies in the file: `e6b/results_wire.jsonl` has 497 metric records; 488
  equal `e6/results_gates_r1.jsonl` exactly; the other 9 are timing rows (vision
  `vision_latency_device_ms` / `vision_latency_e2e_ms` / `vision_host_prep_ms` at 1x2560, 4x2560,
  8x1280) that differ within run-to-run noise.

## 12. Screens (phase 1, phase 2)

`scripts/sweep_p1.sh` (vision 22 + audio 27 single levers), `sweep_p2v.sh` (vision combos on the
SDPA exp lever + bfp4 twins), `sweep_p2a.sh` (audio combos + bfp4 twins as an accuracy screen
first), `sweep_p3a.sh` (audio bfp4 combo). One device session per candidate (accuracy then perf);
the screen reproduces the tests' own logged minima on the base configs (vision soft 0.99653 video8
f3, worst token 0.8638 portrait; audio soft 0.99769 tone3). Per-candidate rows: README tables and
`sweep_results.csv` (`tower` column).

Vision: the only large lever is SDPA (`exp_approx` -32.2 ms, passes; fp32 acc off -32.4 ms, fails
on video frames). Weights and fidelity move 0-4 ms. Blocked: bfp8 QKV output, because
`rotary_embedding_llama` requires bf16 inputs and `rms_norm` keeps the input dtype.
Audio: HiFi2 projections (-3.5 ms), fused QKV output bf16 (-2.1 ms), shift HiFi2 (-0.13 ms),
sharded norms (-0.3 ms), bfp8 weights (-0.2 to -0.4 ms) add up to -6.6 ms. bfp4 fails for the FFN
alone and for the combination; LoFi FFN fails.

## 13. Probe and selection

`scripts/probe_v.sh`, `probe_a.sh`: the finalists, the next slower passing configs and the base
through `EmbeddingGemma2.from_pretrained(precision=...)` on 18 / 17 probe inputs plus the tower's
full-model items and retrieval. The two fastest vision combos (bfp8 gate/up outputs) fail on 3-frame
video (soft 0.98857 / 0.98394); `v_exp_bfp8w4` holds (0.99479). Audio: the bfp4 combo fails, the
bfp8 + HiFi2 + QKV-bf16 + shift + sharded combo holds (0.99114 on a synthetic tone); `hifi2_all`
alone flips the near-tie retrieval row. Selected file = those two tower sections, checked equal to
the measured candidates' policies (`PrecisionConfig` comparison) before the code commit.

## 14. Audio worst-token sensitivity

`scripts/audio_sensitivity.py` (CPU, HF fp32), `scripts/audio_token_trace.py` (device, default
path at the time of the run: E5c audio policy). Result and tables: README "Audio worst-token
sensitivity". The functional-audio README's `[guess]` line now points to it.

## 15. Planted pass / fail / pass

`logs/pfp_e6b_1.log` (7 passed), `pfp_e6b_2_planted.log` (`bfp5`: 2 failed, ValueError),
`pfp_e6b_2b_planted_valid.log` (`bfp4`: 2 passed, models hold BFLOAT4_B), reverted: gate run.

## 16. Gates, soak, post-selection bench (code commit `68f46d1f16`)

`scripts/gates.sh g1` then `scripts/post_bench.sh g1` (`logs/chain_final.log`): precision 7, vision
91, audio 109 (7 deselected), full model + full-model trace 41, demo 2, `tests/pcc -k "not s32768"`
361 passed; watcher over vision + audio 190 passed, 0 error lines; alloc tracking over the two
trace tests 19 passed, no "still alive"; joint vision + audio soak 2052 replays, 0 mismatches
(tracking on; a second run without tracking: `logs/soak_g1_notrk.log`). Post-selection bench and
baseline override: README Gates table. Text gate metrics equal to E6a r1 except one value E6a r1
logged both ways (README).

# E9.1-r re-selection (2026-10-28, text, 12 x 10 grid)

Why, policy, table and choice: README "E9.1-r re-selection". Run folder
`tt-ops/runs/embeddinggemma-2/e9/01r-geometry-precision/` (`$G`); scripts in `$G/scripts/`, every
device command through `run.sh` (card-free check, `timeout -s KILL`, `__tt_exit`), one at a time,
`EMBEDDING_GEMMA2_GRID=12x10` unless stated. Candidate files: `$G/precision/precision_<id>.json`
(the selected file with the candidate's text section; `make_candidates.py`), merged combos
`$G/precision/combos_all.json` (E6 `combos_p2/p3/r.json`).

## 17. Screen (`chain_c1.sh`, `chain_c2.sh`, `chain_c4.sh`)

1. Rank, all 19 (`dtype_sweep.py run --no-acc --shapes 1x512,8x512 --soak 0 --sustained-ms 200`,
   baseline first and last): `logs/c_rank_12x10.log`. Baseline control 45.021 -> 45.016 ms.
   `p3_qkv_gu_exp_ple_exc` errored (static CBs clash with L1 buffers).
2. Five-shape timing for `p3_qkv_gu`, `p3_qkv_exp` between two baseline runs (`logs/c_shapes_12x10.log`);
   both baseline runs errored with the same clash.
3. `dtype_probe.py --inputs e9`, all 19 candidates (`logs/c_probe_e9_12x10.log`):
   fails `p3_qkv_gu` (0.98836) and `p3_qkv_gu_exp` (0.98990); every other config >= 0.99117.
4. In rank order: short + mid probe and 10-bucket contract gate (`prefill.py -k gate`,
   `EMBEDDING_GEMMA2_PRECISION=<candidate file>`) for `p3_qkv_exp`, `act_bfp8_attn_qkv`, baseline
   (`chain_c2.sh`); full gates for `p3_qkv_exp` (`gates.sh c_p3exp_12x10`: 1 failed
   `test_full_model::test_retrieval[audio]`, demo 1 failed `test_multimodal_demo` text -> audio;
   152 + 1 passed otherwise).
5. Next four in rank order (`chain_c4.sh`): retrieval + demo tests, short + mid probe, contract gate.
   `p2_sdpa_fast` and `sdpa_nofp32acc` pass all; `sdpa_exp_approx` and `act_bfp8_mlp_gate_up`
   crash in the contract session <= 8192 (clash); the short-probe process stopped at
   `sdpa_exp_approx` (clash), so `act_bfp8_mlp_gate_up` has no short-probe value.
6. Clash repro (`repro_cb_clash.py`, README open item): `logs/c_repro_cb_clash_base_*`.
7. Tie at 41.843 / 41.845 ms; orchestrator rule: the fastest measured, `p2_sdpa_fast`, unless its
   full gates disqualify it.

## 18. Selection and gates (`chain_c5.sh`)

`selected_precision_config.json`: text section = `$G/precision/precision_p2_sdpa_fast.json` text
(checked equal by a JSON compare; vision / audio unchanged), `config_id` `p2_sdpa_fast`.
Then, with no override: `gates.sh c_p2_12x10` and `gates.sh c_p2_13x10` (153 + 1 + 2 passed
each), 10-bucket contract gate at 12 x 10 with alloc tracking (10 passed), soak 2000 replays at
1 x 2048 on 12 x 10 (0 mismatches), `perf.py` default at 13 x 10 new / `p3_qkv_gu` control / new
(7.099 / 6.857 / 7.098 ms text 1 x 512). Values: README "E9.1-r re-selection".

## 19. Amendment E9.1-r2a (2026-10-28): default `p2_sdpa_fast` -> `sdpa_nofp32acc`

Sections 17 and 18 stay as the record of the E9.1-r choice. The E9.1-r review (verdicts.md row
E9.1-r, finding 4) and the orchestrator changed the tie-break: the two configs tie within noise
(41.843 / 41.845 ms), `sdpa_nofp32acc` has the wider worst-case margin (short probe 0.99240 vs
0.99094) and no `exp_approx`; `p3_qkv_gu` had failed a new input after a 1.1e-3 probe margin.
Run folder `tt-ops/runs/embeddinggemma-2/e9/01r2a/` (`$A`), scripts copied from `$G/scripts/`
(`run.sh` waits while the card is held instead of exiting), one device command at a time.

1. `selected_precision_config.json`: text section = `$G/precision/precision_sdpa_nofp32acc.json`
   text section (JSON compare: equal; vision / audio equal to the previous file); `config_id`
   `sdpa_nofp32acc`; description amended. The only text field that changes: `sdpa.exp_approx`
   true -> false. `dtype_sweep.candidate_dict("sdpa_nofp32acc")["text"]` equals the new default text.
2. `chain_r2a.sh`: `gates.sh r2a_12x10`, `gates.sh r2a_13x10` (153 + 1 + 2 passed each); contract
   gate 10 buckets at 12 x 10 with alloc tracking (10 passed, min 0.99424 / 0.99834 at b1_s128);
   `dtype_probe.py --inputs short / mid / e9` for `sdpa_nofp32acc` (all pass, worst 0.99240 at
   `[511]`); soaks 2000 replays at 1 x 512 / 2048 / 4096 on 12 x 10 (0 mismatches each); `perf.py`
   at 13 x 10 new / `p3_qkv_gu` control / new (7.078 / 6.861 / 7.091 ms text 1 x 512).
   Values: README "Amendment E9.1-r2a".
