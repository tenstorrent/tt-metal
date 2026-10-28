# E5b work log (optimized vision tower)

Host sjc-snva-tp100, one P150a, branch `gtobarTT/embedding_gemma2_bringup`, base `c62d1e8c1f` (E5a).
Env for every run: `source tt-ops/runs/embeddinggemma-2/e1/env.sh` plus `HF_HUB_OFFLINE=1`,
`HF_MODEL=<HF cache>/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b`,
`EMBEDDING_GEMMA2_RESULTS=tt-ops/runs/embeddinggemma-2/e5b/results*.jsonl`
(`e5b/scripts/env.sh`). Every device job ran as a tmux session `tt-bringup_embeddinggemma2_e5b_*`
with its log in `e5b/logs/`. One device job at a time. Times are local (PDT), 2026-10-27.

## Timeline

| time | step | result |
|---|---|---|
| 16:56 | read the four skills, E2 / E5a reports | plan: SDPA first (58 % in E2), then RoPE, projections, MLP, L1 |
| 16:58 | carry-over commit `86c6b9f9f7` | `resolve_snapshot` pinned to 914f7f89 (+ README.md, *.jinja); README 65536 note; contract s32768 evidence. Offline resolve of the hub id -> the 914f7f89 snapshot, README.md and chat_template.jinja present **[M]** |
| 17:02 | baseline bench of the E2 code (`bench_vision.py`, label `baseline_E2`) | 62.80 / 323.26 / 228.31 ms (b1x2560 / b4x2560 / b8x1280); E2 doc said 55.60 / 294.47 / 199.61, see README "Result" |
| 17:03-17:08 | SDPA sweeps (`sdpa_sweep.py` b1, b4, b8) | 13x10 q128 k512 HiFi2: 2.01 -> 0.98 ms (b1), 12.64 -> 3.66 ms (b4); bfp4 mask -1 %; fp32 acc off -50 % at PCC 0.99936 |
| 17:04 | RoPE check (`rope_check.py`) | `rotary_embedding_llama` with trans_mat = R[:32, :32]: same output as the 4-op form (max diff equal), 44 vs 130 us (b1), 152 vs 491 us (b4) |
| 17:10 | first full rewrite behind a temporary lever switch; vision suite 80 passed | 31.3 ms at b1, but worst soft-token cosine 0.624 |
| 17:12-17:39 | ablation chain L0-L7 + X (`ablate.sh`: bench + `test_soft_tokens` per lever set) | bfp8 weights cause the cosine drop; bf16 weights cost 0.6-4.6 %; L1 activations -2.5 ms at b1; SDPA fp32 acc off fails accuracy |
| 17:32 | Tracy mid (`tracy/mid_w16`) | SDPA 48 %, gate matmul with packer gelu 153 vs 101 us |
| 17:41 | matmul sweep (`mm_sweep.py` 2560 L1, 10240) | 2D wins at 2560 for every role in the micro, minimal_matmul at 10240 |
| 17:42-17:54 | L8-L12 | down 2D slower in the model (213 us, DRAM input); gelu as multiply pre-activation kept; o_proj 2D kept; 128 KB L1 rule clashes |
| 18:00 | levers removed, clean code: vision suite 81 passed, bench equal to L11 | commits `0eaea00924` (harness), `c504dd3268` |
| 18:01 | Tracy after (b1, b4) and before (E2 files checked out, restored) | README tables |
| 18:03 | windowed SDPA sweep (`sdpa_windowed.py`) | windowed wins only at 8 x 1280 (1.99 vs 2.63 ms) |
| 18:09 | windowed path for >= 8 frames at 1280; suite 81 passed; bench 105.24 -> 94.65 ms at 8 x 1280 | commit `a410c73278` |
| 18:10-18:24 | gates on `a410c73278` (`e5b/scripts/gates.sh`) | all pass (vision 81, full model 41, demo 2, watcher 81 with 0 error lines, tests/pcc 343); logs `gate_*_a410c73278.log` |
| 18:24-18:28 | L1 screens at 1 x 2560 (`bench_patched.py`, `exp_mlp_l1.py`, `exp_mask_l1.py`) | MLP up/gate + QKV outputs in L1: 28.25 -> 26.27 ms; 128 KB cap clashes; mask in L1 blocked by SDPA |
| 18:29 | named wide intermediates in code; suite 81 passed; bench 26.26 / 121.62 / 94.64 | commit `47824a6e41` |
| 18:30 | gates and after Tracy on `47824a6e41` | see "Gates" |

## Keeps and rejects

Burst medians, traced, `e5b/bench.jsonl`; accuracy = min soft-token PCC / worst-token cosine over
the 7 vision cases (`results_<label>.jsonl`).

| change | 1x2560 ms | 4x2560 ms | 8x1280 ms | accuracy | decision |
|---|---|---|---|---|---|
| SDPA table + HiFi2 | 62.47 -> 46.05 | 322.02 -> 178.59 | 226.89 -> 169.75 | 0.997906 / 0.9636 | keep |
| bfp4 mask | -0.76 | -0.95 | -0.76 | same | keep |
| fused RoPE | -2.60 | -10.45 | -13.12 | 0.997316 / 0.9297 | keep |
| attention projections bfp8 (tt.linear) | -2.73 | -10.61 | -11.99 | 0.995885 / 0.8244 | bfp8 reject, configs keep |
| MLP bfp8, text MLP(optimized) | -8.64 | -33.20 | -36.87 | 0.996286 / 0.6239 | bfp8 reject, configs keep |
| bf16 weights for both (vs bfp8) | +0.20 | +5.18 | +5.18 | 0.997025 / 0.9091 | keep (accuracy) |
| L1 activations, <= 4096 patches | -2.53 | 0.00 | 0.00 | same | keep |
| SDPA fp32 acc off | -8.82 | -23.88 | -28.34 | 0.990183 / 0.5190 | reject (E6) |
| 2D config for o_proj and down | +1.04 | -0.01 | +0.01 | 0.997025 / 0.8870 | reject |
| gelu as multiply pre-activation | -0.29 | -6.91 | -6.91 | 0.997163 / 0.8841 | keep |
| 2D config for o_proj only | -0.45 | -0.01 | -0.02 | 0.997163 / 0.8638 | keep |
| L1 rule 128 KB, 10240 patches | crash | crash | crash | — | reject (CB clash) |
| windowed SDPA, >= 8 frames at 1280 | 0.00 | 0.00 | -10.59 | 0.997163 / 0.8638 | keep |
| 128 KB L1 cap for every tensor, one image | clash | — | — | — | reject (CB clash) |
| gelu-multiply output in L1 (screen) | +0.04 | — | — | — | reject (no gain) |
| MLP up/gate outputs in L1 (screen) | -1.33 | — | — | — | keep |
| + QKV output in L1 (screen) | -0.66 | — | — | — | keep |
| bfp4 mask in L1 | fails | — | — | — | blocked (SDPA asserts a DRAM mask) |
| up/gate + QKV in L1, committed | -1.99 | 0.00 | -0.01 | 0.997163 / 0.8638 | keep (`47824a6e41`) |

Noise: window spread under 0.2 % (largest 0.32 ms at 4 x 2560); every kept change is above it.

## Notes

- The "before" Tracy uses the E2 `tt/vision` files (`git checkout 86c6b9f9f7 -- tt/vision`,
  restored by `before_tracy.sh`; the working tree was verified clean against HEAD afterwards).
- Tracy and watcher ran in separate processes.
- `capture_all` contract unchanged: the new `windows` input is allocated with the other
  persistent inputs before any capture; `trace_matches_eager` stays bit-identical (max abs 0.0).
- The first SDPA b4/b8 sweep's output was lost (the tmux command piped only its last command into
  the log); it was rerun with the helper fixed (`logs/sdpa_b4_b8.log`). The lost run's RoPE b4
  lines are in `logs/rope_b4_lost_sdpa.log`.
- HF cache `refs/main` was not modified (it reads 914f7f89 from the E5a restore); every run used
  `HF_HUB_OFFLINE=1` and the pinned snapshot path.

## Gates

`e5b/scripts/gates.sh` on HEAD `47824a6e41` (the final code commit; `dirty_tt=0`), env as above,
metrics in `e5b/results_gates.jsonl`, logs `e5b/logs/gate_*.log`. HF `refs/main` read 914f7f89
before and after (not modified). The same gates also passed on `a410c73278`
(`logs/gate_*_a410c73278.log`).

| gate | command | result |
|---|---|---|
| vision | `python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/vision -q` | 81 passed (135.8 s) |
| full model + trace | `python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model.py models/demos/blackhole/embedding_gemma2/tests/pcc/test_full_model_trace.py -q` | 41 passed (72.5 s) |
| demo | `python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/demo/demo.py -s` (as `e5a/logs/gate_demo2.log`) | 2 passed; retrieval argmax TT == HF on all 4 tasks |
| watcher | `TT_METAL_WATCHER=10 TT_METAL_LOGS_PATH=e5b/watcher python_env/bin/python -m pytest .../tests/pcc/vision -q` | 81 passed (163.1 s); `watcher.log` 653 lines, 0 matching `error|assert|sanitiz|overflow|hang|exception|fault` |
| all PCC (shared `tt/weights.py` changed in the carry-over) | `python_env/bin/python -m pytest models/demos/blackhole/embedding_gemma2/tests/pcc -q -k "not s32768"` | 343 passed, 1 deselected (441.1 s) |

Accuracy on the gates (min over cases; bars unchanged): per-layer PCC 0.999935 (video8 layer 15;
E2 0.999958), attention 0.999959, MLP 0.999987, encoder-out 0.996938 (floor 0.99), soft-token PCC
0.997163 (E2 0.997739; bar 0.99), worst soft-token cosine 0.863832 (portrait; E2 0.926977 on
landscape, HF-bf16 0.8662 on portrait), end-to-end image / video embedding cosine (full model)
0.999451 (video_text; E5a 0.999312; bar 0.99), trace replay == eager max abs diff 0.0 (vision and
full model). Fallback audit (`test_vision_fallback_audit.py`, in the vision gate): passed, 0 torch
calls, no host transfer inside the forward; Tracy after: 320 / 322 device ops, 0 host ops.
