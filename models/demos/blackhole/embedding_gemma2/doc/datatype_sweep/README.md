# EmbeddingGemma 2 text tower: datatype sweep (E6a)

## Selected config: `p3_qkv_gu_exp`

The E5a text policy (bfp8 weights, HiFi2 + fp32 accumulation matmuls, bf16 activations, HiFi2 +
fp32 SDPA, HiFi4 norms, bfp4 masks) with three changes:

- the fused QKV projection writes its output in bfp8 (`attn_qkv.out_dtype`),
- the MLP gate and up projections write their outputs in bfp8 (`mlp_gate_up.out_dtype`),
- SDPA runs with `exp_approx_mode` on.

It is the default: `tt/precision.py` loads `selected_precision_config.json` for every text
construction path (`TextModel.from_state_dict`, `TextEncoder.from_pretrained`,
`EmbeddingGemma2.from_pretrained`).

| metric (vs live HF fp32, sweep screen cases) | E5a baseline | selected | gate |
|---|---|---|---|
| min per-layer PCC (teacher forced, gated set) | 0.99990 | 0.99970 | >= 0.995 |
| min attention / MLP module PCC (layers 0, 5) | 0.99988 | 0.99959 | >= 0.995 |
| min last-hidden PCC (free running) | 0.99692 | **0.99002** | >= 0.99 |
| min embedding cosine (per prompt) | 0.99657 | 0.99269 | >= 0.99 |
| demo retrieval argmax TT == HF | yes | yes | equal |

Traced device time per forward, one P150a, burst median of 5 windows of 20 replays after 3 s
idle; sweep harness `tests/perf/dtype_sweep.py` **[M]**:

| batch x bucket | E5a baseline ms | selected ms | change |
|---|---|---|---|
| 1 x 128 | 4.92 | 4.69 | -4.7 % |
| 1 x 512 | 7.31 | 6.49 | -11.3 % |
| 1 x 2048 | 22.05 | 17.07 | -22.6 % |
| 8 x 512 | 29.40 | 22.65 | -23.0 % |
| 1 x 8192 | 105.3-109.7 (see caveats) | 79.36 | -25 % |
| ranking aggregate 1x512 + 8x512 | 36.71 | 29.14 | -20.6 % |

Post-selection, normal construction path (`tests/perf/bench_sustained.py`, `TextEncoder.from_pretrained`
with no overrides, code commit `b6e6e9d30e`), and the safe baseline in the same session through
`EMBEDDING_GEMMA2_PRECISION=baseline_precision_config.json` **[M]** (`e6/post_bench.jsonl`):

| batch x bucket | selected burst / sustained ms | baseline burst / sustained ms | burst change |
|---|---|---|---|
| 1 x 128 | 4.691 / 4.687 | 4.923 / 4.920 | -4.7 % |
| 1 x 512 | 6.490 / 6.485 | 7.316 / 7.312 | -11.3 % |
| 1 x 2048 | 17.069 / 17.093 | 22.042 / 22.049 | -22.6 % |
| 8 x 512 | 22.653 / 23.294 | 29.392 / 29.413 | -22.9 % |
| 1 x 8192 | 79.308 / 81.856 | 105.690 / 106.550 | -25.0 % |

Burst: 5 windows of 20 replays after 3 s idle; sustained: after a 10 s soak, 5 windows of 100
replays (30 at 1x8192). The selected config's sustained time rises at 8x512 and 1x8192 (+2.8 % /
+3.2 % over burst), the P150a current-limit throttle seen in E5a.

Acceptance bar (the "minimum allowed accuracy"): every existing text gate passes unchanged with the
candidate as the model default: `pytest tests/pcc -q -k "not s32768"` (per-layer / module PCC
>= 0.995, last-hidden PCC >= 0.99, embedding cosine >= 0.99, trace == eager, multimodal cosine),
`test_full_model.py`, `demo/demo.py`, and `test_text_model[b1_s32768]`. The sweep screens on the
same metrics and cases (all of `tests/common.CASES` plus the batch-8 padded case, and the demo
retrieval), then the selected config runs the full gate set (Gates).

**Margin.** The binding metric is the free-running last-hidden PCC floor: 0.99002 against 0.99
(padded case `b2_s512_s33`). The next-fastest passing configs keep more margin at a cost:
`p3_qkv_gu` (no exp_approx) 31.06 ms / 0.99184, `act_bfp8_attn_qkv` alone 33.55 ms / 0.99308.
The safe E5a baseline is one override away (below).

## Pareto charts

x: traced device ms (burst median, 1x512 + 8x512); every evaluated config is a point, blue passes
the screen, grey fails it; line: non-dominated frontier; red: selected; dotted: gate minimum. Left
panel: every config; right: zoom on the passing ones.

![embedding cosine vs traced ms](text_cosine_perf_pareto.png)

![per-layer PCC vs traced ms](text_pcc_perf_pareto.png)

The per-layer PCC never binds (every config >= 0.9958); the last-hidden PCC and the embedding
cosine do. `sweep_results.csv` lists every metric, including the last-hidden PCC per config.

## What was swept (47 configs, `sweep_results.{json,csv}`)

Ranking aggregate in ms (baseline 36.71); "pass" = all screen bars met.

| lever | configs | best result | decision |
|---|---|---|---|
| bf16 weights | `bf16w_hifi2` | 1x128 +6 %, 1x512 +3.4 %; cosine 0.99944 | rejected: slower; trace replay hung at 1x2048 (below) |
| bfp4 weights, per group, HiFi2 and LoFi twins | 14 (`bfp4_<group>_{hifi2,lofi}`) | -0.15 ms at best | rejected: every group fails; LoFi == HiFi2 accuracy with bfp4 |
| LoFi matmuls (bfp8) | `bfp8_lofi`, `bfp8_lofi_{mlp,attn,ple}` | all: -0.25 ms, last hidden 0.98650 | all-LoFi fails; per-group LoFi passes but is within 0.1 ms of HiFi2: not kept |
| matmul fp32 accumulation off | `bfp8_hifi2_nofp32acc` | -0.14 ms, pass (0.99066) | not kept (noise-level gain, costs margin) |
| SDPA fp32_dest_acc off | `sdpa_nofp32acc` | -1.67 ms, pass (0.99240) | does not combine (see p3_qkv_sdpafast) |
| SDPA exp_approx | `sdpa_exp_approx` | -1.44 ms, pass (0.99608) | **kept** |
| SDPA LoFi | `sdpa_lofi`, `p2_sdpa_lofi_sliding` | -0.39 ms | rejected: attention PCC 0.99047 (layer 5, 8192); with full layers excepted last hidden 0.97774 |
| norms HiFi2 / fp32 acc off | `norms_hifi2`, `norms_nofp32acc` | -0.14 / +0.44 ms | not kept |
| projection outputs bfp8 (activation dtype) | `act_bfp8`, `act_bfp8_nofinal`, `act_bfp8_<group>` | all: -6.56 ms but last hidden 0.97814; qkv -3.16, gate/up -2.49 | **kept for attn_qkv and mlp_gate_up** |
| residual stream bfp8 | `resid_bfp8` | -0.35 ms | rejected: cosine 0.96778 |
| mask dtype bfp8 / bf16 vs bfp4 | `mask_bfp8`, `mask_bf16` | bit-identical accuracy, no speedup | bfp4 kept; context contract unchanged |
| combinations | `p2_*`, `p3_*` | see below | |

Combinations (accuracy erodes additively in 1 - PCC, so only the largest levers combine):

| config | ms | last hidden | cosine | result |
|---|---|---|---|---|
| `p3_qkv_gu_exp_ple` (+ ple outputs bfp8) | 28.39 | 0.98913 | 0.99240 | fail |
| **`p3_qkv_gu_exp`** | **29.14** | **0.99002** | **0.99269** | **pass, selected** |
| `p3_qkv_gu_exp_exc_full` (layers 5/11/17/23 keep bf16 outputs) | 30.75 | 0.99069 | 0.98988 | fail (and slower) |
| `p3_qkv_gu` | 31.06 | 0.99184 | 0.99467 | pass |
| `p3_qkv_sdpafast` (qkv + SDPA fp32 acc off + exp) | 31.17 | 0.98925 | 0.99245 | fail |
| `p3_qkv_exp` | 31.62 | 0.99146 | 0.99461 | pass |
| `p2_sdpa_fast` (SDPA fp32 acc off + exp) | 35.04 | 0.99094 | 0.99605 | pass (exp_approx adds nothing on top of fp32 acc off) |
| `p2_sdpa_fast_mmnofp32`, `_normhifi2`, `_mmnofp32_normhifi2` | 34.77-34.92 | 0.97277-0.98865 | | fail |

Layer exceptions were tried where they could help (`p2_sdpa_lofi_sliding`: LoFi SDPA on sliding
layers only; `p3_qkv_gu_exp_exc_full`: full-attention layers keep bf16 outputs). Neither passed;
the bfp4 groups were too far from the bar (cosine 0.91-0.99, last hidden 0.91-0.99) for first /
last layer exceptions to be worth a run, and none of them was faster than bfp8.

Per material matmul group (weight dtype, fidelity, ms aggregate, accuracy, decision):

| group | bfp8 HiFi2 (base) | bfp8 LoFi | bfp4 HiFi2 | bfp4 LoFi | bfp8 output |
|---|---|---|---|---|---|
| attn_qkv | 36.71 pass | 36.60 pass (with attn_o) | 36.72 fail cos 0.941 | 36.58 fail | **33.55 pass (kept)** |
| attn_o | 36.71 pass | (with attn_qkv) | 36.68 fail cos 0.915 | 36.66 fail | 36.68 pass, no gain |
| mlp_gate_up | 36.71 pass | 36.61 pass (with down) | 36.73 fail cos 0.957 | 36.56 fail | **34.23 pass (kept)** |
| mlp_down | 36.71 pass | (with gate_up) | 36.68 fail cos 0.953 | 36.65 fail | 36.65 pass, no gain |
| ple_proj | 36.71 pass | 36.68 pass (with block) | 36.69 fail cos 0.98982 | 36.68 fail | 36.39 pass; fails in combination |
| ple_block | 36.71 pass | (with ple_proj) | 36.67 fail cos 0.973 | 36.63 fail | 36.30 pass; fails in combination |
| final_proj | 36.71 pass | (in bfp8_lofi) | 36.69 fail last 0.98964 | 36.69 fail | in act_bfp8 only |

Why weights barely matter here **[I]**: bfp8 text weights are about 150 MB per forward, about
0.3 ms of DRAM time at 1x512; the forward is bound by the norms, SDPA and the activation traffic
between ops, which is what the output-dtype and SDPA levers cut.

## The policy file

`selected_precision_config.json` (schema in `tt/precision.py`; unknown or missing keys raise):

- `text.weights.<group>`: `dtype` (bf16 / bfp8 / bfp4), `fidelity`, `fp32_acc`, `out_dtype` for
  `attn_qkv`, `attn_o`, `mlp_gate_up`, `mlp_down`, `ple_proj`, `ple_block`, `final_proj`.
- `text.layer_exceptions`: per-layer overrides of the per-layer groups and SDPA (empty).
- `text.sdpa`: fidelity, fp32_acc, exp_approx. `text.norms`: fidelity, fp32_acc (every text
  RMSNorm). `text.residual_dtype`, `text.mask_dtype`.
- `vision`, `audio`: the current tower policies, recorded only; E6b wires them. The two towers keep
  their hard-coded E5b / E5c policies in E6a (the shared `tt/linear.py` / `tt/rms_norm.py`
  arguments default to the old behaviour).
- `kv_cache_dtype`, `ccl_dtype`, `logits_dtype`, `sampling`: "n/a: encoder, single chip".
- Fixed, not swept: token-embedding table bf16 (row-major `ttnn.embedding`), pooler matmul HiFi4
  with fp32 output.

Return to the safe baseline: `EMBEDDING_GEMMA2_PRECISION=<.../doc/datatype_sweep/baseline_precision_config.json>`,
or `precision=<that path>` in `TextEncoder.from_pretrained` / `EmbeddingGemma2.from_pretrained`.

## Proof the runtime consumes it

`tests/pcc/test_precision_config.py`:

- `test_default_loads_selected_file`: builds the model with no `precision=` argument, checks it was
  loaded from `selected_precision_config.json`, runs one forward with spies on `ttnn.linear`,
  `minimal_matmul`, SDPA and `rms_norm`, and checks every weight dtype, matmul fidelity / fp32 acc /
  output dtype, SDPA fidelity / fp32 acc / exp_approx, norm config, residual and mask dtype of all
  24 layers against the file read with plain `json` (1211 checks on the selected config).
- `test_override_changes_runtime`: an override (mlp_down bfp4 LoFi, layer-0 exception bf16 HiFi4
  bfp8-output, flipped exp_approx, another mask dtype) changes the built tensors and the configs
  the forward passes.
- `test_invalid_value_rejected`: an invalid dtype or a missing group raises.

Pass / fail / pass on a planted value in `selected_precision_config.json` **[M]** (logs
`e6/logs/`): the gate watcher run passes `test_default_loads_selected_file` (1211 checks);
`mlp_down.dtype` planted to `bfp5` makes it fail with `ValueError: text.weights.mlp_down.dtype:
'bfp5' not in ['bf16', 'bfp8', 'bfp4']` (`pfp_2_planted.log`); planted to the valid `bfp4`, the
default-built model holds `BFLOAT4_B` `w_down` weights (`pfp_2b_planted_valid.log`); reverted, it
passes again (`pfp_3_reverted.log`).

The sweep records the same construction evidence per candidate (`construction` in
`sweep_results.json`: per-layer weight dtypes, policies, SDPA config, residual / mask dtype).

## Gates (final code commit)

All on code commit `b6e6e9d30e` (selected config as the default, no `EMBEDDING_GEMMA2_PRECISION`),
`tt-ops/runs/embeddinggemma-2/e6/scripts/gates.sh g1`, logs `e6/logs/gate_g1_*.log`, metrics
`e6/results_gates_g1.jsonl` **[M]**:

| gate | result | worst metric |
|---|---|---|
| `pytest tests/pcc -q -k "not s32768"` | 357 passed, 1 deselected | layer PCC 0.99973, attention 0.99959, MLP 0.99979, last hidden 0.990018 (`[512, 33]`), embedding / trace cosine 0.99269 |
| `test_text_model[b1_s32768]` | 1 passed | last hidden 0.99523, cosine 0.99949 |
| `tests/pcc/test_full_model.py` | 30 passed | full-model cosine 0.99876 (audio_tone3s), batched 0.99938 |
| `demo/demo.py -s` | 2 passed | retrieval argmax TT == HF for every task |
| `tests/pcc/vision -q` (unchanged tower) | 91 passed | E5c: 91 passed |
| `tests/pcc/audio -q -k "not perf"` (unchanged tower) | 109 passed, 7 deselected | |
| watcher (`TT_METAL_WATCHER=10`, text tests + precision test, `-k "not s32768"`) | 101 passed | 0 error lines in `watcher.log` |

The full-model traced checks (`test_full_model_trace.py`, inside `tests/pcc`) passed with full-trace
cosine >= 0.99903.

## Not done / caveats

- `bf16w_hifi2`: the traced benchmark hung at 1x2048 (every core of the k-norm `rms_norm`
  [1, 2, 2048, 256] waiting, `tt-ops/runs/embeddinggemma-2/e6/triage/bf16w_1x2048.txt`). Accuracy
  (cosine 0.99944, last hidden 0.99876) and burst 1x128 / 1x512 were measured before the hang. Not
  root-caused: the config is slower than bfp8 at every measured shape, so it is not a selection
  candidate. Repro **[M]** (`e6/scripts/bf16w_repro.py`): with only the 1x2048 trace captured,
  200 replays complete; with all five perf traces captured, replaying the 1x2048 trace hangs
  after 240-260 replays (`e6/logs/bf16w_repro_{1,2}.log`). The selected config under the same
  five-trace setup ran 2000 replays at 1x2048 without a hang (`e6/logs/selected_soak_1x2048.log`).
- 1x8192 burst of the baseline is 109.7 ms when it is the first candidate of a sweep process and
  105.3-105.9 ms for the bfp8 configs with identical kernels later in a process (`mask_bfp8`,
  `bfp4_final_proj_*`) **[M]**; cause not investigated. Rankings use 1x512 + 8x512.
- Vision / audio precision: E6b, with this machinery.
