# MiMo-V2.6-Flash multimodal evals: OpenRouter conformance, benchmark suite, tt-inference-server plan

Evals workstream of [multimodal_plan.md](multimodal_plan.md). Goal: show that the TT endpoint passes an
OpenRouter-style evaluation, i.e. it accepts every request form OpenRouter sends without server errors, and it scores
within noise of a reference deployment on the benchmarks OpenRouter routes by plus a per-modality suite.

Everything here was measured on bhg4 on 2026-11-07 without a TT server (none existed): the offline conformance run uses
vLLM 0.30.0's own chat front end and the upstream MiMo processor on CPU, and the benchmark harness ran in dry-run mode.
Numbers marked "assumed" are inputs to estimates, not measurements.

## What OpenRouter checks

From OpenRouter's provider and Auto Exacto docs (copies of 2026-11-06):

* **Benchmarks (Auto Exacto).** GPQA-Diamond (temperature 0.5, 10 epochs, options shuffled deterministically per
  question, unlike openbench whose fixed reseed puts every answer at B) and tau2-bench Airline (temperature 0, 1 epoch,
  an LLM user simulator). A run counts only with at least 50 GPQA questions / 45 tau2 tasks; the score is a rolling 32-day
  window; the derank threshold is the median of all endpoints' scores over the model's first ~21 days minus 2 sigma
  (computed once 4 endpoints exist). An endpoint with no benchmark data is deprioritized.
* **Tool calls.** Per request with `finish_reason: tool_calls`: invalid JSON arguments, an unknown function name, or a
  JSON-Schema (draft 7) mismatch marks the request as errored. Truncation by the caller's `max_tokens` does not count.
* **Uptime.** 5xx, mid-stream errors and "successful requests with error finish reasons" count against the endpoint;
  400, 413, 429 and 403 do not. 422 is in neither list.
* **Media.** `xiaomi/mimo-v2.6-flash` advertises text + image + video + audio input (7 providers on 2026-11-06:
  GMICloud bf16, Xiaomi / Cohere / Makora / DeepInfra / Venice fp8, Darkbloom fp4). Image and video input can be turned
  off per endpoint (an empty `sources` list in the feed, or the video-input PATCH route); audio cannot ("audio has no
  sources yet", "Image and audio input stay required"). Baseline tests run the media checks for every source that is on.
  Multimodal quality is not benchmarked by Auto Exacto.

## Files (`tests/evals/`)

| File | What |
|---|---|
| `mm_requests.py` | The OpenRouter request forms and their media fixtures (shapes on white, a solid-colour clip, the MLK speech clip converted to every OpenRouter audio format); 55 cases, each with the wanted outcome: 200 with a sane answer, or a 4xx with a message |
| `test_api_conformance.py` | Online half: the cases against `BASE_URL` as raw JSON; fails on a 5xx, a mid-stream error, a bad tool call, or an unhealthy `/health` after the case |
| `conformance_offline.py` | Offline half: the cases through vLLM's front end (request validation, media fetch / decode, chat template, `MiMoV2OmniMultiModalProcessor`), no engine; the HTTP status the server would answer (exceptions mapped by vLLM's own `create_error_response`) and the placeholder counts. `--requests` measures captured benchmark requests |
| `lmms_plugin/mimo_lmms/` | lmms-eval model `mimo_openrouter` (OpenRouter-shaped requests, resume journal, retry policy by status, stats, dry run) and the `videomme_short` task |
| `openbench_tasks/gpqa_shuffled.py` | GPQA-Diamond with OpenRouter's settings and a per-question shuffle (seeded by a hash of the question) |
| `run_eval.sh`, `summarize_results.py` | One command per benchmark, one `summary.json` |
| `prefetch_datasets.py` | Video-MME short videos only (HTTP range reads of the 101 GB zip set: 300 videos, 3.1 GB, 3 min), tau2 data at the installed tau2 commit, GPQA |

Why a custom lmms-eval model: lmms-eval's own `openai` backends send video as `file://` plus vLLM's `media_io_kwargs`
(or as extracted PNG frames) and audio only from local mp3/wav paths (an HF audio column becomes a "video" part), and
tt-inference-server's pin (v0.4.1) also caps `max_new_tokens` at 4096. OpenRouter accepts none of that, so a TT run
and a reference run would not see the same request.

## API conformance

Offline half, measured (`conformance_offline.py`, vLLM 0.30.0 from the serving env, `--max-model-len 131072`; the
serving env as is, and with torchaudio 2.11.0 added from a private `--target` dir). 55 cases; "as is" / "+torchaudio":

| Group | Result |
|---|---|
| Images: https URL, data URI png / jpeg / webp / gif (static, animated), detail low / high, 3 images, multi-turn, image first, 4096² | 200 in both. Tokens: 448² = 196, 4096² = 8,100 (resized to the 8,388,608-pixel cap), animated GIF = first frame. `detail` is ignored (vLLM logs a warning) |
| Image errors: bad base64, truncated PNG / 404 URL | 400 / 400 / 422 (`VLLMUnprocessableEntityError`) |
| Audio: wav, mp3, flac, m4a, ogg, aiff, aac, 48 kHz stereo, 8 kHz, 10-min mp3, mismatched format label, vLLM `audio_url` | **500 as is** (`ModuleNotFoundError: torchaudio`, every audio request); 200 with torchaudio. 13 s clip = 82 tokens, 10 min = 3,751 (6.25 / s) |
| Audio `pcm16`, `pcm24` (OpenRouter formats) | **400 in both** ("Invalid or corrupted audio data"): headerless PCM has no sample rate |
| FLAC encoded to a pipe (no total-samples field) | 400 in both, "file contains 418293516410647.4 s": the duration guard reads a missing length as garbage |
| Video: data URI mp4 / webm / mov / mpeg, `video/mov` MIME, https URL, with an audio track, 60 s 720p | 200 in both. Every video is 32 frames (vLLM's loader default) = 16 temporal grids: 320×240 = 1,280 tokens, 720p = 14,080, plus ~114 timestamp tokens |
| Legacy `"type": "input_video"` | **400 in both** ("Unsupported chat content part type") |
| Mixed text + image + audio, image + video, streaming, tools with an image, reasoning on | 200 (audio cases need torchaudio) |
| Reasoning off: `chat_template_kwargs.enable_thinking=false`, `reasoning_effort: "none"` | prompt ends `<think></think>` (off) |
| Reasoning off with OpenRouter's own `reasoning: {"enabled": false}` | **ignored**: thinking stays on |

vLLM's limit check, measured with the same front end: the largest item MiMo's processor can produce is 8,192 tokens for
an image and 16,384 (at 32K max-model-len) or 114,688 (at 1M) for a video. The plugin forces
`disable_chunked_mm_input`, and vLLM 0.30 then refuses to start when that maximum exceeds `max_num_batched_tokens`
(`encoder_cache_manager.py`: "Chunked MM input disabled but max_tokens_per_mm_item ... is larger than
max_num_batched_tokens"). The TTI spec runs 4096.

The online half has not run (no server). Its logic was checked against an in-process fake endpoint: 54 of 55 cases
pass, the miss being the fake ignoring `reasoning.enabled` exactly as vLLM does.

### Gap list and proposed fix locations

| # | Gap | Effect on OpenRouter | Proposed fix | Where |
|---|---|---|---|---|
| 1 | torchaudio missing from the serving env (`python_env`, and the TTI image) | every audio request 500 (uptime), and audio input cannot be switched off on OpenRouter | install `torchaudio==2.11.0` (matches torch 2.11.0+cpu) | tt-metal `create_venv` requirements / TTI Dockerfile |
| 2 | max item tokens (image 8,192, video 16K-115K) > `max_num_batched_tokens` 4096 with `disable_chunked_mm_input` | server does not start with images or video enabled | cap per item (`--mm-processor-kwargs '{"max_pixels": ...}'` for images; video `--limit-mm-per-prompt` / processor kwargs) or raise the budget; must cover Video-MME (14,080 tokens per question at 720p) | llm / plugin workstreams, then the TTI spec |
| 3 | `input_audio` `pcm16` / `pcm24` | 400 for a documented format | wrap raw PCM in a WAV header (24 kHz mono LE) before vLLM | request shim in `serving/lb.py` and TTI `engine_proxy.py` (the proxy every request crosses) |
| 4 | legacy `input_video` part | 400 for a documented form | rewrite `input_video` to `video_url` | same shim |
| 5 | `reasoning: {"enabled": false}` / `reasoning.effort` ignored | reasoning stays on: slower, and short `max_tokens` answers come back empty | map to `chat_template_kwargs.enable_thinking` (`reasoning_effort: "none"` already works); agree the field with OpenRouter at onboarding | same shim |
| 6 | media URL fetch failure answers 422 | 422 is in neither OpenRouter list | map to 400 | same shim (or a vLLM front-end patch in the plugin) |
| 7 | pipe-encoded FLAC rejected with a nonsense duration | 400 for a valid file (not uptime) | decode with a length-agnostic reader, or accept | low priority; vLLM upstream |
| 8 | the prefix router hashes base64 media as text (`lb.py` / `engine_proxy.py` `prefix_text`) | a 20 MB video is ~80K chunk hashes per request, evicting a third of the 262,144-entry index | hash each media part by digest | `serving/lb.py`, TTI `engine_proxy.py` |
| 9 | the capability gate (multimodal_plan.md, plugin contract) | a modality the model has not enabled must be a 400, never an engine exception | as planned | plugin workstream |

## Benchmark harness

```bash
# bhg4: venv /data/ttuser-agentx/mm/evalenv (Python 3.12, lmms-eval 0.7.4, openbench 0.5.3, inspect-ai 0.3.125,
# tau2 @558e6cd, mcp<2, openai<3, torchcodec 0.17); data in /data/ttuser-agentx/mm/datasets (HF_HOME .../hf)
cd models/experimental/mimo_v2_6_flash/tests/evals
python prefetch_datasets.py videomme tau2 gpqa --durations short          # once (needs HF_HOME)
./run_eval.sh <bench> <base_url> <model> [--limit N] [--concurrency 16] [--thinking on|off|default] [--dry-run]
./run_eval.sh mmau_mini http://<host>:8000/v1 XiaomiMiMo/MiMo-V2.6-Flash-MOPD          # TT endpoint
./run_eval.sh mmau_mini https://openrouter.ai/api/v1 xiaomi/mimo-v2.6-flash --provider gmicloud   # reference
```

Benches: `mmmu_pro`, `ocrbench`, `videomme_short`, `mmau_mini`, `librispeech_clean` (lmms-eval + `mimo_openrouter`),
`gpqa_diamond` (OpenRouter protocol), `gpqa_openbench` (openbench as published), `tau2_airline`. Rerunning a command
resumes it (journal for lmms-eval benches, `inspect eval-retry` for the others). Output: `summary.json` with the score,
sample counts, requests by status, prompt / completion / reasoning tokens, latency p50 / p95, wall time, versions.

Smoke, measured: all eight benches with `--limit 5 --dry-run` load their data, build requests, score and write a summary
(rc 0). The dry run captures each request as the OpenAI client sends it and checks it against the OpenRouter shape: 0
problems in 3,075 captured requests (all of MMMU-Pro and OCRBench, 100 each of MMAU / LibriSpeech, 45 Video-MME) on the
vLLM target (`chat_template_kwargs`), and on the OpenRouter target for MMAU (`reasoning` + `provider.order` with
`allow_fallbacks: false`). A rerun answered all 5 from the journal. GPQA and tau2 ran with inspect's mock model (tau2
as both agent and user). No endpoint reachable without keys exists, so nothing has been scored for real.

Reference runs need `OPENROUTER_API_KEY` only (tau2 also uses it for its user simulator, `openrouter/openai/gpt-4.1`,
openbench's default user model). Pin GMICloud (the bf16 endpoint) and Xiaomi (first party) as two references.

## References (2026-11-07 update: multimodal only)

The text benchmarks (GPQA-Diamond, tau2) are out of scope for now; their harness code stays as is. References come
from Artificial Analysis (AA) instead of an OpenRouter run.

**AA's MMMU-Pro protocol** (artificialanalysis.ai/methodology/intelligence-benchmarking): MMMU/MMMU_Pro, 1,730
questions, 10 options, 1 repeat, pass@1, AA's own instruction prompt ("Answer the following multiple choice question.
The last line of your response should be in the following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A')."
then the question and `A) ...` options) and a multi-stage regex (primary `(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:...`,
then boxed / "answer is" / letter fallbacks, last match). General settings: temperature 0.6 for reasoning models unless
the lab recommends another (MiMo's card: 1.0, top_p 0.95), maximum output tokens the creator allows. Not stated by AA:
the split (1,730 questions with 10 options and text questions is "standard (10 options)"), image placement, the exact
per-model temperature and token cap. **AA's MiMo-V2.6-Flash score: 73.06%** (1,264 / 1,730; reasoning variant, measured
on Xiaomi's first-party API, which serves the RL checkpoint; we serve MOPD, RL plus a short MOPD stage). AA publishes no
GPQA Diamond, tau2, video or audio score for this model.

`run_eval.sh mmmu_pro_aa` runs that protocol (thinking on, temperature 1.0, top_p 0.95, max_new_tokens 131,072) and its
summary prints TT, AA, the delta, the Wilson 95% interval of TT and the 95% interval of the delta (two independent
proportions on 1,730 items each, about +/-3 points). `mmmu_pro` (lmms-eval's CoT prompt, thinking off) stays as a cheap
second image check, as does OCRBench (no reference: a regression check against our own earlier runs).

**Acceptance bars for video and audio: weaker evidence.** No MiMo-V2.6-Flash score exists, so the bars are floors from
comparable models' published numbers under other protocols (frames, prompts, dataset versions differ); they catch a
broken encoder path, they do not prove parity. `summarize_results.py` prints each against its bar
(`tests/evals/aa_reference.json`, "acceptance_bars").

| Bench | Bar | Basis (published, other protocols) |
|---|---|---|
| Video-MME short, w/o subtitles | >= 75.0% | short videos: Gemini 1.5 Pro 81.7, InternVL2.5-72B 82.8 (64 frames), Qwen2-VL-72B 80.1, GPT-4o 80.0 (video-mme.github.io leaderboard); whole set: Qwen3-Omni-30B-A3B-Instruct 70.5, Qwen2.5-Omni-7B 64.3. We sample 2 frames per second, as MiMo's own endpoint |
| MMAU test-mini | >= 65.6% | Qwen2.5-Omni-7B 65.60 (model card, split not stated); Qwen3-Omni-30B-A3B-Instruct 77.5 (MMAU-v05.15.25) |
| LibriSpeech test-clean | WER <= 2.5% | Qwen3-Omni-30B-A3B-Instruct 1.22, Qwen2.5-Omni-7B 1.8, Whisper-large-v3 1.8 (model cards) |

A stronger check than any of these, once the host fp32 multimodal reference runs end to end: the same items through
the reference and through TT, answer agreement per item.

## Results (2026-11-07)

TT rows: 4 row engines on bhg4 with the MTP release settings and the multimodal class (`serving/serve_mm_eval.sh`,
behind `serving/lb.py`), concurrency 32. TT container: the tt-inference-server image (tt-metal 6312ef6d4dc, vllm-tt-plugin
b4acf686, tt-inference-server 1e0bb95d, `--impl mimo-v2.6-flash-mm`, its engine_proxy), bhg4, concurrency 32, run
directory `/data/ttuser-agentx/mm/eval_runs/20261108T054216Z/tti`; no request failed in any set. OpenRouter: the same client (`run_suite.sh --target openrouter`) pinned to one
provider. "failed = wrong" counts a request that got no answer as wrong; "answered" scores only answered items. Run
directories on bhg4: `/data/ttuser-agentx/mm/eval_runs/20261107T043703Z/` (references and the first TT sweep),
`20261107T150944Z/tt` (Video-MME), `20261107T163302Z/tt` (VGI-Bench).

| Bench | TT rows | TT container | GMICloud (bf16) | Xiaomi (fp8, first party) | Other |
|---|---|---|---|---|---|
| OCRBench (1,000) | 84.0 | 83.6 | 87.6 | 86.4 | |
| MMAU test-mini (1,000) | 61.1 | 61.6 | 62.4 | 63.0 | |
| LibriSpeech test-clean (WER, 2,620) | 2.02 | 1.73 | 1.71 | 1.81 | |
| Video-MME short (900) | 80.8 | 81.6 | 75.6 (78.4 answered) | 81.4 | |
| MMMU-Pro, AA protocol (1,730) | 64.05 | not run | 68.38 | 69.65 | AA: 73.06 |
| VGI-Bench (439) | 36.2 | not run | | 33.0 (38.2 answered; refuses videos > 300 MB) | OpenRouter: 35.5 +/- 4 |

* TT code: OCRBench, MMAU, LibriSpeech and MMMU-Pro on 885d4142564 (before the prefix-cache race fix), Video-MME on
  9c014d841ad, VGI-Bench on fc60826e361 (a video's audio track tokenized on the device). Video-MME was 60.6 on
  885d4142564: the same-step prefix-cache race (142 of 762 answers garbage), vLLM's 32-frame sampling and the missing
  audio track; after the three fixes it is 80.8.
* VGI-Bench ran at temperature 1.0 / top_p 0.95, not OpenRouter's 0: Xiaomi's endpoint samples in thinking mode
  whatever the request asks (3 of 3 different outputs at 0), and MiMo's greedy reasoning loops to the token limit on
  hard long-video questions (4 of 5 smoke questions; also at 0.05 and 0.2, never at 1.0). On the 380 questions Xiaomi
  answered: TT 36.8, Xiaomi 38.2 (95% interval of the difference +/- 4.9). `lb.py --thinking-greedy 1.0,0.95` (and
  tt-inference-server's `TT_PROXY_THINKING_GREEDY`) applies that mapping on request; off by default.
* The image gap (OCRBench -3 to -4, MMMU-Pro -4 to -6) is not explained. OpenRouter serves the RL checkpoint, we serve
  MOPD; on uniform-colour images TT answers as the fp32 host reference does (`reference/answer_image_fp32.py`), as
  does Makora (vLLM) on OpenRouter, while Xiaomi and GMICloud answer differently.
* API conformance (`test_api_conformance.py`) on 885d4142564: 46 of 55 passed; the 9 failures were all uniform-colour
  fixtures (the checkpoint's answer, see above). The fixtures are now coloured shapes on white: 55 of 55 pass in the
  container and on 2 row engines on bhg2 (8574442976a).

### Text regression on this branch (2026-11-08)

The text suite of `mimo-v2.6/dev` rerun on bhg2 with this branch (tt-metal 8574442976a, vllm-tt-plugin b4acf686, 4 row
engines, text-only class), against the 2026-11-03 `mimo-v2.6/dev` runs on the same box and client (AgentX `agentx_head2`,
SPEED-Bench `mergecheck`). Outputs in `/home/zni/mimo/mm/mmimg/text/` on bhg2.

* Device tests: smoke, MoE, spec inputs (12), spec groups, verify sampler, traced, large, decode and spec decode all
  pass. VSB at 1 / 2 / 3 / 4 users: 225.4 / 160.9 / 103.9 / 91.3 tok/s.
* AgentX (InferenceX `inferencex-agentx-mvp`, full, 3,600 s, concurrency 16), metrics from `profile_export.jsonl`:

  | | requests | errors | full-response tok/s/user (p50) | report Out Tok/s/User (p50) | token soup | spec collapse windows |
  |---|---|---|---|---|---|---|
  | dev, 2026-11-03 | 2,102 | 3 | 108.0 (103.4) | 130.5 (104.8) | 0 | 0 |
  | multimodal | 2,115 | 3 | 110.7 (105.4) | 125.9 (107.1) | 0 | 0 |

  The 3 errors in both runs are AIPerf's `InvalidInferenceResultError` (a stream with no content). Cache hit 93.9%
  (theoretical 96.9%).
* SPEED-Bench 1k (throughput set, max 8,192 output tokens), output tok/s and per-user tok/s (p50); 0 of 176 responses
  with token soup in both:

  | concurrency | dev, 2026-11-03 | multimodal |
  |---|---|---|
  | 4 | 516 / 136 | 524 / 142 |
  | 8 | 804 / 109 | 792 / 108 |
  | 12 | 1,027 / 96 | 984 / 98 |
  | 16 | 1,180 / 86 | 1,173 / 88 |

  Per-user rates match; the lower total at concurrency 12 comes with shorter answers (mean output 5,086 vs 5,675
  tokens, sampled), which leave the batch less full.

## Suite and estimates

Chosen suite (~2 per modality plus OpenRouter's two):

| Bench | Why |
|---|---|
| GPQA-Diamond, OpenRouter protocol | Auto Exacto signal; an endpoint without it is deprioritized |
| tau2-bench Airline | Auto Exacto signal; also exercises the `mimo` tool parser end to end |
| MMMU-Pro standard (10 options), CoT | reasoning over images, multi-image prompts (up to 7 images); the benchmark MiMo-V2.5 reports (77.9) and TTI already runs for Gemma 4 |
| OCRBench | fine-grained text reading: the most sensitive to ViT numerics, no judge, 1000 items |
| Video-MME short (no subtitles) | the video benchmark MiMo-V2.5 reports (87.7, settings unstated); short = 300 videos, 3.1 GB; letter answers |
| MMAU test-mini | audio understanding (sound, music, speech), multiple choice, no judge |
| LibriSpeech test-clean (WER) | audio tokenizer / encoder fidelity: transcription errors show directly |

Not chosen: CharXiv and MathVista (GPT judge or GPT answer extraction), ChartQA / AI2D (overlap OCRBench / MMMU-Pro);
Video-MME medium / long are the extension once short passes (same data, 600 more videos).

Estimates for the TT server. Inputs: 4 row engines x 4 users = concurrency 16; decode 12.5 tok/s/user (assumed,
10-15) = 200 tok/s; prefill 5,000 tok/s per row = 20,000 tok/s (assumed); time = N·out / 200 + N·in / 20,000;
encoder (ViT / audio) time not included (not measured yet). Input tokens are measured with MiMo's processor (Video-MME:
resolution of all 300 videos through the processor's resize rule, which matched the processor on 15 of 15; audio:
durations of every clip x 6.25 tokens / s). Output tokens are assumed.

| Bench | Items | Input tok / item (measured) | Output tok / item (assumed) | Prefill | Decode | Wall at c16 |
|---|---|---|---|---|---|---|
| GPQA-Diamond (10 epochs, thinking on) | 1,980 | 252 | 6,000 (3-10K) | 25 s | 16.5 h | **~16.5 h** (1 epoch: 1.7 h) |
| tau2 Airline (thinking on) | 50 tasks x ~15 turns | policy 1,635 + tools + history | 400 per turn | small (prefix cache) | 4 waves x 15 turns x 32 s | **~0.7 h** |
| MMMU-Pro std CoT (thinking off) | 1,730 | 702 (1.2 images of 427, max 7 images, max 6,400 per image) | 600 | 61 s | 1.4 h | **~1.5 h** (thinking on, 4K out: ~10 h) |
| OCRBench | 1,000 | 721 (image 697, p95 3,726, max 8,148) | 15 | 36 s | 75 s | **~2 min** |
| Video-MME short | 900 | 12,600 (video 12,443) | 5 | 570 s (190 s with prefix-cache hits on the 3 questions per video) | 23 s | **~10 min** + ViT |
| MMAU test-mini | 1,000 | 140 (audio 86) | 10 | 7 s | 50 s | **~1 min** + audio encoder |
| LibriSpeech test-clean | 2,620 | 69 (audio 47) | 30 | 9 s | 6.6 min | **~7 min** + audio encoder |

The text benchmarks dominate: the whole multimodal set is ~2 h, GPQA at OpenRouter's 10 epochs ~16 h at the assumed
decode rate (the README's documented text throughput at c16, ~1,000 tok/s with MTP, would make it ~3.3 h). A reference
run costs little: at GMICloud's list price ($0.112 / $0.224 per M) the full suite including GPQA is a few dollars;
tau2's gpt-4.1 user simulator is about 6M input tokens (~$12, assumed 8K tokens per user turn).

## tt-inference-server plan

Done on tt-inference-server `mimo-v2.6-flash/multimodal` (`--impl mimo-v2.6-flash-mm`: the MTP entry's engine settings
plus the multimodal class, frames at 2 per second, the proxy shim, the multimodal eval venvs and tasks); the plan below
is kept as written. Images are built from tt-metal `mimo-v2.6/multimodal` and vllm-tt-plugin
`mimo-v2.6-flash/multimodal`; these multimodal branches take the text branches' changes, they do not merge back.

Read on bhg4 at `mimo-v2.6-flash/inference-server-dev` 81b1f848. MiMo lives only in `workflows/model_specs/dev/llm.yaml`
(three entries: GPU reference, `impl: mimo_v2_6_flash`, and the default MTP entry), `model_type` unset (LLM),
`supported_modalities` unset (`["text"]`). Its evals (`reference_config/evals/eval_config.py:2860-3141`) are text and
agentic: `r1_gpqa_diamond` (lm-eval, temperature 1.0 / top_p 0.95, GPU reference 85.35), `longbench2_generate`,
`terminal_bench_2_1`, `tau3_bench_banking` (MiMo as its own user simulator), `deepswe`. VLM evals run lmms-eval
v0.4.1 (`requirements/evals-vision.txt`) through `openai_compatible`; Gemma 4's `mmmu_pro_standard_cot` (thinking off,
graded against a same-protocol GPU subset) and `fleurs_en` (WAV `input_audio` through `lm_eval_request_overrides.py`)
are the closest templates. No video eval exists anywhere.

1. **Spec** (`workflows/model_specs/dev/llm.yaml`, both MiMo TT entries, or a new multimodal entry so text-only stays
   byte-identical):
   * `model_type: VLM`, `supported_modalities: [text, image, audio, video]` (this alone enables the multimodal eval
     tasks and the guidellm omni scenarios for MiMo; it adds no vLLM flag);
   * `vllm_args`: `hf-overrides: '{"architectures": ["MiMoV2OmniForCausalLM"]}'`, `limit-mm-per-prompt:
     '{"image": 8, "video": 1, "audio": 4}'`, `media-io-kwargs: '{"video": {"num_frames": 32}}'` (pin vLLM's default
     so every deployment samples alike), and the per-item cap / `max_num_batched_tokens` the llm workstream settles
     (gap 2);
   * image: torchaudio in the Docker image (gap 1).
2. **Eval venv.** New `WorkflowVenvType.EVALS_MULTIMODAL` (lmms-eval 0.7.4 + torchcodec + the `mimo_lmms` plugin, vendored
   or installed from this tt-metal path) instead of bumping `EVALS_VISION`, so other models' baselines do not move.
   New `EVALS_OPENBENCH` (openbench 0.5.3, inspect-ai 0.3.125, tau2 @558e6cd with its data at that commit, `mcp<2`,
   `openai<3`) in `workflows/workflow_venvs.py` + `requirements/`.
3. **Eval tasks** (`eval_config.py`, MiMo's `EvalConfig`): `mmmu_pro_standard_cot`, `ocrbench`, `videomme_short`,
   `mmau_test_mini`, `librispeech_test_clean` with `eval_class="mimo_openrouter"`, `modalities` set, `thinking=False`,
   `max_concurrent=16`, `unit: "WER"` for LibriSpeech; `gpqa_diamond_openrouter` and `tau2_airline` with
   `eval_class="openbench"` (command: `inspect eval` / `bench eval` with `openai-api/tt/<model>`, logs parsed from the
   `.eval` file). References: `gpu_reference_score` from a same-protocol run (`run_eval.sh ... --target openrouter
   --provider gmicloud`, or a GPU vLLM server), tolerance 0.05 except tau2 (0.10, as tau3); `limit_samples_map` smoke
   0.01, CI nightly 0.2 (GPQA: 1 epoch).
4. **Command builder** (`llm_module/eval_command.py`): branches for the two new eval classes (`run_eval.sh`'s arguments).
5. **Report check** (`test_module/llm_tests/llm_eval_tests.py:779`): the inference-error scan globs `samples_*.jsonl`,
   lmms-eval writes `{date}_samples_{task}.jsonl`, so failed requests score as wrong silently; glob `*samples_*.jsonl`
   and count `[MIMO_REQUEST_FAILED` / `[LMMS_EVAL_REQUEST_FAILED` texts.
6. **Proxy** (`vllm-tt-metal/src/engine_proxy.py`): the request shim of gaps 3-6 and media-digest prefix keys (gap 8),
   kept identical to tt-metal's `serving/lb.py`.
7. **Benchmarks:** `calculate_vision_tokens` (`reference_config/benchmarking/benchmark_config.py:321`) returns 0 for
   MiMo: add its rule (ceil-to-32 resize, 8,192-token cap, tokens = (H/32)(W/32)); the `--tools vllm` benchmark drops
   image points, so MiMo's multimodal perf runs through the guidellm omni scenarios (MMBench, Video-MME,
   librispeech_asr_dummy), which `supported_modalities` enables.
8. **Release:** `ReleaseWorkflow` already runs evals, benchmarks and spec tests; with the spec and tasks above MiMo's
   release run picks up the multimodal tasks. GPQA at 10 epochs is too long for every release (estimate above): run 1
   epoch in release / nightly and 10 epochs before OpenRouter onboarding.

## Serving image with the shared KV store (2026-11-09)

`ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.26.1-ec294657718ee4133bbd28ae8215b8799dbfa09c-88e5ca8-mm-kvstore`
(private; index digest `sha256:967ca1cae048117bf94a445305f751cc52c907770fbc42c5d53182f8da8ba562` with the package
description as an `org.opencontainers.image.description` index annotation, amd64 manifest
`sha256:b9cce638027b03a92490f43d2e3a2ac7edf05151ec34abe493d008f2c444ac1d`, 21.8 GB, vLLM 0.30.0).
Built on bhg2 with `scripts/build_single_docker.sh --build --ubuntu-version 22.04 --kvstore-bin <ttserve-kvstore>` from
tt-metal `mimo-v2.6/multimodal` ec294657718, vllm-tt-plugin `mimo-v2.6-flash/multimodal` 88e5ca8 and
tt-inference-server `mimo-v2.6-flash/multimodal` 29be1ec3c; `ttserve-kvstore` is a release build of
github.com/zhenghaoniTT/ttserve main (head e8081870c on the build date).

What it adds to the multimodal image: `/usr/local/bin/ttserve-kvstore`. The launcher starts it before the four row
engines when the spec configures vllm-tt-plugin's `TTKVStoreConnector` (the `mimo-v2.6-flash-mm` entry does: 96 GiB
global + 24 GiB sliding-window pools in `/dev/shm`, `docker_shm_size: 128g`), so a prompt prefix one engine computed
is loaded by the others instead of prefilled again. Block keys chain the identifier and offset of every media item
that overlaps the block (as vLLM's own prefix-cache hash does), so two images of the same size never share KV; a
request with an unidentified item is neither loaded nor saved, and a prefill row the model could not serve is not
saved. The launcher serves without the store when the binary is missing, `TT_KVSTORE=0`, or `/dev/shm` is too small.

Container check on bhg2 (`mm_kv_check.py`: a ~13K-token text prompt, then media item A or B of the same size, greedy
one-word answers; A is computed and saved on one engine and read on another):

| | image | video |
|---|---|---|
| reference A / B (own cache salts) | Red / Blue | Red / Blue |
| load A on another engine | 13440 of 13513 tokens from the store, 0.3 s (2.2 s computed), Red | 13696 of 13745, 0.4 s, Red |
| load B (other item, same placeholder tokens) | only the 12288 text tokens, Blue | 12288, Blue |
| API conformance through the proxy | 55/55 | |

Running it (the shape of tt-inference-server's `docker run`): `--ipc private --shm-size 128g`, `/dev/tenstorrent`
and `/dev/hugepages-1G`, the runtime spec JSON with `RUNTIME_MODEL_SPEC_JSON_PATH`, the checkpoint read-only with
`MODEL_WEIGHTS_DIR`, and a persistent `cache_root` for the converted weights (~186 GB; the check mounted an existing
cache read-only and set `TT_CACHE_PATH` to it). Two things to know:

* **Cold start.** With an empty kernel cache the engines took 1899 s to become healthy (warmup compilation 1690 s;
  converted weights loaded in about a minute). tt-inference-server does not set `TT_METAL_CACHE`, so the JIT cache
  (14 GB) lives in the `--rm` container and every start pays it again; pass `-e TT_METAL_CACHE=` a directory inside
  the persistent `cache_root`, as the check did.
* **Offline weights.** vLLM resolves `--model` through the Hugging Face hub cache; with `HF_HUB_OFFLINE=1` (so a cache
  miss cannot start a full checkpoint download) seed `$HF_HOME/hub/models--XiaomiMiMo--MiMo-V2.6-Flash-MOPD` with
  `refs/main` and a snapshot link to the mounted checkpoint.

## OpenRouter's provider checks: two failures (2026-11-12)

OpenRouter's onboarding run against the deployed image (tt-metal ec294657718, vllm-tt-plugin 88e5ca8, vLLM 0.30.0)
failed two checks. Both were reproduced on bhg2 with that code (one row engine, `serve_mm_eval.sh`), and both are fixed
in the proxy (`serving/lb.py` and tt-inference-server's `engine_proxy.py`); the engine is unchanged. The two requests
are conformance cases now (`openrouter_audio_transcription`, `openrouter_youtube_url`; 57 cases).

**input-audio-transcription: an empty completion.** The request is a system prompt ("You are a transcription engine.
Reply with only the exact words spoken in the audio, with no commentary."), a user turn holding only a 2 s WAV (8-bit,
22.05 kHz), streamed, temperature 1, top_p 0.95, no `reasoning` field, so thinking is at its default (on). The answer
had the transcription in `reasoning` and nothing in `content`.

* On the device: 50 of 50 answers that way (and 24 of 24 not streamed, 20 of 24 without the system prompt); with
  thinking off, 24 of 24 have the transcription as content. The raw tokens are `<think><think>the words<|im_end|>`
  (sometimes one `<think>`), also at temperature 0: the block is never closed.
* It is the checkpoint, not the port. The fp32 host reference on the same 48-token prompt, teacher-forced along that
  answer (`reference/answer_audio_fp32.py`, log-probabilities): after `assistant\n`, `<think>` 0.00; after
  `<think>`, `<think>` -0.01, `This` -5.38, `</think>` -16.78; after the words, `<|im_end|>` 0.00. The device has
  `<think>` -0.35 and `This` -1.27 at the second position and the same tokens elsewhere.
* Only transcription-only audio requests do this. An audio question that needs reasoning, an image with the same kind
  of system prompt and text prompts all give `<think>...</think>answer`.
* vLLM's `mimo` reasoning parser (0.30's, and 0.32's, which the plugin backports to 0.31) files an unclosed block
  under reasoning, so moving to vLLM 0.31 does not change it.
* The fix: a chat answer that stops with `finish_reason` "stop", blank content, no tool call and text in reasoning
  gets that text as its content too (`AnswerFromReasoning`; `TT_PROXY_ANSWER_FROM_REASONING=0` turns it off).
  Streamed, it is one content delta before the last event. Through the fixed proxy: 50 of 50 streamed and 16 of 16
  not streamed have the transcription as content.
* Not covered: a request with `include_reasoning: false` (or `reasoning.exclude`), where vLLM drops the reasoning
  before the proxy sees it; that answer stays empty. Not checked: the RL checkpoint (the one OpenRouter's other
  providers serve), which is not on bhg2, and whether those providers pass the check by answering without thinking.

**input-youtube-url: 400 "Could not open video stream".** The request's `video_url` is a YouTube watch link. The proxy
downloaded the watch page (HTML) and the engine's video decoder rejected it. A 400 does not count against uptime, but
the message named a decoder. The proxy now refuses `youtube.com` / `youtu.be` / `youtube-nocookie.com` links by name
before resolving them, and any URL that answers `text/html` or `application/xhtml+xml`, with a 400 that says to send
the file itself or a base64 data URI. YouTube links stay unsupported.

API conformance through each fixed proxy in front of the same engine: 57 of 57 (`lb.py`), 57 of 57
(`engine_proxy.py`). Through the proxy as deployed, `openrouter_audio_transcription` fails and the other 56 pass.

## Not done

* The text benchmarks (GPQA-Diamond, tau2) are out of scope for now.
* MMMU-Pro (8 h) and VGI-Bench have not run in the container.
* The image gap to OpenRouter's providers is open (see Results).
* Encoder latency (ViT, audio) is not in the estimates.
