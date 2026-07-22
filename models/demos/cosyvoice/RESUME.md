# RESUME.md — Fresh-agent entry point

> **Read this first**, then `BRINGUP_PLAN.md` §0 + §1.1 + §11.
> Last updated: 2026-07-22 (**ALL PHASES COMPLETE** — Phase 0 + spikes + Phase 2a/2b/2c +
> Phase 3 E2E + Phase 4 verification/perf + Phase 5 docs. D1–D26 done. Project complete.)

This file is the *concise* handoff. The authoritative plan with full detail is
`BRINGUP_PLAN.md`; this document points you at the right section and gives the
exact next command.

## TL;DR — where we are

Stage-1 bring-up of **CosyVoice2-0.5B** on Tenstorrent **Wormhole N300** via
**TTNN** is **COMPLETE** (D1–D26). All phases done: env + checkpoint + curated deps,
config harness, golden fixtures, all spikes, LLM on N300 (PCC 0.997), flow (PCC 1.0),
vocoder (PCC 1.0), E2E pipeline (20 WAVs, 4 modes × 5 langs), verification (C6–C8
measured), and docs (README + model-card + Stage-2 roadmap).

**Final metrics:** LLM decode 34.1 tok/s ✓ | Token accuracy 96–100% ✓ | WER 0.000 ✓ |
Speaker similarity 82.9 ✓ | E2E RTF 2.17 (Stage-2 target; flow on host is bottleneck).

## What is DONE (verified — see BRINGUP_PLAN.md §11.2 for full table D1–D23)

- Phase 0.1–0.5: repo layout, ref repo @ `074ca6d` + Matcha submodule, HF
  checkpoint @ `eec1ae6c` (4.6 GB), curated `requirements-cosyvoice.txt` (torch
  2.11.0+cpu preserved), full import smoke test.
- **0.6**: `tt/model_config.py` (frozen dataclasses) + `scripts/extract_config.py`
  (yaml→config regression harness, plain-dict load, green). **U1 + U2 RESOLVED.**
- **0.7**: `scripts/gen_golden.py` — 4 modes (zero_shot/cross_lingual/instruct2/sft),
  seed=1986 + RAS sampling (seeded), per-mode `.pt` fixtures (LLM logps/tokens/rng_state,
  flow mu/dphi_dt/mel/x_init/t_span/token/embedding/prompt_feat, hift mel_in/f0/source/waveform)
  + WAVs in `model_data/golden/`. **U2–U5 RESOLVED.** Transformers 5.10 compat:
  eager attention + full decode mask (lesson 12). Token counts: 284/195/287/126.
- **Spikes**: (a) ESPnet rel-pos → fresh module (tt_transformers is RoPE, can't
  express it) — D17; (b) `conv_transpose1d→conv_transpose2d` POC PCC≥0.99999 —
  D16, **U9+U10 RESOLVED**; (c) DSP-glue: iSTFT+SineGen2 → host Stage 1 —
  **U11 RESOLVED** (iSTFT issue still to file).
- **Phase 2a (D21)**: `tt/weights.py` (llm.pt→Meta-format, 292 keys + speech heads),
  `tt/llm/model.py` (CosyVoiceLLM on N300), `tt/llm/sampling.py` (RAS port).
  **On N300**: prefill PCC=0.9969, decode PCC=0.996–0.998, top-25 agreement=96.7%.
  **U8 RESOLVED.** Lessons 14–15 added (bf16 sampling metric, 128-align prefill).
- **Phase 2b (D22)**: `tt/flow/encoder.py` (UpsampleConformerEncoder, host torch,
  ESPnet rel-pos attn, PCC=1.0 vs reference), `tt/flow/flow_matching.py`
  (FlowEncoderModel: tokens→mu/spks/cond), `tt/flow/unet_estimator.py` (wraps
  reference CausalConditionalDecoder), `tt/flow/cfm.py` (Euler solver + CFG).
  **PCC=1.0** for mu/spks/cond + dphi_dt + mel across all 4 modes.
  Lessons 16–18 added (bidirectional attn, ESPnet PE xscale, CFG zeros).
- **Phase 2c (D23)**: `tt/hifigan/generator.py` (HiFTVocoder, host torch, weight-norm
  folded via `remove_parametrizations`). U15 RESOLVED (torch 2.x parametrizations API,
  328→246 keys). U16 RESOLVED (Snake alpha `[C]` unsqueezed to `[1,C,1]`).
  U17 RESOLVED (ConvRNNF0Predictor = 5×Conv1d+ELU + Linear, NOT an RNN).
  `tests/pcc/test_hift_module.py`: **waveform PCC=1.0, f0 PCC=1.0**, MCD 0.82–1.03 dB
  across all 4 modes.
- **Phase 3 (D24)**: `tt/pipeline.py::TtnnCosyVoice` — non-streaming E2E orchestration
  wiring `CosyVoiceLLM` (N300) + `FlowEncoderModel`+`CausalConditionalCFM` (host) +
  `HiFTVocoder` (host). Reuses reference `CosyVoiceFrontEnd` for host glue (text
  normalize, Qwen tokenizer, `speech_tokenizer_v2.onnx`, `campplus.onnx`, mel
  extraction). SFT bridge (lesson 9) in `add_zero_shot_spk`. `demo/data/texts.json`
  (zh/en/ja→katakana/yue/ko × 2 sentences). `demo/demo.py` (pytest, 20 WAVs →
  `demo/output/`) + `demo/try_it.py` (interactive). `tests/e2e/test_modes.py`: 4-mode
  waveform sanity + teacher-forced top-25 token accuracy >95%. **Exit gate MET**:
  demo 20 passed (6m21s), e2e 5 passed, pcc 32 passed (regression green). Lessons 21–22.

## What is REMAINING

**Nothing — all phases complete (D1–D26).** The only outstanding item is filing
the two GitHub issues (drafted in `GITHUB_ISSUES.md` with final results) on
`tenstorrent/tt-metal` — user will do this manually.

Stage-2 work (future): trace+2CQ, flow on device, bi-streaming, batching,
on-device sampling, 2nd N300 chip. See README.md § Stage-2 Roadmap.

## Known unknowns (MUST verify — do not assume)

`BRINGUP_PLAN.md` §11.6 — **U1–U17 are ALL RESOLVED** (moved to §11.2 DONE).
No open unknowns remain. Phase 3 (E2E pipeline) has no blocking unknowns.

## Session-start checklist (run this first)

```bash
# 1. Activate the env (mandatory first step).
source /root/tt-metal/python_env/bin/activate

# 2. Sanity-check heavy artifacts present (regen if missing).
cd /root/tt-metal/models/demos/cosyvoice
ls model_data/CosyVoice_src/cosyvoice/cli/cosyvoice.py        # ref repo
ls model_data/cosyvoice2-0.5B/llm.pt                          # checkpoint
ls model_data/CosyVoice_src/third_party/Matcha-TTS/matcha/    # submodule
ls model_data/golden/llm/zero_shot.pt                          # golden fixtures
# If ref/checkpoint missing: python scripts/clone_reference.py && python scripts/download_model.py
# If golden missing: python scripts/gen_golden.py --modes zero_shot,cross_lingual,instruct2,sft

# 3. Verify curated deps installed + ttnn still imports (~3s).
cd model_data/CosyVoice_src
python - <<'PY'
import sys; sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.flow.flow_matching import CausalConditionalCFM
from cosyvoice.hifigan.generator import HiFTGenerator
import ttnn
print("env OK")
PY

# 4. Verify the config regression gate is green.
cd /root/tt-metal/models/demos/cosyvoice
python scripts/extract_config.py   # must print "OK ... U1 ... RESOLVED."

# 5. Read BRINGUP_PLAN.md §11.2 (DONE) + §11.3 (REMAINING) to find your next step.
```

If step 3 fails after a fresh env, re-run:
```bash
uv pip install --python /root/tt-metal/python_env/bin/python \
  -r /root/tt-metal/models/demos/cosyvoice/requirements-cosyvoice.txt
```
then re-run the smoke test. If `ttnn` fails to import after that, the torch ABI
was disturbed — investigate before proceeding (the curated file never touches
torch, but a transitive dep might).

## Hard-won Phase-0 lessons (do not relearn — full list in §11.4)

1. **Never install CosyVoice's upstream `requirements.txt` verbatim.** It pins
   `torch==2.3.1` + `onnxruntime-gpu`, which downgrades torch and breaks `ttnn`.
   Use `requirements-cosyvoice.txt` (curated, CPU-only).
2. **The Matcha-TTS submodule is a transitive-deps landmine.** It drags in
   `conformer`, `diffusers`, `hydra-core`, `lightning`, `gdown`, `wget` — all
   needed for the import surface even though CV2's estimator doesn't use them.
3. **`pip` is not on the env's PATH.** Use `uv pip install --python
   /root/tt-metal/python_env/bin/python <pkg>`.
4. **`snapshot_download` with `local_dir` returns a path, not a revision SHA.**
   Use `model_info(...).sha` to pin (already handled in `download_model.py`).
5. **`example.py` uses relative `model_dir='pretrained_models/CosyVoice2-0.5B'`.**
   Pass the absolute `model_data/cosyvoice2-0.5B/` path to `AutoModel(...)` to
   sidestep the relative-path assumption.
6. **CV2 HF snapshot has NO `spk2info.pt`.** SFT = bootstrap a zero-shot speaker.
7. **`example.py::cosyvoice2_example()` covers 3 of 4 modes** (zero_shot,
   cross_lingual, instruct2) + the SFT bootstrap, but never calls
   `inference_sft` after the bootstrap, and doesn't cover all 5 languages —
   Phase 3 must add both.
8. **Golden-gen torch-2.11 compat workarounds (CPU-only; do NOT affect the TTNN
   bf16 port):** (a) `pyworld` stub (training-only dep, inference never calls
   processors); (b) reimplement `load_wav`/`torchaudio.save` via `soundfile`
   (torchaudio 2.11 routes through uninstalled `torchcodec`); (c)
   `cv.model.{llm,flow,hift}.float()` cast (Qwen2.5 loads bfloat16 from BlankEN
   but CosyVoice heads are fp32 → dtype mismatch in assembled lm_input).
9. **SFT-mode reference-repo quirk (U5): `frontend_sft` reads
   `spk2info[spk_id]['embedding']` (singular), but `add_zero_shot_spk` stores
   `llm_embedding`/`flow_embedding`.** `gen_golden.py::run_sft` bridges it by
   copying `llm_embedding` into an `embedding` key after bootstrap.
10. **`HiFTGenerator.inference` base class (CV2 path) computes f0 in the
    predictor's NATIVE dtype — it does NOT cast to float64** (only
    `CausalHiFTGenerator` does). Mutating `f0_predictor.to(float64)` in
    instrumentation breaks subsequent modes.
11. **Thread exceptions don't propagate through `Thread.join()`.** The LLM decode
    runs in `llm_job` via `threading.Thread`; if it crashes, `p.join()` returns
    silently and tokens stay empty — always check the LLM produced tokens, not
    just the final audio.
12. **Transformers 5.10 breaks CosyVoice's Qwen2 attention (CRITICAL).** Two
    fixes needed: (a) force `attn_implementation='eager'` (SDPA mishandles
    CosyVoice's custom 1D mask); (b) decode attention mask must cover full
    KV-cache length, not just `[1,1]`. Both in `gen_golden.py::_patch_qwen2_encoder()`.
13. **Pure greedy (argmax) causes degenerate period-2 token loops.** CosyVoice2
    requires RAS (repetition-aware sampling). Golden fixtures use seeded RAS,
    NOT greedy.
14. **Token accuracy with bf16 + RAS: use top-k agreement, not exact match.**
    bf16 logits cause different multinomial draws — exact match is ~4%. Correct
    metric: golden token within RAS window (top_k=25). Measured 96.7%.
15. **tt_transformers prefill requires 128-aligned seq_len.** Decode `current_pos`
    must be `ttnn.Tensor` (shape `[batch]`, int32), not Python int.
16. **Flow encoder uses FULL BIDIRECTIONAL attention in non-streaming mode** (NOT
    causal). Causal mask gives PCC 0.90 instead of 1.0.
17. **ESPnet rel-pos PE requires `xscale=sqrt(d_model)` + flip+concat PE generation.**
    Missing xscale gives PCC 0.05.
18. **CFM CFG uses ZEROS for unconditioned path**, NOT duplicates. `mu_in[0]=mu`,
    `mu_in[1]=0`. Using `repeat_interleave` gives PCC 0.80 instead of 1.0.
19. **Weight-norm fold: use `remove_parametrizations`, NOT legacy `remove_weight_norm`.**
    `hift.pt` uses torch 2.x `parametrizations.weight.original0/original1`. Legacy
    API raises ValueError. Correct: `remove_parametrizations(module, 'weight',
    leave_parametrized=True)`. 328→246 keys after fold.
20. **HiFTGenerator yaml config differs from constructor defaults:**
    `source_resblock_kernel_sizes=[7,7,11]` (3 blocks, NOT the default `[7,11]`
    with 2 blocks). Always instantiate from yaml values, not code defaults.
21. **Mode-specific LLM prefix assembly (Phase 3).** Reference builds
    `lm_input = [sos, embed(concat(prompt_text, text)), task_id, speech_embed(prompt_speech_token)]`.
    zero_shot: `concat(prompt_text,text)` + prompt speech tokens. cross_lingual:
    text only, no LLM prompt speech tokens (flow still gets flow_prompt_speech_token +
    prompt_speech_feat). instruct2: `concat(instruct_text,text)`, no LLM prompt speech
    tokens. sft: text only, flow gets empty prompt token/feat. `min_len=tts_text_len*2`,
    `max_len=tts_text_len*20`.
22. **E2E token-accuracy must be TEACHER-FORCED, not free-run (Phase 3).** Free-run
    pipeline-vs-golden token comparison gives ~10% (RNG state diverges + bf16 changes
    multinomial draws). Correct metric: feed golden `lm_input` + golden tokens
    teacher-forced, check golden token ∈ model top-25 log-probs each step (>95%).

## Working agreement (carried from the original task)

- Keep `BRINGUP_PLAN.md` as the living source of truth. When you resolve an open
  item, record the confirmed fact in §1.1 / §9 / §11 and cross it off.
- Do not start coding a component until its Phase-1 op table is complete and any
  missing-op issue is filed (TTNN guide §2.6).
- File GitHub issues immediately for any genuinely-missing TTNN op.
- PCC gates: ≥0.99 per op/module/component before proceeding; token accuracy
  >95% for the LLM (top-25 agreement metric, RAS seeded — NOT greedy; lesson 14).
- bfloat16 end-to-end in Stage 1; `math_fidelity` per TTNN guide.
- Commit nothing unless asked. Do not run git operations on tt-metal unless asked.
- Non-streaming path only (flow `streaming=False`, hift `finalize=True`).
