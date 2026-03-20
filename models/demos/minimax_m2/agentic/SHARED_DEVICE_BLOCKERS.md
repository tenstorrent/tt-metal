# N300 shared-device agentic: issues, mitigations, open gaps

## Goal

Run **all agentic tools** (LLM, Whisper, BERT QA, OWL-ViT, SpeechT5) on **one** open **N300 (1×2) mesh**: load → warmup (including Whisper decoder trace capture where applicable) → sequential inference, **without hangs** and without requiring separate process-per-model.

The stress script is:

`models/demos/minimax_m2/agentic/tests/run_all_tools.py`
(use `--skip-llm` when Hugging Face access to the gated Llama checkpoint is unavailable).

---

## What we observed (symptoms)

| # | Symptom | When |
|---|---------|------|
| A | First Whisper `transcribe()` / **decoder trace capture** appears to **stall** (no progress past encoder / SDP logs) | **OWL-ViT, BERT, and/or SpeechT5 already loaded** on the same mesh (original “load everything then warm up” order). |
| B | Metal **allocator warning**: allocating device buffers is **unsafe** while a **persistent trace** is active; buffers may corrupt when trace runs | Right after Whisper **persistent decoder trace** capture completes; further allocations on the mesh are flagged. |
| C | Run **stops making log progress** at **`[3/5] BERT warmup call`** (`bert.qa(...)`) | After Whisper warmup + trace release + loading remaining models (empirical; long/indefinite wait — treated as **hang** until proven slow compile). |
| D | New run **blocks on UMD mutex** (`CHIP_IN_USE_0_PCIe` held by another PID) | A **previous Python process** still holding the device (often a **stuck** `run_all_tools.py`). |
| E | Device open shows **dispatch / `run_mailbox`** warnings or `TT_FATAL` lines during init | After **abrupt termination** (`kill -9`) of a stuck workload — firmware/dispatch may be in a bad state. |
| F | LLM load/infer **fails with HTTP 401** | Default HF model is **gated**; no token / cache / login. |

---

## Understanding (working model)

1. **Whisper + co-resident models**
   Whisper’s first generation **captures a persistent decoder trace**. That path is sensitive to **what else is already resident** on the mesh (DRAM, programs, trace region). Loading **OWL / BERT / SpeechT5 first** correlated with **Whisper stalling** during trace capture.
   **Tearing down only SpeechT5** did **not** fix Whisper — **OWL and BERT** could remain active on the mesh, so the failure mode is **broader than a single tool**.

2. **Persistent trace vs. further allocations**
   While the trace is **active**, Metal reports that **new device allocations are unsafe**. Loading large models (e.g. BERT) **after** Whisper warmup **without** releasing the trace plausibly contributes to **undefined behavior or hangs**.
   Mitigation: **release the Whisper decoder trace** after Whisper warmup (`WhisperGenerator.cleanup()` / `WhisperTool.release_decoder_trace()`). The **next** `transcribe()` should **re-capture** the trace.

3. **BERT warmup after other models**
   Even with trace release, **PHASE 0b weight load** completed, but **first BERT `qa()`** still showed **no further logs** in extended waits — suggesting either a **very long** first compile, a **deadlock**, or **interaction between full-mesh BERT and chip0 submesh models** (OWL-ViT, SpeechT5).
   **Load order was changed** to **BERT before OWL-ViT and SpeechT5** to test whether chip0-heavy setup before first BERT forward was involved. **End-to-end PASS through Phase 1b/2 was not confirmed** in session logs.

4. **Process and device hygiene**
   A stuck run must be **terminated** (e.g. kill the stuck `python ... run_all_tools.py` PID) to clear the **chip-in-use** lock. **Do not** rely on the agent to run `tt-smi -r` / hardware reset; per project rules, **user-driven reset** may be needed if the device is left inconsistent after kills (symptom E).

5. **LLM**
   The **five-tool** goal includes LLM; **gated weights** block that path unless `HF_TOKEN` / login / local cache is set.

---

## Mitigations implemented (code)

- **`run_all_tools.py` — staged flow**
  - **PHASE 0a / 1a:** Load **Whisper** (and LLM if not `--skip-llm`) → LLM warmup (optional) → **Whisper warmup** (trace capture).
  - **After Whisper warmup:** **`release_decoder_trace()`** + `gc` + `synchronize_device`.
  - **PHASE 0b:** Load **BERT first**, then **OWL-ViT** and **SpeechT5** on **chip0** submesh (`SpeechT5Tool(..., warmup_on_init=False)`).
  - **PHASE 1b / 2:** Warmups and inference as before.

- **`WhisperTool` / Whisper demo pipeline**
  - Pipeline callable exposes **`whisper_generator`** so agentic code can call **`release_decoder_trace()`** (wraps `WhisperGenerator.cleanup()`).
  - **`close()`** also attempts cleanup.

---

## What is still stopping full goal attainment

| Blocker | Status |
|---------|--------|
| **Prove E2E** `run_all_tools.py` completes **Phase 1b + Phase 2** with all stages **PASS** (or document exact failure) | **Open** — BERT warmup hang or extreme latency not closed out. |
| **Root cause** of BERT stall (mesh vs submesh, CQ, allocator, program cache, or model-specific compile) | **Open** — needs targeted experiments (BERT-only after Whisper+release, profiler, smaller repro). |
| **LLM in the same run** | **Open** unless HF auth for gated checkpoint is available. |
| **Device state after kill** | Operational — user may need **reset** if init warnings / fatal mailbox reads persist. |

---

## Related files

| File | Role |
|------|------|
| `agentic/tests/run_all_tools.py` | Unified multi-tool load/warmup/infer script |
| `agentic/tool_wrappers/whisper_tool.py` | `release_decoder_trace()`, `close()` |
| `models/demos/audio/whisper/demo/demo.py` | `whisper_generator` attached to pipeline |
| `models/demos/audio/whisper/tt/whisper_generator.py` | `_release_decoder_trace()`, `cleanup()` |
| `BRINGUP_LOG.md` | Session status + block hash |

---

## Session log pointer

Update `models/demos/minimax_m2/BRINGUP_LOG.md` when closing a session: **[Status, PCC, Block Hash]** per project rules. This document is the **technical appendix** for shared-device agentic debugging.
