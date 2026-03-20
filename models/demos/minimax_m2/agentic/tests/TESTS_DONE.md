# Systematic Multi-Model N300 Testing - Results Log

This file tracks passing tests for the shared-device multi-model workflow.

## Device Parameters (shared config)

```python
l1_small_size=24_576
trace_region_size=100_000_000
num_command_queues=2
```

---

## Level 0: Individual Model Standalone Tests

| Model | Status | Date | Notes |
|-------|--------|------|-------|
| Whisper | PASS | 2026-03-20 | Trace capture + release works |
| BERT | PASS | 2026-03-20 | Uses full mesh |
| OWL-ViT | PASS | 2026-03-20 | Uses chip0 submesh |
| SpeechT5 | PASS | 2026-03-20 | Uses chip0 submesh, KV-cache mode |
| LLM | SKIP | | Requires HF gated model auth |

---

## Level 1: Pairwise Tests

| Pair | Status | Date | Notes |
|------|--------|------|-------|
| Whisper + BERT | PASS | 2026-03-20 | Whisper-first staged, trace release before BERT |
| Whisper + OWL-ViT | PENDING | | |
| Whisper + SpeechT5 | PENDING | | |
| BERT + OWL-ViT | PASS | 2026-03-20 | Both use different mesh views |
| BERT + SpeechT5 | FAIL→PASS | 2026-03-20 | PASS when BERT uses chip0 instead of full mesh |
| OWL-ViT + SpeechT5 | PASS | 2026-03-20 | Both on chip0 submesh |

---

## Level 2: Three-Model Tests

| Combination | Status | Date | Notes |
|-------------|--------|------|-------|
| BERT + OWL-ViT + SpeechT5 | PASS | 2026-03-20 | All on chip0 submesh |
| Whisper + BERT + OWL-ViT | PENDING | | |

---

## Level 3: Full Four-Model Test (--skip-llm)

| Test | Status | Date | Notes |
|------|--------|------|-------|
| Whisper + BERT + OWL-ViT + SpeechT5 | **PASS** | 2026-03-20 | All models on chip0 submesh |

---

## Key Finding: All Models Must Use chip0 Submesh

**Root Cause**: Mixing full-mesh models with chip0-submesh models causes hangs.

- Full-mesh Whisper + chip0 BERT → BERT warmup hangs
- Full-mesh BERT + chip0 SpeechT5 → SpeechT5 warmup hangs

**Solution**: Run ALL models on chip0 submesh.

This sacrifices multi-chip parallelism but enables all models to coexist without deadlocks.

---

## Final Architecture

```
N300 (1×2 mesh)
├── chip0 (all models run here)
│   ├── Whisper STT
│   ├── BERT QA
│   ├── OWL-ViT detection
│   ├── SpeechT5 TTS
│   └── LLM (if loaded)
└── chip1 (unused — available for future optimization)
```

**DRAM Budget (chip0 only):** ~9.9 GB / 12 GB (~2 GB headroom)
