# Architectural Blockers and Incompatibilities

This file documents combinations that are architecturally impossible or have fundamental conflicts.

---

## Confirmed Blockers

### 1. SpeechT5 Generator (trace mode) + Any Other Model

**Status**: CONFIRMED BLOCKER

**Symptom**: SpeechT5's trace-enabled generator requires `l1_small_size=300_000` which clashes with Whisper's L1 buffers starting at offset 1018400.

**Root cause**: With `l1_small_size=300_000`, the CB region would end at ~1185120, overlapping Whisper's buffers.

**Workaround**: SpeechT5 runs with `generator=None` (KV-cache mode without trace), which works with `l1_small_size=24_576`.

---

### 2. Full-Mesh Models + chip0-Submesh Models (Co-resident)

**Status**: CONFIRMED BLOCKER

**Symptom**: When any model uses full mesh (both chips) while other models use chip0 submesh, warmup of the chip0 models hangs indefinitely.

**Observed conflicts**:
- Full-mesh Whisper + chip0 BERT → BERT warmup hangs
- Full-mesh BERT + chip0 SpeechT5 → SpeechT5 warmup hangs

**Root cause**: Device resource conflicts between full-mesh and submesh views when multiple models are co-resident. The exact mechanism involves compiled program state and cross-chip coordination.

**Solution**: Run ALL models on chip0 submesh. This sacrifices multi-chip parallelism but enables stable co-residency.

---

## Resolved Issues

### BERT + SpeechT5 Conflict (RESOLVED)

**Original symptom**: BERT (full mesh) + SpeechT5 (chip0) caused hangs.

**Resolution**: Run BERT on chip0 instead of full mesh. Both models coexist on chip0.

---

## Notes

- LLM testing requires HuggingFace authentication for `meta-llama/Llama-3.2-3B-Instruct`
- Use `--skip-llm` flag when HF auth is unavailable
- chip1 is currently unused and available for future optimization
