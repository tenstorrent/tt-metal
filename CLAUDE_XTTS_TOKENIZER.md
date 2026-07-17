# XTTS-v2 Text Tokenizer / Normalization — TTNN bringup (Block 0)

Parent: `CLAUDE_XTTS_TTNN.md` (read it first for shared decisions + integration contract).

## Status / Owner / Started
- Status: not started
- Owner: —
- Started: —

## Role in pipeline
Text normalization + custom XTTS BPE tokenization. Runs on **CPU** (no TTNN needed —
pure preprocessing). Produces the `text_tokens` fed to the GPT (Block 3).

## Interface contract (from master)
| Direction | Tensor | Shape | dtype |
|-----------|--------|-------|-------|
| in | raw text string | — | — |
| out | `text_tokens` | (1, N_text) | int → Block 3 (GPT) |

Also owns language handling (default `en`) and any number/text normalization.

## Foundation / template
Net-new, CPU-only. Wraps coqui's XTTS tokenizer directly; no TTNN.

## Reference source
- coqui `TTS/tts/models/xtts.py` (tokenizer construction) and the XTTS `VoiceBpeTokenizer`.
- `speecht5_tts/demo_ttnn.py::normalize_text_for_tts` is a loose precedent for text
  normalization (number→words), but XTTS has its own multilingual normalizer + BPE.

## Build steps
1. Instantiate the coqui XTTS tokenizer from the checkpoint.
2. Wrap it in `reference/tokenizer.py` with a stable `encode(text, lang) -> ids` API.
3. Capture a golden `(text, lang) -> text_tokens` pair for downstream golden tests.

## PCC validation plan
Exact-match token-id comparison vs coqui tokenizer output (not PCC — discrete ids).
Test under `tests/`.

## Findings log (dated)
- (none yet)

## Open questions / TODO
- [ ] Confirm BOS/EOS/pad token handling and how tokens are concatenated with the GPT prefix.
- [ ] Confirm multilingual normalization scope needed for first bringup (en only?).
