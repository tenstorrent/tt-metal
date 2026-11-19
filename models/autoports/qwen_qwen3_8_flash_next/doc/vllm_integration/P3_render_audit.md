# P3 c07 rendering audit — 2026-11-16

The advisory `render_c07` failure identifies an upstream endpoint normalization
inconsistency for the deprecated `reasoning_content` input field. It does not
show that chat completion discarded the history. The saved c07 completion
returned exactly `Ada`, HTTP 200, with `finish_reason="stop"`, 88 prompt tokens,
two completion tokens and zero generated reasoning tokens. The corresponding
c06 request without historical reasoning used 73 prompt tokens and also returned
`Ada`. The actual response is coherent, correctly recalls the name, and contains
no repetition, gibberish, language drift or visible request contamination.

Evidence is [chat_conformance.json](../eval_readiness/chat_conformance.json) and
[`cases.c07_history_with_reasoning` in responses.json](../eval_readiness/chat_conformance/responses.json).
All critical P3 checks passed. Preserve the original advisory failure; this audit
does not relabel it as a passing exact-token comparison.

## Source diagnosis

Installed upstream source root is
`/home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm/`.
The saved response identifies `vllm-0.31.0-b15bbf9b`.

| Source under that root | Relevant behavior |
| --- | --- |
| `entrypoints/openai/chat_completion/protocol.py:521`, `ChatCompletionRequest._normalize_messages_before` | Renames deprecated message `reasoning_content` to `reasoning`, unless a non-null canonical value already exists. |
| `entrypoints/serve/tokenize/protocol.py:46`, `TokenizeChatRequest` | Separate request model with no equivalent reasoning alias normalizer. Its before-validator handles generation-prompt constraints. |
| `entrypoints/serve/tokenize/serving.py:84` | Sends tokenization request messages to the shared chat preprocessor. |
| `entrypoints/chat_utils.py:2016` and `:2040` | Reads only the canonical input `reasoning`, then supplies both `reasoning` and compatibility `reasoning_content` to assistant template messages. |

The pinned checkpoint is
`/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa/`.
Its `chat_template.jinja:111–119` reads `message.reasoning_content` and preserves
historical thinking by default. Thus chat completion normalizes the deprecated
wire field, restores the template-compatible alias, and includes the two prior
reasoning strings. `/tokenize` does not normalize that wire field and renders
empty historical thinking sections instead.

The installed skill harness source is
`codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/runtime/readiness_check/run_chat_conformance.py`
under `/home/ttuser/qwen38-flash/`. `_assistant` at line 301 constructs the
deprecated field; `_render_compare` at line 507 sends those messages unchanged
to `/tokenize`, while its HF comparison at line 522 supplies them directly to
the checkpoint template. This explains the saved mismatch at token 16:
73 `/tokenize` tokens versus 88 HF tokens. The 88-token actual completion usage
also matches the HF length, rather than the defective `/tokenize` surrogate.
Equal length alone is not proof of exact completion prompt-token equality.

## Verification and bounded follow-up

A CPU-only AST extraction executed the installed upstream
`_normalize_messages_before` function against a deep copy of the saved c07
request. Both assistant messages retained their content and reasoning strings,
with `reasoning_content` renamed to `reasoning`. Source inspection verified the
separate tokenization request model lacks this transformation. No vLLM,
transformers, torch or TTNN module was imported; no HTTP or device operation
was performed. No plugin, upstream runtime or skill source was modified.

If an additional exact render check is needed, use the saved c07 payload and
change only the two assistant wire keys sent to `/tokenize` from
`reasoning_content` to `reasoning`. Compare its full token array against HF
rendering of the original messages (the checkpoint expects the compatibility
field). Keep the thinking flags and every message value unchanged. This tests
canonical-field render parity without deleting the reasoning text or weakening
the oracle. That HTTP check has not been run by this audit.

Carry the precise limitation: the pinned upstream `/tokenize` endpoint is not
an exact chat-completion rendering proxy for deprecated reasoning-history
fields. The current artifacts demonstrate correct c07 completion content and
source-supported preservation of history, but do not contain the full actual
completion prompt-token array. This does not justify a model/plugin change.
