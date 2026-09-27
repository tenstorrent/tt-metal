# AutoDebug: one-token native-template length difference

The eight completed MMLU-Pro responses report one more prompt token than direct HF rendering of the retained role/content messages as string-valued content. This does not establish double templating or an extra BOS.

## Cause supported by running-server logs and installed source

`run/server-b32.log:103` records that the actual server detected native chat-template content format `openai`. The installed core is `/home/container_app_user/tt-metal/python_env/lib/python3.10/site-packages/vllm`, version 0.26.0+empty. Its `entrypoints/chat_utils.py:_parse_chat_message_content` converts string content to text parts and preserves dictionary parts when `content_format == "openai"`. Its `renderers/hf.py:safe_apply_chat_template` calls the tokenizer's native template once. Its chat request `add_special_tokens` default is false, and template generation prompt defaults true.

The pinned native template (`4d7ae4984b7db7de8f8457170b3f1a419ee76d52/chat_template.jinja:197-204`) has distinct system-content branches: string content is trimmed, while sequence content emits each trimmed text item plus a trailing space. The MMLU-Pro recipe supplies a system message. Rendering the same messages with OpenAI text-part lists therefore adds one token at the system turn boundary. The earlier one-user readiness probe has no system turn and does not exhibit this difference.

Upstream lm_eval LocalChatCompletion sends structured role/content messages, without HF rendering client-side. Its recorded request kwargs preserve `enable_thinking=false`. The retained regenerated request hashes match all eight response links (separate `partial-accuracy.json` verification).

## Offline discriminating test

`tools/benchmark_template_audit.py` loads only the pinned cached HF tokenizer and renders every completed question twice: original string content and server-normalized OpenAI text-part content. Both variants contain exactly one BOS. The OpenAI-parts variant matches all eight actual API prompt-token counts:

| Actual API | HF string content | HF OpenAI text parts |
|---:|---:|---:|
| 1453 | 1452 | 1453 |
| 1116 | 1115 | 1116 |
| 1553 | 1552 | 1553 |
| 1061 | 1060 | 1061 |
| 1071 | 1070 | 1071 |
| 1512 | 1511 | 1512 |
| 1518 | 1517 | 1518 |
| 1585 | 1584 | 1585 |

Retained artifact: `run/template-audit.json`, including per-question IDs, native token hashes, BOS counts and rendering prefixes. This verifies the one-token explanation without server execution or new inference. There is no justified implementation repair: forcing string content would change the established server preprocessing rather than correct a duplicate template. The benchmark questions and generation settings remain unchanged.

Limitation: accuracy responses retain prompt lengths, not their complete actual server input-token arrays. Consequently this audit proves an exact length match and a source-supported transformation, not a byte-for-byte comparison of those eight live token arrays. The separate native readiness probe supplies that stronger token-hash check only for its own simple prompt. Do not generalize that probe to an unrecorded actual-token hash for every benchmark question.

Command: `/home/container_app_user/tt-metal/python_env/bin/python models/autoports/google_gemma_4_26b_a4b_it/tools/benchmark_template_audit.py` (CPU tokenizer only; local cached revision, no server or inference).
