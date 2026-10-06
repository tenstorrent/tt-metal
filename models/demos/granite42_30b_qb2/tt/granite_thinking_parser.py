from typing import Sequence

import vllm
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser

_VLLM_VERSION = tuple(int(x) for x in vllm.__version__.split(".")[:2])


# --- Monkeypatch for vLLM >= 0.20 (no upstream hook exists) ---
# Patches DelegatingParser.parse_delta to strip leading \n from content
# at the reasoning->content boundary.

if _VLLM_VERSION >= (0, 20):
    try:
        from vllm.parser.abstract_parser import DelegatingParser

        _original_parse_delta = DelegatingParser.parse_delta

        # v0.20-0.22: parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None)
        # v0.23+:     parse_delta(..., *, finished: bool)
        # Use **kwargs to accept both signatures.
        #
        # State is tracked on the DelegatingParser instance (self), not in a
        # global dict. This is safe because vLLM creates a new DelegatingParser
        # per streaming request (confirmed in serving.py for v0.20 through v0.28).
        # This avoids the id(request) reuse bug on v0.20-0.22 where Python
        # recycles memory addresses and stale flags cause intermittent leaks.
        def _patched_parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None, **kwargs):
            result = _original_parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids, **kwargs)
            if result is not None and getattr(result, "content", None) is not None:
                if not getattr(self, "_granite_content_started", False):
                    stripped = result.content.lstrip("\n")
                    if not stripped:
                        result.content = None
                    else:
                        self._granite_content_started = True
                        result.content = stripped
            return result

        DelegatingParser.parse_delta = _patched_parse_delta
    except (ImportError, AttributeError):
        pass


@ReasoningParserManager.register_module("granite_thinking_parser")
class GraniteThinkingParser(DeepSeekR1ReasoningParser):
    def extract_reasoning(self, model_output, request):
        reasoning_content, final_content = super().extract_reasoning(model_output, request)

        if final_content is not None:
            final_content = final_content.lstrip("\n")

        if (
            hasattr(request, "chat_template_kwargs")
            and request.chat_template_kwargs
            and (
                request.chat_template_kwargs.get("enable_thinking") is False
                or request.chat_template_kwargs.get("force_nonempty_content") is True
            )
            and final_content is None
        ):
            reasoning_content, final_content = None, reasoning_content

        return reasoning_content, final_content

    # --- vLLM < 0.20: use extract_reasoning_streaming ---

    if _VLLM_VERSION < (0, 20):

        def extract_reasoning_streaming(
            self,
            previous_text: str,
            current_text: str,
            delta_text: str,
            previous_token_ids: Sequence[int],
            current_token_ids: Sequence[int],
            delta_token_ids: Sequence[int],
        ):
            """Strip leading newlines from streaming content deltas."""
            result = super().extract_reasoning_streaming(
                previous_text,
                current_text,
                delta_text,
                previous_token_ids,
                current_token_ids,
                delta_token_ids,
            )

            if result is None:
                return None

            if result.content is not None and self.end_token_id in previous_token_ids:
                end_pos = None
                for i in range(len(previous_token_ids) - 1, -1, -1):
                    if previous_token_ids[i] == self.end_token_id:
                        end_pos = i
                        break

                if end_pos is not None:
                    content_token_ids_so_far = previous_token_ids[end_pos + 1 :]
                    if len(content_token_ids_so_far) == 0 or all(
                        self.model_tokenizer.decode([tid]).strip("\n") == "" for tid in content_token_ids_so_far
                    ):
                        stripped = result.content.lstrip("\n")
                        if not stripped:
                            return None
                        return type(result)(content=stripped)

            return result
