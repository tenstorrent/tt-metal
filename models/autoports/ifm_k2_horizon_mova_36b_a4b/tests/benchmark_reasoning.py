# SPDX-License-Identifier: Apache-2.0
"""vLLM parser plugin for the pinned native K2 Horizon high-effort template.

Load with --reasoning-parser-plugin <this file> --reasoning-parser k2_horizon.
The native template opens <ifm|think> in the prompt, so generated text begins
inside reasoning even when no opening marker occurs in the generated tokens.
This plugin only separates output channels; it never modifies chat inputs.
"""

from collections.abc import Sequence

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class K2HorizonReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return "<ifm|think>"

    @property
    def end_token(self) -> str:
        return "</ifm|think>"

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text) if delta_text else None
        # Special markers are atomic tokenizer tokens. A scheduler chunk can
        # contain either marker alone or markers together with ordinary text.
        reasoning = delta_text
        if self.start_token_id in delta_token_ids:
            reasoning = reasoning.partition(self.start_token)[2]
        if self.end_token_id in delta_token_ids:
            reasoning, _, content = reasoning.partition(self.end_token)
            return DeltaMessage(reasoning=reasoning or None, content=content or None)
        return DeltaMessage(reasoning=reasoning) if reasoning else None

    def count_reasoning_tokens(self, token_ids: Sequence[int]) -> int:
        # vLLM passes generated IDs only, excluding the prompt's opening tag.
        count = 0
        for token_id in token_ids:
            if token_id == self.end_token_id:
                break
            if token_id != self.start_token_id:
                count += 1
        return count


ReasoningParserManager.register_module("k2_horizon", module=K2HorizonReasoningParser)
