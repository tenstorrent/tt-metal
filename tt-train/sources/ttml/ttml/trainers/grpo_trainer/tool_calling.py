# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tool calling for multi-turn GRPO rollouts: the Tool interface, the tool-call parser of each model family,
and the turn loop on top of a sampler's ``generate_and_get_log_probs``."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type


class Tool(ABC):
    """One tool that the model can call. Subclass it and pass instances to ``GRPOTrainer(tools=...)``."""

    name: str
    description: str
    parameters: dict

    @abstractmethod
    def call(self, arguments: dict) -> str:
        """Run the tool. The return value is the tool response text."""

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class ToolCall:
    name: str
    arguments: dict
    error: Optional[str] = None


class ToolCallParser(ABC):
    """Tool-call protocol of one model family."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    @abstractmethod
    def parse(self, turn_ids: List[int]) -> List[ToolCall]:
        """Tool calls of one finished assistant turn, in order. Empty: the turn is final."""

    @abstractmethod
    def tool_response_text(self, results: List[str]) -> str:
        """Text after the turn: one user turn with all results, then the header of the next assistant turn."""


class Qwen3ToolCallParser(ToolCallParser):
    """A turn that ends with <|im_end|> calls the tools in its complete <tool_call>{...}</tool_call> pairs after </think>."""

    _CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def __init__(self, tokenizer: Any) -> None:
        super().__init__(tokenizer)
        self._end_of_turn_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def parse(self, turn_ids: List[int]) -> List[ToolCall]:
        if not turn_ids or turn_ids[-1] != self._end_of_turn_id:
            return []
        text = self.tokenizer.decode(turn_ids)
        if "<think>" in text:
            if "</think>" not in text:
                return []
            text = text.rsplit("</think>", 1)[1]
        calls = []
        for body in self._CALL.findall(text):
            try:
                call = json.loads(body)
                calls.append(ToolCall(name=str(call["name"]), arguments=dict(call.get("arguments", {}))))
            except (ValueError, KeyError, TypeError) as e:
                calls.append(ToolCall(name="", arguments={}, error=f"invalid tool call ({e})"))
        return calls

    def tool_response_text(self, results: List[str]) -> str:
        body = "".join(f"\n<tool_response>\n{result}\n</tool_response>" for result in results)
        return f"\n<|im_start|>user{body}<|im_end|>\n<|im_start|>assistant\n"


TOOL_CALL_PARSERS: Dict[str, Type[ToolCallParser]] = {"qwen3": Qwen3ToolCallParser}
_MODEL_TYPE_TOOL_CALL_FORMATS: Dict[str, str] = {"qwen3": "qwen3"}


def default_tool_call_format(model_type: str) -> str:
    """The tool_call_format of a HuggingFace model_type, for a config that does not set one."""
    if model_type not in _MODEL_TYPE_TOOL_CALL_FORMATS:
        raise ValueError(
            f"no default tool_call_format for model_type {model_type!r}; "
            f"set grpo_config.multi_turn.tool_call_format to one of {list(TOOL_CALL_PARSERS)}"
        )
    return _MODEL_TYPE_TOOL_CALL_FORMATS[model_type]


def run_tool_calls(calls: List[ToolCall], tools: Mapping[str, Tool], max_chars: int) -> List[str]:
    """Run each call in order. Errors go back to the model as the tool response."""
    results = []
    for call in calls:
        tool = tools.get(call.name)
        if call.error is not None:
            result = f"Error: {call.error}"
        elif tool is None:
            result = f"Error: invalid tool call (unknown tool {call.name!r})"
        else:
            try:
                result = str(tool.call(call.arguments))
            except Exception as e:
                result = f"Error: invalid tool call ({e})"
        results.append(result[:max_chars])
    return results


def generate_tool_turns(
    sampler: Any,
    prompts: List[List[int]],
    *,
    tools: Mapping[str, Tool],
    parser: ToolCallParser,
    max_turns: int,
    max_tool_response_chars: int,
    max_completion_length: int,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]]]:
    """Up to max_turns assistant turns per prompt. After each turn, run its tool calls and append the tool responses;
    the sampler prefills only the appended tokens. Returns (completions, logprobs, loss_masks); tool-response tokens
    have log-prob 0.0 and loss mask 0.0. A row ends when its turn has no tool call, when a turn reaches its token
    limit, or when the tool responses do not fit max_completion_length."""
    completions, logprobs = sampler.generate_and_get_log_probs(prompts, [max_completion_length] * len(prompts))
    loss_masks = [[1.0] * len(c) for c in completions]
    turns = [list(c) for c in completions]
    active = [True] * len(prompts)
    for _ in range(max_turns - 1):
        appended: List[List[int]] = [[] for _ in prompts]
        for i in range(len(prompts)):
            calls = parser.parse(turns[i]) if active[i] else []
            if calls:
                text = parser.tool_response_text(run_tool_calls(calls, tools, max_tool_response_chars))
                tokens = parser.tokenizer.encode(text, add_special_tokens=False)
                if len(completions[i]) + len(tokens) < max_completion_length:
                    appended[i] = tokens
            active[i] = bool(appended[i])
        if not any(active):
            break
        limits = [max_completion_length - len(c) - len(a) if a else 0 for c, a in zip(completions, appended)]
        turns, turn_logprobs = sampler.generate_and_get_log_probs(appended, limits, append_to_previous=True)
        for i, tokens in enumerate(appended):
            if tokens:
                completions[i] += tokens + turns[i]
                logprobs[i] += [0.0] * len(tokens) + turn_logprobs[i]
                loss_masks[i] += [0.0] * len(tokens) + [1.0] * len(turns[i])
    return completions, logprobs, loss_masks
