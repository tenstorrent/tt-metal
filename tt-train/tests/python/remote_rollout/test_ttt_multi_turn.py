# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn tool calling on Qwen3-1.7B.

The tests up to the device section need only the tokenizer. The device tests need one Blackhole chip:
    cd tt-train/tests/python/remote_rollout
    python3 -m pytest -s test_ttt_multi_turn.py
"""

from __future__ import annotations

import ast
import operator
import re

import pytest
from transformers import AutoTokenizer
from ttml.trainers.grpo_trainer.tool_calling import (
    Qwen3ToolCallParser,
    Tool,
    ToolCall,
    generate_tool_turns,
    run_tool_calls,
)

MODEL_ID = "Qwen/Qwen3-1.7B"
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate one arithmetic expression with + - * /."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "For example 17*23"}},
        "required": ["expression"],
    }

    def call(self, arguments: dict) -> str:
        def evaluate(node):
            if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
                return _OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"unsupported expression {arguments['expression']!r}")

        value = evaluate(ast.parse(arguments["expression"], mode="eval").body)
        return str(int(value)) if float(value).is_integer() else str(value)


@pytest.fixture(scope="module", params=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B-Base"])
def qwen3_tokenizer(request):
    return AutoTokenizer.from_pretrained(request.param)


CALC = '<tool_call>\n{"name": "calculator", "arguments": {"expression": "17*23"}}\n</tool_call>'
CALC_CALL = ("calculator", {"expression": "17*23"}, False)
ERROR = ("", {}, True)


@pytest.mark.parametrize(
    "turn_text, expected",
    [
        pytest.param("<think>\nok\n</think>\n\n<answer>391</answer><|im_end|>", [], id="final answer"),
        pytest.param("<think>\nok\n</think>\n\n" + CALC + "<|im_end|>", [CALC_CALL], id="one call after think"),
        pytest.param(CALC + "<|im_end|>", [CALC_CALL], id="one call, no think"),
        pytest.param(
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call><|im_end|>',
            [("a", {}, False), ("b", {"x": 1}, False)],
            id="two calls in order",
        ),
        pytest.param(CALC + "\nI will wait for the result.<|im_end|>", [CALC_CALL], id="text after the last call"),
        pytest.param(CALC + "<tool_response>999</tool_response><|im_end|>", [CALC_CALL], id="own tool response"),
        pytest.param("<think>\n" + CALC + "\n</think>\n\n<answer>3</answer><|im_end|>", [], id="call inside think"),
        pytest.param("<think>\n" + CALC + "<|im_end|>", [], id="call in an open think"),
        pytest.param('<tool_call>\n{"name": "a", "arguments"<|im_end|>', [], id="unclosed call"),
        pytest.param(CALC + "<|endoftext|>", [], id="ends with endoftext"),
        pytest.param(CALC, [], id="cut by the token limit"),
        pytest.param('<tool_call>\n{"name": calc}\n</tool_call><|im_end|>', [ERROR], id="bad JSON"),
        pytest.param('<tool_call>\n{"arguments": {}}\n</tool_call><|im_end|>', [ERROR], id="no name"),
        pytest.param(
            '<tool_call>\n{"name": "a", "arguments": "17*23"}\n</tool_call><|im_end|>',
            [ERROR],
            id="arguments not an object",
        ),
        pytest.param('<tool_call>\n{"name": "a"}\n</tool_call><|im_end|>', [("a", {}, False)], id="no arguments"),
        pytest.param(
            CALC + "\n<tool_call>\n{oops}\n</tool_call><|im_end|>", [CALC_CALL, ERROR], id="good call, then bad call"
        ),
    ],
)
def test_qwen3_parse(qwen3_tokenizer, turn_text, expected):
    turn_ids = qwen3_tokenizer.encode(turn_text, add_special_tokens=False)
    calls = Qwen3ToolCallParser(qwen3_tokenizer).parse(turn_ids)
    assert [(c.name, c.arguments, c.error is not None) for c in calls] == expected


def test_qwen3_tool_response_text(qwen3_tokenizer):
    parser = Qwen3ToolCallParser(qwen3_tokenizer)
    assert parser.tool_response_text(["391"]) == (
        "\n<|im_start|>user\n<tool_response>\n391\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    )
    assert parser.tool_response_text(["1", "2"]) == (
        "\n<|im_start|>user\n<tool_response>\n1\n</tool_response>\n<tool_response>\n2\n</tool_response>"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def test_qwen3_tool_response_text_matches_chat_template(qwen3_tokenizer):
    call = {"type": "function", "function": {"name": "f", "arguments": {}}}
    messages = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": "R1"},
        {"role": "tool", "content": "R2"},
    ]
    rendered = qwen3_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    assert rendered.endswith("<|im_end|>" + Qwen3ToolCallParser(qwen3_tokenizer).tool_response_text(["R1", "R2"]))


class _BrokenTool(Tool):
    name = "broken"
    description = "Always fails."
    parameters = {"type": "object", "properties": {}}

    def call(self, arguments: dict) -> str:
        raise RuntimeError("boom")


def test_run_tool_calls():
    tools = {"calculator": CalculatorTool(), "broken": _BrokenTool()}
    calls = [
        ToolCall("calculator", {"expression": "17*23"}),
        ToolCall("nope", {}),
        ToolCall("broken", {}),
        ToolCall("", {}, error="invalid tool call (bad JSON)"),
    ]
    assert run_tool_calls(calls, tools, max_chars=2048) == [
        "391",
        "Error: invalid tool call (unknown tool 'nope')",
        "Error: invalid tool call (boom)",
        "Error: invalid tool call (bad JSON)",
    ]
    assert run_tool_calls([ToolCall("calculator", {"expression": "123456789*1000"})], tools, max_chars=5) == ["12345"]


class _ScriptedSampler:
    """generate_and_get_log_probs returns scripted turns; it records each call."""

    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = [
            [tokenizer.encode(t, add_special_tokens=False) if t is not None else None for t in row] for row in turns
        ]
        self.calls = []

    def generate_and_get_log_probs(self, token_ids, max_new_tokens, *, append_to_previous=False):
        k = len(self.calls)
        self.calls.append((token_ids, list(max_new_tokens), append_to_previous))
        out = [self.turns[i][k][:limit] if ids else [] for i, (ids, limit) in enumerate(zip(token_ids, max_new_tokens))]
        return out, [[-0.5] * len(t) for t in out]


def test_generate_tool_turns_five_tool_turns(qwen3_tokenizer):
    tok = qwen3_tokenizer
    steps = ["17*23", "391+456", "847-129", "718/2", "359*3"]
    row0 = [
        f'<tool_call>\n{{"name": "calculator", "arguments": {{"expression": "{s}"}}}}\n</tool_call><|im_end|>'
        for s in steps
    ]
    row0.append("ANSWER: 391, 847, 718, 359, 1077<|im_end|>")
    row1 = ["<answer>7</answer><|im_end|>"] + [None] * 5  # no tool call: done after turn 1
    row2 = [CALC + "<|im_end|>", "<think>\nlong" * 400] + [None] * 4  # turn 2 is cut by its limit
    sampler = _ScriptedSampler(tok, [row0, row1, row2])
    parser = Qwen3ToolCallParser(tok)
    prompts = [[1, 2, 3], [4, 5], [6]]
    W = 400

    completions, logprobs = generate_tool_turns(
        sampler,
        prompts,
        tools={"calculator": CalculatorTool()},
        parser=parser,
        max_turns=8,
        max_tool_response_chars=2048,
        max_completion_length=W,
    )

    results = ["391", "847", "718", "359", "1077"]
    responses = [tok.encode(parser.tool_response_text([r]), add_special_tokens=False) for r in results]
    expected0, expected_lp0 = [], []
    for k, turn in enumerate(sampler.turns[0]):
        expected0 += turn
        expected_lp0 += [-0.5] * len(turn)
        if k < len(responses):
            expected0 += responses[k]
            expected_lp0 += [0.0] * len(responses[k])
    assert completions[0] == expected0 and logprobs[0] == expected_lp0
    assert completions[1] == sampler.turns[1][0]
    assert len(completions[2]) == W and completions[2][: len(sampler.turns[2][0])] == sampler.turns[2][0]

    assert len(sampler.calls) == 6
    assert sampler.calls[0] == (prompts, [W, W, W], False)
    assert sampler.calls[1][0] == [responses[0], [], responses[0]] and sampler.calls[1][2] is True
    assert sampler.calls[2][0] == [responses[1], [], []]
    limit = W - len(sampler.turns[0][0]) - len(responses[0])
    assert sampler.calls[1][1][0] == limit and sampler.calls[1][1][1] == 0


def test_generate_tool_turns_stops_at_max_turns(qwen3_tokenizer):
    sampler = _ScriptedSampler(qwen3_tokenizer, [[CALC + "<|im_end|>"] * 3])
    completions, _ = generate_tool_turns(
        sampler,
        [[1]],
        tools={"calculator": CalculatorTool()},
        parser=Qwen3ToolCallParser(qwen3_tokenizer),
        max_turns=2,
        max_tool_response_chars=2048,
        max_completion_length=1000,
    )
    assert len(sampler.calls) == 2
    assert qwen3_tokenizer.decode(completions[0]).count("<tool_call>") == 2


def test_tokenize_prompts_writes_tool_schemas(qwen3_tokenizer):
    from datasets import Dataset
    from ttml.trainers.grpo_trainer.rollout_batch_source import tokenize_prompts

    chat = Dataset.from_list([{"prompt": [{"role": "user", "content": "What is 17*23?"}], "answer": 391}])
    prompts, extra = tokenize_prompts(chat, qwen3_tokenizer, 1, tools=[CalculatorTool()])
    text = qwen3_tokenizer.decode(prompts[0])
    assert '"name": "calculator"' in text and text.endswith("<|im_start|>assistant\n")
    assert extra == {"answer": [391]}
    plain = Dataset.from_list([{"prompt": "plain text", "answer": 1}])
    prompts, _ = tokenize_prompts(plain, qwen3_tokenizer, 1, tools=[CalculatorTool()])
    assert prompts == [qwen3_tokenizer.encode("plain text", add_special_tokens=False)]


# ---------------------------------------------------------------- device tests (one Blackhole chip)

TASK = (
    "Compute these five steps one at a time. Step 1: {a} multiplied by {b}. Step 2: add {c} to the result of step 1. "
    "Step 3: subtract {d} from the result of step 2. Step 4: divide the result of step 3 by 2. "
    "Step 5: multiply the result of step 4 by 3. For each step, call the calculator with only that step, "
    "then wait for its result before the next step. After step 5, reply with 'ANSWER: ' followed by the five "
    "results in order, separated by commas."
)
NUMBERS = dict(a=17, b=23, c=456, d=129)
OTHER_NUMBERS = dict(a=1234, b=56, c=78901, d=2345)


def _gold(numbers):
    results = [numbers["a"] * numbers["b"]]
    results.append(results[-1] + numbers["c"])
    results.append(results[-1] - numbers["d"])
    results.append(results[-1] // 2)
    results.append(results[-1] * 3)
    return results


def _answer(tokenizer, completion):
    text = tokenizer.decode(completion).rsplit("ANSWER:", 1)[-1]
    return [int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]


@pytest.fixture(scope="module")
def sampler():
    import ttnn
    from _ttt_sampler_utils import open_sampler

    if not ttnn.get_device_ids():
        pytest.skip("needs a Blackhole chip")
    with open_sampler(dummy_weights=False, model_source=MODEL_ID, max_batch_size=2, max_seq_len=4096) as s:
        yield s


def _task_ids(tokenizer, numbers):
    messages = [{"role": "user", "content": TASK.format(**numbers)}]
    return tokenizer.apply_chat_template(
        messages, tools=[CalculatorTool().schema()], add_generation_prompt=True, tokenize=True, return_dict=False
    )


def _tool_tokens(tokenizer, turn):
    parser = Qwen3ToolCallParser(tokenizer)
    results = run_tool_calls(parser.parse(turn), {"calculator": CalculatorTool()}, 2048)
    return tokenizer.encode(parser.tool_response_text(results), add_special_tokens=False)


def _record_start_pos(monkeypatch, sampler):
    calls = []
    prefill = sampler.generator.prefill_forward_text

    def spy(*args, **kwargs):
        calls.append(kwargs.get("start_pos"))
        return prefill(*args, **kwargs)

    monkeypatch.setattr(sampler.generator, "prefill_forward_text", spy)
    return calls


@pytest.mark.timeout(0)
def test_append_to_previous_prefills_only_new_tokens(sampler, monkeypatch):
    tok = sampler.tokenizer
    W = 2048
    prompts = [_task_ids(tok, NUMBERS), _task_ids(tok, OTHER_NUMBERS)]
    first, _ = sampler.generate_and_get_log_probs(prompts, [W, W])
    appended = [_tool_tokens(tok, turn) for turn in first]
    assert all(len(a) > 0 for a in appended), "turn 1 of each row must end with a tool call"
    start_pos = [len(p) + len(c) - 1 for p, c in zip(prompts, first)]
    assert start_pos[0] != start_pos[1]
    # The generator floors start_pos to a multiple of 64; from 128 on, the prefill skips cached tokens.
    assert min(start_pos) >= 128
    limits = [W - len(c) - len(a) for c, a in zip(first, appended)]

    calls = _record_start_pos(monkeypatch, sampler)
    second, second_lp = sampler.generate_and_get_log_probs(appended, limits, append_to_previous=True)
    monkeypatch.undo()
    assert calls == [start_pos]
    assert [len(lp) for lp in second_lp] == [len(t) for t in second]

    # Greedy tokens can differ from a full prefill at near-ties (cached K/V come from decode, not from prefill),
    # so compare the answers.
    full, _ = sampler.generate_and_get_log_probs([p + c + a for p, c, a in zip(prompts, first, appended)], limits)
    gold = [_gold(NUMBERS), _gold(OTHER_NUMBERS)]
    assert [_answer(tok, c) for c in second] == [_answer(tok, c) for c in full] == gold

    # The prefill reads the earlier tokens from the KV cache: without them the result changes.
    assert sampler.generate_and_get_log_probs(prompts, [W, W])[0] == first
    sampler._reset_kv_cache()
    assert sampler.generate_and_get_log_probs(appended, limits, append_to_previous=True)[0] != second


@pytest.mark.timeout(0)
def test_append_to_previous_rules(sampler, expect_error):
    tok = sampler.tokenizer
    prompts = [_task_ids(tok, NUMBERS), _task_ids(tok, OTHER_NUMBERS)]
    sampler.generate_and_get_log_probs(prompts, [16, 16])
    with expect_error(ValueError, "one non-negative limit for each of the 2 rows"):
        sampler.generate_and_get_log_probs(prompts, [16])
    with expect_error(ValueError, "rows; the previous call had 2"):
        sampler.generate_and_get_log_probs([[1]], [4], append_to_previous=True)
    with expect_error(ValueError, "at least one row"):
        sampler.generate_and_get_log_probs([[], []], [0, 0], append_to_previous=True)
    with expect_error(ValueError, r"max_new_tokens >= 1"):
        sampler.generate_and_get_log_probs([[1], []], [0, 0], append_to_previous=True)
    sampler.generate_and_get_log_probs(
        [tok.encode("\nok", add_special_tokens=False), []], [4, 0], append_to_previous=True
    )
    with expect_error(ValueError, "row 1 is idle"):
        sampler.generate_and_get_log_probs([[1], [1]], [4, 4], append_to_previous=True)


@pytest.mark.timeout(0)
def test_max_new_tokens_per_row(sampler):
    tok = sampler.tokenizer
    prompts = [_task_ids(tok, NUMBERS), _task_ids(tok, OTHER_NUMBERS)]
    completions, logprobs = sampler.generate_and_get_log_probs(prompts, [5, 17])
    assert [len(c) for c in completions] == [5, 17]
    assert [len(lp) for lp in logprobs] == [5, 17]
