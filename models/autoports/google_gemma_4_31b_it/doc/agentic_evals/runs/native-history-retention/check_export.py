"""Host-only paired test of pinned stock Chat and Terminus export branch."""
import ast
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path("bringup/artifacts/parser-protocol-harbor/src/harbor")
chat_source = ROOT / "llms/chat.py"
tree = ast.parse(chat_source.read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chat")
ns = dict(Path=Path, Any=Any, BaseLLM=object, LLMResponse=object, RolloutDetail=dict)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(chat_source), "exec"), ns)
term_source = ROOT / "agents/terminus_2/terminus_2.py"
branches = [
    n
    for n in ast.walk(ast.parse(term_source.read_text()))
    if isinstance(n, ast.If) and ast.unparse(n.test) == "self.options.store_all_messages"
]
assert len(branches) == 1
branch = compile(ast.Module(body=branches, type_ignores=[]), str(term_source), "exec")


async def run(enabled):
    calls = []

    class Fake:
        async def call(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return SimpleNamespace(
                response_id=None,
                usage=None,
                content='{"commands":[],"analysis":"fixture"}',
                reasoning_content="fixture thinking",
                prompt_token_ids=None,
                completion_token_ids=None,
                logprobs=None,
                extra=None,
            )

    chat = ns["Chat"](Fake(), interleaved_thinking=True)
    await chat.chat("exact initial fixture prompt")
    await chat.chat("exact fixture observation\nwith whitespace")
    before = copy.deepcopy(chat.messages)
    context = SimpleNamespace(metadata={})
    obj = SimpleNamespace(options=SimpleNamespace(store_all_messages=enabled), _chat=chat)
    exec(branch, dict(self=obj, context=context))
    assert chat.messages == before
    return calls, context.metadata, before


if __name__ == "__main__":
    a = asyncio.run(run(False))
    b = asyncio.run(run(True))
    assert a[0] == b[0] and a[2] == b[2]
    assert "all_messages" not in a[1] and b[1]["all_messages"] == b[2]
    assert b[0][1]["message_history"] == b[2][:2]
    assert b[2][2]["content"] == "exact fixture observation\nwith whitespace"
    print("Stock paired export-only check passed; no network or devices")
