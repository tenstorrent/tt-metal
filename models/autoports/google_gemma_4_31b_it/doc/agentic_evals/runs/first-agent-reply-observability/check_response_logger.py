"""Host-only unchanged stock middleware AST with synthetic response bytes."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(__file__).resolve().parents[7]
source = root / "bringup/artifacts/vllm-server-source-v0.31.0/vllm/entrypoints/serve/middleware/log_response.py"
tree = ast.parse(source.read_text())
nodes = [
    n
    for n in tree.body
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    and n.name in ["_log_non_streaming_response", "log_response"]
]
assert len(nodes) == 2
logs = []


async def iterate(items):
    for item in items:
        yield item


ns = {
    "Request": object,
    "logger": NS(info=lambda fmt, *args: logs.append(fmt % args)),
    "iterate_in_threadpool": iterate,
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), ns)


async def main():
    rows = []
    for case, content, reasoning in [
        ("visible", "synthetic answer", None),
        ("reasoning_only", None, "synthetic private reasoning"),
        ("empty", "", None),
    ]:
        logs.clear()
        payload = {
            "id": "chatcmpl-synthetic-" + case,
            "choices": [
                {"index": 0, "message": {"content": content, "reasoning": reasoning}, "finish_reason": "length"}
            ],
        }
        body = json.dumps(payload).encode()
        response = NS(body_iterator=iterate([body]), headers={"content-type": "application/json"})

        async def next_call(_):
            return response

        result = await ns["log_response"](object(), next_call)
        restored = b"".join([x async for x in result.body_iterator])
        assert restored == body
        logged = json.loads(logs[0][len("response_body={") : -1])
        assert logged == payload
        rows.append(
            {
                "case": case,
                "log_count": len(logs),
                "id_preserved": logged["id"] == payload["id"],
                "finish_reason_preserved": logged["choices"][0]["finish_reason"] == "length",
                "bytes_unchanged": True,
            }
        )
    print(
        json.dumps(
            {
                "source": str(source),
                "cases": rows,
                "scope": "host synthetic one-chunk JSONResponse path; no real server/HTTP/TT/client receipt",
            }
        )
    )


asyncio.run(main())
