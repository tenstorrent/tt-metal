"""Host-only execution of stock response-logging AST block; no vLLM import/server."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(__file__).resolve().parents[7]
source = root / "bringup/artifacts/vllm-server-source-v0.31.0/vllm/entrypoints/openai/chat_completion/serving.py"
tree = ast.parse(source.read_text())
blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno == 1207]
assert len(blocks) == 1
fn = ast.FunctionDef(
    name="check",
    args=ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=k) for k in ["self", "choices", "final_res", "request_id"]],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    ),
    body=blocks,
    decorator_list=[],
)
namespace = {}
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(source), "exec"), namespace)
rows = []
for name, content, reasoning in [
    ("visible", "synthetic answer", None),
    ("reasoning_only", None, "synthetic reasoning"),
    ("empty", "", None),
]:
    calls = []
    owner = NS(enable_log_outputs=True, request_logger=NS(log_outputs=lambda **kw: calls.append(kw)))
    choices = [
        NS(
            index=0,
            message=NS(content=content, reasoning=reasoning, tool_calls=None),
            finish_reason="length" if name != "visible" else "stop",
        )
    ]
    namespace["check"](owner, choices, NS(outputs=[NS(token_ids=[1, 2])]), "synthetic-id")
    rows.append(
        {
            "case": name,
            "completion_log_calls": len(calls),
            "id_preserved": all(x["request_id"] == "synthetic-id" for x in calls) if calls else None,
        }
    )
assert [x["completion_log_calls"] for x in rows] == [1, 0, 0]
print(
    json.dumps(
        {
            "source": str(source),
            "source_block_line": 1207,
            "cases": rows,
            "scope": "synthetic host-only extracted unchanged AST logging block; no model execution",
        }
    )
)
