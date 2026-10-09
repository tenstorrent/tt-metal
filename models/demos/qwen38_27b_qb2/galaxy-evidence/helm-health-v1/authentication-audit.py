# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import ast
import asyncio
import hashlib
import importlib.metadata
import json
import secrets
from pathlib import Path

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

path = Path(
    "/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006/serving_env/lib/python3.10/site-packages/vllm/entrypoints/serve/utils/server_utils.py"
)
source = path.read_bytes()
tree = ast.parse(source)
selected = [
    node
    for node in tree.body
    if (isinstance(node, ast.ClassDef) and node.name == "AuthenticationMiddleware")
    or (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GUARDED_PREFIX" for target in node.targets)
    )
]
assert len(selected) == 2
module = ast.Module(
    body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
    type_ignores=[],
)
namespace = dict(hashlib=hashlib, secrets=secrets, Headers=Headers, JSONResponse=JSONResponse)
exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


async def downstream(scope, receive, send):
    await JSONResponse({"reached_downstream": True})(scope, receive, send)


app = namespace["AuthenticationMiddleware"](downstream, ["audit-only-test-key"])


async def check(path, token=None):
    messages = []

    async def send(message):
        messages.append(message)

    async def receive():
        return {"type": "http.request", "body": b""}

    headers = [] if token is None else [(b"authorization", ("Bearer " + token).encode())]
    await app({"type": "http", "method": "GET", "path": path, "root_path": "", "headers": headers}, receive, send)
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


async def main():
    cases = []
    for path, token, expected in [
        ("/v1/models", None, 401),
        ("/v1/models", "audit-only-test-key", 200),
        ("/health", None, 200),
        ("/health", "wrong-key", 200),
    ]:
        status = await check(path, token)
        assert status == expected
        cases.append(
            dict(
                path=path,
                credentials="absent"
                if token is None
                else ("valid_test_key" if token == "audit-only-test-key" else "invalid_test_key"),
                status=status,
            )
        )
    print(
        json.dumps(
            dict(
                state="completed",
                vllm_version=importlib.metadata.version("vllm"),
                source_path=str(pathlib_path),
                source_sha256=hashlib.sha256(source).hexdigest(),
                scope="Unmodified AST-extracted vLLM authentication middleware with a synthetic downstream ASGI app; no engine, hardware or network listener",
                hardware_opened=False,
                cases=cases,
            )
        )
    )


pathlib_path = path
asyncio.run(main())
