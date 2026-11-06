"""Execute the actual nested request function with a mocked HTTP transport."""

import ast
import hashlib
import json
import threading
import urllib.request
from pathlib import Path
from types import SimpleNamespace


def check(source, timeout, over_deadline=False):
    tree = ast.parse(Path(source).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    request = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "request")
    seen = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def __iter__(self):
            return iter([b'data: {"choices":[{"token_ids":[1]}]}\n', b"data: [DONE]\n"])

    def urlopen(req, timeout):
        seen.append(timeout)
        return Response()

    clock = iter([0.0, timeout + 1 if over_deadline else 1.0, 2.0, 3.0, 4.0, 5.0])
    env = dict(
        args=SimpleNamespace(url="http://unused", model="fixture", request_timeout=timeout),
        prompt=lambda length, label: [1] * length,
        time=SimpleNamespace(monotonic=lambda: next(clock)),
        hashlib=hashlib,
        started=0.0,
        records=[],
        lock=threading.Lock(),
        json=json,
        vocabulary_size=2,
        urllib=SimpleNamespace(request=SimpleNamespace(Request=urllib.request.Request, urlopen=urlopen)),
        save=lambda: None,
    )
    exec(compile(ast.Module(body=[request], type_ignores=[]), source, "exec"), env)
    row = env["request"]("fixture", 1, 1)
    assert seen == [timeout], seen
    assert row["status"] == ("failed" if over_deadline else "pass"), row
    if over_deadline:
        assert "total deadline exceeded" in row["error"]


if __name__ == "__main__":
    import sys

    for limit in (30.0, 600.0):
        check(sys.argv[1], limit)
    check(sys.argv[1], 600.0, True)
    print("3 timeout regression checks passed")
