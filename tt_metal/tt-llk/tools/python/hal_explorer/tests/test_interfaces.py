# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The command line, the HTTP server and the compile flags shared with lit."""

import ast
import http.client
import json
import os
import re
import socket
import subprocess
import sys
import time

import pytest
from conftest import TOOL, needs_toolchain
from hal_explorer.toolchain import build

CLI = [sys.executable, os.path.join(TOOL, "cli.py")]
LIT_CONFIG = (
    build.LLK_ROOT
    / "tools"
    / "tests"
    / "unit"
    / "lit"
    / "hal"
    / "blackhole"
    / "lit.local.cfg"
)


def cli(*args, stdin=""):
    return subprocess.run(
        CLI + list(args), input=stdin, capture_output=True, text=True, timeout=600
    )


def test_flags_match_the_lit_configuration():
    tree = ast.parse(LIT_CONFIG.read_text(encoding="utf-8"))
    (command,) = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", "") == "compile_command"
    ]
    lit_flags = [e.value for e in command.elts if isinstance(e, ast.Constant)]
    ours = [f for f in build.compile_flags()[1:] if not f.startswith("-I")]
    thread = ["-DCOMPILE_FOR_TRISC=1", "-DLLK_TRISC_MATH"]
    assert [f for f in ours if f not in thread] == lit_flags


@needs_toolchain
class TestCommandLine:
    def test_counts_one_row_per_path(self):
        run = cli(
            "-", "--counts", stdin="void f(bool a) { if (a) { TTI_NOP; } TTI_NOP; }"
        )
        assert run.returncode == 0, run.stderr
        assert sorted(run.stdout.splitlines()) == ["f(bool)\t1\t!a", "f(bool)\t2\ta"]

    def test_json_follows_the_documented_shape(self):
        run = cli("-", "--json", stdin="void f() { TTI_NOP; }")
        data = json.loads(run.stdout)
        assert set(data) == {
            "ok",
            "mode",
            "diagnostics",
            "errors",
            "functions",
            "comparisons",
        }
        (function,) = data["functions"]
        (group,) = function["groups"]
        assert group["count"] == 1 and group["sequence"][0]["word_text"] == "0x02000000"

    def test_body_mode_variables(self):
        run = cli(
            "-",
            "--mode",
            "body",
            "--var",
            "const bool a",
            "--counts",
            stdin="if (a) { TTI_NOP; }",
        )
        assert run.returncode == 0, run.stderr
        assert sorted(run.stdout.splitlines()) == [
            "replay body\t0\t!a",
            "replay body\t1\ta",
        ]

    @pytest.mark.parametrize(
        "args, stdin, status",
        [
            (["-"], "void f() { nope(); }", 1),
            (["-"], "void reference_f() { TTI_NOP; }\nvoid f() { }", 2),
            (["-"], "void reference_f() { TTI_NOP; }", 2),
            (["-", "--var", "bool a"], "void f() { }", 3),
            (["/nonexistent.cpp"], "", 3),
            (["-", "--function", "missing"], "void f() { }", 3),
            (["-"], "void helper(); void f() { helper(); }", 4),
            (["-", "--json"], "void reference_f() { TTI_NOP; }\nvoid f() { }", 2),
        ],
    )
    def test_exit_status(self, args, stdin, status):
        assert cli(*args, stdin=stdin).returncode == status

    def test_function_filter_keeps_its_comparison(self):
        code = "void reference_f() { TTI_NOP; }\nvoid f() { }\nvoid reference_g() { }\nvoid g() { }"
        run = cli("-", "--function", "g", stdin=code)
        assert (
            run.returncode == 0
            and "IDENTICAL" in run.stdout
            and "== compare f()" not in run.stdout
        )


@needs_toolchain
class TestServer:
    @pytest.fixture(scope="class")
    def server(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        process = subprocess.Popen(
            [
                sys.executable,
                os.path.join(TOOL, "server.py"),
                "--port",
                str(port),
                "--jobs",
                "2",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.2)
        yield port
        process.terminate()
        process.wait(timeout=10)

    def request(self, port, method="POST", path="/compile", body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        payload = (
            body if isinstance(body, (bytes, type(None))) else json.dumps(body).encode()
        )
        sent = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, payload, sent)
        response = connection.getresponse()
        return response.status, response.read().decode()

    def test_page_and_compile(self, server):
        status, page = self.request(server, "GET", "/")
        assert status == 200 and "HAL" in page and "__EXAMPLES__" not in page
        status, text = self.request(
            server,
            body={
                "mode": "body",
                "body": "if (a) { TTI_NOP; }",
                "variables": [{"type": "bool", "name": "a"}],
            },
        )
        assert status == 200 and json.loads(text)["ok"]

    @pytest.mark.parametrize(
        "body, headers, status",
        [
            (b"{not json", None, 400),
            ({"mode": "nope", "body": ""}, None, 400),
            ({"body": 5}, None, 400),
            ({"body": "", "variables": "x"}, None, 400),
            ({"body": "void f() {}"}, {"Content-Type": "text/plain"}, 415),
            ({"body": "void f() {}"}, {"Host": "evil.example"}, 403),
        ],
    )
    def test_rejected_requests(self, server, body, headers, status):
        assert self.request(server, body=body, headers=headers)[0] == status

    def test_oversized_request(self, server):
        assert self.request(server, body={"body": "x" * 400_000})[0] == 413

    def test_unknown_path(self, server):
        assert self.request(server, "GET", "/../../etc/passwd")[0] == 404


def test_readme_documents_every_cli_option():
    readme = (build.LLK_ROOT / "tools/python/hal_explorer/README.md").read_text(
        encoding="utf-8"
    )
    options = set(
        re.findall(
            r"add_argument\(\s*\"(--[\w-]+)",
            (build.LLK_ROOT / "tools/python/hal_explorer/cli.py").read_text(
                encoding="utf-8"
            ),
        )
    )
    assert options and all(option in readme for option in options)
