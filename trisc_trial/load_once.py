"""Load TestConfig._compile_kernel_part_once from a test_config.py without importing the module (no tt deps)."""

import ast, hashlib, json, os, re, shutil, subprocess
from pathlib import Path

from filelock import FileLock


class FakeTestConfig:
    ARTEFACTS_DIR = None
    TESTS_WORKING_DIR = None


def run_shell_command(command, cwd, stdin=None):
    r = subprocess.run(command, cwd=cwd, input=stdin, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"compile failed: {r.stderr[-2000:]}")
    return r


def load(test_config_py):
    tree = ast.parse(Path(test_config_py).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TestConfig")
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_compile_kernel_part_once", "_declared_name")]
    for fn in fns:
        fn.decorator_list = []
    mod = ast.Module(body=fns, type_ignores=[])
    ns = dict(
        TestConfig=FakeTestConfig, Path=Path, run_shell_command=run_shell_command, re=re, json=json,
        sha256=hashlib.sha256, FileLock=FileLock, subprocess=subprocess, shutil=shutil, os=os,
    )
    exec(compile(mod, test_config_py, "exec"), ns)
    once, declared = ns["_compile_kernel_part_once"], ns.get("_declared_name")

    def call(self, command, source):
        if declared:
            self._declared_name = declared
        return once(self, command, source)

    return call
