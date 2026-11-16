import fcntl
import hashlib
import json
import pathlib
import subprocess
import time
import xml.etree.ElementTree as ET

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
P = D / "anomalies/swe85-below-band/correct-actions/psf__requests-5414"
O = P / "semantic-pure-urls"
O.mkdir(exist_ok=False)
O.chmod(0o700)
lock = (D / "setup/.semantic-verifier-worker.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
meta = json.loads((P / "manifest.json").read_text())
cfgpath = D / "setup/swe85-candidate/tasks85-offline-verifier/psf__requests-5414/tests/config.json"
cfg = json.loads(cfgpath.read_text())
patch = P / "reference.tests.private.patch"
assert patch.read_text() == cfg["test_patch"]
tests = ["tests/test_requests.py::TestRequests::test_invalid_url"] + [
    "tests/test_requests.py::TestPreparingURLs::" + s
    for s in [
        "test_preparing_url",
        "test_preparing_bad_url",
        "test_url_mutation",
        "test_parameters_for_nonstandard_schemes",
    ]
]
rows = []
for state, folder in [("before", "precall"), ("chosen30", "chosen30"), ("correct", "candidate")]:
    out = O / state
    out.mkdir()
    name = "v5-requests5414-pure-" + state + "-" + str(time.time_ns())
    source = P / folder / "requests/models.py"
    sourcehash = sha(source)
    calls = []
    started = False

    def run(argv, check=True):
        p = subprocess.run(argv, capture_output=True, text=True, timeout=240)
        idx = len(calls)
        (out / f"{idx:02d}.stdout.txt").write_text(p.stdout)
        (out / f"{idx:02d}.stderr.txt").write_text(p.stderr)
        calls.append({"argv": argv, "exit_code": p.returncode})
        if check and p.returncode:
            raise RuntimeError(p.stderr)
        return p

    try:
        run(
            [
                "docker",
                "create",
                "--name",
                name,
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "4g",
                "--memory-swap",
                "4g",
                "--pids-limit",
                "256",
                "-e",
                "PIP_NO_INDEX=1",
                "--entrypoint",
                "/bin/bash",
                meta["extraction"]["image_id"],
                "-lc",
                "sleep 1800",
            ]
        )
        started = True
        run(["docker", "start", name])
        head = run(["docker", "exec", name, "git", "-C", "/testbed", "rev-parse", "HEAD"]).stdout.strip()
        assert head == meta["image_head"]
        assert run(["docker", "exec", name, "sha256sum", "/testbed/requests/models.py"]).stdout.split()[0] == sha(
            P / "base/requests/models.py"
        )
        run(["docker", "cp", str(source.resolve()), name + ":/testbed/requests/models.py"])
        run(["docker", "cp", str(patch.resolve()), name + ":/tmp/original-test.patch"])
        assert run(["docker", "exec", name, "sha256sum", "/tmp/original-test.patch"]).stdout.split()[0] == sha(patch)
        run(
            [
                "docker",
                "exec",
                name,
                "git",
                "-C",
                "/testbed",
                "checkout",
                cfg["base_commit"],
                "--",
                "tests/test_requests.py",
            ]
        )
        run(["docker", "exec", name, "git", "-C", "/testbed", "apply", "--check", "/tmp/original-test.patch"])
        run(["docker", "exec", name, "git", "-C", "/testbed", "apply", "/tmp/original-test.patch"])
        hashbefore = run(
            ["docker", "exec", name, "sha256sum", "/testbed/tests/test_requests.py", "/testbed/tests/conftest.py"]
        ).stdout
        env = "cd /testbed && source /opt/miniconda3/bin/activate && conda activate testbed && "
        run(["docker", "exec", name, "/bin/bash", "-lc", env + "python --version && python -m pip freeze"])
        run(["docker", "exec", name, "/bin/bash", "-lc", env + "python -m pip install ."])
        r = run(
            [
                "docker",
                "exec",
                name,
                "/bin/bash",
                "-lc",
                env + "pytest -rA --junitxml=/tmp/pure-urls.xml " + " ".join(tests),
            ],
            check=False,
        )
        run(["docker", "cp", name + ":/tmp/pure-urls.xml", str((out / "junit.xml").resolve())])
        hashafter = run(
            ["docker", "exec", name, "sha256sum", "/testbed/tests/test_requests.py", "/testbed/tests/conftest.py"]
        ).stdout
        assert hashbefore == hashafter
        assert run(["docker", "exec", name, "sha256sum", "/testbed/requests/models.py"]).stdout.split()[0] == sourcehash
        assert sha(source) == sourcehash
        xml = ET.parse(out / "junit.xml")
        cases = [
            {
                "name": x.attrib["name"],
                "classname": x.attrib.get("classname"),
                "status": "fail"
                if x.find("failure") is not None
                else "error"
                if x.find("error") is not None
                else "skip"
                if x.find("skipped") is not None
                else "pass",
            }
            for x in xml.iter("testcase")
        ]
        row = {
            "state": state,
            "source_sha256": sourcehash,
            "source": str(source),
            "tests": tests,
            "test_hashes_before_after": hashbefore,
            "tests_unchanged": True,
            "pytest_exit_code": r.returncode,
            "test_cases": cases,
            "counts": {k: sum(c["status"] == k for c in cases) for k in ["pass", "fail", "error", "skip"]},
        }
        rows.append(row)
    finally:
        if started:
            run(["docker", "rm", "-f", name], check=False)
        (out / "commands.json").write_text(json.dumps(calls, indent=2) + "\n")
result = {
    "status": "narrow_original_pure_URL_tests_only",
    "task": "psf__requests-5414",
    "image_id": meta["extraction"]["image_id"],
    "image_head": meta["image_head"],
    "config_sha256": sha(cfgpath),
    "test_patch_sha256": sha(patch),
    "states": rows,
    "limits": {"cpus": 1, "memory_gib": 4, "network": "none"},
    "scope": "Original TestPreparingURLs pure URL parameterizations and original invalid_url tests with official test patch. No fixture/test/source rewrites. Same pinned image, environment and test selection for all3 source states. Full original test runs separately retained with httpbin fixture and endpoint failures; not full suite pass or scored rerun.",
    "chosen30_materialization": "Exact old_block/new_block literals from native57 script; original source restored at native55, no intervening tracked source edits.",
    "source_and_test_hashes_preserved": True,
}
(O / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
for p in O.rglob("*"):
    if p.is_file():
        p.chmod(0o600)
print(
    json.dumps({"summary": str(O / "summary.json"), "states": [{k: r[k] for k in ["state", "counts"]} for r in rows]})
)
