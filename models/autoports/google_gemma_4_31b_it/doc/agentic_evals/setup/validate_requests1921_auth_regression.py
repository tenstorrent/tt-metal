"""Host-only exact-function regression; no HTTP, source changes, or original test edits."""
import ast
import collections
import collections.abc
import hashlib
import json
import pathlib

B = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals/anomalies/swe85-below-band")
C = B / "correct-actions/psf__requests-1921"
O = B / "correct-actions-semantic/psf__requests-1921-local-auth-attempt1"
O.mkdir(exist_ok=False)
O.chmod(0o700)
packet = B / "full-replies/psf__requests-1921.call34.packet.private.json"
reply = json.loads(packet.read_text())["reconstructed_full_reply"]
command = 'sed -i "45s/return session_setting/request_setting = dict_class()/" requests/sessions.py'
assert command in reply
pre = (C / "precall/requests/sessions.py").read_text()
lines = pre.splitlines(keepends=True)
assert "return session_setting" in lines[44]
lines[44] = lines[44].replace("return session_setting", "request_setting = dict_class()")
chosen = "".join(lines)


class DummyAuth:
    def __call__(self, request):
        request.headers["X-Dummy-Auth"] = "inherited"
        return request


class PreparedProbe:
    def __init__(self):
        self.headers = {}


rows = []
for state, source in [
    ("precall", pre),
    ("chosen", chosen),
    ("candidate", (C / "candidate/requests/sessions.py").read_text()),
]:
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "merge_setting")

    # Exact function AST. This auth path never reaches to_key_val_list; fail if it does.
    def unreachable(*a):
        raise AssertionError("Unexpected dictionary conversion on auth path")

    ns = {"OrderedDict": collections.OrderedDict, "Mapping": collections.abc.Mapping, "to_key_val_list": unreachable}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<exact merge_setting>", "exec"), ns)
    auth = DummyAuth()
    effective = ns["merge_setting"](None, auth)
    probe = PreparedProbe()
    if effective:
        effective(probe)
    passed = effective is auth and probe.headers.get("X-Dummy-Auth") == "inherited"
    rows.append(
        {
            "state": state,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "auth_identity_inherited": effective is auth,
            "effective_type": type(effective).__name__,
            "header_present": probe.headers.get("X-Dummy-Auth") == "inherited",
            "passed": passed,
        }
    )
    (O / (state + ".sessions.py")).write_text(source)
assert [r["passed"] for r in rows] == [True, False, True]
result = {
    "status": "narrow_auth_regression_discriminates_chosen_and_candidate",
    "rows": rows,
    "chosen_command": command,
    "packet_sha256": hashlib.sha256(packet.read_bytes()).hexdigest(),
    "scope": "Exact merge_setting function AST executed on host with standard-library bindings. DummyAuth/probe are diagnostic fixtures. Proves absent request auth inherits session callable before/candidate and is lost after exact chosen line45 command. Not full Session.prepare_request execution or full-suite pass; original external test failures remain unresolved.",
    "original_tests_modified": False,
    "network_used": False,
}
(O / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
for p in O.iterdir():
    p.chmod(0o600)
print(json.dumps(result, indent=2))
