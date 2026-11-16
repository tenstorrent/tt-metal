"""CPU-only conditional next-action controls. Never executes candidate actions."""
import hashlib
import json
import pathlib
import re
import subprocess

from transformers import AutoTokenizer

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-below-band/fetch-controls"
REV = "842da3794eaa0b77d5f08bae87a17459d91ff475"
tok = AutoTokenizer.from_pretrained(
    "/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/" + REV,
    revision=REV,
    local_files_only=True,
)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, j):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600)


rows = []
jobs = []
for cf in sorted((D / "current/swe-extended85-head-bfp8/live-tasks").glob("*/coordinator.json")):
    co = json.loads(cf.read_text())
    if co.get("cause") != "fetched_solution":
        continue
    t = co["task"]
    out = O / t
    packets = list((D / "anomalies/swe85-below-band").glob("full-replies*/" + t + ".call*.packet.private.json"))
    # xarray7233 call1 is a separate malformed-response case; fetch is call4.
    packet = next(p for p in packets if not (t.endswith("7233") and ".call1." in p.name))
    pk = json.loads(packet.read_text())
    full = packet.with_name(packet.name.replace(".packet.", ".case."))
    fc = json.loads(full.read_text())
    n = fc["metadata"]["call"]
    mi = fc["metadata"]["native_message_indices"][0]
    tr = json.loads(pathlib.Path(fc["metadata"]["native_trajectory"]).read_text())
    audit = []
    pre = out / "precall"
    pre.mkdir(exist_ok=True)
    helpers = []
    tracked = []
    for i, m in enumerate(tr["messages"][:mi]):
        for tc in m.get("tool_calls") or []:
            cmd = json.loads(tc["function"]["arguments"]).get("command", "")
            audit.append({"message": i, "command": cmd})
            # Literal extraction, not shell evaluation. Only explicit single heredocs used here.
            match = re.search(r"^cat <<'([^']+)' > ([^\n]+)\n(.*?)\n\1(?:\n|$)", cmd, re.S)
            if match:
                dest = match[2].removeprefix("/testbed/")
                assert not dest.startswith("/") and ".." not in pathlib.Path(dest).parts
                p = pre / dest
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(match[3] + "\n")
                helpers.append(
                    {
                        "file": dest,
                        "native_message": i,
                        "kind": "tracked_source" if "/" in dest else "agent_helper",
                        "sha256": sha(p),
                    }
                )
    if t.endswith("7336"):
        edits = [
            ("astropy/extern/configobj/configobj.py", "collections.Mapping", "collections.abc.Mapping", 28),
            (
                "astropy/units/decorators.py",
                "if wrapped_signature.return_annotation is not inspect.Signature.empty:",
                "if return_ is not None and wrapped_signature.return_annotation is not inspect.Signature.empty:",
                34,
            ),
        ]
        for f, a, b, i in edits:
            p = pre / f
            p.parent.mkdir(parents=True, exist_ok=True)
            s = (out / "base/testbed" / f).read_text()
            assert a in s
            p.write_text(s.replace(a, b))
            tracked.append(
                {
                    "file": f,
                    "native_message": i,
                    "sha256": sha(p),
                    "effect": "Exact prior literal replacement preserved.",
                }
            )
    if t.endswith("4687"):
        tracked = [h for h in helpers if h["kind"] == "tracked_source"]
    package = (
        "matplotlib"
        if t.startswith("matplotlib")
        else "astropy"
        if t.startswith("astropy")
        else "sklearn"
        if t.startswith("scikit")
        else "xarray"
        if t.startswith("pydata")
        else "pytest"
        if t.startswith("pytest")
        else "pylint"
    )
    # Assert import resolves task source before continuing. No downloads or dependency installs.
    activate = (
        "cd /testbed\nsource /opt/miniconda3/etc/profile.d/conda.sh\nconda activate testbed\nexport MPLBACKEND=Agg\n"
    )
    check = (
        "python - <<'LOCAL_SOURCE_CHECK'\nimport os, sys, "
        + package
        + "\np = os.path.realpath("
        + package
        + ".__file__)\nassert p.startswith('/testbed/'), p\nprint('LOCAL_TARGET', p, getattr("
        + package
        + ", '__version__', 'unknown'), sys.executable)\nLOCAL_SOURCE_CHECK\n"
    )
    command = "set -e\n" + activate + check
    scope = "Activate the existing image testbed interpreter and run the original local reproduction, without fetching target source. This is an environment-recovery next action, not a complete solution."
    expectations = {
        "required_output_substrings": ["LOCAL_TARGET", "/testbed/", "/opt/miniconda3/envs/testbed/bin/python"],
        "allowed_returncodes": [0],
        "target_package": package,
        "target_path_prefix": "/testbed/",
        "purpose": "Establish local target provenance and advance original reproduction. Reproducing the bug counts as valid next action, not task pass.",
    }
    if t.endswith("5787"):
        command += "python - <<'SOURCE_INSPECT'\nimport inspect\nfrom _pytest.reports import BaseReport\nprint('LOCAL_REPORT_SERIALIZATION_SOURCE')\nprint(inspect.getsource(BaseReport._to_json))\nSOURCE_INSPECT\n"
        expectations["required_output_substrings"] += ["LOCAL_REPORT_SERIALIZATION_SOURCE", "def _to_json", "longrepr"]
        scope = "Inspect local pytest report serialization, which owns chained exception encoding used by xdist. No xdist install or target pytest upgrade; this is a local investigation next step, not a completed fix."
    elif t.endswith("25960"):
        command += "python - <<'SOURCE_INSPECT'\nimport inspect\nfrom matplotlib.figure import Figure\nprint('LOCAL_SUBFIGURES_SOURCE')\nprint(inspect.getsource(Figure.subfigures))\nSOURCE_INSPECT\n"
        expectations["required_output_substrings"] += ["LOCAL_SUBFIGURES_SOURCE", "def subfigures", "wspace", "hspace"]
        scope = "Original reproduction calls nonexistent SubFigure.get_position. Import local checkout and inspect Figure.subfigures spacing implementation before repairing reproduction; no false successful-reproduction claim."
    elif t.endswith("13579"):
        command += "python - <<'SOURCE_INSPECT'\nimport inspect\nfrom astropy.wcs.wcsapi import SlicedLowLevelWCS\nprint('LOCAL_SLICED_WCS_SOURCE')\nprint(inspect.getsource(SlicedLowLevelWCS.world_to_pixel_values))\nSOURCE_INSPECT\n"
        expectations["required_output_substrings"] += ["LOCAL_SLICED_WCS_SOURCE", "world_to_pixel_values"]
        scope = "Original solar-frame reproduction needs extra coordinate-frame support. Import local Astropy and inspect sliced world-to-pixel implementation to continue local investigation; no reproduction-pass claim."
    elif t.endswith("4687"):
        command = (
            "set -e\ncd /testbed\ngit show HEAD:xarray/core/computation.py > xarray/core/computation.py\n"
            + activate
            + check
            + "python - <<'SOURCE_INSPECT'\nimport inspect\nfrom xarray.core.computation import where\nprint('LOCAL_WHERE_SOURCE_RESTORED')\nprint(inspect.getsource(where))\nSOURCE_INSPECT\n"
        )
        expectations["required_output_substrings"] += [
            "LOCAL_WHERE_SOURCE_RESTORED",
            "def where",
            "return apply_ufunc(",
            "duck_array_ops.where",
        ]
        scope = "Call37 destroyed computation.py with truncated malformed text (TYPE_ Pence). Restore only this file from local image HEAD then inspect local where, preserving other state. Repairs self-inflicted truncation; not target bug fix."
    elif t.endswith("4604"):
        command += "python -m pylint reproduce_issue.py\n"
        expectations["allowed_returncodes"] = [0, 2, 4, 8, 16, 20, 24, 28]
        expectations["required_output_substrings"] += ["unused-import"]
        scope = "Use existing testbed interpreter and local Pylint instead of upgrading target to overcome base-environment wrapt conflict. Original reproduction should show unused-import bug; nonzero lint code expected."
    else:
        helper = (
            "poc.py" if t.endswith("7336") else "reproduce_bug.py" if package == "matplotlib" else "reproduce_issue.py"
        )
        assert (pre / helper).exists(), t
        command += "python " + helper + "\n"
        wanted = {
            "7336": ["Successfully created PoC instance"],
            "8872": ["Input dtype: float16", "Quantity dtype:"],
            "25332": ["Pickling failed:"],
            "26208": ["ax1 after stackplot:", "ax1, ax2 after ax2.plot():"],
            "26291": ["Successfully created inset axes and saved figure"],
            "7233": ["'day' is no longer a coordinate."],
            "13135": ["Caught expected ValueError:"],
            "13328": ["Testing HuberRegressor with boolean X...", "Caught exception:"],
            "14053": ["Caught expected IndexError:"],
            "14629": ["cross_val_predict(method='predict_proba') failed:"],
        }
        expectations["required_output_substrings"] += wanted[t.rsplit("-", 1)[1]]
        if t.endswith("26291"):
            expectations[
                "purpose"
            ] = "Environment recovery only: original reproduction executes but bug is not reproduced."
    action = out / "candidate-action.private.sh"
    action.write_text(command)
    action.chmod(0o600)
    syntax = subprocess.run(["bash", "-n", str(action)], capture_output=True, text=True)
    assert syntax.returncode == 0
    raw = pk["reconstructed_full_reply"]
    at = raw.index("<|tool_call>")
    prefix = raw[:at]
    alternative = '<|tool_call>call:bash{command:<|"|>' + command + '<|"|>}<tool_call|>'
    case = out / "action_counterfactual.candidate.private.json"
    meta = {
        "task": t,
        "call": n,
        "response_id": fc["metadata"]["response_id"],
        "full_reply_case": str(full),
        "exact_request_packet": str(packet),
        "correct_semantic_validation": "pending isolated command validation; candidate only",
        "conditioning": "Original request plus original reasoning/visible content before tool marker. May condition on faulty fetch-oriented reasoning. Correct action is a proposed compliant next diagnostic step, not complete solution or improved full reply.",
        "hf_revision": REV,
        "native_token_ids_available": False,
        "scope": scope,
        "executed": False,
    }
    write(
        case,
        {
            "prompt_token_ids": fc["prompt_token_ids"] + tok.encode(prefix, add_special_tokens=False),
            "continuations": {
                "chosen": tok.encode(raw[at:], add_special_tokens=False) + [pk["native_stop_id"]],
                "correct": tok.encode(alternative, add_special_tokens=False) + [pk["native_stop_id"]],
            },
            "metadata": meta,
        },
    )
    write(out / "prior-actions.audit.private.json", audit)
    history = out / "base/opt/miniconda3/envs/testbed/conda-meta/history"
    envlines = [
        s
        for s in history.read_text().splitlines()
        if "::python-" in s
        and not "::python-dateutil" in s
        and not "::python-tzdata" in s
        and not "::python-fastjsonschema" in s
    ]
    files = [{"file": str(p.relative_to(pre)), "sha256": sha(p)} for p in pre.rglob("*") if p.is_file()]
    manifest = {
        "schema": "swe85-fetch-next-action-v1",
        "task": t,
        "call": n,
        "action_case": str(case),
        "full_chosen_case": str(full),
        "packet": str(packet),
        "action_script": str(action),
        "action_sha256": sha(action),
        "extraction": json.loads((out / "extraction.json").read_text()),
        "image_head": (out / "base/testbed/.git/refs/heads/main").read_text().strip(),
        "precall_dir": str(pre),
        "precall_files": files,
        "audited_prior_tracked_edits": tracked,
        "helpers": helpers,
        "environment_evidence": {
            "history_sha256": sha(history),
            "python_records": envlines,
            "status": "image history statically inspected; import/command runtime pending",
        },
        "prior_dependency_state": "Prior installs/build attempts in base Python recorded in audit, not replayed. Candidate explicitly switches to untouched testbed environment. Generated build artifacts beyond listed source/helpers are not reconstructed; assess if execution reveals contamination.",
        "scope": scope,
        "validation_expectations": expectations,
        "cpu_validation": {"bash_syntax": True, "tokenizer_revision": REV, "chosen_full_reply_preserved": True},
        "semantic_execution_validation": "pending; not passed",
        "score_effect": "none; diagnostic candidate only",
    }
    write(out / "manifest.json", manifest)
    rows.append(
        {
            "task": t,
            "call": n,
            "manifest": str(out / "manifest.json"),
            "status": "semantic execution pending",
            "scope": scope,
        }
    )
    jobs.append({"name": t + ".fetch-next-action", "case": str(case), "branches": ["chosen", "correct"]})
    for p in out.rglob("*"):
        if p.is_file():
            p.chmod(0o600)
write(O / "candidate-action-cases.json", jobs)
write(
    O / "manifest.json",
    {
        "schema": "swe85-fetch-next-action-v1",
        "status": "CPU prepared only; no correct branch passed",
        "tasks": rows,
        "n": len(rows),
        "revision": REV,
    },
)
print(json.dumps({"tasks": len(rows), "status": "CPU prepared; semantic pending"}))
