import datetime
import hashlib
import json
import pathlib

from transformers import AutoTokenizer

D = pathlib.Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
L = D / "current/swe-extended85-head-bfp8/live-tasks"
O = D / "anomalies/swe85-below-band/full-replies-update4"
O.mkdir(parents=True, exist_ok=True)
rev = "842da3794eaa0b77d5f08bae87a17459d91ff475"
snap = pathlib.Path("/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots") / rev
tok = AutoTokenizer.from_pretrained(str(snap), revision=rev, local_files_only=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(p, j, private=True):
    p.write_text(json.dumps(j, indent=2) + "\n")
    p.chmod(0o600 if private else 0o644)


def norm(messages):
    messages = json.loads(json.dumps(messages))
    for m in messages:
        for tc in m.get("tool_calls") or []:
            a = tc["function"].get("arguments")
            if isinstance(a, str):
                tc["function"]["arguments"] = json.loads(a) if a else {}
    return messages


# Canonical template macros serialize reasoning/tool payload; preserve original content separately before tools,
# as generated parser order is thought -> visible assistant content -> tool call.
from jinja2 import Environment

source = (snap / "chat_template.jinja").read_text()
macro = source[source.index("{%- macro format_argument") : source.index("{%- macro strip_thinking")]
fmt = Environment().from_string(macro).module.format_argument


def reply(m, stop):
    text = ""
    reason = m.get("reasoning") or m.get("reasoning_content")
    if reason:
        text += "<|channel>thought\n" + reason + "<channel|>"
    text += m.get("content") or ""
    for tc in m.get("tool_calls") or []:
        f = tc["function"]
        a = f.get("arguments")
        a = json.loads(a) if isinstance(a, str) and a else (a or {})
        if not isinstance(a, dict):
            raise ValueError("nonmapping tool args")
        text += (
            "<|tool_call>call:"
            + f["name"]
            + "{"
            + ",".join(k + ":" + fmt(v, False) for k, v in sorted(a.items(), key=lambda x: x[0].lower()))
            + "}<tool_call|>"
        )
    ids = tok.encode(text, add_special_tokens=False)
    if isinstance(stop, int):
        ids.append(stop)
    else:
        raise ValueError("No native integer stop ID")
    return text, ids


special = {
    "sphinx-doc__sphinx-8638": [3],
    "astropy__astropy-7336": [18],
    "pydata__xarray-4687": [39],
    "pylint-dev__pylint-4970": [51, 55],
    "pylint-dev__pylint-6386": [33],
    "pytest-dev__pytest-6197": [35],
    "pylint-dev__pylint-8898": [44],
    "pytest-dev__pytest-5840": [15, 16],
    "matplotlib__matplotlib-20488": [29, 88, 91],
    "pydata__xarray-3095": [53],
    "pytest-dev__pytest-10356": [14],
    "pydata__xarray-7233": [1, 4],
}
selected = []
excluded = []
gaps = []
reviewed = list(L.glob("sphinx-doc__sphinx-8638-first/coordinator.json"))
for cp in sorted(reviewed):
    c = json.loads(cp.read_text())
    task = c["task"]
    if c.get("counted_reward") != 0:
        if c.get("cause") == "infrastructure":
            excluded.append(task)
        continue
    tp = cp.parent / "digest" / f"{task}.triage.json"
    t = json.loads(tp.read_text()) if tp.exists() else {}
    ns = special.get(task, [c.get("first_bad_call", t.get("first_bad_call"))])
    dg = json.loads((cp.parent / "digest" / f"{task}.json").read_text())
    col = json.loads((cp.parent / "collected.json").read_text())["tasks"][0]
    traj = pathlib.Path(col["source_result"]).parent / "agent/mini-swe-agent.trajectory.json"
    native = json.loads(traj.read_text())
    proof = json.loads((cp.parent / "selection-proof.json").read_text())
    for n in ns:
        dr = next((d for d in dg["calls"] if d["n"] == n), None)
        if dr is None:
            gaps.append({"task": task, "reason": "Unresolved call index", "candidate": n})
            continue
        selected.append(
            dict(
                task=task,
                call=n,
                response_id=dr["id"],
                coordinator=str(cp),
                coordinator_sha256=sha(cp.read_bytes()),
                triage=str(tp),
                cause=c.get("cause"),
                native_trajectory=str(traj),
                native_message_indices=[i for i, m in enumerate(native["messages"]) if dr["id"] in json.dumps(m)],
                request_source=proof["source_log"],
                response_source=str(cp.parent / "responses.private.log"),
                native_prompt_tokens=dr["prompt_tokens"],
                native_completion_tokens=dr["tokens"],
                action_excerpt=dr.get("action"),
                anomaly=bool(c.get("anomaly")),
            )
        )
requests = {}
responses = {}
for src in {r["request_source"] for r in selected}:
    wanted = {r["response_id"] for r in selected if r["request_source"] == src}
    with open(src) as f:
        for ln, line in enumerate(f, 1):
            if "eval_call=" not in line:
                continue
            j = json.loads(line.split("eval_call=", 1)[1])
            rid = j.get("id")
            if rid in wanted:
                requests[rid] = (j["request"], ln)
for src in {r["response_source"] for r in selected}:
    for ln, line in enumerate(open(src), 1):
        if "response_body=" not in line:
            continue
        s = line.split("response_body=", 1)[1].strip()
        s = s[1:] if s.startswith("{{") else s
        try:
            j, _ = json.JSONDecoder().raw_decode(s)
        except ValueError:
            continue
        responses[j["id"]] = (j, ln)
jobs = []
manifest = []
for r in selected:
    rid = r["response_id"]
    name = f"{r['task']}.call{r['call']}"
    try:
        raw, line = requests[rid]
        q = json.loads(raw)
        b, rline = responses[rid]
        m = b["choices"][0]["message"]
        settings = q.get("chat_template_kwargs") or {}
        rendered = tok.apply_chat_template(
            norm(q["messages"]), tools=q.get("tools"), tokenize=False, add_generation_prompt=True, **settings
        )
        prompt = tok.encode(rendered, add_special_tokens=False)
        text, chosen = reply(m, b["choices"][0]["stop_reason"])
        packet = O / (name + ".packet.private.json")
        case = O / (name + ".case.private.json")
        write(
            packet,
            {
                "raw_request_body": raw,
                "native_response": b,
                "rendered_prompt": rendered,
                "reconstructed_full_reply": text,
                "native_stop_id": b["choices"][0]["stop_reason"],
            },
        )
        meta = {
            **r,
            "request_line": line,
            "response_line": rline,
            "request_sha256": sha(raw.encode()),
            "native_response_sha256": sha(json.dumps(b, sort_keys=True).encode()),
            "hf_revision": rev,
            "template_sha256": sha(source.encode()),
            "reconstructed_prompt_tokens": len(prompt),
            "reconstructed_completion_tokens": len(chosen),
            "prompt_token_count_gap": len(prompt) - r["native_prompt_tokens"],
            "completion_token_count_gap": len(chosen) - r["native_completion_tokens"],
            "native_prompt_ids_available": b.get("prompt_token_ids") is not None,
            "native_output_ids_available": b["choices"][0].get("token_ids") is not None,
            "correct_branch": "pending validated pre-call state; not constructed",
            "scope": "Full original reply reconstruction, no truncation. Agreement only, not completed precision attempt.",
            "reconstruction": "Exact request body with historical JSON tool arguments deserialized for pinned template. Reply reconstructed as canonical thought header, complete saved reasoning, channel boundary, complete visible content, canonical tool-call macros from pinned template, native stop ID. No native token IDs: generated whitespace/order/protocol spelling cannot be recovered uniquely; counts and hashes are evidence, not identity proof. Canonicalization of malformed tool names is not proof of exact malformed native bytes.",
            "sampling": {k: q.get(k) for k in ["temperature", "top_p", "top_k"]},
            "native_seed": q.get("seed"),
            "executed": False,
        }
        write(case, {"prompt_token_ids": prompt, "continuations": {"chosen": chosen}, "metadata": meta})
        jobs.append({"name": name, "case": str(case), "branches": ["chosen"]})
        manifest.append(
            {
                **meta,
                "private_case": str(case),
                "private_case_sha256": sha(case.read_bytes()),
                "private_packet": str(packet),
                "private_packet_sha256": sha(packet.read_bytes()),
            }
        )
    except Exception as e:
        gaps.append({"task": r["task"], "call": r["call"], "response_id": rid, "error": str(e)})
write(O / "cases.json", jobs, False)
write(O / "validate.build_kwargs.json", {"batch_size": 1}, False)
result = {
    "status": "prepared_not_executed_correct_branches_pending",
    "snapshot_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "reviewed_coordinators": len(reviewed),
    "failed_tasks": len(set(r["task"] for r in selected)),
    "prepared_cases": len(jobs),
    "excluded_unresolved_infrastructure": excluded,
    "gaps": gaps,
    "cases": manifest,
    "mapping_qualifications": {
        "pylint-dev__pylint-4970": "Both call51 initial attempt and call55 actual insertion prepared; root identifies55 as actual edit. Do not label51 executed insertion.",
        "pytest-dev__pytest-5840": "Call15 writes replacement script,16 executes it; both saved for decision/execution comparison.",
        "matplotlib__matplotlib-20488": "Call29 earliest destructive rewrite identified by triage;88/91 separate anomalous repetitions. Root must ratify29 semantic first-bad status.",
        "pydata__xarray-3095": "Recovered malformed53 native ID omitted; triage maps to message106 FormatError. First semantic wrong-edit call still unresolved.",
        "pytest-dev__pytest-10356": "Recovered malformed14; semantic wrong-action identification requires separate adjudication.",
        "pydata__xarray-7233": "Call1 protocol anomaly,4 fetched-release scoring violation; both preserved.",
    },
}
write(O / "case_manifest.json", result, False)
print(json.dumps({k: result[k] for k in ["reviewed_coordinators", "failed_tasks", "prepared_cases", "gaps"]}))
