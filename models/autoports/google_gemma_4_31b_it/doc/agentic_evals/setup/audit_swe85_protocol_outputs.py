"""CPU-only stock-parser audit of completed protocol-onset files; no scoring edits."""
import argparse
import datetime
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"ttnn", "tt_lib"}:
            raise RuntimeError("Forbidden device import: " + fullname)


sys.meta_path.insert(0, Guard())
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.gemma4 import Gemma4Parser

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", type=Path, default=D / "setup/swe85-admitted-controls85/combined-cases.json")
p.add_argument("--runs", type=Path, default=D / "setup/swe85-pilot-split85/runs/rest")
p.add_argument("--output", type=Path, default=D / "setup/remainder-execution85/protocol-parser-audit.json")
a = p.parse_args()


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rev = "842da3794eaa0b77d5f08bae87a17459d91ff475"
snap = Path("/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots") / rev
tok = AutoTokenizer.from_pretrained(str(snap), revision=rev, local_files_only=True)
legacy = {
    "anomaly.xarray7233-call1.canonicalprefix": ("pydata__xarray-7233", 1),
    "anomaly.pytest10356-call14.canonicalprefix": ("pytest-dev__pytest-10356", 14),
    "anomaly.xarray3095-call53.canonicalprefix": ("pydata__xarray-3095", 53),
}
rows = []
pending = []
selected = []
errors = []
for job in read(a.manifest):
    if not job["name"].startswith("anomaly."):
        continue
    c = read(job["case"])
    m = c["metadata"]
    motif = m.get("motif", "")
    isprotocol = job["name"] in legacy or motif.startswith(
        ("call:bash", "GetAll-bash", "bash\n", "bash,command:", "this.thought")
    )
    if not isprotocol:
        continue
    selected.append(job["name"])
    try:
        if job["name"] in legacy:
            task, n = legacy[job["name"]]
            packet = D / f"anomalies/swe85-below-band/full-replies/{task}.call{n}.packet.private.json"
            request_count = m["reconstructed_request_prompt_tokens"]
        else:
            packet = Path(m["raw_packet"])
            request_count = len(read(m["source_full_reply_case"])["prompt_token_ids"])
        pk = read(packet)
        q = json.loads(pk["raw_request_body"])
        request = ChatCompletionRequest(**q)
        prefixids = c["prompt_token_ids"][request_count:]
        prefix = tok.decode(prefixids, skip_special_tokens=False)

        def parse(text, ids):
            parser = Gemma4Parser(tok, tools=request.tools, chat_template_kwargs=q.get("chat_template_kwargs", {}))
            reason, content, tools = parser.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)
            calls = []
            for t in tools or []:
                try:
                    args = json.loads(t.arguments)
                    valid = isinstance(args, dict) and isinstance(args.get("command"), str)
                    json_ok = True
                except (ValueError, TypeError):
                    args = {}
                    valid = False
                    json_ok = False
                calls.append(
                    {
                        "name": t.name,
                        "expected_bash": t.name == "bash",
                        "valid_json": json_ok,
                        "valid_command_schema": valid,
                        "argument_keys": list(args) if isinstance(args, dict) else None,
                    }
                )
            return {
                "structured_tool_count": len(calls),
                "missing_tool": not calls,
                "malformed_name": any(not x["expected_bash"] for x in calls),
                "invalid_arguments": any(not x["valid_command_schema"] for x in calls),
                "tools": calls,
                "visible_content_chars": len(content or ""),
                "visible_content_head": (content or "")[:120],
                "reasoning_chars": len(reason or ""),
                "visible_GetAll_bash": "GetAll-bash" in (content or ""),
                "visible_pseudo_call": "call:bash" in (content or ""),
                "reasoning_literal_thought_tail": (reason or "").rstrip().endswith("thought"),
            }

        native = pk["native_response"]["choices"][0]["message"]
        native_observed = {
            "response_id": pk["native_response"]["id"],
            "structured_tool_names": [t["function"]["name"] for t in native.get("tool_calls") or []],
            "content_only_current_parser": parse(
                native.get("content") or "", tok.encode(native.get("content") or "", add_special_tokens=False)
            ),
            "canonical_full_reply_current_parser": parse(
                pk["reconstructed_full_reply"], tok.encode(pk["reconstructed_full_reply"], add_special_tokens=False)
            ),
        }
        for mode in ["served", "full"]:
            path = a.runs / mode / (job["name"] + ".json")
            if not path.exists():
                pending.append({"job": job["name"], "mode": mode})
                continue
            try:
                r = read(path)
            except ValueError:
                pending.append({"job": job["name"], "mode": mode, "reason": "partial JSON"})
                continue
            if "diagnostic_total_wall_s" not in r:
                pending.append({"job": job["name"], "mode": mode, "reason": "diagnostic still in progress"})
                continue
            trials = []
            for run in r.get("free_running", []):
                ids = run["token_ids"]
                text = tok.decode(ids, skip_special_tokens=False)
                cap = job.get("free_run_tokens", 0)
                trials.append(
                    {
                        "seed": run["sampling"].get("seed"),
                        "tokens": len(ids),
                        "at_requested_token_cap": bool(cap and len(ids) >= cap),
                        "generated_has_tool_close_token": tok.get_vocab()["<tool_call|>"] in ids,
                        "generated_last_id": ids[-1] if ids else None,
                        "generated_head_ids": ids[:12],
                        "generated_text_head": text[:100],
                        "generated_text_tail": text[-60:],
                        "conditioning_plus_generated": parse(prefix + text, prefixids + ids),
                        "generated_only": parse(text, ids),
                        "mechanical_flag_from_runner": run.get("flag"),
                        "mechanical_repetition_not_adjudicated_by_parser": True,
                    }
                )
            rows.append(
                {
                    "job": job["name"],
                    "mode": mode,
                    "result": str(path),
                    "result_sha256": sha(path),
                    "fixture_sha256": sha(job["case"]),
                    "packet_sha256": sha(packet),
                    "assistant_prefix_token_count": len(prefixids),
                    "assistant_prefix_tail_ids": prefixids[-20:],
                    "assistant_prefix_tail": prefix[-160:],
                    "native": native_observed,
                    "trials": trials,
                }
            )
    except Exception as e:
        errors.append({"job": job["name"], "error": type(e).__name__ + ": " + str(e)[:250]})
source_paths = [
    Path("python_env/lib/python3.10/site-packages/vllm") / x
    for x in [
        "parser/gemma4.py",
        "parser/engine/parser_engine.py",
        "parser/engine/adapters.py",
        "tool_parsers/gemma4_engine_tool_parser.py",
        "reasoning/gemma4_engine_reasoning_parser.py",
    ]
] + [snap / "chat_template.jinja", snap / "tokenizer.json", snap / "tokenizer_config.json", Path(__file__)]
out = {
    "status": "partial_completed_outputs_CPU_stock_nonstreaming_parser_only",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest_sha256": sha(a.manifest),
    "protocol_jobs_selected": len(selected),
    "selected_jobs": selected,
    "completed_case_modes": len(rows),
    "completed_trials": sum(len(r["trials"]) for r in rows),
    "pending": pending,
    "errors": errors,
    "sources_sha256": {str(p): sha(p) for p in source_paths},
    "rows": rows,
    "rules": [
        "Expected structured tool is bash with JSON mapping containing string command. Missing tool, malformed function name and invalid arguments are separate outcomes.",
        "Missing tool at token cap may be incomplete; inspect completion/stop and channel context before declaring reproduction.",
        "Valid structured call does not excuse visible GetAll-bash, stray thought, repetition, wrong actions or other semantic anomalies. Mechanical repetition is not assessed by this parser.",
        "Raw special-marker display alone is not reproduction: runner decodes skip_special_tokens=False. Compare actual channel-aware structured results with native response body.",
        "Native token IDs and exact historical stream absent. Canonical reconstruction/current nonstreaming parse is not proof of historical stream/parser equivalence or full-model correctness.",
    ],
}
totals = {}
for mode in ["served", "full"]:
    ts = [t for r in rows if r["mode"] == mode for t in r["trials"]]
    totals[mode] = {
        "case_modes": sum(r["mode"] == mode for r in rows),
        "trials": len(ts),
        "missing_tool": sum(t["conditioning_plus_generated"]["missing_tool"] for t in ts),
        "malformed_name": sum(t["conditioning_plus_generated"]["malformed_name"] for t in ts),
        "invalid_arguments": sum(t["conditioning_plus_generated"]["invalid_arguments"] for t in ts),
        "capped_trials": sum(t["at_requested_token_cap"] for t in ts),
    }
out["partial_counts"] = totals
out["rules"].append(
    "Full-prefix findings may include fixed inherited anomalous text. Compare generated-only findings and exact prefix before attributing anomaly to new sampling. A capped parser-extracted call is not necessarily complete/executable; tool-close-token presence is recorded separately."
)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(out, indent=2) + "\n")
summary = {
    k: out[k]
    for k in [
        "status",
        "created_utc",
        "protocol_jobs_selected",
        "completed_case_modes",
        "completed_trials",
        "partial_counts",
        "pending",
        "errors",
        "rules",
    ]
}
summary["detailed_audit_sha256"] = sha(a.output)
a.output.with_name(a.output.stem + "-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ["protocol_jobs_selected", "completed_case_modes", "completed_trials", "errors"]}))
