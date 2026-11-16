"""Read-only CPU stock-parser interpretation of one saved diagnostic output."""
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"ttnn", "tt_lib"}:
            raise RuntimeError("Device imports forbidden: " + fullname)


sys.meta_path.insert(0, Guard())
from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.gemma4 import Gemma4Parser

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
O = D / "anomalies/swe85-sympy13480-controls"
result = D / "setup/swe85-pilot-split85/runs/rest/served/anomaly.sympy__sympy-13480.call1.onset.json"
r = json.loads(result.read_text())
c = json.loads(Path(r["case"]).read_text())
pk = json.loads(Path(c["metadata"]["raw_packet"]).read_text())
q = json.loads(pk["raw_request_body"])
rev = c["metadata"]["hf_revision"]
snap = Path("/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots") / rev
tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True, revision=rev)
request = ChatCompletionRequest(**q)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse(text, ids):
    p = Gemma4Parser(tok, tools=request.tools, chat_template_kwargs=q.get("chat_template_kwargs", {}))
    reason, content, tools = p.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)
    return {
        "reasoning_chars": len(reason or ""),
        "content_chars": len(content or ""),
        "content_preview": (content or "")[:100],
        "structured_tools": [
            {
                "name": t.name,
                "arguments_valid_json": isinstance(json.loads(t.arguments), dict),
                "argument_keys": list(json.loads(t.arguments)),
            }
            for t in tools or []
        ],
    }


full = json.loads(Path(c["metadata"]["source_full_reply_case"]).read_text())
n = len(full["prompt_token_ids"])
prefixids = c["prompt_token_ids"][n:]
prefix = tok.decode(prefixids, skip_special_tokens=False)
rows = []
for run in r["free_running"]:
    ids = run["token_ids"]
    text = tok.decode(ids, skip_special_tokens=False)
    rows.append(
        {
            "seed": run["sampling"]["seed"],
            "generated_token_count": len(ids),
            "generated_head_ids": ids[:12],
            "generated_text_head": text[:100],
            "generated_text_tail": text[-80:],
            "full_reconstructed_assistant_parse": parse(prefix + text, prefixids + ids),
            "generated_only_parse": parse(text, ids),
        }
    )
native = pk["native_response"]["choices"][0]["message"]
observed = native.get("content") or ""
output = {
    "status": "CPU_stock_parser_test_no_inference",
    "fixture_sha256": digest(r["case"]),
    "served_result_sha256": digest(result),
    "packet_sha256": digest(c["metadata"]["raw_packet"]),
    "prefix_token_count": len(prefixids),
    "prefix_tail": prefix[-180:],
    "prefix_tail_ids": prefixids[-20:],
    "template_sha256": digest(snap / "chat_template.jinja"),
    "native_observation": {
        "structured_tool_count": len(native.get("tool_calls") or []),
        "content_head": observed[:100],
        "content_only_parse": parse(observed, tok.encode(observed, add_special_tokens=False)),
    },
    "trials": rows,
    "sources": {
        str(p): digest(p)
        for p in [
            Path("python_env/lib/python3.10/site-packages/vllm/parser/gemma4.py"),
            Path("python_env/lib/python3.10/site-packages/vllm/parser/engine/parser_engine.py"),
            Path("python_env/lib/python3.10/site-packages/vllm/tool_parsers/gemma4_engine_tool_parser.py"),
            Path("python_env/lib/python3.10/site-packages/vllm/reasoning/gemma4_engine_reasoning_parser.py"),
            Path("models/autoports/google_gemma_4_31b_it/tools/precision_repro_batch.py"),
        ]
    },
    "limitations": "Current stock non-streaming parser test on canonical reconstructed assistant prefix plus generated token IDs. Native original token IDs and exact serialized stream absent; parser-version/stream equivalence to historical response not proven. Visible protocol delimiters in raw decoder text alone do not prove a protocol anomaly.",
}
output["interpretation"] = {
    "classification": "native_pseudo_tool_anomaly_not_reproduced_in_three_served_free_trials_under_current_stock_parser",
    "basis": "All seeds42/43/44 produce actual tool opener token48 and parse to bash(command) with JSON arguments. Native captured content lacks structured tool call and parses as visible content. Additional channel close token101 is redundant at this content-state onset, but parser tolerates it; this alone is not the native missing-tool anomaly.",
    "raw_decoder_policy": "precision_repro_batch decodes with skip_special_tokens=False; ordinary tool serialization therefore remains visible. tool_response is a turn boundary/stop marker, not an executed or invented response payload.",
    "triage_rule": [
        "Reconstruct full assistant prefix plus generated tokens and retain exact channel state; parse with served stock parser and actual tools/request settings.",
        "Require structured call with allowed function name and schema-valid arguments; raw delimiter appearance alone is insufficient.",
        "For native missing-tool anomalies, count reproduction only if pseudo-call persists in visible content or no valid structured call is produced; classify malformed names/arguments separately.",
        "Distinguish intentionally capped incomplete tool calls from completed parse failures, and do not infer tool execution from serialized calls.",
        "Keep mechanical repetition metrics separate from protocol validity and semantic action correctness; lack of native IDs prevents exact native-equivalence claims.",
    ],
}
(O / "served-protocol-parser-interpretation.json").write_text(json.dumps(output, indent=2) + "\n")
print(
    json.dumps({"prefix_tail": output["prefix_tail"], "native": output["native_observation"], "trials": rows}, indent=2)
)
