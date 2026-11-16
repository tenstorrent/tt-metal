#!/usr/bin/env python3
"""Deferred CPU-only exact-prefix audit of P5 outputs with pinned Gemma4Parser.

Default behavior is inert and does not import transformers/vLLM. Use --ast-helper-only
for the stdlib-only converter probe. --execute runs only the stock parser on saved
text/IDs; it performs no inference, model load, tool execution, or device work.
"""
import argparse
import ast
import hashlib
import importlib.abc
import json
import sys
from pathlib import Path

D = Path("models/autoports/google_gemma_4_31b_it/doc/agentic_evals")
BASE = D / "setup/swe85-precision-bisect/down-bf16-hifi4-seven-case-candidate"
REV = "842da3794eaa0b77d5f08bae87a17459d91ff475"
SNAP = Path("/home/mvasiljevic/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots") / REV
TOKENIZER = SNAP / "tokenizer.json"
PARSER_SOURCE = Path("python_env/lib/python3.10/site-packages/vllm/parser/gemma4.py")
RESULT_NAMES = [
    "anomaly.pytest-dev__pytest-5787.call65.onset.seed42.p5_downbf16_hifi4.cap2048.json",
    "anomaly.sphinx-doc__sphinx-11510.call101.contraction.onset.seed43.p5_downbf16_hifi4.cap2048.json",
    "anomaly.sphinx-doc__sphinx-11510.call119.protocol.onset.seed42.p5_downbf16_hifi4.cap1024.json",
    "anomaly.sphinx-doc__sphinx-11510.call119.protocol.onset.seed43.p5_downbf16_hifi4.cap2048.json",
    "anomaly.sphinx-doc__sphinx-11510.call119.protocol.onset.seed44.p5_downbf16_hifi4.cap1024.json",
    "anomaly.sphinx-doc__sphinx-11510.call86.protocol.onset.seed44.p5_downbf16_hifi4.cap1024.json",
    "anomaly.sympy__sympy-14711.call1.onset.seed42.p5_downbf16_hifi4.cap256.json",
]


class DeviceImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split(".")[0] in {"ttnn", "tt_lib"}:
            raise RuntimeError("Forbidden device import in parser audit: " + fullname)
        return None


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_text())


def ast_helper_probe():
    """Run only pure Gemma arg converter helpers extracted from pinned source AST."""
    source = PARSER_SOURCE.read_text()
    tree = ast.parse(source)
    wanted = {"_parse_gemma4_args", "_parse_gemma4_array", "_strip_partial_delim", "_gemma4_arg_converter"}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in wanted]
    assert {n.name for n in nodes} == wanted

    class NullLogger:
        def warning(self, *args, **kwargs):
            pass

    ns = {"json": json, "STRING_DELIM": '<|"|>', "_DELIM_LEN": 5, "logger": NullLogger()}
    ns["_PARTIAL_DELIM_SUFFIXES"] = tuple(ns["STRING_DELIM"][:k] for k in range(len(ns["STRING_DELIM"]), 0, -1))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PARSER_SOURCE), "exec"), ns)
    examples = {
        "canonical_gemma": 'command:<|"|>echo OK<|"|>',
        "json_object_raw_args": '"command":"echo OK"',
        "json_invalid_escape_raw_args": '"command":"cat \\\'EOF\\\'"',
    }
    return {
        "parser_source_sha256": sha(PARSER_SOURCE),
        "helper_scope": "AST-extracted pure converter/argument parser only; no vLLM imports; does not test ParserEngine token/state handling",
        "outputs": {
            k: {
                "raw_args": v,
                "converted_json": ns["_gemma4_arg_converter"](v, False),
                "decoded_mapping": json.loads(ns["_gemma4_arg_converter"](v, False)),
            }
            for k, v in examples.items()
        },
    }


def actual_parse(parser_cls, tokenizer, request, template_kwargs, text, ids):
    parser = parser_cls(tokenizer, tools=request.tools, chat_template_kwargs=template_kwargs)
    reason, content, tools = parser.parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)
    parsed = []
    for t in tools or []:
        try:
            args = json.loads(t.arguments)
            valid = isinstance(args, dict) and isinstance(args.get("command"), str)
            error = None
        except Exception as e:
            args = {}
            valid = False
            error = type(e).__name__ + ": " + str(e)[:160]
        parsed.append(
            {
                "name": t.name,
                "arguments_raw": t.arguments,
                "arguments_json_parsed": args,
                "valid_bash_command_schema": t.name == "bash" and valid,
                "json_parse_error": error,
            }
        )
    return {
        "tool_count": len(parsed),
        "tools": parsed,
        "missing_tool": not parsed,
        "malformed_name": any(t["name"] != "bash" for t in parsed),
        "invalid_arguments": any(not t["valid_bash_command_schema"] for t in parsed),
        "visible_content_chars": len(content or ""),
        "visible_content_head": (content or "")[:120],
        "reasoning_chars": len(reason or ""),
    }


def execute(output):
    sys.meta_path.insert(0, DeviceImportGuard())
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.parser.gemma4 import Gemma4Parser

    tok = AutoTokenizer.from_pretrained(str(SNAP), revision=REV, local_files_only=True)
    rows = []
    errors = []
    for name in RESULT_NAMES:
        path = BASE / "runs/down_bf16_hifi4" / name
        try:
            r = read(path)
            case = Path(r["case"])
            fixture = read(case)
            meta = r["metadata"]
            packet = Path(meta["raw_packet"])
            pk = read(packet)
            q = json.loads(pk["raw_request_body"])
            request = ChatCompletionRequest(**q)
            original = read(meta["source_full_reply_case"])
            request_n = len(original["prompt_token_ids"])
            full_ids = fixture["prompt_token_ids"]
            prefix_ids = full_ids[request_n:]
            prefix = tok.decode(prefix_ids, skip_special_tokens=False)
            run = r["free_running"][0]
            out_ids = run["token_ids"]
            generated = tok.decode(out_ids, skip_special_tokens=False)
            joined_ids = prefix_ids + out_ids
            joined = prefix + generated
            # Three controls are formed at the same exact saved open-tool prefix.
            # Canonical Gemma args, syntactically valid JSON args, and invalid JSON escape.
            control_suffixes = {
                "canonical_gemma": '{command:<|"|>echo OK<|"|>}<tool_call|>',
                "valid_json_style": '{"command":"echo OK"}<tool_call|>',
                "invalid_escape_json_style": '{"command":"cat \\\'EOF\\\'"}<tool_call|>',
            }
            controls = {}
            for cname, suffix in control_suffixes.items():
                suffix_ids = tok.encode(suffix, add_special_tokens=False)
                controls[cname] = {
                    "suffix": suffix,
                    "parse": actual_parse(
                        Gemma4Parser,
                        tok,
                        request,
                        q.get("chat_template_kwargs", {}),
                        prefix + suffix,
                        prefix_ids + suffix_ids,
                    ),
                }
            rows.append(
                {
                    "result": str(path),
                    "result_sha256": sha(path),
                    "fixture": str(case),
                    "fixture_sha256": sha(case),
                    "raw_packet": str(packet),
                    "raw_packet_sha256": sha(packet),
                    "prompt_token_count": len(full_ids),
                    "assistant_prefix_token_count": len(prefix_ids),
                    "assistant_prefix_tail": prefix[-220:],
                    "assistant_prefix_tail_ids": prefix_ids[-24:],
                    "seed": run["sampling"]["seed"],
                    "cap": len(run["token_ids"]) if False else int(name.rsplit("cap", 1)[1].split(".")[0]),
                    "generated_token_count": len(out_ids),
                    "generated_text_sha256": sha_bytes(generated.encode()),
                    "generated_text_head": generated[:240],
                    "generated_text_tail": generated[-240:],
                    "conditioned_plus_generation": actual_parse(
                        Gemma4Parser, tok, request, q.get("chat_template_kwargs", {}), joined, joined_ids
                    ),
                    "generated_only": actual_parse(
                        Gemma4Parser, tok, request, q.get("chat_template_kwargs", {}), generated, out_ids
                    ),
                    "synthetic_same_prefix_controls": controls,
                }
            )
        except Exception as e:
            errors.append({"result": name, "error": type(e).__name__ + ": " + str(e)[:400]})
    sources = [
        PARSER_SOURCE,
        Path("python_env/lib/python3.10/site-packages/vllm/parser/engine/parser_engine.py"),
        Path("python_env/lib/python3.10/site-packages/vllm/parser/engine/adapters.py"),
        TOKENIZER,
    ]
    result = {
        "status": "read_only_pinned_stock_parser_audit",
        "hf_revision": REV,
        "tokenizer_sha256": sha(TOKENIZER),
        "p5_triage_sha256": sha(BASE / "triage-d.json"),
        "parser_source_hashes": {str(p): sha(p) for p in sources},
        "synthetic_controls": "All three controls are appended to each exact saved open-tool assistant prefix and parsed with the same request/tools. No tool execution.",
        "rows": rows,
        "errors": errors,
        "limitations": "Non-streaming current parser reconstruction only; historical server stream/native token IDs are not available. JSON syntax by itself is not treated as evidence that Gemma custom arguments yield a command field.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps({"status": result["status"], "rows": len(rows), "errors": errors, "output": str(output)}, indent=2)
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--execute",
        action="store_true",
        help="Actually import and run pinned parser on saved CPU text only; no inference/device.",
    )
    ap.add_argument(
        "--ast-helper-only",
        action="store_true",
        help="stdlib-only AST extraction of argument converter helpers; no parser/vLLM imports.",
    )
    ap.add_argument("--output", type=Path, default=BASE / "p5-gemma4-parser-audit.json")
    a = ap.parse_args()
    if a.ast_helper_only:
        print(json.dumps(ast_helper_probe(), indent=2, ensure_ascii=False))
    elif a.execute:
        execute(a.output)
    else:
        print(
            json.dumps(
                {
                    "status": "inert_plan",
                    "message": "No imports or parsing performed. Use --ast-helper-only for converter-only probe or --execute after device window is clear.",
                    "results": RESULT_NAMES,
                    "output": str(a.output),
                },
                indent=2,
            )
        )
