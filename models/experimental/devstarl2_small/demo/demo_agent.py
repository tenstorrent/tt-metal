import argparse
import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from models.experimental.devstarl2_small.devstral_utils.fp8_dequantize_compat import apply_fp8_dequantize_compat


apply_fp8_dequantize_compat()

DEFAULT_MODEL = "mistralai/Devstral-Small-2-24B-Instruct-2512"
DEFAULT_SYSTEM_PROMPT = (
    "You are Devstral Agent, a concise and practical coding assistant. "
    "Help with debugging, implementation, and code review in a local repository."
)
DEFAULT_AGENT_RULES = """
You have these tools available:
- terminal(command)
- bash(command)
- read_file(path, offset=1, limit=200)
- write_file(path, content, append=false)
- search_replace(path, search, replace, max_replacements=0)
- grep(pattern, path=".", glob="*", case_insensitive=false, max_results=200)
- web_search(query, max_results=5)
- web_fetch(url, max_chars=12000)
- todo(action, id="", content="", status="", items=[])
- ask_user_question(question)
- load_skill(path)
- inspect_codebase(path=".", max_entries=300)
- delegate_task(command, description="")

Tool-call format (only this content when calling a tool):
<tool_call>
{"name":"tool_name","arguments":{"arg":"value"}}
</tool_call>

When you receive <tool_result>, use it and continue.
Prefer read_file/write_file/search_replace/grep for repository tasks over generic terminal commands.
"""


@dataclass
class ChatConfig:
    model_id: str
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    device: str
    system_prompt: str
    workspace_root: str
    command_timeout_sec: int
    max_tool_calls_per_turn: int


@dataclass
class AgentState:
    todos: List[Dict[str, str]] = field(default_factory=list)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_model_and_tokenizer(config: ChatConfig):
    print(f"Loading model: {config.model_id}")
    tokenizer = MistralCommonBackend.from_pretrained(config.model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(config.model_id, device_map="auto")
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
        # Avoid noisy warning when max_new_tokens is provided at generation time.
        model.generation_config.max_length = None
    model.eval()
    torch.set_grad_enabled(False)
    print("Model loaded.")
    return model, tokenizer


def build_inputs(
    tokenizer: MistralCommonBackend,
    messages: List[Dict[str, str]],
    device: str,
) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer.apply_chat_template(
        conversation=messages,
        return_tensors="pt",
        return_dict=True,
    )
    return {k: v.to(device=device) for k, v in tokenized.items()}


def generate_assistant_text(
    model: Mistral3ForConditionalGeneration,
    tokenizer: MistralCommonBackend,
    messages: List[Dict[str, str]],
    config: ChatConfig,
) -> str:
    inputs = build_inputs(tokenizer, messages, config.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature if config.do_sample else None,
            top_p=config.top_p if config.do_sample else None,
        )[0]
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _limit_text(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _resolve_workspace_path(workspace_root: str, raw_path: str) -> Path:
    root = Path(workspace_root).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("Path escapes workspace_root.")
    return candidate


def run_shell(command: str, workspace_root: str, timeout_sec: int) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = (
            (completed.stdout or "")
            + ("\n" if completed.stdout and completed.stderr else "")
            + (completed.stderr or "")
        ).strip()
        return {
            "ok": completed.returncode == 0,
            "exit_code": completed.returncode,
            "output": _limit_text(output),
        }
    except subprocess.TimeoutExpired as exc:
        partial = ((exc.stdout or "") + ("\n" if exc.stdout and exc.stderr else "") + (exc.stderr or "")).strip()
        return {
            "ok": False,
            "exit_code": None,
            "output": _limit_text(partial),
            "error": f"Command timed out after {timeout_sec} seconds.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exit_code": None, "output": "", "error": str(exc)}


def tool_read_file(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, str(args.get("path", "")))
        offset = int(args.get("offset", 1))
        limit = int(args.get("limit", 200))
        lines = path.read_text(encoding="utf-8").splitlines()
        start = max(offset - 1, 0)
        end = min(start + max(limit, 1), len(lines))
        rendered = "\n".join(f"{idx + 1}|{lines[idx]}" for idx in range(start, end))
        return {"ok": True, "path": str(path), "output": _limit_text(rendered)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_write_file(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, str(args.get("path", "")))
        content = str(args.get("content", ""))
        append = bool(args.get("append", False))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": str(path), "bytes_written": len(content.encode("utf-8"))}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_search_replace(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, str(args.get("path", "")))
        search = str(args.get("search", ""))
        replace = str(args.get("replace", ""))
        max_replacements = int(args.get("max_replacements", 0))
        text = path.read_text(encoding="utf-8")
        if max_replacements <= 0:
            count = text.count(search)
            updated = text.replace(search, replace)
        else:
            updated = text.replace(search, replace, max_replacements)
            count = min(text.count(search), max_replacements)
        path.write_text(updated, encoding="utf-8")
        return {"ok": True, "path": str(path), "replacements": count}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_grep(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    pattern = str(args.get("pattern", ""))
    rel_path = str(args.get("path", "."))
    glob = str(args.get("glob", "*"))
    case_insensitive = bool(args.get("case_insensitive", False))
    max_results = int(args.get("max_results", 200))
    if not pattern:
        return {"ok": False, "error": "pattern is required"}
    target = _resolve_workspace_path(config.workspace_root, rel_path)
    cmd = ["rg", "-n", "--glob", glob, "--max-count", str(max_results), pattern, str(target)]
    if case_insensitive:
        cmd.insert(1, "-i")
    return run_shell(
        " ".join(subprocess.list2cmdline([part]) for part in cmd), config.workspace_root, config.command_timeout_sec
    )


def tool_web_fetch(args: Dict[str, Any]) -> Dict[str, Any]:
    url = str(args.get("url", ""))
    max_chars = int(args.get("max_chars", 12000))
    if not url:
        return {"ok": False, "error": "url is required"}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "devstral-demo-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return {"ok": True, "url": url, "output": _limit_text(body, max_chars=max_chars)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    query = str(args.get("query", "")).strip()
    max_results = int(args.get("max_results", 5))
    if not query:
        return {"ok": False, "error": "query is required"}
    encoded = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/html/?q={encoded}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "devstral-demo-agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        matches = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html)
        results: List[Dict[str, str]] = []
        for href, title_html in matches[:max_results]:
            title = re.sub(r"<.*?>", "", title_html)
            results.append({"title": title, "url": href})
        return {"ok": True, "query": query, "results": results}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Search failed: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_todo(args: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    action = str(args.get("action", "list")).lower()
    if action == "list":
        return {"ok": True, "todos": state.todos}
    if action == "clear":
        state.todos = []
        return {"ok": True, "todos": state.todos}
    if action == "add":
        item = {
            "id": str(args.get("id", f"todo-{len(state.todos) + 1}")),
            "content": str(args.get("content", "")),
            "status": str(args.get("status", "pending")),
        }
        state.todos.append(item)
        return {"ok": True, "todos": state.todos}
    if action == "update":
        todo_id = str(args.get("id", ""))
        for item in state.todos:
            if item["id"] == todo_id:
                if "content" in args:
                    item["content"] = str(args["content"])
                if "status" in args:
                    item["status"] = str(args["status"])
                return {"ok": True, "todos": state.todos}
        return {"ok": False, "error": f"Todo id not found: {todo_id}"}
    if action == "set":
        raw_items = args.get("items", [])
        if not isinstance(raw_items, list):
            return {"ok": False, "error": "items must be a list"}
        parsed: List[Dict[str, str]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            parsed.append(
                {
                    "id": str(raw.get("id", f"todo-{len(parsed) + 1}")),
                    "content": str(raw.get("content", "")),
                    "status": str(raw.get("status", "pending")),
                }
            )
        state.todos = parsed
        return {"ok": True, "todos": state.todos}
    return {"ok": False, "error": f"Unsupported todo action: {action}"}


def tool_ask_user_question(args: Dict[str, Any]) -> Dict[str, Any]:
    question = str(args.get("question", "Please provide more detail:")).strip()
    answer = input(f"[Agent question] {question}\nYour answer: ").strip()
    return {"ok": True, "question": question, "answer": answer}


def tool_load_skill(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        path = _resolve_workspace_path(config.workspace_root, str(args.get("path", "")))
        content = path.read_text(encoding="utf-8")
        return {"ok": True, "path": str(path), "content": _limit_text(content)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def tool_inspect_codebase(args: Dict[str, Any], config: ChatConfig) -> Dict[str, Any]:
    try:
        base = _resolve_workspace_path(config.workspace_root, str(args.get("path", ".")))
        max_entries = int(args.get("max_entries", 300))
        entries: List[str] = []
        for p in sorted(base.rglob("*")):
            if ".git" in p.parts:
                continue
            entries.append(str(p.relative_to(config.workspace_root)))
            if len(entries) >= max_entries:
                break
        return {"ok": True, "path": str(base), "entries": entries}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def execute_tool_call(tool_call: Dict[str, Any], config: ChatConfig, state: AgentState) -> Dict[str, Any]:
    name = str(tool_call.get("name", ""))
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return {"ok": False, "error": "arguments must be an object"}

    if name in ("terminal", "bash"):
        if "command" not in args:
            return {"ok": False, "error": "command is required"}
        return run_shell(str(args["command"]), config.workspace_root, config.command_timeout_sec)
    if name == "read_file":
        return tool_read_file(args, config)
    if name == "write_file":
        return tool_write_file(args, config)
    if name == "search_replace":
        return tool_search_replace(args, config)
    if name == "grep":
        return tool_grep(args, config)
    if name == "web_search":
        return tool_web_search(args)
    if name == "web_fetch":
        return tool_web_fetch(args)
    if name == "todo":
        return tool_todo(args, state)
    if name == "ask_user_question":
        return tool_ask_user_question(args)
    if name == "load_skill":
        return tool_load_skill(args, config)
    if name == "inspect_codebase":
        return tool_inspect_codebase(args, config)
    if name == "delegate_task":
        if "command" not in args:
            return {"ok": False, "error": "command is required"}
        result = run_shell(str(args["command"]), config.workspace_root, config.command_timeout_sec)
        result["description"] = str(args.get("description", ""))
        result["delegated"] = True
        return result
    return {"ok": False, "error": f"Unknown tool: {name}"}


def chat_loop(model: Mistral3ForConditionalGeneration, tokenizer: MistralCommonBackend, config: ChatConfig):
    system_content = f"{config.system_prompt}\n\n{DEFAULT_AGENT_RULES}".strip()
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    state = AgentState()

    print("\n--- Devstral2 Small Agent Demo ---")
    print("Type 'quit' or 'exit' to stop. Use '/clear' to reset conversation history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_content}]
            state = AgentState()
            print("Conversation history and todo state cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        final_response: Optional[str] = None
        last_tool_result: Optional[Dict[str, Any]] = None
        last_tool_name: Optional[str] = None
        last_tool_signature: Optional[str] = None
        repeated_tool_calls = 0

        for _ in range(config.max_tool_calls_per_turn):
            assistant_text = generate_assistant_text(model, tokenizer, messages, config)
            tool_call = parse_tool_call(assistant_text)

            if tool_call is None:
                final_response = assistant_text
                messages.append({"role": "assistant", "content": assistant_text})
                break

            messages.append({"role": "assistant", "content": assistant_text})
            call_signature = json.dumps(tool_call, sort_keys=True, ensure_ascii=True)
            if call_signature == last_tool_signature:
                repeated_tool_calls += 1
            else:
                repeated_tool_calls = 0
            last_tool_signature = call_signature

            tool_result = execute_tool_call(tool_call, config, state)
            last_tool_result = tool_result
            last_tool_name = str(tool_call.get("name", ""))
            result_content = f"<tool_result>\n{json.dumps(tool_result, ensure_ascii=True)}\n</tool_result>"
            messages.append({"role": "user", "content": result_content})

            if repeated_tool_calls >= 1:
                if tool_result.get("ok", False):
                    output = str(tool_result.get("output", "")).strip()
                    if output:
                        final_response = output
                    else:
                        final_response = f"Completed `{last_tool_name}` successfully."
                else:
                    err = str(tool_result.get("error", "")).strip()
                    out = str(tool_result.get("output", "")).strip()
                    details = err if err else out
                    final_response = f"`{last_tool_name}` failed." + (f" Details: {details}" if details else "")
                messages.append({"role": "assistant", "content": final_response})
                break

        if final_response is None:
            if last_tool_result is not None:
                if last_tool_result.get("ok", False):
                    output = str(last_tool_result.get("output", "")).strip()
                    final_response = output if output else "Tool execution completed successfully."
                else:
                    err = str(last_tool_result.get("error", "")).strip()
                    out = str(last_tool_result.get("output", "")).strip()
                    details = err if err else out
                    final_response = "Tool execution did not complete cleanly." + (
                        f" Details: {details}" if details else ""
                    )
            else:
                final_response = (
                    "I reached the per-turn tool-call limit before producing a final answer. "
                    "Try narrowing the request."
                )
            messages.append({"role": "assistant", "content": final_response})

        print(f"\nAssistant: {final_response}\n")


def parse_args() -> ChatConfig:
    parser = argparse.ArgumentParser(description="Interactive terminal agent demo for Devstral-Small-2")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens per answer")
    parser.add_argument("--temperature", type=float, default=0.15, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling value")
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (greedy decoding)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Input tensor placement device (model uses HF device_map='auto')",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt for the assistant",
    )
    parser.add_argument(
        "--workspace-root",
        default=str(Path.cwd()),
        help="Workspace root for tool execution",
    )
    parser.add_argument(
        "--command-timeout-sec",
        type=int,
        default=20,
        help="Timeout in seconds for shell and delegation commands",
    )
    parser.add_argument(
        "--max-tool-calls-per-turn",
        type=int,
        default=6,
        help="Maximum number of tool calls the assistant can make per user turn",
    )
    args = parser.parse_args()

    return ChatConfig(
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        device=resolve_device(args.device),
        system_prompt=args.system_prompt,
        workspace_root=str(Path(args.workspace_root).resolve()),
        command_timeout_sec=args.command_timeout_sec,
        max_tool_calls_per_turn=args.max_tool_calls_per_turn,
    )


def main():
    config = parse_args()
    model, tokenizer = load_model_and_tokenizer(config)
    chat_loop(model, tokenizer, config)


if __name__ == "__main__":
    main()
