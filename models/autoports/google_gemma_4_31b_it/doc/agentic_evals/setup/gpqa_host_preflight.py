import importlib.metadata as md
import json
from pathlib import Path

from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import TaskManager, get_task_dict

p = Path(__file__).resolve().parent
cmd = json.loads((p / "gpqa_command.json").read_text())["argv"]
model = LocalChatCompletion.create_from_arg_string(cmd[cmd.index("--model_args") + 1], {"trust_remote_code": True})
t = model.tokenizer
messages = [{"role": "user", "content": "What is 2 + 2?"}]
rendered = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
encoded = t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=True)
ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
assert len(ids) > 2
manager = TaskManager()
task = get_task_dict(["r1_gpqa_diamond"], task_manager=manager)["r1_gpqa_diamond"]
assert len(task.eval_docs) == 198
selected = json.loads((p / "gpqa10_tasks.json").read_text())["tasks"]
for row in selected:
    doc = task.eval_docs[row["doc_id"]]
    assert doc["Record ID"] == row["record_id"]
    prompt = task.doc_to_text(doc)
    assert prompt and len(t.encode(prompt)) > 0
record = {
    "LocalChatCompletion_construction": "pass",
    "pinned_tokenizer_path": str(t.name_or_path),
    "chat_template_thinking_true": "pass",
    "template_tokens": len(ids),
    "task_docs": len(task.eval_docs),
    "selected_doc_identity_and_prompt_tokenization": "pass",
    "versions": {
        k: md.version(k)
        for k in ["lm-eval", "transformers", "protobuf", "tokenizers", "huggingface-hub", "datasets", "torch", "pyjwt"]
    },
    "model_calls": 0,
}
(p / "gpqa_full_host_preflight.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
