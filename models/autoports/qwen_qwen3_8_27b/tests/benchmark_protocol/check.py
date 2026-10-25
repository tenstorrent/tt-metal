"""Host-only check of the frozen GPQA population and native-template transport."""
import hashlib
import json
from pathlib import Path

import benchmark_stage
from benchmark_stage.gpqa import task_spec
from benchmark_stage.subsets import digest, flatten
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import get_task_dict
from transformers import AutoTokenizer

model_dir = Path(__file__).resolve().parents[2]
manifest = json.loads((model_dir / "doc/benchmark/subset/manifest.json").read_text())
name = "gpqa_diamond_cot_zeroshot"
task = flatten(get_task_dict([task_spec(name)]))[name]
docs = list(task.eval_docs)
frozen = manifest["tasks"][name]
assert digest(docs) == frozen["population_sha256"]
assert all(digest(docs[i]) == h for i, h in zip(frozen["indices"], frozen["document_sha256"]))
messages = [{"role": "user", "content": task.doc_to_text(docs[frozen["indices"][0]])}]
backend = LocalChatCompletion(model="Qwen/Qwen3.8-27B",
    base_url="http://127.0.0.1:8000/v1/chat/completions",
    tokenized_requests=False, tokenizer_backend=None)
structured = backend.apply_chat_template(messages)
assert json.loads(structured.prompt) == messages
payload = backend._create_payload(messages, gen_kwargs={"max_gen_toks": 16384,
    "temperature": 1.0, "do_sample": True, "until": [], "top_p": 0.95, "top_k": 20,
    "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh",
                             "preserve_thinking": True}})
assert payload["messages"] == messages and payload["stop"] == []
tokenizer_dir = model_dir.parents[2] / ".cache/qwen38-full"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
rendered = tokenizer.apply_chat_template(payload["messages"], tokenize=False,
    add_generation_prompt=True, **payload["chat_template_kwargs"])
assert rendered.count("<|im_start|>user\n") == 1
assert rendered.endswith("<|im_start|>assistant\n<think>\n")
print(json.dumps({"status": "pass", "benchmark_stage_module": benchmark_stage.__file__,
    "population": len(docs), "population_sha256": digest(docs),
    "selected": len(frozen["indices"]), "num_fewshot": task.config.num_fewshot,
    "native_template_sha256": hashlib.sha256((tokenizer_dir / "chat_template.jinja").read_bytes()).hexdigest(),
    "structured_messages_preserved": True, "native_template_application_count": 1,
    "native_thinking_prefix": True,
    "first_document_prompt_tokens": len(tokenizer.encode(rendered, add_special_tokens=False))}, indent=2))
