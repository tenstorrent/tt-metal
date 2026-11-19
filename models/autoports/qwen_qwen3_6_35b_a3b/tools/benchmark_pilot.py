# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Time four frozen native-thinking requests before sizing the timed stage."""

import concurrent.futures
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from benchmark_stage.subsets import digest, flatten
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import get_task_dict

MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL / "doc/benchmark"
GENERATION = dict(
    max_gen_toks=32768,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
    do_sample=True,
    until=[],
    chat_template_kwargs=dict(enable_thinking=True),
)


def main():
    manifest = json.loads((ROOT / "subset-common-accessible/manifest.json").read_text())
    backend = LocalChatCompletion(
        model="Qwen/Qwen3.6-35B-A3B",
        base_url="http://127.0.0.1:8000/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
    )
    requests = []
    for group in ("mmlu_pro", "ifeval"):
        tasks = flatten(get_task_dict([group]))
        name = sorted(tasks)[0]
        task = tasks[name]
        task.set_fewshot_seed(seed=1234)
        task.set_config(key="generation_kwargs", value=GENERATION, update=True)
        task.build_all_requests(
            samples=manifest["tasks"][name]["indices"][:2],
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            chat_template=backend.apply_chat_template,
        )
        for instance in task.instances:
            messages, kwargs = instance.args
            payload = backend._create_payload(json.loads(messages.prompt), generate=True, gen_kwargs=dict(kwargs))
            requests.append(
                dict(
                    group=group,
                    task=name,
                    doc_id=instance.doc_id,
                    document=instance.doc,
                    document_sha256=digest(instance.doc),
                    payload=payload,
                )
            )
    (ROOT / "setup/pilot-requests.json").write_text(json.dumps(requests, indent=2) + "\n")

    def send(index, row):
        start = time.perf_counter()
        req = Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(row["payload"]).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=3600) as response:
            raw = json.load(response)
        result = dict(
            index=index,
            task=row["task"],
            doc_id=row["doc_id"],
            start=start,
            seconds=time.perf_counter() - start,
            response=raw,
        )
        (ROOT / f"setup/pilot-response-{index}.json").write_text(json.dumps(result, indent=2) + "\n")
        return {key: value for key, value in result.items() if key != "response"} | {
            "usage": raw["usage"],
            "finish_reason": raw["choices"][0]["finish_reason"],
            "final_characters": len(raw["choices"][0]["message"].get("content") or ""),
        }

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(send, i, row) for i, row in enumerate(requests)]
        rows = [future.result() for future in futures]
    result = dict(
        seconds=time.perf_counter() - started,
        server_slots=32,
        concurrent_requests=4,
        rows=rows,
        generation=GENERATION,
        template="Structured messages; native pinned HF template applied once by server",
    )
    (ROOT / "setup/pilot-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
