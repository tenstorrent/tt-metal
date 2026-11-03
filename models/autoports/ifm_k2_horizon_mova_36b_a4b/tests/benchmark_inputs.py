"""Retain all frozen upstream documents and structured zero-shot inputs."""
import json
from pathlib import Path

from benchmark_stage.gpqa import task_spec
from benchmark_stage.subsets import digest, flatten
from lm_eval.tasks import get_task_dict

root = Path(__file__).resolve().parents[1] / "doc/benchmark/run"
manifest = json.loads((root / "manifest.json").read_text())
config = json.loads((root / "run_config.json").read_text())
with (root / "benchmark_inputs.jsonl").open("w") as stream:
    for group in config["tasks"]:
        for name, task in flatten(get_task_dict([task_spec(group)])).items():
            docs = list(task.eval_docs)
            frozen = manifest["tasks"][name]
            if digest(docs) != frozen["population_sha256"]:
                raise ValueError("dataset mismatch")
            for index, expected in zip(frozen["indices"], frozen["document_sha256"]):
                doc = docs[index]
                if digest(doc) != expected:
                    raise ValueError("document mismatch")
                messages = task.fewshot_context(
                    doc,
                    num_fewshot=0,
                    apply_chat_template=True,
                    fewshot_as_multiturn=True,
                    chat_template=lambda messages, **kwargs: messages,
                )
                stream.write(
                    json.dumps(
                        dict(
                            group=group,
                            task=name,
                            doc_id=index,
                            document=doc,
                            document_sha256=expected,
                            messages=messages,
                            generation=config["generation"][group],
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
print("Archived all frozen upstream documents and structured messages")
