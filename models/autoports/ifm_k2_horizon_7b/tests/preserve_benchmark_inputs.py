"""Rebuild frozen upstream request inputs without making an inference call."""

import json
from pathlib import Path

from benchmark_stage.evaluate import evaluate_groups
from benchmark_stage.subsets import digest
from lm_eval.models.openai_completions import LocalChatCompletion

ROOT = Path(__file__).resolve().parents[1] / "doc/benchmark/run"


class InputsSaved(Exception):
    pass


def main():
    config = json.loads((ROOT / "run_config.json").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    output = ROOT / "input-reconstruction"

    def preserve(self, requests, **kwargs):
        rows = []
        for request in requests:
            task = request.task_name
            rows.append(
                {
                    "task": task,
                    "doc_id": manifest["tasks"][task]["indices"][request.doc_id],
                    "doc": request.doc,
                    "doc_sha256": digest(request.doc),
                    "arguments": request.args,
                    "messages": self.create_message([request.args[0]]),
                    "request_sha256": digest([request.args[0], request.args[1]]),
                }
            )
        with (ROOT / "benchmark-inputs.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes = {row["request_sha256"] for row in rows}
        linked = []
        for path in ROOT.glob("*/request_links.jsonl"):
            linked.extend(json.loads(line) for line in path.read_text().splitlines())
        assert len(rows) == sum(manifest["groups"][task]["sample_count"] for task in config["tasks"])
        assert all(row["request_sha256"] in hashes for row in linked)
        (ROOT / "input-reconstruction.json").write_text(
            json.dumps(
                {
                    "method": "Same pinned upstream task/request construction and seeds; intercepted before HTTP",
                    "requests": len(rows),
                    "linked_completed_requests_checked": len(linked),
                    "all_completed_request_hashes_match": True,
                    "inference_calls": 0,
                },
                indent=2,
            )
            + "\n"
        )
        raise InputsSaved

    LocalChatCompletion.generate_until = preserve
    try:
        evaluate_groups(
            model=config["model"],
            base_url=config["base_url"],
            manifest_path=ROOT / "manifest.json",
            groups=config["tasks"],
            output=output,
            generation=config["generation"][config["tasks"][0]],
            shared=True,
        )
    except InputsSaved:
        print((ROOT / "input-reconstruction.json").read_text())


if __name__ == "__main__":
    main()
