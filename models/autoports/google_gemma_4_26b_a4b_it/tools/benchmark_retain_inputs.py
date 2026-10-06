# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct frozen upstream requests without inference, retaining interrupted inputs."""

import argparse
import json
from pathlib import Path

from benchmark_stage.gpqa import task_spec
from benchmark_stage.subsets import digest
from lm_eval import evaluator
from lm_eval.models.openai_completions import LocalChatCompletion


class InputsRetained(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    root = parser.parse_args().run_dir
    config = json.loads((root / "run_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    membership = {child: group for group in config["tasks"] for child in manifest["groups"][group]["tasks"]}
    samples = {child: manifest["tasks"][child]["indices"] for child in membership}
    expected = sum(len(v) for v in samples.values())

    class Capture(LocalChatCompletion):
        def generate_until(self, requests, **kwargs):
            rows = []
            for req in requests:
                task = req.task_name
                original_id = samples[task][req.doc_id]
                document_hash = digest(req.doc)
                if document_hash != manifest["tasks"][task]["document_sha256"][req.doc_id]:
                    raise ValueError("Reconstructed document differs from frozen content")
                rows.append(
                    {
                        "group": membership[task],
                        "task": task,
                        "doc_id": original_id,
                        "doc": req.doc,
                        "document_sha256": document_hash,
                        "arguments": list(req.args),
                        "request_sha256": digest(list(req.args)),
                    }
                )
            if len(rows) != expected:
                raise ValueError("Incomplete reconstructed input set")
            mapping = {(row["task"], row["doc_id"]): row["request_sha256"] for row in rows}
            matched = 0
            for path in root.glob("*/request_links.jsonl"):
                for line in path.read_text().splitlines():
                    link = json.loads(line)
                    if mapping[(link["task"], link["doc_id"])] != link["request_sha256"]:
                        raise ValueError("Reconstructed messages differ from an actual request")
                    matched += 1
            with (root / "benchmark-inputs.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            (root / "input-retention.json").write_text(
                json.dumps(
                    {
                        "captured_inputs": len(rows),
                        "actual_request_hashes_verified": matched,
                        "inference_requests_sent": 0,
                        "method": "Upstream request construction with identical seeds, frozen documents and native structured-chat transport; stopped before any API inference or scoring.",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"Retained {len(rows)} inputs; verified {matched} actual request hashes; no inference or synthetic scores"
            )
            raise InputsRetained

    backend = Capture(
        model=config["model"],
        base_url=config["base_url"] + "/v1/chat/completions",
        num_concurrent=32,
        max_retries=0,
        timeout=3600,
        tokenized_requests=False,
        tokenizer_backend=None,
        max_gen_toks=2048,
    )
    try:
        evaluator.simple_evaluate(
            model=backend,
            tasks=[task_spec(group) for group in config["tasks"]],
            samples=samples,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            log_samples=True,
            gen_kwargs=config["generation"][config["tasks"][0]],
            random_seed=0,
            numpy_random_seed=1234,
            torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
    except InputsRetained:
        return
    raise RuntimeError("Expected input-capture boundary was not reached")


if __name__ == "__main__":
    main()
