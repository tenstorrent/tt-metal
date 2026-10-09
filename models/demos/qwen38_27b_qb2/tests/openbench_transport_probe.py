# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic local HTTP proof of the pinned evaluator's request and scorer path."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def run(output):
    from inspect_ai import Task
    from inspect_ai import eval as evaluate
    from inspect_ai.dataset import MemoryDataset
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import generate
    from openbench.evals.gpqa_diamond import record_to_mcq_sample
    from openbench.scorers.mcq import create_mcq_scorer

    from models.demos.qwen38_27b_qb2.demo.run_openbench_gpqa import create_model

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            time.sleep(0.15)
            payload = dict(
                id="synthetic",
                object="chat.completion",
                created=1,
                model="Qwen/Qwen3.8-27B",
                choices=[
                    dict(
                        index=0,
                        finish_reason="stop",
                        message=dict(
                            role="assistant",
                            content="Answer: (A)",
                            reasoning_content="This is an evaluator transport probe. Answer: (D) is not the final answer.",
                        ),
                    )
                ],
                usage=dict(prompt_tokens=8, completion_tokens=8, total_tokens=16),
            )
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass  # Expected when the short-timeout regression probe disconnects.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        sample = record_to_mcq_sample(
            dict(
                Question="Synthetic transport test: preserve [X] notation.",
                **{
                    "Correct Answer": "alpha [X]",
                    "Incorrect Answer 1": "beta [Y]",
                    "Incorrect Answer 2": "gamma [Z]",
                    "Incorrect Answer 3": "delta [W]",
                },
            )
        )
        assert "[X]" in sample.input
        sample.target = "A"  # This probe scores the synthetic response, never a benchmark question.
        task = Task(
            dataset=MemoryDataset([sample]),
            solver=generate(),
            scorer=create_mcq_scorer()(),
            config=GenerateConfig(temperature=0.5),
        )
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        short_model = create_model(base_url, request_timeout=0.02)
        assert short_model.api.client.timeout == 0.02 and short_model.api.client.max_retries == 0
        short_logs = evaluate(
            task,
            model=short_model,
            epochs=1,
            max_tokens=65536,
            max_retries=0,
            retry_on_error=0,
            fail_on_error=False,
            timeout=1,
            log_dir=str(output / "short-timeout"),
            display="none",
        )
        assert short_logs[0].samples[0].error is not None
        assert len(requests) == 1  # No hidden SDK retries after the timeout.
        model = create_model(base_url, request_timeout=2)
        assert model.api.client.timeout == 2 and model.api.client.max_retries == 0
        logs = evaluate(
            task,
            model=model,
            epochs=1,
            max_tokens=65536,
            max_connections=128,
            max_samples=128,
            max_retries=0,
            retry_on_error=0,
            fail_on_error=False,
            timeout=30,
            log_dir=str(output / "logs"),
            display="none",
        )
        log = logs[0]
        assert log.status == "success" and len(log.samples) == 1
        assert list(log.samples[0].scores.values())[0].value == "C"
        assert len(requests) == 2 and requests[0]["model"] == "Qwen/Qwen3.8-27B"
        assert requests[0]["temperature"] == 0.5 and requests[0]["max_tokens"] == 65536
        assert log.samples[0].output.choices[0].stop_reason == "stop"
        receipt = dict(
            passed=True,
            synthetic_only=True,
            hardware_opened=False,
            model=requests[0]["model"],
            max_tokens=requests[0]["max_tokens"],
            temperature=requests[0]["temperature"],
            answer_scored_from_final=True,
            reasoning_did_not_override_final=True,
            short_timeout_failed_after_one_request=True,
            longer_timeout_received_delayed_response=True,
            sdk_max_retries=model.api.client.max_retries,
            results=log.results.model_dump(mode="json"),
            log_location=log.location,
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "probe.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt))
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
