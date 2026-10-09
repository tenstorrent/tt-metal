# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exercise official Tau/LiteLLM transport with synthetic local HTTP responses."""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def run(output):
    from tau2.data_model.message import UserMessage
    from tau2.domains.airline.environment import get_environment
    from tau2.utils import llm_utils

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                dict(
                    path=self.path,
                    payload=payload,
                    authorization=self.headers.get("Authorization"),
                    timeout_header=self.headers.get("x-stainless-read-timeout"),
                )
            )
            if self.path.startswith("/timeout/"):
                time.sleep(0.2)
            message = dict(role="assistant", content="Synthetic simulator response")
            if payload["model"] == "Qwen/Qwen3.8-27B":
                message = dict(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        dict(
                            id="synthetic-tool",
                            type="function",
                            function=dict(name="get_user_details", arguments='{"user_id":"synthetic_user"}'),
                        )
                    ],
                )
            response = dict(
                id="synthetic",
                object="chat.completion",
                created=1,
                model=payload["model"],
                choices=[
                    dict(index=0, finish_reason="tool_calls" if message.get("tool_calls") else "stop", message=message)
                ],
                usage=dict(prompt_tokens=8, completion_tokens=8, total_tokens=16),
            )
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    original = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "synthetic-reference-key"
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        agent = llm_utils.generate(
            model="openai/Qwen/Qwen3.8-27B",
            messages=[UserMessage(role="user", content="Synthetic tool test")],
            tools=get_environment().get_tools(),
            api_base=base + "/agent/v1",
            api_key="EMPTY",
            timeout=900,
            num_retries=0,
            max_retries=0,
            temperature=0,
            max_tokens=16384,
            extra_body={"top_k": 1, "chat_template_kwargs": {"enable_thinking": True}},
        )
        simulator = llm_utils.generate(
            model="openai/openai/gpt-5.1",
            messages=[UserMessage(role="user", content="Synthetic simulator test")],
            api_base=base + "/reference/v1",
            timeout=900,
            num_retries=0,
            max_retries=0,
            temperature=0,
        )
        assert len(requests) == 2
        assert requests[0]["path"].startswith("/agent/") and requests[0]["authorization"] == "Bearer EMPTY"
        assert (
            requests[1]["path"].startswith("/reference/")
            and requests[1]["authorization"] == "Bearer synthetic-reference-key"
        )
        payload = requests[0]["payload"]
        assert payload["top_k"] == 1 and payload["chat_template_kwargs"]["enable_thinking"] is True
        assert payload["tools"] and payload["max_tokens"] == 16384
        assert agent.tool_calls[0].name == "get_user_details"
        assert agent.tool_calls[0].arguments == {"user_id": "synthetic_user"}
        assert simulator.content == "Synthetic simulator response"
        assert requests[1]["payload"]["model"] == "openai/gpt-5.1"
        assert float(requests[0]["timeout_header"]) == 900
        timeout_error = None
        try:
            llm_utils.generate(
                model="openai/Qwen/Qwen3.8-27B",
                messages=[UserMessage(role="user", content="Synthetic timeout test")],
                api_base=base + "/timeout/v1",
                api_key="EMPTY",
                timeout=0.02,
                num_retries=0,
                max_retries=0,
            )
        except Exception as error:
            timeout_error = type(error).__name__
        assert timeout_error and len(requests) == 3
        result = dict(
            passed=True,
            synthetic_only=True,
            hardware_opened=False,
            external_model_calls=0,
            agent_and_simulator_routes_separate=True,
            simulator_credential_not_sent_to_agent=True,
            model_tool_parser_used=True,
            sampling_parameters_preserved=True,
            sdk_timeout_seconds=900,
            short_timeout_failed_without_retries=True,
            requests=len(requests),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        if original is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
