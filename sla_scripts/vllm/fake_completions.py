# Empty-answer /v1/completions server: lets lm-eval score a partial cache, counting unanswered questions as wrong.
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        k = (
            len(body.get("prompt", [0]))
            if isinstance(body.get("prompt"), list) and body["prompt"] and isinstance(body["prompt"][0], list)
            else 1
        )
        out = {
            "choices": [{"index": i, "text": "", "finish_reason": "stop"} for i in range(k)],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = json.dumps({"data": [{"id": "meta-llama/Llama-3.3-70B-Instruct"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", 8099), H).serve_forever()
