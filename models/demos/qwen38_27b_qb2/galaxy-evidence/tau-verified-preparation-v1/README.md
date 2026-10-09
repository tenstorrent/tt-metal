# Independent Tau-Verified airline preparation

Prepared Oct 9, 2026. **No scored evaluation or external model call yet.**
The user selected OpenRouter for the independent simulator. No
`OPENROUTER_API_KEY` was configured in the inspected local or SSH environments;
the runner fails explicitly rather than using Qwen as its own simulator.

The isolated host environment pins
[Amazon's Tau2-Bench-Verified](https://github.com/amazon-agi/tau2-bench-verified)
at `864350a8971a8f8ee9e7b8472e2edc380a806b0c`, including hashes of the database,
policy, tasks and split. CPU preflight loaded all 50 airline tasks and their
tools successfully. Its dependency freeze is retained here. The environment is
`/home/ttuser/qwen38-artifacts-20261007/tau-verified-v1`.

The prepared runner defaults to OpenRouter's `https://openrouter.ai/api/v1`,
GPT-5.1 simulator and GPT-4o-mini assertion judge, separate from the local Qwen
agent endpoint. It uses all 50 tasks, one trial, seed 300, concurrency 8,
temperature 0, at most 200 steps and a two-hour default wall limit. Requests
use 900 seconds and zero SDK retries. Qwen gets 16K output budget, thinking
enabled and top_k=1. Raw responses, transport failures, truncations and partial
trials are retained. Incomplete runs receive no claimed reference accuracy.

The real Tau/LiteLLM transport was tested against synthetic local HTTP servers:
agent/simulator routing and credentials stay separate, tool JSON is parsed by
the actual library, sampling fields survive transport, the SDK receives the
timeout, and a delayed response triggers exactly one request without retry.
Transport v2 also verifies OpenRouter's `openai/gpt-5.1` model identifier.
These passes establish harness wiring, not agent quality or OpenRouter availability.

The initial `runner-sha256.txt` identifies the earlier preflight runner before
OpenRouter defaults were added. The current source and repaired fixture tests
were subsequently included in the BFP8 follow-up snapshot and its full CPU
preflight: 456 tests plus 40 subtests passed, one skipped. The preparation
receipts have not been rewritten as proof of the later source bytes.

This is the official verified airline task through its native runner, not a
claim to reproduce OpenRouter's complete private benchmark execution settings.
The earlier 3/12 self-simulator Tau pilot remains historical and does not
qualify current BFP8 agentic quality. No full Tau job has been queued yet.
