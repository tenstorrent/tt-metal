# Xing4.0-29B-A4B serve stack: every token through tt-d-gen's prefill engine

An OpenAI-compatible server whose every generated token is a **prefill** that tt-d-gen's real engine
(`te.BackendRuntime`, PREFILL role) schedules onto tt-metal's prefill runner ("decode via prefill"). It is slow
(about 1 s per token at short context) by design. It exists to hammer the engine's prefill path for correctness
and stability: admission, slot choice, prefix reuse (a remount on every step), interleaving concurrent requests,
eviction, the pulled-back last chunk, and the H2D / layer-ack channels, for hours.

tt-d-gen's design has a prefill node that never produces tokens (`PREFILL_DONE` only); a decode node generates them
after KV migration. Xing has no decode node, so the next token comes from the runner side: after each chunk, the
runner puts the chunk's last real row through the final norm and the LM head and leaves the logits in `/dev/shm`.
The engine never sees generated tokens as decode; each one is a follow-up "turn" (prompt + tokens so far) that the
engine remounts from the slot's resident prefix (64-token blocks) and pads to one 5120-token chunk.

## Processes

| process | Python | does |
|---|---|---|
| `server.py` (front end, under pytest) | python_env | HTTP, chat template, tokenizer, sampling, per-step checks, heartbeat; starts and watches the runner |
| `runner.py` | python_env | tt-metal `prefill_runner.main` on the 4x2 mesh (FABRIC_2D, D2H layer acks), + the LM head per chunk; starts the engine daemon |
| `engine_daemon.py` | tt-d-gen `venv312` (stdlib + `tt_engine`) | one `BackendRuntime` for the server's life; admits over a Unix socket, answers on `PREFILL_DONE` |

One generation step: front end `admit(ids)` -> daemon -> engine chunks it onto the H2D stream -> runner prefills,
acks 40 layers, writes `s<slot>_e<end>_<seq>.bin` -> daemon gets `PREFILL_DONE` -> front end takes the logits file of
the chunk the server plan says is last (`server_rules.chunk_plan(n, resident)`), samples, appends, admits again.

## Start / stop (on the box)

```bash
cd <this checkout>
BRINGUP_HF=/localdev/$USER/hf_models/Xing4.0-29B-A4B models/demos/xing40_a4b_d_p/serve/start.sh
grep -m1 'XING_SERVE: listening' generated/xing_serve/server.log     # ~1 min with the weight cache warm
models/demos/xing40_a4b_d_p/serve/stop.sh
```

`start.sh` runs `test_serve.py` through `scripts/run_safe_pytest.sh` (detached): the server holds the device lock while
it runs, gets hang triage, and the device is reset afterwards. Defaults: `BRINGUP_SPEC` = this model's spec,
`BRINGUP_SERVER_REPO` = `/localdev/$USER/tt-d-gen` (needs the engine build, `agents/dgen-build.md`). Pass settings as
`start.sh VAR=val ...`.

Before it opens the port, the server answers the spec's smoke prompt ("Paris", greedy) through the engine. If the
runner or the daemon dies, or a step stalls, the whole stack stops (logs below).

Env: `XING_SERVE_PORT` (8000), `XING_SERVE_SLOTS` (4: engine `max_slots` = `PREFILL_NUM_USERS`),
`XING_SERVE_POOLS` (`web:8,hammer:8`), `XING_SERVE_MAX_TOKENS` (1024), `XING_SERVE_TEMPERATURE` / `XING_SERVE_TOP_P` (0 / 1: greedy unless the request
says otherwise), `XING_SERVE_THINKING` (0), `XING_SERVE_HEARTBEAT_S` (60), `XING_SERVE_STEP_S` (600: one step's
bound), `XING_SERVE_DIR` (`generated/xing_serve`).

## Web chat

`GET /` serves `chat.html`: a plain streaming chat (no tools) with telemetry under every answer and a live engine
panel. From your machine: `ssh -N -L 8000:localhost:8000 <box>`, then open http://localhost:8000/.

For others without SSH: `python models/demos/xing40_a4b_d_p/serve/forward.py` copies container port 5555 to the
server's port. On the bh-lb-17 reservation container the host publishes 54210 (`P_USER_DBD_PORT`) to container port
5555, so the chat is then at http://bh-lb-17:54210/ for anyone who can reach the host (no auth). Stop it with
`kill $(cat generated/xing_serve/forward.pid)`.

- Under each answer, **from tt-d-gen** (its ADMITTED events): for your prompt, cache hit or miss, tokens reused from
  a cached prefix, tokens computed, slot; for the generation, how many of its re-admits hit the cache and the
  average reuse per step. A bar per step: height = step time, green = prefix hit, amber = cold.
- **Measured by the server**: tok/s, s/token (the prefill role has no token counter of its own).
- Side panel: tt-d-gen's `TelemetrySnapshot` every 2 s (slots free / idle-resident, hits / admitted, reused vs
  prompt tokens, evictions, KV blocks running / cached / free, chunk latency), plus the pools and our check counters.

Pools: requests carry `X-Xing-Pool` (`web`, the default, or `hammer`, which `hammer.py` sends);
`XING_SERVE_POOLS` (default `web:8,hammer:8`) caps each pool's concurrent generations. The engine still picks slots
and evicts across all of them, so an idle chat can lose its cached prefix to the hammer (its next turn goes cold).

## Use it

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
curl -s localhost:8000/v1/completions -d '{"prompt":"The capital of France is","max_tokens":8}'   # raw, no template
curl -s localhost:8000/stats      # per-step checks, step latency, engine telemetry
```

From your machine: `ssh -N -L 8000:localhost:8000 <box>`. Responses carry `token_ids` and a `timings` object (steps,
s/token, the slots used, the first step's resident prefix).

## Hammer

```bash
python models/demos/xing40_a4b_d_p/serve/hammer.py --concurrency 3 --duration-min 60 --long 52000
```

Greedy cases: one-turn facts, a two-turn memory check, the same prompt twice (token ids must match), and a long
needle-in-a-haystack prompt (several chunks; past 51200 tokens also the pulled-back last chunk). Short cases start
with a ~150-token system turn so every step remounts; `--no-system` sends them cold. Exit 1 on any failure.

## What the server checks on every step

Counted in `/stats` (`check_fail` must stay 0; failures logged as `CHECK FAIL` in `server.log`):

- `PREFILL_DONE` at exactly the prompt length (daemon), no REJECTED / ABORTED (`engine_errors`)
- the engine's resident prefix: a multiple of 32 and at most the reusable cap `(n - 1) // 64 * 64`
- a logits file for the last chunk of the server plan for `(n, resident)`, written after the admit
- finite logits
- the idle heartbeat (also keeps the runner's 180 s dispatch timeout from firing on its wait for input): its greedy
  token never changes (`heartbeat_mismatch`)

## Logs (`generated/xing_serve/`)

`server.log` (front end + run_safe_pytest), `runner.log` (one line per chunk: slot, range, greedy next token,
logits time), `engine.log` (daemon), `stats.json` (every 30 s), `engine_status.json` (every 5 s).

## Limits

- About 1 s per token at short context (0.6 s chunk + about 0.7 s for logits: the whole chunk's final hidden comes
  to the host, then the LM head runs in fp32 on the CPU). Under 64 tokens there is no reusable prefix block, so
  each step goes in cold and may land on any slot.
- The engine's own telemetry and prefix index see one "prompt" per token; its hit counters are meaningful, its
  token totals are not throughput.
- No repetition penalty; `temperature` / `top_p` / `seed` / `stop` / `max_tokens` work.
