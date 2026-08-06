# Getting started with `pool` (Poolside's coding agent) on the TT Laguna server

`pool` is Poolside's terminal coding agent (github.com/poolsideai/pool). In **standalone mode** it talks to any
OpenAI-compatible endpoint — here, our vLLM server serving **Laguna-XS-2.1 on P150×4** with Laguna's published
`poolside_v1` tool-call + reasoning parsers. This guide gets you from zero to a working agent session.

---

## 1. Start the Laguna model server (the endpoint `pool` will use)

From `/tmp`, with the installed-tree env (see `README.md`), serve with the published parsers:

```bash
cd /tmp
export TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal
export PYTHONPATH=/home/ttuser/dev/tt-metal:$TT_METAL_HOME/vllm:$TT_METAL_HOME/vllm/plugins/vllm-tt-plugin/src
export TT_LAGUNA_PIPE_CHUNK=2048 TT_LAGUNA_PREFIX_CACHE=1 TT_LAGUNA_PREFILL_FAST=1

python -m models.common.readiness_check.run_vllm_server \
  --model-dir /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1 \
  --hf-model poolside/Laguna-XS-2.1 --mesh-device P150x4 --stages serve \
  --max-num-seqs 8 --block-size 64 --max-model-len 131072 \
  --tt-config '{"trace_region_size":1500000000,"fabric_config":"FABRIC_1D_RING","env_passthrough":["VLLM_*","MESH_DEVICE","TT_LAGUNA_*","TT_METAL_*","PYTHONPATH"]}' \
  --additional-server-args='--trust-remote-code --max-num-batched-tokens 131072 --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1'
```

Boot ≈ 10–15 min (MoE load → warmup). Ready when:
```bash
curl -s http://localhost:8000/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])'   # -> poolside/Laguna-XS-2.1
```
Recovery after a hard kill: `tt-smi -r all`, and truncate `readiness_vllm/server.log` before relaunch.

## 2. Install `pool`

```bash
curl -fsSL https://downloads.poolside.ai/pool/install.sh | sh      # accept the EULA prompt (https://poolside.ai/eula)
export PATH="$HOME/.local/bin:$PATH"                               # installer target; add to your shell rc
pool --version                                                     # -> 1.0.15
```
(Headless/CI: prefix with `POOL_INSTALL_ACCEPT_EULA=1` to accept non-interactively.)

## 3. Point `pool` at the Laguna server (standalone mode)

**`POOLSIDE_STANDALONE_BASE_URL` is the mode switch.** Setting it puts pool in *standalone* mode (local
OpenAI-compatible endpoint). Without it, pool stays in *tenant* mode and tries to fetch a cloud agent — you'll
see `failed to find default agent: 404`. The `exec --api-url` flag alone is **not** enough; set the env var.

```bash
export POOLSIDE_STANDALONE_BASE_URL=http://localhost:8000/v1     # <-- the switch (note the /v1)
export POOLSIDE_API_KEY=EMPTY                                    # any value; the local server ignores it
export POOLSIDE_STANDALONE_MODEL="poolside/Laguna-XS-2.1"        # set explicitly (skips model auto-listing)
export POOLSIDE_STANDALONE_CONTEXT_LENGTH=131072
```
Put these in your shell rc (or a `pool-laguna.env` you `source`) so every session picks them up.

### Interactive session (TUI)
```bash
cd /path/to/your/project
pool                       # opens the ACP client against the standalone endpoint above
```

### Non-interactive (scripts / one-shot tasks) — VERIFIED
```bash
cd /path/to/your/project
pool exec --unsafe-auto-allow --sandbox disabled \
  -p "Add a docstring to the top-level function in main.py and run the tests."
```
- `--unsafe-auto-allow` auto-approves tool actions (required when there's no TTY to confirm). This works in
  standalone mode; in tenant mode it needs the `auto-approve-commands` permission from an admin.
- `--sandbox disabled|required` controls sandboxing; `-o json` gives machine-readable NLJSON output.
- Exit codes: `0` success, `4` task-failure, anything else = unexpected error.

> Verified 2026-08-06 on this deployment: `pool exec` drove Laguna to emit a `poolside_v1` tool call (`write`),
> executed it, created the file, and returned `exit(success:true)` — the full model → tool-call → execution loop.

## 4. Notes / gotchas specific to this deployment

- **Tool-calling is automatic** via `poolside_v1` — no extra flags; `pool`'s tool calls round-trip through the
  server's parser. Reasoning (`<think>`) is split by the `poolside_v1` reasoning parser.
- **The model is throughput-limited on TT (~25 t/s/u decode).** Agentic tasks that need many steps are SLOW —
  each turn re-prefills the growing transcript. Prefer focused prompts; expect minutes per task, not seconds.
- **Run one `pool` session at a time.** The server is fine at `--max-num-seqs 8`, but concurrent long agent
  decode is unstable on this stack — keep it to a single active agent.
- **API key** is ignored by the local server but `pool` still requires one to be set (`POOLSIDE_API_KEY=EMPTY`).
- Config/logs: `~/.config/poolside/settings.yaml`, logs under `~/.local/state/poolside/pool/logs`.
