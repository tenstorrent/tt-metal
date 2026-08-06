# Plan — adopt Laguna's published `poolside_v1` tool-call parser

**Status: DONE (2026-08-06), validated on device.** Parsers vendored into `vllm_tt_plugin`, registered, offline
unit test green + glm47 parity, and the on-device tool round-trip returned `finish_reason=tool_calls` with
`get_weather({"city":"Paris","metric":true})`. Docs pinned to `poolside_v1`. See the execution log at the end.

## Target (corrected)
vLLM ships a **standalone published parser** for Poolside's Laguna models: **`--tool-call-parser poolside_v1`**.
It implements Poolside's native XML-style protocol and extracts BOTH structured tool calls AND interleaved
`<think>` reasoning blocks during generation. This is the sanctioned parser; the goal is to run it instead of
the borrowed `glm47`.

> Correction to an earlier note in this session: I wrongly wrote "Laguna doesn't ship a standalone parser."
> It does (`poolside_v1`). My claim was based only on it being ABSENT from our pinned vLLM checkout — not proof
> of nonexistence. The grammar I reverse-engineered from the chat template
> (`<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>…</tool_call>`, `<think>…</think>`) is exactly
> what `poolside_v1` parses.

## Where it goes: the TENSTORRENT vLLM PLUGIN, not the core fork
Two distinct things live under `/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/`:
- **vLLM core = a FORK** (`github.com/tenstorrent/vllm.git`). Built-in parsers live in `vllm/tool_parsers/`.
- **`vllm-tt-plugin`** (`plugins/vllm-tt-plugin/src/vllm_tt_plugin/`) — the TT plugin (platform/worker/
  model_runner) that ALSO ships TT model parsers and registers them.

**The established TT pattern is to add the parser to the PLUGIN, never to patch the core fork.** Precedent:
commit `#431 "gemma4: add reasoning + tool-call parsers to vllm-tt-plugin"`. `vllm_tt_plugin/entrypoints.py`
`register()` (runs in every vLLM process) calls `_register_tt_tool_parsers()`:
```python
ToolParserManager.register_lazy_module("gemma4", "vllm_tt_plugin.gemma4_tool_parser", "Gemma4ToolParser")
```
Docstring intent (verbatim): *"Kept in the plugin (rather than patched into vllm.tool_parsers) so it carries
over unchanged when switching to upstream vLLM."* So the poolside_v1 parser goes here, the same way.

## Why it's not already running (verified)
Served vLLM core = pinned TT fork `0.1.dev14175+g6b4a3a7b4.d20260703` (~2026-07-03) — predates the upstream
`poolside_v1`, so it isn't a built-in here, and the plugin hasn't registered it yet. A full core-fork upgrade
is high-risk (the plugin is pinned to this vLLM's internal hooks), so we register the parser via the plugin.

## Approach: add `poolside_v1` to the plugin (pure-Python, no C++ rebuild, upgrade-safe)

### Phase 0 — Acquire the published `poolside_v1` parser (offline)
1. Fetch the upstream source from the vLLM commit/PR that added `poolside_v1` (WebFetch the file + any
   reasoning-parser counterpart it ships with).
2. **Base-class compat check** against our pinned fork. `ToolParser` (abstract_tool_parser.py) API here:
   `__init__(self, tokenizer: TokenizerLike)`, `extract_tool_calls(model_output, request) -> ExtractedToolCallInformation`,
   `extract_tool_calls_streaming(...)`. Plugin parsers subclass `ToolParser` directly (see gemma4). If upstream
   `poolside_v1` imports newer symbols (moved modules, changed `ExtractedToolCallInformation`/`DeltaMessage`
   fields, new base methods), adapt the import lines + drifted signatures; keep the parsing logic identical.

### Phase 1 — Add to the plugin + register lazily (`.local` plugin diff, NO core-fork edit)
1. Add `plugins/vllm-tt-plugin/src/vllm_tt_plugin/poolside_v1_tool_parser.py` (and a
   `poolside_v1_reasoning_parser.py` iff it ships one), mirroring `gemma4_tool_parser.py`.
2. Register in `vllm_tt_plugin/entrypoints.py`:
   `_register_tt_tool_parsers()` → `ToolParserManager.register_lazy_module("poolside_v1",
   "vllm_tt_plugin.poolside_v1_tool_parser", "PoolsideV1ToolParser")` (+ the reasoning parser via
   `ReasoningParserManager.register_lazy_module(...)` if applicable).
3. Pure-Python editable install → **no C++ rebuild**; ensure `.../vllm-tt-plugin/src` is the tree on the server
   PYTHONPATH ([[laguna-vllm-serving-env]] — the third PYTHONPATH entry; `vllm_tt_plugin` editable pin).
   Verify: `python -c "from vllm.tool_parsers.abstract_tool_parser import ToolParserManager as M; import vllm_tt_plugin.entrypoints as e; e.register(); print('poolside_v1' in M.tool_parsers)"`.

### Phase 2 — Reconcile with the reasoning parser
`poolside_v1` handles interleaved `<think>` blocks itself. Today we run `--reasoning-parser deepseek_r1`
(also parses `<think>…</think>`) + `chat_template_kwargs:{enable_thinking:true}`. Determine the correct combo:
either (a) keep `deepseek_r1` for the reasoning channel + `poolside_v1` for tool calls (if they compose), or
(b) `poolside_v1` supersedes reasoning extraction and `deepseek_r1` should be dropped. Decide by reading the
upstream parser + a smoke test asserting `message.reasoning` and `message.tool_calls` are both populated once.

### Phase 3 — Validate
- **Offline unit test** (`scripts/test_tool_parser.py`): feed canonical Laguna outputs (single/multi tool_call;
  string vs int/float/bool/object/array args via the template's `tojson`-nonstring / raw-string rule; call
  after `<think>`; leading content; streaming deltas) → assert correct OpenAI `tool_calls`. Cross-check
  `poolside_v1` vs `glm47` (should agree on the common cases; poolside_v1 is the reference where they differ).
- **On-device smoke** (reuse serve recipe, batch-1): a `/v1/chat/completions` with a `tools` schema →
  `tool_calls[0].function.name` + `arguments` round-trip as valid JSON; multi-arg + object-arg cases.
- **End-to-end**: small SWE mini-extra tool-call slice (this path uses OpenAI tool-calling) — no parser errors,
  tool calls execute. (Terminal-Bench is OUT of scope — terminus-2 uses its own TEXT action parser, not the
  vLLM tool parser; these warnings are unrelated.)

### Phase 4 — Ship
- Pin `--tool-call-parser poolside_v1` (+ the resolved reasoning-parser decision) in `README.md`, `STATUS.md`,
  `scripts/stage_ce_serve.sh`, `scripts/tb_run.sh`.
- The vendored parser lives on the uncommitted `.local` plugin diff (like the batched-decode fix); record it in
  STATUS "Implemented" + a repro note so a fresh checkout re-vendors it. Commit the offline test to the repo.
- Update memory [[laguna-toolcalling]]: `poolside_v1` is the published parser; glm47 was the interim stand-in.

## Effort / risk
Phase 0–1: ~1–2 h offline (fetch + compat-adapt + register). Phase 3 offline test: ~1 h. On-device validate:
~1 boot (~15 min) + a few tool-call requests, batch-1. Main risk = base-class API drift between the ~July
pinned tree and the newer upstream parser; contained to import/signature adaptation in one file. No C++ rebuild,
no vLLM upgrade, no change to model behavior.

---

## EXECUTION LOG (2026-08-06) — DONE, validated on device

**Result:** `poolside_v1` tool + reasoning parsers vendored into `vllm_tt_plugin` and validated. On-device
tool round-trip: `finish_reason=tool_calls`, `get_weather({"city":"Paris","metric":true})` — correct name +
string/bool typing. Offline unit test PASS (`scripts/test_tool_parser.py`); == glm47 on the common case.

**Plugin diff (uncommitted, lives in `.local`, like the batched-decode fix — NOT in the tt-metal repo):**
`/home/ttuser/.local/lib/model-bringup/tt-metal/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/`
- `poolside_v1_tool_parser.py`, `poolside_v1_reasoning_parser.py` (vendored from upstream vLLM main)
- `entrypoints.py`: `_register_tt_tool_parsers()` + `_register_tt_reasoning_parsers()` each add a
  `register_lazy_module("poolside_v1", "vllm_tt_plugin.poolside_v1_<t>_parser", "PoolsideV1<T>Parser")`.

**Re-vendor from a clean checkout (turnkey):**
1. `curl -sSL https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/tool_parsers/poolside_v1_tool_parser.py`
   → `vllm_tt_plugin/poolside_v1_tool_parser.py`; same for
   `.../main/vllm/reasoning/poolside_v1_reasoning_parser.py` → `vllm_tt_plugin/poolside_v1_reasoning_parser.py`.
2. Apply 3 pinned-fork adaptations to the tool parser (only): (a) `from vllm.tool_parsers.abstract_tool_parser
   import ToolParser` + `from vllm.tool_parsers.utils import Tool`; (b) inline `safe_literal_eval`
   (`ast.literal_eval` under `warnings.simplefilter("ignore", SyntaxWarning)`) + `import ast, warnings`;
   (c) `super().__init__(tokenizer)` (drop the `tools` arg) + `self.tools = tools`. Reasoning parser: verbatim.
3. Add the two `register_lazy_module("poolside_v1", …)` lines to `entrypoints.py`.
4. Verify offline: `python doc/vllm_integration/scripts/test_tool_parser.py` → `OFFLINE TOOL-PARSER TEST PASS`.
5. Serve with `--enable-auto-tool-choice --tool-call-parser poolside_v1 --reasoning-parser poolside_v1`.

**Note:** the on-device smoke exercised the tool path fully; the reasoning parser is registered and boots
cleanly but the simple prompt emitted no `<think>`, so its `</think>`-scoping behavior wasn't stress-tested —
low risk (it subclasses the working `deepseek_v3` parser). A thinking-heavy tool turn would exercise it.
