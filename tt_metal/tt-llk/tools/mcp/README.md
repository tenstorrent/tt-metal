# llk-tools MCP server

Gives AI assistants (Claude Code, Cursor, any MCP client) LLK tools that run on the
machine with the card. Use it instead of a PR gate to check a change for perf
regressions.

| tool | what it does |
|---|---|
| `list_perf_tests` | the perf test modules that `perf_compare` can measure |
| `perf_compare` | one perf test, two commits, on this card: verdict, report, log tail. Wraps `.claude/scripts/perf_compare_commits.sh`, so it uses the same rule as the gates (`perf/regression_compare.py`) |

## Set up

1. In your LLK Python environment: `pip install mcp` (1.x or 2.x).
2. Start the assistant from `tt_metal/tt-llk`. `.mcp.json` (Claude Code) and
   `.cursor/mcp.json` (Cursor) already list the server as `llk-tools`; approve it
   when asked.

A full comparison takes from minutes to about an hour. If your client stops waiting
first, raise its tool timeout (Claude Code: `MCP_TOOL_TIMEOUT`, in ms), or ask for a
`dry_run` first and run fewer `iterations`. Runs are cached per commit, so a repeated
or interrupted comparison only measures what is missing.

Ask in plain words, for example "check my branch for perf regressions in
perf_math_matmul on blackhole".
