# Per-Architecture Profiles for Issue-Solver

The issue-solver orchestrator uses this table to resolve arch-specific values from `TARGET_ARCH`.

## Profile table

| Field | blackhole | quasar | wormhole |
|---|---|---|---|
| `ARCH` | `blackhole` | `quasar` | `wormhole` |
| `LLK_DIR` | `tt_llk_blackhole` | `tt_llk_quasar` | `tt_llk_wormhole_b0` |
| `TESTS_DIR` | `tests/python_tests/blackhole` | `tests/python_tests/quasar` | `tests/python_tests/wormhole` |
| `SIM_PORT` | `5555` | `5556` | `5557` |
| `REF_ARCH` | `wormhole` | `blackhole` | *(none — wormhole is oldest)* |
| `LOGS_BASE` | `/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver` | `/proj_sw/user_dev/llk_code_gen/quasar_issue_solver` | `/proj_sw/user_dev/llk_code_gen/wormhole_issue_solver` |
| `CONFLUENCE_ROOT_PAGE` | `48300268` | `48300268` | `48300268` |
| `CONFLUENCE_SFPU_SPEC` | *(no dedicated page)* | `1256423592` | *(no dedicated page)* |
| `DASHBOARD_PROJECT_ID` | `blackhole_issue_solver` | `quasar_issue_solver` | `wormhole_issue_solver` |

## How the orchestrator consumes this

Set `TARGET_ARCH` from the top-level router. Look up each field by name in the table above and substitute into agent prompts and bash commands (shown in `codegen/agents/issue-solver/orchestrator.md`). No other file hardcodes arch-specific values.
