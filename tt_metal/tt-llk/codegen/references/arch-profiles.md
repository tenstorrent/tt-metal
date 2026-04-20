# Per-Architecture Profiles for Issue-Solver

The issue-solver orchestrators use this table to resolve arch-specific values from their target arch(es):
- `codegen/agents/issue-solver/orchestrator.md` — single-arch, reads the column for `TARGET_ARCH`
- `codegen/agents/issue-solver/orchestrator-multi.md` — multi-arch, reads one column per entry in `TARGET_ARCHES` and builds per-arch shell associative maps (`LLK_DIR_OF[$arch]`, etc.)

## Profile table

| Field | blackhole | quasar | wormhole |
|---|---|---|---|
| `ARCH` | `blackhole` | `quasar` | `wormhole` |
| `LLK_DIR` | `tt_llk_blackhole` | `tt_llk_quasar` | `tt_llk_wormhole_b0` |
| `TESTS_DIR` | `tests/python_tests/blackhole` | `tests/python_tests/quasar` | `tests/python_tests/wormhole` |
| `SIM_PORT` | *(n/a — hardware only)* | `5556` | *(n/a — hardware only)* |
| `REF_ARCH` | `wormhole` | `blackhole` | *(none — wormhole is oldest)* |
| `REF_LLK_DIR` | `tt_llk_wormhole_b0` | `tt_llk_blackhole` | *(none)* |
| `LOGS_BASE` | `/proj_sw/user_dev/llk_code_gen/blackhole_issue_solver` | `/proj_sw/user_dev/llk_code_gen/quasar_issue_solver` | `/proj_sw/user_dev/llk_code_gen/wormhole_issue_solver` |
| `CONFLUENCE_ROOT_PAGE` | `48300268` | `48300268` | `48300268` |
| `CONFLUENCE_SFPU_SPEC` | *(no dedicated page)* | `1256423592` | *(no dedicated page)* |
| `DASHBOARD_PROJECT_ID` | `blackhole_issue_solver` | `quasar_issue_solver` | `wormhole_issue_solver` |

`SIM_PORT` is only populated for Quasar (the lone sim carve-out). Blackhole and Wormhole run on the locally-attached card via `run_hw` in `tester.md` — no simulator, no `--run-simulator`, no `flock`. Hosts without a matching BH/WH card finalize the run as `failed` with `ENV_ERROR`. Quasar has no silicon, so it always runs on `emu-quasar-1x3` with port 5556, flock-serialized on `/tmp/tt-llk-test-simulator.lock` — same sim path the Quasar kernel-gen playbooks have used since day one.

## How the orchestrators consume this

**Single-arch** (`orchestrator.md`): set `TARGET_ARCH` from the top-level router, look up each field by name, substitute into agent prompts and bash commands. No other file hardcodes arch-specific values.

**Multi-arch** (`orchestrator-multi.md`): iterate over `TARGET_ARCHES` and build per-arch bash associative maps (`LLK_DIR_OF`, `TESTS_DIR_OF`, `REF_ARCH_OF`, `REF_LLK_DIR_OF`, `LOGS_BASE_OF`, `DASHBOARD_ID_OF`, and `SIM_PORT_OF` — populated for Quasar only) from the same table. Per-arch subagent prompts index these maps by arch name. See the `Step 0a` section of `orchestrator-multi.md` for the canonical pattern.

`REF_LLK_DIR` is the `LLK_DIR` value of the reference architecture (i.e., the arch whose existing code is used as a porting reference); it's listed here as an explicit row rather than derived at runtime so the table stays the single source of truth.
