# Per-Architecture Profiles for Issue-Solver

The issue-solver orchestrators use this table to resolve arch-specific values from their target arch(es):
- `codegen/agents/issue-solver/orchestrator.md` — single-arch, reads the column for `TARGET_ARCH`
- `codegen/agents/issue-solver/orchestrator-multi.md` — multi-arch, reads one column per entry in `TARGET_ARCHES`, creates one dashboard run, and passes the ordered arch list to the shared analyzer/fixer/tester

## Profile table

| Field | blackhole | quasar | wormhole |
|---|---|---|---|
| `ARCH` | `blackhole` | `quasar` | `wormhole` |
| `LLK_DIR` | `tt_llk_blackhole` | `tt_llk_quasar` | `tt_llk_wormhole_b0` |
| `TESTS_DIR` | `tests/python_tests/blackhole` | `tests/python_tests/quasar` | `tests/python_tests/wormhole` |
| `SIM_PORT` | *(n/a — hardware only)* | `5556` | *(n/a — hardware only)* |
| `REF_ARCH` | `wormhole` | `blackhole` | *(none — wormhole is oldest)* |
| `REF_LLK_DIR` | `tt_llk_wormhole_b0` | `tt_llk_blackhole` | *(none)* |
| `LOGS_BASE` | `${CODEGEN_LOGS_ROOT}/blackhole_issue_solver` | `${CODEGEN_LOGS_ROOT}/quasar_issue_solver` | `${CODEGEN_LOGS_ROOT}/wormhole_issue_solver` |
| `MULTI_LOGS_BASE` | `${CODEGEN_LOGS_ROOT}/issue_solver` | `${CODEGEN_LOGS_ROOT}/issue_solver` | `${CODEGEN_LOGS_ROOT}/issue_solver` |
| `CONFLUENCE_ROOT_PAGE` | `48300268` | `48300268` | `48300268` |
| `CONFLUENCE_SFPU_SPEC` | *(no dedicated page)* | `1256423592` | *(no dedicated page)* |
| `DASHBOARD_PROJECT_ID` | `blackhole_issue_solver` | `quasar_issue_solver` | `wormhole_issue_solver` |
| `MULTI_DASHBOARD_PROJECT_ID` | `issue_solver` | `issue_solver` | `issue_solver` |

`CODEGEN_LOGS_ROOT` is the log root the orchestrators resolve in Step 0, by
precedence: (1) an explicit `CODEGEN_LOGS_ROOT` env value wins; (2) else the
shared dashboard tree `/proj_sw/user_dev/llk_code_gen` **if that path exists**
(so the dashboard is fed automatically wherever the NFS location is mounted);
(3) else an **in-repo, gitignored** folder in the main checkout —
`<main_repo>/tt_metal/tt-llk/codegen/logs` (resolved via `git rev-parse
--git-common-dir` so it lands in the main repo, not the removed worktree). The
`*_issue_solver` / `issue_solver` suffixes above are always appended, so the
dashboard folder shape is preserved in every case.

`SIM_PORT` is only populated for Quasar (the lone sim carve-out). Blackhole and Wormhole run on the locally-attached card — no simulator, no `--run-simulator`. Hosts without a matching BH/WH card finalize the run as `failed` with `ENV_ERROR`. Quasar has no silicon, so it always runs on `emu-quasar-1x3` with port 5556. On every arch, `.claude/scripts/run_test.sh` serialises the consumer step internally via the single global lock `/tmp/tt-llk-test.lock` — agents never flock manually.

## How the orchestrators consume this

**Single-arch** (`orchestrator.md`): set `TARGET_ARCH` from the top-level router, look up each field by name, substitute into agent prompts and bash commands. No other file hardcodes arch-specific values.

**Multi-arch** (`orchestrator-multi.md`): iterate over `TARGET_ARCHES` and build per-arch bash associative maps (`LLK_DIR_OF`, `TESTS_DIR_OF`, `REF_ARCH_OF`, `REF_LLK_DIR_OF`, and `SIM_PORT_OF` — populated for Quasar only) from the same table. The run itself uses the shared `MULTI_LOGS_BASE` and `arch="multi"`; per-arch results are recorded inside one `arch_results` field instead of creating per-arch sibling runs.

`REF_LLK_DIR` is the `LLK_DIR` value of the reference architecture (i.e., the arch whose existing code is used as a porting reference); it's listed here as an explicit row rather than derived at runtime so the table stays the single source of truth.
