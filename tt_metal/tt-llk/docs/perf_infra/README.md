# LLK perf infrastructure

## The two rules

Break either one and something fails. They cover most of what you need to know.

| # | Rule | If you skip it |
|---|---|---|
| 1 | A new CSV column must exist in `DB_SCHEMA` (`helpers/perf/wide_schema.py`). | The Parquet writer raises `PerfSchemaError` at the end of the run. |
| 2 | A change to a perf test's columns must be recorded in `helpers/perf/test_schemas.py`, with `version` increased. | `test_perf_header_gate.py` fails. |

## I want to…

| Your task | Go to |
|---|---|
| Fix a gate that just failed | [tasks.md §1](tasks.md#1-a-gate-failed-find-the-fix) |
| Add a sweep parameter to a perf test | [tasks.md §2](tasks.md#2-add-a-sweep-parameter-to-a-perf-test) |
| Add a new parameter class | [tasks.md §3](tasks.md#3-add-a-new-parameter-class) |
| Add a new perf test | [tasks.md §4](tasks.md#4-add-a-new-perf-test) |
| Rename a perf test | [tasks.md §5](tasks.md#5-rename-a-perf-test) |
| Delete or absorb a perf test | [tasks.md §6](tasks.md#6-delete-or-absorb-a-perf-test) |
| Rename a column | [tasks.md §7](tasks.md#7-rename-a-column) |
| Add a run type or an efficiency metric | [tasks.md §8](tasks.md#8-add-a-run-type-or-an-efficiency-metric) |
| Produce a perf report | [tasks.md §9](tasks.md#9-produce-a-perf-report) |
| Find out whether my branch regressed perf | [tasks.md §10](tasks.md#10-find-out-whether-my-branch-regressed-perf) |
| Find out what regressed in nightly, and who caused it | [tasks.md §11](tasks.md#11-find-out-what-regressed-in-nightly) |
| Read a run as a table, or query history | [tasks.md §12](tasks.md#12-read-a-run-as-a-table) |
| Avoid a mistake people keep making | [pitfalls.md](pitfalls.md) |
| Understand how a part works | [reference.md](reference.md) |

## The four documents

| Document | Contents |
|---|---|
| [tasks.md](tasks.md) | One procedure per task. Start here. |
| [pitfalls.md](pitfalls.md) | Mistakes that pass the gates, or fail somewhere far from the cause. |
| [reference.md](reference.md) | How each part works. Context for engineers and agents. |
| README.md | This index. |

All commands run from `tt_metal/tt-llk/tests/python_tests`. No gate needs hardware.
