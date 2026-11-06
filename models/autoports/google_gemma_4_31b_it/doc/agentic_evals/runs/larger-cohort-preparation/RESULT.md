# Deterministic larger cohorts ready

Original selection reproduced exactly. Frozen SWE nightly5 and Terminal nightly10 are exact prefixes of the regenerated order. Candidates are prepared only; central cohorts and rung eligibility remain unchanged.

- SWE:10,20,40,80,160,320,500;500 unique catalog entries, full500 exactly covers catalog.
- Terminal:20,40,80,89;89 unique catalog entries, full89 exactly covers catalog.

All sizes are nested prefixes of a single score-independent order. Ordering is SHA256(UTF8(seed + NUL + name)) for both strata and tasks, followed by round-robin over nonempty strata. Seed is unchanged from cohorts.json. Original SWE stratum is the task owner prefix before `__` (named repository in original evidence), not owner/repo; Terminal uses task.toml metadata.category and full terminal-bench/name identifiers. No scores or transcripts are read by prepare.py.

Pinned SWE source is Harbor registry.json@e76c762fca5267dfb36e1f900f0a7724ed72e735, swebench-verified1.0, with all500 tasks pointing to harbor-datasets@86723674f04e4209ac479d0fb75d9d9f44b4377e. Parsed registry exactly matches original stored catalog. Terminal source is harbor-framework/terminal-bench-2-1@12b7565679ce5006230b820666353673a133db93. Both task metadata checkouts are clean at exact commits. Only task.toml definitions were read; no solution access or network fetch needed.

Every candidate task has original CPU/memory values captured with task-definition SHA256 in task_resources.json. SWE all500 specify1CPU/4G. Terminal68 specify1CPU/2048MB,10 specify1CPU/4096MB,5 specify1CPU/8192MB,3 specify4CPU/8192MB,3 specify2CPU/4096MB. Keep per-task resources; no blanket override or host-capacity claim follows. Original agent/verifier definition timeouts are recorded for provenance, not replacements for stage release limits.

Reproduction command:

`/home/mvasiljevic/gemma4-31b-pipeline-v3/tt-inference-server-stage12/.pre-commit/bin/python prepare.py --execution /home/mvasiljevic/tt-metal-gemma4-31b-pipeline --metadata-root /home/mvasiljevic/gemma4-31b-pipeline-v3`

Assertions passed for pinned clean sources, original-prefix equality, full counts, uniqueness, every nested prefix, exact full coverage, CPU/memory presence and unchanged central cohorts bytes. Output candidates.json includes all lists/source/rule; task_resources.json maps every task to its definition/resources. No tests of model behavior, source publication, dispatch or devices; predicted/actual device hours0 and no speed credit.
