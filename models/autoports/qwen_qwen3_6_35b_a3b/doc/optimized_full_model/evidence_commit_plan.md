# Minimal evidence and local commit plan

Planning only: no staging, commits, evidence conversion or source edits performed.
The installed [tt-device-usage skill](/home/ttuser/qwen36-35b-a3b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/tt-device-usage/SKILL.md:50)
says: “Evidence is the logs plus one manifest. Do not compress, hash or archive
individual artifacts.” Preserve already-generated provenance as historical
records; do not create additional per-artifact checksum/archive machinery.

1. Keep complete raw logs, profiler CSVs, failure captures and prompt/completion
   files in their existing persistent stage locations. One stage evidence
   manifest should link each gate/candidate to its command, result, exact scope,
   raw paths, exit status and relevant source/commit identity. Reference existing
   provenance directly. Do not copy raw evidence into a new archive or normalize
   its contents to satisfy formatting hooks.
2. Locally commit stage-owned implementation/tests and concise stage documents,
   manifest, compact structured results and required perf tables/CSV. Use an
   explicit path list; inspect it against unrelated checkout changes. Large raw
   reports/logs remain at manifest-listed persistent paths. The current hook
   rejects files above500KiB; compression to evade this limit is not appropriate.
   Keep review evidence readable and raw files recoverable in the workspace.
3. Run normal hooks on the selected source/docs/results **before** final staging.
   Do not include raw rendered prompts or completions in a batch that runs
   `trailing-whitespace` or `end-of-file-fixer`: these hooks alter model output
   bytes. Current examples without a final newline include
   `final/aime_autoregressive/hf_completion.txt` and
   `final/qualitative/q01/tt_completion.txt`. Adding one would change evidence.
   Leave these raw text files outside the commit and reference them in the
   manifest; existing suite JSON preserves exact completion strings and token
   IDs. Do not globally disable hooks or silently stage hook-modified raw output.
4. If repository-contained raw text is specifically required at review, first
   adopt an explicit narrowly scoped preservation exception for those exact
   evidence paths, with the reason documented, and run the remaining hooks.
   That would be a separate reviewable repo-config change; this plan does not
   make it. No exception is needed merely to retain raw files on persistent disk.
5. After required gates and stage review finish, commit only the reviewed
   stage-owned paths locally, record each SHA in the parent's work log, and
   never push. A code/docs commit and a subsequent compact final-evidence commit
   are sufficient if validation finishes later. Do not mistake a commit for
   acceptance of pending perf/profile/context/replay/serving or review gates.

Most `.log` files are already ignored by the checkout. Their omission from a git
commit must not be mistaken for permission to delete them. The single manifest
must identify that persistence boundary clearly.
