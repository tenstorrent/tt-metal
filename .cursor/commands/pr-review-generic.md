# pr-review-generic

Do a comprehensive review of this branch as a PR into 'origin/main'. Identify files with `git diff origin/main --name-only`.
**Do not consider this branch's history or untracked files in your evaluation.**
Use `git diff --histogram origin/main` as the diff invocation.
Evaluate for best practices, security, correctness, completeness, readability, maintainability, organization, test coverage, and documentation.
You may dispatch subagents or swarm to help you.
If a file touched by this branch contains thread-safe items such as std::mutex or std::atomic, additionally evaluate for thread-safety, race conditions, and deadlocks. Prefer std::atomic_flag over std::atomic<bool>.
If a file touched by this branch uses MPI, consider that there may be operations running on the same or different hosts for purposes of synchronization. Multithreading, multiprocess, and multihost are all possible.
C++: For functions which were changed by this branch, evaluate for any Undefined Behavior and include mitigations as a part of the plan. Prefer std::filesystem::path over std::string for file path operations. Target C++20.
Python: Prefer pathlib.Path over string and os.* operations where possible, targeting Python 3.10.
Bash: Use the bash-safety skill.
System: Primary target is Linux x86_64, but ARM and RISC-V are also potential targets.
Produce the review in a Plan markdown document which includes implementation details for actionable feedback in the 'to-dos' section.
If there are any 'to-dos', append an instruction at the end of the 'to-dos' to update the Plan document with implementation progress (to account for some subagents that may not update the plan without instruction).
The first section of this Plan markdown document should be concise high-level summary of what this branch changes in a format appropriate for filing this PR.
If the review covers architectural changes, create a mermaid format flowchart showing the architectural change. If two mermaid subgraphs are unrelated or unconnected, you may generate separate ```mermaid ``` sections as appropriate instead of grouping them all together for the purposes of readability.
If this branch changes a README markdown file, you may suggest a context-relevant mermaid format flowchart if the edited section of the README could have improved clarity.
