## AI playbook: discover TTNN operation folders

This guide provides clear prompts and procedures for an AI agent (or a developer) to recursively discover TTNN "operation" folders under `ttnn/cpp/ttnn/operations`.

### Objective
- **Goal**: Find all directories under `ttnn/cpp/ttnn/operations` that represent a valid operation.
- **A directory is a valid operation folder if and only if** it contains:
  - **`device/`** subdirectory, and
  - Program creation files in that device folder

### Traversal and stop conditions
- **Recurse** through subdirectories starting at `ttnn/cpp/ttnn/operations`.
- Continue recursion on every folder until every folder and sub-folder is explored .

### Deliverables
- A list (JSON or newline-separated) of absolute or repo-root–relative paths to all valid operation directories.
