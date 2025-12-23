# Refactoring Process Workflow

This document explains the high-level process flow for refactoring the `register_operation` calls in the `tt-metal` repository.

## Overview

The goal is to iteratively refactor all device operations identified in the `device_ops_to_process.txt` file. For each operation, the agent follows the steps outlined in the `REFACTORING_GUIDE.md`.

## Workflow Steps

### 1. Identify the Next Task
The agent reads `device_ops_to_process.txt` to find the next entry marked as `[ ]` (not started).

### 2. Mark as In Progress
The agent updates `device_ops_to_process.txt` to change the status of the chosen entry to `[/]` (in progress) to prevent duplicate work and track current activity.

### 3. Analyze and Implement
Following the `REFACTORING_GUIDE.md`, the agent:
- Extracts the operation name and class from the `.hpp` file.
- Identifies the `invoke` function arguments.
- Replaces the `register_operation` call with a direct function declaration in the `.hpp` file.
- Implements the new function in the corresponding `.cpp` file, mapping arguments to `operation_attributes_t` and `tensor_args_t` and calling `launch_on_device`.

### 4. Build and Verify
The agent runs the verification build:
```bash
./build_metal.sh -c -e --debug
```

### 5. Finalize the Task
- **If the build succeeds:** The agent marks the entry as `[x]` (done) in `device_ops_to_process.txt`.
- **If the build fails:** The agent analyzes the errors, fixes the implementation, and repeats Step 4.

## Tracking Progress

The `device_ops_to_process.txt` file serves as the source of truth for the entire refactoring project:
- `[ ]` Pending
- `[/]` In Progress
- `[x]` Done

This cycle repeats until all 207 entries in the list are marked as done.
