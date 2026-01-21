#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
AI Agent Test TODO Completer

Uses Claude Sonnet to automatically complete TODO sections in generated test files.
Each agent works on one operation independently.

Usage:
    python complete_test_todos.py --operation eltwise_binary
    python complete_test_todos.py --all  # Run all operations
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Try to import anthropic (check at runtime, not import time)
ANTHROPIC_AVAILABLE = False


def check_anthropic():
    """Check if anthropic package is available."""
    global ANTHROPIC_AVAILABLE
    try:
        import anthropic

        ANTHROPIC_AVAILABLE = True
        return True
    except ImportError:
        ANTHROPIC_AVAILABLE = False
        return False


# ===========================================================================
# Configuration
# ===========================================================================

OPERATIONS = {
    "eltwise_binary": {
        "test_file": "test_eltwise_binary_block.cpp",
        "operations": ["add", "sub", "mul"],
        "priority": 1,
    },
    "broadcast": {
        "test_file": "test_broadcast_block.cpp",
        "operations": ["add_tiles_bcast", "sub_tiles_bcast", "mul_tiles_bcast"],
        "priority": 2,
    },
    "transpose": {
        "test_file": "test_transpose_block.cpp",
        "operations": ["transpose_wh"],
        "priority": 3,
    },
    "reduce": {
        "test_file": "test_reduce_block.cpp",
        "operations": ["reduce"],
        "priority": 2,
    },
    "pack": {
        "test_file": "test_pack_block.cpp",
        "operations": ["pack"],
        "priority": 3,
    },
}

# ===========================================================================
# AI Agent
# ===========================================================================


class TestCompletionAgent:
    """AI agent that completes TODO sections in test files."""

    def __init__(self):
        """Initialize the AI agent with credentials from environment."""
        # Import here after checking availability
        from anthropic import Anthropic as AnthropicClient

        # Get credentials from environment (set in ~/.bashrc)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        model = os.environ.get("ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-20250514")

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")

        print(f"[INFO] Using model: {model}")
        if base_url:
            print(f"[INFO] Using base URL: {base_url}")
            self.client = AnthropicClient(api_key=api_key, base_url=base_url)
        else:
            self.client = AnthropicClient(api_key=api_key)

        self.model = model

    def complete_test_todo(self, test_file_path: Path, operation: str, context_files: dict) -> str:
        """
        Use AI to complete TODO sections in a test file.

        Args:
            test_file_path: Path to test file with TODOs
            operation: Operation name (e.g., "eltwise_binary")
            context_files: Dict of context file paths to provide to agent

        Returns:
            Completed test file content
        """
        # Read the test file
        test_content = test_file_path.read_text()

        # Check if already completed
        if "GTEST_SKIP" not in test_content and "TODO" not in test_content:
            print(f"[INFO] Test file already completed: {test_file_path.name}")
            return test_content

        # Build context
        context = self._build_context(context_files, operation)

        # Create prompt
        prompt = f"""You are an expert C++ test engineer working on tt-metal Compute API tests.

## Your Task

Complete the TODO section in the test file below. The test validates that block variant functions
produce IDENTICAL results to tile-by-tile processing.

## Test File to Complete

```cpp
{test_content}
```

## What You Need to Implement

Replace the TODO section with actual test implementation that:

1. **Creates two programs**:
   - `program_ref`: Uses tile-by-tile kernel (kernels/compute_*_tiles.cpp)
   - `program_test`: Uses block operation kernel (kernels/compute_*_block.cpp)

2. **Sets up buffers and circular buffers**:
   - Input buffers in DRAM
   - Circular buffers for both programs (CB 0, 1, 2)
   - Output buffers for results

3. **Creates kernels**:
   - Reference kernel using tile-by-tile operations
   - Test kernel using block operations
   - Both should have compile-time args: Ht, Wt, num_blocks

4. **Generates test data**:
   - Random bfloat16 input data
   - Write to device buffers

5. **Runs both programs and compares**:
   - Execute both programs
   - Read results
   - Compare with PCC >= 0.9999
   - Also validate against golden reference

## Context and Examples

{context}

## Requirements

- Use existing tt-metal test infrastructure (CommandQueueFixture, CreateProgram, etc.)
- Follow the pattern from existing tests in tests/tt_metal/tt_metal/test_eltwise_binary.cpp
- Ensure PCC >= 0.9999 for validation
- Remove GTEST_SKIP() line when implementation is complete
- Keep all existing test cases (15 block sizes)

## Output Format

Provide the COMPLETE test file with TODO sections filled in. Include all necessary:
- Headers and includes
- Helper functions if needed
- Complete run_*_block_test() implementation
- All test cases (keep existing TEST_F declarations)

Output ONLY the complete C++ file, no explanations before or after.
"""

        print(f"[INFO] Sending request to AI agent...")
        print(f"[INFO] This may take 1-2 minutes...")

        # Call AI
        try:
            response = self.client.messages.create(
                model=self.model, max_tokens=16000, messages=[{"role": "user", "content": prompt}]
            )

            completed_content = response.content[0].text

            # Extract code if wrapped in markdown
            if "```cpp" in completed_content:
                # Find the code block
                start = completed_content.find("```cpp") + 6
                end = completed_content.find("```", start)
                completed_content = completed_content[start:end].strip()
            elif "```" in completed_content:
                start = completed_content.find("```") + 3
                end = completed_content.find("```", start)
                completed_content = completed_content[start:end].strip()

            return completed_content

        except Exception as e:
            print(f"[ERROR] AI agent failed: {e}")
            raise

    def _build_context(self, context_files: dict, operation: str) -> str:
        """Build context string from relevant files."""
        context_parts = []

        # Add testing plan excerpt
        if "testing_plan" in context_files:
            plan_path = context_files["testing_plan"]
            if plan_path.exists():
                content = plan_path.read_text()
                # Extract relevant section for this operation
                context_parts.append(f"### From TESTING_PLAN.md\n\n{content[:3000]}...")

        # Add example test if available
        if "example_test" in context_files:
            example_path = context_files["example_test"]
            if example_path.exists():
                content = example_path.read_text()
                context_parts.append(f"### Example Test Structure\n\n```cpp\n{content[:2000]}...\n```")

        return "\n\n".join(context_parts)


# ===========================================================================
# Main Logic
# ===========================================================================


def complete_operation_tests(operation: str, repo_root: Path, test_output_dir: Path, dry_run: bool = False) -> bool:
    """
    Complete TODO sections for a specific operation.

    Args:
        operation: Operation name
        repo_root: Path to tt-metal repository
        test_output_dir: Path to generated test files
        dry_run: If True, don't actually write files

    Returns:
        True if successful
    """
    if operation not in OPERATIONS:
        print(f"[ERROR] Unknown operation: {operation}")
        return False

    config = OPERATIONS[operation]
    test_file = test_output_dir / config["test_file"]

    if not test_file.exists():
        print(f"[ERROR] Test file not found: {test_file}")
        print(f"[INFO] Run './run_test_generation.sh --operation {operation}' first")
        return False

    print(f"\n{'='*70}")
    print(f"📝 Completing TODOs for: {operation}")
    print(f"   Test file: {test_file.name}")
    print(f"   Operations: {', '.join(config['operations'])}")
    print(f"{'='*70}\n")

    # Build context files
    context_files = {
        "testing_plan": Path("TESTING_PLAN.md"),
        "example_test": repo_root / "tests/tt_metal/tt_metal/test_eltwise_binary.cpp",
    }

    # Create agent
    try:
        agent = TestCompletionAgent()
    except Exception as e:
        print(f"[ERROR] Failed to create agent: {e}")
        return False

    # Complete the test
    try:
        completed_content = agent.complete_test_todo(test_file, operation, context_files)

        if dry_run:
            print(f"[DRY RUN] Would write completed test to: {test_file}")
            print(f"[DRY RUN] Preview (first 500 chars):\n{completed_content[:500]}...")
        else:
            # Write back
            test_file.write_text(completed_content)
            print(f"[✓] Completed: {test_file}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed to complete {operation}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AI agent to complete TODO sections in generated test files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete tests for specific operation
  python complete_test_todos.py --operation eltwise_binary

  # Complete all operations (one after another)
  python complete_test_todos.py --all

  # Dry run (preview only)
  python complete_test_todos.py --operation reduce --dry-run
        """,
    )

    parser.add_argument("--operation", choices=list(OPERATIONS.keys()), help="Operation to complete tests for")

    parser.add_argument("--all", action="store_true", help="Complete tests for all operations")

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("tt-metal/tests/tt_metal/tt_metal/block_variants"),
        help="Directory containing generated test files",
    )

    parser.add_argument("--repo-root", type=Path, default=Path("tt-metal"), help="Path to tt-metal repository root")

    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing files")

    args = parser.parse_args()

    # Validate
    if not args.all and not args.operation:
        parser.error("Must specify --operation or --all")

    # Check anthropic availability
    if not check_anthropic():
        print("[ERROR] anthropic package not installed")
        print("[INFO] Install with: pip install anthropic")
        return 1

    print("╔════════════════════════════════════════════════════════════╗")
    print("║  AI Agent Test TODO Completer                         ║")
    print("║  tt-metal Compute API - Issue #35739                   ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    # Process operations
    operations_to_process = []
    if args.all:
        # Sort by priority
        operations_to_process = sorted(OPERATIONS.keys(), key=lambda k: OPERATIONS[k]["priority"])
    else:
        operations_to_process = [args.operation]

    print(f"[INFO] Processing {len(operations_to_process)} operation(s)")
    if args.dry_run:
        print("[INFO] DRY RUN mode - no files will be modified\n")

    # Process each operation
    success_count = 0
    for operation in operations_to_process:
        if complete_operation_tests(operation, args.repo_root, args.test_dir, args.dry_run):
            success_count += 1

    print(f"\n{'='*70}")
    print(f"✅ Completed {success_count}/{len(operations_to_process)} operations")
    print(f"{'='*70}\n")

    if success_count == len(operations_to_process):
        print("🎉 All operations completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Review completed tests")
        print("   2. Build: cd tt-metal && ./build_metal.sh --build-tests")
        print("   3. Run: ./build/test/tt_metal/test_*_block")
        return 0
    else:
        print(f"⚠️  Some operations failed ({len(operations_to_process) - success_count} failures)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
