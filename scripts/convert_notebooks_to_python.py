#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
This script is intended to be used as a Git pre-commit hook for maintaining Jupyter notebooks tutorials
in a 'notebooks/' directory that require synchronized Python script exports in a 'python/' directory.

The script performs the following actions:
- Identifies all staged Jupyter notebook (.ipynb) files under the 'notebooks/' directory.
- Converts each staged notebook to a Python script using Jupyter nbconvert with a custom template.
- Places the generated Python scripts in the 'python/' directory, updating them only if the content has changed.
- Stages any new or updated Python scripts for commit, ensuring that the exported scripts are always in sync with the notebooks.
- Exits with a non-zero status code if any files were added or updated, signaling to the Git hook that further action may be required.

This automation helps enforce consistency between notebooks and their corresponding Python scripts in version control.
"""

from pathlib import Path
import subprocess
import sys
import shutil

NOTEBOOKS_DIR = Path("ttnn/tutorials")
OUTPUT_DIR = Path("ttnn/tutorials/basic_python")
TEMPLATE_DIR = Path("scripts/nbconvert_template")
TEMPLATE_NAME = "ttnn_examples_convert"
EXCLUDED_TUTORIALS = ["ttnn/tutorials/ttnn_intro.ipynb"]


def get_staged_notebooks() -> list[Path]:
    """
    Returns a list of staged .ipynb files under the notebooks/ directory.
    """
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    staged_files = result.stdout.splitlines()
    return [
        Path(f)
        for f in staged_files
        if f.endswith(".ipynb") and Path(f).is_relative_to(NOTEBOOKS_DIR) and f not in EXCLUDED_TUTORIALS
    ]


def convert_with_nbconvert(notebook: Path, output_tmp: Path) -> None:
    """
    Converts a Jupyter notebook to a Python script using nbconvert with a custom template.
    """
    subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "python",
            f"--template={TEMPLATE_NAME}",
            f"--TemplateExporter.extra_template_basedirs={TEMPLATE_DIR}",
            "--TemplateExporter.exclude_markdown=True",
            f"--output={output_tmp.name}",
            f"--output-dir={output_tmp.parent}",
            str(notebook),
        ],
        check=True,
    )


def has_file_changed(file: Path) -> bool:
    """
    Check if file has changed since last commit
    """
    result = subprocess.run(
        ["git", "diff", "--quiet", str(file)],
        stdout=subprocess.PIPE,
        text=True,
    )
    return result.returncode != 0


def filter_modified_files(files: list[Path]) -> list[Path]:
    """
    Filter files that have changed since last commit
    """

    # If input is empty list, then `git diff --name-only` will list unstaged files
    # which is not what we want.
    if not files:
        return []

    # Make sure that untracked files are also added to the index
    # Otherwise, `git diff` will not show them
    subprocess.run(["git", "add", "--intent-to-add"] + [str(f) for f in files], check=True)

    # Check if any files have changed since last commit
    result = subprocess.run(
        ["git", "diff", "--name-only"] + [str(f) for f in files],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )

    return [Path(f) for f in result.stdout.splitlines()]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    staged_notebooks = get_staged_notebooks()
    if not staged_notebooks:
        print("✅ No staged notebooks to process.")
        return

    # Get all currently staged .py files (under OUTPUT_DIR)
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        stdout=subprocess.PIPE,
        text=True,
    )
    staged_files = set(Path(f) for f in result.stdout.splitlines())
    staged_py_files = {f for f in staged_files if f.suffix == ".py" and f.is_relative_to(OUTPUT_DIR)}

    new_files = []
    updated_files = []

    for notebook in staged_notebooks:
        output_file = OUTPUT_DIR / f"{notebook.stem}.py"

        # If the .py file is already staged, skip processing
        if output_file in staged_py_files:
            print(f"⏭ Skipping {notebook} — {output_file.name} already staged.")
            continue

        tmp_file = OUTPUT_DIR / f".tmp_{notebook.stem}.py"
        try:
            convert_with_nbconvert(notebook, tmp_file)
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to convert {notebook}, continuing.")
            continue

        if not output_file.exists():
            shutil.move(str(tmp_file), str(output_file))
            new_files.append(output_file)
        else:
            if output_file.read_text() != tmp_file.read_text():
                shutil.move(str(tmp_file), str(output_file))
                updated_files.append(output_file)
            else:
                tmp_file.unlink()

    # Stage new or changed files
    files_to_add = new_files + updated_files

    if files_to_add:
        subprocess.run(["black"] + [str(f) for f in files_to_add])

    # Only add files that have changed since last commit
    files_to_add = filter_modified_files(files_to_add)
    print(f"files to add: {files_to_add}")

    if files_to_add:
        subprocess.run(["git", "add"] + [str(f) for f in files_to_add], check=True)
        print("🟢 Staged files:")
        for f in files_to_add:
            print(f" → {f}")
        sys.exit(1)  # Fail to force user to review
    else:
        print("✅ No changes to generated files.")


if __name__ == "__main__":
    main()
