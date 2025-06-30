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

NOTEBOOKS_DIR = Path("ttnn/tutorials/2025_dx_rework")
OUTPUT_DIR = Path("ttnn/tutorials/basic_python")
TEMPLATE_DIR = Path("scripts/nbconvert_template")
TEMPLATE_NAME = "ttnn_examples_convert"


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
    return [Path(f) for f in staged_files if f.endswith(".ipynb") and Path(f).is_relative_to(NOTEBOOKS_DIR)]


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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    staged_notebooks = get_staged_notebooks()
    if not staged_notebooks:
        print("✅ No staged notebooks to process.")
        return

    new_files = []
    updated_files = []

    for notebook in staged_notebooks:
        output_file = OUTPUT_DIR / f"{notebook.stem}.py"
        tmp_file = OUTPUT_DIR / f".tmp_{notebook.stem}.py"

        convert_with_nbconvert(notebook, tmp_file)

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
        subprocess.run(["git", "add"] + [str(f) for f in files_to_add], check=True)
        print("🟢 Staged files:")
        for f in files_to_add:
            print(f" → {f}")
        sys.exit(1)  # Exit with error code to indicate changes were made
    else:
        print("✅ No changes to generated files.")


if __name__ == "__main__":
    main()
