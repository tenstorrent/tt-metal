import os
import re


def remove_commented_lines(start_directory=".", comment_string="--commented out by unused variable remover"):
    """
    Recursively searches for files in the specified directory and removes lines
    containing the specified comment string.

    Args:
        start_directory (str): The directory to start the search from (defaults to the current directory).
        comment_string (str): The comment string to search for and remove lines containing it.
    """

    files_modified = 0
    lines_removed = 0

    for root, _, files in os.walk(start_directory):
        for filename in files:
            if filename.endswith((".cpp", ".c", ".hpp", ".h", ".cxx", ".hxx")):  # Adjust file extensions as needed
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, "r") as f:
                        lines = f.readlines()

                    modified_lines = []
                    removed_count = 0
                    for line in lines:
                        if comment_string not in line:
                            modified_lines.append(line)
                        else:
                            removed_count += 1

                    if removed_count > 0:
                        with open(filepath, "w") as f:
                            f.writelines(modified_lines)
                        files_modified += 1
                        lines_removed += removed_count
                        print(f"Removed {removed_count} line(s) from {filepath}")

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    print(f"\nFinished processing. Modified {files_modified} file(s) and removed {lines_removed} line(s) in total.")


# Example Usage
remove_commented_lines()  # Starts from the current directory by default
# remove_commented_lines("/path/to/your/codebase")  # Specify a different starting directory
