import re
import logging


def remove_inline_comments(text):
    """Removes inline comments (//...) from a string."""
    return re.sub(r"//.*", "", text).rstrip()


def is_variable_written_later(filename, line_number, variable_name, source_lines):
    """
    Checks if a variable is written to after its declaration line in the source file.

    Args:
        filename (str): The path to the source file.
        line_number (int): The line number where the variable is declared.
        variable_name (str): The name of the variable to check.
        source_lines (list): A list of strings, where each string is a line
            from the source file.

    Returns:
        bool: True if the variable is written to on a subsequent line,
            False otherwise.
    """

    for i in range(line_number, len(source_lines)):  # Start *after* the declaration
        line = remove_inline_comments(source_lines[i])
        # Simple check for assignment. Could be improved for more complex cases.
        if variable_name + " =" in line or variable_name + "=" in line or variable_name + " " in line:
            # Exclude assignments within the same line as declaration to not count the initializing assignment
            if i + 1 != line_number:
                # Make sure we are not inside a comment
                if "//" not in line or line.find("//") > line.find(variable_name):
                    logging.info(f"Variable '{variable_name}' is written to on line {i+1} of {filename}")
                    return True  # Found a write

    logging.info(f"Variable '{variable_name}' is NOT written to after line {line_number} of {filename}")
    return False


def comment_out_unused_variables(log_file, num_variables_to_comment=5, output_log_file="build_removing_variables.log"):
    """
    Reads a build log, finds lines with "unused variable" warnings, and comments out the
    corresponding lines in the source files. Handles multi-line declarations, adds a comment flag,
    skips already commented out lines and for loops, and handles single-line multi-variable declarations.

    Args:
        log_file (str): Path to the build log file.
        num_variables_to_comment (int): The maximum number of unused variables to comment out.
        output_log_file (str): Path to the file where log messages will be written.
    """

    # Configure logging to write to a file
    logging.basicConfig(
        filename=output_log_file, filemode="w", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    common_variable_types = [
        "int",
        "short",
        "long",
        "long long",
        "unsigned int",
        "unsigned short",
        "unsigned long",
        "unsigned long long",
        "char",
        "signed char",
        "unsigned char",
        "wchar_t",
        "float",
        "double",
        "long double",
        "bool",  # C++ only
        "void",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "int_least8_t",
        "int_least16_t",
        "int_least32_t",
        "int_least64_t",
        "uint_least8_t",
        "uint_least16_t",
        "uint_least32_t",
        "uint_least64_t",
        "int_fast8_t",
        "int_fast16_t",
        "int_fast32_t",
        "int_fast64_t",
        "uint_fast8_t",
        "uint_fast16_t",
        "uint_fast32_t",
        "uint_fast64_t",
        "intptr_t",
        "uintptr_t",
        # C++ Standard Library Types (most common)
        "std::string",  # string class
        "std::wstring",  # wide string class
        "std::vector",  # dynamic array
        "std::array",  # fixed size array
        "std::list",  # doubly-linked list
        "std::deque",  # double ended queue
        "std::queue",  # queue
        "std::stack",  # stack
        "std::set",  # set (unique elements)
        "std::multiset",  # set (multiple elements allowed)
        "std::map",  # map (key-value pairs, unique keys)
        "std::multimap",  # map (key-value pairs, multiple keys allowed)
        "std::pair",  # for storing pairs of values
        "std::tuple",  # for storing multiple values
        "std::optional"  # for storing values that may or may not exist
        # Smart Pointers
        "std::unique_ptr",  # Smart pointer
        "std::shared_ptr",  # Smart pointer
        "std::weak_ptr",  # Smart pointer
    ]

    count = 0
    with open(log_file, "r") as f:
        for line in f:
            if (
                "warning: unused variable" in line or "-Wunused-but-set-variable" in line
            ) and count < num_variables_to_comment:
                try:
                    # Extract filename and line number from the warning message
                    match = re.search(r"(.*?):(\d+):", line)
                    if not match:
                        logging.info(f"Could not parse line for filename and line number: {line.strip()}")
                        continue  # Skip to the next line

                    filename = match.group(1)
                    line_number = int(match.group(2))

                    # Read the entire source file
                    with open(filename, "r") as source_file:
                        source_lines = source_file.readlines()

                    if line_number <= 0 or line_number > len(source_lines):
                        logging.info(f"Line number {line_number} is out of range in file {filename}:{line_number}")
                        continue

                    # Get the problematic line
                    original_line = source_lines[line_number - 1].strip()

                    if (
                        line_number > 2
                        and remove_inline_comments(source_lines[line_number - 2]).strip() != ""
                        and (
                            (
                                remove_inline_comments(source_lines[line_number - 2].strip())[-1]
                                not in [";", "{", "}", "(", "[", "<"]
                            )
                            and ("//" not in source_lines[line_number - 2].strip()[0:2])
                        )
                    ):
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} is a continuation of a multi-line declaration. Skipping for now."
                        )
                        continue

                    if ";" not in original_line:
                        logging.info(f"Line {line_number} in {filename}:{line_number} is not a single line. Skipping.")
                        continue

                    common_type = False
                    for variable_type in common_variable_types:
                        if variable_type in original_line:
                            common_type = True
                            break
                    if not common_type:
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} is not a common variable type. Skipping."
                        )
                        continue

                    # Extract the variable name from the warning
                    variable_name_match = re.search(r"variable '(\w+)'", line)
                    if not variable_name_match:
                        logging.info(f"Could not extract variable name from warning: {line.strip()}")
                        continue
                    variable_name = variable_name_match.group(1)

                    if "-Wunused-but-set-variable" in line and is_variable_written_later(
                        filename, line_number, variable_name, source_lines
                    ):
                        logging.info(
                            f"Variable '{variable_name}' is written to after line {line_number} of {filename}. Skipping."
                        )
                        continue

                    # Check if the line is already commented out
                    if "//" in original_line.lstrip()[0:2]:
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} is already commented out. Skipping."
                        )
                        continue

                    # check if the line is a for loop.
                    if "for(" in original_line.strip() or "for (" in original_line.strip():
                        logging.info(f"Line {line_number} in {filename}:{line_number} is a for loop. Skipping.")
                        continue

                    # Check for un-nested commas
                    def has_unnested_comma(text):
                        balance = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0, "<": 0, ">": 0}
                        for char in text:
                            if char in balance:
                                if char in "([{<":
                                    balance[char] += 1
                                else:
                                    balance[{")": "(", "]": "[", "}": "{", ">": "<"}[char]] -= 1
                            if char == "," and all(v == 0 for v in balance.values()):
                                return True
                        return False

                    if has_unnested_comma(original_line):
                        logging.info(
                            f"Line {line_number} in {filename}:{line_number} contains an un-nested comma. Skipping."
                        )
                        continue

                    if "tt_metal" in original_line.strip():
                        logging.info(f"Line {line_number} in {filename}:{line_number} contains tt_metal. Skipping.")
                        continue

                    # Add comment to the line
                    source_lines[line_number - 1] = (
                        " //" + source_lines[line_number - 1].rstrip() + " --commented out by unused variable remover\n"
                    )

                    # Write the modified content back to the source file
                    with open(filename, "w") as source_file:
                        source_file.writelines(source_lines)

                    logging.info(f"Added comment to line {line_number} in {filename}:{line_number}")
                    count += 1

                except FileNotFoundError:
                    logging.error(f"File not found: {filename}:{line_number}")
                except Exception as e:
                    logging.error(f"Error processing line: {line.strip()}. Error: {e}")

    logging.info(f"Commented out {count} unused variables.")


# Example usage:
log_file_path = "build.log"  # Replace with your actual log file path
number_of_variables = 999999999  # adjust to your needs
comment_out_unused_variables(log_file_path, number_of_variables)
