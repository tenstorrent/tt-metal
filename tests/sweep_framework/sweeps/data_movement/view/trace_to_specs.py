import re
import os


def extract_brackets_content(line):
    # Function to extract the content inside brackets
    brackets_content = []
    open_brackets = 0
    current_content = ""

    for char in line:
        if char == "[":
            open_brackets += 1
            if open_brackets > 0:
                current_content = ""  # Reset content inside the brackets
        elif char == "]":
            if open_brackets > 0:
                brackets_content.append(current_content.strip())
            open_brackets -= 1
        elif open_brackets > 0:
            current_content += char

    return brackets_content


def parse_md_file_simple_no_regex(file_path):
    view_specs = []

    with open(file_path, "r") as file:
        for line in file:
            # Extract all sets of content inside brackets
            brackets_content = extract_brackets_content(line)

            if len(brackets_content) >= 3:  # Ensure we have both shape and size
                shape_str = brackets_content[0]  # First set of brackets for shape
                size_str = brackets_content[2]  # Third set of brackets for size

                # Convert the shape and size strings to lists of integers
                if "s" in shape_str or "s" in size_str:
                    continue
                shape = list(map(int, shape_str.split(",")))
                size = list(map(int, size_str.split(",")))

                # Append the dictionary to the list
                view_specs.append({"shape": shape, "size": size})

    return view_specs


# Example usage
file_path = "sweeps/data_movement/view/view_trace.md"  # Replace with the actual path to your md file
print("hello")
print(os.getcwd())
view_specs = parse_md_file_simple_no_regex(file_path)

print(len(view_specs))
# Output the parsed data
for i in range(0, 20):
    print(view_specs[i])
