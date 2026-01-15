import os
import re

base_path = "/home/boxx/tt-metal/ttnn/cpp/ttnn/operations"


def fix_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Find the namespace ttnn::operations::... {
    ns_match = re.search(r"namespace ttnn::operations::([\w_:]+) \{", content)
    if not ns_match:
        return

    ns_full = ns_match.group(0)

    # Check if namespace ttnn::prim is nested
    prim_match = re.search(r"namespace ttnn::prim \{", content)
    if not prim_match:
        return

    prim_start = prim_match.start()

    # Find the last occurrence of any namespace opening BEFORE ttnn::prim
    # and check if there's a } after it but before ttnn::prim.

    before_prim = content[:prim_start]
    last_ns = before_prim.rfind("namespace ")
    if last_ns == -1:
        return

    # Simple check: is there a } between last_ns and prim_start?
    if "}" not in before_prim[last_ns:]:
        # It's likely nested!
        print(f"Fixing nested namespace in {file_path}")

        # Also need to find if there's a re-opening after ttnn::prim
        prim_end_match = re.search(r"\}  // namespace ttnn::prim", content[prim_start:])
        if prim_end_match:
            prim_end = prim_start + prim_end_match.end()

            # Re-open the previous namespace if there was more code after prim
            after_prim = content[prim_end:]

            # Find the last open namespace name to re-open it
            ns_lines = re.findall(r"namespace ([\w_:]+) \{", before_prim)
            if not ns_lines:
                return
            last_ns_name = ns_lines[-1]

            new_content = (
                content[:prim_start].rstrip()
                + f"\n\n}}  // namespace {last_ns_name}\n\n"
                + content[prim_start:prim_end]
            )

            if after_prim.strip():
                # Check if it already re-opens
                if not re.search(r"namespace " + re.escape(last_ns_name) + r" \{", after_prim):
                    new_content += f"\n\nnamespace {last_ns_name} {{\n" + after_prim
                else:
                    new_content += after_prim
            else:
                new_content += after_prim

            with open(file_path, "w") as f:
                f.write(new_content)


for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".cpp") or file.endswith(".hpp"):
            fix_file(os.path.join(root, file))
