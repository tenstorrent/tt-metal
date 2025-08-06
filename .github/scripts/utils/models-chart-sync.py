import re
import sys
import json

# This script is used to parse and compare the model charts in the featured section of tt_metal/README.md and the full list of models in tt_metal/models/READMe.md.
# It relies on the structure of the Markdown files to extract model information and validate that featured models are correctly represented in the full model list.


def parse_featured_models_from_file(md_file_path):
    """Parse the "Featured Models" section from a Markdown file.

    Args:
        md_file_path (str): Path to the Markdown file.

    Returns:
        list: A list of dictionaries representing each model with its attributes.
    """

    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    models = []
    # 1. Isolate the "## Featured Models" section
    featured_section = re.split(r"^##\s+", md_text, flags=re.MULTILINE)
    for part in featured_section:
        if part.strip().startswith("Featured Models"):
            section = part
            break
    else:
        return []

    # 2. Find all "###" model headers and their tables
    model_iter = re.finditer(
        r"###\s*(\[[^\]]+\]\([^)]+\))\s*\n"  # match [Name](link)
        r"((?:.|\n)+?)"  # match everything after (non-greedy)
        r"(?=^###|\Z)",  # until next ### or end of string
        section,
        flags=re.MULTILINE,
    )

    for match in model_iter:
        model_ref = match.group(1)  # [Name](link)
        body = match.group(2)

        # Get name and link
        link_match = re.match(r"\[([^\]]+)\]\(([^)]+)\)", model_ref)
        if link_match:
            model_name = link_match.group(1).strip()
            model_link = link_match.group(2).strip()
        else:
            model_name = model_ref.strip()
            model_link = None

        # Find the first table in body (header, divider, and one or more rows)
        table_match = re.search(r"((?:\|.+\n)+)", body)
        if not table_match:
            continue  # skip if there's no table

        # Split table into lines and parse
        table_lines = [l for l in table_match.group(1).strip().splitlines() if l.strip()]
        if len(table_lines) < 3:
            continue  # malformed table

        # Take header, divider, and at least one row
        header_line = table_lines[0]
        field_names = [h.strip().replace("<br>", " ") for h in header_line.strip("| \n").split("|")]

        # Find the index of the divider line
        divider_idx = next((i for i, line in enumerate(table_lines) if re.match(r"^\|\s*[-:]+", line)), 1)
        # Data rows are after divider
        data_lines = table_lines[divider_idx + 1 :]

        for row_line in data_lines:
            row_items = [item.strip() for item in row_line.strip("| \n").split("|")]
            if len(row_items) != len(field_names):
                continue  # skip malformed rows
            attrib = dict(zip(field_names, row_items))
            model = {"name": model_name, "link": model_link, "attributes": attrib}
            models.append(model)
    return models


def parse_model_charts_from_file(md_file_path):
    """Parse all model charts from a Markdown file.

    Args:
        md_file_path (str): Path to the Markdown file.

    Returns:
        list: A list of dictionaries representing each model with its attributes.
    """

    with open(md_file_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    models = []
    # Find all markdown tables whose first (leftmost) column is 'Model' (optionally with spaces)
    table_pattern = re.compile(
        r"(^\s*\|.+?\|\s*\n"  # header
        r"\|(?:\s*:?-+:?\s*\|)+\s*\n"  # divider line
        r"(?:\|.+?\|\s*\n)+)",  # at least one row
        flags=re.MULTILINE,
    )

    for table_match in table_pattern.finditer(md_text):
        table = table_match.group(0)
        lines = [line.strip() for line in table.strip().splitlines() if line.strip()]
        if len(lines) < 3:
            continue  # Not enough rows

        header_cells = [c.strip().replace("<br>", " ") for c in lines[0].strip("| ").split("|")]
        if not header_cells or header_cells[0].lower() != "model":
            continue  # Only process tables whose first cell is 'Model'

        # Find the divider row (should be the 2nd row)
        divider_index = 1
        for i, line in enumerate(lines):
            if re.match(r"^\|[\s:-]+\|$", line):
                divider_index = i
                break
        rows = lines[divider_index + 1 :]

        for row in rows:
            row_cells = [c.strip() for c in row.strip("| ").split("|")]
            if len(row_cells) != len(header_cells):
                continue  # skip malformed rows

            model_cell = row_cells[0]
            # Parse [name](link) in the Model cell
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", model_cell)
            if m:
                name = m.group(1).strip()
                link = m.group(2).strip()
            else:
                name = model_cell.strip()
                link = None
            # Create attribute dict (exclude "Model")
            attrib = dict(zip(header_cells[1:], row_cells[1:]))
            models.append({"name": name, "link": link, "attributes": attrib})
    return models


def compare_featured_to_full_list(featured_models, all_models):
    """Compare featured models against the full model list.

    Args:
        featured_models (list): List of featured models with attributes.
        all_models (list): List of all models with attributes.

    Returns:
        tuple: (missing, mismatches) where:
            - missing: List of models that are in featured but not in full list.
            - mismatches: List of models with attribute mismatches.
    """

    mismatches = []
    missing = []
    for feat in featured_models:
        # Only match by name
        candidates = [m for m in all_models if m["name"] == feat["name"]]
        if not candidates:
            missing.append({"name": feat["name"], "reason": "Model missing in full list", "featured_entry": feat})
            continue

        # Lowercase attribute keys for comparison
        feat_attribs = {k.lower(): v for k, v in feat["attributes"].items()}
        match_found = False

        for cand in candidates:
            cand_attribs = {k.lower(): v for k, v in cand["attributes"].items()}
            all_match = True
            for fk, fv in feat_attribs.items():
                # Ignore trailing asterisks in both featured and candidate attribute values
                fv_clean = str(fv).strip().rstrip("*").strip()
                cand_val = str(cand_attribs.get(fk, "")).strip().rstrip("*").strip()
                if fk not in cand_attribs or fv_clean != cand_val:
                    all_match = False
                    break
            if all_match:
                match_found = True
                break
        if not match_found:
            mismatches.append(
                {
                    "name": feat["name"],
                    "reason": "Attribute(s) mismatch",
                    "featured_entry": feat,
                    "full_list_candidates": candidates,
                }
            )
    return missing, mismatches


if __name__ == "__main__":
    featured_md = "./README.md"
    full_list_md = "./models/README.md"

    print(f"--- Fetching models from featured file: {featured_md}")
    featured_models = parse_featured_models_from_file(featured_md)
    if not featured_models:
        print("❌ No featured models found in the README.md file.")
        sys.exit(1)
    print(f"Found {len(featured_models)} featured models.")

    print(f"--- Fetching models from full list file: {full_list_md}")
    all_models = parse_model_charts_from_file(full_list_md)
    if not all_models:
        print("❌ No models found in the full list file.")
        sys.exit(1)
    print(f"Found {len(all_models)} models in the full list.")

    missing, mismatches = compare_featured_to_full_list(featured_models, all_models)

    if missing or mismatches:
        for entry in missing:
            print(f"\n❌ Missing model: {entry['name']}")
        for entry in mismatches:
            print(f"\n❌ Attribute mismatch in model: {entry['name']}")
            print("  Featured attributes:", entry["featured_entry"]["attributes"])
            print("  Full list candidates:")
            for c in entry["full_list_candidates"]:
                print("   ->", c["attributes"])
        sys.exit(1)
    else:
        print(f"\n✅ All featured models are matched and attributes are correct in the full model list.")
        sys.exit(0)
