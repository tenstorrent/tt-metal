import re
import sys
import json

# This script is used to parse and compare the model charts in the featured section of tt_metal/README.md and the full list of models in tt_metal/models/READMe.md.
# It relies on the structure of the Markdown files to extract model information and validate that featured models are correctly represented in the full model list.
# It enforces that featured models reflect the attributes of models in the full list, ensuring consistency across the documentation.


def strip_md_links(text):
    if not text:
        return None
    result = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    result = result.strip()
    return result if result else None


def _norm(text):
    if text is None:
        return None
    # Remove whitespace and '*'', lowercase.
    return re.sub(r"[\s\*]", "", str(text)).lower()


def parse_featured_models(md_file_path):
    """Parse the ## Featured Models section from a Markdown file.
    This function expects the Markdown file to have a section starting with '## Featured Models'
    and each model entry to be under a '### Model Name' header followed by an attributes chart.
    Args:
        md_file_path (str): Path to the Markdown file.
    Returns:
        list: A list of dictionaries representing each featured model with its attributes.
        list: A list of error messages encountered during parsing.
    """
    models = []
    errors = []
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_lines = f.readlines()

    found_featured = False
    section_start_idx = None

    for idx, line in enumerate(md_lines):
        if re.match(r"^\s*##+\s*Featured Models\s*$", line, re.IGNORECASE):
            section_start_idx = idx
            found_featured = True
            break

    if not found_featured:
        errors.append("ERROR: No ## Featured Models section found")
        return models, errors

    i = section_start_idx + 1
    last_norm_header = None

    while i < len(md_lines):
        line = md_lines[i].rstrip("\n")
        abs_line_number = i + 1
        ls = line.strip()
        # Model header
        if ls.startswith("###"):
            m = re.match(r"^###\s*(\[[^\]]+\]\([^)]+\)|.+)$", ls)
            model_name = strip_md_links(m.group(1).strip() if m else "")
            if not model_name:
                errors.append(f"ERROR: ### header found without model name at line {abs_line_number}")
                i += 1
                continue
            # Check for table
            table_lines = []
            j = i + 1
            while j < len(md_lines) and md_lines[j].strip().startswith("|"):
                table_lines.append(md_lines[j].strip())
                j += 1
            if not table_lines:
                errors.append(
                    f"ERROR: ### header for '{model_name}' at line {abs_line_number} not followed by attributes chart"
                )
                i += 1
                continue
            if len(table_lines) != 3:
                table_start_line = i + 2
                errors.append(
                    f"ERROR: Attributes chart under '{model_name}' at line {table_start_line} seems to be malformed."
                    f"\nExpected 3 rows (header / markdown chars / values) but found {len(table_lines)} rows instead."
                )
                i = j
                continue
            header_cells = [strip_md_links(cell) for cell in table_lines[0].strip("|").split("|")]
            value_cells = [
                strip_md_links(cell) if cell.strip() else None for cell in table_lines[2].strip("|").split("|")
            ]
            if len(header_cells) != len(value_cells):
                table_start_line = i + 2
                errors.append(
                    f"ERROR: Malformed attributes chart under '{model_name}' at line {table_start_line} (header/value cell count mismatch)"
                )
                i = j
                continue
            attrib = dict(zip(header_cells, value_cells))
            models.append({"name": model_name, "attributes": attrib})
            last_norm_header = [_norm(h) for h in header_cells]
            i = j
            continue

        # Check for section boundary ('##' not followed by att chart)
        if re.match(r"^##(?!#)", ls):
            if not models:
                break
            j = i + 1
            while j < len(md_lines) and md_lines[j].strip() == "":
                j += 1
            if j >= len(md_lines) or not md_lines[j].strip().startswith("|"):
                break
            hdr_cells = [strip_md_links(x) for x in md_lines[j].strip().strip("|").split("|")]
            norm_hdr = [_norm(h) for h in hdr_cells]
            if last_norm_header is not None and norm_hdr == last_norm_header:
                errors.append(
                    f"ERROR: Attributes chart after '##' header at line {j+1} matches previous model attribute headers (likely missing ### model header above table)."
                )
                k = j
                while k < len(md_lines) and md_lines[k].strip().startswith("|"):
                    k += 1
                i = k
                continue
            else:
                break
        # Standalone chart
        elif ls.startswith("|"):
            errors.append(f"ERROR: Attribute chart found without proper ### Model Name at line {abs_line_number}")
            while i < len(md_lines) and md_lines[i].strip().startswith("|"):
                i += 1
            continue
        else:
            i += 1

    return models, errors


def parse_all_models_charts(md_file_path):
    """Parse all model charts from a Markdown file.
    Args:
        md_file_path (str): Path to the Markdown file.
    Returns:
        list: A list of dictionaries representing each model with its attributes.
        list: A list of error messages encountered during parsing.
    """
    models = []
    errors = []
    with open(md_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    found_any_chart = False
    n = len(lines)
    i = 0

    def parse_table(start):
        table_lines = [lines[start].strip()]
        table_line_indices = [start]
        cur = start + 1
        while cur < n and lines[cur].strip().startswith("|"):
            table_lines.append(lines[cur].strip())
            table_line_indices.append(cur)
            cur += 1
        header_cells = [strip_md_links(cell) for cell in table_lines[0].strip("|").split("|")]
        return header_cells, table_lines, table_line_indices, cur

    while i < n:
        line = lines[i].strip()
        # Only begin table parsing when line starts with '|'
        if not (line.startswith("|") and "|" in line):
            i += 1
            continue
        header_cells, table_lines, table_line_indices, next_i = parse_table(i)

        # Only process tables where the first header cell is 'model'
        if not header_cells or not header_cells[0] or header_cells[0].lower() != "model":
            i = next_i
            continue

        found_any_chart = True
        # Check for separator row after header
        if len(table_lines) < 2 or not table_lines[1].startswith("|"):
            errors.append(f"ERROR: Model chart at line {i+1} missing separator row")
            i = next_i
            continue
        # Check for at least one model entry
        if len(table_lines) < 3:
            errors.append(f"ERROR: Model chart at line {i+1} has no model entries")
            i = next_i
            continue

        # Iterate rows (skip header and separator)
        for row_idx, row_line in enumerate(table_lines[2:], start=2):
            row_file_index = table_line_indices[row_idx]
            value_cells = [strip_md_links(cell) if cell.strip() else None for cell in row_line.strip("|").split("|")]
            # Value/header count mismatch
            if len(value_cells) != len(header_cells):
                errors.append(
                    f"ERROR: Model chart row at line {row_file_index+1} has {len(value_cells)} cells but header has {len(header_cells)}"
                )
                continue
            row_data = dict(zip(header_cells, value_cells))
            model_name = row_data.get(header_cells[0])
            if not model_name:
                errors.append(f"ERROR: Model entry with no name at line {row_file_index+1}")
                continue
            # Extract attributes
            attributes = {k: v for k, v in row_data.items() if k != header_cells[0]}
            if not attributes:
                errors.append(f"ERROR: Model '{model_name}' at line {row_file_index+1} has no attributes")
                continue
            models.append({"name": model_name, "attributes": attributes})
        i = next_i

    if not found_any_chart:
        errors.append("ERROR: No model chart found in file")
    return models, errors


def compare_featured_vs_all_models(featured_models, all_models):
    """
    For each featured model, check for a match in all_models (by name and all attributes presented).
    Matching is case-insensitive, ignores whitespace and '*' in keys/values.
    Args:
      featured_models: [{'name': ..., 'attributes': {...}}, ...]
      all_models:      [{'name': ..., 'attributes': {...}}, ...]
    Returns:
      errors (list of str)
    """
    errors = []

    # Build lookup for all models by normalized name
    all_by_name = {}
    for entry in all_models:
        nm = _norm(entry["name"])
        all_by_name.setdefault(nm, []).append(entry)

    for fmodel in featured_models:
        fname = _norm(fmodel["name"])
        fattrib_norm = {_norm(k): _norm(v) for k, v in fmodel.get("attributes", {}).items()}
        matches = all_by_name.get(fname, [])
        if not matches:
            errors.append(
                f"ERROR: Featured model '{fmodel['name']}' not found in all models.\n" f"Featured model entry: {fmodel}"
            )
            continue

        # Match attributes
        exact = []
        candidates_lines = []
        for candidate in matches:
            attribs = candidate.get("attributes", {})
            attribs_norm = {_norm(k): _norm(v) for k, v in attribs.items()}
            if all(attribs_norm.get(ak) == av for ak, av in fattrib_norm.items()):
                exact.append(candidate)
            else:
                # collect diffs
                diffs = []
                for fkey, fval in fattrib_norm.items():
                    cval = attribs_norm.get(fkey)
                    if cval != fval:
                        diffs.append(f"    [{fkey}]: featured='{fval}', candidate='{cval}'")
                candidates_lines.append(f"  Candidate: {candidate}\n" + "\n".join(diffs))
        if exact:  # at least one match
            continue

        errors.append(
            f"ERROR: Missmatch for Featured model '{fmodel['name']}' (attributes: {fmodel['attributes']})"
            f"\nFound {len(matches)} candidates with same name but different attributes. Compare:"
            f"\n" + "\n".join(candidates_lines)
        )

    return errors


if __name__ == "__main__":
    # Set the paths to the Markdown files
    featured_md = "./README.md"
    full_list_md = "./models/README.md"

    # Fetch models from the Markdown files
    print(f"--- Fetching models from featured file: {featured_md}")
    featured_models, featured_errors = parse_featured_models(featured_md)
    print(f"Found {len(featured_models)} featured models.")

    print(f"--- Fetching models from full list file: {full_list_md}")
    all_models, all_errors = parse_all_models_charts(full_list_md)
    print(f"Found {len(all_models)} models in the full list.")

    # Check if no models were found
    failed = False
    if not featured_models:
        print(f"❌ No featured models found in the {featured_md} file.")
        failed = True
    if not all_models:
        print(f"❌ No models found in the {full_list_md} file.")
        failed = True
    if failed:
        sys.exit(1)

    # Report parsing errors
    if featured_errors:
        failed = True
        print("❌ Errors found while parsing featured models:")
        for err in featured_errors:
            print(f" - {err}")
    if all_errors:
        failed = True
        print("❌ Errors found while parsing full model list:")
        for err in all_errors:
            print(f" - {err}")
    if failed:
        sys.exit(1)

    # Compare featured models to the full models list
    compare_errors = compare_featured_vs_all_models(featured_models, all_models)

    # Report results
    if compare_errors:
        for err in compare_errors:
            print(f"\n❌ {err}")
        sys.exit(1)
    else:
        print(f"\n✅ All featured models and their attributes match entries in the full model list.")
        sys.exit(0)
