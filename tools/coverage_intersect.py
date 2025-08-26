#!/usr/bin/env python3

import sys
import os
from collections import defaultdict


def parse_lcov_info(path):
    files = {}  # sf_path -> { 'functions': {start_line: name}, 'fnda': {name: hits}, 'lines': {lineno: hits} }
    current_sf = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("SF:"):
                current_sf = line[3:]
                files.setdefault(current_sf, {"functions": {}, "fnda": {}, "lines": {}})
            elif line.startswith("FN:") and current_sf is not None:
                # FN:<line number>,<function name>
                try:
                    payload = line[3:]
                    ln_str, name = payload.split(",", 1)
                    ln = int(ln_str)
                    files[current_sf]["functions"][ln] = name
                except Exception:
                    pass
            elif line.startswith("FNDA:") and current_sf is not None:
                # FNDA:<execution count>,<function name>
                try:
                    payload = line[5:]
                    hits_str, name = payload.split(",", 1)
                    hits = int(float(hits_str))
                    files[current_sf]["fnda"][name] = hits
                except Exception:
                    pass
            elif line.startswith("DA:") and current_sf is not None:
                # DA:<line number>,<execution count>[,checksum]
                try:
                    payload = line[3:]
                    parts = payload.split(",")
                    ln = int(parts[0])
                    hits = int(float(parts[1]))
                    files[current_sf]["lines"][ln] = hits
                except Exception:
                    pass
            elif line == "end_of_record":
                current_sf = None
    return files


def build_function_ranges(functions_map):
    # Convert {start_line: name} into sorted list of (start, end, name)
    starts = sorted(functions_map.items())
    ranges = []
    for i, (start, name) in enumerate(starts):
        if i + 1 < len(starts):
            end = starts[i + 1][0] - 1
        else:
            end = 10**9
        ranges.append((start, end, name))
    return ranges


def assign_lines_to_functions(lines_map, function_ranges):
    fn_lines = defaultdict(set)  # name -> set(line numbers with hits > 0)
    for ln, hits in lines_map.items():
        if hits <= 0:
            continue
        # binary search would be nicer; linear acceptable for typical sizes
        for start, end, name in function_ranges:
            if start <= ln <= end:
                fn_lines[name].add(ln)
                break
    return fn_lines


def intersect_infos(info_a, info_b):
    out = []
    for sf in sorted(set(info_a.keys()) & set(info_b.keys())):
        a = info_a[sf]
        b = info_b[sf]
        a_ranges = build_function_ranges(a["functions"])
        b_ranges = build_function_ranges(b["functions"])

        a_fn_lines = assign_lines_to_functions(a["lines"], a_ranges)
        b_fn_lines = assign_lines_to_functions(b["lines"], b_ranges)

        # For strictness, a function must exist by name in both and have identical executed line sets
        common_fn_names = set(a_fn_lines.keys()) & set(b_fn_lines.keys())
        intersected_fn_names = [
            name for name in common_fn_names if a_fn_lines[name] == b_fn_lines[name] and len(a_fn_lines[name]) > 0
        ]

        if not intersected_fn_names:
            continue

        # Emit SF block
        out.append(f"SF:{sf}")

        # Emit FN and FNDA for intersected functions. Use A's start lines where possible.
        # Build name->start lookup from A (fallback to B if missing)
        name_to_start = {name: start for start, name in a["functions"].items()}
        for start_b, name_b in b["functions"].items():
            name_to_start.setdefault(name_b, start_b)

        for name in sorted(intersected_fn_names):
            start = name_to_start.get(name, 0)
            out.append(f"FN:{start},{name}")

        for name in sorted(intersected_fn_names):
            # Optional: require identical FNDA counts; comment next 2 lines to skip FNDA equality
            a_hits = a["fnda"].get(name, 0)
            b_hits = b["fnda"].get(name, 0)
            if a_hits != b_hits:
                # If strict FNDA equality is desired, skip mismatched
                continue
            out.append(f"FNDA:{a_hits},{name}")

        # Emit DA lines that belong to intersected functions and are executed in both
        # Build combined function ranges map by name (from A preferred)
        fn_ranges_by_name = {}
        for start, end, name in a_ranges:
            fn_ranges_by_name[name] = (start, end)
        for start, end, name in b_ranges:
            fn_ranges_by_name.setdefault(name, (start, end))

        # Lines: intersection of executed lines, restricted to function blocks we keep
        lines_written = set()
        for name in intersected_fn_names:
            # Ensure FNDA equality was maintained (only include those we output FNDA for)
            a_hits = a["fnda"].get(name, 0)
            b_hits = b["fnda"].get(name, 0)
            if a_hits != b_hits:
                continue
            same_lines = a_fn_lines[name] & b_fn_lines[name]
            if name in fn_ranges_by_name:
                start, end = fn_ranges_by_name[name]
                for ln in sorted(same_lines):
                    if ln < start or ln > end:
                        continue
                    hits = min(a["lines"].get(ln, 0), b["lines"].get(ln, 0))
                    if hits > 0 and ln not in lines_written:
                        out.append(f"DA:{ln},{hits}")
                        lines_written.add(ln)

        # Summary metrics
        fns_total = len(intersected_fn_names)
        fns_hit = fns_total  # by construction, these are hit
        out.append(f"FNF:{fns_total}")
        out.append(f"FNH:{fns_hit}")
        # Let genhtml compute line totals; we wrote DA for executed lines only
        out.append("end_of_record")
    return "\n".join(out) + ("\n" if out else "")


def main():
    if len(sys.argv) != 4:
        print("Usage: coverage_intersect.py <merged_A.info> <merged_B.info> <out.info>", file=sys.stderr)
        sys.exit(1)
    a_path, b_path, out_path = sys.argv[1:4]
    if not os.path.isfile(a_path) or not os.path.isfile(b_path):
        print("ERROR: Inputs must be files", file=sys.stderr)
        sys.exit(1)
    a = parse_lcov_info(a_path)
    b = parse_lcov_info(b_path)
    out_text = intersect_infos(a, b)
    with open(out_path, "w") as f:
        f.write(out_text)


if __name__ == "__main__":
    main()
