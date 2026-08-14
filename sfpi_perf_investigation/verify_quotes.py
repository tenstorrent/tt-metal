#!/usr/bin/env python3
"""Check that code quoted in the issue drafts still matches the probes.

Every C++ block in issues/*.md that opens with

    // verified: <probe file> :: <function>

must appear verbatim in that probe file, so a quote cannot drift away from the
source that produced the instruction counts it claims.

Usage: verify_quotes.py            # checks issues/*.md against ./*.cc
Exit status is non-zero if any quote does not match.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
MARKER = re.compile(r"^// verified:\s*(\S+)\s*::\s*(\S+)\s*$")
BLOCK = re.compile(r"```cpp\n(.*?)```", re.S)


def main():
    drafts = sorted((HERE / "issues").glob("*.md"))
    if not drafts:
        sys.exit("no issue drafts found next to this script")

    checked = failed = 0
    for draft in drafts:
        for block in BLOCK.findall(draft.read_text()):
            head, _, body = block.partition("\n")
            m = MARKER.match(head.strip())
            if not m:
                continue  # illustrative snippet, nothing to check against
            probe_name, fn = m.groups()
            checked += 1
            probe = HERE / probe_name
            label = f"{draft.name}: {probe_name} :: {fn}"
            if not probe.is_file():
                print(f"FAIL {label} -- no such probe file")
                failed += 1
            elif body not in probe.read_text():
                print(f"FAIL {label} -- quote does not match the probe verbatim")
                failed += 1
            elif f"void {fn}(" not in body:
                print(f"FAIL {label} -- quote does not define {fn}")
                failed += 1
            else:
                print(f"ok   {label}")

    print(f"\n{checked} quote(s) checked, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
