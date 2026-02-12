#!/usr/bin/env python3
"""
Resolve merge conflicts from rebasing select_program_factory + validate_on_program_cache_hit removals.

Pattern: HEAD has one removed, incoming has the other removed.
Resolution: remove both (take neither side of the conflict).

Special case: operation_concepts.hpp needs both concepts kept.
"""

import re
import subprocess


def resolve_file(filepath):
    with open(filepath) as f:
        content = f.read()

    if '<<<<<<< HEAD' not in content:
        return False

    # Special handling for operation_concepts.hpp - keep both concepts
    if filepath.endswith('operation_concepts.hpp'):
        return resolve_keep_both(filepath, content)

    lines = content.split('\n')
    result = []
    i = 0
    resolved = False

    while i < len(lines):
        line = lines[i]

        if line.startswith('<<<<<<< HEAD'):
            ours_start = i + 1
            separator = None
            theirs_end = None

            j = i + 1
            while j < len(lines):
                if lines[j].startswith('======='):
                    separator = j
                elif lines[j].startswith('>>>>>>>'):
                    theirs_end = j
                    break
                j += 1

            if separator is None or theirs_end is None:
                result.append(line)
                i += 1
                continue

            ours_block = '\n'.join(lines[ours_start:separator])
            theirs_block = '\n'.join(lines[separator+1:theirs_end])

            has_select_factory_ours = 'select_program_factory' in ours_block
            has_validate_hit_theirs = 'validate_on_program_cache_hit' in theirs_block
            has_select_factory_theirs = 'select_program_factory' in theirs_block
            has_validate_hit_ours = 'validate_on_program_cache_hit' in ours_block

            # Both sides remove different things - take neither
            if (has_select_factory_ours and has_validate_hit_theirs) or \
               (has_validate_hit_ours and has_select_factory_theirs):
                resolved = True
            elif not ours_block.strip() and has_validate_hit_theirs:
                # Ours is empty (already removed), theirs has validate_on_program_cache_hit
                resolved = True
            elif not ours_block.strip() and has_select_factory_theirs:
                resolved = True
            elif has_select_factory_ours and not theirs_block.strip():
                resolved = True
            elif has_validate_hit_ours and not theirs_block.strip():
                resolved = True
            elif not ours_block.strip() and not theirs_block.strip():
                resolved = True
            else:
                # Unknown conflict - take ours but print warning
                print(f"  WARNING: unexpected conflict in {filepath} at line {i+1}")
                print(f"    OURS: {ours_block[:120]}")
                print(f"    THEIRS: {theirs_block[:120]}")
                for line_idx in range(ours_start, separator):
                    result.append(lines[line_idx])
                resolved = True

            i = theirs_end + 1
            continue

        result.append(line)
        i += 1

    if resolved:
        new_content = '\n'.join(result)
        new_content = re.sub(r'\n{3,}', '\n\n', new_content)
        with open(filepath, 'w') as f:
            f.write(new_content)

    return resolved


def resolve_keep_both(filepath, content):
    """Keep both sides of the conflict (for operation_concepts.hpp)."""
    lines = content.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.startswith('<<<<<<< HEAD'):
            separator = None
            theirs_end = None
            j = i + 1
            while j < len(lines):
                if lines[j].startswith('======='):
                    separator = j
                elif lines[j].startswith('>>>>>>>'):
                    theirs_end = j
                    break
                j += 1

            if separator and theirs_end:
                for k in range(i + 1, separator):
                    result.append(lines[k])
                for k in range(separator + 1, theirs_end):
                    result.append(lines[k])
                i = theirs_end + 1
                continue

        result.append(line)
        i += 1

    with open(filepath, 'w') as f:
        f.write('\n'.join(result))
    return True


def get_conflicted_files():
    proc = subprocess.run(
        ['git', 'diff', '--name-only', '--diff-filter=U'],
        capture_output=True, text=True
    )
    return [f.strip() for f in proc.stdout.strip().split('\n') if f.strip()]


def main():
    files = get_conflicted_files()
    if not files:
        print("No conflicted files found")
        return 0

    print(f"Found {len(files)} conflicted files")
    resolved = 0
    for filepath in files:
        if resolve_file(filepath):
            resolved += 1

    print(f"Resolved {resolved}/{len(files)} files")

    # Check for any remaining conflicts
    still_conflicted = []
    for filepath in files:
        try:
            with open(filepath) as f:
                if '<<<<<<< HEAD' in f.read():
                    still_conflicted.append(filepath)
        except:
            pass

    if still_conflicted:
        print(f"\nStill conflicted:")
        for f in still_conflicted:
            print(f"  {f}")
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
