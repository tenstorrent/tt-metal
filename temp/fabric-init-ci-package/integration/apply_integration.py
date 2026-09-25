#!/usr/bin/env python3
"""Narrow, reviewed integration edits. Default prints a diff; --apply writes it.
Run FROM a tt-metal checkout after copying the package's .github/ and tests/.
Uses PyYAML's node marks to retain original YAML formatting/comments.
Refuses missing/ambiguous anchors. Does NOT add/duplicate the user's five zones.
"""
import argparse
import difflib
from pathlib import Path
import re
import yaml

NEW_INPUT = "tracy-debug-categories"
BASE = "tests/tt_metal/tt_fabric/fabric_init_perf"

def node_at(text, *keys):
    node = yaml.compose(text)
    for key in keys:
        if not isinstance(node, yaml.MappingNode):
            raise ValueError(f"Expected YAML map while resolving {keys}")
        matches = [value for field, value in node.value if field.value == key]
        if len(matches) != 1:
            raise ValueError(f"Missing/ambiguous YAML key {key} in {keys}")
        node = matches[0]
    return node

def insert_mapping_entries(text, keys, entries):
    node = node_at(text, *keys)
    if not isinstance(node, yaml.MappingNode) or not node.value:
        raise ValueError(f"Expected nonempty map: {keys}")
    first = node.value[0][0]
    line = first.start_mark.line
    indent = first.start_mark.column
    lines = text.splitlines(keepends=True)
    block = "\n".join(" " * indent + s if s else "" for s in entries.splitlines()) + "\n"
    lines.insert(line, block)
    return "".join(lines)

def once(text, pattern, replacement):
    text, count = re.subn(pattern, replacement, text, flags=re.M)
    if count != 1:
        raise ValueError(f"Expected one source anchor, got {count}: {pattern}")
    return text

def patch_build(text):
    if "# fabric-init-ci: category forwarding" in text:
        return text
    text = insert_mapping_entries(text, ("on", "workflow_call", "inputs"), '''tracy-debug-categories:
  description: "Host Tracy debug categories (off or fabric-init initially)"
  type: string
  required: false
  default: "off"''')
    # Find the shell step which actually invokes build_metal.sh; inject env safely.
    tree = yaml.compose(text)
    def visit(node):
        if isinstance(node, yaml.MappingNode):
            values = {k.value: v for k, v in node.value}
            if "run" in values and ('./build_metal.sh' in values["run"].value and re.search(
                    r'^\s*(?:\./build_metal\.sh\b.*|"\$\{build_args\[@\]\}")\s*$', values["run"].value, re.M)):
                yield values
            for _, v in node.value:
                yield from visit(v)
        elif isinstance(node, yaml.SequenceNode):
            for v in node.value:
                yield from visit(v)
    steps = list(visit(tree))
    if len(steps) != 1:
        raise ValueError("Cannot identify the unique build_metal.sh step; adapt this patch to checkout")
    step = steps[0]
    lines = text.splitlines(keepends=True)
    env_line = "FABRIC_INIT_TRACY_CATEGORIES: ${{ inputs.tracy-debug-categories || 'off' }}\n"
    if "env" in step:
        first = step["env"].value[0][0]
        lines.insert(first.start_mark.line, ' ' * first.start_mark.column + env_line)
    else:
        run_key = next(k for k, v in find_parent_mapping(tree, step["run"]).value if k.value == 'run')
        lines.insert(run_key.start_mark.line, ' ' * run_key.start_mark.column + 'env:\n' +
                     ' ' * (run_key.start_mark.column + 2) + env_line)
    text = ''.join(lines)
    def forward(m):
        indent = m.group('indent')
        block = [
            '# fabric-init-ci: category forwarding',
            'case "$FABRIC_INIT_TRACY_CATEGORIES" in',
            '  off|fabric-init) ;;',
            '  *) echo "Unsupported debug category selection" >&2; exit 2 ;;',
            'esac',
            'build_args+=(--build-perf-debug "$FABRIC_INIT_TRACY_CATEGORIES")',
        ]
        return ''.join(indent+s+'\n' for s in block) + m.group(0)
    text = once(text, r'^(?P<indent> +)(?:\./build_metal\.sh[^\n]*|"\$\{build_args\[@\]\}")$', forward)
    # Distinguish category-enabled libraries from other profiler build artifacts.
    if 'TRACY_SUFFIX=' in text:
        def suffix(m):
            return (m.group(0) + '\n' + m.group('indent') +
                    'TRACY_SUFFIX="${TRACY_SUFFIX%_fabric-init}${{ inputs.tracy-debug-categories == \'fabric-init\' && \'_fabric-init\' || \'\' }}"')
        # Several workflow steps/branches can independently initialize the suffix.
        # Update every assignment so artifact creation and lookup stay consistent.
        # subn visits only the original matches, not the assignments we insert.
        text, count = re.subn(
            r'^(?P<indent> +)TRACY_SUFFIX=[^\n]*$', suffix, text, flags=re.M)
        if count == 0:
            raise ValueError("Found TRACY_SUFFIX but no supported assignment; inspect the workflow")
    return text

def find_parent_mapping(node, target):
    if isinstance(node, yaml.MappingNode):
        if any(v is target for _, v in node.value): return node
        for _, value in node.value:
            found = find_parent_mapping(value, target)
            if found: return found
    elif isinstance(node, yaml.SequenceNode):
        for value in node.value:
            found = find_parent_mapping(value, target)
            if found: return found
    return None

def patch_parent(text):
    if 'run-fabric-init-perf-tests:' in text: return text
    text = insert_mapping_entries(text, ('on','workflow_dispatch','inputs'), '''run-fabric-init-perf-tests:
  description: "Include Galaxy FABRIC_2D cold/hot initialization profiling under perf tests"
  type: boolean
  default: false
fabric-init-baseline-mode:
  description: "Report first; enforce after baseline calibration"
  type: choice
  options: [report, enforce]
  default: report''')
    text = insert_mapping_entries(text, ('jobs','build-artifact','with'), '''tracy-debug-categories: ${{ inputs.run-fabric-init-perf-tests && 'fabric-init' || 'off' }}''')
    text = insert_mapping_entries(text, ('jobs','fabric-perf-tests','with'), '''run-fabric-init-tests: ${{ inputs.run-fabric-init-perf-tests == true }}
init-baseline-mode: ${{ inputs.fabric-init-baseline-mode || 'report' }}''')
    # Selecting only init still schedules the perf group. Original push/schedule rules remain.
    return once(text,
        r"inputs\.run-fabric-perf-tests != false",
        "(inputs.run-fabric-perf-tests != false || inputs.run-fabric-init-perf-tests == true)")

def patch_perf(text):
    if '  fabric-init-tests:' in text: return text
    text = insert_mapping_entries(text, ('on','workflow_call','inputs'), '''run-fabric-init-tests:
  type: boolean
  default: false
init-baseline-mode:
  type: string
  default: report''')
    block = '''
  # fabric-init-ci: logically nested under the existing fabric-perf-tests group.
  fabric-init-tests:
    name: Fabric Init Tests
    if: ${{ inputs.run-fabric-init-tests }}
    uses: ./.github/workflows/fabric-init-perf-impl.yaml
    secrets: inherit
    with:
      build-artifact-name: ${{ inputs.build-artifact-name }}
      wheel-artifact-name: ${{ inputs.wheel-artifact-name }}
      docker-image: ${{ inputs.docker-image }}
      enabled-skus: wh_galaxy_perf,bh_galaxy_perf
      baseline-mode: ${{ inputs.init-baseline-mode }}
      pairs: 3
'''
    jobs = node_at(text, 'jobs')
    at = jobs.end_mark.index
    return text[:at].rstrip() + '\n' + block + text[at:]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    transforms = {
        '.github/workflows/build-artifact.yaml': patch_build,
        '.github/workflows/tm-fabric-tests.yaml': patch_parent,
        '.github/workflows/tm-fabric-tests-perf-impl.yaml': patch_perf,
        'tt_metal/tools/profiler/tracy_debug_categories.txt': lambda s: s if 'fabric-init' in s.splitlines() else s.rstrip()+'\nfabric-init\n',
        'tests/tt_metal/tt_fabric/CMakeLists.txt': lambda s: s if 'add_subdirectory(fabric_init_perf)' in s else s.rstrip()+'\n\nadd_subdirectory(fabric_init_perf)\n',
    }
    # Compute and validate everything before modifying any file.
    changes = []
    for name, fn in transforms.items():
        p = Path(name); old = p.read_text(); new = fn(old)
        if p.suffix == '.yaml': yaml.compose(new)
        changes.append((p, old, new))
    for p, old, new in changes:
        print(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(p),tofile=str(p))),end='')
    if args.apply:
        for p, old, new in changes:
            if old != new: p.write_text(new)
        print('Applied integration edits. Review git diff before committing.')

if __name__ == '__main__':
    main()
