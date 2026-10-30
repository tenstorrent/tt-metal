# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Plain-text rendering of a Result for the command line."""

from hal_explorer.model import Access, Instruction, Loop


def when_text(conditions):
    return " && ".join(conditions) or "always"


def origin_text(origin):
    if origin is None:
        return ""
    if origin.in_input:
        return f"  <- input:{origin.line}"
    return f"  <- {origin.function or '?'} ({origin.file}:{origin.line})"


def instruction_line(item, number):
    fields = " ".join(
        f"{f['name']}={'rt' if f['runtime'] else f['value']}" for f in item.fields
    )
    runtime = f" rt=({item.runtime_expr})" if item.runtime_mask else ""
    word = f"0x{item.word:08x}" + (" + rt" if item.runtime_mask else "")
    issue = {"immediate": "imm", "store": "store", "builtin": "builtin"}[item.issue]
    return f"{number:>4}  {item.mnemonic:<12} {word:<16} [{issue}] {fields}{runtime}{origin_text(item.origin)}"


def item_lines(items, indent="    ", counter=None):
    counter = counter if counter is not None else [0]
    for item in items:
        if isinstance(item, Loop):
            yield f"{indent}loop, repeats while {item.repeat_while} {{"
            if item.variants:
                for variant in item.variants:
                    when = " | ".join(when_text(c) for c in variant["when"])
                    yield f"{indent}  when {when}:"
                    yield from item_lines(variant["body"], indent + "    ", [None])
            else:
                yield from item_lines(item.body, indent + "  ", [None])
            yield f"{indent}}}"
        elif isinstance(item, Access):
            yield f"{indent}   -  MMIO {item.operation:<5} {item.width}b {item.target} = {item.value_text}{origin_text(item.origin)}"
        elif isinstance(item, Instruction):
            number = "-" if counter[0] is None else counter[0]
            if counter[0] is not None:
                counter[0] += 1
            yield indent + instruction_line(item, number)


def function_text(function, show_gimple=False):
    lines = [
        f"== {function.signature}: {function.path_count} path(s), {len(function.groups)} distinct sequence(s)"
    ]
    for warning in function.warnings:
        lines.append(f"  !! {warning['message']}{origin_text(warning['origin'])}")
    for group in function.groups:
        when = " | ".join(when_text(c) for c in group.when)
        lines.append(
            f"  -- count {group.count_text}  when {when}  fingerprint {group.fingerprint}"
        )
        lines.extend(item_lines(group.sequence))
    if function.samples and any(g.count is None for g in function.groups):
        lines.append("  -- samples")
        for sample in function.samples[:16]:
            inputs = (
                ", ".join(f"{k}={v}" for k, v in sample.inputs.items()) or "no inputs"
            )
            lines.append(
                f"     {inputs}: {sample.count}{'' if sample.complete else ' (stopped at a limit)'}"
            )
    if show_gimple:
        lines.append("  -- compiler output (optimized GIMPLE)")
        lines.extend(
            f"   {'●' if l['mark'] == 'insn' else '○' if l['mark'] else ' '} {l['text']}"
            for l in function.gimple
        )
    return "\n".join(lines)


def counts_rows(function):
    """function, count, conditions: one tab-separated row per path."""
    for group in function.groups:
        for conditions in group.when:
            yield f"{function.signature}\t{group.count_text}\t{when_text(conditions)}"


def comparison_text(comparison):
    lines = [
        f"== compare {comparison.name} vs {comparison.reference}: {comparison.verdict.upper()}"
    ]
    lines.extend(f"  !! {reason}" for reason in comparison.reasons)
    if comparison.counterexample:
        inputs = ", ".join(
            f"{k}={v}" for k, v in comparison.counterexample["inputs"].items()
        )
        lines.append(f"  -- counterexample: {inputs or 'no inputs'}")
        lines.extend(f"    {line}" for line in comparison.counterexample["diff"])
    elif comparison.verdict == "identical":
        lines.append(f"  -- issued the same on all {comparison.samples} input vectors")
    return "\n".join(lines)
