# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compare <name> against reference_<name> by running both on the same concrete inputs.

Parameters are matched by position, so renaming them changes nothing. Every input
vector runs both functions; any difference in what they issue is reported with
that input as a counterexample.
"""

import difflib

from hal_explorer.analysis import samples as sampling
from hal_explorer.analysis.interp import Interpreter
from hal_explorer.model import Comparison

READ_MODELS = ((0, "zero"), ("derived", "a derived value per read"))
STEP_BUDGET = 1_000_000
MIN_VECTORS = 16

REFERENCE_PREFIX = "reference_"


def partner_name(name):
    """ns::reference_f for ns::f."""
    scope, _, last = name.rpartition("::")
    return f"{scope}::{REFERENCE_PREFIX}{last}" if scope else REFERENCE_PREFIX + last


def is_reference(name):
    return name.rpartition("::")[2].startswith(REFERENCE_PREFIX)


def pair(contexts):
    """[(context, reference context)] plus Comparisons for references without a partner."""
    by_signature = {}
    for context in contexts:
        types = tuple(t for t, _ in context.function.parameters)
        by_signature[(context.function.name, types)] = context
    pairs, unpaired = [], []
    for (name, types), context in by_signature.items():
        if not is_reference(name):
            continue
        scope, _, last = name.rpartition("::")
        actual_name = (f"{scope}::" if scope else "") + last[len(REFERENCE_PREFIX) :]
        actual = by_signature.get((actual_name, types))
        if actual is None:
            others = [n for n, t in by_signature if n == actual_name]
            reason = (
                f"{actual_name} exists but with different parameter types"
                if others
                else f"no function named {actual_name} was emitted (missing, misspelled, or inline/static and unused)"
            )
            unpaired.append(
                Comparison(
                    actual_name, context.function.signature, "unpaired", 0, [reason]
                )
            )
        else:
            pairs.append((actual, context))
    return pairs, unpaired


def compare(actual, reference, actual_result, reference_result):
    """Comparison of two analysed functions with the same parameter types.

    Both run on the same input vectors, once with every MMIO read returning zero and once
    with each read returning a fixed derived value. Values loaded through pointer
    parameters are derived from the parameter position and offset on both sides.
    """
    name, ref_name = actual.function.signature, reference.function.signature
    reasons = []
    for label, result in (("rewrite", actual_result), ("reference", reference_result)):
        for warning in result.warnings:
            reasons.append(f"{label}: {warning['message']}")
    globals_found = {**reference.globals, **actual.globals}
    space = sampling.input_space(reference.function.parameters, globals_found)
    names = [n for _, n in actual.function.parameters] + sorted(globals_found)
    positional = [n for _, n in reference.function.parameters]
    actual_positional = [n for _, n in actual.function.parameters]
    vectors = sampling.vectors(space)
    stopped, compared, steps = 0, 0, 0
    for read_value, read_label in READ_MODELS:
        machines = [
            Interpreter(
                c.function,
                c.loops,
                c.tables,
                c.sources,
                c.definitions,
                read_value,
                derive_loads=True,
            )
            for c in (reference, actual)
        ]
        for inputs in vectors:
            if steps + sum(m.steps for m in machines) > STEP_BUDGET:
                break
            compared += 1
            actual_inputs = rename(inputs, positional, actual_positional)
            ref_traces = machines[0].run(inputs)
            act_traces = machines[1].run(actual_inputs)
            if any(not t.complete for t in ref_traces + act_traces):
                stopped += 1
            if sorted(map(trace_keys, ref_traces)) != sorted(
                map(trace_keys, act_traces)
            ):
                shown = {
                    n: actual_inputs.get(n)
                    for n in names
                    if isinstance(actual_inputs.get(n), (int, type(None)))
                }
                example = {
                    "inputs": shown,
                    "diff": first_diff(ref_traces, act_traces),
                    "mmio_reads": read_label,
                }
                return Comparison(name, ref_name, "differs", compared, reasons, example)
        steps += sum(m.steps for m in machines)
    total = len(vectors) * len(READ_MODELS)
    if stopped:
        reasons.append(
            f"{stopped} of {compared} runs stopped at a step or branch limit; both sides were compared up to that point"
        )
    if compared < total:
        reasons.append(
            f"compared {compared} of {total} runs before the step budget ran out"
        )
    inconclusive = compared < min(MIN_VECTORS, total) or any(
        r.startswith(("rewrite:", "reference:")) for r in reasons
    )
    return Comparison(
        name,
        ref_name,
        "inconclusive" if inconclusive else "identical",
        compared,
        reasons,
    )


def rename(inputs, from_names, to_names):
    """The same input vector with parameters renamed by position; globals keep their names."""
    renamed = {k: v for k, v in inputs.items() if k not in from_names}
    for old, new in zip(from_names, to_names):
        renamed[new] = inputs[old]
    return renamed


def trace_keys(trace):
    return tuple(item.key() for item in trace.items)


def first_diff(reference_traces, actual_traces):
    for ref, act in zip(reference_traces, actual_traces):
        if trace_keys(ref) != trace_keys(act):
            return list(
                difflib.unified_diff(
                    [i.line() for i in ref.items],
                    [i.line() for i in act.items],
                    lineterm="",
                    n=2,
                )
            )[2:]
    longer = (
        reference_traces
        if len(reference_traces) > len(actual_traces)
        else actual_traces
    )
    return [
        f"the two functions branch differently on MMIO reads ({len(reference_traces)} vs {len(actual_traces)} outcomes)"
    ] + [i.line() for i in longer[-1].items]
