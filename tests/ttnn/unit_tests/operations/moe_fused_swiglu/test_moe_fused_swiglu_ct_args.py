# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""The compile-time argument order is written twice; this asserts the two copies agree.

The host builds each kernel's argument vector from `KERNEL_CT_ORDER`; the kernel reads it back
through an enum generated from the X-macro lists in `kernels/moe_fused_swiglu_ct_args.hpp`. A
disagreement is not a compile error — it silently hands every argument to the wrong name, which is
the class of bug the whole named-argument scheme exists to remove. So the header is parsed here
and compared element by element.

No device needed.
"""

import pathlib
import re

import pytest

from ttnn.operations.moe_fused_swiglu.moe_fused_swiglu_program_descriptor import KERNEL_CT_ORDER

HEADER = (
    pathlib.Path(__file__).resolve().parents[5]
    / "ttnn/ttnn/operations/moe_fused_swiglu/kernels/moe_fused_swiglu_ct_args.hpp"
)


def header_order(kernel):
    """The X(NAME) sequence of one MOE_<KERNEL>_CT_ARGS list, in declaration order."""
    text = HEADER.read_text()
    macro = f"#define MOE_{kernel.upper()}_CT_ARGS(X)"
    start = text.index(macro) + len(macro)
    # the list ends at the first line that does not continue with a backslash
    body, line_start = [], start
    for line in text[start:].split("\n"):
        body.append(line)
        if not line.rstrip().endswith("\\"):
            break
    return tuple(re.findall(r"X\((\w+)\)", "\n".join(body)))


@pytest.mark.parametrize("kernel", sorted(KERNEL_CT_ORDER))
def test_ct_arg_order_matches_header(kernel):
    assert HEADER.exists(), f"missing {HEADER}"
    from_header = header_order(kernel)
    from_host = KERNEL_CT_ORDER[kernel]
    assert from_header, f"parsed no arguments for {kernel} out of {HEADER.name}"
    assert from_header == from_host, (
        f"{kernel} compile-time arg order disagrees between the host and the kernel header.\n"
        f"  host  : {from_host}\n"
        f"  header: {from_header}\n"
        f"  first divergence at index "
        f"{next((i for i, (a, b) in enumerate(zip(from_host, from_header)) if a != b), min(len(from_host), len(from_header)))}"
    )


def test_no_duplicate_names():
    """A repeated name would make `_ct_args` silently emit the same value twice."""
    for kernel, order in KERNEL_CT_ORDER.items():
        dupes = {n for n in order if order.count(n) > 1}
        assert not dupes, f"{kernel} lists {sorted(dupes)} more than once"
