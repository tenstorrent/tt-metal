# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in structural experiment; never changes production construction defaults.

Compile narrowly edited copies of the current methods in private namespaces so
the probe follows the selected math exactly. Anchors must match once or setup
fails. No native kernel, dtype, reduction tree, or constructor knob is changed.
"""

import hashlib
import inspect
import textwrap
import types


def _edited(method, edits):
    original = method.__func__
    source = textwrap.dedent(inspect.getsource(original))
    before = source
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError(f"Structural probe source anchor changed in {original.__qualname__}: {old!r}")
        source = source.replace(old, new, 1)
    namespace = dict(original.__globals__)
    exec(compile(source, f"<stage7-structural:{original.__qualname__}>", "exec"), namespace)
    function = namespace[original.__name__]
    return types.MethodType(function, method.__self__), {
        "method": original.__qualname__,
        "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "candidate_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_source": source,
    }


def install(decoder, *, recurrent_permute=False, collective_padding=False):
    if not (recurrent_permute or collective_padding):
        raise ValueError("Select at least one structural hypothesis")
    if getattr(decoder, "stage7_structural_probe", None) is not None:
        raise ValueError("Structural probe already installed")
    evidence = {"recurrent_permute": recurrent_permute, "collective_padding": collective_padding, "edits": []}
    if recurrent_permute and decoder.kind == "linear_attention":
        decoder._structural_original_sum = decoder._recurrent_sum
        decoder._structural_original_step = decoder._recurrent_step
        decoder._recurrent_sum_head_rows, row = _edited(
            decoder._recurrent_sum,
            [("return ttnn.permute(total[:, :, :8, :], (0, 2, 1, 3))", "return total[:, :, :8, :]")],
        )
        evidence["edits"].append(row)
        decoder._recurrent_step, row = _edited(
            decoder._recurrent_step,
            [
                ("output = self._recurrent_sum(", "output = self._recurrent_sum_head_rows("),
                ("return (ttnn.permute(output, (0, 2, 1, 3)), new)", "return (output, new)"),
            ],
        )
        evidence["edits"].append(row)
    if collective_padding:
        if decoder.replicated_collective != "direct" or decoder.residual_layout != "replicated":
            raise ValueError("Padding probe requires the current replicated/direct collective contract")
        decoder._structural_original_reduce = decoder._reduce
        decoder._reduce, row = _edited(
            decoder._reduce,
            [
                (
                    "        if rows < 32:\n"
                    "            local = ttnn.pad(local, [(0, 0), (0, 0), (0, 32 - rows), (0, 0)], 0.0)\n",
                    "",
                ),
                ("return reshape(total[:, :, :rows, :], (1, rows, 2048))", "return reshape(total, (1, rows, 2048))"),
            ],
        )
        evidence["edits"].append(row)
    decoder.stage7_structural_probe = evidence
    return decoder
