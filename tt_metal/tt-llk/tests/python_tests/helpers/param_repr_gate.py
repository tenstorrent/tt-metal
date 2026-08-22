# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Value-complete-repr tripwire for variant-hash inputs (lane FQ, FO-2).

The variant hash (``TestConfig.generate_variant_hash``) keys the ``str()`` of
the templates/runtimes lists, i.e. every parameter's ``__repr__``. A parameter
class that subclasses the ``@dataclass`` base WITHOUT ``@dataclass`` and stores
its values in a hand-written ``__init__`` inherits the base's EMPTY dataclass
repr: ``repr(ReciprocalImpl(0)) == repr(ReciprocalImpl(2)) == "ReciprocalImpl()"``.
The hash is then value-blind for that parameter and the ``.build_complete``
fast path reuses the FIRST impl's ELF for every other impl in the same
ARTEFACTS_DIR — a measurement-integrity defect (lane FO finding; the FI/EK
collision was only caught downstream by the .text gate).

This module is import-free of device libraries on purpose so the corpus
selftest can load it by path and prove the tripwire fails loudly.
"""


class ParamReprError(RuntimeError):
    """A template/runtime parameter's repr hides its values from the hash."""


def empty_repr_params(params) -> list:
    """Params whose repr equals their class's empty-instance repr while the
    instance actually carries state.

    A dataclass repr is ``Cls(field=value, ...)``; the empty-instance repr is
    ``Cls()``. A parameterless param class (no fields, no instance state, e.g.
    EN_DEST_REUSE) legitimately reprs as ``Cls()`` and is NOT flagged — it has
    no values a hash could miss.
    """
    offenders = []
    for param in params:
        state = [k for k in vars(param) if not k.startswith("_")]
        if state and repr(param) == f"{type(param).__qualname__}()":
            offenders.append(param)
    return offenders


def assert_value_complete_reprs(params, context: str = "") -> None:
    """Refuse (loudly) any param whose values are invisible to ``repr``.

    Raises ParamReprError naming every offending class, its hidden attributes,
    and the fix — this must stay FATAL: a value-blind repr silently reuses the
    wrong impl's ELF via the .build_complete fast path.
    """
    offenders = empty_repr_params(params)
    if not offenders:
        return
    lines = [
        f"variant-hash tripwire{f' ({context})' if context else ''}: "
        "parameter repr(s) hide their values from the variant hash — every "
        "value of such a parameter hashes IDENTICALLY and .build_complete "
        "reuses the wrong impl's ELF within one ARTEFACTS_DIR.",
    ]
    for param in offenders:
        hidden = {k: v for k, v in vars(param).items() if not k.startswith("_")}
        lines.append(
            f"  {type(param).__qualname__}: repr is "
            f"'{type(param).__qualname__}()' but the instance carries {hidden}"
        )
    lines.append(
        "Fix: decorate the class with @dataclass and declare its values as "
        "annotated fields (do not hand-write __init__); see "
        "helpers/test_variant_parameters.py for the pattern."
    )
    raise ParamReprError("\n".join(lines))
