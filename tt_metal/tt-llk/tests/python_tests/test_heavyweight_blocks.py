# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the heavyweight golden's warnings and its dispatch.

No kernel, no device. Two things here rot without failing, so each gets a test
rather than a convention.

* A warning blaming the wrong file. A hard-coded ``stacklevel`` is a claim
  about the call stack that any new layer makes wrong, and a wrong one does
  not raise. This package had six such constants and they disagreed.
* A step builder reaching a block by a name built from a string, which fails
  at run time on one path only.
"""

import os
import warnings

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generator.heavyweight.data_transfer_blocks import (
    DataTransferBlocks,
    WormholeArch,
)
from helpers.golden_generator.heavyweight.data_transfer_blocks.units import Fpu
from helpers.golden_generator.heavyweight.data_transfer_blocks.units.warn import (
    _PACKAGE_ROOT,
)
from helpers.golden_generator.heavyweight.operations import (
    BLACKHOLE,
    QUASAR,
    WORMHOLE,
    DataCopyGolden,
    EltwiseBinaryGolden,
    OpConfig,
)
from helpers.llk_params import (
    MathFidelity,
    MathOperation,
    StochasticRounding,
)

N = 1024


# ----------------------------------------------------------------------
# Warning provenance
# ----------------------------------------------------------------------


def _origins(trigger):
    """The files the warnings raised by `trigger` blame."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trigger()
    assert caught, "expected a warning, got none"
    return [os.path.realpath(w.filename) for w in caught]


class _NoSplitArch(WormholeArch):
    """An architecture whose multiplier split is genuinely unmodelled."""

    fpu = Fpu(WormholeArch.l1)


def _unmodelled_split():
    EltwiseBinaryGolden(
        MathOperation.Elwmul,
        MathFidelity.HiFi4,
        blocks=DataTransferBlocks(_NoSplitArch()),
    ).run(
        [torch.rand(N) + 1, torch.rand(N) + 1],
        DataFormat.Float16_b,
        DataFormat.Float16_b,
    )


def _unmodelled_ties():
    # 1 + 2**-8 is an exact tie for a 7-mantissa-bit Float16_b Dest write, and
    # Wormhole leaves its rounding mode unset.
    DataCopyGolden(arch=WORMHOLE).run(
        torch.full((N,), 1.0 + 2.0**-8),
        DataFormat.Float16,
        DataFormat.Float16_b,
        dest_format=DataFormat.Float16_b,
    )


def _stochastic_rounding():
    DataCopyGolden(arch=WORMHOLE).run(
        torch.rand(N),
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        stoch_rnd=StochasticRounding.All,
    )


@pytest.mark.parametrize(
    "trigger",
    [_unmodelled_split, _unmodelled_ties, _stochastic_rounding],
    ids=["unmodelled_split", "unmodelled_ties", "stochastic_rounding"],
)
def test_warning_blames_the_caller(trigger):
    """A warning points outside the blocks, at whoever asked for the transfer.

    Before ``caller_stacklevel``, the stochastic-rounding warning blamed
    ``data_transfer_blocks.py`` rather than the operation, because its
    ``stacklevel`` was one short. Its neighbour in the same function, at the
    same depth, used a different number, so one of the two had to be wrong.
    """
    for origin in _origins(trigger):
        assert not origin.startswith(_PACKAGE_ROOT), (
            f"warning blames {origin}, which is inside data_transfer_blocks/. "
            f"It should blame whoever asked. A stacklevel has gone stale; see "
            f"units/warn.py"
        )


def test_no_hard_coded_stacklevel_comes_back():
    """Every warn site computes who to blame rather than counting frames."""
    offenders = []
    for root, dirs, names in os.walk(_PACKAGE_ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if "stacklevel=" in line and "caller_stacklevel()" not in line:
                        offenders.append(f"{os.path.relpath(path)}:{number}")
    assert not offenders, (
        "hard-coded stacklevel, which goes stale the next time a layer is "
        "added:\n" + "\n".join(offenders)
    )


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


@pytest.mark.parametrize("arch", [QUASAR, WORMHOLE, BLACKHOLE], ids=lambda a: a.name)
def test_both_src_registers_dispatch(arch):
    """SrcA and SrcB both reach a real block, on every architecture.

    These used to be reached as ``getattr(blocks, f"l1_to_{register}")``, so a
    rename failed at run time and only on the path that used that register.
    """
    golden = DataCopyGolden(arch=arch)
    cfg = OpConfig(
        in_formats=[DataFormat.Float16_b, DataFormat.Float16_b],
        out_format=DataFormat.Float16_b,
        dest_format=DataFormat.Float16_b,
        geometry=dict(num_faces=4, face_r_dim=16),
    )
    for register in ("srcA", "srcB"):
        assert golden._l1_to_register(cfg, register, "in0", register, 0, None)
        assert golden._dest_to_register(cfg, register, "dest", register, None)
