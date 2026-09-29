# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test templates the tests step fills: one component test per (block type, step), one swap test per column of the
swap order. The rendered files live in <model_dir>/tests/bringup/ and are frozen before any implementation exists.
The test role may edit a rendered file (comparison mode, threshold, extra checks) before freezing, never after.
A swap test gates every swapped step's own output (CHECKS = "steps", F49), so by default swap tests are frozen without
a test-role review (spec agents.swap_review opts back in).
"""

from __future__ import annotations

from pathlib import Path

HEADER = """# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""

COMPONENT = (
    HEADER
    + '''"""{title}

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Golden input and output: layer {layer}, the component rung's last dumped chunk (see testing/harness.component_golden).
"""

from models.demos.common.bringup.testing.component import run_component_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
STEP = {step!r}
LAYER = {layer}
COMPARE = {mode!r}  # None = PCC for float outputs, exact match for integer outputs
THRESHOLD = {thr!r}  # None = spec thresholds.component (default 0.99)


@mesh_parametrize
def test_component(mesh_device):
    assert run_component_test(S, STEP, LAYER, mesh_device, COMPARE, THRESHOLD)
'''
)

SWAP = (
    HEADER
    + '''"""{title}

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer {layer} ({block_type}) with these steps on the device and the rest on the CPU reference:
{swapped_list}
"""

from models.demos.common.bringup.testing.component import run_swap_test
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BLOCK_TYPE = {block_type!r}
SWAPPED = {swapped!r}
THRESHOLD = {thr!r}  # None = spec thresholds.block (default 0.98)
CHECKS = "steps"  # also gate every swapped step's own output (testing/component.py); None = block out only


@mesh_parametrize
def test_swap(mesh_device):
    assert run_swap_test(S, BLOCK_TYPE, SWAPPED, mesh_device, THRESHOLD, CHECKS)
'''
)


def tests_dir(spec) -> Path:
    return spec.model_dir / "tests" / "bringup"


def component_test_path(spec, block_type: str, step: str) -> Path:
    return tests_dir(spec) / f"test_c_{block_type}_{step}.py"


def swap_test_path(spec, block_type: str, n: int, step: str) -> Path:
    return tests_dir(spec) / f"test_swap_{block_type}_{n:02d}_{step}.py"


def _write(path: Path, text: str, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for d in (path.parent, path.parent.parent):
        if not (d / "__init__.py").exists():
            (d / "__init__.py").write_text(HEADER)
    path.write_text(text)
    return path


def render_component_test(spec, block_type: str, step: str, mode=None, thr=None, overwrite=False) -> Path:
    layer = spec.representative_layer(block_type)
    title = f"Component test: {step} of block type {block_type} (layer {layer}) on device vs golden."
    return _write(
        component_test_path(spec, block_type, step),
        COMPONENT.format(title=title, step=step, layer=layer, mode=mode, thr=thr),
        overwrite,
    )


def render_swap_test(spec, block_type: str, swapped: list[str], thr=None, overwrite=False) -> Path:
    layer = spec.representative_layer(block_type)
    title = f"Swap test {len(swapped)}: block type {block_type} (layer {layer}) with {swapped[-1]} swapped in last."
    text = SWAP.format(
        title=title,
        layer=layer,
        block_type=block_type,
        swapped=list(swapped),
        thr=thr,
        swapped_list="\n".join(f"    {s}" for s in swapped),
    )
    return _write(swap_test_path(spec, block_type, len(swapped), swapped[-1]), text, overwrite)
