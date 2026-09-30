# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only cache prepare step for V4.1 device tests (bead 8y7.18.3): fills the oracle and device-weight caches
outside the device lock, so the device run only reads them.

Usage (plain python, NOT through run_safe_pytest.sh; takes no lock and opens no hardware)::

    python models/demos/deepseek_v3_d_p/tests/v41/prepare_caches.py <test id> [<test id> ...]

with the same test ids as the safe runner, e.g.
``models/demos/deepseek_v3_d_p/tests/v41/test_block_v41.py::test_v41_blocks_on_device_state[blackhole-fabric2d-mesh-2x4-small-single_chunk-stack-full-bf16]``,
``.../test_transformer_v41.py::test_v41_transformer_small[blackhole-fabric2d-mesh-2x4-one_chunk-sharing]``,
``.../test_transformer_v41.py::test_v41_transformer_production[blackhole-fabric2d-mesh-2x4-real-one_chunk]``.

It opens a MOCK mesh (TT_METAL_MOCK_CLUSTER_DESC_PATH -> the LoudBox 8x P150 descriptor: no PCIe access, no
command queues or firmware) and runs each test's own setup function (setup_blocks / setup_small /
setup_production, + reference_data for the transformer gate), so the caches are written by the canonical code:
mock-written weight-cache files are byte-identical to device-written ones. Idempotent: a second run is all hits.
"""

import os
import re
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[5]
MOCK_CLUSTER = _REPO / "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_8xP150.yaml"
os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = str(MOCK_CLUSTER)  # before ttnn is imported: never real hardware

import pytest  # noqa: E402

import ttnn  # noqa: E402

_ID = re.compile(r"(?P<file>[^:]+)::(?P<test>\w+)\[blackhole-fabric2d-mesh-(?P<sp>\d)x(?P<tp>\d)-(?P<params>[^\]]+)\]")
_CHUNKS = {"single_chunk": 1, "two_chunks": 2, "one_chunk": 1}


def _prepare(mesh, test: str, params: list[str]) -> str:
    if test == "test_v41_blocks_on_device_state":
        from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import setup_blocks

        weights, chunks, schedule, prompt, kv_format = params
        setup = setup_blocks(mesh, weights, _CHUNKS[chunks], schedule, prompt, kv_format)
        return f"reference built: {setup.reference.built}"
    from models.demos.deepseek_v3_d_p.tests.v41 import test_transformer_v41 as T

    if test == "test_v41_transformer_small":
        case, schedule = params
        model, spec, tokens, reference, _ = T.setup_small(mesh, case, schedule)
    elif test == "test_v41_transformer_production":
        weights, chunks = params
        setup = T.setup_production(mesh, weights, chunks)
        if setup is None:
            return "skipped: checkpoint not downloaded"
        model, spec, tokens, reference = setup
        if chunks in T.KV_FORMAT_CASES:
            T.kv_format_reference(spec, tokens, reference)
        else:
            T.reference_data(spec, tokens, reference, T.last_chunk_rows(chunks))
        return f"reference built: {reference.built}"
    else:
        raise ValueError(f"no prepare step for {test}")
    T.reference_data(spec, tokens, reference)
    built = getattr(reference, "built", True)
    return f"reference built: {built}"


def main(ids: list[str]) -> None:
    parsed = []
    for test_id in ids:
        m = _ID.fullmatch(test_id)
        if m is None:
            raise SystemExit(f"cannot parse test id {test_id!r}")
        parsed.append(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    meshes = {}
    try:
        for m in parsed:
            shape = (int(m["sp"]), int(m["tp"]))
            if shape not in meshes:
                meshes[shape] = ttnn.open_mesh_device(ttnn.MeshShape(*shape))
            start = time.perf_counter()
            try:
                note = _prepare(meshes[shape], m["test"], m["params"].split("-"))
            except pytest.skip.Exception as skip:
                note = f"skipped: {skip}"
            print(f"prepare {m['test']}[{m['params']}]: {note}, {time.perf_counter() - start:.1f}s", flush=True)
    finally:
        for mesh in meshes.values():
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main(sys.argv[1:])
