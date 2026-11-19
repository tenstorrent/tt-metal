# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regression for the sender-coordinate contract behind the triaged hang."""

import ast
from pathlib import Path


def test_exact_k_sender_coordinates():
    source = Path(__file__).resolve().parents[1] / "tt/multichip_decoder.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "exact_k_shards")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    planner = namespace["exact_k_shards"]
    for k in [1024, 3072, 4096]:
        for requested in [16, 32, 64]:
            storage, shard = planner(k, requested)
            assert storage <= requested and storage * shard == k // 32
            for block in range(1, shard + 1):
                if shard % block:
                    continue
                blocks = k // 32 // block
                per_shard = blocks // storage
                assert blocks // per_shard == storage
                assert per_shard * block == shard
    assert planner(3072, 64) == (48, 2)
    # The original oversharded layout made the kernel address a 96-entry table
    # even though the factory supplied only 64 X and 64 Y coordinates.
    assert (96 // (96 // 64)) == 96


if __name__ == "__main__":
    test_exact_k_sender_coordinates()
    print("EXACT_K_PLANNER_PASS")
