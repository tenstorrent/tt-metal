# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The two registered Kimi-K3.1 FP64 chain goldens the K3.1 scorers open by id are complete, of the producer version
the scorers were checked on, and readable by other users (CI). Reads only metadata and the token ids. CPU only."""

import stat
from pathlib import Path

from models.demos.deepseek_v3_d_p.tests.accuracy.golden_registry import MANIFEST, TOKEN_IDS
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import GOLDEN_1M, open_chain, segment_path, state_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_deep_chain import GOLDEN_256K, open_deep_chain, pass_dir, passes


def _unreadable(files: list[Path]) -> list[str]:
    """Files some other user cannot read, by mode bits alone: every directory from ``/`` down world-traversable,
    the file world-readable."""
    bad = []
    for path in files:
        path = path.resolve()
        if not (all(p.stat().st_mode & stat.S_IXOTH for p in path.parents) and path.stat().st_mode & stat.S_IROTH):
            bad.append(str(path))
    return bad


def test_registered_k31_layers_0_3_chain_serves_the_scorers() -> None:
    golden = open_chain(GOLDEN_1M)  # registered, complete, fp64_chain version 2
    assert golden.rows == 1044480 and golden.token_ids().shape == (golden.rows,)
    assert {"embed_tokens", "layers.1.experts", "layers.3"} <= set(golden.identity["weights_sha256"])
    assert {"moe_output_1", "routing_ids_2", "dsa_input_3", "topk_indices_3", "latent", "attn_output_3"} <= set(
        golden.manifest["products"]
    )
    segments = range(len(golden.manifest["segments"]))
    files = [golden.path / MANIFEST, golden.path / TOKEN_IDS]
    files += [p for i in segments for p in (segment_path(golden.path, i), state_path(golden.path, i))]
    missing = [str(p) for p in files if not p.is_file()]
    assert not missing, f"files missing: {missing[:10]}"
    assert not _unreadable(files), f"not readable by other users: {_unreadable(files)[:10]}"


def test_registered_k31_layers_0_7_deep_chain_serves_the_scorers() -> None:
    golden = open_deep_chain(GOLDEN_256K)  # registered, every pass and row segment complete, version 2
    assert golden.rows == 262144 and golden.token_ids().shape == (golden.rows,)
    assert [(p["layer"], p["site"]) for p in golden.manifest["passes"]] == passes(8)
    assert {"embed_tokens", "attn.layers.3", "ffn.layers.7.experts"} <= set(golden.identity["weights_sha256"])
    segments = len(golden.manifest["segments"])
    files = [golden.path / MANIFEST, golden.path / TOKEN_IDS]
    files += [pass_dir(golden.path, l, s) / f"segment_{i:06d}.pt" for l, s in passes(8) for i in range(segments)]
    missing = [str(p) for p in files if not p.is_file()]
    assert not missing, f"files missing: {missing[:10]}"
    assert not _unreadable(files), f"not readable by other users: {_unreadable(files)[:10]}"
