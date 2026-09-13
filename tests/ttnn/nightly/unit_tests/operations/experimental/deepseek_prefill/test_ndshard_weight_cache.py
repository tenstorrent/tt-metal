# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The ND-sharded weight cache must skip the reshard without changing a single value.

Builds the same expert twice against one cache directory -- first a miss, which reshards and writes
the ND cache, then a hit, which loads the final placement straight from disk. The hit is only worth
having if it is bit-identical to the miss, so that is what is asserted.
"""

import pathlib

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

_MODELS = [("kimi_k26", 7168, 2048), ("gptoss_120b", 2880, 2880), ("kimi_k3", 3584, 3072)]


def _build(device, emb, hidden, weights, cache_dir, sharded):
    idx = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.uint32
    )
    return TtRoutedExpert(
        mesh_device=device,
        experts_per_chip=1,
        global_expert_idx_table=idx,
        emb_dim=emb,
        hidden_dim=hidden,
        max_tokens=5120,
        torch_weights=[weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        weights_dram_sharded=sharded,
        activation=ttnn.RoutedExpertActivation.Silu,
        weight_cache_path=pathlib.Path(cache_dir),
        cache_name_prefix="ndcache_probe",
    )


@pytest.mark.parametrize("name, emb, hidden", [pytest.param(*m, id=m[0]) for m in _MODELS])
@pytest.mark.skipif(not is_blackhole(), reason="shard widths are sized for the Blackhole grid")
def test_ndshard_weight_cache_is_bit_exact(device, tmp_path, name, emb, hidden):
    torch.manual_seed(0)
    weights = {
        "gate_proj": torch.randn(hidden, emb) * 0.02,
        "up_proj": torch.randn(hidden, emb) * 0.02,
        "down_proj": torch.randn(emb, hidden) * 0.02,
    }
    cache = tmp_path / name

    miss = _build(device, emb, hidden, weights, cache, sharded=True)
    written = sorted(p.name for p in cache.glob("*ndshard*"))
    print(f"CACHE {name} wrote {len(written)} ND files: {written}")
    assert len(written) == 3, f"expected one ND cache file per projection, got {written}"

    hit = _build(device, emb, hidden, weights, cache, sharded=True)

    for role in ("gate_projs", "up_projs", "down_projs"):
        a, b = getattr(miss, role)[0], getattr(hit, role)[0]
        assert "created_with_nd_shard_spec=1" in str(b.memory_config()), f"{role}: hit is not ND-sharded"
        assert tuple(a.memory_config().nd_shard_spec.shard_shape) == tuple(b.memory_config().nd_shard_spec.shard_shape)
        ta, tb = ttnn.to_torch(a), ttnn.to_torch(b)
        print(
            f"CACHE {name} {role}: equal={torch.equal(ta, tb)} shard={tuple(b.memory_config().nd_shard_spec.shard_shape)}"
        )
        assert torch.equal(ta, tb), f"{role}: cache hit differs from the resharded build"


@pytest.mark.parametrize("name, emb, hidden", [pytest.param(*m, id=m[0]) for m in _MODELS[:1]])
@pytest.mark.skipif(not is_blackhole(), reason="shard widths are sized for the Blackhole grid")
def test_interleaved_path_writes_no_nd_cache(device, tmp_path, name, emb, hidden):
    """weights_dram_sharded=False must be untouched by any of this."""
    torch.manual_seed(0)
    weights = {
        "gate_proj": torch.randn(hidden, emb) * 0.02,
        "up_proj": torch.randn(hidden, emb) * 0.02,
        "down_proj": torch.randn(emb, hidden) * 0.02,
    }
    cache = tmp_path / f"{name}_interleaved"
    built = _build(device, emb, hidden, weights, cache, sharded=False)
    assert list(cache.glob("*ndshard*")) == [], "interleaved build must not write an ND cache"
    assert built.gate_projs[0].memory_config().nd_shard_spec is None, "interleaved build must stay interleaved"


@pytest.mark.parametrize("name, emb, hidden", [pytest.param(*m, id=m[0]) for m in _MODELS[:2]])
@pytest.mark.skipif(not is_blackhole(), reason="shard widths are sized for the Blackhole grid")
def test_build_ttnn_cache_generates_nd_files(device, tmp_path, name, emb, hidden):
    """build_ttnn_cache(dram_sharded=True) must pre-populate what a first run would otherwise write."""
    torch.manual_seed(0)
    weights = {
        "gate_proj": torch.randn(hidden, emb) * 0.02,
        "up_proj": torch.randn(hidden, emb) * 0.02,
        "down_proj": torch.randn(emb, hidden) * 0.02,
    }
    cache = tmp_path / f"{name}_gen"
    cache.mkdir(parents=True, exist_ok=True)

    TtRoutedExpert.build_ttnn_cache([weights], 1, device, ttnn.bfloat4_b, cache, "ndcache_probe", dram_sharded=True)
    nd_files = sorted(p.name for p in cache.glob("*ndshard*"))
    print(f"GEN {name} produced {len(nd_files)} ND files: {nd_files}")
    assert len(nd_files) == 3, f"expected 3 ND cache files, got {nd_files}"

    # A module built against this cache must now load the ND placement, not reshard into it.
    built = _build(device, emb, hidden, weights, cache, sharded=True)
    ref = _build(device, emb, hidden, weights, tmp_path / f"{name}_ref", sharded=True)
    for role in ("gate_projs", "up_projs", "down_projs"):
        got = getattr(built, role)[0]
        assert "created_with_nd_shard_spec=1" in str(got.memory_config()), f"{role}: not ND-sharded"
        assert torch.equal(ttnn.to_torch(got), ttnn.to_torch(getattr(ref, role)[0])), f"{role}: differs"
    print(f"GEN {name}: pre-populated cache loads ND and matches a fresh reshard")


@pytest.mark.skipif(not is_blackhole(), reason="shard widths are sized for the Blackhole grid")
def test_check_cache_complete_distinguishes_nd(device, tmp_path):
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    torch.manual_seed(0)
    emb, hidden = 7168, 2048
    weights = {
        "gate_proj": torch.randn(hidden, emb) * 0.02,
        "up_proj": torch.randn(hidden, emb) * 0.02,
        "down_proj": torch.randn(emb, hidden) * 0.02,
    }

    def complete(path, sharded):
        init_checker(path)  # the checker snapshots the dir, so re-init after every write
        return TtRoutedExpert.check_cache_complete(path, "cc", 1, ttnn.bfloat4_b, dram_sharded=sharded)

    # Interleaved-only cache: complete for interleaved, incomplete for ND.
    plain = tmp_path / "plain"
    plain.mkdir()
    TtRoutedExpert.build_ttnn_cache([weights], 1, device, ttnn.bfloat4_b, plain, "cc")
    print(f"CC interleaved-only: plain={complete(plain, False)} nd={complete(plain, True)}")
    assert complete(plain, False) is True
    assert complete(plain, True) is False, "ND cache reported complete without any ND file"

    # ND cache: complete both ways.
    both = tmp_path / "both"
    both.mkdir()
    TtRoutedExpert.build_ttnn_cache([weights], 1, device, ttnn.bfloat4_b, both, "cc", dram_sharded=True)
    print(f"CC nd-built: plain={complete(both, False)} nd={complete(both, True)}")
    assert complete(both, False) is True
    assert complete(both, True) is True

    # The bug this fixes: ND files alone must NOT satisfy the interleaved check.
    for f in both.glob("*_layout_*.tensorbin"):
        f.unlink()
    print(f"CC nd-only (layout files removed): plain={complete(both, False)} nd={complete(both, True)}")
    assert complete(both, False) is False, "ND files satisfied the interleaved check"
