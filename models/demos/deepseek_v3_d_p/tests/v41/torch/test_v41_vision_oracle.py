# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the oracle package's image encoder oracle (``vision_oracle``; host only, no device).

The oracle output is checked against the vendored ``vision.ViT`` / ``vision.Aligner`` built independently of the
oracle, the token grid against hand-computed sizes, the disk cache (miss/hit, key sensitivity, bit-identical
reload), synthetic weight derivation, checkpoint weight loading (needs shard 1) and the test-image inputs.
Small vision dims except for the weight and image checks.
"""

from dataclasses import replace

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as o
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import vision

SMALL = replace(o.vision_args(), vision_n_layers=2, vision_dim=64, vision_n_heads=4, vision_inter_dim=96, dim=48)


@pytest.fixture(autouse=True)
def _few_threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(prev)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "CACHE_DIR", tmp_path)
    return tmp_path


def _shard1_present() -> bool:
    return (o.HF_SNAPSHOT / "model-00001-of-00048.safetensors").is_file()


@pytest.mark.parametrize("n_h, n_w, n_llm", [(4, 5, (2, 2)), (3, 3, (1, 1)), (7, 2, (3, 1))], ids=["4x5", "3x3", "7x2"])
def test_vision_oracle_matches_reference_modules(cache, n_h, n_w, n_llm):
    """Grids that are and are not multiples of the 3x3 aligner downsample (zero padding)."""
    patches = o.random_patches(n_h, n_w, SMALL, seed=3)
    got = o.vision_oracle(patches, n_h, n_w, args=SMALL, seed=5)

    weights = o.synthetic_vision_weights(SMALL, 5)
    with v41.set_dtype(torch.bfloat16):
        vit, aligner = vision.ViT(SMALL), vision.Aligner(SMALL)
    for prefix, module in (("vision.", vit), ("aligner.", aligner)):
        module.load_state_dict({k: weights[prefix + k].to(v.dtype) for k, v in module.state_dict().items()})
    with torch.inference_mode(), v41.set_dtype(torch.bfloat16):
        hidden = vit.eval()(patches, n_h, n_w)
        aligned = aligner.eval()(hidden, n_h, n_w)

    assert got["meta"] == {"n_h": n_h, "n_w": n_w, "n_llm_h": n_llm[0], "n_llm_w": n_llm[1]}
    assert got["hidden"].dtype == torch.bfloat16 and got["hidden"].shape == (n_h * n_w, SMALL.vision_dim)
    assert got["aligned"].shape == (n_llm[0] * n_llm[1], SMALL.dim)
    assert torch.equal(got["hidden"], hidden) and torch.equal(got["aligned"], aligned)
    assert vit.norm.weight.dtype == torch.float32  # module dtypes as upstream builds them


def test_vision_oracle_cache_hit_miss_and_keys(cache, monkeypatch, expect_error):
    patches = o.random_patches(4, 5, SMALL, seed=3)
    miss = o.vision_oracle(patches, 4, 5, args=SMALL)
    path = o.vision_cache_path(SMALL, patches, 4, 5, 0, None)
    assert path.is_file() and sorted(cache.iterdir()) == [path]

    def no_run(*args):
        raise AssertionError("cache hit must not run the reference")

    monkeypatch.setattr(o, "build_vision_reference", no_run)
    hit = o.vision_oracle(patches, 4, 5, args=SMALL)
    assert hit["meta"] == miss["meta"]
    assert torch.equal(hit["hidden"], miss["hidden"]) and torch.equal(hit["aligned"], miss["aligned"])

    other_keys = {
        o.vision_cache_path(SMALL, patches, 4, 5, 1, None),  # seed
        o.vision_cache_path(SMALL, o.random_patches(4, 5, SMALL, seed=4), 4, 5, 0, None),  # patches
        o.vision_cache_path(SMALL, patches, 5, 4, 0, None),  # grid
        o.vision_cache_path(replace(SMALL, vision_rope_theta=500.0), patches, 4, 5, 0, None),  # args
        o.vision_cache_path(SMALL, patches, 4, 5, 0, o.HF_SNAPSHOT),  # weights source
    }
    monkeypatch.setattr(o, "PACKAGE_VERSION", o.PACKAGE_VERSION + 1)
    other_keys.add(o.vision_cache_path(SMALL, patches, 4, 5, 0, None))
    assert path not in other_keys and len(other_keys) == 6

    with expect_error(ValueError, "patches must be"):
        o.vision_oracle(patches, 5, 5, args=SMALL)


def test_synthetic_vision_weights_are_per_tensor():
    w = o.synthetic_vision_weights(SMALL, 0)
    assert sorted(w) == sorted(o.vision_weight_shapes(SMALL))
    assert all(t.dtype == torch.bfloat16 and t.shape == o.vision_weight_shapes(SMALL)[n] for n, t in w.items())
    # a tensor depends on (seed, name) only: the same with fewer layers, different with another seed
    fewer = o.synthetic_vision_weights(replace(SMALL, vision_n_layers=1), 0)
    assert all(torch.equal(t, w[n]) for n, t in fewer.items())
    other = o.synthetic_vision_weights(SMALL, 1)
    assert not torch.equal(other["vision.blocks.0.attn.wqkv.weight"], w["vision.blocks.0.attn.wqkv.weight"])
    # distributions: norms around 1, small biases, matrices of RMS 1/sqrt(fan_in)
    assert abs(w["vision.norm.weight"].float().mean() - 1) < 0.05
    assert w["aligner.w1.bias"].float().std() < 0.03
    rms = w["vision.blocks.1.mlp.w2.weight"].float().pow(2).mean().sqrt()
    assert abs(rms * SMALL.vision_inter_dim**0.5 - 1) < 0.05


@pytest.mark.skipif(not _shard1_present(), reason="needs V4.1 checkpoint shard 1 (vision.*, aligner.*)")
def test_checkpoint_vision_weights_match_reference_layout():
    args = o.vision_args()
    w = o.checkpoint_vision_weights(o.HF_SNAPSHOT)
    shapes = o.vision_weight_shapes(args)
    assert sorted(w) == sorted(shapes) and len(w) == 3 + 8 * args.vision_n_layers + 4
    assert all(t.shape == shapes[n] and t.dtype == torch.bfloat16 for n, t in w.items())
    vit, aligner = o.build_vision_reference(args, w)  # strict load
    assert torch.equal(vit.blocks[31].attn.wo.bias, w["vision.blocks.31.attn.wo.bias"])
    assert torch.equal(aligner.w2.weight, w["aligner.w2.weight"])


def test_test_images_and_patch_grids():
    png = o.synthetic_image(640, 480, 1)
    assert png == o.synthetic_image(640, 480, 1) and png != o.synthetic_image(640, 480, 2)
    # grids image_processor plans for the released args (<= 1024 LLM tokens)
    for (w, h), (n_h, n_w) in {(640, 480): (35, 46), (1000, 1000): (72, 72), (1920, 1080): (69, 122)}.items():
        patches, got_h, got_w = o.image_patches(o.synthetic_image(w, h, 1))
        assert (got_h, got_w) == (n_h, n_w) and patches.shape == (n_h * n_w, 3, 14, 14)
        assert patches.dtype == torch.bfloat16 and patches.abs().max() <= 1
    grid = o.random_patches(4, 5, seed=1)
    assert grid.shape == (20, 3, 14, 14) and torch.equal(grid, o.random_patches(4, 5, seed=1))
