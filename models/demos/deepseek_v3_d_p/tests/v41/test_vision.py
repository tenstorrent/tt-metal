# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 ViT x32 + aligner (bead 10.1) vs the oracle package's ``vision_oracle`` (reference ``vision.ViT`` /
``vision.Aligner``, cached on disk).

Grids: a direct 4x5 patch grid, and grids ``image_processor.load_image`` plans for 640x480 (35x46 patches), 1000x1000
(72x72) and 1920x1080 (69x122, 968 of the 1024 LLM tokens). 4x5, 35x46 and 69x122 are not multiples of 3, so the
aligner's zero padding is exercised. Synthetic and real weights. Contracts: ViT output, aligner output from the
device ViT (end to end), aligner output from the reference ViT output (isolates the unfold ordering), each
PCC >= 0.99 (G1); bit-identical repeat.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tt.v41.vision import TtV41Vision, load_vision_weights, vision_weight_names
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from tests.ttnn.utils_for_testing import comp_pcc

PCC = 0.99
SEED = 101  # synthetic weights, patches and test images
ARGS = orc.vision_args()


def _weights(source: str) -> tuple[dict, object]:
    """Device weights and the oracle's checkpoint (None = synthetic weights of SEED)."""
    if source == "synthetic":
        weights = orc.synthetic_vision_weights(ARGS, SEED)
        assert sorted(weights) == sorted(vision_weight_names()), "reference and loader names differ"
        return weights, None
    ckpt = resolve_checkpoint()
    if ckpt is None or not all((ckpt.root / ckpt.weight_map[n]).is_file() for n in vision_weight_names()[:1]):
        pytest.skip("V4.1 checkpoint shard with vision.*/aligner.* (model-00001-of-00048) not downloaded")
    return load_vision_weights(ckpt), ckpt.root


def _patches(case: str):
    if case == "grid4x5":
        return orc.random_patches(4, 5, ARGS, seed=SEED), 4, 5
    width, height = (int(v) for v in case.removeprefix("img").split("x"))
    return orc.image_patches(orc.synthetic_image(width, height, SEED), ARGS)


def _replica(t, mesh_device):
    """Device 0's copy of a replicated tensor; asserts every device holds the same values."""
    copies = ttnn.get_device_tensors(t)
    first = ttnn.to_torch(copies[0])
    for other in copies[1:]:
        assert torch.equal(first, ttnn.to_torch(other)), "replicas differ across devices"
    return first


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("case", ["grid4x5", "img640x480", "img1000x1000", "img1920x1080"])
@pytest.mark.parametrize("weights_source", ["synthetic", "real"])
@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param(
            (2, 4),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        )
    ],
    indirect=True,
)
def test_v41_vision(mesh_device, weights_source, case):
    t0 = time.time()
    weights, checkpoint = _weights(weights_source)
    patches, n_h, n_w = _patches(case)
    n, tokens = n_h * n_w, -(-n_h // 3) * -(-n_w // 3)
    logger.info(f"weights ({weights_source}) + patches {n_h}x{n_w}: {time.time() - t0:.1f}s")

    t0 = time.time()
    hit = orc.vision_cache_path(ARGS, patches, n_h, n_w, SEED, checkpoint).is_file()
    expected = orc.vision_oracle(patches, n_h, n_w, args=ARGS, seed=SEED, checkpoint=checkpoint)
    ref_hidden, ref_out = expected["hidden"], expected["aligned"]
    logger.info(f"vision oracle ({'cached' if hit else 'computed'}): {time.time() - t0:.1f}s")
    assert ref_out.shape == (tokens, C.EMB_SIZE)

    t0 = time.time()
    tt = TtV41Vision(mesh_device, weights)
    logger.info(f"device weights: {time.time() - t0:.1f}s")

    def run():
        hidden = tt.vit(patches, n_h, n_w)
        out = tt.aligner(hidden, n_h, n_w)
        return _replica(hidden, mesh_device)[0, 0, :n], _replica(out, mesh_device)[0, 0]

    t0 = time.time()
    hidden, out = run()
    hidden2, out2 = run()
    logger.info(f"device forward x2: {time.time() - t0:.1f}s")
    assert out.shape == (tokens, C.EMB_SIZE)
    assert torch.equal(hidden, hidden2) and torch.equal(out, out2), "vision encoder is not bit-identical across repeats"

    # aligner alone, fed the reference ViT output (sequence-padded like the device ViT output)
    rows = -(-n // 128) * 128
    ref_in = torch.zeros(1, 1, rows, C.VISION_DIM, dtype=torch.bfloat16)
    ref_in[0, 0, :n] = ref_hidden
    tt_in = ttnn.from_torch(
        ref_in,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    aligner_only = _replica(tt.aligner(tt_in, n_h, n_w), mesh_device)[0, 0]

    results = {
        "vit": comp_pcc(ref_hidden.float(), hidden.float(), 0.0)[1],
        "aligner_e2e": comp_pcc(ref_out.float(), out.float(), 0.0)[1],
        "aligner_only": comp_pcc(ref_out.float(), aligner_only.float(), 0.0)[1],
    }
    logger.info(f"vision {weights_source} {case} ({n_h}x{n_w} patches -> {tokens} tokens): {results}")
    for key, value in results.items():
        assert value >= PCC, (key, results)
