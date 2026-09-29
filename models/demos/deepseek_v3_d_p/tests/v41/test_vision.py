# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 ViT x32 + aligner (bead 10.1) vs the reference ``vision.ViT`` / ``vision.Aligner``.

Grids: a direct 4x5 patch grid, and grids ``image_processor.load_image`` plans for 640x480 (35x46 patches), 1000x1000
(72x72) and 1920x1080 (69x122, 968 of the 1024 LLM tokens). 4x5, 35x46 and 69x122 are not multiples of 3, so the
aligner's zero padding is exercised. Synthetic and real weights. Contracts: ViT output, aligner output from the
device ViT (end to end), aligner output from the reference ViT output (isolates the unfold ordering), each
PCC >= 0.99 (G1); bit-identical repeat.
"""

import io

import numpy as np
import pytest
import torch
from PIL import Image

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import image_processor
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import vision
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tt.v41.vision import TtV41Vision, load_vision_weights, vision_weight_names
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from tests.ttnn.utils_for_testing import comp_pcc

PCC = 0.99
ARGS = v41.ModelArgs(
    dim=C.EMB_SIZE,
    vision_n_layers=C.VISION_N_LAYERS,
    vision_dim=C.VISION_DIM,
    vision_n_heads=C.VISION_N_HEADS,
    vision_inter_dim=C.VISION_INTER_DIM,
    vision_patch_size=C.VISION_PATCH_SIZE,
    vision_rope_theta=C.VISION_ROPE_THETA,
    vision_downsample_ratio=C.VISION_DOWNSAMPLE_RATIO,
    vision_max_n_token=C.VISION_MAX_N_TOKEN,
)


def _synthetic_weights(gen) -> dict:
    ref_shapes = {f"vision.{k}": v.shape for k, v in vision.ViT(ARGS).state_dict().items()}
    ref_shapes |= {f"aligner.{k}": v.shape for k, v in vision.Aligner(ARGS).state_dict().items()}
    assert sorted(ref_shapes) == sorted(vision_weight_names()), "reference and loader names differ"
    out = {}
    for name, shape in ref_shapes.items():
        if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name.endswith("norm.weight"):
            t = 1 + 0.1 * torch.randn(shape, generator=gen)
        elif name.endswith(".bias"):
            t = 0.02 * torch.randn(shape, generator=gen)
        else:
            t = torch.randn(shape, generator=gen) * shape[-1] ** -0.5
        out[name] = t.to(torch.bfloat16)
    return out


def _weights(source: str, gen) -> dict:
    if source == "synthetic":
        return _synthetic_weights(gen)
    ckpt = resolve_checkpoint()
    if ckpt is None or not all((ckpt.root / ckpt.weight_map[n]).is_file() for n in vision_weight_names()[:1]):
        pytest.skip("V4.1 checkpoint shard with vision.*/aligner.* (model-00001-of-00048) not downloaded")
    return load_vision_weights(ckpt)


def _reference_modules(weights):
    with v41.set_dtype(torch.bfloat16):
        vit, aligner = vision.ViT(ARGS), vision.Aligner(ARGS)
    for prefix, module in (("vision.", vit), ("aligner.", aligner)):
        state = {k: weights[prefix + k].to(v.dtype) for k, v in module.state_dict().items()}
        module.load_state_dict(state)
    return vit.eval(), aligner.eval()


def _test_image(width: int, height: int, gen) -> bytes:
    """Deterministic smooth image with structure (gradients, a disc) plus mild noise, PNG-encoded."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.hypot(x - width * 0.4, y - height * 0.6) < min(width, height) * 0.25
    img = np.stack([255 * x / width, 255 * y / height, 128 + 100 * r], axis=-1)
    img += torch.randn(img.shape, generator=gen).numpy() * 8
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _patches(case: str, gen):
    if case == "grid4x5":
        return torch.randn(20, 3, C.VISION_PATCH_SIZE, C.VISION_PATCH_SIZE, generator=gen).to(torch.bfloat16), 4, 5
    width, height = (int(v) for v in case.removeprefix("img").split("x"))
    patches, n_h, n_w, n_llm_h, n_llm_w = image_processor.load_image({"data": _test_image(width, height, gen)}, ARGS)
    assert (n_llm_h, n_llm_w) == (-(-n_h // 3), -(-n_w // 3))
    return patches, n_h, n_w


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
    gen = torch.Generator().manual_seed(10_1)
    weights = _weights(weights_source, gen)
    patches, n_h, n_w = _patches(case, gen)
    n, tokens = n_h * n_w, -(-n_h // 3) * -(-n_w // 3)

    ref_vit, ref_aligner = _reference_modules(weights)
    with torch.no_grad():
        ref_hidden = ref_vit(patches, n_h, n_w)
        ref_out = ref_aligner(ref_hidden, n_h, n_w)
    assert ref_out.shape == (tokens, C.EMB_SIZE)

    tt = TtV41Vision(mesh_device, weights)

    def run():
        hidden = tt.vit(patches, n_h, n_w)
        out = tt.aligner(hidden, n_h, n_w)
        return _replica(hidden, mesh_device)[0, 0, :n], _replica(out, mesh_device)[0, 0]

    hidden, out = run()
    hidden2, out2 = run()
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
    print(f"vision {weights_source} {case} ({n_h}x{n_w} patches -> {tokens} tokens): {results}")
    for key, value in results.items():
        assert value >= PCC, (key, results)
