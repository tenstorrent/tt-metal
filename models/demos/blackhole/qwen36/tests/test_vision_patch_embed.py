# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""PCC for on-device vision patch embed plus interpolated position embed.
Reference is the HF pair this replaces. Only the two weight tensors are loaded."""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.blackhole.qwen36.tt.vision.patch_embed import VisionEmbed
from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs


def _load_embed_weights(model_args, keys):
    """Read just ``keys`` from a sharded safetensors checkpoint."""
    from safetensors.torch import load_file

    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    ckpt_dir = model_args.CKPT_DIR
    if tpc.wh_9b_n300_vision(model_args) and not os.path.isfile(os.path.join(ckpt_dir, "model.safetensors.index.json")):
        from huggingface_hub import snapshot_download

        offline = os.getenv("HF_HUB_OFFLINE") == "1" or os.getenv("CI") == "true"
        ckpt_dir = snapshot_download(ckpt_dir, local_files_only=offline)

    index = json.loads((Path(ckpt_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in {index[k] for k in keys}:
        tensors = load_file(str(Path(ckpt_dir) / shard))
        for k in keys:
            if k in tensors:
                out[k] = tensors[k]
    missing = [k for k in keys if k not in out]
    assert not missing, f"missing from checkpoint: {missing}"
    return out


class _RefEmbed(torch.nn.Module):
    """Minimal stand-in exposing the two attributes VisionEmbed reads off the HF vision model."""

    def __init__(self, proj_weight, proj_bias, pos_weight):
        super().__init__()
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Module()
        self.patch_embed.proj.weight = torch.nn.Parameter(proj_weight, requires_grad=False)
        self.patch_embed.proj.bias = torch.nn.Parameter(proj_bias, requires_grad=False)
        self.pos_embed = torch.nn.Embedding(pos_weight.shape[0], pos_weight.shape[1])
        self.pos_embed.weight = torch.nn.Parameter(pos_weight, requires_grad=False)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("grid_hw", [(28, 42), (16, 16)], ids=["hw28x42", "hw16x16"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_patch_embed(grid_hw, mesh_device, reset_seeds, ensure_gc):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed
    from transformers.vision_utils import get_vision_bilinear_indices_and_weights

    h, w = grid_hw
    grid_thw = torch.tensor([[1, h, w]], dtype=torch.long)
    n_patches = int(grid_thw.prod(dim=1).sum())
    seq_len = ((n_patches // 2048) + 1) * 2048

    # Real weights: the Conv3d fold and bilinear interpolation must match the checkpoint.
    model_args = VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=seq_len)
    vcfg = model_args.hf_config.vision_config

    weights = _load_embed_weights(
        model_args,
        ["model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias", "model.visual.pos_embed.weight"],
    )
    ref = _RefEmbed(
        weights["model.visual.patch_embed.proj.weight"].float(),
        weights["model.visual.patch_embed.proj.bias"].float(),
        weights["model.visual.pos_embed.weight"].float(),
    )

    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
    pixel_values = torch.randn(n_patches, patch_dim, dtype=torch.float32)

    hf_patch_embed = Qwen3_5VisionPatchEmbed(vcfg).float()
    hf_patch_embed.proj.weight = ref.patch_embed.proj.weight
    hf_patch_embed.proj.bias = ref.patch_embed.proj.bias

    num_grid_per_side = int(vcfg.num_position_embeddings**0.5)
    idx, wts = get_vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=num_grid_per_side, spatial_merge_size=vcfg.spatial_merge_size
    )
    ref_out = hf_patch_embed(pixel_values) + (ref.pos_embed(idx) * wts[:, :, None]).sum(0)
    ref_padded = torch.nn.functional.pad(ref_out, (0, 0, 0, seq_len - n_patches))

    tt_embed = VisionEmbed(mesh_device, model_args, ref, weight_cache_path=None)
    tt_out = tt_embed.forward(pixel_values, idx, wts, seq_len)

    shape = tuple(int(d) for d in tt_out.shape)
    assert shape[:3] == (1, 1, seq_len), f"unexpected shape {shape}"

    # Concat the hidden fracture. vision_replicated_acts may be absent; default it the way VisionEmbed does.
    if getattr(model_args, "vision_replicated_acts", False):
        out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1]
    else:
        out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    out = out.reshape(seq_len, vcfg.hidden_size).float()

    passing, pcc = comp_pcc(ref_padded, out, 0.99)
    logger.info(comp_allclose(ref_padded, out))
    logger.info(f"patch+pos embed PCC: {pcc}")

    # The padded tail must be exactly zero, as the host F.pad made it.
    tail_max = out[n_patches:].abs().max().item() if seq_len > n_patches else 0.0
    logger.info(f"padded-tail max abs: {tail_max}")
    assert tail_max == 0.0, f"padded rows are not zero (max {tail_max}) — bias leaked into the pad"
    assert passing, f"patch+pos embed PCC {pcc} below 0.99"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x4": (1, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_vision_dropin_embed_paths_agree(mesh_device, reset_seeds, ensure_gc, tmp_path):
    """Device embed and host embed must match through a random 2-block tower."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer

    # 1176 patches is not a multiple of 32, so the device cos/sin pad is exercised.
    grid_thw = torch.tensor([[1, 28, 42]], dtype=torch.long)
    n_patches = int(grid_thw.prod(dim=1).sum())
    seq_len = ((n_patches // 2048) + 1) * 2048

    model_args = VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=seq_len)
    model_args.hf_config.vision_config.depth = 2
    vcfg = model_args.hf_config.vision_config

    ref = Qwen3_5VisionModel(vcfg).float().eval()

    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
    pixel_values = torch.randn(n_patches, patch_dim, dtype=torch.float32)

    def run(host_embed):
        prev = os.environ.get("QWEN36_HOST_VISION_EMBED")
        os.environ["QWEN36_HOST_VISION_EMBED"] = "1" if host_embed else "0"
        try:
            # Random weights must use an isolated cache dir; the ttnn cache is keyed by tensor name only.
            tower = DropInVisionTransformer(
                ref, model_args, dtype=ttnn.bfloat8_b, weight_cache_path=tmp_path / ("host" if host_embed else "dev")
            )
            assert (tower.tt_embed is None) == host_embed, "QWEN36_HOST_VISION_EMBED did not take effect"
            out = tower.forward(pixel_values, grid_thw)
            return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
        finally:
            if prev is None:
                os.environ.pop("QWEN36_HOST_VISION_EMBED", None)
            else:
                os.environ["QWEN36_HOST_VISION_EMBED"] = prev

    host_out = run(host_embed=True)
    dev_out = run(host_embed=False)

    passing, pcc = comp_pcc(host_out, dev_out, 0.99)
    logger.info(comp_allclose(host_out, dev_out))
    logger.info(f"host-embed vs device-embed tower output PCC: {pcc}")
    assert passing, f"device embed path diverges from host embed path: PCC {pcc}"
