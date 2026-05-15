# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class VariantSpec:
    name: str
    best_use_case: str
    repo_id: str
    subfolder: str


VARIANTS: list[VariantSpec] = [
    VariantSpec(
        name="ACE-Step v1.5 Turbo",
        best_use_case="Fast generation (2–10 seconds). Great for quick prototyping.",
        repo_id="ACE-Step/Ace-Step1.5",
        subfolder="acestep-v15-turbo",
    ),
    # The public docs mention SFT/Base. If they are not present in the repo snapshot, we skip.
    VariantSpec(
        name="ACE-Step v1.5 SFT",
        best_use_case="Highest quality and best prompt adherence. Uses more steps for better sound.",
        repo_id="ACE-Step/acestep-v15-sft",
        subfolder="",
    ),
    VariantSpec(
        name="ACE-Step v1.5 Base",
        best_use_case="The Swiss Army Knife for advanced tasks like track extraction and expansion.",
        repo_id="ACE-Step/acestep-v15-base",
        subfolder="",
    ),
]


def _find_safetensors_shard(model_dir: Path) -> Path:
    shards = sorted(model_dir.rglob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors under {model_dir}")
    named = [p for p in shards if p.name == "model.safetensors"]
    if named:
        return named[0]
    return shards[0]


def _ttnn_output_from_demo_tensors(*, device, model_dir: Path, picked_prefix: str, cfg: dict, tensors: dict):
    """Replay patch-embed + output-head on TTNN using the same host tensors as the torch demo."""
    import torch

    import ttnn
    from models.demos.ace_step_v1_5.ttnn_impl.output_head import TtAceStepDiTOutputHead
    from models.demos.ace_step_v1_5.ttnn_impl.patchify import PatchifyMetadata, TtAceStepPatchEmbed1D
    from models.demos.ace_step_v1_5.ttnn_impl.safetensors_loader import load_safetensors_state_dict

    prefix = picked_prefix if picked_prefix.endswith(".") else f"{picked_prefix}."
    model_path = _find_safetensors_shard(model_dir)
    sd_np = load_safetensors_state_dict(str(model_path), prefix=prefix).tensors

    @dataclass(frozen=True)
    class _Cfg:
        patch_size: int
        in_channels: int
        hidden_size: int
        audio_acoustic_hidden_dim: int
        rms_norm_eps: float

    ace_cfg = _Cfg(
        patch_size=int(cfg["patch_size"]),
        in_channels=int(cfg["in_channels"]),
        hidden_size=int(cfg["hidden_size"]),
        audio_acoustic_hidden_dim=int(cfg["out_channels"]),
        rms_norm_eps=float(cfg["eps"]),
    )

    x = tensors["x_latent"].to(torch.bfloat16)
    temb = tensors["temb"].to(torch.bfloat16)
    meta_d = tensors["meta"]
    meta = PatchifyMetadata(
        original_seq_len=int(meta_d["original_seq_len"]),
        pad_length=int(meta_d["pad_length"]),
        patch_size=int(meta_d["patch_size"]),
    )

    tt_proj_in = TtAceStepPatchEmbed1D(
        config=ace_cfg,
        state_dict=sd_np,
        base_address="proj_in",
        device=device,
        activation_dtype=ttnn.bfloat16,
    )
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    patches_tt, _ = tt_proj_in.forward(x_tt)

    tt_head = TtAceStepDiTOutputHead(
        config=ace_cfg,
        state_dict=sd_np,
        base_address="",
        device=device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )
    temb_tt = ttnn.from_torch(temb, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    y_tt = tt_head.forward(patches_tt, temb_tt, meta)
    return ttnn.to_torch(y_tt).float()


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: v.name)
def test_hf_output_head_demo_variants_smoke(device, variant: VariantSpec):
    """
    Download HF weights, run torch ``hf_output_head_demo``, then TTNN patch+head PCC vs torch.
    """
    from huggingface_hub import snapshot_download

    from models.demos.ace_step_v1_5.tests._dit_decoder_pcc_common import assert_pcc_print
    from models.demos.ace_step_v1_5.torch_ref.hf_output_head_demo import run_hf_output_head_demo

    snapshot_dir = snapshot_download(repo_id=variant.repo_id)
    model_dir = Path(snapshot_dir) / variant.subfolder
    if not model_dir.exists():
        pytest.skip(f"Variant not present in snapshot: {variant.subfolder}")

    sig = run_hf_output_head_demo(
        repo_id=variant.repo_id,
        subfolder=variant.subfolder,
        seed=0,
        batch=1,
        original_seq_len=257,
        input_pt=None,
        noise_std=1.0,
        offline=False,
        return_tensors=True,
    )

    assert sig["repo_id"] == variant.repo_id
    assert sig["model_dir"].endswith(variant.subfolder)

    B, T, C = sig["output"]["shape"]
    assert B == 1
    assert T == 257
    assert C > 0

    y_tt = _ttnn_output_from_demo_tensors(
        device=device,
        model_dir=model_dir,
        picked_prefix=str(sig["picked_prefix"]),
        cfg=sig["config"],
        tensors=sig["tensors"],
    )
    y_ref = sig["tensors"]["y_torch"].float()
    label = variant.name.replace(" ", "_").lower()
    assert_pcc_print(f"hf_output_head_demo_{label}", y_ref, y_tt)
