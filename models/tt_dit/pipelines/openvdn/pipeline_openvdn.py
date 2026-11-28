# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN (VDN-MiniMax-H3) text-to-audio-video on a Blackhole mesh: the MiniMax-H3 pipeline with the hybrid-attention
transformer and the VDN checkpoint's weights.

Reference: OpenVDN/vdn-minimax-h3 @e262cb57. Its diffusers path (`src/inference/infer_diffusers.py`) is the stock
MiniMax-H3 modular pipeline with `VDNMiniMaxH3Transformer3DModel` as the transformer, so everything but the transformer
is the H3 pipeline's here too: prompt encoder, packing, schedulers (shifts 12 / 3), VAE and audio decode. The released
8-step checkpoint (`stage-dmd-step-250`) runs at `num_inference_steps=9` (sigma grid points; 8 model evaluations).

The transformer weights are built on the host the way `VDNMiniMaxH3Transformer3DModel.from_pretrained` builds them:
the H3 base transformer (bf16), the checkpoint's linear-branch tensors, then each LoRA adapter in the spec's order
folded in as `W += B @ A` (scale 1: alpha equals rank in both released adapters), the delta in fp32 and added in the
weight's dtype. The adapters' `attn.orig.*` targets are the H3 attention's own weights.

`weights_dir` is an H3 diffusers snapshot (transformer, text_encoder, tokenizer, processor, vae, audio_vae, schedulers;
the VDN repo's `h3-base/` plus MiniMax-H3's text encoder); `checkpoint_dir` is one VDN checkpoint, e.g.
`<vdn repo>/stage-dmd-step-250`.

Eager and unbucketed for now: no denoise trace, no dynamic length, no warmup. Coresident (every stage stays loaded
across requests) since stage 4 measured it fits: the hybrid layers no longer gather the whole hidden state. The packed
sequence goes to the transformer frame-aligned (see `_denoise`).
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import torch

import ttnn

from ...models.transformers.openvdn.hybrid_attention_openvdn import HybridLayout
from ...models.transformers.openvdn.transformer_openvdn import OpenVDNTransformer3DModel
from ..minimax_h3.packing import MiniMaxH3PackedSequence
from ..minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

#: Model evaluations the released DMD checkpoint was distilled for; `num_inference_steps` is one more.
OPENVDN_NUM_FORWARDS = 8

# The transform configuration this port implements (`stage-*/diffusers/config.json`, `vdn.transform`).
_SUPPORTED_TRANSFORM = {
    "anchor_frames": "both",
    "enable_softmax_gate": True,
    "linear_attention": {
        "a_fp32": True,
        "bridge": "alpha",
        "delta_rule": "vdn_solve",
        "enable_text_state": True,
        "short_conv": {"targets": ["k", "v"]},
    },
}


def _check_transform(transform: dict, head_dim: int) -> None:
    for key, value in _SUPPORTED_TRANSFORM.items():
        if key == "linear_attention":
            for sub, expected in value.items():
                if transform[key][sub] != expected:
                    raise NotImplementedError(f"vdn transform linear_attention.{sub}={transform[key][sub]!r}")
        elif transform[key] != value:
            raise NotImplementedError(f"vdn transform {key}={transform[key]!r}")
    if transform["linear_attention"]["linear_head_dim"] != head_dim:
        raise NotImplementedError("the linear branch's head_dim must equal the attention head_dim")


def hybrid_layout(layout: MiniMaxH3PackedSequence) -> HybridLayout:
    """The hybrid layers' view of a packed sequence, as the model repo's `_layout` derives it: the target video rows
    are the trailing contiguous run of `video_indices`, their (h, w) rotary coordinates the frame grid."""
    video = layout.video_indices
    breaks = (video[1:] != video[:-1] + 1).nonzero()
    target = video[int(breaks[-1]) + 1 :] if breaks.numel() else video
    grid = layout.position_ids[target, 1:]
    frame_height, frame_width = torch.unique(grid[:, 0]).numel(), torch.unique(grid[:, 1]).numel()
    per_frame = frame_height * frame_width
    if target.numel() % per_frame:
        raise ValueError(f"{target.numel()} target video rows do not fill {frame_height}x{frame_width} frames")
    text = layout.text_indices
    if text.numel() and not torch.equal(text, torch.arange(int(text[0]), int(text[0]) + text.numel())):
        raise ValueError("the prompt rows must be contiguous")
    return HybridLayout(
        seq_len=layout.sequence_length,
        video_start=int(target[0]),
        num_frames=target.numel() // per_frame,
        frame_height=frame_height,
        frame_width=frame_width,
        text_start=int(text[0]) if text.numel() else 0,
        text_len=text.numel(),
    )


class OpenVDNPipeline(MiniMaxH3Pipeline):
    """`t2va` (and `fl2va`) with the VDN transformer. Build with `create_pipeline`."""

    def __init__(self, *, checkpoint_dir: str | os.PathLike, **kwargs) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.vdn_spec = json.loads((self.checkpoint_dir / "diffusers" / "config.json").read_text())["vdn"]
        self.merged_adapter_pairs: int | None = None
        self.branch_tensors: int | None = None
        if kwargs.get("task", "t2va") != "t2va":
            raise NotImplementedError("OpenVDN releases t2va/fl2va checkpoints only")
        super().__init__(**kwargs)

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        checkpoint_dir: str | os.PathLike,
        weights_dir: str | os.PathLike | None = None,
        **kwargs,
    ) -> OpenVDNPipeline:
        """`MiniMaxH3Pipeline.create_pipeline` on the eager, unbucketed path (overridable through `kwargs`)."""
        defaults = dict(
            dynamic_length=False,
            bucket_denoise=False,
            trace_denoise=False,
            trace_audio=False,
            use_persistent_ccl_buffers=False,
            dit_fsdp=False,
            warmup=False,
            # Every stage stays loaded (stage 4: fits at 5 s on the 4x8, total 183 -> 92 s per request).
            coresident=True,
        )
        return super().create_pipeline(
            mesh_device=mesh_device,
            weights_dir=weights_dir,
            checkpoint_dir=checkpoint_dir,
            **{**defaults, **kwargs},
        )

    # ------------------------------------------------------------------ transformer

    #: Bumped whenever the transformer's device parameters change (names or layout), so a weight cache written by
    #: earlier code is not loaded. 2: row-broadcast conv taps for the fused feature kernels (stage 4).
    PARAMETERS_VERSION = 2

    def _dit_weight_mode(self) -> str:
        return f"{super()._dit_weight_mode()}_openvdn_{self.checkpoint_dir.name}_p{self.PARAMETERS_VERSION}"

    def _build_transformer(self) -> OpenVDNTransformer3DModel:
        config = {k: v for k, v in self.transformer_config.items() if k not in ("rope_freq_dim", "rope_theta")}
        config["patch_size"] = tuple(config["patch_size"])
        transform = self.vdn_spec["transform"]
        _check_transform(transform, config["attention_head_dim"])
        self._host_log(
            f"building the {config['num_layers']}-layer OpenVDN transformer ({self.checkpoint_dir.name}), "
            f"TP={self.tp_factor}/SP={self.sp_factor}"
        )
        return OpenVDNTransformer3DModel(
            **config,
            radius=transform["softmax_attention"]["radius"],
            chunk=transform["softmax_attention"]["chunk"],
            anchor_frames=transform["anchor_frames"],
            enable_text_state=transform["linear_attention"]["enable_text_state"],
            mesh_device=self.mesh_device,
            ccl_manager=self.ccl_manager,
            parallel_config=self.dit_parallel_config,
            is_fsdp=self.dit_fsdp,
            kv_gather_capacity=self.kv_gather_capacity,
            dynamic_length=self.dynamic_length,
            agmm_mode=self.agmm_mode,
            ff1_output_dtype=self.ff1_output_dtype,
        )

    def _read_safetensors(self, subfolder: str) -> dict[str, torch.Tensor]:
        state = super()._read_safetensors(subfolder)
        if subfolder != self.transformer_subfolder:
            return state
        from safetensors.torch import load_file

        branch = load_file(str(self.checkpoint_dir / self.vdn_spec["branch"]))
        clash = sorted(set(branch) & set(state))
        if clash:
            raise RuntimeError(f"linear-branch keys already in the base checkpoint, e.g. {clash[:4]}")
        state.update({k: v.to(torch.bfloat16) for k, v in branch.items()})
        self.branch_tensors = len(branch)

        merged = 0
        for adapter in self.vdn_spec["adapters"]:
            lora = load_file(str(self.checkpoint_dir / adapter))
            for name, a in lora.items():
                if ".lora_A." not in name:
                    continue
                b = lora[name.replace(".lora_A.", ".lora_B.")]
                target = name.split(".lora_A.")[0].replace(".attn.orig.", ".attn.") + ".weight"
                if target not in state:
                    raise KeyError(f"adapter {adapter}: target {target} is not in the transformer checkpoint")
                weight = state[target]
                delta = b.float() @ a.float()
                if delta.shape != weight.shape:
                    raise ValueError(f"adapter {adapter}: {target} delta {tuple(delta.shape)} != {tuple(weight.shape)}")
                state[target] = weight + delta.to(weight.dtype)
                merged += 1
        self.merged_adapter_pairs = merged
        self._host_log(
            f"OpenVDN weights: {len(branch)} linear-branch tensors, {merged} LoRA pairs merged from "
            f"{len(self.vdn_spec['adapters'])} adapters"
        )
        return state

    # ------------------------------------------------------------------ frame-aligned packing (stage 4)
    #
    # With OPENVDN_ALIGNED=1 (default) the packed sequence goes to the transformer in the frame-aligned order
    # (`HybridLayout.aligned`): the head globals padded to a multiple of 64 rows, each frame to its own slot, so the
    # hybrid layers run their window and linear branch on each device's own rows. The H3 assembly gathers rows by
    # index, so only the indices change: pad rows read source row 0 and copy the metadata (rotary position, modality
    # tag) of the real row before them; the output indices pick the real rows.

    def _denoise(self, transformer, layout, *args, **kwargs):
        hybrid = hybrid_layout(layout)
        if os.environ.get("OPENVDN_ALIGNED", "1") == "1" and self.sp_factor > 1:
            hybrid = dataclasses.replace(hybrid, aligned=True)
            self._aligned_rows, total = hybrid.aligned_rows(self.sp_factor)
            layout = _aligned_packed_sequence(layout, self._aligned_rows, total)
        transformer.set_layout(hybrid)
        try:
            return super()._denoise(transformer, layout, *args, **kwargs)
        finally:
            transformer.set_layout(None)
            self._aligned_rows = None

    _aligned_rows: torch.Tensor | None = None

    def _assembly_indices(self, condition_spec, caps, l_len, a_len, v_len, rung, vsa_geometry=None):
        if self._aligned_rows is None:
            return super()._assembly_indices(condition_spec, caps, l_len, a_len, v_len, rung, vsa_geometry)
        assert vsa_geometry is None
        # The logical order, as `MiniMaxH3Pipeline._assembly_indices` builds it, then moved to the aligned rows.
        src = self._assembly_source_offsets(caps)
        logical = [torch.arange(l_len)]
        cursors = {"video": 0, "audio": 0}
        for modality, rows in condition_spec:
            base = src[f"condition_{modality}"] + cursors[modality]
            cursors[modality] += rows
            logical.append(torch.arange(base, base + rows))
        logical.append(torch.arange(src["audio"], src["audio"] + a_len))
        logical.append(torch.arange(src["video"], src["video"] + v_len))
        logical = torch.cat(logical).to(torch.int32)
        indices = torch.zeros(rung, dtype=torch.int32)
        indices[self._aligned_rows] = logical
        if self._shard_assembly_indices:
            return self._row_indices(indices, rung)
        return self._replicated_indices(indices)

    def _select_bucket(self, seq_len: int) -> int:
        # The aligned length is already a multiple of 64 * SP and the hybrid layers take exactly it, so a traced
        # (bucketed) denoise runs at that length rather than the ladder's next rung.
        if self._aligned_rows is None:
            return super()._select_bucket(seq_len)
        return self._padded_length(seq_len)

    def _output_indices(self, start, count, capacity, vsa_geometry=None):
        if self._aligned_rows is None:
            return super()._output_indices(start, count, capacity, vsa_geometry)
        assert vsa_geometry is None
        rows = self._aligned_rows.to(torch.int32)
        indices = torch.full((capacity,), int(rows[start]), dtype=torch.int32)
        indices[:count] = rows[start : start + count]
        return self._replicated_indices(indices)


def _aligned_packed_sequence(
    layout: MiniMaxH3PackedSequence, rows: torch.Tensor, total: int
) -> MiniMaxH3PackedSequence:
    """`layout` with its rows at `rows` of a `total`-row sequence; every pad row copies the metadata of the real row
    before it."""
    valid = torch.zeros(total, dtype=torch.bool)
    valid[rows] = True
    source = (torch.cumsum(valid.long(), 0) - 1).clamp(min=0)
    return dataclasses.replace(
        layout,
        sequence_length=total,
        position_ids=layout.position_ids[source],
        token_tags=layout.token_tags[source],
        video_indices=rows[layout.video_indices],
        audio_indices=rows[layout.audio_indices],
        text_indices=rows[layout.text_indices],
    )


__all__ = ["OPENVDN_NUM_FORWARDS", "OpenVDNPipeline", "hybrid_layout"]
