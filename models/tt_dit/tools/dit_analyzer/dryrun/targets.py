# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Named dry-run targets: what to build, on which mesh, with which inputs.

A target is the small amount of context a dry run cannot read off the source:
mesh shape, which mesh axis carries which parallel role, and the activation
shapes a generation would produce. Everything else -- module structure, weight
distribution, which collectives happen -- comes from running the real code.

The branch flags and shapes a target fixes are exactly what phase 12's
``ditcheck matrix`` sweeps, so they are kept in one declarative place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from ..ir import Dist, Graph
from . import recorder
from .context import CTX
from .install import install
from .weights import load_meta_weights


@dataclass
class Preset:
    """One mesh configuration of a target."""

    name: str
    mesh_shape: Tuple[int, int]
    arch: str = "blackhole"
    topology: str = "Ring"
    sp_axis: int = 1  # sequence parallel
    tp_axis: int = 0  # tensor parallel
    cfg_axis: int = 0
    oracle: Optional[str] = None  # hand-written `examples/` graph to diff against
    description: str = ""

    @property
    def axis_names(self) -> Tuple[str, ...]:
        names = ["axis%d" % i for i in range(len(self.mesh_shape))]
        names[self.sp_axis] = "sp"
        names[self.tp_axis] = "tp"
        return tuple(names)


@dataclass
class Target:
    name: str
    description: str
    build: Callable[[Preset], Graph]
    presets: Dict[str, Preset] = field(default_factory=dict)

    def preset(self, name: Optional[str]) -> Preset:
        if name is None:
            return next(iter(self.presets.values()))
        if name not in self.presets:
            raise KeyError(
                "unknown preset '%s' for target '%s' (have: %s)" % (name, self.name, ", ".join(self.presets))
            )
        return self.presets[name]


# -----------------------------------------------------------------------------
# LTX-2.3 transformer block (audio + video)
# -----------------------------------------------------------------------------
VIDEO_DIM, VIDEO_HEADS, VIDEO_HEAD_DIM, VIDEO_N = 4096, 32, 128, 38912
AUDIO_DIM, AUDIO_HEADS, AUDIO_HEAD_DIM, AUDIO_N = 2048, 32, 64, 256
TEXT_L, LAYERS, STEPS = 32, 48, 8
LTX_ADALN_MODULATIONS = 9  # adaln_single.linear.weight rows = 9 * inner_dim for A+V


def _ltx_checkpoint():
    """The A+V checkpoint as a metadata-only index (blocker 38).

    Points at a real ``.safetensors`` header/index if ``DITCHECK_LTX_CHECKPOINT``
    names one -- reading shapes, never weights -- and otherwise falls back to a
    declared manifest that reproduces the audio+video checkpoint's detectable
    flags: a ``to_gate_logits`` key (``has_gate``) and an ``adaln_single`` weight
    whose first dim (``9 * inner_dim``) exceeds ``6 * inner_dim``
    (``cross_attention_adaln``). Either way the flags are *derived*, not asserted.
    """
    import os

    from . import checkpoint as ckpt

    path = os.environ.get("DITCHECK_LTX_CHECKPOINT")
    if path:
        if path.endswith(".index.json"):
            return ckpt.from_index_json(path)
        return ckpt.from_safetensors_header(path)
    return ckpt.declared(
        keys=[
            ckpt.LTX_ADALN_KEY,
            "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight",
        ],
        shapes={ckpt.LTX_ADALN_KEY: (LTX_ADALN_MODULATIONS * VIDEO_DIM, VIDEO_DIM)},
    )


def _ltx_block(preset: Preset) -> Graph:
    """One `LTXTransformerBlock`, built and called for real.

    ``apply_gated_attention`` (``has_gate``) and ``cross_attention_adaln`` are
    checkpoint-derived (blocker 38): they come from a metadata-only index via
    the same rule tt_dit uses, not a hardcoded boolean. ``has_audio`` is a
    per-instance flag the checkpoint never encodes, so it stays declared.
    """
    from .checkpoint import ltx_flags

    mesh_device = install(preset.mesh_shape, preset.arch)
    import ttnn  # the shim

    index = _ltx_checkpoint()
    flags = ltx_flags(index, inner_dim=VIDEO_DIM)
    has_gate = bool(flags["has_gate"])
    # a shape-less index can't evaluate the adaln size test; keep the A+V default
    cross_attention_adaln = True if flags["cross_attention_adaln"] is None else flags["cross_attention_adaln"]

    graph = _start(mesh_device, preset, name="ltx_block_dryrun_%s" % preset.name, calls=LAYERS, steps=STEPS)
    graph.meta.update(
        {
            "model": "LTX-2.3 (audio + video), dry run from source",
            "note": "one LTXTransformerBlock x%d layers; %s topology" % (LAYERS, preset.topology),
            "checkpoint": index.source,
            "checkpoint_flags": {"has_gate": has_gate, "cross_attention_adaln": cross_attention_adaln},
        }
    )

    from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerBlock
    from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
    from models.tt_dit.parallel.manager import CCLManager

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=preset.cfg_axis),
        sequence_parallel=ParallelFactor(factor=preset.mesh_shape[preset.sp_axis], mesh_axis=preset.sp_axis),
        tensor_parallel=ParallelFactor(factor=preset.mesh_shape[preset.tp_axis], mesh_axis=preset.tp_axis),
    )
    ccl = CCLManager(
        mesh_device,
        num_links=2,
        topology=ttnn.Topology.Ring if preset.topology == "Ring" else ttnn.Topology.Linear,
    )

    block = LTXTransformerBlock(
        video_dim=VIDEO_DIM,
        audio_dim=AUDIO_DIM,
        video_ffn_dim=4 * VIDEO_DIM,
        audio_ffn_dim=4 * AUDIO_DIM,
        video_num_heads=VIDEO_HEADS,
        audio_num_heads=AUDIO_HEADS,
        video_cross_attention_dim=VIDEO_DIM,
        audio_cross_attention_dim=AUDIO_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        parallel_config=parallel_config,
        has_audio=True,  # per-instance, not checkpoint-encoded
        apply_gated_attention=has_gate,  # checkpoint-derived (blocker 38)
        cross_attention_adaln=cross_attention_adaln,  # checkpoint-derived
    )
    graph.meta["parameters"] = load_meta_weights(block)

    out = block(**_ltx_inputs(preset))
    graph.outputs = [t.sym for t in out if t is not None]
    return graph


def _ltx_inputs(preset: Preset) -> Dict[str, Any]:
    """Activations as a generation would present them: sp on sequence, tp on features."""
    sp, tp = preset.sp_axis, preset.tp_axis
    seq_shard = {sp: 2}
    feature_shard = {tp: 3}
    activation = dict(seq_shard)
    activation.update(feature_shard)
    adaln = 9  # cross_attention_adaln=True

    def tensor(logical, shard=None):
        return recorder.entry(logical, Dist.make(CTX.mesh, shard or {}), base="input")

    return dict(
        video_1BND=tensor([1, 1, VIDEO_N, VIDEO_DIM], activation),
        video_prompt=tensor([1, 1, TEXT_L, VIDEO_DIM]),  # replicated text embeddings
        video_temb=tensor([adaln, 1, 1, VIDEO_DIM], feature_shard),
        video_N=VIDEO_N,
        video_rope_cos=tensor([1, 1, VIDEO_N, VIDEO_HEAD_DIM], seq_shard),
        video_rope_sin=tensor([1, 1, VIDEO_N, VIDEO_HEAD_DIM], seq_shard),
        trans_mat=tensor([1, 1, 32, 32]),
        video_prompt_temb=tensor([2, 1, 1, VIDEO_DIM]),
        audio_1BND=tensor([1, 1, AUDIO_N, AUDIO_DIM], activation),
        audio_prompt=tensor([1, 1, TEXT_L, AUDIO_DIM]),
        audio_temb=tensor([adaln, 1, 1, AUDIO_DIM], feature_shard),
        av_ca_temb=tensor([5, 1, 1, VIDEO_DIM], feature_shard),
        audio_N=AUDIO_N,
        audio_rope_cos=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], seq_shard),
        audio_rope_sin=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], seq_shard),
        av_ca_audio_temb=tensor([5, 1, 1, AUDIO_DIM], feature_shard),
        audio_prompt_temb=tensor([2, 1, 1, AUDIO_DIM]),
        video_cross_pe_cos=tensor([1, 1, VIDEO_N, AUDIO_HEAD_DIM], seq_shard),
        video_cross_pe_sin=tensor([1, 1, VIDEO_N, AUDIO_HEAD_DIM], seq_shard),
        audio_cross_pe_cos=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], seq_shard),
        audio_cross_pe_sin=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM], seq_shard),
        audio_cross_pe_cos_full=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM]),
        audio_cross_pe_sin_full=tensor([1, 1, AUDIO_N, AUDIO_HEAD_DIM]),
        audio_attn_mask=tensor([1, 1, AUDIO_N, AUDIO_N]),
        audio_padding_mask_full=tensor([1, 1, AUDIO_N, 1]),
    )


def _start(mesh_device, preset: Preset, name: str, calls: int = 1, steps: int = 1) -> Graph:
    from . import start

    return start(
        mesh_device,
        axis_names=preset.axis_names,
        name=name,
        steps=steps,
        calls=calls,
        topology=preset.topology,
    )


TARGETS: Dict[str, Target] = {
    "ltx_block": Target(
        name="ltx_block",
        description="LTX-2.3 audio+video transformer block, from models/tt_dit source",
        build=_ltx_block,
        presets={
            "bh_4x8": Preset(
                name="bh_4x8",
                mesh_shape=(4, 8),
                topology="Ring",
                sp_axis=1,
                tp_axis=0,
                oracle="ltx_block_bh_4x8",
                description="Blackhole Galaxy 4x8, Ring: the fused AGMM path",
            ),
            "bh_2x4": Preset(
                name="bh_2x4",
                mesh_shape=(2, 4),
                topology="Linear",
                sp_axis=1,
                tp_axis=0,
                oracle="ltx_block_bh_2x4",
                description="Blackhole 2x4, Linear: the explicit-gather path",
            ),
        },
    ),
}


def build(target: str, preset: Optional[str] = None) -> Tuple[Graph, Preset]:
    """Run a named target and return its graph. Installs the shim as a side effect."""
    if target not in TARGETS:
        raise KeyError("unknown target '%s' (have: %s)" % (target, ", ".join(sorted(TARGETS))))
    spec = TARGETS[target]
    chosen = spec.preset(preset)
    return spec.build(chosen), chosen


def describe() -> str:
    lines = []
    for name in sorted(TARGETS):
        spec = TARGETS[name]
        lines.append("  %-14s %s" % (name, spec.description))
        for pname, preset in spec.presets.items():
            oracle = ("  oracle: example:%s" % preset.oracle) if preset.oracle else ""
            lines.append(
                "  %-14s   --preset %-8s %dx%d %s%s" % ("", pname, *preset.mesh_shape, preset.topology, oracle)
            )
    return "\n".join(lines)
