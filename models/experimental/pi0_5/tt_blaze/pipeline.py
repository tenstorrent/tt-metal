# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pi0_5BlazePipeline — driver for pi0.5 on a 4-loudbox tt-blaze deployment.

Composes the four FusedOps from `layers.py` into a `PipelineGraph`,
sets up persistent D2D sockets between loudboxes, runs the
PipelineManager writer/reader threads, and drives a single
`run_inference(pixel_values, lang_tokens, noisy_actions) → actions`.

Scaffold only — sub-op bodies are stubs in `layers.py`. This driver
shows the wiring a contributor filling in those bodies would target.

Reference shape:
  blaze/models/deepseek_v3_b1/pipeline.py — DSv3 PipelineConfiguration factory
  tt-blaze/pipeline_builder/design.md §4 — PipelineBuilder API
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:  # pragma: no cover
    import ttnn
    from pipeline_builder import PipelineBuilder, PipelineRunMode
    from pipeline_builder.graph import PipelineGraph, Node, Edge
    from blaze.socket_config import ReceiverSocketConfig, SenderSocketConfig
    from blaze.kernel_codegen import LoopingConfig
except ImportError:  # pragma: no cover
    ttnn = None  # type: ignore

    class PipelineBuilder:  # type: ignore[no-redef]
        def __init__(self, **kw):
            ...

        def build(self, stages):
            return None

    class PipelineRunMode:  # type: ignore[no-redef]
        FULL = "full"
        COMPUTE = "compute"
        SOCKETS = "sockets"

    class PipelineGraph:  # type: ignore[no-redef]
        def __init__(self, nodes=None, edges=None):
            ...

    class Node:  # type: ignore[no-redef]
        def __init__(self, **kw):
            ...

    class Edge:  # type: ignore[no-redef]
        def __init__(self, *a):
            ...

    ReceiverSocketConfig = SenderSocketConfig = object  # type: ignore

    class LoopingConfig:  # type: ignore[no-redef]
        def __init__(self, run_persistently=False, num_iterations=None):
            ...


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────

# Galaxy = 32 chips = 4 loudboxes × 8 chips. One loudbox per stage.
NUM_LOUDBOXES = 4
CHIPS_PER_LOUDBOX = 8
LOUDBOX_SHAPE = (4, 2)  # rows × cols — tt-blaze canonical (blaze_101.md §4.2)

# pi0.5 model constants
NUM_SIGLIP_LAYERS = 27
NUM_VLM_LAYERS = 18
NUM_EXPERT_LAYERS = 18
DENOISE_STEPS = 10  # Euler integrator steps

# Layer-to-loudbox packing:
#   A: 27 SigLIP layers chained into one FusedOp at TP=8 (~73 MB/chip)
#   B: VLM layers 0..8  (9 layers, TP=8, ~134 MB/chip incl. KV)
#   C: VLM layers 9..17 (same)
#   D: Expert 0..17 + Suffix MLP (~60 MB/chip incl. migrated VLM KV)
VLM_LAYERS_PER_LOUDBOX = 9
EXPERT_LAYERS_PER_LOUDBOX = 18


# ──────────────────────────────────────────────────────────────────────────
# Top-level driver
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class Pi0_5BlazeStages:
    """Per-stage state once the pipeline is built."""

    vision: Optional[object] = None  # FusedOp on loudbox A (SigLIP)
    vlm_a: Optional[object] = None  # FusedOp on loudbox B (VLM 0..8)
    vlm_b: Optional[object] = None  # FusedOp on loudbox C (VLM 9..17)
    denoise: Optional[object] = None  # FusedOp on loudbox D (Expert + Suffix)
    # One persistent KV-migration socket per VLM layer (18 total),
    # bridging stages B/C → D once per inference.
    kv_migration_sockets: List[object] = field(default_factory=list)


@dataclass
class Pi0_5BlazePipeline:
    """Drives a single pi0.5 inference end-to-end on a 4-loudbox Galaxy.

    Usage:
        pipe = Pi0_5BlazePipeline.build(mesh_device, hf_weights)
        pipe.start()
        actions = pipe.run_inference(pixel_values, lang_tokens, noisy_actions)
        pipe.stop()
    """

    mesh_device: object
    hf_weights: Dict[str, object]
    stages: Pi0_5BlazeStages = field(default_factory=Pi0_5BlazeStages)
    submeshes: List[object] = field(default_factory=list)
    pipeline: Optional[object] = None  # the tt-blaze Pipeline object
    is_started: bool = False

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        mesh_device,
        hf_weights: Dict[str, object],
        *,
        tp_size: int = 8,
        denoise_steps: int = DENOISE_STEPS,
    ) -> "Pi0_5BlazePipeline":
        self = cls(mesh_device=mesh_device, hf_weights=hf_weights)

        # 1. Partition Galaxy into 4 loudbox submeshes (uniform 4×2 each).
        #    Simplest case of pipeline_builder/design.md §2's create_submeshes.
        if ttnn is not None:  # pragma: no cover
            self.submeshes = mesh_device.create_submeshes(ttnn.MeshShape(*LOUDBOX_SHAPE))

        # 2. Build the four stage FusedOps.
        self._build_vision_stage(self._submesh(0))
        self._build_vlm_stage_a(self._submesh(1))
        self._build_vlm_stage_b(self._submesh(2))
        self._build_denoise_stage(self._submesh(3))

        # 3. Wire the PipelineGraph (linear chain).
        graph = PipelineGraph(
            nodes={
                "vision": Node(shape=LOUDBOX_SHAPE, factory=lambda m: self.stages.vision),
                "vlm_a": Node(shape=LOUDBOX_SHAPE, factory=lambda m: self.stages.vlm_a),
                "vlm_b": Node(shape=LOUDBOX_SHAPE, factory=lambda m: self.stages.vlm_b),
                "denoise": Node(shape=LOUDBOX_SHAPE, factory=lambda m: self.stages.denoise),
            },
            edges=[
                Edge("vision", "vlm_a"),
                Edge("vlm_a", "vlm_b"),
                Edge("vlm_b", "denoise"),
                # Plus the non-standard one-shot KV migration edge —
                # see _setup_kv_migration_sockets.
            ],
        )

        # 4. Build the tt-blaze Pipeline. The canonical PipelineBuilder.build
        # takes list[Callable[[MeshDevice], StageKind]]; the DAG-shaped
        # `build_graph` from design.md §3d isn't shipped yet, so we collapse
        # to a linear chain.
        builder = PipelineBuilder(mesh_device=mesh_device, submesh_shape=LOUDBOX_SHAPE)
        stage_factories = [
            lambda m: self.stages.vision,
            lambda m: self.stages.vlm_a,
            lambda m: self.stages.vlm_b,
            lambda m: self.stages.denoise,
        ]
        self.pipeline = builder.build(stage_factories)

        # 5. KV migration edge (one-shot at prefill completion).
        self._setup_kv_migration_sockets()

        return self

    def _submesh(self, i: int):
        return self.submeshes[i] if self.submeshes else None

    def _build_vision_stage(self, submesh) -> None:
        """Stage 0 — SigLIP-27 + multimodal projector.

        27 layers chained in ONE FusedOp at TP=8 across the loudbox. Each
        layer is a SiglipEncoderLayer.emit() invocation; layers chain via
        L1 CBs (no D2D between layers — that would be 27 socket
        crossings of ~1 µs each, wasted against the 8.9 ms target).

        Only the FIRST emit takes the ReceiverSocket from host; only the
        LAST emit takes the SenderSocket to vlm_a (loudbox B).
        Intermediates use CB-to-CB passing.

        This pattern mirrors how dense_layer/op.py chains MLA + CbReconfig
        + MLPMixed in one fused emit, just deeper (27 sub-emits).
        See mapping_notes.md §3 for the packing rationale.

        Sketch:
            tensors = [get_siglip_layer_tensors(submesh, layer_idx=i, ...)
                       for i in range(NUM_SIGLIP_LAYERS)]
            args = [get_siglip_layer_args(submesh, layer_idx=i, ...)
                    for i in range(NUM_SIGLIP_LAYERS)]
            self.stages.vision = ChainedSiglipStage(tensors, args)
        """
        # TODO(blaze): see sketch above.

    def _build_vlm_stage_a(self, submesh) -> None:
        """Stage 1 — VLM PaliGemma layers 0..8.

        9 layers chained as one FusedOp at TP=8. KV cache for layers 0..8
        lives stage-local. H2D-ish receiver from vision, D2D sender to
        vlm_b.
        """
        # TODO(blaze): mirror _build_vision_stage with VlmDecoderLayer ×9.

    def _build_vlm_stage_b(self, submesh) -> None:
        """Stage 2 — VLM PaliGemma layers 9..17.

        Same shape as Stage 1, layers 9..17. After this stage the prefill
        ACTIVATIONS are not forwarded to denoise — only the layer-paired
        K/V tensors are (via KV migration). The denoise stage receives
        its starting activation (noisy actions x_t) from host via a
        separate H2D socket on loudbox D.
        """
        # TODO(blaze): VlmDecoderLayer × 9, layers 9..17.

    def _build_denoise_stage(self, submesh) -> None:
        """Stage 3 — Expert decoder layers 0..17 + Suffix MLP + Euler loop.

        Per inference, this stage runs 10 iterations of:
            1. SuffixMlp.emit_input(x_t, t) → suffix activation [50, 1024]
            2. ExpertDecoderLayer.emit() × 18  (adaRMS modulation indexed
               by metadata.position_id = t; joint attn vs migrated VLM KV)
            3. SuffixMlp.emit_output(...) → dx_t
            4. Euler step: x_{t+1} = x_t + (1/N) * dx_t

        Two ways to drive the loop (see mapping_notes.md §5):
          (a) Host-driven: orchestrator pushes x_t / reads dx_t each
              step. 10 host roundtrips, ~100 µs overhead.
          (b) Device-driven via LoopingConfig(num_iterations=10):
              host pushes x_0 once, reads x_10 once. Latency-optimal.
              Requires the FusedOp to advance metadata.position_id
              internally and gate D2H on iteration 9.

        Default scaffold: (b). Fallback path called out in run_inference.

        The Euler step itself can be fused into SuffixMlp.emit_output
        (add x_t to dx_t before the D2H), keeping the loop on-chip.
        Reference for an internal-looping FusedOp: blaze/ops/sparse_layer/op.py
        per_loop_iter_setup() pattern.
        """
        # TODO(blaze): SuffixMlp.emit_input + ExpertDecoderLayer × 18 +
        # SuffixMlp.emit_output, wrapped in a LoopingConfig(num_iterations=10).

    # ── KV migration edge (out-of-band) ────────────────────────────────

    def _setup_kv_migration_sockets(self) -> None:
        """One-shot D2D migration from VLM stages → denoise stage.

        For each VLM layer L ∈ 0..17, open a persistent ttnn.MeshSocket
        from the chip(s) on the VLM stage holding L's K/V to the chip(s)
        on the denoise stage holding expert layer L's vlm_k_prefix /
        vlm_v_prefix slot.

        Fired ONCE per inference at prefill completion (not per token /
        per step). Persistent through all 10 denoise steps.

        Routing:
          VLM 0..8  → Loudbox B (stage 1) → ── 9 sockets ──→ Loudbox D
          VLM 9..17 → Loudbox C (stage 2) → ── 9 sockets ──→ Loudbox D
        18 sockets total, ~9 MB aggregate, ~1-2 ms wall clock at BH
        ethernet (deployment plan §3.3.a).

        Open question: tt-blaze's PipelineGraph today only represents
        per-token activation edges. This is a one-shot edge built
        outside the FusedOp's main socket path. See mapping_notes.md §4
        / open question 2.

        Sketch:
            for layer_idx in range(NUM_VLM_LAYERS):
                src = self.submeshes[1 if layer_idx < 9 else 2]
                dst = self.submeshes[3]
                sock = ttnn.MeshSocket(src, dst, ...)
                self.stages.kv_migration_sockets.append(sock)
        """
        # TODO(blaze): see sketch.

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start PipelineManager threads + FusedOps in persistent looping mode."""
        if self.is_started:
            return
        # TODO(blaze): self.pipeline.setup_and_run(mode=PipelineRunMode.FULL)
        # then start PipelineManager (pipeline_manager/architecture.md §5).
        self.is_started = True

    def stop(self) -> None:
        """Tear down sockets and stop FusedOps."""
        if not self.is_started:
            return
        # TODO(blaze): self.pipeline.terminate()
        self.is_started = False

    # ── One-shot inference path ────────────────────────────────────────

    def run_inference(self, pixel_values, lang_tokens, noisy_actions):
        """Run a single inference end-to-end.

        Inputs:
            pixel_values   [bs=2, 3, 224, 224] bf16   (base + wrist cams)
            lang_tokens    [200] int32                (padded)
            noisy_actions  [50, 32] bf16              (initial x_0 noise)

        Returns:
            actions [50, 32] bf16 — denoised x_10.

        Flow:
          1. Host → Stage 0 (vision) via H2D. (Text embeds pre-looked-up on
             host per mapping_notes.md §7; saves 527 MB / chip on loudbox A.)
          2. Stage 0: SigLIP-27 + projector → D2D → Stage 1.
          3. Stage 1: VLM layers 0..8 prefill → D2D → Stage 2.
          4. Stage 2: VLM layers 9..17 prefill → KV migration fires
             (18 sockets fire in parallel; ~1-2 ms).
          5. Stage 3: 10-step Euler denoise loop (device-driven or
             host-driven, see _build_denoise_stage).
          6. Stage 3 → host via D2H: x_10.
        """
        assert self.is_started, "call .start() first"
        # Sketch:
        # 1. vision_h2d_socket.write_tensor(pack(pixel_values, lang_embeds))
        # 2. wait for prefill-complete signal from stage 2
        # 3. fire KV migration sockets (all 18 in parallel)
        # 4. device-driven loop:
        #       denoise_h2d_socket.write_tensor(pack(noisy_actions, 0))
        #       return denoise_d2h_socket.read_tensor()
        #    OR host-driven loop:
        #       x = noisy_actions
        #       for step in range(DENOISE_STEPS):
        #           denoise_h2d_socket.write_tensor(pack(x, step))
        #           dx = denoise_d2h_socket.read_tensor()
        #           x = x + (1.0 / DENOISE_STEPS) * dx
        #       return x
        raise NotImplementedError("scaffold — driver bodies stubbed")


# ──────────────────────────────────────────────────────────────────────────
# PipelineConfiguration factory (canonical tt-blaze handoff format)
# ──────────────────────────────────────────────────────────────────────────


def create_pi0_5_pipeline_configuration(
    hf_weights: Dict[str, object],
    *,
    tp_size: int = 8,
    denoise_steps: int = DENOISE_STEPS,
):
    """Mirror blaze.models.deepseek_v3_b1.pipeline.create_single_pod_pipeline_configuration.

    Returns a PipelineConfiguration (fused_layer_contract.md §11) the
    PipelineManager / PipelineBuilder can consume directly:

        {0: lambda m: VisionStage(m, hf_weights),
         1: lambda m: VlmPrefillStage(m, hf_weights, layer_range=(0, 9)),
         2: lambda m: VlmPrefillStage(m, hf_weights, layer_range=(9, 18)),
         3: lambda m: DenoiseStage(m, hf_weights, num_steps=denoise_steps)}

    Each StageKind is a subclass; canonical example
    blaze/stages/deepseek_decoder.py.
    """
    raise NotImplementedError("populate per pi0.5 stage classes")
