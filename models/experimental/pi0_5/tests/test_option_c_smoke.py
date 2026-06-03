# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Option C smoke tests — heterogeneous 3-stage pipeline.

Layered exactly like `test_option_b_smoke.py`, each step stricter than
the previous:

  1. test_default_layout_shape_c           — pure dataclass, no HW.
  2. test_open_32_chip_mesh_partition_c    — opens 8×4 mesh + 3 heterogeneous
                                             submeshes (vision 4 / prefill 18 /
                                             denoise 6), no weights, no compute.
  3. test_vlm_slice_forward_one_layer_c    — one VLM layer (random weights)
                                             on the 18-chip prefill submesh,
                                             dummy forward, shape check.
  4. test_expert_slice_forward_one_layer_c — one expert layer (random
                                             weights + adaRMS) on the 6-chip
                                             denoise submesh, dummy forward,
                                             shape check.
  5. test_inter_submesh_host_bounce_c      — round-trip a tensor between two
                                             heterogeneous submeshes.
  6. test_e2e_vlm_to_expert_shrunk_c       — synthetic prefix → 1-layer
                                             prefill → layer-paired KV
                                             migrate → 1-layer expert step.
  7. test_full_pipeline_object_dry_run_c   — `Pi0_5PipelineC.initialize()`
                                             succeeds end-to-end on real
                                             weights with the shrunk layout
                                             (no full forward — that's the
                                             benchmark file's job).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig, PaliGemmaConfig
from models.experimental.pi0_5.tt.option_c.stages import (
    build_default_layout,
    build_shrunk_layout,
)
from models.experimental.pi0_5.tt.option_c.mesh_setup import (
    open_galaxy_mesh,
    describe_submesh,
)
from models.experimental.pi0_5.tt.option_c.vlm_slice import (
    Pi0_5OptionCVLMSlice,
    Pi0_5OptionCVLMSlicePaired,
)
from models.experimental.pi0_5.tt.option_c.expert_slice import (
    Pi0_5OptionCExpertSlice,
    Pi0_5OptionCExpertSlicePaired,
)
from models.experimental.pi0_5.tt.option_c.stage_prefill import StagePrefill
from models.experimental.pi0_5.tt.option_c.stage_denoise import StageDenoise
from models.experimental.pi0_5.tt.option_c.kv_migration import KVMigration
from models.experimental.pi0_5.tt.option_c.mesh_setup import (
    create_per_chip_submeshes,
    create_tp_submeshes_2x1,
)
from models.experimental.pi0_5.tt.option_c.transport import send_activation_via_host


# pi0.5 upstream-openpi LIBERO finetune (used only by the dry-run test;
# every test that needs it skips if the file isn't present).
_REAL_CKPT = "/home/tt-admin/pi05_cache/pi05_libero_upstream"


# ---------------------------------------------------------------------------- #
# Shared fixture: open the 8x4 parent mesh + 3 heterogeneous submeshes ONCE     #
# per pytest session.                                                           #
#                                                                               #
# tt-metal currently leaks `MetalContext` slots when `open_mesh_device` /       #
# `close_mesh_device` is called repeatedly in the same Python process — after   #
# ~5 cycles you hit "context_id out of range (max 32)" during teardown. So we   #
# open the parent mesh once and let each test carve / use the existing          #
# submeshes.                                                                    #
# ---------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def galaxy_mesh():
    """(parent, [vision, prefill, denoise]) on the canonical 4/18/6 layout."""
    layout = build_default_layout()
    with open_galaxy_mesh(layout) as (parent, submeshes):
        yield layout, parent, submeshes


@pytest.fixture(autouse=True)
def _release_l1_between_tests():
    """L1 leaks across function-scoped tests when the parent mesh is shared:
    pytest's traceback machinery keeps a failed test's locals alive (and
    successful tests can hold large ttnn.Tensor refs until the next gen-2 GC).
    Force a collection after every test so the slice/tensor objects from one
    test don't push the next one into L1 OOM.
    """
    yield
    import gc

    gc.collect()


def _close_micro_submeshes(*micro_submesh_lists):
    """Explicitly release sub-sub-submeshes created by `create_per_chip_submeshes`.

    tt-metal's parent-mesh `close_mesh_device` can't safely tear down deeply
    nested carved submeshes during fixture exit — it trips a "cq in use by
    parent mesh" assertion. Closing each micro-submesh at end-of-test
    (before fixture teardown reaches the parent) avoids that path.
    """
    for ms_list in micro_submesh_lists:
        if ms_list is None:
            continue
        for sm in ms_list:
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                # Best-effort cleanup; if a submesh is already gone or
                # tt-metal refuses, swallow — we just don't want to leave
                # the parent unable to close.
                pass


# ---------------------------------------------------------------------------- #
# Helpers (mirror test_option_b_smoke.py)                                       #
# ---------------------------------------------------------------------------- #


def _random_vlm_layer_weights(layer_idx: int, cfg: GemmaConfig) -> dict:
    """Random torch tensors keyed exactly as Pi0_5OptionCVLMSlice expects."""
    p = f"model.layers.{layer_idx}."
    W, M, H, KV, D = cfg.width, cfg.mlp_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    gen = torch.Generator().manual_seed(0xB1 + layer_idx)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    return {
        f"{p}self_attn.q_proj.weight": randn(H * D, W),
        f"{p}self_attn.k_proj.weight": randn(KV * D, W),
        f"{p}self_attn.v_proj.weight": randn(KV * D, W),
        f"{p}self_attn.o_proj.weight": randn(W, H * D),
        f"{p}mlp.gate_proj.weight": randn(M, W),
        f"{p}mlp.up_proj.weight": randn(M, W),
        f"{p}mlp.down_proj.weight": randn(W, M),
        f"{p}input_layernorm.weight": torch.zeros(W),
        f"{p}post_attention_layernorm.weight": torch.zeros(W),
    }


def _random_expert_layer_weights(layer_idx: int, cfg: GemmaConfig) -> dict:
    """Random torch tensors keyed exactly as Pi0_5OptionCExpertSlice expects."""
    p = f"model.layers.{layer_idx}."
    W, M, H, KV, D = cfg.width, cfg.mlp_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    gen = torch.Generator().manual_seed(0xE0 + layer_idx)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    return {
        f"{p}self_attn.q_proj.weight": randn(H * D, W),
        f"{p}self_attn.k_proj.weight": randn(KV * D, W),
        f"{p}self_attn.v_proj.weight": randn(KV * D, W),
        f"{p}self_attn.o_proj.weight": randn(W, H * D),
        f"{p}mlp.gate_proj.weight": randn(M, W),
        f"{p}mlp.up_proj.weight": randn(M, W),
        f"{p}mlp.down_proj.weight": randn(W, M),
        # adaRMS modulation Denses (out = 3*W, fused at upload).
        f"{p}input_layernorm.dense.weight": randn(3 * W, W),
        f"{p}input_layernorm.dense.bias": randn(3 * W),
        f"{p}post_attention_layernorm.dense.weight": randn(3 * W, W),
        f"{p}post_attention_layernorm.dense.bias": randn(3 * W),
    }


def _replicate(t: torch.Tensor, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh),
    )


# ---------------------------------------------------------------------------- #
# Layer 1 — pure dataclass                                                      #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(60)
def test_default_layout_shape_c():
    layout = build_default_layout()
    assert layout.parent_mesh_shape == (8, 4)
    assert len(layout.stages) == 3

    vision, prefill, denoise = layout.stages
    assert vision.name == "vision"
    assert vision.submesh_shape == (2, 4)
    assert vision.num_chips == 8
    assert vision.siglip_layer_range == (0, 27)
    assert vision.holds_mm_projector is True
    assert vision.holds_embed_tokens is True

    assert prefill.name == "prefill"
    assert prefill.submesh_shape == (6, 3)
    assert prefill.num_chips == 18
    assert prefill.vlm_layer_range == (0, 18)
    assert prefill.emits_kv_migration is True
    assert prefill.holds_vlm_final_norm is True

    assert denoise.name == "denoise"
    assert denoise.submesh_shape == (6, 1)
    assert denoise.num_chips == 6
    assert denoise.expert_layer_range == (0, 18)
    assert denoise.runs_denoise_loop is True
    assert denoise.receives_kv_migration is True
    assert denoise.holds_suffix_mlp is True


# ---------------------------------------------------------------------------- #
# Layer 2 — open the mesh + 3 heterogeneous submeshes                           #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(180)
def test_open_32_chip_mesh_partition_c(galaxy_mesh):
    """Confirm the shared parent mesh + 3 submeshes have the expected shapes."""
    layout, parent, submeshes = galaxy_mesh
    assert parent.shape[0] == 8 and parent.shape[1] == 4
    assert parent.get_num_devices() == 32, f"Expected 32 chips, got {parent.get_num_devices()}"
    assert len(submeshes) == 3, f"Expected 3 submeshes, got {len(submeshes)}"

    expected = [(2, 4, 8), (6, 3, 18), (6, 1, 6)]
    for i, (sm, (er, ec, en)) in enumerate(zip(submeshes, expected)):
        assert sm.get_num_devices() == en, (
            f"Submesh {i} has {sm.get_num_devices()} devices, expected {en} " f"({describe_submesh(sm)})"
        )
        assert (
            sm.shape[0] == er and sm.shape[1] == ec
        ), f"Submesh {i} shape is ({sm.shape[0]},{sm.shape[1]}), expected ({er},{ec})"
        print(f"stage {i} ({layout.stages[i].name}): {describe_submesh(sm)}")


# ---------------------------------------------------------------------------- #
# Layer 3 — one VLM layer forward on the prefill submesh                        #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(420)
def test_vlm_slice_forward_one_layer_c(galaxy_mesh):
    """One real-config VLM layer on the 18-chip prefill submesh."""
    _layout, _parent, submeshes = galaxy_mesh
    cfg = PaliGemmaConfig()  # depth=18
    weights = {"vlm_language": _random_vlm_layer_weights(0, cfg.vlm_config)}

    submesh = submeshes[1]  # prefill
    slice_ = Pi0_5OptionCVLMSlice(
        config=cfg,
        weights=weights,
        submesh=submesh,
        layer_range=(0, 1),
        holds_embed_tokens=False,
        holds_vlm_final_norm=False,
    )
    assert slice_.num_layers == 1 and len(slice_.vlm_blocks) == 1

    S = 64
    h = _replicate(torch.randn(1, S, cfg.vlm_config.width) * 0.02, submesh)
    mask = _replicate(torch.zeros(1, 1, S, S, dtype=torch.float32), submesh)

    out, _new_kv = slice_.forward(h, attention_mask=mask, use_cache=False)
    assert out.shape[-1] == cfg.vlm_config.width
    print(f"option_c VLM slice (prefill submesh) OK — out.shape={out.shape}")


# ---------------------------------------------------------------------------- #
# Layer 4 — one expert layer forward on the denoise submesh                     #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(420)
def test_expert_slice_forward_one_layer_c(galaxy_mesh):
    _layout, _parent, submeshes = galaxy_mesh
    cfg = PaliGemmaConfig()
    W = cfg.expert_config.width
    gen = torch.Generator().manual_seed(0xE2)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02

    expert = _random_expert_layer_weights(0, cfg.expert_config)
    expert["model.norm.dense.weight"] = randn(3 * W, W)
    expert["model.norm.dense.bias"] = randn(3 * W)
    weights = {"action_expert": expert}

    submesh = submeshes[2]  # denoise
    slice_ = Pi0_5OptionCExpertSlice(
        config=cfg,
        weights=weights,
        submesh=submesh,
        expert_layer_range=(0, 1),
    )
    assert slice_.num_layers == 1 and len(slice_.expert_blocks) == 1

    S = 64
    h = _replicate(torch.randn(1, S, W) * 0.02, submesh)
    adarms_cond = _replicate(torch.randn(1, W) * 0.02, submesh)
    mask = _replicate(torch.zeros(1, 1, S, S, dtype=torch.float32), submesh)

    out = slice_.forward(h, adarms_cond, attention_mask=mask)
    assert out.shape[-1] == W
    print(f"option_c expert slice (denoise submesh) OK — out.shape={out.shape}")


# ---------------------------------------------------------------------------- #
# Layer 5 — host-bounce transport between two heterogeneous submeshes           #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(180)
def test_inter_submesh_host_bounce_c(galaxy_mesh):
    """Validate host-bounce transport between the prefill and denoise submeshes."""
    _layout, _parent, submeshes = galaxy_mesh
    src_mesh, dst_mesh = submeshes[1], submeshes[2]
    original = torch.randn(1, 64, 2048, dtype=torch.float32) * 0.5
    src = _replicate(original, src_mesh)
    moved = send_activation_via_host(src, dst_mesh)
    assert moved.shape == src.shape, f"shape mismatch: {moved.shape} vs {src.shape}"
    assert moved.dtype == src.dtype
    print(f"option_c transport OK — moved.shape={moved.shape}, dtype={moved.dtype}")


# ---------------------------------------------------------------------------- #
# Layer 6 — shrunk e2e: prefill → KV migrate → expert step                      #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(600)
def test_e2e_vlm_to_expert_shrunk_c(galaxy_mesh):
    """Synthetic prefix → 1-layer prefill → layer-paired KV migrate → 1-layer
    expert step. Skips the vision stage (host SigLIP needs HF-compatible
    weights). Uses a shrunk layout for layer-range metadata; the submeshes
    come from the shared fixture.
    """
    _layout, _parent, submeshes = galaxy_mesh
    shrunk = build_shrunk_layout(vlm_depth=1, expert_depth=1)
    cfg = PaliGemmaConfig(
        vlm_config=GemmaConfig(width=2048, depth=1, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256),
        expert_config=GemmaConfig(width=1024, depth=1, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256),
    )

    vlm_lang = _random_vlm_layer_weights(0, cfg.vlm_config)
    vlm_lang["model.norm.weight"] = torch.zeros(cfg.vlm_config.width)

    W = cfg.expert_config.width
    expert_w = _random_expert_layer_weights(0, cfg.expert_config)
    gen = torch.Generator().manual_seed(0xEF)
    randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02
    expert_w["model.norm.dense.weight"] = randn(3 * W, W)
    expert_w["model.norm.dense.bias"] = randn(3 * W)

    weights = {"vlm_language": vlm_lang, "action_expert": expert_w}

    s = shrunk.stages
    prefill = StagePrefill(s[1], submeshes[1], cfg, weights)
    denoise = StageDenoise(s[2], submeshes[2], cfg, weights)
    migrator = KVMigration(denoise_submesh=submeshes[2])

    prefill.initialize()
    denoise.initialize(kv_migrator=migrator)

    S_prefix = 64
    h_on_1 = _replicate(torch.randn(1, S_prefix, cfg.vlm_config.width) * 0.02, submeshes[1])
    prefix_mask = _replicate(torch.zeros(1, 1, S_prefix, S_prefix, dtype=torch.float32), submeshes[1])

    _ = prefill.forward(h_on_1, attention_mask=prefix_mask, use_cache=True)
    kv1 = prefill.get_kv_cache()
    assert len(kv1) == 1 and kv1[0][0] == 0, f"prefill KV: {[(i, type(v)) for i, v in kv1]}"

    per_layer_kv = [None] * cfg.vlm_config.depth
    for gi, kv in kv1:
        per_layer_kv[gi] = kv
    migrator.migrate_layer_paired(per_layer_kv)
    migrated = migrator.as_list(cfg.vlm_config.depth)
    assert migrated[0] is not None, "KV migration produced no entry for layer 0"

    S_suffix = 64
    suffix_h = _replicate(torch.randn(1, S_suffix, W) * 0.02, submeshes[2])
    adarms_cond = _replicate(torch.randn(1, W) * 0.02, submeshes[2])
    joint_mask = _replicate(
        torch.zeros(1, 1, S_suffix, S_prefix + S_suffix, dtype=torch.float32),
        submeshes[2],
    )

    out = denoise.forward_expert_step(suffix_h, adarms_cond, prefix_kv_cache=migrated, attention_mask=joint_mask)
    assert out.shape[-1] == W, f"got {out.shape}"
    print(f"option_c shrunk e2e OK — out.shape={out.shape}, " f"K0.shape={migrated[0][0].shape}")


# ---------------------------------------------------------------------------- #
# Layer 7 — Pi0_5PipelineC.initialize() on real weights (dry run)               #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(900)
def test_full_pipeline_object_dry_run_c(galaxy_mesh):
    """`Pi0_5PipelineC(...).initialize()` succeeds on real pi05 weights.

    Confirms all stage orchestrators wire together cleanly — no forward
    is exercised (that's `tests/perf/test_option_c_benchmark.py`'s job
    once it's added). This is the structural completeness check
    referenced in `tt/option_c/README.md`.
    """
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.option_c.pipeline import Pi0_5PipelineC

    _layout, _parent, submeshes = galaxy_mesh
    loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = loader.categorized_weights
    cfg = PaliGemmaConfig()

    # Shrunk layout so the (currently replicated) per-layer uploads fit
    # the per-chip budget without needing layer-paired sharding yet.
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)

    pipe = Pi0_5PipelineC(
        layout=shrunk,
        submeshes=submeshes,
        config=cfg,
        weights=weights,
        denoise_steps=2,
    )
    pipe.initialize()
    assert pipe.stage_0 is not None
    assert pipe.stage_1 is not None
    assert pipe.stage_2 is not None
    assert pipe.kv_migrator is not None
    assert pipe.stage_1.slice is not None and pipe.stage_1.slice.num_layers == 2
    assert pipe.stage_2.slice is not None and pipe.stage_2.slice.num_layers == 1
    print(
        "option_c full-pipeline dry-run OK — "
        f"vision={describe_submesh(submeshes[0])}, "
        f"prefill={describe_submesh(submeshes[1])}, "
        f"denoise={describe_submesh(submeshes[2])}"
    )


# ---------------------------------------------------------------------------- #
# Layer 8 — layer-paired L1 prefill (1 VLM layer per chip, no replication)      #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(600)
def test_vlm_slice_layer_paired_l1_two_layers(galaxy_mesh):
    """Two VLM layers placed on TWO different prefill chips, each L1-resident.

    Exercises the Option C target placement: weights for layer 0 live only on
    the first prefill chip, weights for layer 1 live only on the second; the
    activation host-bounces between them. This is the path the README's
    "layer-paired L1 placement" deferred item refers to.
    """
    _layout, _parent, submeshes = galaxy_mesh
    prefill_mesh = submeshes[1]

    micro_submeshes = create_per_chip_submeshes(prefill_mesh, count=2)
    try:
        assert len(micro_submeshes) == 2
        for sm in micro_submeshes:
            assert sm.get_num_devices() == 1

        cfg = PaliGemmaConfig()
        weights = {
            "vlm_language": {
                **_random_vlm_layer_weights(0, cfg.vlm_config),
                **_random_vlm_layer_weights(1, cfg.vlm_config),
            }
        }

        slice_ = Pi0_5OptionCVLMSlicePaired(
            config=cfg,
            weights=weights,
            micro_submeshes=micro_submeshes,
            layer_range=(0, 2),
            holds_embed_tokens=False,
            holds_vlm_final_norm=False,
        )
        assert slice_.num_layers == 2 and len(slice_.vlm_blocks) == 2

        S = 64
        # Input activation lives on the FIRST chip (the layer-0 chip).
        h0 = _replicate(torch.randn(1, S, cfg.vlm_config.width) * 0.02, micro_submeshes[0])
        mask0 = _replicate(torch.zeros(1, 1, S, S, dtype=torch.float32), micro_submeshes[0])

        out, new_cache = slice_.forward(h0, attention_mask=mask0, use_cache=True)
        assert out.shape[-1] == cfg.vlm_config.width
        # Output must end up on the LAST chip in the chain.
        out_shards = ttnn.get_device_tensors(out)
        assert len(out_shards) == 1, f"layer-paired output must reside on one chip; got {len(out_shards)} shards"
        # KV cache: one entry per layer, keyed by global idx (0, 1).
        assert new_cache is not None
        assert new_cache[0] is not None and new_cache[1] is not None
        print(f"option_c layer-paired VLM slice OK — out.shape={out.shape}, " f"kv keys = [0, 1], chips used = 2")
    finally:
        _close_micro_submeshes(micro_submeshes)


@pytest.mark.timeout(600)
def test_stage_prefill_layer_paired_l1_dry_run(galaxy_mesh):
    """StagePrefill in layer-paired mode end-to-end on 2 chips.

    Same coverage as `test_vlm_slice_layer_paired_l1_two_layers` but through
    the stage orchestrator: confirms `StagePrefill(..., layer_paired_l1=True)`
    constructs the paired slice, exposes `first_chip_submesh` / `last_chip_submesh`,
    and produces a KV cache the migrator can consume.
    """
    _layout, _parent, submeshes = galaxy_mesh
    shrunk = build_shrunk_layout(vlm_depth=2, expert_depth=1)
    cfg = PaliGemmaConfig()
    weights = {
        "vlm_language": {
            **_random_vlm_layer_weights(0, cfg.vlm_config),
            **_random_vlm_layer_weights(1, cfg.vlm_config),
            "model.norm.weight": torch.zeros(cfg.vlm_config.width),
        }
    }

    prefill = StagePrefill(
        shrunk.stages[1],
        submeshes[1],
        cfg,
        weights,
        layer_paired_l1=True,
    )
    prefill.initialize()
    try:
        assert prefill.layer_paired_l1 is True
        assert prefill.micro_submeshes is not None and len(prefill.micro_submeshes) == 2
        assert isinstance(prefill.slice, Pi0_5OptionCVLMSlicePaired)
        assert prefill.first_chip_submesh is prefill.micro_submeshes[0]
        assert prefill.last_chip_submesh is prefill.micro_submeshes[1]

        S = 64
        h0 = _replicate(torch.randn(1, S, cfg.vlm_config.width) * 0.02, prefill.first_chip_submesh)
        mask0 = _replicate(torch.zeros(1, 1, S, S, dtype=torch.float32), prefill.first_chip_submesh)
        h_after = prefill.forward(h0, attention_mask=mask0, use_cache=True)
        assert h_after.shape[-1] == cfg.vlm_config.width

        kv_entries = prefill.get_kv_cache()
        assert len(kv_entries) == 2, f"expected 2 KV entries, got {len(kv_entries)}"
        assert kv_entries[0][0] == 0 and kv_entries[1][0] == 1
        print(
            f"option_c layer-paired prefill stage OK — kv entries = "
            f"{[g for g, _ in kv_entries]}, out.shape={h_after.shape}"
        )
    finally:
        _close_micro_submeshes(prefill.micro_submeshes)


# ---------------------------------------------------------------------------- #
# Layer 9 — layer-paired L1 denoise (3 expert layers per chip × N chips)        #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(600)
def test_expert_slice_layer_paired_l1_two_chips(galaxy_mesh):
    """Two-chip layer-paired expert: 3 layers on chip 0, 3 on chip 1.

    Exercises chip-to-chip activation transport inside the expert backbone —
    the target Option C placement for the denoise submesh.
    """
    _layout, _parent, submeshes = galaxy_mesh
    denoise_mesh = submeshes[2]

    micro_submeshes = create_per_chip_submeshes(denoise_mesh, count=2)
    try:
        assert len(micro_submeshes) == 2

        cfg = PaliGemmaConfig()
        W = cfg.expert_config.width
        gen = torch.Generator().manual_seed(0xEC)
        randn = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float32) * 0.02

        expert = {}
        for layer_idx in range(6):  # 2 chips * 3 layers
            expert.update(_random_expert_layer_weights(layer_idx, cfg.expert_config))
        expert["model.norm.dense.weight"] = randn(3 * W, W)
        expert["model.norm.dense.bias"] = randn(3 * W)

        slice_ = Pi0_5OptionCExpertSlicePaired(
            config=cfg,
            weights={"action_expert": expert},
            micro_submeshes=micro_submeshes,
            expert_layer_range=(0, 6),
            layers_per_chip=3,
        )
        assert slice_.num_layers == 6 and len(slice_.expert_blocks) == 6
        assert slice_.chip_for_layer == [0, 0, 0, 1, 1, 1]

        S = 64
        h0 = _replicate(torch.randn(1, S, W) * 0.02, micro_submeshes[0])
        adarms_cond0 = _replicate(torch.randn(1, W) * 0.02, micro_submeshes[0])
        mask0 = _replicate(torch.zeros(1, 1, S, S, dtype=torch.float32), micro_submeshes[0])

        out = slice_.forward(h0, adarms_cond0, attention_mask=mask0)
        assert out.shape[-1] == W
        out_shards = ttnn.get_device_tensors(out)
        assert len(out_shards) == 1, f"layer-paired expert output must end on one chip; got {len(out_shards)} shards"
        print(
            f"option_c layer-paired expert slice OK — out.shape={out.shape}, " f"chip_for_layer={slice_.chip_for_layer}"
        )
    finally:
        _close_micro_submeshes(micro_submeshes)


# ---------------------------------------------------------------------------- #
# Layer 10 — on-device 3-chip SigLIP split + mm_projector chip                  #
# ---------------------------------------------------------------------------- #


@pytest.mark.timeout(900)
@pytest.mark.timeout(120)
def test_prefill_tp_2x1_submesh_carving(galaxy_mesh):
    """Carve the (6,3) prefill submesh into 9 (2,1) col-pair sub-meshes.

    Step 1 sanity check for OPTION_C_TP_WITHIN_STAGE_PLAN.md — confirms
    `create_tp_submeshes_2x1` returns exactly 9 sub-meshes, each with
    exactly 2 devices. No forward, no weight upload, no fabric required.
    """
    _layout, _parent, submeshes = galaxy_mesh
    prefill_mesh = submeshes[1]
    assert prefill_mesh.shape[0] == 6 and prefill_mesh.shape[1] == 3, (
        f"prefill submesh shape ({prefill_mesh.shape[0]},{prefill_mesh.shape[1]}) " f"is not (6,3)"
    )

    tp_subs = create_tp_submeshes_2x1(prefill_mesh)
    try:
        assert len(tp_subs) == 9, f"expected 9 (2,1) sub-meshes, got {len(tp_subs)}"
        for i, sm in enumerate(tp_subs):
            assert sm.get_num_devices() == 2, f"tp_submeshes[{i}] has {sm.get_num_devices()} devices, expected 2"
            assert (
                sm.shape[0] == 2 and sm.shape[1] == 1
            ), f"tp_submeshes[{i}] shape ({sm.shape[0]},{sm.shape[1]}) is not (2,1)"
        print(f"prefill (6,3) → 9 × (2,1) sub-meshes (2 chips each) OK")
    finally:
        _close_micro_submeshes(tp_subs)


def test_vision_slice_device_siglip_split_dry_run(galaxy_mesh):
    """Pi0_5OptionCVisionSliceSplit constructs on real weights (no forward).

    The on-device SigLIP split now spans the full 8-chip (2,4) vision
    submesh (post full-L1 redesign). This test confirms the carving +
    per-chip weight upload path succeeds for the wider layout. The
    forward path is exercised separately by the benchmark file once
    it's wired into Pi0_5PipelineC.
    """
    from pathlib import Path

    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader
    from models.experimental.pi0_5.tt.option_c.vision_slice import Pi0_5OptionCVisionSliceSplit

    _layout, _parent, submeshes = galaxy_mesh
    vision_mesh = submeshes[0]
    assert (
        vision_mesh.get_num_devices() == 8
    ), f"vision submesh should be 8 chips ((2,4)), got {vision_mesh.get_num_devices()}"

    micro_submeshes = create_per_chip_submeshes(vision_mesh, count=8)
    try:
        assert len(micro_submeshes) == 8

        loader = Pi0_5WeightLoader(_REAL_CKPT)
        weights = loader.categorized_weights
        cfg = PaliGemmaConfig()

        # Default layout: 8 SigLIP chunks (4+4+4+4+3+3+3+2 layers) with
        # mm_projector co-located on the last chip. See
        # Pi0_5OptionCVisionSliceSplit docstring for the L1 fit
        # rationale behind the redistribution.
        expected_layers_per_chip = [4, 4, 4, 4, 3, 3, 3, 2]
        assert sum(expected_layers_per_chip) == 27
        split = Pi0_5OptionCVisionSliceSplit(
            config=cfg,
            weights=weights,
            micro_submeshes=micro_submeshes,
            siglip_depth=27,
        )
        assert len(split.siglip_chunks) == 8
        assert split.layers_per_chip == expected_layers_per_chip
        assert split.projector_chip_idx == 7
        expected_lo = 0
        last_chunk_idx = len(expected_layers_per_chip) - 1
        for chunk_idx, chunk in enumerate(split.siglip_chunks):
            n = expected_layers_per_chip[chunk_idx]
            assert chunk.layer_lo == expected_lo
            assert chunk.layer_hi == expected_lo + n
            assert chunk.holds_patch_embed == (chunk_idx == 0)
            assert chunk.holds_pos_embed == (chunk_idx == 0)
            assert chunk.holds_post_ln == (chunk_idx == last_chunk_idx)
            expected_lo += n
        assert split.mm_projector is not None
        print(
            f"option_c device-SigLIP split "
            f"(8 chips × {'+'.join(map(str, expected_layers_per_chip))}, "
            f"projector co-located on chip {last_chunk_idx}) OK"
        )
    finally:
        _close_micro_submeshes(micro_submeshes)
