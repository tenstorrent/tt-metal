"""Tests for the constraint catalog, checker, recipe library, and
prompt injection.

Focus: pin the rule logic + recipe-lookup + markdown formatting. The
catalog is meant to be cheap and additive, so adding new constraints
in the future should NOT break these tests — they test the rules that
exist today by name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.constraints import (  # noqa: E402
    Catalog,
    Constraint,
    OpCallContext,
    Violation,
    check_component,
    format_constraint_hints,
)
from scripts.tt_hw_planner.constraints.catalog import default_catalog  # noqa: E402
from scripts.tt_hw_planner.constraints.recipes import RECIPES  # noqa: E402


# ---------------------------------------------------------------------------
# Catalog plumbing — base shape
# ---------------------------------------------------------------------------


def test_default_catalog_has_constraints() -> None:
    cat = default_catalog()
    assert len(cat.constraints) > 0
    names = {c.name for c in cat.constraints}
    # The constraints we shipped are present
    assert "layer_norm_sub_tile_dim" in names
    assert "conv_bias_dtype_mismatch" in names


def test_catalog_run_returns_empty_when_no_rule_fires() -> None:
    cat = default_catalog()
    ctx = OpCallContext(component_name="x", hf_class_name="UnrelatedClass")
    assert cat.run(ctx) == []


def test_buggy_constraint_does_not_take_down_catalog() -> None:
    def _explodes(ctx: OpCallContext):
        raise RuntimeError("kaboom")

    cat = Catalog(constraints=[Constraint("buggy", _explodes)])
    # Must not raise; just returns no violations.
    assert cat.run(OpCallContext(component_name="x", hf_class_name="y")) == []


# ---------------------------------------------------------------------------
# layer_norm small-dim rule
# ---------------------------------------------------------------------------


def test_layer_norm_fires_on_sub_tile_channel_dim() -> None:
    """SAM2 case: video_layer_norm with [1, 4, 8, 8] input — channel
    dim of 4 should fire."""
    ctx = OpCallContext(
        component_name="video_layer_norm",
        hf_class_name="Sam2VideoLayerNorm",
        input_shapes=[[1, 4, 8, 8]],
        input_dtypes=["torch.float32"],
    )
    violations = default_catalog().run(ctx)
    names = {v.constraint_name for v in violations}
    assert "layer_norm_sub_tile_dim" in names


def test_layer_norm_does_not_fire_on_aligned_dim() -> None:
    ctx = OpCallContext(
        component_name="layer_norm",
        hf_class_name="LayerNorm",
        input_shapes=[[1, 64, 32, 32]],
    )
    violations = [v for v in default_catalog().run(ctx) if v.constraint_name == "layer_norm_sub_tile_dim"]
    assert violations == []


def test_layer_norm_doesnt_fire_for_non_norm_component() -> None:
    ctx = OpCallContext(
        component_name="conv_thing",
        hf_class_name="ConvBlock",
        input_shapes=[[1, 4, 8, 8]],  # has sub-tile dim
    )
    violations = [v for v in default_catalog().run(ctx) if v.constraint_name == "layer_norm_sub_tile_dim"]
    assert violations == []


def test_layer_norm_last_dim_unaligned_fires() -> None:
    """Last dim 96 isn't a multiple of 32 (96 = 3*32, so this WOULD be
    aligned). Test a true unaligned case: last_dim = 80."""
    ctx = OpCallContext(
        component_name="ln",
        hf_class_name="LayerNorm",
        input_shapes=[[1, 16, 80]],
    )
    violations = default_catalog().run(ctx)
    assert any(v.constraint_name == "layer_norm_last_dim_not_tile_aligned" for v in violations)


def test_layer_norm_last_dim_aligned_no_fire() -> None:
    ctx = OpCallContext(
        component_name="ln",
        hf_class_name="LayerNorm",
        input_shapes=[[1, 16, 64]],
    )
    violations = [v for v in default_catalog().run(ctx) if v.constraint_name == "layer_norm_last_dim_not_tile_aligned"]
    assert violations == []


# ---------------------------------------------------------------------------
# Conv bias dtype rule
# ---------------------------------------------------------------------------


def test_conv_bias_dtype_fires_for_bfloat16_input() -> None:
    ctx = OpCallContext(
        component_name="video_mask_down_sampler",
        hf_class_name="Sam2VideoMaskDownSampler",
        input_shapes=[[1, 1, 256, 256]],
        input_dtypes=["torch.bfloat16"],
    )
    violations = default_catalog().run(ctx)
    assert any(v.constraint_name == "conv_bias_dtype_mismatch" for v in violations)


def test_conv_bias_dtype_no_fire_for_fp32_input() -> None:
    ctx = OpCallContext(
        component_name="video_mask_down_sampler",
        hf_class_name="Sam2VideoMaskDownSampler",
        input_shapes=[[1, 1, 256, 256]],
        input_dtypes=["torch.float32"],
    )
    violations = [v for v in default_catalog().run(ctx) if v.constraint_name == "conv_bias_dtype_mismatch"]
    assert violations == []


# ---------------------------------------------------------------------------
# Position-embedding rule
# ---------------------------------------------------------------------------


def test_position_embedding_shape_arg_fires() -> None:
    ctx = OpCallContext(
        component_name="video_position_embedding_sine",
        hf_class_name="Sam2VideoPositionEmbeddingSine",
        input_shapes=[[1, 256, 32, 32]],
    )
    violations = default_catalog().run(ctx)
    assert any(v.constraint_name == "position_embedding_shape_arg" for v in violations)


def test_position_embedding_no_fire_for_unrelated_component() -> None:
    ctx = OpCallContext(
        component_name="something_else",
        hf_class_name="Linear",
        input_shapes=[[1, 256]],
    )
    violations = [v for v in default_catalog().run(ctx) if v.constraint_name == "position_embedding_shape_arg"]
    assert violations == []


# ---------------------------------------------------------------------------
# Matmul K-dim rule
# ---------------------------------------------------------------------------


def test_matmul_k_dim_fires_for_unaligned_k() -> None:
    ctx = OpCallContext(
        component_name="linear_proj",
        hf_class_name="LinearProjection",
        input_shapes=[[1, 64, 80]],  # K=80, not a multiple of 32
    )
    violations = default_catalog().run(ctx)
    assert any(v.constraint_name == "matmul_k_dim_not_tile_aligned" for v in violations)


def test_matmul_k_dim_no_fire_for_aligned_k() -> None:
    ctx = OpCallContext(
        component_name="linear_proj",
        hf_class_name="LinearProjection",
        input_shapes=[[1, 64, 64]],
    )
    violations = [v for v in default_catalog().run(ctx) if v.constraint_name == "matmul_k_dim_not_tile_aligned"]
    assert violations == []


# ---------------------------------------------------------------------------
# check_component reads manifest + opplan from disk
# ---------------------------------------------------------------------------


def test_check_component_loads_manifest(tmp_path: Path) -> None:
    manifest = {
        "component": "video_layer_norm",
        "args": {
            "kind": "tuple",
            "items": [{"kind": "tensor", "shape": [1, 4, 8, 8], "dtype": "torch.float32"}],
        },
        "kwargs": {"kind": "dict", "items": {}},
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    violations = check_component(
        component_name="video_layer_norm",
        hf_class_name="Sam2VideoLayerNorm",
        manifest_path=p,
    )
    assert any(v.constraint_name == "layer_norm_sub_tile_dim" for v in violations)


def test_check_component_handles_missing_files(tmp_path: Path) -> None:
    """No manifest, no opplan — should NOT raise; returns whatever
    rules fire based on the bare name (typically none)."""
    violations = check_component(
        component_name="video_layer_norm",
        hf_class_name="Sam2VideoLayerNorm",
        manifest_path=tmp_path / "does_not_exist.json",
        opplan_path=tmp_path / "neither.json",
    )
    # No shapes → small-dim rule does not fire
    assert not any(v.constraint_name == "layer_norm_sub_tile_dim" for v in violations)


def test_check_component_handles_malformed_manifest(tmp_path: Path) -> None:
    p = tmp_path / "manifest.json"
    p.write_text("not json")
    violations = check_component(
        component_name="x",
        hf_class_name="X",
        manifest_path=p,
    )
    assert isinstance(violations, list)


# ---------------------------------------------------------------------------
# Recipes — every catalog rule has a registered recipe
# ---------------------------------------------------------------------------


def test_every_catalog_rule_has_a_recipe() -> None:
    """Whenever a constraint fires, the prompt injector looks up its
    recipe_id. Every shipped recipe_id must be present in RECIPES."""
    cat = default_catalog()
    # Run all rules against synthetic contexts that maximize fire rate
    ctxs = [
        OpCallContext("x", "LayerNorm", input_shapes=[[1, 4, 8, 8]]),
        OpCallContext("x", "LayerNorm", input_shapes=[[1, 16, 80]]),
        OpCallContext("x", "ConvBlock", input_shapes=[[1, 1, 4, 4]], input_dtypes=["bfloat16"]),
        OpCallContext("x", "PositionEmbeddingSine", input_shapes=[[1, 4]]),
        OpCallContext("x", "LinearProjection", input_shapes=[[1, 80]]),
    ]
    seen_recipe_ids = set()
    for ctx in ctxs:
        for v in cat.run(ctx):
            seen_recipe_ids.add(v.recipe_id)
    assert seen_recipe_ids, "expected at least one violation from synthetic ctxs"
    for rid in seen_recipe_ids:
        assert rid in RECIPES, f"recipe_id `{rid}` referenced by a constraint but not registered"


def test_recipe_body_renders_with_details() -> None:
    """Recipe bodies use .format() substitution against violation.details.
    The render call must succeed for the synthetic violations we ship."""
    cat = default_catalog()
    ctx = OpCallContext("video_layer_norm", "LayerNorm", input_shapes=[[1, 4, 8, 8]])
    violations = cat.run(ctx)
    block = format_constraint_hints(violations)
    assert block != ""
    assert "small_dim_layer_norm" in block or "Small-dim LayerNorm" in block
    # The substituted small_dim value should appear in the block
    assert "4" in block


# ---------------------------------------------------------------------------
# format_constraint_hints — block shape
# ---------------------------------------------------------------------------


def test_format_empty_when_no_violations() -> None:
    assert format_constraint_hints([]) == ""


def test_format_contains_header_and_each_violation_section() -> None:
    violations = [
        Violation(
            constraint_name="example_constraint",
            description="example description",
            recipe_id="small_dim_layer_norm",
            details={"shape": [1, 4, 8, 8], "small_dim": 4, "tile_width": 32},
        )
    ]
    block = format_constraint_hints(violations)
    assert "TTNN CONSTRAINT WARNINGS" in block
    assert "example_constraint" in block
    assert "example description" in block


def test_format_robust_to_missing_recipe_id() -> None:
    """If a Violation references a recipe_id that isn't in RECIPES, the
    formatter must NOT raise — it should mark the gap and continue."""
    violations = [
        Violation(
            constraint_name="invented",
            description="d",
            recipe_id="does_not_exist",
        )
    ]
    block = format_constraint_hints(violations)
    assert "does_not_exist" in block


def test_format_robust_to_missing_template_placeholders() -> None:
    """If a recipe's template references a placeholder that isn't in
    details, the formatter falls back to the raw body, not a crash."""
    violations = [
        Violation(
            constraint_name="example",
            description="d",
            recipe_id="small_dim_layer_norm",
            details={},  # missing all placeholders
        )
    ]
    block = format_constraint_hints(violations)
    assert block != ""  # didn't crash
    assert "small_dim" in block.lower() or "Small-dim" in block


# ---------------------------------------------------------------------------
# Regression: parallel-agent prompt MUST include the constraint block
# (bug: assemble_iter_prompt previously dropped it, leaving extras blind)
# ---------------------------------------------------------------------------


def test_parallel_prompt_assembler_threads_constraint_block_through() -> None:
    """Pins the iter_prompt.assemble_iter_prompt signature: it must
    accept the constraint_block kwarg and splice it BEFORE failure_context.
    Without this, parallel-agent extras get no catalog hints — a silent
    regression we want to catch in CI."""
    from scripts.tt_hw_planner._cli_helpers.iter_prompt import assemble_iter_prompt

    out = assemble_iter_prompt(
        hw_header="HW\n",
        task_block="TASK\n",
        systemic_block="",
        shape_probe_block="",
        agentic_block="",
        budget_clause="",
        failure_context="FAILURE\n",
        strategy_directive="strat",
        escalated_scope_block="",
        native_directive="",
        cross_component_block="",
        components_block="COMP\n",
        constraint_block="CONSTRAINT_BLOCK_MARKER\n",
    )
    assert "CONSTRAINT_BLOCK_MARKER" in out
    # Block appears BEFORE failure_context so the LLM sees the
    # actionable hint above the (often noisy) raw failure trace.
    assert out.index("CONSTRAINT_BLOCK_MARKER") < out.index("FAILURE")


def test_build_constraint_block_for_sub_tile_case(tmp_path) -> None:
    """End-to-end: given a manifest with a sub-tile dim, the shared
    helper should produce a non-empty constraint block with the
    catalog recipe text."""
    import json

    from scripts.tt_hw_planner._cli_helpers.iter_prompt import build_constraint_block

    demo = tmp_path
    (demo / "_captured" / "video_layer_norm").mkdir(parents=True)
    (demo / "_stubs").mkdir(parents=True)
    (demo / "_captured" / "video_layer_norm" / "manifest.json").write_text(
        json.dumps(
            {
                "component": "video_layer_norm",
                "hf_class": "Sam2VideoLayerNorm",
                "args": {
                    "kind": "tuple",
                    "items": [{"kind": "tensor", "shape": [1, 4, 8, 8], "dtype": "torch.float32"}],
                },
                "kwargs": {"kind": "dict", "items": {}},
            }
        )
    )
    block = build_constraint_block(demo_dir=demo, target_component="video_layer_norm")
    assert "TTNN CONSTRAINT WARNINGS" in block
    assert "padding-poisoned" in block
