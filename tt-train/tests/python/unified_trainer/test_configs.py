# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Configs and the YAML recipe loader. Device-free."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ttml.unified_trainer import SFTTrainerConfig, load_config
from ttml.unified_trainer.configs import apply_override

RECIPES = Path(__file__).resolve().parents[3] / "configs" / "training_configs" / "unified_trainer"

MINIMAL = {"dataset_id": "sst2", "model_name": "meta-llama/Llama-3.2-1B", "learning_rate": 1e-4, "batch_size": 8}


def sft(**overrides):
    return SFTTrainerConfig.model_validate({**MINIMAL, "max_length": 64, **overrides})


def test_defaults_train_every_weight_without_checkpoints():
    config = sft()
    assert config.lora is None
    assert config.checkpoint.save_strategy == "none"
    assert (config.num_epochs, config.gradient_accumulation_steps, config.max_steps, config.val_steps_freq) == (
        1,
        1,
        0,
        0,
    )
    assert config.metrics.train_metrics == ["loss"]
    assert not config.logging.use_wandb


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 2},
        {"lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "rank": 4}},
        {"checkpoint": {"save_every": 10}},
        {"metrics": {"log_freq": 10}},
        {"logging": {"wandb_watch_mode": "all"}},
    ],
    ids=["top level", "lora", "checkpoint", "metrics", "logging"],
)
def test_unknown_fields_are_rejected(overrides, expect_error):
    with expect_error(ValidationError, "Extra inputs are not permitted"):
        sft(**overrides)


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"learning_rate": -1.0}, "learning_rate"),
        ({"max_length": 0}, "max_length"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"lora": {"r": 0, "alpha": 8, "target_modules": ["q_proj"]}}, "lora.r"),
        ({"lora": {"r": 4, "alpha": 8, "target_modules": []}}, "lora.target_modules"),
        ({"lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "dropout": 1.0}}, "lora.dropout"),
        ({"checkpoint": {"save_strategy": "always"}}, "checkpoint.save_strategy"),
        ({"checkpoint": {"keep_last_n": 0}}, "checkpoint.keep_last_n"),
        ({"master_dtype": "float16"}, "master_dtype"),
    ],
)
def test_invalid_values_are_rejected(overrides, field, expect_error):
    with expect_error(ValidationError, rf"\n{field}\n"):
        sft(**overrides)


def test_a_custom_dataset_needs_its_files(expect_error):
    with expect_error(ValidationError, "custom_dataset is required"):
        sft(dataset_id="custom")
    assert (
        sft(dataset_id="custom", custom_dataset={"train_dataset_path": "train.jsonl"}).custom_dataset.template
        == "alpaca"
    )


def test_model_name_is_a_field():
    assert sft().model_name == MINIMAL["model_name"]
    assert "model_name" in sft().model_dump(mode="json")


@pytest.mark.parametrize(
    "override, expected",
    [
        ("lora.r=16", {"lora": {"r": 16}}),
        ("learning_rate=1.0e-5", {"learning_rate": 1e-5}),
        ("lora.target_modules=[q_proj, k_proj]", {"lora": {"target_modules": ["q_proj", "k_proj"]}}),
        ("checkpoint.save_strategy=step", {"checkpoint": {"save_strategy": "step"}}),
        (
            "custom_dataset.column_mapping={output: answer}",
            {"custom_dataset": {"column_mapping": {"output": "answer"}}},
        ),
    ],
)
def test_apply_override_parses_values_as_yaml(override, expected):
    data = {}
    apply_override(data, override)
    assert data == expected


@pytest.mark.parametrize(
    "override, error",
    [
        ("lora.r", "expected dotted.key=value"),
        ("=4", "expected dotted.key=value"),
        ("learning_rate.x=1", "'learning_rate' is not a section"),
    ],
)
def test_apply_override_rejects_malformed_overrides(override, error, expect_error):
    with expect_error(ValueError, error):
        apply_override({"learning_rate": 1.0}, override)


def test_load_config_applies_overrides(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("dataset_id: sst2\nmodel_name: m\nlearning_rate: 1.0e-4\nbatch_size: 8\nmax_length: 64\n")
    config = load_config(recipe, ["batch_size=2", "lora.r=4", "lora.alpha=8", "lora.target_modules=[o_proj]"])
    assert config.batch_size == 2 and config.lora.target_modules == ["o_proj"]


def test_load_config_rejects_a_non_mapping(tmp_path, expect_error):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("- a\n- b\n")
    with expect_error(ValueError, "expected a mapping"):
        load_config(recipe)


@pytest.mark.parametrize("recipe", sorted(RECIPES.glob("*.yaml")), ids=lambda path: path.stem)
def test_every_recipe_validates(recipe):
    config = load_config(recipe)
    assert config.max_length % 32 == 0


def test_there_are_recipes():
    assert len(list(RECIPES.glob("*.yaml"))) >= 4
