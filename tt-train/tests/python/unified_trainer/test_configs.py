# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Configs and the YAML recipe loader. Device-free."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ttml.unified_trainer import (
    AdamWConfig,
    BestRotation,
    EpochSave,
    OptimizerConfig,
    SFTTrainerConfig,
    StepSave,
    load_config,
)
from ttml.unified_trainer.configs import apply_override

RECIPES = Path(__file__).resolve().parents[3] / "configs" / "training_configs" / "unified_trainer"

MINIMAL = {"dataset_id": "sst2", "model_name": "meta-llama/Llama-3.2-1B", "optimizer": {"lr": 1e-4}, "batch_size": 8}


def sft(**overrides):
    return SFTTrainerConfig.model_validate({**MINIMAL, "max_length": 64, **overrides})


def test_defaults_train_every_weight_without_checkpoints():
    config = sft()
    assert config.lora is None
    assert config.checkpoint.save_strategy.type == "none"
    assert config.checkpoint.init_strategy is None
    assert not config.checkpoint.save_optim
    assert (config.num_epochs, config.gradient_accumulation_steps, config.max_steps, config.val_steps_freq) == (
        1,
        1,
        0,
        0,
    )
    assert config.metrics.train_metrics == ["loss"]
    assert not config.logging.use_wandb
    assert len(config.optimizer) == 1 and config.optimizer[0].modules is None
    assert isinstance(config.optimizer[0], AdamWConfig) and config.optimizer[0].weight_decay == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 2},
        {"lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "rank": 4}},
        {"optimizer": {"lr": 1e-4, "momentum": 0.9}},
        {"checkpoint": {"save_every": 10}},
        {"checkpoint": {"save_strategy": {"type": "none", "frequency": 10}}},
        {"checkpoint": {"rotation_strategy": {"type": "last", "metric": "val/loss"}}},
        {"metrics": {"log_freq": 10}},
        {"logging": {"wandb_watch_mode": "all"}},
    ],
    ids=["top level", "lora", "optimizer", "checkpoint", "save strategy", "rotation strategy", "metrics", "logging"],
)
def test_unknown_fields_are_rejected(overrides, expect_error):
    with expect_error(ValidationError, "Extra inputs are not permitted"):
        sft(**overrides)


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"optimizer": {"lr": -1.0}}, "optimizer.0.lr"),
        ({"optimizer": {"lr": 1e-4, "weight_decay": -1.0}}, "optimizer.0.weight_decay"),
        ({"optimizer": {"lr": 1e-4, "beta2": 1.0}}, "optimizer.0.beta2"),
        ({"optimizer": {"type": "sgd", "lr": 1e-4}}, "optimizer.0"),
        ({"optimizer": OptimizerConfig(lr=1e-4)}, "optimizer.0"),
        ({"optimizer": {}}, "optimizer.0.lr"),
        ({"optimizer": []}, "optimizer"),
        ({"optimizer": {"lr": 1e-4, "modules": []}}, "optimizer.0.modules"),
        ({"max_length": 0}, "max_length"),
        ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps"),
        ({"lora": {"r": 0, "alpha": 8, "target_modules": ["q_proj"]}}, "lora.r"),
        ({"lora": {"r": 4, "alpha": 8, "target_modules": []}}, "lora.target_modules"),
        ({"lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "dropout": 1.0}}, "lora.dropout"),
        ({"checkpoint": {"save_strategy": "step"}}, "checkpoint.save_strategy"),
        ({"checkpoint": {"save_strategy": {"type": "always"}}}, "checkpoint.save_strategy"),
        (
            {"checkpoint": {"save_strategy": {"type": "step", "frequency": 0}}},
            "checkpoint.save_strategy.step.frequency",
        ),
        (
            {"checkpoint": {"rotation_strategy": {"type": "last", "keep_n": 0}}},
            "checkpoint.rotation_strategy.last.keep_n",
        ),
        (
            {"checkpoint": {"rotation_strategy": {"type": "best", "mode": "lowest"}}},
            "checkpoint.rotation_strategy.best.mode",
        ),
        ({"checkpoint": {"init_strategy": {"type": "last", "path": "runs"}}}, "checkpoint.init_strategy.type"),
        ({"checkpoint": {"init_strategy": {"type": "resume"}}}, "checkpoint.init_strategy.path"),
        ({"master_dtype": "float16"}, "master_dtype"),
    ],
)
def test_invalid_values_are_rejected(overrides, field, expect_error):
    with expect_error(ValidationError, rf"\n{field}\n"):
        sft(**overrides)


def test_the_optimizer_type_picks_its_config():
    assert sft(optimizer={"type": "adamw", "lr": 1e-4, "beta1": 0.8}).optimizer[0].beta1 == 0.8
    assert sft(optimizer=AdamWConfig(lr=1e-4)).optimizer[0].lr == 1e-4


def test_each_checkpoint_strategy_type_takes_its_own_options():
    checkpoint = sft(
        val_steps_freq=10,
        checkpoint={
            "save_strategy": {"type": "epoch"},
            "rotation_strategy": {"type": "best", "keep_n": 2},
            "init_strategy": {"type": "resume", "path": "runs/checkpoints/final"},
        },
    ).checkpoint
    assert isinstance(checkpoint.save_strategy, EpochSave) and checkpoint.save_strategy.frequency == 1
    assert isinstance(checkpoint.rotation_strategy, BestRotation)
    assert (checkpoint.rotation_strategy.metric, checkpoint.rotation_strategy.mode) == ("val/loss", "min")
    assert sft(checkpoint={"save_strategy": StepSave(frequency=5)}).checkpoint.save_strategy.frequency == 5


def test_best_rotation_needs_validation(expect_error):
    with expect_error(ValidationError, "best ranks checkpoints by validation, so set val_steps_freq"):
        sft(checkpoint={"rotation_strategy": {"type": "best"}})


def test_unset_optimizer_fields_keep_ttmls_defaults_and_modules_stay_out():
    assert sft(optimizer={"lr": 1e-4, "epsilon": 1e-6, "modules": ["gamma"]}).optimizer[0].to_ttml() == {
        "type": "AdamW",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "epsilon": 1e-6,
    }


def test_the_run_config_logs_the_optimizers_own_fields():
    assert sft(optimizer={"lr": 1e-4, "weight_decay": 0.01}).model_dump(mode="json")["optimizer"] == [
        {
            "lr": 1e-4,
            "modules": None,
            "type": "adamw",
            "weight_decay": 0.01,
            "beta1": None,
            "beta2": None,
            "epsilon": None,
            "amsgrad": None,
            "stochastic_rounding": None,
        }
    ]


def test_several_optimizers_each_name_their_weights():
    config = sft(
        optimizer=[
            {"lr": 1e-3, "modules": ["gamma"]},
            {"lr": 1e-4, "weight_decay": 0.01},
        ]
    )
    assert [(optimizer.lr, optimizer.modules) for optimizer in config.optimizer] == [(1e-3, ["gamma"]), (1e-4, None)]


def test_only_one_optimizer_may_take_the_remaining_weights(expect_error):
    with expect_error(ValidationError, "at most one optimizer may leave modules unset"):
        sft(optimizer=[{"lr": 1e-3}, {"lr": 1e-4}])


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
        ("optimizer.lr=1.0e-5", {"optimizer": {"lr": 1e-5}}),
        ("lora.target_modules=[q_proj, k_proj]", {"lora": {"target_modules": ["q_proj", "k_proj"]}}),
        ("checkpoint.save_strategy={type: none}", {"checkpoint": {"save_strategy": {"type": "none"}}}),
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
        ("batch_size.x=1", "'batch_size' is not a section"),
    ],
)
def test_apply_override_rejects_malformed_overrides(override, error, expect_error):
    with expect_error(ValueError, error):
        apply_override({"batch_size": 1}, override)


def test_load_config_applies_overrides(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("dataset_id: sst2\nmodel_name: m\noptimizer:\n  lr: 1.0e-4\nbatch_size: 8\nmax_length: 64\n")
    config = load_config(
        recipe, ["batch_size=2", "optimizer.lr=1.0e-5", "lora.r=4", "lora.alpha=8", "lora.target_modules=[o_proj]"]
    )
    assert config.batch_size == 2 and config.lora.target_modules == ["o_proj"]
    assert config.optimizer[0].lr == 1e-5


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
