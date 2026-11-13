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

MINIMAL = {
    "dataset": {"id": "sst2"},
    "model": {"name": "meta-llama/Llama-3.2-1B"},
    "optimizer": {"lr": 1e-4},
    "batch_size": 8,
}


def sft(**overrides):
    return SFTTrainerConfig.model_validate({**MINIMAL, "max_length": 64, **overrides})


def test_defaults_train_every_weight_without_checkpoints():
    config = sft()
    assert config.model.lora is None
    assert config.model.init_strategy is None
    assert config.checkpoint.save_strategy.type == "none"
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
        {"gradient_checkpointing": True},
        {"dataset": {"id": "sst2", "split": "train"}},
        {"dataset": {"id": "custom", "custom_options": {"train_dataset_path": "train.jsonl", "delimiter": ","}}},
        {"model": {"name": "m", "revision": "main"}},
        {"model": {"name": "m", "init_strategy": {"type": "init", "path": "runs", "strict": True}}},
        {"model": {"name": "m", "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "rank": 4}}},
        {"optimizer": {"lr": 1e-4, "momentum": 0.9}},
        {"checkpoint": {"save_every": 10}},
        {"checkpoint": {"save_strategy": {"type": "none", "frequency": 10}}},
        {"checkpoint": {"rotation_strategy": {"type": "last", "metric": "val/loss"}}},
        {"metrics": {"log_freq": 10}},
        {"logging": {"wandb_watch_mode": "all"}},
    ],
    ids=[
        "top level",
        "a model field at the top level",
        "dataset",
        "custom options",
        "model",
        "init strategy",
        "lora",
        "optimizer",
        "checkpoint",
        "save strategy",
        "rotation strategy",
        "metrics",
        "logging",
    ],
)
def test_unknown_fields_are_rejected(overrides, expect_error):
    with expect_error(ValidationError, "Extra inputs are not permitted"):
        sft(**overrides)


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"batch_size": 0}, "batch_size"),
        ({"dataset": "sst2"}, "dataset"),
        ({"dataset": {}}, "dataset.id"),
        ({"dataset": {"id": ""}}, "dataset.id"),
        ({"model": "m"}, "model"),
        ({"model": {}}, "model.name"),
        ({"model": {"name": ""}}, "model.name"),
        ({"model": {"name": "m", "gradient_checkpointing": "sometimes"}}, "model.gradient_checkpointing"),
        ({"model": {"name": "m", "init_strategy": {"type": "last", "path": "runs"}}}, "model.init_strategy.type"),
        ({"model": {"name": "m", "init_strategy": {"type": "resume"}}}, "model.init_strategy.path"),
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
        ({"model": {"name": "m", "lora": {"r": 0, "alpha": 8, "target_modules": ["q_proj"]}}}, "model.lora.r"),
        ({"model": {"name": "m", "lora": {"r": 4, "alpha": 8, "target_modules": []}}}, "model.lora.target_modules"),
        (
            {"model": {"name": "m", "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "dropout": 1.0}}},
            "model.lora.dropout",
        ),
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


@pytest.mark.parametrize("dataset_id", ["custom", "Custom"])
def test_a_custom_dataset_needs_its_options(dataset_id, expect_error):
    with expect_error(ValidationError, r"\ndataset\n  Value error, custom_options is required when id is 'custom'"):
        sft(dataset={"id": dataset_id})


def test_the_dataset_section_names_the_dataset_and_the_options_of_a_custom_one():
    assert sft().model_dump(mode="json")["dataset"] == {"id": "sst2", "custom_options": None}
    custom = sft(dataset={"id": "custom", "custom_options": {"train_dataset_path": "train.jsonl"}})
    assert custom.model_dump(mode="json")["dataset"] == {
        "id": "custom",
        "custom_options": {
            "file_type": "json",
            "train_dataset_path": "train.jsonl",
            "val_dataset_path": None,
            "template": "alpaca",
            "column_mapping": None,
        },
    }


def test_the_model_section_names_the_model_how_it_is_built_and_the_checkpoint_it_starts_from():
    assert sft().model_dump(mode="json")["model"] == {
        "name": "meta-llama/Llama-3.2-1B",
        "lora": None,
        "gradient_checkpointing": False,
        "init_strategy": None,
    }
    lora = {"r": 4, "alpha": 8, "target_modules": ["o_proj"]}
    init = {"type": "resume", "path": "runs/checkpoints/final"}
    model = {"name": "m", "lora": lora, "gradient_checkpointing": True, "init_strategy": init}
    assert sft(model=model).model_dump(mode="json")["model"] == {
        "name": "m",
        "lora": {**lora, "dropout": 0.0, "task_type": "CAUSAL_LM"},
        "gradient_checkpointing": True,
        "init_strategy": init,
    }


@pytest.mark.parametrize("section", ["dataset", "model"])
def test_a_config_needs_a_dataset_and_a_model(section, expect_error):
    without_section = {key: value for key, value in MINIMAL.items() if key != section}
    with expect_error(ValidationError, rf"\n{section}\n  Field required"):
        SFTTrainerConfig.model_validate({**without_section, "max_length": 64})


@pytest.mark.parametrize(
    "override, expected",
    [
        ("model.lora.r=16", {"model": {"lora": {"r": 16}}}),
        ("optimizer.lr=1.0e-5", {"optimizer": {"lr": 1e-5}}),
        (
            "model.lora.target_modules=[q_proj, k_proj]",
            {"model": {"lora": {"target_modules": ["q_proj", "k_proj"]}}},
        ),
        ("checkpoint.save_strategy={type: none}", {"checkpoint": {"save_strategy": {"type": "none"}}}),
        (
            "dataset.custom_options.column_mapping={output: answer}",
            {"dataset": {"custom_options": {"column_mapping": {"output": "answer"}}}},
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
        ("model.lora.r", "expected dotted.key=value"),
        ("=4", "expected dotted.key=value"),
        ("batch_size.x=1", "'batch_size' is not a section"),
    ],
)
def test_apply_override_rejects_malformed_overrides(override, error, expect_error):
    with expect_error(ValueError, error):
        apply_override({"batch_size": 1}, override)


def test_load_config_applies_overrides(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "dataset:\n  id: sst2\nmodel:\n  name: m\noptimizer:\n  lr: 1.0e-4\nbatch_size: 8\nmax_length: 64\n"
    )
    config = load_config(
        recipe,
        [
            "batch_size=2",
            "optimizer.lr=1.0e-5",
            "model.lora.r=4",
            "model.lora.alpha=8",
            "model.lora.target_modules=[o_proj]",
            "model.init_strategy={type: init, path: runs/final}",
        ],
    )
    assert config.batch_size == 2 and config.model.lora.target_modules == ["o_proj"]
    assert config.optimizer[0].lr == 1e-5
    # An override into a section keeps the section's other fields.
    assert (config.model.name, config.model.init_strategy.path) == ("m", "runs/final")


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
