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
    StoppingCriteria,
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


def test_defaults_name_no_lora_and_save_no_checkpoints():
    config = sft()
    assert config.model.lora is None
    assert config.checkpoint.save_strategy.type == "none"
    assert config.checkpoint.init_strategy is None
    assert not config.checkpoint.save_optim
    assert config.gradient_accumulation_steps == 1
    # One epoch unless asked for otherwise.
    assert (config.stopping_criteria.type, config.stopping_criteria.limit) == ("epoch", 1)
    assert (config.metrics.train.steps_freq, config.metrics.train.metric_names) == (10, ["loss"])
    # No validation unless asked for.
    assert (config.metrics.validation.steps_freq, config.metrics.validation.metric_names) == (0, ["loss"])
    assert not config.logging.use_wandb
    assert len(config.optimizer) == 1 and config.optimizer[0].modules is None
    assert isinstance(config.optimizer[0], AdamWConfig) and config.optimizer[0].weight_decay == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"epochs": 2},
        {"master_dtype": "float32"},
        {"gradient_checkpointing": True},
        {"val_steps_freq": 10},
        {"num_epochs": 2},
        {"max_steps": 100},
        {"stopping_criteria": {"type": "step", "limit": 100, "patience": 3}},
        {"dataset": {"id": "sst2", "split": "train"}},
        {"dataset": {"id": "custom", "custom_options": {"train_dataset_path": "train.jsonl", "delimiter": ","}}},
        {"model": {"name": "m", "revision": "main"}},
        {"device": {"mesh": [1, 1], "device_ids": [0]}},
        {"model": {"name": "m", "init_strategy": {"type": "init", "path": "runs"}}},
        {"checkpoint": {"init_strategy": {"type": "init", "path": "runs", "strict": True}}},
        {"model": {"name": "m", "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"], "rank": 4}}},
        {"optimizer": {"lr": 1e-4, "momentum": 0.9}},
        {"checkpoint": {"save_every": 10}},
        {"checkpoint": {"save_strategy": {"type": "none", "frequency": 10}}},
        {"checkpoint": {"rotation_strategy": {"type": "last", "metric": "val/loss"}}},
        {"metrics": {"steps_freq": 10}},
        {"metrics": {"train": {"steps_freq": 10, "metrics": ["loss"]}}},
        {"metrics": {"validation": {"steps_freq": 10, "split": "validation"}}},
        {"logging": {"wandb_watch_mode": "all"}},
    ],
    ids=[
        "top level",
        "master weights",
        "a model field at the top level",
        "a metrics field at the top level",
        "epochs at the top level",
        "steps at the top level",
        "stopping criteria",
        "dataset",
        "custom options",
        "model",
        "device",
        "a checkpoint field in model",
        "init strategy",
        "lora",
        "optimizer",
        "checkpoint",
        "save strategy",
        "rotation strategy",
        "metrics",
        "train metrics",
        "validation metrics",
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
        ({"checkpoint": {"init_strategy": {"type": "last", "path": "runs"}}}, "checkpoint.init_strategy.type"),
        ({"checkpoint": {"init_strategy": {"type": "resume"}}}, "checkpoint.init_strategy.path"),
        ({"device": [1, 1]}, "device"),
        ({"device": {"mesh": "1x1"}}, "device.mesh"),
        ({"device": {"mesh": [1]}}, "device.mesh.1"),
        ({"device": {"mesh": [1, 1, 1]}}, "device.mesh"),
        ({"device": {"mesh": [0, 1]}}, "device.mesh.0"),
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
        ({"stopping_criteria": 100}, "stopping_criteria"),
        ({"stopping_criteria": {"limit": 100}}, "stopping_criteria.type"),
        ({"stopping_criteria": {"type": "step"}}, "stopping_criteria.limit"),
        ({"stopping_criteria": {"type": "loss", "limit": 1}}, "stopping_criteria.type"),
        ({"stopping_criteria": {"type": "epoch", "limit": 0}}, "stopping_criteria.limit"),
        ({"stopping_criteria": {"type": "step", "limit": 1.5}}, "stopping_criteria.limit"),
        ({"metrics": {"train": {"steps_freq": 0}}}, "metrics.train.steps_freq"),
        ({"metrics": {"validation": {"steps_freq": -1}}}, "metrics.validation.steps_freq"),
        ({"metrics": {"validation": {"metric_names": "loss"}}}, "metrics.validation.metric_names"),
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
        metrics={"validation": {"steps_freq": 10}},
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
    with expect_error(ValidationError, "best ranks checkpoints by validation, so set metrics.validation.steps_freq"):
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


def test_the_model_section_names_the_model_and_how_it_is_built():
    assert sft().model_dump(mode="json")["model"] == {
        "name": "meta-llama/Llama-3.2-1B",
        "lora": None,
        "gradient_checkpointing": False,
    }
    lora = {"r": 4, "alpha": 8, "target_modules": ["o_proj"]}
    model = {"name": "m", "lora": lora, "gradient_checkpointing": True}
    assert sft(model=model).model_dump(mode="json")["model"] == {
        "name": "m",
        "lora": {**lora, "dropout": 0.0, "task_type": "CAUSAL_LM"},
        "gradient_checkpointing": True,
    }


def test_the_checkpoint_section_names_the_checkpoint_a_run_starts_from():
    init = {"type": "resume", "path": "runs/checkpoints/final"}
    config = sft(checkpoint={"init_strategy": init})
    assert config.model_dump(mode="json")["checkpoint"]["init_strategy"] == init


@pytest.mark.parametrize("criteria", [{"type": "epoch", "limit": 3}, {"type": "step", "limit": 500}])
def test_the_stopping_criteria_limit_epochs_or_steps(criteria):
    assert sft().model_dump(mode="json")["stopping_criteria"] == {"type": "epoch", "limit": 1}
    assert sft(stopping_criteria=criteria).model_dump(mode="json")["stopping_criteria"] == criteria
    assert sft(stopping_criteria=StoppingCriteria(**criteria)).stopping_criteria.limit == criteria["limit"]


def test_the_metrics_section_has_the_same_fields_for_training_and_validation():
    assert sft().model_dump(mode="json")["metrics"] == {
        "train": {"steps_freq": 10, "metric_names": ["loss"]},
        "validation": {"steps_freq": 0, "metric_names": ["loss"]},
    }
    metrics = {
        "train": {"steps_freq": 5, "metric_names": ["loss", "learning_rate"]},
        "validation": {"steps_freq": 50, "metric_names": []},
    }
    assert sft(metrics=metrics).model_dump(mode="json")["metrics"] == metrics
    # Setting one of the two keeps the other's defaults.
    config = sft(metrics={"validation": {"steps_freq": 50}})
    assert (config.metrics.train.steps_freq, config.metrics.validation.steps_freq) == (10, 50)


def test_the_device_section_names_the_mesh_a_single_chip_unless_set():
    assert sft().device.mesh == (1, 1)
    assert sft().model_dump(mode="json")["device"] == {"mesh": [1, 1]}
    # Any mesh is a valid config; the model provider decides which ones it trains on.
    assert sft(device={"mesh": [1, 2]}).device.mesh == (1, 2)


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
        ("stopping_criteria.limit=20", {"stopping_criteria": {"limit": 20}}),
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
            "checkpoint.init_strategy={type: init, path: runs/final}",
        ],
    )
    assert config.batch_size == 2 and config.model.lora.target_modules == ["o_proj"]
    assert config.optimizer[0].lr == 1e-5
    # An override into a section keeps the section's other fields.
    assert config.model.name == "m"
    assert (config.checkpoint.init_strategy.path, config.checkpoint.save_strategy.type) == ("runs/final", "none")


def test_load_config_rejects_a_non_mapping(tmp_path, expect_error):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("- a\n- b\n")
    with expect_error(ValueError, "expected a mapping"):
        load_config(recipe)


@pytest.mark.parametrize("recipe", sorted(RECIPES.glob("*.yaml")), ids=lambda path: path.stem)
def test_every_recipe_validates_and_trains_lora_with_adamw(recipe):
    config = load_config(recipe)
    assert config.max_length % 32 == 0
    # What a run can train so far.
    assert config.model.lora is not None
    assert all(isinstance(optimizer, AdamWConfig) for optimizer in config.optimizer)


def test_there_are_recipes():
    assert len(list(RECIPES.glob("*.yaml"))) >= 3
