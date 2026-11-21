# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The Llama 3.2 1B Instruct SST-2 LoRA recipe, run as ``python -m ttml.experimental.trainer`` runs it, for two steps:
it trains, validates, logs the metrics of both, and saves PEFT adapter checkpoints that rotate and load back.

Downloads ``meta-llama/Llama-3.2-1B-Instruct``, a gated repo (``HF_TOKEN``), and SST-2 unless they are cached.
"""

import gc
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ttml
import ttml.experimental
from ttml.experimental.models.utils.checkpoint import ADAPTER_CONFIG, ADAPTER_WEIGHTS, read_adapter
from ttml.experimental.models.utils.hub import load_hf_config
from ttml.experimental.trainer.__main__ import main
from ttml.experimental.trainer.configs import load_config
from ttml.experimental.trainer.managers.checkpoint import HISTORY_FILE
from ttml.experimental.trainer.managers.logging import TrainingLogger
from ttml.experimental.trainer.utils import TRAINER_STATE

RECIPE = Path(ttml.experimental.__file__).parent / "training_configs" / "llama_3_2_1b_instruct_sst2_lora.yaml"
STEPS = 2

pytestmark = [pytest.mark.requires_device, pytest.mark.slow]


def run_recipe(overrides: list[str]) -> SimpleNamespace:
    """Run the recipe with ``--set`` overrides; return the ``(metrics, step)`` it logged and the messages it logged."""
    logged = SimpleNamespace(metrics=[], messages=[])
    log_metrics, info = TrainingLogger.log_metrics, TrainingLogger.info

    def record_metrics(self, metrics, step=None, commit=True):
        if metrics:
            logged.metrics.append((dict(metrics), step))
        log_metrics(self, metrics, step=step, commit=commit)

    def record_message(self, message):
        logged.messages.append(message)
        info(self, message)

    # The trainer opens its own single-chip mesh; another module's device must not be open.
    ttml.close_device_mesh()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(TrainingLogger, "log_metrics", record_metrics)
            patch.setattr(TrainingLogger, "info", record_message)
            main(["--config", str(RECIPE), *(arg for override in overrides for arg in ("--set", override))])
    finally:
        gc.collect()
        ttml.close_device_mesh()
    return logged


@pytest.fixture(scope="module")
def recipe(tmp_path_factory):
    """The recipe's run, cut to two steps that each log, validate and save a checkpoint; keeping only the last."""
    project = tmp_path_factory.mktemp("sst2_lora")
    overrides = [
        f"stopping_criteria.limit={STEPS}",
        "metrics.train.frequency=1",
        "metrics.validation.frequency=1",
        f"checkpoint.save_strategy.project_dir={project}",
        "checkpoint.save_strategy.frequency=1",
        "checkpoint.rotation_strategy.type=last",
        "checkpoint.rotation_strategy.keep_n=1",
    ]
    logged = run_recipe(overrides)
    config = load_config(RECIPE, overrides)
    return SimpleNamespace(config=config, checkpoints=project / "checkpoints", logged=logged)


def logged_values(logged, name: str) -> dict[int, float]:
    """The values of the metric ``name`` the run logged, by step."""
    return {step: metrics[name] for metrics, step in logged.metrics if name in metrics}


def test_logs_the_train_and_validation_losses(recipe):
    train, val = logged_values(recipe.logged, "train/loss"), logged_values(recipe.logged, "val/loss")
    assert sorted(train) == [1, 2], "train/loss is logged every step"
    # Before the first step too.
    assert sorted(val) == [0, 1, 2], "val/loss is logged before training and every step"
    for name, value in [*train.items(), *val.items()]:
        assert math.isfinite(value) and value > 0, f"{name}: {value}"
    assert sorted(logged_values(recipe.logged, "train/epoch")) == [1, 2]


def test_saves_every_step_and_keeps_the_last_checkpoint(recipe):
    """Step 1's checkpoint is saved, then rotated away once the final one is saved at step 2."""
    saved = [message for message in recipe.logged.messages if message.startswith("Saved checkpoint: ")]
    removed = [message for message in recipe.logged.messages if message.startswith("Removed old checkpoint: ")]
    step_1 = saved[0].removeprefix("Saved checkpoint: ")
    assert len(saved) == 2 and Path(step_1).name.startswith("checkpoint_step1_epoch0_"), saved
    assert saved[1] == f"Saved checkpoint: {recipe.checkpoints / 'final'}"
    assert removed == [f"Removed old checkpoint: {step_1}"]
    # Nothing left of a save: no staging directory, nor the one a save replaced.
    assert sorted(path.name for path in recipe.checkpoints.iterdir()) == [HISTORY_FILE, "final"]


def test_a_checkpoint_carries_the_metrics_of_its_step(recipe):
    final = recipe.checkpoints / "final"
    metrics = {"train/loss": logged_values(recipe.logged, "train/loss")[STEPS]}
    metrics["val/loss"] = logged_values(recipe.logged, "val/loss")[STEPS]

    state = json.loads((final / TRAINER_STATE).read_text())
    assert (state["step"], state["epoch"], state["metrics"]) == (STEPS, 0, metrics)
    assert state["format"] == {"type": "hf"}
    assert state["base_model"] == {"name_or_path": recipe.config.model.name}
    (entry,) = json.loads((recipe.checkpoints / HISTORY_FILE).read_text())["checkpoints"]
    assert (entry["name"], entry["path"], entry["step"], entry["metrics"]) == ("final", str(final), STEPS, metrics)


def test_the_final_checkpoint_is_a_peft_adapter(recipe):
    final = recipe.checkpoints / "final"
    lora = recipe.config.model.lora
    hf_config = load_hf_config(recipe.config.model.name)
    hidden, heads, kv_heads = (hf_config[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads"))
    head_dim = hf_config.get("head_dim") or hidden // heads
    rows = {"q_proj": heads * head_dim, "k_proj": kv_heads * head_dim, "v_proj": kv_heads * head_dim, "o_proj": hidden}
    assert sorted(rows) == sorted(lora.target_modules), "premise: the recipe adapts q, k, v and o"

    assert (final / ADAPTER_CONFIG).is_file() and (final / ADAPTER_WEIGHTS).is_file()
    adapter = read_adapter(final)
    expected_shapes = {}
    for layer in range(hf_config["num_hidden_layers"]):
        for projection, out_features in rows.items():
            module = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
            expected_shapes[f"{module}.lora_A.weight"] = (
                lora.r,
                heads * head_dim if projection == "o_proj" else hidden,
            )
            expected_shapes[f"{module}.lora_B.weight"] = (out_features, lora.r)
    assert {name: value.shape for name, value in adapter.tensors.items()} == expected_shapes
    assert all(np.isfinite(value).all() for value in adapter.tensors.values())
    # lora_B starts at zero, so two steps must have moved it.
    assert all(value.any() for name, value in adapter.tensors.items() if name.endswith("lora_B.weight"))

    config = adapter.config
    assert config["peft_type"] == "LORA" and config["task_type"] == lora.task_type
    assert (config["r"], config["lora_alpha"]) == (lora.r, lora.alpha)
    assert config["target_modules"] == sorted(lora.target_modules)
    assert config["base_model_name_or_path"] == recipe.config.model.name


def test_a_run_initialized_from_the_checkpoint_starts_from_its_adapters(recipe, tmp_path):
    """One step at a learning rate of 0 keeps the weights the run loaded, so its final checkpoint is the one it
    started from. The run saves to the project_dir it started from, and keeps that checkpoint."""
    init = tmp_path / "checkpoints" / "final"
    shutil.copytree(recipe.checkpoints / "final", init)
    logged = run_recipe(
        [
            "stopping_criteria.limit=1",
            "optimizer.lr=0",
            "metrics.validation.frequency=0",
            f"checkpoint.init_strategy.path={init}",
            f"checkpoint.save_strategy.project_dir={tmp_path}",
        ]
    )

    (saved_path,) = [
        m.removeprefix("Saved checkpoint: ") for m in logged.messages if m.startswith("Saved checkpoint: ")
    ]
    assert Path(saved_path).parent == init.parent and Path(saved_path).name.startswith("final_"), saved_path
    started_from, saved = read_adapter(recipe.checkpoints / "final"), read_adapter(saved_path)
    assert sorted(saved.tensors) == sorted(started_from.tensors)
    for name, value in started_from.tensors.items():
        np.testing.assert_array_equal(saved.tensors[name], value, err_msg=name)
    # The saved adapters equal the ones it started from, so the step-2 trainer_state tells the kept checkpoint apart.
    assert (init / TRAINER_STATE).read_text() == (recipe.checkpoints / "final" / TRAINER_STATE).read_text()
