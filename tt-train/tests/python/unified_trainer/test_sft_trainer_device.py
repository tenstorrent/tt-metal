# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SFTTrainer and the Llama provider on a device, on a tiny random HF Llama saved locally (no hub access):

- the HF export of a loaded model reproduces its checkpoint bit for bit;
- an exported LoRA adapter makes the same logits in HF PEFT as in ttml;
- LoRA and full fine-tuning runs learn, save checkpoints HF loads, rotate them, and load them back;
- a run resumed from a checkpoint trains as the run it continues would have;
- one trainer runs a LoRA job and then a full one.
"""

import gc
import json

import numpy as np
import pytest

import ttnn
import ttml
from ttml.unified_trainer import (
    CheckpointCallback,
    MetricsCallback,
    SFTTrainer,
    SFTTrainerConfig,
    UnsupportedTrainingConfig,
)
from ttml.unified_trainer.checkpoint_io import (
    is_adapter_dir,
    is_model_dir,
    read_adapter,
    read_trainer_state,
    write_adapter,
)
from ttml.unified_trainer.managers.device import DeviceManager
from ttml.unified_trainer.providers import LlamaProvider, load_hf_config

safetensors_numpy = pytest.importorskip("safetensors.numpy")

N_LAYERS, N_HEADS, N_KV_HEADS, HIDDEN, INTERMEDIATE = 2, 4, 2, 128, 128
# Covers CharTokenizer's ids.
VOCAB = 256
# The runs' max_length: one example of the dataset below is 230 tokens.
MAX_LENGTH = 256

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
EVERY_PROJECTION = ATTENTION + ["gate_proj", "up_proj", "down_proj"]
PEFT_MODULES = [f"self_attn.{p}" for p in ATTENTION] + [f"mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]

pytestmark = pytest.mark.requires_device


class CharTokenizer:
    """One token per character, a BOS first, and "</s>" as EOS."""

    eos_token = "</s>"
    pad_token_id = 0

    def __call__(self, text, truncation=False, padding=False):
        ids = [1]
        for i, part in enumerate(text.split(self.eos_token)):
            if i:
                ids.append(2)
            ids += [3 + ord(c) % 200 for c in part]
        return {"input_ids": ids}


def save_hf_llama(directory, tied=False):
    """A small random ``LlamaForCausalLM`` in bf16, as real checkpoints ship."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        vocab_size=VOCAB,
        max_position_embeddings=MAX_LENGTH,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.copy_(torch.rand_like(param) + 0.5)
            else:
                # HF's 0.02 init leaves attention near uniform, where a wrong RoPE row order would pass.
                param.copy_(torch.randn_like(param) * HIDDEN**-0.5)
    model.to(torch.bfloat16).save_pretrained(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    return save_hf_llama(tmp_path_factory.mktemp("tiny_llama"))


@pytest.fixture
def no_open_device():
    """The trainer opens its own single-chip mesh; another module's device must not be open."""
    ttml.close_device_mesh()
    yield
    gc.collect()
    ttml.close_device_mesh()


@pytest.fixture
def char_tokenizer(monkeypatch):
    monkeypatch.setattr("ttml.unified_trainer.sft_trainer.load_tokenizer", lambda model_name: CharTokenizer())


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def sft_config(model_dir, tmp_path, checkpoint=None, rows=None, lora=None, init_strategy=None, **overrides):
    rows = rows or [{"instruction": "Say the word.", "output": "tenstorrent"}] * 16
    fields = {
        "dataset": {
            "id": "custom",
            "custom_options": {
                "file_type": "jsonl",
                "train_dataset_path": write_jsonl(tmp_path / "train.jsonl", rows),
                "val_dataset_path": write_jsonl(tmp_path / "val.jsonl", rows[:4]),
            },
        },
        "model": {"name": str(model_dir), "lora": lora, "init_strategy": init_strategy},
        "max_length": MAX_LENGTH,
        "optimizer": {"lr": 1e-2},
        "batch_size": 2,
        "max_steps": 8,
        "val_steps_freq": 4,
        "metrics": {"steps_freq": 1},
        "checkpoint": {
            "save_strategy": {"type": "step", "frequency": 4},
            "project_dir": str(tmp_path / "run"),
            **(checkpoint or {}),
        },
    }
    return SFTTrainerConfig.model_validate({**fields, **overrides})


def lora(targets=EVERY_PROJECTION):
    return {"r": 4, "alpha": 8, "target_modules": targets}


class Losses:
    def __init__(self):
        self.train, self.val = [], []

    def on_optimizer_step_end(self, trainer, window_loss):
        self.train.append(trainer.loss_to_float(window_loss))

    def on_validation_end(self, trainer, val_loss):
        self.val.append(trainer.loss_to_float(val_loss))


def ttml_logits(model, ids):
    tokens = ttml.autograd.Tensor.from_numpy(
        ids.reshape(1, 1, 1, -1).astype(np.uint32), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
    )
    values = model(tokens, None).to_numpy(ttnn.DataType.FLOAT32)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return values.reshape(ids.size, -1)[:, :VOCAB]


def trained_logits(trainer, ids):
    trainer.model.eval()
    return ttml_logits(trainer.model, ids)


def hf_logits(model, ids):
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        return model(torch.from_numpy(ids)).logits[0].float().numpy()


def load_hf(directory):
    transformers = pytest.importorskip("transformers")
    return transformers.AutoModelForCausalLM.from_pretrained(str(directory)).float().eval()


def pcc(a, b):
    return np.corrcoef(a.ravel(), b.ravel())[0, 1]


def assert_close(actual, expected, what, min_pcc=0.999, max_rel=0.05):
    rel = np.abs(actual - expected).max() / np.abs(expected).max()
    assert pcc(actual, expected) > min_pcc and rel < max_rel, f"{what}: pcc={pcc(actual, expected)}, rel={rel}"


def sample_ids(length=MAX_LENGTH, seed=7):
    """Token ids as long as the model was built for (its RoPE tables)."""
    return np.random.default_rng(seed).integers(3, VOCAB, (1, length))


def assert_same_bits(a_file, b_file):
    a, b = safetensors_numpy.load_file(str(a_file)), safetensors_numpy.load_file(str(b_file))
    assert sorted(a) == sorted(b)
    for name, value in a.items():
        np.testing.assert_array_equal(b[name].view(np.uint16), value.view(np.uint16), err_msg=name)


def final_checkpoint(tmp_path, project="run"):
    return tmp_path / project / "checkpoints" / "final"


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_the_hf_export_reproduces_the_checkpoint(tmp_path, tied, no_open_device):
    from ttml.models.llama import Llama, LlamaConfig, export_hf_model, load_from_safetensors

    source = save_hf_llama(tmp_path / "source", tied)
    device = DeviceManager()
    try:
        model = Llama(LlamaConfig.from_hf(str(source), max_position_embeddings=MAX_LENGTH))
        load_from_safetensors(model, str(source))
        export_hf_model(model, tmp_path / "export", hf_source_dir=source)
        del model
    finally:
        gc.collect()
        device.close()

    # bf16 in, bf16 out: the round trip is exact, and a tied embedding is written once, as HF does.
    assert_same_bits(source / "model.safetensors", tmp_path / "export" / "model.safetensors")
    assert json.loads((tmp_path / "export" / "config.json").read_text()) == json.loads(
        (source / "config.json").read_text()
    )


def test_an_exported_adapter_makes_the_same_logits_in_peft(tiny_llama, tmp_path, no_open_device):
    peft = pytest.importorskip("peft")
    config = sft_config(tiny_llama, tmp_path, lora=lora())
    provider = LlamaProvider()
    ids = sample_ids()
    device = DeviceManager()
    try:
        model = provider.build(config, load_hf_config(config.model.name))
        model.eval()
        base = ttml_logits(model, ids)
        rng = np.random.default_rng(0)
        for name, tensor in model.parameters().items():
            if name.rsplit("/", 1)[-1] in ("lora_A", "lora_B"):
                values = rng.standard_normal(tensor.shape()).astype(np.float32) * 0.1
                ttml.autograd.assign_numpy(tensor, values)
        adapted = ttml_logits(model, ids)
        (tmp_path / "adapter").mkdir()
        write_adapter(tmp_path / "adapter", provider.export_adapter(model, config))
        del model
    finally:
        gc.collect()
        device.close()

    hf_model = load_hf(tiny_llama)
    peft_model = peft.PeftModel.from_pretrained(hf_model, str(tmp_path / "adapter")).eval()
    hf_adapted = hf_logits(peft_model, ids)
    with peft_model.disable_adapter():
        hf_base = hf_logits(peft_model, ids)

    assert_close(base, hf_base, "base logits")
    assert_close(adapted, hf_adapted, "adapted logits")
    # The adapter changes the logits a lot, and the same way in both: a wrong q/k row order would not.
    assert pcc(hf_adapted, hf_base) < 0.98
    assert pcc(adapted - base, hf_adapted - hf_base) > 0.99


def test_a_lora_run_learns_and_saves_rotated_peft_adapters(tiny_llama, tmp_path, no_open_device, char_tokenizer):
    config = sft_config(
        tiny_llama, tmp_path, checkpoint={"rotation_strategy": {"type": "last", "keep_n": 2}}, lora=lora()
    )
    losses = Losses()
    trainer = SFTTrainer(callbacks=[losses, MetricsCallback(), CheckpointCallback()])
    try:
        trainer.setup(config)
        trainer.train()
        total, trainable = trainer.parameter_counts()
        exported = trainer.provider.export_adapter(trainer.model, trainer.config)
    finally:
        trainer.cleanup()

    assert 0 < trainable < total
    assert len(losses.train) == 8 and np.isfinite(losses.train).all()
    assert losses.train[-1] < losses.train[0]
    assert len(losses.val) == 3 and losses.val[-1] < losses.val[0]

    final = final_checkpoint(tmp_path)
    assert is_adapter_dir(final) and not is_model_dir(final)
    state = read_adapter(final)
    assert set(state.tensors) == {
        f"base_model.model.model.layers.{i}.{module}.lora_{ab}.weight"
        for i in range(N_LAYERS)
        for module in PEFT_MODULES
        for ab in "AB"
    }
    for name, value in exported.tensors.items():
        np.testing.assert_array_equal(state.tensors[name], value)
    assert any(np.abs(value).max() > 0 for name, value in state.tensors.items() if "lora_B" in name)
    assert state.config["target_modules"] == sorted(EVERY_PROJECTION)
    assert read_trainer_state(final)["step"] == 8

    # Saved at steps 4 and 8 and at the end; the last two are kept.
    history = json.loads((final.parent / "checkpoint_history.json").read_text())
    assert [entry["step"] for entry in history["checkpoints"]] == [8, 8]
    kept = sorted(p.name for p in final.parent.iterdir() if p.is_dir())
    assert len(kept) == 2 and kept[0].startswith("checkpoint_step8_") and kept[1] == "final"


def test_init_restores_the_adapters(tiny_llama, tmp_path, no_open_device, char_tokenizer):
    ids = sample_ids()
    first = SFTTrainer(callbacks=[CheckpointCallback()])
    try:
        first.setup(sft_config(tiny_llama, tmp_path, lora=lora(ATTENTION), max_steps=2, val_steps_freq=0))
        first.train()
        trained = trained_logits(first, ids)
    finally:
        first.cleanup()

    # The next run starts from the final checkpoint of the first; one step at a learning rate of 0 keeps it. It
    # has no CheckpointCallback: the trainer loads the checkpoint, and nothing overwrites it.
    second = SFTTrainer()
    try:
        second.setup(
            sft_config(
                tiny_llama,
                tmp_path,
                init_strategy={"type": "init", "path": str(final_checkpoint(tmp_path))},
                lora=lora(ATTENTION),
                optimizer={"lr": 0.0},
                max_steps=1,
                val_steps_freq=0,
            )
        )
        second.train()
        reloaded = second.provider.export_adapter(second.model, second.config)
        restored = trained_logits(second, ids)
    finally:
        second.cleanup()

    saved = read_adapter(final_checkpoint(tmp_path))
    assert set(reloaded.tensors) == set(saved.tensors)
    for name, value in saved.tensors.items():
        np.testing.assert_array_equal(reloaded.tensors[name], value, err_msg=name)
    assert_close(restored, trained, "restored logits", min_pcc=0.9999, max_rel=0.01)


def test_a_full_fine_tune_saves_an_hf_model_directory(tiny_llama, tmp_path, no_open_device, char_tokenizer):
    config = sft_config(
        tiny_llama, tmp_path, optimizer={"lr": 1e-3}, max_steps=4, max_grad_norm=1.0, master_dtype="float32"
    )
    losses = Losses()
    ids = sample_ids()
    trainer = SFTTrainer(callbacks=[losses, CheckpointCallback()])
    try:
        trainer.setup(config)
        trainer.train()
        trained = trained_logits(trainer, ids)
    finally:
        trainer.cleanup()
    assert np.isfinite(losses.train).all() and losses.train[-1] < losses.train[0]

    final = final_checkpoint(tmp_path)
    assert is_model_dir(final) and not is_adapter_dir(final)
    assert read_trainer_state(final)["step"] == 4
    assert sorted(safetensors_numpy.load_file(str(final / "model.safetensors"))) == sorted(
        safetensors_numpy.load_file(str(tiny_llama / "model.safetensors"))
    )
    assert_close(trained, hf_logits(load_hf(final), ids), "trained logits in HF")

    second = SFTTrainer()
    try:
        second.setup(sft_config(tiny_llama, tmp_path, val_steps_freq=0, master_dtype="float32"))
        second.load_checkpoint(str(final))
        second.provider.export_weights(second.model, second.config, tmp_path / "reloaded")
        reloaded = trained_logits(second, ids)
    finally:
        second.cleanup()
    assert_same_bits(final / "model.safetensors", tmp_path / "reloaded" / "model.safetensors")
    assert_close(reloaded, trained, "reloaded logits", min_pcc=0.9999, max_rel=0.01)


@pytest.mark.parametrize(
    "overrides, weights",
    [
        ({"lora": {**lora(ATTENTION), "dropout": 0.2}}, "adapter_model.safetensors"),
        ({"optimizer": {"lr": 1e-3}, "master_dtype": "float32"}, "model.safetensors"),
    ],
    ids=["lora with dropout", "full in float32"],
)
def test_a_resumed_run_trains_as_the_run_it_continues(
    overrides, weights, tiny_llama, tmp_path, no_open_device, char_tokenizer
):
    # Distinct rows, so training on other batches than the uninterrupted run would show in the losses.
    rows = [{"instruction": f"Say word {i}.", "output": "tenstorrent"[: 3 + i % 9]} for i in range(16)]

    def train(project, checkpoint, init_strategy=None):
        losses = Losses()
        trainer = SFTTrainer(callbacks=[losses, CheckpointCallback()])
        checkpoint = {"project_dir": str(tmp_path / project), "save_optim": True, **checkpoint}
        try:
            trainer.setup(
                sft_config(
                    tiny_llama,
                    tmp_path,
                    checkpoint=checkpoint,
                    init_strategy=init_strategy,
                    rows=rows,
                    num_epochs=2,
                    gradient_accumulation_steps=2,
                    val_steps_freq=0,
                    **overrides,
                )
            )
            trainer.train()
        finally:
            trainer.cleanup()
        return losses.train

    # 8 batches of 2 rows, 2 batches a step: 4 steps an epoch, 8 in all.
    uninterrupted = train("run", {"save_strategy": {"type": "step", "frequency": 3}})
    (step_3,) = (tmp_path / "run" / "checkpoints").glob("checkpoint_step3_*")
    # From the middle of the first epoch, through the second.
    resumed = train("resumed", {}, init_strategy={"type": "resume", "path": str(step_3)})

    assert len(uninterrupted) == 8 and resumed == uninterrupted[3:]
    assert_same_bits(final_checkpoint(tmp_path) / weights, final_checkpoint(tmp_path, "resumed") / weights)


def test_one_trainer_runs_a_lora_job_and_then_a_full_one(tiny_llama, tmp_path, no_open_device, char_tokenizer):
    losses = Losses()
    trainer = SFTTrainer(callbacks=[losses, CheckpointCallback()])
    jobs = [
        sft_config(tiny_llama, tmp_path, checkpoint={"project_dir": str(tmp_path / "lora")}, lora=lora()),
        sft_config(
            tiny_llama,
            tmp_path,
            checkpoint={"project_dir": str(tmp_path / "full")},
            optimizer={"lr": 1e-3},
            max_steps=4,
        ),
    ]
    for job in jobs:
        try:
            trainer.setup(job)
            trainer.train()
        finally:
            trainer.cleanup()

    assert len(losses.train) == 12 and np.isfinite(losses.train).all()
    assert is_adapter_dir(final_checkpoint(tmp_path, "lora"))
    assert is_model_dir(final_checkpoint(tmp_path, "full"))
    assert trainer.config is None and not hasattr(trainer, "model")


def test_a_partly_targeted_fused_layer_fails_before_the_device_opens(
    tiny_llama, tmp_path, char_tokenizer, expect_error
):
    trainer = SFTTrainer()
    try:
        with expect_error(UnsupportedTrainingConfig, "one fused layer"):
            trainer.setup(sft_config(tiny_llama, tmp_path, lora=lora(["q_proj", "v_proj"])))
        assert not hasattr(trainer, "device_manager")
    finally:
        trainer.cleanup()
