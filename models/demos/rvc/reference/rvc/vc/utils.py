import json
import os

from safetensors.torch import load_file

from .hubert import (
    HubertModel,
    HubertPretrainingConfig,
    HubertPretrainingTask,
)


def get_maybe_sharded_checkpoint_filename(filename: str, suffix: str) -> str:
    filename = filename.replace(".pt", suffix + ".pt")
    fsdp_filename = filename[:-3] + "-shard0.pt"
    if os.path.exists(fsdp_filename):
        return fsdp_filename
    else:
        return filename


def setup_task():
    return HubertPretrainingTask(HubertPretrainingConfig())


def load_model_ensemble_and_task(
    model_path,
    model_cfg_path,
    suffix="",
):
    cfg = None
    task = setup_task()

    with open(model_cfg_path) as f:
        cfg = json.load(f)
    model = HubertModel.build_model(cfg["model"], task)
    hubert_state_safetensors = load_file(model_path)

    model.load_state_dict(hubert_state_safetensors, strict=True)
    return model


def load_hubert(config, hubert_path: str, hubert_cfg_path: str):
    hubert_model = load_model_ensemble_and_task(
        hubert_path,
        hubert_cfg_path,
        suffix="",
    )
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.float()
    return hubert_model.eval()
