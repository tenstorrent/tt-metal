import os
from typing import Any

import torch

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


def setup_task(state_dict: dict[str, Any]):
    return HubertPretrainingTask(HubertPretrainingConfig(), state_dict)


def load_model_ensemble_and_task(
    filename,
    strict=True,
    suffix="",
):
    cfg = None
    orig_filename = filename
    filename = get_maybe_sharded_checkpoint_filename(orig_filename, suffix)

    if not os.path.exists(filename):
        raise IOError("Model file not found: {}".format(filename))
    with open(filename, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"), weights_only=False)

    if state.get("cfg", None) is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(f"No cfg exist in state keys = {state.keys()}")

    task = setup_task(state["task_state"])

    model = HubertModel.build_model(cfg["model"], task)

    model.load_state_dict(state["model"], strict=strict)
    return model


def load_hubert(config, hubert_path: str):
    hubert_model = load_model_ensemble_and_task(
        hubert_path,
        suffix="",
    )
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    return hubert_model.eval()
