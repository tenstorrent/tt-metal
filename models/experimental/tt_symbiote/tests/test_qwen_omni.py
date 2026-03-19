# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3-Omni model with TTNN backend.

Run modes (set ``TT_SYMBIOTE_RUN_MODE`` before pytest):

- ``CPU`` — symbiote reference path; fits 30B + multimodal in host memory.
- ``NORMAL`` or ``NORMAL_WITH_FALLBACK`` — execute TTNN modules on silicon.

For thinker MoE, symbiote expects ``MESH_DEVICE=T3K`` (see ``run_on_devices`` on
``TTNNQwen3OmniThinkerNaiveMoE``).

**Why a mesh (e.g. 1×8) and not the default ``device`` fixture?**  The root ``device``
fixture uses ``CreateDevice`` → **one** Wormhole chip (~1 GiB usable DRAM per bank).
MoE activations/weights plus ``to_device`` buffers can exceed that and hit
``BankManager`` OOM.  These tests use the ``mesh_device`` fixture (same pattern as
``test_glm.py``) so **all available chips** are opened as a mesh when
``MESH_DEVICE=T3K`` (shape ``(1, 8)``), with ``FABRIC_1D_RING``.  That raises total
cluster memory and enables distributed routing; per-chip DRAM can still fragment, so
if OOM persists, use ``TT_SYMBIOTE_RUN_MODE=CPU``, trim inputs, or preload/evict MoE
weights explicitly.

Example (8-chip mesh on T3K):

.. code-block:: bash

   export MESH_DEVICE=T3K
   export TT_SYMBIOTE_RUN_MODE=NORMAL
   pytest models/experimental/tt_symbiote/tests/test_qwen_omni.py::test_qwen_omni -v -s -p no:timeout

Symbiote registration is **thinker SparseMoE only** (``register_qwen_omni_symbiote_modules``); Linear,
attention, talker, audio, and vision stay on stock PyTorch modules.

**Note:** Full-resolution image+audio keeps large vision token counts in **PyTorch** paths;
if those are moved to TTNN later, watch DRAM there too.
"""

import os

import pytest
import soundfile as sf
import torch
import ttnn
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3OmniThinkerNaiveMoE
from models.experimental.tt_symbiote.qwen3omni.hf_generation_compat import apply_qwen3_omni_talker_prepare_inputs_fix
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

# CPU: fast / reference. NORMAL / NORMAL_WITH_FALLBACK: real TTNN on device (see README).
# Thinker MoE uses @run_on_devices(DeviceArch.T3K) — set MESH_DEVICE=T3K when using a T3K-class mesh.
_ALLOWED_SYMBIOTE_RUN_MODES = frozenset({"CPU", "NORMAL", "NORMAL_WITH_FALLBACK"})


def _require_symbiote_run_mode():
    mode = os.environ.get("TT_SYMBIOTE_RUN_MODE")
    assert (
        mode in _ALLOWED_SYMBIOTE_RUN_MODES
    ), f"Set TT_SYMBIOTE_RUN_MODE to one of {_ALLOWED_SYMBIOTE_RUN_MODES}, got {mode!r}"


def _patch_thinker_talker_device_dtype(model):
    """HF generate() reads thinker/talker .device/.dtype; TTNN submodules don't support .to()."""
    _cpu = torch.device("cpu")
    _dtype = torch.bfloat16
    for sub in (getattr(model, "thinker", None), getattr(model, "talker", None)):
        if sub is not None:
            cls = type(sub)
            if not hasattr(cls, "_tt_symbiote_device_patched"):
                if isinstance(getattr(cls, "device", None), property):
                    cls.device = property(lambda self, d=_cpu: d)
                if isinstance(getattr(cls, "dtype", None), property):
                    cls.dtype = property(lambda self, d=_dtype: d)
                cls._tt_symbiote_device_patched = True


def register_qwen_omni_symbiote_modules(model) -> dict:
    """Replace thinker text SparseMoE blocks with ``TTNNQwen3OmniThinkerNaiveMoE`` (naive expert loop, no Linear/attn/talker)."""
    thinker_mlp_class = type(model.thinker.model.layers[0].mlp)
    return register_module_replacement_dict(
        model.thinker,
        {thinker_mlp_class: TTNNQwen3OmniThinkerNaiveMoE},
        model_config=None,
    )


_MESH_SHAPE_BY_ENV = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}


def _mesh_param_for_request():
    return _MESH_SHAPE_BY_ENV.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))


# Open multi-chip mesh (not single-chip ``device``) — see module docstring.
pytestmark = [
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 245760,
                "num_command_queues": 1,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            }
        ],
        indirect=True,
    ),
    pytest.mark.parametrize("mesh_device", [_mesh_param_for_request()], indirect=True),
]


def test_qwen_omni_symbiote_replacements_verified(mesh_device):
    """Load model, apply symbiote replacements, assert thinker MoE layers are TTNN (no generate)."""
    _require_symbiote_run_mode()
    apply_qwen3_omni_talker_prepare_inputs_fix()

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(dtype=torch.bfloat16)

    register_qwen_omni_symbiote_modules(model)
    set_device(model, mesh_device)
    _patch_thinker_talker_device_dtype(model)

    n_thinker = len(model.thinker.model.layers)
    for i, layer in enumerate(model.thinker.model.layers):
        assert isinstance(
            layer.mlp, TTNNQwen3OmniThinkerNaiveMoE
        ), f"thinker.layers[{i}].mlp expected TTNN thinker naive MoE, got {type(layer.mlp)}"

    print(f"Replacements OK: thinker {n_thinker} layers (MoE only; mesh " f"{mesh_device.get_num_devices()} device(s))")


def test_qwen_omni(mesh_device):
    """Test Qwen3-Omni model with TTNN acceleration."""
    _require_symbiote_run_mode()

    # HF: talker prepare_inputs_for_generation vs GenerationMixin next_sequence_length (transformers >= 4.49)
    apply_qwen3_omni_talker_prepare_inputs_fix()

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    print(f"Loading Qwen3-Omni model from {MODEL_PATH}...")
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(dtype=torch.bfloat16)

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
                {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    print("Registering TTNN module replacements...")
    register_qwen_omni_symbiote_modules(model)

    # Set device for all TTNN modules
    print(f"Setting device for TTNN modules (mesh: {mesh_device.get_num_devices()} device(s))...")
    set_device(model, mesh_device)

    _patch_thinker_talker_device_dtype(model)

    # Preprocess and move weights to device
    print("Preprocessing and moving weights to device...")
    # for k, v in tqdm(modules.items(), desc="Processing modules"):
    #     v.preprocess_weights()
    #     v.move_weights_to_device()

    model.eval()  # Disables dropout, batch norm updates
    torch.set_grad_enabled(False)  # Disables autograd overhead

    # Preparation for inference
    print("Preparing inputs...")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    input_device = getattr(model, "device", None) or torch.device("cpu")
    input_dtype = getattr(model, "dtype", None) or torch.bfloat16
    inputs = inputs.to(input_device).to(input_dtype)
    print("Running inference...")
    DispatchManager.clear_timings()

    # Inference: Generation of the output text and audio
    text_ids, audio = model.generate(
        **inputs, speaker="Ethan", thinker_return_dict_in_generate=True, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    DispatchManager.save_stats_to_file("qwen_omni_timing_stats.csv")

    text = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(f"QWEN-OMNI OUTPUT: {text}")

    if audio is not None:
        sf.write(
            "output.wav",
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        print("Audio output saved to output.wav")
