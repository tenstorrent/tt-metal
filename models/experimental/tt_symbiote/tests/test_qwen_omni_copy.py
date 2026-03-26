# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Test for Qwen3-Omni model with TTNN backend.

Functionally the same as ``test_qwen_omni.py`` except:

- **Static ``NN_TO_TTNN_*`` maps** + ``register_module_replacement_dict`` on thinker/talker/code2wav
  (instead of ``register_qwen_omni_symbiote_modules``).
- **Vision** blocks use ``TTNNQwen3VLMoeVisionAttention`` (reference test keeps vision on PyTorch).

Everything else matches the reference: same weights dtype and flow, ``set_device(model, mesh_device)``,
same replacement targets as ``register_qwen_omni_symbiote_modules`` for non-vision paths
(``TTNNQwenAudioAttentionOptimized``, talker MoE + attn, code2wav when present).

**Load:** Some transformers builds omit ``initializer_range`` on the composite ``Qwen3OmniMoeConfig`` while
``_init_weights`` still reads ``model.config.initializer_range``. This file loads config from the hub, patches
the root ``initializer_range`` when missing, then calls ``from_pretrained(..., config=...)`` (same pattern as
earlier symbiote Qwen-Omni tests).

Run modes (set ``TT_SYMBIOTE_RUN_MODE`` before pytest):

- ``CPU`` — symbiote reference path; fits 30B + multimodal in host memory.
- ``NORMAL`` or ``NORMAL_WITH_FALLBACK`` — execute TTNN modules on silicon.

For thinker MoE, symbiote expects ``MESH_DEVICE=T3K`` (see ``run_on_devices`` on
``TTNNQwen3OmniThinkerNaiveMoE``).

Example (8-chip mesh on T3K):

.. code-block:: bash

   export MESH_DEVICE=T3K
   export TT_SYMBIOTE_RUN_MODE=NORMAL
   pytest models/experimental/tt_symbiote/tests/test_qwen_omni_copy.py::test_qwen_omni -v -s -p no:timeout

**nn_to_ttnn** (``NN_TO_TTNN_THINKER`` / ``NN_TO_TTNN_TALKER`` + optional ``model.code2wav``):

- Thinker text MoE → ``TTNNQwen3OmniThinkerNaiveMoE``
- Thinker self-attention → ``TTNNQwen3OmniAttention``
- Vision block attention → ``TTNNQwen3VLMoeVisionAttention`` *(extra vs reference test)*
- Thinker audio encoder self-attention → ``TTNNQwenAudioAttentionOptimized``
- Talker MoE → ``TTNNQwen3TalkerMoE``
- Talker self-attention → ``TTNNQwen3Attention``
- Code2Wav pre-transformer self-attention → ``TTNNQwen3OmniMoeCode2WavAttention`` (when present)

**Note:** Full-resolution image+audio can stress DRAM; use ``TT_SYMBIOTE_RUN_MODE=CPU`` or trim inputs if needed.
"""

import os

import pytest
import soundfile as sf
import torch
import ttnn
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioAttention,
    Qwen3OmniMoeCode2WavAttention,
    Qwen3OmniMoeTalkerTextSparseMoeBlock,
    Qwen3OmniMoeThinkerTextAttention,
    Qwen3OmniMoeThinkerTextSparseMoeBlock,
    Qwen3OmniMoeVisionAttention,
)
from qwen_omni_utils import process_mm_info

from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3OmniThinkerNaiveMoE, TTNNQwen3TalkerMoE
from models.experimental.tt_symbiote.qwen3omni.hf_generation_compat import apply_qwen3_omni_talker_prepare_inputs_fix
from models.experimental.tt_symbiote.qwen3omni.tt.audio_attention import TTNNQwenAudioAttentionOptimized
from models.experimental.tt_symbiote.qwen3omni.tt.code2wav_attn import TTNNQwen3OmniMoeCode2WavAttention
from models.experimental.tt_symbiote.qwen3omni.tt.talker_attention import TTNNQwen3Attention
from models.experimental.tt_symbiote.qwen3omni.tt.thinker_attention import TTNNQwen3OmniAttention
from models.experimental.tt_symbiote.qwen3omni.tt.vision_attn import TTNNQwen3VLMoeVisionAttention
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

# CPU: fast / reference. NORMAL / NORMAL_WITH_FALLBACK: real TTNN on device (see README).
_ALLOWED_SYMBIOTE_RUN_MODES = frozenset({"CPU", "NORMAL", "NORMAL_WITH_FALLBACK"})

# Static HF → TTNN maps (equivalent to ``register_qwen_omni_symbiote_modules`` + vision row).
NN_TO_TTNN_THINKER = {
    Qwen3OmniMoeThinkerTextSparseMoeBlock: TTNNQwen3OmniThinkerNaiveMoE,
    Qwen3OmniMoeThinkerTextAttention: TTNNQwen3OmniAttention,
    Qwen3OmniMoeVisionAttention: TTNNQwen3VLMoeVisionAttention,
    Qwen3OmniMoeAudioAttention: TTNNQwenAudioAttentionOptimized,
}
NN_TO_TTNN_CODE2WAV = {
    Qwen3OmniMoeCode2WavAttention: TTNNQwen3OmniMoeCode2WavAttention,
}
NN_TO_TTNN_TALKER = {
    Qwen3OmniMoeTalkerTextSparseMoeBlock: TTNNQwen3TalkerMoE,
    Qwen3OmniMoeThinkerTextAttention: TTNNQwen3Attention,
}


def patch_qwen3_omni_moe_root_config(config):
    """Set ``config.initializer_range`` on composite Qwen3-Omni-MoE config if missing (HF ``_init_weights``)."""
    if getattr(config, "initializer_range", None) is not None:
        return
    val = None
    thinker = getattr(config, "thinker_config", None)
    if thinker is not None:
        val = getattr(thinker, "initializer_range", None)
        if val is None:
            text_cfg = getattr(thinker, "text_config", None)
            if text_cfg is not None:
                val = getattr(text_cfg, "initializer_range", None)
    config.initializer_range = 0.02 if val is None else val


def load_qwen3_omni_moe_for_conditional_generation_bf16(model_path: str) -> Qwen3OmniMoeForConditionalGeneration:
    omni_config = Qwen3OmniMoeConfig.from_pretrained(model_path)
    patch_qwen3_omni_moe_root_config(omni_config)
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        config=omni_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )


def _register_code2wav_nn_to_ttnn(model) -> dict:
    """``code2wav`` is not under ``model.thinker``; register its pre-transformer attention separately."""
    code2wav = getattr(model, "code2wav", None)
    if code2wav is None or getattr(code2wav, "pre_transformer", None) is None:
        return {}
    return register_module_replacement_dict(code2wav, NN_TO_TTNN_CODE2WAV, model_config=None)


def _require_symbiote_run_mode():
    mode = os.environ.get("TT_SYMBIOTE_RUN_MODE")
    assert (
        mode in _ALLOWED_SYMBIOTE_RUN_MODES
    ), f"Set TT_SYMBIOTE_RUN_MODE to one of {_ALLOWED_SYMBIOTE_RUN_MODES}, got {mode!r}"


def _patch_thinker_talker_device_dtype(model):
    """HF generate() reads thinker/talker/code2wav .device/.dtype; TTNN submodules don't support .to()."""
    _cpu = torch.device("cpu")
    _dtype = torch.bfloat16
    for sub in (
        getattr(model, "thinker", None),
        getattr(model, "talker", None),
        getattr(model, "code2wav", None),
    ):
        if sub is None:
            continue
        cls = type(sub)
        if hasattr(cls, "_tt_symbiote_device_patched"):
            continue
        # HF may call .to(submodule.device); expose stable host placeholders (same as thinker/talker).
        dev_attr = getattr(cls, "device", None)
        if dev_attr is None or isinstance(dev_attr, property):
            cls.device = property(lambda self, d=_cpu: d)
        dtype_attr = getattr(cls, "dtype", None)
        if dtype_attr is None or isinstance(dtype_attr, property):
            cls.dtype = property(lambda self, d=_dtype: d)
        cls._tt_symbiote_device_patched = True


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
    """Load model, apply ``nn_to_ttnn``; assert TTNN modules (incl. vision in this test) are installed."""
    _require_symbiote_run_mode()
    apply_qwen3_omni_talker_prepare_inputs_fix()

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model = load_qwen3_omni_moe_for_conditional_generation_bf16(MODEL_PATH)
    model.to(dtype=torch.bfloat16)

    # Talker may mix MoE and dense MLPs; only layers matching layer0's MLP class are replaced (see test_qwen3_talker_moe).
    talker_moe_class = type(model.talker.model.layers[0].mlp)
    talker_moe_layer_indices = [
        i for i, layer in enumerate(model.talker.model.layers) if type(layer.mlp) == talker_moe_class
    ]

    register_module_replacement_dict(model.thinker, NN_TO_TTNN_THINKER, model_config=None)
    register_module_replacement_dict(model.talker, NN_TO_TTNN_TALKER, model_config=None)
    _register_code2wav_nn_to_ttnn(model)
    set_device(model, mesh_device)
    _patch_thinker_talker_device_dtype(model)

    n_thinker = len(model.thinker.model.layers)
    for i, layer in enumerate(model.thinker.model.layers):
        assert isinstance(
            layer.mlp, TTNNQwen3OmniThinkerNaiveMoE
        ), f"thinker.layers[{i}].mlp expected TTNN thinker naive MoE, got {type(layer.mlp)}"
        assert isinstance(
            layer.self_attn, TTNNQwen3OmniAttention
        ), f"thinker.layers[{i}].self_attn expected TTNNQwen3OmniAttention, got {type(layer.self_attn)}"

    n_vision = len(model.thinker.visual.blocks)
    for i, block in enumerate(model.thinker.visual.blocks):
        assert isinstance(
            block.attn, TTNNQwen3VLMoeVisionAttention
        ), f"thinker.visual.blocks[{i}].attn expected TTNNQwen3VLMoeVisionAttention, got {type(block.attn)}"

    for i in talker_moe_layer_indices:
        assert isinstance(
            model.talker.model.layers[i].mlp, TTNNQwen3TalkerMoE
        ), f"talker.layers[{i}].mlp expected TTNNQwen3TalkerMoE, got {type(model.talker.model.layers[i].mlp)}"

    n_talker = len(model.talker.model.layers)
    for i, layer in enumerate(model.talker.model.layers):
        assert isinstance(
            layer.self_attn, TTNNQwen3Attention
        ), f"talker.layers[{i}].self_attn expected TTNNQwen3Attention, got {type(layer.self_attn)}"

    n_audio = len(model.thinker.audio_tower.layers)
    for i, layer in enumerate(model.thinker.audio_tower.layers):
        assert isinstance(
            layer.self_attn, TTNNQwenAudioAttentionOptimized
        ), f"thinker.audio_tower.layers[{i}].self_attn expected TTNNQwenAudioAttentionOptimized, got {type(layer.self_attn)}"

    code2wav = getattr(model, "code2wav", None)
    n_code2wav = 0
    if code2wav is not None and getattr(code2wav, "pre_transformer", None) is not None:
        n_code2wav = len(code2wav.pre_transformer.layers)
        for i, layer in enumerate(code2wav.pre_transformer.layers):
            assert isinstance(
                layer.self_attn, TTNNQwen3OmniMoeCode2WavAttention
            ), f"code2wav.pre_transformer.layers[{i}].self_attn expected TTNNQwen3OmniMoeCode2WavAttention, got {type(layer.self_attn)}"

    print(
        f"Replacements OK: thinker {n_thinker} (MoE+attn), vision {n_vision} (TTNN attn), talker MoE+attn "
        f"(mesh {mesh_device.get_num_devices()} device(s)); "
        f"audio_tower {n_audio}, code2wav {n_code2wav}, talker {n_talker} layers"
    )


def test_qwen_omni(mesh_device):
    """Test Qwen3-Omni model with TTNN acceleration (``nn_to_ttnn``); mirrors ``test_qwen_omni`` + vision TTNN."""
    _require_symbiote_run_mode()

    # HF: talker prepare_inputs_for_generation vs GenerationMixin next_sequence_length (transformers >= 4.49)
    apply_qwen3_omni_talker_prepare_inputs_fix()

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    print(f"Loading Qwen3-Omni model from {MODEL_PATH}...")
    model = load_qwen3_omni_moe_for_conditional_generation_bf16(MODEL_PATH)
    model.to(dtype=torch.bfloat16)

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
                {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
                {
                    "type": "text",
                    "text": "What can you see and hear? Answer in one short sentence.Give the car names in a comma separated list.",
                },
            ],
        },
    ]

    # Set whether to use audio in video
    USE_AUDIO_IN_VIDEO = True

    print("Registering TTNN module replacements...")
    register_module_replacement_dict(model.thinker, NN_TO_TTNN_THINKER, model_config=None)
    register_module_replacement_dict(model.talker, NN_TO_TTNN_TALKER, model_config=None)
    _register_code2wav_nn_to_ttnn(model)

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
    # Use deterministic talker decoding to make waveform quality reproducible.
    text_ids, audio = model.generate(
        **inputs,
        speaker="Ethan",
        thinker_return_dict_in_generate=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        talker_do_sample=False,
        talker_max_new_tokens=1024,
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
