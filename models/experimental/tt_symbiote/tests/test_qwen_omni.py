# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
    Qwen3OmniMoeVisionMLP,
    Qwen3OmniMoeVisionAttention,
)
from qwen_omni_utils import process_mm_info

from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.run_config import DispatchManager
from models.experimental.tt_symbiote.modules.moe import TTNNQwen3OmniThinkerNaiveMoE, TTNNQwen3TalkerMoE
from models.experimental.tt_symbiote.core.hf_generation_compat import apply_qwen3_omni_talker_prepare_inputs_fix
from models.experimental.tt_symbiote.modules.attention import TTNNQwenAudioAttention
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3OmniMoeCode2WavAttention
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3Attention
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3OmniAttention
from models.experimental.tt_symbiote.modules.attention import TTNNQwen3VLMoeVisionAttention
from models.experimental.tt_symbiote.modules.linear import TTNNQwen3OmniVisionMLP
from models.experimental.tt_symbiote.utils.device_management import set_device
from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

_ALLOWED_SYMBIOTE_RUN_MODES = frozenset({"CPU", "NORMAL", "NORMAL_WITH_FALLBACK"})

# Static HF → TTNN maps
NN_TO_TTNN_THINKER = {
    Qwen3OmniMoeThinkerTextSparseMoeBlock: TTNNQwen3OmniThinkerNaiveMoE,
    Qwen3OmniMoeThinkerTextAttention: TTNNQwen3OmniAttention,
    Qwen3OmniMoeVisionAttention: TTNNQwen3VLMoeVisionAttention,
    Qwen3OmniMoeVisionMLP: TTNNQwen3OmniVisionMLP,
    Qwen3OmniMoeAudioAttention: TTNNQwenAudioAttention,
}
NN_TO_TTNN_CODE2WAV = {
    Qwen3OmniMoeCode2WavAttention: TTNNQwen3OmniMoeCode2WavAttention,
}
NN_TO_TTNN_TALKER = {
    Qwen3OmniMoeTalkerTextSparseMoeBlock: TTNNQwen3TalkerMoE,
    Qwen3OmniMoeThinkerTextAttention: TTNNQwen3Attention,
}

MODEL_NAME = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


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


def _force_t3k_runtime_guard_for_pcc(mesh_device):
    """Thinker/talker MoE PCC runs TTNN paths tagged for T3K; require an 8-device mesh."""
    if mesh_device.get_num_devices() < 8:
        pytest.skip(
            f"PCC MoE tests expect an 8-device T3K-class mesh (got {mesh_device.get_num_devices()}); "
            "set MESH_DEVICE=T3K."
        )


def _replicate_mesh_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


def _rotary_cos_sin_mesh_mapper(mesh_device, cos_torch: torch.Tensor):
    """Mapper for uploading HF ``cos``/``sin``.

    ``TTNNQwen3OmniAttention`` / ``TTNNQwen3Attention`` call ``_maybe_all_gather`` on rotaries.
    If ``cos``/``sin`` are **replicated**, all-gather concatenates identical shards on the last
    dim (e.g. 128 → 1024 on 8 devices) and RoPE then fails in ``ttnn.pad``. Shard the last dim
    so all-gather reconstructs the true rotary width (``qwen_attention`` uses the same pattern).
    """
    n = mesh_device.get_num_devices()
    rotary_dim = int(cos_torch.shape[-1])
    if n > 1 and rotary_dim % n == 0:
        return ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    if n > 1 and rotary_dim % n != 0:
        pytest.skip(
            f"PCC needs rotary_dim ({rotary_dim}) divisible by mesh size ({n}) for width-sharded "
            "cos/sin upload; use a different mesh or checkpoint."
        )
    return _replicate_mesh_mapper(mesh_device)


def _shape_int_tuple(shape):
    """Mesh / ttnn logical shapes may expose dims as tensors; ``torch.empty`` needs ``int``."""
    out = []
    for s in shape:
        if isinstance(s, torch.Tensor):
            s = int(s.item())
        else:
            s = int(s)
        out.append(s)
    return tuple(out)


def _to_plain_torch_tensor(x: torch.Tensor) -> torch.Tensor:
    """Copy tensor subclasses (e.g. ``TorchTTNNTensor``) into a plain ``torch.Tensor`` for ``comp_pcc``."""
    x = x.detach().contiguous()
    if type(x) is torch.Tensor:
        return x.clone()
    out = torch.empty(_shape_int_tuple(x.shape), dtype=x.dtype, device=x.device)
    out.copy_(x)
    return out


def _resolve_torch_tensor_for_pcc(x, mesh_device=None, tuple_first: bool = True):
    """HF / TTNN / symbiote outputs → plain ``torch.Tensor`` for ``comp_pcc`` (never ``None``)."""
    # Symbiote sometimes returns ``[ttnn_tensor, None]`` (list), not ``(attn, None, ...)`` (tuple).
    if tuple_first and isinstance(x, (tuple, list)):
        x = x[0]
    if x is None:
        raise TypeError("_resolve_torch_tensor_for_pcc: got None (missing output tensor)")
    if isinstance(x, TorchTTNNTensor):
        tt = x
        x = tt.to_torch
        if x is None:
            x = getattr(tt, "elem", None)
        if x is None and getattr(tt, "ttnn_tensor", None) is not None:
            x = ttnn.to_torch(tt.ttnn_tensor)
        if x is None and hasattr(tt, "to_ttnn"):
            raw = tt.to_ttnn
            if isinstance(raw, ttnn.Tensor):
                x = ttnn.to_torch(raw)
    if isinstance(x, ttnn.Tensor):
        if mesh_device is None or mesh_device.get_num_devices() <= 1:
            x = ttnn.to_torch(x)
        else:
            try:
                x = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
            except Exception:
                out = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                if out.dim() >= 1 and out.shape[0] > 1:
                    out = out[0:1]
                x = out
    if x is None:
        raise TypeError("_resolve_torch_tensor_for_pcc: could not convert to torch tensor")
    if not isinstance(x, torch.Tensor):
        return torch.as_tensor(x).detach().clone().contiguous()
    return _to_plain_torch_tensor(x)


def _to_torch_output(x, mesh_device, tuple_first: bool = True):
    """Backward-compatible alias for PCC tests (unwrap TTNN output to torch)."""
    return _resolve_torch_tensor_for_pcc(x, mesh_device, tuple_first=tuple_first)


def _comp_pcc_in_test(golden, calculated, mesh_device=None, pcc=0.99):
    # Golden: HF tensor. Calculated: ``(attn, …)``, ``[attn, None]`` from symbiote, or raw ttnn.
    g = _resolve_torch_tensor_for_pcc(golden, mesh_device=None, tuple_first=False)
    c = _resolve_torch_tensor_for_pcc(calculated, mesh_device, tuple_first=True)
    return comp_pcc(g, c, pcc=pcc)


@pytest.fixture(scope="module")
def full_omni_model_for_pcc():
    """Single full HF load (bf16) shared by PCC tests."""
    omni_config = Qwen3OmniMoeConfig.from_pretrained(MODEL_NAME)
    patch_qwen3_omni_moe_root_config(omni_config)
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        config=omni_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )


def _first_talker_moe_layer_index(model):
    ref_cls = type(model.talker.model.layers[0].mlp)
    for i, layer in enumerate(model.talker.model.layers):
        if type(layer.mlp) == ref_cls:
            return i
    return 0


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


def test_qwen_omni_thinker_attention_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    m = full_omni_model_for_pcc
    torch_attn = m.thinker.model.layers[0].self_attn
    torch_attn.rotary_emb = m.thinker.model.rotary_emb
    text_config = m.config.thinker_config.text_config
    tt_attn = TTNNQwen3OmniAttention.from_torch(torch_attn)
    set_device(tt_attn, mesh_device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    batch, seq_len, hidden = 1, 32, text_config.hidden_size
    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = torch_attn.rotary_emb(hidden_states, position_ids)

    with torch.no_grad():
        torch_out, _ = torch_attn(
            hidden_states, position_embeddings=(cos, sin), attention_mask=None, past_key_values=None
        )
    # Hidden states: replicate (matches vision/audio). Cos/sin: last-dim shard on multi-device so
    # ``_maybe_all_gather`` on rotary tensors does not concat replicated rotary width.
    _repl = _replicate_mesh_mapper(mesh_device)
    _rot_mapper = _rotary_cos_sin_mesh_mapper(mesh_device, cos)
    tt_hs = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_repl,
    )
    tt_cos = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=_rot_mapper
    )
    tt_sin = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=_rot_mapper
    )
    tt_ret = tt_attn(
        tt_hs,
        position_embeddings=(tt_cos, tt_sin),
        attention_mask=None,
        past_key_values=None,
    )
    passing, pcc = _comp_pcc_in_test(torch_out, tt_ret, mesh_device)
    assert passing, f"Thinker attention PCC too low: {pcc}"


def test_qwen_omni_talker_attention_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    m = full_omni_model_for_pcc
    torch_attn = m.talker.model.layers[0].self_attn
    torch_attn.rotary_emb = m.talker.model.rotary_emb
    text_config = m.config.talker_config.text_config
    tt_attn = TTNNQwen3Attention.from_torch(torch_attn)
    set_device(tt_attn, mesh_device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    batch, seq_len, hidden = 1, 32, text_config.hidden_size
    hidden_states = torch.randn(batch, seq_len, hidden, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cos, sin = torch_attn.rotary_emb(hidden_states, position_ids)
    with torch.no_grad():
        torch_out, _ = torch_attn(
            hidden_states, position_embeddings=(cos, sin), attention_mask=None, past_key_values=None
        )
    _repl = _replicate_mesh_mapper(mesh_device)
    _rot_mapper = _rotary_cos_sin_mesh_mapper(mesh_device, cos)
    tt_hs = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_repl,
    )
    tt_cos = ttnn.from_torch(
        cos, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=_rot_mapper
    )
    tt_sin = ttnn.from_torch(
        sin, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=_rot_mapper
    )
    tt_ret = tt_attn(
        tt_hs,
        position_embeddings=(tt_cos, tt_sin),
        attention_mask=None,
        past_key_values=None,
    )
    passing, pcc = _comp_pcc_in_test(torch_out, tt_ret, mesh_device)
    assert passing, f"Talker attention PCC too low: {pcc}"


def test_qwen_omni_vision_attention_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    m = full_omni_model_for_pcc
    torch_attn = m.thinker.visual.blocks[0].attn
    config = m.config.thinker_config.vision_config

    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    tt_attn = TTNNQwen3VLMoeVisionAttention.from_torch(torch_attn)
    set_device(tt_attn, mesh_device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    seq_len = 128
    hidden_states = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16)

    head_dim = config.hidden_size // config.num_heads

    # Match HF / ``apply_rotary_pos_emb_vision``: ``cos``/``sin`` are ``[seq_len, head_dim]``.
    # 3D ``[seq, 1, head_dim]`` breaks TTNN: ``unsqueeze(-2)`` on 3D makes 4D rotary, so ``q * cos``
    # becomes 4D and ``permute(q, (1,0,2))`` fails (expects rank 3).
    cos = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)
    sin = torch.randn(seq_len, head_dim, dtype=torch.bfloat16)

    with torch.no_grad():
        torch_out = torch_attn(
            hidden_states,
            cu_seqlens=torch.tensor([0, seq_len]),
            position_embeddings=(cos, sin),
        )

    _repl = _replicate_mesh_mapper(mesh_device)
    # ``TTNNQwen3VLMoeVisionAttention`` all-gathers cos/sin on the last dim; replicate + multi-device
    # stacks rotary width vs width-sharded Q/K; shard last dim when divisible by mesh size.
    _rot_mapper = _rotary_cos_sin_mesh_mapper(mesh_device, cos)

    tt_input = ttnn.from_torch(
        hidden_states,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_repl,
    )

    tt_cos = ttnn.from_torch(
        cos,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_rot_mapper,
    )

    tt_sin = ttnn.from_torch(
        sin,
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=_rot_mapper,
    )

    tt_out = tt_attn(
        tt_input,
        cu_seqlens=None,
        position_embeddings=(tt_cos, tt_sin),
    )

    passing, pcc = _comp_pcc_in_test(torch_out, tt_out, mesh_device)

    assert passing, f"Vision attention PCC too low: {pcc}"


def test_qwen_omni_audio_attention_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    m = full_omni_model_for_pcc
    torch_attn = m.thinker.audio_tower.layers[0].self_attn
    config = m.config.thinker_config.audio_config

    tt_attn = TTNNQwenAudioAttention.from_torch(torch_attn)
    set_device(tt_attn, mesh_device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    seq_len = 128
    inp = torch.randn(seq_len, config.d_model, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.long)
    with torch.no_grad():
        torch_out = torch_attn(inp, cu_seqlens=cu_seqlens)
    _repl = _replicate_mesh_mapper(mesh_device)
    tt_inp = ttnn.from_torch(inp, dtype=ttnn.bfloat16, device=mesh_device, layout=ttnn.TILE_LAYOUT, mesh_mapper=_repl)
    tt_out = tt_attn(tt_inp, cu_seqlens=cu_seqlens)
    passing, pcc = _comp_pcc_in_test(torch_out, tt_out, mesh_device)
    assert passing, f"Audio attention PCC too low: {pcc}"


def test_qwen_omni_code2wav_attention_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    m = full_omni_model_for_pcc
    code2wav = getattr(m, "code2wav", None)
    if code2wav is None or getattr(code2wav, "pre_transformer", None) is None:
        pytest.skip("This checkpoint has no code2wav.pre_transformer")
    torch_attn = code2wav.pre_transformer.layers[0].self_attn
    config = m.config.code2wav_config
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"

    tt_attn = TTNNQwen3OmniMoeCode2WavAttention.from_torch(torch_attn)
    tt_attn.use_windowed_attention = False
    set_device(tt_attn, mesh_device)
    tt_attn.preprocess_weights()
    tt_attn.move_weights_to_device()

    batch, seq_len = 1, 64
    hs = torch.randn(batch, seq_len, config.hidden_size, dtype=torch.bfloat16)
    head_dim = int(torch_attn.head_dim)
    cos = torch.cos(torch.randn(batch, seq_len, head_dim, dtype=torch.bfloat16))
    sin = torch.sin(torch.randn(batch, seq_len, head_dim, dtype=torch.bfloat16))
    with torch.no_grad():
        torch_out, _ = torch_attn(hs, position_embeddings=(cos, sin), attention_mask=None)
    tt_ret = tt_attn(
        TorchTTNNTensor(hs),
        position_embeddings=(TorchTTNNTensor(cos), TorchTTNNTensor(sin)),
        attention_mask=None,
    )
    passing, pcc = _comp_pcc_in_test(torch_out, tt_ret, mesh_device)
    assert passing, f"Code2Wav attention PCC too low: {pcc}"


def test_qwen_omni_thinker_moe_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    _force_t3k_runtime_guard_for_pcc(mesh_device)
    m = full_omni_model_for_pcc
    torch_mlp = m.thinker.model.layers[0].mlp
    text_config = m.config.thinker_config.text_config

    tt_mlp = TTNNQwen3OmniThinkerNaiveMoE.from_torch(torch_mlp)
    set_device(tt_mlp, mesh_device)
    tt_mlp.preprocess_weights()
    tt_mlp.move_weights_to_device()

    x = torch.randn(1, 32, text_config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        torch_out = torch_mlp(x)
        tt_out = tt_mlp(x)
    passing, pcc = _comp_pcc_in_test(torch_out, tt_out, mesh_device)
    assert passing, f"Thinker MoE PCC too low: {pcc}"


def test_qwen_omni_talker_moe_pcc(mesh_device, full_omni_model_for_pcc):
    _require_symbiote_run_mode()
    _force_t3k_runtime_guard_for_pcc(mesh_device)
    m = full_omni_model_for_pcc
    idx = _first_talker_moe_layer_index(m)
    torch_mlp = m.talker.model.layers[idx].mlp
    text_config = m.config.talker_config.text_config

    tt_mlp = TTNNQwen3TalkerMoE.from_torch(torch_mlp)
    set_device(tt_mlp, mesh_device)
    tt_mlp.preprocess_weights()
    tt_mlp.move_weights_to_device()

    x = torch.randn(1, 32, text_config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        torch_out = torch_mlp(x)
        tt_out = tt_mlp(TorchTTNNTensor(x))
    passing, pcc = _comp_pcc_in_test(torch_out, tt_out, mesh_device)
    assert passing, f"Talker MoE PCC too low: {pcc}"


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
        assert isinstance(
            block.mlp, TTNNQwen3OmniVisionMLP
        ), f"thinker.visual.blocks[{i}].mlp expected TTNNQwen3OmniVisionMLP, got {type(block.mlp)}"

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
            layer.self_attn, TTNNQwenAudioAttention
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
                    "text": "What can you see and hear? Answer in one short sentence.Give the car names as a list of strings.",
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
    talker_max_new_tokens = int(os.environ.get("TT_SYMBIOTE_QWEN_OMNI_TALKER_MAX_NEW_TOKENS", "1024"))
    text_ids, audio = model.generate(
        **inputs,
        speaker="Ethan",
        thinker_return_dict_in_generate=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        talker_do_sample=False,
        talker_max_new_tokens=talker_max_new_tokens,
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
