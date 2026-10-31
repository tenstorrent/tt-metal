# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MoE-LLaVA-Qwen-1.8B-4e Integration and Device PCC Tests

Validates the MoE-LLaVA port with CPU integration tests and hardware-gated
device PCC tests. CPU tests avoid downloading the full checkpoint:

    1. ModelArgs reads the MoE config (num_experts, moe_layer_ids, use_hf_rope,
       Qwen1's half-intermediate 5504) from Qwen1-1.8B-moe/config.json
    2. The checkpoint converter splits dense/MoE layers correctly
    3. extract_qwen_moe_state_dict_from_llava handles the real
       `transformer.*` checkpoint layout (vision tower / projector skipped)
    4. LlavaModel extracts the projector from both `model.mm_projector.*`
       (LLaVA-MORE) and `transformer.mm_projector.*` (MoE-LLaVA) prefixes
    5. VisionModelArgs qwen-moe auto-defaults (projector 1024→2048→2048)
    6. The QWenTokenizer loads from the HF hub (LanguageBind/MoE-LLaVA-Qwen-1.8B-4e
       with remote code from Qwen/Qwen-1_8B) and round-trips — network-gated
    7. The MoE-LLaVA conversation template matches conv_templates["qwen"]

Converter tests use scaled-down dimensions (key mapping is what matters); the
ModelArgs test asserts the real production values from config.json.

Device tests cover the dense block, single MoE layer, full prefill, and decode
PCC gates. They require a healthy n150 or n300 mesh.

Usage:
    python -m pytest models/demos/llava/tests/test_moe_llava.py -v

Reference:
    - models/demos/llava/tt/qwen1_moe_checkpoint.py
    - models/demos/llava/tt/llm_wrapper.py (extract_qwen_moe_state_dict_from_llava)
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

# Compatibility shim for ttnn builds lacking ttnn.tools.trace_allocation_tracker
import models.demos.llava.ttnn_tools_shim  # noqa: F401
from models.demos.llava.tt.llava_model import build_moe_llava_prompt
from models.demos.llava.tt.llm_wrapper import extract_qwen_moe_state_dict_from_llava
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.qwen1_moe_checkpoint import convert_qwen_moe_to_meta, qwen_mlp_to_meta

QWEN1_MOE_CONFIG_DIR = str(
    Path(__file__).resolve().parents[4] / "models" / "tt_transformers" / "model_params" / "Qwen1-1.8B-moe"
)
LLAMA31_CONFIG_DIR = str(
    Path(__file__).resolve().parents[4] / "models" / "tt_transformers" / "model_params" / "Llama-3.1-8B-Instruct"
)

# Real production values (asserted against config.json via ModelArgs)
N_LAYERS = 24
MOE_LAYER_IDS = list(range(0, 24, 2))
VOCAB = 151936
I_HALF = 5504  # Qwen1 effective FFN width (intermediate_size // 2)

# Scaled-down dims for synthetic-checkpoint converter tests (memory-light)
TH = 64  # test hidden size
TI = 128  # test FFN width (TH * 2, mirrors 5504 = 2048*2.6875 ratio-free)
TVOCAB = 256
NUM_EXPERTS = 4


def _hf_hub_reachable() -> bool:
    """The QWenTokenizer ships as hub remote code; tokenizer tests need the hub."""
    import socket

    try:
        socket.create_connection(("huggingface.co", 443), timeout=2).close()
        return True
    except OSError:
        return False


requires_hf_hub = pytest.mark.skipif(
    not _hf_hub_reachable(), reason="requires network to fetch the QWenTokenizer remote code from the HF hub"
)


@pytest.fixture
def _no_tokenizer(monkeypatch):
    """ModelArgs.__init__ builds a tokenizer (hub/asset dependent); stub it for offline config tests.

    Nothing consumes args.tokenizer in these tests (assertions live in the
    network-gated hub test), and create_processor() already swallows load
    failures, so stubbing create_tokenizer keeps the ModelArgs tests offline.
    """
    from models.tt_transformers.tt.model_config import ModelArgs

    monkeypatch.setattr(ModelArgs, "create_tokenizer", lambda self: None)


# =============================================================================
# Phase 1.1/1.2: ModelArgs MoE config loading
# =============================================================================


def test_model_args_reads_moe_config(_no_tokenizer):
    """ModelArgs must pick up MoE fields + Qwen1 quirks from Qwen1-1.8B-moe/config.json."""
    os.environ["HF_MODEL"] = QWEN1_MOE_CONFIG_DIR
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh_device=None, instruct=True, dummy_weights=False, max_batch_size=1, max_seq_len=1024)

    # Architecture (matches LanguageBind/MoE-LLaVA-Qwen-1.8B-4e config.json)
    assert args.dim == 2048
    assert args.n_layers == N_LAYERS
    assert args.n_heads == 16 and args.n_kv_heads == 16  # MHA
    assert args.head_dim == 128
    assert args.vocab_size == VOCAB and args.padded_vocab_size == VOCAB  # 64-divisible: no padding
    assert args.norm_eps == 1e-06
    assert args.rope_theta == 10000

    # Qwen1 QWenMLP uses intermediate_size // 2 — config.json must carry the
    # effective 5504 so args.hidden_dim matches the checkpoint weights.
    assert args.hidden_dim == I_HALF, "hidden_dim must be Qwen1's half-intermediate 5504"

    # MoE fields (Phase 1.1 fix — previously lost in rebase)
    assert args.is_mixture_of_experts is True
    assert args.num_experts == NUM_EXPERTS
    assert args.num_experts_per_tok == 2
    assert args.moe_layer_ids == MOE_LAYER_IDS

    # Qwen1 weights stay in HF layout (converter does no QKV permute) → HF rope
    assert args.use_hf_rope is True
    assert args.use_qk_fused is False

    # Tokenizer identity is covered by test_qwen_tokenizer_hub_load (network-gated);
    # here create_tokenizer is stubbed to keep this config test offline.


def test_model_args_llama_regression(_no_tokenizer):
    """The MoE config gating must not affect the dense LLaMA path."""
    os.environ["HF_MODEL"] = LLAMA31_CONFIG_DIR
    from models.tt_transformers.tt.model_config import ModelArgs

    args = ModelArgs(mesh_device=None, instruct=True, dummy_weights=False, max_batch_size=1, max_seq_len=1024)
    assert not getattr(args, "is_mixture_of_experts", False)
    assert getattr(args, "moe_layer_ids", None) is None
    assert args.use_hf_rope is False
    assert args.hidden_dim == 14336


# =============================================================================
# Phase 2.1: checkpoint converter + Meta-format extraction
# =============================================================================


def _synthetic_moe_llava_state_dict():
    """Minimal MoE-LLaVA-Qwen-shaped checkpoint (transformer.* keys, 1 MoE + 1 dense layer)."""
    sd = {}
    sd["transformer.wte.weight"] = torch.zeros(TVOCAB, TH)
    sd["transformer.ln_f.weight"] = torch.ones(TH)
    sd["lm_head.weight"] = torch.zeros(TVOCAB, TH)
    for l in range(2):
        sd[f"transformer.h.{l}.attn.c_attn.weight"] = torch.zeros(3 * TH, TH)
        sd[f"transformer.h.{l}.attn.c_attn.bias"] = torch.zeros(3 * TH)
        sd[f"transformer.h.{l}.attn.c_proj.weight"] = torch.zeros(TH, TH)
        sd[f"transformer.h.{l}.ln_1.weight"] = torch.ones(TH)
        sd[f"transformer.h.{l}.ln_2.weight"] = torch.ones(TH)
        if l % 2 == 0:  # MoE layer
            sd[f"transformer.h.{l}.mlp.wg.weight"] = torch.zeros(TH, NUM_EXPERTS)
            for e in range(NUM_EXPERTS):
                sd[f"transformer.h.{l}.mlp.experts.{e}.w1.weight"] = torch.zeros(TI, TH)
                sd[f"transformer.h.{l}.mlp.experts.{e}.w2.weight"] = torch.zeros(TI, TH)
                sd[f"transformer.h.{l}.mlp.experts.{e}.c_proj.weight"] = torch.zeros(TH, TI)
        else:  # dense layer
            sd[f"transformer.h.{l}.mlp.w1.weight"] = torch.zeros(TI, TH)
            sd[f"transformer.h.{l}.mlp.w2.weight"] = torch.zeros(TI, TH)
            sd[f"transformer.h.{l}.mlp.c_proj.weight"] = torch.zeros(TH, TI)
    # Vision tower + projector live in the full checkpoint; extraction must skip them
    sd["transformer.image_tower.vision_model.class_embedding"] = torch.zeros(TH)
    sd["transformer.mm_projector.0.weight"] = torch.zeros(TH, 1024)
    sd["transformer.mm_projector.0.bias"] = torch.zeros(TH)
    sd["transformer.mm_projector.2.weight"] = torch.zeros(TH, TH)
    sd["transformer.mm_projector.2.bias"] = torch.zeros(TH)
    return sd


def test_convert_qwen_moe_to_meta_layer_split():
    """Even layers convert to block_sparse_moe; odd layers convert to feed_forward."""
    sd = _synthetic_moe_llava_state_dict()
    meta_sd, leftover = convert_qwen_moe_to_meta(dict(sd), n_layers=2, moe_layer_ids=[0], num_heads=16, head_dim=128)

    # MoE layer 0
    assert "layers.0.block_sparse_moe.gate.weight" in meta_sd
    assert tuple(meta_sd["layers.0.block_sparse_moe.gate.weight"].shape) == (TH, NUM_EXPERTS)
    for e in range(NUM_EXPERTS):
        assert tuple(meta_sd[f"layers.0.block_sparse_moe.experts.{e}.w1.weight"].shape) == (TI, TH)
        assert tuple(meta_sd[f"layers.0.block_sparse_moe.experts.{e}.w2.weight"].shape) == (TH, TI)
        assert tuple(meta_sd[f"layers.0.block_sparse_moe.experts.{e}.w3.weight"].shape) == (TI, TH)
    # Dense layer 1
    assert "layers.1.feed_forward.w1.weight" in meta_sd
    assert "layers.1.block_sparse_moe.gate.weight" not in meta_sd
    # Attention split from fused c_attn
    assert tuple(meta_sd["layers.0.attention.wq.weight"].shape) == (TH, TH)
    assert tuple(meta_sd["layers.0.attention.wk.weight"].shape) == (TH, TH)
    assert tuple(meta_sd["layers.0.attention.wv.weight"].shape) == (TH, TH)
    # Only the vision/projector keys remain as leftovers
    assert set(leftover.keys()) == {
        "transformer.image_tower.vision_model.class_embedding",
        "transformer.mm_projector.0.weight",
        "transformer.mm_projector.0.bias",
        "transformer.mm_projector.2.weight",
        "transformer.mm_projector.2.bias",
    }


def test_extract_qwen_moe_state_dict_from_llava():
    """The llm_wrapper extractor skips vision/projector, audits leftovers, and produces Meta keys."""
    sd = _synthetic_moe_llava_state_dict()
    sd["transformer.h.0.attn.rotary_emb.inv_freq"] = torch.zeros(32)  # non-persistent buffer
    meta_sd = extract_qwen_moe_state_dict_from_llava(sd, moe_layer_ids=[0], n_layers=2, num_heads=16, head_dim=128)

    assert "tok_embeddings.weight" in meta_sd and "norm.weight" in meta_sd and "output.weight" in meta_sd
    assert "layers.0.block_sparse_moe.gate.weight" in meta_sd  # MoE layer
    assert "layers.1.feed_forward.w1.weight" in meta_sd  # dense layer
    # vision tower + projector + inv_freq must be gone
    assert not any("image_tower" in k or "mm_projector" in k or "inv_freq" in k for k in meta_sd)


def test_extract_qwen_moe_state_dict_rejects_wrong_layout(expect_error):
    """A LLaVA-MORE-style (model.*) dict must be rejected with a clear error."""
    with expect_error(KeyError, "does not look like MoE-LLaVA"):
        extract_qwen_moe_state_dict_from_llava(
            {"model.layers.0.self_attn.q_proj.weight": torch.zeros(TH, TH)},
            moe_layer_ids=[0],
            n_layers=2,
        )


def test_qwen_swiglu_remap_numeric():
    """The SwiGLU remap (Qwen w2→gate) must be numerically exact."""
    import torch.nn.functional as F

    torch.manual_seed(0)
    qwen_w1, qwen_w2 = (torch.randn(TI, TH) for _ in range(2))
    qwen_cproj = torch.randn(TH, TI)  # c_proj: Linear(intermediate -> hidden), weight [TH, TI]
    x = torch.randn(1, 8, TH)
    # Qwen1 forward: a1 = w1(x) up; a2 = w2(x) gate; out = c_proj(a1 * silu(a2))
    qwen_out = ((x @ qwen_w1.T) * F.silu(x @ qwen_w2.T)) @ qwen_cproj.T
    meta = qwen_mlp_to_meta({"w1.weight": qwen_w1, "w2.weight": qwen_w2, "c_proj.weight": qwen_cproj})
    # Meta forward: w2(silu(w1(x)) * w3(x)) with w1=Qwen.w2 gate, w3=Qwen.w1 up
    meta_out = (F.silu(x @ meta["w1.weight"].T) * (x @ meta["w3.weight"].T)) @ meta["w2.weight"].T
    assert torch.allclose(qwen_out, meta_out, atol=1e-5)


# =============================================================================
# Phase 2.4: LlavaModel backbone integration
# =============================================================================


def _make_llava_model(llava_state_dict, llm_backbone, tmp_path):
    from models.demos.llava.tt.llava_model import LlavaModel

    args = VisionModelArgs(mesh_device=None, max_batch_size=1, llm_backbone=llm_backbone)
    return LlavaModel(
        args=args,
        dtype=None,  # no device ops in this test
        llava_state_dict=llava_state_dict,
        tokenizer=None,
        weight_cache_path=tmp_path,
        llm_backbone=llm_backbone,
    )


def test_llava_model_projector_extraction_both_prefixes(tmp_path):
    """MoE-LLaVA nests the projector under transformer.mm_projector.* (not model.*)."""
    moe_sd = _synthetic_moe_llava_state_dict()
    model = _make_llava_model(moe_sd, "qwen-moe", tmp_path)
    assert set(model.projector_sd.keys()) == {
        "mm_projector.0.weight",
        "mm_projector.0.bias",
        "mm_projector.2.weight",
        "mm_projector.2.bias",
    }
    assert tuple(model.projector_sd["mm_projector.0.weight"].shape) == (TH, 1024)
    # Backbone-level defaults (real dims, from VisionModelArgs)
    assert model.llm_hidden_size == 2048
    assert model.llm_model_config == "Qwen1-1.8B-moe"

    # LLaVA-MORE layout must keep working (model.mm_projector.*)
    llama_sd = {f"model.mm_projector.{i}.{k}": torch.zeros(4096, 4096) for i in (0, 2) for k in ("weight", "bias")}
    model = _make_llava_model(llama_sd, "llama3.1", tmp_path)
    assert len(model.projector_sd) == 4
    assert model.llm_hidden_size == 4096
    assert model.llm_model_config == "Llama-3.1-8B-Instruct"


def test_llava_model_rejects_missing_projector(expect_error, tmp_path):
    with expect_error(KeyError, "projector"):
        _make_llava_model({"transformer.wte.weight": torch.zeros(2, 2)}, "qwen-moe", tmp_path)


def test_qwen_cpu_embed_from_state_dict(tmp_path):
    """qwen-moe builds the embedding lookup from transformer.wte.weight (no HF download)."""
    moe_sd = _synthetic_moe_llava_state_dict()
    model = _make_llava_model(moe_sd, "qwen-moe", tmp_path)
    embed_model = model._load_cpu_embed_model()
    embeds = embed_model.model.embed_tokens(torch.tensor([[1, 2, 3]]))
    assert tuple(embeds.shape) == (1, 3, TH)


# =============================================================================
# Phase 1.3: VisionModelArgs defaults
# =============================================================================


def test_vision_model_args_qwen_moe_defaults():
    args = VisionModelArgs(llm_backbone="qwen-moe")
    assert args.projector_hidden_size == 2048
    assert args.llm_hidden_size == 2048

    # Explicit overrides win
    args = VisionModelArgs(llm_backbone="qwen-moe", projector_hidden_size=1234)
    assert args.projector_hidden_size == 1234

    # LLaMA default untouched
    args = VisionModelArgs()
    assert args.projector_hidden_size == 4096 and args.llm_hidden_size == 4096


# =============================================================================
# Tokenizer + conversation template
# =============================================================================


@requires_hf_hub
def test_qwen_tokenizer_hub_load():
    """The Qwen1 tokenizer loads from the hub repo (remote code via Qwen/Qwen-1_8B) and round-trips."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("LanguageBind/MoE-LLaVA-Qwen-1.8B-4e", trust_remote_code=True)
    assert type(tok).__name__ == "QWenTokenizer"
    text = "A chat between a curious user and an artificial intelligence assistant."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == text
    assert tok.eos_token_id == 151643  # 151643


def test_moe_llava_conversation_template():
    """The prompt must mirror MoE-LLaVA conv_templates['qwen'] (conv_phi, SeparatorStyle.TWO)."""
    prompt = build_moe_llava_prompt("What is in this image?")
    assert prompt.startswith(
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: <image>\n"
    )
    assert prompt.endswith("What is in this image? ASSISTANT:")
    # <image> splits into exactly two parts for the interleaver
    assert len(prompt.split("<image>")) == 2


# =============================================================================
# Device validation (n300 / n150) — PCC gates against the PyTorch reference.
# Mesh shape follows
# MESH_DEVICE (default N150/single chip; the 2-chip mesh is MESH_DEVICE=N300).
# =============================================================================

MOE_LLAVA_CKPT = "LanguageBind/MoE-LLaVA-Qwen-1.8B-4e"
_SEQ_LEN = 128


def _load_real_checkpoint():
    """Load the serialized LanguageBind checkpoint (DeepSpeed MoE key layout)."""
    import glob

    from huggingface_hub import snapshot_download

    path = snapshot_download(MOE_LLAVA_CKPT, allow_patterns=["*.bin", "*.json", "qwen.tiktoken"])
    sd = {}
    for f in sorted(glob.glob(f"{path}/pytorch_model*.bin")):
        sd.update(torch.load(f, map_location="cpu", weights_only=True))
    assert len(sd) > 600, f"Unexpected checkpoint size: {len(sd)} keys"
    return sd


def _reference_logits(n_layers, tokens):
    """Last-token logits from the CPU PyTorch reference, truncated to n_layers."""
    from models.demos.llava.reference.qwen_moe_reference import load_model as load_moe_reference_model

    model, _config = load_moe_reference_model(MOE_LLAVA_CKPT, dtype=torch.float32)
    if n_layers is not None and n_layers < len(model.transformer.h):
        model.transformer.h = model.transformer.h[:n_layers]
    with torch.no_grad():
        out = model(input_ids=tokens)
    return out.logits[0, -1, :].float()


def _device_prefill_logits(mesh_device, moe_sd, n_layers, tokens):
    """Build the TTNN Qwen1-MoE model and run chunked prefill; return host logits."""
    import ttnn
    from models.demos.llava.tt.llm_wrapper import create_qwen_moe_llm

    llm_dtype = ttnn.bfloat8_b
    tt_model, tt_model_args, tt_kv_cache, page_table = create_qwen_moe_llm(
        mesh_device,
        moe_sd,
        dtype=llm_dtype,
        max_seq_len=1024,
        n_layers=n_layers,
    )

    # Host embeddings from the checkpoint's wte (mirrors the demo embed shim)
    wte = moe_sd["transformer.wte.weight"]
    input_embeds = torch.nn.functional.embedding(tokens[0], wte.float()).unsqueeze(0)
    total_len = input_embeds.shape[1]
    seq_len = input_embeds.shape[1]  # 128 — already tile-aligned

    chunk_size = 128
    last_token_idx = total_len - 1
    logits = None
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_embeds = input_embeds[:, chunk_start:chunk_end, :]
        block_start = chunk_start // 32
        block_end = (chunk_end + 31) // 32
        chunk_pt = page_table[:, block_start:block_end]

        x = ttnn.from_torch(
            chunk_embeds,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        x = ttnn.unsqueeze_to_4D(x)

        mat_len = tt_model.rope_setup.cos_matrix_prefill.shape[2]
        slice_end = min(mat_len, chunk_start + (chunk_end - chunk_start))
        cos_slice = tt_model.rope_setup.cos_matrix_prefill[:, :, chunk_start:slice_end, :]
        sin_slice = tt_model.rope_setup.sin_matrix_prefill[:, :, chunk_start:slice_end, :]
        rot_mats_global = [cos_slice, sin_slice]

        tt_page_table = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_chunk_page_table = ttnn.from_torch(
            chunk_pt,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt_chunk_start_idx = None
        if chunk_start > 0:
            tt_chunk_start_idx = ttnn.from_torch(
                torch.tensor([chunk_start], dtype=torch.int32),
                device=mesh_device,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        is_last_chunk = chunk_end >= seq_len
        get_last_token = (last_token_idx - chunk_start) // 32 * 32 if is_last_chunk else -1

        tt_out = tt_model.ttnn_prefill_forward(
            x,
            rot_mats_global=rot_mats_global,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            chunk_start_idx=tt_chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=tt_kv_cache,
            batch_size=1,
        )
        if is_last_chunk:
            logits = tt_model.process_output_prefill(tt_out.cpu(), last_token_idx=last_token_idx % 32)
        else:
            ttnn.deallocate(tt_out, force=True)

    return tt_model, tt_kv_cache, page_table, logits.float()


def _device_decode_logits(tt_model, tt_kv_cache, page_table, next_token, current_pos, mesh_device):
    """One decode step; returns host logits for the sampled position."""

    tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_pt_decode = tt_model.prepare_inputs_decode(
        next_token, current_pos, page_table
    )
    tt_logits, _ = tt_model.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rot_mat_idxs,
        page_table=tt_pt_decode,
        kv_cache=tt_kv_cache,
    )
    decode_logits = tt_model.process_output_decode(tt_logits, B=1)
    return decode_logits[:, -1, :].float()


def _mesh_shape_from_env():
    """Device tests follow MESH_DEVICE (N150 -> 1 chip default, N300 -> 2 chips)."""
    import os

    return (1, 2) if os.environ.get("MESH_DEVICE", "N150") == "N300" else (1, 1)


@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": True} if _mesh_shape_from_env() == (1, 2) else {}], indirect=True
)
def test_moe_llava_qwen_prefill_pcc(mesh_device):
    """PCC validation: full 24-layer Qwen1-MoE prefill vs PyTorch reference (>= 0.91)."""
    from models.common.utility_functions import comp_pcc

    torch.manual_seed(0)
    tokens = torch.randint(1000, 100000, (1, _SEQ_LEN))
    moe_sd = _load_real_checkpoint()

    ref_logits = _reference_logits(None, tokens)
    tt_model, tt_kv_cache, _page_table, dev_logits = _device_prefill_logits(mesh_device, moe_sd, None, tokens)
    del tt_model, tt_kv_cache

    passing, pcc = comp_pcc(ref_logits, dev_logits.flatten(), pcc=0.91)
    logger.info(f"  Full prefill PCC: {pcc:.4f} (>= 0.91)")
    assert passing, f"Prefill PCC {pcc:.4f} < 0.91"


@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": True} if _mesh_shape_from_env() == (1, 2) else {}], indirect=True
)
def test_moe_llava_qwen_decode_pcc(mesh_device):
    """PCC validation: Qwen1-MoE decode vs PyTorch reference (>= 0.99)."""
    from models.common.utility_functions import comp_pcc

    torch.manual_seed(0)
    tokens = torch.randint(1000, 100000, (1, _SEQ_LEN))
    moe_sd = _load_real_checkpoint()

    tt_model, tt_kv_cache, page_table, prefill_logits = _device_prefill_logits(mesh_device, moe_sd, None, tokens)
    next_token = torch.argmax(prefill_logits.flatten()).view(1, 1)
    current_pos = torch.tensor([tokens.shape[1]], dtype=torch.int32)

    ref_next = _reference_logits(None, torch.cat([tokens, next_token], dim=1))
    dev_next = _device_decode_logits(tt_model, tt_kv_cache, page_table, next_token, current_pos, mesh_device)
    del tt_model, tt_kv_cache

    passing, pcc = comp_pcc(ref_next.flatten(), dev_next.flatten(), pcc=0.99)
    logger.info(f"  Decode PCC: {pcc:.4f} (>= 0.99)")
    assert passing, f"Decode PCC {pcc:.4f} < 0.99"


@pytest.mark.parametrize("mesh_device", [_mesh_shape_from_env()], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": True} if _mesh_shape_from_env() == (1, 2) else {}], indirect=True
)
def test_moe_layer_single_block_pcc(mesh_device):
    """PCC: 2-layer slice (MoE layer 0 + dense layer 1) vs PyTorch reference (>= 0.98).

    A truncated 2-layer model covers one full MoE block and one full dense
    block — the doc's Step 3.1 (dense block) and Step 3.2 (single MoE layer)
    gates in a single pass over the real checkpoint.
    """
    from models.common.utility_functions import comp_pcc

    torch.manual_seed(0)
    tokens = torch.randint(1000, 100000, (1, _SEQ_LEN))
    moe_sd = _load_real_checkpoint()

    ref_logits = _reference_logits(2, tokens)
    tt_model, tt_kv_cache, _page_table, dev_logits = _device_prefill_logits(mesh_device, moe_sd, 2, tokens)
    del tt_model, tt_kv_cache

    passing, pcc = comp_pcc(ref_logits, dev_logits.flatten(), pcc=0.98)
    logger.info(f"  Single MoE + dense block PCC: {pcc:.4f} (>= 0.98)")
    assert passing, f"Block PCC {pcc:.4f} < 0.98"
