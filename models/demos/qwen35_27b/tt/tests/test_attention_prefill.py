# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 prefill attention PCC tests against the HuggingFace reference.

"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen35_27b.tt.attention import Qwen35Attention
from models.demos.qwen35_27b.tt.model_config import (
    Qwen35ModelArgs,
    _replicate,
    _shard_w,
    load_qwen35_state_dict,
    prepare_attn_qg,
)
from models.demos.qwen35_27b.tt.rope import Qwen35PartialRopeSetup, get_prefill_rot_mats
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import PagedAttentionConfig


class HFReferenceAttention:
    """One-stop HF reference for PCC testing tt-metal's full_attention forward.

    Wraps Qwen3_5Attention + Qwen3_5TextRotaryEmbedding for one layer of the
    Qwen3.5-27B checkpoint. An internal DynamicCache is populated by prefill()
    so callers don't have to thread past_key_values manually. Use one instance
    per test scenario; instantiate fresh between tests so cache state doesn't
    leak.
    """

    def __init__(self, model_path, layer_idx, state_dict):
        # TODO Currently tests text attention only since TT implementation supports text-only
        text_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True).text_config
        self.text_cfg = text_cfg
        # Storing `layer_idx` so expected_kv() and any downstream caller can read the same
        # cache slot the HF attention writes KVCaches to.
        self.layer_idx = layer_idx

        self.attn = Qwen3_5Attention(text_cfg, layer_idx=layer_idx).to(torch.bfloat16).eval()

        # load_qwen35_state_dict (model_config.py:645-654) renames HF's
        # self_attn.q_proj.weight → attention.wqkv.weight etc., but the
        # tensor VALUES are identical. Copy from meta-format keys directly
        # into the HF module's slots.
        p = f"layers.{layer_idx}."
        self.attn.q_proj.weight.data.copy_(state_dict[p + "attention.wqkv.weight"].to(torch.bfloat16))
        self.attn.k_proj.weight.data.copy_(state_dict[p + "attention.wk.weight"].to(torch.bfloat16))
        self.attn.v_proj.weight.data.copy_(state_dict[p + "attention.wv.weight"].to(torch.bfloat16))
        self.attn.o_proj.weight.data.copy_(state_dict[p + "attention.wo.weight"].to(torch.bfloat16))
        self.attn.q_norm.weight.data.copy_(state_dict[p + "attention.q_norm.weight"].to(torch.bfloat16))
        self.attn.k_norm.weight.data.copy_(state_dict[p + "attention.k_norm.weight"].to(torch.bfloat16))

        self.rope = Qwen3_5TextRotaryEmbedding(text_cfg)
        # DynamicCache K/V slot populated by prefill(); read via expected_kv().
        self.past = DynamicCache()

    @staticmethod
    def _causal_mask(seq_len, dtype):
        """Causal mask [1, 1, seq_len, seq_len] with -inf above the diagonal.

        HF eager_attention_forward (modeling_qwen3_5.py:619-620) silently
        produces full (not causal) attention when attention_mask=None — it
        just skips the mask-add. We always pass this explicit mask for prefill.
        """
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype)
        mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        return mask

    def prefill(self, hidden):
        """Run reference prefill on `hidden` shaped [1, seq_len, dim].

        Populates internal cache. Returns attention output [1, seq_len, dim].
        Uses 2D position_ids so apply_interleaved_mrope (the model's
        multimodal rotary expansion) collapses to the identity for text.
        """
        seq_len = hidden.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        cos, sin = self.rope(hidden, position_ids)
        mask = self._causal_mask(seq_len, hidden.dtype)
        out, _ = self.attn(
            hidden,
            position_embeddings=(cos, sin),
            attention_mask=mask,
            past_key_values=self.past,
        )
        return out

    def expected_kv(self):
        """Return (k, v) torch tensors from internal cache for paged-KV PCC.

        Shape: [1, n_kv_heads, seq_len, head_dim] — matches the unpaged
        tt-metal cache slice the test compares against. seq_len reflects
        whatever prefill() has written so far.
        """
        return (self.past.layers[self.layer_idx].keys, self.past.layers[self.layer_idx].values)


def _find_first_full_attn_layer(layer_types):
    """Find the first full_attention layer index."""
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            return i
    raise ValueError("No full_attention layer found")


def _load_attention_weights_for_layer(state_dict, layer_idx, mesh, args, cache_dir):
    """Load attention mesh tensors for a single layer.

    TODO(tech-debt): duplicates the per-layer body of
    model.py:_load_and_wire_attention_weights; extract a shared helper.
    """
    os.makedirs(cache_dir, exist_ok=True)
    p = f"layers.{layer_idx}."
    tp = args.num_devices

    qg_reordered = prepare_attn_qg(state_dict, p, args.n_heads, args.head_dim, tp)

    tw = {}
    tw["wqkv"] = _shard_w(
        qg_reordered,
        mesh,
        dim=-1,
        memory_config=args.attn_qg_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wqkv"),
    )
    tw["wk"] = _shard_w(
        state_dict[p + "attention.wk.weight"],
        mesh,
        dim=-1,
        memory_config=args.attn_k_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wk"),
    )
    tw["wv"] = _shard_w(
        state_dict[p + "attention.wv.weight"],
        mesh,
        dim=-1,
        memory_config=args.attn_v_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wv"),
    )
    tw["wo"] = _shard_w(
        state_dict[p + "attention.wo.weight"],
        mesh,
        dim=0,
        memory_config=args.attn_wo_weight_memcfg,
        cache_path=os.path.join(cache_dir, "wo"),
    )
    # Bake +1 into the loaded q_norm/k_norm weight so attention forward stays
    # a single multiply instead of an add+multiply per token. Mirrors the
    # production loader (model.py:_load_and_wire_attention_weights).
    #   * HF weight (w) is what the safetensors checkpoint stores.
    #   * HF Qwen3_5RMSNorm computes  out = norm(x) * (1 + w)
    #     (modeling_qwen3_5.py:735); w is initialized near 0.
    #   * tt-metal forward computes   out = norm(x) * w_baked
    #     (attention.py:179-180, 463-464, 677-678); with w_baked = 1+w loaded
    #     here, the two match.
    tw["q_norm"] = _replicate(
        1.0 + state_dict[p + "attention.q_norm.weight"],
        mesh,
        os.path.join(cache_dir, "q_norm"),
    )
    tw["k_norm"] = _replicate(
        1.0 + state_dict[p + "attention.k_norm.weight"],
        mesh,
        os.path.join(cache_dir, "k_norm"),
    )
    return tw


def _allocate_vllm_kv_cache(args, paged_cfg, mesh_device):
    """Allocate the K/V cache pair vLLM hands attention forward.

    Shape per tensor: [max_num_blocks, num_local_kv_heads, block_size, head_dim].
    Dtype is bfloat16 to match the production tt-metal cache; the dtype vLLM
    itself picks isn't load-bearing for PCC since we read both sides at this
    precision.
    """
    shape = (
        paged_cfg.max_num_blocks,
        args.n_local_kv_heads,
        paged_cfg.block_size,
        args.head_dim,
    )
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    return tuple(
        ttnn.from_torch(
            zeros,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        for _ in range(2)
    )


def _build_page_table(seq_len, paged_cfg, mesh_device):
    """Build a contiguous virtual→physical page table for one user.

    At runtime vLLM's KV-cache manager produces the page table; for a unit
    test we use the simplest layout it might emit — identity arange. Shape:
    [batch=1, num_blocks], int32, ROW_MAJOR_LAYOUT, replicated across the
    mesh. Mirrors the inline single-user variant in test_e2e_generate.py.

    TODO(tech-debt): test_e2e_generate.py and test_vllm_prefill.py build
    near-identical tables inline; a shared test_paged_utils module would
    cover all three callers.
    """
    num_blocks = (seq_len + paged_cfg.block_size - 1) // paged_cfg.block_size
    pt_torch = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    return ttnn.from_torch(
        pt_torch,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


@pytest.fixture(scope="session")
def model_path():
    """Resolve HF model path from env, defaulting to the standard checkpoint location.

    Session-scoped: resolved once per pytest invocation. Also back-fills os.environ
    so downstream code (e.g. ModelArgs) that re-reads HF_MODEL gets a consistent value.
    """
    path = os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))
    os.environ["HF_MODEL"] = path  # ensure ModelArgs sees an absolute path
    return path


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        True,
        False,
    ),
    ids=(
        "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize("seq_len", [64, 256, 2048], ids=["s64", "s256", "s2048"])
@pytest.mark.parametrize("max_seq_len", [2048])
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_attention_forward_prefill(
    mesh_device, model_path, seq_len, paged_attention, page_params, max_seq_len, reset_seeds, device_params, ensure_gc
):
    """
    This test checks that TT Qwen3.5 attention prefill implementation provides the correct outputs.

    We compare using `comp_pcc` that:
    - output matches huggingface reference output
    - TT kv caches match HF kv cache.

    """
    batch_size = 1
    pcc = 0.99

    args = Qwen35ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len)
    layer_idx = _find_first_full_attn_layer(args.layer_types)
    logger.info(f"Testing layer {layer_idx} on {mesh_device.get_num_devices()} device(s), seq_len={seq_len}")

    state_dict = load_qwen35_state_dict(model_path)
    cache_dir = os.path.expanduser(f"~/models/Qwen3.5-27B-mesh-tp{args.num_devices}/test_attn_layer_{layer_idx}")
    tw = _load_attention_weights_for_layer(state_dict, layer_idx, mesh_device, args, cache_dir)
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    attn = Qwen35Attention(
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
        args=args,
        state_dict={},
        weight_cache_path=Path(cache_dir),
        layer_num=layer_idx,
        dtype=ttnn.bfloat8_b,
        transformation_mats=None,
        configuration=args,
    )
    attn.set_weights(tw)

    rope = Qwen35PartialRopeSetup(
        device=mesh_device,
        batch_size=batch_size,
        head_dim=args.head_dim,
        max_seq_len=max_seq_len,
        rope_theta=args.rope_theta,
    )
    # block_size=32 matches vLLM default; max_num_blocks is sized by page_params.
    paged_cfg = PagedAttentionConfig(
        block_size=page_params["page_block_size"], max_num_blocks=page_params["page_max_num_blocks"]
    )

    kv_cache = _allocate_vllm_kv_cache(args, paged_cfg, mesh_device)
    page_table_tt = _build_page_table(seq_len, paged_cfg, mesh_device)

    # Random input shared between the HF and tt-metal sides.
    x = torch.randn(1, seq_len, args.dim, dtype=torch.bfloat16)

    # HuggingFace reference output
    logger.info("Instantiating HF Reference and Running Prefill... ")
    ref = HFReferenceAttention(model_path, layer_idx, state_dict)
    ref_out = ref.prefill(x)  # [1, seq_len, dim] bfloat16; populates ref.past for kv caches

    # Tenstorrent forward — tt-metal expects [1, 1, seq_len, dim] replicated across the mesh.
    x_tt = ttnn.from_torch(
        x.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # Mirror what vLLM hands forward_prefill_paged during trace-capture warmup.
    #   * vLLM's trace path builds a single set of rot_mats sized to args.max_seq_len
    #     (tt_transformers/tt/model.py:337, `slice_end = max_seq_len if trace_enabled`)
    #     and reuses it across every captured prefill length — that way one trace
    #     covers every seq_len bucket without re-baking cos/sin.
    #   * That path also bypasses Qwen35Model.prefill_forward_text (which per-chunk-
    #     slices cos_full at model.py:516); it goes ttnn_prefill_forward → parent
    #     forward → layer loop, so attention receives the full max_seq_len matrices
    #     directly.
    cos_tt, sin_tt = get_prefill_rot_mats(rope, max_seq_len, mesh_device)

    if paged_attention:
        # Paged path (vLLM). chunk_page_table = page_table_tt because this is
        # a single-chunk prefill; prefill_layer_chunked only slices
        # chunk_page_table to a sub-range when the chunk doesn't span the
        # whole prompt (model.py:514-524).
        out_tt = attn.forward(
            x_tt,
            current_pos=None,
            rot_mats=[cos_tt, sin_tt],
            mode="prefill",
            page_table=page_table_tt,
            chunk_page_table=page_table_tt,
            chunk_start_idx=0,
            kv_cache=kv_cache,
            user_id=0,
        )
    else:
        # Non-paged: forward_prefill builds its own cos/sin internally,
        # so rot_mats=None is fine here. kv_cache=None routes the
        # dispatcher (attention.py:132-145) to forward_prefill instead
        # of forward_prefill_paged.
        out_tt = attn.forward(
            x_tt,
            current_pos=None,
            rot_mats=None,
            mode="prefill",
            kv_cache=None,
        )

    # _all_reduce reduce-scatters along dim=3 (attention.py:321-330), so the
    # multi-device output is fractured (dim/num_devices per device); concat
    # along dim=3 reassembles the full output dim.
    if mesh_device.get_num_devices() > 1:
        out_tt = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        out_tt = ttnn.to_torch(out_tt)

    passing, msg = comp_pcc(ref_out, out_tt, pcc=pcc)
    logger.info(f"TT attention outputs match HF reference: {msg}")
    assert passing, msg

    if paged_attention:

        def unpage(c):
            pt = ttnn.to_torch(ttnn.get_device_tensors(page_table_tt)[0]).flatten().tolist()
            return torch.cat([c[b] for b in pt], dim=-2)[None, :, :seq_len]  # [1, num_heads, seq_len, head_dim]

        # reference: [batch, num_heads, seq_len, head_dim]
        k_ref, v_ref = ref.expected_kv()

        # tt cache: [max_num_blocks, num_heads, block_size, head_dim]
        k_cache = ttnn.to_torch(kv_cache[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))
        v_cache = ttnn.to_torch(kv_cache[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1))

        # unpage to extract exact k, v tensors up to seq_len
        k_tt, v_tt = unpage(k_cache), unpage(v_cache)

        ok_k, msg_k = comp_pcc(k_ref, k_tt, pcc=pcc)
        ok_v, msg_v = comp_pcc(v_ref, v_tt, pcc=pcc)
        logger.info(f"K cache: {msg_k}\nV cache: {msg_v}")
        assert ok_k and ok_v, f"k={msg_k}, v={msg_v}"
