# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer vLLM KV cache specs for Qwen3.6 / Qwen3.8 (QWEN36_MAMBA_SPEC=1).

One FullAttentionSpec per full-attention layer and one MambaSpec per GDN layer, with the GDN page padded to the
attention page. vLLM then puts every layer into groups of equal page size, and block id b names the same bytes in
the attention group and in each GDN group. No ttnn import: host tests build the specs with vLLM alone.
"""

import torch

# BFLOAT8_B stores a 32x32 tile as 1024 one-byte mantissas plus 64 shared exponents (one per 16 values): 17 bytes per
# 16 values.
BF8_BYTES_PER_16 = 17


def _cfg(tc, name, default=None):
    return tc.get(name, default) if isinstance(tc, dict) else getattr(tc, name, default)


def bf8_bytes(values):
    """Bytes of ``values`` BFLOAT8_B values (a multiple of 16). Per token this is an average: a KV page of whole tiles
    (block_size and head_dim multiples of 32) holds exactly block_size times this."""
    assert values % 16 == 0, f"{values} BF8 values do not fill whole exponent groups"
    return values // 16 * BF8_BYTES_PER_16


def gdn_layer_name(i):
    """vLLM's name of GDN layer i (its Qwen3.5 model's module name): the key of its MambaSpec and its entry in the
    layer_names of its group in the plugin's mamba_state."""
    return f"model.layers.{i}.linear_attn"


def qwen36_layer_names(text_config):
    """(attention layer names, GDN layer names) in layer order, named as vLLM's Qwen3.5 model names them."""
    n = int(_cfg(text_config, "num_hidden_layers"))
    types = _cfg(text_config, "layer_types", None) or ["linear_attention"] * 3 + ["full_attention"]
    attn, gdn = [], []
    for i in range(n):
        if types[i % len(types)] == "full_attention":
            attn.append(f"model.layers.{i}.self_attn")
        else:
            gdn.append(gdn_layer_name(i))
    return attn, gdn


def qwen36_kv_cache_spec(
    text_config,
    tp,
    block_size,
    kv_bf8=True,
    num_speculative_blocks=0,
    mamba_cache_mode="none",
    mamba_block_size=None,
    prefill_checkpoint_alignment=None,
):
    """dict[layer name, KVCacheSpec] for vLLM's hybrid KV cache manager. All sizes are per device at ``tp`` devices.

    Attention: ``num_kv_heads`` is the model total (the plugin divides it by the device count for the cache shape);
    ``num_head_slots`` is the per-device head count, so ``page_size_bytes`` is the per-device page. With ``kv_bf8``
    the page is stated in bytes (``state_content_bytes``, K and V of one token and head), because BFLOAT8_B has no
    torch dtype.
    GDN: conv history (kernel - 1, conv_dim / tp) bf16 and recurrent state (value_heads / tp, key_dim, value_dim) fp32,
    padded to the attention page. ``num_speculative_blocks`` = the MTP draft length K (state blocks per request
    beyond the running one). ``mamba_block_size`` (default ``block_size``): tokens per GDN state block; mode "none"
    needs at least max_model_len so a request holds one state block per group. ``prefill_checkpoint_alignment``
    (None: no checkpoints): one prefill checkpoint block per GDN group, placed only at offsets from the chunk start
    that are multiples of it (mode align; the prefill must write the state there)."""
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    tp = int(tp)
    dim = int(_cfg(text_config, "hidden_size"))
    n_heads = int(_cfg(text_config, "num_attention_heads"))
    n_kv = int(_cfg(text_config, "num_key_value_heads"))
    head_dim = int(_cfg(text_config, "head_dim", None) or dim // n_heads)
    nk = int(_cfg(text_config, "linear_num_key_heads", 16))
    nv = int(_cfg(text_config, "linear_num_value_heads", 32))
    dk = int(_cfg(text_config, "linear_key_head_dim", 128))
    dv = int(_cfg(text_config, "linear_value_head_dim", 128))
    kconv = int(_cfg(text_config, "linear_conv_kernel_dim", 4))
    assert nk % tp == 0 and nv % tp == 0, f"GDN heads ({nk} key, {nv} value) must divide by tp={tp}"
    n_local_kv = max(1, n_kv // tp)  # TPAttention replicates the KV heads when tp > n_kv

    attn_kw = {"state_content_bytes": bf8_bytes(2 * head_dim)} if kv_bf8 else {}
    attn = FullAttentionSpec(
        block_size=int(block_size),
        num_kv_heads=n_kv,
        num_head_slots=n_local_kv,
        head_size=head_dim,
        dtype=torch.bfloat16,
        **attn_kw,
    )
    conv_dim_tp = (2 * nk * dk + nv * dv) // tp
    gdn_bytes = (kconv - 1) * conv_dim_tp * 2 + (nv // tp) * dk * dv * 4
    if gdn_bytes > attn.page_size_bytes:
        raise ValueError(
            f"GDN state {gdn_bytes} B per layer does not fit the {attn.page_size_bytes} B attention page "
            f"(block_size={block_size}, tp={tp}, kv_bf8={kv_bf8}); use a larger block"
        )
    ckpt_kw = (
        {}
        if prefill_checkpoint_alignment is None
        else dict(num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=int(prefill_checkpoint_alignment))
    )
    gdn = MambaSpec(
        block_size=int(block_size if mamba_block_size is None else mamba_block_size),
        shapes=((kconv - 1, conv_dim_tp), (nv // tp, dk, dv)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=attn.page_size_bytes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=int(num_speculative_blocks),
        **ckpt_kw,
    )
    attn_names, gdn_names = qwen36_layer_names(text_config)
    return {**{name: attn for name in attn_names}, **{name: gdn for name in gdn_names}}
