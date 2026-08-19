# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Mistral Small 4 119B Model Configuration.

Single source of truth for model dimension constants.
Values from HuggingFace config.json for `mistralai/Mistral-Small-4-119B-2603`
(the ``text_config`` block; the checkpoint is a ``Mistral3ForConditionalGeneration``
wrapper carrying a Pixtral ``vision_config`` that this text-only prefill path ignores).

Architecture summary: **DeepSeek-family MLA attention with GPT-OSS-family MoE routing.**
The attention weight names are DeepSeek's exactly (``q_a_proj`` / ``q_a_layernorm`` /
``q_b_proj`` / ``kv_a_proj_with_mqa`` / ``kv_a_layernorm`` / ``kv_b_proj`` / ``o_proj``),
but the router is a plain softmax top-4 with no correction bias and no expert groups,
and the expert weights ship as ONE stacked fp8 tensor per projection with gate and up
fused, not as per-expert ``mlp.experts.{i}.*`` entries.
"""


class MistralSmall4Config:
    """Mistral Small 4 119B model dimensions."""

    # Core dimensions
    EMB_SIZE = 4096  # embedding dimension (hidden_size)
    FABRIC_PAYLOAD_SIZE = EMB_SIZE  # max fabric packet payload; must stay in sync with migration code
    MOE_INTERMEDIATE_SIZE = 2048  # MoE FFN hidden dimension
    INTERMEDIATE_SIZE = 12288  # shared-expert / dense FFN hidden dimension

    # MoE configuration
    NUM_ROUTED_EXPERTS = 128
    NUM_EXPERTS_PER_TOKEN = 4
    NUM_SHARED_EXPERTS = 1
    # n_group / topk_group are both 1: no expert-group routing, the gate collapses to plain top-k.
    NUM_EXPERT_GROUPS = 1
    NUM_LIMITED_GROUPS = 1
    # routed_scaling_factor = 1.0 (DeepSeek-V3 uses 2.5, GLM 2.5, V4-Flash 1.5).
    ROUTE_SCALE = 1.0
    # ⚠ NOT USED by the gate mode this model must run under, and deliberately left at the family
    # default rather than set to "softmax". The grouped-topk / hash-gate kernels accept ONLY
    # "sigmoid" or "sqrtsoftplus" (moe_grouped_topk.cpp:17-23 TT_THROWs on anything else) — there
    # is no softmax score_func. Mistral's router (softmax over all experts -> top-4 -> renormalize,
    # `norm_topk_prob: true`) is mathematically IDENTICAL to GPT-OSS routing (top-k on raw logits ->
    # softmax over just the selected k), because softmax is monotonic so the top-k indices agree and
    # the renormalization is exactly the softmax over the selected subset. So Mistral runs
    # GateComputeMode.GPT_DEVICE, where score_func is never read.
    # If someone runs this model under DEVICE / DEVICE_FP32 instead, they get a SIGMOID router
    # affinity silently — wrong numbers, not a crash. That is the tripwire this comment exists for.
    SCORE_FUNC = "sigmoid"

    # Model architecture
    NUM_LAYERS = 36
    # first_k_dense_replace = 0 -> there are NO leading dense-FFN layers. Every one of the 36 layers
    # is MoE. DeepSeek-V3 and GLM-5.1 both have 3; this is the first resident with zero.
    NUM_DENSE_LAYERS = 0
    VOCAB_SIZE = 131072
    MAX_POSITION_EMBEDDINGS = 1048576  # 1M context

    # MLA dimensions
    NUM_ATTENTION_HEADS = 32
    NUM_KEY_VALUE_HEADS = 32
    Q_LORA_RANK = 1024
    # ⚠ 256 is unprecedented in this family (DeepSeek-V3 / Kimi / GLM all use 512). It makes the
    # packed-FP8 KV cache's rope_offset_bytes = 264, which is not 16-byte aligned and fails
    # MlaKvCacheFormat.SCALED_FP8's validate_scaled(). Dense MLA's allocate_kv_cache does not select
    # a packed format by default, so this only bites when SCALED_FP8 is opted into.
    KV_LORA_RANK = 256
    QK_NOPE_HEAD_DIM = 64
    QK_ROPE_HEAD_DIM = 64
    QK_HEAD_DIM = 128  # = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
    V_HEAD_DIM = 128
    HEAD_DIM = 128

    # Other
    RMS_NORM_EPS = 1e-6
    ROPE_THETA = 10000.0
    INITIALIZER_RANGE = 0.02


def mistral4_hf_config(max_seq: int = 8192):
    """The HF-attribute config both the device path and the CPU reference read.

    Returns a **real** ``Mistral4Config`` (transformers >= 5.12 ships ``mistral4`` natively), built
    explicitly from ``MistralSmall4Config`` so this module stays the single source of truth for the
    dims -- a transformers upgrade cannot silently move the model out from under us.

    Why a real config rather than a ``SimpleNamespace`` like GLM's builder:

      * ``PreTrainedConfig`` already aliases ``rope_scaling`` -> ``rope_parameters``, so the
        transformers 5.x rename resolves itself. The dict it returns is a *superset* of the six keys
        ``tt/mla/rope.py`` indexes (factor, original_max_position_embeddings, beta_fast, beta_slow,
        mscale, mscale_all_dim), so no remapping is needed.
      * It is the config the HF reference model (``Mistral4Model`` / ``Mistral4Attention`` /
        ``Mistral4MoE``) requires. A namespace cannot drive those, and without them there is no
        reference to take PCC against -- and no random-weight transformer run at all, since
        ``create_hf_model`` builds the TT state_dict from an HF model instance.
      * A namespace silently omits whatever nobody thought to list. ``attention_dropout`` and
        ``initializer_range`` were both found missing only by hitting AttributeError at runtime.

    Two fields still have to be attached by hand, because no HF config carries them:

      * ``rope_theta`` -- Mistral nests it INSIDE ``rope_parameters``; ``tt/mla/rope.py`` reads it as
        a top-level attribute. This is a hoist, not a rename.
      * ``max_seq_len`` -- a convention of this model family.

    Plus ``quantization_config``, which lives at the TOP level of Mistral's ``config.json`` (next to
    the vision tower), not inside ``text_config``, so constructing from the text config alone drops
    it. The honest per-tensor value is passed through; see the note at the bottom.
    """
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    C = MistralSmall4Config
    cfg = Mistral4Config(
        vocab_size=C.VOCAB_SIZE,
        hidden_size=C.EMB_SIZE,
        intermediate_size=C.INTERMEDIATE_SIZE,
        moe_intermediate_size=C.MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=C.NUM_LAYERS,
        num_attention_heads=C.NUM_ATTENTION_HEADS,
        num_key_value_heads=C.NUM_KEY_VALUE_HEADS,
        n_shared_experts=C.NUM_SHARED_EXPERTS,
        n_routed_experts=C.NUM_ROUTED_EXPERTS,
        routed_scaling_factor=C.ROUTE_SCALE,
        kv_lora_rank=C.KV_LORA_RANK,
        q_lora_rank=C.Q_LORA_RANK,
        qk_rope_head_dim=C.QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=C.QK_NOPE_HEAD_DIM,
        v_head_dim=C.V_HEAD_DIM,
        n_group=C.NUM_EXPERT_GROUPS,
        topk_group=C.NUM_LIMITED_GROUPS,
        num_experts_per_tok=C.NUM_EXPERTS_PER_TOKEN,
        first_k_dense_replace=C.NUM_DENSE_LAYERS,
        norm_topk_prob=True,
        max_position_embeddings=C.MAX_POSITION_EMBEDDINGS,
        initializer_range=C.INITIALIZER_RANGE,
        rms_norm_eps=C.RMS_NORM_EPS,
        attention_bias=False,
        attention_dropout=0.0,
        rope_interleave=True,
    )

    # The normalizations AND the invariant guards both live in the shared normalizer, so a config
    # arriving via AutoConfig gets checked exactly as strictly as a hand-built one.
    return normalize_mistral4_config(cfg, max_seq=max_seq)


def normalize_mistral4_config(cfg, max_seq: int | None = None):
    """Apply the post-construction fixups a `Mistral4Config` needs before the TT/reference paths
    can use it. Mutates and returns `cfg`.

    **This must be applied to a config loaded by `AutoConfig` too, not just to a hand-built one.**
    There are two config paths in the test suite and they had silently diverged: `config_only` ->
    `_resolve_config_only` -> the builder below (which applied all of this) versus `hf_config` ->
    `_resolve_hf_config` -> `AutoConfig.from_pretrained` (which applied none of it). The chunked MLA
    test reaches the second one via `pretrained_transformer_weights`, so it failed on the missing
    `rope_theta` -- and would then have run with the YaRN mscale bug below, which does NOT crash and
    just produces a wrong softmax temperature. Factored out here so both paths get the same config.

    `max_seq` is optional because the runner/`run_model` overwrites `max_seq_len` anyway; pass it
    when building fresh so the rope config is consistent from the start.
    """
    # rope_parameters is populated by Mistral4Config.__post_init__ with the checkpoint's YaRN block.
    # transformers 5.x keeps rope_theta only in there; the rest of the code reads cfg.rope_theta.
    if getattr(cfg, "rope_theta", None) is None:
        cfg.rope_theta = float(cfg.rope_parameters["rope_theta"])
    if max_seq is not None:
        cfg.max_seq_len = max_seq

    # Mistral carries a full YaRN block (factor=128, mscale=1, mscale_all_dim=1) but applies NO
    # mscale amplitude anywhere: `Mistral4Attention.__init__` sets `self.scaling = qk_head_dim**-0.5`
    # unconditionally, and `Mistral4RotaryEmbedding.attention_scaling` is 1.0 so nothing is baked
    # into cos/sin either. DeepSeek folds mscale^2 into the softmax scale under exactly these config
    # values, which at factor=128 is a 2.2058x multiplier on the attention logits -- a wrong softmax
    # temperature, with no crash to reveal it. This flag turns that off in both tt/mla/mla.py and the
    # CPU MLAReference. Verified against transformers 5.12's mistral4 implementation.
    cfg.mla_disable_yarn_mscale = True

    # --- quantization ---
    # Deliberately NOT the [128,128] block shape every other resident carries. Mistral ships
    # PER-TENSOR fp8: `weight_block_size` is null in config.json, dense weights carry a rank-0 scalar
    # `*_scale_inv`, and the stacked expert tensors carry `[128,1,1]` (one scale per expert). The
    # shared dequantizer (deepseek_v3/utils/hf_model_utils.py) asserts `tensor.ndim == inv_scale.ndim`
    # and `len(block_shape) == tensor.ndim`, so it RAISES on both of Mistral's shapes rather than
    # silently mis-scaling. The honest value is passed through so the per-tensor path in
    # utils/test_utils.py (`is_per_tensor_fp8`) is the one selected, and anything else still fails loud.
    # Only when absent: a config loaded from the real checkpoint already carries the genuine block
    # (with `modules_to_not_convert`, `dequantize`, ...), and overwriting it with this reconstruction
    # would throw that away. Verified equivalent on the real checkpoint: `weight_block_size` is null
    # there too.
    if getattr(cfg, "quantization_config", None) is None:
        cfg.quantization_config = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "static",
            "weight_block_size": None,
        }

    # Guard the invariants the device path depends on, so a transformers change fails here loudly
    # rather than as a PCC drift 20 minutes into a galaxy run.
    C = MistralSmall4Config
    assert cfg.qk_head_dim == C.QK_HEAD_DIM, f"qk_head_dim {cfg.qk_head_dim} != {C.QK_HEAD_DIM}"
    assert cfg.head_dim == C.HEAD_DIM, f"head_dim {cfg.head_dim} != {C.HEAD_DIM}"
    assert cfg.num_local_experts == C.NUM_ROUTED_EXPERTS  # attribute_map alias used by Mistral4NaiveMoe
    for key in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
        assert key in cfg.rope_scaling, f"rope_scaling missing {key!r} that tt/mla/rope.py indexes"
    return cfg
