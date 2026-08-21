# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Small-4-119B MoE FFN CPU reference (plain routed MoE, no device dispatch/combine).

Mistral's router is *not* the DeepSeek/GLM ``noaux_tc`` gate, and no pre-existing reference in this
repo can express it:

* HF ``Mistral4TopkRouter`` owns only ``weight`` — there is **no** ``e_score_correction_bias``.
  ``Mistral4MoE.route_tokens_to_experts`` scores with ``router_logits.softmax(-1)``, takes top-4,
  renormalizes (``norm_topk_prob``) and multiplies by ``routed_scaling_factor`` (1.0).
* ``reference.modeling_deepseek.MoEGate`` — used by ``glm_moe_reference`` and by
  ``reference.tt.moe.moe`` — hard-raises ``NotImplementedError`` for anything but
  ``scoring_func="sigmoid"`` / ``topk_method="noaux_tc"``, and its ``noaux_tc`` branch *requires*
  the correction bias.
* The shared expert is **added** to the routed sum (``modeling_mistral4.Mistral4MoE.forward``), and
  ``moe_intermediate_size`` (2048) is used for both the routed and the shared experts.

So the routing here is HF's own code: ``mistral4_route_tokens_to_experts`` calls
``Mistral4MoE.route_tokens_to_experts`` unbound, on a tiny attribute shim, and the logits come from a
real ``Mistral4TopkRouter``. Nothing in this module re-implements the router math — a hand-written
copy would be the single most likely place for a silent divergence, since with no correction bias the
top-4 selection is decided purely by the logits.

With ``n_group = topk_group = 1`` the group-limited-routing block inside
``route_tokens_to_experts`` is a provable no-op (the single group is always selected, so the mask is
all-ones); it is *not* load-bearing for this model. ``topk(..., sorted=False)`` order is likewise
irrelevant because ``norm_topk_prob`` divides by the sum.

Expert application mirrors ``glm_moe_reference``: ``TorchExpert`` (verified identical to HF
``Mistral4MLP.forward``) instantiated lazily per hit expert, weighted and summed with the shared
expert. That is the mathematical MoE result the distributed ``TtMoe`` (dispatch -> experts -> combine)
computes, so it composes into the block reference without replicating the dispatch machinery.
"""

from types import SimpleNamespace

import torch
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE, Mistral4TopkRouter

from models.demos.deepseek_v3_d_p.reference.mistral_small_4_119b_config import Mistral4Small119BConfig
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert

# Fields shared between the DeepSeek-named namespace that ``mistral4_hf_config()`` returns and the
# real ``Mistral4Config`` constructor. Deliberately excluded:
#   qk_head_dim / head_dim  -- Mistral4Config.__post_init__ derives them
#   max_seq_len             -- a device-path field, not an HF one
#   rope_theta / rope_scaling / quantization_config -- see mistral4_torch_config's docstring
_SHARED_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "n_shared_experts",
    "n_routed_experts",
    "routed_scaling_factor",
    "kv_lora_rank",
    "q_lora_rank",
    "qk_rope_head_dim",
    "v_head_dim",
    "qk_nope_head_dim",
    "n_group",
    "topk_group",
    "num_experts_per_tok",
    "first_k_dense_replace",
    "norm_topk_prob",
    "hidden_act",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
    "pretraining_tp",
    "rope_interleave",
    "attention_bias",
    "attention_dropout",
)


def mistral4_torch_config(config=None, **overrides) -> Mistral4Config:
    """Build a real ``Mistral4Config`` — the thing HF's own modules need — from our namespace config.

    ``mistral4_hf_config()`` returns a DeepSeek-*named* ``SimpleNamespace`` (``rope_scaling`` instead
    of ``rope_parameters``, top-level ``rope_theta``, zeroed ``mscale``) because that is what ttMLA and
    the vendored DeepSeek reference read. HF's ``Mistral4*`` classes read the upstream names, and
    ``Mistral4MoE`` additionally needs ``num_local_experts`` — which only exists via
    ``Mistral4Config.attribute_map``, so a namespace can never supply it. Hence this translation.

    Verified: bare ``Mistral4Config()`` defaults are *exactly* this checkpoint's ``text_config``
    (hidden 4096, moe_intermediate 2048, 36 layers, 128 routed / 1 shared expert, top-4, n_group =
    topk_group = 1, norm_topk_prob, routed_scaling_factor 1.0, kv_lora 256, q_lora 1024, 64/64/128
    head dims, rms_eps 1e-6, rope_parameters {yarn, theta 10000, factor 128, orig_max 8192, beta
    32/1, mscale 1.0, llama_4_scaling_beta 0.1}). So ``mistral4_torch_config()`` with no arguments is
    the production model, and ``config``/``overrides`` exist only for shrunken test configs.

    ``rope_parameters`` is intentionally left to ``__post_init__``: it reproduces the checkpoint
    verbatim, including the ``mscale = 1.0`` that ``mistral4_hf_config`` deliberately zeroes. That
    zeroing is a device/reference softmax-scale convention (see
    ``reference/mistral_small_4_119b_config.py``); it does not belong in an upstream HF config, and
    HF's ``Mistral4Attention`` ignores mscale anyway (``self.scaling = qk_head_dim ** -0.5``).

    Args:
        config: optional namespace/config to copy the shared fields off (e.g. ``mistral4_hf_config()``,
            or a test config with a shrunken expert count).
        **overrides: ``Mistral4Config`` kwargs applied last.
    """
    kwargs = {}
    if config is not None:
        for field in _SHARED_CONFIG_FIELDS:
            if hasattr(config, field):
                kwargs[field] = getattr(config, field)
    kwargs.update(overrides)
    return Mistral4Config(**kwargs)


def unpack_stacked_expert_weights(
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    moe_intermediate_size: int | None = None,
) -> list[dict]:
    """Split Mistral's packed routed-expert tensors into per-expert ``TorchExpert`` weight dicts.

    The checkpoint stacks the routed experts: ``mlp.experts.gate_up_proj`` is one
    ``[n_experts, 2 * moe_intermediate, hidden]`` tensor (``[128, 4096, 4096]`` for this model) and
    ``mlp.experts.down_proj`` is ``[n_experts, hidden, moe_intermediate]``. ``Mistral4NaiveMoe.forward``
    consumes them as ``F.linear(x, gate_up_proj[e]).chunk(2, dim=-1)`` -> ``(gate, up)``; since
    ``F.linear`` weights are ``[out, in]`` and ``chunk`` splits the *output* dim in order, the first
    half of dim 1 is gate and the second half is up:

        gate_proj[e] = gate_up_proj[e][:moe_intermediate, :]   # [moe_intermediate, hidden]
        up_proj[e]   = gate_up_proj[e][moe_intermediate:, :]   # [moe_intermediate, hidden]
        down_proj[e] = down_proj[e]                            # [hidden, moe_intermediate]

    which is already the ``[out, in]`` convention ``TorchExpert(torch_weights=...)`` expects.

    fp8 checkpoints must be dequantized first: this model is per-tensor fp8 (``weight_block_size``
    null), so it goes through ``utils.test_utils.is_per_tensor_fp8`` ->
    ``_dequantize_per_tensor_fp8_state_dict``, whose plain broadcast already handles the stacked
    experts' ``[n_experts, 1, 1]`` scale. Passing raw ``float8_e4m3fn`` tensors here is rejected.
    """
    if gate_up_proj.dtype == torch.float8_e4m3fn or down_proj.dtype == torch.float8_e4m3fn:
        raise ValueError(
            "packed expert tensors are still fp8; dequantize first "
            "(utils.test_utils.dequantize_state_dict -> per-tensor path)"
        )
    if gate_up_proj.ndim != 3 or down_proj.ndim != 3:
        raise ValueError(f"expected 3-D stacked experts, got {tuple(gate_up_proj.shape)} / {tuple(down_proj.shape)}")
    n_experts = gate_up_proj.shape[0]
    if down_proj.shape[0] != n_experts:
        raise ValueError(f"expert-count mismatch: {n_experts} vs {down_proj.shape[0]}")
    if moe_intermediate_size is None:
        moe_intermediate_size = gate_up_proj.shape[1] // 2
    if gate_up_proj.shape[1] != 2 * moe_intermediate_size:
        raise ValueError(
            f"gate_up_proj dim1 {gate_up_proj.shape[1]} != 2 * moe_intermediate_size {moe_intermediate_size}"
        )
    return [
        {
            "gate_proj": gate_up_proj[e, :moe_intermediate_size, :],
            "up_proj": gate_up_proj[e, moe_intermediate_size:, :],
            "down_proj": down_proj[e],
        }
        for e in range(n_experts)
    ]


def mistral4_router_logits(hidden_states: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """Router logits via HF's own ``Mistral4TopkRouter`` (``F.linear(x.view(-1, H), weight)``).

    ``gate_weight`` is used at bf16 *value* with fp32 *compute*, matching the device on the device's
    own authority: ``TtMoEGatePrefill._convert_and_cache_gate_weights`` uploads the gate weight as
    ``dtype=ttnn.bfloat16`` unconditionally (``tt/moe/tt_moe_gate_prefill.py:220``), while the
    adapter's ``default_gate_mode = "DEVICE_FP32"`` only typecasts the *logits* to fp32
    (``:833``). This differs from ``glm_5_1/moe.py``, which rounds only the correction *bias* and
    leaves the weight as handed in — Mistral has no correction bias, so the top-4 choice out of 128
    rides entirely on these logits, and the weight is the tensor whose precision decides near-ties.
    In practice the cast is a no-op for every current caller (the fp8 dequantizer already emits bf16);
    it is here so an fp32-valued weight cannot silently route differently than the device.
    """
    n_routed, hidden = gate_weight.shape
    router = Mistral4TopkRouter(SimpleNamespace(n_routed_experts=n_routed, hidden_size=hidden))
    with torch.no_grad():
        router.weight.copy_(gate_weight.to(torch.bfloat16).float())
        return router(hidden_states.reshape(-1, hidden).float())


def mistral4_route_tokens_to_experts(
    router_logits: torch.Tensor,
    *,
    num_experts_per_tok: int,
    n_group: int = Mistral4Small119BConfig.NUM_EXPERT_GROUPS,
    topk_group: int = Mistral4Small119BConfig.NUM_LIMITED_GROUPS,
    norm_topk_prob: bool = Mistral4Small119BConfig.NORM_TOPK_PROB,
    routed_scaling_factor: float = Mistral4Small119BConfig.ROUTE_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax top-k routing, executed by HF's ``Mistral4MoE.route_tokens_to_experts`` itself.

    Called unbound on an attribute shim: the method reads only the scalars below (no parameters, no
    submodules), so this is the exact upstream code object with zero weight allocation — a full-size
    ``Mistral4MoE`` would allocate a ``[128, 4096, 4096]`` ``gate_up_proj`` (~8.6 GB fp32) that the
    routing does not touch.

    Returns ``(topk_indices [tokens, k], topk_weights [tokens, k])``.
    """
    shim = SimpleNamespace(
        n_routed_experts=router_logits.shape[-1],
        n_group=n_group,
        topk_group=topk_group,
        top_k=num_experts_per_tok,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )
    return Mistral4MoE.route_tokens_to_experts(shim, router_logits)


def mistral4_moe_reference(
    hidden_states: torch.Tensor,
    *,
    gate_weights: dict,
    routed_expert_weights: list[dict],
    shared_expert_weights: dict,
    emb_dim: int,
    num_experts_per_tok: int = Mistral4Small119BConfig.NUM_EXPERTS_PER_TOKEN,
    n_group: int = Mistral4Small119BConfig.NUM_EXPERT_GROUPS,
    topk_group: int = Mistral4Small119BConfig.NUM_LIMITED_GROUPS,
    norm_topk_prob: bool = Mistral4Small119BConfig.NORM_TOPK_PROB,
    routed_scaling_factor: float = Mistral4Small119BConfig.ROUTE_SCALE,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Plain routed-MoE forward. ``hidden_states`` [1, seq, hidden] -> [1, seq, hidden].

    Args:
        gate_weights: ``{"weight": [n_routed, hidden]}``. Any ``e_score_correction_bias`` entry is
            ignored — Mistral's router has none (a stray one would mean the weights came from a
            DeepSeek-shaped loader).
        routed_expert_weights: per-expert ``{"gate_proj","up_proj","down_proj"}``, ``[out, in]`` each.
            Use ``unpack_stacked_expert_weights`` for a Mistral checkpoint's packed tensors.
        shared_expert_weights: same shape; its output is **added**, not gated.
        emb_dim: hidden size.
        compute_dtype: dtype the experts run in. bf16 by default to match the device; the accumulation
            is fp32 regardless. Pass ``torch.float32`` for an exact cross-check against HF.

    Returns:
        [1, seq, hidden] in ``hidden_states.dtype``.
    """
    b, s, h = hidden_states.shape
    if h != emb_dim:
        raise ValueError(f"hidden_states last dim {h} != emb_dim {emb_dim}")
    flat = hidden_states.reshape(-1, h)
    n_routed = len(routed_expert_weights)
    if gate_weights["weight"].shape[0] != n_routed:
        raise ValueError(f"gate weight has {gate_weights['weight'].shape[0]} experts, got {n_routed} weight dicts")

    router_logits = mistral4_router_logits(hidden_states, gate_weights["weight"])
    topk_idx, topk_weight = mistral4_route_tokens_to_experts(
        router_logits,
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
    )

    routed_hidden = routed_expert_weights[0]["gate_proj"].shape[0]
    out = torch.zeros_like(flat, dtype=torch.float32)
    for e in range(n_routed):
        sel = topk_idx == e  # [tokens, top_k]
        if not bool(sel.any()):
            continue
        tok, slot = sel.nonzero(as_tuple=True)
        expert = TorchExpert(emb_dim, routed_hidden, torch_weights=routed_expert_weights[e]).eval().to(compute_dtype)
        with torch.no_grad():
            ex_out = expert(flat[tok].to(compute_dtype))
        out[tok] += topk_weight[tok, slot].unsqueeze(-1).float() * ex_out.float()

    # Shared expert: added unconditionally (Mistral4MoE.forward), intermediate = moe_intermediate_size
    # * n_shared_experts, which is 2048 * 1 here.
    shared_hidden = shared_expert_weights["gate_proj"].shape[0]
    shared = TorchExpert(emb_dim, shared_hidden, torch_weights=shared_expert_weights).eval().to(compute_dtype)
    with torch.no_grad():
        out += shared(flat.to(compute_dtype)).float()

    return out.reshape(b, s, h).to(hidden_states.dtype)
