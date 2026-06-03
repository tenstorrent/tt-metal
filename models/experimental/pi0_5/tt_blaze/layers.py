# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tt-blaze FusedOp stubs for pi0.5.

Four FusedOp classes, one per pi0.5 layer family:

    SiglipEncoderLayer   one SigLIP-27 vision encoder layer
    VlmDecoderLayer      one PaliGemma (Gemma-2B) decoder layer
    ExpertDecoderLayer   one action-expert layer (adaRMS + joint attn vs VLM KV)
    SuffixMlp            time/action proj at the head and tail of the expert tower

Each subclasses FusedOp, declares Input/Output sockets matching the dtype map
in PI0_5_GALAXY_DEPLOYMENT_PLAN.md §1, and exposes the same
ReceiverSocket → ... → SenderSocket + CrossDeviceSignal skeleton as
`blaze/ops/dense_layer/op.py`.

Internal sub-op composition is sketched in comment blocks with
`# TODO(blaze):` markers — body fillers should mirror dense_layer.op
(blaze/ops/dense_layer/op.py) and llama31_decoder_layer.op
(closest pre-existing analog to VlmDecoderLayer).
"""

from __future__ import annotations

# tt-blaze imports — fall back to stubs so the scaffold parses without it.
try:  # pragma: no cover
    import ttnn
    from blaze.blaze_op import BlazeOp, FusedOp, Input, Output
    from blaze.socket_config import ReceiverSocketConfig, SenderSocketConfig
    from blaze.ops.receiver_socket import ReceiverSocket
    from blaze.ops.sender_socket import SenderSocket
    from blaze.ops.cross_device_signal import CrossDeviceSignal
    from blaze.ops.cb_reconfig.op import CbReconfig
    from blaze.ops.mcast import McastGridConfig
except ImportError:  # pragma: no cover

    class FusedOp:  # type: ignore[no-redef]
        name: str = ""

    class Input:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            ...

    class Output:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            ...

    class BlazeOp:  # type: ignore[no-redef]
        @staticmethod
        def child_prefix(p, c):
            return f"{p}/{c}"

    class ReceiverSocketConfig:
        pass  # type: ignore[no-redef]

    class SenderSocketConfig:
        pass  # type: ignore[no-redef]

    class _StubOp:
        @staticmethod
        def emit(*a, **kw):
            return None

    ReceiverSocket = SenderSocket = CrossDeviceSignal = CbReconfig = _StubOp  # type: ignore
    McastGridConfig = object  # type: ignore
    ttnn = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
# Helper — common compose() boilerplate
# ──────────────────────────────────────────────────────────────────────────


def _common_compose(cls, f, tensors, output, user_args, tensor_keys, extra_ua_keys=()):
    ua = user_args or {}
    kwargs = {k: tensors[k] for k in tensor_keys}
    kwargs["output_tensor"] = output
    kwargs["prefix"] = ua.get("prefix", cls.name)
    kwargs["layer_idx"] = ua.get("layer_idx", 0)
    kwargs["tp_size"] = ua.get("tp_size", 8)
    kwargs["mcast_grid_configs"] = ua.get("mcast_grid_configs")
    kwargs["receiver_socket_config"] = ua.get("receiver_socket_config")
    kwargs["sender_socket_config"] = ua.get("sender_socket_config")
    kwargs["execute_socket_logic"] = ua.get("execute_socket_logic", False)
    for k in extra_ua_keys:
        kwargs[k] = ua.get(k)
    cls.emit(f, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Stage 0 — SigLIP encoder layer (one of 27)
# ──────────────────────────────────────────────────────────────────────────


class SiglipEncoderLayer(FusedOp):
    """One SigLIP-27 vision encoder layer.

    Shape contract (PI0_5_GALAXY_DEPLOYMENT_PLAN.md §1.1):
      hidden=1152, mlp_intermediate=4304, num_heads=16, head_dim=72 (pad→96),
      seq_len=256 patches, bs=2 (base cam + wrist cam — BS-attn reshape).
    Per-layer weights ~20.7 MB (attn q/k/v/o bf16, mlp fc1/fc2 bf8_b).

    Parallelism:
      TP=8 mesh_axis=0; attn head-parallel (16 heads / 8 = 2/chip);
      MLP intermediate-dim parallel (4304/8 ≈ 538 / chip, padded).
      CP=1.
    Padding:
      receiver/sender both unpadded (1, 2, 256, 1152); no padding bytes.
    Metadata trailer:
      Vision is a one-shot forward, not a decode loop — named
      position_id/slot_id/token_id slots unused.
      Reserved[0] = `layer_idx` (0..26), selects which weight set per
      chained sub-emit.
    Persistent tensor table:
      | Tensor              | Dtype | Memory | Sharded along     |
      | q/k/v/o weight (×4) | bf16  | L1     | head-dim (TP=8)   |
      | q/k/v/o bias (×4)   | bf16  | L1     | replicated        |
      | fc1 weight          | bf8_b | L1     | intermediate(TP=8)|
      | fc1 bias            | bf16  | L1     | intermediate(TP=8)|
      | fc2 weight          | bf8_b | L1     | hidden-in (TP=8)  |
      | fc2 bias            | bf16  | L1     | replicated        |
      | pre/mlp/attn LN γ,β | bf16  | L1     | replicated        |
    Per-chip footprint at TP=8: ~2.4 MB / layer (weights) + ~0.05 MB
    (norms) + scratch. 27 layers chained on one loudbox → ~73 MB / chip.
    """

    name: str = "siglip_encoder_layer"

    # attention
    act: Input = Input()
    pre_attn_ln_gamma: Input = Input()
    pre_attn_ln_beta: Input = Input()
    q_weight: Input = Input()
    q_bias: Input = Input()
    k_weight: Input = Input()
    k_bias: Input = Input()
    v_weight: Input = Input()
    v_bias: Input = Input()
    o_weight: Input = Input()
    o_bias: Input = Input()
    # mlp
    pre_mlp_ln_gamma: Input = Input()
    pre_mlp_ln_beta: Input = Input()
    fc1_weight: Input = Input()
    fc1_bias: Input = Input()
    fc2_weight: Input = Input()
    fc2_bias: Input = Input()
    # tt-blaze plumbing
    metadata_persistent: Input = Input()
    out: Output = Output()

    _TENSOR_KEYS = (
        "act",
        "pre_attn_ln_gamma",
        "pre_attn_ln_beta",
        "q_weight",
        "q_bias",
        "k_weight",
        "k_bias",
        "v_weight",
        "v_bias",
        "o_weight",
        "o_bias",
        "pre_mlp_ln_gamma",
        "pre_mlp_ln_beta",
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
        "metadata_persistent",
    )

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        _common_compose(cls, f, tensors, output, user_args, cls._TENSOR_KEYS)

    @staticmethod
    def emit(
        f,
        *,
        act,
        pre_attn_ln_gamma,
        pre_attn_ln_beta,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        o_weight,
        o_bias,
        pre_mlp_ln_gamma,
        pre_mlp_ln_beta,
        fc1_weight,
        fc1_bias,
        fc2_weight,
        fc2_bias,
        metadata_persistent,
        output_tensor,
        prefix="siglip_encoder_layer",
        layer_idx=0,
        tp_size=8,
        mcast_grid_configs=None,
        receiver_socket_config: ReceiverSocketConfig = None,
        sender_socket_config: SenderSocketConfig = None,
        execute_socket_logic: bool = False,
    ):
        # Body sketch — mirror blaze/ops/dense_layer/op.py:244-366.
        #
        # 1. ReceiverSocket.emit() → (act_cb, metadata_cb) via
        #    get_activation_and_metadata_handles.
        # 2. LayerNorm pre-attn. TODO(blaze): SigLIP uses LayerNorm not
        #    RMSNorm — need a `layernorm` micro-op (gamma + beta + mean
        #    + var) or wrap rmsnorm with explicit mean-subtraction.
        # 3. Attention: QKV proj → BS-attention reshape (bs=2 × 256 →
        #    512-seq for matmul, split back before SDPA) → SDPA →
        #    O proj. The BS-attn reshape is in ttnn_siglip.py:
        #    Pi0_5SigLIPAttentionTTNN. head_dim=72 pads to 96 on
        #    tt-metal — TODO(blaze): confirm SDPA supports padded head
        #    dims.
        # 4. ResidualAdd.
        # 5. CbReconfig.
        # 6. LayerNorm pre-mlp.
        # 7. MLP intermediate-dim parallel TP=8: fc1 [1152→4304] fused
        #    GELU, fc2 [4304→1152]. all_reduce on fc2 output.
        # 8. ResidualAdd → SenderSocket → CrossDeviceSignal back-edge.
        return output_tensor


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 / 2 — VLM PaliGemma decoder layer (one of 18)
# ──────────────────────────────────────────────────────────────────────────


class VlmDecoderLayer(FusedOp):
    """One PaliGemma (Gemma-2B) decoder layer.

    Closest analog: blaze/ops/llama31_decoder_layer/op.py (GQA + dense
    MLP, no MoE). Same socket skeleton; sub-ops only differ in
    PaliGemma's sandwich-norm + post-MLP norm.

    Shape contract (§1.2):
      hidden=2048, mlp_intermediate=16384, num_q_heads=16, num_kv_heads=2,
      head_dim=128. Prefill seq up to 968 tokens (or 544 for LIBERO
      finetune). batch=1.
    Per-layer weights ~110 MB (all bf8_b except RMSNorm γ).
    Parallelism:
      TP=8 mesh_axis=0; attn head-parallel; MLP intermediate-dim
      parallel. KV REPLICATED across TP axis (num_kv_heads=2 doesn't
      divide 8) — mirrors option_b/tp_block.py:79.
    Padding: unpadded sockets (D2D-only path between stages).
    Metadata trailer: position_id / slot_id / token_id (named slots),
      reserved[0] = layer_idx (0..17).
    KV cache:
      Per layer per chip: 968 tokens × 256-dim-kv × 1 B (bf8) replicated
      = ~247 KB K + ~247 KB V = ~494 KB / layer. 9 layers/chip ≈ ~4.5
      MB. Stage-local during prefill, read by Stage 3 via the one-shot
      KV migration edge (mapping_notes §4).
    Per-chip total at TP=8: ~124 MB weights + 4.5 MB KV + ~5 MB scratch
      = ~134 MB / chip → fits 175 MB cap, ~40 MB headroom.
      The TP=8 shape shrinks the matmul kernel's static CB region ~7×
      (out_block_w 43 → ~6), so L1-resident weights don't collide with
      kernel CBs the way Option C did at single-chip shape — see
      OPTION_B_L1_ASSESSMENT.md.
    """

    name: str = "vlm_decoder_layer"

    act: Input = Input()
    pre_attn_norm_gamma: Input = Input()
    q_weight: Input = Input()
    k_weight: Input = Input()
    v_weight: Input = Input()
    o_weight: Input = Input()
    post_attn_norm_gamma: Input = Input()
    pre_mlp_norm_gamma: Input = Input()
    mlp_gate_weight: Input = Input()
    mlp_up_weight: Input = Input()
    mlp_down_weight: Input = Input()
    post_mlp_norm_gamma: Input = Input()
    # KV cache (stage-local; also read by Stage 3 via migration)
    k_cache: Input = Input()
    v_cache: Input = Input()
    # plumbing
    metadata_persistent: Input = Input()
    ar_attn_intermediate: Input = Input()
    ar_mlp_intermediate: Input = Input()
    out: Output = Output()

    _TENSOR_KEYS = (
        "act",
        "pre_attn_norm_gamma",
        "q_weight",
        "k_weight",
        "v_weight",
        "o_weight",
        "post_attn_norm_gamma",
        "pre_mlp_norm_gamma",
        "mlp_gate_weight",
        "mlp_up_weight",
        "mlp_down_weight",
        "post_mlp_norm_gamma",
        "k_cache",
        "v_cache",
        "metadata_persistent",
        "ar_attn_intermediate",
        "ar_mlp_intermediate",
    )

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        _common_compose(cls, f, tensors, output, user_args, cls._TENSOR_KEYS, extra_ua_keys=("kv_replicate",))

    @staticmethod
    def emit(
        f,
        *,
        act,
        pre_attn_norm_gamma,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        post_attn_norm_gamma,
        pre_mlp_norm_gamma,
        mlp_gate_weight,
        mlp_up_weight,
        mlp_down_weight,
        post_mlp_norm_gamma,
        k_cache,
        v_cache,
        metadata_persistent,
        ar_attn_intermediate,
        ar_mlp_intermediate,
        output_tensor,
        prefix="vlm_decoder_layer",
        layer_idx=0,
        tp_size=8,
        kv_replicate=True,
        mcast_grid_configs=None,
        receiver_socket_config: ReceiverSocketConfig = None,
        sender_socket_config: SenderSocketConfig = None,
        execute_socket_logic: bool = False,
    ):
        # Body sketch — closest existing analog is
        # blaze/ops/llama31_decoder_layer/op.py.
        #
        # 1. ReceiverSocket.emit() — standard pattern.
        # 2. GQA-like attention: pre_attn RMSNorm + GQAFused (q_heads=16,
        #    kv_heads=2, head_dim=128, kv_replicate=True) + post_attn
        #    RMSNorm. PaliGemma sandwich-norm has BOTH pre and post
        #    norms (unlike Llama 3.1 which only has pre).
        # 3. Residual add.
        # 4. CbReconfig.
        # 5. MLP: pre_mlp RMSNorm + SwiGLU(gate, up, down) + post_mlp
        #    RMSNorm. TODO(blaze): reuse matmul_swiglu or dense_swiglu;
        #    needs to thread the post-MLP norm.
        # 6. all_reduce(down output, cluster_axis=TP) — TP=8 sharded
        #    intermediate; collect partial sums.
        # 7. Residual add → SenderSocket → CrossDeviceSignal.
        return output_tensor


# ──────────────────────────────────────────────────────────────────────────
# Stage 3 — Expert decoder layer (one of 18)
# ──────────────────────────────────────────────────────────────────────────


class ExpertDecoderLayer(FusedOp):
    """One action-expert decoder layer with adaRMS modulation and joint
    attention reading VLM K/V from the corresponding prefill stage.

    The expert is Gemma-300M shape (hidden=1024) but its joint
    attention reads VLM K/V (head_dim=128 same as VLM) at matching
    layer indices. The concat is along the seq dim:
        K = concat(vlm_k_prefix [968], expert_k_suffix [50], dim=seq)
        V = concat(vlm_v_prefix [968], expert_v_suffix [50], dim=seq)
        attn = SDPA(expert_q, K, V, causal=True past prefix boundary)

    Shape contract (§1.3):
      hidden=1024, mlp_intermediate=4096, num_q_heads=16, num_kv_heads=2,
      head_dim=128 (SAME as VLM by construction — that's what lets the
      VLM K/V flow directly into the expert without a re-projection).
      Prefix tokens = up to 968. Suffix tokens = 50.
    Per-layer weights ~19.9 MB excluding adaRMS Dense (see below).

    adaRMS modulation:
      The pre_attn / pre_ffw adaRMSNorm scale/shift/gate are driven by
      a per-step `modulation` tensor whose shape is [3072, 1024] @
      adarms_cond [1024] → [3072] split into (scale+1, shift, gate)
      of 1024 each. Crucially these only depend on the integer denoise
      step, so we precompute all 10 modulation outputs per layer
      ([18, 10, 3072] bf16 = 1.1 MB total) at startup and gather them
      via metadata.position_id (= denoise_step). Saves ~120 MB / chip
      of on-chip Dense weights. Matches the existing TT-NN behavior
      (ttnn_suffix.py / ttnn_pi0_5_model.py).

    Parallelism: TP=8 (matches VLM stages; KV replicated).
    Metadata trailer:
      position_id = denoise_step (0..9), slot_id=0, reserved[0] =
      layer_idx (0..17), reserved[1] = prefix_seq_len (1 / inference).
    KV cache:
      Owns the 50-token suffix K/V. The vlm_k_prefix / vlm_v_prefix
      slots are pre-allocated; populated once per inference by the
      KV migration edge (mapping_notes §4); persistent across all 10
      Euler steps.
    """

    name: str = "expert_decoder_layer"

    act: Input = Input()
    pre_attn_norm_gamma: Input = Input()
    q_weight: Input = Input()
    k_weight: Input = Input()
    v_weight: Input = Input()
    o_weight: Input = Input()
    post_attn_norm_gamma: Input = Input()
    pre_mlp_norm_gamma: Input = Input()
    mlp_gate_weight: Input = Input()
    mlp_up_weight: Input = Input()
    mlp_down_weight: Input = Input()
    post_mlp_norm_gamma: Input = Input()
    # joint attention: KV from VLM prefill (one-shot migrated)
    vlm_k_prefix: Input = Input()
    vlm_v_prefix: Input = Input()
    # adaRMS precomputed modulations [18 layers, 10 steps, 3072]
    adarms_pre_attn_modulations: Input = Input()
    adarms_pre_ffw_modulations: Input = Input()
    # expert KV cache (suffix)
    k_cache_suffix: Input = Input()
    v_cache_suffix: Input = Input()
    # plumbing
    metadata_persistent: Input = Input()
    ar_attn_intermediate: Input = Input()
    ar_mlp_intermediate: Input = Input()
    out: Output = Output()

    _TENSOR_KEYS = (
        "act",
        "pre_attn_norm_gamma",
        "q_weight",
        "k_weight",
        "v_weight",
        "o_weight",
        "post_attn_norm_gamma",
        "pre_mlp_norm_gamma",
        "mlp_gate_weight",
        "mlp_up_weight",
        "mlp_down_weight",
        "post_mlp_norm_gamma",
        "vlm_k_prefix",
        "vlm_v_prefix",
        "adarms_pre_attn_modulations",
        "adarms_pre_ffw_modulations",
        "k_cache_suffix",
        "v_cache_suffix",
        "metadata_persistent",
        "ar_attn_intermediate",
        "ar_mlp_intermediate",
    )

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        _common_compose(cls, f, tensors, output, user_args, cls._TENSOR_KEYS)

    @staticmethod
    def emit(
        f,
        *,
        act,
        pre_attn_norm_gamma,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        post_attn_norm_gamma,
        pre_mlp_norm_gamma,
        mlp_gate_weight,
        mlp_up_weight,
        mlp_down_weight,
        post_mlp_norm_gamma,
        vlm_k_prefix,
        vlm_v_prefix,
        adarms_pre_attn_modulations,
        adarms_pre_ffw_modulations,
        k_cache_suffix,
        v_cache_suffix,
        metadata_persistent,
        ar_attn_intermediate,
        ar_mlp_intermediate,
        output_tensor,
        prefix="expert_decoder_layer",
        layer_idx=0,
        tp_size=8,
        mcast_grid_configs=None,
        receiver_socket_config: ReceiverSocketConfig = None,
        sender_socket_config: SenderSocketConfig = None,
        execute_socket_logic: bool = False,
    ):
        # Body sketch:
        #
        # 1. ReceiverSocket.emit() — standard. Metadata carries
        #    denoise_step (position_id) and layer_idx (reserved[0]).
        # 2. adaRMS pre-attn. RMSNorm(act, pre_attn_norm_gamma) then
        #    affine driven by gathered modulation[layer_idx,
        #    denoise_step].
        #    TODO(blaze): author `adarms_norm` micro-op:
        #      a. RMSNorm(act, gamma).
        #      b. Gather (scale+1, shift, gate) from modulation tensor
        #         using metadata.position_id + layer_idx.
        #      c. out = scale * normed + shift  (gate consumed at step 4).
        # 3. Joint attention: q_proj(adarms_out), k_proj/v_proj of
        #    suffix, concat(vlm_k_prefix, k_suffix) along seq, same for
        #    V, SDPA, o_proj, all_reduce.
        #    TODO(blaze): two ways to handle the KV prefix:
        #      (a) custom `joint_sdpa` that accepts vlm_k_prefix +
        #          suffix_k as separate L1 buffers, concats in-kernel.
        #      (b) at init, pre-write the VLM prefix into k_cache_suffix
        #          positions [0..968]; suffix goes at [968..1018].
        #          Standard GQA op works as-is with the longer cache.
        #    (b) is simpler; (a) is closer to deployment plan §3.3.a.
        # 4. ResidualAdd with adaRMS gate: h = h + gate_attn * attn_out.
        # 5. CbReconfig.
        # 6. adaRMS pre-ffw — same as step 2.
        # 7. SwiGLU MLP (same shape as VLM, smaller dims) + all_reduce.
        # 8. ResidualAdd with adaRMS gate_ffw → SenderSocket →
        #    CrossDeviceSignal.
        return output_tensor


# ──────────────────────────────────────────────────────────────────────────
# Stage 3 tail — Suffix MLP (action input + time embedding; action output)
# ──────────────────────────────────────────────────────────────────────────


class SuffixMlp(FusedOp):
    """Suffix MLP — runs ONCE per denoise step on Stage 3, twice per
    step actually: once at the top of the expert tower (action_in_proj
    + time embedding merge) and once at the bottom (action_out_proj).

    Two emits in one FusedOp class (similar to how
    blaze.ops.lm_head_sampling_layer covers related sub-paths in one
    op):
      emit_input():  x_t [50,32] → in_proj → +time → suffix activation [50,1024]
      emit_output(): expert_out [50,1024] → out_proj → dx_t [50,32]

    Shape contract (§1.4):
      action_in_proj [1024,32], time_mlp_in [1024,1024],
      time_mlp_out [1024,1024], action_out_proj [32,1024]; all bf8_b
      weights. ~2.5 MB total. Runs replicated on chip 0 of loudbox D,
      mcast'd to the rest.
    Metadata trailer:
      position_id = denoise_step (0..9). Drives sincos time embedding
      (we use a precomputed [10, 1024] table — gather rather than
      compute sincos in-kernel).
    """

    name: str = "suffix_mlp"

    # input path
    x_t: Input = Input()  # [1,1,50,32] bf16
    action_in_proj_weight: Input = Input()
    action_in_proj_bias: Input = Input()
    time_scalar: Input = Input()  # [1] bf16 (or precomputed table gather)
    time_mlp_in_weight: Input = Input()
    time_mlp_in_bias: Input = Input()
    time_mlp_out_weight: Input = Input()
    time_mlp_out_bias: Input = Input()
    # output path
    expert_out: Input = Input()  # [1,1,50,1024] bf8
    action_out_proj_weight: Input = Input()
    action_out_proj_bias: Input = Input()
    # plumbing
    metadata_persistent: Input = Input()

    # Two outputs because the FusedOp covers two paths.
    suffix_input: Output = Output()
    denoised_action: Output = Output()

    @classmethod
    def compose(cls, f, tensors, output, user_args):
        ua = user_args or {}
        mode = ua.get("mode", "input")  # "input" or "output"
        if mode == "input":
            cls.emit_input(
                f,
                x_t=tensors["x_t"],
                action_in_proj_weight=tensors["action_in_proj_weight"],
                action_in_proj_bias=tensors["action_in_proj_bias"],
                time_scalar=tensors["time_scalar"],
                time_mlp_in_weight=tensors["time_mlp_in_weight"],
                time_mlp_in_bias=tensors["time_mlp_in_bias"],
                time_mlp_out_weight=tensors["time_mlp_out_weight"],
                time_mlp_out_bias=tensors["time_mlp_out_bias"],
                metadata_persistent=tensors["metadata_persistent"],
                output_tensor=output,
                prefix=ua.get("prefix", "suffix_mlp_input"),
            )
        else:
            cls.emit_output(
                f,
                expert_out=tensors["expert_out"],
                action_out_proj_weight=tensors["action_out_proj_weight"],
                action_out_proj_bias=tensors["action_out_proj_bias"],
                metadata_persistent=tensors["metadata_persistent"],
                output_tensor=output,
                prefix=ua.get("prefix", "suffix_mlp_output"),
            )

    @staticmethod
    def emit_input(
        f,
        *,
        x_t,
        action_in_proj_weight,
        action_in_proj_bias,
        time_scalar,
        time_mlp_in_weight,
        time_mlp_in_bias,
        time_mlp_out_weight,
        time_mlp_out_bias,
        metadata_persistent,
        output_tensor,
        prefix="suffix_mlp_input",
    ):
        # 1. action_in_proj: x_t [50,32] @ W [32,1024] + b → [50,1024].
        # 2. Time embedding: Gather row from precomputed [10,1024] sincos
        #    table using metadata.position_id (= denoise_step).
        # 3. time_mlp_in (silu) → time_mlp_out → broadcast-add to step 1.
        # 4. No socket — feeds directly into ExpertDecoderLayer 0 on
        #    the same loudbox via L1 CB.
        return output_tensor

    @staticmethod
    def emit_output(
        f,
        *,
        expert_out,
        action_out_proj_weight,
        action_out_proj_bias,
        metadata_persistent,
        output_tensor,
        prefix="suffix_mlp_output",
    ):
        # 1. expert_out [50,1024] @ W [1024,32] + b → [50,32].
        # 2. D2HSenderSocket — push back to host.
        #    For minimal latency, gate emission on
        #    metadata.position_id == 9 (final denoise step). Otherwise
        #    emit every step and let host filter. TODO(blaze): existing
        #    d2h_sender_socket has no conditional-emit pattern; needs
        #    either a new conditional D2H or host-side filter.
        return output_tensor


# ──────────────────────────────────────────────────────────────────────────
# Per-FusedOp persistent tensor / args allocators
# ──────────────────────────────────────────────────────────────────────────

# Pattern from fused_layer_contract.md §5
# (`get_<stage>_tensors` + `get_<stage>_args`). Backed test and assembly
# pipeline call these identically; whatever the single-stage test does
# at setup time belongs in here.


def get_siglip_layer_tensors(mesh_device, *, layer_idx, hf_weights, tp_size=8, seed=None):
    """Allocate persistent tensors for one SiglipEncoderLayer.

    Mirror tests/blaze/utils/weight_helpers.py::create_mla_weights for
    the shard / preprocessing. HF SigLIP layout for pi05_libero is
    documented at models/experimental/pi0_5/tt/ttnn_siglip.py (search
    `state_dict` handling).
    """
    raise NotImplementedError("populate per pi0.5 SigLIP weight names")


def get_vlm_layer_tensors(mesh_device, *, layer_idx, hf_weights, tp_size=8, seed=None):
    raise NotImplementedError("populate per pi0.5 VLM weight names")


def get_expert_layer_tensors(mesh_device, *, layer_idx, hf_weights, tp_size=8, seed=None):
    raise NotImplementedError("populate per pi0.5 expert weight names")


def get_suffix_mlp_tensors(mesh_device, *, hf_weights, seed=None):
    raise NotImplementedError("populate per pi0.5 suffix MLP weight names")


def get_siglip_layer_args(mesh_device, *, layer_idx, receiver_mesh_coord, sender_mesh_coord, tp_size=8):
    """Return non-tensor kwargs for SiglipEncoderLayer.emit().

    Wire mcast_grid_configs, fused-op program configs, tp_size-dependent
    matmul subblock sizes. dense_layer uses
    `ua["mcast_grid_configs"]` — pi0.5 needs its own per-stage set.
    """
    raise NotImplementedError


def get_vlm_layer_args(mesh_device, *, layer_idx, receiver_mesh_coord, sender_mesh_coord, tp_size=8):
    raise NotImplementedError


def get_expert_layer_args(mesh_device, *, layer_idx, receiver_mesh_coord, sender_mesh_coord, tp_size=8):
    raise NotImplementedError


def get_suffix_mlp_args(mesh_device, *, receiver_mesh_coord, sender_mesh_coord):
    raise NotImplementedError
