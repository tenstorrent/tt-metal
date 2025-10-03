"""
Implementation of the Grok-2 architecture in pure PyTorch.

The Grok-2 model is a very large mixture-of-experts (MoE) transformer with
64 transformer blocks, a hidden size of 8192 and a vocabulary of 131,072 tokens.
Each transformer block contains a multi-head self attention sublayer followed
by a mixture-of-experts feed-forward network.  The MoE uses eight experts per
block and selects the top-2 experts for each token.  A small shared MLP is
added to the MoE and the two outputs are averaged when `residual_moe` is set
in the configuration.  Rotary positional embeddings are used in the attention
module and the model relies on RMS normalization throughout.  The design
follows the architecture described in the SGLang implementation【901089138700521†L552-L618【901089138700521†L729-L767】 and the HuggingFace configuration for Grok-2
configuration for Grok-2【796741125451584†L59-L89】.

This file provides a self contained implementation of Grok-2 along with a
convenience ``from_pretrained`` classmethod that can load weights from the
`xai-org/grok-2` repository on Hugging Face.  The implementation borrows
heavily from the open source Grok-1 implementation (see the ``keyfan/grok-1-hf``
repository) and adapts it to the Grok-2 hyperparameters.

Note: this code is intended for illustrative purposes.  Loading the
270 billion-parameter Grok-2 model requires multiple high-memory GPUs and
cannot be executed in the constrained environment used here.  Nevertheless
the code shows how to construct the model and how to load the official
checkpoint using the Hugging Face APIs.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    # These imports are optional and used only in the ``from_pretrained``
    # convenience method.  They allow downloading the model configuration and
    # weights directly from Hugging Face.  If Hugging Face is not installed
    # they will raise ImportError, in which case the user will need to
    # manually download the weights.
    from transformers import AutoConfig, AutoModelForCausalLM
except Exception:
    AutoConfig = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore


@dataclass
class GrokConfig:
    """Configuration for the Grok-2 model.

    The default values correspond to the released Grok-2 checkpoint.  See the
    official ``config.json`` on Hugging Face for reference【796741125451584†L59-L99】.

    The default values correspond to the released Grok-2 checkpoint.  See the
    official ``config.json`` on Hugging Face for reference【796741125451584†L59-L99】.
    """

    vocab_size: int = 131_072
    hidden_size: int = 8_192
    intermediate_size: int = 32_768
    moe_intermediate_size: int = 16_384
    num_hidden_layers: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    num_local_experts: int = 8
    num_experts_per_tok: int = 2
    residual_moe: bool = True
    head_dim: int = 128
    rms_norm_eps: float = 1.0e-5
    rope_theta: float = 208_533_496
    max_position_embeddings: int = 131_072
    embedding_multiplier_scale: float = 90.50966799187809
    output_multiplier_scale: float = 0.5
    hidden_act: str = "gelu"
    # Soft capping for attention logits and router logits.  See
    # ``Grok1Attention`` and ``Grok1MoE`` in sglang for details【901089138700521†L447-L464】.
    attn_logit_softcapping: float = 30.0
    router_logit_softcapping: float = 30.0
    final_logit_softcapping: float = 50.0
    attn_dropout: float = 0.0
    # Temperature length used in xAI's implementation.  Not used here.
    attn_temperature_len: int = 1024


class GrokRMSNorm(nn.Module):
    """Root mean square (RMS) layer normalisation.

    This normalisation normalises the hidden states by their root mean square
    value and then applies a learned rescaling.  It is equivalent to the
    ``T5LayerNorm`` used in the T5 and Llama architectures.  The SGLang
    implementation uses RMSNorm before the attention, after attention and
    after the MoE feed‑forward【901089138700521†L626-L633】.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class GrokRotaryEmbedding(nn.Module):
    """Rotary positional embedding used by Grok.

    The implementation caches cosine and sine matrices up to the maximum
    sequence length.  On every call it returns the first ``seq_len`` rows
    of the cached tables.  The shape of the returned tensors is
    ``(seq_len, 2 * dim)`` so that they can be easily broadcasted to the
    query/key tensors.  This mirrors the SGLang implementation of
    ``RotaryEmbedding``【901089138700521†L342-L449】.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # compute inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype))
        # In Grok and related models the real and imaginary parts are interleaved
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the cached cosine and sine embeddings for a given sequence length.

        Args:
            x (torch.Tensor): a tensor used only to determine the device and
                dtype for the cache.  Typically this is the value tensor from
                the attention layer.
            seq_len (int, optional): the required sequence length.  If not
                provided the sequence length of ``x`` is used.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cosine and sine tables of shape
                ``(seq_len, 2 * dim)``.
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        if seq_len > self.max_seq_len_cached:
            # extend the cache if necessary
            self._set_cos_sin_cache(seq_len, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:seq_len].to(dtype=x.dtype, device=x.device),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the last dimension of the tensor by half.  If the last dimension
    contains pairs of values (real and imaginary parts), this corresponds to
    multiplying by the complex number ``-i``.  See the SGLang code for the
    equivalent implementation【205582949680871†L302-L310】.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embedding to query and key tensors.

    This function broadcasts the cosine and sine values to match the shape
    of ``q`` and ``k`` and performs the rotation.  It is a direct port of
    the SGLang helper function【205582949680871†L315-L370】.

    Args:
        q (torch.Tensor): query tensor of shape ``(batch, heads, seq_len, head_dim)``.
        k (torch.Tensor): key tensor of shape ``(batch, kv_heads, seq_len, head_dim)``.
        cos (torch.Tensor): cosine embeddings of shape ``(seq_len, 2 * head_dim)``.
        sin (torch.Tensor): sine embeddings of shape ``(seq_len, 2 * head_dim)``.
        position_ids (torch.Tensor): position indices for the tokens.
        unsqueeze_dim (int): dimension along which to unsqueeze ``cos`` and ``sin``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: rotated query and key tensors.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads to match the number of query heads.

    This helper function expands the key/value heads when the number of
    key/value heads is smaller than the number of query heads.  It is
    equivalent to ``torch.repeat_interleave`` along the head dimension and
    mirrors the SGLang implementation【205582949680871†L374-L395】.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # shape becomes (batch, num_kv_heads, n_rep, slen, head_dim)
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GrokAttention(nn.Module):
    """Multi‑head self attention with multi‑query keys and values.

    Grok uses 64 query heads but only 8 key/value heads.  The keys and values
    are expanded to 64 heads using ``repeat_kv``【205582949680871†L524-L569】.  Rotary
    positional embeddings are applied to the query and key states and the
    attention logits are soft‑capped using a hyperbolic tangent【205582949680871†L571-L575】.
    """

    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        assert self.num_heads * self.head_dim == self.hidden_size, "hidden_size must equal num_heads * head_dim"
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings
        # projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # rotary embedding
        self.rotary_emb = GrokRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=int(self.rope_theta),
        )
        # Soft cap for attention logits
        self.logit_cap = max(config.attn_logit_softcapping, 0.0)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        # Precompute scaling factor for dot product
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the self‑attention for a batch of tokens.

        Args:
            hidden_states (torch.Tensor): input tensor of shape
                ``(batch_size, seq_len, hidden_size)``.
            attention_mask (torch.Tensor, optional): attention mask broadcastable
                to ``(batch_size, 1, seq_len, seq_len)`` with zeros in valid
                positions and ``-inf`` in masked positions.  If ``None`` the
                mask is not applied.
            position_ids (torch.LongTensor, optional): tensor of shape
                ``(batch_size, seq_len)`` containing the position of each token.
                If ``None`` positions are assumed to be ``range(seq_len)``.

        Returns:
            torch.Tensor: tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        bsz, q_len, _ = hidden_states.size()
        # Linear projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # Reshape to (batch, heads, seq_len, head_dim)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # Position ids
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, q_len)
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(v, seq_len=q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=2)
        # Expand k/v to match q heads
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)
        # Compute attention scores.  The scaling by 1/sqrt(d_k) is implicit in the
        # ``scale`` attribute and the logits are soft‑capped using tanh【205582949680871†L571-L575】.
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.logit_cap > 0:
            attn_scores = self.logit_cap * torch.tanh(attn_scores / self.logit_cap)
        # Apply mask (if provided).  The mask should broadcast to
        # (batch_size, heads, seq_len, seq_len).
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        # Reshape back to (batch_size, seq_len, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.head_dim)
        # Final linear projection
        output = self.o_proj(attn_output)
        return output


class GrokBlockSparseTop2MLP(nn.Module):
    """Single expert feed‑forward network used in the Grok MoE.

    The expert contains two parallel projections (`linear` and `linear_v`) from
    hidden size to ``ffn_dim``.  The output is computed as

        gelu(linear(x)) * linear_v(x) → linear_1

    which corresponds to a gated activation as used in Mistral and Mixtral.  The
    default activation is GELU but other activations could be used by
    overriding ``config.hidden_act``.  This matches the SGLang definition of
    ``Grok1MLP``【901089138700521†L86-L124】 and the PyTorch implementation in
    ``keyfan/grok-1-hf``【205582949680871†L633-L658】.
    """

    def __init__(self, config: GrokConfig, ffn_dim: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = ffn_dim if ffn_dim is not None else config.moe_intermediate_size
        # Two projections for the gated activation
        self.linear = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.linear_v = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        # Final projection back to hidden size
        self.linear_1 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        # Activation function
        if config.hidden_act == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_act}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x1 = self.linear(hidden_states)
        x2 = self.linear_v(hidden_states)
        return self.linear_1(self.act_fn(x1) * x2)


class GrokDecoderLayer(nn.Module):
    """A single Grok transformer block.

    Each block consists of a pre‑normed multi‑head self attention followed by a
    mixture‑of‑experts feed‑forward sublayer.  When ``config.residual_moe`` is
    ``True`` a shared MLP of dimension ``config.intermediate_size`` is added
    alongside the expert outputs and the two are averaged【901089138700521†L614-L623】.  Four
    RMS norms are used as per the SGLang implementation【901089138700521†L626-L633】.
    """

    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        # Attention sublayer
        self.self_attn = GrokAttention(config)
        # Router for top‑k expert selection.  We keep this in float32 for
        # numerical stability as recommended in the SGLang code【901089138700521†L156-L164】.
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=torch.float32)
        # Experts: a module list of individual MLPs
        self.experts = nn.ModuleList([GrokBlockSparseTop2MLP(config) for _ in range(self.num_experts)])
        # Optional shared MLP when using residual MoE
        if config.residual_moe:
            # Shared MLP uses the non‑MoE intermediate size
            self.mlp = GrokBlockSparseTop2MLP(config, ffn_dim=config.intermediate_size)
        # Normalisation layers
        self.rms_norm = GrokRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_1 = GrokRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_2 = GrokRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.rms_norm_3 = GrokRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        # Constant used to average shared MLP and MoE outputs
        self.sqrt_two = 1.4142135623730951

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre‑attention RMS norm
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states)
        # Self attention
        attn_output = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        # Residual connection and second RMS norm
        hidden_states = residual + self.rms_norm_1(attn_output)
        # Pre‑MoE normalisation
        residual = hidden_states
        hidden_states = self.rms_norm_2(hidden_states)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # Flatten batch and sequence dimensions for the router
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        # Compute router logits and softmax in float32
        router_logits = self.router(hidden_states_flat.to(torch.float32))
        # Soft cap the router logits (optional).  The SGLang code applies a
        # soft capping via ``fused_moe_router_shim``; here we emulate this by
        # applying tanh with the configured cap【901089138700521†L156-L171】.
        cap = self.config.router_logit_softcapping
        router_logits = cap * torch.tanh(router_logits / cap)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        # Select top‑k experts for each token
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)
        # Prepare tensor for accumulating expert outputs
        moe_output = torch.zeros_like(hidden_states_flat)
        # Create one‑hot mask indicating which expert is selected for each token
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # Loop over experts: gather the tokens assigned to each expert, run
        # their MLP and scatter back the results weighted by the routing
        # coefficients
        for expert_idx, expert_layer in enumerate(self.experts):
            idx, tok_indices = torch.where(expert_mask[expert_idx])
            if tok_indices.numel() == 0:
                continue
            tok_list = tok_indices.tolist()
            idx_list = idx.tolist()
            # Gather the states for this expert
            current_states = hidden_states_flat.index_select(0, torch.tensor(tok_list, device=hidden_states.device))
            # Compute expert output and multiply by routing weight
            current_output = expert_layer(current_states) * routing_weights[tok_list, idx_list, None]
            # Scatter add into the output tensor
            moe_output.index_add_(0, tok_indices, current_output.to(hidden_states_flat.dtype))
        # Reshape back to (batch, seq_len, hidden_size)
        moe_output = moe_output.view(batch_size, seq_len, hidden_dim)
        # If residual MoE is enabled, compute the shared MLP and average
        if self.config.residual_moe:
            shared_output = self.mlp(hidden_states)
            moe_output = (shared_output + moe_output) / self.sqrt_two
        # Residual connection and final RMS norm
        hidden_states = residual + self.rms_norm_3(moe_output)
        return hidden_states


class GrokModel(nn.Module):
    """Grok language model without the LM head.

    The model embeds input tokens, processes them through a stack of
    ``config.num_hidden_layers`` transformer blocks and applies a final
    RMS normalisation【901089138700521†L729-L767】.  The embedding vectors are scaled
    by ``config.embedding_multiplier_scale`` after lookup as in the official
    checkpoint.
    """

    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GrokDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = GrokRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        # Embed tokens and scale
        hidden_states = self.embed_tokens(input_ids) * self.config.embedding_multiplier_scale
        # Create position ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        # Broadcast attention mask if necessary
        if attention_mask is not None:
            # Convert to 4D mask expected by attention: (batch, 1, seq_len, seq_len)
            # Mask values are 0 for valid tokens and -inf for masked positions
            mask = attention_mask[:, None, None, :]
            attn_mask = (1.0 - mask.to(hidden_states.dtype)) * torch.finfo(hidden_states.dtype).min
        else:
            attn_mask = None
        # Iterate through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attn_mask, position_ids=position_ids)
        # Final normalisation
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GrokForCausalLM(nn.Module):
    """Grok language model with a language modelling head.

    This wrapper adds a linear layer on top of ``GrokModel`` to project the
    hidden states back to the vocabulary.  The output is scaled by
    ``config.output_multiplier_scale`` and soft‑capped using
    ``config.final_logit_softcapping`` as in the SGLang implementation【901089138700521†L920-L938】.
    """

    def __init__(self, config: GrokConfig) -> None:
        super().__init__()
        self.config = config
        self.model = GrokModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(hidden_states)
        # Apply output multiplier and soft cap
        logits = logits.to(torch.float32) * self.config.output_multiplier_scale
        cap = self.config.final_logit_softcapping
        logits = cap * torch.tanh(logits / cap)
        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device_map: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> "GrokForCausalLM":
        """
        Load a Grok‑2 checkpoint from Hugging Face and return a ``GrokForCausalLM`` model.

        This convenience method will download the configuration and weights from
        the specified ``model_name`` using the Hugging Face transformers API.
        Internally it loads the official implementation via ``AutoModelForCausalLM``
        with ``trust_remote_code=True`` and copies the state dictionary into
        the pure PyTorch implementation defined in this file.  Using the official
        model ensures that the weight names and shapes match exactly.

        Args:
            model_name (str): The name of the model on Hugging Face, e.g.
                ``"xai-org/grok-2"``.
            device_map (Optional[str]): Optional device mapping passed to
                ``AutoModelForCausalLM.from_pretrained``.  Use ``"auto"`` to
                dispatch layers across available GPUs.
            dtype (Optional[torch.dtype]): The desired dtype (e.g. ``torch.bfloat16``).
            trust_remote_code (bool): Whether to trust the remote code when
                loading the official model.  Should be left ``True`` for Grok.

        Returns:
            GrokForCausalLM: the model with loaded weights.
        """
        if AutoConfig is None or AutoModelForCausalLM is None:
            raise ImportError("transformers is not installed; please install transformers to use from_pretrained")
        # Load the official configuration
        official_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        # Construct our config from the official values
        cfg = GrokConfig(
            vocab_size=getattr(official_config, "vocab_size", 131_072),
            hidden_size=getattr(official_config, "hidden_size", 8_192),
            intermediate_size=getattr(official_config, "intermediate_size", 32_768),
            moe_intermediate_size=getattr(official_config, "moe_intermediate_size", 16_384),
            num_hidden_layers=getattr(official_config, "num_hidden_layers", 64),
            num_attention_heads=getattr(official_config, "num_attention_heads", 64),
            num_key_value_heads=getattr(official_config, "num_key_value_heads", 8),
            num_local_experts=getattr(official_config, "num_local_experts", 8),
            num_experts_per_tok=getattr(official_config, "num_experts_per_tok", 2),
            residual_moe=getattr(official_config, "residual_moe", True),
            head_dim=getattr(official_config, "head_dim", 128),
            rms_norm_eps=getattr(official_config, "rms_norm_eps", 1e-5),
            rope_theta=getattr(official_config, "rope_theta", 208_533_496),
            max_position_embeddings=getattr(official_config, "max_position_embeddings", 131_072),
            embedding_multiplier_scale=getattr(official_config, "embedding_multiplier_scale", 90.50966799187809),
            output_multiplier_scale=getattr(official_config, "output_multiplier_scale", 0.5),
            hidden_act=getattr(official_config, "hidden_act", "gelu"),
            attn_logit_softcapping=getattr(official_config, "attn_logit_softcapping", 30.0),
            router_logit_softcapping=getattr(official_config, "router_logit_softcapping", 30.0),
            final_logit_softcapping=getattr(official_config, "final_logit_softcapping", 50.0),
            attn_dropout=getattr(official_config, "attention_dropout", 0.0),
            attn_temperature_len=getattr(official_config, "attn_temperature_len", 1024),
        )
        # Instantiate our model
        model = cls(cfg)
        # Load the official model to get the weights
        official_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        # Copy the state dict
        state_dict = official_model.state_dict()
        # The Hugging Face model uses the prefix "transformer" for the base
        # model.  We strip this prefix so that parameters map directly to our
        # implementation.  For example ``transformer.embed_tokens.weight``
        # becomes ``model.embed_tokens.weight``.  Any leftover parameters (e.g.
        # cached rotary embeddings) are ignored.
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("transformer."):
                new_key = key[len("transformer.") :]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        # Load the weights into our model
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            print(f"Warning: missing parameters when loading {model_name}: {missing}")
        if len(unexpected) > 0:
            print(f"Warning: unexpected parameters when loading {model_name}: {unexpected}")
        return model
