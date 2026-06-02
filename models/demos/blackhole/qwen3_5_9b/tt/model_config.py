# models/demos/blackhole/qwen3_5_9b/tt/model_config.py
"""Qwen3.5-9B model configuration for Blackhole P150.

Subclasses ``tt_transformers.ModelArgs`` (the framework convention). ``HF_MODEL`` is
the single source of truth and must be exported (the base raises a clear error if it
is unset). It may be a local checkpoint directory OR a hub id; a hub id is resolved to
a local snapshot directory via ``snapshot_download`` (same as the vLLM wrapper) because
``AutoConfig.from_pretrained`` on a bare hub id is unreliable in this transformers
version. Following the ``gpt_oss`` / ``gemma4`` convention, config, weights and the
tokenizer are loaded with ``trust_remote_code=True``. The base class resolves the
checkpoint from ``HF_MODEL`` (``self.CKPT_DIR``) and parses the HF config via
``AutoConfig``. No JSON config override and no ``checkpoint_dir`` constructor param remain.

Everything Qwen3.5-specific (hybrid Gated DeltaNet + Gated Full Attention layers,
DeltaNet key/value heads + conv kernel, partial rotary factor) is read from the
parsed HF text config and set on top of the base params after ``super().__init__()``.

Weight loading (``load_state_dict``) and the bfp8/bf16 cache path
(``weight_cache_path``) are overridden: the 9B uses its own remapped key scheme,
NOT the framework's meta-style wq/wk/wv keys. Weights come from
``transformers.AutoModelForCausalLM.from_pretrained`` (resolves to the text-only
``Qwen3_5ForCausalLM`` — no vision tower) and are remapped to the internal scheme.
"""
import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs


class Qwen35ModelArgs(ModelArgs):
    """Model configuration for Qwen3.5-9B on Blackhole P150."""

    def __init__(
        self,
        mesh_device=None,
        max_batch_size=1,
        max_seq_len=2048,
        **kwargs,
    ):
        # HF_MODEL (set in the environment) is the single source of truth: the base
        # ModelArgs reads it into self.CKPT_DIR and raises a clear error if it is unset.
        # Unless HF_MODEL already points at a local checkpoint dir (one containing
        # config.json), resolve it to a local snapshot dir via snapshot_download (same
        # as the vLLM wrapper): AutoConfig.from_pretrained on a bare hub id is unreliable
        # in this transformers version, but works on a directory path. The config.json
        # check (rather than os.path.isdir) avoids being fooled by a stray relative dir
        # created by the weight tensor cache when an unresolved hub id was used before.
        hf_model = os.getenv("HF_MODEL")
        if hf_model and not os.path.isfile(os.path.join(hf_model, "config.json")):
            from huggingface_hub import snapshot_download

            os.environ["HF_MODEL"] = snapshot_download(hf_model)
        super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)

        # The base resolves the checkpoint dir from HF_MODEL into self.CKPT_DIR; mirror
        # it onto self.checkpoint_dir so weight_cache_path / load_state_dict keep working.
        self.checkpoint_dir = self.CKPT_DIR

        # ------------------------------------------------------------------
        # Qwen3.5-specific params the base does not know about. Read from the
        # parsed HF text config (Qwen3_5TextConfig). The base already set
        # head_dim, dim, n_heads, n_kv_heads, n_layers, vocab_size, norm_eps, etc.
        # ------------------------------------------------------------------
        text_config = self.hf_config.get_text_config()

        # RoPE — rope_theta is nested under rope_parameters; the base reads a top-level
        # "rope_theta" (absent here) so self.rope_theta is None.
        rope_params = getattr(text_config, "rope_parameters", None) or {}
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = getattr(text_config, "partial_rotary_factor", 1.0)
        self.rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # DeltaNet-specific parameters (base does not know about these)
        self.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)
        self.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)

        # Layer type list — base only reads layer_types into a local (to derive
        # sliding_window_pattern); the 9B needs the full list to dispatch DeltaNet
        # vs. full-attention layers.
        self.attention_type_list = getattr(text_config, "layer_types", None) or (
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8
        )

        # Derived
        self.linear_q_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_k_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_v_dim = self.linear_num_value_heads * self.linear_value_head_dim

        # Blackhole P150 device config (lazy import to allow CPU-only testing)
        if mesh_device is not None:
            import ttnn

            self.weight_dtype = ttnn.bfloat8_b
            self.act_dtype = ttnn.bfloat16
        else:
            self.weight_dtype = None
            self.act_dtype = None

    def _set_hf_params(self, checkpoint_dir):
        # Match gpt_oss/gemma4: load the HF config with trust_remote_code=True. Set the
        # flag before delegating so the base AutoConfig.from_pretrained call uses it.
        self.trust_remote_code_hf = True
        super()._set_hf_params(checkpoint_dir)

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "full_attention"

    def is_deltanet_layer(self, layer_idx: int) -> bool:
        return self.attention_type_list[layer_idx] == "linear_attention"

    def weight_cache_path(self, dtype=None):
        """Return cache directory path for converted weight tensors.

        Directory is created automatically by ttnn.as_tensor when first cache file is written.
        """
        if dtype is None:
            dtype = self.weight_dtype
        import ttnn

        if dtype == ttnn.bfloat8_b:
            suffix = "tensor_cache_bfp8"
        else:
            suffix = "tensor_cache_bf16"
        return Path(self.checkpoint_dir) / suffix

    def load_state_dict(self):
        """Load + remap this checkpoint's weights via transformers from_pretrained.

        HF_MODEL (self.CKPT_DIR) is the single source — a hub name or local path.
        AutoModelForCausalLM resolves to the TEXT-ONLY Qwen3_5ForCausalLM (no vision
        tower is built), whose state_dict uses the `model.` prefix; remap_qwen35_state_dict
        normalizes that to the internal key scheme. This OVERRIDES the base meta-key
        (wq/wk/wv) loader — the 9B uses its own scheme.
        """
        from transformers import AutoModelForCausalLM

        from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

        model = AutoModelForCausalLM.from_pretrained(self.CKPT_DIR, dtype="auto", trust_remote_code=True)
        state_dict = remap_qwen35_state_dict(model.state_dict())
        del model
        return state_dict
