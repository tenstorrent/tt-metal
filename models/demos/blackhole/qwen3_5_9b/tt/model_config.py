# models/demos/blackhole/qwen3_5_9b/tt/model_config.py
"""Qwen3.5-9B model configuration for Blackhole P150.

Subclasses ``tt_transformers.ModelArgs`` (the framework convention) while keeping
the 9B's custom weight-key scheme. The base class resolves the checkpoint from the
``HF_MODEL`` env var and reads the HF config via ``AutoConfig``; neither works for
this checkpoint, so two methods are overridden (see ``_set_hf_params`` /
``create_tokenizer``). Everything Qwen3.5-specific (hybrid Gated DeltaNet + Gated
Full Attention layers, DeltaNet key/value heads + conv kernel, partial rotary
factor) is set on top of the base params after ``super().__init__()``.

Weight loading (``load_state_dict``) and the bfp8/bf16 cache path
(``weight_cache_path``) are overridden verbatim from the standalone config: the
9B uses its own remapped key scheme, NOT the framework's meta-style wq/wk/wv keys.
"""
import json
import os
from pathlib import Path

from models.tt_transformers.tt.model_config import ModelArgs


class Qwen35ModelArgs(ModelArgs):
    """Model configuration for Qwen3.5-9B on Blackhole P150."""

    def __init__(
        self,
        mesh_device=None,
        checkpoint_dir="/local/ttuser/atupe/Qwen9b",
        max_batch_size=1,
        max_seq_len=2048,
        **kwargs,
    ):
        # checkpoint_dir is authoritative for the 9B. The base ModelArgs resolves the
        # checkpoint directory exclusively from the HF_MODEL env var (it raises
        # ValueError if unset and has no checkpoint_dir param), so bridge our explicit
        # checkpoint_dir into that convention BEFORE calling super().__init__().
        self.checkpoint_dir = checkpoint_dir
        # The base ModelArgs resolves its checkpoint from the HF_MODEL env var (it has no
        # checkpoint_dir param). Set it only for the duration of super().__init__() and
        # restore the prior value, so constructing this config has no lasting global side effect.
        _prev_hf_model = os.environ.get("HF_MODEL")
        os.environ["HF_MODEL"] = checkpoint_dir
        try:
            super().__init__(mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, **kwargs)
        finally:
            if _prev_hf_model is None:
                os.environ.pop("HF_MODEL", None)
            else:
                os.environ["HF_MODEL"] = _prev_hf_model

        # ------------------------------------------------------------------
        # Qwen3.5-specific params not set (or set differently) by the base.
        # Read from the merged text config stashed during _set_hf_params; do NOT
        # re-open config.json (avoid double I/O).
        # ------------------------------------------------------------------
        text_config = self._hf_text_config

        # RoPE — rope_theta is nested under rope_parameters in config.json; the base
        # reads a top-level "rope_theta" (absent here) so self.rope_theta is None.
        rope_params = text_config.get("rope_parameters", {})
        self.rope_theta = rope_params.get("rope_theta", 10_000_000)
        self.partial_rotary_factor = text_config.get("partial_rotary_factor", 1.0)
        self.rope_head_dim = int(self.head_dim * self.partial_rotary_factor)

        # DeltaNet-specific parameters (base does not know about these)
        self.linear_num_key_heads = text_config.get("linear_num_key_heads", 16)
        self.linear_num_value_heads = text_config.get("linear_num_value_heads", 32)
        self.linear_key_head_dim = text_config.get("linear_key_head_dim", 128)
        self.linear_value_head_dim = text_config.get("linear_value_head_dim", 128)
        self.linear_conv_kernel_dim = text_config.get("linear_conv_kernel_dim", 4)

        # Layer type list — base only reads layer_types into a local (to derive
        # sliding_window_pattern); the 9B needs the full list to dispatch DeltaNet
        # vs. full-attention layers.
        self.attention_type_list = text_config.get(
            "layer_types",
            ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
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
        """Read config.json directly instead of via transformers AutoConfig.

        The base implementation calls ``AutoConfig.from_pretrained(checkpoint_dir)``,
        which raises ``KeyError: 'qwen3_5'`` ("Transformers does not recognize this
        architecture") for this checkpoint. We load config.json with json.load,
        mirror the base's text_config merge, stash the merged dict for the
        Qwen3.5-specific attrs set in __init__, then feed the base's
        ``_set_params_from_dict`` (which consumes a plain dict).
        """
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            config = json.load(f)

        if "text_config" in config or "vision_config" in config:
            # Mirror base merge_text_config: start from text_config, then merge in
            # the non-nested top-level keys.
            # Mirrors ModelArgs._set_hf_params.merge_text_config — keep in sync if the base changes.
            text_config = config.get("text_config", {})
            text_config.update({k: v for k, v in config.items() if k not in ["text_config", "vision_config"]})
        else:
            text_config = config

        # Stash for the Qwen3.5-specific attrs read in __init__.
        self._hf_text_config = text_config

        self._set_params_from_dict(text_config)

        # The 9B is driven by raw token IDs and uses its own attention modules; the
        # vision tower is unused. Record multimodal flag from the full config but do
        # not build a vision config beyond what _set_params_from_dict already did.
        # Intentionally overrides the is_multimodal that _set_params_from_dict just
        # derived from the stripped text_config, so it reflects the FULL config's
        # vision presence. Informational only — the 9B never reads is_multimodal.
        self.is_multimodal = "vision_config" in config
        self.hf_config = None

    def create_tokenizer(self):
        """The 9B is driven by raw token IDs and never uses args.tokenizer.

        The base create_tokenizer raises ("No fallback tokenizer") for this
        checkpoint, so return None.
        """
        return None

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
        """Load + remap this checkpoint's HF safetensors. The config owns weight loading.

        Behavior-identical to the load that previously lived in Qwen35Model.from_pretrained:
        glob the sharded safetensors, read every tensor on CPU, then run the Qwen3.5 remap.
        This OVERRIDES the base meta-key (wq/wk/wv) loader — the 9B uses its own scheme.
        """
        import glob

        from safetensors import safe_open

        from models.demos.blackhole.qwen3_5_9b.tt.weight_mapping import remap_qwen35_state_dict

        files = sorted(glob.glob(f"{self.checkpoint_dir}/model.safetensors-*.safetensors"))
        if not files:
            raise FileNotFoundError(f"No safetensors matching '{self.checkpoint_dir}/model.safetensors-*.safetensors'")
        raw_state_dict = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    raw_state_dict[key] = f.get_tensor(key)
        state_dict = remap_qwen35_state_dict(raw_state_dict)
        del raw_state_dict
        return state_dict
