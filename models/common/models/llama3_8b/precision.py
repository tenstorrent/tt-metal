# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTTv2-owned precision policy for Llama-3.1-8B."""

import ttnn

TENSOR_GROUPS = ("ff1_ff3", "ff2", "wqkv", "wo", "kv_cache", "activation")
OP_GROUPS = (
    "li_ff1_ff3",
    "li_ff2",
    "li_qkv_decode",
    "li_o_decode",
    "sdpa_decode",
    "li_qkv_prefill",
    "li_o_prefill",
    "sdpa_prefill",
    "accuracy",
)


class Llama31DecoderPrecision:
    """Per-decoder tensor dtype and math-fidelity selection.

    This mirrors the TTTv1 policy for the Llama-3.1-8B path, including the
    existing performance override for decoder 31.
    """

    _DTYPES = {
        "bfp4": ttnn.bfloat4_b,
        "bfp8": ttnn.bfloat8_b,
        "bf16": ttnn.bfloat16,
        None: None,
    }

    @classmethod
    def from_string(cls, optimizations: str):
        if optimizations == "performance":
            return cls.performance
        if optimizations == "accuracy":
            return cls.accuracy
        raise ValueError(
            f"Invalid optimization configuration: {optimizations}. Allowed values are 'performance' or 'accuracy'"
        )

    @classmethod
    def performance(cls, num_decoders: int, model_name: str):
        inst = cls(num_decoders, model_name, cls._performance_settings(model_name))
        if model_name == "Llama-3.1-8B-Instruct" and num_decoders > 31:
            inst._tensor_precision[31]["ff1_ff3"] = "bfp8"
            inst._op_fidelity[31]["li_ff1_ff3"] = "hifi2fp16"
            inst._update_full_name()
        inst.__name__ = "performance"
        return inst

    @classmethod
    def accuracy(cls, num_decoders: int, model_name: str):
        inst = cls(num_decoders, model_name, cls._accuracy_settings(model_name))
        inst.__name__ = "accuracy"
        return inst

    def __init__(self, num_decoders: int, model_name: str, settings: dict | None = None):
        self.model_name = model_name
        default_tensor_precision, default_op_fidelity = self._default_settings()
        settings = settings or {}
        default_tensor_precision.update(settings.get("tensor_precision", {}))
        default_op_fidelity.update(settings.get("op_fidelity", {}))
        self._tensor_precision = {decoder_id: dict(default_tensor_precision) for decoder_id in range(num_decoders)}
        self._op_fidelity = {decoder_id: dict(default_op_fidelity) for decoder_id in range(num_decoders)}
        self._update_full_name()

    @staticmethod
    def _base_model_name(model_name: str):
        for suffix in ("-Instruct", "-instruct"):
            if model_name.endswith(suffix):
                return model_name[: -len(suffix)]
        return model_name

    @classmethod
    def _accuracy_settings(cls, model_name: str):
        base_model_name = cls._base_model_name(model_name)
        if base_model_name.startswith("Llama-3") or base_model_name.startswith("Meta-Llama-3"):
            return {
                "tensor_precision": {
                    "wqkv": "bfp8",
                    "kv_cache": "bfp8",
                    "wo": "bfp8",
                },
                "op_fidelity": {
                    "li_ff1_ff3": "hifi2fp16",
                    "li_ff2": "hifi2fp16",
                },
            }
        return {
            "tensor_precision": {
                "wqkv": "bf16",
                "kv_cache": "bf16",
                "wo": "bf16",
            },
            "op_fidelity": {
                "li_qkv_decode": "hifi4",
                "li_qkv_prefill": "hifi4",
                "sdpa_decode": "hifi4",
                "sdpa_prefill": "hifi4",
                "li_o_decode": "hifi4",
                "li_o_prefill": "hifi4",
            },
        }

    @classmethod
    def _performance_settings(cls, model_name: str):
        return {
            "tensor_precision": {"ff1_ff3": "bfp4"},
            "op_fidelity": {"li_ff1_ff3": "lofi"},
        }

    @staticmethod
    def _default_settings():
        return (
            {
                "ff1_ff3": "bfp8",
                "ff2": "bfp8",
                "wqkv": "bfp8",
                "wo": "bfp8",
                "kv_cache": "bfp8",
                "activation": None,
            },
            {
                "li_ff1_ff3": "hifi2fp16",
                "li_ff2": "hifi2fp16",
                "li_qkv_decode": "hifi2",
                "sdpa_decode": "hifi2",
                "li_o_decode": "hifi2",
                "li_qkv_prefill": "hifi2",
                "sdpa_prefill": "hifi4",
                "li_o_prefill": "hifi2",
                "accuracy": "hifi4fp32",
            },
        )

    def get_tensor_dtype(self, decoder_id: int, tensor: str, prefetcher: bool = False):
        effective_decoder_id = 0 if prefetcher else decoder_id
        value = self._tensor_precision.get(effective_decoder_id, {}).get(tensor)
        if prefetcher and value is None and tensor != "activation":
            return ttnn.bfloat8_b
        return self._DTYPES.get(value)

    def get_math_fidelity(self, decoder_id: int, op: str, configuration):
        kernel_lookup = {
            "lofi": configuration.compute_kernel_config_lofi,
            "hifi2": configuration.compute_kernel_config_hifi2,
            "hifi2na": configuration.compute_kernel_config_hifi2_na,
            "hifi2fp16": configuration.compute_kernel_config_hifi2_fp16,
            "hifi2nol1acc": configuration.compute_kernel_config_hifi2_nol1acc,
            "hifi4": configuration.compute_kernel_config_hifi4,
            "hifi4fp32": configuration.compute_kernel_config_hifi4_fp32,
        }
        return kernel_lookup[self._op_fidelity[decoder_id][op]]

    def _update_full_name(self):
        self._full_name = " | ".join(
            f"Decoder {decoder_id}: precision_cfg = {self._tensor_precision[decoder_id]}, fidelity_cfg = {self._op_fidelity[decoder_id]}"
            for decoder_id in self._tensor_precision
        )
