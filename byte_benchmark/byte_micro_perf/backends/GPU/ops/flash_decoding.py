import sys
import pathlib
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.llm_ops import FlashDecodingOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}

try:
    from flash_attn import flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA2Op(FlashDecodingOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v2"]

            if self.dtype == "bfloat16_c8":
                raise NotImplementedError

        def flash_decoding_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            cache_seqlens = tensor_mapping["cache_seqlens"]

            if self.dtype == "bfloat16":
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                )
                return out
            elif self.dtype == "bfloat16_c8":
                k_scale = tensor_mapping["k_scale"]
                v_scale = tensor_mapping["v_scale"]
                raise NotImplementedError

    OP_MAPPING["flash_attn_v2"] = FA2Op
except:
    pass


try:
    from flash_attn_interface import flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA3Op(FlashDecodingOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v3"]

            if self.dtype == "bfloat16_c8":
                raise NotImplementedError

        def flash_decoding_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            cache_seqlens = tensor_mapping["cache_seqlens"]

            if self.dtype == "bfloat16":
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_seqlens,
                    causal=True,
                )
                return out
            elif self.dtype == "bfloat16_c8":
                k_scale = tensor_mapping["k_scale"]
                v_scale = tensor_mapping["v_scale"]
                raise NotImplementedError

    OP_MAPPING["flash_attn_v3"] = FA3Op
except:
    pass
