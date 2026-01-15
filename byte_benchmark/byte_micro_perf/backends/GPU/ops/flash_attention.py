import sys
import pathlib
from functools import partial
import torch

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA2Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v2"]

            self.input_tensor_info.update(
                {
                    "q": OpTensorInfo(
                        shape=[self.batch_size, self.num_tokens // self.batch_size, self.q_head_num, self.head_dim],
                        dtype=self.torch_dtype,
                        device=self.backend.get_torch_device_name(),
                    ),
                    "k_cache": OpTensorInfo(
                        shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim],
                        dtype=self.cache_torch_dtype,
                        device=self.backend.get_torch_device_name(),
                        creator=torch.empty,
                    ),
                    "v_cache": OpTensorInfo(
                        shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim],
                        dtype=self.cache_torch_dtype,
                        device=self.backend.get_torch_device_name(),
                        creator=torch.empty,
                    ),
                }
            )

            # currently not support prefill_session_cache mode
            # cause not support different q_lens
            if self.mode in ["prefill_session_cache"]:
                raise NotImplementedError("not support prefill_session_cache")

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")

        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            # ignore k_scale/v_scale
            if self.mode == "prefill" and self.cache_len == 0:
                # q: [1, q_seq_len, q_head_num, head_dim]
                # k: [1, q_seq_len, kv_head_num, head_dim]
                # v: [1, q_seq_len, kv_head_num, head_dim]
                out = flash_attn_func(q, k_cache, v_cache, softmax_scale=self.softmax_scale, causal=True)
            else:
                # q: [batch_size, q_seq_len, q_head_num, head_dim]
                # k: [batch_size, max_kv_len, kv_head_num, head_dim]
                # v: [batch_size, max_kv_len, kv_head_num, head_dim]
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_lens,
                    cache_batch_idx=cache_slot_ids,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                )
            return out

    OP_MAPPING["flash_attn_v2"] = FA2Op
except:
    pass


try:
    from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache

    # https://github.com/Dao-AILab/flash-attention
    class FA3Op(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flash_attn_v3"]

            self.input_tensor_info.update(
                {
                    "q": OpTensorInfo(
                        shape=[self.batch_size, self.num_tokens // self.batch_size, self.q_head_num, self.head_dim],
                        dtype=self.torch_dtype,
                        device=self.backend.get_torch_device_name(),
                    ),
                    "k_cache": OpTensorInfo(
                        shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim],
                        dtype=self.cache_torch_dtype,
                        device=self.backend.get_torch_device_name(),
                        creator=torch.empty,
                    ),
                    "v_cache": OpTensorInfo(
                        shape=[self.batch_size, self.max_kv_len, self.kv_head_num, self.head_dim],
                        dtype=self.cache_torch_dtype,
                        device=self.backend.get_torch_device_name(),
                        creator=torch.empty,
                    ),
                }
            )

            # currently not support prefill_session_cache mode
            # cause not support different q_lens
            if self.mode in ["prefill_session_cache"]:
                raise NotImplementedError("not support prefill_session_cache")

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")

        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            # ignore k_scale/v_scale
            if self.mode == "prefill" and self.cache_len == 0:
                # q: [1, q_seq_len, q_head_num, head_dim]
                # k: [1, q_seq_len, kv_head_num, head_dim]
                # v: [1, q_seq_len, kv_head_num, head_dim]
                out = flash_attn_func(q, k_cache, v_cache, softmax_scale=self.softmax_scale, causal=True)
            else:
                # q: [batch_size, q_seq_len, q_head_num, head_dim]
                # k: [batch_size, max_kv_len, kv_head_num, head_dim]
                # v: [batch_size, max_kv_len, kv_head_num, head_dim]
                out = flash_attn_with_kvcache(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    cache_seqlens=cache_lens,
                    cache_batch_idx=cache_slot_ids,
                    softmax_scale=self.softmax_scale,
                    causal=True,
                )
            return out

    OP_MAPPING["flash_attn_v3"] = FA3Op
except:
    pass
