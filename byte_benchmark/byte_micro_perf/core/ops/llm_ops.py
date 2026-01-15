import sys
import pathlib
import torch
import random
from functools import partial
from itertools import combinations


sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[2]))

from core.utils import OpTensorInfo, calc_tensor_size
from core.op import BasicOp


def smooth_per_token_dynamic_quant(hidden_states, smooth_scale, dst_torch_dtype=torch.int8):
    smoothed_input = torch.mul(hidden_states, smooth_scale).type(torch.float32)
    per_token_scale = torch.div(torch.max(smoothed_input.abs(), -1, keepdim=False)[0], 127.0)
    quant_tokens = torch.div(smoothed_input, per_token_scale.unsqueeze(-1)).round().type(dst_torch_dtype)
    return quant_tokens, per_token_scale


def fake_quant_gemm(tokens, per_token_scale, weights, weight_scale, dst_torch_dtype=torch.bfloat16):
    fake_gemm_output = torch.matmul(tokens.type(torch.bfloat16), weights.type(torch.bfloat16))
    dequant_scale = torch.matmul(per_token_scale.unsqueeze(-1), weight_scale.unsqueeze(0))
    y = torch.mul(fake_gemm_output, dequant_scale).type(dst_torch_dtype)
    return y


"""
******************************************
Common
******************************************
"""


class AddRmsNormDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.add_residual = self.args_dict.get("add_residual", True)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        self.eps = 1e-5

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            # use 1 as norm weight
            "norm_weight": OpTensorInfo(
                shape=[self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
        }
        if self.add_residual:
            self.input_tensor_info["residual"] = OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )

        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], dtype=torch.float32, device=self.backend.get_torch_device_name()
            ),
            "after_res": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "after_norm": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        # run func
        self._run_func = self.add_rms_norm_dynamic_quant_run

    def add_rms_norm_dynamic_quant_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        residual = tensor_mapping.get("residual", None)
        norm_weight = tensor_mapping["norm_weight"]
        smooth_scale = tensor_mapping["smooth_scale"]

        # add residual
        after_res = hidden_states
        if residual is not None:
            after_res = hidden_states + residual

        # rms norm
        after_norm = torch.nn.functional.rms_norm(
            after_res, normalized_shape=norm_weight.shape, weight=norm_weight, eps=self.eps
        )

        # dynamic quant
        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(after_norm, smooth_scale)

        return quant_tokens, per_token_scale, after_res, after_norm


class ScaleDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], dtype=torch.float32, device=self.backend.get_torch_device_name()
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        # run func
        self._run_func = self.scale_dynamic_quant

    def scale_dynamic_quant(self, tensor_mapping):
        return smooth_per_token_dynamic_quant(
            tensor_mapping["hidden_states"], tensor_mapping["smooth_scale"], self.dst_torch_dtype
        )


class MoeQuantMatmulOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["bfloat16"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            "expert_weight": OpTensorInfo(
                shape=[self.hidden_size, self.new_hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "expert_scale": OpTensorInfo(
                shape=[self.new_hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(shape=[self.num_tokens, self.new_hidden_size], dtype=self.dst_torch_dtype)
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        # run func
        self._run_func = self.moe_quant_matmul_run

    def moe_quant_matmul_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        per_token_scale = tensor_mapping["per_token_scale"]
        expert_weight = tensor_mapping["expert_weight"]
        expert_scale = tensor_mapping["expert_scale"]

        y = fake_quant_gemm(
            hidden_states, per_token_scale, expert_weight, expert_scale, dst_torch_dtype=self.dst_torch_dtype
        )
        return y


"""
******************************************
Attn
******************************************
"""


class HeadRMSNormOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # pre-defined attrs
        self.num_tokens = self.args_dict["num_tokens"]
        self.total_head_num = self.args_dict["total_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.norm_head_start = self.args_dict["norm_head_start"]
        self.norm_head_num = self.args_dict["norm_head_num"]
        self.norm_head_end = self.norm_head_start + self.norm_head_num

        self.eps = 1e-5

        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "weight": OpTensorInfo(
                shape=[self.norm_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = calc_tensor_size(
            self.input_tensor_info["token_data"]
        ) / self.total_head_num * self.norm_head_num + calc_tensor_size(self.input_tensor_info["weight"])
        self.write_bytes = calc_tensor_size(self.output_tensor_info["y"]) / self.total_head_num * self.norm_head_num
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.head_rms_norm_run

    def head_rms_norm_run(self, tensor_mapping):
        # get pre-allocated input tensors
        token_data = tensor_mapping["token_data"]
        weight = tensor_mapping["weight"]

        # get pre-allocated output tensors
        y = tensor_mapping["y"]

        # per head rms_norm
        for head_idx in range(self.norm_head_num):
            head_data = token_data[:, head_idx, :]
            head_weight = weight[head_idx, :]
            y[:, head_idx, :] = torch.nn.functional.rms_norm(
                head_data, normalized_shape=head_weight.shape, weight=head_weight, eps=self.eps
            )
        return y


# generator for prefill mode
def generate_prefill_data(q_seq_len, cache_len):
    q_lens = [q_seq_len]
    accum_q_lens = [0, q_seq_len]
    cache_lens = [cache_len]
    cache_slot_ids = [0]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


# generator for prefill_session_cache mode
def generate_prefill_session_cache_data(batch_size, target_q_len, aver_cache_len):
    # random q_len, accum to target_q_len
    aver_q_len = target_q_len // batch_size
    q_len_remainder = target_q_len % batch_size
    q_len_offset = aver_q_len // 10
    q_lens = []
    for i in range(batch_size):
        q_lens.append(aver_q_len + (1 if i < q_len_remainder else 0))
    for i in range(batch_size):
        q_lens[i] += random.randint(-q_len_offset, q_len_offset)

    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


# generator for decode mode
def generate_decode_data(batch_size, q_seq_len, aver_cache_len):
    # fixed q_len
    q_lens = [q_seq_len for _ in range(batch_size)]

    # accum q_lens
    accum_q_lens = [0]
    for i in range(batch_size):
        accum_q_lens.append(accum_q_lens[-1] + q_lens[i])

    # random cache_lens
    cache_lens = [aver_cache_len for _ in range(batch_size)]
    cache_offset = aver_cache_len // 10
    for i in range(batch_size):
        cache_lens[i] += random.randint(-cache_offset, cache_offset)

    # sequential cache_slot_ids
    cache_slot_ids = [i for i in range(batch_size)]

    kv_lens = [q_len + kv_len for q_len, kv_len in zip(q_lens, cache_lens)]

    return q_lens, accum_q_lens, cache_lens, cache_slot_ids, kv_lens


class RotaryEmbeddingOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

        self.rope_offset = self.args_dict.get("rope_offset", 0)
        self.rope_dim = self.args_dict["rope_dim"]

        self.mode = self.args_dict.get("mode", "prefill")
        if self.mode == "prefill":
            # [q_seq_len, total_head_num, head_dim]
            self.batch_size = 1
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = generate_prefill_data(
                self.q_seq_len, self.cache_len
            )

        elif self.mode == "prefill_session_cache":
            # [accumed_num_tokens, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            (
                self.q_lens,
                self.accum_q_lens,
                self.cache_lens,
                self.cache_slot_ids,
                self.kv_lens,
            ) = generate_prefill_session_cache_data(self.batch_size, self.q_seq_len, self.cache_len)

        elif self.mode == "decode":
            # [batch_size * q_seq_len, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = generate_decode_data(
                self.batch_size, self.q_seq_len, self.cache_len
            )

        else:
            raise NotImplementedError

        # accum q_lens
        self.num_tokens = sum(self.q_lens)
        # accum cache_lens
        self.num_cache_tokens = sum(self.cache_lens)
        # max q_len + cache_len
        self.max_kv_len = max(self.kv_lens)

        # return cos/sin
        def precompute_freqs_cis(dim, max_seq_len, theta: float = 10000.0):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
            t = torch.arange(max_seq_len, device=freqs.device)
            freqs = torch.outer(t, freqs).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            return torch.real(freqs_cis), torch.imag(freqs_cis)

        cos_tensor, sin_tensor = precompute_freqs_cis(self.rope_dim, self.max_kv_len)

        self.input_tensor_info = {
            "packed_qkv": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "q_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device),
            ),
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device),
            ),
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device),
            ),
            "cos": OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: cos_tensor.to(device=device),
            ),
            "sin": OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: sin_tensor.to(device=device),
            ),
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["packed_qkv"])
            / self.total_head_num
            * (self.q_head_num + self.kv_head_num)
            + calc_tensor_size(self.input_tensor_info["q_lens"])
            + calc_tensor_size(self.input_tensor_info["accum_q_lens"])
            + calc_tensor_size(self.input_tensor_info["cache_lens"])
            + calc_tensor_size(self.input_tensor_info["cos"])
            + calc_tensor_size(self.input_tensor_info["sin"])
        )

        self.write_bytes = (
            calc_tensor_size(self.output_tensor_info["y"]) / self.total_head_num * (self.q_head_num + self.kv_head_num)
        )

        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.rotary_embedding_run

    def rotary_embedding_run(self, tensor_mapping):
        # get pre-allocated input tensors
        packed_qkv = tensor_mapping["packed_qkv"]
        q_lens = tensor_mapping["q_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        cache_lens = tensor_mapping["cache_lens"]
        cos = tensor_mapping["cos"]
        sin = tensor_mapping["sin"]

        # get pre-allocated output tensors
        y = tensor_mapping["y"]

        def rotate(qk, cos, sin):
            # [q_seq_len, q_head_num + kv_head_num, head_dim]
            x1 = qk[..., : self.rope_dim // 2]
            x2 = qk[..., self.rope_dim // 2 :]

            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

            x1_rot = x1 * cos - x2 * sin
            x2_rot = x1 * sin + x2 * cos

            return torch.cat([x1_rot, x2_rot], dim=-1)

        # for each batch
        for batch_idx in range(self.batch_size):
            q_len = q_lens[batch_idx]
            q_offset = accum_q_lens[batch_idx]
            cur_cache_len = cache_lens[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            qk_head_start = 0
            qk_head_end = self.q_head_num + self.kv_head_num

            dim_start = self.rope_offset
            dim_end = self.rope_offset + self.rope_dim

            cache_start = cur_cache_len
            cache_end = cur_cache_len + q_len

            y[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end] = rotate(
                packed_qkv[token_start:token_end, qk_head_start:qk_head_end, dim_start:dim_end],
                cos[cache_start:cache_end],
                sin[cache_start:cache_end],
            )

        return y


class StoreKVCacheOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict.get("dst_dtype", "int8")
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.torch_dst_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

        self.mode = self.args_dict.get("mode", "prefill")
        if self.mode == "prefill":
            # [q_seq_len, total_head_num, head_dim]
            self.batch_size = 1
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = generate_prefill_data(
                self.q_seq_len, self.cache_len
            )

        elif self.mode == "prefill_session_cache":
            # [accumed_num_tokens, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            (
                self.q_lens,
                self.accum_q_lens,
                self.cache_lens,
                self.cache_slot_ids,
                self.kv_lens,
            ) = generate_prefill_session_cache_data(self.batch_size, self.q_seq_len, self.cache_len)

        elif self.mode == "decode":
            # [batch_size * q_seq_len, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = generate_decode_data(
                self.batch_size, self.q_seq_len, self.cache_len
            )

        else:
            raise NotImplementedError

        # accum q_lens
        self.num_tokens = sum(self.q_lens)
        # accum cache_lens
        self.num_cache_tokens = sum(self.cache_lens)
        # max q_len + cache_len
        self.max_kv_len = max(self.kv_lens)

        self.input_tensor_info = {
            "packed_qkv": OpTensorInfo(
                shape=[self.num_tokens, self.total_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "q_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device),
            ),
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device),
            ),
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device),
            ),
            "cache_slot_ids": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_slot_ids, dtype=dtype, device=device),
            ),
            "k_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dst_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "v_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dst_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
            "k_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            "v_scale": OpTensorInfo(
                shape=[self.kv_head_num, self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
        }

        self.output_tensor_info = {}

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * 2 * self.kv_head_num
            + calc_tensor_size(self.input_tensor_info["q_lens"])
            + calc_tensor_size(self.input_tensor_info["accum_q_lens"])
            + calc_tensor_size(self.input_tensor_info["cache_lens"])
            + calc_tensor_size(self.input_tensor_info["cache_slot_ids"])
            + calc_tensor_size(self.input_tensor_info["k_scale"])
            + calc_tensor_size(self.input_tensor_info["v_scale"])
        )

        self.write_bytes = (
            calc_tensor_size(self.input_tensor_info["k_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens
            + calc_tensor_size(self.input_tensor_info["v_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens
        )

        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.store_kv_cache_run

    def store_kv_cache_run(self, tensor_mapping):
        # get pre-allocated input tensors
        packed_qkv = tensor_mapping["packed_qkv"]
        q_lens = tensor_mapping["q_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        cache_lens = tensor_mapping["cache_lens"]
        cache_slot_ids = tensor_mapping["cache_slot_ids"]
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]
        k_scale = tensor_mapping["k_scale"]
        v_scale = tensor_mapping["v_scale"]

        # for each batch
        for batch_idx in range(self.batch_size):
            q_len = q_lens[batch_idx]
            q_offset = accum_q_lens[batch_idx]
            cur_cache_len = cache_lens[batch_idx]
            cur_slot_id = cache_slot_ids[batch_idx]

            token_start = q_offset
            token_end = q_offset + q_len

            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            cache_start = cur_cache_len
            cache_end = cur_cache_len + q_len

            # [num_tokens, total_head_num, head_dim]
            # --> [q_len, kv_head_num, head_dim]
            # --> [kv_head_num, q_len, head_dim]
            cur_k = packed_qkv[token_start:token_end, k_head_start:k_head_end].transpose(0, 1)
            cur_v = packed_qkv[token_start:token_end, v_head_start:v_head_end].transpose(0, 1)

            # [max_batch_size, total_head_num, max_seq_len, head_dim]
            # --> [kv_head_num, q_len, head_dim]

            k_cache[cur_slot_id, k_head_start:k_head_end, cache_start:cache_end] = torch.round(
                torch.mul(cur_k, k_scale.unsqueeze(1))
            ).type(self.torch_dst_dtype)
            v_cache[cur_slot_id, v_head_start:v_head_end, cache_start:cache_end] = torch.round(
                torch.mul(cur_v, v_scale.unsqueeze(1))
            ).type(self.torch_dst_dtype)


class FlashAttentionOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        # parse args
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        # q in bf16, out bf16
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # cache_dtype
        # bf16 or int8
        self.cache_dtype = self.args_dict.get("cache_dtype", "bfloat16")
        if not self.dtype in ["bfloat16", "int8"]:
            raise NotImplementedError
        self.cache_torch_dtype = getattr(torch, self.cache_dtype)

        # pre-defined attrs
        self.is_causal = self.args_dict.get("is_causal", True)
        if not self.is_causal:
            raise NotImplementedError

        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.mode = self.args_dict.get("mode", "prefill")
        if self.mode == "prefill":
            # [q_seq_len, total_head_num, head_dim]
            self.batch_size = 1
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = generate_prefill_data(
                self.q_seq_len, self.cache_len
            )

        elif self.mode == "prefill_session_cache":
            # [accumed_num_tokens, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            (
                self.q_lens,
                self.accum_q_lens,
                self.cache_lens,
                self.cache_slot_ids,
                self.kv_lens,
            ) = generate_prefill_session_cache_data(self.batch_size, self.q_seq_len, self.cache_len)

        elif self.mode == "decode":
            # [batch_size * q_seq_len, total_head_num, head_dim]
            self.batch_size = self.args_dict.get("batch_size", 1)
            self.q_seq_len = self.args_dict["q_seq_len"]
            self.cache_len = self.args_dict["cache_len"]

            self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = generate_decode_data(
                self.batch_size, self.q_seq_len, self.cache_len
            )

        else:
            raise NotImplementedError

        self.softmax_scale = self.head_dim ** (-0.5)

        # accum q_lens
        self.num_tokens = sum(self.q_lens)
        # accum cache_lens
        self.num_cache_tokens = sum(self.cache_lens)
        # max q_len + cache_len
        self.max_kv_len = max(self.kv_lens)

        # define basic input/outpus, could be overrided
        self.input_tensor_info = {
            # slice from packed_qkv, stride on head_num
            "q": OpTensorInfo(
                shape=[self.num_tokens, self.q_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "q_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device),
            ),
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device),
            ),
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device),
            ),
            "cache_slot_ids": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.cache_slot_ids, dtype=dtype, device=device),
            ),
            # reference kv_cache format
            "k_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
            "v_cache": OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
        }
        if self.dtype != self.cache_dtype:
            self.input_tensor_info.update(
                {
                    "k_scale": OpTensorInfo(
                        shape=[self.kv_head_num, self.head_dim],
                        dtype=torch.float32,
                        device=self.backend.get_torch_device_name(),
                        creator=torch.ones,
                    ),
                    "v_scale": OpTensorInfo(
                        shape=[self.kv_head_num, self.head_dim],
                        dtype=torch.float32,
                        device=self.backend.get_torch_device_name(),
                        creator=torch.ones,
                    ),
                }
            )

        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.num_tokens, self.q_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["q"])
            + calc_tensor_size(self.input_tensor_info["q_lens"])
            + calc_tensor_size(self.input_tensor_info["accum_q_lens"])
            + calc_tensor_size(self.input_tensor_info["cache_lens"])
            + calc_tensor_size(self.input_tensor_info["cache_slot_ids"])
            + calc_tensor_size(self.input_tensor_info["k_cache"])
            / self.batch_size
            / self.max_kv_len
            * self.num_cache_tokens
            + calc_tensor_size(self.input_tensor_info["v_cache"])
            / self.batch_size
            / self.max_kv_len
            * self.num_cache_tokens
        )
        if self.dtype != self.cache_dtype:
            self.read_bytes += calc_tensor_size(self.input_tensor_info["k_scale"]) + calc_tensor_size(
                self.input_tensor_info["v_scale"]
            )
        self.write_bytes = self.output_tensor_size

        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 0

        for idx in range(self.batch_size):
            q_len = self.q_lens[idx]
            cache_len = self.cache_lens[idx]
            kv_len = q_len + cache_len

            # p = q * v, bf16 batch_gemm
            # o = p * v, bf16 batch_gemm
            gemm_flops = 2 * self.q_head_num * q_len * self.head_dim * kv_len * 2

            if self.is_causal:
                flops_ratio = (cache_len + 1 + kv_len) * q_len / 2 / (kv_len * kv_len)
            else:
                flops_ratio = 1.0

            self.calc_flops += gemm_flops * flops_ratio

        # specify create input/output tensors func
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

        # specify run function
        self._run_func = self.flash_attention_run

    def flash_attention_run(self, tensor_mapping):
        raise NotImplementedError


"""
******************************************
MOE
******************************************
"""


class MoeGatingGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16", "float32"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict.get("dst_dtype", "float32")
        if not self.dst_dtype in ["float32"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.num_experts = self.args_dict["num_experts"]
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        # input/output tensors
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "gating_weight": OpTensorInfo(
                shape=[self.hidden_size, self.num_experts],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }
        self.output_tensor_info = {
            "gating_output": OpTensorInfo(
                shape=[self.num_tokens, self.num_experts],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.num_experts

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        # run func
        self._run_func = self.moe_gating_gemm_run

    def moe_gating_gemm_run(self, tensor_mapping):
        gating_output = torch.mm(tensor_mapping["hidden_states"], tensor_mapping["gating_weight"]).type(
            self.dst_torch_dtype
        )
        return gating_output


class MoeSoftmaxTopkOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # pre-defined attrs
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]

        self.compute_mode = self.args_dict["compute_mode"]
        if not self.compute_mode in ["pre-softmax", "post-softmax"]:
            raise NotImplementedError

        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.hidden_size = self.args_dict["hidden_size"]

        # input/output tensors
        self.input_tensor_info = {
            "gating_output": OpTensorInfo(
                shape=[self.num_tokens, self.num_experts],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "selected_experts": OpTensorInfo(
                shape=[self.num_tokens, self.topk],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "moe_weights": OpTensorInfo(
                shape=[self.num_tokens, self.topk],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=False)

        # run func
        self._run_func = self.moe_softmax_topk_run

    def moe_softmax_topk_run(self, tensor_mapping):
        gating_output = tensor_mapping["gating_output"]

        # softmax --> topk --> normlize
        if self.compute_mode == "pre-softmax":
            softmax_output = torch.softmax(gating_output, dim=-1)
            moe_weights, selected_experts = torch.topk(softmax_output, self.topk, dim=-1)
            moe_weights = moe_weights / moe_weights.sum(dim=-1, keepdim=True)
            return selected_experts, moe_weights
        # topk --> softmax
        elif self.compute_mode == "post-softmax":
            topk_output, selected_experts = torch.topk(gating_output, self.topk, dim=-1)
            softmax_output = torch.softmax(topk_output, dim=-1)
            return selected_experts, softmax_output
        else:
            raise NotImplementedError


class MoeScatterDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.world_size = self.args_dict.get("world_size", 1)
        self.rank = self.args_dict.get("rank", 0)
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.dp_size = self.args_dict.get("dp_size", 1)
        self.sp_size = self.args_dict.get("sp_size", 1)

        self.num_shared_experts = self.args_dict.get("num_shared_experts", 0)
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        """
        select shared experts based on dp_size/dp_rank
        """
        self.dp_rank = self.rank // self.sp_size
        self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

        """
        select tokens based on sp_size/sp_rank
        """
        self.sp_rank = self.rank % self.sp_size
        self.shared_tokens_per_sp = self.num_tokens // self.sp_size
        self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
        self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

        """
        select experts based on ep_rank
        """
        self.experts_per_rank = self.num_experts // self.ep_size
        self.ep_rank = self.rank
        self.expert_idx_start = self.ep_rank * self.experts_per_rank
        self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
        self.other_experts_set = set(range(self.num_experts)) - set(range(self.expert_idx_start, self.expert_idx_end))

        """
        for convinience, we also split num_tokens to ep_size parts to generate selected_experts
        and selected tokens are also distributed to corresponding experts
        if rank == 0, experts_per_rank == 4, and topk == 5, and num_tokens == 32, so num_tokens_per_ep == 8
        token 0: 0, 1, 2, 3, 0
        token 1: 1, 2, 3, 0, 1
        token 2: 2, 3, 0, 1, 2
        token 3: 3, 0, 1, 2, 3
        ...

        other tokens will select other tokens randomly
        """
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

        # [tokens_per_ep, topk]
        self.actual_output_tokens = self.tokens_per_ep * self.topk
        self.experts_repeat_time = 1
        if self.actual_output_tokens > self.experts_per_rank:
            self.experts_repeat_time = (self.actual_output_tokens + self.experts_per_rank - 1) // self.experts_per_rank
        self.refer_expert_seq = (
            torch.arange(start=self.expert_idx_start, end=self.expert_idx_end, dtype=torch.int32)
            .repeat(self.experts_repeat_time)[: self.actual_output_tokens]
            .view(self.tokens_per_ep, self.topk)
        )

        # all tokens topk
        dummy_experts = list(next(combinations(self.other_experts_set, self.topk)))
        self.refer_selected_experts = (
            torch.tensor(dummy_experts, dtype=torch.int32).unsqueeze(0).repeat(self.num_tokens, 1)
        )
        self.refer_selected_experts[self.tokens_ep_start : self.tokens_ep_end] = self.refer_expert_seq

        self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

        # reserve tokens memory for shared_tokens_per_sp/allocated_tokens
        self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

        # for extreme case
        self.max_allocated_tokens = self.num_tokens * self.topk
        self.max_scatter_tokens = self.total_shared_tokens + self.max_allocated_tokens

        # for designed real case
        self.real_allocated_tokens = self.actual_output_tokens
        self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens

        # input/output tensors
        self.input_tensor_info = {
            # complete tokens
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            # complete selected_experts
            "selected_experts": OpTensorInfo(
                shape=[self.num_tokens, self.topk],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: self.refer_selected_experts.to(
                    self.backend.get_torch_device_name()
                ),
            ),
            # complete moe_weights
            "moe_weights": OpTensorInfo(
                shape=[self.num_tokens, self.topk],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            # partial (shared_experts + experts) smooth_scale
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.total_experts_num, self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
        }
        self.output_tensor_info = {
            # partial, reserved for max
            "scatter_tokens": OpTensorInfo(
                shape=[self.max_scatter_tokens, self.hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros,
            ),
            # partial, reserved for max
            "scatter_per_token_scale": OpTensorInfo(
                shape=[self.max_scatter_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            # partial, reserved for max
            "scatter_tokens_offset": OpTensorInfo(
                shape=[self.max_scatter_tokens],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1,
            ),
            # partial (shared_experts + experts) token count
            "experts_token_count": OpTensorInfo(
                shape=[self.total_experts_num],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros,
            ),
            # partial (shared_experts + experts) token start
            "experts_token_start": OpTensorInfo(
                shape=[self.total_experts_num],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros,
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["hidden_states"]) / self.num_tokens * self.tokens_per_ep
            + calc_tensor_size(self.input_tensor_info["selected_experts"])
            + calc_tensor_size(self.input_tensor_info["moe_weights"])
            + calc_tensor_size(self.input_tensor_info["smooth_scale"])
        )
        self.write_bytes = (
            calc_tensor_size(self.output_tensor_info["scatter_tokens"])
            / self.max_scatter_tokens
            * self.real_scatter_tokens
            + calc_tensor_size(self.output_tensor_info["scatter_per_token_scale"])
            / self.max_scatter_tokens
            * self.real_scatter_tokens
            + calc_tensor_size(self.output_tensor_info["scatter_tokens_offset"])
            / self.max_scatter_tokens
            * self.real_scatter_tokens
            + calc_tensor_size(self.output_tensor_info["experts_token_count"])
            + calc_tensor_size(self.output_tensor_info["experts_token_start"])
        )
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.moe_scatter_dynamic_quant_run

    def moe_scatter_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        hidden_states = tensor_mapping["hidden_states"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        smooth_scale = tensor_mapping["smooth_scale"]

        # get pre-allocated output tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]
        scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_start = tensor_mapping["experts_token_start"]

        # shared experts
        for idx in range(self.shared_experts_per_rank):
            expert_idx = idx

            # dynamic quant
            input_token_start = self.shared_token_sp_start
            input_token_end = self.shared_token_sp_end

            quant_tokens, tokens_scale = smooth_per_token_dynamic_quant(
                hidden_states[input_token_start:input_token_end],
                smooth_scale[expert_idx],
                dst_torch_dtype=self.dst_torch_dtype,
            )

            # assign output
            output_token_start = idx * self.shared_tokens_per_sp
            output_token_end = output_token_start + self.shared_tokens_per_sp

            scatter_tokens[output_token_start:output_token_end] = quant_tokens
            scatter_per_token_scale[output_token_start:output_token_end] = tokens_scale
            experts_token_count[expert_idx] = self.num_tokens
            experts_token_start[expert_idx] = output_token_start
            scatter_tokens_offset[output_token_start:output_token_end] = torch.arange(
                start=input_token_start,
                end=input_token_end,
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            )

        # experts
        cur_output_token_offset = self.total_shared_tokens
        for idx in range(self.experts_per_rank):
            expert_idx = self.shared_experts_per_rank + idx

            # get token indices
            token_indices, topk_indices = torch.where(selected_experts == self.expert_idx_start + idx)
            cur_token_count = token_indices.numel()

            experts_token_count[expert_idx] = cur_token_count
            experts_token_start[expert_idx] = cur_output_token_offset

            output_token_start = cur_output_token_offset
            cur_output_token_offset += cur_token_count
            output_token_end = cur_output_token_offset

            if cur_token_count > 0:
                cur_tokens = hidden_states[token_indices]
                cur_tokens_weight = moe_weights[token_indices, topk_indices]

                # dynamic quant
                quant_tokens, tokens_scale = smooth_per_token_dynamic_quant(
                    torch.mul(cur_tokens, cur_tokens_weight.view(cur_token_count, 1)),
                    smooth_scale[expert_idx],
                    dst_torch_dtype=self.dst_torch_dtype,
                )

                # assign output
                scatter_tokens[output_token_start:output_token_end] = quant_tokens
                scatter_per_token_scale[output_token_start:output_token_end] = tokens_scale
                scatter_tokens_offset[output_token_start:output_token_end] = token_indices.type(
                    scatter_tokens_offset.dtype
                )


class MoeQuantGroupGemmOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["bfloat16"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # predefined attrs
        self.world_size = self.args_dict.get("world_size", 1)
        self.rank = self.args_dict.get("rank", 0)
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.dp_size = self.args_dict.get("dp_size", 1)
        self.sp_size = self.args_dict.get("sp_size", 1)

        self.num_shared_experts = self.args_dict.get("num_shared_experts", 0)
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]

        """
        select shared experts based on dp_size/dp_rank
        """
        self.dp_rank = self.rank // self.sp_size
        self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

        """
        select tokens based on sp_size/sp_rank
        """
        self.sp_rank = self.rank % self.sp_size
        self.shared_tokens_per_sp = self.num_tokens // self.sp_size
        self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
        self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

        """
        select experts based on ep_rank
        """
        self.experts_per_rank = self.num_experts // self.ep_size
        self.ep_rank = self.rank
        self.expert_idx_start = self.ep_rank * self.experts_per_rank
        self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
        self.other_experts_set = set(range(self.num_experts)) - set(range(self.expert_idx_start, self.expert_idx_end))

        """
        for convinience, we also split num_tokens to ep_size parts to generate selected_experts
        """
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

        # [tokens_per_ep, topk]
        self.allocated_tokens = self.tokens_per_ep * self.topk
        self.allocated_tokens_per_expert = self.allocated_tokens // self.experts_per_rank
        self.allocated_tokens_per_expert_remainder = self.allocated_tokens % self.experts_per_rank

        self.token_list = []
        self.token_start_list = []
        temp_token_start = 0
        for i in range(self.shared_experts_per_rank):
            self.token_start_list.append(temp_token_start)
            self.token_list.append(self.shared_tokens_per_sp)
            temp_token_start += self.token_list[-1]
        for i in range(self.experts_per_rank):
            self.token_start_list.append(temp_token_start)
            if i < self.allocated_tokens_per_expert_remainder:
                self.token_list.append(self.allocated_tokens_per_expert + 1)
            else:
                self.token_list.append(self.allocated_tokens_per_expert)
            temp_token_start += self.token_list[-1]

        self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

        self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

        self.real_allocated_tokens = self.allocated_tokens
        self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens

        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.real_scatter_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.real_scatter_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            "experts_weight": OpTensorInfo(
                shape=[self.total_experts_num, self.hidden_size, self.new_hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "experts_scale": OpTensorInfo(
                shape=[self.total_experts_num, self.new_hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            "experts_token_count": OpTensorInfo(
                shape=[self.total_experts_num],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.token_list, dtype=dtype, device=device),
            ),
            "experts_token_start": OpTensorInfo(
                shape=[self.total_experts_num],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.token_start_list, dtype=dtype, device=device),
            ),
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(
                shape=[self.real_scatter_tokens, self.new_hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = 2 * self.real_scatter_tokens * self.hidden_size * self.new_hidden_size

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.moe_quant_group_gemm_run

    def moe_quant_group_gemm_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]
        experts_weight = tensor_mapping["experts_weight"]
        experts_scale = tensor_mapping["experts_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_start = tensor_mapping["experts_token_start"]

        # get pre-allocated output tensor
        y = tensor_mapping["y"]

        # use loop gemm and fp32 to simulate int8 group_gemm
        for i in range(self.total_experts_num):
            cur_token_start = experts_token_start[i]
            cur_token_end = cur_token_start + experts_token_count[i]

            cur_tokens = scatter_tokens[cur_token_start:cur_token_end]
            cur_tokens_scale = per_token_scale[cur_token_start:cur_token_end]

            cur_weight = experts_weight[i]
            cur_weight_scale = experts_scale[i]

            y[cur_token_start:cur_token_end] = fake_quant_gemm(
                cur_tokens, cur_tokens_scale, cur_weight, cur_weight_scale, self.dst_torch_dtype
            )
        return y


class MoeSwigluDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # dst_dtype
        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8"]:
            raise NotImplementedError
        self.dst_torch_dtype = getattr(torch, self.dst_dtype)

        # pre-defined attrs
        self.world_size = self.args_dict.get("world_size", 1)
        self.rank = self.args_dict.get("rank", 0)
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.dp_size = self.args_dict.get("dp_size", 1)
        self.sp_size = self.args_dict.get("sp_size", 1)

        self.num_shared_experts = self.args_dict.get("num_shared_experts", 0)
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        """
        select shared experts based on dp_size/dp_rank
        """
        self.dp_rank = self.rank // self.sp_size
        self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

        """
        select tokens based on sp_size/sp_rank
        """
        self.sp_rank = self.rank % self.sp_size
        self.shared_tokens_per_sp = self.num_tokens // self.sp_size
        self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
        self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

        """
        select experts based on ep_rank
        no remainder on **num_experts**
        """
        self.experts_per_rank = self.num_experts // self.ep_size
        self.ep_rank = self.rank
        self.expert_idx_start = self.ep_rank * self.experts_per_rank
        self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
        self.other_experts_set = set(range(self.num_experts)) - set(range(self.expert_idx_start, self.expert_idx_end))

        """
        for convinience, we also split num_tokens to ep_size parts to generate selected_experts
        """
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

        # [tokens_per_ep, topk]
        self.allocated_tokens = self.tokens_per_ep * self.topk
        self.allocated_tokens_per_expert = self.allocated_tokens // self.experts_per_rank
        self.allocated_tokens_per_expert_remainder = self.allocated_tokens % self.experts_per_rank

        self.token_list = []
        self.token_start_list = []
        temp_token_start = 0
        for i in range(self.shared_experts_per_rank):
            self.token_start_list.append(temp_token_start)
            self.token_list.append(self.shared_tokens_per_sp)
            temp_token_start += self.token_list[-1]
        for i in range(self.experts_per_rank):
            self.token_start_list.append(temp_token_start)
            if i < self.allocated_tokens_per_expert_remainder:
                self.token_list.append(self.allocated_tokens_per_expert + 1)
            else:
                self.token_list.append(self.allocated_tokens_per_expert)
            temp_token_start += self.token_list[-1]

        self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

        self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

        self.real_allocated_tokens = self.allocated_tokens
        self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens

        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.real_scatter_tokens, self.hidden_size * 2],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.total_experts_num, self.hidden_size],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones,
            ),
            "experts_token_count": OpTensorInfo(
                shape=[self.total_experts_num],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.token_list, dtype=dtype, device=device),
            ),
            "experts_token_start": OpTensorInfo(
                shape=[self.total_experts_num],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.token_start_list, dtype=dtype, device=device),
            ),
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.real_scatter_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.real_scatter_tokens],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.moe_swiglu_dynamic_quant_run

    def moe_swiglu_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        smooth_scale = tensor_mapping["smooth_scale"]
        experts_token_count = tensor_mapping["experts_token_count"]
        experts_token_start = tensor_mapping["experts_token_start"]

        # get per-allocated output tensors
        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        # swiglu, x1 used as gating, x2 used as up
        x1, x2 = torch.chunk(scatter_tokens, 2, dim=-1)
        swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

        # per expert dynamic quant
        for i in range(self.total_experts_num):
            cur_token_start = experts_token_start[i]
            cur_token_end = cur_token_start + experts_token_count[i]

            cur_quant_tokens, cur_per_token_scale = smooth_per_token_dynamic_quant(
                swiglu_tokens[cur_token_start:cur_token_end], smooth_scale[i]
            )
            quant_tokens[cur_token_start:cur_token_end] = cur_quant_tokens
            per_token_scale[cur_token_start:cur_token_end] = cur_per_token_scale

        return quant_tokens, per_token_scale


class MoeGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        # src_dtype
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        # predefined attrs
        self.world_size = self.args_dict.get("world_size", 1)
        self.rank = self.args_dict.get("rank", 0)
        self.ep_size = self.args_dict.get("ep_size", 1)
        self.dp_size = self.args_dict.get("dp_size", 1)
        self.sp_size = self.args_dict.get("sp_size", 1)

        self.num_shared_experts = self.args_dict.get("num_shared_experts", 0)
        self.num_experts = self.args_dict["num_experts"]
        self.topk = self.args_dict["topk"]
        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]

        """
        select shared experts based on dp_size/dp_rank
        """
        self.dp_rank = self.rank // self.sp_size
        self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

        """
        select tokens based on sp_size/sp_rank
        """
        self.sp_rank = self.rank % self.sp_size
        self.shared_tokens_per_sp = self.num_tokens // self.sp_size
        self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
        self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

        """
        select experts based on ep_rank
        """
        self.experts_per_rank = self.num_experts // self.ep_size
        self.ep_rank = self.rank
        self.expert_idx_start = self.ep_rank * self.experts_per_rank
        self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
        self.other_experts_set = set(range(self.num_experts)) - set(range(self.expert_idx_start, self.expert_idx_end))

        """
        for convinience, we also split num_tokens to ep_size parts to generate selected_experts
        """
        self.tokens_per_ep = self.num_tokens // self.ep_size
        self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
        self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

        # [tokens_per_ep, topk]
        self.allocated_tokens = self.tokens_per_ep * self.topk
        self.allocated_tokens_per_expert = self.allocated_tokens // self.experts_per_rank
        self.allocated_tokens_per_expert_remainder = self.allocated_tokens % self.experts_per_rank

        self.token_offset_list = []

        # for shared experts
        for _ in range(self.shared_experts_per_rank):
            self.token_offset_list.extend(range(self.shared_token_sp_start, self.shared_token_sp_end))

        # for experts
        for expert_idx in range(self.experts_per_rank):
            cur_select_token = expert_idx
            while cur_select_token // self.topk < self.tokens_per_ep:
                self.token_offset_list.append(cur_select_token // self.topk + self.tokens_ep_start)
                cur_select_token += self.experts_per_rank

        self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

        self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

        self.real_allocated_tokens = self.allocated_tokens
        self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens

        # input/output tensors
        self.input_tensor_info = {
            "scatter_tokens": OpTensorInfo(
                shape=[self.real_scatter_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "scatter_tokens_offset": OpTensorInfo(
                shape=[self.real_scatter_tokens],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(self.token_offset_list, dtype=dtype, device=device),
            ),
        }
        self.output_tensor_info = {
            # init zero
            "convergent_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.zeros,
            ),
        }

        # calculator
        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = 2 * calc_tensor_size(self.input_tensor_info["scatter_tokens"]) + calc_tensor_size(
            self.input_tensor_info["scatter_tokens_offset"]
        )
        self.write_bytes = calc_tensor_size(self.input_tensor_info["scatter_tokens"])
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # creator func
        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        # run func
        self._run_func = self.moe_gather_run

    def moe_gather_run(self, tensor_mapping):
        # get pre-allocated input tensors
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]

        # get pre-allocated output tensors
        convergent_tokens = tensor_mapping["convergent_tokens"]

        # [num_scatter_tokens, hidden_size] --> [num_tokens, hidden_size]
        convergent_tokens.index_add_(0, scatter_tokens_offset, scatter_tokens)

        return convergent_tokens


"""
quant_matmul and moe_group_gemm
"""


class QuantMatmulOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["int8"]:
            raise NotImplementedError

        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.new_hidden_size = self.args_dict["new_hidden_size"]

        self.input_tensor_info = {
            "hidden_states": OpTensorInfo(
                shape=[self.num_tokens, self.hidden_size], dtype=torch.int8, device=self.backend.get_torch_device_name()
            ),
            "weight": OpTensorInfo(
                shape=[self.hidden_size, self.new_hidden_size],
                dtype=torch.int8,
                device=self.backend.get_torch_device_name(),
            ),
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], dtype=torch.float32, device=self.backend.get_torch_device_name()
            ),
            "weight_scale": OpTensorInfo(
                shape=[self.new_hidden_size], dtype=torch.float32, device=self.backend.get_torch_device_name()
            ),
        }
        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.num_tokens, self.new_hidden_size],
                dtype=torch.int8,
                device=self.backend.get_torch_device_name(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        # gemm:         [M, K] * [K, N] = [M, N]
        # scale:        [M, 1] * [1, N] --> [M, N]
        # cast:         [M, N] --> [M, N]
        # dequantize:   [M, N] --> [M, N]
        self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

        self._run_func = self.quant_matmul_run

    def quant_matmul_run(self, tensor_mapping):
        raise NotImplementedError


def generate_balanced_random_list(target_sum, length):
    average = target_sum // length
    remainder = target_sum % length
    adjust_max = average // 10

    result = [average] * length
    if remainder > 0:
        indices = random.sample(range(length), remainder)
        for idx in indices:
            result[idx] += 1

    source_list = []
    for _ in range(length // 2):
        random_value = random.randint(0, adjust_max)
        source_list.append(random_value)
        source_list.append(-random_value)
    if length % 2 == 1:
        source_list.append(0)

    target_adjust = random.sample(source_list, length)
    for i in range(length):
        result[i] += target_adjust[i]

    return result


"""
each rank has:
- [num_tokens // world_size, hidden_size]

after gating/softmax/topk, each rank has:
- [num_tokens // world_size * topk, hidden_size]

dispatch tokens to corresponding rank, dispatched tokens are possibly unbalanced

"""


class MoeDispatchTokensOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict.get("dtype", "bfloat16")
        if not self.dtype in ["bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        assert self.world_size % 2 == 0
        dist_module = self.backend.get_dist_module()
        self.local_rank = dist_module.get_rank(group=self.op_group)

        self.num_tokens = self.args_dict["num_tokens"]
        self.hidden_size = self.args_dict["hidden_size"]
        self.topk = self.args_dict["topk"]

        # 4096 // 8 = 512 tokens per rank
        self.num_tokens_per_rank = self.num_tokens // self.world_size

        # 1. 512 * topk tokens on each rank
        # 2. 512 * topk tokens may be dispatched to other ranks
        # 3. split 512 * topk tokens to list
        # 4. all_to_all
        self.after_num_tokens = self.num_tokens_per_rank * self.topk

        random.seed(1)
        all_ranks_distributions = []
        for _ in range(self.world_size):
            all_ranks_distributions.append(generate_balanced_random_list(self.after_num_tokens, self.world_size))

        self.input_split_sizes = all_ranks_distributions[self.local_rank]
        self.output_split_sizes = [data[self.local_rank] for data in all_ranks_distributions]

        self.input_token_num = self.after_num_tokens
        self.output_token_num = sum(self.output_split_sizes)

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.input_token_num, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.output_token_num, self.hidden_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)

        self._run_func = self.moe_dispatch_tokens_run

    def moe_dispatch_tokens_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_to_all_single(dst, src, self.output_split_sizes, self.input_split_sizes, group=self.op_group)
        return dst
