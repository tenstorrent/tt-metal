import sys
import pathlib
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[3]))

from core.ops.vector_norm_ops import RMSNormOp

OP_MAPPING = {
    "torch": RMSNormOp,
}

try:
    from vllm import _custom_ops as vllm_ops

    class VLLMRMSNormOp(RMSNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["vllm"]

            if self.add_residual:
                self._create_tensors_func = partial(
                    self._create_in_out_tensors,
                    create_inputs=True,
                    create_outputs=False,
                )
            else:
                self._create_tensors_func = partial(
                    self._create_in_out_tensors,
                    create_inputs=True,
                    create_outputs=True,
                )

        def rms_norm_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            orig_shape = src.shape
            src = src.view(-1, src.shape[-1])

            if self.add_residual:
                residual = tensor_mapping["residual"]
                residual = residual.view(-1, residual.shape[-1])
                vllm_ops.fused_add_rms_norm(src, residual, weight, self.epsilon)
                return src.view(orig_shape)
            else:
                dst = tensor_mapping["dst"]
                vllm_ops.rms_norm(dst, src, weight, self.epsilon)
                return dst.view(orig_shape)

    OP_MAPPING["vllm"] = VLLMRMSNormOp
except:
    pass


try:
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    class FlashInferRMSNormOp(RMSNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["flashinfer"]

        def rms_norm_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            orig_shape = src.shape
            src = src.view(-1, src.shape[-1])

            if self.add_residual:
                residual = tensor_mapping["residual"]
                residual = residual.view(-1, residual.shape[-1])
                fused_add_rmsnorm(src, residual, weight, self.epsilon)
                return src.view(orig_shape)
            else:
                dst = rmsnorm(src, weight, self.epsilon)
                return dst.view(orig_shape)

    OP_MAPPING["flashinfer"] = FlashInferRMSNormOp
except:
    pass
