"""POC wrapper for pi05_siglip_ops::Softmax Op-struct."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from softmax_op import SigLIPSoftmaxOp, build_tensors_for_softmax_test  # noqa: F401,E402


class SigLIPSoftmaxOpStruct(SigLIPSoftmaxOp):
    KERNEL_SOURCE = "models/experimental/pi0/fused_ops/attention_block/kernels/standalone_softmax_kernel.cpp"
