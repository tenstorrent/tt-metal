"""POC wrappers for pi05_siglip_ops::EncoderMatmul Op-struct.

Subclasses both SigLIPQKVMatmulOp and SigLIPOprojMatmulOp (which share the
same underlying encoder-shape matmul kernel) and points KERNEL_SOURCE at the
new Op-struct kernel main. Used to confirm the Op-struct port produces
bit-identical output to the monolithic qkv_matmul_kernel.cpp for both
N_TILES_PER_CORE=3 (QKV) and N_TILES_PER_CORE=1 (O-proj) shapes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from qkv_op import SigLIPQKVMatmulOp, build_tensors_for_test as build_qkv_tensors  # noqa: F401,E402
from oproj_op import SigLIPOprojMatmulOp, build_tensors_for_oproj_test  # noqa: F401,E402

_KERNEL_SOURCE = "models/experimental/pi0/fused_ops/attention_block/kernels/standalone_matmul_kernel.cpp"


class SigLIPQKVMatmulOpStruct(SigLIPQKVMatmulOp):
    KERNEL_SOURCE = _KERNEL_SOURCE


class SigLIPOprojMatmulOpStruct(SigLIPOprojMatmulOp):
    KERNEL_SOURCE = _KERNEL_SOURCE
