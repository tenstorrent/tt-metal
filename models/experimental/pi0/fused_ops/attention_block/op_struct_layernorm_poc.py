"""POC wrapper for pi05_siglip_ops::LayerNorm Op-struct.

Subclasses the existing layernorm_op.py wrapper but points KERNEL_SOURCE
at the new Op-struct kernel main. Used by test_layernorm_op_struct.py to
confirm the Op-struct port preserves LN's two-stage reduce +
binary_op_init_common reset (load-bearing per
pi05-ln-kernel-multi-reduce-pattern and pi05-llk-binary-op-init-common).
"""
import sys
from pathlib import Path

# Reuse the existing op + tensor builder.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "perf"))
from layernorm_op import SigLIPLayerNormOp, build_tensors_for_ln_test  # noqa: F401,E402


class SigLIPLayerNormOpStruct(SigLIPLayerNormOp):
    KERNEL_SOURCE = "models/experimental/pi0/fused_ops/attention_block/kernels/standalone_layernorm_kernel.cpp"
