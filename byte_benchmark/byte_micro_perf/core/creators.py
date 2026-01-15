import sys
import pathlib
import importlib
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).absolute().parents[1]))


def create_backend_instance(backend_type: str):
    backend_module = importlib.import_module("backends." + backend_type + ".backend_" + backend_type.lower())
    backend_cls = getattr(backend_module, "Backend" + backend_type)

    backend_instance = backend_cls()
    backend_instance.backend_type = backend_type
    backend_instance.backend_cls = backend_cls
    backend_instance.torch_device_name = backend_instance.get_torch_device_name()
    backend_instance.device_name = backend_instance.get_device_name(0)
    backend_instance.device_count, backend_instance.avail_devices = backend_instance.get_device_count()
    backend_instance.env_dict = backend_instance.get_backend_env()
    return backend_instance


# collect all backends OP_MAPPING
OP_INFO_MAPPING = {
    # xccl_ops
    "all_reduce": {"test_mode": "concurrent"},
    "reduce_scatter": {"test_mode": "concurrent"},
    "all_gather": {"test_mode": "concurrent"},
    "all_to_all": {"test_mode": "concurrent"},
    "broadcast": {"test_mode": "concurrent"},
    "p2p": {"test_mode": "concurrent_p2p"},
    "host2device": {"test_mode": "concurrent"},
    "device2host": {"test_mode": "concurrent"},
    "device2device": {"test_mode": "single"},
    # vector_linear_ops
    "add": {"test_mode": "single"},
    "sub": {"test_mode": "single"},
    "mul": {"test_mode": "single"},
    "cast": {"test_mode": "single"},
    # vector_sfu_ops
    "div": {"test_mode": "single"},
    "sin": {"test_mode": "single"},
    "cos": {"test_mode": "single"},
    "exp": {"test_mode": "single"},
    "log": {"test_mode": "single"},
    "sqrt": {"test_mode": "single"},
    # vector_reduction_ops
    "reduce_max": {"test_mode": "single"},
    "reduce_min": {"test_mode": "single"},
    "reduce_sum": {"test_mode": "single"},
    "topk": {"test_mode": "single"},
    # vector_norm_ops
    "layer_norm": {"test_mode": "single"},
    "rms_norm": {"test_mode": "single"},
    "softmax": {"test_mode": "single"},
    # vector_activation_ops
    "gelu": {"test_mode": "single"},
    "silu": {"test_mode": "single"},
    # vector_index_ops
    "sort": {"test_mode": "single"},
    "embedding": {"test_mode": "single"},
    "gather": {"test_mode": "single"},
    "index_select": {"test_mode": "single"},
    "scatter": {"test_mode": "single"},
    "index_add": {"test_mode": "single"},
    # tensor_gemm_ops
    "gemm": {"test_mode": "single"},
    # llm: basic
    "scale_dynamic_quant": {"test_mode": "single"},
    "add_rms_norm_dynamic_quant": {"test_mode": "single"},
    # all_reduce
    # llm: MOE
    "moe_gating_gemm": {"test_mode": "single"},
    "moe_softmax_topk": {"test_mode": "single"},
    "moe_scatter_dynamic_quant": {"test_mode": "single"},
    "moe_quant_matmul": {"test_mode": "single"},
    "moe_quant_group_gemm": {"test_mode": "single"},
    "moe_swiglu_dynamic_quant": {"test_mode": "single"},
    "moe_gather": {"test_mode": "single"},
    # llm: ATTN
    "head_rms_norm": {"test_mode": "single"},
    "rotary_embedding": {"test_mode": "single"},
    "store_kv_cache": {"test_mode": "single"},
    "store_paged_kv_cache": {"test_mode": "single"},
    "flash_attention": {"test_mode": "single"},
    "flash_attention_session_cache": {"test_mode": "single"},
    "flash_decoding": {"test_mode": "single"},
    "moe_dispatch_tokens": {"test_mode": "concurrent"},
    # gemm ops
    "quant_matmul": {"test_mode": "single"},
}


def get_op_info(backend_type: str, op_type: str):
    if op_type in OP_INFO_MAPPING:
        if "op_mapping" not in OP_INFO_MAPPING[op_type] or backend_type not in OP_INFO_MAPPING[op_type]["op_mapping"]:
            OP_INFO_MAPPING[op_type]["op_mapping"] = {}
            try:
                backend_ops = importlib.import_module(f"backends.{backend_type}.ops.{op_type}")
                OP_INFO_MAPPING[op_type]["op_mapping"][backend_type] = getattr(backend_ops, "OP_MAPPING")
            except:
                traceback.print_exc()
                OP_INFO_MAPPING[op_type]["op_mapping"][backend_type] = []

            # Check if backend defines custom TEST_MODE override
            # This allows backends like Tenstorrent to use "single" mode for CCL ops
            # (ttnn native CCL uses single-process MeshDevice instead of multi-process gloo)
            test_mode = OP_INFO_MAPPING[op_type]["test_mode"]
            try:
                backend_ops = importlib.import_module(f"backends.{backend_type}.ops.{op_type}")
                if hasattr(backend_ops, "TEST_MODE"):
                    test_mode = getattr(backend_ops, "TEST_MODE")
            except:
                pass

            return test_mode, OP_INFO_MAPPING[op_type]["op_mapping"][backend_type]
    else:
        return "single", []


def create_op_instance(op_cls, args_dict, backend_instance, op_group=None, group_size=1):
    op_instance = op_cls(args_dict, backend_instance, op_group=op_group, group_size=group_size)
    return op_instance
