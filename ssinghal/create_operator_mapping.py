import json


def create_pytorch_to_ttnn_mapping():
    """Create comprehensive mapping from PyTorch operators to TTNN equivalents"""

    # Comprehensive mapping based on TTNN documentation and common PyTorch operations
    operator_mapping = {
        # Tensor operations
        "add": {
            "ttnn_function": "ttnn.add",
            "torch_function": "torch.add",
            "num_inputs": 2,
            "category": "elementwise",
            "supported": True,
        },
        "mul": {
            "ttnn_function": "ttnn.mul",
            "torch_function": "torch.mul",
            "num_inputs": 2,
            "category": "elementwise",
            "supported": True,
        },
        "div": {
            "ttnn_function": "ttnn.div",
            "torch_function": "torch.div",
            "num_inputs": 2,
            "category": "elementwise",
            "supported": True,
        },
        "clone": {
            "ttnn_function": "ttnn.clone",
            "torch_function": "torch.clone",
            "num_inputs": 1,
            "category": "tensor_ops",
            "supported": True,
        },
        "copy": {
            "ttnn_function": "ttnn.copy",
            "torch_function": "torch.copy_",
            "num_inputs": 2,
            "category": "tensor_ops",
            "supported": True,
        },
        # Activation functions
        "relu": {
            "ttnn_function": "ttnn.relu",
            "torch_function": "torch.relu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "silu": {
            "ttnn_function": "ttnn.silu",
            "torch_function": "torch.nn.functional.silu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "sigmoid": {
            "ttnn_function": "ttnn.sigmoid",
            "torch_function": "torch.sigmoid",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "tanh": {
            "ttnn_function": "ttnn.tanh",
            "torch_function": "torch.tanh",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "softmax": {
            "ttnn_function": "ttnn.softmax",
            "torch_function": "torch.softmax",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "softplus": {
            "ttnn_function": "ttnn.softplus",
            "torch_function": "torch.nn.functional.softplus",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "leakyrelu": {
            "ttnn_function": "ttnn.leaky_relu",
            "torch_function": "torch.nn.functional.leaky_relu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "hardtanh": {
            "ttnn_function": "ttnn.hardtanh",
            "torch_function": "torch.nn.functional.hardtanh",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "GELU": {
            "ttnn_function": "ttnn.gelu",
            "torch_function": "torch.nn.functional.gelu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "GELUActivation": {
            "ttnn_function": "ttnn.gelu",
            "torch_function": "torch.nn.functional.gelu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "SiLU": {
            "ttnn_function": "ttnn.silu",
            "torch_function": "torch.nn.functional.silu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "ReLU": {
            "ttnn_function": "ttnn.relu",
            "torch_function": "torch.nn.functional.relu",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "Sigmoid": {
            "ttnn_function": "ttnn.sigmoid",
            "torch_function": "torch.nn.functional.sigmoid",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        "Mish": {
            "ttnn_function": "ttnn.mish",
            "torch_function": "torch.nn.functional.mish",
            "num_inputs": 1,
            "category": "activation",
            "supported": True,
        },
        # Shape operations
        "view": {
            "ttnn_function": "ttnn.reshape",
            "torch_function": "torch.view",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Requires shape parameter",
        },
        "unsafeview": {
            "ttnn_function": "ttnn.reshape",
            "torch_function": "torch.view",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Requires shape parameter",
        },
        "permute": {
            "ttnn_function": "ttnn.permute",
            "torch_function": "torch.permute",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Requires permutation dims",
        },
        "transpose": {
            "ttnn_function": "ttnn.transpose",
            "torch_function": "torch.transpose",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Requires dim0, dim1",
        },
        "unsqueeze": {
            "ttnn_function": "ttnn.unsqueeze",
            "torch_function": "torch.unsqueeze",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Requires dim parameter",
        },
        "expand": {
            "ttnn_function": "ttnn.expand",
            "torch_function": "torch.expand",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Requires size parameter",
        },
        # Linear algebra
        "mm": {
            "ttnn_function": "ttnn.matmul",
            "torch_function": "torch.mm",
            "num_inputs": 2,
            "category": "linalg",
            "supported": True,
        },
        "bmm": {
            "ttnn_function": "ttnn.matmul",
            "torch_function": "torch.bmm",
            "num_inputs": 2,
            "category": "linalg",
            "supported": True,
        },
        "addmm": {
            "ttnn_function": "ttnn.addmm",
            "torch_function": "torch.addmm",
            "num_inputs": 3,
            "category": "linalg",
            "supported": True,
        },
        "Linear": {
            "ttnn_function": "ttnn.linear",
            "torch_function": "torch.nn.functional.linear",
            "num_inputs": 2,
            "category": "linalg",
            "supported": True,
            "note": "Requires weight, optional bias",
        },
        # Reduction operations
        "mean": {
            "ttnn_function": "ttnn.mean",
            "torch_function": "torch.mean",
            "num_inputs": 1,
            "category": "reduction",
            "supported": True,
        },
        "linalgvectornorm": {
            "ttnn_function": "ttnn.norm",
            "torch_function": "torch.linalg.vector_norm",
            "num_inputs": 1,
            "category": "reduction",
            "supported": True,
        },
        # Concatenation and stacking
        "cat": {
            "ttnn_function": "ttnn.concat",
            "torch_function": "torch.cat",
            "num_inputs": "variable",
            "category": "concat",
            "supported": True,
            "note": "Requires dim parameter",
        },
        "Concat": {
            "ttnn_function": "ttnn.concat",
            "torch_function": "torch.cat",
            "num_inputs": "variable",
            "category": "concat",
            "supported": True,
        },
        "stack": {
            "ttnn_function": "ttnn.stack",
            "torch_function": "torch.stack",
            "num_inputs": "variable",
            "category": "concat",
            "supported": True,
        },
        "splitwithsizes": {
            "ttnn_function": "ttnn.split",
            "torch_function": "torch.split",
            "num_inputs": 1,
            "category": "concat",
            "supported": True,
            "note": "Requires split_sizes, dim",
        },
        # Clamping operations
        "clamp": {
            "ttnn_function": "ttnn.clip",
            "torch_function": "torch.clamp",
            "num_inputs": 1,
            "category": "elementwise",
            "supported": True,
            "note": "Requires min, max",
        },
        "clampmin": {
            "ttnn_function": "ttnn.clip",
            "torch_function": "torch.clamp_min",
            "num_inputs": 1,
            "category": "elementwise",
            "supported": True,
            "note": "Requires min value",
        },
        # Pooling operations
        "MaxPool2d": {
            "ttnn_function": "ttnn.max_pool2d",
            "torch_function": "torch.nn.functional.max_pool2d",
            "num_inputs": 1,
            "category": "pooling",
            "supported": True,
            "note": "Requires kernel_size",
        },
        "AdaptiveAvgPool2d": {
            "ttnn_function": "ttnn.adaptive_avg_pool2d",
            "torch_function": "torch.nn.functional.adaptive_avg_pool2d",
            "num_inputs": 1,
            "category": "pooling",
            "supported": False,
            "note": "May not be directly supported",
        },
        "AdaptiveAvgPool1d": {
            "ttnn_function": "ttnn.adaptive_avg_pool1d",
            "torch_function": "torch.nn.functional.adaptive_avg_pool1d",
            "num_inputs": 1,
            "category": "pooling",
            "supported": False,
            "note": "May not be directly supported",
        },
        "AdaptiveMaxPool2d": {
            "ttnn_function": "ttnn.adaptive_max_pool2d",
            "torch_function": "torch.nn.functional.adaptive_max_pool2d",
            "num_inputs": 1,
            "category": "pooling",
            "supported": False,
            "note": "May not be directly supported",
        },
        # Upsampling
        "Upsample": {
            "ttnn_function": "ttnn.upsample",
            "torch_function": "torch.nn.functional.interpolate",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Nearest mode supported; input expected NHWC on device",
        },
        "upsamplenearest2d": {
            "ttnn_function": "ttnn.upsample",
            "torch_function": "torch.nn.functional.interpolate",
            "num_inputs": 1,
            "category": "shape",
            "supported": True,
            "note": "Maps to ttnn.upsample with nearest mode",
        },
        # Other operations
        "Dropout": {
            "ttnn_function": "ttnn.dropout",
            "torch_function": "torch.nn.functional.dropout",
            "num_inputs": 1,
            "category": "regularization",
            "supported": True,
        },
        "log": {
            "ttnn_function": "ttnn.log",
            "torch_function": "torch.log",
            "num_inputs": 1,
            "category": "elementwise",
            "supported": True,
        },
        "roll": {
            "ttnn_function": "ttnn.roll",
            "torch_function": "torch.roll",
            "num_inputs": 1,
            "category": "shape",
            "supported": False,
            "note": "May not be directly supported",
        },
        "topk": {
            "ttnn_function": "ttnn.topk",
            "torch_function": "torch.topk",
            "num_inputs": 1,
            "category": "reduction",
            "supported": True,
        },
        "constantpadnd": {
            "ttnn_function": "ttnn.pad",
            "torch_function": "torch.nn.functional.pad",
            "num_inputs": 1,
            "category": "shape",
            "supported": False,
            "note": "May not be directly supported",
        },
        # Meta operations (likely not directly testable)
        "Tensor": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 0,
            "category": "meta",
            "supported": False,
            "note": "Meta operation - tensor creation",
        },
        "TensorScalar": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 0,
            "category": "meta",
            "supported": False,
            "note": "Meta operation - scalar tensor",
        },
        "Scalar": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 0,
            "category": "meta",
            "supported": False,
            "note": "Meta operation - scalar value",
        },
        "int": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 0,
            "category": "meta",
            "supported": False,
            "note": "Type conversion",
        },
        "dim": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 0,
            "category": "meta",
            "supported": False,
            "note": "Dimension query",
        },
        "asstrided": {
            "ttnn_function": None,
            "torch_function": "torch.as_strided",
            "num_inputs": 1,
            "category": "meta",
            "supported": False,
            "note": "Low-level stride manipulation",
        },
        # Special operations
        "Identity": {
            "ttnn_function": "ttnn.identity",
            "torch_function": "torch.nn.Identity",
            "num_inputs": 1,
            "category": "special",
            "supported": True,
        },
        "SwinDropPath": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 1,
            "category": "special",
            "supported": False,
            "note": "Swin transformer specific dropout",
        },
        "ContrastiveHead": {
            "ttnn_function": None,
            "torch_function": None,
            "num_inputs": 1,
            "category": "special",
            "supported": False,
            "note": "Custom head layer",
        },
    }

    return operator_mapping


def analyze_mapping():
    """Analyze the operator mapping and generate statistics"""

    mapping = create_pytorch_to_ttnn_mapping()

    # Load basic operators data
    with open("ssinghal/basic_operators.json", "r") as f:
        basic_ops = json.load(f)

    stats = {
        "total_operators": len(basic_ops),
        "supported": 0,
        "unsupported": 0,
        "meta_operations": 0,
        "by_category": {},
    }

    detailed_analysis = {}

    for op_name, op_data in basic_ops.items():
        if op_name in mapping:
            op_mapping = mapping[op_name]
            detailed_analysis[op_name] = {"occurrences": len(op_data), "mapping": op_mapping}

            category = op_mapping["category"]
            if category not in stats["by_category"]:
                stats["by_category"][category] = {"supported": 0, "unsupported": 0}

            if op_mapping["supported"]:
                stats["supported"] += 1
                stats["by_category"][category]["supported"] += 1
            else:
                stats["unsupported"] += 1
                stats["by_category"][category]["unsupported"] += 1

            if category == "meta":
                stats["meta_operations"] += 1
        else:
            # Operator not in mapping
            detailed_analysis[op_name] = {"occurrences": len(op_data), "mapping": None, "note": "No mapping defined"}
            stats["unsupported"] += 1

    return mapping, stats, detailed_analysis


def main():
    print("Creating PyTorch to TTNN operator mapping...")

    mapping, stats, detailed_analysis = analyze_mapping()

    # Save mapping and analysis
    with open("ssinghal/operator_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    with open("ssinghal/mapping_analysis.json", "w") as f:
        json.dump({"statistics": stats, "detailed_analysis": detailed_analysis}, f, indent=2)

    # Print summary
    print(f"\n=== OPERATOR MAPPING SUMMARY ===")
    print(f"Total operators: {stats['total_operators']}")
    print(f"Supported: {stats['supported']}")
    print(f"Unsupported: {stats['unsupported']}")
    print(f"Meta operations: {stats['meta_operations']}")

    print(f"\n=== BY CATEGORY ===")
    for category, counts in stats["by_category"].items():
        total = counts["supported"] + counts["unsupported"]
        print(f"{category}: {counts['supported']}/{total} supported")

    # Show supported operators for testing
    print(f"\n=== SUPPORTED OPERATORS FOR TESTING ===")
    testable_ops = []
    for op_name, analysis in detailed_analysis.items():
        if analysis.get("mapping") and analysis["mapping"]["supported"] and analysis["mapping"]["category"] != "meta":
            testable_ops.append((op_name, analysis["occurrences"]))

    testable_ops.sort(key=lambda x: x[1], reverse=True)  # Sort by occurrence count

    for op_name, count in testable_ops:
        print(f"  {op_name}: {count} occurrences")

    print(f"\nTotal testable operators: {len(testable_ops)}")
    print(f"\nFiles saved:")
    print(f"  - ssinghal/operator_mapping.json")
    print(f"  - ssinghal/mapping_analysis.json")


if __name__ == "__main__":
    main()
