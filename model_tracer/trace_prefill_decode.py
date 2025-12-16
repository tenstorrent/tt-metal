# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Prefill/Decode Operations Tracer

Captures TTNN operations separately for prefill and decode phases of LLM inference.
This allows you to analyze what operations are executed during each phase and
understand the differences in tensor shapes, memory configurations, etc.

Usage:
    # Basic usage (uses HF_MODEL env var)
    HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python model_tracer/trace_prefill_decode.py

    # With specific model
    python model_tracer/trace_prefill_decode.py --model meta-llama/Llama-3.2-1B-Instruct

    # With custom parameters
    python model_tracer/trace_prefill_decode.py --model meta-llama/Llama-3.2-1B-Instruct \
        --batch-size 1 --max-seq-len 1024 --num-decode-iterations 5

    # Trace only prefill or decode
    python model_tracer/trace_prefill_decode.py --mode prefill
    python model_tracer/trace_prefill_decode.py --mode decode

    # Use fewer layers for faster tracing
    python model_tracer/trace_prefill_decode.py --num-layers 4

Output:
    - model_tracer/traced_operations/prefill_ops_<model>_<timestamp>.json
    - model_tracer/traced_operations/decode_ops_<model>_<timestamp>.json
    - model_tracer/traced_operations/trace_summary_<model>_<timestamp>.txt
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger


def get_base_dir():
    """Get the tt-metal base directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from model_tracer to tt-metal
    return os.path.dirname(current_dir)


BASE_DIR = get_base_dir()


def load_valid_operations():
    """Load valid operations from Allops.txt for filtering"""
    valid_ops = set()
    allops_file = os.path.join(BASE_DIR, "tests/sweep_framework/Allops.txt")

    # Operations to exclude from tracing
    excluded_operations = {
        "ttnn::unary_chain",
        "ttnn::view",
        "ttnn::pearson_correlation_coefficient",
        "ttnn::dump_tensor",
        "ttnn::dram_prefetcher",
        "ttnn::complex_tensor",
        "ttnn::as_tensor",
        "ttnn::allocate_tensor_on_device",
        "ttnn::to_device",
        "ttnn::to_dtype",
        "ttnn::to_layout",
        "ttnn::to_memory_config",
        "ttnn::to_torch",
        "ttnn::prim::binary",
        "ttnn::prim::example",
        "ttnn::prim::example_multiple_return",
        "ttnn::from_device",
        "ttnn::from_torch",
        "ttnn::composite_example",
        "ttnn::composite_example_multiple_return",
        "ttnn::deallocate",
        "ttnn::move",
        "ttnn::reallocate",
        "ttnn::load_tensor",
    }

    try:
        with open(allops_file, "r") as f:
            for line in f:
                op_name = line.strip()
                if op_name:
                    op_name_colons = op_name.replace(".", "::")
                    if op_name_colons not in excluded_operations:
                        valid_ops.add(op_name_colons)
        logger.info(f"Loaded {len(valid_ops)} valid operations from Allops.txt")
        return valid_ops
    except FileNotFoundError:
        logger.warning(f"Allops.txt not found, using prefix filtering")
        return None


def is_valid_operation(op_name, valid_ops):
    """Check if operation is valid for tracing"""
    if valid_ops is None:
        return op_name.startswith("ttnn::") or op_name.startswith("ttnn::experimental::")
    return op_name in valid_ops


def clean_operation_data(operation):
    """Clean operation data to ensure it's JSON serializable"""
    if not isinstance(operation, dict):
        return None

    def clean_recursive(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                try:
                    cleaned_value = clean_recursive(value)
                    json.dumps(cleaned_value)
                    cleaned[key] = cleaned_value
                except Exception as e:
                    value_str = str(value)
                    if value_str.startswith("{") and value_str.endswith("}"):
                        try:
                            import re

                            fixed_json_str = value_str
                            fixed_json_str = re.sub(r':\s*"{\s*([^}]+)\s*}"', r': "[\1]"', fixed_json_str)
                            fixed_json_str = re.sub(
                                r'"grid"\s*:\s*\{(\[.*?\](?:\s*,\s*\[.*?\])*)\}', r'"grid":[\1]', fixed_json_str
                            )
                            fixed_json_str = re.sub(r"(\{[^}]+\})\s*-\s*(\{[^}]+\})", r"\1, \2", fixed_json_str)
                            parsed_data = json.loads(fixed_json_str)
                            cleaned[key] = clean_recursive(parsed_data)
                            continue
                        except:
                            pass
                    cleaned[key] = {
                        "UnparsedElement": {
                            "error": str(e),
                            "element_info": value_str[:500] if len(value_str) > 500 else value_str,
                        }
                    }
            return cleaned
        elif isinstance(obj, list):
            return [clean_recursive(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    try:
        cleaned_op = clean_recursive(operation)
        json.dumps(cleaned_op)
        return cleaned_op
    except (TypeError, ValueError) as e:
        return {
            "operation": str(operation.get("operation", "unknown")),
            "arguments": [],
            "error": f"Serialization failure: {str(e)}",
        }


def filter_and_clean_operations(trace_data, valid_ops):
    """Filter trace data to only include valid TTNN operations"""
    if not isinstance(trace_data, dict) or "content" not in trace_data:
        return []

    filtered_operations = []
    for op in trace_data["content"]:
        if isinstance(op, dict) and "operation" in op:
            op_name = op["operation"]
            if is_valid_operation(op_name, valid_ops):
                cleaned_op = clean_operation_data(op)
                if cleaned_op:
                    filtered_operations.append(cleaned_op)

    return filtered_operations


def analyze_operations(operations, phase_name):
    """Analyze and summarize operations for a phase"""
    op_counts = defaultdict(int)
    op_shapes = defaultdict(list)

    for op in operations:
        op_name = op.get("operation", "unknown")
        op_counts[op_name] += 1

        # Try to extract tensor shapes from arguments
        args = op.get("arguments", [])
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    if isinstance(value, dict) and "Tensor" in value:
                        tensor_info = value["Tensor"]
                        if "tensor_spec" in tensor_info:
                            shape = tensor_info["tensor_spec"].get("logical_shape", [])
                            if shape:
                                op_shapes[op_name].append(shape)

    summary = {
        "phase": phase_name,
        "total_operations": len(operations),
        "unique_operations": len(op_counts),
        "operation_counts": dict(sorted(op_counts.items(), key=lambda x: x[1], reverse=True)),
        "sample_shapes": {k: v[:3] for k, v in op_shapes.items()},  # Keep first 3 shapes per op
    }

    return summary


def print_summary(summary, file=None):
    """Print operation summary"""
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"📊 {summary['phase']} Phase Summary")
    output.append(f"{'='*60}")
    output.append(f"Total operations: {summary['total_operations']}")
    output.append(f"Unique operation types: {summary['unique_operations']}")
    output.append(f"\n🔧 Operation counts:")

    for op_name, count in list(summary["operation_counts"].items())[:20]:
        output.append(f"  {op_name}: {count}x")

    if len(summary["operation_counts"]) > 20:
        output.append(f"  ... and {len(summary['operation_counts']) - 20} more operation types")

    if summary["sample_shapes"]:
        output.append(f"\n📐 Sample tensor shapes (first 3 per operation):")
        for op_name, shapes in list(summary["sample_shapes"].items())[:10]:
            output.append(f"  {op_name}: {shapes}")

    output.append(f"{'='*60}\n")

    text = "\n".join(output)
    print(text)
    if file:
        file.write(text + "\n")


def setup_mesh_device(num_devices=None, trace_region_size=50000000):
    """Set up mesh device for inference"""
    import ttnn

    # Determine mesh shape based on available devices or environment
    mesh_device_env = os.environ.get("MESH_DEVICE")
    mesh_shapes = {
        "N150": (1, 1),
        "N300": (1, 2),
        "N150x4": (1, 4),
        "T3K": (1, 8),
        "TG": (8, 4),
        "P150": (1, 1),
        "P300": (1, 2),
        "P150x4": (1, 4),
        "P150x8": (1, 8),
        "BHGLX": (8, 4),
    }

    if mesh_device_env and mesh_device_env in mesh_shapes:
        mesh_shape = ttnn.MeshShape(*mesh_shapes[mesh_device_env])
        logger.info(f"Using MESH_DEVICE={mesh_device_env} -> shape {mesh_shapes[mesh_device_env]}")
    elif num_devices:
        mesh_shape = ttnn.MeshShape(1, num_devices)
        logger.info(f"Using {num_devices} devices")
    else:
        available_devices = len(ttnn.get_device_ids())
        mesh_shape = ttnn.MeshShape(1, available_devices)
        logger.info(f"Using all available devices: {available_devices}")

    # Check if we need fabric config
    total_devices = mesh_shape.num_rows * mesh_shape.num_cols
    is_multi_device = total_devices > 1

    device_params = {
        "trace_region_size": trace_region_size,
        "num_command_queues": 1,
    }

    if is_multi_device:
        # Set up fabric for multi-device
        fabric_config = ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(fabric_config)
        logger.info(f"Enabled fabric config: {fabric_config}")

    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **device_params)
    logger.info(f"Opened mesh device with {mesh_device.get_num_devices()} devices")

    return mesh_device


def close_mesh_device(mesh_device):
    """Close mesh device and cleanup"""
    import ttnn

    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric if it was set
    try:
        ttnn.set_fabric_config(None)
    except:
        pass


def run_traced_inference(
    model_name,
    batch_size=1,
    max_seq_len=1024,
    max_generated_tokens=200,
    num_decode_iterations=5,
    num_layers=None,
    mode="both",
    output_dir=None,
):
    """
    Run inference with graph tracing to capture prefill and decode operations.

    Args:
        model_name: HuggingFace model name
        batch_size: Batch size for inference
        max_seq_len: Maximum sequence length
        max_generated_tokens: Maximum tokens to generate
        num_decode_iterations: Number of decode iterations to trace
        num_layers: Number of layers to use (None = all)
        mode: "prefill", "decode", or "both"
        output_dir: Directory to save trace files

    Returns:
        dict: Contains prefill_ops, decode_ops, and summaries
    """
    import ttnn
    from ttnn.graph_tracer_utils import GraphTracerUtils

    # Import model-related modules
    from models.tt_transformers.tt.common import (
        PagedAttentionConfig,
        create_tt_model,
        preprocess_inputs_prefill,
    )
    from models.tt_transformers.tt.generator import Generator, create_submeshes
    from models.tt_transformers.tt.model_config import DecodersPrecision

    # Set HF_MODEL environment variable
    os.environ["HF_MODEL"] = model_name

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "model_tracer/traced_operations")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split("/")[-1].replace("-", "_")

    # Load valid operations for filtering
    valid_ops = load_valid_operations()

    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "num_layers": num_layers,
        "timestamp": timestamp,
        "prefill_ops": [],
        "decode_ops": [],
        "prefill_summary": None,
        "decode_summary": None,
    }

    mesh_device = None

    try:
        # Setup mesh device
        logger.info("Setting up mesh device...")
        mesh_device = setup_mesh_device()
        num_devices = mesh_device.get_num_devices()

        # Model parameters
        instruct = True
        data_parallel = 1
        global_batch_size = batch_size * data_parallel
        paged_attention = True
        page_params = {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024}

        # Create optimization function
        optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

        # Setup model
        logger.info(f"Setting up model: {model_name}...")
        submesh_devices = create_submeshes(mesh_device, data_parallel)
        state_dict = None

        model_args_list = []
        model_list = []
        tt_kv_cache_list = []

        paged_attention_config = (
            PagedAttentionConfig(
                block_size=page_params["page_block_size"],
                max_num_blocks=page_params["page_max_num_blocks_per_dp"],
            )
            if paged_attention
            else None
        )

        for submesh in submesh_devices:
            model_args_i, model_i, tt_kv_cache_i, state_dict = create_tt_model(
                submesh,
                instruct=instruct,
                max_batch_size=global_batch_size // data_parallel,
                optimizations=optimizations,
                max_seq_len=max_seq_len,
                paged_attention_config=paged_attention_config,
                dtype=ttnn.bfloat8_b,
                state_dict=state_dict,
                num_layers=num_layers,
            )
            model_args_list.append(model_args_i)
            model_list.append(model_i)
            tt_kv_cache_list.append(tt_kv_cache_i)

        # Create page table
        page_table = None
        if paged_attention_config:
            permutation = torch.randperm(paged_attention_config.max_num_blocks)
            reverse_permutation = torch.argsort(permutation).repeat(data_parallel)
            page_table = reverse_permutation.reshape(
                global_batch_size, paged_attention_config.max_num_blocks // (global_batch_size // data_parallel)
            )

        tokenizer = model_args_list[0].tokenizer
        processor = model_args_list[0].processor

        # Create generator
        generator = Generator(model_list, model_args_list, mesh_device, processor=processor, tokenizer=tokenizer)

        # Create sample input prompts
        input_prompts = ["What is the capital of France? Please explain in detail."] * global_batch_size

        # Preprocess inputs
        logger.info("Preprocessing inputs...")
        (
            input_tokens_prefill_pt,
            encoded_prompts,
            decoding_pos,
            prefill_lens,
        ) = preprocess_inputs_prefill(
            input_prompts, tokenizer, model_args_list, instruct, max_generated_tokens, max_prefill_len=max_seq_len
        )

        input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

        # ============================================
        # PREFILL PHASE TRACING
        # ============================================
        if mode in ["prefill", "both"]:
            logger.info("=" * 60)
            logger.info("🔍 Starting PREFILL phase tracing...")
            logger.info("=" * 60)

            # Warmup run (not traced)
            logger.info("Running prefill warmup (not traced)...")
            _ = generator.prefill_forward_text(
                input_tokens_prefill_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache_list,
                prompt_lens=decoding_pos,
            )

            # Reset KV cache for traced run
            for i in range(len(model_list)):
                for layer in model_list[i].layers:
                    k_cache, v_cache = layer.attention.layer_past
                    k_cache = ttnn.mul(k_cache, 0, output_tensor=k_cache)
                    v_cache = ttnn.mul(v_cache, 0, output_tensor=v_cache)
            generator.prev_page_table = None

            # Traced prefill run
            logger.info("Running prefill with tracing enabled...")
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

            prefill_logits = generator.prefill_forward_text(
                input_tokens_prefill_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache_list,
                prompt_lens=decoding_pos,
            )

            prefill_graph = ttnn.graph.end_graph_capture()
            prefill_data = GraphTracerUtils.serialize_graph(prefill_graph)

            # Filter and clean prefill operations
            results["prefill_ops"] = filter_and_clean_operations(prefill_data, valid_ops)
            results["prefill_summary"] = analyze_operations(results["prefill_ops"], "PREFILL")

            logger.info(f"✅ Captured {len(results['prefill_ops'])} prefill operations")
            print_summary(results["prefill_summary"])

            # Save prefill operations
            prefill_file = os.path.join(output_dir, f"prefill_ops_{model_short_name}_{timestamp}.json")
            with open(prefill_file, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "model_name": model_name,
                            "phase": "prefill",
                            "batch_size": batch_size,
                            "max_seq_len": max_seq_len,
                            "num_layers": num_layers or "all",
                            "prefill_len": prefill_lens[0],
                            "timestamp": timestamp,
                        },
                        "operations": results["prefill_ops"],
                        "summary": results["prefill_summary"],
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"💾 Saved prefill operations to: {prefill_file}")

            # Get first token for decode
            prefilled_token = torch.argmax(prefill_logits, dim=-1)
        else:
            # Skip prefill, generate dummy token
            prefilled_token = torch.zeros(global_batch_size, 1, dtype=torch.long)

        # ============================================
        # DECODE PHASE TRACING
        # ============================================
        if mode in ["decode", "both"]:
            logger.info("=" * 60)
            logger.info("🔍 Starting DECODE phase tracing...")
            logger.info("=" * 60)

            current_pos = torch.tensor([decoding_pos[b] for b in range(global_batch_size)])
            out_tok = prefilled_token

            # Warmup decode run (not traced)
            logger.info("Running decode warmup (not traced)...")
            _ = generator.decode_forward_text(
                out_tok,
                current_pos,
                enable_trace=False,  # Disable internal tracing for warmup
                page_table=page_table,
                kv_cache=tt_kv_cache_list,
            )
            current_pos += 1

            # Traced decode iterations
            all_decode_ops = []

            for iteration in range(num_decode_iterations):
                logger.info(f"Running decode iteration {iteration + 1}/{num_decode_iterations} with tracing...")

                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

                decode_logits, _ = generator.decode_forward_text(
                    out_tok,
                    current_pos,
                    enable_trace=False,  # Disable internal tracing to get clean capture
                    page_table=page_table,
                    kv_cache=tt_kv_cache_list,
                )

                decode_graph = ttnn.graph.end_graph_capture()
                decode_data = GraphTracerUtils.serialize_graph(decode_graph)

                # Filter and collect decode operations
                iter_ops = filter_and_clean_operations(decode_data, valid_ops)
                all_decode_ops.extend(iter_ops)

                logger.info(f"  Iteration {iteration + 1}: captured {len(iter_ops)} operations")

                # Update for next iteration
                out_tok = torch.argmax(decode_logits, dim=-1).unsqueeze(1)
                current_pos += 1

            results["decode_ops"] = all_decode_ops
            results["decode_summary"] = analyze_operations(results["decode_ops"], "DECODE")

            logger.info(
                f"✅ Captured {len(results['decode_ops'])} decode operations across {num_decode_iterations} iterations"
            )
            print_summary(results["decode_summary"])

            # Save decode operations
            decode_file = os.path.join(output_dir, f"decode_ops_{model_short_name}_{timestamp}.json")
            with open(decode_file, "w") as f:
                json.dump(
                    {
                        "metadata": {
                            "model_name": model_name,
                            "phase": "decode",
                            "batch_size": batch_size,
                            "max_seq_len": max_seq_len,
                            "num_layers": num_layers or "all",
                            "num_iterations": num_decode_iterations,
                            "timestamp": timestamp,
                        },
                        "operations": results["decode_ops"],
                        "summary": results["decode_summary"],
                    },
                    f,
                    indent=2,
                    default=str,
                )
            logger.info(f"💾 Saved decode operations to: {decode_file}")

        # ============================================
        # SAVE COMBINED SUMMARY
        # ============================================
        summary_file = os.path.join(output_dir, f"trace_summary_{model_short_name}_{timestamp}.txt")
        with open(summary_file, "w") as f:
            f.write(f"TTNN Operations Trace Summary\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Max Seq Len: {max_seq_len}\n")
            f.write(f"Num Layers: {num_layers or 'all'}\n")
            f.write(f"\n")

            if results["prefill_summary"]:
                print_summary(results["prefill_summary"], f)

            if results["decode_summary"]:
                print_summary(results["decode_summary"], f)

            # Compare prefill vs decode operations
            if results["prefill_summary"] and results["decode_summary"]:
                f.write("\n" + "=" * 60 + "\n")
                f.write("📊 PREFILL vs DECODE Comparison\n")
                f.write("=" * 60 + "\n")

                prefill_ops_set = set(results["prefill_summary"]["operation_counts"].keys())
                decode_ops_set = set(results["decode_summary"]["operation_counts"].keys())

                common_ops = prefill_ops_set & decode_ops_set
                prefill_only = prefill_ops_set - decode_ops_set
                decode_only = decode_ops_set - prefill_ops_set

                f.write(f"\nOperations in BOTH phases ({len(common_ops)}):\n")
                for op in sorted(common_ops):
                    p_count = results["prefill_summary"]["operation_counts"][op]
                    d_count = results["decode_summary"]["operation_counts"][op]
                    f.write(f"  {op}: prefill={p_count}x, decode={d_count}x\n")

                if prefill_only:
                    f.write(f"\nOperations ONLY in PREFILL ({len(prefill_only)}):\n")
                    for op in sorted(prefill_only):
                        f.write(f"  {op}: {results['prefill_summary']['operation_counts'][op]}x\n")

                if decode_only:
                    f.write(f"\nOperations ONLY in DECODE ({len(decode_only)}):\n")
                    for op in sorted(decode_only):
                        f.write(f"  {op}: {results['decode_summary']['operation_counts'][op]}x\n")

        logger.info(f"💾 Saved summary to: {summary_file}")

        return results

    except Exception as e:
        logger.error(f"Error during tracing: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Cleanup
        if mesh_device is not None:
            logger.info("Closing mesh device...")
            close_mesh_device(mesh_device)


def main():
    parser = argparse.ArgumentParser(
        description="Trace TTNN operations for prefill and decode phases separately",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with environment variable
    HF_MODEL=meta-llama/Llama-3.2-1B-Instruct python model_tracer/trace_prefill_decode.py

    # Specify model directly
    python model_tracer/trace_prefill_decode.py --model meta-llama/Llama-3.2-1B-Instruct

    # Trace only prefill phase
    python model_tracer/trace_prefill_decode.py --mode prefill

    # Use fewer layers for faster tracing
    python model_tracer/trace_prefill_decode.py --num-layers 4

    # Custom decode iterations
    python model_tracer/trace_prefill_decode.py --num-decode-iterations 10
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
        help="HuggingFace model name (default: HF_MODEL env var or Llama-3.2-1B-Instruct)",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size for inference (default: 1)")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Maximum sequence length (default: 1024)")
    parser.add_argument(
        "--max-generated-tokens", type=int, default=200, help="Maximum tokens to generate (default: 200)"
    )
    parser.add_argument(
        "--num-decode-iterations", type=int, default=3, help="Number of decode iterations to trace (default: 3)"
    )
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers to use (default: all layers)")
    parser.add_argument(
        "--mode", choices=["prefill", "decode", "both"], default="both", help="Which phase(s) to trace (default: both)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for trace files (default: model_tracer/traced_operations)",
    )

    args = parser.parse_args()

    print("🚀 TTNN Prefill/Decode Operations Tracer")
    print("=" * 60)
    print(f"📦 Model: {args.model}")
    print(f"📊 Batch Size: {args.batch_size}")
    print(f"📏 Max Seq Len: {args.max_seq_len}")
    print(f"🔄 Decode Iterations: {args.num_decode_iterations}")
    print(f"📚 Num Layers: {args.num_layers or 'all'}")
    print(f"🎯 Mode: {args.mode}")
    print("=" * 60)

    try:
        results = run_traced_inference(
            model_name=args.model,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_generated_tokens=args.max_generated_tokens,
            num_decode_iterations=args.num_decode_iterations,
            num_layers=args.num_layers,
            mode=args.mode,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 60)
        print("✅ Tracing completed successfully!")
        print("=" * 60)

        if results["prefill_summary"]:
            print(
                f"📊 Prefill: {results['prefill_summary']['total_operations']} operations, "
                f"{results['prefill_summary']['unique_operations']} unique types"
            )

        if results["decode_summary"]:
            print(
                f"📊 Decode: {results['decode_summary']['total_operations']} operations, "
                f"{results['decode_summary']['unique_operations']} unique types"
            )

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
