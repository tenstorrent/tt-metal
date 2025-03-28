# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
import pandas as pd
from collections import defaultdict
from models.demos.llama3_subdevices.tt.llama_common import (
    precompute_freqs,
    PagedAttentionConfig,
)
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.llama3_subdevices.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_subdevices.tt.llama_rope import TtLlamaRotarySetup
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import TransformerBlock
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL
from models.perf.device_perf_utils import run_device_perf
from tt_metal.tools.profiler.process_model_log import (
    get_latest_ops_log_filename,
)
from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "paged_attention",
    (
        # True,
        False,
    ),
    ids=(
        # "paged_attention",
        "default_attention",
    ),
)
@pytest.mark.parametrize(
    "page_params",
    [{"page_block_size": 32, "page_max_num_blocks": 1024}],
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_llama_decoder_inference(
    max_seq_len,
    batch_size,
    paged_attention,
    page_params,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    mesh_device.enable_async(True)

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=True)
    model_args.n_layers = 1
    generation_length = 10
    seqlen = 1
    generation_start_pos = 127
    all_tests_pass = True

    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = TransformerBlock(layer_id=0, args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare page table for paged attention
    page_table_tt = None
    paged_attention_config = None

    if paged_attention:
        paged_attention_config = PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        # Implied shuffling of blocks
        permutation = torch.randperm(paged_attention_config.max_num_blocks)
        # Page table which maps virtual blocks to physical
        reverse_permutation = torch.argsort(permutation)
        page_table = reverse_permutation.reshape(
            model_args.max_batch_size, paged_attention_config.max_num_blocks // model_args.max_batch_size
        )
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, -2) if (model_args.is_galaxy and batch_size > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        n_layers=model_args.n_layers,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )
    # input = torch.randn(1, 32, 4096)
    tt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1

    decode_input = model_args.prepare_residual_tensor_decode(
        tt_decode_input,
        # ttnn.DRAM_MEMORY_CONFIG,
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    # Get cos/sin matrices for the current position of each user
    rot_mats = rope_setup.get_rot_mats(current_pos)
    tt_pf = prefetcher_setup.get_input_tensors()
    ttnn.dram_prefetcher(
        tt_pf,
        num_layers=1,
        global_cb=prefetcher_setup.global_circular_buffer,
    )
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

    res = None
    for i in range(generation_length):
        # Run TT model

        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        res = tt_out
    tt_out = ttnn.to_torch(
        tt_out,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    ttnn.synchronize_device(mesh_device)
    tt_ccl.close()


def merge_device_rows(df):
    block_by_device = defaultdict(list)

    for _, row in df.iterrows():
        op_name = row["OP CODE"]
        op_type = row["OP TYPE"]

        if op_type == "tt_dnn_device":
            device_id = int(row["DEVICE ID"])
            block_by_device[device_id].append((op_name, row.to_dict()))

    device_ids = sorted(block_by_device.keys())
    merged_blocks = []

    global_index = 0
    while max(len(block_by_device[device_id]) for device_id in device_ids) > 0:
        blocks = []
        op_name = None
        missing_devices = []
        for device_id in device_ids:
            if not len(block_by_device[device_id]):
                print(
                    colored(
                        f"Warning: Device {device_id} is missing operation {op_name} at index {global_index}", "yellow"
                    )
                )
                continue
            if op_name is None:
                op_name = block_by_device[device_id][0][0]
            elif op_name != block_by_device[device_id][0][0]:
                missing_devices.append(device_id)
                continue

            blocks.append(block_by_device[device_id].pop(0))

        if missing_devices:
            print(
                colored(
                    f"Warning: {op_name} at index {global_index} not present in CSV for {len(missing_devices)} devices {missing_devices} - do not trust data for this op or directly subsequent ops with the same name",
                    "yellow",
                )
            )

        if not blocks:
            break

        if "AllGather" in op_name or "ReduceScatter" in op_name:
            # For collective ops, take the row with minimum duration
            min_duration_block = min(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(min_duration_block[1])
        else:
            # For non-collective ops, take the row with maximum duration
            max_duration_block = max(blocks, key=lambda x: x[1]["DEVICE KERNEL DURATION [ns]"])
            merged_blocks.append(max_duration_block[1])

        global_index += 1

    return pd.DataFrame(merged_blocks)


@pytest.mark.parametrize(
    "abs_tolerance_ns",
    (1000,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_reduce",
    (30000000,),
)
@pytest.mark.parametrize(
    "abs_tolerance_ns_all_gather",
    (10000,),
)
@pytest.mark.models_device_performance_bare_metal
def test_llama_TG_perf_device(reset_seeds, abs_tolerance_ns, abs_tolerance_ns_all_reduce, abs_tolerance_ns_all_gather):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "tg-llama-decoder"

    batch_size = 32
    subdir = "tg-llama-decoder"
    num_iterations = 1
    generation_length = 10

    command = f"pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_decoder_inference"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    profiler.start("run")
    profiler.start(step_name)
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    profiler.end(step_name)
    profiler.end("run")

    filename = get_latest_ops_log_filename(subdir)

    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)
    # df = df[int(len(df) / generation_length) :] # Excluding first layer
    input_data = df[["OP CODE", "DEVICE KERNEL DURATION [ns]"]].to_dict(orient="records")
    kernel_duration_dict = {}
    for entry in input_data:
        op_code = entry["OP CODE"]
        if op_code in ["Embeddings", "DramPrefetcher"]:
            continue
        duration = entry["DEVICE KERNEL DURATION [ns]"]
        if op_code not in kernel_duration_dict:
            kernel_duration_dict[op_code] = []
        kernel_duration_dict[op_code].append(duration)

    # Average over all generations
    kernel_duration_per_instance_dict = {}
    for op_code in kernel_duration_dict:
        num_ops_with_op_code = len(kernel_duration_dict[op_code])
        num_instances = num_ops_with_op_code // generation_length
        assert num_ops_with_op_code % generation_length == 0
        for iteration_id in range(generation_length):
            for instance_id in range(num_instances):
                op_code_with_id = f"{op_code}_{instance_id}"
                if op_code_with_id not in kernel_duration_per_instance_dict:
                    kernel_duration_per_instance_dict[op_code_with_id] = []
                kernel_duration_per_instance_dict[op_code_with_id].append(
                    kernel_duration_dict[op_code][iteration_id * num_instances + instance_id]
                )

    kernel_duration_per_instance_averaged_dict = {}
    for op_code_with_id in kernel_duration_per_instance_dict:
        kernel_duration_per_instance_averaged_dict[op_code_with_id] = sum(
            kernel_duration_per_instance_dict[op_code_with_id]
        ) / len(kernel_duration_per_instance_dict[op_code_with_id])

    print(kernel_duration_per_instance_averaged_dict)

    expected_times_dict = {
        "LayerNorm_0": 6443.3,
        "LayerNorm_1": 6491.0,
        "LayerNorm_2": 6148.1,
        "LayerNorm_3": 6390.6,
        "AllGatherAsync_0": 2710.3,
        "AllGatherAsync_1": 5837.7,
        "AllGatherAsync_2": 2486.7,
        "ReshardDeviceOperation_0": 1803.7,
        "ReshardDeviceOperation_1": 1851.5,
        "ReshardDeviceOperation_2": 1520.3,
        "Matmul_0": 8429.8,
        "Matmul_1": 8926.8,
        "Matmul_2": 9583.1,
        "Matmul_3": 9631.9,
        "Matmul_4": 17304.9,
        "AllReduceAsync_0": 515885.6,
        "AllReduceAsync_1": 350838.1,
        "AllReduceAsync_2": 3828933.1,
        "AllReduceAsync_3": 133560.4,
        "AllReduceAsync_4": 1007790.5,
        "NLPCreateHeadsDecodeDeviceOperation_0": 8496.5,
        "RotaryEmbeddingLlamaFusedQK_0": 5177.4,
        "PagedUpdateCacheDeviceOperation_0": 4613.0,
        "ScaledDotProductAttentionDecode_0": 19761.8,
        "NLPConcatHeadsDecodeDeviceOperation_0": 6234.9,
        "BinaryDeviceOperation_0": 871.3,
        "BinaryDeviceOperation_1": 13043.2,
        "BinaryDeviceOperation_2": 1860.9,
    }

    mapping_op_code_to_name = {
        "LayerNorm_0": "PreAllGatherLN_0",
        "LayerNorm_1": "PostAllGatherLN_0",
        "LayerNorm_2": "PreAllGatherLN_1",
        "LayerNorm_3": "PostAllGatherLN_1",
        "AllGatherAsync_0": "AllGatherAsync_LN_0",
        "AllGatherAsync_1": "AllGatherAsync_SDPA_0",
        "AllGatherAsync_2": "AllGatherAsync_LN_1",
        "ReshardDeviceOperation_0": "ReshardDeviceOperation_LN_0",
        "ReshardDeviceOperation_1": "ReshardDeviceOperation_CreateHeads",
        "ReshardDeviceOperation_2": "ReshardDeviceOperation_LN_1",
        "Matmul_0": "QKV_MM",
        "Matmul_1": "DO_MM",
        "Matmul_2": "FF1_MM",
        "Matmul_3": "FF3_MM",
        "Matmul_4": "FF2_MM",
        "AllReduceAsync_0": "AllReduceAsync_QKV",
        "AllReduceAsync_1": "AllReduceAsync_DO",
        "AllReduceAsync_2": "AllReduceAsync_FF1",
        "AllReduceAsync_3": "AllReduceAsync_FF3",
        "AllReduceAsync_4": "AllReduceAsync_FF2",
        "NLPCreateHeadsDecodeDeviceOperation_0": "CreateHeads",
        "RotaryEmbeddingLlamaFusedQK_0": "RotaryEmbeddingLlamaFusedQK",
        "PagedUpdateCacheDeviceOperation_0": "PagedUpdateCache",
        "ScaledDotProductAttentionDecode_0": "SDPA",
        "NLPConcatHeadsDecodeDeviceOperation_0": "ConcatHeads",
        "BinaryDeviceOperation_0": "Binary_Residual_0",
        "BinaryDeviceOperation_1": "Binary_Mult_Silu",
        "BinaryDeviceOperation_2": "Binary_Residual_1",
    }

    assert len(kernel_duration_per_instance_averaged_dict) == len(
        expected_times_dict
    ), f"Expected {len(expected_times_dict)} operations, got {len(kernel_duration_per_instance_averaged_dict)}. If the number or type of operations changed, expected times must be updated."

    passing = True
    for op_code_with_id, avg_duration in kernel_duration_per_instance_averaged_dict.items():
        if op_code_with_id in expected_times_dict:
            expected_time = expected_times_dict[op_code_with_id]
            op_name = mapping_op_code_to_name[op_code_with_id]
            benchmark_data.add_measurement(profiler, 0, step_name, op_name, avg_duration)
            if "AllReduceAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_reduce
            elif "AllGatherAsync" in op_code_with_id:
                tolerance = abs_tolerance_ns_all_gather
            else:
                tolerance = abs_tolerance_ns
            if avg_duration > expected_time + tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id}: {avg_duration} ns larger than expected {expected_time} ns by {abs(avg_duration - expected_time)} ns (tolerance {tolerance} ns)"
                )
            elif avg_duration < expected_time - tolerance:
                passing = False
                logger.info(
                    f"{op_code_with_id}: {avg_duration} ns smaller than expected {expected_time} ns by {abs(expected_time - avg_duration)} ns (tolerance {tolerance} ns)"
                )
        else:
            passing = False
            logger.info(f"Warning: {op_code_with_id} not found in expected_times_dict")

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"tg-llama-decoder",
        ml_model_name="llama70b-tg-decoder",
    )

    assert passing
