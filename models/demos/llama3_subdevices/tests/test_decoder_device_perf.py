# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import time
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
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
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

    generation_start_pos = 127
    generation_length = 2
    all_tests_pass = True

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

    seqlen = 1

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
    for i in range(generation_length):
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

        # Run TT model
        res = None
        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        # for device in mesh_device.get_devices():
        #     ttnn.DumpDeviceProfiler(device)
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


@pytest.mark.models_device_performance_bare_metal
def test_llama_TG_perf_device(reset_seeds):
    profiler = BenchmarkProfiler()
    benchmark_data = BenchmarkData()
    step_name = "llama-80-decoder"

    batch_size = 32
    subdir = "llama-80-decoder"
    margin = 0.03
    num_iterations = 1

    command = f"pytest models/demos/llama3_subdevices/tests/test_decoder_device_perf.py::test_llama_decoder_inference"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    profiler.start("run")
    profiler.start(step_name)
    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    profiler.end(step_name)
    profiler.end("run")

    filename = get_latest_ops_log_filename(subdir)

    # filename = "/localdev/sraizada/tracy-logs/llama-80-decoder/reports/2025_03_03_13_14_57/ops_perf_results_2025_03_03_13_14_57.csv"
    df = pd.read_csv(filename)
    df = df[df["OP TYPE"].isin(["tt_dnn_device"])]
    df = merge_device_rows(df)
    df = df[int(len(df) / 2) :]
    input_data = df[["OP CODE", "DEVICE KERNEL DURATION [ns]"]].to_dict(orient="records")
    kernel_duration_dict = {}
    for entry in input_data:
        op_code = entry["OP CODE"]
        duration = entry["DEVICE KERNEL DURATION [ns]"]
        if op_code not in kernel_duration_dict:
            kernel_duration_dict[op_code] = []
        kernel_duration_dict[op_code].append(duration)

    expected_times_dict = {
        "Embeddings": [7230.0, 7241.0],
        "DramPrefetcher": [20261638.0],
        "LayerNorm": [6230.0, 6449.0, 7743.0, 7426.0],
        "AllGatherAsync": [2572.0, 11958.0, 2570.0],
        "ReshardDeviceOperation": [1456.0, 1939.0, 2776.0],
        "Matmul": [8022.0, 9004.0, 9727.0, 10140.0, 16769.0],
        "AllReduceAsync": [219371.0, 11674841.0, 772010.0, 33516.0, 3713658.0],
        "NLPCreateHeadsDecodeDeviceOperation": [8840.0],
        "RotaryEmbeddingLlamaFusedQK": [4637.0],
        "PagedUpdateCacheDeviceOperation": [4579.0],
        "ScaledDotProductAttentionDecode": [19907.0],
        "NLPConcatHeadsDecodeDeviceOperation": [7039.0],
        "BinaryDeviceOperation": [2577.0, 13666.0, 2574.0],
    }
    passing = True
    for op_code, durations in kernel_duration_dict.items():
        if op_code in expected_times_dict:
            thresholds = expected_times_dict[op_code]
            # Ensure the lists are of the same length to compare corresponding elements
            if len(durations) == len(thresholds):
                for id, (duration, threshold) in enumerate(zip(durations, thresholds)):  # Compare corresponding items
                    benchmark_data.add_measurement(profiler, 0, step_name, f"{op_code}_{id}", duration)
                    if duration > threshold:
                        passing = False
                        logger.info(f"{op_code}: {duration} ns exceeds the threshold of {threshold} ns")
            else:
                passing = False
                logger.info(
                    f"Warning: Length mismatch for {op_code}. Kernel durations: {len(durations)}, Expected thresholds: {len(thresholds)}"
                )

    benchmark_data.save_partial_run_json(
        profiler,
        run_type=f"llama-80-decoder",
        ml_model_name="llama70b-tg-decoder",
    )

    assert passing
