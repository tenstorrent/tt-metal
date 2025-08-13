# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def run_resnet50_pipeline(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
    pipeline_config,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        dealloc_input=True,
        final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
        model_location_generator=model_location_generator,
    )

    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    dram_input_memory_config = sharded_mem_config_DRAM  # This option is ignored for non-trace single CQ

    def model_wrapper(l1_input_tensor):
        test_infra.input_tensor = l1_input_tensor
        return test_infra.run()

    pipeline = create_pipeline_from_config(
        config=pipeline_config,
        model=model_wrapper,
        device=device,
        dram_input_memory_config=dram_input_memory_config,
        l1_input_memory_config=input_mem_config,
    )

    pipeline.compile(tt_inputs_host)

    if use_signpost:
        signpost(header="start")
    outputs = pipeline.enqueue([tt_inputs_host]).pop_all()
    if use_signpost:
        signpost(header="stop")

    for output in outputs:
        test_infra.validate(output)

    pipeline.cleanup()


def run_resnet50_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    run_resnet50_pipeline(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
        PipelineConfig(use_trace=False, num_command_queues=1, all_transfers_on_separate_command_queue=False),
    )


def run_resnet50_trace_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    run_resnet50_pipeline(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
        PipelineConfig(use_trace=True, num_command_queues=1, all_transfers_on_separate_command_queue=False),
    )


def run_resnet50_2cqs_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    run_resnet50_pipeline(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
        PipelineConfig(use_trace=False, num_command_queues=2, all_transfers_on_separate_command_queue=False),
    )


def run_resnet50_trace_2cqs_inference(
    device,
    device_batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    model_location_generator,
):
    run_resnet50_pipeline(
        device,
        device_batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        model_location_generator,
        PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
    )
