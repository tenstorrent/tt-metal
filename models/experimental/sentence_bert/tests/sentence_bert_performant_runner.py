# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.tests.sentence_bert_performant_runner_infra import (
    SentenceBERTPerformanceRunnerInfra,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


class SentenceBERTPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=8,
        sequence_length=384,
        input_ids=None,
        extended_mask=None,
        token_type_ids=None,
        position_ids=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    ):
        self.device = device
        self.input_ids = input_ids
        self.extended_mask = extended_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        print("ibnfwubrf", self.input_ids, self.extended_mask, self.token_type_ids, self.position_ids)
        self.runner_infra = SentenceBERTPerformanceRunnerInfra(
            device=self.device,
            batch_size=device_batch_size,
            sequence_length=sequence_length,
            input_ids=self.input_ids,
            extended_mask=self.extended_mask,
            token_type_ids=self.token_type_ids,
            position_ids=self.position_ids,
        )
        self.tt_inputs_host = self.runner_infra.setup_input()
        self.input_mem_config = ttnn.L1_MEMORY_CONFIG
        self.tt_image_res = self.tt_inputs_host.to(device, ttnn.DRAM_MEMORY_CONFIG)

    def _capture_sentencebert_trace_2cqs(self):
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.runner_infra.ttnn_input_ids.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.ttnn_input_ids.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.ttnn_input_ids = ttnn.allocate_tensor_on_device(spec, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        assert trace_input_addr == self.ttnn_input_ids.buffer_address()

    def _execute_sentencebert_trace_2cqs_inference(self, tt_inputs_host=None):
        # tt_inputs_host = self.tt_inputs_host if tt_inputs_host is None else tt_inputs_host
        # tt_inputs_host = ttnn.from_torch(tt_inputs_host, dtype=ttnn.uint32)
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        # TODO: Add in place support to ttnn to_memory_config
        if self.ttnn_input_ids.is_sharded():
            self.ttnn_input_ids = ttnn.reshard(self.tt_image_res, self.input_mem_config, self.ttnn_input_ids)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        self.runner_infra.validate()
        return self.runner_infra.ttnn_output_tensor[0]

    def _validate(self, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output.last_hidden_state
        assert_with_pcc(torch_output_tensor, result_output_tensor, self.runner_infra.valid_pcc)

    def release(self):
        ttnn.release_trace(self.device, self.tid)

    def run(self, input_ids, check_pcc=True):
        tt_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32)
        output = self._execute_sentencebert_trace_2cqs_inference(tt_input_ids)
        if check_pcc:
            self._validate(result_output_tensor=ttnn.to_torch(output).squeeze(dim=1))
        return output
