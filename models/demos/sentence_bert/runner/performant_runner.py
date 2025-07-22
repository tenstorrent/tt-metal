# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.sentence_bert.runner.performant_runner_infra import SentenceBERTPerformanceRunnerInfra
from tests.ttnn.utils_for_testing import assert_with_pcc


class SentenceBERTPerformantRunner:
    def __init__(
        self,
        device,
        device_batch_size=8,
        sequence_length=384,
        input_ids=None,
        extended_mask=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    ):
        self.device = device
        self.input_ids = input_ids
        self.extended_mask = extended_mask
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.runner_infra = SentenceBERTPerformanceRunnerInfra(
            device=self.device,
            batch_size=device_batch_size,
            sequence_length=sequence_length,
            input_ids=self.input_ids,
            extended_mask=self.extended_mask,
            attention_mask=self.attention_mask,
            token_type_ids=self.token_type_ids,
            position_ids=self.position_ids,
        )
        (
            self.tt_inputs_host,
            sharded_mem_config_DRAM,
            self.input_mem_config,
            self.tt_tokens_host,
            self.tt_posids_host,
            self.tt_ext_att_mask_host,
            self.tt_att_mask_host,
        ) = self.runner_infra.setup_dram_sharded_input(device)
        self.tt_inputs = self.tt_inputs_host.to(device, sharded_mem_config_DRAM)
        self.tt_tokens = self.tt_tokens_host.to(device, sharded_mem_config_DRAM)
        self.tt_pos = self.tt_posids_host.to(device, sharded_mem_config_DRAM)
        self.tt_ext_att_mask = self.tt_ext_att_mask_host.to(device, sharded_mem_config_DRAM)
        self.tt_att_mask = self.tt_att_mask_host.to(device, sharded_mem_config_DRAM)

    def _capture_sentencebert_trace_2cqs(self):
        self.op_event = ttnn.record_event(self.device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 1)
        ttnn.copy_host_to_device_tensor(self.tt_tokens_host, self.tt_tokens, 1)
        ttnn.copy_host_to_device_tensor(self.tt_posids_host, self.tt_pos, 1)
        ttnn.copy_host_to_device_tensor(self.tt_ext_att_mask_host, self.tt_ext_att_mask, 1)
        ttnn.copy_host_to_device_tensor(self.tt_att_mask_host, self.tt_att_mask, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
        self.runner_infra.ttnn_token_ids = ttnn.to_memory_config(self.tt_tokens, self.input_mem_config)
        self.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(self.tt_pos, self.input_mem_config)
        self.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(self.tt_ext_att_mask, self.input_mem_config)
        self.runner_infra.ttnn_att_mask = ttnn.to_memory_config(self.tt_att_mask, self.input_mem_config)
        spec_input = self.runner_infra.ttnn_input_ids.spec
        spec_token = self.runner_infra.ttnn_token_ids.spec
        spec_pos = self.runner_infra.ttnn_pos_ids.spec
        spec_att = self.runner_infra.ttnn_ext_att_mask.spec
        spec_att_2 = self.runner_infra.ttnn_att_mask.spec
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()
        self.runner_infra.dealloc_output()
        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 1)
        ttnn.copy_host_to_device_tensor(self.tt_tokens_host, self.tt_tokens, 1)
        ttnn.copy_host_to_device_tensor(self.tt_posids_host, self.tt_pos, 1)
        ttnn.copy_host_to_device_tensor(self.tt_ext_att_mask_host, self.tt_ext_att_mask, 1)
        ttnn.copy_host_to_device_tensor(self.tt_att_mask_host, self.tt_att_mask, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
        self.runner_infra.ttnn_token_ids = ttnn.to_memory_config(self.tt_tokens, self.input_mem_config)
        self.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(self.tt_pos, self.input_mem_config)
        self.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(self.tt_ext_att_mask, self.input_mem_config)
        self.runner_infra.ttnn_att_mask = ttnn.to_memory_config(self.tt_att_mask, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.run()
        self.runner_infra.validate()

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_inputs, 1)
        ttnn.copy_host_to_device_tensor(self.tt_tokens_host, self.tt_tokens, 1)
        ttnn.copy_host_to_device_tensor(self.tt_posids_host, self.tt_pos, 1)
        ttnn.copy_host_to_device_tensor(self.tt_ext_att_mask_host, self.tt_ext_att_mask, 1)
        ttnn.copy_host_to_device_tensor(self.tt_att_mask_host, self.tt_att_mask, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.runner_infra.ttnn_input_ids = ttnn.to_memory_config(self.tt_inputs, self.input_mem_config)
        self.runner_infra.ttnn_token_ids = ttnn.to_memory_config(self.tt_tokens, self.input_mem_config)
        self.runner_infra.ttnn_pos_ids = ttnn.to_memory_config(self.tt_pos, self.input_mem_config)
        self.runner_infra.ttnn_ext_att_mask = ttnn.to_memory_config(self.tt_ext_att_mask, self.input_mem_config)
        self.runner_infra.ttnn_att_mask = ttnn.to_memory_config(self.tt_att_mask, self.input_mem_config)
        self.op_event = ttnn.record_event(self.device, 0)
        self.runner_infra.dealloc_output()
        trace_input_addr = self.runner_infra.ttnn_input_ids.buffer_address()
        trace_input_addr2 = self.runner_infra.ttnn_token_ids.buffer_address()
        trace_input_addr3 = self.runner_infra.ttnn_pos_ids.buffer_address()
        trace_input_addr4 = self.runner_infra.ttnn_ext_att_mask.buffer_address()
        trace_input_addr5 = self.runner_infra.ttnn_att_mask.buffer_address()
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.runner_infra.run()
        self.ttnn_input_ids = ttnn.allocate_tensor_on_device(spec_input, self.device)
        self.ttnn_token_ids = ttnn.allocate_tensor_on_device(spec_token, self.device)
        self.ttnn_pos_ids = ttnn.allocate_tensor_on_device(spec_pos, self.device)
        self.ttnn_ext_att_mask = ttnn.allocate_tensor_on_device(spec_att, self.device)
        self.ttnn_att_mask = ttnn.allocate_tensor_on_device(spec_att_2, self.device)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        assert trace_input_addr == self.ttnn_input_ids.buffer_address()
        assert trace_input_addr2 == self.ttnn_token_ids.buffer_address()
        assert trace_input_addr3 == self.ttnn_pos_ids.buffer_address()
        assert trace_input_addr4 == self.ttnn_ext_att_mask.buffer_address()
        assert trace_input_addr5 == self.ttnn_att_mask.buffer_address()

    def _execute_sentencebert_trace_2cqs_inference(
        self, tt_inputs_host=None, tt_tokens=None, tt_posids=None, tt_ext_att_mask=None, tt_att_mask=None
    ):
        if tt_inputs_host is None:
            tt_inputs_host = self.tt_inputs_host
            tt_tokens = self.tt_tokens_host
            tt_posids = self.tt_posids_host
            tt_ext_att_mask = self.tt_ext_att_mask_host
            tt_att_mask = self.tt_att_mask_host

        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs, 1)
        ttnn.copy_host_to_device_tensor(tt_tokens, self.tt_tokens, 1)
        ttnn.copy_host_to_device_tensor(tt_posids, self.tt_pos, 1)
        ttnn.copy_host_to_device_tensor(tt_ext_att_mask, self.tt_ext_att_mask, 1)
        ttnn.copy_host_to_device_tensor(tt_att_mask, self.tt_att_mask, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.ttnn_input_ids = ttnn.reshard(self.tt_inputs, self.input_mem_config, self.ttnn_input_ids)
        self.ttnn_token_ids = ttnn.reshard(self.tt_tokens, self.input_mem_config, self.ttnn_token_ids)
        self.ttnn_pos_ids = ttnn.reshard(self.tt_pos, self.input_mem_config, self.ttnn_pos_ids)
        self.ttnn_ext_att_mask = ttnn.reshard(self.tt_ext_att_mask, self.input_mem_config, self.ttnn_ext_att_mask)
        self.ttnn_att_mask = ttnn.reshard(self.tt_att_mask, self.input_mem_config, self.ttnn_att_mask)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        return self.runner_infra.ttnn_output_tensor[0]

    def _validate(self, result_output_tensor):
        torch_output_tensor = self.runner_infra.torch_output.post_processed_output
        assert_with_pcc(torch_output_tensor, result_output_tensor, self.runner_infra.valid_pcc)

    def release(self):
        ttnn.release_trace(self.device, self.tid)

    def run(self, input_ids=None, tokens=None, posids=None, ext_att_mask=None, att_mask=None):
        (
            tt_inputs_host,
            _,
            tt_tokens,
            tt_posids,
            tt_ext_att_mask,
            tt_att_mask,
        ) = self.runner_infra.setup_l1_sharded_input(input_ids, tokens, posids, ext_att_mask, att_mask)
        output = self._execute_sentencebert_trace_2cqs_inference(
            tt_inputs_host, tt_tokens, tt_posids, tt_ext_att_mask, tt_att_mask
        )
        return output
