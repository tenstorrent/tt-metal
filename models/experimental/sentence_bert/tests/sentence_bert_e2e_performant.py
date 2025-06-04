# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn, torch
from models.experimental.sentence_bert.tests.sentence_bert_test_infra import create_test_infra
from tests.ttnn.utils_for_testing import assert_with_pcc

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


ttnn.buffer_address = buffer_address


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class SentenceBERTrace2CQ:
    def __init__(self):
        ...

    def initialize_sentence_bert_trace_2cqs_inference(
        self,
        device,
        input_ids,
        extended_mask,
        token_type_ids,
        position_ids,
        device_batch_size=8,
        act_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat16,
        sequence_length=384,
    ):
        self.test_infra = create_test_infra(
            device,
            device_batch_size,
            sequence_length,
            input_ids=input_ids,
            extended_mask=extended_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        self.device = device
        self.tt_inputs_host, self.input_mem_config = self.test_infra.setup_l1_sharded_input(device)
        self.tt_image_res = self.tt_inputs_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
        self.op_event = ttnn.record_event(device, 0)

        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        spec = self.test_infra.input_tensor.spec
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.test_infra.validate()
        self.test_infra.dealloc_output()  # output_tensor_1[0].deallocate(force=True)

        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.run()
        self.test_infra.validate()

        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.test_infra.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        self.op_event = ttnn.record_event(device, 0)
        self.test_infra.dealloc_output()  # .output_tensor_1[0].deallocate(force=True)
        trace_input_addr = ttnn.buffer_address(self.test_infra.input_tensor)
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.test_infra.run()
        self.input_tensor = ttnn.allocate_tensor_on_device(spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        assert trace_input_addr == ttnn.buffer_address(self.input_tensor)

    def _validate(self, result_output_tensor):
        torch_output_tensor = self.test_infra.torch_output_tensor_1
        assert_with_pcc(torch_output_tensor, ttnn.to_torch(result_output_tensor).squeeze(dim=1), 0.986)

    def execute_sentence_bert_trace_2cqs_inference(self, iter=0, tt_inputs_host=None):
        if tt_inputs_host is None:
            tt_inputs_host = self.tt_inputs_host

        # torch.save(ttnn.to_torch(tt_inputs_host),f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/trace_inputs_{iter}")
        # print("before copy")
        # p(self.tt_image_res,"tt_image_res")
        # p(tt_inputs_host,"tt_inputs_host")
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(self.tt_inputs_host, self.tt_image_res, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.input_tensor = ttnn.to_memory_config(self.tt_image_res, self.input_mem_config)
        # print("before exec")
        torch.save(
            ttnn.to_torch(self.tt_image_res),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_tt_image_res_{iter}",
        )
        torch.save(
            ttnn.to_torch(tt_inputs_host),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/tt_inputs_host_{iter}",
        )
        torch.save(
            ttnn.to_torch(self.tt_inputs_host),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_tt_inputs_host_{iter}",
        )
        torch.save(
            ttnn.to_torch(self.input_tensor),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_input_tensor_{iter}",
        )
        torch.save(
            ttnn.to_torch(self.ttnn_input_ids),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_ttnn_input_ids_{iter}",
        )
        torch.save(
            ttnn.to_torch(self.ttnn_token_type_ids),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_ttnn_token_type_ids_{iter}",
        )
        torch.save(
            ttnn.to_torch(self.ttnn_position_ids),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_ttnn_position_ids_{iter}",
        )
        torch.save(
            ttnn.to_torch(self.ttnn_attention_mask),
            f"/home/ubuntu/venkatesh_latest/tt-metal/models/experimental/sentence_bert/dumps/self_ttnn_attention_mask_{iter}",
        )
        # p(self.tt_image_res,"tt_image_res")
        # p(tt_inputs_host,"tt_inputs_host")
        # p(self.tt_inputs_host,"self.tt_inputs_host")
        # p(self.input_tensor,"self.input_tensor")
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        # p(self.test_infra.output_tensor_1[0],"output")
        # print("trace after execution")
        self.test_infra.validate()

        outputs = ttnn.from_device(self.test_infra.output_tensor_1[0], blocking=True)
        # print("ttn output after trace", outputs.shape)
        # self._validate(outputs)
        return self.test_infra.output_tensor_1[0]

    def run(self, torch_input_ids, iter):
        input_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = self.execute_sentence_bert_trace_2cqs_inference(tt_inputs_host=input_ids, iter=iter)
        return output

    def release_sentence_bert_trace_2cqs_inference(self):
        ttnn.release_trace(self.device, self.tid)
