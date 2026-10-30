# SPDX-License-Identifier: Apache-2.0
"""Host-only output-finalization routing; no TT devices are opened."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import ttnn

from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM


@pytest.mark.parametrize("already_read", [False, True])
def test_sampled_output_uses_one_replica_and_never_rereads_async_host(already_read):
    adapter=Qwen38ForCausalLM.__new__(Qwen38ForCausalLM)
    adapter.batch_size=2
    adapter.generator=SimpleNamespace(counters={'readbacks':0})
    values=torch.tensor([23,71],dtype=torch.int32)
    host=SimpleNamespace(storage_type=lambda:ttnn.StorageType.HOST)
    calls=[]
    replica=SimpleNamespace(cpu=lambda **kwargs:(calls.append(kwargs) or host))
    device=SimpleNamespace(storage_type=lambda:ttnn.StorageType.DEVICE)
    with patch.object(ttnn,'get_device_tensors',return_value=[replica]) as get, \
         patch.object(ttnn,'to_torch',side_effect=lambda tensor: values if tensor is host else None):
        output=adapter.process_decode_output_host(host if already_read else device,is_tokens=True)
    assert output.tolist()==[[23],[71]] and output.dtype==torch.int64
    assert adapter.generator.counters['readbacks']==(0 if already_read else 1)
    assert get.call_count==(0 if already_read else 1)
    assert calls==([] if already_read else [{'blocking':True}])


def test_torch_output_is_formatted_without_device_api():
    adapter=Qwen38ForCausalLM.__new__(Qwen38ForCausalLM)
    adapter.batch_size=2
    with patch.object(ttnn,'to_torch',side_effect=AssertionError('unexpected conversion')), \
         patch.object(ttnn,'get_device_tensors',side_effect=AssertionError('unexpected read')):
        output=adapter.process_decode_output_host(torch.tensor([5,6]),is_tokens=True)
    assert output.tolist()==[[5],[6]]
    with pytest.raises(ValueError,match='sampled tokens'):
        adapter.process_decode_output_host(output,is_tokens=False)
