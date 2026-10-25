# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of intermediate prefill output suppression."""
from types import SimpleNamespace

import pytest
import torch

from models.autoports.qwen_qwen3_8_27b.tt.generator_vllm import Qwen38ForCausalLM


def adapter(allow_host=False):
    obj=Qwen38ForCausalLM.__new__(Qwen38ForCausalLM)
    obj.batch_size=2
    obj.model=SimpleNamespace(config=SimpleNamespace(vocab_size=5))
    obj.kv_cache=object()
    obj.state_pool=None
    obj.scheduler_block_size=32
    obj.allow_host_sampling=allow_host
    obj.host_logits_reads=0
    obj.decode_slots=set()
    obj.slot_ends=[None,None]
    obj.request_seeds=[1,2]
    obj._event=lambda *args,**kwargs:None
    calls=[]
    logits=torch.arange(10).reshape(2,1,5).float()
    def forward(values,**kwargs):
        calls.append(('forward',kwargs['prompt_lens']))
        return logits
    def host(value):
        calls.append(('read',None))
        return value
    def sample(value,active_slots=None):
        calls.append(('sample',active_slots))
        return torch.tensor([[7],[8]])
    obj.generator=SimpleNamespace(prefill_chunk_size=4096,page_table_host=torch.zeros(2,8,dtype=torch.int32),
        reset_state_slots=lambda slots:calls.append(('reset',slots)),
        prefill_continuation=forward,host_logits=host,sample_prefill=sample)
    obj._sampling=lambda *args,**kwargs:calls.append(('params',kwargs.get('reset_slots')))
    obj.read_decode_output=lambda output:output
    obj.process_decode_output_host=lambda output,**kwargs:output
    return obj,calls


def run(obj,mask=None,params=None):
    return obj.prefill_forward(torch.ones(2,33,dtype=torch.int64),[0,0],[33,33],
        torch.ones(2,8,dtype=torch.int32),obj.kv_cache,
        intermediate_prefill_mask=mask,sampling_params=params)


def test_all_intermediate_updates_state_without_host_opt_in_or_readback():
    obj,calls=adapter()
    output,deltas=run(obj,[True,True])
    assert output.shape==(2,1)
    assert deltas.tolist()==[0,0]
    assert obj.slot_ends==[33,33]
    assert calls==[('reset',[0,1]),('forward',[33,33])]
    assert obj.host_logits_reads==0


def test_mixed_host_prefill_reads_only_final_group():
    obj,calls=adapter(allow_host=True)
    output,_=run(obj,[True,False])
    assert output.shape==(2,1,5)
    assert output[0].eq(0).all()
    assert output[1].tolist()==[[5,6,7,8,9]]
    assert calls.count(('read',None))==1
    assert not any(kind=='sample' for kind,_ in calls)
    assert obj.slot_ends==[33,33]


def test_final_host_prefill_still_requires_explicit_compatibility():
    obj,_=adapter()
    with pytest.raises(ValueError,match='QWEN38_HOST_SAMPLING'):
        run(obj,[False,False])


def test_mixed_device_prefill_does_not_sample_intermediate_group():
    obj,calls=adapter()
    output,_=run(obj,[True,False],params=object())
    assert output.tolist()==[[0],[8]]
    assert calls.count(('sample',[1]))==1
    assert calls.count(('params',[1]))==1
    assert not any(kind=='read' for kind,_ in calls)


def test_events_record_physical_pieces_and_non_aligned_tail():
    obj,calls=adapter()
    obj.generator.prefill_chunk_size=32
    events=[]
    obj._event=lambda name,**fields:events.append((name,fields))
    run(obj,[True,True])
    pieces=[(fields['start'],fields['length']) for name,fields in events
            if name=='prefill_chunk']
    assert pieces==[(0,32),(32,1)]
    assert [value for name,value in calls if name=='forward']==[[32,32],[1,1]]
    assert obj.slot_ends==[33,33]


@pytest.mark.parametrize('suppressed',[False,True])
def test_terminal_work_only_for_final_emitted_chunk(suppressed):
    obj,calls=adapter()
    obj.generator.prefill_chunk_size=32
    forward=obj.generator.prefill_continuation
    terminal=[]
    def record(values,**kwargs):
        terminal.append(kwargs['compute_logits'])
        return forward(values,**kwargs)
    obj.generator.prefill_continuation=record
    run(obj,[suppressed,suppressed],params=object())
    assert terminal==[False,not suppressed]
    assert obj.slot_ends==[33,33]
