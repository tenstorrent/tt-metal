# SPDX-License-Identifier: Apache-2.0
"""Host boundary tests. No TT devices are opened; device replay is a model stub."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator


def skeleton(batch=2):
    gen=Generator.__new__(Generator)
    gen.batch_size=batch
    gen.context_length=262144
    gen.sampling_strategy='topk'
    gen.k=torch.zeros(batch,dtype=torch.int32)
    gen.p=torch.zeros(batch)
    gen.temp=torch.zeros(batch)
    gen.seeds=torch.arange(32,dtype=torch.int32)+1234
    gen.seed_values=gen.seeds.clone()
    gen.counters=dict(token_refreshes=0,position_refreshes=0,page_table_refreshes=0,
                      model_replays=0,sampling_replays=0,readbacks=0)
    gen.uploads=[]
    def refresh(value,target):
        gen.uploads.append((id(target),value.clone()))
        target.copy_(value)
    gen.refresh=refresh
    return gen


def test_public_temperature_reciprocal_and_parameter_only_reload():
    gen=skeleton(3)
    gen.seeds.add_(37)
    live=gen.seeds.clone(); initial=gen.seed_values.clone()
    gen.set_sampling_params(top_k=[1,2,32],top_p=[0,1,.9],temperature=[.5,1,2],reset_seed=False)
    assert torch.equal(gen.temp,torch.tensor([2.,1.,.5]))
    assert torch.equal(gen.seeds,live)
    assert torch.equal(gen.seed_values,initial)
    assert all(target!=id(gen.seeds) for target,_ in gen.uploads)


def test_sampling_reset_is_separate_and_accepts_emitted_offsets():
    gen=skeleton()
    gen.reset_sampling_state(seed=[90,170],offsets=[7,11])
    assert gen.seed_values[:2].tolist()==[90,170]
    assert gen.seeds[:2].tolist()==[97,181]
    before=gen.seeds.clone()
    gen.set_sampling_params(top_k=2,temperature=2,reset_seed=False)
    assert torch.equal(gen.seeds,before)
    # Legacy standalone setter still resets by default.
    gen.set_sampling_params(seed=[12,13])
    assert gen.seeds[:2].tolist()==[12,13]


def test_reserved_rng_sentinel_is_rejected():
    gen=skeleton()
    with pytest.raises(ValueError,match='sentinel'):
        gen.reset_sampling_state(seed=(1<<32)-1-gen.context_length,offsets=gen.context_length)


def decode_skeleton():
    gen=skeleton()
    gen.mesh=object(); gen.kv_cache=object()
    gen.page_table=torch.tensor([[7,8],[12,13]],dtype=torch.int32)
    gen.page_table_host=gen.page_table.clone(); gen._page_table_host_valid=True
    gen.tokens=torch.zeros(1,1,1,2,dtype=torch.int32)
    gen.positions=torch.zeros(2,dtype=torch.int32)
    gen.rope_indices=torch.zeros(2,1,dtype=torch.int32)
    gen.active_mask=torch.ones(2,1,1)
    gen.model_trace='model'; gen.sample_trace='sample'; gen.decode_traces={False:'model',True:'masked'}
    gen._decode_active=None
    def execute(mesh,trace,*,cq_id,blocking):
        assert blocking is False
        if trace in ('model','masked'):
            gen.positions.add_(1); gen.rope_indices.add_(1)
        else:
            gen.tokens.add_(10)
    return gen,execute


class Poison:
    def __getattr__(self,name):
        raise AssertionError(f'Stale host input was consulted: {name}')


def test_resident_feedback_and_page_only_update_ignore_poisoned_host_inputs():
    gen,execute=decode_skeleton()
    with patch('ttnn.execute_trace',side_effect=execute):
        gen.decode_forward(torch.tensor([3,4]),torch.tensor([30,50]),
            page_table=gen.page_table_host,kv_cache=gen.kv_cache,return_device=True)
        counts=dict(gen.counters)
        output=gen.decode_forward(Poison(),Poison(),page_table=Poison(),kv_cache=gen.kv_cache,
            return_device=True,reload_inputs=False,reload_page_table=False)
        assert output is gen.tokens
        assert gen.tokens.flatten().tolist()==[23,24]
        assert gen.positions.tolist()==[32,52]
        new_map=torch.tensor([[7,19],[12,21]],dtype=torch.int32)
        gen.decode_forward(Poison(),Poison(),page_table=new_map,kv_cache=gen.kv_cache,
            return_device=True,reload_inputs=False,reload_page_table=True)
        assert torch.equal(gen.page_table,new_map)
        assert gen.tokens.flatten().tolist()==[33,34]
        assert gen.positions.tolist()==[33,53]
        assert gen.counters['token_refreshes']==counts['token_refreshes']
        assert gen.counters['position_refreshes']==counts['position_refreshes']
        assert gen.counters['page_table_refreshes']==counts['page_table_refreshes']+1


def test_resident_requires_authoritative_first_submission():
    gen,_=decode_skeleton()
    with pytest.raises(RuntimeError,match='authoritative'):
        gen.decode_forward(Poison(),Poison(),page_table=Poison(),kv_cache=gen.kv_cache,
            return_device=True,reload_inputs=False,reload_page_table=False)


def test_remap_moves_live_rng_and_dormant_parameter_state_once():
    gen=skeleton(3)
    gen._decode_active=torch.ones(3,dtype=torch.bool)
    conv=torch.arange(18).reshape(3,2,3).float()
    gen.kv_cache=[SimpleNamespace(conv=conv)]
    gen.state_ops={'conv':dict(indices=torch.zeros_like(conv,dtype=torch.int64),scratch=torch.zeros_like(conv))}
    gen.sampling_state_ops={name:dict(indices=torch.zeros_like(getattr(gen,name),dtype=torch.int64),
                                     scratch=torch.zeros_like(getattr(gen,name)))
                            for name in ('seeds','k','p','temp')}
    gen.seeds.add_(97); gen.k.copy_(torch.tensor([1,2,3])); gen.p.copy_(torch.tensor([.1,.2,.3]))
    gen.temp.copy_(torch.tensor([2.,1.,.5]))
    old={name:getattr(gen,name).clone() for name in ('seeds','k','p','temp','seed_values')}
    oldconv=conv.clone(); mapping=torch.tensor([2,0,1])
    def gather(value,dim,indices,out):
        # Match the non-last-axis composite path: returned storage may differ
        # from the supplied optional output. Copying `out` would lose the move.
        return torch.gather(value,dim,indices.long())
    with patch('ttnn.gather',side_effect=gather),patch('ttnn.copy',side_effect=lambda src,dst:dst.copy_(src)):
        gen.remap_state_slots(mapping)
    assert torch.equal(conv,oldconv[mapping])
    for name in old:
        assert torch.equal(getattr(gen,name)[:3],old[name][mapping])
    assert gen._decode_active is None


def test_subset_rng_reset_and_prefill_preserve_live_peer_counters():
    gen=skeleton()
    gen.sampling_merge={name:torch.zeros(1,1,1,32,dtype=torch.int32)
                        for name in ('mask','update','output')}
    gen.sampling_merge['backup']=gen.seeds.clone()
    gen.seeds[:2]=torch.tensor([0x7fffff13,-1024],dtype=torch.int32)
    before=gen.seeds.clone(); initial=gen.seed_values.clone()
    def merge(selected,untouched):
        gen.seeds.copy_(torch.where(gen.sampling_merge['mask'].reshape(-1).bool(),
                                   selected.reshape(-1),untouched.reshape(-1)))
    gen._merge_sampling_counters=merge
    gen.reset_sampling_state(seed=[0x1000001,42],offsets=[7,9],slots=[0])
    assert gen.seeds[0].item()==0x1000008
    assert torch.equal(gen.seeds[1:],before[1:])
    assert torch.equal(gen.seed_values[1:],initial[1:])
    gen.sample_trace='canonical';gen.mesh=object();gen.logits=object();gen.tokens=object()
    before=gen.seeds.clone()
    with patch('ttnn.copy',side_effect=lambda x,y:y.copy_(x)), \
         patch('ttnn.execute_trace',side_effect=lambda *a,**k:gen.seeds.add_(1)):
        assert gen.sample_prefill(gen.logits,active_slots=[0]) is gen.tokens
    assert gen.seeds[0].item()==before[0].item()+1
    assert torch.equal(gen.seeds[1:],before[1:])
    with pytest.raises(ValueError,match='unique'):
        gen.reset_sampling_state(slots=[0,0])


def test_single_slot_remap_validates_then_preserves_without_device_gather():
    gen=skeleton(1)
    gen._decode_active=(True,)
    before=gen.seeds.clone();initial=gen.seed_values.clone()
    with patch('ttnn.gather',side_effect=AssertionError('singleton gather forbidden')):
        gen.remap_state_slots([0])
        assert gen._decode_active is None
        assert torch.equal(before,gen.seeds)
        assert torch.equal(initial,gen.seed_values)
        assert not gen.uploads
        for mapping in ([1],[-1],[],[0,0]):
            with pytest.raises(ValueError,match='every slot'):
                gen.remap_state_slots(mapping)
