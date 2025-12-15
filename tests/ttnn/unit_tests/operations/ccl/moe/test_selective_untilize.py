import random

import pytest
import torch
import ttnn

from tests.nightly.t3000.ccl.nightly.test_all_to_all_combine import check_results, gen_tensors as gen_tensors_combine

CHUNK_COMPLETE_VAL = torch.iinfo(torch.uint32).max

def gen_combine_semaphore_table(batch, seq ,devices, experts, ready_device_chunk_counts: list[int]):
    assert experts % devices ==0
    assert max(ready_device_chunk_counts) <= devices
    assert min(ready_device_chunk_counts) >=0
    assert len(ready_device_chunk_counts) == devices
    
    num_experts_per_device = experts // devices
    
    # initialize this to the complete signal, so the Op doesn't wait for additional batches
    combine_semaphore_table = torch.full([devices,num_experts_per_device,devices], CHUNK_COMPLETE_VAL, dtype=torch.uint32)
    
    # randomly assign chunk entries on different devices to be in a completed state
    # the op waits for whole columns of this table to be ready before proceeding so
    # need to set whole columns as ready
    for dest_d, ready_cols in zip(range(devices), ready_device_chunk_counts):
        chunk_indexes = list(range(ready_cols * num_experts_per_device))
        random.shuffle(chunk_indexes)
        for r in range(ready):
            e = randint(0,num_experts_per_device-1)
            for _ in range(len(devices))
                source_d = chunk_indexes.pop()
                combine_semaphore_table[dest_d,e,source_d]
    
    return combine_semaphore_table
                


def gen_compute_output_buffers(
    input_sparse_contribs_tensor,expert_mapping,metadata_tensor, combine_semaphore_table
):
    experts = 
    
    buffer_tensor = torch.zeros((experts,devices,chunk_size,hidden_size), dtype=torch.bfloat16)
    
    # experts on the receiver devices
    for e in range(experts):
            
    

    
    
def gen_tensors(
    batch,
    experts,
    selected_experts_k,
    hidden_size,
    seq,
    mesh_shape,
    replication_axis,
    devices,
    chunk_size,
    ready_device_chunks, # op waits for all D of a given E set of chunks to proceed
    scheme="random",
): 

    ( 
        sparse_dispatched_tokens, # probably not needed
        input_sparse_contribs_tensor,
        expert_mapping,
        metadata_tensor,
        output_tensor,
        output_data_map 
    ) = gen_tensors_combine(
        batch,
        experts,
        selected_experts_k,
        hidden_size,
        seq,
        mesh_shape,
        replication_axis,
        devices,
        scheme,
        local_reduce=True,
    )
    
    input_buffer = gen_compute_output_buffers(
        batch,seq ,devices, experts, input_sparse_contribs_tensor,expert_mapping,metadata_tensor, ready_chunks
    )