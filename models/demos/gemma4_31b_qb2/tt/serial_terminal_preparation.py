"""Startup-only serial final-head, compact sampling-copy and cache-restore leaves.

No trace capture, sampling, admission, KV writes, or retained request state.
Caller must subsequently warm/capture the real full-model decode/sampler path.
"""
def prepare(generator,state,record):
    import torch
    import ttnn
    model=generator.model;mesh=model.mesh
    assert not generator.traces and not state.prefill_history
    assert state.history_bank is not None and not state.history_bank.leases
    assert generator.runtime_page_selector is not None
    # Serial final-token head consumes logical one-row activations. Compile
    # both valid incoming hidden widths, preserving actual mesh distribution.
    outputs=[]
    for sharded in (False,True):
        hidden=model.tensor(torch.zeros(1,1,1,model.config.hidden_size),
                            dim=3 if sharded else None,dtype=ttnn.bfloat16)
        logits=model.terminal(hidden)
        outputs.append(logits)
        del hidden
    # Canonical sampling always consumes32 rows, including partial entry rows.
    # Distinct clone owners mirror separate serial-request outputs.
    row=outputs[-1]
    compact=[ttnn.clone(row) for _ in range(state.batch)]
    target=model.tensor(torch.zeros(1,1,32,model.config.vocab_size),dim=3,dtype=ttnn.bfloat16)
    for batch in range(1,state.batch+1):
        merged=compact[0] if batch==1 else ttnn.concat(compact[:batch],dim=2)
        if batch>1 and batch<32:
            merged=ttnn.pad(merged,[(0,0),(0,0),(0,32-batch),(0,0)],value=0.)
        shape=ttnn.Shape([1,1,32,model.config.vocab_size//4])
        ttnn.copy(ttnn.reshape(merged,shape,shape),target)
        del merged
    del compact,row,outputs,target,logits
    # Cache restore selects1..8 sliding pages; physical IDs are runtime inputs.
    # The actual BF8->BF16 native concat/cast geometry is shared across layers.
    index=next(i for i,l in enumerate(model.layers) if l.sliding)
    for cache in state.kv[index]:
        for count in range(1,9):
            parts=[generator.runtime_page_selector.select(cache,page) for page in range(1,count+1)]
            tail=parts[0] if count==1 else ttnn.concat(parts,dim=2)
            result=ttnn.typecast(tail,ttnn.bfloat16)
            del result,tail,parts
    ttnn.synchronize_device(mesh)
    memory=ttnn.get_memory_view(mesh,ttnn.BufferType.DRAM)
    record(dict(label='terminal_sampling_handoff_restore_complete',programs=mesh.num_program_cache_entries(),
        allocated_per_bank=memory.total_bytes_allocated_per_bank,
        free_per_bank=memory.total_bytes_free_per_bank,
        largest_free_per_bank=memory.largest_contiguous_bytes_free_per_bank))
    assert not generator.traces and not state.prefill_history and not state.history_bank.leases
