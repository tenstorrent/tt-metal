"""Experimental eager staging preparation, before any trace capture.

Uses existing production embedding/selector and exact serial slice/pad calls.
This prepares staging leaves only, not attention/MLP/history/sampling families.
Caller supplies the actual admitted max span (including partial-page rewind).
No runtime fallback, changed lengths, cache allocation, or trace capture.
"""
def prepare(model, mesh, maximum_span, record, *, spans=None):
    import torch
    import ttnn
    if not 1 <= maximum_span <= 8319:
        raise ValueError('This diagnostic inventory covers up to8192 scheduled+127rewind tokens')
    kinds={}
    for layer in model.layers:
        kinds.setdefault(layer.kind,layer)
    assert len(kinds)==2
    admitted=set(range(1,maximum_span+1)) if spans is None else set(spans)
    if not admitted or any(type(s) is not int or not 1<=s<=maximum_span for s in admitted):
        raise ValueError('Invalid staging span subset')
    rows=[]
    for physical in sorted({(s+31)//32*32 for s in admitted}):
        token_ids=ttnn.from_torch(torch.zeros((1,physical),dtype=torch.int32),
            device=mesh,dtype=ttnn.uint32,layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
        backing=model.embed(token_ids,sequence_length=physical)
        width=backing.shape[-1]
        for span in range(physical-31,min(physical,maximum_span)+1):
            if span not in admitted:continue
            shape=ttnn.Shape([1,1,span,width]);padded=ttnn.Shape([1,1,physical,width])
            # Short prefill gathers before returning embed; its native key
            # includes true logical height, so a later view cannot substitute.
            x=(model.embed(token_ids,sequence_length=span) if span<=128 else
               ttnn.reshape(backing,shape,padded))
            for layer in kinds.values():
                stream=span>layer.PREFILL_CHUNK
                if stream:
                    output_shape=ttnn.Shape([(span+31)//32,1,32,width])
                    output=ttnn.reshape(x,output_shape,output_shape)
                for begin in range(0,span,layer.PREFILL_CHUNK):
                    end=min(begin+layer.PREFILL_CHUNK,span)
                    inp=(ttnn.reshape(x,ttnn.Shape([1,1,end,width]),
                            ttnn.Shape([1,1,(end+31)//32*32,width]))
                         if begin==0 else x[:,:,begin:end,:])
                    pad=-inp.shape[2]%128
                    if pad:inp=ttnn.pad(inp,[(0,0),(0,0),(0,pad),(0,0)],value=0.0)
                    # Reproduce stream-copy native metadata on this private
                    # preparation embedding, never on model KV/request history.
                    if stream:
                        result=ttnn.clone(inp)
                        assert result.buffer_unique_id()!=x.buffer_unique_id()
                        if begin==0:output.update_tensor_topology(result.tensor_topology())
                        tiled=ttnn.Shape([1,1,(end-begin+31)//32*32,width])
                        part=ttnn.reshape(result,tiled,tiled)
                        table=layer.prefill_output_page_table[:,begin//32:(end+31)//32]
                        ttnn.experimental.paged_fill_cache(output,part,table,batch_idx=0)
                        del table,part,result
                    del inp
                if stream:del output
            if model.runtime_final_tile is not None:
                final=model.runtime_final_tile.select(x,(span-1)//32*32)
                # Production last-row scalar is bounded0..31 after tile selection.
                row=(span-1)%32
                final=final[:,:,row:row+1,:]
                del final
            del x
        del backing,token_ids
        if physical%256==0 or physical==(maximum_span+31)//32*32:
            ttnn.synchronize_device(mesh)
            mem=ttnn.get_memory_view(mesh,ttnn.BufferType.DRAM)
            item=dict(physical=physical,programs=mesh.num_program_cache_entries(),
                allocated_per_bank=mem.total_bytes_allocated_per_bank,
                free_per_bank=mem.total_bytes_free_per_bank,
                largest_free_per_bank=mem.largest_contiguous_bytes_free_per_bank)
            rows.append(item);record(item)
    return rows
