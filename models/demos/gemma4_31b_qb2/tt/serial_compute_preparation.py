"""Experimental representative prefill computation preparation before capture.

Caller owns fresh, empty request state. This intentionally mutates only specified
startup KV pages; it is not permitted after admission. Full60 closure must prove
that representatives plus actual scalar leaves cover the complete model.
"""
def prepare(generator, state, record):
    import torch
    import ttnn
    from models.demos.gemma4_31b_qb2.tt.history_bank import HistoryTail
    model=generator.model;mesh=model.mesh;bank=state.history_bank
    assert not generator.traces and not state.prefill_history and bank is not None and not bank.leases
    representatives={}
    for i,layer in enumerate(model.layers):representatives.setdefault(layer.kind,i)
    assert len(representatives)==2
    def upload(x,dtype=ttnn.int32):return generator._upload(x.contiguous(),dtype)
    def memory(label):
        ttnn.synchronize_device(mesh)
        v=ttnn.get_memory_view(mesh,ttnn.BufferType.DRAM)
        record(dict(label=label,programs=mesh.num_program_cache_entries(),
            allocated_per_bank=v.total_bytes_allocated_per_bank,
            free_per_bank=v.total_bytes_free_per_bank,
            largest_free_per_bank=v.largest_contiguous_bytes_free_per_bank))
    for kind,i in representatives.items():
        layer=model.layers[i];pair=state.kv[i]
        factor=1 if layer.sliding else 2
        # Actual table width remains2048; only startup pages1..68 are addressed.
        host_table=(factor*(1+torch.arange(2048,dtype=torch.int32))).reshape(1,-1)
        assert int(host_table.max()) < pair[0].shape[0]
        table=upload(host_table)
        spans=list(range(128,layer.PREFILL_CHUNK+1,128))+[layer.PREFILL_CHUNK+128]
        histories=list(range(0,1025,128)) if layer.sliding else [0]
        for span in spans:
            # Long sliding invocation's final128 has a1024-row internal history;
            # initial history variants were already covered by shorter spans.
            hs=[0] if span>layer.PREFILL_CHUNK else histories
            for h in hs:
                starts=[h] if layer.sliding else [128,256]
                for start in starts:
                    ids=upload(torch.zeros((1,span),dtype=torch.int32),ttnn.uint32)
                    x=model.embed(ids,sequence_length=span)
                    history=None
                    if h:
                        history=tuple(ttnn.zeros([1,4,h,256],device=mesh,dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(2))
                    fills={}
                    for begin in range(0,span,layer.PREFILL_CHUNK):
                        end=min(begin+layer.PREFILL_CHUNK,span)
                        a=(start+begin)//128;b=(start+end)//128
                        fills[begin]=upload(host_table[:,a:b])
                    output=layer.prefill_forward(x,page_table=table,kv_cache=pair,
                        consume_input=True,start_pos=start,history=history,
                        return_history=layer.sliding,tail_descriptor=layer.sliding,
                        fill_tables=fills)
                    if layer.sliding:
                        output,tail=output
                        assert isinstance(tail,HistoryTail)
                        bank.write_descriptor(bank.roots[0][i],tail)
                        del tail
                    del output,x,ids,history,fills
            memory(kind+':'+str(span))
        # Restore every touched physical page, including the two-page start.
        clear_table=upload(host_table[:,:68])
        for dtype in dict.fromkeys(cache.dtype for cache in pair):
            clear=ttnn.zeros([1,layer.kv_heads,68*128,layer.head_dim],device=mesh,
                dtype=dtype,layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for cache in pair:
                if cache.dtype == dtype:
                    ttnn.experimental.paged_fill_cache(cache,clear,clear_table,batch_idx=0)
            del clear
        del clear_table,table
    # Every actual scalar is compiled into a native fused residual-add program.
    # Interleaved BinaryNg omits logical size from its key; native closure later
    # validates this source-derived economy against all60 real layers.
    a=ttnn.zeros([1,1,128,1344],device=mesh,dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for scalar in sorted({layer.layer_scalar for layer in model.layers}):
        output=ttnn.add(a,a,dtype=ttnn.bfloat16,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU,scalar)])
        del output
    del a
    # Bank gathers use eight fixed index widths, independently of layer count.
    sliding=next(i for i,layer in enumerate(model.layers) if layer.sliding)
    for h in range(128,1025,128):
        result=bank.read_attention(bank.roots[0][sliding],1152-h,h)
        del result
    memory('representative_compute_complete')
    assert not bank.leases and not state.prefill_history and not generator.traces
