# SPDX-License-Identifier: Apache-2.0
"""Metadata-only tensor inventory; no device compute or tensor readback."""
import math
import ttnn


def tensor_bytes(tensor):
    local=ttnn.get_device_tensors(tensor)[0]
    shape=list(local.padded_shape)
    elements=math.prod(shape)
    dtype=str(local.dtype)
    if local.dtype==ttnn.bfloat4_b:
        size=elements//1024*576
    elif local.dtype==ttnn.bfloat8_b:
        size=elements//1024*1088
    else:
        size=elements*(2 if local.dtype==ttnn.bfloat16 else 4)
    return dict(padded_shape=shape,dtype=dtype,layout=str(local.layout),payload_bytes_per_device=size)


def inventory(model):
    rows=[]
    for i,layer in zip(model.layer_indices,model.layers):
        for name in ('in_norm','post_norm','gate','up','gateup','down','mix','out','l2_weight','neg_A','dt','out_norm','q_norm','k_norm'):
            tensor=getattr(layer,name,None)
            if tensor is not None:
                rows.append(dict(name=f'layers.{i}.{name}',layer_type=layer.layer_type,**tensor_bytes(tensor)))
        for name in ('taps','norm_shards','gdn_constants'):
            values=getattr(layer,name,())
            values=values.values() if isinstance(values,dict) else values
            for j,tensor in enumerate(values):
                rows.append(dict(name=f'layers.{i}.{name}.{j}',layer_type=layer.layer_type,**tensor_bytes(tensor)))
    for name in ('embedding','head','final_norm','cos_table','sin_table'):
        rows.append(dict(name=name,**tensor_bytes(getattr(model,name))))
    return dict(tensors=rows,total_payload_bytes_per_device=sum(r['payload_bytes_per_device'] for r in rows),
        basis='Local device padded tensor shape; BFP4/8 tile bytes576/1088, BF16 two bytes, FP32/int32 four bytes. Excludes allocator bank padding, shared workspaces and request/cache/trace buffers.',
        copies={'gate_up':dict(copies=2,reason='Accepted stage05 separate DRAM-sharded prefill projections plus packed decode/one-row projection; same BF4/LoFi policy, measured latency selection retained.'),
                'norm':dict(copies=2,reason='Full norm weights for replicated small-batch decode and hidden-sharded norm weights for prefill/nativeRS batches; no per-request conversion.')})


def request_inventory(generator):
    rows=[]
    for name in ('page_table','tokens','positions','rope_indices','k','p','temp','seeds','prefill_output','active_mask','logits','output_history','output_index'):
        rows.append(dict(name=name,**tensor_bytes(getattr(generator,name))))
    for name in ('_index_offsets','_invalid_vocab_mask','_invalid_vocab_tail_mask','_seeds','_user_ids'):
        tensor=getattr(generator.sampler,name,None)
        if tensor is not None:
            rows.append(dict(name='sampler.'+name,**tensor_bytes(tensor)))
    for physical,bucket in generator.buckets.items():
        for name,tensor in bucket.items():
            rows.append(dict(name=f'bucket{physical}.{name}',**tensor_bytes(tensor)))
    return dict(tensors=rows,total_payload_bytes_per_device=sum(r['payload_bytes_per_device'] for r in rows),
        basis='Persistent request buffers allocated before capture. Tile-padded masks are included; cache and shared workspace are separate; common sampler static buffers are included.')


def runtime_build():
    """Identify the backend actually mapped into this Python process."""
    import hashlib
    from pathlib import Path
    libraries=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/_ttnncpp.so' in line})
    assert len(libraries)==1,libraries
    library=Path(libraries[0])
    return dict(cpp_library=str(library.resolve()),cpp_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                build_provenance='rotary_build_provenance.json')
